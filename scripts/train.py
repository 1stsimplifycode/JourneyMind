"""Train an edge travel-time model and export it for CPU serving.

    python scripts/train.py --encoder graphsage
    python scripts/train.py --encoder gat
    python scripts/train.py --encoder mlp        # baseline 4, the ablation

Splitting is TEMPORAL, not random. A random split would put 09:00 Tuesday in
training and 09:15 Tuesday in test; the model would have effectively seen the
answer and the reported accuracy would be fiction. Weeks 1-5 train, week 6
validate, weeks 7-8 test -- which is the task the model actually has to do:
predict a future it has not observed.

The output is a `.npz` that the serving path replays in NumPy, so the deployed
service never imports PyTorch.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from datetime import datetime

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "backend"))

from app.data.static_provider import get_provider              # noqa: E402
from app.graph.builder import get_graph                        # noqa: E402
from app.graph.features import TimeContext                     # noqa: E402
from app.models.gnn_torch import (                             # noqa: E402
    EdgeTravelTimeModel, huber_log_loss, to_minutes,
)

# Weeks 1-5 train, week 6 validate, weeks 7-8 test. The bundle starts on a Monday.
TRAIN_END_DAY = 35
VAL_END_DAY = 42


def build_dataset(verbose: bool = True):
    """One row per observation, aligned to the graph's static edges."""
    graph = get_graph()
    provider = get_provider()

    # every static edge, keyed by its data-layer edge id (both directions of an
    # edge share one id and therefore share observations)
    rows_by_edge_id: dict[str, int] = {}
    uv, edge_feats = [], []
    for row, idx in enumerate(graph.static_edge_idx):
        e = graph.edges[idx]
        if e.edge_id in rows_by_edge_id:
            continue
        rows_by_edge_id[e.edge_id] = len(uv)
        uv.append([graph.node_pos[e.u], graph.node_pos[e.v]])
        edge_feats.append(graph.edge_features[row])
    uv = np.asarray(uv, dtype=np.int64)
    edge_feats = np.asarray(edge_feats, dtype=np.float32)

    start = datetime(2025, 1, 6)
    X_edge, X_time, Y, DAY, HOUR, WEEKEND, EDGE_ID, BASE = [], [], [], [], [], [], [], []
    skipped = 0
    for o in provider.get_travel_times():
        r = rows_by_edge_id.get(o.edge_id)
        if r is None:
            skipped += 1
            continue
        ts = datetime.fromisoformat(o.ts)
        ctx = TimeContext(hour=o.hour, dow=o.dow, rain=o.rain)
        X_edge.append(r)
        X_time.append(ctx.vector())
        Y.append(o.observed_min)
        DAY.append((ts - start).days)
        HOUR.append(o.hour)
        WEEKEND.append(1 if o.is_weekend else 0)
        EDGE_ID.append(o.edge_id)
        BASE.append(o.base_min)

    data = dict(
        node_feats=graph.node_features.astype(np.float32),
        adj=graph.adj_index,
        uv=uv, edge_feats=edge_feats,
        edge_row=np.asarray(X_edge, dtype=np.int64),
        time_feats=np.asarray(X_time, dtype=np.float32),
        y=np.asarray(Y, dtype=np.float32),
        day=np.asarray(DAY, dtype=np.int32),
        hour=np.asarray(HOUR, dtype=np.float32),
        weekend=np.asarray(WEEKEND, dtype=np.int32),
        edge_ids=np.asarray(EDGE_ID),
        base_min=np.asarray(BASE, dtype=np.float32),
    )
    if verbose:
        print(f"dataset: {len(Y)} observations over {len(uv)} distinct edges "
              f"({skipped} skipped as unmatched)")
    return graph, data


def temporal_split(day: np.ndarray):
    tr = day < TRAIN_END_DAY
    va = (day >= TRAIN_END_DAY) & (day < VAL_END_DAY)
    te = day >= VAL_END_DAY
    return tr, va, te


def normalisation(node_feats, edge_feats, mask_rows, edge_row):
    """Standardisation statistics, fitted over the WHOLE graph.

    Deliberately not fitted on observed edges only. Transfer edges carry no
    travel-time observations at all -- that is the sparse-label situation the
    documentation describes, and it is exactly the case the GNN is supposed to
    generalise into. Fitting the scaler on observed edges alone gives the
    `class_transfer` one-hot column zero variance, and every transfer edge then
    arrives at inference scaled by 1/epsilon. Feature statistics are not labels,
    the full graph is known at training time, and using it here is what keeps
    unobserved edges on the same scale as observed ones.

    The 1e-3 floor is a second guard: a genuinely constant column contributes
    nothing and must not be amplified.
    """
    n_mean, n_std = node_feats.mean(axis=0), node_feats.std(axis=0)
    e_mean, e_std = edge_feats.mean(axis=0), edge_feats.std(axis=0)
    return dict(node_mean=n_mean, node_std=np.maximum(n_std, 1e-3),
                edge_mean=e_mean, edge_std=np.maximum(e_std, 1e-3))


def metrics(y_true, y_pred, hour, weekend) -> dict:
    from app.models.baselines import is_peak, mae, mape, rmse
    peak = np.asarray([is_peak(float(h), bool(w)) for h, w in zip(hour, weekend)])
    out = {
        "MAE_min": round(mae(y_true, y_pred), 4),
        "RMSE_min": round(rmse(y_true, y_pred), 4),
        "MAPE_pct": round(mape(y_true, y_pred), 3),
        "n": int(len(y_true)),
    }
    if peak.any():
        out["peak_MAE_min"] = round(mae(y_true[peak], y_pred[peak]), 4)
        out["peak_MAPE_pct"] = round(mape(y_true[peak], y_pred[peak]), 3)
        out["peak_n"] = int(peak.sum())
    return out


def train(encoder="graphsage", epochs=140, lr=3e-3, hidden=48, layers=2, heads=2,
          head_hidden=64, dropout=0.1, batch=8192, seed=0, patience=60,
          out_dir=None, verbose=True):
    torch.manual_seed(seed)
    np.random.seed(seed)
    graph, d = build_dataset(verbose=verbose)
    tr, va, te = temporal_split(d["day"])
    if verbose:
        print(f"split: train={tr.sum()} val={va.sum()} test={te.sum()}")

    norm = normalisation(d["node_feats"], d["edge_feats"], tr, d["edge_row"])
    node_x = torch.tensor((d["node_feats"] - norm["node_mean"]) / norm["node_std"])
    edge_x = torch.tensor((d["edge_feats"] - norm["edge_mean"]) / norm["edge_std"])
    src = torch.tensor(d["adj"][0])
    dst = torch.tensor(d["adj"][1])
    uv = torch.tensor(d["uv"])
    time_x = torch.tensor(d["time_feats"])
    y = torch.tensor(d["y"])
    row = torch.tensor(d["edge_row"])

    model = EdgeTravelTimeModel(encoder=encoder, hidden=hidden, layers=layers,
                                heads=heads, head_hidden=head_hidden, dropout=dropout)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    idx_tr = np.flatnonzero(tr)
    idx_va = np.flatnonzero(va)
    idx_te = np.flatnonzero(te)

    def forward(idx):
        r = row[idx]
        return model(node_x, src, dst, uv[r], edge_x[r], time_x[idx])

    best_val, best_state, bad = float("inf"), None, 0
    for ep in range(1, epochs + 1):
        model.train()
        perm = np.random.permutation(idx_tr)
        total = 0.0
        for i in range(0, len(perm), batch):
            sl = torch.tensor(perm[i:i + batch])
            opt.zero_grad()
            loss = huber_log_loss(forward(sl), y[sl])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            total += float(loss) * len(sl)
        sched.step()

        model.eval()
        with torch.no_grad():
            vi = torch.tensor(idx_va)
            val_loss = float(huber_log_loss(forward(vi), y[vi]))
            val_mae = float(torch.mean(torch.abs(to_minutes(forward(vi)) - y[vi])))
        if val_loss < best_val - 1e-5:
            best_val, bad = val_loss, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
        if verbose and (ep % 20 == 0 or ep == 1):
            print(f"  epoch {ep:3d}  train {total / len(perm):.5f}  "
                  f"val {val_loss:.5f}  val MAE {val_mae:.3f} min")
        if bad >= patience:
            if verbose:
                print(f"  early stop at epoch {ep} (no val improvement for {patience})")
            break

    if best_state:
        model.load_state_dict(best_state)
    model.eval()

    results = {}
    with torch.no_grad():
        for name, idx in (("validation", idx_va), (" test", idx_te)):
            ii = torch.tensor(idx)
            pred = to_minutes(forward(ii)).numpy()
            results[name.strip()] = metrics(d["y"][idx], pred, d["hour"][idx],
                                            d["weekend"][idx])
    if verbose:
        for k, v in results.items():
            print(f"  {k:<11} {v}")

    out_dir = out_dir or os.environ.get("JM_MODELS_DIR") or os.path.join(ROOT, "models")
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{encoder}_model.npz")
    meta = model.export_npz(
        path, norm=norm, metrics=results,
        extra=dict(
            trained_on="bundled synthetic travel-time observations "
                       "(weeks 1-5 train, week 6 validate, weeks 7-8 test)",
            split="temporal", seed=seed, epochs_run=ep,
            honesty_note=(
                "Trained on the bundled synthetic dataset. These metrics describe "
                "performance on that generator, not on a real city."),
        ),
    )
    if verbose:
        print(f"  wrote {path} ({os.path.getsize(path) / 1024:.0f} KB)")
    return model, results, path, meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", default="graphsage", choices=["graphsage", "gat", "mlp"])
    ap.add_argument("--epochs", type=int, default=140)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--hidden", type=int, default=48)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--heads", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", default=None,
                    help="where the .npz goes. Defaults to $JM_MODELS_DIR, "
                         "then models/. Training a second city into the "
                         "first city's directory overwrites its weights.")
    ap.add_argument("--patience", type=int, default=60,
                    help="Identical for every encoder, so the ablation is fair.")
    ap.add_argument("--all", action="store_true",
                    help="train graphsage, gat and the mlp ablation in one go")
    a = ap.parse_args()

    encoders = ["graphsage", "gat", "mlp"] if a.all else [a.encoder]
    summary = {}
    for enc in encoders:
        print(f"\n=== training {enc} ===")
        _, res, path, _ = train(encoder=enc, epochs=a.epochs, lr=a.lr,
                                hidden=a.hidden, layers=a.layers, heads=a.heads,
                                seed=a.seed, patience=a.patience,
                                out_dir=a.out_dir)
        summary[enc] = res
    print("\n=== summary (test split) ===")
    for enc, res in summary.items():
        t = res["test"]
        print(f"  {enc:<10} MAE {t['MAE_min']:.3f} min  RMSE {t['RMSE_min']:.3f}  "
              f"MAPE {t['MAPE_pct']:.2f}%")
    print(json.dumps(summary, indent=2)[:0])  # keep json import honest


if __name__ == "__main__":
    main()
