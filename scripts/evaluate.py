"""Evaluate every model in the comparison set, and the recommendations themselves.

    python scripts/evaluate.py                 # prediction + recommendation metrics
    python scripts/evaluate.py --spatial       # also run the held-out-region test

Two things are evaluated separately, because they fail independently:

  A. Is the travel-time prediction accurate?   MAE, RMSE, MAPE, peak-hour error
  B. Do the recommendations actually help?     constraint satisfaction, regret

Splitting is temporal (weeks 1-5 / 6 / 7-8), never random. `--spatial`
additionally holds out a geographic region: train on the rest of the map, test
on that region. That is the stricter test and the more interesting result --
edges in a held-out region cannot be memorised, so it is where graph structure
should matter if it matters anywhere.

Whatever the numbers say is what gets written to EVALUATION.md. This script has
no opinion about which model ought to win.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "backend"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.data.static_provider import get_provider          # noqa: E402
from app.graph.builder import get_graph                    # noqa: E402
from app.graph.features import TimeContext                 # noqa: E402
from app.models.baselines import (                         # noqa: E402
    GradientBoostedPredictor, HistoricalMeanPredictor, is_peak, mae, mape, rmse,
)
from train import TRAIN_END_DAY, VAL_END_DAY, build_dataset, train  # noqa: E402


# --------------------------------------------------------------------------
# A. prediction accuracy
# --------------------------------------------------------------------------
def score(y_true, y_pred, hour, weekend) -> dict:
    peak = np.asarray([is_peak(float(h), bool(w)) for h, w in zip(hour, weekend)])
    out = {
        "MAE_min": round(mae(y_true, y_pred), 4),
        "RMSE_min": round(rmse(y_true, y_pred), 4),
        "MAPE_pct": round(mape(y_true, y_pred), 3),
        "n": int(len(y_true)),
    }
    if peak.any():
        out["peak_MAE_min"] = round(mae(y_true[peak], y_pred[peak]), 4)
        out["peak_RMSE_min"] = round(rmse(y_true[peak], y_pred[peak]), 4)
        out["peak_MAPE_pct"] = round(mape(y_true[peak], y_pred[peak]), 3)
        out["peak_n"] = int(peak.sum())
    return out


def evaluate_predictions(seeds=(0,), epochs=450, patience=60, verbose=True,
                         out_dir=None) -> dict:
    graph, d = build_dataset(verbose=verbose)
    tr = d["day"] < TRAIN_END_DAY
    te = d["day"] >= VAL_END_DAY
    idx_te = np.flatnonzero(te)
    y_te = d["y"][idx_te]
    hour_te, wk_te = d["hour"][idx_te], d["weekend"][idx_te]

    results: dict[str, dict] = {}

    # -- baseline 1: free-flow -------------------------------------------
    results["1. Free-flow time"] = score(y_te, d["base_min"][idx_te], hour_te, wk_te)

    # -- baseline 2: historical mean per edge per hour --------------------
    obs = list(get_provider().get_travel_times())
    start = datetime(2025, 1, 6)
    train_obs = [o for o in obs
                 if (datetime.fromisoformat(o.ts) - start).days < TRAIN_END_DAY]
    hist = HistoricalMeanPredictor().fit(train_obs)
    pred = np.asarray([
        hist._lookup(str(d["edge_ids"][i]), int(d["hour"][i]),
                     int(d["weekend"][i]), float(d["base_min"][i]))
        for i in idx_te
    ])
    results["2. Historical mean"] = score(y_te, pred, hour_te, wk_te)

    # -- baseline 3: gradient-boosted trees, no graph ---------------------
    if GradientBoostedPredictor.available():
        X = np.hstack([d["edge_feats"][d["edge_row"]], d["time_feats"]])
        gbt = GradientBoostedPredictor().fit(X[tr], np.log1p(d["y"][tr]))
        results["3. Gradient-boosted trees"] = score(
            y_te, np.maximum(np.expm1(gbt.model.predict(X[idx_te])), 0.05),
            hour_te, wk_te)
    else:
        results["3. Gradient-boosted trees"] = {"skipped": "scikit-learn not installed"}

    # -- baselines 4-6: the neural models, across seeds --------------------
    for key, label in (("mlp", "4. MLP (graph removed)"),
                       ("graphsage", "5. GraphSAGE"),
                       ("gat", "6. GAT")):
        runs = []
        for s in seeds:
            if verbose:
                print(f"  training {key} (seed {s}) ...")
            _, res, _, _ = train(encoder=key, epochs=epochs, patience=patience,
                                 seed=s, verbose=False, out_dir=out_dir)
            runs.append(res["test"])
        agg = {
            "MAE_min": round(float(np.mean([r["MAE_min"] for r in runs])), 4),
            "RMSE_min": round(float(np.mean([r["RMSE_min"] for r in runs])), 4),
            "MAPE_pct": round(float(np.mean([r["MAPE_pct"] for r in runs])), 3),
            "peak_MAE_min": round(float(np.mean([r["peak_MAE_min"] for r in runs])), 4),
            "n": runs[0]["n"], "seeds": list(seeds),
        }
        if len(runs) > 1:
            agg["MAE_min_std"] = round(float(np.std([r["MAE_min"] for r in runs])), 4)
            agg["MAE_min_range"] = [round(min(r["MAE_min"] for r in runs), 4),
                                    round(max(r["MAE_min"] for r in runs), 4)]
        results[label] = agg

    return results


# --------------------------------------------------------------------------
# B. do the recommendations actually help?
# --------------------------------------------------------------------------
SCENARIOS = [
    # (origin, destination, budget, max_time, preference)
    ("pl_majestic_bus", "pl_indiranagar_100ft", 100, 30, "balanced"),
    ("pl_home", "pl_college", 100, 45, "balanced"),
    ("pl_home", "pl_domlur", 150, 50, "fastest"),
    ("pl_home", "pl_domlur", 150, 90, "cheapest"),
    ("pl_banashankari_home", "pl_lalbagh_gate", 80, 35, "balanced"),
    ("pl_jayanagar_4b", "pl_mg_road_shops", 90, 45, "balanced"),
    ("pl_rv_college", "pl_home", 100, 45, "balanced"),
    ("pl_koramangala", "pl_indiranagar_100ft", 120, 40, "fastest"),
    ("pl_lalbagh_gate", "pl_majestic_bus", 60, 40, "cheapest"),
    ("pl_home", "pl_indiranagar_100ft", 100, 50, "balanced"),
]
HOURS = (8, 9, 13, 18, 21)


def evaluate_recommendations(verbose=True) -> dict:
    """Constraint satisfaction, regret, and whether mixing modes actually wins.

    Regret here is measured in the optimiser's own weighted-score units against
    the best journey in the full candidate pool -- i.e. how much worse the
    recommendation was than the best available choice in hindsight. It is a
    self-consistency measure, not ground truth: there is no ground truth for
    "the best journey", which is precisely why the documentation asks for a
    user study as well.
    """
    from app.optimisation import constraints as C
    from app.optimisation.scoring import score_all, weights_for
    from app.services.engine import JourneyMindEngine, JourneyRequest, RoutingError

    eng = JourneyMindEngine()
    places = {p.place_id: p for p in eng.graph.places}

    total = feasible = satisfied = 0
    multimodal = 0
    mm_better_cost = mm_better_time = 0
    regrets: list[float] = []
    savings: list[float] = []
    elapsed: list[float] = []
    no_fit_with_fallbacks = 0
    no_fit = 0

    for o, d, budget, max_time, pref in SCENARIOS:
        if o not in places or d not in places:
            continue
        for hour in HOURS:
            total += 1
            a, b = places[o], places[d]
            try:
                rec = eng.recommend(JourneyRequest(
                    origin_lat=a.lat, origin_lon=a.lon, origin_label=a.name,
                    dest_lat=b.lat, dest_lon=b.lon, dest_label=b.name,
                    departure=datetime(2025, 1, 7, hour, 0),
                    budget=float(budget), max_time_min=float(max_time),
                    preference=pref))
            except RoutingError:
                continue
            elapsed.append(rec.pipeline.get("elapsed_ms", 0.0))

            if not rec.feasible:
                no_fit += 1
                if rec.fallbacks:
                    no_fit_with_fallbacks += 1
                continue

            feasible += 1
            j = rec.recommended
            st = C.evaluate(j, budget, max_time)
            if st.feasible:
                satisfied += 1

            pool = [j] + [alt["journey"] for alt in rec.alternatives
                          if alt["kind"] == "feasible"]
            if len(pool) > 1:
                w, _ = weights_for(pref)
                ranked = score_all(pool, w)
                regrets.append(max(0.0, (j.score or 0.0) - (ranked[0].score or 0.0)))

            vehicles = {m for m in j.modes if m != "walk"}
            if len(vehicles) >= 2:
                multimodal += 1
                singles = [x for x in ([j] + [alt["journey"] for alt in rec.alternatives])
                           if len({m for m in x.modes if m != "walk"}) == 1]
                if singles:
                    cheapest_single = min(singles, key=lambda x: x.cost)
                    fastest_single = min(singles, key=lambda x: x.total_min)
                    if j.cost < cheapest_single.cost - 0.5:
                        mm_better_cost += 1
                        savings.append(cheapest_single.cost - j.cost)
                    if j.total_min < fastest_single.total_min - 0.5:
                        mm_better_time += 1

    return {
        "scenarios_run": total,
        "feasible_recommendation_returned": feasible,
        "constraint_satisfaction_rate": round(satisfied / max(feasible, 1), 4),
        "no_feasible_journey": no_fit,
        "no_feasible_journey_with_labelled_fallbacks": no_fit_with_fallbacks,
        "recommendation_is_multimodal_rate": round(multimodal / max(feasible, 1), 4),
        "multimodal_beat_every_single_mode_on_cost": mm_better_cost,
        "multimodal_beat_every_single_mode_on_time": mm_better_time,
        "median_saving_vs_cheapest_single_mode_inr":
            round(float(np.median(savings)), 2) if savings else None,
        "mean_regret_score_units": round(float(np.mean(regrets)), 6) if regrets else 0.0,
        "max_regret_score_units": round(float(np.max(regrets)), 6) if regrets else 0.0,
        "median_latency_ms": round(float(np.median(elapsed)), 1) if elapsed else None,
        "p95_latency_ms": round(float(np.percentile(elapsed, 95)), 1) if elapsed else None,
        "note": ("Regret is measured against the best journey in this request's own "
                 "feasible pool, in the optimiser's weighted-score units. It measures "
                 "self-consistency, not correctness -- there is no ground truth for "
                 "'the best journey'."),
    }


# --------------------------------------------------------------------------
# spatial hold-out (RQ5)
# --------------------------------------------------------------------------
def evaluate_spatial_holdout(verbose=True) -> dict:
    """Train on most of the map, test on a region the model never saw.

    This is the stricter test the documentation asks for. Edges in a held-out
    region cannot be memorised from their own history, so it is the setting in
    which neighbourhood structure should help if it helps anywhere.
    """
    graph = get_graph()
    lons = [n.lon for n in graph.nodes.values()]
    cut = float(np.percentile(lons, 72))       # hold out the eastern ~28%
    held = {nid for nid, n in graph.nodes.items() if n.lon > cut}
    if verbose:
        print(f"  spatial hold-out: lon > {cut:.4f} — {len(held)}/{len(graph.nodes)} nodes")
    return {
        "held_out_nodes": len(held),
        "total_nodes": len(graph.nodes),
        "cut_lon": round(cut, 5),
        "status": "not run",
        "why": ("Requires a training run that masks every observation whose edge "
                "touches the held-out region, which scripts/train.py does not yet "
                "support. The split is computed here so the experiment is defined "
                "and reproducible; running it is the single highest-value next "
                "step for RQ1 and RQ5."),
    }


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--epochs", type=int, default=450)
    ap.add_argument("--patience", type=int, default=60)
    ap.add_argument("--skip-training", action="store_true",
                    help="only run the recommendation-level metrics")
    ap.add_argument("--spatial", action="store_true")
    ap.add_argument("--out", default=os.path.join(ROOT, "models", "evaluation.json"))
    ap.add_argument("--out-dir", default=None,
                    help="where seed checkpoints go. Defaults to the directory "
                         "of --out, so evaluating one city never overwrites "
                         "another city's served weights.")
    a = ap.parse_args()

    report: dict = {"generated_from": "bundled synthetic study-area dataset"}

    if not a.skip_training:
        print("=== A. travel-time prediction accuracy (temporal split) ===")
        out_dir = a.out_dir or os.path.dirname(os.path.abspath(a.out))
        pred = evaluate_predictions(seeds=tuple(a.seeds), epochs=a.epochs,
                                    patience=a.patience, out_dir=out_dir)
        report["prediction"] = pred
        print(f"\n{'model':<28}{'MAE':>8}{'RMSE':>8}{'MAPE%':>8}{'peakMAE':>9}")
        for name, m in pred.items():
            if "skipped" in m:
                print(f"{name:<28}{'—':>8}  ({m['skipped']})")
                continue
            print(f"{name:<28}{m['MAE_min']:>8.3f}{m['RMSE_min']:>8.3f}"
                  f"{m['MAPE_pct']:>8.2f}{m.get('peak_MAE_min', float('nan')):>9.3f}"
                  + (f"   ±{m['MAE_min_std']:.3f} over seeds {m['seeds']}"
                     if "MAE_min_std" in m else ""))

    print("\n=== B. do the recommendations help? ===")
    rec = evaluate_recommendations()
    report["recommendation"] = rec
    for k, v in rec.items():
        if k != "note":
            print(f"  {k:<52} {v}")

    if a.spatial:
        print("\n=== C. spatial hold-out (RQ5) ===")
        sp = evaluate_spatial_holdout()
        report["spatial_holdout"] = sp
        for k, v in sp.items():
            print(f"  {k:<22} {v}")

    report["honesty_note"] = (
        "Every number here is measured on synthetic data whose generator makes "
        "neighbourhood averaging useful by construction. These figures describe "
        "that generator, not a real city, and must never be quoted as evidence "
        "about real-world travel-time prediction."
    )
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
