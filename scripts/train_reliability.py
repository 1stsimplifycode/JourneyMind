"""Train and evaluate the reliability heads.

    python scripts/train_reliability.py

Fits three calibrated logistic heads (match / accept / cancel) on the simulated
booking history, exports them to models/reliability_model.npz for NumPy
serving, and reports them against a ladder of baselines.

THE BASELINE LADDER
-------------------
A probability model that is not compared against a lookup table is not
evaluated, it is advertised. The ladder here mirrors EVALUATION.md's:

    1. global rate            one number for everything
    2. per-provider rate      "bike taxis cancel 24% of the time"
    3. provider x hour bucket a lookup table -- the real bar, and much
                              stronger than people expect
    4. logistic regression    the served model
    5. gradient-boosted trees does good classical ML beat it?

If the lookup table wins, this script says so and the served model should
change. That has to be a possible outcome or the comparison is theatre.

THE SPLIT IS TEMPORAL, NEVER RANDOM
-----------------------------------
Weeks 1-7 train, week 8 validate, weeks 9-10 test. A random split would put
09:00 Tuesday in training and 09:05 Tuesday in test, and the reported accuracy
would be fiction -- the same argument scripts/train.py makes for travel time.

WHAT THESE NUMBERS MEAN
-----------------------
They describe the generator in scripts/generate_mobility_data.py. They are not
evidence about any real operator's cancellation behaviour. See SOURCES.md.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from datetime import datetime

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "backend"))

from app.reliability.features import (                       # noqa: E402
    FEATURE_NAMES, HEADS, encode_rows, feature_signature,
)

DATA = os.path.join(ROOT, "data", "mobility", "bookings.csv")
MODELS = os.path.join(ROOT, "models")
OUT = os.path.join(MODELS, "reliability_model.npz")
REPORT = os.path.join(MODELS, "reliability_evaluation.json")

TRAIN_END_DAY = 49      # weeks 1-7
VAL_END_DAY = 56        # week 8

#: Which rows each head is asked about, and what it predicts.
HEAD_SPEC = {
    "match":  dict(subset=lambda r: True,                  label=lambda r: int(r["matched"])),
    "accept": dict(subset=lambda r: int(r["matched"]) == 1, label=lambda r: int(r["accepted"])),
    "cancel": dict(subset=lambda r: int(r["accepted"]) == 1, label=lambda r: int(r["cancelled"])),
}


# --------------------------------------------------------------------------
# metrics
# --------------------------------------------------------------------------
def brier(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.mean((p - y) ** 2))


def log_loss(y: np.ndarray, p: np.ndarray) -> float:
    p = np.clip(p, 1e-9, 1 - 1e-9)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def auc(y: np.ndarray, p: np.ndarray) -> float:
    """Rank-based AUC. Ties handled by averaging ranks."""
    if len(np.unique(y)) < 2:
        return float("nan")
    order = np.argsort(p, kind="mergesort")
    ranks = np.empty(len(p), dtype=float)
    sp = p[order]
    i = 0
    while i < len(sp):
        j = i
        while j + 1 < len(sp) and sp[j + 1] == sp[i]:
            j += 1
        ranks[order[i:j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    n1 = float(y.sum())
    n0 = float(len(y) - n1)
    return float((ranks[y == 1].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def ece(y: np.ndarray, p: np.ndarray, bins: int = 12) -> float:
    """Expected calibration error — the number that decides whether a
    probability may be multiplied into a rupee figure."""
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (p >= lo) & (p < hi if hi < 1.0 else p <= hi)
        if not m.any():
            continue
        total += m.mean() * abs(y[m].mean() - p[m].mean())
    return float(total)


def reliability_curve(y: np.ndarray, p: np.ndarray, bins: int = 10) -> list[dict]:
    edges = np.linspace(0.0, 1.0, bins + 1)
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (p >= lo) & (p < hi if hi < 1.0 else p <= hi)
        if not m.any():
            continue
        out.append(dict(bin_lo=round(float(lo), 3), bin_hi=round(float(hi), 3),
                        n=int(m.sum()), predicted=round(float(p[m].mean()), 4),
                        observed=round(float(y[m].mean()), 4)))
    return out


def score(y, p) -> dict:
    return dict(brier=round(brier(y, p), 5), log_loss=round(log_loss(y, p), 5),
                auc=round(auc(y, p), 4), ece=round(ece(y, p), 5), n=int(len(y)))


# --------------------------------------------------------------------------
def load_rows():
    with open(DATA, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    start = datetime.fromisoformat(rows[0]["ts"])
    for r in rows:
        r["_day"] = (datetime.fromisoformat(r["ts"]) - start).days
    return rows


def split(rows):
    tr = [r for r in rows if r["_day"] <= TRAIN_END_DAY]
    va = [r for r in rows if TRAIN_END_DAY < r["_day"] <= VAL_END_DAY]
    te = [r for r in rows if r["_day"] > VAL_END_DAY]
    return tr, va, te


def hour_bucket(r) -> tuple:
    return (r["provider_id"], int(float(r["hour"])) // 3, int(r["is_weekend"]))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--C", type=float, default=1.0, help="inverse regularisation")
    args = ap.parse_args()

    if not os.path.exists(DATA):
        raise SystemExit("no booking history — run scripts/generate_mobility_data.py first")
    from sklearn.linear_model import LogisticRegression

    rows = load_rows()
    tr_all, va_all, te_all = split(rows)
    print(f"bookings: {len(rows)}  train {len(tr_all)}  val {len(va_all)}  test {len(te_all)}")
    print("split: temporal (weeks 1-7 / 8 / 9-10)\n")

    export: dict[str, np.ndarray] = {}
    report: dict[str, dict] = {}

    # one shared standardiser, fitted on the training rows of the widest head
    X_all_train = encode_rows(tr_all)
    mean = X_all_train.mean(axis=0)
    scale = X_all_train.std(axis=0)
    scale[scale < 1e-8] = 1.0

    for head in HEADS:
        spec = HEAD_SPEC[head]
        tr = [r for r in tr_all if spec["subset"](r)]
        va = [r for r in va_all if spec["subset"](r)]
        te = [r for r in te_all if spec["subset"](r)]
        ytr = np.array([spec["label"](r) for r in tr], dtype=float)
        yva = np.array([spec["label"](r) for r in va], dtype=float)
        yte = np.array([spec["label"](r) for r in te], dtype=float)
        Xtr = (encode_rows(tr) - mean) / scale
        Xva = (encode_rows(va) - mean) / scale
        Xte = (encode_rows(te) - mean) / scale

        print(f"=== head: {head}  (predicting P({head}))  "
              f"train {len(tr)} / test {len(te)}, base rate {ytr.mean():.3f} ===")

        results: dict[str, dict] = {}

        # 1. global rate
        results["1. global rate"] = score(yte, np.full(len(yte), ytr.mean()))

        # 2. per-provider rate
        prov_rate = {}
        for p in {r["provider_id"] for r in tr}:
            ys = [spec["label"](r) for r in tr if r["provider_id"] == p]
            prov_rate[p] = float(np.mean(ys)) if ys else float(ytr.mean())
        results["2. per-provider rate"] = score(
            yte, np.array([prov_rate.get(r["provider_id"], ytr.mean()) for r in te]))

        # 3. provider x 3-hour x weekday lookup — the real bar
        buckets: dict[tuple, list] = {}
        for r in tr:
            buckets.setdefault(hour_bucket(r), []).append(spec["label"](r))
        lut = {k: float(np.mean(v)) for k, v in buckets.items() if len(v) >= 20}
        results["3. provider x hour lookup"] = score(
            yte, np.array([lut.get(hour_bucket(r),
                                   prov_rate.get(r["provider_id"], ytr.mean())) for r in te]))

        # 4. logistic regression — the served model
        lr = LogisticRegression(C=args.C, max_iter=2000, random_state=args.seed)
        lr.fit(Xtr, ytr)
        p_te = lr.predict_proba(Xte)[:, 1]
        results["4. logistic regression"] = score(yte, p_te)

        # 5. gradient-boosted trees — comparison only, never served
        try:
            from sklearn.ensemble import HistGradientBoostingClassifier
            gbt = HistGradientBoostingClassifier(
                max_iter=250, learning_rate=0.07, max_depth=6, random_state=args.seed)
            gbt.fit(Xtr, ytr)
            results["5. gradient-boosted trees"] = score(yte, gbt.predict_proba(Xte)[:, 1])
        except Exception as exc:
            print(f"  (gbt skipped: {exc})")

        for name, m in results.items():
            print(f"  {name:28s} brier {m['brier']:.5f}  logloss {m['log_loss']:.5f}  "
                  f"auc {m['auc']:.4f}  ece {m['ece']:.5f}")

        # THE DECISION RULE, STATED BEFORE THE NUMBERS ARE READ
        # Brier ranks; ECE decides. This probability is multiplied into a rupee
        # figure, so a model that discriminates slightly better but is less
        # well calibrated makes the expected-cost number worse, not better.
        # Ties on Brier within 0.0005 are treated as ties, because they are.
        served = "4. logistic regression"
        best_brier = min(results.items(), key=lambda kv: kv[1]["brier"])[0]
        best_ece = min(results.items(), key=lambda kv: kv[1]["ece"])[0]
        gap = results[served]["brier"] - results[best_brier]["brier"]
        print(f"  best by Brier: {best_brier}   best by ECE: {best_ece}")
        if best_brier == served:
            print(f"  -> serving {served} (wins on both counts)")
        elif gap <= 0.0005:
            print(f"  -> serving {served}: {best_brier} is ahead on Brier by "
                  f"{gap:.5f}, which is a tie, and {served} is better calibrated")
        elif best_ece == served:
            print(f"  -> serving {served} DESPITE losing on Brier by {gap:.5f}, "
                  f"because it is better calibrated and this number is "
                  f"multiplied into money. Recorded, not hidden.")
        else:
            print(f"  -> WARNING: {best_brier} beats the served model on Brier "
                  f"by {gap:.5f} AND on ECE. The served model should change.")

        export[f"coef_{head}"] = lr.coef_[0].astype(np.float64)
        export[f"intercept_{head}"] = np.float64(lr.intercept_[0])
        report[head] = dict(
            base_rate=round(float(ytr.mean()), 5),
            n_train=len(tr), n_test=len(te),
            baselines=results,
            best_by_brier=best_brier,
            best_by_ece=best_ece,
            served="logistic regression",
            selection_rule=("Brier ranks, ECE decides: the output is multiplied "
                            "into an expected-cost figure, so calibration is "
                            "worth more than a marginal discrimination gain."),
            calibration=reliability_curve(yte, p_te),
            coefficients={n: round(float(c), 4)
                          for n, c in zip(FEATURE_NAMES, lr.coef_[0])},
        )
        top = sorted(zip(FEATURE_NAMES, lr.coef_[0]), key=lambda t: -abs(t[1]))[:5]
        print("  strongest signals: " +
              ", ".join(f"{n} {c:+.2f}" for n, c in top) + "\n")

    os.makedirs(MODELS, exist_ok=True)
    np.savez(
        OUT, mean=mean, scale=scale,
        feature_names=np.array(list(FEATURE_NAMES)),
        version=np.array(f"reliability-v1-seed{args.seed}"),
        trained_on=np.array("simulated booking history (data/mobility/bookings.csv)"),
        data_class=np.array("SIMULATED"),
        n_train=np.int64(len(tr_all)),
        **export,
    )
    size_kb = os.path.getsize(OUT) / 1024
    print(f"wrote {OUT} ({size_kb:.0f} KB)")

    with open(REPORT, "w", encoding="utf-8") as fh:
        json.dump(dict(
            generated_from="scripts/train_reliability.py",
            split="temporal: weeks 1-7 train, 8 validate, 9-10 test",
            features=feature_signature(),
            heads=report,
            honesty_note=(
                "Measured on the simulated booking history generated by "
                "scripts/generate_mobility_data.py. These figures describe that "
                "generator. They are not evidence about any real ride-hailing "
                "operator's cancellation behaviour."),
        ), fh, indent=2)
    print(f"wrote {REPORT}")


if __name__ == "__main__":
    main()
