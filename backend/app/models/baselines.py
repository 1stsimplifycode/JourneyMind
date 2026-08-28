"""Non-graph baselines (documentation section 17, baselines 1-3).

Baseline 4 -- the MLP with identical features and the graph removed -- is the
critical ablation, so it lives in the neural module beside GraphSAGE and GAT
and shares their training code. Putting it there is deliberate: same features,
same capacity, same optimiser, same loss. Only message passing differs.
"""

from __future__ import annotations

import math
from collections import defaultdict

import numpy as np

from ..graph.features import TimeContext
from .base import ModelInfo, TravelTimePredictor


class FreeFlowPredictor(TravelTimePredictor):
    """Baseline 1: distance / speed limit. Naive physics, no learning at all."""

    info = ModelInfo(
        name="freeflow", display_name="Free-flow time", family="analytic",
        predicts="estimated edge travel time", status="baseline",
        trained_on="nothing — closed form",
        notes="Distance divided by the free-flow speed. Ignores time of day entirely.",
    )

    def predict_static(self, graph, ctx: TimeContext) -> np.ndarray:
        return np.asarray(
            [graph.edges[i].base_min for i in graph.static_edge_idx], dtype=np.float32
        )

    def predict_rows(self, node_feats, edge_feats, src, dst, edge_uv, time_feats,
                     base_min=None):
        if base_min is None:
            raise ValueError("free-flow baseline needs the per-row base_min column")
        return np.asarray(base_min, dtype=np.float32)


class HistoricalMeanPredictor(TravelTimePredictor):
    """Baseline 2: a lookup table of the mean observed time per edge, per hour
    bucket, per weekday/weekend. Deceptively strong -- most of the signal in
    travel time is "this road, at this hour, usually takes this long"."""

    info = ModelInfo(
        name="historical", display_name="Historical mean", family="lookup",
        predicts="estimated edge travel time", status="baseline",
        trained_on="bundled travel-time observations",
        notes="Mean observed minutes for this edge in this hour bucket. "
              "Falls back to the edge's own mean, then to free-flow, when a "
              "bucket was never observed.",
    )

    def __init__(self, provider=None):
        self._provider = provider
        self._table: dict[tuple[str, int, int], float] | None = None
        self._edge_mean: dict[str, float] = {}

    # -- fitting -----------------------------------------------------------
    def fit(self, observations) -> "HistoricalMeanPredictor":
        acc: dict[tuple[str, int, int], list[float]] = defaultdict(list)
        per_edge: dict[str, list[float]] = defaultdict(list)
        for o in observations:
            key = (o.edge_id, int(o.hour), 1 if o.is_weekend else 0)
            acc[key].append(o.observed_min)
            per_edge[o.edge_id].append(o.observed_min)
        self._table = {k: float(np.mean(v)) for k, v in acc.items()}
        self._edge_mean = {k: float(np.mean(v)) for k, v in per_edge.items()}
        return self

    def _ensure(self) -> None:
        if self._table is None:
            if self._provider is None:
                raise RuntimeError(
                    "HistoricalMeanPredictor needs either a fitted table or a provider"
                )
            self.fit(self._provider.get_travel_times())

    # -- prediction --------------------------------------------------------
    def _lookup(self, edge_id: str, hour: int, weekend: int, fallback: float) -> float:
        assert self._table is not None
        v = self._table.get((edge_id, hour, weekend))
        if v is not None:
            return v
        # nearest observed hour for this edge before giving up
        for delta in (1, -1, 2, -2, 3, -3):
            v = self._table.get((edge_id, (hour + delta) % 24, weekend))
            if v is not None:
                return v
        return self._edge_mean.get(edge_id, fallback)

    def predict_static(self, graph, ctx: TimeContext) -> np.ndarray:
        self._ensure()
        hour, weekend = int(ctx.hour), 1 if ctx.is_weekend else 0
        out = np.empty(len(graph.static_edge_idx), dtype=np.float32)
        for row, idx in enumerate(graph.static_edge_idx):
            e = graph.edges[idx]
            out[row] = self._lookup(e.edge_id, hour, weekend, e.base_min)
        return out

    def predict_rows(self, node_feats, edge_feats, src, dst, edge_uv, time_feats,
                     edge_ids=None, hours=None, weekends=None, base_min=None):
        self._ensure()
        n = len(edge_ids)
        out = np.empty(n, dtype=np.float32)
        for i in range(n):
            out[i] = self._lookup(edge_ids[i], int(hours[i]), int(weekends[i]),
                                  float(base_min[i]))
        return out


class GradientBoostedPredictor(TravelTimePredictor):
    """Baseline 3: gradient-boosted trees on edge + time features, no graph.

    Requires scikit-learn, which is an offline training dependency and is NOT
    installed in the deployed image. The registry reports it as unavailable
    at serving time rather than pretending it is there.
    """

    info = ModelInfo(
        name="gbt", display_name="Gradient-boosted trees", family="tree",
        predicts="estimated edge travel time", status="baseline",
        trained_on="bundled travel-time observations (edge + time features)",
        notes="Good classical ML on a flat feature table. Sees no graph structure.",
    )

    def __init__(self, model=None):
        self.model = model

    @staticmethod
    def available() -> bool:
        try:
            import sklearn  # noqa: F401
            return True
        except ImportError:
            return False

    def fit(self, X: np.ndarray, y: np.ndarray, **kw) -> "GradientBoostedPredictor":
        from sklearn.ensemble import HistGradientBoostingRegressor
        self.model = HistGradientBoostingRegressor(
            loss="absolute_error", max_iter=kw.get("max_iter", 300),
            learning_rate=kw.get("learning_rate", 0.08),
            max_depth=kw.get("max_depth", 8), random_state=kw.get("seed", 0),
        )
        self.model.fit(X, y)
        return self

    def _predict_log(self, X: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("GradientBoostedPredictor has not been fitted")
        return np.expm1(self.model.predict(X)).astype(np.float32)

    def predict_static(self, graph, ctx: TimeContext) -> np.ndarray:
        tv = ctx.vector()
        X = np.hstack([graph.edge_features,
                       np.tile(tv, (graph.edge_features.shape[0], 1))])
        return np.maximum(self._predict_log(X), 0.05)

    def predict_rows(self, node_feats, edge_feats, src, dst, edge_uv, time_feats, **kw):
        return np.maximum(self._predict_log(np.hstack([edge_feats, time_feats])), 0.05)


def free_flow_minutes(distance_km: float, speed_kmph: float) -> float:
    return distance_km / max(speed_kmph, 1e-6) * 60.0


def peak_hours() -> tuple[tuple[float, float], ...]:
    """Windows used by the peak-hour error metric."""
    return ((7.5, 10.5), (17.0, 20.5))


def is_peak(hour: float, is_weekend: bool) -> bool:
    if is_weekend:
        return False
    return any(lo <= hour <= hi for lo, hi in peak_hours())


def mape(y_true: np.ndarray, y_pred: np.ndarray, floor: float = 0.5) -> float:
    denom = np.maximum(np.abs(y_true), floor)
    return float(np.mean(np.abs(y_true - y_pred) / denom) * 100.0)


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(math.sqrt(float(np.mean((y_true - y_pred) ** 2))))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))
