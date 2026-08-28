"""Serving the reliability models.

Three heads answer three different questions about one request:

    match    will any vehicle respond?
    accept   given one responded, will the driver take this fare?
    cancel   given the driver took it, will they abandon before pickup?

WHY LOGISTIC REGRESSION AND NOT SOMETHING LARGER
------------------------------------------------
This is the question the whole product turns on, so the model choice is argued
rather than assumed.

1. **The output must be a calibrated probability, not a score.** It is
   multiplied into an expected-cost calculation, so "0.3" has to mean "happens
   three times in ten" or every rupee downstream is wrong. Logistic regression
   optimises exactly that (log loss), and its calibration is checked in
   `scripts/evaluate_reliability.py` rather than assumed.
2. **The coefficients are the explanation.** The product promises to say *why*
   an option was not recommended. A readable weight on `short_trip_penalty` is
   that sentence; a tree ensemble's feature importance is not.
3. **It serves without scikit-learn.** The deployed image deliberately carries
   no sklearn and no torch (see `backend/requirements.txt`); the GNN already
   ships as exported weights replayed in NumPy. A linear model is four lines of
   NumPy, so the reliability layer costs the image nothing.
4. **It is not assumed to win.** `scripts/evaluate_reliability.py` runs it
   against a global rate, a per-provider rate, a per-provider-per-hour lookup
   and gradient-boosted trees, and reports whichever comes out ahead — the same
   discipline `EVALUATION.md` applies to the GNN, which currently reports
   *against* the graph model.

WHY NOT A GNN HERE
------------------
A GNN earns its place when a prediction depends on the *neighbourhood* of a
node, which is true of road congestion and is why the travel-time model is a
graph model. Cancellation is not that shape: it is driven by properties of the
individual request — how short the fare is, how far the pickup is, what hour it
is, whether it is raining. The one genuinely spatial input, neighbourhood
congestion, is already available as a per-zone scalar computed by the graph, so
the graph's contribution arrives as a feature rather than as an architecture.
Adding message passing here would add parameters, training cost and opacity to
buy nothing measurable. That judgement is recorded so it can be revisited if
zone-level supply spillover ever gets real data behind it.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .features import FEATURE_DIM, FEATURE_NAMES, HEADS, RequestFeatures

log = logging.getLogger("journeymind.reliability")

WEIGHTS_FILENAME = "reliability_model.npz"

#: Used when no trained checkpoint is present. Deliberately pessimistic and
#: deliberately flat: a fallback must never look like a confident prediction.
FALLBACK_RATES = {
    "bike_taxi": dict(p_match=0.86, p_accept=0.74, p_cancel=0.22),
    "auto": dict(p_match=0.83, p_accept=0.82, p_cancel=0.14),
    "cab": dict(p_match=0.91, p_accept=0.87, p_cancel=0.11),
    "carpool": dict(p_match=0.61, p_accept=0.76, p_cancel=0.15),
}
DEFAULT_FALLBACK = dict(p_match=0.85, p_accept=0.80, p_cancel=0.16)


def _sigmoid(z: np.ndarray | float):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30.0, 30.0)))


@dataclass(frozen=True)
class ReliabilityPrediction:
    p_match: float
    p_accept: float
    p_cancel: float
    source: str                 # "model" | "fallback"
    model_version: str | None
    drivers_basis: str
    #: Per-head contribution of each feature, largest first. This is what the
    #: interface turns into "not recommended because ...".
    drivers: tuple[tuple[str, float], ...] = ()

    @property
    def p_success_per_attempt(self) -> float:
        return self.p_match * self.p_accept * (1.0 - self.p_cancel)


class ReliabilityModel:
    """Calibrated linear heads, replayed in NumPy."""

    def __init__(self, coef: dict[str, np.ndarray], intercept: dict[str, float],
                 mean: np.ndarray, scale: np.ndarray, meta: dict):
        self.coef = coef
        self.intercept = intercept
        self.mean = mean
        self.scale = scale
        self.meta = meta

    # -- loading -----------------------------------------------------------
    @classmethod
    def load(cls, path: str | Path) -> "ReliabilityModel":
        z = np.load(path, allow_pickle=False)
        names = [str(n) for n in z["feature_names"]]
        if names != list(FEATURE_NAMES):
            raise ValueError(
                "reliability checkpoint was fitted on different features — "
                f"expected {len(FEATURE_NAMES)}, checkpoint has {len(names)}. "
                "Re-run scripts/train_reliability.py.")
        coef = {h: z[f"coef_{h}"] for h in HEADS}
        intercept = {h: float(z[f"intercept_{h}"]) for h in HEADS}
        meta = {"version": str(z["version"]), "trained_on": str(z["trained_on"]),
                "n_train": int(z["n_train"]), "data_class": str(z["data_class"])}
        return cls(coef, intercept, z["mean"], z["scale"], meta)

    # -- prediction --------------------------------------------------------
    def _head(self, head: str, x: np.ndarray) -> tuple[float, list[tuple[str, float]]]:
        xs = (x - self.mean) / self.scale
        contrib = self.coef[head] * xs
        z = float(contrib.sum() + self.intercept[head])
        drivers = sorted(zip(FEATURE_NAMES, contrib.tolist()),
                         key=lambda t: -abs(t[1]))
        return float(_sigmoid(z)), drivers

    def predict(self, f: RequestFeatures) -> ReliabilityPrediction:
        x = f.vector()
        p_match, _ = self._head("match", x)
        p_accept, _ = self._head("accept", x)
        p_cancel, cancel_drivers = self._head("cancel", x)
        top = tuple((n, round(v, 4)) for n, v in cancel_drivers[:4] if abs(v) > 0.01)
        return ReliabilityPrediction(
            p_match=p_match, p_accept=p_accept, p_cancel=p_cancel,
            source="model", model_version=self.meta.get("version"),
            # Rendered in the interface, so it reads as a sentence rather than
            # a provenance tag. The machine-readable class stays on the API.
            drivers_basis=(
                f"calibrated from {self.meta.get('n_train', 0):,} historical "
                f"bookings ({self.meta.get('version', 'v1')})"),
            drivers=top,
        )


class FallbackReliability:
    """Flat per-provider rates, used when no checkpoint is present.

    It exists so the service starts and answers rather than failing, and it
    says `source="fallback"` on every prediction so nothing downstream can
    mistake a constant for a model.
    """

    meta = {"version": "fallback", "data_class": "assumption"}

    def predict(self, f: RequestFeatures) -> ReliabilityPrediction:
        r = FALLBACK_RATES.get(f.provider_id, DEFAULT_FALLBACK)
        return ReliabilityPrediction(
            p_match=r["p_match"], p_accept=r["p_accept"], p_cancel=r["p_cancel"],
            source="fallback", model_version=None,
            drivers_basis=("no trained reliability model is loaded — these are "
                           "flat per-provider assumptions, not predictions"),
            drivers=(),
        )


_model: ReliabilityModel | FallbackReliability | None = None


def get_reliability_model(models_dir: Path | None = None):
    """Load once, cache. Falls back rather than failing, and says which."""
    global _model
    if _model is not None:
        return _model
    if models_dir is None:
        from ..config import get_settings
        models_dir = get_settings().models_dir
    path = Path(models_dir) / WEIGHTS_FILENAME
    if path.exists():
        try:
            _model = ReliabilityModel.load(path)
            log.info("reliability model: %s (%s bookings)",
                     _model.meta.get("version"), _model.meta.get("n_train"))
            return _model
        except Exception as exc:            # a bad checkpoint must not kill boot
            log.warning("reliability checkpoint unusable (%s) — using fallback", exc)
    else:
        log.warning("no reliability checkpoint at %s — using flat fallback rates", path)
    _model = FallbackReliability()
    return _model


def reset_cache() -> None:
    """Test hook."""
    global _model
    _model = None
