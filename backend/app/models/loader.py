"""Model registry and startup loading.

Rules this module enforces:

  * The service starts even when no trained weights exist. It falls back to a
    baseline and says so in the API response -- it does not crash and it does
    not silently pretend a GNN produced the numbers.
  * Whatever actually produced a number is what gets reported. `model_info` on
    every response names the model that ran, not the model that was requested.
"""

from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path

from ..config import get_settings
from ..data.static_provider import get_provider
from .base import ModelInfo, TravelTimePredictor
from .baselines import FreeFlowPredictor, GradientBoostedPredictor, HistoricalMeanPredictor

log = logging.getLogger("journeymind.models")

NEURAL_KEYS = ("graphsage", "gat", "mlp")
WEIGHT_FILENAMES = {
    "graphsage": "graphsage_model.npz",
    "gat": "gat_model.npz",
    "mlp": "mlp_model.npz",
}


class ModelUnavailable(RuntimeError):
    pass


def weights_path(key: str) -> Path:
    return get_settings().models_dir / WEIGHT_FILENAMES.get(key, f"{key}_model.npz")


def _build(key: str) -> TravelTimePredictor:
    if key == "freeflow":
        return FreeFlowPredictor()
    if key == "historical":
        return HistoricalMeanPredictor(provider=get_provider())
    if key == "gbt":
        raise ModelUnavailable(
            "The gradient-boosted-trees baseline is an offline evaluation model. "
            "It needs scikit-learn, which is not installed in the serving image. "
            "Run scripts/evaluate.py locally to compare it."
        )
    if key in NEURAL_KEYS:
        from .gnn_numpy import NeuralEdgePredictor
        path = weights_path(key)
        if not path.exists():
            raise ModelUnavailable(
                f"No trained weights at {path.name}. Run "
                f"`python scripts/train.py --encoder {key}` to produce them."
            )
        return NeuralEdgePredictor.load(path)
    raise ModelUnavailable(f"Unknown model '{key}'")


@lru_cache(maxsize=8)
def get_predictor(key: str | None = None) -> TravelTimePredictor:
    """Resolve the requested model, falling back rather than failing."""
    s = get_settings()
    want = key or s.travel_time_model
    try:
        p = _build(want)
        log.info("travel-time model: %s", p.info.display_name)
        return p
    except ModelUnavailable as exc:
        log.warning("model '%s' unavailable (%s); falling back to '%s'",
                    want, exc, s.model_fallback)
    except Exception as exc:  # a corrupt checkpoint must not take the app down
        log.exception("model '%s' failed to load (%s); falling back to '%s'",
                      want, exc, s.model_fallback)
    try:
        return _build(s.model_fallback)
    except Exception:
        log.exception("fallback model failed too; using free-flow")
        return FreeFlowPredictor()


def registry() -> list[dict]:
    """Every model in the comparison set and whether it can run right now.

    This is what section 18 of the documentation asks for: one interface, six
    implementations, no claim that any of them is better without evaluation.
    """
    s = get_settings()
    rows: list[dict] = []
    order = ("freeflow", "historical", "gbt", "mlp", "graphsage", "gat")
    static_info = {
        "freeflow": FreeFlowPredictor.info,
        "historical": HistoricalMeanPredictor.info,
        "gbt": GradientBoostedPredictor.info,
    }
    from .gnn_numpy import DISPLAY
    for key in order:
        if key in static_info:
            info: ModelInfo = static_info[key]
            available = key != "gbt" or GradientBoostedPredictor.available()
            reason = (None if available else
                      "offline evaluation only — scikit-learn is not in the serving image")
            rows.append({**info.as_dict(), "available": available, "reason": reason,
                         "baseline_number": order.index(key) + 1})
        else:
            name, family, note = DISPLAY[key]
            path = weights_path(key)
            available = path.exists()
            rows.append({
                "model": name, "key": key, "family": family,
                "prediction": "estimated edge travel time",
                "status": "prototype", "notes": note,
                "trained_on": "bundled travel-time observations",
                "available": available,
                "reason": None if available else f"no trained weights ({path.name})",
                "baseline_number": order.index(key) + 1,
            })
    for r in rows:
        r["active"] = (r["key"] == s.travel_time_model)
    return rows


def active_model_info(predictor: TravelTimePredictor) -> dict:
    s = get_settings()
    d = predictor.info.as_dict()
    d["requested"] = s.travel_time_model
    d["fell_back"] = predictor.info.name != s.travel_time_model
    if hasattr(predictor, "metrics") and predictor.metrics:
        d["validation_metrics"] = predictor.metrics
    return d
