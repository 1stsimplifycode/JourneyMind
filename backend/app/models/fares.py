"""Fare estimation.

Two genuinely different things live here and the difference is carried all the
way to the UI:

  PUBLISHED  metro and bus fares come from operator fare tables. A number, not
             a guess. Still transcribed by hand, so the source is named.
  ESTIMATED  ride-hailing fares come from a transparent
                 base + distance x per_km + duration x per_min
             model with an uncertainty band. There is no public price feed and
             surge pricing is proprietary, so we do not model it and we never
             present these as quotes.

Nothing in this module contacts any operator.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..data.provider import FareModel


@dataclass(frozen=True)
class FareEstimate:
    amount: float          # point estimate, rupees
    low: float             # lower end of the band (== amount when exact)
    high: float            # upper end of the band
    provenance: str        # exact | published | estimated
    label: str             # human label for the mode
    note: str
    source: str | None = None

    @property
    def is_range(self) -> bool:
        return self.high - self.low > 0.51

    def display(self, symbol: str = "₹") -> str:
        if not self.is_range:
            return f"{symbol}{self.amount:.0f}"
        return f"{symbol}{self.low:.0f}–{symbol}{self.high:.0f}"


def _slab_fare(model: FareModel, distance_km: float) -> float:
    for upper, fare in model.slabs:
        if distance_km <= upper:
            return float(fare)
    return float(model.above_top_slab_fare)


def _metered_fare(model: FareModel, distance_km: float, duration_min: float) -> float:
    extra_km = max(0.0, distance_km - model.base_distance_km)
    fare = model.base_fare + extra_km * model.per_km + duration_min * model.per_min
    return max(fare, model.minimum_fare)


def estimate_fare(model: FareModel, distance_km: float, duration_min: float) -> FareEstimate:
    """One leg's fare under one mode's fare rule."""
    if model.kind == "flat":
        amount = float(model.flat_fare)
    elif model.kind == "distance_slab":
        amount = _slab_fare(model, distance_km)
    elif model.kind == "metered":
        amount = _metered_fare(model, distance_km, duration_min)
    else:  # unknown rule: refuse to invent a number
        raise ValueError(f"Unsupported fare rule '{model.kind}' for mode '{model.mode}'")

    band = model.uncertainty_pct
    low = amount * (1.0 - band)
    high = amount * (1.0 + band)
    if model.kind == "metered" and band > 0:
        # round the band outward to whole rupees so the UI never implies
        # more precision than the model has
        low, high = max(0.0, round(low)), round(high)
    return FareEstimate(
        amount=round(amount, 2), low=round(low, 2), high=round(high, 2),
        provenance=model.provenance, label=model.label,
        note=model.note, source=model.source,
    )


class FareEstimator:
    """Applies the right fare rule per mode, and knows how a journey's legs
    combine into a total (a metro fare is charged once end-to-end, not per
    inter-station hop)."""

    def __init__(self, fares: dict[str, FareModel]):
        self.fares = fares

    def has(self, mode: str) -> bool:
        return mode in self.fares

    def model_for(self, mode: str) -> FareModel:
        if mode not in self.fares:
            raise KeyError(f"No fare model configured for mode '{mode}'")
        return self.fares[mode]

    def leg_fare(self, mode: str, distance_km: float, duration_min: float) -> FareEstimate:
        return estimate_fare(self.model_for(mode), distance_km, duration_min)

    def combine(self, estimates: list[FareEstimate]) -> FareEstimate:
        """Total across a journey. Provenance degrades to the weakest link:
        one estimated leg makes the whole total an estimate."""
        if not estimates:
            return FareEstimate(0.0, 0.0, 0.0, "exact", "Free", "No paid legs.")
        rank = {"exact": 0, "published": 1, "estimated": 2}
        worst = max(estimates, key=lambda e: rank.get(e.provenance, 2)).provenance
        return FareEstimate(
            amount=round(sum(e.amount for e in estimates), 2),
            low=round(sum(e.low for e in estimates), 2),
            high=round(sum(e.high for e in estimates), 2),
            provenance=worst, label="Total",
            note=("Includes at least one estimated ride-hailing fare."
                  if worst == "estimated" else
                  "Built from published operator fare tables."),
        )

    def provenance_summary(self, modes: list[str]) -> dict[str, str]:
        return {m: self.fares[m].provenance for m in modes if m in self.fares}
