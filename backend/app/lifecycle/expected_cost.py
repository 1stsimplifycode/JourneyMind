"""Expected cost: what the trip will actually cost you, not what it advertises.

THE PROBLEM THIS SOLVES
-----------------------
A bike-taxi advertises 20 rupees. A third of the time the driver cancels after
accepting, you have already lost six minutes, and the re-request lands in a
market that just proved tight, so the second ride costs 28. The number on the
card is 20. The number you pay is not.

Every consumer app shows the 20. This module computes the rest.

HOW
---
The lifecycle in `states.py` is an absorbing Markov chain, and it is small
enough to solve exactly rather than simulate. One attempt succeeds with

    q = p_match x p_accept x (1 - p_cancel)

so attempt k is reached with probability (1-q)^(k-1), and the outcome space of
the whole booking is a short, exact list:

    success on attempt k   (1-q)^(k-1) . q      you pay fare_k
    abandoned after K      (1-q)^K              you pay for the fallback instead

That enumeration is the whole model. It gives an exact mean, an exact
distribution, and therefore honest quantiles -- not a point estimate with a
confidence adjective bolted on.

WHY A FALLBACK TERM
-------------------
If every attempt fails you do not teleport home; you take the next best thing.
Ignoring that makes unreliable options look cheap, because their failure mass
gets costed at zero. The fallback is the most reliable alternative available
for the same trip -- usually transit or walking -- and it is priced in.

WHAT IS ASSUMED, STATED PLAINLY
-------------------------------
* `surge_per_retry` -- that a re-request after a failure is dearer. Real, and
  the direction is not in doubt, but the magnitude here is an assumption.
* `cancel_discovery_frac` -- that a driver cancellation is discovered late,
  after most of the pickup wait. Set from the pickup estimate, not observed.
* Rider-side cancellation fees are modelled as zero, because the failures here
  are driver-side.
These are parameters with defaults, not constants buried in an expression, so
they can be fitted the moment real outcome data exists.
"""

from __future__ import annotations

from dataclasses import dataclass, field

#: Retry budget. Beyond a handful of failed requests a real person stops trying
#: and does something else, and the tail contributes almost nothing anyway.
DEFAULT_MAX_ATTEMPTS = 3


@dataclass(frozen=True)
class LifecycleParams:
    """The behavioural assumptions, all overridable, none hidden."""

    max_attempts: int = DEFAULT_MAX_ATTEMPTS
    surge_per_retry: float = 0.18      # each re-request costs ~18% more
    search_timeout_min: float = 2.5    # how long you wait before "no drivers"
    match_min: float = 0.6             # matching round-trip
    cancel_discovery_frac: float = 0.7  # fraction of pickup wasted on a cancel
    cancellation_fee: float = 0.0      # driver-side failure: rider is not charged
    value_of_time_per_min: float = 0.0  # money value of wasted minutes; 0 = off


@dataclass(frozen=True)
class Outcome:
    """One leaf of the exact outcome space."""

    label: str
    probability: float
    cost: float
    minutes: float


@dataclass(frozen=True)
class ExpectedCost:
    displayed_fare: float
    expected_cost: float
    expected_minutes: float
    p_success: float
    p_abandon: float
    expected_attempts: float
    expected_wasted_min: float
    cost_p10: float
    cost_p50: float
    cost_p90: float
    surcharge: float                    # expected_cost - displayed_fare
    outcomes: tuple[Outcome, ...] = field(default_factory=tuple)
    fallback_label: str | None = None
    fallback_cost: float | None = None

    @property
    def substitution_share(self) -> float:
        """How much of the expected cost is actually a *different* journey.

        When an option fails every attempt the rider takes the fallback, so the
        expectation blends two outcomes. If that share is large the headline
        number stops describing the option you clicked on, and the interface
        has to say so -- otherwise an unreliable option that fails into a cheap
        bus looks like a bargain. This is the flag that prevents that.
        """
        return self.p_abandon

    @property
    def is_blended(self) -> bool:
        """True when the expected cost is materially not about this option."""
        return self.p_abandon >= 0.10

    @property
    def surcharge_pct(self) -> float:
        if self.displayed_fare <= 0:
            return 0.0
        return 100.0 * self.surcharge / self.displayed_fare

    def as_dict(self) -> dict:
        return {
            "displayed_fare": round(self.displayed_fare, 2),
            "expected_cost": round(self.expected_cost, 2),
            "surcharge": round(self.surcharge, 2),
            "surcharge_pct": round(self.surcharge_pct, 1),
            "expected_minutes": round(self.expected_minutes, 1),
            "expected_wasted_min": round(self.expected_wasted_min, 1),
            "p_success": round(self.p_success, 4),
            "p_abandon": round(self.p_abandon, 4),
            "expected_attempts": round(self.expected_attempts, 2),
            "cost_p10": round(self.cost_p10, 2),
            "cost_p50": round(self.cost_p50, 2),
            "cost_p90": round(self.cost_p90, 2),
            "is_blended": self.is_blended,
            "substitution_share": round(self.substitution_share, 4),
            "fallback_label": self.fallback_label,
            "fallback_cost": (round(self.fallback_cost, 2)
                              if self.fallback_cost is not None else None),
            "outcomes": [
                {"label": o.label, "probability": round(o.probability, 4),
                 "cost": round(o.cost, 2), "minutes": round(o.minutes, 1)}
                for o in self.outcomes
            ],
        }


def _quantile(outcomes: list[Outcome], p: float) -> float:
    """Quantile of the exact discrete cost distribution."""
    ordered = sorted(outcomes, key=lambda o: o.cost)
    cum = 0.0
    for o in ordered:
        cum += o.probability
        if cum >= p - 1e-9:
            return o.cost
    return ordered[-1].cost if ordered else 0.0


def solve(*, displayed_fare: float, p_match: float, p_accept: float, p_cancel: float,
          pickup_min: float, ride_min: float,
          fallback_cost: float | None = None, fallback_min: float | None = None,
          fallback_label: str | None = None,
          params: LifecycleParams | None = None) -> ExpectedCost:
    """Exact expected cost and time over the booking lifecycle.

    Degenerates correctly: a scheduled service with p_match = p_accept = 1 and
    p_cancel = 0 returns its fare and its timetable, with a point distribution
    and zero surcharge. That is what makes the metro row on the comparison
    honestly read "very low uncertainty" rather than merely optimistic.
    """
    p = params or LifecycleParams()
    p_match = min(max(p_match, 0.0), 1.0)
    p_accept = min(max(p_accept, 0.0), 1.0)
    p_cancel = min(max(p_cancel, 0.0), 1.0)
    q = p_match * p_accept * (1.0 - p_cancel)
    K = max(1, p.max_attempts)

    # Time lost by one failed attempt, weighted by how it failed. A cancellation
    # after acceptance is far more expensive than never being matched, which is
    # exactly the distinction a single "cancellation rate" throws away.
    fail_mass = 1.0 - q
    if fail_mass > 1e-9:
        waste = (
            (1.0 - p_match) * p.search_timeout_min
            + p_match * (1.0 - p_accept) * p.match_min
            + p_match * p_accept * p_cancel * (p.match_min + pickup_min * p.cancel_discovery_frac)
        ) / fail_mass
    else:
        waste = 0.0

    if fallback_cost is None:
        # No alternative supplied: assume the rider eventually pays the escalated
        # fare anyway. Conservative -- it never flatters an unreliable option.
        fallback_cost = displayed_fare * (1.0 + p.surge_per_retry) ** K
    if fallback_min is None:
        fallback_min = pickup_min + ride_min

    outcomes: list[Outcome] = []
    for k in range(1, K + 1):
        prob = ((1.0 - q) ** (k - 1)) * q
        if prob <= 0.0:
            continue
        fare_k = displayed_fare * (1.0 + p.surge_per_retry) ** (k - 1)
        cost_k = fare_k + p.cancellation_fee * (k - 1)
        minutes_k = (k - 1) * waste + pickup_min + ride_min
        cost_k += p.value_of_time_per_min * ((k - 1) * waste)
        outcomes.append(Outcome(
            label=(f"ride on attempt {k}" if k > 1 else "ride on the first request"),
            probability=prob, cost=cost_k, minutes=minutes_k))

    p_abandon = (1.0 - q) ** K
    if p_abandon > 0.0:
        outcomes.append(Outcome(
            label=f"gave up after {K} tries — took {fallback_label or 'the fallback'}",
            probability=p_abandon,
            cost=fallback_cost + p.value_of_time_per_min * (K * waste),
            minutes=K * waste + fallback_min))

    total_p = sum(o.probability for o in outcomes) or 1.0
    expected_cost = sum(o.probability * o.cost for o in outcomes) / total_p
    expected_minutes = sum(o.probability * o.minutes for o in outcomes) / total_p

    # Expected number of failed attempts: attempt j is reached with (1-q)^(j-1)
    # and fails with (1-q), so the failures sum to a truncated geometric series.
    expected_failures = sum((1.0 - q) ** j for j in range(1, K + 1))

    return ExpectedCost(
        displayed_fare=displayed_fare,
        expected_cost=expected_cost,
        expected_minutes=expected_minutes,
        p_success=1.0 - p_abandon,
        p_abandon=p_abandon,
        expected_attempts=1.0 + expected_failures,
        expected_wasted_min=expected_failures * waste,
        cost_p10=_quantile(outcomes, 0.10),
        cost_p50=_quantile(outcomes, 0.50),
        cost_p90=_quantile(outcomes, 0.90),
        surcharge=expected_cost - displayed_fare,
        outcomes=tuple(outcomes),
        fallback_label=fallback_label,
        fallback_cost=fallback_cost,
    )
