"""Stage 1 of the optimiser: throw out what is impossible.

Any journey costing more than the budget is deleted. Any journey longer than
the deadline is deleted. No arguing, no weighting -- a journey you cannot pay
for is not a cheap journey, it is not a journey.

The one subtlety is *which* cost is compared against the budget. Ride-hailing
fares are estimates with a band. Comparing the point estimate against the
budget would quietly recommend trips that are more likely than not to come in
over budget. The comparison therefore uses the point estimate but records the
band, and a journey whose upper bound exceeds the budget is flagged `at_risk`
so the UI can say so.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..routing.journey import Journey


@dataclass(frozen=True)
class ConstraintStatus:
    within_budget: bool
    within_time: bool
    budget_headroom: float        # rupees left over (negative when over)
    time_headroom: float          # minutes left over (negative when over)
    cost_at_risk: bool            # point estimate fits, upper band does not
    reasons: tuple[str, ...] = ()

    @property
    def feasible(self) -> bool:
        return self.within_budget and self.within_time

    def as_dict(self) -> dict:
        return {
            "feasible": self.feasible,
            "within_budget": self.within_budget,
            "within_time": self.within_time,
            "budget_headroom": round(self.budget_headroom, 2),
            "time_headroom": round(self.time_headroom, 2),
            "cost_at_risk": self.cost_at_risk,
            "reasons": list(self.reasons),
        }


def evaluate(journey: Journey, budget: float, max_time_min: float) -> ConstraintStatus:
    cost = journey.cost
    within_budget = cost <= budget + 1e-9
    within_time = journey.total_min <= max_time_min + 1e-9
    at_risk = within_budget and journey.total_cost.high > budget + 1e-9

    reasons: list[str] = []
    if not within_budget:
        reasons.append(f"₹{cost - budget:.0f} over your budget")
    if not within_time:
        reasons.append(f"{journey.total_min - max_time_min:.0f} min over your time limit")
    if at_risk:
        reasons.append("fits on the estimate, but the upper end of the fare range does not")

    return ConstraintStatus(
        within_budget=within_budget, within_time=within_time,
        budget_headroom=budget - cost,
        time_headroom=max_time_min - journey.total_min,
        cost_at_risk=at_risk, reasons=tuple(reasons),
    )


def partition(journeys: list[Journey], budget: float, max_time_min: float
              ) -> tuple[list[tuple[Journey, ConstraintStatus]],
                         list[tuple[Journey, ConstraintStatus]]]:
    """Split into (feasible, infeasible), each paired with its status."""
    feasible, infeasible = [], []
    for j in journeys:
        st = evaluate(j, budget, max_time_min)
        (feasible if st.feasible else infeasible).append((j, st))
    return feasible, infeasible


def near_miss_alternatives(infeasible: list[tuple[Journey, ConstraintStatus]]
                           ) -> list[dict]:
    """When nothing fits both limits, offer the honest next-best options --
    each clearly labelled with the constraint it breaks.

    Never silently returns an invalid route as if it were valid.
    """
    if not infeasible:
        return []
    out: list[dict] = []

    def add(label: str, why: str, pair):
        j, st = pair
        if any(o["journey"].journey_id == j.journey_id for o in out):
            return
        out.append({"label": label, "why": why, "journey": j, "status": st})

    under_budget = [p for p in infeasible if p[1].within_budget]
    if under_budget:
        add("Closest under budget", "Fits your budget but not your time limit.",
            min(under_budget, key=lambda p: p[0].total_min))

    add("Fastest available", "The quickest option we found, whatever it costs.",
        min(infeasible, key=lambda p: p[0].total_min))
    add("Cheapest available", "The least expensive option we found, however long it takes.",
        min(infeasible, key=lambda p: p[0].cost))
    return out[:3]
