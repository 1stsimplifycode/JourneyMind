"""Why this journey, in a sentence a person would actually say.

Every sentence here is generated from the journey's own attributes compared
against the alternatives it beat. There is no language model involved and no
network call: the same inputs always produce the same words, which is what you
want from an explanation that a user is going to trust.

The interesting comparisons are the ones a person would make themselves:

  * against the fastest option that was available  -> "₹15 cheaper, 3 min slower"
  * against the cheapest option that was available -> "₹20 more, but 22 min sooner"
  * against the constraints the user typed          -> "fits your ₹100 budget"
  * against the single-mode options                -> "beats any one ride on its own"

A comparison is only mentioned when it is actually true and actually
interesting. Saying "₹0 cheaper than the alternative" is worse than saying
nothing.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..optimisation.constraints import ConstraintStatus
from ..routing.journey import Journey

RUPEE = "₹"

MODE_WORDS = {
    "walk": "walking", "metro": "the metro", "bus": "the bus",
    "bike_taxi": "a bike taxi", "auto": "an auto",
    "cab": "a cab",
}


@dataclass
class Explanation:
    headline: str
    reasons: list[str]
    comparisons: list[str]
    caveats: list[str]

    def as_dict(self) -> dict:
        return {"headline": self.headline, "reasons": self.reasons,
                "comparisons": self.comparisons, "caveats": self.caveats}


def _money(x: float) -> str:
    return f"{RUPEE}{abs(x):.0f}"


def _mins(x: float) -> str:
    m = abs(x)
    return "1 minute" if 0.5 <= m < 1.5 else f"{m:.0f} minutes"


def _route_phrase(j: Journey) -> str:
    """'the metro then a Rapido' — the shape of the trip in words."""
    parts = [MODE_WORDS.get(m, m) for m in j.modes if m != "walk"]
    if not parts:
        return "walking the whole way"
    if len(parts) == 1:
        return parts[0]
    return ", then ".join(parts[:-1]) + ", then " + parts[-1]


def _is_multimodal(j: Journey) -> bool:
    return len({m for m in j.modes if m != "walk"}) >= 2


def explain(chosen: Journey, status: ConstraintStatus, pool: list[Journey],
            budget: float, max_time_min: float, preset: str) -> Explanation:
    """Build the explanation for the recommended journey."""
    others = [j for j in pool if j.journey_id != chosen.journey_id]
    reasons: list[str] = []
    comparisons: list[str] = []
    caveats: list[str] = []

    # --- headline ---------------------------------------------------------
    shape = _route_phrase(chosen)
    if not any(m != "walk" for m in chosen.modes):
        headline = "Walk the whole way — nothing you could pay for beats it here."
    elif _is_multimodal(chosen):
        headline = (f"Take {shape} — it fits both your limits with room to spare."
                    if status.budget_headroom > 5 and status.time_headroom > 3
                    else f"Take {shape} — the best complete trip inside your limits.")
    else:
        headline = f"Take {shape} — the best option inside your limits."

    # --- constraint reasons ----------------------------------------------
    if status.within_budget:
        if status.budget_headroom >= 1:
            reasons.append(
                f"Fits your {_money(budget)} budget with {_money(status.budget_headroom)} left over.")
        else:
            reasons.append(f"Fits your {_money(budget)} budget exactly.")
    if status.within_time:
        if status.time_headroom >= 1:
            reasons.append(
                f"Gets you there {_mins(status.time_headroom)} before your "
                f"{max_time_min:.0f}-minute limit.")
        else:
            reasons.append(f"Arrives right on your {max_time_min:.0f}-minute limit.")

    # --- comparison with the fastest option available ---------------------
    # Compared only against journeys the user could actually have taken. A
    # five-hour walk is technically the cheapest thing in the pool, and
    # "248 minutes quicker than walking" is not a comparison anyone asked for.
    viable = [j for j in others
              if j.total_min <= max_time_min and j.cost <= budget]
    if not viable:
        # Nothing else fit both limits. Still refuse to compare against a
        # journey that would have blown the deadline -- if there is no honest
        # comparison to make, make none.
        viable = [j for j in others if j.total_min <= max_time_min]
    if viable:
        fastest = min(viable, key=lambda j: j.total_min)
        if fastest.total_min < chosen.total_min - 0.5:
            saved = fastest.cost - chosen.cost
            lost = chosen.total_min - fastest.total_min
            if saved > 1:
                comparisons.append(
                    f"{_money(saved)} cheaper than the fastest option, and only "
                    f"{_mins(lost)} slower.")
        elif chosen.total_min < fastest.total_min - 0.5:
            comparisons.append("It is also the fastest option we found.")

        cheapest = min(viable, key=lambda j: j.cost)
        if cheapest.cost < chosen.cost - 1:
            extra = chosen.cost - cheapest.cost
            saved_time = cheapest.total_min - chosen.total_min
            if saved_time > 1:
                comparisons.append(
                    f"{_money(extra)} more than the cheapest option, but "
                    f"{_mins(saved_time)} quicker.")
        elif chosen.cost < cheapest.cost - 1:
            comparisons.append("It is also the cheapest option we found.")

        # transfers: the thing people quietly hate
        fewer = [j for j in viable if j.transfers < chosen.transfers]
        more = [j for j in viable if j.transfers > chosen.transfers]
        if more and not fewer:
            comparisons.append(
                "It avoids an extra change that the other options need.")

    # --- the multimodal point, when it is actually the point --------------
    # Compare only against single-mode options that would actually have got the
    # user there in time. "Mixing modes beats walking for an hour" is true and
    # useless; "mixing modes beats the Rapido you would have booked" is the
    # comparison this product exists to make.
    if _is_multimodal(chosen):
        singles = [j for j in others
                   if not _is_multimodal(j) and any(m != "walk" for m in j.modes)]
        usable = [j for j in singles if j.total_min <= max_time_min * 1.05]
        if not usable:
            usable = [j for j in singles if j.total_min <= chosen.total_min * 1.4]
        if usable:
            best_single = min(usable, key=lambda j: (j.cost, j.total_min))
            gap = best_single.cost - chosen.cost
            gap_t = best_single.total_min - chosen.total_min
            if gap > 1:
                comparisons.append(
                    f"Mixing modes saves {_money(gap)} against the best "
                    f"single-mode trip that would still get you there in time "
                    f"({_route_phrase(best_single)}).")
            elif gap_t > 1:
                comparisons.append(
                    f"Mixing modes saves {_mins(gap_t)} against the best "
                    f"single-mode trip ({_route_phrase(best_single)}).")

    # --- preference acknowledgement ---------------------------------------
    if preset == "cheapest":
        reasons.append("You asked for the cheapest option, so price was weighted heaviest.")
    elif preset == "fastest":
        reasons.append("You asked for the fastest option, so time was weighted heaviest.")
    elif preset == "custom":
        reasons.append("Ranked using the priorities you set on the sliders.")

    # --- caveats: never let an estimate pass as a fact --------------------
    if chosen.total_cost.provenance == "estimated":
        caveats.append(
            "The total includes an estimated ride-hailing fare "
            f"({chosen.total_cost.display()} range). It is not a quote and surge "
            "pricing is not modelled.")
    if status.cost_at_risk:
        caveats.append(
            "At the top of the estimated fare range this trip would go over your budget.")
    if chosen.wait_min > 6:
        caveats.append(
            f"About {_mins(chosen.wait_min)} of this is waiting, estimated from "
            "published headways rather than live vehicle positions.")
    if chosen.walk_min > 12:
        caveats.append(f"It involves about {_mins(chosen.walk_min)} of walking.")

    return Explanation(headline=headline, reasons=reasons,
                       comparisons=comparisons, caveats=caveats)


def explain_alternative(alt: Journey, status: ConstraintStatus,
                        chosen: Journey) -> str:
    """One line per alternative, always relative to the recommendation."""
    d_cost = alt.cost - chosen.cost
    d_time = alt.total_min - chosen.total_min
    bits: list[str] = []

    if d_cost < -1:
        bits.append(f"{_money(d_cost)} cheaper")
    elif d_cost > 1:
        bits.append(f"{_money(d_cost)} more")
    if d_time < -0.5:
        bits.append(f"{_mins(d_time)} faster")
    elif d_time > 0.5:
        bits.append(f"{_mins(d_time)} slower")

    if alt.transfers < chosen.transfers:
        bits.append("one fewer change")
    elif alt.transfers > chosen.transfers:
        bits.append("one more change")

    if not bits:
        head = f"A different way to travel: {_route_phrase(alt)}."
    else:
        head = f"{_route_phrase(alt).capitalize()} — " + ", ".join(bits) + "."

    if not status.feasible:
        head += " " + " ".join(f"Breaks a limit: {r}." for r in status.reasons)
    return head


def no_feasible_message(budget: float, max_time_min: float) -> str:
    return (f"No journey fits both your {_money(budget)} budget and your "
            f"{max_time_min:.0f}-minute limit.")
