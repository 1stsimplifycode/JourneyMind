"""Stage 2: drop dominated journeys.

If journey B is both cheaper than A *and* faster than A, then A is dominated
and is removed -- regardless of any weighting. No user preference can make A
the right answer, because B beats it on everything the user was asked about.

This is what stops the system from ever recommending something silly. It runs
before scoring, not after.

THE FRONTIER MUST USE EVERY OBJECTIVE THE SCORE USES
----------------------------------------------------
This module previously ran on (cost, time) alone, on the grounds that adding
transfers and comfort "would make almost everything non-dominated and the
filter would stop doing any work". Measured on this study area, that turned out
to be false, and the cost of believing it was severe:

    frontier axes          mean size   ride modes ever reachable
    cost, time                   4.4   rapido
    + comfort                    7.0   rapido, auto, namma_yatri, cab
    + comfort + transfers        8.0   rapido, auto, namma_yatri, cab
    (60 origin-destination pairs, budget 600, limit 180 min)

A bike-taxi is cheaper AND faster than an auto, a Namma Yatri and a cab, so on
a two-axis frontier it dominates all three -- every time, on every route. Those
three modes were deleted before the comfort weight was ever applied, which made
two of the four objectives in `scoring.score()` incapable of changing the
answer. A rider asking for maximum comfort was still handed a bike-taxi.

So dominance now runs over all four scored objectives. The filter still does
real work -- it removes roughly a third of the feasible set -- and a mode that
loses on price and speed can now survive on comfort and be ranked on it.

This is a deliberate departure from v1 section 15, which specifies (cost, time).
The departure is recorded rather than hidden: the documentation's own scoring
formula has four terms, and a frontier that pre-filters on two of them makes the
other two decorative.
"""

from __future__ import annotations

from ..routing.journey import Journey

COST_EPS = 0.5      # rupees: below this two fares are "the same price"
TIME_EPS = 0.5      # minutes
COMFORT_EPS = 0.02  # discomfort is 0..1; below this two rides feel the same
TRANSFER_EPS = 0    # a change is a change


def _axes(j: Journey) -> tuple[float, float, float, float]:
    """The four objectives, all oriented so that lower is better -- the same
    orientation and the same four quantities that `scoring.score()` weighs."""
    return (j.cost, j.total_min, float(j.transfers), j.discomfort)


_EPS = (COST_EPS, TIME_EPS, TRANSFER_EPS, COMFORT_EPS)


def dominates(a: Journey, b: Journey) -> bool:
    """Does `a` dominate `b`? No worse on every objective, better on at least one.

    "Every objective" means all four that the ranking stage weighs. A journey
    that is dearer and slower but genuinely more comfortable is NOT dominated,
    because a rider who cares about comfort could rationally choose it.
    """
    av, bv = _axes(a), _axes(b)
    no_worse = all(x <= y + e for x, y, e in zip(av, bv, _EPS))
    strictly_better = any(x < y - e for x, y, e in zip(av, bv, _EPS))
    return no_worse and strictly_better


def frontier(journeys: list[Journey]) -> list[Journey]:
    """The non-dominated set over all four scored objectives, cheapest first."""
    kept: list[Journey] = []
    for j in journeys:
        if any(dominates(other, j) for other in journeys if other is not j):
            continue
        kept.append(j)

    # Near-twins can tie on every axis (a metro variant and its mirror). Collapse
    # them, keyed on all four objectives so that a genuinely different option --
    # a cab at the same price and time as a bike-taxi -- is never silently
    # dropped for being in the same cost/time cell.
    best: dict[tuple, Journey] = {}
    for j in kept:
        cell = (round(j.cost / max(COST_EPS, 1e-6)),
                round(j.total_min / max(TIME_EPS, 1e-6)),
                j.transfers,
                round(j.discomfort / max(COMFORT_EPS, 1e-6)))
        if cell not in best:
            best[cell] = j
    return sorted(best.values(), key=lambda j: (j.cost, j.total_min))


def dominated_by(journeys: list[Journey]) -> dict[str, list[str]]:
    """Diagnostics: which journey knocked each one out. Used by the API's
    pipeline trace so the filtering is inspectable rather than magic."""
    out: dict[str, list[str]] = {}
    for j in journeys:
        killers = [o.journey_id for o in journeys if o is not j and dominates(o, j)]
        if killers:
            out[j.journey_id] = killers
    return out
