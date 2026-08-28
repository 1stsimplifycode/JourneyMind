"""Time-dependent edge costs.

Time of day changes edge weights *during* a journey: a hop entered at 09:14 is
not the hop the departure-time prediction described. The documentation calls
for time-dependent Dijkstra for exactly this reason.

Doing that continuously would mean re-running the model at every relaxation.
Instead the model is run once per elapsed-time bucket (0, 15, 30, 45, 60, 90
minutes after departure) and an edge entered at elapsed time t is priced from
the bucket containing t. That is an approximation -- piecewise-constant in
elapsed time -- and it is described as one rather than sold as exact.

The cost model also carries a search-time *money* weight. Real fares are
slab-based and per-boarding, so they cannot be decomposed per edge exactly;
`marginal_cost` is a deliberate approximation used only to steer the search
towards cheap candidates. Every journey that survives is then priced exactly by
`FareEstimator` during assembly.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..graph.builder import MODE_DISCOMFORT
from ..graph.features import TimeContext
from ..models.base import predict_edge_minutes

BUCKET_STARTS: tuple[float, ...] = (0.0, 15.0, 30.0, 45.0, 60.0, 90.0)

# Approximate per-km money rates used only to bias path search. Exact fares
# come from the published/estimated fare models during journey assembly.
SEARCH_RATE_PER_KM = {"metro": 3.2, "bus": 1.9, "walk": 0.0}

#: A SEARCH-ONLY shadow price on walking, in rupees per minute. Nobody is ever
#: charged this and no fare anywhere includes it.
#:
#: It exists because walking is free and the cheapest-blend search will
#: therefore choose any amount of it: the planner offered a bike taxi to the
#: metro followed by a SEVENTY-SEVEN MINUTE WALK to the door, which is cheap,
#: is what the maths asked for, and is not a commute. Rejecting it afterwards
#: only lost the candidate. Pricing walking inside the search makes the same
#: optimiser reach for a last-mile ride instead, which is the answer a person
#: would give.
#:
#: Calibrated against the alternative, not invented: a short station approach
#: (~5 min, Rs 15 of shadow) stays cheaper than hailing anything, while half an
#: hour on foot (Rs 90) costs more than the ride that would replace it.
WALK_SHADOW_PER_MIN = 3.0


@dataclass
class CostTable:
    """Per-edge travel minutes at each elapsed-time bucket, plus a static
    money approximation and the diagnostics from the prediction pass."""

    travel: np.ndarray          # [n_buckets, n_edges] minutes
    money: np.ndarray           # [n_edges] approximate rupees
    boarding_wait: np.ndarray   # [n_buckets, n_edges] minutes charged on boarding
    ride_wait: np.ndarray       # [n_edges] pickup wait for ride edges
    diagnostics: dict

    def bucket_for(self, elapsed_min: float) -> int:
        b = 0
        for i, start in enumerate(BUCKET_STARTS):
            if elapsed_min >= start:
                b = i
        return b

    def travel_min(self, edge_i: int, elapsed_min: float) -> float:
        return float(self.travel[self.bucket_for(elapsed_min), edge_i])

    def wait_min(self, edge, edge_i: int, prev_route: str | None,
                 elapsed_min: float) -> float:
        """Waiting is charged when you board, not at every stop."""
        if edge.kind == "transit":
            if prev_route == edge.route_id:
                return 0.0
            return float(self.boarding_wait[self.bucket_for(elapsed_min), edge_i])
        if edge.kind == "ride":
            return float(self.ride_wait[edge_i])
        return 0.0

    def total_min(self, edge, edge_i: int, prev_route: str | None,
                  elapsed_min: float) -> float:
        return (self.travel_min(edge_i, elapsed_min)
                + self.wait_min(edge, edge_i, prev_route, elapsed_min))


def build_cost_table(request_graph, base_graph, predictor, ctx: TimeContext) -> CostTable:
    n = len(request_graph.edges)
    travel = np.zeros((len(BUCKET_STARTS), n), dtype=np.float32)
    wait = np.zeros((len(BUCKET_STARTS), n), dtype=np.float32)
    diagnostics: dict = {}

    for b, offset in enumerate(BUCKET_STARTS):
        shifted = ctx.shifted(offset)
        minutes, diag = predict_edge_minutes(request_graph, base_graph, predictor, shifted)
        travel[b] = minutes
        if b == 0:
            diagnostics = diag
        for i, e in enumerate(request_graph.edges):
            if e.kind == "transit":
                # Half the headway while the route runs; the wait until the
                # first departure when it does not. A scheduling assumption,
                # not an observation.
                wait[b, i] = base_graph.boarding_wait_for(e.route_id, shifted)

    money = np.zeros(n, dtype=np.float32)
    ride_wait = np.zeros(n, dtype=np.float32)
    fares = base_graph.fares
    for i, e in enumerate(request_graph.edges):
        if e.kind == "ride":
            ride_wait[i] = e.wait_min
            m = fares.get(e.mode)
            if m is not None:
                extra = max(0.0, e.distance_km - m.base_distance_km)
                money[i] = m.base_fare + extra * m.per_km + e.base_min * m.per_min
        elif e.kind == "transit":
            money[i] = e.distance_km * SEARCH_RATE_PER_KM.get(e.mode, 2.0)
        elif e.mode == "walk":
            # Search-only. See WALK_SHADOW_PER_MIN -- the fare is still zero.
            money[i] = (e.walk_min or e.base_min) * WALK_SHADOW_PER_MIN

    in_service = base_graph.routes_in_service(ctx)
    diagnostics["buckets_min"] = list(BUCKET_STARTS)
    diagnostics["hour_local"] = round(ctx.hour, 2)
    diagnostics["routes_out_of_service"] = sorted(
        base_graph.routes[rid].name for rid, ok in in_service.items() if not ok)
    return CostTable(travel=travel, money=money, boarding_wait=wait,
                     ride_wait=ride_wait, diagnostics=diagnostics)


def discomfort_of(mode: str) -> float:
    return MODE_DISCOMFORT.get(mode, 0.5)
