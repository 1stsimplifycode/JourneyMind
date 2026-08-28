"""Turning a path through the graph into a journey a person can read.

A path is a list of edges. A journey is a list of *legs*: "walk 4 minutes",
"metro 6 stops", "Rapido to the door". Consecutive edges that a traveller would
experience as one continuous action are collapsed into one leg.

Fares are applied per *fare unit*, not per leg, because that is how operators
actually charge:

  metro   one ticket for the whole metro portion, priced on total metro
          distance -- an interchange between two lines is not a second fare
  bus     one fare per boarding
  ride    one fare per hailed vehicle
  walk    free

Every number a leg carries is tagged with where it came from: published,
estimated or predicted.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..graph.builder import MODE_DISCOMFORT
from ..models.fares import FareEstimate, FareEstimator
from .costs import CostTable

# Modes whose fare is charged once for the whole journey rather than per leg.
NETWORK_FARE_MODES = ("metro",)

# A walk shorter than this is not a leg a person would describe. It appears
# when the user's coordinates sit on top of a graph node, and if it is kept it
# makes three identical trips look like three different ones.
NEGLIGIBLE_WALK_KM = 0.05
NEGLIGIBLE_WALK_MIN = 0.8


@dataclass
class Leg:
    index: int
    mode: str
    kind: str                      # access | road | transit | transfer | ride
    from_node: str
    from_name: str
    to_node: str
    to_name: str
    distance_km: float
    travel_min: float
    wait_min: float
    route_id: str | None
    route_name: str | None
    route_colour: str | None
    stops: int
    fare: FareEstimate | None
    geometry: list[tuple[float, float]]
    time_provenance: str = "predicted"
    #: Walking folded into this leg -- reaching the vehicle, or leaving it at
    #: the far end. Counted in the time and the distance, never priced, and
    #: never shown as a leg of its own: walking is not a mode this product
    #: recommends, but you still cannot reach a metro platform without covering
    #: the last fifty metres on foot.
    access_min: float = 0.0
    access_km: float = 0.0
    #: For a transit leg that continues on a different service: one leg for the
    #: rider, several boardings underneath. Empty for a single-service leg.
    segments: list[dict] = field(default_factory=list)
    #: Where you actually get on and off the vehicle.
    #:
    #: Absorbing the walk moves `from_node` back to where the walk began, which
    #: keeps the route continuous and the map unbroken -- but it also made the
    #: metro leg claim it started at "Sarjapur Road stop 10", a bus stop. You
    #: do not board a train at a bus stop. These carry the vehicle's own
    #: endpoints so the itinerary can name them.
    board_name: str = ""
    alight_name: str = ""

    @property
    def total_min(self) -> float:
        return self.travel_min + self.wait_min + self.access_min

    @property
    def total_km(self) -> float:
        return self.distance_km + self.access_km

    @property
    def interchanges(self) -> int:
        return max(0, len(self.segments) - 1)


@dataclass
class Journey:
    journey_id: str
    legs: list[Leg]
    total_min: float
    total_cost: FareEstimate
    transfers: int
    modes: list[str]
    distance_km: float
    walk_min: float
    wait_min: float
    discomfort: float
    reliability: float
    blend: str = ""
    signature: tuple = field(default_factory=tuple)
    score: float | None = None
    score_parts: dict = field(default_factory=dict)
    #: Non-fatal notes from the logic validator, repeated to the rider rather
    #: than acted on by the engine.
    warnings: list[str] = field(default_factory=list)

    @property
    def cost(self) -> float:
        return self.total_cost.amount

    def shape(self) -> list[str]:
        """The journey as a rider would say it out loud.

        Consecutive legs of the same mode are one step, because a line change
        is not a change of transport: "Metro -> Metro" describes an interchange
        at Rashtreeya Vidyalaya Road as though it were two separate trains, and
        reads as a bug even though the routing is right. The legs keep their own
        route names; only this summary collapses them.
        """
        out: list[str] = []
        for lg in self.legs:
            if not out or out[-1] != lg.mode:
                out.append(lg.mode)
        return out

    def interchange_indices(self) -> list[int]:
        """Legs that continue the previous mode on a different service."""
        return [i for i, (a, b) in enumerate(zip(self.legs, self.legs[1:]), start=1)
                if a.mode == b.mode]

    def mode_summary(self) -> str:
        seen, out = set(), []
        for leg in self.legs:
            if leg.mode == "walk":
                continue
            if leg.mode not in seen:
                seen.add(leg.mode)
                out.append(leg.mode)
        return " + ".join(out) if out else "walk"


def _node_name(graph, node_id: str) -> str:
    n = graph.nodes.get(node_id)
    return n.name if n else node_id


def _node_ll(graph, node_id: str) -> tuple[float, float]:
    n = graph.nodes.get(node_id)
    return (n.lat, n.lon) if n else (0.0, 0.0)


def _groupable(prev, cur) -> bool:
    """Would a traveller experience these two edges as one continuous action?

    Two metro edges on DIFFERENT lines now merge. Changing from the Yellow to
    the Green line at Rashtreeya Vidyalaya Road is a real interchange, but it
    is one metro journey, and splitting it produced "Metro -> Metro" in every
    summary -- an interchange described as two separate trains. The change is
    kept as a `segment` inside the leg, so the line names and the per-boarding
    fares survive.
    """
    if prev.mode != cur.mode:
        return False
    if prev.kind == "transit" or cur.kind == "transit":
        return prev.kind == cur.kind
    if prev.kind == "ride" or cur.kind == "ride":
        return False                        # each hailed vehicle is its own leg
    return True                             # walking of any kind merges


def build_journey(graph, costs: CostTable, path: tuple[int, ...], fares: FareEstimator,
                  journey_id: str, blend: str = "") -> Journey:
    """Assemble, time and price one candidate path."""
    # -- 1. walk the path, accumulating elapsed time and grouping into legs --
    groups: list[list[tuple[int, float, float]]] = []   # (edge_i, travel, wait)
    t = 0.0
    cur_route: str | None = None
    for edge_i in path:
        e = graph.edges[edge_i]
        travel = costs.travel_min(edge_i, t)
        wait = costs.wait_min(e, edge_i, cur_route, t)
        if groups and _groupable(graph.edges[groups[-1][-1][0]], e):
            groups[-1].append((edge_i, travel, wait))
        else:
            groups.append([(edge_i, travel, wait)])
        t += travel + wait
        if e.kind == "transit":
            cur_route = e.route_id
        elif e.kind == "ride":
            cur_route = None

    # -- 2. materialise legs -------------------------------------------------
    legs: list[Leg] = []
    for gi, group in enumerate(groups):
        first = graph.edges[group[0][0]]
        last = graph.edges[group[-1][0]]
        dist = sum(graph.edges[i].distance_km for i, _, _ in group)
        travel = sum(tv for _, tv, _ in group)
        wait = sum(w for _, _, w in group)
        geom = [_node_ll(graph, graph.edges[group[0][0]].u)]
        geom += [_node_ll(graph, graph.edges[i].v) for i, _, _ in group]
        # One leg for the rider, one segment per service underneath it. Bus
        # fares are charged per boarding, so the split has to survive even
        # though the summary no longer shows it.
        segments: list[dict] = []
        if first.kind == "transit":
            for i, tv, w in group:
                e = graph.edges[i]
                if segments and segments[-1]["route_id"] == e.route_id:
                    seg = segments[-1]
                    seg["distance_km"] += e.distance_km
                    seg["minutes"] += tv + w
                    seg["stops"] += 1
                    seg["to_name"] = _node_name(graph, e.v)
                else:
                    segments.append({
                        "route_id": e.route_id, "route_name": e.route_name,
                        "route_colour": e.route_colour,
                        "from_name": _node_name(graph, e.u),
                        "to_name": _node_name(graph, e.v),
                        "distance_km": e.distance_km, "minutes": tv + w,
                        "stops": 1})

        legs.append(Leg(
            index=gi, mode=first.mode, kind=first.kind,
            from_node=first.u, from_name=_node_name(graph, first.u),
            to_node=last.v, to_name=_node_name(graph, last.v),
            distance_km=round(dist, 4), travel_min=round(travel, 3),
            wait_min=round(wait, 3), route_id=first.route_id,
            route_name=first.route_name, route_colour=first.route_colour,
            stops=len(group) if first.kind == "transit" else 0,
            fare=None, geometry=geom,
            time_provenance="predicted" if first.kind != "access" else "estimated",
            segments=segments,
        ))

    legs, groups = _drop_negligible_walks(legs, groups)

    # -- 3. shape the legs a rider would recognise ---------------------------
    # Walking folds into the leg it serves, then two services of one mode
    # become one leg with two boardings inside it. Both happen BEFORE pricing,
    # so a fare is charged per boarding and never charged for a walk.
    legs, groups, walk_min = _absorb_walks(legs, groups)
    legs, groups = _merge_same_mode_transit(legs, groups)

    # -- 4. price, per fare unit --------------------------------------------
    estimates: list[FareEstimate] = []
    network_totals: dict[str, list[float]] = {}
    for leg in legs:
        if leg.mode in NETWORK_FARE_MODES:
            acc = network_totals.setdefault(leg.mode, [0.0, 0.0])
            acc[0] += leg.distance_km
            acc[1] += leg.travel_min + leg.wait_min
            continue
        if not fares.has(leg.mode):
            continue
        if len(leg.segments) > 1:
            # A bus fare is charged per boarding. Merging two services into one
            # leg for display must not merge them into one ticket.
            parts = [fares.leg_fare(leg.mode, sg["distance_km"], sg["minutes"])
                     for sg in leg.segments]
            est = fares.combine(parts)
            est = FareEstimate(est.amount, est.low, est.high, parts[0].provenance,
                               parts[0].label,
                               f"{len(parts)} boardings, charged separately.",
                               parts[0].source)
        else:
            est = fares.leg_fare(leg.mode, leg.distance_km,
                                 leg.travel_min + leg.wait_min)
        leg.fare = est
        if est.amount > 0 or est.provenance != "exact":
            estimates.append(est)

    for mode, (dist, mins) in network_totals.items():
        est = fares.leg_fare(mode, dist, mins)
        estimates.append(est)
        # attribute the single ticket to the first leg of that mode and mark
        # the rest as covered by it, so the UI never double-counts
        first = True
        for leg in legs:
            if leg.mode != mode:
                continue
            leg.fare = est if first else FareEstimate(
                0.0, 0.0, 0.0, est.provenance, est.label,
                "Covered by the same ticket as the earlier leg of this mode.")
            first = False

    total_cost = fares.combine(estimates)

    # -- 5. journey-level summary -------------------------------------------
    boardings = sum(len(lg.segments) if lg.segments else 1
                    for lg in legs if lg.kind in ("transit", "ride"))
    transfers = max(0, boardings - 1)
    total_min = sum(lg.total_min for lg in legs)
    wait_min = sum(lg.wait_min for lg in legs)
    dist = sum(lg.total_km for lg in legs)

    discomfort = (
        sum(MODE_DISCOMFORT.get(lg.mode, 0.5) * lg.total_min for lg in legs)
        / max(total_min, 1e-6)
    )
    reliability = 1.0
    for grp in groups:
        reliability = min(reliability,
                          min(graph.edges[i].reliability for i, _, _ in grp))

    modes: list[str] = []
    for lg in legs:
        if lg.mode not in modes:
            modes.append(lg.mode)

    return Journey(
        journey_id=journey_id, legs=legs, total_min=round(total_min, 2),
        total_cost=total_cost, transfers=transfers, modes=modes,
        distance_km=round(dist, 3), walk_min=round(walk_min, 2),
        wait_min=round(wait_min, 2), discomfort=round(discomfort, 4),
        reliability=round(reliability, 3), blend=blend,
        signature=journey_signature(legs),
    )


def _absorb_walks(legs: list[Leg], groups: list) -> tuple[list[Leg], list, float]:
    """Fold every walking leg into the vehicle leg it serves.

    Walking is not a mode JourneyMind recommends. It is also unavoidable: you
    reach a metro platform on foot whether or not anybody calls it a leg. So
    the minutes and the metres are kept -- inside `access_min` / `access_km` on
    the neighbouring vehicle leg, counted in the journey total and never priced
    -- and the walk stops being a step in the itinerary.

    A journey made mostly of walking is not saved by this. It keeps its walking
    total, and `routing/validate` rejects it: if the rider has to walk half an
    hour, the honest answer is a first-mile ride, not a relabelled hike.

    Returns (legs, groups, walk_min). A walk-only journey is returned untouched
    -- there is nothing to absorb it into, and the validator will reject it.
    """
    walk_min = sum(lg.total_min for lg in legs if lg.mode == "walk")
    vehicles = [lg for lg in legs if lg.mode != "walk"]
    if not vehicles:
        return legs, groups, walk_min

    kept: list[tuple[Leg, list]] = []
    pending_min = pending_km = 0.0
    pending_geom: list = []
    pending_from: tuple[str, str] | None = None

    for leg, grp in zip(legs, groups):
        if leg.mode == "walk":
            if pending_from is None:
                pending_from = (leg.from_node, leg.from_name)
            pending_min += leg.total_min
            pending_km += leg.distance_km
            pending_geom.extend(leg.geometry)
            continue
        if pending_min or pending_km:
            leg.board_name = leg.board_name or leg.from_name
            leg.access_min = round(leg.access_min + pending_min, 3)
            leg.access_km = round(leg.access_km + pending_km, 4)
            # the leg now starts where the walk started, so the drawn route and
            # the continuity check both stay unbroken
            leg.from_node, leg.from_name = pending_from
            leg.geometry[:0] = pending_geom[:-1]
            pending_min = pending_km = 0.0
            pending_geom, pending_from = [], None
        kept.append((leg, grp))

    if pending_min or pending_km:               # a walk at the very end
        last, _ = kept[-1]
        last.access_min = round(last.access_min + pending_min, 3)
        last.access_km = round(last.access_km + pending_km, 4)
        last.geometry.extend(pending_geom[1:])
        last.alight_name = last.alight_name or last.to_name
        # the tail walk ends at the destination, and so now does this leg
        last.to_node, last.to_name = legs[-1].to_node, legs[-1].to_name

    kept_legs = [lg for lg, _ in kept]
    for i, lg in enumerate(kept_legs):
        lg.index = i
    return kept_legs, [g for _, g in kept], walk_min


def _merge_same_mode_transit(legs: list[Leg], groups: list) -> tuple[list[Leg], list]:
    """Two bus rides with a walk between them are one bus leg, two boardings.

    Absorbing the walk (above) leaves the two services sitting next to each
    other, and "Bus -> Bus" is not how anybody describes changing buses at
    Kanakapura Road. They merge into one leg whose `segments` keep both routes,
    both fares and the interchange -- exactly what already happens when the
    change is on the same platform.

    Only TRANSIT merges. Two hailed vehicles in a row is not an interchange, it
    is a journey that should have stayed in the first vehicle, and the
    validator rejects it.
    """
    if len(legs) < 2:
        return legs, groups

    kept: list[tuple[Leg, list]] = []
    for leg, grp in zip(legs, groups):
        if kept:
            prev, prev_grp = kept[-1]
            if prev.kind == "transit" and leg.kind == "transit" and prev.mode == leg.mode:
                prev.segments = (prev.segments or []) + (leg.segments or [])
                prev.travel_min = round(prev.travel_min + leg.travel_min, 3)
                prev.wait_min = round(prev.wait_min + leg.wait_min, 3)
                prev.access_min = round(prev.access_min + leg.access_min, 3)
                prev.access_km = round(prev.access_km + leg.access_km, 4)
                prev.distance_km = round(prev.distance_km + leg.distance_km, 4)
                prev.stops += leg.stops
                prev.to_node, prev.to_name = leg.to_node, leg.to_name
                prev.alight_name = leg.alight_name or leg.to_name
                prev.geometry.extend(leg.geometry[1:])
                kept[-1] = (prev, prev_grp + grp)
                continue
        kept.append((leg, grp))

    kept_legs = [lg for lg, _ in kept]
    for i, lg in enumerate(kept_legs):
        lg.index = i
    return kept_legs, [g for _, g in kept]


def _drop_negligible_walks(legs: list[Leg], groups: list) -> tuple[list[Leg], list]:
    """Remove zero-length access walks, stitching their geometry into the
    neighbouring leg so the drawn route stays continuous.

    These appear whenever the user's coordinates sit on top of a graph node.
    Left in, they make one trip look like three different ones -- "walk 0 m,
    then Rapido" and "Rapido, then walk 0 m" would get separate signatures.
    """
    if len(legs) <= 1:
        return legs, groups

    keep: list[tuple[Leg, list]] = []
    carried: list = []          # geometry of a dropped leading walk
    for leg, grp in zip(legs, groups):
        if (leg.mode == "walk"
                and leg.distance_km < NEGLIGIBLE_WALK_KM
                and leg.total_min < NEGLIGIBLE_WALK_MIN):
            if keep:
                prev = keep[-1][0]
                prev.geometry.extend(leg.geometry[1:])
                prev.to_node, prev.to_name = leg.to_node, leg.to_name
            else:
                carried = leg.geometry[:-1]
            continue
        if carried:
            leg.geometry[:0] = carried
            carried = []
        keep.append((leg, grp))

    if not keep:                            # never return an empty journey
        return legs, groups
    kept_legs = [lg for lg, _ in keep]
    for i, lg in enumerate(kept_legs):
        lg.index = i
    return kept_legs, [g for _, g in keep]


def journey_signature(legs: list[Leg]) -> tuple:
    """What makes two journeys 'the same trip' to a human.

    Deliberately coarse: the sequence of (mode, route) with walking collapsed.
    Two paths that differ only by which back street the walk used are the same
    journey and only one should be shown.
    """
    sig: list[tuple[str, str]] = []
    for lg in legs:
        if lg.mode == "walk":
            if sig and sig[-1][0] == "walk":
                continue
            sig.append(("walk", ""))
        else:
            sig.append((lg.mode, lg.route_id or ""))
    return tuple(sig)


def deduplicate(journeys: list[Journey]) -> list[Journey]:
    """Collapse trivial variants, keeping the fastest of each signature."""
    best: dict[tuple, Journey] = {}
    for j in journeys:
        cur = best.get(j.signature)
        if cur is None or (j.total_min, j.cost) < (cur.total_min, cur.cost):
            best[j.signature] = j
    return sorted(best.values(), key=lambda j: (j.total_min, j.cost))
