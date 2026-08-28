"""The multimodal transport graph.

One graph holds four kinds of edge, which is what makes mixed-mode journeys
findable at all -- "Walk then Metro then Rapido" is not a special case anyone
coded, it is simply a path through this graph:

    road      walking along a street segment between junctions/stops
    transit   one stop to the next on a metro or bus route
    transfer  walking between two nearby transit nodes (metro exit -> bus stop)
    ride      a hailed vehicle (bike-taxi / auto / cab) between two hubs
    access    walking from the user's actual coordinates to the network

`MultimodalGraph` is the immutable, cached, city-wide part (road + transit +
transfer). `RequestGraph` overlays the per-request part (access + ride) without
mutating it, so concurrent requests cannot interfere with each other.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from functools import lru_cache

import numpy as np

from ..config import get_settings
from ..data.geo import ROAD_DETOUR, WALK_DETOUR, haversine_km
from ..data.provider import Node, TransportDataProvider
from ..data.static_provider import get_provider
from .features import (
    NODE_FEATURE_DIM, TimeContext, edge_class_of, encode_edge, encode_node,
)

ORIGIN_ID = "__origin__"
DESTINATION_ID = "__destination__"

#: The hailed VEHICLE types. A brand is not a mode: Rapido is a provider of a
#: bike taxi and Namma Yatri is a provider of an auto, so both live in
#: providers/simulated.py and neither appears here. Keeping "rapido" and
#: "namma_yatri" as modes brand-locked the router and made one vehicle appear
#: twice in every comparison.
RIDE_MODES = ("bike_taxi", "auto", "cab")
# FREE-FLOW road speed by ride mode -- the speed with the roads empty. The
# travel-time model's predicted congestion is applied on top of this, so these
# must not already have congestion baked in. A bike-taxi filters through
# stationary traffic, so it sits highest.
RIDE_FREE_SPEED_KMPH = {"bike_taxi": 34.0, "auto": 28.0, "cab": 31.0}
# Time spent waiting for the vehicle to arrive, in minutes. A modelling
# assumption, not an availability claim -- see the limitations section.
RIDE_PICKUP_WAIT_MIN = {"bike_taxi": 3.0, "auto": 4.0, "cab": 5.0}

# Crude comfort proxies in [0, 1], where 1 is least comfortable. Subjective by
# nature; documented as such rather than dressed up as measurement.
MODE_DISCOMFORT = {
    "walk": 0.85, "bus": 0.60, "metro": 0.25,
    "bike_taxi": 0.55, "auto": 0.45, "cab": 0.15,
}


@dataclass
class GraphEdge:
    """One directed hop. `base_min` is the model-free estimate; `predicted_min`
    is filled in by the travel-time model for a specific time context."""

    idx: int
    edge_id: str
    u: str
    v: str
    kind: str                      # road | transit | transfer | ride | access
    mode: str                      # walk | metro | bus | rapido | auto | ...
    distance_km: float
    base_min: float                # free-flow VEHICLE time: the model's target
    walk_min: float = 0.0          # time on foot, when this edge is walkable
    free_speed_kmph: float = 0.0
    lanes: int = 1
    route_id: str | None = None
    route_name: str | None = None
    route_colour: str | None = None
    headway_min: float = 0.0
    wait_min: float = 0.0
    predicted_min: float | None = None
    reliability: float = 0.85      # 0..1, higher is more predictable

    @property
    def travel_min(self) -> float:
        """In-vehicle / on-foot time, excluding waiting."""
        if self.predicted_min is not None:
            return self.predicted_min
        return self.walk_min if self.mode == "walk" and self.walk_min else self.base_min

    @property
    def total_min(self) -> float:
        return self.travel_min + self.wait_min

    @property
    def is_static(self) -> bool:
        """Static edges are the ones the GNN is trained on."""
        return self.kind in ("road", "transit", "transfer")


class MultimodalGraph:
    """City-wide, request-independent. Built once and cached."""

    def __init__(self, provider: TransportDataProvider):
        self.provider = provider
        self.city = provider.get_city()
        self.nodes: dict[str, Node] = provider.node_index()
        self.routes = provider.route_index()
        self.fares = provider.get_fares()
        self.places = list(provider.get_places())

        self.edges: list[GraphEdge] = []
        self.out_adj: dict[str, list[int]] = {n: [] for n in self.nodes}
        self._build_edges()

        self.node_order: list[str] = sorted(self.nodes)
        self.node_pos: dict[str, int] = {n: i for i, n in enumerate(self.node_order)}
        self.node_features = self._build_node_features()
        self.adj_index = self._build_adjacency_index()

        self.static_edge_idx = [e.idx for e in self.edges if e.is_static]
        self.edge_features = self._build_edge_features()
        self._ride_hubs = self._pick_ride_hubs()

    # -- construction ------------------------------------------------------
    def _add(self, **kw) -> GraphEdge:
        e = GraphEdge(idx=len(self.edges), **kw)
        self.edges.append(e)
        self.out_adj.setdefault(e.u, []).append(e.idx)
        return e

    def _build_edges(self) -> None:
        s = get_settings()

        # Road segments. A street has two traversal times and they are not the
        # same thing: the VEHICLE time, which is what congestion is about and
        # what the model is trained to predict, and the WALKING time, which is
        # distance over walking pace and does not care about traffic. `base_min`
        # carries the vehicle time so it matches the training observations;
        # `walk_min` is what the router charges, because in this graph a road
        # edge is something you walk along (rides are hailed hub to hub).
        for r in self.provider.get_road_edges():
            drive_min = r.distance_km / max(r.free_speed_kmph, 1.0) * 60.0
            walk_min = r.distance_km / s.walk_speed_kmph * 60.0
            for u, v in ((r.u, r.v), (r.v, r.u)):
                self._add(edge_id=r.edge_id, u=u, v=v, kind="road", mode="walk",
                          distance_km=r.distance_km, base_min=drive_min,
                          walk_min=walk_min, free_speed_kmph=r.free_speed_kmph,
                          lanes=r.lanes, reliability=0.95)

        # transit: both running directions of each route
        for t in self.provider.get_transit_edges():
            route = self.routes.get(t.route_id)
            colour = route.colour if route else "#666"
            name = route.name if route else t.route_id
            for u, v in ((t.u, t.v), (t.v, t.u)):
                self._add(edge_id=t.edge_id, u=u, v=v, kind="transit", mode=t.mode,
                          distance_km=t.distance_km, base_min=t.scheduled_min,
                          route_id=t.route_id, route_name=name, route_colour=colour,
                          reliability=0.92 if t.mode == "metro" else 0.70)

        # walking transfers between nearby transit nodes
        for tf in self.provider.get_transfer_edges():
            for u, v in ((tf.u, tf.v), (tf.v, tf.u)):
                self._add(edge_id=tf.edge_id, u=u, v=v, kind="transfer", mode="walk",
                          distance_km=tf.distance_km, base_min=tf.walk_min,
                          walk_min=tf.walk_min, reliability=0.95)

    def _build_node_features(self) -> np.ndarray:
        speeds: dict[str, list[float]] = {n: [] for n in self.nodes}
        for e in self.edges:
            if e.kind == "road" and e.free_speed_kmph:
                speeds[e.u].append(e.free_speed_kmph)
        feats = np.zeros((len(self.node_order), NODE_FEATURE_DIM), dtype=np.float32)
        for i, nid in enumerate(self.node_order):
            n = self.nodes[nid]
            adj = speeds.get(nid) or [25.0]
            feats[i] = encode_node(
                kind=n.kind, degree=n.degree or len(self.out_adj.get(nid, [])),
                observed_congestion=n.observed_congestion,
                is_interchange=n.is_interchange,
                mean_adjacent_free_speed=float(np.mean(adj)),
                lat=n.lat, lon=n.lon, bbox=self.city.bbox,
            )
        return feats

    def _build_adjacency_index(self) -> tuple[np.ndarray, np.ndarray]:
        """(src, dst) index arrays for message passing. Includes both
        directions of every static edge plus a self-loop per node."""
        src, dst = [], []
        for e in self.edges:
            if not e.is_static:
                continue
            src.append(self.node_pos[e.u])
            dst.append(self.node_pos[e.v])
        for i in range(len(self.node_order)):
            src.append(i)
            dst.append(i)
        return np.asarray(src, dtype=np.int64), np.asarray(dst, dtype=np.int64)

    def _build_edge_features(self) -> np.ndarray:
        """Static per-edge features. Time context is concatenated at predict
        time, so this matrix is built once."""
        rows = []
        for idx in self.static_edge_idx:
            e = self.edges[idx]
            cu = self.nodes[e.u].observed_congestion
            cv = self.nodes[e.v].observed_congestion
            headway = 0.0
            if e.route_id and e.route_id in self.routes:
                r = self.routes[e.route_id]
                headway = (r.headway_peak_min + r.headway_offpeak_min) / 2.0
            rows.append(encode_edge(
                edge_class=edge_class_of(e.kind, e.mode),
                distance_km=e.distance_km,
                free_speed_kmph=e.free_speed_kmph or (e.distance_km / max(e.base_min, 1e-6) * 60.0),
                base_min=e.base_min, lanes=e.lanes, headway_min=headway,
                endpoint_congestion_mean=(cu + cv) / 2.0,
            ))
        return np.asarray(rows, dtype=np.float32)

    def _pick_ride_hubs(self) -> list[str]:
        """Where a hailed vehicle can plausibly pick you up or drop you: every
        metro station, every named place, and bus stops served by 2+ routes."""
        route_count: dict[str, int] = {}
        for r in self.routes.values():
            for st in r.stops:
                route_count[st] = route_count.get(st, 0) + 1
        hubs = [
            nid for nid, n in self.nodes.items()
            if n.kind in ("metro_station", "place") or route_count.get(nid, 0) >= 2
        ]
        return sorted(hubs)

    # -- lookups -----------------------------------------------------------
    def edge_id_to_static_row(self) -> dict[int, int]:
        return {idx: row for row, idx in enumerate(self.static_edge_idx)}

    def nearest_nodes(self, lat: float, lon: float, max_km: float, limit: int
                      ) -> list[tuple[str, float]]:
        out = [
            (nid, haversine_km(lat, lon, n.lat, n.lon))
            for nid, n in self.nodes.items()
        ]
        out.sort(key=lambda t: t[1])
        near = [t for t in out if t[1] <= max_km][:limit]
        return near or out[:1]  # never strand a request with nothing at all

    def headway_for(self, route_id: str | None, ctx: TimeContext) -> float:
        if not route_id or route_id not in self.routes:
            return 0.0
        return self.routes[route_id].headway_at(ctx.hour, ctx.is_weekend)

    def boarding_wait_for(self, route_id: str | None, ctx: TimeContext) -> float:
        """Expected wait to board this route at this moment.

        Half the headway while the route is running -- the expected wait for a
        passenger turning up at a random time. Outside service hours it is the
        real wait until the first departure, which is what makes a 02:00 request
        stop being offered a metro it cannot catch.
        """
        if not route_id or route_id not in self.routes:
            return 0.0
        r = self.routes[route_id]
        until = r.minutes_until_service(ctx.hour, ctx.is_weekend)
        return until + r.headway_at(ctx.hour, ctx.is_weekend) / 2.0

    def routes_in_service(self, ctx: TimeContext) -> dict[str, bool]:
        return {rid: r.in_service(ctx.hour, ctx.is_weekend)
                for rid, r in self.routes.items()}


class RequestGraph:
    """A per-request view: the shared city graph plus this request's access
    and ride edges. Read-only with respect to the shared graph."""

    def __init__(self, base: MultimodalGraph, origin: tuple[float, float],
                 destination: tuple[float, float], allowed_modes: set[str] | None = None):
        self.base = base
        self.nodes = dict(base.nodes)
        # Clone: a predictor writes predicted_min onto these, and the shared
        # city graph must stay untouched so concurrent requests cannot collide.
        self.edges: list[GraphEdge] = [replace(e) for e in base.edges]
        self.out_adj: dict[str, list[int]] = {k: list(v) for k, v in base.out_adj.items()}
        self.allowed_modes = allowed_modes
        self.origin_lat, self.origin_lon = origin
        self.dest_lat, self.dest_lon = destination
        self._extra_start = len(base.edges)
        self._add_endpoints()
        self._add_access_edges()
        self._add_ride_edges()

    # -- construction ------------------------------------------------------
    def _add(self, **kw) -> GraphEdge:
        e = GraphEdge(idx=len(self.edges), **kw)
        self.edges.append(e)
        self.out_adj.setdefault(e.u, []).append(e.idx)
        return e

    def _add_endpoints(self) -> None:
        for nid, lat, lon, name in (
            (ORIGIN_ID, self.origin_lat, self.origin_lon, "Origin"),
            (DESTINATION_ID, self.dest_lat, self.dest_lon, "Destination"),
        ):
            self.nodes[nid] = Node(node_id=nid, name=name, kind="place",
                                   lat=lat, lon=lon, category="endpoint")
            self.out_adj.setdefault(nid, [])

    def _add_access_edges(self) -> None:
        """Walk between the user's actual coordinates and the nearby network."""
        s = get_settings()
        for endpoint, lat, lon in (
            (ORIGIN_ID, self.origin_lat, self.origin_lon),
            (DESTINATION_ID, self.dest_lat, self.dest_lon),
        ):
            near = self.base.nearest_nodes(lat, lon, s.max_access_walk_km, limit=10)
            for nid, straight_km in near:
                walk_km = straight_km * WALK_DETOUR
                walk_min = walk_km / s.walk_speed_kmph * 60.0
                for u, v in ((endpoint, nid), (nid, endpoint)):
                    self._add(edge_id=f"ac_{u}_{v}", u=u, v=v, kind="access",
                              mode="walk", distance_km=walk_km, base_min=walk_min,
                              walk_min=walk_min, reliability=0.97)

    def _nearby_transit(self, lat: float, lon: float) -> list[str]:
        """Stops and stations a hailed vehicle would plausibly run you to."""
        s = get_settings()
        near = self.base.nearest_nodes(lat, lon, s.access_ride_km, limit=40)
        out = [nid for nid, _ in near
               if self.base.nodes[nid].kind in ("bus_stop", "metro_station")]
        return out[:s.access_ride_stops]

    def _ride_pairs(self) -> list[tuple[str, str, bool]]:
        """Where a hailed vehicle can take you: the door, a hub, or a stop.

        The flag says whether this is the door-to-door ride. That one is the
        option a rider always has and must always be offered; everything else
        is a first/last-mile hop and is bounded far more tightly.

        The third case is what makes a cheap journey possible at all. A ride
        used to reach only a "hub" -- a metro station, a named place, or a stop
        served by two routes -- and the nearest of those to the Wipro campus is
        6.7 km away, so every cheap itinerary started with a half-hour walk and
        was rightly thrown out. There are bus stops 500 m from that gate, and a
        bike taxi will happily take you to one.
        """
        pairs: list[tuple[str, str, bool]] = [(ORIGIN_ID, DESTINATION_ID, True)]
        for hub in self.base._ride_hubs:
            pairs.append((ORIGIN_ID, hub, False))
            pairs.append((hub, DESTINATION_ID, False))
        for nid in self._nearby_transit(self.origin_lat, self.origin_lon):
            pairs.append((ORIGIN_ID, nid, False))
        for nid in self._nearby_transit(self.dest_lat, self.dest_lon):
            pairs.append((nid, DESTINATION_ID, False))
        seen, out = set(), []
        for u, v, direct in pairs:
            if (u, v) in seen:
                continue
            seen.add((u, v))
            out.append((u, v, direct))
        return out

    def _add_ride_edges(self) -> None:
        s = get_settings()
        modes = [m for m in RIDE_MODES
                 if self.allowed_modes is None or m in self.allowed_modes]
        for u, v, is_direct in self._ride_pairs():
            if u == v:
                continue
            nu, nv = self.nodes.get(u), self.nodes.get(v)
            if nu is None or nv is None:
                continue
            straight = haversine_km(nu.lat, nu.lon, nv.lat, nv.lon)
            road_km = straight * ROAD_DETOUR
            cap = s.max_direct_ride_km if is_direct else s.max_ride_leg_km
            if road_km < 0.35 or road_km > cap:
                continue
            for mode in modes:
                speed = RIDE_FREE_SPEED_KMPH[mode]
                self._add(edge_id=f"rd_{mode}_{u}_{v}", u=u, v=v, kind="ride",
                          mode=mode, distance_km=road_km,
                          base_min=road_km / speed * 60.0, free_speed_kmph=speed,
                          wait_min=RIDE_PICKUP_WAIT_MIN[mode], reliability=0.72)

    # -- accessors ---------------------------------------------------------
    @property
    def request_edges(self) -> list[GraphEdge]:
        return self.edges[self._extra_start:]

    def out_edges(self, node_id: str) -> list[GraphEdge]:
        return [self.edges[i] for i in self.out_adj.get(node_id, ())]

    def clone_edges(self) -> list[GraphEdge]:
        """Deep-ish copy so a predictor can write predicted_min without
        touching the shared city graph."""
        return [replace(e) for e in self.edges]


@lru_cache(maxsize=4)
def get_graph(city_id: str | None = None) -> MultimodalGraph:
    return MultimodalGraph(get_provider(city_id))
