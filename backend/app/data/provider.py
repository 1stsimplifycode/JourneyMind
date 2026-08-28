"""The data abstraction layer.

Everything above this module talks to `TransportDataProvider` and never to a
file, a feed or an API. Swapping the bundled demo bundle for a real OSM +
GTFS pipeline means writing one new subclass and changing one line in the
factory at the bottom -- nothing in the graph, model, routing or optimisation
layers should need to change.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Iterable, Literal, Sequence

Provenance = Literal["exact", "published", "estimated", "predicted", "demo"]


# --------------------------------------------------------------------------
# domain records
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class Node:
    node_id: str
    name: str
    kind: str  # metro_station | bus_stop | junction | place
    lat: float
    lon: float
    lines: tuple[str, ...] = ()
    is_interchange: bool = False
    category: str | None = None
    degree: int = 0
    observed_congestion: float = 0.0

    @property
    def is_transit(self) -> bool:
        return self.kind in ("metro_station", "bus_stop")


@dataclass(frozen=True)
class RoadEdge:
    edge_id: str
    u: str
    v: str
    distance_km: float
    road_class: str
    free_speed_kmph: float
    lanes: int


@dataclass(frozen=True)
class TransitEdge:
    edge_id: str
    route_id: str
    mode: str  # metro | bus
    u: str
    v: str
    seq: int
    distance_km: float
    scheduled_min: float


@dataclass(frozen=True)
class TransferEdge:
    edge_id: str
    u: str
    v: str
    distance_km: float
    walk_min: float


@dataclass(frozen=True)
class TransitRoute:
    route_id: str
    mode: str
    name: str
    colour: str
    headway_peak_min: float
    headway_offpeak_min: float
    stops: tuple[str, ...]
    # Local clock hours the route actually runs. A journey planned at 02:00
    # must not be told to catch a train that is in the depot.
    service_start_h: float = 5.0
    service_end_h: float = 23.5
    service_start_weekend_h: float | None = None

    def headway_at(self, hour: float, is_weekend: bool) -> float:
        peak = (not is_weekend) and (7.5 <= hour <= 10.5 or 17.0 <= hour <= 20.5)
        return self.headway_peak_min if peak else self.headway_offpeak_min

    def first_departure_h(self, is_weekend: bool) -> float:
        if is_weekend and self.service_start_weekend_h is not None:
            return self.service_start_weekend_h
        return self.service_start_h

    def in_service(self, hour: float, is_weekend: bool) -> bool:
        return self.first_departure_h(is_weekend) <= hour <= self.service_end_h

    def minutes_until_service(self, hour: float, is_weekend: bool) -> float:
        """0 while the route is running, otherwise the wait until it starts.

        Charged as real waiting time rather than used to hide the route, so a
        journey that genuinely has to wait for the first train says so instead
        of silently disappearing.
        """
        if self.in_service(hour, is_weekend):
            return 0.0
        start = self.first_departure_h(True if (is_weekend and hour > self.service_end_h)
                                       else is_weekend)
        delta = start - hour
        if delta < 0:                      # service is over for today
            delta += 24.0
        return delta * 60.0


@dataclass(frozen=True)
class TravelTimeObservation:
    edge_id: str
    edge_kind: str
    mode: str
    ts: str
    hour: float
    dow: int
    is_weekend: bool
    rain: bool
    base_min: float
    observed_min: float


@dataclass(frozen=True)
class Place:
    place_id: str
    name: str
    lat: float
    lon: float
    category: str


@dataclass(frozen=True)
class FareModel:
    """One mode's fare rule plus its honesty label."""

    mode: str
    label: str
    kind: str  # flat | distance_slab | metered
    provenance: Provenance
    note: str
    source: str | None = None
    flat_fare: float = 0.0
    slabs: tuple[tuple[float, float], ...] = ()
    above_top_slab_fare: float = 0.0
    base_fare: float = 0.0
    base_distance_km: float = 0.0
    per_km: float = 0.0
    per_min: float = 0.0
    minimum_fare: float = 0.0
    uncertainty_pct: float = 0.0


@dataclass(frozen=True)
class CityMeta:
    city_id: str
    display_name: str
    currency: str
    currency_symbol: str
    timezone: str
    centre: dict
    bbox: dict
    data_status: str
    data_status_label: str
    notes: str
    counts: dict = field(default_factory=dict)


# --------------------------------------------------------------------------
# the interface
# --------------------------------------------------------------------------
class TransportDataProvider(ABC):
    """Read-only access to one study area."""

    @abstractmethod
    def get_city(self) -> CityMeta: ...

    @abstractmethod
    def get_nodes(self) -> Sequence[Node]: ...

    @abstractmethod
    def get_road_edges(self) -> Sequence[RoadEdge]: ...

    @abstractmethod
    def get_transit_edges(self) -> Sequence[TransitEdge]: ...

    @abstractmethod
    def get_transfer_edges(self) -> Sequence[TransferEdge]: ...

    @abstractmethod
    def get_transit_routes(self) -> Sequence[TransitRoute]: ...

    @abstractmethod
    def get_fares(self) -> dict[str, FareModel]: ...

    @abstractmethod
    def get_travel_times(self) -> Iterable[TravelTimeObservation]: ...

    @abstractmethod
    def get_places(self) -> Sequence[Place]: ...

    # -- convenience shared by every implementation ------------------------
    def node_index(self) -> dict[str, Node]:
        return {n.node_id: n for n in self.get_nodes()}

    def route_index(self) -> dict[str, TransitRoute]:
        return {r.route_id: r for r in self.get_transit_routes()}
