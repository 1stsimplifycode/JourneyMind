"""`TransportDataProvider` backed by the bundled static study-area files.

This is the provider the deployed MVP uses. It reads CSV/JSON from disk once,
caches in memory, and never touches the network -- so the application cannot
fail because an external feed is down.
"""

from __future__ import annotations

import csv
import json
from functools import lru_cache
from pathlib import Path
from typing import Iterable, Sequence

from ..config import get_settings
from .provider import (
    CityMeta, FareModel, Node, Place, RoadEdge, TransferEdge, TransitEdge,
    TransitRoute, TransportDataProvider, TravelTimeObservation,
)


def _f(v, default=0.0) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def _i(v, default=0) -> int:
    try:
        return int(float(v))
    except (TypeError, ValueError):
        return default


class MissingDataError(RuntimeError):
    """Raised at startup when the study-area bundle is absent or incomplete."""


class StaticFileProvider(TransportDataProvider):
    REQUIRED = (
        "city.json", "nodes.csv", "road_edges.csv", "transit_edges.csv",
        "transfer_edges.csv", "transit_routes.json", "fares.json", "places.json",
    )

    def __init__(self, city_dir: Path):
        self.dir = Path(city_dir)
        missing = [f for f in self.REQUIRED if not (self.dir / f).exists()]
        if missing:
            raise MissingDataError(
                f"Study-area bundle at {self.dir} is missing: {', '.join(missing)}. "
                f"Run `python scripts/generate_dataset.py` to rebuild it."
            )
        self._nodes: list[Node] | None = None
        self._road: list[RoadEdge] | None = None
        self._transit: list[TransitEdge] | None = None
        self._transfer: list[TransferEdge] | None = None
        self._routes: list[TransitRoute] | None = None
        self._fares: dict[str, FareModel] | None = None
        self._places: list[Place] | None = None

    # -- helpers -----------------------------------------------------------
    def _rows(self, name: str):
        with open(self.dir / name, newline="", encoding="utf-8") as fh:
            yield from csv.DictReader(fh)

    def _json(self, name: str):
        with open(self.dir / name, encoding="utf-8") as fh:
            return json.load(fh)

    # -- interface ---------------------------------------------------------
    def get_city(self) -> CityMeta:
        raw = self._json("city.json")
        return CityMeta(
            city_id=raw["city_id"], display_name=raw["display_name"],
            currency=raw.get("currency", "INR"),
            currency_symbol=raw.get("currency_symbol", "₹"),
            timezone=raw.get("timezone", "Asia/Kolkata"),
            centre=raw["centre"], bbox=raw["bbox"],
            data_status=raw.get("data_status", "demo"),
            data_status_label=raw.get("data_status_label", "Demo / estimated data"),
            notes=raw.get("notes", ""),
            counts=dict(
                nodes=len(self.get_nodes()),
                road_edges=len(self.get_road_edges()),
                transit_edges=len(self.get_transit_edges()),
                transfer_edges=len(self.get_transfer_edges()),
                routes=len(self.get_transit_routes()),
            ),
        )

    def get_nodes(self) -> Sequence[Node]:
        if self._nodes is None:
            self._nodes = [
                Node(
                    node_id=r["node_id"], name=r["name"], kind=r["kind"],
                    lat=_f(r["lat"]), lon=_f(r["lon"]),
                    lines=tuple(x for x in (r.get("lines") or "").split("|") if x),
                    is_interchange=bool(_i(r.get("is_interchange"))),
                    category=(r.get("category") or None),
                    degree=_i(r.get("degree")),
                    observed_congestion=_f(r.get("observed_congestion")),
                )
                for r in self._rows("nodes.csv")
            ]
        return self._nodes

    def get_road_edges(self) -> Sequence[RoadEdge]:
        if self._road is None:
            self._road = [
                RoadEdge(
                    edge_id=r["edge_id"], u=r["u"], v=r["v"],
                    distance_km=_f(r["distance_km"]), road_class=r["road_class"],
                    free_speed_kmph=_f(r["free_speed_kmph"], 30.0), lanes=_i(r["lanes"], 1),
                )
                for r in self._rows("road_edges.csv")
            ]
        return self._road

    def get_transit_edges(self) -> Sequence[TransitEdge]:
        if self._transit is None:
            self._transit = [
                TransitEdge(
                    edge_id=r["edge_id"], route_id=r["route_id"], mode=r["mode"],
                    u=r["u"], v=r["v"], seq=_i(r["seq"]),
                    distance_km=_f(r["distance_km"]), scheduled_min=_f(r["scheduled_min"]),
                )
                for r in self._rows("transit_edges.csv")
            ]
        return self._transit

    def get_transfer_edges(self) -> Sequence[TransferEdge]:
        if self._transfer is None:
            self._transfer = [
                TransferEdge(
                    edge_id=r["edge_id"], u=r["u"], v=r["v"],
                    distance_km=_f(r["distance_km"]), walk_min=_f(r["walk_min"]),
                )
                for r in self._rows("transfer_edges.csv")
            ]
        return self._transfer

    def get_transit_routes(self) -> Sequence[TransitRoute]:
        if self._routes is None:
            self._routes = [
                TransitRoute(
                    route_id=r["route_id"], mode=r["mode"], name=r["name"],
                    colour=r.get("colour", "#666666"),
                    headway_peak_min=_f(r["headway_peak_min"], 10.0),
                    headway_offpeak_min=_f(r["headway_offpeak_min"], 20.0),
                    stops=tuple(r["stops"]),
                    service_start_h=_f(r.get("service_start_h"), 5.0),
                    service_end_h=_f(r.get("service_end_h"), 23.5),
                    service_start_weekend_h=(
                        _f(r["service_start_weekend_h"])
                        if r.get("service_start_weekend_h") is not None else None),
                )
                for r in self._json("transit_routes.json")
            ]
        return self._routes

    def get_fares(self) -> dict[str, FareModel]:
        if self._fares is None:
            raw = self._json("fares.json")
            out: dict[str, FareModel] = {}
            for mode, spec in raw["modes"].items():
                out[mode] = FareModel(
                    mode=mode, label=spec.get("label", mode.title()),
                    kind=spec["kind"], provenance=spec.get("provenance", "estimated"),
                    note=spec.get("note", ""), source=spec.get("source"),
                    flat_fare=_f(spec.get("flat_fare")),
                    slabs=tuple((float(a), float(b)) for a, b in spec.get("slabs", [])),
                    above_top_slab_fare=_f(spec.get("above_top_slab_fare")),
                    base_fare=_f(spec.get("base_fare")),
                    base_distance_km=_f(spec.get("base_distance_km")),
                    per_km=_f(spec.get("per_km")), per_min=_f(spec.get("per_min")),
                    minimum_fare=_f(spec.get("minimum_fare")),
                    uncertainty_pct=_f(spec.get("uncertainty_pct")),
                )
            self._fares = out
        return self._fares

    def get_travel_times(self) -> Iterable[TravelTimeObservation]:
        """Streamed, not cached -- this is the only large file and it is used
        by the offline training scripts, not by the request path."""
        path = self.dir / "travel_times.csv"
        if not path.exists():
            return
        for r in self._rows("travel_times.csv"):
            yield TravelTimeObservation(
                edge_id=r["edge_id"], edge_kind=r["edge_kind"], mode=r["mode"],
                ts=r["ts"], hour=_f(r["hour"]), dow=_i(r["dow"]),
                is_weekend=bool(_i(r["is_weekend"])), rain=bool(_i(r["rain"])),
                base_min=_f(r["base_min"]), observed_min=_f(r["observed_min"]),
            )

    def get_places(self) -> Sequence[Place]:
        if self._places is None:
            self._places = [
                Place(place_id=p["place_id"], name=p["name"], lat=_f(p["lat"]),
                      lon=_f(p["lon"]), category=p.get("category", "other"))
                for p in self._json("places.json")
            ]
        return self._places


@lru_cache(maxsize=4)
def get_provider(city_id: str | None = None) -> TransportDataProvider:
    """Factory. Swap the implementation here to move off the static bundle."""
    s = get_settings()
    city = city_id or s.city_id
    return StaticFileProvider(s.data_dir / "city" / city)
