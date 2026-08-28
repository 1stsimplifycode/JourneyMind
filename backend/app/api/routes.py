"""HTTP API.

    GET  /health         liveness + what is actually loaded
    GET  /api/city       study area, bounds, transit lines, data honesty notice
    GET  /api/places     the named places the UI offers for From / To
    GET  /api/models     every model in the comparison set and its availability
    GET  /api/demo       the bundled demo scenario, run through the real pipeline
    POST /api/recommend  the product

Nothing here computes a recommendation itself. Everything goes through the
engine, so the API and the demo cannot drift apart.
"""

from __future__ import annotations

import logging
from datetime import datetime

from fastapi import APIRouter, HTTPException, Request

from ..config import get_settings
from ..data.geo import haversine_km
from ..demo_scenario import DEMO_SCENARIO
from ..models.loader import registry
from ..schemas import RecommendRequest
from ..services.geocode import geocode
from ..services.clock import now_local as _now_local, to_local as _to_local
from ..services.engine import JourneyRequest, RoutingError, get_engine
from .serialise import data_notice, recommendation_out

log = logging.getLogger("journeymind.api")
router = APIRouter()

# The bundled demo. Origin and destination are real addresses at opposite ends
# of the study corridor, and the answer is computed for the moment you press
# the button -- nothing about it is precomputed or hard-coded.


def now_local() -> datetime:
    """Right now, on the study area's own clock."""
    return _now_local(get_engine().city.timezone)


def to_local(dt: datetime | None) -> datetime:
    """A client timestamp, expressed on the study area's clock."""
    return _to_local(dt, get_engine().city.timezone)


def default_departure() -> datetime:
    """Leave now. The product is a live advisor, not a timetable browser."""
    return now_local()


def _best_local_match(needle: str, candidates: list):
    """The bundled place a typed name means, if it means one.

    Plain substring matching sent "Jayanagar" to **Vi**jayanagar, because one
    name contains the other. A typed name has to line up with a word boundary
    to count, and an exact name beats a prefix beats a word inside the name.
    """
    def score(name: str) -> int | None:
        low = name.lower()
        if low == needle:
            return 0
        words = low.replace("(", " ").replace(")", " ").replace(",", " ").split()
        if low.startswith(needle):
            return 1
        if any(w.startswith(needle) for w in words):
            return 2
        # a multi-word query that appears whole inside the name
        if " " in needle and needle in low:
            return 3
        return None

    ranked = []
    for c in candidates:
        s = score(c.name)
        if s is not None:
            ranked.append((s, len(c.name), c))
    if not ranked:
        return None
    ranked.sort(key=lambda t: (t[0], t[1]))
    return ranked[0][2]


def resolve_point(point, engine, what: str) -> tuple[float, float, str]:
    """A named place, an explicit coordinate, or a label we can match."""
    places = {p.place_id: p for p in engine.graph.places}

    if getattr(point, "place_id", None):
        p = places.get(point.place_id)
        if p is None:
            raise HTTPException(status_code=422, detail={
                "error": f"Unknown place id '{point.place_id}' for {what}.",
                "code": "unknown_place",
                "detail": "Call GET /api/places for the list this study area supports.",
            })
        return p.lat, p.lon, point.label or p.name

    if point.lat is not None and point.lon is not None:
        label = point.label or f"{point.lat:.4f}, {point.lon:.4f}"
        return point.lat, point.lon, label

    if point.label:
        needle = point.label.strip().lower()
        hit = _best_local_match(needle, list(places.values()))
        if hit is not None:
            return hit.lat, hit.lon, hit.name

        # Not one of the bundled fifteen. Ask OpenStreetMap, bounded to the
        # corridor -- "Whitefield" and "BTM Layout" are real places a rider
        # might type, and answering "could not find that" was only ever true
        # of our own list.
        found = geocode(point.label, engine.city.bbox)
        if found is not None:
            lat, lon, name = found
            return lat, lon, name

        raise HTTPException(status_code=422, detail={
            "error": f"Could not find '{point.label}' in this study area.",
            "code": "unknown_place",
            "detail": ("This build covers one bounded corridor of Bengaluru. "
                       "Try a place inside it, pick one from GET /api/places, "
                       "or send lat and lon directly."),
        })

    raise HTTPException(status_code=422, detail={
        "error": f"No usable {what}.", "code": "invalid_point",
        "detail": "Send a place_id, or a lat/lon pair.",
    })


# --------------------------------------------------------------------------
@router.get("/health", tags=["meta"])
def health():
    """Liveness probe. Reports what is actually loaded, not what was configured."""
    s = get_settings()
    try:
        engine = get_engine()
        from ..models.loader import get_predictor
        predictor = get_predictor()
        return {
            "status": "ok",
            "app": s.app_name, "version": s.version,
            "demo_mode": s.demo_mode,
            "city": engine.city.display_name,
            "graph": {"nodes": len(engine.graph.nodes), "edges": len(engine.graph.edges)},
            "model": {"requested": s.travel_time_model,
                      "loaded": predictor.info.name,
                      "display": predictor.info.display_name,
                      "fell_back": predictor.info.name != s.travel_time_model},
        }
    except Exception as exc:                       # never 500 a health check
        log.exception("health check degraded")
        return {"status": "degraded", "app": s.app_name, "version": s.version,
                "error": str(exc)[:300]}


@router.get("/api/city", tags=["meta"])
def city():
    engine = get_engine()
    c = engine.city
    now = now_local()
    return {
        "city_id": c.city_id, "display_name": c.display_name,
        "currency": c.currency, "currency_symbol": c.currency_symbol,
        "timezone": c.timezone, "centre": c.centre, "bbox": c.bbox,
        "counts": c.counts,
        "routes": [
            {"route_id": r.route_id, "mode": r.mode, "name": r.name,
             "colour": r.colour, "headway_peak_min": r.headway_peak_min,
             "headway_offpeak_min": r.headway_offpeak_min,
             "service_start_h": r.first_departure_h(now.weekday() >= 5),
             "service_end_h": r.service_end_h,
             "in_service": r.in_service(now.hour + now.minute / 60.0,
                                        now.weekday() >= 5),
             "stops": [
                 {"node_id": s, "name": engine.graph.nodes[s].name,
                  "lat": engine.graph.nodes[s].lat, "lon": engine.graph.nodes[s].lon}
                 for s in r.stops if s in engine.graph.nodes
             ]}
            for r in engine.graph.routes.values()
        ],
        "data_notice": data_notice(c, engine.graph.fares),
        "now": now,
        "default_departure": now,
        "demo_scenario": {
            "origin": DEMO_SCENARIO["origin"],
            "destination": DEMO_SCENARIO["destination"],
            "budget": DEMO_SCENARIO["budget"],
            "max_time": DEMO_SCENARIO["max_time"],
            "preference": DEMO_SCENARIO["preference"],
            "title": DEMO_SCENARIO["title"],
            "meeting_title": DEMO_SCENARIO["meeting_title"],
            "meeting_hour": DEMO_SCENARIO["meeting_hour"],
        },
    }


@router.get("/api/places", tags=["meta"])
def places():
    engine = get_engine()
    return {"places": [
        {"place_id": p.place_id, "name": p.name, "lat": p.lat, "lon": p.lon,
         "category": p.category}
        for p in sorted(engine.graph.places, key=lambda p: p.name)
    ]}


@router.get("/api/models", tags=["meta"])
def models():
    """The baseline comparison set from the documentation.

    Availability is reported honestly: a model with no trained weights, or one
    whose library is not installed in the serving image, says so.
    """
    return {
        "models": registry(),
        "note": ("Six models behind one interface. Which one is better is an "
                 "experimental question — see scripts/evaluate.py and "
                 "EVALUATION.md. No accuracy claim is made here."),
    }


@router.get("/api/demo", tags=["journeys"])
def demo():
    """The documented demo scenario, computed live by the same engine."""
    engine = get_engine()
    places_by_id = {p.place_id: p for p in engine.graph.places}
    o = places_by_id[DEMO_SCENARIO["origin"]]
    d = places_by_id[DEMO_SCENARIO["destination"]]
    dep = now_local()          # the demo is live: this minute, not a fixed hour

    req = JourneyRequest(
        origin_lat=o.lat, origin_lon=o.lon, origin_label=o.name,
        dest_lat=d.lat, dest_lon=d.lon, dest_label=d.name,
        departure=dep, budget=DEMO_SCENARIO["budget"],
        max_time_min=DEMO_SCENARIO["max_time"],
        preference=DEMO_SCENARIO["preference"],
    )
    try:
        rec = engine.recommend(req)
    except RoutingError as exc:
        raise HTTPException(status_code=422, detail={
            "error": exc.message, "code": exc.code}) from exc

    payload = recommendation_out(rec, engine)
    payload["scenario"] = {
        "title": DEMO_SCENARIO["title"],
        "description": DEMO_SCENARIO["description"],
        "request": {
            "origin": DEMO_SCENARIO["origin"], "destination": DEMO_SCENARIO["destination"],
            "budget": DEMO_SCENARIO["budget"], "max_time": DEMO_SCENARIO["max_time"],
            "preference": DEMO_SCENARIO["preference"],
            "departure_time": dep.isoformat(),
        },
    }
    return payload


@router.post("/api/recommend", tags=["journeys"])
def recommend(body: RecommendRequest, request: Request):
    s = get_settings()
    engine = get_engine()

    o_lat, o_lon, o_label = resolve_point(body.origin, engine, "origin")
    d_lat, d_lon, d_label = resolve_point(body.destination, engine, "destination")

    if haversine_km(o_lat, o_lon, d_lat, d_lon) < 0.05:
        raise HTTPException(status_code=422, detail={
            "error": "Your start and destination are the same place.",
            "code": "same_endpoints",
            "detail": "Pick two different points.",
        })
    if body.budget > s.max_budget_inr or body.max_time > s.max_time_min:
        raise HTTPException(status_code=422, detail={
            "error": "Budget or time limit is outside the supported range.",
            "code": "out_of_range",
        })

    req = JourneyRequest(
        origin_lat=o_lat, origin_lon=o_lon, origin_label=o_label,
        dest_lat=d_lat, dest_lon=d_lon, dest_label=d_label,
        departure=to_local(body.departure_time),
        budget=float(body.budget), max_time_min=float(body.max_time),
        preference=body.preference,
        manual_weights=body.weights.model_dump() if body.weights else None,
        max_transfers=body.max_transfers,
        allowed_modes=set(body.modes) if body.modes else None,
        rain=body.rain,
    )

    try:
        rec = engine.recommend(req)
    except RoutingError as exc:
        raise HTTPException(status_code=422, detail={
            "error": exc.message, "code": exc.code}) from exc
    except Exception as exc:
        # Log the detail, return something safe. No stack traces to the client.
        log.exception("recommendation failed")
        raise HTTPException(status_code=500, detail={
            "error": "The recommendation engine could not complete this request.",
            "code": "engine_error",
        }) from exc

    return recommendation_out(rec, engine)
