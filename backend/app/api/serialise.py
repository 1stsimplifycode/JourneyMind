"""Turning engine objects into API payloads.

Kept apart from the routes so the shape of the response is defined in one
place, and so every number that leaves the building carries its provenance
label with it.
"""

from __future__ import annotations

from ..config import get_settings
from ..services.clock import now_local
from ..models.fares import FareEstimate
from ..optimisation.constraints import ConstraintStatus
from ..routing.journey import Journey
from ..services.engine import Recommendation


def fare_out(f: FareEstimate | None, symbol: str = "₹") -> dict | None:
    if f is None:
        return None
    return {
        "amount": round(f.amount, 2), "low": round(f.low, 2), "high": round(f.high, 2),
        "display": f.display(symbol), "provenance": f.provenance, "label": f.label,
        "note": f.note, "source": f.source, "is_range": f.is_range,
    }


def leg_out(leg, symbol: str = "₹") -> dict:
    return {
        "index": leg.index, "mode": leg.mode, "kind": leg.kind,
        "from_name": leg.from_name, "to_name": leg.to_name,
        "distance_km": round(leg.distance_km, 3),
        "travel_min": round(leg.travel_min, 1),
        "wait_min": round(leg.wait_min, 1),
        "total_min": round(leg.total_min, 1),
        "stops": leg.stops, "route_name": leg.route_name,
        "route_colour": leg.route_colour, "fare": fare_out(leg.fare, symbol),
        "time_provenance": leg.time_provenance,
        "geometry": [[round(a, 6), round(b, 6)] for a, b in leg.geometry],
    }


def journey_out(j: Journey, status: ConstraintStatus, symbol: str = "₹") -> dict:
    return {
        "journey_id": j.journey_id,
        "summary": j.mode_summary(),
        "modes": j.modes,
        "legs": [leg_out(lg, symbol) for lg in j.legs],
        "total_cost": fare_out(j.total_cost, symbol),
        "total_min": round(j.total_min, 1),
        "transfers": j.transfers,
        "distance_km": round(j.distance_km, 2),
        "walk_min": round(j.walk_min, 1),
        "wait_min": round(j.wait_min, 1),
        "reliability": j.reliability,
        "score": j.score,
        "score_breakdown": j.score_parts or None,
        "constraints": status.as_dict(),
    }


def data_notice(city, fares) -> dict:
    s = get_settings()
    return {
        "demo_mode": s.demo_mode,
        "label": city.data_status_label,
        "city": city.display_name,
        "notes": city.notes,
        "fare_provenance": {m: f.provenance for m, f in fares.items()},
    }


def recommendation_out(rec: Recommendation, engine) -> dict:
    symbol = engine.city.currency_symbol
    req = rec.request
    payload = {
        "feasible": rec.feasible,
        "message": rec.message,
        "origin": {"label": req.origin_label, "lat": req.origin_lat, "lon": req.origin_lon},
        "destination": {"label": req.dest_label, "lat": req.dest_lat, "lon": req.dest_lon},
        "departure_time": req.departure,
        # When this answer was produced, on the study area's clock. The UI
        # shows it so a result that has been sitting on screen for ten minutes
        # cannot pass itself off as current.
        "computed_at": now_local(engine.city.timezone),
        # What each single-mode option would have cost, priced whether or not it
        # survived filtering. v1 section 3's worked example, returned as data.
        "mode_comparison": [
            {**{k: v for k, v in row.items() if k != "total_cost"},
             "total_cost": fare_out(row["total_cost"], symbol)}
            for row in rec.mode_comparison
        ],
        "preference": rec.preset,
        "weights": rec.weights,
        "recommended": None,
        "explanation": None,
        "alternatives": [],
        "fallbacks": [],
        "model_info": rec.model_info,
        "data_notice": data_notice(engine.city, engine.graph.fares),
        "pipeline": rec.pipeline,
    }
    if rec.recommended is not None and rec.recommended_status is not None:
        payload["recommended"] = journey_out(rec.recommended, rec.recommended_status, symbol)
        payload["explanation"] = rec.explanation.as_dict()
        payload["alternatives"] = [
            {"kind": a.get("kind", "feasible"), "reason": a["reason"],
             "journey": journey_out(a["journey"], a["status"], symbol)}
            for a in rec.alternatives
        ]
    payload["fallbacks"] = [
        {"label": f["label"], "why": f["why"], "reason": f["reason"],
         "journey": journey_out(f["journey"], f["status"], symbol)}
        for f in rec.fallbacks
    ]
    return payload
