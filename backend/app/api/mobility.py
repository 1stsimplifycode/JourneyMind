"""Mobility intelligence endpoints.

    POST /api/compare              the product: expected cost across providers
    GET  /api/providers            the provider registry and what each one is
    GET  /api/lifecycle            the booking state machine, as data
    GET  /api/enterprise/facets    filter options for the dashboard   [analyst]
    GET  /api/enterprise/overview  the enterprise dashboard payload    [analyst]
    GET  /api/enterprise/audit     recorded AI decisions               [analyst]

The rider endpoints are open; the enterprise ones require a key with at least
the analyst role, because they expose population-level data. See security.py
for why the gate sits exactly there.
"""

from __future__ import annotations

import logging
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, Query, Request

from ..enterprise.analytics import (
    DEFAULT_MINUTE_COST, Filters, build, facets, load_bookings,
)
from ..lifecycle.states import ABSORBING, LEGAL_TRANSITIONS
from ..providers.simulated import registry as provider_registry
from ..reliability.model import get_reliability_model
from ..schemas import CompareRequest
from ..security import Principal, Role, audit, get_audit_log, require_role
from ..services.clock import now_local, to_local
from ..services.compare import PRIORITIES, compare
from ..services.engine import RoutingError, get_engine

log = logging.getLogger("journeymind.api.mobility")
router = APIRouter()


def _resolve(point, engine, what: str) -> tuple[float, float, str]:
    from .routes import resolve_point
    return resolve_point(point, engine, what)


# --------------------------------------------------------------------------
@router.post("/api/compare", tags=["mobility"])
def compare_options(body: CompareRequest, request: Request):
    """Compare every way of making this trip, priced for what it will really cost."""
    engine = get_engine()
    o_lat, o_lon, o_label = _resolve(body.origin, engine, "origin")
    d_lat, d_lon, d_label = _resolve(body.destination, engine, "destination")
    departure = to_local(body.departure_time, engine.city.timezone)

    try:
        result = compare(
            origin_lat=o_lat, origin_lon=o_lon, origin_label=o_label,
            dest_lat=d_lat, dest_lon=d_lon, dest_label=d_label,
            departure=departure, priority=body.priority,
            budget=body.budget, max_time_min=body.max_time, rain=body.rain)
        # what the rider typed, so the screen can admit it matched something
        # else rather than quietly relabelling their origin
        result.origin_typed = _typed(body.origin)
        result.dest_typed = _typed(body.destination)
    except RoutingError as exc:
        raise HTTPException(status_code=422, detail={
            "error": exc.message, "code": exc.code}) from exc
    except Exception as exc:
        log.exception("comparison failed")
        raise HTTPException(status_code=500, detail={
            "error": "The comparison engine could not complete this request.",
            "code": "engine_error"}) from exc

    payload = _serialise(result, engine)

    # Governance: every recommendation is recorded with the evidence behind it.
    rel = get_reliability_model()
    audit(
        kind="recommendation",
        actor=getattr(getattr(request.state, "principal", None), "key_id", "anonymous"),
        request={"origin": o_label, "destination": d_label,
                 "priority": body.priority, "budget": body.budget,
                 "max_time": body.max_time,
                 "departure": departure.isoformat(timespec="seconds")},
        decision={
            "recommended": (result.recommended.quote.provider_id
                            if result.recommended else None),
            "headline": result.headline,
            "expected_cost": (round(result.recommended.expected.expected_cost, 2)
                              if result.recommended else None),
            "displayed_fare": (round(result.recommended.quote.fare.amount, 2)
                               if result.recommended else None),
            "reasons": result.reasoning,
        },
        model_versions={
            "reliability": rel.meta.get("version", "fallback"),
            "travel_time": engine.graph.city.city_id,
        },
        confidence=(result.recommended.expected.p_success
                    if result.recommended else None),
        data_classes=sorted({o.quote.data_class.value for o in result.options}),
    )
    return payload


def _typed(point) -> str | None:
    """The free text on the request, if it was free text at all."""
    if isinstance(point, str):
        return point
    return getattr(point, "label", None)


def _resolved(label: str, typed: str | None) -> dict:
    """What the endpoint is, and what the rider called it if that differs."""
    out = {"label": label}
    if typed and typed.strip().lower() != label.strip().lower():
        out["typed"] = typed.strip()
    return out


def _journeys_for(result) -> list[dict]:
    """The multi-vehicle journeys the planner found for this trip.

    A journey is shown when it genuinely mixes modes -- walk-and-metro,
    bike-taxi-then-metro. Single-vehicle trips are already on the ride cards
    and repeating them as "journeys" would be noise.
    """
    # The rule itself lives in services/compare.offerable, and the comparison
    # already applied it -- to the list AND to the headline, so the two cannot
    # contradict each other. This layer only decides how many to show.
    return (getattr(result, "journeys", []) or [])[:3]


def _serialise(result, engine) -> dict:
    symbol = engine.city.currency_symbol

    def money(x: float) -> str:
        return f"{symbol}{x:,.0f}"

    options = []
    for o in result.options:
        q, e = o.quote, o.expected
        r = q.reliability
        # An option with no route has no numbers. The lifecycle solver still
        # returns a figure for it -- it prices the fallback you would take
        # instead -- and publishing that as the option's own cost read as
        # "Walk: ₹25, 0 min", which is not a thing. Unrouted options carry
        # their reason and nothing else.
        unrouted = not q.available and q.distance_km <= 0
        options.append({
            "provider_id": q.provider_id,
            "display_name": q.display_name,
            # The vehicle and the operator are different facts. Keeping them
            # apart is what stops "recommend Rapido" standing in for
            # "recommend a bike taxi".
            "provider_name": q.provider_name,
            "mode": q.mode,
            "service_class": q.service_class.value,
            "data_class": q.data_class.value,
            "rank": o.rank,
            "recommended": o is result.recommended,
            "available": q.available,
            "unavailable_reason": q.unavailable_reason,
            "feasible": o.feasible,
            # Priced but never advised -- see ProviderQuote.recommendable.
            "recommendable": o.quote.recommendable,
            "within_budget": o.within_budget,
            "within_time": o.within_time,
            # Fits the limit on the trip itself, but not once the cost of
            # failing and rebooking is priced in.
            "budget_at_risk": o.budget_at_risk,
            "time_at_risk": o.time_at_risk,
            "fare": {
                "amount": round(q.fare.amount, 2),
                "low": round(q.fare.low, 2), "high": round(q.fare.high, 2),
                "display": (money(q.fare.amount) if not q.fare.is_range
                            else f"{money(q.fare.low)}–{money(q.fare.high)}"),
                "provenance": q.fare.provenance,
                "surge_multiplier": q.fare.surge_multiplier,
            },
            "pickup_min": None if unrouted else round(q.pickup_min, 1),
            "ride_min": None if unrouted else round(q.ride_min, 1),
            "door_to_door_min": None if unrouted else round(q.door_to_door_min, 1),
            "distance_km": None if unrouted else round(q.distance_km, 2),
            "reliability": {
                "p_match": round(r.p_match, 4),
                "p_accept": round(r.p_accept, 4),
                "p_cancel": round(r.p_cancel, 4),
                "p_success_per_attempt": round(r.p_success_per_attempt, 4),
                "basis": r.basis,
                "data_class": r.data_class.value,
            },
            "expected": (None if unrouted else
                         {**e.as_dict(),
                          "expected_cost_display": money(e.expected_cost)}),
            "notes": list(q.notes),
        })

    return {
        # `typed` is only present when the match differs from what was
        # entered. A pasted office address can resolve to a nearby feature --
        # 0.7 km away, in one real case -- and quietly relabelling somebody's
        # origin is not something to do in silence.
        "origin": _resolved(result.origin_label, getattr(result, "origin_typed", None)),
        "destination": _resolved(result.dest_label,
                                 getattr(result, "dest_typed", None)),
        # Multimodal journeys, from the SAME planner the Journey planner uses.
        # Without these the booking screen shows only single-vehicle rides and
        # the product looks like it can only ever suggest one hailed option --
        # which is the opposite of what the routing engine actually does.
        "journeys": _journeys_for(result),
        "departure_time": result.departure,
        "computed_at": now_local(engine.city.timezone),
        "priority": result.priority,
        "budget": result.budget,
        "max_time": result.max_time_min,
        "headline": result.headline,
        "reasoning": result.reasoning,
        "recommended_provider": (result.recommended.quote.provider_id
                                 if result.recommended else None),
        "options": options,
        "pipeline": result.trace,
        # Human-facing wording. The machine-readable `data_class` on every
        # option is unchanged and still carries the exact provenance -- this is
        # the display string only.
        "data_notice": {
            "label": "Demo dataset",
            "detail": ("Routes and travel times come from the bundled study-area "
                       "graph and the travel-time model. Ride-hailing fares, "
                       "availability and cancellation rates are modelled estimates "
                       "rather than live operator data — no commercial provider "
                       "API is contacted. Metro and bus fares are transcribed from "
                       "published tables."),
        },
    }


@router.get("/api/providers", tags=["mobility"])
def providers():
    """Every provider behind the abstraction, and how honest each one is."""
    rel = get_reliability_model()
    return {
        "providers": provider_registry(),
        "reliability_model": {
            "version": rel.meta.get("version", "fallback"),
            "trained_on": rel.meta.get("trained_on", "n/a"),
            "data_class": rel.meta.get("data_class", "assumption"),
            "is_fallback": rel.meta.get("version") == "fallback",
        },
        "note": ("Every hailed-vehicle adapter is a simulation. The interface is "
                 "the deliverable: a real adapter implements the same five "
                 "methods and nothing else changes."),
    }


@router.get("/api/lifecycle", tags=["mobility"])
def lifecycle():
    """The booking state machine, served as data so the UI cannot drift from it."""
    return {
        "states": sorted({s.value for s in LEGAL_TRANSITIONS}),
        "absorbing": sorted(s.value for s in ABSORBING),
        "transitions": {a.value: sorted(b.value for b in bs)
                        for a, bs in LEGAL_TRANSITIONS.items()},
        "note": ("A search is not a ride. The three failure edges are modelled "
                 "separately because they cost different amounts of time."),
    }


# --------------------------------------------------------------------------
# enterprise — gated
# --------------------------------------------------------------------------
analyst = require_role(Role.ANALYST)


@router.get("/api/enterprise/facets", tags=["enterprise"])
def enterprise_facets(principal: Principal = Depends(analyst)):
    return {"facets": facets(load_bookings()),
            "principal": principal.as_dict()}


@router.get("/api/enterprise/overview", tags=["enterprise"])
def enterprise_overview(
    request: Request,
    principal: Principal = Depends(analyst),
    campus: str | None = Query(None),
    provider: str | None = Query(None),
    employee_group: str | None = Query(None),
    mode: str | None = Query(None),
    date_from: str | None = Query(None),
    date_to: str | None = Query(None),
    hour_from: int | None = Query(None, ge=0, le=23),
    hour_to: int | None = Query(None, ge=1, le=24),
    minute_cost: float = Query(DEFAULT_MINUTE_COST, gt=0, le=1000),
):
    filters = Filters(campus=campus, provider=provider,
                      employee_group=employee_group, mode=mode,
                      date_from=date_from, date_to=date_to,
                      hour_from=hour_from, hour_to=hour_to)
    payload = build(load_bookings(), filters, minute_cost)
    payload["principal"] = principal.as_dict()
    audit(kind="enterprise_query", actor=principal.key_id,
          request={"filters": payload["filters_applied"]},
          decision={"bookings_in_scope": payload["overview"].get("bookings", 0),
                    "insights": len(payload["insights"])},
          data_classes=["SIMULATED"])
    return payload


@router.get("/api/enterprise/audit", tags=["enterprise"])
def enterprise_audit(principal: Principal = Depends(analyst),
                     limit: int = Query(100, ge=1, le=500),
                     kind: str | None = Query(None)):
    """Every AI decision this instance has made, newest first."""
    logbook = get_audit_log()
    return {
        "entries": logbook.recent(limit=limit, kind=kind),
        "total_held": len(logbook),
        "durable": logbook.path is not None,
        "note": ("In-memory ring buffer unless JM_AUDIT_LOG is set to a file "
                 "path, in which case entries are also appended durably."),
        "principal": principal.as_dict(),
    }
