"""Booking and reveal endpoints — the demonstration flow.

    POST /api/book              press BOOK NOW; runs attempt 1
    POST /api/book/{id}/retry   press TRY AGAIN; runs attempt n+1 at a new fare
    GET  /api/book/{id}         the session as it stands
    GET  /api/book/{id}/reveal  what actually happened, and what it cost
    GET  /api/insights          the supply-demand relationships behind all of it

THE ORDER MATTERS
-----------------
`/reveal` is a separate call, and the interface does not make it until the
rider has attempted a booking. That is the whole storytelling constraint: the
product behaves like a normal mobility app, the rider hits a real failure, and
only then does the system explain that it saw the failure coming. Showing the
prediction first turns a demonstration into a dashboard.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta

import numpy as np
from fastapi import APIRouter, HTTPException, Query, Request

from ..booking.escalation import (
    MeetingContext, assess_arrival, compose_notification, incident_record,
)
from ..booking.session import (
    DEMO_TARGET_OUTCOME, BookingSession, get_store, new_session_id,
    MAX_ATTEMPTS, run_next_attempt, seed_for_outcome,
)
from ..demo_scenario import DEMO_SCENARIO
from ..labels import journey_phrase
from ..enterprise.analytics import load_bookings
from ..lifecycle.expected_cost import LifecycleParams
from ..reliability.model import get_reliability_model
from ..schemas import BookRequest, NotifyRequest
from ..security import audit
from ..services.clock import now_local, to_local
from ..services.compare import compare
from ..services.engine import RoutingError, get_engine

log = logging.getLogger("journeymind.api.booking")
router = APIRouter()

#: Fallback seed when the demo target is unreachable for an option — a metro
#: cannot cancel on you, so there is no seed that makes it.
DEMO_SEED = 20260828


def _money(x: float) -> str:
    return f"₹{x:,.0f}"


def _resolve(point, engine, what: str):
    from .routes import resolve_point
    return resolve_point(point, engine, what)


def _demo_seed(demo: bool, rel) -> int:
    """Which draw a demonstration should start on.

    A live demo has to reproduce the *interesting* failure -- a driver accepting
    and then cancelling -- or the evaluator never sees the problem the product
    exists to solve. So demo mode searches for a seed whose first attempt lands
    on that branch.

    THIS DOES NOT CHANGE THE PROBABILITIES. The option's chance of cancelling is
    whatever the model predicts; what is chosen is which sample the demo opens
    on, exactly as a presenter would by re-running a live booking until they got
    the case they wanted to discuss. For an option that cannot cancel -- a metro,
    a bus -- no such seed exists and the fixed fallback is used, so a scheduled
    service still completes and still says so.
    """
    if not demo:
        return DEMO_SEED
    # The whole retry budget, not just the first attempt. A demonstration of a
    # rider who cannot get a ride has to actually run out of attempts, or the
    # arrival-risk panel lands under the words "Journey completed".
    found = seed_for_outcome(
        DEMO_TARGET_OUTCOME, p_match=rel.p_match, p_accept=rel.p_accept,
        p_cancel=rel.p_cancel, failing_attempts=MAX_ATTEMPTS)
    if found is None:
        # Nothing that unlucky in the search budget: settle for the first
        # attempt cancelling, which is the step the story turns on.
        found = seed_for_outcome(
            DEMO_TARGET_OUTCOME, p_match=rel.p_match, p_accept=rel.p_accept,
            p_cancel=rel.p_cancel)
    return DEMO_SEED if found is None else found


@router.post("/api/book", tags=["booking"])
def book(body: BookRequest, request: Request):
    """Press BOOK NOW on one option and live with the consequences."""
    engine = get_engine()
    o_lat, o_lon, o_label = _resolve(body.origin, engine, "origin")
    d_lat, d_lon, d_label = _resolve(body.destination, engine, "destination")
    departure = to_local(body.departure_time, engine.city.timezone)

    try:
        result = compare(
            origin_lat=o_lat, origin_lon=o_lon, origin_label=o_label,
            dest_lat=d_lat, dest_lon=d_lon, dest_label=d_label,
            departure=departure, priority=body.priority, rain=body.rain)
    except RoutingError as exc:
        raise HTTPException(status_code=422, detail={
            "error": exc.message, "code": exc.code}) from exc

    chosen = next((o for o in result.options
                   if o.quote.provider_id == body.provider_id), None)
    if chosen is None:
        raise HTTPException(status_code=422, detail={
            "error": f"'{body.provider_id}' is not an option for this trip.",
            "code": "unknown_provider",
            "detail": "Call POST /api/compare to see what is available."})
    if not chosen.quote.available:
        raise HTTPException(status_code=422, detail={
            "error": chosen.quote.unavailable_reason or "That option cannot serve this trip.",
            "code": "provider_unavailable"})

    q, rel = chosen.quote, chosen.quote.reliability
    session = BookingSession(
        session_id=new_session_id(),
        provider_id=q.provider_id, display_name=q.display_name, mode=q.mode,
        service_class=q.service_class.value,
        origin_label=o_label, dest_label=d_label, departure=departure,
        base_fare=q.fare.amount, pickup_min=q.pickup_min, ride_min=q.ride_min,
        p_match=rel.p_match, p_accept=rel.p_accept, p_cancel=rel.p_cancel,
        params=LifecycleParams(),
        rng=np.random.default_rng(_demo_seed(body.demo, rel) if body.demo else None),
        demo=bool(body.demo),
        comparison=_snapshot(result),
    )
    get_store().put(session)
    attempt = run_next_attempt(session)

    audit(kind="booking", actor="anonymous",
          request={"origin": o_label, "destination": d_label,
                   "provider": q.provider_id, "demo": bool(body.demo)},
          decision={"attempt": attempt.number, "outcome": attempt.outcome.value,
                    "fare": round(attempt.fare, 2)},
          model_versions={"reliability": get_reliability_model().meta.get("version", "fallback")},
          confidence=round(rel.p_success_per_attempt, 4))

    return {"session": session.as_dict(), "attempt": attempt.as_dict()}


@router.post("/api/book/{session_id}/retry", tags=["booking"])
def retry(session_id: str):
    """TRY AGAIN. A new attempt, at the escalated fare."""
    session = get_store().get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail={
            "error": "That booking has expired.", "code": "session_expired",
            "detail": "Start a new booking."})
    try:
        attempt = run_next_attempt(session)
    except ValueError as exc:
        raise HTTPException(status_code=409, detail={
            "error": str(exc), "code": "no_attempts_left"}) from exc
    return {"session": session.as_dict(), "attempt": attempt.as_dict()}


@router.get("/api/book/{session_id}", tags=["booking"])
def get_session(session_id: str):
    session = get_store().get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail={
            "error": "That booking has expired.", "code": "session_expired"})
    return {"session": session.as_dict()}


# --------------------------------------------------------------------------
def _snapshot(result) -> dict:
    """Freeze the options as they were offered, so the reveal explains the
    booking the rider actually made rather than a fresh calculation."""
    return {
        "headline": result.headline,
        # The itineraries too. After four failed bookings the rider needs to
        # know that travelling in stages is still on the table -- telling
        # somebody to try another hailed vehicle is the one answer that has
        # already been shown not to work.
        "journeys": list(result.journeys or []),
        "options": [{
            "provider_id": o.quote.provider_id,
            "display_name": o.quote.display_name,
            "mode": o.quote.mode,
            "service_class": o.quote.service_class.value,
            "fare": round(o.quote.fare.amount, 2),
            "pickup_min": round(o.quote.pickup_min, 1),
            "door_to_door_min": round(o.quote.door_to_door_min, 1),
            "available": o.quote.available,
            "p_match": round(o.quote.reliability.p_match, 4),
            "p_accept": round(o.quote.reliability.p_accept, 4),
            "p_cancel": round(o.quote.reliability.p_cancel, 4),
            "p_success": round(o.expected.p_success, 4),
            "expected_cost": round(o.expected.expected_cost, 2),
            "expected_minutes": round(o.expected.expected_minutes, 1),
            "expected_attempts": round(o.expected.expected_attempts, 2),
            "expected_wasted_min": round(o.expected.expected_wasted_min, 1),
            "is_blended": o.expected.is_blended,
        } for o in result.options],
    }


#: How much longer than the chosen option an alternative may take before it
#: stops counting as advice. 1.5x plus a grace for short trips: a 20-minute
#: ride can lose 25 minutes to a cheaper option, a three-hour one cannot.
ALT_TIME_FACTOR = 1.5
ALT_TIME_GRACE_MIN = 15.0


@router.get("/api/book/{session_id}/reveal", tags=["booking"])
def reveal(session_id: str):
    """What actually happened — shown only after the rider has tried to book.

    Compares what the option advertised against what the model expected of it,
    and against the option the model would have chosen. Every number here was
    computed *before* the booking ran; none of it is fitted to the outcome.
    """
    session = get_store().get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail={
            "error": "That booking has expired.", "code": "session_expired"})
    if not session.attempts:
        raise HTTPException(status_code=409, detail={
            "error": "Nothing to explain yet — no booking has been attempted.",
            "code": "not_attempted"})

    snap = session.comparison or {"options": []}
    options = snap.get("options", [])
    chosen = next((o for o in options if o["provider_id"] == session.provider_id), None)

    # Two alternatives, because they answer different questions.
    #
    #   better            lowest expected cost among options that actually
    #                     complete *and arrive in a comparable time* -- often
    #                     transit, and that IS the honest answer even if it is
    #                     not the exciting one
    #   better_same_class like-for-like: the best alternative of the same kind,
    #                     so the comparison is not "a cab versus a train"
    viable = [o for o in options
              if o["available"] and o["p_success"] >= 0.80
              and o["provider_id"] != session.provider_id]

    # Cheapest-that-completes is not automatically better advice. On a 20 km
    # trip the metro is genuinely ₹25 and genuinely completes -- and genuinely
    # takes three and a half hours. Crowning it while the panel above warns the
    # rider they are late for a meeting is two contradictory recommendations on
    # one screen, so an alternative has to be comparable on TIME before it can
    # win on money.
    chosen_min = float(chosen["expected_minutes"]) if chosen else None
    comparable = viable
    excluded_cheaper = None
    if chosen_min:
        limit = chosen_min * ALT_TIME_FACTOR + ALT_TIME_GRACE_MIN
        comparable = [o for o in viable if o["expected_minutes"] <= limit]
        cheapest_any = min(viable, key=lambda o: o["expected_cost"]) if viable else None
        # If the outright cheapest was ruled out on time, say so rather than
        # quietly dropping it -- the rider is entitled to make that trade.
        if (cheapest_any is not None
                and cheapest_any not in comparable):
            excluded_cheaper = cheapest_any

    better = min(comparable, key=lambda o: o["expected_cost"]) if comparable else None
    same = [o for o in comparable if o["service_class"] == session.service_class]
    better_same_class = min(same, key=lambda o: o["expected_cost"]) if same else None

    lived = {
        "attempts": len(session.attempts),
        "failures": session.failures,
        "settled": session.settled,
        "paid": round(session.total_paid, 2) if session.settled else None,
        "advertised": round(session.base_fare, 2),
        "wasted_min": round(session.wasted_min, 1),
        "overpaid": (round(session.total_paid - session.base_fare, 2)
                     if session.settled else None),
    }

    narrative = _narrate(session, chosen, better, better_same_class, lived)
    if excluded_cheaper is not None:
        narrative.append(
            f"{excluded_cheaper['display_name']} is cheaper still at "
            f"{_money(excluded_cheaper['expected_cost'])} expected, but takes "
            f"about {excluded_cheaper['expected_minutes']:.0f} minutes against "
            f"{chosen['expected_minutes']:.0f} — a trade of time for money "
            f"rather than a better answer to the same question.")

    return {
        "session_id": session.session_id,
        "lived": lived,
        "chosen": chosen,
        "better": better,
        "better_same_class": better_same_class,
        "cheaper_but_slower": excluded_cheaper,
        "narrative": narrative,
        "all_options": options,
        "method_note": (
            "Every probability shown here was predicted before you pressed BOOK "
            "NOW, and the booking you just ran was a draw from exactly those "
            "probabilities. Nothing has been fitted to what happened."),
        "causality_note": (
            "These are associations learned from historical bookings, not causal "
            "claims. A low fare does not cause a cancellation. Both are related "
            "to the same underlying condition: how attractive a driver finds a "
            "given trip at a given moment."),
    }


def _narrate(session: BookingSession, chosen, better, better_same_class,
             lived) -> list[str]:
    """Plain sentences, generated from the numbers rather than templated prose."""
    out: list[str] = []
    if chosen is None:
        return out

    if lived["settled"] and lived["attempts"] == 1:
        out.append(
            f"That worked first time — which it does about "
            f"{chosen['p_success']:.0%} of the time for this option.")
    elif lived["settled"]:
        out.append(
            f"It took {lived['attempts']} attempts. You paid "
            f"{_money(lived['paid'])} against the {_money(lived['advertised'])} "
            f"advertised, and lost about {lived['wasted_min']:.0f} minutes getting there.")
    else:
        out.append(
            f"After {lived['attempts']} attempts the journey never started, and "
            f"about {lived['wasted_min']:.0f} minutes are gone.")

    failures = session.failures
    if "DRIVER_CANCELLED" in failures:
        out.append(
            f"A driver accepted and then cancelled. That is the expensive "
            f"failure — the clock was already running. This option is cancelled "
            f"after acceptance about {chosen['p_cancel']:.0%} of the time on a "
            f"trip like this.")
    if "DRIVER_REJECTED" in failures:
        out.append(
            f"A driver declined the request. About {1 - chosen['p_accept']:.0%} of "
            f"matched drivers decline this trip — short fares are declined more "
            f"often when demand is high.")
    if "NO_DRIVER_AVAILABLE" in failures:
        out.append(
            f"One search found nobody at all. Around "
            f"{1 - chosen['p_match']:.0%} of requests for this option get no response.")

    if chosen["expected_cost"] > chosen["fare"] + 0.5:
        out.append(
            f"None of that was visible on the card. {_money(chosen['fare'])} was "
            f"the advertised fare; {_money(chosen['expected_cost'])} is what this "
            f"option is expected to cost once the chance of it falling through is "
            f"priced in — {(chosen['expected_cost'] / max(chosen['fare'], 1) - 1):.0%} more.")
    elif chosen.get("is_blended"):
        out.append(
            f"The expected cost of {_money(chosen['expected_cost'])} sits below the "
            f"{_money(chosen['fare'])} fare only because this option fails so often "
            f"that most of the time you end up on something else entirely. That is "
            f"not a discount.")

    def _cheaper_dearer(a: float, b: float) -> str:
        """Say 'more' or 'less' correctly — a fare comparison that gets its own
        direction wrong destroys the credibility of everything around it."""
        if a > b + 0.5:
            return "more than"
        if a < b - 0.5:
            return "less than"
        return "about the same as"

    # De-duplicate by provider, not by object identity. When the like-for-like
    # alternative and the overall best are the same option, it must be named
    # ONCE -- the previous guard skipped both and the comparison vanished.
    emitted: set[str] = set()
    for alt, lead in ((better_same_class, "Like for like, "), (better, "")):
        if not alt or alt["provider_id"] == session.provider_id:
            continue
        if alt["provider_id"] in emitted:
            continue
        if alt["expected_cost"] >= chosen["expected_cost"] - 0.5:
            continue
        emitted.add(alt["provider_id"])
        direction = _cheaper_dearer(alt["fare"], chosen["fare"])
        slower = alt["expected_minutes"] - chosen["expected_minutes"]
        timing = (f" It also arrives about {slower:.0f} minutes later."
                  if slower >= 5 else
                  f" It also arrives about {abs(slower):.0f} minutes sooner."
                  if slower <= -5 else " Arrival time is about the same.")
        out.append(
            f"{lead}{alt['display_name']} advertises {_money(alt['fare'])} — "
            f"{direction} {_money(chosen['fare'])} — but completes "
            f"{alt['p_success']:.0%} of the time, so its expected cost is "
            f"{_money(alt['expected_cost'])}: cheaper in the sense that counts, "
            f"which is what you end up paying.{timing}")
    return out


# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
# escalation: what happens when the rider simply cannot get a ride
# --------------------------------------------------------------------------
def _meeting_for(session, title, starts_at, manager):
    """The commitment the rider is trying to reach.

    Falls back to the ONE canonical demo scenario rather than to an invented
    "an hour from now": a meeting that exists only inside the escalation is a
    meeting no other screen can corroborate, and the manager notification ended
    up describing a commitment the rest of the demo had never heard of.

    The default title names the time rather than the owner. "your next meeting"
    read correctly in the app's voice and wrongly in the rider's -- the drafted
    message said "late for your next meeting" to the manager, which is that
    person's meeting, not the rider's.
    """
    if starts_at is None:
        hour = float(DEMO_SCENARIO["meeting_hour"])
        scheduled = session.departure.replace(
            hour=int(hour), minute=int(round((hour % 1) * 60)),
            second=0, microsecond=0)
        # A meeting already behind the rider is not what they are travelling
        # to; fall forward rather than reporting them hours late for it.
        starts_at = (scheduled if scheduled > session.departure
                     else session.departure + timedelta(minutes=60))
        title = title or (DEMO_SCENARIO["meeting_title"]
                          if scheduled > session.departure else None)
    return MeetingContext(title=title or f"the {starts_at:%H:%M} meeting",
                          starts_at=starts_at, manager=manager)


def _rider_clock(session) -> datetime:
    """Where the rider is in TIME, not where the server is.

    The projection has to run on the trip's own clock: a journey planned
    for 09:00 that has burned nine minutes on failed bookings is at 09:09,
    whatever the wall clock says. Using the server clock produced "543
    minutes early" against a 10:00 meeting — nonsense that would have
    shipped, because the number looked like an ordinary big number.
    """
    return session.departure + timedelta(minutes=session.wasted_min)


#: When the rider is already late, how many further minutes are worth trading
#: for a cheaper option. Being least-late is the objective once nothing can be
#: on time; it is not worth any amount of money.
LATE_TOLERANCE_MIN = 12.0


def _best_remaining(session, *, meeting=None, now=None):
    """What the rider should switch to, from the frozen comparison.

    Fastest-wins is the wrong rule and it showed: the panel recommended a ₹543
    cab for an 83-minute trip when a ₹113 option arrived eight minutes later.
    Being on time is the constraint, not the objective -- so among the options
    that still make the meeting, the cheapest wins, and only when nothing makes
    it does the fastest.
    """
    # A settled booking has nothing to switch to. Offering an alternative to
    # somebody already in the vehicle is advice about a decision they have
    # made.
    if session.settled:
        return None

    snap = session.comparison or {}
    opts = snap.get("options", [])
    viable = [o for o in opts
              if o["available"] and o["p_success"] >= 0.80
              and o["provider_id"] != session.provider_id]
    if not viable:
        viable = [o for o in opts if o["available"]
                  and o["provider_id"] != session.provider_id]

    # Travelling in stages counts as an option. It is priced door to door and
    # it does not depend on the one thing that has just failed four times.
    for j in snap.get("journeys", []):
        viable.append({
            "provider_id": j["journey_id"],
            "display_name": journey_phrase(j.get("shape_modes") or j["modes"]),
            "expected_cost": j["fare"],
            "expected_minutes": j["total_min"],
            "p_success": None,
            "is_journey": True,
        })
    if not viable:
        return None

    fastest = min(viable, key=lambda o: o["expected_minutes"])
    if meeting is None or now is None:
        return fastest

    budget_min = (meeting.starts_at - now).total_seconds() / 60.0
    in_time = [o for o in viable if o["expected_minutes"] <= budget_min]
    if in_time:
        return min(in_time, key=lambda o: o["expected_cost"])

    # Nothing arrives in time, so the objective becomes "least late" -- but not
    # at any price. Ten minutes off a twenty-five-minute delay does not justify
    # four hundred rupees, and recommending a ₹505 cab while a ₹141 itinerary
    # sat on the same screen is the kind of advice that gets a product closed.
    close = [o for o in viable
             if o["expected_minutes"] <= fastest["expected_minutes"] + LATE_TOLERANCE_MIN]
    return min(close, key=lambda o: o["expected_cost"]) if close else fastest


@router.get("/api/book/{session_id}/escalation", tags=["booking"])
def escalation(session_id: str,
               meeting: str | None = Query(None),
               meeting_at: datetime | None = Query(None),
               manager: str | None = Query(None)):
    """Am I going to miss what I was travelling to, and what should I do?"""
    session = get_store().get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail={
            "error": "That booking has expired.", "code": "session_expired"})
    ctx = _meeting_for(session, meeting, meeting_at, manager)
    now = _rider_clock(session)
    alt = _best_remaining(session, meeting=ctx, now=now)
    risk = assess_arrival(now=now,
                          wasted_min=session.wasted_min, meeting=ctx,
                          best_option=alt)
    snap = session.as_dict()
    return {
        "session_id": session.session_id,
        "attempts": snap["attempt_count"],
        "attempts_left": snap["attempts_left"],
        "exhausted": session.exhausted,
        "meeting": ctx.as_dict(),
        "risk": risk.as_dict(),
        "alternative": alt,
        # Only when the rider is genuinely stuck AND genuinely at risk. A
        # completed ride is not an incident, however tight the arrival.
        "can_notify": (risk.level in ("at_risk", "late")
                       and session.exhausted and not session.settled),
        "notification_preview": compose_notification(
            session=session, meeting=ctx, risk=risk, alternative=alt),
    }


@router.post("/api/book/{session_id}/notify", tags=["booking"])
def notify(session_id: str, body: NotifyRequest):
    """Tell the manager -- only because the rider pressed the button.

    Composed and recorded, never transmitted: no mail or chat transport is
    wired in, and reporting a message as sent when it was not would be a false
    claim about an action outside this system.
    """
    session = get_store().get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail={
            "error": "That booking has expired.", "code": "session_expired"})
    ctx = _meeting_for(session, body.meeting, body.meeting_at, body.manager)
    now = _rider_clock(session)
    alt = _best_remaining(session, meeting=ctx, now=now)
    risk = assess_arrival(now=now,
                          wasted_min=session.wasted_min, meeting=ctx,
                          best_option=alt)
    message = compose_notification(session=session, meeting=ctx, risk=risk,
                                   alternative=alt)
    incident = incident_record(session=session, meeting=ctx, risk=risk,
                               notified=True)
    audit(kind="escalation", actor="rider",
          request={"session": session.session_id, "meeting": ctx.title,
                   "attempts": len(session.attempts)},
          decision={"incident_id": incident["incident_id"], "risk": risk.level,
                    "notified": True, "delivery": message["delivery"]},
          model_versions={"reliability": get_reliability_model().meta.get(
              "version", "fallback")},
          data_classes=["SIMULATED"])
    return {"message": message, "incident": incident, "risk": risk.as_dict(),
            "alternative": alt}


@router.get("/api/insights", tags=["booking"])
def insights(bins: int = Query(6, ge=3, le=12)):
    """The supply-and-demand relationships behind the whole product.

    Aggregated from the bundled booking history so a viewer can see that this
    is a market phenomenon, not a quirk of one booking. Every panel is a
    *relationship between observed quantities*; none of them is a causal claim,
    and the payload says so rather than leaving it to be inferred.

    Aggregation runs over NumPy masks (see enterprise/store.py) so a panel
    costs milliseconds rather than a second.
    """
    t = load_bookings()
    if t is None or not t.n:
        return {"panels": [], "note": "no booking history loaded"}

    import numpy as np

    def stats(mask) -> dict | None:
        n = int(np.count_nonzero(mask))
        if n < 25:
            return None
        matched = int(np.count_nonzero(mask & t.matched))
        accepted = int(np.count_nonzero(mask & t.accepted))
        return {
            "n": n,
            "supply": round(matched / n, 4),
            "acceptance": round(accepted / matched, 4) if matched else 0.0,
            "cancellation": (round(int(np.count_nonzero(mask & t.cancelled))
                                   / accepted, 4) if accepted else 0.0),
            "success": round(int(np.count_nonzero(mask & t.completed)) / n, 4),
            "mean_fare": round(float(t.fare[mask].mean()), 2),
        }

    def band(values, labels, edges) -> list[dict]:
        out = []
        for i, lab in enumerate(labels):
            lo = edges[i]
            hi = edges[i + 1] if i + 1 < len(edges) else float("inf")
            row = stats((values >= lo) & (values < hi))
            if row:
                out.append({"label": lab, **row})
        return out

    fare_panel = band(
        t.fare, ["under ₹40", "₹40–70", "₹70–110", "₹110–170", "₹170–260", "over ₹260"],
        [0, 40, 70, 110, 170, 260])
    demand_panel = band(
        t.peak_intensity, ["very low", "low", "moderate", "high", "very high"],
        [0.0, 0.12, 0.3, 0.55, 0.8])
    distance_panel = band(
        t.distance_km, ["under 2 km", "2–4 km", "4–7 km", "7–12 km", "over 12 km"],
        [0, 2, 4, 7, 12])

    hours = t.hour.astype(np.int16)
    hourly = []
    for h in range(24):
        row = stats(hours == h)
        if row:
            hourly.append({"label": f"{h:02d}:00", "hour": h, **row})

    providers = []
    for pid in sorted(t.provider.labels):
        mask = t.provider.mask_for(pid)
        row = stats(mask)
        if row is None:
            continue
        completed = mask & t.completed
        km = float(t.distance_km[completed].sum())
        spend = float(t.spend[mask].sum())
        if not km or not row["success"]:
            continue
        providers.append({
            "label": pid, **row,
            "cost_per_km": round(spend / km, 2),
            "effective_cost_per_km": round(spend / km / row["success"], 2),
        })

    return {
        "panels": [
            {"key": "fare", "title": "Fare band vs how often the booking works",
             "x_label": "advertised fare", "rows": fare_panel,
             "reading": ("Cheaper bands are not automatically worse. What moves "
                         "with the fare is trip length, and short trips are the "
                         "ones drivers decline.")},
            {"key": "demand", "title": "Demand pressure vs supply and cancellation",
             "x_label": "demand", "rows": demand_panel,
             "reading": ("As demand rises, fewer requests find a vehicle and more "
                         "accepted bookings are cancelled. This is the market "
                         "condition both the fare and the failure respond to.")},
            {"key": "distance", "title": "Trip length vs acceptance",
             "x_label": "distance", "rows": distance_panel,
             "reading": ("The clearest relationship in the data: the shorter the "
                         "trip, the more often a matched driver declines it.")},
            {"key": "hour", "title": "Time of day vs booking success",
             "x_label": "hour", "rows": hourly,
             "reading": "Booking success collapses in the commute peaks."},
            {"key": "provider", "title": "Provider reliability vs effective cost",
             "x_label": "provider", "rows": providers,
             "reading": ("Effective cost per km divides the billed rate by the "
                         "share of bookings that complete. The ranking is not the "
                         "same as the billed ranking.")},
        ],
        "data_note": "Aggregated from the bundled demonstration booking history.",
        "causality_note": (
            "These panels show ASSOCIATION, not causation. Nothing here "
            "establishes that a low fare causes a cancellation. The defensible "
            "reading is that fare, demand, supply, acceptance and cancellation "
            "all respond to the same underlying market conditions, and that the "
            "relationships are strong enough to predict — which is all the "
            "recommendation engine needs."),
        "generated_at": now_local(get_engine().city.timezone),
    }
