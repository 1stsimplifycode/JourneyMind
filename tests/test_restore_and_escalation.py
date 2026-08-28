"""The multimodal planner, the demo cancellation script, and the escalation.

Two of these are regression guards against the product *looking* broken:

  * `test_planner_is_not_a_rapido_machine` fails if the routing engine stops
    producing genuine multi-mode journeys. The engine was accused of always
    recommending a bike taxi; it does not, but a generous budget makes it look
    that way, and this pins the behaviour so the question can be settled by
    running the tests rather than by argument.

  * `test_demo_reproduces_the_cancellation_script` fails if a demonstration
    stops showing driver-accepted-then-cancelled, which is the one failure the
    whole product story depends on the evaluator seeing.
"""

from __future__ import annotations

import itertools
import os
import sys
from datetime import datetime

import pytest
from fastapi.testclient import TestClient

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "backend"))

from app.booking.escalation import (                                # noqa: E402
    AT_RISK_MARGIN_MIN, MeetingContext, assess_arrival,
)
from app.main import app                                            # noqa: E402
from app.services.engine import JourneyRequest, get_engine          # noqa: E402

LONG = {"origin": "Wipro Campus", "destination": "PES University",
        "departure_time": "2026-08-28T09:00:00"}


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


# ==========================================================================
# the planner still plans
# ==========================================================================
def test_the_planner_is_not_locked_to_one_mode():
    """Squeeze the budget and the planner must reach for other vehicles.

    Not a demand that multimodal always wins -- at an unconstrained budget a
    direct bike taxi legitimately does. The claim is that the planner is
    *capable*, which is what "it always recommends Rapido" would disprove.

    The budgets stop at ₹150. Below that nothing on this corridor is reachable
    without half an hour on foot, and refusing to call that a commute is the
    point rather than a gap -- `test_an_unaffordable_trip_says_so_with_numbers`
    covers what happens instead.
    """
    eng = get_engine()
    P = {p.place_id: p for p in eng.graph.places}
    o, d = P["pl_wipro_sarjapur"], P["pl_pes_university"]

    seen_modes: set[str] = set()
    multimodal = 0
    for budget, max_time in ((400, 180), (250, 120), (150, 150)):
        r = eng.recommend(JourneyRequest(
            origin_lat=o.lat, origin_lon=o.lon, origin_label=o.name,
            dest_lat=d.lat, dest_lon=d.lon, dest_label=d.name,
            departure=datetime(2026, 8, 28, 9, 0), budget=float(budget),
            max_time_min=float(max_time), preference="balanced"))
        j = r.recommended
        assert j is not None, f"no journey at ₹{budget}/{max_time}min"
        seen_modes |= set(j.modes)
        assert "walk" not in j.modes, "walking is not a mode we recommend"
        if len(set(j.modes)) > 1:
            multimodal += 1

    assert multimodal >= 1, "the planner never combined modes at any budget"
    for mode in ("bike_taxi", "metro", "bus"):
        assert mode in seen_modes, (
            f"{mode} never appeared in any recommendation — the planner has "
            f"stopped considering it. Saw: {sorted(seen_modes)}")


def test_an_unaffordable_trip_says_so_with_numbers():
    """Below the floor the answer is a sentence, not an empty screen."""
    eng = get_engine()
    P = {p.place_id: p for p in eng.graph.places}
    o, d = P["pl_wipro_sarjapur"], P["pl_pes_university"]
    r = eng.recommend(JourneyRequest(
        origin_lat=o.lat, origin_lon=o.lon, origin_label=o.name,
        dest_lat=d.lat, dest_lon=d.lon, dest_label=d.name,
        departure=datetime(2026, 8, 28, 9, 0), budget=20.0,
        max_time_min=240.0, preference="balanced"))
    assert r.recommended is None
    assert not r.feasible
    assert r.message and "20" in r.message
    assert r.fallbacks, "nothing fits, and nothing was offered instead"
    for f in r.fallbacks:
        assert f["label"] and f["why"]
        assert f["journey"].cost > 20.0


def test_cheaper_budgets_shift_the_recommendation_away_from_hailed_rides():
    eng = get_engine()
    P = {p.place_id: p for p in eng.graph.places}
    o, d = P["pl_wipro_sarjapur"], P["pl_pes_university"]

    def modes_at(budget, max_time):
        r = eng.recommend(JourneyRequest(
            origin_lat=o.lat, origin_lon=o.lon, origin_label=o.name,
            dest_lat=d.lat, dest_lon=d.lon, dest_label=d.name,
            departure=datetime(2026, 8, 28, 9, 0), budget=float(budget),
            max_time_min=float(max_time), preference="balanced"))
        return {m for m in r.recommended.modes if m != "walk"}

    rich, poor = modes_at(400, 180), modes_at(150, 180)
    assert rich != poor, "budget makes no difference to the recommendation"
    assert "bike_taxi" not in poor or len(poor) > 1, (
        "even on a tight budget the planner offers only a bike taxi")


def test_multimodal_journeys_reach_the_booking_screen(client):
    """The gap that made the product look one-dimensional: the booking screen
    used to show single-vehicle rides only."""
    d = client.post("/api/compare", json=LONG).json()
    assert "journeys" in d
    assert d["journeys"], "no multimodal journey offered for a cross-city trip"
    for j in d["journeys"]:
        vehicles = {m for m in j["modes"] if m != "walk"}
        assert len(vehicles) >= 2, f"{j['shape']} is not multimodal"
        assert j["legs"] and j["fare_display"] and j["total_min"] > 0


def test_walking_is_never_offered_as_a_bookable_ride(client):
    """You do not book a walk."""
    d = client.post("/api/compare", json=LONG).json()
    for o in d["options"]:
        if o["service_class"] == "self":
            assert o["provider_id"] in ("walk", "cycle")


# ==========================================================================
# the demo script
# ==========================================================================
def test_demo_reproduces_the_cancellation_script(client):
    """Searching → Driver found → Driver accepted → Driver cancelled."""
    want = ["Searching for driver…", "Driver found",
            "Driver accepted your request", "Driver cancelled your ride"]
    for _ in range(3):
        r = client.post("/api/book", json={
            **LONG, "provider_id": "bike_taxi", "demo": True}).json()
        got = [s["label"] for s in r["attempt"]["steps"]]
        assert got == want, f"demo script drifted: {got}"


def test_a_scheduled_service_still_completes_in_demo_mode(client):
    """The cancellation seed must not be forced onto something that cannot
    cancel — a metro does not strand you."""
    c = client.post("/api/compare", json=LONG).json()
    scheduled = [o for o in c["options"]
                 if o["service_class"] == "scheduled" and o["available"]]
    if not scheduled:
        pytest.skip("no scheduled service on this trip")
    r = client.post("/api/book", json={
        **LONG, "provider_id": scheduled[0]["provider_id"], "demo": True}).json()
    assert r["attempt"]["outcome"] == "RIDE_COMPLETED"


def test_the_retry_budget_is_four(client):
    r = client.post("/api/book", json={
        **LONG, "provider_id": "bike_taxi", "demo": True}).json()
    assert r["session"]["max_attempts"] == 4
    assert r["session"]["attempts_left"] == 3


# ==========================================================================
# escalation
# ==========================================================================
def test_arrival_risk_uses_the_trip_clock_not_the_server_clock():
    """Regression: the projection once ran on the wall clock and reported a
    rider 543 minutes early for a meeting an hour after departure."""
    departure = datetime(2026, 8, 28, 9, 0)
    meeting = MeetingContext(title="the 10:00 review",
                             starts_at=datetime(2026, 8, 28, 10, 0))
    risk = assess_arrival(
        now=departure, wasted_min=0.0, meeting=meeting,
        best_option={"expected_minutes": 30.0, "display_name": "Cab"})
    assert 25 <= risk.minutes_spare <= 35, risk.minutes_spare
    assert risk.level == "on_track"


def test_lost_minutes_push_a_rider_from_on_track_to_late():
    meeting = MeetingContext(title="the 10:00 review",
                             starts_at=datetime(2026, 8, 28, 10, 0))
    best = {"expected_minutes": 50.0, "display_name": "Cab"}
    calm = assess_arrival(now=datetime(2026, 8, 28, 9, 0), wasted_min=0.0,
                          meeting=meeting, best_option=best)
    burned = assess_arrival(now=datetime(2026, 8, 28, 9, 25), wasted_min=25.0,
                            meeting=meeting, best_option=best)
    assert calm.level != "late"
    assert burned.level == "late"
    assert burned.minutes_spare < calm.minutes_spare


def test_escalation_offers_an_alternative_and_a_notification(client):
    r = client.post("/api/book", json={
        **LONG, "provider_id": "bike_taxi", "demo": True}).json()
    sid = r["session"]["session_id"]
    while r["session"]["can_retry"]:
        r = client.post(f"/api/book/{sid}/retry").json()

    e = client.get(f"/api/book/{sid}/escalation", params={
        "meeting": "the 10:15 review", "meeting_at": "2026-08-28T10:15:00",
        "manager": "Priya"}).json()
    assert e["risk"]["level"] in ("on_track", "at_risk", "late")
    assert e["notification_preview"]["delivery"] == "composed_not_sent"
    if e["alternative"]:
        assert e["alternative"]["provider_id"] != "bike_taxi"


def test_notify_is_never_reported_as_sent(client):
    """It composes and records. Claiming delivery would be a false statement
    about an action outside this system."""
    r = client.post("/api/book", json={
        **LONG, "provider_id": "bike_taxi", "demo": True}).json()
    sid = r["session"]["session_id"]
    n = client.post(f"/api/book/{sid}/notify", json={
        "meeting": "the 10:15 review", "meeting_at": "2026-08-28T10:15:00",
        "manager": "Priya"}).json()
    assert n["message"]["delivery"] == "composed_not_sent"
    assert "not transmitted" in n["message"]["delivery_note"]


def test_the_notification_does_not_claim_every_booking_failed(client):
    """Regression: it said 'none of them held' even after one held."""
    r = client.post("/api/book", json={
        **LONG, "provider_id": "bike_taxi", "demo": True}).json()
    sid = r["session"]["session_id"]
    while r["session"]["can_retry"]:
        r = client.post(f"/api/book/{sid}/retry").json()
    n = client.post(f"/api/book/{sid}/notify", json={}).json()
    if r["session"]["settled"]:
        assert "none of them held" not in n["message"]["body"]


def test_an_incident_is_recorded_without_an_employee_identity(client):
    r = client.post("/api/book", json={
        **LONG, "provider_id": "bike_taxi", "demo": True}).json()
    sid = r["session"]["session_id"]
    inc = client.post(f"/api/book/{sid}/notify", json={}).json()["incident"]
    assert inc["incident_id"].startswith("inc_")
    assert inc["severity"] in ("low", "medium", "high")
    assert inc["attempts"] >= 1 and inc["minutes_lost"] >= 0
    blob = " ".join(str(v).lower() for v in inc.values())
    for forbidden in ("employee_id", "email", "@", "staff_id"):
        assert forbidden not in blob, f"the incident leaks {forbidden!r}"


def test_escalation_is_audited(client):
    r = client.post("/api/book", json={
        **LONG, "provider_id": "bike_taxi", "demo": True}).json()
    sid = r["session"]["session_id"]
    client.post(f"/api/book/{sid}/notify", json={})
    log = client.get("/api/enterprise/audit?limit=50",
                     headers={"X-API-Key": "demo-analyst-key"}).json()
    kinds = {e["kind"] for e in log["entries"]}
    assert "escalation" in kinds, "notifying a manager was not recorded"


# ==========================================================================
# the reveal must not recommend something worse
# ==========================================================================
def test_the_crowned_alternative_is_not_hours_slower(client):
    """Regression, caught by looking at a screenshot rather than an assertion.

    On the 20 km demo route the metro is genuinely ₹25 and genuinely completes
    -- and takes three and a half hours. Crowning it as "what the engine
    picked" on the same screen that warns the rider they are late for a meeting
    is two contradictory recommendations at once.
    """
    r = client.post("/api/book", json={
        **LONG, "provider_id": "bike_taxi", "demo": True}).json()
    sid = r["session"]["session_id"]
    rev = client.get(f"/api/book/{sid}/reveal").json()
    chosen, better = rev["chosen"], rev["better"]
    if better:
        limit = chosen["expected_minutes"] * 1.5 + 15.0
        assert better["expected_minutes"] <= limit, (
            f"{better['display_name']} takes {better['expected_minutes']:.0f} "
            f"min against {chosen['expected_minutes']:.0f} — that is a "
            f"different trip, not a better answer")


def test_a_cheaper_but_slower_option_is_named_rather_than_hidden(client):
    """Filtering it out silently would be its own dishonesty."""
    r = client.post("/api/book", json={
        **LONG, "provider_id": "bike_taxi", "demo": True}).json()
    sid = r["session"]["session_id"]
    rev = client.get(f"/api/book/{sid}/reveal").json()
    if rev.get("cheaper_but_slower"):
        name = rev["cheaper_but_slower"]["display_name"]
        assert any(name in n for n in rev["narrative"]), (
            f"{name} was excluded on time but never mentioned")


def test_the_escalation_does_not_recommend_the_priciest_way_to_be_on_time(client):
    """Fastest-wins recommended a ₹543 cab over a ₹113 option eight minutes
    slower. Being on time is the constraint; cost is the objective."""
    r = client.post("/api/book", json={
        **LONG, "provider_id": "bike_taxi", "demo": True}).json()
    sid = r["session"]["session_id"]
    while r["session"]["can_retry"]:
        r = client.post(f"/api/book/{sid}/retry").json()
    e = client.get(f"/api/book/{sid}/escalation", params={
        "meeting_at": "2026-08-28T12:00:00"}).json()   # comfortably reachable
    alt = e["alternative"]
    assert alt, "no alternative offered"
    opts = {o["provider_id"]: o for o in client.post(
        "/api/compare", json=LONG).json()["options"]}
    # Anything that also arrives before noon and costs less would be better.
    spare = 180.0 - r["session"]["wasted_min"]
    cheaper_and_in_time = [
        o for o in opts.values()
        if o["available"] and o["provider_id"] != "bike_taxi"
        and o["expected"]["expected_minutes"] <= spare
        and o["expected"]["expected_cost"] < alt["expected_cost"] - 0.5
        and o["expected"] and o["expected"]["p_success"] >= 0.80]
    assert not cheaper_and_in_time, (
        f"{alt['display_name']} at ₹{alt['expected_cost']:.0f} was recommended "
        f"over " + ", ".join(f"{o['display_name']} ₹{o['expected']['expected_cost']:.0f}"
                             for o in cheaper_and_in_time))


def test_the_drafted_message_is_in_the_riders_voice(client):
    """It said "late for your next meeting" — to the manager, that is the
    manager's meeting."""
    r = client.post("/api/book", json={
        **LONG, "provider_id": "bike_taxi", "demo": True}).json()
    sid = r["session"]["session_id"]
    body = client.post(f"/api/book/{sid}/notify", json={}).json()["message"]["body"]
    assert "your next meeting" not in body
    # and the time is stated once, not once in the title and once in brackets
    assert body.count("(10:00)") <= 1


def test_the_attempt_budget_is_configurable_not_baked_in():
    """Four attempts is a demonstration choice, not a finding about riders."""
    from app.booking.session import MAX_ATTEMPTS, attempt_budget
    assert attempt_budget("2") == 2
    assert attempt_budget(None) == 4
    assert attempt_budget("") == 4
    assert attempt_budget("0") == 1          # a budget of zero is not a budget
    assert attempt_budget("nonsense") == 4   # bad config must not crash a demo
    assert MAX_ATTEMPTS == attempt_budget(os.getenv("JM_MAX_BOOKING_ATTEMPTS"))


def test_the_message_counts_attempts_in_english(client):
    """"I have tried 1 times" is exactly the sort of thing a manager notices."""
    r = client.post("/api/book", json={
        **LONG, "provider_id": "bike_taxi", "demo": True}).json()
    sid = r["session"]["session_id"]
    one = client.post(f"/api/book/{sid}/notify", json={}).json()["message"]["body"]
    assert "1 times" not in one
    assert "tried once" in one

    while r["session"]["can_retry"]:
        r = client.post(f"/api/book/{sid}/retry").json()
    many = client.post(f"/api/book/{sid}/notify", json={}).json()["message"]["body"]
    assert f"tried {len(r['session']['attempts'])} times" in many
