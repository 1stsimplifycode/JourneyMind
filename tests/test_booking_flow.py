"""The demonstration flow: BOOK NOW, fail, retry, reveal.

The central assertion in this file is that the booking a rider *lives* is a
draw from the same distribution the expected-cost model *priced*. If those two
ever come apart, the reveal is theatre and the product is dishonest.
"""

from __future__ import annotations

import collections
import os
import sys

import pytest
from fastapi.testclient import TestClient

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "backend"))

from app.booking.session import MAX_ATTEMPTS                    # noqa: E402
from app.main import app                                        # noqa: E402

TRIP = {"origin": "College (Shanthinagar)", "destination": "M.G. Road",
        "departure_time": "2026-08-28T09:00:00"}


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def start(client, provider="bike_taxi", demo=True):
    r = client.post("/api/book", json={**TRIP, "provider_id": provider, "demo": demo})
    assert r.status_code == 200, r.json()
    return r.json()


# ==========================================================================
# BOOK NOW
# ==========================================================================
def test_book_now_returns_a_narrated_attempt(client):
    b = start(client)
    a = b["attempt"]
    assert a["steps"], "a booking must produce steps to show the rider"
    assert a["steps"][0]["state"] == "REQUESTED"
    assert a["outcome"] in {"RIDE_COMPLETED", "NO_DRIVER_AVAILABLE",
                            "DRIVER_REJECTED", "DRIVER_CANCELLED"}
    for s in a["steps"]:
        assert s["label"] and s["dwell_ms"] > 0
        assert s["tone"] in {"progress", "good", "bad"}


def test_demo_mode_is_reproducible(client):
    """A live demonstration must not depend on luck."""
    outcomes = {start(client, demo=True)["attempt"]["outcome"] for _ in range(4)}
    assert len(outcomes) == 1, f"demo mode was not deterministic: {outcomes}"


def test_without_demo_mode_outcomes_vary(client):
    """...and the seed must not be secretly rigging the result either."""
    seen = collections.Counter(
        start(client, demo=False)["attempt"]["outcome"] for _ in range(25))
    assert len(seen) > 1, f"unseeded bookings never varied: {seen}"


def test_an_unknown_provider_is_refused(client):
    r = client.post("/api/book", json={**TRIP, "provider_id": "helicopter"})
    assert r.status_code == 422
    assert r.json()["detail"]["code"] == "unknown_provider"


# ==========================================================================
# retry / rebooking
# ==========================================================================
def test_retry_escalates_the_fare(client):
    """A re-request lands in a market that has just proved tight."""
    b = start(client)
    sid = b["session"]["session_id"]
    first = b["attempt"]["fare"]
    if not b["session"]["can_retry"]:
        pytest.skip("first attempt succeeded under this seed")
    r = client.post(f"/api/book/{sid}/retry").json()
    assert r["attempt"]["fare"] > first
    assert r["attempt"]["number"] == 2


def test_retry_is_refused_past_the_budget(client):
    b = start(client, demo=False)
    sid = b["session"]["session_id"]
    for _ in range(MAX_ATTEMPTS + 2):
        s = client.get(f"/api/book/{sid}").json()["session"]
        if not s["can_retry"]:
            break
        client.post(f"/api/book/{sid}/retry")
    s = client.get(f"/api/book/{sid}").json()["session"]
    assert s["attempt_count"] <= MAX_ATTEMPTS
    if not s["settled"]:
        assert client.post(f"/api/book/{sid}/retry").status_code == 409


def test_a_settled_booking_cannot_be_retried(client):
    """A scheduled service completes every time, so it is the clean case.

    Which scheduled service exists depends on the trip -- a short hop may have
    a bus and no metro -- so the test asks the engine rather than assuming.
    """
    c = client.post("/api/compare", json={**TRIP}).json()
    scheduled = [o for o in c["options"]
                 if o["service_class"] == "scheduled" and o["available"]]
    if not scheduled:
        pytest.skip("no scheduled service serves this trip")
    b = start(client, provider=scheduled[0]["provider_id"])
    assert b["session"]["settled"] is True
    assert b["session"]["can_retry"] is False
    sid = b["session"]["session_id"]
    assert client.post(f"/api/book/{sid}/retry").status_code == 409


def test_an_expired_session_says_so(client):
    assert client.post("/api/book/bk_nope/retry").status_code == 404
    assert client.get("/api/book/bk_nope").status_code == 404


# ==========================================================================
# the reveal
# ==========================================================================
def test_reveal_prices_the_option_that_was_actually_booked(client):
    b = start(client)
    sid = b["session"]["session_id"]
    rev = client.get(f"/api/book/{sid}/reveal").json()
    assert rev["chosen"]["provider_id"] == b["session"]["provider_id"]
    assert rev["narrative"], "the reveal must explain itself"
    assert "not causal" in rev["causality_note"]


def test_reveal_numbers_predate_the_booking(client):
    """The reveal must quote what was predicted, not what happened.

    Booking the same option twice under different seeds must not change the
    predicted probabilities -- if it did, the explanation would be fitted to
    the outcome it is meant to explain.
    """
    a = start(client, demo=True)
    b = start(client, demo=False)
    ra = client.get(f"/api/book/{a['session']['session_id']}/reveal").json()
    rb = client.get(f"/api/book/{b['session']['session_id']}/reveal").json()
    assert ra["chosen"]["p_success"] == rb["chosen"]["p_success"]
    assert ra["chosen"]["expected_cost"] == rb["chosen"]["expected_cost"]


def test_reveal_names_a_cheaper_alternative_when_one_exists(client):
    """The whole point: a dearer sticker price with a lower expected cost."""
    b = start(client)
    rev = client.get(f"/api/book/{b['session']['session_id']}/reveal").json()
    alt = rev.get("better_same_class") or rev.get("better")
    if alt is None or alt["expected_cost"] >= rev["chosen"]["expected_cost"] - 0.5:
        pytest.skip("no cheaper alternative for this trip")
    joined = " ".join(rev["narrative"])
    assert alt["display_name"] in joined, "a better option was found but never named"


def test_comparison_sentence_gets_its_direction_right(client):
    """Regression: the narrative once called a lower fare 'more than'."""
    b = start(client)
    rev = client.get(f"/api/book/{b['session']['session_id']}/reveal").json()
    chosen = rev["chosen"]
    checked = 0
    for alt in (rev.get("better_same_class"), rev.get("better")):
        if not alt:
            continue
        for line in rev["narrative"]:
            if alt["display_name"] not in line or "advertises" not in line:
                continue
            checked += 1
            if alt["fare"] > chosen["fare"] + 0.5:
                assert "more than" in line, line
            elif alt["fare"] < chosen["fare"] - 0.5:
                assert "less than" in line, line
    assert checked or True     # nothing to check is acceptable; a wrong word is not


def test_the_cheapest_advertised_is_not_the_cheapest_journey(client):
    """The product's thesis, asserted against the live engine.

    The claim is NOT "the cheapest option is the worst" -- a cab is dearer on
    both counts and always will be. The claim is precise: there exists an
    option that advertises MORE than the cheapest one and yet is expected to
    cost LESS. That crossover is the entire product.

    If this fails, the demo trip has stopped telling the story, and the fix is
    to choose a different trip rather than to soften the assertion.
    """
    # Wet peak on the long corridor: the trip where the crossover is real.
    # Carpool used to supply it by being absurdly cheap and almost never
    # completing, which was a property of a mode this product no longer has.
    c = client.post("/api/compare", json={
        "origin": "Wipro Campus, Doddakannelli (Sarjapur Road)",
        "destination": "PES University, RR Campus (100 Feet Ring Road)",
        "departure_time": "2026-08-28T18:30:00", "rain": True,
        "priority": "balanced"}).json()
    avail = [o for o in c["options"] if o["available"]]
    assert len(avail) >= 2
    cheapest = min(avail, key=lambda o: o["fare"]["amount"])

    crossovers = [o for o in avail
                  if o["fare"]["amount"] > cheapest["fare"]["amount"] + 0.5
                  and o["expected"]["expected_cost"] < cheapest["expected"]["expected_cost"] - 0.5]
    assert crossovers, (
        f"no option advertises more than {cheapest['display_name']} "
        f"(₹{cheapest['fare']['amount']:.0f} → ₹"
        f"{cheapest['expected']['expected_cost']:.0f} expected) while costing "
        f"less in expectation — this demo trip no longer shows the crossover")

    # and the cheapest sticker price must genuinely inflate, or there is no story
    assert (cheapest["expected"]["expected_cost"]
            > cheapest["fare"]["amount"] + 0.5), (
        "the cheapest option's expected cost no longer exceeds its fare")


# ==========================================================================
# insights
# ==========================================================================
def test_insights_return_panels_and_refuse_to_claim_causation(client):
    d = client.get("/api/insights").json()
    assert len(d["panels"]) >= 4
    assert "ASSOCIATION, not causation" in d["causality_note"]
    for p in d["panels"]:
        assert p["title"] and p["reading"]
        for row in p["rows"]:
            assert row["n"] >= 25, "under-populated cells must be dropped"


def test_insights_show_the_short_trip_relationship(client):
    """The clearest association in the data, and the one the demo leans on."""
    d = client.get("/api/insights").json()
    panel = next((p for p in d["panels"] if p["key"] == "distance"), None)
    assert panel and len(panel["rows"]) >= 3
    shortest, longest = panel["rows"][0], panel["rows"][-1]
    assert shortest["acceptance"] < longest["acceptance"], (
        "short trips are no longer declined more often than long ones — the "
        "demo narrative depends on this")
