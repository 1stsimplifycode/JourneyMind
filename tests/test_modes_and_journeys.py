"""What JourneyMind will and will not put in front of a rider.

Six modes, no brands as modes, no walking as a commute, no forced multimodality
and no forced direct ride. Each test names the defect it guards.
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

from app.graph.builder import RequestGraph                            # noqa: E402
from app.graph.features import TimeContext                            # noqa: E402
from app.main import app                                              # noqa: E402
from app.models.loader import get_predictor                           # noqa: E402
from app.providers.simulated import ALL_PROVIDERS                     # noqa: E402
from app.routing.costs import build_cost_table                        # noqa: E402
from app.routing.journey import build_journey, deduplicate            # noqa: E402
from app.routing.kshortest import generate_candidates                 # noqa: E402
from app.routing.validate import (                                    # noqa: E402
    ALLOWED_MODES, MAX_JOURNEY_WALK_MIN, duplicates, partition_valid,
)
from app.services.engine import JourneyRequest, get_engine, primary_mode  # noqa: E402

PEAK = datetime(2026, 8, 28, 9, 0)
WIPRO = "Wipro Campus, Doddakannelli (Sarjapur Road)"
PES = "PES University, RR Campus (100 Feet Ring Road)"
LONG = {"origin": WIPRO, "destination": PES,
        "departure_time": "2026-08-28T09:00:00"}
SHORT = {"origin": "College (Shanthinagar)", "destination": "M.G. Road",
         "departure_time": "2026-08-28T09:00:00"}

OFFERED = {"bike_taxi", "auto", "cab", "metro", "bus"}


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


@pytest.fixture(scope="module")
def engine():
    return get_engine()


def places(engine):
    return {p.place_id: p for p in engine.graph.places}


def candidates(engine, o, d, departure=PEAK):
    rg = RequestGraph(engine.graph, (o.lat, o.lon), (d.lat, d.lon), None)
    costs = build_cost_table(rg, engine.graph, get_predictor(),
                             TimeContext.from_datetime(departure))
    js = deduplicate([
        build_journey(rg, costs, p.edges, engine.fares, f"J{i}", p.origin_blend)
        for i, p in enumerate(generate_candidates(rg, costs, 4, 3))])
    from app.data.geo import haversine_km
    straight = haversine_km(o.lat, o.lon, d.lat, d.lon)
    return partition_valid(js, straight)


def plan(engine, o, d, budget, max_time, preference="balanced", departure=PEAK):
    return engine.recommend(JourneyRequest(
        origin_lat=o.lat, origin_lon=o.lon, origin_label=o.name,
        dest_lat=d.lat, dest_lon=d.lon, dest_label=d.name,
        departure=departure, budget=float(budget), max_time_min=float(max_time),
        preference=preference))


# ==========================================================================
# the mode vocabulary
# ==========================================================================
def test_the_product_offers_exactly_six_modes():
    assert ALLOWED_MODES == OFFERED
    assert {p.mode for p in ALL_PROVIDERS} == OFFERED


def test_a_brand_is_never_a_mode():
    """Rapido is how you book a bike taxi. Namma Yatri is how you book an auto.

    Both used to BE modes in the graph, which brand-locked the router and made
    one vehicle appear twice in every comparison.
    """
    for p in ALL_PROVIDERS:
        assert p.mode in OFFERED
        assert p.mode not in {"rapido", "namma_yatri"}
        assert p.provider_name and p.provider_name != p.display_name or \
            p.mode in {"metro", "bus", "cab"} or p.provider_name


def test_two_providers_share_the_auto(client):
    """The clearest case for separating mode from provider."""
    d = client.post("/api/compare", json=LONG).json()
    autos = [o for o in d["options"] if o["mode"] == "auto"]
    assert len(autos) == 2
    assert {o["provider_name"] for o in autos} == {"Metered auto", "Namma Yatri"}
    assert {o["display_name"] for o in autos} == {"Auto"}


@pytest.mark.parametrize("gone", ["carpool", "walk", "cycle"])
def test_a_removed_mode_is_gone_from_every_surface(client, gone):
    for payload in (client.post("/api/compare", json=LONG).json(),
                    client.post("/api/compare", json=SHORT).json()):
        assert gone not in {o["provider_id"] for o in payload["options"]}
        assert gone not in {o["mode"] for o in payload["options"]}
        for j in payload["journeys"]:
            assert gone not in j["modes"]
    assert gone not in {p.provider_id for p in ALL_PROVIDERS}
    assert gone not in {r["provider_id"] for r in
                        client.get("/api/providers").json()["providers"]}


def test_carpool_history_is_excluded_from_the_enterprise_view(client):
    """A scorecard row for a mode nobody can book is not an insight."""
    from app.enterprise.analytics import load_bookings
    t = load_bookings()
    assert t.excluded_rows > 0, "the bundled history has no carpool to exclude?"
    d = client.get("/api/enterprise/overview",
                   headers={"X-API-Key": "demo-analyst-key"}).json()
    blob = str(d).lower()
    assert "carpool" not in blob


# ==========================================================================
# walking is access, not a commute
# ==========================================================================
def test_no_recommendation_anywhere_contains_a_walking_leg(engine):
    P = list(engine.graph.places)
    for o, d in list(itertools.permutations(P, 2))[::6]:
        kept, _ = candidates(engine, o, d)
        for j in kept:
            assert "walk" not in j.modes, f"{o.name} -> {d.name}: {j.shape()}"
            for lg in j.legs:
                assert lg.mode != "walk"


def test_the_walking_that_remains_is_a_station_approach(engine):
    P = list(engine.graph.places)
    for o, d in list(itertools.permutations(P, 2))[::6]:
        kept, _ = candidates(engine, o, d)
        for j in kept:
            assert j.walk_min <= MAX_JOURNEY_WALK_MIN, (
                f"{o.name} -> {d.name}: {j.walk_min:.0f} min on foot")


def test_absorbed_walking_is_still_counted_in_the_total(engine):
    """Folding a walk into a leg must not make its minutes disappear."""
    P = places(engine)
    kept, _ = candidates(engine, P["pl_wipro_sarjapur"], P["pl_pes_university"])
    for j in kept:
        assert abs(sum(l.total_min for l in j.legs) - j.total_min) < 0.05
        assert abs(sum(l.total_km for l in j.legs) - j.distance_km) < 0.02
        walked = sum(l.access_min for l in j.legs)
        assert abs(walked - j.walk_min) < 0.05 or j.walk_min == 0


def test_a_fare_is_never_charged_for_walking(engine):
    """Access minutes go into the time, never into the meter."""
    P = places(engine)
    kept, _ = candidates(engine, P["pl_wipro_sarjapur"], P["pl_pes_university"])
    for j in kept:
        for lg in j.legs:
            if lg.access_km <= 0 or lg.fare is None:
                continue
            priced = engine.fares.leg_fare(lg.mode, lg.distance_km,
                                           lg.travel_min + lg.wait_min)
            if len(lg.segments) <= 1:
                assert lg.fare.amount == pytest.approx(priced.amount, abs=0.01)


# ==========================================================================
# neither shape is forced
# ==========================================================================
def test_a_direct_ride_wins_when_it_deserves_to(engine):
    """On a short hop one vehicle is the whole answer.

    Not asserted on the long corridor: there the metro option is cheaper and
    quicker than the direct ride, and insisting on one vehicle would force the
    shape the brief says never to force.
    """
    P = places(engine)
    r = plan(engine, P["pl_college"], P["pl_mg_road_shops"], 400, 180)
    assert r.recommended.transfers == 0
    assert r.recommended.shape() == ["bike_taxi"]


def test_multimodal_wins_when_it_deserves_to(engine):
    P = places(engine)
    r = plan(engine, P["pl_wipro_sarjapur"], P["pl_pes_university"], 150, 150)
    assert len(set(r.recommended.modes)) >= 2
    assert "metro" in r.recommended.modes or "bus" in r.recommended.modes


def test_between_two_equivalent_answers_the_simpler_one_wins(engine):
    """A rupee and a minute apart is the same answer to a rider."""
    from app.optimisation.scoring import (
        SIMPLICITY_COST_BAND, SIMPLICITY_TIME_BAND, _prefer_the_simpler_winner)

    class Fake:
        def __init__(self, cost, mins, transfers, score):
            self.cost, self.total_min = cost, mins
            self.transfers, self.score = transfers, score

    complicated = Fake(100.0, 60.0, 3, 0.10)
    simple = Fake(104.0, 62.0, 0, 0.11)
    assert _prefer_the_simpler_winner([complicated, simple])[0] is simple

    # ...but a real trade-off is left to the weights
    much_dearer = Fake(100.0 + SIMPLICITY_COST_BAND + 20, 62.0, 0, 0.11)
    assert _prefer_the_simpler_winner([complicated, much_dearer])[0] is complicated
    much_slower = Fake(104.0, 60.0 + SIMPLICITY_TIME_BAND + 10, 0, 0.11)
    assert _prefer_the_simpler_winner([complicated, much_slower])[0] is complicated


def test_a_journey_is_named_by_its_transit_spine(engine):
    """"Bike taxi, metro, bike taxi" is how you take the metro."""
    P = places(engine)
    kept, _ = candidates(engine, P["pl_wipro_sarjapur"], P["pl_pes_university"])
    # exactly one transit mode: that is the spine. Two of them (bike taxi,
    # bus, metro, bike taxi) is a genuinely mixed itinerary with no single
    # name, which is why it belongs in the stages list and not on a card.
    spined = [j for j in kept
              if len({m for m in j.modes if m in ("metro", "bus")}) == 1]
    assert spined
    for j in spined:
        transit = next(m for m in j.modes if m in ("metro", "bus"))
        assert primary_mode(j) == transit, j.shape()
    mixed = [j for j in kept
             if len({m for m in j.modes if m in ("metro", "bus")}) > 1]
    for j in mixed:
        assert primary_mode(j) is None, j.shape()


# ==========================================================================
# the cards price the whole journey
# ==========================================================================
def test_a_metro_card_prices_the_ride_to_the_station(client):
    """₹25 for a journey that also needs two bike taxis is not the price."""
    d = client.post("/api/compare", json=LONG).json()
    metro = next(o for o in d["options"] if o["mode"] == "metro")
    assert metro["available"], metro["unavailable_reason"]
    assert metro["fare"]["amount"] > 25.0
    assert metro["door_to_door_min"] < 150
    assert any("door to door" in n.lower() for n in metro["notes"]), metro["notes"]


def test_a_metro_reached_by_bike_taxi_does_not_claim_certainty(client):
    """The train is certain. The booking that gets you to it is not."""
    d = client.post("/api/compare", json=LONG).json()
    metro = next(o for o in d["options"] if o["mode"] == "metro")
    assert metro["expected"]["p_success"] < 1.0
    assert metro["reliability"]["p_cancel"] > 0.0
    assert "hailed" in metro["reliability"]["basis"]


def test_an_option_is_never_its_own_fallback(client):
    """The cost of the bus failing was the cost of taking the bus."""
    d = client.post("/api/compare", json={**LONG, "rain": True}).json()
    for o in d["options"]:
        if o["expected"] and o["expected"].get("fallback_label"):
            assert o["expected"]["fallback_label"] != o["display_name"]


def test_the_crossover_the_product_exists_to_show_still_happens(client):
    """Advertised price is not the price of a journey that works."""
    d = client.post("/api/compare", json={**LONG, "rain": True,
                                          "departure_time": "2026-08-28T18:30:00"}).json()
    avail = [o for o in d["options"] if o["available"]]
    cheapest = min(avail, key=lambda o: o["fare"]["amount"])
    crossover = [o for o in avail
                 if o["fare"]["amount"] > cheapest["fare"]["amount"] + 0.5
                 and o["expected"]["expected_cost"]
                 < cheapest["expected"]["expected_cost"] - 0.5]
    assert crossover, (
        f"{cheapest['display_name']} advertises "
        f"₹{cheapest['fare']['amount']:.0f} and is expected to cost "
        f"₹{cheapest['expected']['expected_cost']:.0f}; nothing dearer costs less")


# ==========================================================================
# budgets, including small ones
# ==========================================================================
@pytest.mark.parametrize("budget", [5, 10, 20, 50, 100, 150, 250, 400])
def test_every_budget_gets_an_answer_or_a_reason(engine, budget):
    """"No options" is not an answer. Either a journey, or why not and what
    the cheapest one actually costs."""
    P = places(engine)
    r = plan(engine, P["pl_wipro_sarjapur"], P["pl_pes_university"],
             budget, 240)
    if r.recommended is not None:
        assert r.recommended.cost <= budget + 1e-9
        assert "walk" not in r.recommended.modes
        return
    assert r.message and str(budget) in r.message
    assert r.fallbacks, "nothing fitted and nothing was offered instead"
    cheapest = min(f["journey"].cost for f in r.fallbacks)
    assert cheapest > budget, "a fallback that fits should have been recommended"


def test_the_planner_reaches_for_transit_as_the_budget_tightens(engine):
    P = places(engine)
    rich = plan(engine, P["pl_wipro_sarjapur"], P["pl_pes_university"], 400, 240)
    poor = plan(engine, P["pl_wipro_sarjapur"], P["pl_pes_university"], 150, 240)
    assert poor.recommended.cost < rich.recommended.cost
    assert len(set(poor.recommended.modes)) > len(set(rich.recommended.modes))


def test_a_hailed_vehicle_can_reach_a_stop_not_only_a_hub(engine):
    """The nearest ride hub to the Wipro gate is 6.7 km away, so every cheap
    itinerary used to begin with a half-hour walk. There are bus stops 500 m
    from that gate."""
    P = places(engine)
    o = P["pl_wipro_sarjapur"]
    rg = RequestGraph(engine.graph, (o.lat, o.lon),
                      (P["pl_pes_university"].lat, P["pl_pes_university"].lon), None)
    from app.graph.builder import ORIGIN_ID
    hops = [e for e in rg.edges
            if e.kind == "ride" and e.u == ORIGIN_ID and e.distance_km < 2.0]
    assert hops, "no short first-mile ride exists at all"
    assert any(engine.graph.nodes[e.v].kind in ("bus_stop", "metro_station")
               for e in hops)


# ==========================================================================
# the validator
# ==========================================================================
def test_a_detour_is_not_a_route(client):
    """A 78 m trip was answered with a 1 km bike taxi that looped out to a bus
    stop and back, because that was the only ride edge in reach."""
    r = client.post("/api/compare", json={
        "origin": "12.9345,77.6100", "destination": "12.9350,77.6105",
        "departure_time": "2026-08-28T09:00:00"})
    if r.status_code == 422:
        assert r.json()["detail"]["error"]
        return
    d = r.json()
    for o in d["options"]:
        if o["available"] and o["distance_km"]:
            assert o["distance_km"] < 0.7, o["display_name"]


def test_no_two_journeys_offered_are_the_same_trip(engine):
    P = list(engine.graph.places)
    for o, d in list(itertools.permutations(P, 2))[::9]:
        kept, _ = candidates(engine, o, d)
        dupes = duplicates(kept)
        assert not dupes, f"{o.name} -> {d.name}: {[x[0].shape() for x in dupes]}"


def test_the_booking_screen_never_repeats_a_card_as_an_itinerary(client):
    """The Metro card IS the bike-taxi-metro-bike-taxi journey now."""
    d = client.post("/api/compare", json=LONG).json()
    carded = {o["mode"] for o in d["options"] if o["available"]}
    for j in d["journeys"]:
        vehicles = set(j["modes"])
        assert len(vehicles) >= 2
        # an itinerary must not be exactly what one card already describes
        transit = vehicles & {"metro", "bus"}
        assert not (len(transit) == 1 and vehicles - transit == {"bike_taxi"}
                    and next(iter(transit)) in carded
                    and len(j["legs"]) == 3), j["shape"]
