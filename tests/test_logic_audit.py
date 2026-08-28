"""The rules a journey has to obey to be shown to anybody.

Every test here corresponds to a defect that was found in the running system,
not to a rule invented in the abstract. The comment above each one says what it
was. They are grouped the way the pipeline runs:

    graph -> candidates -> validation -> constraints -> ranking -> presentation
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

from app.data.geo import ROAD_DETOUR, haversine_km                   # noqa: E402
from app.graph.builder import (                                      # noqa: E402
    DESTINATION_ID, ORIGIN_ID, RequestGraph,
)
from app.graph.features import TimeContext                           # noqa: E402
from app.main import app                                             # noqa: E402
from app.models.loader import get_predictor                          # noqa: E402
from app.routing.costs import build_cost_table                       # noqa: E402
from app.routing.journey import build_journey, deduplicate           # noqa: E402
from app.routing.kshortest import generate_candidates                # noqa: E402
from app.routing.validate import (                                   # noqa: E402
    HAILED_MODES, partition_valid, validate_journey,
)
from app.services.engine import JourneyRequest, get_engine           # noqa: E402

PEAK = datetime(2026, 8, 28, 9, 0)
WIPRO = "Wipro Campus, Doddakannelli (Sarjapur Road)"
PES = "PES University, RR Campus (100 Feet Ring Road)"
LONG = {"origin": WIPRO, "destination": PES,
        "departure_time": "2026-08-28T09:00:00"}
SHORT = {"origin": "College (Shanthinagar)", "destination": "M.G. Road",
         "departure_time": "2026-08-28T09:00:00"}


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


@pytest.fixture(scope="module")
def engine():
    return get_engine()


def candidates(engine, o, d, departure=PEAK):
    """Every journey the search produces for one trip, before ranking."""
    rg = RequestGraph(engine.graph, (o.lat, o.lon), (d.lat, d.lon), None)
    costs = build_cost_table(rg, engine.graph, get_predictor(),
                             TimeContext.from_datetime(departure))
    paths = generate_candidates(rg, costs, 4, 3)
    return deduplicate([
        build_journey(rg, costs, p.edges, engine.fares, f"J{i}", p.origin_blend)
        for i, p in enumerate(paths)])


def places(engine):
    return {p.place_id: p for p in engine.graph.places}


def plan(engine, o, d, budget, max_time, preference="balanced", departure=PEAK):
    return engine.recommend(JourneyRequest(
        origin_lat=o.lat, origin_lon=o.lon, origin_label=o.name,
        dest_lat=d.lat, dest_lon=d.lon, dest_label=d.name,
        departure=departure, budget=float(budget), max_time_min=float(max_time),
        preference=preference))


# ==========================================================================
# the graph: a direct ride always exists
# ==========================================================================
def test_a_direct_ride_exists_for_every_pair_of_places(engine):
    """The cap on a first/last-mile hop was applied to the door-to-door ride.

    Twelve of the 210 place pairs -- including the demonstration route -- lost
    their direct ride entirely, and the router replaced it with two hailed
    vehicles in a row.
    """
    P = list(engine.graph.places)
    missing = []
    for o, d in itertools.permutations(P, 2):
        rg = RequestGraph(engine.graph, (o.lat, o.lon), (d.lat, d.lon), None)
        direct = {e.mode for e in rg.edges
                  if e.kind == "ride" and e.u == ORIGIN_ID and e.v == DESTINATION_ID}
        if not direct:
            km = haversine_km(o.lat, o.lon, d.lat, d.lon) * ROAD_DETOUR
            missing.append(f"{o.name} -> {d.name} ({km:.1f} km)")
    assert not missing, "no door-to-door ride offered for:\n  " + "\n  ".join(missing)


def test_splitting_a_ride_in_two_does_not_make_it_faster(engine):
    """Ride time was scaled by the congestion at the two ENDPOINTS only.

    A long edge was scored on its ends while the same ground split across a hub
    picked up that hub's reading, so two hops could beat one direct ride. That
    non-additivity is what manufactured the two-vehicle journeys.
    """
    P = places(engine)
    o, d = P["pl_wipro_sarjapur"], P["pl_pes_university"]
    rg = RequestGraph(engine.graph, (o.lat, o.lon), (d.lat, d.lon), None)
    costs = build_cost_table(rg, engine.graph, get_predictor(),
                             TimeContext.from_datetime(PEAK))

    direct = next(e for e in rg.edges if e.kind == "ride" and e.mode == "bike_taxi"
                  and e.u == ORIGIN_ID and e.v == DESTINATION_ID)
    direct_min = costs.travel_min(direct.idx, 0.0)

    # every hub you could break the trip at, as a two-hop alternative
    hops = {}
    for e in rg.edges:
        if e.kind != "ride" or e.mode != "bike_taxi":
            continue
        if e.u == ORIGIN_ID and e.v != DESTINATION_ID:
            hops.setdefault(e.v, {})["out"] = e
        elif e.v == DESTINATION_ID and e.u != ORIGIN_ID:
            hops.setdefault(e.u, {})["in"] = e

    for hub, pair in hops.items():
        if "out" not in pair or "in" not in pair:
            continue
        two_hop = (costs.travel_min(pair["out"].idx, 0.0)
                   + costs.travel_min(pair["in"].idx, 0.0))
        detour = pair["out"].distance_km + pair["in"].distance_km - direct.distance_km
        if detour > 0.5:
            continue        # a genuinely longer road may honestly take longer
        assert two_hop >= direct_min - 2.0, (
            f"breaking the trip at {hub} makes it {direct_min - two_hop:.1f} min "
            f"FASTER for no extra distance — travel time is not additive")


# ==========================================================================
# the validator
# ==========================================================================
def test_no_candidate_hails_the_same_vehicle_twice_in_a_row(engine):
    """You would have stayed in the first one."""
    P = list(engine.graph.places)
    for o, d in list(itertools.permutations(P, 2))[::7]:
        kept, _ = partition_valid(candidates(engine, o, d))
        for j in kept:
            for a, b in zip(j.legs, j.legs[1:]):
                assert not (a.mode == b.mode and a.mode in HAILED_MODES), (
                    f"{o.name} -> {d.name}: {'>'.join(l.mode for l in j.legs)}")


def test_every_surviving_candidate_is_physically_continuous(engine):
    P = list(engine.graph.places)
    for o, d in list(itertools.permutations(P, 2))[::7]:
        kept, _ = partition_valid(candidates(engine, o, d))
        for j in kept:
            for a, b in zip(j.legs, j.legs[1:]):
                assert a.to_node == b.from_node, (
                    f"{o.name} -> {d.name}: leg {a.index} ends at {a.to_name}, "
                    f"leg {b.index} starts at {b.from_name}")


def test_every_surviving_candidate_adds_up(engine):
    """Legs must sum to the journey, in time, distance and fare band."""
    P = list(engine.graph.places)
    for o, d in list(itertools.permutations(P, 2))[::7]:
        kept, _ = partition_valid(candidates(engine, o, d))
        for j in kept:
            assert abs(sum(l.total_min for l in j.legs) - j.total_min) < 0.05
            assert abs(sum(l.total_km for l in j.legs) - j.distance_km) < 0.02
            assert j.total_cost.low - 0.51 <= j.cost <= j.total_cost.high + 0.51
            boardings = sum(len(l.segments) if l.segments else 1
                            for l in j.legs if l.kind in ("transit", "ride"))
            assert j.transfers == max(0, boardings - 1)
            assert j.total_min > 0


def test_an_interchange_lives_inside_one_leg(engine):
    """Changing from the Yellow line to the Green line is one metro journey.

    It used to be two legs, which read as "Metro -> Metro" everywhere it was
    summarised. Now it is one leg with two `segments`: the services keep their
    names and their per-boarding fares, and no summary claims two trains.
    """
    P = list(engine.graph.places)
    saw_interchange = False
    for o, d in list(itertools.permutations(P, 2))[::5]:
        kept, _ = partition_valid(candidates(engine, o, d))
        for j in kept:
            for a, b in zip(j.legs, j.legs[1:]):
                assert a.mode != b.mode, (
                    f"{o.name} -> {d.name}: two {a.mode} legs side by side")
            for lg in j.legs:
                if len(lg.segments) > 1:
                    saw_interchange = True
                    routes = [sg["route_id"] for sg in lg.segments]
                    assert len(set(routes)) == len(routes), (
                        f"one service split across segments on {lg.route_name}")
                    assert all(sg["route_name"] for sg in lg.segments)
    assert saw_interchange, "no interchange anywhere — the check proved nothing"


def test_the_summary_never_says_a_mode_twice_in_a_row(engine):
    """"Metro → Metro" describes an interchange as two separate trains."""
    P = list(engine.graph.places)
    for o, d in list(itertools.permutations(P, 2))[::7]:
        kept, _ = partition_valid(candidates(engine, o, d))
        for j in kept:
            shape = j.shape()
            for a, b in zip(shape, shape[1:]):
                assert a != b, f"{o.name} -> {d.name}: {' → '.join(shape)}"


def test_a_leg_that_is_the_whole_trip_makes_the_others_decoration(engine):
    P = list(engine.graph.places)
    for o, d in list(itertools.permutations(P, 2))[::7]:
        kept, _ = partition_valid(candidates(engine, o, d))
        for j in kept:
            vehicles = [l for l in j.legs if l.mode != "walk"]
            if len(vehicles) < 2 or j.distance_km <= 0:
                continue
            biggest = max(l.total_km for l in vehicles) / j.distance_km
            assert biggest < 0.85, (
                f"{o.name} -> {d.name}: one leg is {biggest:.0%} of the "
                f"distance and the transfers earn nothing")


def test_the_validator_actually_rejects_a_broken_journey(engine):
    """A gate that never closes is not a gate."""
    P = places(engine)
    js = candidates(engine, P["pl_wipro_sarjapur"], P["pl_pes_university"])
    j = js[0]

    j.total_min += 30.0                       # claim a duration the legs deny
    codes = {v.code for v in validate_journey(j)}
    assert "time_mismatch" in codes
    j.total_min -= 30.0
    assert not [v for v in validate_journey(j) if v.fatal]


def test_rejections_are_recorded_rather_than_silent(client):
    """A candidate set that loses half its members must be visible."""
    d = client.post("/api/compare", json=LONG).json()
    val = d["pipeline"]["engine"].get("candidates", {})
    assert "after_validation" in val
    assert val["after_validation"] <= val["after_deduplication"]


# ==========================================================================
# direct versus multimodal: neither is forced
# ==========================================================================
def test_a_direct_ride_can_win(engine):
    """Sometimes the smartest recommendation is one vehicle.

    Asserted on a trip where a direct ride genuinely dominates rather than on
    the long corridor, where it does not: there the metro option is both
    cheaper and quicker, and making the direct ride win anyway would be the
    forced-direct-ride failure in the other direction.
    """
    P = places(engine)
    r = plan(engine, P["pl_college"], P["pl_mg_road_shops"], 400, 180)
    assert r.recommended.shape() == ["bike_taxi"], r.recommended.shape()
    assert r.recommended.transfers == 0


def test_the_recommendation_is_never_dominated(engine):
    """Whatever wins, nothing on the same screen beats it on BOTH axes.

    The rule the corridor-specific assertion was trying to express, stated so
    that it holds everywhere: a journey that is cheaper *and* faster than the
    recommendation means the ranking got it wrong, whatever shape either is.
    """
    P = list(engine.graph.places)
    for o, d in list(itertools.permutations(P, 2))[::11]:
        r = plan(engine, o, d, 400, 180)
        best = r.recommended
        if best is None:
            continue
        for j in r.candidates:
            if j.signature == best.signature:
                continue
            assert not (j.cost < best.cost - 0.5
                        and j.total_min < best.total_min - 0.5), (
                f"{o.name} -> {d.name}: {' > '.join(j.shape())} at "
                f"₹{j.cost:.0f}/{j.total_min:.0f}min beats the recommended "
                f"{' > '.join(best.shape())} at ₹{best.cost:.0f}/"
                f"{best.total_min:.0f}min on both")


def test_multimodal_can_win(engine):
    """...and sometimes it is three."""
    P = places(engine)
    r = plan(engine, P["pl_wipro_sarjapur"], P["pl_pes_university"], 150, 150)
    vehicles = {m for m in r.recommended.modes if m != "walk"}
    assert len(vehicles) >= 2, (
        f"a tight budget was still answered with a single "
        f"{' > '.join(r.recommended.shape())}")


def test_a_trip_too_short_to_ride_fails_gracefully(client):
    """A hundred metres is a walk, and walking is not a mode we offer.

    The honest answer is that none of the six modes serves this trip -- said in
    a sentence, with a reason on every row, rather than by inventing an option
    or returning an empty screen.
    """
    r = client.post("/api/compare", json={
        "origin": "12.9345,77.6100", "destination": "12.9350,77.6105",
        "departure_time": "2026-08-28T09:00:00"})
    if r.status_code == 422:
        assert r.json()["detail"]["error"]
        return
    d = r.json()
    assert d["recommended_provider"] is None
    assert d["headline"]
    for o in d["options"]:
        assert not o["available"]
        assert o["unavailable_reason"]


def test_walking_is_never_a_leg_a_rider_is_shown(client):
    """Walking is how you reach a vehicle here, not how you travel.

    It stays in the graph -- there is no other way onto a metro platform -- but
    it is folded into the leg it serves and never named as a step.
    """
    for trip in (SHORT, LONG):
        d = client.post("/api/compare", json=trip).json()
        assert all(o["mode"] != "walk" for o in d["options"])
        for j in d["journeys"]:
            assert "walk" not in j["modes"], j["shape"]
            for leg in j["legs"]:
                assert leg["mode"] != "walk"


def test_only_the_six_modes_are_ever_offered(client):
    """Carpool, walking and cycling are gone, and gone from every surface.

    Carpool was the cheapest card on almost every trip at an 11% completion
    rate -- a mode nobody could book steering the whole comparison. A cycle
    assumes a bicycle nobody told us the rider owns. Walking is access.
    """
    allowed = {"bike_taxi", "auto", "cab", "metro", "bus"}
    for trip in (SHORT, LONG):
        d = client.post("/api/compare", json=trip).json()
        assert {o["mode"] for o in d["options"]} <= allowed
        for gone in ("carpool", "walk", "cycle"):
            assert gone not in {o["provider_id"] for o in d["options"]}
        for j in d["journeys"]:
            assert set(j["modes"]) <= allowed, j["shape"]


# ==========================================================================
# hard constraints beat soft objectives
# ==========================================================================
def test_a_time_limit_is_not_negotiable(client):
    d = client.post("/api/compare", json={**LONG, "max_time": 65}).json()
    for o in d["options"]:
        if o["feasible"]:
            assert o["door_to_door_min"] <= 65 + 1e-6, o["display_name"]


def test_a_budget_is_what_you_are_charged_not_what_you_average(client):
    """Gating on EXPECTED cost admitted a ₹300 ride against a ₹250 budget on
    the grounds that you probably would not get it."""
    d = client.post("/api/compare", json={**LONG, "budget": 200}).json()
    for o in d["options"]:
        if o["feasible"]:
            assert o["fare"]["amount"] <= 200 + 1e-6, (
                f"{o['display_name']} charges ₹{o['fare']['amount']:.0f} "
                f"against a ₹200 budget")


def test_an_option_over_budget_in_expectation_is_flagged_not_hidden(client):
    d = client.post("/api/compare", json={**LONG, "budget": 220}).json()
    for o in d["options"]:
        if o["feasible"] and o["expected"] and o["expected"]["expected_cost"] > 220:
            assert o["budget_at_risk"], o["display_name"]


def test_a_cheaper_slower_option_cannot_beat_the_time_limit(client):
    """The metro is ₹25 and takes three and a half hours."""
    d = client.post("/api/compare", json={**LONG, "max_time": 90,
                                          "priority": "cheapest"}).json()
    if d["recommended_provider"]:
        rec = next(o for o in d["options"]
                   if o["provider_id"] == d["recommended_provider"])
        assert rec["door_to_door_min"] <= 90 + 1e-6


def test_an_offered_journey_respects_the_stated_limits(client):
    d = client.post("/api/compare", json={**LONG, "budget": 200,
                                          "max_time": 65}).json()
    for j in d["journeys"]:
        assert j["fare"] <= 200 + 1e-6
        assert j["total_min"] <= 65 + 1e-6


def test_the_headline_never_contradicts_the_journeys_below_it(client):
    """"Nothing fits ₹200" sat directly above a ₹167 journey that fitted."""
    d = client.post("/api/compare", json={**LONG, "budget": 200,
                                          "max_time": 65}).json()
    if d["journeys"]:
        assert "nothing fits" not in d["headline"].lower(), d["headline"]


# ==========================================================================
# preference actually changes the answer
# ==========================================================================
def test_each_preference_changes_the_plan(engine):
    P = places(engine)
    o, d = P["pl_wipro_sarjapur"], P["pl_pes_university"]
    shapes = {p: tuple(plan(engine, o, d, 400, 180, preference=p).recommended.shape())
              for p in ("cheapest", "balanced", "fastest")}
    assert len(set(shapes.values())) > 1, f"every preference agreed: {shapes}"


def test_every_preference_still_respects_the_hard_limits(engine):
    P = places(engine)
    o, d = P["pl_wipro_sarjapur"], P["pl_pes_university"]
    for p in ("cheapest", "balanced", "fastest"):
        r = plan(engine, o, d, 200, 100, preference=p)
        if r.recommended is not None:
            assert r.recommended.cost <= 200 + 1e-6
            assert r.recommended.total_min <= 100 + 1e-6


# ==========================================================================
# alternatives must be alternatives
# ==========================================================================
def test_alternatives_are_not_the_recommendation_again(engine):
    P = list(engine.graph.places)
    for o, d in list(itertools.permutations(P, 2))[::11]:
        r = plan(engine, o, d, 400, 180)
        if r.recommended is None:
            continue
        seen = {r.recommended.signature}
        for a in r.alternatives:
            sig = a["journey"].signature
            assert sig not in seen, f"{o.name} -> {d.name}: duplicate alternative"
            seen.add(sig)


def test_two_providers_of_the_same_vehicle_do_not_quote_identically(client):
    """Auto and Namma Yatri shared a fare table AND a reliability class, so
    they were one option printed twice."""
    d = client.post("/api/compare", json=LONG).json()
    by_id = {o["provider_id"]: o for o in d["options"]}
    auto, ny = by_id["auto"], by_id["namma_yatri"]
    assert abs(auto["fare"]["amount"] - ny["fare"]["amount"]) > 0.5, (
        "Auto and Namma Yatri quote the same number for the same trip")


# ==========================================================================
# service availability and the clock
# ==========================================================================
def test_a_shut_network_is_said_out_loud_not_silently_routed_around(client):
    d = client.post("/api/recommend", json={
        "origin": WIPRO, "destination": PES, "budget": 400, "max_time": 300,
        "departure_time": "2026-08-28T02:00:00"}).json()
    modes = {m for leg in d["recommended"]["legs"] for m in [leg["mode"]]}
    caveats = " ".join(d["explanation"]["caveats"]).lower()
    if "metro" not in modes:
        assert "service" in caveats or "running" in caveats, d["explanation"]["caveats"]


def test_a_journey_never_boards_a_service_without_paying_for_the_wait(engine):
    """Out-of-service routes are not banned -- waiting for the first train is a
    real journey. What must never happen is boarding one for free."""
    P = places(engine)
    for j in candidates(engine, P["pl_wipro_sarjapur"], P["pl_pes_university"],
                        departure=datetime(2026, 8, 28, 2, 0)):
        for lg in j.legs:
            if lg.kind == "transit":
                assert lg.wait_min > 0, (
                    f"boarded {lg.route_name} at 02:00 with no wait at all")


# ==========================================================================
# presentation
# ==========================================================================
def test_an_unroutable_option_publishes_no_numbers(client):
    """"Walk: ₹25, 0 min" -- the lifecycle solver was pricing the FALLBACK and
    the payload published it as the option's own cost."""
    d = client.post("/api/compare", json=SHORT).json()
    for o in d["options"]:
        if not o["available"] and o["distance_km"] is None:
            assert o["expected"] is None
            assert o["door_to_door_min"] is None
            assert o["unavailable_reason"]


def test_every_offered_journey_mixes_modes(client):
    d = client.post("/api/compare", json=LONG).json()
    for j in d["journeys"]:
        assert len({m for m in j["modes"] if m != "walk"}) >= 2, j["shape"]


def test_a_legs_route_name_survives_the_collapsed_summary(client):
    """Collapsing "Metro → Metro" must not lose which lines they were."""
    d = client.post("/api/compare", json=LONG).json()
    for j in d["journeys"]:
        for leg in j["legs"]:
            if leg.get("interchange"):
                assert leg["route"], "an interchange with no service name"


# ==========================================================================
# edge cases: fail gracefully, never invent
# ==========================================================================
@pytest.mark.parametrize("body,expect", [
    ({"origin": WIPRO, "destination": WIPRO}, "same_endpoints"),
    ({"origin": "Atlantis", "destination": PES}, "unknown_place"),
    ({"origin": "19.0760,72.8777", "destination": PES}, "outside_study_area"),
])
def test_impossible_requests_fail_with_a_reason(client, body, expect):
    r = client.post("/api/compare", json={**body,
                                          "departure_time": "2026-08-28T09:00:00"})
    assert r.status_code == 422
    assert r.json()["detail"]["code"] == expect


@pytest.mark.parametrize("typed", [
    "12.9345, 77.6100", "12.9345,77.6100", "12.9345 77.6100"])
def test_a_typed_coordinate_is_a_coordinate(client, typed):
    """It arrived as a place name and came back "could not find
    '12.9345, 77.6100' in this study area" -- true, and useless."""
    r = client.post("/api/compare", json={
        "origin": typed, "destination": PES,
        "departure_time": "2026-08-28T09:00:00"})
    assert r.status_code == 200, r.json()


def test_nothing_affordable_is_said_plainly(client):
    d = client.post("/api/compare", json={**LONG, "budget": 5}).json()
    assert d["recommended_provider"] is None
    assert not [o for o in d["options"] if o["feasible"]]
    assert d["headline"]


def test_a_failed_booking_does_not_become_an_enterprise_pattern(client):
    """One stranded rider is an incident. It is not a finding about a fleet."""
    before = client.get("/api/enterprise/overview",
                        headers={"X-API-Key": "demo-analyst-key"}).json()
    r = client.post("/api/book", json={
        **LONG, "provider_id": "bike_taxi", "demo": True}).json()
    sid = r["session"]["session_id"]
    while r["session"]["can_retry"]:
        r = client.post(f"/api/book/{sid}/retry").json()
    client.post(f"/api/book/{sid}/notify", json={})
    after = client.get("/api/enterprise/overview",
                       headers={"X-API-Key": "demo-analyst-key"}).json()
    assert before["overview"]["bookings"] == after["overview"]["bookings"]


# ==========================================================================
# one demo scenario, named once
# ==========================================================================
def test_the_demo_scenario_has_exactly_one_definition(client):
    """The booking page hard-coded a different pair of places from the planner,
    and the escalation invented a meeting neither had heard of."""
    from app.demo_scenario import DEMO_SCENARIO
    city = client.get("/api/city").json()["demo_scenario"]
    for key in ("origin", "destination", "budget", "max_time", "preference"):
        assert city[key] == DEMO_SCENARIO[key]
    assert city["meeting_title"] == DEMO_SCENARIO["meeting_title"]

    named = {p["place_id"]: p["name"]
             for p in client.get("/api/places").json()["places"]}
    demo = client.get("/api/demo").json()
    assert demo["origin"]["label"] == named[DEMO_SCENARIO["origin"]]
    assert demo["destination"]["label"] == named[DEMO_SCENARIO["destination"]]


def test_the_escalation_names_the_scenario_meeting(client):
    from app.demo_scenario import DEMO_SCENARIO
    r = client.post("/api/book", json={
        **LONG, "provider_id": "bike_taxi", "demo": True}).json()
    sid = r["session"]["session_id"]
    e = client.get(f"/api/book/{sid}/escalation").json()
    assert e["meeting"]["title"] == DEMO_SCENARIO["meeting_title"]
    assert e["meeting"]["starts_at"].endswith("10:00")


def test_a_slow_cheap_winner_admits_what_it_costs_in_time(client):
    """"Best value: Bus" without the three hours attached is the product
    hiding its own trade-off."""
    d = client.post("/api/compare", json={
        "origin": WIPRO, "destination": PES, "rain": True,
        "departure_time": "2026-08-28T18:30:00"}).json()
    rec = next(o for o in d["options"]
               if o["provider_id"] == d["recommended_provider"])
    faster = [o for o in d["options"]
              if o["feasible"] and o["expected"]
              and o["expected"]["expected_minutes"]
              < rec["expected"]["expected_minutes"] - 15]
    if faster:
        joined = " ".join(d["reasoning"]).lower()
        assert "sooner" in joined, (
            f"{rec['display_name']} was recommended at "
            f"{rec['expected']['expected_minutes']:.0f} min with a "
            f"{min(o['expected']['expected_minutes'] for o in faster):.0f} min "
            f"option available, and the reasoning never mentions it")


# ==========================================================================
# a timetabled service is not a hailed one
# ==========================================================================
@pytest.mark.parametrize("provider", ["metro", "bus"])
def test_a_scheduled_service_has_no_driver_to_search_for(client, provider):
    """Booking a metro narrated "Searching for driver… / Kiran K. is on the
    way", which is the least believable thing the product could say."""
    r = client.post("/api/book", json={
        **LONG, "provider_id": provider, "demo": True})
    if r.status_code != 200:
        pytest.skip(f"{provider} does not serve this trip")
    steps = r.json()["attempt"]["steps"]
    text = " ".join(s["label"] + " " + s["detail"] for s in steps).lower()
    for word in ("driver", "accepted your request", "cancelled"):
        assert word not in text, f"{provider} narrated: {text}"
    assert r.json()["attempt"]["outcome"] == "RIDE_COMPLETED"


def test_every_narrated_transition_is_legal(client):
    """Including the scheduled path, which was added for exactly this."""
    from app.lifecycle.states import LEGAL_TRANSITIONS, BookingState
    for provider in ("metro", "bus", "bike_taxi", "auto", "cab", "carpool"):
        for _ in range(6):
            r = client.post("/api/book", json={
                **LONG, "provider_id": provider, "demo": False})
            if r.status_code != 200:
                break
            d = r.json()
            sid = d["session"]["session_id"]
            while True:
                states = [BookingState(s["state"]) for s in d["attempt"]["steps"]]
                for a, b in zip(states, states[1:]):
                    assert b in LEGAL_TRANSITIONS[a], f"{provider}: {a} -> {b}"
                if not d["session"]["can_retry"]:
                    break
                d = client.post(f"/api/book/{sid}/retry").json()
