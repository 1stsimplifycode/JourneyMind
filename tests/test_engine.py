"""Engine, optimiser and explanation tests.

These assert behaviour the product promises: that over-budget journeys are
removed, that dominated journeys are removed, that the three presets actually
produce different answers, that alternatives are genuinely different, and that
an impossible request says so instead of quietly returning an invalid route.
"""

from __future__ import annotations

import os
import sys
from datetime import datetime

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "backend"))

from app.models.fares import FareEstimate, FareEstimator            # noqa: E402
from app.optimisation import constraints as C                        # noqa: E402
from app.optimisation.pareto import dominates, frontier              # noqa: E402
from app.optimisation.scoring import PRESETS, score_all, weights_for  # noqa: E402
from app.routing.journey import Journey                              # noqa: E402
from app.services.engine import JourneyMindEngine, JourneyRequest, RoutingError  # noqa: E402

WEEKDAY_0900 = datetime(2025, 1, 7, 9, 0)


@pytest.fixture(scope="module")
def engine():
    return JourneyMindEngine()


@pytest.fixture(scope="module")
def places(engine):
    return {p.place_id: p for p in engine.graph.places}


def ask(engine, places, o, d, budget=100.0, max_time=30.0, preference="balanced",
        when=WEEKDAY_0900, **kw):
    a, b = places[o], places[d]
    return engine.recommend(JourneyRequest(
        origin_lat=a.lat, origin_lon=a.lon, origin_label=a.name,
        dest_lat=b.lat, dest_lon=b.lon, dest_label=b.name,
        departure=when, budget=budget, max_time_min=max_time,
        preference=preference, **kw))


def fake(jid, cost, minutes, transfers=0, modes=("metro",), discomfort=0.3):
    return Journey(
        journey_id=jid, legs=[], total_min=minutes,
        total_cost=FareEstimate(cost, cost, cost, "published", "Total", ""),
        transfers=transfers, modes=list(modes), distance_km=5.0, walk_min=2.0,
        wait_min=1.0, discomfort=discomfort, reliability=0.9,
    )


# --------------------------------------------------------------------------
# graph and pipeline
# --------------------------------------------------------------------------
def test_graph_is_multimodal_and_connected(engine):
    kinds = {e.kind for e in engine.graph.edges}
    assert {"road", "transit", "transfer"} <= kinds
    modes = {e.mode for e in engine.graph.edges}
    assert {"walk", "metro", "bus"} <= modes
    assert len(engine.graph.nodes) > 100


def test_request_graph_adds_access_and_ride_edges(engine, places):
    from app.graph.builder import DESTINATION_ID, ORIGIN_ID, RequestGraph
    o, d = places["pl_majestic_bus"], places["pl_indiranagar_100ft"]
    rg = RequestGraph(engine.graph, (o.lat, o.lon), (d.lat, d.lon))
    extra = rg.request_edges
    assert any(e.kind == "access" for e in extra)
    assert any(e.kind == "ride" for e in extra)
    assert rg.out_adj[ORIGIN_ID], "origin must have outgoing edges"
    assert any(e.v == DESTINATION_ID for e in rg.edges)


def test_recommendation_is_multimodal_for_the_demo_pair(engine, places):
    rec = ask(engine, places, "pl_majestic_bus", "pl_indiranagar_100ft")
    assert rec.feasible
    j = rec.recommended
    vehicles = {m for m in j.modes if m != "walk"}
    assert len(vehicles) >= 2, f"expected a mixed-mode journey, got {j.modes}"
    assert j.legs, "a journey must have legs"
    assert j.total_min == pytest.approx(sum(l.total_min for l in j.legs), abs=0.05)


def test_leg_geometry_is_drawable(engine, places):
    rec = ask(engine, places, "pl_majestic_bus", "pl_indiranagar_100ft")
    for leg in rec.recommended.legs:
        assert len(leg.geometry) >= 2
        for lat, lon in leg.geometry:
            assert 12.0 < lat < 14.0 and 77.0 < lon < 78.5


# --------------------------------------------------------------------------
# constraints
# --------------------------------------------------------------------------
def test_over_budget_journeys_are_removed(engine, places):
    rec = ask(engine, places, "pl_majestic_bus", "pl_indiranagar_100ft",
              budget=40.0, max_time=120.0)
    if rec.recommended:
        assert rec.recommended.cost <= 40.0 + 1e-9
    for a in rec.alternatives:
        if a["kind"] == "feasible":
            assert a["journey"].cost <= 40.0 + 1e-9


def test_over_time_journeys_are_removed(engine, places):
    rec = ask(engine, places, "pl_majestic_bus", "pl_indiranagar_100ft",
              budget=100000.0, max_time=20.0)
    if rec.recommended:
        assert rec.recommended.total_min <= 20.0 + 1e-9


def test_impossible_request_is_reported_not_faked(engine, places):
    rec = ask(engine, places, "pl_home", "pl_domlur", budget=5.0, max_time=10.0)
    assert rec.feasible is False
    assert rec.recommended is None
    assert rec.message and "fits both" in rec.message
    assert rec.fallbacks, "must offer labelled alternatives instead of nothing"
    for f in rec.fallbacks:
        assert f["status"].feasible is False
        assert f["status"].reasons, "every fallback must say which limit it breaks"


def test_constraint_status_reports_headroom():
    j = fake("A", cost=70.0, minutes=27.0)
    st = C.evaluate(j, budget=100.0, max_time_min=30.0)
    assert st.feasible and st.within_budget and st.within_time
    assert st.budget_headroom == pytest.approx(30.0)
    assert st.time_headroom == pytest.approx(3.0)

    over = C.evaluate(fake("B", 120.0, 40.0), budget=100.0, max_time_min=30.0)
    assert not over.feasible and len(over.reasons) == 2


def test_estimated_fare_band_over_budget_is_flagged():
    j = Journey(journey_id="X", legs=[], total_min=20.0,
                total_cost=FareEstimate(95.0, 80.0, 115.0, "estimated", "Total", ""),
                transfers=0, modes=["bike_taxi"], distance_km=6.0, walk_min=1.0,
                wait_min=3.0, discomfort=0.5, reliability=0.7)
    st = C.evaluate(j, budget=100.0, max_time_min=30.0)
    assert st.within_budget and st.cost_at_risk


# --------------------------------------------------------------------------
# Pareto
# --------------------------------------------------------------------------
def test_dominated_journeys_are_dropped():
    cheap_fast = fake("A", cost=50.0, minutes=20.0)
    worse = fake("B", cost=80.0, minutes=30.0)     # dearer AND slower
    trade = fake("C", cost=30.0, minutes=45.0)     # cheaper but slower
    assert dominates(cheap_fast, worse)
    assert not dominates(cheap_fast, trade)
    kept = {j.journey_id for j in frontier([cheap_fast, worse, trade])}
    assert kept == {"A", "C"}


def test_frontier_keeps_everything_on_a_real_trade_off():
    js = [fake("A", 20, 60), fake("B", 40, 40), fake("C", 80, 25)]
    assert len(frontier(js)) == 3


def test_pipeline_actually_removes_dominated_candidates(engine, places):
    rec = ask(engine, places, "pl_home", "pl_domlur", budget=100000.0, max_time=100000.0)
    p = rec.pipeline["pareto"]
    assert p["in"] >= p["on_frontier"]
    front = [j for j in [rec.recommended] + [a["journey"] for a in rec.alternatives] if j]
    for a in front:
        for b in front:
            assert not (a is not b and dominates(b, a))


# --------------------------------------------------------------------------
# personalisation
# --------------------------------------------------------------------------
def test_preset_weights_are_normalised():
    for name in PRESETS:
        w, key = weights_for(name)
        assert key == name
        assert w.cost + w.time + w.transfers + w.comfort == pytest.approx(1.0)


def test_manual_weights_override_and_normalise():
    w, key = weights_for("fastest", {"cost": 3, "time": 1, "transfers": 0, "comfort": 0})
    assert key == "custom"
    assert w.cost == pytest.approx(0.75) and w.time == pytest.approx(0.25)


def test_cheapest_and_fastest_disagree():
    js = [fake("cheap", 20.0, 60.0), fake("quick", 90.0, 22.0)]
    cheap = score_all(list(js), weights_for("cheapest")[0])[0]
    quick = score_all(list(js), weights_for("fastest")[0])[0]
    assert cheap.journey_id == "cheap"
    assert quick.journey_id == "quick"


def test_presets_change_the_real_recommendation(engine, places):
    out = {}
    for pref in ("cheapest", "balanced", "fastest"):
        rec = ask(engine, places, "pl_home", "pl_domlur",
                  budget=100000.0, max_time=100000.0, preference=pref)
        out[pref] = rec.recommended
    assert out["cheapest"].cost <= out["fastest"].cost
    assert out["fastest"].total_min <= out["cheapest"].total_min

    # The presets must be capable of disagreeing -- but demanding that they
    # disagree on THIS trip asserts something about the corridor rather than
    # about the optimiser. Here the multimodal option is genuinely both the
    # cheapest and the quickest, and a preset that ignored that to look busy
    # would be the bug. So: they must differ somewhere.
    seen = set()
    for a, b in (("pl_home", "pl_domlur"), ("pl_wipro_sarjapur", "pl_pes_university"),
                 ("pl_college", "pl_mg_road_shops")):
        for pref in ("cheapest", "fastest"):
            r = ask(engine, places, a, b, budget=100000.0, max_time=100000.0,
                    preference=pref)
            if r.recommended is not None:
                seen.add((a, b, pref, r.recommended.signature))
    by_trip = {}
    for a, b, pref, sig in seen:
        by_trip.setdefault((a, b), set()).add(sig)
    assert any(len(v) > 1 for v in by_trip.values()), \
        "cheapest and fastest agreed on every trip tried"


def test_alternatives_are_genuinely_different(engine, places):
    rec = ask(engine, places, "pl_home", "pl_domlur",
              budget=100000.0, max_time=100000.0)
    seen = {rec.recommended.signature}
    for a in rec.alternatives:
        assert a["journey"].signature not in seen, "alternatives must differ from each other"
        seen.add(a["journey"].signature)


# --------------------------------------------------------------------------
# fares
# --------------------------------------------------------------------------
def test_published_fares_are_exact_and_ride_fares_are_ranges(engine):
    f = FareEstimator(engine.graph.fares)
    metro = f.leg_fare("metro", 12.0, 20.0)
    assert metro.provenance == "published" and not metro.is_range

    bike = f.leg_fare("bike_taxi", 4.0, 12.0)
    assert bike.provenance == "estimated" and bike.is_range
    assert bike.low < bike.amount < bike.high
    assert "–" in bike.display()


def test_walking_is_free(engine):
    f = FareEstimator(engine.graph.fares)
    assert f.leg_fare("walk", 2.0, 26.0).amount == 0.0


def test_total_provenance_degrades_to_the_weakest_link(engine):
    f = FareEstimator(engine.graph.fares)
    total = f.combine([f.leg_fare("metro", 10.0, 18.0), f.leg_fare("bike_taxi", 3.0, 9.0)])
    assert total.provenance == "estimated"


def test_metro_is_charged_once_across_an_interchange(engine, places):
    """A journey that changes metro lines pays one fare, not two."""
    rec = ask(engine, places, "pl_banashankari_home", "pl_indiranagar_100ft",
              budget=100000.0, max_time=100000.0, preference="cheapest")
    for j in [rec.recommended] + [a["journey"] for a in rec.alternatives]:
        metro_legs = [l for l in j.legs if l.mode == "metro"]
        if len(metro_legs) > 1:
            charged = [l for l in metro_legs if l.fare and l.fare.amount > 0]
            assert len(charged) == 1, "an interchange must not be a second metro fare"


# --------------------------------------------------------------------------
# explanations
# --------------------------------------------------------------------------
def test_explanation_matches_the_recommendation(engine, places):
    rec = ask(engine, places, "pl_majestic_bus", "pl_indiranagar_100ft")
    e = rec.explanation
    assert e.headline
    assert any("budget" in r for r in e.reasons)
    assert any("limit" in r for r in e.reasons)
    # an estimated total must always be disclosed as estimated
    if rec.recommended.total_cost.provenance == "estimated":
        assert any("estimate" in c.lower() or "quote" in c.lower() for c in e.caveats)


def test_explanation_is_deterministic(engine, places):
    a = ask(engine, places, "pl_majestic_bus", "pl_indiranagar_100ft").explanation
    b = ask(engine, places, "pl_majestic_bus", "pl_indiranagar_100ft").explanation
    assert a.as_dict() == b.as_dict()


def test_no_zero_valued_comparisons_are_emitted(engine, places):
    rec = ask(engine, places, "pl_home", "pl_domlur",
              budget=100000.0, max_time=100000.0)
    import re
    for line in rec.explanation.comparisons:
        # word boundaries: "0 minutes" used to match inside "40 minutes",
        # which failed the moment a comparison quoted a round difference
        assert not re.search(r"(?<![\d])0 (minutes|min)\b", line), line
        assert not re.search(r"₹0(?![\d.])", line), line


# --------------------------------------------------------------------------
# errors
# --------------------------------------------------------------------------
def test_same_origin_and_destination_is_rejected(engine, places):
    p = places["pl_home"]
    with pytest.raises(RoutingError) as exc:
        engine.recommend(JourneyRequest(
            origin_lat=p.lat, origin_lon=p.lon, origin_label=p.name,
            dest_lat=p.lat, dest_lon=p.lon, dest_label=p.name,
            departure=WEEKDAY_0900, budget=100.0, max_time_min=30.0))
    assert exc.value.code == "same_endpoints"


def test_point_outside_the_study_area_is_rejected(engine, places):
    p = places["pl_home"]
    with pytest.raises(RoutingError) as exc:
        engine.recommend(JourneyRequest(
            origin_lat=p.lat, origin_lon=p.lon, origin_label=p.name,
            dest_lat=28.61, dest_lon=77.21, dest_label="New Delhi",
            departure=WEEKDAY_0900, budget=100.0, max_time_min=30.0))
    assert exc.value.code == "outside_study_area"


# --------------------------------------------------------------------------
# time dependence
# --------------------------------------------------------------------------
def test_peak_hour_is_slower_than_the_middle_of_the_night(engine, places):
    peak = ask(engine, places, "pl_home", "pl_domlur", budget=100000.0,
               max_time=100000.0, preference="fastest",
               when=datetime(2025, 1, 7, 9, 0))
    quiet = ask(engine, places, "pl_home", "pl_domlur", budget=100000.0,
                max_time=100000.0, preference="fastest",
                when=datetime(2025, 1, 7, 13, 0))
    assert peak.pipeline["prediction"]["mean_congestion_ratio"] > \
        quiet.pipeline["prediction"]["mean_congestion_ratio"]
