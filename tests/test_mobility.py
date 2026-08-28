"""The mobility-intelligence layer: lifecycle, expected cost, providers, compare.

The expected-cost solver is the product's central claim, so it is tested three
ways: against a closed form, against a Monte Carlo of the same state machine,
and for the degenerate cases where it must return the boring answer.
"""

from __future__ import annotations

import os
import sys
from datetime import datetime

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "backend"))

from app.lifecycle.expected_cost import LifecycleParams, solve            # noqa: E402
from app.lifecycle.states import (                                        # noqa: E402
    ABSORBING, BookingState, BookingTrajectory, IllegalTransition,
    can_transition, simulate,
)
from app.providers.base import DataClass, ServiceClass                    # noqa: E402
from app.providers.simulated import ALL_PROVIDERS, registry               # noqa: E402
from app.reliability.features import RequestFeatures, short_trip_penalty  # noqa: E402
from app.reliability.model import get_reliability_model                   # noqa: E402
from app.services.compare import PRIORITIES, compare                      # noqa: E402
from app.services.engine import get_engine                                # noqa: E402

PARAMS = LifecycleParams(surge_per_retry=0.40, max_attempts=3)


# ==========================================================================
# the state machine
# ==========================================================================
def test_illegal_transitions_are_rejected():
    """An event stream that claims the impossible must fail at the door.

    Otherwise a corrupt stream quietly poisons every aggregate downstream.
    """
    t = BookingTrajectory(provider_id="x")
    now = datetime(2026, 8, 28, 9, 0)
    t.push(now, BookingState.SEARCHING, 0)
    t.push(now, BookingState.REQUESTED, 1)
    with pytest.raises(IllegalTransition):
        t.push(now, BookingState.RIDE_COMPLETED, 1)     # cannot skip the middle


def test_absorbing_states_have_no_exits():
    for s in ABSORBING:
        assert not any(can_transition(s, other) for other in BookingState)


def test_a_completed_trajectory_is_a_legal_path():
    rng = np.random.default_rng(3)
    t = simulate(rng, "cab", datetime(2026, 8, 28, 9, 0),
                 p_match=1.0, p_accept=1.0, p_cancel=0.0, base_fare=100,
                 surge_per_retry=0.2, pickup_min=5, ride_min=20,
                 search_timeout_min=2.5, match_min=0.6,
                 cancel_discovery_frac=0.7, max_attempts=3)
    assert t.completed and t.fare_paid == 100
    states = [e.state for e in t.events]
    for a, b in zip(states, states[1:]):
        assert can_transition(a, b), f"{a} -> {b}"


# ==========================================================================
# expected cost — the central claim
# ==========================================================================
def test_expected_cost_matches_the_closed_form():
    """Enumerated outcomes must equal the geometric series by hand."""
    ec = solve(displayed_fare=20, p_match=1.0, p_accept=1.0, p_cancel=0.30,
               pickup_min=6, ride_min=14, fallback_cost=100.0, params=PARAMS)
    q = 0.7
    expected = (q * 20 + (1 - q) * q * 28 + (1 - q) ** 2 * q * 39.2
                + (1 - q) ** 3 * 100.0)
    assert ec.expected_cost == pytest.approx(expected, rel=1e-9)
    assert ec.p_success == pytest.approx(1 - (1 - q) ** 3)
    assert sum(o.probability for o in ec.outcomes) == pytest.approx(1.0)


def test_expected_cost_matches_a_monte_carlo_of_the_state_machine():
    """The analytic solver and the simulator must agree, or one is wrong."""
    rng = np.random.default_rng(11)
    pm, pa, pc = 0.80, 0.85, 0.18
    ec = solve(displayed_fare=60, p_match=pm, p_accept=pa, p_cancel=pc,
               pickup_min=5, ride_min=18, fallback_cost=40.0,
               fallback_min=50.0, params=PARAMS)
    costs = []
    for _ in range(20000):
        t = simulate(rng, "p", datetime(2026, 8, 28, 9, 0), p_match=pm,
                     p_accept=pa, p_cancel=pc, base_fare=60,
                     surge_per_retry=PARAMS.surge_per_retry, pickup_min=5,
                     ride_min=18, search_timeout_min=PARAMS.search_timeout_min,
                     match_min=PARAMS.match_min,
                     cancel_discovery_frac=PARAMS.cancel_discovery_frac,
                     max_attempts=PARAMS.max_attempts)
        costs.append(t.fare_paid if t.completed else 40.0)
    assert float(np.mean(costs)) == pytest.approx(ec.expected_cost, rel=0.03)


def test_a_scheduled_service_degenerates_to_its_fare():
    """A metro does not cancel on you, so expected cost IS the fare.

    This is what lets the comparison honestly show transit as the reliable
    floor rather than merely asserting it.
    """
    ec = solve(displayed_fare=25, p_match=1.0, p_accept=1.0, p_cancel=0.0,
               pickup_min=0, ride_min=30, params=PARAMS)
    assert ec.expected_cost == pytest.approx(25.0)
    assert ec.surcharge == pytest.approx(0.0)
    assert ec.p_success == 1.0
    assert ec.cost_p10 == ec.cost_p90 == pytest.approx(25.0)
    assert not ec.is_blended


def test_more_cancellation_never_lowers_the_expected_cost():
    """Monotonicity. If this ever inverts, the model is not modelling risk."""
    prev = -1.0
    for pc in (0.0, 0.1, 0.2, 0.35, 0.5):
        ec = solve(displayed_fare=50, p_match=1.0, p_accept=1.0, p_cancel=pc,
                   pickup_min=5, ride_min=20, fallback_cost=90.0, params=PARAMS)
        assert ec.expected_cost >= prev - 1e-9
        prev = ec.expected_cost


def test_a_cheap_fallback_flags_the_expectation_as_blended():
    """An option that usually fails into a cheap bus is not cheap.

    The number is arithmetically right but stops describing the option you
    chose, so it must be flagged or it reads as a discount.
    """
    ec = solve(displayed_fare=90, p_match=0.5, p_accept=0.6, p_cancel=0.3,
               pickup_min=8, ride_min=25, fallback_cost=12.0,
               fallback_label="Bus", params=PARAMS)
    assert ec.expected_cost < 90
    assert ec.is_blended and ec.substitution_share > 0.1
    assert ec.fallback_label == "Bus"


def test_wasted_time_is_charged_for_failed_attempts():
    clean = solve(displayed_fare=50, p_match=1.0, p_accept=1.0, p_cancel=0.0,
                  pickup_min=5, ride_min=20, params=PARAMS)
    messy = solve(displayed_fare=50, p_match=1.0, p_accept=1.0, p_cancel=0.4,
                  pickup_min=5, ride_min=20, params=PARAMS)
    assert clean.expected_wasted_min == pytest.approx(0.0)
    assert messy.expected_wasted_min > 1.0
    assert messy.expected_minutes > clean.expected_minutes


# ==========================================================================
# reliability
# ==========================================================================
def test_short_trips_are_penalised_and_long_ones_are_not():
    assert short_trip_penalty(0.8) > short_trip_penalty(3.0) > 0
    assert short_trip_penalty(6.0) == 0.0
    assert short_trip_penalty(20.0) == 0.0


def test_reliability_model_returns_probabilities_and_says_where_they_came_from():
    m = get_reliability_model()
    p = m.predict(RequestFeatures(provider_id="bike_taxi", distance_km=2.0,
                                  pickup_km=1.5, hour=9.0, dow=1))
    for v in (p.p_match, p.p_accept, p.p_cancel):
        assert 0.0 <= v <= 1.0
    assert p.source in ("model", "fallback")
    assert p.drivers_basis, "a prediction must say what it rests on"


def test_a_short_trip_is_predicted_to_cancel_more_than_a_long_one():
    """The strongest documented effect in the generator must survive training."""
    m = get_reliability_model()
    if m.meta.get("version") == "fallback":
        pytest.skip("no trained reliability model")
    short = m.predict(RequestFeatures(provider_id="bike_taxi", distance_km=1.2,
                                      pickup_km=1.2, hour=9.0, dow=1))
    long = m.predict(RequestFeatures(provider_id="bike_taxi", distance_km=12.0,
                                     pickup_km=1.2, hour=9.0, dow=1))
    assert short.p_cancel > long.p_cancel


# ==========================================================================
# providers
# ==========================================================================
def test_every_provider_declares_its_provenance():
    for p in ALL_PROVIDERS:
        assert isinstance(p.data_class, DataClass)
        assert isinstance(p.service_class, ServiceClass)


def test_hailed_providers_are_labelled_simulated_and_transit_is_not():
    """The honesty rule: no adapter here talks to a live ride-hailing API."""
    for p in ALL_PROVIDERS:
        if p.service_class is ServiceClass.HAILED:
            assert p.data_class is DataClass.SIMULATED, p.provider_id
        else:
            assert p.data_class is DataClass.PUBLISHED, p.provider_id


def test_registry_covers_every_mode_the_brief_asks_for():
    """Six modes, five providers, and nothing else.

    Carpool was removed: it is not a mode JourneyMind covers, and while it
    stayed it was the cheapest card on almost every trip at an 11% completion
    rate. Walking and cycling went with it -- walking is how you reach a
    vehicle here, not a commute this product recommends, and a cycle assumes a
    bicycle nobody has told us the rider owns.
    """
    rows = registry()
    ids = {r["provider_id"] for r in rows}
    assert ids == {"bike_taxi", "auto", "namma_yatri", "cab", "bus", "metro"}
    assert {r["mode"] for r in rows} == {"bike_taxi", "auto", "cab", "bus", "metro"}
    for gone in ("carpool", "walk", "cycle"):
        assert gone not in ids

    # mode and provider are different facts, and two providers share one auto
    by_mode = {}
    for r in rows:
        by_mode.setdefault(r["mode"], []).append(r["provider_name"])
    assert len(by_mode["auto"]) == 2, by_mode["auto"]


# ==========================================================================
# the comparison service
# ==========================================================================
@pytest.fixture(scope="module")
def trip():
    engine = get_engine()
    places = {p.place_id: p for p in engine.graph.places}
    o, d = places["pl_koramangala"], places["pl_indiranagar_100ft"]
    return dict(origin_lat=o.lat, origin_lon=o.lon, origin_label=o.name,
                dest_lat=d.lat, dest_lon=d.lon, dest_label=d.name,
                departure=datetime(2026, 8, 28, 9, 0))


def test_comparison_prices_every_available_mode(trip):
    c = compare(**trip, priority="balanced")
    ids = {o.quote.provider_id for o in c.options}
    assert {"bike_taxi", "auto", "cab", "metro"} <= ids
    assert c.recommended is not None
    assert c.reasoning, "a recommendation must explain itself"


def test_priority_actually_changes_the_answer(trip):
    """If every priority returns the same option, the control is decoration."""
    picks = {}
    for p in PRIORITIES:
        c = compare(**trip, priority=p)
        picks[p] = c.recommended.quote.provider_id if c.recommended else None
    assert len(set(picks.values())) > 1, f"all priorities agreed: {picks}"


def test_cheapest_ranks_on_expected_cost_not_the_advertised_fare(trip):
    """The thesis, asserted."""
    c = compare(**trip, priority="cheapest")
    feasible = [o for o in c.options if o.feasible]
    assert c.recommended is not None
    best_expected = min(o.expected.expected_cost for o in feasible)
    assert c.recommended.expected.expected_cost == pytest.approx(best_expected)


def test_scheduled_options_carry_no_surcharge(trip):
    """A timetable does not surcharge. Reaching it by bike taxi can.

    A scheduled quote is now door to door, so when the journey needs a hailed
    first or last mile it inherits that booking's retry cost and its chance of
    failing. `p_cancel > 0` is the quote saying exactly that -- and a metro
    journey that begins with a bike taxi reporting 100% would be the overclaim
    this product exists to argue against.
    """
    c = compare(**trip, priority="balanced")
    for o in c.options:
        if o.quote.service_class is ServiceClass.SCHEDULED and o.quote.available:
            if o.quote.reliability.p_cancel <= 1e-9:
                assert o.expected.surcharge == pytest.approx(0.0, abs=0.01)
                assert o.expected.p_success == 1.0
            else:
                assert o.expected.surcharge > 0.0
                assert o.expected.p_success < 1.0
                assert "hailed" in o.quote.reliability.basis


def test_hailed_options_are_never_cheaper_than_their_fare_without_a_reason(trip):
    """Expected cost below the advertised fare is only legitimate when the
    expectation has been blended with a cheaper fallback — and then it is
    flagged. Anything else would be a modelling error."""
    c = compare(**trip, priority="balanced")
    for o in c.options:
        if o.quote.service_class is ServiceClass.HAILED and o.quote.available:
            if o.expected.expected_cost < o.quote.fare.amount - 0.5:
                assert o.expected.is_blended, o.quote.provider_id


def test_budget_filters_on_expected_cost(trip):
    c = compare(**trip, priority="cheapest", budget=20.0)
    for o in c.options:
        if o.feasible:
            assert o.expected.expected_cost <= 20.0
