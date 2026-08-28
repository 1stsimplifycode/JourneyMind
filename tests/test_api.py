"""API contract tests: status codes, validation, and honest error messages."""

from __future__ import annotations

import os
import sys

import pytest
from fastapi.testclient import TestClient

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "backend"))

from app.main import app  # noqa: E402


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


DEMO_BODY = {
    "origin": "pl_majestic_bus",
    "destination": "pl_indiranagar_100ft",
    "budget": 100,
    "max_time": 30,
    "preference": "balanced",
    "departure_time": "2025-01-07T09:00:00",
}


# --------------------------------------------------------------------------
def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    b = r.json()
    assert b["status"] == "ok"
    assert b["graph"]["nodes"] > 100
    assert b["model"]["loaded"]


def test_city_exposes_bounds_and_data_notice(client):
    b = client.get("/api/city").json()
    assert b["bbox"]["min_lat"] < b["bbox"]["max_lat"]
    assert b["data_notice"]["demo_mode"] is True
    assert b["data_notice"]["label"]
    assert any(r["mode"] == "metro" for r in b["routes"])


def test_places(client):
    b = client.get("/api/places").json()
    assert len(b["places"]) >= 5
    assert all({"place_id", "name", "lat", "lon"} <= set(p) for p in b["places"])


def test_models_registry_lists_all_six_baselines(client):
    b = client.get("/api/models").json()
    keys = {m["key"] for m in b["models"]}
    assert keys == {"freeflow", "historical", "gbt", "mlp", "graphsage", "gat"}
    assert sum(1 for m in b["models"] if m["active"]) == 1
    for m in b["models"]:
        assert m["available"] or m["reason"], "an unavailable model must say why"


def test_demo_runs_through_the_real_pipeline(client):
    """The demo endpoint computes for *now*, so this asserts only what is true
    at every hour. Which modes win is a function of the clock -- late at night
    a direct bike-taxi legitimately beats bike-plus-metro -- so the multimodal
    claim is tested separately, against a pinned departure time."""
    b = client.get("/api/demo").json()
    assert b["feasible"] is True
    assert b["scenario"]["title"]
    j = b["recommended"]
    assert j["legs"], "a recommendation must have legs"
    assert j["constraints"]["feasible"] is True
    assert b["explanation"]["headline"]
    assert b["pipeline"]["candidates"]["paths_found"] > 0


def test_the_demo_trip_is_multimodal_in_the_morning_peak(client):
    """Pinned to 09:00, where the mixed journey is the right answer and stays
    the right answer regardless of when the suite happens to run."""
    b = client.post("/api/recommend", json={
        "origin": {"place_id": "pl_wipro_sarjapur"},
        "destination": {"place_id": "pl_pes_university"},
        "departure_time": "2026-08-28T09:00:00",
        "budget": 250, "max_time": 120, "preference": "balanced",
    }).json()
    j = b["recommended"]
    assert len({m for m in j["modes"] if m != "walk"}) >= 2, j["summary"]


# --------------------------------------------------------------------------
def test_recommend_happy_path(client):
    r = client.post("/api/recommend", json=DEMO_BODY)
    assert r.status_code == 200
    b = r.json()
    assert b["feasible"] is True
    assert b["recommended"]["total_cost"]["display"]
    assert len(b["alternatives"]) >= 1
    assert b["model_info"]["model"]
    assert b["data_notice"]["demo_mode"] is True


def test_recommend_accepts_manual_weights(client):
    body = {**DEMO_BODY, "weights": {"cost": 0.9, "time": 0.1,
                                     "transfers": 0.0, "comfort": 0.0}}
    b = client.post("/api/recommend", json=body).json()
    assert b["preference"] == "custom"
    assert sum(b["weights"].values()) == pytest.approx(1.0, abs=1e-6)


@pytest.mark.parametrize("patch,expect_code", [
    ({"budget": 0}, 422),
    ({"budget": -5}, 422),
    ({"max_time": 0}, 422),
    ({"preference": "quickest"}, 422),
    ({"origin": "pl_not_a_place"}, 422),
    ({"modes": ["helicopter"]}, 422),
    ({"max_transfers": 99}, 422),
])
def test_invalid_input_is_rejected_with_422(client, patch, expect_code):
    r = client.post("/api/recommend", json={**DEMO_BODY, **patch})
    assert r.status_code == expect_code


def test_missing_fields_are_rejected(client):
    r = client.post("/api/recommend", json={"origin": "pl_home"})
    assert r.status_code == 422


def test_same_endpoints_rejected_with_a_useful_message(client):
    r = client.post("/api/recommend",
                    json={**DEMO_BODY, "destination": DEMO_BODY["origin"]})
    assert r.status_code == 422
    assert r.json()["detail"]["code"] == "same_endpoints"


def test_point_outside_study_area_rejected(client):
    r = client.post("/api/recommend", json={
        **DEMO_BODY, "destination": {"lat": 28.6139, "lon": 77.2090, "label": "Delhi"}})
    assert r.status_code == 422
    assert r.json()["detail"]["code"] == "outside_study_area"


def test_no_feasible_journey_returns_200_with_labelled_fallbacks(client):
    r = client.post("/api/recommend", json={**DEMO_BODY, "budget": 5, "max_time": 8})
    assert r.status_code == 200
    b = r.json()
    assert b["feasible"] is False
    assert b["recommended"] is None
    assert b["message"]
    assert b["fallbacks"], "must offer labelled near-misses"
    for f in b["fallbacks"]:
        assert f["journey"]["constraints"]["feasible"] is False
        assert f["label"] and f["why"]


def test_budget_too_low_only(client):
    b = client.post("/api/recommend",
                    json={**DEMO_BODY, "budget": 3, "max_time": 240}).json()
    if b["feasible"]:
        assert b["recommended"]["total_cost"]["amount"] <= 3
    else:
        assert b["fallbacks"]


def test_time_too_strict_only(client):
    b = client.post("/api/recommend",
                    json={**DEMO_BODY, "budget": 100000, "max_time": 4}).json()
    assert b["feasible"] is False
    assert b["fallbacks"]


# --------------------------------------------------------------------------
def test_coordinates_are_accepted_directly(client):
    b = client.post("/api/recommend", json={
        **DEMO_BODY,
        "origin": {"lat": 12.9776, "lon": 77.5715, "label": "Majestic"},
        "destination": {"lat": 12.9719, "lon": 77.6412, "label": "Indiranagar"},
    }).json()
    assert b["origin"]["label"] == "Majestic"
    assert b["recommended"] or b["fallbacks"]


def test_a_typed_place_name_is_accepted(client):
    """Free text is what a person types, so it has to reach the matcher.

    `resolve_point` has always been able to match a typed place name, but the
    PointInput validator used to reject label-only points before it ran, so
    every typed name failed at the schema boundary. Regression test for that.
    """
    b = client.post("/api/recommend", json={
        **DEMO_BODY,
        "origin": {"label": "Majestic Bus Station"},
        "destination": {"label": "Indiranagar 100ft Road"},
    })
    assert b.status_code == 200, b.json()
    assert b.json()["origin"]["label"] == "Majestic Bus Station"


def test_a_partial_lowercase_name_is_matched(client):
    b = client.post("/api/recommend", json={
        **DEMO_BODY, "origin": {"label": "majestic"}, "destination": {"label": "lalbagh"},
    })
    assert b.status_code == 200, b.json()
    assert "Majestic" in b.json()["origin"]["label"]


def test_a_bare_string_endpoint_is_accepted(client):
    """The documented shorthand: origin/destination as plain strings."""
    b = client.post("/api/recommend", json={
        **DEMO_BODY, "origin": "Majestic Bus Station", "destination": "M.G. Road",
    })
    assert b.status_code == 200, b.json()


def test_an_unmatched_name_explains_itself(client):
    """A name we do not know is a 422 with a sentence, never a schema dump."""
    b = client.post("/api/recommend", json={
        **DEMO_BODY, "origin": {"label": "Hogwarts"}, "destination": {"label": "Lalbagh West Gate"},
    })
    assert b.status_code == 422
    detail = b.json()["detail"]
    assert isinstance(detail, dict), "must be our error shape, not pydantic's list"
    assert detail["code"] == "unknown_place"
    assert "Hogwarts" in detail["error"]


def test_a_point_with_nothing_usable_is_rejected(client):
    b = client.post("/api/recommend", json={
        **DEMO_BODY, "origin": {"label": "   "}, "destination": {"label": "Lalbagh West Gate"},
    })
    assert b.status_code == 422


def test_every_ride_provider_is_priced_for_comparison(client):
    """The rider must see what each single app would have said.

    v1 section 3's worked example is a table of Bus / Rapido / Namma Yatri /
    Uber next to the winning combination. The reference journeys behind it have
    always been generated (baseline 5) and were being discarded by the budget
    and Pareto filters before anyone saw them.
    """
    b = client.post("/api/recommend", json={
        "origin": {"place_id": "pl_wipro_sarjapur"},
        "destination": {"place_id": "pl_pes_university"},
        "departure_time": "2026-08-28T09:00:00",
        "budget": 250, "max_time": 120, "preference": "balanced",
    }).json()
    rows = {r["mode"]: r for r in b["mode_comparison"]}
    for mode in ("bike_taxi", "auto", "cab"):
        assert mode in rows, f"{mode} missing from the comparison"
    for mode in ("metro", "bus"):
        assert mode in rows

    for row in rows.values():
        assert row["total_cost"]["display"]
        assert row["total_min"] > 0
        assert row["verdict"], "every row must say where it stands"


def test_unaffordable_providers_are_shown_and_labelled(client):
    """"You cannot afford it" is information, not a reason to hide the row."""
    b = client.post("/api/recommend", json={
        **DEMO_BODY,
        "origin": {"place_id": "pl_wipro_sarjapur"},
        "destination": {"place_id": "pl_pes_university"},
        "budget": 250, "max_time": 120,
    }).json()
    rows = {r["mode"]: r for r in b["mode_comparison"]}
    over = [r for r in rows.values() if not r["feasible"]]
    assert over, "expected at least one option to break a limit on this long trip"
    for r in over:
        assert ("budget" in r["verdict"] or "late" in r["verdict"]
                or "slow" in r["verdict"]), r["verdict"]


def test_a_comparison_row_is_one_answer_priced_door_to_door(client):
    """A row is ONE way of making the trip, whole.

    It used to be one vehicle, which meant the Metro row quoted ₹25 for a
    journey that also needed a bike taxi at each end -- a station is not a
    doorstep. A row is now the complete journey built around one spine, and it
    declares the hailed legs that got you to it.
    """
    b = client.get("/api/demo").json()
    rows = b["mode_comparison"]
    assert rows
    for r in rows:
        assert r["total_min"] > 0 and r["total_cost"]["display"]
        acc = r.get("access") or {}
        if r["mode"] in ("metro", "bus"):
            # a transit row may need a first/last mile, and must own up to it
            assert acc.get("rides", 0) == 0 or acc["mode"] in (
                "bike_taxi", "auto", "cab")
            if acc.get("rides"):
                assert acc["minutes"] > 0 and acc["fare"] > 0
        else:
            # a hailed row is one vehicle, door to door, with nothing bolted on
            assert r["transfers"] == 0
            assert acc.get("rides", 0) == 0


def test_every_fare_carries_a_provenance_label(client):
    b = client.get("/api/demo").json()
    total = b["recommended"]["total_cost"]
    assert total["provenance"] in {"exact", "published", "estimated"}
    for leg in b["recommended"]["legs"]:
        if leg["fare"]:
            assert leg["fare"]["provenance"] in {"exact", "published", "estimated"}
        assert leg["time_provenance"] in {"predicted", "estimated"}


def test_ride_hailing_fare_is_a_range_never_a_quote(client):
    b = client.get("/api/demo").json()
    ride = [l for l in b["recommended"]["legs"]
            if l["mode"] in {"bike_taxi", "auto", "cab"}]
    for leg in ride:
        assert leg["fare"]["provenance"] == "estimated"
        assert leg["fare"]["is_range"]
        assert "surge" in leg["fare"]["note"].lower() or "estimate" in leg["fare"]["note"].lower()


def test_unknown_api_route_is_404_not_the_spa(client):
    r = client.get("/api/definitely-not-a-route")
    assert r.status_code == 404


def test_openapi_document_builds(client):
    assert client.get("/api/openapi.json").status_code == 200
