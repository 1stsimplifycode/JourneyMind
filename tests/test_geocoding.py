"""Typing a place name that is not one of the bundled fifteen.

Most of this runs offline. The one test that actually calls Nominatim is marked
and skips when there is no network, because a suite that depends on donated
infrastructure is slow, flaky and somebody else's rate limit.
"""

from __future__ import annotations

import os
import socket
import sys

import pytest
from fastapi.testclient import TestClient

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "backend"))

from app.api.routes import _best_local_match                      # noqa: E402
from app.main import app                                          # noqa: E402
from app.services.geocode import _is_a_place, geocode             # noqa: E402

PES = "PES University, RR Campus (100 Feet Ring Road)"
TRIP = {"destination": PES, "budget": 400, "max_time": 240,
        "departure_time": "2026-08-28T09:00:00"}


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


class Place:
    def __init__(self, name):
        self.name, self.lat, self.lon = name, 0.0, 0.0


NAMES = ["Home (Vijayanagar)", "Jayanagar 4th Block", "College (Shanthinagar)",
         "Koramangala 5th Block", "Office (HSR Layout edge)", "M.G. Road",
         "PES University, RR Campus (100 Feet Ring Road)"]
PLACES = [Place(n) for n in NAMES]


# ==========================================================================
# matching the bundled list
# ==========================================================================
def test_a_name_inside_another_name_is_not_a_match():
    """"Jayanagar" resolved to **Vi**jayanagar, because one contains the other."""
    hit = _best_local_match("jayanagar", PLACES)
    assert hit is not None and hit.name == "Jayanagar 4th Block"


def test_an_exact_name_beats_a_prefix():
    assert _best_local_match("m.g. road", PLACES).name == "M.G. Road"


def test_a_word_inside_the_name_still_matches():
    """"HSR Layout" is a word in "Office (HSR Layout edge)"."""
    assert _best_local_match("hsr layout", PLACES).name == "Office (HSR Layout edge)"


def test_nonsense_matches_nothing():
    assert _best_local_match("zzzzz", PLACES) is None


# ==========================================================================
# what the geocoder will and will not accept back
# ==========================================================================
def test_a_bus_route_is_not_a_destination():
    """"Hebbal" came back as the relation "Red Line (Sarjapur to Hebbal)" and
    put the rider on a point somewhere along a bus route."""
    assert not _is_a_place({"class": "route", "type": "bus"})
    assert not _is_a_place({"class": "railway", "type": "rail"})
    assert not _is_a_place({"class": "highway", "type": "bus_stop"})
    assert _is_a_place({"class": "place", "type": "suburb"})
    assert _is_a_place({"class": "building", "type": "office"})


def test_the_geocoder_is_off_in_this_suite():
    """conftest sets JM_GEOCODER=0 so the tests never call out."""
    assert geocode("Whitefield", {"min_lat": 12.8, "max_lat": 13.1,
                                  "min_lon": 77.4, "max_lon": 77.8}) is None


def test_an_unknown_place_still_fails_with_a_reason(client):
    r = client.post("/api/recommend", json={**TRIP, "origin": "Whitefield"})
    assert r.status_code == 422
    d = r.json()["detail"]
    assert d["code"] == "unknown_place"
    assert "corridor" in d["detail"] or "places" in d["detail"]


def test_a_bundled_place_never_needs_the_network(client):
    """With the geocoder off, the fifteen still work."""
    r = client.post("/api/recommend", json={**TRIP, "origin": "Jayanagar"})
    assert r.status_code == 200, r.json()
    assert r.json()["origin"]["label"] == "Jayanagar 4th Block"


def test_a_typed_coordinate_never_needs_the_network(client):
    r = client.post("/api/recommend", json={**TRIP, "origin": "12.9345, 77.6100"})
    assert r.status_code == 200, r.json()


# ==========================================================================
# the real thing, when there is a network
# ==========================================================================
def _online() -> bool:
    try:
        socket.create_connection(("nominatim.openstreetmap.org", 443), timeout=4).close()
        return True
    except OSError:
        return False


@pytest.mark.skipif(not _online(), reason="no network for Nominatim")
def test_nominatim_resolves_a_real_place_inside_the_corridor(monkeypatch):
    """The point of the whole thing: a name nobody typed into places.json."""
    from app.config import get_settings
    from app.services.geocode import geocode as live_geocode
    monkeypatch.setattr(get_settings(), "geocoder_enabled", True)

    bbox = {"min_lat": 12.895, "max_lat": 13.006,
            "min_lon": 77.525, "max_lon": 77.696}
    hit = live_geocode("BTM Layout", bbox)
    assert hit is not None, "BTM Layout is inside the corridor and real"
    lat, lon, name = hit
    assert bbox["min_lat"] <= lat <= bbox["max_lat"]
    assert bbox["min_lon"] <= lon <= bbox["max_lon"]
    assert name


@pytest.mark.skipif(not _online(), reason="no network for Nominatim")
def test_a_place_outside_the_corridor_is_refused(monkeypatch):
    """`bounded=1` is what stops "Springfield" resolving to Illinois."""
    from app.config import get_settings
    from app.services.geocode import geocode as live_geocode
    monkeypatch.setattr(get_settings(), "geocoder_enabled", True)
    bbox = {"min_lat": 12.895, "max_lat": 13.006,
            "min_lon": 77.525, "max_lon": 77.696}
    assert live_geocode("Springfield Illinois", bbox) is None


# ==========================================================================
# a pasted address
# ==========================================================================
LONG_ADDRESS = ("Ericsson Global, A Block, Citrine Block SEZ, Bagmane World "
                "Technology Centre, Outer Ring Rd, Laxmi Sagar Layout, "
                "Mahadevapura, Bengaluru, Karnataka 560048")


def test_a_pasted_address_is_not_too_long_for_the_schema(client):
    """154 characters against a 120-character limit: the rider got a raw
    Pydantic error before any place logic ran."""
    assert len(LONG_ADDRESS) > 120
    r = client.post("/api/compare", json={
        "origin": LONG_ADDRESS, "destination": PES,
        "departure_time": "2026-08-28T09:00:00"})
    # the geocoder is off in this suite, so this must be OUR error, with a
    # reason -- never a schema rejection
    assert r.status_code in (200, 422)
    if r.status_code == 422:
        detail = r.json()["detail"]
        assert isinstance(detail, dict), f"raw schema error: {detail}"
        assert detail["code"] == "unknown_place"


def test_an_over_specified_address_is_retried_shorter():
    """Nominatim answers "Mahadevapura, Bengaluru" and does not answer the
    same building written out in full."""
    from app.services.geocode import _shorten
    tries = _shorten(LONG_ADDRESS)
    assert tries[0] == LONG_ADDRESS
    assert len(tries) > 1
    # each attempt drops a leading component
    assert all(len(t) <= len(tries[0]) for t in tries)
    assert any("Mahadevapura" in t and "Ericsson" not in t for t in tries)


def test_a_single_component_query_is_not_shortened():
    from app.services.geocode import _shorten
    assert _shorten("Whitefield") == ["Whitefield"]


def test_a_station_is_a_place_but_a_rail_line_is_not():
    """Banning the whole `railway` class to stop route relations also threw
    away Mahadevapura, which is a station inside the corridor."""
    assert _is_a_place({"class": "railway", "type": "station"})
    assert _is_a_place({"class": "railway", "type": "halt"})
    assert not _is_a_place({"class": "railway", "type": "rail"})
    assert not _is_a_place({"class": "route", "type": "subway"})
