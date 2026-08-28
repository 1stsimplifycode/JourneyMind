"""Turning a typed place name into a point, using OpenStreetMap.

Before this existed, only the fifteen names in `places.json` worked. Anything
else -- "Whitefield", "BTM Layout", "Electronic City Phase 1" -- came back
`Could not find 'Whitefield' in this study area`, which was true of the bundled
list and useless to somebody who lives there.

    local places   exact, then substring        offline, instant, always first
    Nominatim      bounded to the study bbox    free, keyless, ODbL 1.0
    lat,lon        parsed by the schema         no lookup needed at all

WHY IT CANNOT HANG A REQUEST
----------------------------
A geocoder is a network call on the critical path of a page load, so every
failure mode ends the same way: return None, let the caller give the honest
"could not find that" answer it always gave. A short timeout, no retries, a
disk cache that survives restarts, and one request per second as Nominatim
asks. Turn it off entirely with `JM_GEOCODER=0` and the product behaves exactly
as it did before.

WHAT LEAVES THIS PROCESS
------------------------
The place name the rider typed, and nothing else. No coordinates of theirs, no
identifier, no other field of the request. That is inherent to asking a
geocoder a question, and it is the only outbound call the application makes.
"""

from __future__ import annotations

import json
import logging
import threading
import time
import urllib.parse
import urllib.request
from pathlib import Path

from ..config import get_settings

log = logging.getLogger("journeymind.geocode")

NOMINATIM = "https://nominatim.openstreetmap.org/search"
UA = {"User-Agent": "JourneyMind/1.0 (mobility research prototype)"}

#: Nominatim asks for at most one request a second. Honoured process-wide.
_MIN_INTERVAL_S = 1.1
_lock = threading.Lock()
_last_call = 0.0

_memory: dict[str, tuple[float, float, str] | None] = {}
_disk_loaded = False


def _cache_path() -> Path:
    return Path(get_settings().data_dir) / "_geocode_cache.json"


def _load_disk() -> None:
    global _disk_loaded
    if _disk_loaded:
        return
    _disk_loaded = True
    p = _cache_path()
    if not p.exists():
        return
    try:
        for k, v in json.loads(p.read_text(encoding="utf-8")).items():
            _memory[k] = tuple(v) if v else None
    except Exception:
        log.warning("geocode cache at %s is unreadable; starting empty", p)


def _save_disk() -> None:
    try:
        p = _cache_path()
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(
            {k: list(v) if v else None for k, v in _memory.items()},
            indent=0), encoding="utf-8")
    except OSError:
        pass                      # a read-only deployment is not a failure


def geocode(query: str, bbox: dict) -> tuple[float, float, str] | None:
    """A point inside `bbox` for this name, or None.

    None covers every unhappy path -- not found, outside the study area, no
    network, geocoder disabled -- because the caller's answer is the same in
    all of them and a page load must not wait to learn which it was.
    """
    s = get_settings()
    if not s.geocoder_enabled:
        return None
    q = " ".join(query.strip().split())
    if len(q) < 3:
        return None

    _load_disk()
    key = q.lower()
    if key in _memory:
        return _memory[key]

    hit = None
    for attempt in _shorten(q):
        hit = _query_nominatim(attempt, bbox, s.geocoder_timeout_s)
        if hit is not None:
            break
    _memory[key] = hit
    _save_disk()
    return hit


#: How many progressively shorter forms of an address to try.
MAX_SHORTENINGS = 4


def _shorten(q: str) -> list[str]:
    """The address, then less of it, the way a person would retype it.

    Nominatim answers "Mahadevapura Bengaluru" and does NOT answer "Ericsson
    Global, A Block, Citrine Block SEZ, Bagmane World Technology Centre, Outer
    Ring Rd, Laxmi Sagar Layout, Mahadevapura, Bengaluru, Karnataka 560048" --
    the same building, over-specified. Somebody pasting an address from a
    signature block should not have to know that, so the leading components are
    dropped one at a time until something resolves.
    """
    parts = [p.strip() for p in q.split(",") if p.strip()]
    if len(parts) < 2:
        return [q]
    tries = [q]
    for drop in range(1, min(len(parts) - 1, MAX_SHORTENINGS + 1)):
        tries.append(", ".join(parts[drop:]))
    # ...and the tail on its own: "Mahadevapura, Bengaluru"
    if len(parts) >= 3:
        tail = ", ".join(parts[-3:])
        if tail not in tries:
            tries.append(tail)
    return tries[:MAX_SHORTENINGS + 2]


#: Object classes that are somewhere you can go. Everything else Nominatim
#: knows about -- bus routes, rail lines, administrative boundaries -- can
#: match a name without being a destination: "Hebbal" returned the relation
#: "Red Line (Sarjapur to Hebbal)" and put the rider on a point somewhere along
#: a bus route.
PLACE_CLASSES = frozenset({
    "place", "building", "amenity", "office", "shop", "highway",
    "landuse", "leisure", "tourism", "healthcare", "education", "man_made",
})

#: ...and these are never a destination even when their class looks fine.
NOT_A_PLACE_TYPES = frozenset({"bus_route", "route", "bus_stop", "platform"})

#: Railway objects that ARE somewhere you can go. Banning the whole `railway`
#: class to stop route relations also threw away stations, and "Mahadevapura"
#: -- a real metro station inside the corridor -- came back unfindable.
RAILWAY_PLACE_TYPES = frozenset({
    "station", "halt", "stop", "tram_stop", "subway_entrance",
})


def _is_a_place(hit: dict) -> bool:
    klass, typ = hit.get("class"), hit.get("type")
    if klass == "route":                       # a bus route is not a place
        return False
    if klass == "railway":
        return typ in RAILWAY_PLACE_TYPES      # the station yes, the line no
    if typ in NOT_A_PLACE_TYPES:
        return False
    return klass in PLACE_CLASSES


def _query_nominatim(q: str, bbox: dict, timeout: float):
    global _last_call
    # `bounded=1` with a viewbox is what keeps "Springfield" from resolving to
    # Illinois: the answer has to be inside the corridor or there is no answer.
    params = urllib.parse.urlencode({
        "q": q, "format": "json", "limit": 8, "bounded": 1,
        "viewbox": f"{bbox['min_lon']},{bbox['max_lat']},"
                   f"{bbox['max_lon']},{bbox['min_lat']}",
    })
    try:
        with _lock:
            wait = _MIN_INTERVAL_S - (time.monotonic() - _last_call)
            if wait > 0:
                time.sleep(wait)
            _last_call = time.monotonic()
        req = urllib.request.Request(f"{NOMINATIM}?{params}", headers=UA)
        with urllib.request.urlopen(req, timeout=timeout) as r:
            results = json.load(r)
    except Exception as exc:                    # network, DNS, timeout, 4xx
        log.info("geocoder unavailable for %r (%s); falling back to the "
                 "bundled place list", q, type(exc).__name__)
        return None

    results = [r for r in results if _is_a_place(r)]
    if not results:
        return None
    top = results[0]
    try:
        lat, lon = float(top["lat"]), float(top["lon"])
    except (KeyError, TypeError, ValueError):
        return None
    if not (bbox["min_lat"] <= lat <= bbox["max_lat"]
            and bbox["min_lon"] <= lon <= bbox["max_lon"]):
        return None
    name = ", ".join(
        p.strip() for p in str(top.get("display_name", q)).split(",")[:2])
    return lat, lon, name or q
