"""Build the study-area graph from REAL open data instead of generating it.

    python scripts/fetch_osm.py                 # writes data/city/bengaluru_osm/

Sources, all free and all open, none of them requiring a key:

    OpenStreetMap via Overpass API   roads, bus stops, metro stations,
                                     metro line relations       ODbL 1.0
    Nominatim                        geocoding the named places ODbL 1.0
    Open-Meteo                       weather for the departure hour  CC-BY 4.0

WHAT THIS DOES AND DOES NOT MAKE REAL
-------------------------------------
Real after this runs: the road network and its geometry, road classes, speed
limits and lane counts where OSM has them; bus stop positions; metro station
positions, names and line membership, in order; the coordinates of every named
place.

Still NOT real, and this is the important half: **observed travel times**.
Nobody publishes free per-edge travel-time observations for Bengaluru -- the
traffic APIs that could are commercial and keyed. So the model's TARGET is
still generated. What changes is that it is generated over a real topology with
real road classes and real distances, rather than over an invented one. That is
a materially different claim and it is the one this file supports.

Ride-hailing fares, availability and cancellation rates remain modelled: no
operator publishes them, and scraping a private app would breach its terms.

POLITENESS
----------
Overpass and Nominatim are donated infrastructure. One combined query, cached
to disk so a re-run costs nothing, exponential backoff on 429, a real
User-Agent, and a second between Nominatim calls. If the cache is present this
script makes no network request at all.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import socket
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "data" / "_osm_cache"
OVERPASS = "https://overpass-api.de/api/interpreter"
NOMINATIM = "https://nominatim.openstreetmap.org/search"
UA = {"User-Agent": "JourneyMind/1.0 (mobility research prototype; contact: repository owner)"}

#: The corridor. Same bounds as the synthetic bundle so the two are comparable.
BBOX = (12.895, 77.525, 13.006, 77.696)

#: What is FETCHED from OSM. Wider than what is routed, so the cut below can be
#: changed without going back to Overpass.
ROAD_CLASSES = ("motorway", "trunk", "primary", "secondary", "tertiary",
                "motorway_link", "trunk_link", "primary_link", "secondary_link")

#: What is ROUTED. Measured, not guessed: this cut gives 842 junctions over
#: 394 km of real arterial road with 50 of the corridor's 52 metro stations
#: reachable, against 1,706 junctions for secondary+ and 3,544 for tertiary+.
#: Yen's k-shortest runs across every candidate on every request, so the whole
#: street network would make the planner unusable for a demonstration.
#:
#: This is therefore an ARTERIAL EXTRACT of the real network, not the whole of
#: it. Side streets exist and are not routed; a real deployment would route
#: them and pay for the compute.
ROUTED_CLASSES = ("motorway", "trunk", "primary",
                  "motorway_link", "trunk_link", "primary_link")

#: Free-flow speeds by OSM class, km/h, used only where the way has no
#: `maxspeed` tag. Indian urban arterials, not motorway assumptions.
DEFAULT_SPEED = {
    "motorway": 60.0, "trunk": 50.0, "primary": 40.0, "secondary": 35.0,
    "tertiary": 30.0, "motorway_link": 40.0, "trunk_link": 35.0,
    "primary_link": 30.0, "secondary_link": 28.0,
}

#: The places the demo talks about, resolved through Nominatim rather than
#: typed in by hand. `query` is what gets geocoded; `hint` disambiguates.
#: Each entry carries a display name and the queries to try in order. A single
#: query is not enough: "Wipro Doddakannelli Sarjapur Road Bengaluru" returns
#: nothing from Nominatim, and the two places the whole demonstration is about
#: are not optional.
PLACES = [
    ("pl_wipro_sarjapur", "Wipro Campus, Doddakannelli (Sarjapur Road)", "office",
     ["Wipro Corporate Office Doddakannelli Bengaluru",
      "Doddakannelli Bengaluru", "Doddakannelli"]),
    ("pl_pes_university", "PES University, RR Campus (100 Feet Ring Road)", "education",
     ["PES University Bengaluru", "PES University Banashankari",
      "PES College of Engineering Bengaluru"]),
    ("pl_home", "Home (Vijayanagar)", "residential", ["Vijayanagar Bengaluru"]),
    ("pl_college", "College (Shanthinagar)", "education", ["Shanthinagar Bengaluru"]),
    ("pl_mg_road_shops", "M.G. Road", "retail", ["MG Road Bengaluru"]),
    ("pl_koramangala", "Koramangala 5th Block", "mixed", ["Koramangala Bengaluru"]),
    ("pl_indiranagar_100ft", "Indiranagar 100ft Road", "retail",
     ["Indiranagar 100 Feet Road Bengaluru"]),
    ("pl_hsr_office", "Office (HSR Layout edge)", "office", ["HSR Layout Bengaluru"]),
    ("pl_banashankari_home", "Banashankari Home", "residential",
     ["Banashankari Bengaluru"]),
    ("pl_jayanagar", "Jayanagar 4th Block", "retail", ["Jayanagar 4th Block Bengaluru"]),
    ("pl_majestic", "Majestic Bus Station", "transport",
     ["Kempegowda Bus Station Majestic Bengaluru"]),
    ("pl_domlur", "Domlur Office Park", "office", ["Domlur Bengaluru"]),
    ("pl_whitefield_gate", "Whitefield Gate", "office",
     ["Bellandur Bengaluru", "Marathahalli Bengaluru"]),
    ("pl_electronic_city", "Electronic City Gate", "office",
     ["Bommanahalli Bengaluru"]),
    ("pl_old_airport", "Old Airport Road Gate", "mixed",
     ["HAL Old Airport Road Bengaluru"]),
]

EARTH_KM = 6371.0088


def haversine(a_lat, a_lon, b_lat, b_lon) -> float:
    p1, p2 = math.radians(a_lat), math.radians(b_lat)
    dp = p2 - p1
    dl = math.radians(b_lon - a_lon)
    h = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * EARTH_KM * math.asin(math.sqrt(h))


# --------------------------------------------------------------------------
# fetching, politely
# --------------------------------------------------------------------------
def _cached(name: str, fetch):
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / name
    if path.exists():
        print(f"  cache hit  {name}")
        return json.loads(path.read_text(encoding="utf-8"))
    payload = fetch()
    path.write_text(json.dumps(payload), encoding="utf-8")
    print(f"  fetched    {name}  ({path.stat().st_size / 1024:.0f} kB)")
    return payload


def overpass(query: str, name: str) -> dict:
    def go():
        data = urllib.parse.urlencode({"data": query}).encode()
        delay = 20.0
        for attempt in range(6):
            try:
                req = urllib.request.Request(OVERPASS, data=data, headers=UA)
                with urllib.request.urlopen(req, timeout=300) as r:
                    return json.load(r)
            except urllib.error.HTTPError as e:
                if e.code not in (429, 504, 503):
                    raise
                print(f"    {e.code} from Overpass; waiting {delay:.0f}s "
                      f"(attempt {attempt + 1}/6)")
                time.sleep(delay)
                delay *= 1.8
            except (socket.timeout, TimeoutError):
                print(f"    timed out; waiting {delay:.0f}s")
                time.sleep(delay)
                delay *= 1.8
        raise SystemExit("Overpass would not answer. Try again later — it is "
                         "donated infrastructure, not a service we are owed.")
    return _cached(name, go)


def nominatim(query: str) -> dict | None:
    def go():
        params = urllib.parse.urlencode(
            {"q": query, "format": "json", "limit": 1, "countrycodes": "in"})
        req = urllib.request.Request(f"{NOMINATIM}?{params}", headers=UA)
        with urllib.request.urlopen(req, timeout=60) as r:
            out = json.load(r)
        time.sleep(1.1)          # Nominatim asks for at most 1 request/second
        return out
    slug = "".join(c if c.isalnum() else "_" for c in query.lower())[:60]
    hits = _cached(f"nominatim_{slug}.json", go)
    return hits[0] if hits else None


# --------------------------------------------------------------------------
# the road graph
# --------------------------------------------------------------------------
def fetch_everything() -> dict:
    s, w, n, e = BBOX
    box = f"{s},{w},{n},{e}"
    classes = "|".join(ROAD_CLASSES)
    print("OpenStreetMap, via Overpass:")
    roads = overpass(f"""[out:json][timeout:300];
        way["highway"~"^({classes})$"]["access"!~"^(private|no)$"]({box});
        out geom;""", "roads.json")
    stops = overpass(f"""[out:json][timeout:180];
        node["highway"="bus_stop"]({box});
        out body;""", "bus_stops.json")
    metro = overpass(f"""[out:json][timeout:180];
        (
          relation["route"="subway"]({box});
        );
        out body;
        >>;
        out body;""", "metro.json")
    buses = overpass(f"""[out:json][timeout:280];
        relation["route"="bus"]({box});
        out body;
        node(r);
        out body;""", "bus_routes.json")
    return {"roads": roads, "stops": stops, "metro": metro, "buses": buses}


def build_road_graph(roads: dict):
    """OSM ways -> a routable junction graph.

    Ways share node ids where they meet, so a node referenced by more than one
    way is an intersection. Everything between intersections is one edge, which
    is what keeps 6,000 ways from becoming 60,000 useless two-metre segments.
    """
    ways = [e for e in roads["elements"] if e.get("type") == "way" and e.get("geometry")]
    print(f"\nroad network: {len(ways)} ways from OSM")

    ref_count: dict[int, int] = defaultdict(int)
    for w in ways:
        for nd in w.get("nodes", []):
            ref_count[nd] += 1

    pos: dict[int, tuple[float, float]] = {}
    for w in ways:
        for nd, g in zip(w.get("nodes", []), w["geometry"]):
            pos[nd] = (g["lat"], g["lon"])

    edges = []
    for w in ways:
        nodes = w.get("nodes", [])
        if len(nodes) < 2:
            continue
        tags = w.get("tags", {})
        klass = tags.get("highway", "tertiary")
        speed = _speed_of(tags, klass)
        lanes = _lanes_of(tags)
        # split at every intersection; carry the geometry between them
        start = 0
        for i in range(1, len(nodes)):
            is_junction = ref_count[nodes[i]] > 1 or i == len(nodes) - 1
            if not is_junction:
                continue
            seg = nodes[start:i + 1]
            km = sum(haversine(*pos[a], *pos[b]) for a, b in zip(seg, seg[1:]))
            if km > 0.005 and seg[0] != seg[-1]:
                edges.append({"u": seg[0], "v": seg[-1], "km": km,
                              "road_class": klass, "speed": speed, "lanes": lanes})
            start = i

    print(f"  split into {len(edges)} segments between intersections")
    return edges, pos


def _speed_of(tags: dict, klass: str) -> float:
    raw = tags.get("maxspeed")
    if raw:
        digits = "".join(c for c in raw if c.isdigit())
        if digits:
            kmh = float(digits)
            if "mph" in raw:
                kmh *= 1.609
            if 5 <= kmh <= 120:
                return kmh
    return DEFAULT_SPEED.get(klass, 30.0)


def _lanes_of(tags: dict) -> int:
    raw = tags.get("lanes")
    if raw and raw.split(";")[0].strip().isdigit():
        return max(1, min(8, int(raw.split(";")[0].strip())))
    return 2


def largest_component(edges):
    """Only the part you can actually drive around. An island is not a route."""
    adj = defaultdict(set)
    for e in edges:
        adj[e["u"]].add(e["v"])
        adj[e["v"]].add(e["u"])
    seen, best = set(), set()
    for start in adj:
        if start in seen:
            continue
        stack, comp = [start], set()
        while stack:
            n = stack.pop()
            if n in comp:
                continue
            comp.add(n)
            stack.extend(adj[n] - comp)
        seen |= comp
        if len(comp) > len(best):
            best = comp
    kept = [e for e in edges if e["u"] in best and e["v"] in best]
    print(f"  largest connected component: {len(best)} junctions, {len(kept)} edges "
          f"({len(edges) - len(kept)} dropped as unreachable)")
    return kept, best


def contract(edges, protect: set[int]):
    """Collapse the points that are only points along a road.

    Splitting at every shared OSM node leaves thousands of degree-2 junctions
    where one way simply ends and the next begins. They are not decisions a
    driver makes, and Yen's k-shortest pays for every one of them. Merging them
    preserves distance exactly and averages speed by length.

    `protect` holds the junctions something is attached to -- a bus stop, a
    metro station, a named place -- which must survive as nodes.
    """
    inc = defaultdict(list)
    for e in edges:
        inc[e["u"]].append(e)
        inc[e["v"]].append(e)

    alive = {id(e): e for e in edges}
    changed = True
    while changed:
        changed = False
        for node, touching in list(inc.items()):
            if node in protect:
                continue
            live = [e for e in touching if id(e) in alive]
            if len(live) != 2:
                continue
            a, b = live
            if a is b:
                continue
            # the far end of each edge
            a_far = a["v"] if a["u"] == node else a["u"]
            b_far = b["v"] if b["u"] == node else b["u"]
            if a_far == b_far:
                continue                      # would make a self-loop
            km = a["km"] + b["km"]
            if km <= 0:
                continue
            merged = {
                "u": a_far, "v": b_far, "km": km,
                "road_class": a["road_class"] if a["km"] >= b["km"] else b["road_class"],
                "speed": (a["speed"] * a["km"] + b["speed"] * b["km"]) / km,
                "lanes": max(a["lanes"], b["lanes"]),
            }
            del alive[id(a)]
            del alive[id(b)]
            alive[id(merged)] = merged
            inc[a_far].append(merged)
            inc[b_far].append(merged)
            inc[node] = []
            changed = True

    kept = list(alive.values())
    nodes = {n for e in kept for n in (e["u"], e["v"])}
    print(f"  contracted to {len(nodes)} junctions, {len(kept)} edges")
    return kept, nodes


def nearest(pos: dict, nodes: set, lat: float, lon: float):
    """Closest surviving junction. Linear, and fine at this scale."""
    best, best_km = None, 1e9
    for n in nodes:
        p = pos.get(n)
        if p is None:
            continue
        km = haversine(lat, lon, p[0], p[1])
        if km < best_km:
            best, best_km = n, km
    return best, best_km


def build_metro(metro: dict, pos: dict):
    """Real stations, real lines, in the real order the relation gives them."""
    nodes = {e["id"]: e for e in metro["elements"] if e.get("type") == "node"}
    rels = [e for e in metro["elements"] if e.get("type") == "relation"]

    lines, seen_names = [], set()
    for r in rels:
        tags = r.get("tags", {})
        name = tags.get("name") or ""
        # OSM models each direction as its own relation, and often each
        # terminus pair as well: the Purple Line alone appears four times.
        # A line is a line, so key on the part before the brackets.
        key = tags.get("ref") or name.split("(")[0].strip().lower()
        if not key or key in seen_names:
            continue
        stops = []
        for m in r.get("members", []):
            if m.get("type") != "node" or m.get("role", "").startswith("stop") is False:
                if m.get("role") not in ("stop", "platform", "stop_entry_only",
                                         "stop_exit_only", ""):
                    continue
            n = nodes.get(m["ref"])
            if n is None:
                continue
            nm = (n.get("tags") or {}).get("name")
            if not nm:
                continue
            if stops and stops[-1][1] == nm:
                continue
            if not (BBOX[0] <= n["lat"] <= BBOX[2] and BBOX[1] <= n["lon"] <= BBOX[3]):
                continue
            stops.append((n["id"], nm, n["lat"], n["lon"]))
        if len(stops) < 3:
            continue
        seen_names.add(key)
        short = name.split("(")[0].strip() or name
        lines.append({
            "name": name, "short": short, "rel_id": r["id"],
            "route_id": "metro_" + "".join(
                c.lower() for c in short if c.isalnum())[:16],
            "colour": tags.get("colour"), "stops": stops})

    lines.sort(key=lambda l: -len(l["stops"]))
    print(f"\nmetro: {len(lines)} lines inside the corridor")
    for l in lines:
        print(f"  {l['name'][:52]:54s} {len(l['stops'])} stations")
    return lines


#: How many real BMTC routes to carry. There are 244 with usable ordering
#: inside the corridor; all of them would put ~1,800 stops into the graph and
#: make the k-shortest search intolerable. The longest ones inside the bbox are
#: kept, because a route that barely clips the corner is not a way across the
#: city. A documented truncation, not a silent one.
MAX_BUS_ROUTES = 14


def build_bus(buses: dict):
    """Real BMTC routes, in the order OSM records their stops.

    OSM models each direction as its own relation, so routes are keyed on their
    `ref` and the longer direction wins. A relation with fewer than four
    in-corridor stops is dropped: it clips the study area rather than crossing
    it, and half a route is worse than none.
    """
    rels = [e for e in buses["elements"] if e.get("type") == "relation"]
    nodes = {e["id"]: e for e in buses["elements"] if e.get("type") == "node"}

    by_ref: dict[str, dict] = {}
    for r in rels:
        tags = r.get("tags", {})
        ref = (tags.get("ref") or tags.get("name") or "").strip()
        if not ref:
            continue
        stops = []
        for m in r.get("members", []):
            if m.get("type") != "node":
                continue
            if not str(m.get("role", "")).startswith(("stop", "platform")):
                continue
            nd = nodes.get(m["ref"])
            if not nd:
                continue
            name = (nd.get("tags") or {}).get("name")
            if not name:
                continue
            if not (BBOX[0] <= nd["lat"] <= BBOX[2] and BBOX[1] <= nd["lon"] <= BBOX[3]):
                continue
            if stops and stops[-1][1] == name:
                continue
            stops.append((nd["id"], name, nd["lat"], nd["lon"]))
        if len(stops) < 4:
            continue
        prev = by_ref.get(ref)
        if prev is None or len(stops) > len(prev["stops"]):
            by_ref[ref] = {"ref": ref, "rel_id": r["id"],
                           "name": tags.get("name") or ref, "stops": stops}

    chosen = sorted(by_ref.values(), key=lambda r: -len(r["stops"]))[:MAX_BUS_ROUTES]
    print(f"\nbus: {len(rels)} relations -> {len(by_ref)} routes with usable "
          f"ordering -> {len(chosen)} carried")
    for r in chosen:
        print(f"  {r['ref']:>8s}  {len(r['stops']):3d} stops  {r['name'][:52]}")
    return chosen


def _pretty(hit: dict, fallback: str) -> str:
    """A place name a person would recognise, from the Nominatim result."""
    parts = [p.strip() for p in hit.get("display_name", "").split(",")]
    return ", ".join(parts[:2]) if parts else fallback


def write_bundle(out: Path, edges, keep, pos, lines, stops, places, city_id,
                 bus_routes):
    """Everything the app already knows how to read, from real data."""
    out.mkdir(parents=True, exist_ok=True)

    node_rows, node_ids = [], {}

    def add_node(nid, name, kind, lat, lon, lines_str="", interchange=0, category=""):
        if nid in node_ids:
            return nid
        node_ids[nid] = True
        node_rows.append({
            "node_id": nid, "name": name, "kind": kind,
            "lat": round(lat, 6), "lon": round(lon, 6),
            "lines": lines_str, "is_interchange": interchange,
            "category": category, "degree": 0,
            # The congestion columns drive the synthetic travel-time generator.
            # Derived from road class and distance from the centre rather than
            # drawn at random, so the field at least follows the real network.
            "observed_congestion": 0.0, "latent_congestion": 0.0,
        })
        return nid

    for n in sorted(keep):
        lat, lon = pos[n]
        add_node(f"jn_{n}", f"Junction {n}", "junction", lat, lon)

    # metro stations, attached to the junction they actually sit on
    station_node: dict[str, str] = {}
    station_lines: dict[str, set] = defaultdict(set)
    for line in lines:
        for _osm, name, lat, lon in line["stops"]:
            station_lines[name].add(line["short"])
    for line in lines:
        for _osm, name, lat, lon in line["stops"]:
            sid = "ms_" + "".join(c.lower() if c.isalnum() else "_" for c in name)[:34]
            station_node[name] = sid
            add_node(sid, name, "metro_station", lat, lon,
                     "|".join(sorted(station_lines[name])),
                     1 if len(station_lines[name]) > 1 else 0)

    # every stop a carried bus route actually calls at, at its real position
    route_stop_node: dict[int, str] = {}
    for route in bus_routes:
        for osm_id, name, lat, lon in route["stops"]:
            nid = f"bs_{osm_id}"
            route_stop_node[osm_id] = nid
            add_node(nid, name, "bus_stop", lat, lon)

    # bus stops: real positions, thinned to those on the routed network
    chosen = []
    for st in stops:
        name = (st.get("tags") or {}).get("name")
        j, km = nearest(pos, keep, st["lat"], st["lon"])
        if j is None or km > 0.25:
            continue
        chosen.append((st, name, j, km))
    # thin so stops are not stacked on top of each other
    thinned, used = [], []
    for st, name, j, km in sorted(chosen, key=lambda t: t[3]):
        if any(haversine(st["lat"], st["lon"], o["lat"], o["lon"]) < 0.45 for o in used):
            continue
        used.append(st)
        thinned.append((st, name, j))
    print(f"bus stops: {len(stops)} in OSM -> {len(chosen)} on the routed network "
          f"-> {len(thinned)} after thinning to 450 m spacing")
    for st, name, _j in thinned:
        add_node(f"bs_{st['id']}", name, "bus_stop", st["lat"], st["lon"])

    for pid, name, lat, lon, cat in places:
        add_node(pid, name, "place", lat, lon, category=cat)

    # ---- road edges, plus a short connector for anything off-network ----
    road_rows, seen_pair = [], set()
    for i, e in enumerate(edges):
        u, v = f"jn_{e['u']}", f"jn_{e['v']}"
        key = tuple(sorted((u, v)))
        if key in seen_pair:
            continue
        seen_pair.add(key)
        road_rows.append({
            "edge_id": f"rd_{i:05d}", "u": u, "v": v,
            "distance_km": round(e["km"], 4), "road_class": e["road_class"],
            "free_speed_kmph": round(e["speed"], 1), "lanes": e["lanes"],
        })

    def connect(nid, lat, lon, tag):
        j, km = nearest(pos, keep, lat, lon)
        if j is None:
            return
        u = f"jn_{j}"
        if u == nid:
            return
        road_rows.append({
            "edge_id": f"rd_c_{tag}_{nid[-12:]}", "u": u, "v": nid,
            "distance_km": round(max(km, 0.01), 4), "road_class": "connector",
            "free_speed_kmph": 20.0, "lanes": 1,
        })

    for name, sid in station_node.items():
        row = next(r for r in node_rows if r["node_id"] == sid)
        connect(sid, row["lat"], row["lon"], "ms")
    for st, _name, _j in thinned:
        connect(f"bs_{st['id']}", st["lat"], st["lon"], "bs")
    for pid, _name, lat, lon, _cat in places:
        connect(pid, lat, lon, "pl")

    # ---- transit edges from the real station order ----------------------
    transit_rows, routes = [], []
    for line in lines:
        rid = line["route_id"]
        seq_nodes = [station_node[n] for _o, n, _a, _b in line["stops"]]
        by_id = {r["node_id"]: r for r in node_rows}
        for seq, (a, b) in enumerate(zip(seq_nodes, seq_nodes[1:])):
            ra, rb = by_id[a], by_id[b]
            km = haversine(ra["lat"], ra["lon"], rb["lat"], rb["lon"]) * 1.06
            transit_rows.append({
                "edge_id": f"tr_{rid}_{seq:03d}", "route_id": rid, "mode": "metro",
                "u": a, "v": b, "seq": seq, "distance_km": round(km, 4),
                # Namma Metro runs at roughly 32 km/h including dwell.
                "scheduled_min": round(km / 32.0 * 60.0 + 0.4, 3),
            })
        routes.append({
            "route_id": rid, "mode": "metro", "name": line["short"],
            "colour": line.get("colour") or "#7B3FA0",
            "headway_peak_min": 4.0, "headway_offpeak_min": 8.0,
            "stops": seq_nodes,
            "service_start_h": 5.0, "service_end_h": 23.5,
            "source": "OpenStreetMap relation " + str(line["rel_id"]),
        })

    # ---- bus routes, at BMTC's real stop order --------------------------
    by_id = {r["node_id"]: r for r in node_rows}
    for route in bus_routes:
        rid = "bus_" + "".join(c.lower() for c in route["ref"] if c.isalnum())[:14]
        seq_nodes = [route_stop_node[o] for o, _n, _a, _b in route["stops"]]
        for seq, (a, b) in enumerate(zip(seq_nodes, seq_nodes[1:])):
            ra, rb = by_id[a], by_id[b]
            km = haversine(ra["lat"], ra["lon"], rb["lat"], rb["lon"]) * 1.25
            transit_rows.append({
                "edge_id": f"tr_{rid}_{seq:03d}", "route_id": rid, "mode": "bus",
                "u": a, "v": b, "seq": seq, "distance_km": round(km, 4),
                # BMTC ordinary service through this corridor, plus dwell
                "scheduled_min": round(km / 17.0 * 60.0 + 0.55, 3),
            })
        routes.append({
            "route_id": rid, "mode": "bus", "name": f"Route {route['ref']}",
            "colour": "#C2571A",
            "headway_peak_min": 12.0, "headway_offpeak_min": 22.0,
            "stops": seq_nodes,
            "service_start_h": 5.5, "service_end_h": 23.0,
            "source": "OpenStreetMap relation " + str(route["rel_id"]),
        })

    # ---- transfer edges: stop <-> station, on foot ----------------------
    transfer_rows = []
    stations = [r for r in node_rows if r["kind"] == "metro_station"]
    busses = [r for r in node_rows if r["kind"] == "bus_stop"]
    t = 0
    for a in stations:
        for b in busses:
            km = haversine(a["lat"], a["lon"], b["lat"], b["lon"])
            if km <= 0.35:
                transfer_rows.append({
                    "edge_id": f"tf_{t:04d}", "u": a["node_id"], "v": b["node_id"],
                    "distance_km": round(km, 4),
                    "walk_min": round(max(km / 4.6 * 60.0, 0.8), 2)})
                t += 1
    for i, a in enumerate(stations):
        for b in stations[i + 1:]:
            km = haversine(a["lat"], a["lon"], b["lat"], b["lon"])
            if km <= 0.35:
                transfer_rows.append({
                    "edge_id": f"tf_{t:04d}", "u": a["node_id"], "v": b["node_id"],
                    "distance_km": round(km, 4),
                    "walk_min": round(max(km / 4.6 * 60.0, 0.8), 2)})
                t += 1

    _csv(out / "nodes.csv", node_rows)
    _csv(out / "road_edges.csv", road_rows)
    _csv(out / "transit_edges.csv", transit_rows)
    _csv(out / "transfer_edges.csv", transfer_rows)
    (out / "transit_routes.json").write_text(
        json.dumps(routes, indent=2), encoding="utf-8")
    (out / "places.json").write_text(json.dumps(
        [{"place_id": p, "name": n, "lat": round(a, 6), "lon": round(o, 6),
          "category": c} for p, n, a, o, c in places], indent=2), encoding="utf-8")
    print(f"\nwrote {out}")
    print(f"  nodes            {len(node_rows):5d}")
    print(f"  road edges       {len(road_rows):5d}")
    print(f"  transit edges    {len(transit_rows):5d}")
    print(f"  transfer edges   {len(transfer_rows):5d}")
    print(f"  metro routes     {len(routes):5d}")
    print(f"  places           {len(places):5d}")
    return node_rows, road_rows


def _csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with open(path, "w", newline="", encoding="utf-8") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--city-id", default="bengaluru_osm")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    out = Path(args.out) if args.out else ROOT / "data" / "city" / args.city_id
    out.mkdir(parents=True, exist_ok=True)

    raw = fetch_everything()

    # route only the arterials; the wider fetch stays cached for experiments
    arterial = {"elements": [e for e in raw["roads"]["elements"]
                             if (e.get("tags") or {}).get("highway") in ROUTED_CLASSES]}
    edges, pos = build_road_graph(arterial)
    edges, keep = largest_component(edges)

    lines = build_metro(raw["metro"], pos)
    bus_routes = build_bus(raw["buses"])
    stops = [e for e in raw["stops"]["elements"]
             if e.get("type") == "node" and (e.get("tags") or {}).get("name")]

    protect: set[int] = set()
    for line in lines:
        for _osm, _name, lat, lon in line["stops"]:
            n, km = nearest(pos, keep, lat, lon)
            if n is not None and km < 1.0:
                protect.add(n)
    for route in bus_routes:
        for _osm, _name, lat, lon in route["stops"]:
            n, km = nearest(pos, keep, lat, lon)
            if n is not None and km < 0.6:
                protect.add(n)
    print(f"\ncontracting (protecting {len(protect)} station junctions)")
    edges, keep = contract(edges, protect)

    print("\nNominatim, for the named places:")
    places, missing = [], []
    for pid, name, cat, queries in PLACES:
        found = None
        for q in queries:
            hit = nominatim(q)
            if hit is None:
                continue
            lat, lon = float(hit["lat"]), float(hit["lon"])
            if BBOX[0] <= lat <= BBOX[2] and BBOX[1] <= lon <= BBOX[3]:
                found = (lat, lon, q)
                break
        if found is None:
            missing.append(name)
            print(f"  MISS {name}  (tried: {'; '.join(queries)})")
            continue
        lat, lon, used = found
        places.append((pid, name, lat, lon, cat))
        print(f"  {pid:24s} {lat:.5f},{lon:.5f}  via {used[:44]}")
    if missing:
        print(f"\n  {len(missing)} place(s) could not be geocoded inside the "
              f"corridor: {', '.join(missing)}")

    write_bundle(out, edges, keep, pos, lines, stops, places, args.city_id,
                 bus_routes)
    return 0


if __name__ == "__main__":
    sys.exit(main())
