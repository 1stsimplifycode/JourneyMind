"""Build the bundled JourneyMind study-area dataset.

Deterministic and reproducible: `python scripts/generate_dataset.py` always
produces byte-identical output for a given --seed.

WHAT IS REAL AND WHAT IS NOT
----------------------------
Real (public knowledge / OSM-derived):
  * Metro station names, their line assignment and approximate coordinates.
  * Published fare structures (kept separately in fares.json).
Synthetic (generated here):
  * Road junctions and the road graph between them.
  * Bus stop positions, bus route stopping patterns and headways.
  * ALL travel-time observations.

The travel-time generator uses a latent per-node "congestion susceptibility"
field. An edge's true delay depends on the *neighbourhood mean* of that latent,
while node features only expose a NOISY per-node reading of it. That makes
neighbourhood averaging genuinely useful, which is why a graph model can beat
an otherwise identical non-graph model on this data. That advantage is a
property of this synthetic generator, NOT evidence about real cities. Any
GraphSAGE-vs-MLP number measured on this bundle must be reported as such.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from datetime import datetime, timedelta

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _stations import BUS_CORRIDORS, METRO_LINES, METRO_STATIONS, PLACES  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(ROOT, "data", "city", "bengaluru_south")

EARTH_R_KM = 6371.0088


# --------------------------------------------------------------------------
# geometry helpers
# --------------------------------------------------------------------------
def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = p2 - p1
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * EARTH_R_KM * math.asin(math.sqrt(a))


def interpolate(waypoints, spacing_km: float):
    """Walk a polyline and drop a point every `spacing_km`."""
    pts = [waypoints[0]]
    carry = 0.0
    for (la1, lo1), (la2, lo2) in zip(waypoints, waypoints[1:]):
        seg = haversine_km(la1, lo1, la2, lo2)
        if seg <= 1e-9:
            continue
        t = (spacing_km - carry) / seg
        while t <= 1.0:
            pts.append((la1 + (la2 - la1) * t, lo1 + (lo2 - lo1) * t))
            t += spacing_km / seg
        carry = (carry + seg) % spacing_km
    if haversine_km(*pts[-1], *waypoints[-1]) > spacing_km * 0.4:
        pts.append(waypoints[-1])
    return pts


# --------------------------------------------------------------------------
# node construction
# --------------------------------------------------------------------------
def build_nodes(rng, cfg):
    nodes = {}

    def add(node_id, name, kind, lat, lon, **extra):
        nodes[node_id] = dict(
            node_id=node_id, name=name, kind=kind,
            lat=round(lat, 6), lon=round(lon, 6), **extra,
        )

    # 1. metro stations -----------------------------------------------------
    for sid, name, lat, lon, lines in METRO_STATIONS:
        add(sid, name, "metro_station", lat, lon,
            lines="|".join(lines), is_interchange=int(len(lines) > 1))

    # 2. bus stops along each corridor -------------------------------------
    corridor_stops = {}
    for corr in BUS_CORRIDORS:
        pts = interpolate(corr["waypoints"], cfg["bus_stop_spacing_km"])
        seq = []
        for i, (lat, lon) in enumerate(pts):
            sid = f"bs_{corr['route_id'].split('_')[1]}_{i:02d}"
            # snap onto an existing stop if one is already within 250 m
            reuse = None
            for other_id, o in nodes.items():
                if o["kind"] == "bus_stop" and haversine_km(lat, lon, o["lat"], o["lon"]) < 0.25:
                    reuse = other_id
                    break
            if reuse:
                seq.append(reuse)
                continue
            area = corr.get("stop_area") or corr["name"].split(" ")[0]
            add(sid, f"{area} stop {i + 1}", "bus_stop", lat, lon)
            seq.append(sid)
        corridor_stops[corr["route_id"]] = seq

    # 3. road junctions on a jittered grid, kept only where the network is --
    bbox = cfg["bbox"]
    anchors = [(n["lat"], n["lon"]) for n in nodes.values()]
    anchors += [(p[2], p[3]) for p in PLACES]
    step = cfg["junction_grid_km"] / 111.0
    j = 0
    lat = bbox["min_lat"]
    while lat <= bbox["max_lat"]:
        lon = bbox["min_lon"]
        while lon <= bbox["max_lon"]:
            jlat = lat + float(rng.uniform(-0.28, 0.28)) * step
            jlon = lon + float(rng.uniform(-0.28, 0.28)) * step
            near = min(haversine_km(jlat, jlon, a, b) for a, b in anchors)
            if near <= cfg["junction_keep_radius_km"]:
                add(f"jn_{j:03d}", f"Junction {j:03d}", "junction", jlat, jlon)
                j += 1
            lon += step
        lat += step

    # 4. named places -------------------------------------------------------
    for pid, name, lat, lon, cat in PLACES:
        add(f"pl_{pid}", name, "place", lat, lon, category=cat)

    return nodes, corridor_stops


# --------------------------------------------------------------------------
# road network
# --------------------------------------------------------------------------
ROAD_CLASSES = [
    # (name, free_speed_kmph, lanes, weight)
    ("arterial", 42.0, 3, 0.20),
    ("secondary", 32.0, 2, 0.45),
    ("residential", 22.0, 1, 0.35),
]


def build_road_edges(rng, nodes, cfg):
    """k-nearest-neighbour road graph over every non-place node, plus
    connectors that attach places to the network."""
    ids = [n for n, v in nodes.items() if v["kind"] != "place"]
    coords = np.array([[nodes[n]["lat"], nodes[n]["lon"]] for n in ids])

    edges, seen = {}, set()

    def add_edge(u, v):
        if u == v:
            return
        key = tuple(sorted((u, v)))
        if key in seen:
            return
        d = haversine_km(nodes[u]["lat"], nodes[u]["lon"], nodes[v]["lat"], nodes[v]["lon"])
        if d < 1e-4:
            return
        # straight-line -> road distance (detour factor for a real street grid)
        d_road = d * cfg["detour_factor"]
        names = [c[0] for c in ROAD_CLASSES]
        probs = np.array([c[3] for c in ROAD_CLASSES], dtype=float)
        cls = names[int(rng.choice(len(names), p=probs / probs.sum()))]
        spec = next(c for c in ROAD_CLASSES if c[0] == cls)
        seen.add(key)
        eid = f"rd_{len(seen):04d}"
        edges[eid] = dict(
            edge_id=eid, u=key[0], v=key[1],
            distance_km=round(d_road, 4), road_class=cls,
            free_speed_kmph=spec[1], lanes=spec[2],
        )

    k = cfg["road_knn"]
    for i, nid in enumerate(ids):
        d = np.sqrt(((coords - coords[i]) ** 2).sum(axis=1))
        for jx in np.argsort(d)[1:k + 1]:
            other = ids[int(jx)]
            gap = haversine_km(*coords[i], *coords[int(jx)])
            if gap <= cfg["road_max_link_km"]:
                add_edge(nid, other)

    # attach places to their nearest few network nodes
    for pid, v in nodes.items():
        if v["kind"] != "place":
            continue
        d = [(haversine_km(v["lat"], v["lon"], nodes[n]["lat"], nodes[n]["lon"]), n) for n in ids]
        d.sort()
        for _, n in d[:cfg["place_connectors"]]:
            add_edge(pid, n)

    return edges


# --------------------------------------------------------------------------
# transit
# --------------------------------------------------------------------------
def build_transit(nodes, corridor_stops, cfg):
    routes, tedges = [], {}

    def add_leg(route_id, mode, u, v, seq, speed_kmph, dwell_min):
        d = haversine_km(nodes[u]["lat"], nodes[u]["lon"], nodes[v]["lat"], nodes[v]["lon"])
        d *= cfg["transit_detour_factor"] if mode == "bus" else 1.02
        run = d / speed_kmph * 60.0 + dwell_min
        eid = f"tr_{len(tedges):04d}"
        tedges[eid] = dict(
            edge_id=eid, route_id=route_id, mode=mode, u=u, v=v, seq=seq,
            distance_km=round(d, 4), scheduled_min=round(run, 3),
        )

    for line_id, spec in METRO_LINES.items():
        routes.append(dict(
            route_id=f"metro_{line_id}", mode="metro", name=spec["name"],
            colour=spec["colour"], headway_peak_min=cfg["metro_headway_peak"],
            headway_offpeak_min=cfg["metro_headway_offpeak"], stops=spec["stations"],
            service_start_h=cfg["metro_service_start_h"],
            service_end_h=cfg["metro_service_end_h"],
            service_start_weekend_h=cfg["metro_service_start_weekend_h"],
        ))
        for i, (u, v) in enumerate(zip(spec["stations"], spec["stations"][1:])):
            add_leg(f"metro_{line_id}", "metro", u, v, i,
                    cfg["metro_speed_kmph"], cfg["metro_dwell_min"])

    for corr in BUS_CORRIDORS:
        stops = corridor_stops[corr["route_id"]]
        routes.append(dict(
            route_id=corr["route_id"], mode="bus", name=corr["name"],
            colour="#C2571A", headway_peak_min=corr["headway_peak_min"],
            headway_offpeak_min=corr["headway_offpeak_min"], stops=stops,
            service_start_h=cfg["bus_service_start_h"],
            service_end_h=cfg["bus_service_end_h"],
            service_start_weekend_h=cfg["bus_service_start_h"],
        ))
        for i, (u, v) in enumerate(zip(stops, stops[1:])):
            add_leg(corr["route_id"], "bus", u, v, i,
                    cfg["bus_speed_kmph"], cfg["bus_dwell_min"])

    return routes, tedges


def build_transfer_edges(nodes, cfg):
    """Walking links between nearby transit nodes (metro exit -> bus stop)."""
    tids = [n for n, v in nodes.items() if v["kind"] in ("metro_station", "bus_stop")]
    out, seen = {}, set()
    for i, a in enumerate(tids):
        for b in tids[i + 1:]:
            d = haversine_km(nodes[a]["lat"], nodes[a]["lon"], nodes[b]["lat"], nodes[b]["lon"])
            if d > cfg["transfer_max_km"]:
                continue
            key = tuple(sorted((a, b)))
            if key in seen:
                continue
            seen.add(key)
            dw = d * cfg["walk_detour_factor"]
            eid = f"tf_{len(out):04d}"
            out[eid] = dict(
                edge_id=eid, u=key[0], v=key[1], distance_km=round(dw, 4),
                walk_min=round(dw / cfg["walk_speed_kmph"] * 60.0 + cfg["transfer_penalty_min"], 3),
            )
    return out


# --------------------------------------------------------------------------
# latent congestion field + travel-time observations
# --------------------------------------------------------------------------
def latent_field(rng, nodes, cfg):
    """Spatially smooth per-node congestion susceptibility in [0, 1]."""
    ids = sorted(nodes)
    pts = np.array([[nodes[n]["lat"], nodes[n]["lon"]] for n in ids])
    bumps = []
    for _ in range(cfg["latent_bumps"]):
        c = pts[int(rng.integers(len(pts)))]
        bumps.append((c, float(rng.uniform(0.35, 1.0)), float(rng.uniform(0.010, 0.030))))
    raw = np.zeros(len(ids))
    for c, amp, sigma in bumps:
        d2 = ((pts - c) ** 2).sum(axis=1)
        raw += amp * np.exp(-d2 / (2 * sigma ** 2))
    raw = (raw - raw.min()) / max(raw.max() - raw.min(), 1e-9)
    return {n: float(raw[i]) for i, n in enumerate(ids)}


def peak_shape(hour: float, dow: int) -> float:
    """0..1 congestion intensity by hour and day of week."""
    if dow >= 5:  # weekend: one gentle afternoon hump
        return 0.45 * math.exp(-((hour - 14.0) ** 2) / (2 * 3.4 ** 2))
    morning = math.exp(-((hour - 9.2) ** 2) / (2 * 1.30 ** 2))
    evening = math.exp(-((hour - 18.6) ** 2) / (2 * 1.65 ** 2))
    return min(1.0, 1.05 * morning + 1.0 * evening)


def neighbourhood_latent(adj, latent, u, v):
    """Mean latent over the endpoints and their 1-hop neighbours."""
    bag = {u, v} | adj.get(u, set()) | adj.get(v, set())
    return float(np.mean([latent[n] for n in bag]))


def generate_observations(rng, nodes, road_edges, transit_edges, latent, adj, cfg):
    rows = []
    start = datetime(2025, 1, 6, 0, 0)  # a Monday
    horizon_days = cfg["weeks"] * 7
    rain_days = set(rng.choice(horizon_days, size=cfg["rain_days"], replace=False).tolist())

    catalogue = []
    for e in road_edges.values():
        catalogue.append(("road", e["edge_id"], e["u"], e["v"], e["distance_km"],
                          e["distance_km"] / e["free_speed_kmph"] * 60.0, "road"))
    for e in transit_edges.values():
        catalogue.append(("transit", e["edge_id"], e["u"], e["v"], e["distance_km"],
                          e["scheduled_min"], e["mode"]))

    for kind, eid, u, v, dist_km, base_min, mode in catalogue:
        nb = neighbourhood_latent(adj, latent, u, v)
        # metro runs on its own alignment: essentially immune to road congestion
        sensitivity = cfg["sensitivity"][mode if mode in cfg["sensitivity"] else "road"]
        for _ in range(cfg["obs_per_edge"]):
            day = int(rng.integers(horizon_days))
            hour = float(rng.integers(cfg["service_start_h"], cfg["service_end_h"])) + float(rng.random())
            ts = start + timedelta(days=day, hours=hour)
            dow = ts.weekday()
            rain = 1 if day in rain_days and 0.35 < rng.random() else 0
            pk = peak_shape(hour, dow)
            mult = 1.0 + sensitivity * pk * (0.35 + 1.35 * nb) + rain * 0.22 * sensitivity * (0.4 + pk)
            noise = float(np.exp(rng.normal(0.0, cfg["obs_noise_sigma"])))
            observed = max(0.25, base_min * mult * noise)
            rows.append(dict(
                edge_id=eid, edge_kind=kind, mode=mode,
                ts=ts.replace(microsecond=0).isoformat(),
                hour=round(hour, 3), dow=dow, is_weekend=int(dow >= 5), rain=rain,
                base_min=round(base_min, 4), observed_min=round(observed, 4),
            ))
    rows.sort(key=lambda r: (r["ts"], r["edge_id"]))
    return rows


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------
CFG = dict(
    # Widened so the corridor reaches Doddakannelli / Sarjapur Road in the east
    # and the PES University stretch of 100 Feet Ring Road in the west.
    bbox=dict(min_lat=12.8950, min_lon=77.5250, max_lat=13.0060, max_lon=77.6960),
    bus_stop_spacing_km=0.85,
    junction_grid_km=0.85,
    junction_keep_radius_km=0.75,
    road_knn=4,
    road_max_link_km=1.5,
    place_connectors=3,
    detour_factor=1.28,
    walk_detour_factor=1.20,
    walk_speed_kmph=4.6,
    transit_detour_factor=1.18,
    metro_speed_kmph=41.0,
    metro_dwell_min=0.42,
    metro_headway_peak=4.0,
    metro_headway_offpeak=8.0,
    # Local-clock service spans. Approximate, and documented as approximate:
    # they follow published first/last-service practice rather than a timetable.
    metro_service_start_h=5.0,
    metro_service_end_h=23.5,
    metro_service_start_weekend_h=6.0,
    bus_service_start_h=5.5,
    bus_service_end_h=23.0,
    bus_speed_kmph=17.0,
    bus_dwell_min=0.55,
    transfer_max_km=0.85,
    transfer_penalty_min=1.2,
    latent_bumps=9,
    weeks=8,
    rain_days=7,
    obs_per_edge=110,
    obs_noise_sigma=0.115,
    service_start_h=6,
    service_end_h=23,
    sensitivity=dict(road=0.85, bus=0.95, metro=0.06),
)


def write_csv(path, rows, fields):
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=20250827)
    ap.add_argument("--out", default=OUT_DIR)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    os.makedirs(args.out, exist_ok=True)

    nodes, corridor_stops = build_nodes(rng, CFG)
    road_edges = build_road_edges(rng, nodes, CFG)
    routes, transit_edges = build_transit(nodes, corridor_stops, CFG)
    transfer_edges = build_transfer_edges(nodes, CFG)

    # adjacency across every static edge kind, for the latent neighbourhood
    adj: dict[str, set] = {n: set() for n in nodes}
    for coll in (road_edges, transit_edges, transfer_edges):
        for e in coll.values():
            adj[e["u"]].add(e["v"])
            adj[e["v"]].add(e["u"])

    latent = latent_field(rng, nodes, CFG)
    for nid, v in nodes.items():
        v["latent_congestion"] = round(latent[nid], 5)
        # what the model is actually allowed to see: a noisy reading
        v["observed_congestion"] = round(
            float(np.clip(latent[nid] + rng.normal(0, CFG["obs_noise_sigma"] * 2.4), 0.0, 1.0)), 5)
        v["degree"] = len(adj[nid])

    obs = generate_observations(rng, nodes, road_edges, transit_edges, latent, adj, CFG)

    write_csv(os.path.join(args.out, "nodes.csv"), list(nodes.values()),
              ["node_id", "name", "kind", "lat", "lon", "lines", "is_interchange",
               "category", "degree", "observed_congestion", "latent_congestion"])
    write_csv(os.path.join(args.out, "road_edges.csv"), list(road_edges.values()),
              ["edge_id", "u", "v", "distance_km", "road_class", "free_speed_kmph", "lanes"])
    write_csv(os.path.join(args.out, "transit_edges.csv"), list(transit_edges.values()),
              ["edge_id", "route_id", "mode", "u", "v", "seq", "distance_km", "scheduled_min"])
    write_csv(os.path.join(args.out, "transfer_edges.csv"), list(transfer_edges.values()),
              ["edge_id", "u", "v", "distance_km", "walk_min"])
    write_csv(os.path.join(args.out, "travel_times.csv"), obs,
              ["edge_id", "edge_kind", "mode", "ts", "hour", "dow", "is_weekend",
               "rain", "base_min", "observed_min"])

    with open(os.path.join(args.out, "transit_routes.json"), "w", encoding="utf-8") as fh:
        json.dump(routes, fh, indent=2)
    with open(os.path.join(args.out, "places.json"), "w", encoding="utf-8") as fh:
        json.dump([
            dict(place_id=f"pl_{p[0]}", name=p[1], lat=p[2], lon=p[3], category=p[4])
            for p in PLACES
        ], fh, indent=2)
    # city.json is hand-maintained, but its bbox must agree with the one the
    # nodes were generated inside -- so it is rewritten here rather than trusted.
    city_path = os.path.join(args.out, "city.json")
    if os.path.exists(city_path):
        with open(city_path, encoding="utf-8") as fh:
            city = json.load(fh)
        city["bbox"] = dict(CFG["bbox"])
        with open(city_path, "w", encoding="utf-8") as fh:
            json.dump(city, fh, indent=2, ensure_ascii=False)

    with open(os.path.join(args.out, "generation_manifest.json"), "w", encoding="utf-8") as fh:
        json.dump(dict(
            seed=args.seed, config=CFG,
            counts=dict(nodes=len(nodes), road_edges=len(road_edges),
                        transit_edges=len(transit_edges), transfer_edges=len(transfer_edges),
                        routes=len(routes), observations=len(obs)),
            honesty_note=(
                "Road junctions, bus stops, headways and every travel-time observation "
                "in this bundle are synthetic. Metro station names/positions follow the "
                "public Namma Metro network. The latent-congestion design makes "
                "neighbourhood averaging useful by construction, so any GNN-vs-MLP gap "
                "measured here is a statement about this generator, not about real cities."
            ),
        ), fh, indent=2)

    kinds: dict[str, int] = {}
    for v in nodes.values():
        kinds[v["kind"]] = kinds.get(v["kind"], 0) + 1
    print("nodes      :", len(nodes), kinds)
    print("road edges :", len(road_edges))
    print("transit    :", len(transit_edges), "edges over", len(routes), "routes")
    print("transfers  :", len(transfer_edges))
    print("observations:", len(obs))


if __name__ == "__main__":
    main()
