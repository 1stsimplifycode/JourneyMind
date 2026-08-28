"""Finish the real-data bundle: city record, fares, and travel-time observations.

    python scripts/fetch_osm.py          # stage 1: the real network
    python scripts/build_osm_bundle.py   # stage 2: this file

WHY THERE IS A GENERATOR HERE AT ALL
------------------------------------
Because nobody gives away the numbers. The road network, the stations, the
lines and the place coordinates in `bengaluru_osm` are real OpenStreetMap and
Nominatim data. Per-edge OBSERVED TRAVEL TIMES are not available free for
Bengaluru: the APIs that carry them are commercial and keyed, and scraping one
would breach its terms. So the model's training target is still simulated.

What changed is the ground it is simulated over. The congestion field is no
longer bumps scattered across an invented lattice -- it is anchored to the real
network: a road's class and its distance from the real city centre set its
susceptibility, and the delay an edge suffers depends on the neighbourhood mean
of that field, exactly as in the synthetic bundle. The generator is deliberately
THE SAME ONE (`scripts/generate_dataset.py`), so the two bundles differ in
their geography and in nothing else, and "does the graph help?" stays a fair
question rather than two experiments with different rules.

This is the honest position: **real topology, simulated observations.** Neither
half is dressed up as the other.
"""

from __future__ import annotations

import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import generate_dataset as G           # noqa: E402  the same generator

OUT = ROOT / "data" / "city" / "bengaluru_osm"
SYNTH = ROOT / "data" / "city" / "bengaluru_south"
CENTRE = (12.9767, 77.5713)            # Vidhana Soudha, the real city centre
SEED = 20260828


def read_csv(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def haversine(a_lat, a_lon, b_lat, b_lon) -> float:
    R = 6371.0088
    p1, p2 = math.radians(a_lat), math.radians(b_lat)
    h = (math.sin((p2 - p1) / 2) ** 2
         + math.cos(p1) * math.cos(p2) * math.sin(math.radians(b_lon - a_lon) / 2) ** 2)
    return 2 * R * math.asin(math.sqrt(h))


def congestion_field(nodes: list[dict], roads: list[dict], rng) -> dict[str, float]:
    """Susceptibility per node, anchored to the real network.

    Three real signals, and they are the reason this is not just noise on a
    different lattice:

      * distance from the real city centre -- the core is slower
      * the class of the roads that actually meet here -- an arterial junction
        carries more traffic than a link road
      * how many of them meet -- degree is a real property of the real graph

    A small random component remains, because a latent field the model could
    reconstruct exactly from features it can see would make the whole
    experiment vacuous.
    """
    by_node = defaultdict(list)
    for r in roads:
        by_node[r["u"]].append(r)
        by_node[r["v"]].append(r)

    weight = {"motorway": 1.0, "trunk": 0.9, "primary": 0.8,
              "motorway_link": 0.7, "trunk_link": 0.65, "primary_link": 0.6,
              "connector": 0.3}
    out = {}
    for n in nodes:
        lat, lon = float(n["lat"]), float(n["lon"])
        km = haversine(lat, lon, *CENTRE)
        central = math.exp(-(km / 6.0) ** 2)                 # 0..1, peaks downtown
        touching = by_node.get(n["node_id"], [])
        cls = (max(weight.get(r["road_class"], 0.5) for r in touching)
               if touching else 0.5)
        degree = min(len(touching), 6) / 6.0
        base = 0.18 + 0.45 * central + 0.22 * cls + 0.15 * degree
        out[n["node_id"]] = float(np.clip(base + rng.normal(0, 0.06), 0.05, 0.98))
    return out


def main() -> int:
    if not (OUT / "nodes.csv").exists():
        print("run scripts/fetch_osm.py first", file=sys.stderr)
        return 1

    rng = np.random.default_rng(SEED)
    nodes = read_csv(OUT / "nodes.csv")
    roads = read_csv(OUT / "road_edges.csv")
    transit = read_csv(OUT / "transit_edges.csv")
    transfers = read_csv(OUT / "transfer_edges.csv")

    # ---- congestion, from the real network ------------------------------
    latent = congestion_field(nodes, roads, rng)
    for n in nodes:
        lat_v = latent[n["node_id"]]
        n["latent_congestion"] = round(lat_v, 5)
        # what the app is allowed to SEE is a noisy reading of the latent
        # field, never the field itself -- the same asymmetry the synthetic
        # bundle uses, and what leaves room for a model to lose to a lookup
        n["observed_congestion"] = round(
            float(np.clip(lat_v + rng.normal(0, 0.09), 0.02, 1.0)), 5)
        n["degree"] = sum(1 for r in roads
                          if r["u"] == n["node_id"] or r["v"] == n["node_id"])

    # ---- the SAME observation generator, over the real graph ------------
    adj = defaultdict(set)
    for e in roads + transit + transfers:
        adj[e["u"]].add(e["v"])
        adj[e["v"]].add(e["u"])

    node_rows = [{**n, "lat": float(n["lat"]), "lon": float(n["lon"])} for n in nodes]
    road_rows = [{**r, "distance_km": float(r["distance_km"]),
                  "free_speed_kmph": float(r["free_speed_kmph"]),
                  "lanes": int(r["lanes"])} for r in roads]
    transit_rows = [{**t, "distance_km": float(t["distance_km"]),
                     "scheduled_min": float(t["scheduled_min"])} for t in transit]

    cfg = dict(G.CFG)
    obs = G.generate_observations(
        rng, {n["node_id"]: n for n in node_rows},
        {r["edge_id"]: r for r in road_rows},
        {t["edge_id"]: t for t in transit_rows},
        latent, adj, cfg)
    print(f"generated {len(obs):,} travel-time observations over the real graph")

    G.write_csv(OUT / "travel_times.csv", obs, list(obs[0]))
    G.write_csv(OUT / "nodes.csv", nodes, list(nodes[0]))

    # ---- the city record ------------------------------------------------
    lats = [float(n["lat"]) for n in nodes]
    lons = [float(n["lon"]) for n in nodes]
    city = {
        "city_id": "bengaluru_osm",
        "display_name": "Bengaluru — OpenStreetMap corridor",
        "country": "IN", "currency": "INR", "currency_symbol": "₹",
        "timezone": "Asia/Kolkata",
        "bbox": {"min_lat": min(lats), "min_lon": min(lons),
                 "max_lat": max(lats), "max_lon": max(lons)},
        "centre": {"lat": CENTRE[0], "lon": CENTRE[1]},
        "data_status": "osm",
        "data_status_label": "OpenStreetMap network · estimated times",
        "notes": (
            "Road network, junctions, bus stop positions, metro stations and "
            "line membership come from OpenStreetMap via the Overpass API "
            "(ODbL 1.0). Named place coordinates come from Nominatim (ODbL "
            "1.0). Only motorway, trunk and primary roads are routed, which "
            "makes this an arterial extract rather than the whole street "
            "network. Travel-time observations are SIMULATED over that real "
            "topology — no free source publishes per-edge observed times for "
            "Bengaluru. Fares are the same published tables as the synthetic "
            "bundle. See SOURCES.md."),
        "attribution": [
            "Map data © OpenStreetMap contributors, ODbL 1.0 "
            "(https://www.openstreetmap.org/copyright)",
            "Geocoding © OpenStreetMap contributors via Nominatim, ODbL 1.0",
        ],
    }
    (OUT / "city.json").write_text(json.dumps(city, indent=2, ensure_ascii=False),
                                   encoding="utf-8")

    # fares are transcribed published tables; the same ones apply
    (OUT / "fares.json").write_text(
        (SYNTH / "fares.json").read_text(encoding="utf-8"), encoding="utf-8")

    (OUT / "generation_manifest.json").write_text(json.dumps({
        "seed": SEED,
        "real": {
            "road_network": "OpenStreetMap via Overpass API, ODbL 1.0",
            "routed_classes": "motorway, trunk, primary (+ links)",
            "bus_stop_positions": "OpenStreetMap highway=bus_stop",
            "metro_stations_and_lines": "OpenStreetMap route=subway relations",
            "place_coordinates": "Nominatim",
            "fares": "BMRCL / BMTC / Karnataka RTO tables, transcribed",
        },
        "simulated": {
            "travel_time_observations": (
                f"{len(obs)} rows, generated by scripts/generate_dataset.py "
                f"over the real topology"),
            "congestion_field": (
                "anchored to distance from the real centre, real road class "
                "and real junction degree, plus noise"),
            "ride_hailing_fares_availability_cancellations": (
                "modelled; no operator publishes these"),
        },
        "not_used": {
            "real_time_traffic": "no free source; commercial APIs are keyed",
            "gtfs_realtime": "not consumed",
        },
        "counts": {"nodes": len(nodes), "road_edges": len(roads),
                   "transit_edges": len(transit), "transfer_edges": len(transfers),
                   "observations": len(obs)},
        "honesty_note": (
            "Real topology, simulated observations. The network is genuinely "
            "OpenStreetMap; the travel times over it are not measurements of "
            "anything and must never be presented as such."),
    }, indent=2), encoding="utf-8")

    print(f"wrote {OUT}")
    for f in sorted(OUT.iterdir()):
        print(f"  {f.name:26s} {f.stat().st_size / 1024:8.0f} kB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
