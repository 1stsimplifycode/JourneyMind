"""Generate the simulated booking history.

    python scripts/generate_mobility_data.py --seed 20260828

WHAT THIS IS, SAID PLAINLY
--------------------------
Every booking event produced here is SIMULATED. No commercial ride-hailing
platform publishes cancellation data, and reverse-engineering a private app's
endpoints would breach its terms and would not be OSINT (see SOURCES.md §33).
So the reliability layer is trained on a process this repository writes down,
and the honest claim is:

    "the pipeline works, and the model recovers a structure we planted"

not

    "bike taxis cancel 31% of the time in Bengaluru".

Any figure derived from this bundle is a statement about this generator.

THE GENERATIVE PROCESS
----------------------
Cancellation is not noise. It has causes a driver would recognise, and the
generator encodes the ones that are well documented in the ride-hailing
literature and in any Indian commuter's experience:

  * SHORT TRIPS GET DROPPED. A driver who has queued for twenty minutes does
    not want a 1.2 km fare. This is the single strongest effect and it is why
    a flat per-provider cancellation rate is a bad model.
  * PEAK HOURS ARE WORSE. When demand outruns supply the driver can afford to
    be selective, so acceptance falls and post-acceptance cancellation rises.
  * RAIN IS WORSE STILL, for the same reason, harder.
  * LATE NIGHT IS THIN. Few drivers, long pickups, more abandonment.
  * CONGESTED NEIGHBOURHOODS ARE WORSE. A long, slow pickup through traffic is
    unpaid work, so drivers abandon it. This is drawn from the *same* latent
    congestion field the travel-time graph uses, which is what ties the
    mobility layer to the city graph instead of inventing a second city.
  * PROVIDERS DIFFER. A bike taxi filters through traffic and cancels less on
    pickup, but is pickier about short fares. A cab is the reverse.

The model is only allowed to see a NOISY reading of neighbourhood congestion
(`observed_congestion` on each node), never the latent value that actually
drives the outcome -- the same discipline `generate_dataset.py` uses, and the
reason a model can be beaten by a lookup table here rather than trivially
winning.
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

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "backend"))

OUT_DIR = os.path.join(ROOT, "data", "mobility")

# --------------------------------------------------------------------------
# provider behaviour: the coefficients of the generative process
# --------------------------------------------------------------------------
# Each provider is a set of log-odds offsets. These are ASSUMPTIONS with a
# defensible direction, not measurements, and they are listed here rather than
# buried so that they can be argued with.
PROVIDERS = {
    "bike_taxi": dict(
        label="Bike taxi", mode="rapido",
        base_cancel=-2.30,        # ~9% at the reference trip
        short_trip_sensitivity=1.45,   # dislikes short fares most
        pickup_sensitivity=0.55,
        peak_sensitivity=0.60,
        rain_sensitivity=1.05,         # exposed to weather: worst rain effect
        base_reject=-1.95, base_no_supply=-2.60,
    ),
    "auto": dict(
        label="Auto", mode="auto",
        base_cancel=-2.75,
        short_trip_sensitivity=0.85,
        pickup_sensitivity=0.80,
        peak_sensitivity=0.45,
        rain_sensitivity=0.40,
        base_reject=-2.15, base_no_supply=-2.20,
    ),
    "cab": dict(
        label="Cab", mode="cab",
        base_cancel=-3.05,        # most reliable once accepted
        short_trip_sensitivity=0.55,
        pickup_sensitivity=1.05,       # long pickups hurt a car most
        peak_sensitivity=0.70,
        rain_sensitivity=0.25,
        base_reject=-2.45, base_no_supply=-2.95,
    ),
    "carpool": dict(
        label="Carpool", mode="carpool",
        base_cancel=-2.05,        # depends on a stranger's plans
        short_trip_sensitivity=0.25,
        pickup_sensitivity=0.35,
        peak_sensitivity=-0.35,        # more matches at commute times
        rain_sensitivity=0.15,
        base_reject=-1.20, base_no_supply=-0.95,   # thin market
    ),
}

# --------------------------------------------------------------------------
# enterprise dimensions
# --------------------------------------------------------------------------
# An enterprise deployment does not care about one rider, it cares about a
# population: which campus, which team, which cost centre. These are the filter
# axes of the enterprise dashboard, and they are SIMULATED like everything else
# in this file. Campuses are anchored to real named places in the study area so
# that a campus filter and a map pin agree.
CAMPUSES = [
    ("cmp_sarjapur",  "Sarjapur Road Campus",  "pl_wipro_sarjapur",   0.34),
    ("cmp_electronic", "Electronic City Hub",  "mg_bommanahalli",     0.22),
    ("cmp_mgroad",    "MG Road Office",        "pl_mg_road_shops",    0.18),
    ("cmp_koramangala", "Koramangala Annexe",  "pl_koramangala",      0.16),
    ("cmp_whitefield", "Old Airport Road Site", "pl_whitefield_gate", 0.10),
]
EMPLOYEE_GROUPS = [
    ("Engineering", 0.42), ("Operations", 0.24),
    ("Sales", 0.14), ("Support", 0.13), ("Leadership", 0.07),
]
#: Minutes beyond which a commute counts as an SLA breach for the employer.
SLA_DOOR_TO_DOOR_MIN = 75.0

REFERENCE_KM = 6.0          # a trip of this length gets no short-trip penalty
REFERENCE_PICKUP_KM = 1.2


def peak_intensity(hour: float, dow: int) -> float:
    """0..1 demand pressure. Same shape as the travel-time generator uses."""
    if dow >= 5:
        return 0.45 * math.exp(-((hour - 14.0) ** 2) / (2 * 3.4 ** 2))
    morning = math.exp(-((hour - 9.2) ** 2) / (2 * 1.30 ** 2))
    evening = math.exp(-((hour - 18.6) ** 2) / (2 * 1.65 ** 2))
    return min(1.0, 1.05 * morning + 1.0 * evening)


def short_trip_penalty(distance_km: float) -> float:
    """How unattractive this fare is purely for being short.

    Smoothly 1 at a very short hop, 0 at the reference distance and beyond.
    A driver's reluctance is about the floor fare, not a cliff at 3 km.
    """
    if distance_km >= REFERENCE_KM:
        return 0.0
    return float((1.0 - distance_km / REFERENCE_KM) ** 1.5)


def logistic(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def load_zones(city_id: str = "bengaluru_south"):
    """Zones are the study-area graph's own nodes, so the mobility layer and
    the routing layer describe the same city rather than two cities.

    Read straight from nodes.csv rather than through the serving graph, on
    purpose: `latent_congestion` is ground truth and the serving `Node` object
    deliberately does not carry it. Only the generator is allowed to see the
    latent field; everything downstream sees the noisy reading.
    """
    path = os.path.join(ROOT, "data", "city", city_id, "nodes.csv")
    if not os.path.exists(path):
        raise SystemExit(
            f"{path} not found — run scripts/generate_dataset.py first")
    zones = []
    with open(path, newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            if r["kind"] == "junction":
                continue    # places, stops and stations are where trips start
            zones.append(dict(
                zone_id=r["node_id"], name=r["name"],
                lat=float(r["lat"]), lon=float(r["lon"]), kind=r["kind"],
                latent_congestion=float(r.get("latent_congestion") or 0.0),
                observed_congestion=float(r.get("observed_congestion") or 0.0),
            ))
    return zones


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=20260828)
    ap.add_argument("--bookings", type=int, default=60000)
    ap.add_argument("--weeks", type=int, default=10)
    ap.add_argument("--out", default=OUT_DIR)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    os.makedirs(args.out, exist_ok=True)
    zones = load_zones()
    if not zones:
        raise SystemExit("no zones — run scripts/generate_dataset.py first")

    start = datetime(2026, 6, 1, 0, 0)          # a Monday
    horizon_days = args.weeks * 7
    rain_days = set(rng.choice(horizon_days, size=max(1, horizon_days // 8),
                               replace=False).tolist())

    rows = []
    zone_by_id = {z["zone_id"]: z for z in zones}
    provider_ids = list(PROVIDERS)
    for _ in range(args.bookings):
        day = int(rng.integers(horizon_days))
        # request times follow demand, not a uniform clock
        hour = float(np.clip(rng.normal(13.0, 4.6), 0.0, 23.99))
        if rng.random() < 0.45:
            hour = float(np.clip(rng.choice([9.0, 18.5]) + rng.normal(0, 1.1), 0, 23.99))
        ts = start + timedelta(days=day, hours=hour)
        dow = ts.weekday()
        rain = int(day in rain_days and rng.random() < 0.55)

        # A trip belongs to a campus; the campus biases where it starts, which
        # is what makes "cancellations are worse at Sarjapur Road" a finding
        # rather than noise.
        ci = int(rng.choice(len(CAMPUSES), p=np.array([c[3] for c in CAMPUSES])))
        campus_id, campus_name, anchor_id, _ = CAMPUSES[ci]
        anchor = zone_by_id.get(anchor_id)
        if anchor is not None and rng.random() < 0.62:
            z = anchor
        else:
            z = zones[int(rng.integers(len(zones)))]
        gi = int(rng.choice(len(EMPLOYEE_GROUPS), p=np.array([g[1] for g in EMPLOYEE_GROUPS])))
        employee_group = EMPLOYEE_GROUPS[gi][0]
        pid = provider_ids[int(rng.integers(len(provider_ids)))]
        spec = PROVIDERS[pid]

        distance_km = float(np.clip(rng.lognormal(mean=1.15, sigma=0.72), 0.6, 34.0))
        pickup_km = float(np.clip(rng.gamma(shape=2.1, scale=0.55), 0.05, 7.0))
        pk = peak_intensity(hour, dow)
        short = short_trip_penalty(distance_km)
        late_night = 1.0 if (hour < 5.5 or hour >= 23.0) else 0.0
        congestion = z["latent_congestion"]          # the model never sees this

        # --- supply: is anyone there at all? ------------------------------
        supply_logit = (
            spec["base_no_supply"] + 1.55 * late_night + 0.95 * rain
            + 1.20 * pk + 0.85 * congestion - 0.30 * math.log1p(distance_km)
        )
        p_no_supply = logistic(supply_logit)
        matched = int(rng.random() >= p_no_supply)

        # --- acceptance: will the driver take THIS fare? ------------------
        reject_logit = (
            spec["base_reject"] + spec["short_trip_sensitivity"] * 1.30 * short
            + spec["pickup_sensitivity"] * 0.75 * (pickup_km / REFERENCE_PICKUP_KM - 1.0)
            + spec["peak_sensitivity"] * 0.55 * pk + 0.45 * rain
        )
        p_reject = logistic(reject_logit)
        accepted = int(matched and rng.random() >= p_reject)

        # --- cancellation after acceptance: the expensive failure ---------
        cancel_logit = (
            spec["base_cancel"]
            + spec["short_trip_sensitivity"] * short
            + spec["pickup_sensitivity"] * 0.60 * (pickup_km / REFERENCE_PICKUP_KM - 1.0)
            + spec["peak_sensitivity"] * pk
            + spec["rain_sensitivity"] * rain
            + 1.25 * congestion
            + 0.55 * late_night
            + float(rng.normal(0.0, 0.22))          # irreducible driver variation
        )
        p_cancel = logistic(cancel_logit)
        cancelled = int(accepted and rng.random() < p_cancel)
        completed = int(accepted and not cancelled)

        rows.append(dict(
            booking_id=f"bk_{len(rows):06d}",
            ts=ts.replace(microsecond=0).isoformat(),
            hour=round(hour, 3), dow=dow, is_weekend=int(dow >= 5),
            late_night=int(late_night), rain=rain,
            provider_id=pid, mode=spec["mode"],
            campus_id=campus_id, campus=campus_name,
            employee_group=employee_group,
            cost_centre=f"CC-{campus_id[4:8].upper()}-{employee_group[:3].upper()}",
            zone_id=z["zone_id"], zone_kind=z["kind"],
            zone_congestion_observed=round(z["observed_congestion"], 5),
            distance_km=round(distance_km, 3),
            pickup_km=round(pickup_km, 3),
            peak_intensity=round(pk, 4),
            short_trip_penalty=round(short, 4),
            matched=matched, accepted=accepted,
            cancelled=cancelled, completed=completed,
        ))

    rows.sort(key=lambda r: r["ts"])
    fields = list(rows[0])
    with open(os.path.join(args.out, "bookings.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    with open(os.path.join(args.out, "zones.json"), "w", encoding="utf-8") as fh:
        json.dump(zones, fh, indent=2)

    n = len(rows)
    manifest = dict(
        seed=args.seed, bookings=n, weeks=args.weeks, zones=len(zones),
        generated_from="scripts/generate_mobility_data.py",
        data_class="SIMULATED",
        honesty_note=(
            "Every booking event in this bundle is simulated by a generative "
            "process written down in generate_mobility_data.py. No commercial "
            "ride-hailing platform publishes cancellation data and none was "
            "accessed. Accuracy figures measured here describe this generator, "
            "not any real operator."
        ),
        observed_rates={
            pid: dict(
                bookings=sum(1 for r in rows if r["provider_id"] == pid),
                no_supply_rate=round(1 - np.mean([r["matched"] for r in rows if r["provider_id"] == pid]), 4),
                reject_rate=round(1 - np.mean([r["accepted"] for r in rows if r["provider_id"] == pid and r["matched"]]), 4),
                cancel_rate=round(float(np.mean([r["cancelled"] for r in rows if r["provider_id"] == pid and r["accepted"]])), 4),
                completion_rate=round(float(np.mean([r["completed"] for r in rows if r["provider_id"] == pid])), 4),
            ) for pid in provider_ids
        },
    )
    with open(os.path.join(args.out, "generation_manifest.json"), "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)

    print(f"bookings   : {n}")
    print(f"zones      : {len(zones)}")
    for pid, s in manifest["observed_rates"].items():
        print(f"  {pid:10s} n={s['bookings']:6d}  no-supply {s['no_supply_rate']:.1%}  "
              f"reject {s['reject_rate']:.1%}  cancel {s['cancel_rate']:.1%}  "
              f"completed {s['completion_rate']:.1%}")


if __name__ == "__main__":
    main()
