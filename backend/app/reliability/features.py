"""Feature encoding for the reliability models.

Lives here, outside both the training script and the serving path, for the same
reason `graph/features.py` does: the vector the model was fitted on and the
vector it is asked to score must provably be the same one. `FEATURE_NAMES` is
written into the exported checkpoint and asserted on load, so a stale weights
file fails loudly instead of silently scoring the wrong columns.

WHAT THE MODEL IS ALLOWED TO SEE
--------------------------------
Everything here is knowable at the moment a rider asks for a quote -- before
any driver is contacted. That constraint is what makes the prediction useful
rather than a post-hoc description: a feature like "how long the driver took to
respond" would improve every metric and be worthless in production, because you
do not have it when you need to decide.

Neighbourhood congestion enters as the NOISY per-node reading
(`observed_congestion`), never the latent field that actually drives outcomes
in the generator. That is deliberate and it is what leaves room for a model to
lose to a lookup table.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

PROVIDERS = ("bike_taxi", "auto", "cab", "carpool")

FEATURE_NAMES = (
    *[f"provider_{p}" for p in PROVIDERS],
    "short_trip_penalty",
    "log_distance_km",
    "pickup_km_norm",
    "peak_intensity",
    "sin_hour",
    "cos_hour",
    "is_weekend",
    "late_night",
    "rain",
    "zone_congestion",
    "log_distance_x_peak",     # a long trip in peak is a different animal
    "short_trip_x_peak",       # a short fare when the driver has options
)
FEATURE_DIM = len(FEATURE_NAMES)

REFERENCE_KM = 6.0
REFERENCE_PICKUP_KM = 1.2

#: The three heads. Each answers a different question about the same request,
#: and each is trained on the subset of history where that question was asked:
#:   match   over every request
#:   accept  over requests that found a vehicle
#:   cancel  over requests a driver accepted
HEADS = ("match", "accept", "cancel")


def short_trip_penalty(distance_km: float) -> float:
    """How unattractive this fare is purely for being short."""
    if distance_km >= REFERENCE_KM:
        return 0.0
    return float((1.0 - distance_km / REFERENCE_KM) ** 1.5)


def peak_intensity(hour: float, dow: int) -> float:
    """0..1 demand pressure — the same shape the travel-time layer uses."""
    if dow >= 5:
        return 0.45 * math.exp(-((hour - 14.0) ** 2) / (2 * 3.4 ** 2))
    morning = math.exp(-((hour - 9.2) ** 2) / (2 * 1.30 ** 2))
    evening = math.exp(-((hour - 18.6) ** 2) / (2 * 1.65 ** 2))
    return min(1.0, 1.05 * morning + 1.0 * evening)


@dataclass(frozen=True)
class RequestFeatures:
    """One quote request, in the terms the reliability models reason about."""

    provider_id: str
    distance_km: float
    pickup_km: float
    hour: float
    dow: int
    rain: bool = False
    zone_congestion: float = 0.35     # neighbourhood mean if the zone is unknown

    @property
    def is_weekend(self) -> bool:
        return self.dow >= 5

    @property
    def late_night(self) -> bool:
        return self.hour < 5.5 or self.hour >= 23.0

    def vector(self) -> np.ndarray:
        onehot = [1.0 if self.provider_id == p else 0.0 for p in PROVIDERS]
        short = short_trip_penalty(self.distance_km)
        log_d = math.log1p(max(self.distance_km, 0.0))
        pk = peak_intensity(self.hour, self.dow)
        a = 2.0 * math.pi * self.hour / 24.0
        return np.array([
            *onehot,
            short,
            log_d,
            self.pickup_km / REFERENCE_PICKUP_KM - 1.0,
            pk,
            math.sin(a), math.cos(a),
            1.0 if self.is_weekend else 0.0,
            1.0 if self.late_night else 0.0,
            1.0 if self.rain else 0.0,
            float(self.zone_congestion),
            log_d * pk,
            short * pk,
        ], dtype=np.float64)


def encode_rows(rows) -> np.ndarray:
    """Encode a batch of booking-history rows into the design matrix."""
    out = np.zeros((len(rows), FEATURE_DIM), dtype=np.float64)
    for i, r in enumerate(rows):
        out[i] = RequestFeatures(
            provider_id=r["provider_id"],
            distance_km=float(r["distance_km"]),
            pickup_km=float(r["pickup_km"]),
            hour=float(r["hour"]),
            dow=int(r["dow"]),
            rain=bool(int(r["rain"])),
            zone_congestion=float(r["zone_congestion_observed"]),
        ).vector()
    return out


def feature_signature() -> dict:
    return {"dim": FEATURE_DIM, "names": list(FEATURE_NAMES),
            "providers": list(PROVIDERS), "heads": list(HEADS)}
