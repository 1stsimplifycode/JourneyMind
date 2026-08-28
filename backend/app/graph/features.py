"""Feature encoding for the multimodal graph.

Three feature blocks, exactly as described in the project documentation:

  node features   -- what kind of place this is and what it is like
  edge features   -- length, mode, speed, scheduled time, reliability
  time context    -- hour and day cyclically encoded, plus a rain flag

The encoders live here (rather than inside the model) so that the training
script and the serving path provably build identical vectors. `NODE_FEATURE_DIM`
etc. are asserted against the exported model metadata at load time.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime

import numpy as np

# --------------------------------------------------------------------------
# node features
# --------------------------------------------------------------------------
NODE_KINDS = ("metro_station", "bus_stop", "junction", "place")
NODE_FEATURE_NAMES = (
    *[f"kind_{k}" for k in NODE_KINDS],
    "degree_norm",
    "observed_congestion",
    "is_interchange",
    "mean_adjacent_free_speed_norm",
    "lat_norm",
    "lon_norm",
)
NODE_FEATURE_DIM = len(NODE_FEATURE_NAMES)

# --------------------------------------------------------------------------
# edge features
# --------------------------------------------------------------------------
EDGE_CLASSES = ("road", "transit_metro", "transit_bus", "transfer")
EDGE_FEATURE_NAMES = (
    *[f"class_{c}" for c in EDGE_CLASSES],
    "log_distance_km",
    "free_speed_norm",
    "log_base_min",
    "lanes_norm",
    "headway_norm",
    "endpoint_congestion_mean",
)
EDGE_FEATURE_DIM = len(EDGE_FEATURE_NAMES)

# --------------------------------------------------------------------------
# time context
# --------------------------------------------------------------------------
TIME_FEATURE_NAMES = ("sin_hour", "cos_hour", "sin_dow", "cos_dow", "is_weekend", "rain")
TIME_FEATURE_DIM = len(TIME_FEATURE_NAMES)

SPEED_NORM = 50.0  # km/h divisor, keeps speeds around 0.4-0.9
LANES_NORM = 4.0
HEADWAY_NORM = 30.0
DEGREE_NORM = 10.0


@dataclass(frozen=True)
class TimeContext:
    """The 'when' half of a prediction request."""

    hour: float          # 0..24, fractional
    dow: int             # 0 = Monday
    rain: bool = False

    @property
    def is_weekend(self) -> bool:
        return self.dow >= 5

    @classmethod
    def from_datetime(cls, dt: datetime, rain: bool = False) -> "TimeContext":
        return cls(hour=dt.hour + dt.minute / 60.0 + dt.second / 3600.0,
                   dow=dt.weekday(), rain=rain)

    def shifted(self, minutes: float) -> "TimeContext":
        """The same day-of-week clock advanced by N minutes -- used by the
        time-dependent search, where later legs are priced at a later hour."""
        total = self.hour + minutes / 60.0
        day_roll = int(total // 24)
        return TimeContext(hour=total % 24.0,
                           dow=(self.dow + day_roll) % 7,
                           rain=self.rain)

    def vector(self) -> np.ndarray:
        a = 2.0 * math.pi * self.hour / 24.0
        b = 2.0 * math.pi * self.dow / 7.0
        return np.array(
            [math.sin(a), math.cos(a), math.sin(b), math.cos(b),
             1.0 if self.is_weekend else 0.0, 1.0 if self.rain else 0.0],
            dtype=np.float32,
        )

    def bucket(self) -> tuple[int, int, int]:
        """Coarse key for the historical-mean baseline and for caching."""
        return (int(self.hour), 1 if self.is_weekend else 0, 1 if self.rain else 0)


def encode_time(ctx: TimeContext) -> np.ndarray:
    return ctx.vector()


def encode_node(kind: str, degree: int, observed_congestion: float,
                is_interchange: bool, mean_adjacent_free_speed: float,
                lat: float, lon: float, bbox: dict) -> np.ndarray:
    onehot = [1.0 if kind == k else 0.0 for k in NODE_KINDS]
    span_lat = max(bbox["max_lat"] - bbox["min_lat"], 1e-6)
    span_lon = max(bbox["max_lon"] - bbox["min_lon"], 1e-6)
    return np.array(
        [
            *onehot,
            min(degree / DEGREE_NORM, 3.0),
            float(observed_congestion),
            1.0 if is_interchange else 0.0,
            mean_adjacent_free_speed / SPEED_NORM,
            (lat - bbox["min_lat"]) / span_lat,
            (lon - bbox["min_lon"]) / span_lon,
        ],
        dtype=np.float32,
    )


def edge_class_of(kind: str, mode: str) -> str:
    if kind == "transit":
        return "transit_metro" if mode == "metro" else "transit_bus"
    if kind == "transfer":
        return "transfer"
    return "road"


def encode_edge(edge_class: str, distance_km: float, free_speed_kmph: float,
                base_min: float, lanes: int, headway_min: float,
                endpoint_congestion_mean: float) -> np.ndarray:
    onehot = [1.0 if edge_class == c else 0.0 for c in EDGE_CLASSES]
    return np.array(
        [
            *onehot,
            math.log1p(max(distance_km, 0.0)),
            free_speed_kmph / SPEED_NORM,
            math.log1p(max(base_min, 0.0)),
            min(lanes / LANES_NORM, 2.0),
            min(headway_min / HEADWAY_NORM, 2.0),
            float(endpoint_congestion_mean),
        ],
        dtype=np.float32,
    )


def feature_signature() -> dict:
    """Written into the model checkpoint and checked on load, so a stale
    weights file fails loudly instead of predicting nonsense."""
    return {
        "node_dim": NODE_FEATURE_DIM,
        "edge_dim": EDGE_FEATURE_DIM,
        "time_dim": TIME_FEATURE_DIM,
        "node_features": list(NODE_FEATURE_NAMES),
        "edge_features": list(EDGE_FEATURE_NAMES),
        "time_features": list(TIME_FEATURE_NAMES),
    }
