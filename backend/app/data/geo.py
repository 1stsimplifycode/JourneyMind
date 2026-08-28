"""Small geodesy helpers. No dependencies beyond the standard library."""

from __future__ import annotations

import math

EARTH_R_KM = 6371.0088


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in kilometres."""
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = p2 - p1
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * EARTH_R_KM * math.asin(math.sqrt(a))


def bbox_contains(bbox: dict, lat: float, lon: float, pad_deg: float = 0.0) -> bool:
    return (
        bbox["min_lat"] - pad_deg <= lat <= bbox["max_lat"] + pad_deg
        and bbox["min_lon"] - pad_deg <= lon <= bbox["max_lon"] + pad_deg
    )


# Straight-line distance under-states how far you actually walk or drive.
# These multipliers convert crow-flies to on-network distance.
WALK_DETOUR = 1.20
ROAD_DETOUR = 1.28
