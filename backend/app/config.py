"""Runtime configuration. Everything is environment-overridable; nothing here
is a secret and the application must start with all of it unset."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent.parent
PROJECT_ROOT = BACKEND_DIR.parent


def _bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


def _int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, "") or default)
    except ValueError:
        return default


def _float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, "") or default)
    except ValueError:
        return default


class Settings:
    """Resolved once at import time and cached."""

    def __init__(self) -> None:
        self.app_name = "JourneyMind"
        self.tagline = "A travel advisor that plans your whole trip — not just one ride."
        self.version = "1.0.0"

        # --- data ----------------------------------------------------------
        self.data_dir = Path(os.getenv("JM_DATA_DIR", PROJECT_ROOT / "data")).resolve()
        self.models_dir = Path(os.getenv("JM_MODELS_DIR", PROJECT_ROOT / "models")).resolve()
        self.city_id = os.getenv("JM_CITY", "bengaluru_south")

        # DEMO_MODE means: serve entirely from the bundled static bundle, never
        # reach out to a network data source. It is the default and it is the
        # only mode the deployed MVP supports.
        self.demo_mode = _bool("DEMO_MODE", True)

        # --- model ---------------------------------------------------------
        # graphsage | gat | mlp | gbt | historical | freeflow
        # GAT is the default because it is the better of the two graph
        # encoders and stable enough to serve -- NOT because it is the most
        # accurate model available. On the bundled test split the graph-free
        # MLP measures better; see EVALUATION.md, which says so. Set JM_MODEL
        # to serve any of the six and compare for yourself.
        self.travel_time_model = os.getenv("JM_MODEL", "gat")
        # If the requested model's weights are missing, fall back rather than crash.
        self.model_fallback = os.getenv("JM_MODEL_FALLBACK", "historical")

        # --- routing / optimisation ----------------------------------------
        self.k_candidates = _int("JM_K_CANDIDATES", 20)
        self.max_alternatives = _int("JM_MAX_ALTERNATIVES", 2)
        # Two different caps, because they answer two different questions.
        # `max_ride_leg_km` bounds a FIRST/LAST-MILE hop to a hub: past ~16 km
        # a "short hop to the metro" is not a short hop. `max_direct_ride_km`
        # bounds the DOOR-TO-DOOR ride, which a rider can genuinely book at any
        # length inside the corridor. Using one cap for both silently deleted
        # the direct ride on every trip over 16 km, and the router replaced it
        # with two hailed vehicles in a row -- twice the base fare, twice the
        # pickup wait, and a journey no rider would ever take.
        self.max_ride_leg_km = _float("JM_MAX_RIDE_KM", 16.0)
        self.max_direct_ride_km = _float("JM_MAX_DIRECT_RIDE_KM", 60.0)
        # How far a hailed vehicle will go to put you on a train or a bus.
        # Without this, a ride could only reach a "hub" -- a metro station, a
        # named place, or a stop served by two routes -- and the nearest one to
        # the Wipro campus is 6.7 km away. Every cheap journey therefore began
        # with a half-hour walk, and the planner looked like it could not build
        # one. There are bus stops 500 m from that gate.
        self.access_ride_km = _float("JM_ACCESS_RIDE_KM", 3.0)
        self.access_ride_stops = _int("JM_ACCESS_RIDE_STOPS", 8)

        # --- geocoding -----------------------------------------------------
        # Typing a place that is not one of the bundled fifteen used to be a
        # dead end. Nominatim is free, keyless and ODbL; the lookup is bounded
        # to the study area, cached to disk, and fails to the old behaviour
        # rather than hanging a page load. Set JM_GEOCODER=0 to disable it.
        self.geocoder_enabled = _bool("JM_GEOCODER", True)
        self.geocoder_timeout_s = _float("JM_GEOCODER_TIMEOUT", 4.0)
        self.max_access_walk_km = _float("JM_MAX_ACCESS_WALK_KM", 1.1)
        self.walk_speed_kmph = _float("JM_WALK_SPEED_KMPH", 4.6)

        # --- server --------------------------------------------------------
        self.port = _int("PORT", 8000)
        self.host = os.getenv("HOST", "0.0.0.0")
        # Comma-separated origins. "*" is the default because the API is public,
        # read-only, unauthenticated and carries no cookies or credentials.
        self.cors_origins = [
            o.strip() for o in os.getenv("JM_CORS_ORIGINS", "*").split(",") if o.strip()
        ]
        self.serve_frontend = _bool("JM_SERVE_FRONTEND", True)
        self.static_dir = Path(
            os.getenv("JM_STATIC_DIR", BACKEND_DIR / "app" / "static")
        ).resolve()

        # --- limits (basic abuse hygiene on a public endpoint) --------------
        self.max_budget_inr = _float("JM_MAX_BUDGET", 100000.0)
        self.max_time_min = _float("JM_MAX_TIME_MIN", 1440.0)
        self.rate_limit_per_min = _int("JM_RATE_LIMIT_PER_MIN", 60)

    @property
    def city_dir(self) -> Path:
        return self.data_dir / "city" / self.city_id

    def as_public_dict(self) -> dict:
        """Safe to expose over the API. No paths, no secrets."""
        return {
            "app": self.app_name,
            "version": self.version,
            "city_id": self.city_id,
            "demo_mode": self.demo_mode,
            "travel_time_model": self.travel_time_model,
            "k_candidates": self.k_candidates,
        }


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
