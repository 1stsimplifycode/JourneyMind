"""Request and response schemas.

Validation lives here rather than in the engine so that a bad request is
rejected with a clear message before any work is done, and so the OpenAPI
document at /docs describes the real contract.
"""

from __future__ import annotations

import re

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator

Preference = Literal["cheapest", "balanced", "fastest"]
Provenance = Literal["exact", "published", "estimated", "predicted", "demo"]


# --------------------------------------------------------------------------
# request
# --------------------------------------------------------------------------
class PointInput(BaseModel):
    """A named place from /api/places, a free-text label, or coordinates.

    All three are accepted because all three are things a person types. The
    label branch matters most: `resolve_point` has always been able to match a
    typed place name, but this validator used to reject label-only points
    before it ever ran, so free text failed at the door with a schema error
    instead of a sentence about the place.
    """

    place_id: str | None = Field(None, max_length=64)
    # A real pasted address is long: "Ericsson Global, A Block, Citrine Block
    # SEZ, Bagmane World Technology Centre, Outer Ring Rd, ... 560048" is 154
    # characters, and at 120 it was rejected by the schema before any of the
    # place logic ran -- the rider got a raw validation error, not an answer.
    label: str | None = Field(None, max_length=300)
    lat: float | None = Field(None, ge=-90, le=90)
    lon: float | None = Field(None, ge=-180, le=180)

    @model_validator(mode="before")
    @classmethod
    def _coordinates_typed_as_text(cls, data):
        """"12.9345, 77.6100" is a coordinate, not the name of a place.

        The docstring promised coordinates and the resolver could use them, but
        a typed pair only ever arrived as `label` -- so it went down the
        place-name branch and came back "Could not find '12.9345, 77.6100' in
        this study area", which is true and useless.
        """
        if not isinstance(data, dict):
            return data
        label = data.get("label")
        if not isinstance(label, str) or data.get("lat") is not None:
            return data
        parts = [p for p in re.split(r"[,\s]+", label.strip()) if p]
        if len(parts) != 2:
            return data
        try:
            lat, lon = float(parts[0]), float(parts[1])
        except ValueError:
            return data
        if not (-90 <= lat <= 90 and -180 <= lon <= 180):
            return data
        return {**data, "lat": lat, "lon": lon, "label": None}

    @model_validator(mode="after")
    def _need_one(self):
        has_label = bool(self.label and self.label.strip())
        if self.place_id is None and not has_label and (self.lat is None or self.lon is None):
            raise ValueError(
                "give a place_id, a place name as `label`, or both lat and lon")
        return self


class ManualWeights(BaseModel):
    cost: float = Field(0.25, ge=0, le=1)
    time: float = Field(0.25, ge=0, le=1)
    transfers: float = Field(0.25, ge=0, le=1)
    comfort: float = Field(0.25, ge=0, le=1)

    @model_validator(mode="after")
    def _not_all_zero(self):
        if self.cost + self.time + self.transfers + self.comfort <= 0:
            raise ValueError("at least one preference weight must be above zero")
        return self


class RecommendRequest(BaseModel):
    origin: PointInput | str
    destination: PointInput | str
    departure_time: datetime | None = Field(
        None,
        description=("ISO 8601. Defaults to now, on the study area's clock. An "
                     "offset-aware timestamp is converted to that clock before "
                     "the travel-time model sees it."))
    budget: float = Field(..., gt=0, le=100000, description="Maximum spend, in rupees")
    max_time: float = Field(..., gt=0, le=1440, description="Maximum journey time, in minutes")
    preference: Preference = "balanced"
    weights: ManualWeights | None = Field(
        None, description="Manual sliders. When present these override the preset.")
    max_transfers: int = Field(3, ge=0, le=6)
    modes: list[str] | None = Field(
        None, description="Restrict ride-hailing modes, e.g. ['bike_taxi'].")
    rain: bool = Field(False, description="Treat conditions as wet in the time context.")

    @field_validator("origin", "destination", mode="before")
    @classmethod
    def _coerce_string(cls, v):
        """Accept a bare string as a place id or a free-text label."""
        if isinstance(v, str):
            return {"place_id": v} if v.startswith("pl_") else {"label": v}
        return v

    @field_validator("modes")
    @classmethod
    def _known_modes(cls, v):
        if v is None:
            return v
        allowed = {"bike_taxi", "auto", "cab"}
        bad = [m for m in v if m not in allowed]
        if bad:
            raise ValueError(
                f"unknown ride mode(s): {', '.join(bad)}. Allowed: {', '.join(sorted(allowed))}")
        return v


Priority = Literal["cheapest", "fastest", "reliable", "balanced"]


class CompareRequest(BaseModel):
    """Compare every way of making one trip.

    Budget and time are OPTIONAL here, unlike the journey planner. Comparing is
    something you do before you know what you can afford, and forcing a budget
    would make the tool refuse the question it exists to answer.
    """

    origin: PointInput | str
    destination: PointInput | str
    departure_time: datetime | None = Field(
        None, description="ISO 8601. Defaults to now, on the study area's clock.")
    priority: Priority = Field(
        "balanced",
        description=("cheapest ranks on EXPECTED cost, not the advertised fare; "
                     "fastest includes time lost to failed bookings; reliable "
                     "maximises the chance of completing without starting over."))
    budget: float | None = Field(None, gt=0, le=100000)
    max_time: float | None = Field(None, gt=0, le=1440)
    rain: bool = False

    @field_validator("origin", "destination", mode="before")
    @classmethod
    def _coerce_string(cls, v):
        if isinstance(v, str):
            return {"place_id": v} if v.startswith("pl_") else {"label": v}
        return v


class BookRequest(BaseModel):
    """Press BOOK NOW on one option."""

    origin: PointInput | str
    destination: PointInput | str
    provider_id: str = Field(..., max_length=40)
    departure_time: datetime | None = None
    priority: Priority = "balanced"
    rain: bool = False
    demo: bool = Field(
        False,
        description=("Fix the random seed so a live demonstration is "
                     "reproducible. Fixes the dice, not the outcome — the "
                     "probabilities remain the model's."))

    @field_validator("origin", "destination", mode="before")
    @classmethod
    def _coerce_string(cls, v):
        if isinstance(v, str):
            return {"place_id": v} if v.startswith("pl_") else {"label": v}
        return v


class NotifyRequest(BaseModel):
    """Tell someone the rider is late. Sent only when the rider asks."""

    meeting: str | None = Field(None, max_length=120)
    meeting_at: datetime | None = None
    manager: str | None = Field(None, max_length=120)


# --------------------------------------------------------------------------
# response
# --------------------------------------------------------------------------
class FareOut(BaseModel):
    amount: float
    low: float
    high: float
    display: str
    provenance: Provenance
    label: str
    note: str
    source: str | None = None
    is_range: bool


class LegOut(BaseModel):
    index: int
    mode: str
    kind: str
    from_name: str
    to_name: str
    distance_km: float
    travel_min: float
    wait_min: float
    total_min: float
    stops: int
    route_name: str | None = None
    route_colour: str | None = None
    fare: FareOut | None = None
    time_provenance: Provenance
    geometry: list[list[float]]


class ConstraintOut(BaseModel):
    feasible: bool
    within_budget: bool
    within_time: bool
    budget_headroom: float
    time_headroom: float
    cost_at_risk: bool
    reasons: list[str]


class JourneyOut(BaseModel):
    journey_id: str
    summary: str
    modes: list[str]
    legs: list[LegOut]
    total_cost: FareOut
    total_min: float
    transfers: int
    distance_km: float
    walk_min: float
    wait_min: float
    reliability: float
    score: float | None = None
    score_breakdown: dict | None = None
    constraints: ConstraintOut


class ExplanationOut(BaseModel):
    headline: str
    reasons: list[str]
    comparisons: list[str]
    caveats: list[str]


class AlternativeOut(BaseModel):
    kind: Literal["feasible", "near_miss"]
    reason: str
    journey: JourneyOut


class FallbackOut(BaseModel):
    label: str
    why: str
    reason: str
    journey: JourneyOut


class ModelInfoOut(BaseModel):
    model: str
    key: str
    family: str
    prediction: str
    status: str
    trained_on: str
    notes: str
    requested: str
    fell_back: bool
    validation_metrics: dict | None = None


class DataNoticeOut(BaseModel):
    demo_mode: bool
    label: str
    city: str
    notes: str
    fare_provenance: dict[str, str]


class ModeComparisonRow(BaseModel):
    """One single-vehicle option, priced for comparison — never a recommendation."""

    mode: str
    journey_id: str
    cost: float
    total_cost: FareOut
    total_min: float
    transfers: int
    feasible: bool
    verdict: str
    beaten_by_recommendation: bool


class RecommendResponse(BaseModel):
    feasible: bool
    message: str | None = None
    origin: dict
    destination: dict
    departure_time: datetime
    computed_at: datetime
    preference: str
    weights: dict
    recommended: JourneyOut | None = None
    explanation: ExplanationOut | None = None
    alternatives: list[AlternativeOut] = []
    fallbacks: list[FallbackOut] = []
    mode_comparison: list[ModeComparisonRow] = []
    model_info: ModelInfoOut
    data_notice: DataNoticeOut
    pipeline: dict


class ErrorResponse(BaseModel):
    error: str
    code: str
    detail: str | None = None
