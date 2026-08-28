"""The mobility-provider abstraction.

One interface, every way of getting across a city. A provider answers five
questions about a specific trip at a specific moment:

    get_fare()                     what it advertises
    get_eta()                      how long until it reaches you, and the ride itself
    get_availability()             whether it can serve this trip at all
    get_route()                    the path it would take
    get_cancellation_probability() how likely it is to fall through

The point of the abstraction is the last one. Every consumer mobility app in
existence answers the first two. This project exists because the advertised
fare of an option with a 34% cancellation rate is not its cost, and no app
tells you that.

WHAT IS REAL AND WHAT IS NOT
----------------------------
Every provider declares a `data_class`, and it is carried on every number that
leaves the building:

    REAL       observed from an authoritative source
    PUBLISHED  transcribed from an operator's published table
    SIMULATED  produced by a documented generative model in this repository
    PREDICTED  output of a model in this repository

No adapter in this repository talks to a live commercial ride-hailing API,
because no such API is open. The ride adapters are SIMULATED, they say so in
every response, and the interface is shaped so a real adapter can replace one
without any other code changing.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum


class DataClass(str, Enum):
    """Provenance of a number. Never inferred, always declared."""

    REAL = "real"
    PUBLISHED = "published"
    SIMULATED = "simulated"
    PREDICTED = "predicted"


class ServiceClass(str, Enum):
    """What kind of thing this is, which determines how it can fail.

    A scheduled service does not cancel on you personally -- it runs or it does
    not. A hailed vehicle has a driver who can decline or abandon the trip.
    Self-powered modes cannot fail at all. These three classes have genuinely
    different reliability models and the distinction drives §lifecycle.
    """

    HAILED = "hailed"          # a driver must accept and then turn up
    SCHEDULED = "scheduled"    # runs to a timetable, or does not run
    SELF_POWERED = "self"      # walking, cycling: always available


@dataclass(frozen=True)
class TripContext:
    """Everything a provider needs to know about the trip being priced."""

    origin_lat: float
    origin_lon: float
    dest_lat: float
    dest_lon: float
    departure: datetime
    straight_km: float
    rain: bool = False
    # Per-mode routing already computed by the JourneyMind engine: mode ->
    # (distance_km, in_vehicle_min, fare_amount, fare_low, fare_high).
    # Providers reuse it rather than re-routing, so a quote and a journey can
    # never disagree about the same trip.
    routed: dict[str, "RoutedLeg"] = field(default_factory=dict)
    #: Nearest graph node to the origin, and its NOISY congestion reading. The
    #: latent field that actually drives outcomes in the generator is never
    #: exposed here -- the serving path only ever sees the observed value.
    zone_id: str | None = None
    zone_congestion: float = 0.35

    @property
    def hour(self) -> float:
        return self.departure.hour + self.departure.minute / 60.0

    @property
    def is_weekend(self) -> bool:
        return self.departure.weekday() >= 5

    @property
    def is_peak(self) -> bool:
        return (not self.is_weekend) and (7.5 <= self.hour <= 10.5 or 17.0 <= self.hour <= 20.5)

    @property
    def is_late_night(self) -> bool:
        return self.hour < 5.5 or self.hour >= 23.0


@dataclass(frozen=True)
class RoutedLeg:
    """What the routing engine already worked out for one mode."""

    distance_km: float
    in_vehicle_min: float
    fare_amount: float
    fare_low: float
    fare_high: float
    fare_provenance: str
    transfers: int = 0
    feasible: bool = True
    #: Getting to the platform and away from it at the far end. A metro fare is
    #: 25 rupees and a metro does not reach anybody's front door; quoting the
    #: ticket alone made a two-hour journey look like a 25-rupee one. These are
    #: the hailed legs at each end -- their fare, their minutes, and how many
    #: of them there are, because each is a booking that can fail.
    access_fare_amount: float = 0.0
    access_fare_low: float = 0.0
    access_fare_high: float = 0.0
    access_min: float = 0.0
    access_rides: int = 0
    access_mode: str | None = None


@dataclass(frozen=True)
class Fare:
    amount: float
    low: float
    high: float
    provenance: str
    surge_multiplier: float = 1.0

    @property
    def is_range(self) -> bool:
        return self.high - self.low > 0.5


@dataclass(frozen=True)
class Reliability:
    """How likely this option is to actually happen, and why.

    The three probabilities are conditional and multiply into one attempt's
    success chance. They are kept separate rather than collapsed because they
    have different causes, different fixes, and different owners: no supply is
    a market problem, rejection is a driver-incentive problem, and cancellation
    after acceptance is a behaviour problem.
    """

    p_match: float            # a vehicle is found at all
    p_accept: float           # given matched, the driver accepts
    p_cancel: float           # given accepted, the driver abandons before pickup
    drivers_nearby: float | None = None
    basis: str = ""           # one sentence: where these numbers came from
    data_class: DataClass = DataClass.SIMULATED

    @property
    def p_success_per_attempt(self) -> float:
        return max(0.0, min(1.0, self.p_match * self.p_accept * (1.0 - self.p_cancel)))


@dataclass(frozen=True)
class ProviderQuote:
    """One provider's complete answer for one trip."""

    provider_id: str
    display_name: str          # the vehicle: "Bike taxi", "Auto", "Metro"
    provider_name: str         # who you book it through: "Rapido", "BMTC"
    mode: str
    service_class: ServiceClass
    data_class: DataClass

    fare: Fare
    pickup_min: float          # wait before you are moving
    ride_min: float            # time in the vehicle / on foot
    distance_km: float
    reliability: Reliability
    available: bool
    unavailable_reason: str | None = None
    notes: tuple[str, ...] = ()
    #: Whether this option may WIN, as opposed to merely being priced.
    #: An option can be perfectly available and still not be advice: a cycle
    #: costs nothing, never cancels and beats everything on a six-kilometre
    #: trip, but only if the rider owns a bicycle, and this study area has no
    #: bike-share to hire one from. Pricing it is useful; recommending it
    #: assumes a fact about the rider that nobody checked.
    recommendable: bool = True

    @property
    def door_to_door_min(self) -> float:
        return self.pickup_min + self.ride_min


class MobilityProvider(ABC):
    """A way of getting across the city.

    Subclasses implement the five questions. `quote()` composes them and is
    what callers use; it exists so that the composition order and the
    unavailability rules live in one place rather than in every adapter.
    """

    provider_id: str = "abstract"
    #: What the rider is travelling IN -- the vehicle. Shared by providers that
    #: run the same vehicle, so this is never a brand.
    display_name: str = "Abstract provider"
    #: WHO they book it through. A metered auto and Namma Yatri are one mode and
    #: two providers; keeping these apart is what stops the engine becoming
    #: "recommend Rapido" instead of "recommend a bike taxi".
    provider_name: str = "Unknown operator"
    mode: str = "walk"
    service_class: ServiceClass = ServiceClass.SELF_POWERED
    data_class: DataClass = DataClass.SIMULATED
    #: See ProviderQuote.recommendable. False means "price it, do not advise it".
    recommendable: bool = True

    # -- the five questions -------------------------------------------------
    @abstractmethod
    def get_route(self, ctx: TripContext) -> RoutedLeg | None:
        """The path this provider would take, or None if it cannot serve the trip."""

    @abstractmethod
    def get_fare(self, ctx: TripContext, route: RoutedLeg) -> Fare:
        """What this provider advertises for that route."""

    @abstractmethod
    def get_eta(self, ctx: TripContext, route: RoutedLeg) -> tuple[float, float]:
        """(minutes until you are moving, minutes in motion)."""

    @abstractmethod
    def get_availability(self, ctx: TripContext) -> tuple[bool, str | None]:
        """(can it serve this trip now, why not)."""

    @abstractmethod
    def get_cancellation_probability(self, ctx: TripContext, route: RoutedLeg) -> Reliability:
        """The three conditional failure probabilities, with their basis."""

    # -- composition --------------------------------------------------------
    def quote(self, ctx: TripContext) -> ProviderQuote:
        """Always returns a quote, even when the answer is "not this trip".

        A provider that cannot serve the trip is reported as unavailable with a
        reason, never omitted. A row that silently disappears reads as "we did
        not consider it"; a row that says "no metro route from here" is an
        answer. This is the same rule the journey planner applies to
        over-budget options.
        """
        route = self.get_route(ctx)
        if route is None:
            return ProviderQuote(
                provider_id=self.provider_id, display_name=self.display_name,
                provider_name=self.provider_name,
                mode=self.mode, service_class=self.service_class,
                data_class=self.data_class,
                fare=Fare(amount=0.0, low=0.0, high=0.0, provenance="estimated"),
                pickup_min=0.0, ride_min=0.0, distance_km=0.0,
                reliability=Reliability(
                    p_match=0.0, p_accept=0.0, p_cancel=0.0,
                    basis="not applicable — this mode cannot serve the trip",
                    data_class=self.data_class),
                available=False,
                unavailable_reason=f"no {self.display_name.lower()} route between "
                                   f"these points in the study area",
            )
        available, reason = self.get_availability(ctx)
        fare = self.get_fare(ctx, route)
        pickup_min, ride_min = self.get_eta(ctx, route)
        reliability = self.get_cancellation_probability(ctx, route)
        return ProviderQuote(
            provider_id=self.provider_id,
            display_name=self.display_name,
            provider_name=self.provider_name,
            mode=self.mode,
            service_class=self.service_class,
            data_class=self.data_class,
            fare=fare,
            pickup_min=pickup_min,
            ride_min=ride_min,
            distance_km=route.distance_km,
            reliability=reliability,
            available=available,
            unavailable_reason=reason,
            notes=self.notes(ctx, route),
            recommendable=self.recommendable,
        )

    def notes(self, ctx: TripContext, route: RoutedLeg) -> tuple[str, ...]:
        return ()
