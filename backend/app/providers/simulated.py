"""Provider adapters.

Every adapter here is a SIMULATION and says so on every quote it returns. No
open API publishes live ride-hailing supply, fares or cancellation rates, and
scraping a private app's endpoints would breach its terms and would not be
OSINT. What is real in this file is the *shape*: when a real adapter becomes
available it implements the same five methods and nothing else in the codebase
changes.

WHERE EACH NUMBER COMES FROM
----------------------------
    route      the JourneyMind graph and the travel-time GNN   PREDICTED
    fare       fares.json -- published tables for metro and bus,
               a transparent base + per-km + per-min model for
               hailed modes                                    PUBLISHED / SIMULATED
    pickup     a supply-driven wait model, this file           SIMULATED
    p_match    reliability head, trained on simulated history  PREDICTED
    p_accept   reliability head                                PREDICTED
    p_cancel   reliability head                                PREDICTED

The transit adapters are the honest ones: a metro fare is a published slab and
a metro does not cancel on you, so those quotes carry PUBLISHED provenance and
a degenerate reliability model. That asymmetry is the point -- it is *why* the
comparison finds transit more trustworthy, rather than an assumption fed in.
"""

from __future__ import annotations

import math

from ..reliability.features import RequestFeatures
from ..reliability.model import get_reliability_model
from .base import (
    DataClass, Fare, MobilityProvider, Reliability, RoutedLeg, ServiceClass,
    TripContext,
)

# --------------------------------------------------------------------------
# shared pickup-wait model
# --------------------------------------------------------------------------
#: Base minutes to reach you when supply is healthy. A bike gets through
#: traffic fastest; a carpool has to already be going your way.
BASE_PICKUP_MIN = {
    "bike_taxi": 3.2, "auto": 4.4, "cab": 5.4,
}


def pickup_wait(provider_id: str, ctx: TripContext, p_match: float) -> float:
    """Minutes until the vehicle reaches you.

    Rises as supply thins, because the same scarcity that makes a match
    unlikely also makes the nearest vehicle further away. Deriving the wait
    from `p_match` rather than drawing it independently keeps the two
    consistent: an option cannot be simultaneously hard to match and quick to
    arrive.
    """
    base = BASE_PICKUP_MIN.get(provider_id, 5.0)
    scarcity = 1.0 + 2.2 * (1.0 - min(max(p_match, 0.05), 1.0))
    weather = 1.25 if ctx.rain else 1.0
    night = 1.35 if ctx.is_late_night else 1.0
    return round(base * scarcity * weather * night, 2)


def estimated_pickup_km(provider_id: str, p_match: float) -> float:
    """How far away the vehicle is, implied by the same scarcity signal."""
    base = {"bike_taxi": 0.9, "auto": 1.2, "cab": 1.5}.get(provider_id, 1.2)
    return round(base * (1.0 + 2.0 * (1.0 - min(max(p_match, 0.05), 1.0))), 3)


# --------------------------------------------------------------------------
# hailed vehicles
# --------------------------------------------------------------------------
class HailedProvider(MobilityProvider):
    """A vehicle with a driver who can decline, and can leave after accepting."""

    service_class = ServiceClass.HAILED
    data_class = DataClass.SIMULATED
    #: which reliability class this platform's vehicles belong to. Namma Yatri
    #: and a generic auto are the same vehicle with different economics, so
    #: they share a reliability class and differ only in fare.
    reliability_class = "auto"

    def get_route(self, ctx: TripContext) -> RoutedLeg | None:
        return ctx.routed.get(self.mode)

    def get_fare(self, ctx: TripContext, route: RoutedLeg) -> Fare:
        surge = self.surge_multiplier(ctx)
        return Fare(
            amount=route.fare_amount * surge,
            low=route.fare_low * surge,
            high=route.fare_high * surge,
            provenance=route.fare_provenance,
            surge_multiplier=surge,
        )

    def surge_multiplier(self, ctx: TripContext) -> float:
        """Demand pricing. An assumption with a defensible direction, not a
        measurement -- surge algorithms are proprietary and unpublished."""
        m = 1.0
        if ctx.is_peak:
            m += 0.16
        if ctx.rain:
            m += 0.22
        if ctx.is_late_night:
            m += 0.10
        return round(m, 3)

    def _reliability_raw(self, ctx: TripContext, route: RoutedLeg):
        model = get_reliability_model()
        # p_match is needed to estimate the pickup distance, which is itself an
        # input to p_accept and p_cancel. One cheap fixed-point pass resolves
        # the circularity: predict with a nominal pickup, then re-predict with
        # the implied one. Two passes is enough -- the second move is small.
        provisional = model.predict(RequestFeatures(
            provider_id=self.reliability_class, distance_km=route.distance_km,
            pickup_km=1.2, hour=ctx.hour, dow=ctx.departure.weekday(),
            rain=ctx.rain, zone_congestion=self.zone_congestion(ctx)))
        pickup_km = estimated_pickup_km(self.reliability_class, provisional.p_match)
        return model.predict(RequestFeatures(
            provider_id=self.reliability_class, distance_km=route.distance_km,
            pickup_km=pickup_km, hour=ctx.hour, dow=ctx.departure.weekday(),
            rain=ctx.rain, zone_congestion=self.zone_congestion(ctx)))

    @staticmethod
    def zone_congestion(ctx: TripContext) -> float:
        return 0.35 if ctx.zone_id is None else ctx.zone_congestion

    def get_cancellation_probability(self, ctx: TripContext, route: RoutedLeg) -> Reliability:
        pred = self._reliability_raw(ctx, route)
        return Reliability(
            p_match=pred.p_match, p_accept=pred.p_accept, p_cancel=pred.p_cancel,
            drivers_nearby=None,
            basis=pred.drivers_basis,
            data_class=(DataClass.PREDICTED if pred.source == "model"
                        else DataClass.SIMULATED),
        )

    def get_eta(self, ctx: TripContext, route: RoutedLeg) -> tuple[float, float]:
        pred = self._reliability_raw(ctx, route)
        return pickup_wait(self.reliability_class, ctx, pred.p_match), route.in_vehicle_min

    def get_availability(self, ctx: TripContext) -> tuple[bool, str | None]:
        route = self.get_route(ctx)
        if route is None:
            return False, "no route for this mode in the study area"
        pred = self._reliability_raw(ctx, route)
        if pred.p_match < 0.12:
            return False, "almost no vehicles responding in this area right now"
        return True, None

    def notes(self, ctx: TripContext, route: RoutedLeg) -> tuple[str, ...]:
        out = []
        s = self.surge_multiplier(ctx)
        if s > 1.001:
            reasons = []
            if ctx.is_peak:
                reasons.append("peak demand")
            if ctx.rain:
                reasons.append("rain")
            if ctx.is_late_night:
                reasons.append("late night")
            out.append(f"Fare includes an estimated {(s - 1) * 100:.0f}% surge "
                       f"({', '.join(reasons)}). Surge is modelled, not quoted.")
        return tuple(out)


class RapidoProvider(HailedProvider):
    """A bike taxi, hailed through Rapido."""

    provider_id = "bike_taxi"
    display_name = "Bike taxi"
    provider_name = "Rapido"
    mode = "bike_taxi"
    reliability_class = "bike_taxi"


class AutoProvider(HailedProvider):
    """A metered auto flagged down or hailed through a generic aggregator."""

    provider_id = "auto"
    display_name = "Auto"
    provider_name = "Metered auto"
    mode = "auto"
    reliability_class = "auto"


class NammaYatriProvider(HailedProvider):
    """The SAME auto, hailed through a platform modelled without surge.

    Mode and provider are different things, and this pair is why. The vehicle
    is an auto, the fare table is the same government meter, and the route is
    the same route -- so if the two options are to be distinguishable at all,
    the difference has to be something real about the platform rather than
    about the vehicle. What is modelled is demand pricing: this platform is
    treated as not applying a peak or weather multiplier. That is an ASSUMPTION
    about platform economics, stated here and on every quote, not a measurement
    of anyone's pricing.
    """

    provider_id = "namma_yatri"
    display_name = "Auto"
    provider_name = "Namma Yatri"
    mode = "auto"
    reliability_class = "auto"

    def surge_multiplier(self, ctx: TripContext) -> float:
        return 1.0

    def notes(self, ctx: TripContext, route: RoutedLeg) -> tuple[str, ...]:
        return super().notes(ctx, route) + (
            "Same vehicle and same meter as any other auto. Modelled without "
            "demand pricing, which is an assumption about how this platform "
            "charges rather than an observed rate.",)


class CabProvider(HailedProvider):
    provider_id = "cab"
    display_name = "Cab"
    provider_name = "Cab aggregator"
    mode = "cab"
    reliability_class = "cab"


# --------------------------------------------------------------------------
# scheduled services
# --------------------------------------------------------------------------
class ScheduledProvider(MobilityProvider):
    """Runs to a timetable or does not run. It cannot cancel on you personally.

    This is where the comparison gets its backbone: a published fare and a
    degenerate failure model mean the expected cost equals the advertised cost,
    which is exactly what makes transit the reliable floor an unreliable
    hailed option has to beat.
    """

    service_class = ServiceClass.SCHEDULED
    data_class = DataClass.PUBLISHED

    def get_route(self, ctx: TripContext) -> RoutedLeg | None:
        return ctx.routed.get(self.mode)

    def get_fare(self, ctx: TripContext, route: RoutedLeg) -> Fare:
        return Fare(amount=route.fare_amount, low=route.fare_low,
                    high=route.fare_high, provenance=route.fare_provenance)

    def get_eta(self, ctx: TripContext, route: RoutedLeg) -> tuple[float, float]:
        # Waiting is already inside the routed journey (headway / 2), so the
        # pickup component is zero rather than double-counted.
        return 0.0, route.in_vehicle_min

    def get_availability(self, ctx: TripContext) -> tuple[bool, str | None]:
        route = self.get_route(ctx)
        if route is None:
            return False, "not reachable by this mode in the study area"
        if not route.feasible:
            return False, "not running at this hour"
        return True, None

    def notes(self, ctx: TripContext, route: RoutedLeg) -> tuple[str, ...]:
        if not route.access_rides:
            return ()
        n = route.access_rides
        return (f"Door to door: this includes {n} hailed "
                f"{'leg' if n == 1 else 'legs'} to and from the service, about "
                f"{route.access_min:.0f} min and ₹{route.access_fare_amount:.0f} "
                f"of the total. A station is not a doorstep.",)

    def get_cancellation_probability(self, ctx: TripContext, route: RoutedLeg) -> Reliability:
        """A train does not cancel on you. Getting to it might.

        When the journey needs a hailed first or last mile, the weak link is
        that ride, not the timetable -- and reporting 100% for a metro journey
        that starts with a bike taxi would be exactly the overclaim this
        product exists to argue against. The train's own certainty is combined
        with each access booking's.
        """
        if not route.access_rides or not route.access_mode:
            return Reliability(
                p_match=1.0, p_accept=1.0, p_cancel=0.0,
                basis=("a scheduled service does not cancel on an individual "
                       "rider; service hours are checked separately"),
                data_class=DataClass.PUBLISHED,
            )

        model = get_reliability_model()
        per_km = route.distance_km / max(route.access_rides, 1)
        pred = model.predict(RequestFeatures(
            provider_id=route.access_mode, distance_km=per_km, pickup_km=1.2,
            hour=ctx.hour, dow=ctx.departure.weekday(), rain=ctx.rain,
            zone_congestion=0.35 if ctx.zone_id is None else ctx.zone_congestion))
        n = route.access_rides
        return Reliability(
            p_match=pred.p_match ** n,
            p_accept=pred.p_accept ** n,
            # at least one of the n access rides cancelling
            p_cancel=1.0 - (1.0 - pred.p_cancel) ** n,
            basis=(f"the timetable is certain; the {n} hailed "
                   f"{'leg' if n == 1 else 'legs'} to and from the service are "
                   f"not, and this combines them"),
            data_class=DataClass.PREDICTED,
        )


class MetroProvider(ScheduledProvider):
    provider_id = "metro"
    display_name = "Metro"
    provider_name = "Namma Metro"
    mode = "metro"


class BusProvider(ScheduledProvider):
    provider_id = "bus"
    display_name = "Bus"
    provider_name = "BMTC"
    mode = "bus"


# --------------------------------------------------------------------------
# what is NOT here, and why
# --------------------------------------------------------------------------
# CARPOOL was removed. It is not one of the modes JourneyMind covers, and it
# was doing real damage while it stayed: a thin market at a fraction of a cab
# fare made it the cheapest card on almost every trip, and its 11% completion
# rate then dragged the whole comparison around a mode nobody could actually
# book.
#
# WALKING and CYCLING were removed as user-facing options. Walking still exists
# inside the graph -- you cannot reach a metro platform without covering the
# last fifty metres on foot, and the router needs those edges for connectivity
# -- but it is not a commute this product recommends, and journeys that lean on
# it are rejected (see routing/validate.MAX_JOURNEY_WALK_MIN). A cycle needs a
# bicycle the rider may not own, on a corridor with no shared-cycle service in
# the data.

#: The six modes JourneyMind covers, across five providers. Two of them --
#: a metered auto and Namma Yatri -- are the same vehicle on different
#: platforms, which is the whole point of separating mode from provider.
ALL_PROVIDERS: tuple[MobilityProvider, ...] = (
    RapidoProvider(), AutoProvider(), NammaYatriProvider(), CabProvider(),
    MetroProvider(), BusProvider(),
)


def registry() -> list[dict]:
    """What the API reports about the provider set."""
    return [
        {"provider_id": p.provider_id, "display_name": p.display_name,
         "provider_name": p.provider_name,
         "mode": p.mode, "service_class": p.service_class.value,
         "data_class": p.data_class.value,
         "adapter": "simulated" if p.data_class is DataClass.SIMULATED else "bundled"}
        for p in ALL_PROVIDERS
    ]
