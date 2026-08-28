"""The comparison service: what will this trip actually cost?

    request
      -> JourneyMind engine        real routes, GNN travel times, real fares
      -> provider adapters         one quote per way of getting there
      -> reliability heads         P(match), P(accept), P(cancel)
      -> lifecycle solver          expected cost, expected time, distribution
      -> constraint filter         drop what breaks the rider's limits
      -> ranking                   by the rider's stated priority
      -> explanation               why this one, and why not the cheaper one

THE ONE IDEA
------------
Every stage before the lifecycle solver exists in every fare-comparison app.
The solver is the product: it turns an advertised fare into an expected cost by
pricing the failure modes, and it is why the recommendation can differ from the
cheapest row on the screen and still be right.

RANKING
-------
Four priorities, and they rank on genuinely different quantities rather than on
re-weightings of one:

    cheapest      lowest expected cost      (not lowest advertised fare)
    fastest       lowest expected time      (including time lost to retries)
    reliable      highest P(success)        then expected cost
    balanced      expected cost, with a reliability floor

"Cheapest" ranking on expected rather than advertised cost is the whole thesis
in one line of code.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime

from ..data.geo import haversine_km
from ..lifecycle.expected_cost import ExpectedCost, LifecycleParams, solve
from ..providers.base import (
    DataClass, MobilityProvider, ProviderQuote, RoutedLeg, ServiceClass, TripContext,
)
from ..providers.simulated import ALL_PROVIDERS
from .engine import JourneyRequest, RoutingError, get_engine, single_vehicle_mode

log = logging.getLogger("journeymind.compare")

PRIORITIES = ("cheapest", "fastest", "reliable", "balanced")
DEFAULT_PRIORITY = "balanced"

#: A balanced recommendation will not hand you an option that fails more than
#: this often, however cheap it is. Expressed as a rule rather than folded into
#: a weight so that it can be stated to the rider and argued with.
BALANCED_RELIABILITY_FLOOR = 0.80


@dataclass
class Option:
    """One provider's quote, priced through the lifecycle."""

    quote: ProviderQuote
    expected: ExpectedCost
    within_budget: bool = True
    within_time: bool = True
    #: Fits the limit on the trip itself, but not once failure is priced in.
    budget_at_risk: bool = False
    time_at_risk: bool = False
    excluded_reason: str | None = None
    rank: int | None = None
    reasons: list[str] = field(default_factory=list)

    @property
    def feasible(self) -> bool:
        """Can this option be RECOMMENDED -- not merely priced."""
        return (self.quote.available and self.quote.recommendable
                and self.within_budget and self.within_time
                and self.excluded_reason is None)

    @property
    def p_success(self) -> float:
        return self.expected.p_success


@dataclass
class Comparison:
    origin_label: str
    dest_label: str
    departure: datetime
    priority: str
    budget: float | None
    max_time_min: float | None
    options: list[Option]
    recommended: Option | None
    headline: str
    reasoning: list[str]
    trace: dict
    #: Multimodal journeys from the routing engine, so the booking screen can
    #: offer "walk -> metro -> walk" beside the single-ride cards.
    journeys: list[dict] = field(default_factory=list)
    #: What the rider actually typed, when the resolver matched it to
    #: something else. Defaulted, so it has to live after the required fields.
    origin_typed: str | None = None
    dest_typed: str | None = None


# --------------------------------------------------------------------------
def _describe_journey(j, symbol: str = "₹", budget: float | None = None,
                     max_time_min: float | None = None) -> dict:
    """One planner journey, in the terms the booking screen needs."""
    interchanges = set(j.interchange_indices())
    legs = [{"mode": lg.mode,
             # where you get ON, not where the absorbed walk began
             "from": lg.board_name or lg.from_name,
             "to": lg.alight_name or lg.to_name,
             "minutes": round(lg.total_min, 1),
             # the approach on foot, named as what it is
             "access_min": round(lg.access_min, 1),
             "route": lg.route_name,
             # True when this leg continues the previous mode on a different
             # service -- a line change, not a change of transport.
             "interchange": i in interchanges,
             "distance_km": round(lg.distance_km, 2)} for i, lg in enumerate(j.legs)]
    return {
        "journey_id": j.journey_id,
        "summary": j.mode_summary(),
        "modes": list(j.modes),
        # Consecutive legs of one mode are one step here: an interchange
        # between two metro lines is a metro trip, not "Metro then Metro".
        "shape": " → ".join(j.shape()),
        # the sequence, with repeats. `modes` is deduplicated and drops the
        # bike taxi at the far end, so "Bike taxi then Bus then Metro" was one
        # leg short of the journey it described.
        "shape_modes": j.shape(),
        "warnings": list(getattr(j, "warnings", []) or []),
        "fare": round(j.cost, 2),
        "fare_display": j.total_cost.display(symbol),
        "total_min": round(j.total_min, 1),
        "transfers": j.transfers,
        "walk_min": round(j.walk_min, 1),
        "distance_km": round(j.distance_km, 2),
        # The planner runs unconstrained so that the ride cards can price
        # options the rider cannot afford -- "you cannot afford it" is
        # information. The journeys still have to be measured against the
        # limits the rider actually stated, or the booking screen would offer
        # a ₹300 itinerary to somebody who said ₹250.
        "within_budget": budget is None or j.cost <= budget + 1e-9,
        "within_time": max_time_min is None or j.total_min <= max_time_min + 1e-9,
        "legs": legs,
    }


def offerable(j: dict) -> bool:
    """Is this planner journey something to put in front of the rider?

    One rule, applied once, so the headline and the list below it cannot
    disagree. They did: "Nothing fits ₹200" sat directly above a ₹167 journey
    that fitted ₹200 and the time limit, because the sentence was written from
    the provider cards and the list was filtered somewhere else entirely.

      * it must mix modes -- a single-vehicle trip is already a ride card, and
        repeating it as an itinerary is noise
      * it must respect the limits the rider actually stated
    """
    vehicles = {m for m in j.get("modes", []) if m != "walk"}
    return (len(vehicles) >= 2
            and j.get("within_budget", True) and j.get("within_time", True))


def _routed_from_engine(engine, req: JourneyRequest) -> tuple[dict[str, RoutedLeg], dict]:
    """Run the real engine once and harvest one single-mode leg per mode.

    Reuses JourneyMind's graph, GNN travel times and fare models rather than
    re-deriving them, so a comparison card and a planned journey can never
    disagree about the same trip.
    """
    rec = engine.recommend(req)
    routed: dict[str, RoutedLeg] = {}
    pool = [rec.recommended] if rec.recommended else []
    pool += [a["journey"] for a in rec.alternatives]
    pool += [f["journey"] for f in rec.fallbacks]

    for row in rec.mode_comparison:
        mode = row["mode"]
        fare = row["total_cost"]
        acc = row.get("access") or {}
        routed[mode] = RoutedLeg(
            distance_km=row.get("distance_km") or 0.0,
            in_vehicle_min=row["total_min"],
            fare_amount=fare.amount, fare_low=fare.low, fare_high=fare.high,
            fare_provenance=fare.provenance,
            transfers=row["transfers"], feasible=True,
            access_fare_amount=acc.get("fare", 0.0),
            access_fare_low=acc.get("fare_low", 0.0),
            access_fare_high=acc.get("fare_high", 0.0),
            access_min=acc.get("minutes", 0.0),
            access_rides=acc.get("rides", 0),
            access_mode=acc.get("mode"),
        )
    # Walking is not one of the app-by-app comparison modes, but it is always
    # physically possible and the self-powered providers need it as their
    # floor. Taken from the engine's walk-only reference rather than from the
    # presentation pool: the pool contains a walk-only journey only when one
    # happened to be recommended, which made "Walk" report "no walk route
    # between these points" on a two-kilometre trip.
    walk_ref = getattr(rec, "walk_reference", None)
    if walk_ref is not None:
        routed["walk"] = RoutedLeg(
            distance_km=walk_ref.distance_km, in_vehicle_min=walk_ref.total_min,
            fare_amount=0.0, fare_low=0.0, fare_high=0.0,
            fare_provenance="exact")
    # Itineraries for the booking screen, drawn from every validated candidate
    # rather than from the two alternatives that happened to be ranked. A mode
    # already represented by its own card is skipped: the Metro card IS the
    # bike-taxi-metro-bike-taxi journey now that cards are priced door to door,
    # and printing it twice is not two options.
    carded = {row["journey_id"] for row in rec.mode_comparison}
    journeys, seen = [], set()
    for j in sorted(rec.candidates, key=lambda x: (x.cost, x.total_min)):
        if j.journey_id in carded or j.signature in seen:
            continue
        seen.add(j.signature)
        journeys.append(j)
    return routed, rec.pipeline, journeys


def _nearest_zone(engine, lat: float, lon: float) -> tuple[str | None, float]:
    """Nearest graph node and its observed congestion — the model never sees
    the latent field that drives outcomes in the generator."""
    near = engine.graph.nearest_nodes(lat, lon, max_km=3.0, limit=1)
    if not near:
        return None, 0.35
    nid = near[0][0]
    return nid, float(engine.graph.nodes[nid].observed_congestion)


def _fallback_for(options: list[Option],
                  exclude: str | None = None) -> tuple[str, float, float] | None:
    """What you actually do when every attempt at an option fails.

    A SCHEDULED service -- a metro or a bus. It runs whether or not a driver
    feels like it, so it is the thing that is still there after four failed
    bookings, and pricing the failure mass against it is what stops an
    unreliable option from looking free.

    `exclude` is the option being priced, and it matters: without it the
    cheapest scheduled service was handed itself as its own fallback, so the
    cost of the bus failing was the cost of taking the bus. A bus that fails
    72% of the time came out barely dearer than its own fare, and the crossover
    this product exists to show disappeared entirely.

    Deliberately NOT self-powered. Walking is always available and costs
    nothing, so admitting it here made every failure free: a ₹29 carpool came
    out at ₹7 expected, because seven times in ten the model had the rider walk
    for half an hour instead and charged them nothing for it. Somebody willing
    to walk would have walked; the fallback has to be a service you would
    actually switch to. With no scheduled service on the trip there is no
    substitute, and the solver prices abandonment as the loss it is.
    """
    reliable = [o for o in options
                if o.quote.available and o.quote.recommendable
                and o.quote.service_class is ServiceClass.SCHEDULED
                and o.quote.provider_id != exclude]
    if not reliable:
        return None
    # The most likely to actually happen, cheapest among equals. Not simply the
    # cheapest: since a scheduled journey now includes its hailed first and last
    # mile, the cheapest timetabled option is no longer automatically the
    # dependable one, and falling back onto something that fails a third of the
    # time is not a floor.
    best = max(reliable, key=lambda o: (round(o.p_success, 2),
                                        -o.quote.fare.amount))
    return best.quote.display_name, best.quote.fare.amount, best.quote.door_to_door_min


def compare(*, origin_lat: float, origin_lon: float, origin_label: str,
            dest_lat: float, dest_lon: float, dest_label: str,
            departure: datetime, priority: str = DEFAULT_PRIORITY,
            budget: float | None = None, max_time_min: float | None = None,
            rain: bool = False, providers: tuple[MobilityProvider, ...] = ALL_PROVIDERS,
            params: LifecycleParams | None = None) -> Comparison:
    t0 = time.perf_counter()
    engine = get_engine()
    priority = priority if priority in PRIORITIES else DEFAULT_PRIORITY

    straight = haversine_km(origin_lat, origin_lon, dest_lat, dest_lon)
    if straight < 0.05:
        raise RoutingError("Your start and destination are the same place.",
                           code="same_endpoints")

    # 1. route once, through the real engine ------------------------------
    engine_req = JourneyRequest(
        origin_lat=origin_lat, origin_lon=origin_lon, origin_label=origin_label,
        dest_lat=dest_lat, dest_lon=dest_lon, dest_label=dest_label,
        departure=departure,
        budget=budget if budget is not None else 100000.0,
        max_time_min=max_time_min if max_time_min is not None else 1440.0,
        preference="balanced", rain=rain)
    routed, engine_trace, planner_journeys = _routed_from_engine(engine, engine_req)

    zone_id, zone_congestion = _nearest_zone(engine, origin_lat, origin_lon)
    ctx = TripContext(
        origin_lat=origin_lat, origin_lon=origin_lon,
        dest_lat=dest_lat, dest_lon=dest_lon, departure=departure,
        straight_km=straight, rain=rain, routed=routed,
        zone_id=zone_id, zone_congestion=zone_congestion)

    # 2. one quote per provider -------------------------------------------
    # Every provider answers, including "not this trip" — see MobilityProvider.quote.
    quotes = [p.quote(ctx) for p in providers]

    # 3. price each through the lifecycle ----------------------------------
    #    Two passes: the fallback has to be chosen from the quotes themselves,
    #    so priced-with-no-fallback comes first, then everything is repriced
    #    against the reliable floor that pass found.
    provisional = [
        Option(quote=q, expected=solve(
            displayed_fare=q.fare.amount, p_match=q.reliability.p_match,
            p_accept=q.reliability.p_accept, p_cancel=q.reliability.p_cancel,
            pickup_min=q.pickup_min, ride_min=q.ride_min, params=params))
        for q in quotes
    ]
    fb = _fallback_for(provisional)

    options: list[Option] = []
    for q in quotes:
        # What THIS rider falls back on, which is never the option that just
        # failed them.
        own_fb = _fallback_for(provisional, exclude=q.provider_id)
        ec = solve(
            displayed_fare=q.fare.amount,
            p_match=q.reliability.p_match, p_accept=q.reliability.p_accept,
            p_cancel=q.reliability.p_cancel,
            pickup_min=q.pickup_min, ride_min=q.ride_min,
            fallback_label=own_fb[0] if own_fb else None,
            fallback_cost=own_fb[1] if own_fb else None,
            fallback_min=own_fb[2] if own_fb else None,
            params=params)
        opt = Option(quote=q, expected=ec)

        # HARD constraint: what the rider is actually charged, and how long the
        # trip itself takes. Expected cost is an average over outcomes and can
        # sit BELOW the fare precisely because the option often fails -- gating
        # on it admitted a ₹300 ride against a ₹250 budget on the grounds that
        # you probably would not get it. That is not what a budget means.
        #
        # SOFT signal: the same limits measured against the expectation, which
        # is where retries and rebooking show up. Reported, never used to
        # exclude -- a rider is entitled to accept that risk.
        if budget is not None:
            opt.within_budget = q.fare.amount <= budget + 1e-9
            opt.budget_at_risk = opt.within_budget and (
                ec.expected_cost > budget + 1e-9 or q.fare.high > budget + 1e-9)
        if max_time_min is not None:
            opt.within_time = q.door_to_door_min <= max_time_min + 1e-9
            opt.time_at_risk = opt.within_time and ec.expected_minutes > max_time_min + 1e-9
        if not q.available:
            opt.excluded_reason = q.unavailable_reason
        options.append(opt)

    # 4. rank ---------------------------------------------------------------
    feasible = [o for o in options if o.feasible]
    ranked = _rank(feasible, priority)
    for i, o in enumerate(ranked):
        o.rank = i + 1
    best = ranked[0] if ranked else None

    described = [_describe_journey(j, engine.city.currency_symbol, budget, max_time_min)
                 for j in planner_journeys]
    offered = [j for j in described if offerable(j)]
    headline, reasoning = _explain(best, ranked, options, priority, budget,
                                   max_time_min, offered)

    options.sort(key=lambda o: (o.rank is None, o.rank if o.rank else 0,
                                o.expected.expected_cost))
    trace = {
        "straight_line_km": round(straight, 3),
        "providers_queried": len(providers),
        "quotes_returned": len(quotes),
        "unavailable": [q.provider_id for q in quotes if not q.available],
        "feasible": len(feasible),
        "priority": priority,
        "zone_id": zone_id,
        "fallback": {"label": fb[0], "cost": round(fb[1], 2)} if fb else None,
        "engine": {k: engine_trace.get(k) for k in ("graph", "prediction", "candidates")},
        "elapsed_ms": round((time.perf_counter() - t0) * 1000, 1),
        "note": ("Expected cost is computed over the booking lifecycle, not read "
                 "off the fare. See backend/app/lifecycle/expected_cost.py."),
    }
    symbol = engine.city.currency_symbol
    return Comparison(
        origin_label=origin_label, dest_label=dest_label, departure=departure,
        priority=priority, budget=budget, max_time_min=max_time_min,
        options=options, recommended=best, headline=headline,
        reasoning=reasoning, trace=trace,
        journeys=offered)


def _rank(options: list[Option], priority: str) -> list[Option]:
    if priority == "cheapest":
        # expected, not advertised. The entire point.
        return sorted(options, key=lambda o: (o.expected.expected_cost,
                                              o.expected.expected_minutes))
    if priority == "fastest":
        return sorted(options, key=lambda o: (o.expected.expected_minutes,
                                              o.expected.expected_cost))
    if priority == "reliable":
        return sorted(options, key=lambda o: (-o.p_success,
                                              o.expected.expected_cost))
    # balanced: money and time both matter, so normalise each across the
    # options actually on offer and weigh them evenly -- the same min-max
    # normalisation the journey optimiser uses, for the same reason (rupees and
    # minutes must never be added together raw).
    #
    # The reliability floor is a gate, not a term. An option that fails one time
    # in five is not "slightly worse", it is a different kind of thing, and
    # averaging cannot express that.
    solid = [o for o in options if o.p_success >= BALANCED_RELIABILITY_FLOOR]
    rest = [o for o in options if o.p_success < BALANCED_RELIABILITY_FLOOR]

    def blended(group: list[Option]) -> list[Option]:
        if len(group) < 2:
            return list(group)
        costs = [o.expected.expected_cost for o in group]
        times = [o.expected.expected_minutes for o in group]
        c_lo, c_hi = min(costs), max(costs)
        t_lo, t_hi = min(times), max(times)
        c_span = max(c_hi - c_lo, 1e-6)
        t_span = max(t_hi - t_lo, 1e-6)
        return sorted(group, key=lambda o: (
            0.5 * (o.expected.expected_cost - c_lo) / c_span
            + 0.5 * (o.expected.expected_minutes - t_lo) / t_span))

    return blended(solid) + sorted(rest, key=lambda o: (-o.p_success,
                                                        o.expected.expected_cost))


def _money(x: float) -> str:
    return f"₹{x:,.0f}"


def _explain(best: Option | None, ranked: list[Option], all_options: list[Option],
             priority: str, budget: float | None,
             max_time_min: float | None,
             journeys: list[dict] | None = None) -> tuple[str, list[str]]:
    """Say why, in the terms the rider asked in."""
    journeys = journeys or []
    if best is None:
        blocked = [o for o in all_options if not o.quote.available]
        reasons = [f"{o.quote.display_name}: {o.quote.unavailable_reason}"
                   for o in blocked if o.quote.unavailable_reason]

        # No single ride fits, but travelling in stages might -- and saying
        # "nothing fits" above a journey that does is the product disowning its
        # own best answer.
        if journeys:
            j = min(journeys, key=lambda x: x["fare"])
            limits = _money(budget) if budget is not None else "your limits"
            if budget is not None and max_time_min is not None:
                limits = f"{_money(budget)} in {max_time_min:.0f} minutes"
            elif max_time_min is not None:
                limits = f"{max_time_min:.0f} minutes"
            return (f"No single ride fits {limits}, but travelling in stages "
                    f"does: {j['shape'].replace(' → ', ' then ')}, "
                    f"{j['fare_display']} in {j['total_min']:.0f} minutes."), reasons

        msg = "No option fits."
        if budget is not None:
            over = [o for o in all_options if o.quote.available and not o.within_budget]
            if over:
                cheapest = min(over, key=lambda o: o.expected.expected_cost)
                msg = (f"Nothing fits {_money(budget)}. The cheapest option once "
                       f"cancellation risk is priced in is "
                       f"{cheapest.quote.display_name} at "
                       f"{_money(cheapest.expected.expected_cost)}.")
        return msg, reasons

    q, ec = best.quote, best.expected
    reasons = [
        f"{_money(q.fare.amount)} advertised, {_money(ec.expected_cost)} expected once "
        f"cancellation and rebooking are priced in.",
        f"{ec.expected_minutes:.0f} min door to door, including "
        f"{ec.expected_wasted_min:.0f} min typically lost to failed requests.",
        f"{ec.p_success:.0%} of riders complete this without having to start over.",
    ]
    if q.service_class is ServiceClass.HAILED:
        r = q.reliability
        reasons.append(
            f"Failure breakdown: {1 - r.p_match:.0%} no vehicle, "
            f"{1 - r.p_accept:.0%} driver declines, "
            f"{r.p_cancel:.0%} cancels after accepting.")

    if ec.is_blended:
        reasons.append(
            f"Note: {ec.p_abandon:.0%} of the time every attempt fails and you end "
            f"up taking {ec.fallback_label or 'something else'} instead, so the "
            f"expected cost blends two different journeys.")

    # ...and the mirror of it: what does this recommendation COST you in time?
    # "Best value: Bus" is a defensible answer when a cab is ten times the price
    # for twice the speed -- but stating it without the three hours attached is
    # the product hiding its own trade-off. The rider gets to disagree only if
    # they are told.
    faster = [o for o in ranked
              if o is not best
              and o.expected.expected_minutes < ec.expected_minutes - 15]
    if faster:
        quickest = min(faster, key=lambda o: o.expected.expected_minutes)
        saved = ec.expected_minutes - quickest.expected.expected_minutes
        extra = quickest.expected.expected_cost - ec.expected_cost
        if extra > 0.5:
            reasons.append(
                f"{quickest.quote.display_name} would get you there about "
                f"{saved:.0f} minutes sooner for {_money(extra)} more — "
                f"{_money(quickest.expected.expected_cost)} against "
                f"{_money(ec.expected_cost)} expected. Worth it or not is your "
                f"call, not the model's.")

    # The sentence the product exists to produce: why not the cheaper row?
    cheaper_advertised = [o for o in all_options
                          if o is not best and o.quote.available
                          and o.quote.fare.amount < q.fare.amount - 0.5]
    if cheaper_advertised:
        rival = min(cheaper_advertised, key=lambda o: o.quote.fare.amount)
        if rival.expected.expected_cost > ec.expected_cost:
            reasons.append(
                f"{rival.quote.display_name} advertises less "
                f"({_money(rival.quote.fare.amount)}) but is expected to cost "
                f"{_money(rival.expected.expected_cost)} — its "
                f"{rival.quote.reliability.p_cancel:.0%} cancellation risk and "
                f"{1 - rival.expected.p_success:.0%} chance of giving up entirely "
                f"outweigh the lower sticker price.")
        elif not rival.feasible:
            reasons.append(
                f"{rival.quote.display_name} is cheaper but "
                f"{rival.excluded_reason or 'breaks one of your limits'}.")

    label = {"cheapest": "Lowest expected cost", "fastest": "Fastest overall",
             "reliable": "Most reliable", "balanced": "Best value"}[priority]
    headline = f"{label}: {q.display_name}"
    return headline, reasons
