"""The logic gate every candidate journey passes before it can be ranked.

A path through a graph is not automatically a journey a person could take. The
search will happily return a route that hails two bike taxis in a row, or that
teleports between two stations because both happen to be graph nodes. Those are
arithmetically fine and physically absurd, and the only reliable place to catch
them is between assembly and ranking -- before a bad candidate can win.

    build_journey  ->  VALIDATE  ->  constraints  ->  Pareto  ->  rank

Two severities, and the distinction matters:

    REJECT   the journey is not a thing that can happen. Dropped.
    WARN     the journey is possible but unusual. Kept, and the reason is
             carried on the journey so the interface can say it out loud.

Nothing here is a preference. "Expensive" is not a violation; "arrives after
your deadline" is not a violation. Those are the optimiser's job, and folding
them in here would quietly delete options the rider is entitled to see and
reject for themselves.

WHAT COUNTS AS A REPEATED MODE
------------------------------
No two consecutive legs may share a mode. Two hailed vehicles in a row means
you would have stayed in the first one. A metro line change is real -- Yellow
to Green at Rashtreeya Vidyalaya Road is a journey thousands of people make
daily -- but it is ONE metro journey, so `build_journey` merges it into a
single leg carrying both services as `segments`. By the time a candidate
reaches this gate, "Metro -> Metro" can only mean the assembly went wrong.

WALKING
-------
Walking is not a mode this product recommends. It is also how anybody reaches a
platform, so it stays in the graph and `_absorb_walks` folds it into the leg it
serves. What this gate enforces is the limit: a journey leaning on more than
`MAX_JOURNEY_WALK_MIN` of walking is not a commute, it is a hike with a bus in
the middle, and the honest answer is a first-mile ride instead.
"""

from __future__ import annotations

from dataclasses import dataclass

#: Modes where a second boarding of the same mode means a second vehicle you
#: hailed and paid for separately.
HAILED_MODES = frozenset({"bike_taxi", "auto", "cab"})

#: The only vehicles JourneyMind covers. Anything else in a journey is a bug in
#: the graph, or a mode somebody added without wiring up its fares, availability
#: and reliability -- carpool was exactly that, and it won cards it could not
#: honour. `walk` is deliberately absent: see the module docstring.
ALLOWED_MODES = frozenset({"bike_taxi", "auto", "cab", "metro", "bus"})

#: How far a journey may travel relative to the straight line between the two
#: points, before it stops being a route and becomes a detour. Road networks
#: bend; they do not bend thirteen-fold. The floor stops a very short trip from
#: being punished for a normal corner.
MAX_DETOUR_RATIO = 3.0
MIN_DETOUR_ALLOWANCE_KM = 0.6

#: Total walking a journey may lean on, once folded into the legs it serves.
#: Roughly the far end of a station approach. Past this the rider is not
#: commuting, and a first-mile ride is the real answer.
MAX_JOURNEY_WALK_MIN = 12.0

#: A vehicle leg shorter than this is not a ride, it is a rounding error.
MIN_VEHICLE_LEG_KM = 0.05

#: Tolerance when checking that the legs add up to the journey.
SUM_TOLERANCE_MIN = 0.05
SUM_TOLERANCE_KM = 0.02

#: A leg carrying at least this share of the total distance IS the journey.
#: Anything else in the itinerary is decoration -- the classic "bike taxi 19 km,
#: then one metro stop, then walk" that looks clever and helps nobody.
DOMINANT_LEG_SHARE = 0.85

#: Walking beyond this in one journey is possible but worth saying out loud.
LONG_WALK_WARN_MIN = 35.0


@dataclass(frozen=True)
class Violation:
    code: str
    severity: str          # reject | warn
    message: str

    @property
    def fatal(self) -> bool:
        return self.severity == "reject"

    def as_dict(self) -> dict:
        return {"code": self.code, "severity": self.severity,
                "message": self.message}


def _vehicle_legs(journey):
    return [lg for lg in journey.legs if lg.mode != "walk"]


def duplicates(journeys) -> list[tuple[object, object]]:
    """Journeys a rider would call the same trip, whatever the graph did.

    Two paths differing only in which back street a transfer used are one
    journey. `Journey.signature` collapses that, but a candidate set pooled
    across five weightings can still carry near-identical twins, so cost and
    duration are part of the key here as well.
    """
    seen: dict[tuple, object] = {}
    dupes = []
    for j in journeys:
        key = (j.signature, round(j.cost, 0), round(j.total_min, 0))
        if key in seen:
            dupes.append((seen[key], j))
        else:
            seen[key] = j
    return dupes


def validate_journey(journey, straight_km: float | None = None) -> list[Violation]:
    """Every way this candidate could be nonsense, checked in one place.

    `straight_km` is the crow-flight distance between the rider's two points.
    Without it the detour check is skipped; with it, a seventy-eight metre trip
    can no longer be answered with a one-kilometre bike taxi that loops out to
    a bus stop and back because that was the only ride edge in reach.
    """
    out: list[Violation] = []
    legs = journey.legs

    if not legs:
        return [Violation("empty", "reject", "The journey has no legs.")]

    # -- 1. physical continuity ------------------------------------------
    for a, b in zip(legs, legs[1:]):
        if a.to_node != b.from_node:
            out.append(Violation(
                "discontinuous", "reject",
                f"Leg {a.index + 1} ends at {a.to_name} but leg {b.index + 1} "
                f"starts at {b.from_name}."))

    # -- 2. only the modes this product actually covers -------------------
    for lg in legs:
        if lg.mode not in ALLOWED_MODES:
            out.append(Violation(
                "mode_not_offered", "reject",
                f"A {lg.mode} leg. JourneyMind covers "
                + ", ".join(sorted(ALLOWED_MODES)) + "."))
            break

    # -- 3. repeated modes back to back ----------------------------------
    for a, b in zip(legs, legs[1:]):
        if a.mode != b.mode:
            continue
        if a.mode in HAILED_MODES:
            out.append(Violation(
                "consecutive_hailed", "reject",
                f"Two {a.mode} rides in a row via {a.to_name}. A rider would "
                f"have stayed in the first vehicle."))
        else:
            out.append(Violation(
                "consecutive_same_mode", "reject",
                f"Two {a.mode} legs in a row via {a.to_name}. An interchange "
                f"belongs inside one leg, not beside it."))

    # -- 3. legs that are not really legs ---------------------------------
    for lg in legs:
        if lg.mode != "walk" and lg.distance_km < MIN_VEHICLE_LEG_KM:
            out.append(Violation(
                "zero_length_vehicle", "reject",
                f"A {lg.mode} leg of {lg.distance_km * 1000:.0f} m."))
        if lg.total_min < 0 or lg.distance_km < 0:
            out.append(Violation("negative_leg", "reject",
                                 f"Leg {lg.index + 1} has a negative time or distance."))

    # -- 4. one leg that is the whole trip --------------------------------
    vehicles = _vehicle_legs(journey)
    if len(vehicles) > 1 and journey.distance_km > 0:
        for lg in vehicles:
            if lg.total_km / journey.distance_km >= DOMINANT_LEG_SHARE:
                out.append(Violation(
                    "decorative_transfer", "reject",
                    f"The {lg.mode} leg covers "
                    f"{100 * lg.total_km / journey.distance_km:.0f}% of the "
                    f"distance; the other legs do not earn their transfers."))
                break

    # -- 5. the arithmetic has to close -----------------------------------
    leg_min = sum(lg.total_min for lg in legs)
    if abs(leg_min - journey.total_min) > SUM_TOLERANCE_MIN:
        out.append(Violation(
            "time_mismatch", "reject",
            f"Legs total {leg_min:.1f} min but the journey claims "
            f"{journey.total_min:.1f} min."))
    leg_km = sum(lg.total_km for lg in legs)
    if abs(leg_km - journey.distance_km) > SUM_TOLERANCE_KM:
        out.append(Violation(
            "distance_mismatch", "reject",
            f"Legs total {leg_km:.2f} km but the journey claims "
            f"{journey.distance_km:.2f} km."))
    if journey.total_min <= 0:
        out.append(Violation("zero_duration", "reject",
                             "The journey takes no time at all."))

    # -- 6. the fare has to be a fare -------------------------------------
    fare = journey.total_cost
    if fare.amount < 0 or fare.low < 0:
        out.append(Violation("negative_fare", "reject",
                             f"A fare of {fare.amount:.2f}."))
    elif not (fare.low - 0.51 <= fare.amount <= fare.high + 0.51):
        out.append(Violation(
            "fare_band", "reject",
            f"The point estimate {fare.amount:.0f} sits outside its own band "
            f"{fare.low:.0f}-{fare.high:.0f}."))

    # -- 7. transfers must describe the boardings -------------------------
    boardings = sum(len(lg.segments) if lg.segments else 1
                    for lg in legs if lg.kind in ("transit", "ride"))
    if journey.transfers != max(0, boardings - 1):
        out.append(Violation(
            "transfer_count", "reject",
            f"{boardings} boardings reported as {journey.transfers} transfers."))

    # -- 7b. a route, not a detour ----------------------------------------
    if straight_km and straight_km > 0 and journey.distance_km > 0:
        allowance = max(MIN_DETOUR_ALLOWANCE_KM, straight_km * MAX_DETOUR_RATIO)
        if journey.distance_km > allowance:
            out.append(Violation(
                "absurd_detour", "reject",
                f"{journey.distance_km:.2f} km travelled for a "
                f"{straight_km:.2f} km trip. That is a detour, not a route."))

    # -- 8. walking is access, not a commute ------------------------------
    if journey.walk_min > MAX_JOURNEY_WALK_MIN:
        out.append(Violation(
            "walking_commute", "reject",
            f"{journey.walk_min:.0f} minutes of this journey are on foot. "
            f"Walking is how you reach a vehicle here, not how you travel."))
    elif journey.walk_min > MAX_JOURNEY_WALK_MIN * 0.6:
        out.append(Violation(
            "notable_walk", "warn",
            f"About {journey.walk_min:.0f} minutes of this journey are spent "
            f"getting to and from the vehicles."))

    return out


def partition_valid(journeys, straight_km: float | None = None
                    ) -> tuple[list, list[tuple[object, list[Violation]]]]:
    """Split candidates into (kept, rejected-with-reasons).

    Warnings are attached to the journey rather than acted on, so the interface
    can repeat them to the rider instead of the engine deciding for them.
    """
    kept, rejected = [], []
    for j in journeys:
        problems = validate_journey(j, straight_km)
        fatal = [v for v in problems if v.fatal]
        if fatal:
            rejected.append((j, fatal))
            continue
        j.warnings = [v.message for v in problems if not v.fatal]
        kept.append(j)
    return kept, rejected


def rejection_summary(rejected: list[tuple[object, list[Violation]]]) -> dict:
    """What the validator threw away, for the pipeline trace.

    Recorded rather than silent: a candidate set that suddenly loses half its
    members is something an engineer needs to be able to see.
    """
    counts: dict[str, int] = {}
    examples: dict[str, str] = {}
    for j, problems in rejected:
        for v in problems:
            counts[v.code] = counts.get(v.code, 0) + 1
            examples.setdefault(v.code, v.message)
    return {
        "rejected": len(rejected),
        "by_rule": counts,
        "examples": examples,
    }
