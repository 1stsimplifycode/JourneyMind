"""The booking lifecycle.

A search is not a ride. Between "show me options" and "I arrived" sits a
sequence that can fail in three distinct ways, and the whole premise of this
product is that those failures have a price:

    SEARCHING
       |
    REQUESTED ------------------> NO_DRIVER_AVAILABLE ---+
       |                                                 |
    DRIVER_MATCHED --> DRIVER_REJECTED ------------------+
       |                                                 |
    DRIVER_ACCEPTED --> DRIVER_CANCELLED ----------------+
       |                                                 |
    RIDE_STARTED                                    REBOOKING
       |                                                 |
    RIDE_COMPLETED                          (back to REQUESTED, or ABANDONED
                                             once the retry budget is spent)

The three failure edges are kept separate rather than merged into one
"cancelled" bucket because they have different causes, cost different amounts
of time, and are fixed by different people:

    NO_DRIVER_AVAILABLE  no supply        cheapest failure -- you learn quickly
    DRIVER_REJECTED      driver declined  cheap -- a few seconds of matching
    DRIVER_CANCELLED     accepted, left   EXPENSIVE -- you waited for a pickup
                                          that never came, and the clock ran

That asymmetry is why a single "cancellation rate" percentage is a poor
summary of a provider, and why this module models the states rather than a
scalar.

This module is also the generator behind the simulated booking history: the
same transition probabilities that price a quote are the ones sampled to
produce trip events, so the analytic model and the simulation cannot drift
apart.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum


class BookingState(str, Enum):
    SEARCHING = "SEARCHING"
    REQUESTED = "REQUESTED"
    DRIVER_MATCHED = "DRIVER_MATCHED"
    DRIVER_ACCEPTED = "DRIVER_ACCEPTED"
    RIDE_STARTED = "RIDE_STARTED"
    RIDE_COMPLETED = "RIDE_COMPLETED"

    NO_DRIVER_AVAILABLE = "NO_DRIVER_AVAILABLE"
    DRIVER_REJECTED = "DRIVER_REJECTED"
    DRIVER_CANCELLED = "DRIVER_CANCELLED"
    REBOOKING = "REBOOKING"
    RIDER_CANCELLED = "RIDER_CANCELLED"
    ABANDONED = "ABANDONED"


#: Terminal states. A trajectory ends here or it is not finished.
ABSORBING = {
    BookingState.RIDE_COMPLETED,
    BookingState.ABANDONED,
    BookingState.RIDER_CANCELLED,
}

#: Only these transitions are legal. Enforced rather than documented, so an
#: event stream that claims something impossible is rejected at the door
#: instead of quietly corrupting the analytics that sit downstream.
LEGAL_TRANSITIONS: dict[BookingState, frozenset[BookingState]] = {
    BookingState.SEARCHING: frozenset({
        BookingState.REQUESTED, BookingState.ABANDONED}),
    BookingState.REQUESTED: frozenset({
        BookingState.DRIVER_MATCHED, BookingState.NO_DRIVER_AVAILABLE,
        # A SCHEDULED service is boarded, not matched. There is no driver to
        # find and nobody to accept your request, so a metro or a bus goes
        # straight from "is it running" to "on board". Forcing it through the
        # hailed path produced "Searching for driver… / Kiran K. is on the way"
        # for a train, which is the single least believable thing the product
        # could say.
        BookingState.RIDE_STARTED,
        BookingState.RIDER_CANCELLED}),
    BookingState.DRIVER_MATCHED: frozenset({
        BookingState.DRIVER_ACCEPTED, BookingState.DRIVER_REJECTED,
        BookingState.RIDER_CANCELLED}),
    BookingState.DRIVER_ACCEPTED: frozenset({
        BookingState.RIDE_STARTED, BookingState.DRIVER_CANCELLED,
        BookingState.RIDER_CANCELLED}),
    BookingState.RIDE_STARTED: frozenset({
        BookingState.RIDE_COMPLETED, BookingState.RIDER_CANCELLED}),
    BookingState.NO_DRIVER_AVAILABLE: frozenset({
        BookingState.REBOOKING, BookingState.ABANDONED}),
    BookingState.DRIVER_REJECTED: frozenset({
        BookingState.REBOOKING, BookingState.ABANDONED}),
    BookingState.DRIVER_CANCELLED: frozenset({
        BookingState.REBOOKING, BookingState.ABANDONED}),
    BookingState.REBOOKING: frozenset({
        BookingState.REQUESTED, BookingState.ABANDONED}),
    BookingState.RIDE_COMPLETED: frozenset(),
    BookingState.ABANDONED: frozenset(),
    BookingState.RIDER_CANCELLED: frozenset(),
}

#: The failures that cost the rider a whole attempt, in the order they can occur.
FAILURE_STATES = (
    BookingState.NO_DRIVER_AVAILABLE,
    BookingState.DRIVER_REJECTED,
    BookingState.DRIVER_CANCELLED,
)


class IllegalTransition(ValueError):
    """Raised when an event stream claims a transition the model forbids."""


def can_transition(a: BookingState, b: BookingState) -> bool:
    return b in LEGAL_TRANSITIONS.get(a, frozenset())


def assert_transition(a: BookingState, b: BookingState) -> None:
    if not can_transition(a, b):
        raise IllegalTransition(f"{a.value} -> {b.value} is not a legal transition")


@dataclass
class BookingEvent:
    at: datetime
    state: BookingState
    attempt: int
    fare_quoted: float | None = None
    note: str = ""

    def as_dict(self) -> dict:
        return {"at": self.at.isoformat(timespec="seconds"), "state": self.state.value,
                "attempt": self.attempt, "fare_quoted": self.fare_quoted, "note": self.note}


@dataclass
class BookingTrajectory:
    """One rider's actual path through the lifecycle."""

    provider_id: str
    events: list[BookingEvent] = field(default_factory=list)
    fare_paid: float | None = None
    total_min: float = 0.0
    wasted_min: float = 0.0
    attempts: int = 0

    @property
    def final_state(self) -> BookingState:
        return self.events[-1].state if self.events else BookingState.SEARCHING

    @property
    def completed(self) -> bool:
        return self.final_state is BookingState.RIDE_COMPLETED

    @property
    def failure_states(self) -> list[BookingState]:
        return [e.state for e in self.events if e.state in FAILURE_STATES]

    def push(self, at: datetime, state: BookingState, attempt: int,
             fare_quoted: float | None = None, note: str = "") -> None:
        if self.events:
            assert_transition(self.events[-1].state, state)
        self.events.append(BookingEvent(at, state, attempt, fare_quoted, note))

    def as_dict(self) -> dict:
        return {
            "provider_id": self.provider_id,
            "final_state": self.final_state.value,
            "completed": self.completed,
            "attempts": self.attempts,
            "fare_paid": self.fare_paid,
            "total_min": round(self.total_min, 1),
            "wasted_min": round(self.wasted_min, 1),
            "failures": [s.value for s in self.failure_states],
            "events": [e.as_dict() for e in self.events],
        }


def simulate(rng, provider_id: str, start: datetime, *, p_match: float, p_accept: float,
             p_cancel: float, base_fare: float, surge_per_retry: float,
             pickup_min: float, ride_min: float, search_timeout_min: float,
             match_min: float, cancel_discovery_frac: float,
             max_attempts: int) -> BookingTrajectory:
    """Sample one trajectory. Same probabilities the analytic model uses.

    Used to generate the simulated booking history that the reliability models
    train on and the enterprise dashboard aggregates. Because it shares its
    parameters with `expected_cost.solve()`, a disagreement between the
    simulated mean and the analytic mean is a bug, and `tests/` asserts they
    agree.
    """
    t = BookingTrajectory(provider_id=provider_id)
    now = start
    t.push(now, BookingState.SEARCHING, 0)

    for attempt in range(1, max_attempts + 1):
        t.attempts = attempt
        fare = base_fare * (1.0 + surge_per_retry) ** (attempt - 1)
        t.push(now, BookingState.REQUESTED, attempt, round(fare, 2))

        if rng.random() >= p_match:
            now += timedelta(minutes=search_timeout_min)
            t.wasted_min += search_timeout_min
            t.push(now, BookingState.NO_DRIVER_AVAILABLE, attempt,
                   note="no vehicle responded")
        else:
            now += timedelta(minutes=match_min)
            t.wasted_min += match_min
            t.push(now, BookingState.DRIVER_MATCHED, attempt, round(fare, 2))

            if rng.random() >= p_accept:
                t.push(now, BookingState.DRIVER_REJECTED, attempt,
                       note="driver declined the trip")
            else:
                t.push(now, BookingState.DRIVER_ACCEPTED, attempt, round(fare, 2))
                if rng.random() < p_cancel:
                    # The expensive failure: you already waited most of a pickup.
                    lost = pickup_min * cancel_discovery_frac
                    now += timedelta(minutes=lost)
                    t.wasted_min += lost
                    t.push(now, BookingState.DRIVER_CANCELLED, attempt,
                           note="driver cancelled after accepting")
                else:
                    now += timedelta(minutes=pickup_min)
                    t.push(now, BookingState.RIDE_STARTED, attempt, round(fare, 2))
                    now += timedelta(minutes=ride_min)
                    t.push(now, BookingState.RIDE_COMPLETED, attempt, round(fare, 2))
                    t.fare_paid = round(fare, 2)
                    t.total_min = t.wasted_min + pickup_min + ride_min
                    return t

        if attempt < max_attempts:
            t.push(now, BookingState.REBOOKING, attempt)
        else:
            t.push(now, BookingState.ABANDONED, attempt,
                   note=f"gave up after {max_attempts} attempts")
            t.total_min = t.wasted_min
            return t

    t.total_min = t.wasted_min
    return t
