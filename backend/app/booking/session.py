"""Live booking sessions — what happens when you actually press BOOK NOW.

This is the demonstration half of the product. The rider presses a button, a
driver is searched for, and the booking either works or falls apart in one of
three ways. The point is that the failure is not theatre:

    THE OUTCOME IS SAMPLED FROM THE SAME PROBABILITIES THE EXPECTED-COST MODEL
    USED TO PRICE THE OPTION.

That single constraint is what makes the reveal honest. When the interface
later says "this option completes 38% of the time", the rider has just lived a
draw from that exact distribution — not a scripted animation with a percentage
bolted on afterwards.

WHY PROBABILITIES ARE FROZEN AT SESSION START
---------------------------------------------
`p_match`, `p_accept` and `p_cancel` are predicted once, when the session opens,
and reused for every retry. Re-predicting per attempt would let the numbers
drift between the failure and the explanation of that failure, and the
explanation would then be describing a different booking than the one that
failed.

WHAT CHANGES BETWEEN ATTEMPTS
-----------------------------
The fare. A re-request lands in a market that has just demonstrated it is
tight, so attempt *n* is priced at `fare x (1 + surge_per_retry)^(n-1)` — the
same escalation the expected-cost solver assumes, so the lived sequence and the
predicted average cannot disagree.

DETERMINISM
-----------
Every session carries its own seeded generator. `demo_seed` fixes it so a live
demonstration is reproducible. **The seed fixes the dice, not the outcome** —
the probabilities are still the model's, and a fixed seed on a 90%-reliable
option still produces a successful booking.
"""

from __future__ import annotations

import logging
import os
import secrets
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np

from ..lifecycle.expected_cost import LifecycleParams
from ..lifecycle.states import BookingState

log = logging.getLogger("journeymind.booking")

SESSION_TTL_SECONDS = 60 * 45
MAX_SESSIONS = 500

def attempt_budget(raw: str | None, default: int = 4) -> int:
    """How many attempts a rider gets before the product escalates instead.

    Four is a demonstration choice, not a finding about riders, so it moves
    with `JM_MAX_BOOKING_ATTEMPTS`. A separate function because a constant read
    at import time can only be tested by reloading the module, and reloading
    this one swaps the session classes underneath a live store.
    """
    try:
        return max(1, int(raw)) if raw else default
    except ValueError:
        log.warning("JM_MAX_BOOKING_ATTEMPTS=%r is not a number; using %d",
                    raw, default)
        return default


MAX_ATTEMPTS = attempt_budget(os.getenv("JM_MAX_BOOKING_ATTEMPTS"))

#: How long the interface should dwell on each step. Chosen so a whole attempt
#: takes 4-7 seconds: long enough to read, short enough that a retry does not
#: stall a two-minute demo.
DWELL_MS = {
    BookingState.REQUESTED: 900,
    BookingState.DRIVER_MATCHED: 1500,
    BookingState.DRIVER_ACCEPTED: 1200,
    BookingState.RIDE_STARTED: 1100,
    BookingState.NO_DRIVER_AVAILABLE: 2200,
    BookingState.DRIVER_REJECTED: 1400,
    BookingState.DRIVER_CANCELLED: 1800,
    BookingState.RIDE_COMPLETED: 600,
}


@dataclass(frozen=True)
class Step:
    """One frame of the booking, as the rider sees it."""

    state: str
    label: str
    detail: str
    dwell_ms: int
    tone: str          # progress | good | bad

    def as_dict(self) -> dict:
        return {"state": self.state, "label": self.label, "detail": self.detail,
                "dwell_ms": self.dwell_ms, "tone": self.tone}


@dataclass
class Attempt:
    number: int
    fare: float
    steps: list[Step]
    outcome: BookingState
    wasted_min: float
    driver_name: str | None = None
    eta_min: float | None = None

    @property
    def succeeded(self) -> bool:
        return self.outcome is BookingState.RIDE_COMPLETED

    def as_dict(self) -> dict:
        return {
            "number": self.number,
            "fare": round(self.fare, 2),
            "outcome": self.outcome.value,
            "succeeded": self.succeeded,
            "wasted_min": round(self.wasted_min, 1),
            "driver_name": self.driver_name,
            "eta_min": round(self.eta_min, 1) if self.eta_min is not None else None,
            "steps": [s.as_dict() for s in self.steps],
        }


@dataclass
class BookingSession:
    session_id: str
    provider_id: str
    display_name: str
    mode: str
    service_class: str
    origin_label: str
    dest_label: str
    departure: datetime
    base_fare: float
    pickup_min: float
    ride_min: float
    p_match: float
    p_accept: float
    p_cancel: float
    params: LifecycleParams
    rng: np.random.Generator
    created_at: float = field(default_factory=time.time)
    attempts: list[Attempt] = field(default_factory=list)
    demo: bool = False
    #: The full comparison as it was at session start, so the reveal describes
    #: the options the rider was actually shown rather than a fresh calculation.
    comparison: dict | None = None

    @property
    def total_paid(self) -> float:
        return self.attempts[-1].fare if self.settled else 0.0

    @property
    def settled(self) -> bool:
        return bool(self.attempts) and self.attempts[-1].succeeded

    @property
    def exhausted(self) -> bool:
        return len(self.attempts) >= MAX_ATTEMPTS and not self.settled

    @property
    def wasted_min(self) -> float:
        return sum(a.wasted_min for a in self.attempts)

    @property
    def failures(self) -> list[str]:
        return [a.outcome.value for a in self.attempts if not a.succeeded]

    #: Minutes the rider has now lost, plus the journey still ahead of them.
    @property
    def elapsed_min(self) -> float:
        return self.wasted_min + (0.0 if self.settled else 0.0)

    @property
    def escalated(self) -> bool:
        """Out of attempts, still not moving. This is where a consumer app
        leaves the rider stranded and an enterprise product does not."""
        return self.exhausted

    def as_dict(self) -> dict:
        return {
            "session_id": self.session_id,
            "provider_id": self.provider_id,
            "display_name": self.display_name,
            "mode": self.mode,
            "service_class": self.service_class,
            "origin": self.origin_label,
            "destination": self.dest_label,
            "attempts": [a.as_dict() for a in self.attempts],
            "attempt_count": len(self.attempts),
            "max_attempts": MAX_ATTEMPTS,
            "settled": self.settled,
            "exhausted": self.exhausted,
            "can_retry": not self.settled and not self.exhausted,
            "advertised_fare": round(self.base_fare, 2),
            "paid": round(self.total_paid, 2) if self.settled else None,
            "wasted_min": round(self.wasted_min, 1),
            "failures": self.failures,
            "demo": self.demo,
            "escalated": self.escalated,
            "attempts_left": max(0, MAX_ATTEMPTS - len(self.attempts)),
        }


# --------------------------------------------------------------------------
# driver names: cosmetic, and deliberately generic
# --------------------------------------------------------------------------
_FIRST = ("Ravi", "Suresh", "Manjunath", "Anil", "Prakash", "Ganesh",
          "Kiran", "Naveen", "Shivu", "Mahesh", "Vinod", "Basavaraj")


def _driver_name(rng: np.random.Generator) -> str:
    return f"{_FIRST[int(rng.integers(len(_FIRST)))]} {chr(65 + int(rng.integers(26)))}."


# --------------------------------------------------------------------------
def _run_attempt(session: BookingSession, number: int) -> Attempt:
    """Sample one attempt through the state machine and narrate it.

    The three failure branches are drawn against the frozen probabilities in
    the order they occur in reality: is anyone there, will they take it, will
    they stay. Their *time* costs differ sharply, and that difference is what
    the expected-cost model prices.
    """
    p = session.params
    rng = session.rng
    fare = session.base_fare * (1.0 + p.surge_per_retry) ** (number - 1)
    steps: list[Step] = []
    wasted = 0.0

    # A timetabled service is a different sequence, not the same one with the
    # probabilities set to certainty. Nobody searches for the driver of a
    # metro.
    if session.service_class == "scheduled":
        return _run_scheduled(session, number, fare)

    steps.append(Step(
        state=BookingState.REQUESTED.value,
        label="Searching for driver…",
        detail=f"{session.display_name} · ₹{fare:,.0f}",
        dwell_ms=DWELL_MS[BookingState.REQUESTED], tone="progress"))

    # 1. is anyone there?
    if rng.random() >= session.p_match:
        wasted += p.search_timeout_min
        steps.append(Step(
            state=BookingState.NO_DRIVER_AVAILABLE.value,
            label="No driver available",
            detail=("Nobody responded nearby. Waiting longer rarely helps once a "
                    "search has timed out."),
            dwell_ms=DWELL_MS[BookingState.NO_DRIVER_AVAILABLE], tone="bad"))
        return Attempt(number, fare, steps, BookingState.NO_DRIVER_AVAILABLE, wasted)

    driver = _driver_name(rng)
    eta = session.pickup_min
    wasted += p.match_min
    steps.append(Step(
        state=BookingState.DRIVER_MATCHED.value,
        label="Driver found",
        detail=f"{driver} · {eta:.0f} min away",
        dwell_ms=DWELL_MS[BookingState.DRIVER_MATCHED], tone="progress"))

    # 2. will they take it?
    if rng.random() >= session.p_accept:
        steps.append(Step(
            state=BookingState.DRIVER_REJECTED.value,
            label="Driver declined the trip",
            detail=("Drivers can decline a request. Short fares are declined more "
                    "often when demand is high."),
            dwell_ms=DWELL_MS[BookingState.DRIVER_REJECTED], tone="bad"))
        return Attempt(number, fare, steps, BookingState.DRIVER_REJECTED, wasted, driver, eta)

    steps.append(Step(
        state=BookingState.DRIVER_ACCEPTED.value,
        label="Driver accepted your request",
        detail=f"{driver} is on the way · arriving in {eta:.0f} min",
        dwell_ms=DWELL_MS[BookingState.DRIVER_ACCEPTED], tone="good"))

    # 3. will they stay? — the expensive failure: the clock has been running
    if rng.random() < session.p_cancel:
        lost = eta * p.cancel_discovery_frac
        wasted += lost
        steps.append(Step(
            state=BookingState.DRIVER_CANCELLED.value,
            label="Driver cancelled your ride",
            detail=(f"{driver} cancelled after accepting. You have already waited "
                    f"about {lost:.0f} minutes."),
            dwell_ms=DWELL_MS[BookingState.DRIVER_CANCELLED], tone="bad"))
        return Attempt(number, fare, steps, BookingState.DRIVER_CANCELLED, wasted, driver, eta)

    steps.append(Step(
        state=BookingState.RIDE_STARTED.value,
        label="Ride started",
        detail=f"On the way · about {session.ride_min:.0f} min",
        dwell_ms=DWELL_MS[BookingState.RIDE_STARTED], tone="progress"))
    steps.append(Step(
        state=BookingState.RIDE_COMPLETED.value,
        label="Journey completed",
        detail=f"You paid ₹{fare:,.0f}",
        dwell_ms=DWELL_MS[BookingState.RIDE_COMPLETED], tone="good"))
    return Attempt(number, fare, steps, BookingState.RIDE_COMPLETED, wasted, driver, eta)


def _run_scheduled(session: BookingSession, number: int, fare: float) -> Attempt:
    """Boarding a timetabled service.

    No driver is matched, nobody accepts, and nobody cancels on you personally
    -- a train that is running is a train you can get on. What CAN go wrong is
    that it is not running at all, and that is already priced into the wait the
    router charged.
    """
    steps = [
        Step(state=BookingState.REQUESTED.value,
             label="Checking the service",
             detail=f"{session.display_name} · ₹{fare:,.0f}",
             dwell_ms=DWELL_MS[BookingState.REQUESTED], tone="progress"),
        Step(state=BookingState.RIDE_STARTED.value,
             label="On board",
             detail=(f"No booking needed — turn up and travel · about "
                     f"{session.ride_min:.0f} min"),
             dwell_ms=DWELL_MS[BookingState.RIDE_STARTED], tone="progress"),
        Step(state=BookingState.RIDE_COMPLETED.value,
             label="Journey completed",
             detail=f"You paid ₹{fare:,.0f}",
             dwell_ms=DWELL_MS[BookingState.RIDE_COMPLETED], tone="good"),
    ]
    return Attempt(number, fare, steps, BookingState.RIDE_COMPLETED, 0.0)


#: The failure a demonstration should show. Driver-accepted-then-cancelled is
#: the pedagogically important one: the rider has already waited most of a
#: pickup, so it is the expensive failure that a flat "cancellation rate"
#: percentage hides. NO_DRIVER_AVAILABLE is cheap by comparison and teaches
#: nothing about why the expected cost moves.
DEMO_TARGET_OUTCOME = BookingState.DRIVER_CANCELLED


def _draw(rng, p_match: float, p_accept: float, p_cancel: float) -> BookingState:
    """One attempt, drawing EXACTLY what `_run_attempt` draws, in order.

    Including the driver's name. It is cosmetic in the narration and anything
    but here: `_driver_name` pulls two integers from the same bit stream, so a
    search that skipped it was predicting a different sequence from the one the
    simulator would then run. Demo mode asked for a cancellation and got
    whatever that misalignment produced.
    """
    if rng.random() >= p_match:
        return BookingState.NO_DRIVER_AVAILABLE
    _driver_name(rng)                       # same two integers, same order
    if rng.random() >= p_accept:
        return BookingState.DRIVER_REJECTED
    if rng.random() < p_cancel:
        return BookingState.DRIVER_CANCELLED
    return BookingState.RIDE_COMPLETED


def seed_for_outcome(target: BookingState, *, p_match: float, p_accept: float,
                     p_cancel: float, max_search: int = 4000,
                     failing_attempts: int = 1) -> int | None:
    """Find a seed whose first attempt lands on `target`.

    This is how demo mode stays reproducible on stage without lying. The
    probabilities are untouched -- what is chosen is which draw from them the
    demonstration starts on, exactly as a presenter would if they re-ran a live
    booking until they got the case they wanted to talk about.

    `failing_attempts` extends that to the whole sequence. A demonstration of
    what happens when a rider CANNOT get a ride has to actually run out of
    attempts: seeding only the first one meant the second frequently succeeded,
    and the arrival-risk panel -- the entire point of the enterprise story --
    appeared under the words "Journey completed". Every attempt is still an
    honest draw from the model's own probabilities; this only chooses where the
    sequence starts.

    Returns None when the run is not reachable, which is the honest answer for
    an option that never cancels (a metro cannot), or for one so reliable that
    four failures in a row do not occur within the search budget.
    """
    if target is BookingState.DRIVER_CANCELLED and (
            p_match <= 0 or p_accept <= 0 or p_cancel <= 0):
        return None
    for seed in range(max_search):
        rng = np.random.default_rng(seed)
        if _draw(rng, p_match, p_accept, p_cancel) is not target:
            continue
        if all(_draw(rng, p_match, p_accept, p_cancel)
               is not BookingState.RIDE_COMPLETED
               for _ in range(max(0, failing_attempts - 1))):
            return seed
    return None


def run_next_attempt(session: BookingSession) -> Attempt:
    if session.settled:
        raise ValueError("this booking has already completed")
    if session.exhausted:
        raise ValueError("no attempts left on this booking")
    attempt = _run_attempt(session, len(session.attempts) + 1)
    session.attempts.append(attempt)
    return attempt


# --------------------------------------------------------------------------
class SessionStore:
    """In-memory, TTL'd, capped. A demo does not need a database, and a bounded
    dictionary cannot become a memory leak."""

    def __init__(self) -> None:
        self._items: dict[str, BookingSession] = {}
        self._lock = threading.Lock()

    def _evict(self) -> None:
        cutoff = time.time() - SESSION_TTL_SECONDS
        stale = [k for k, v in self._items.items() if v.created_at < cutoff]
        for k in stale:
            self._items.pop(k, None)
        if len(self._items) > MAX_SESSIONS:
            for k in sorted(self._items, key=lambda k: self._items[k].created_at
                            )[:len(self._items) - MAX_SESSIONS]:
                self._items.pop(k, None)

    def put(self, session: BookingSession) -> BookingSession:
        with self._lock:
            self._evict()
            self._items[session.session_id] = session
        return session

    def get(self, session_id: str) -> BookingSession | None:
        with self._lock:
            return self._items.get(session_id)

    def __len__(self) -> int:
        return len(self._items)


_store = SessionStore()


def get_store() -> SessionStore:
    return _store


def new_session_id() -> str:
    return "bk_" + secrets.token_urlsafe(9)
