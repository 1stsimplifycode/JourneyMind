"""Columnar store for the booking history.

WHY THIS EXISTS
---------------
The obvious way to hold 60,000 bookings is a list of dicts, and that is what
this project did first. Measured, it cost:

    89 MB of memory                       on a 512 MB free-tier instance
    671 ms to load                        paid on the first enterprise request
    1.6 s per unfiltered aggregate        paid on every filter click

A demo where every filter click costs a second and a half is a demo that feels
broken, and 89 MB of dictionary overhead to hold 10 MB of CSV is simply waste:
almost all of it is per-dict and per-key overhead repeated sixty thousand times.

So the rows are held as typed NumPy columns instead. Filtering becomes a
boolean mask, aggregation becomes a sum over that mask, and the numbers that
come out are identical.

WHAT IT IS NOT
--------------
Not a database, and not trying to be. It is a read-only table loaded once from
a bundled CSV. If this ever needs joins, writes or more than one process, that
is the moment to reach for Postgres — not now.
"""

from __future__ import annotations

import csv
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

log = logging.getLogger("journeymind.enterprise.store")

#: Per-km rates and base fares used to price historical trips, in rupees.
#: Drawn from the same fare families as fares.json so the enterprise figures
#: and the consumer quotes speak the same currency.
#: The hailed providers the product actually offers. History for anything else
#: -- carpool, in the bundled dataset -- is excluded at load: a scorecard row
#: for a mode nobody can book is not an insight.
OFFERED_PROVIDERS = frozenset({"bike_taxi", "auto", "cab"})

RATE_PER_KM = {"bike_taxi": 8.5, "auto": 14.0, "cab": 20.0}
BASE_FARE = {"bike_taxi": 20.0, "auto": 30.0, "cab": 55.0}

#: Minutes lost to each kind of failure. A booking that never happened still
#: cost the employee time, which is the cost conventional reporting misses.
WASTE_NO_SUPPLY = 2.5
WASTE_REJECTED = 0.6
WASTE_CANCELLED = 5.5


def fare_for(mode: str, distance_km: float) -> float:
    return BASE_FARE.get(mode, 30.0) + RATE_PER_KM.get(mode, 14.0) * distance_km


@dataclass(frozen=True)
class Categorical:
    """Integer codes plus their labels — one small array, not 60,000 strings."""

    codes: np.ndarray
    labels: tuple[str, ...]

    def mask_for(self, value: str) -> np.ndarray:
        try:
            return self.codes == self.labels.index(value)
        except ValueError:
            return np.zeros(len(self.codes), dtype=bool)

    def label_of(self, code: int) -> str:
        return self.labels[code]


class BookingTable:
    """The booking history, column by column."""

    __slots__ = ("n", "excluded_rows", "distance_km", "pickup_km", "hour", "peak_intensity",
                 "zone_congestion", "matched", "accepted", "cancelled",
                 "completed", "rain", "dow", "is_weekend", "late_night",
                 "fare", "wasted_min", "spend", "provider", "mode", "campus",
                 "campus_id", "employee_group", "date", "door_to_door_min")

    def __init__(self, cols: dict[str, tuple[str, ...]]) -> None:
        #: Rows left out because their mode is no longer offered.
        self.excluded_rows = 0
        """Build from raw string columns.

        Conversion happens once per column inside NumPy rather than once per
        cell in Python. Parsing cell by cell cost 3.4 seconds on 60,000 rows;
        this is an order of magnitude cheaper, and the cost is paid on the
        first enterprise request of a cold instance where it is most visible.
        """
        n = self.n = len(next(iter(cols.values()))) if cols else 0

        # `np.fromiter(map(float, col))` beats `np.array(col, dtype="U")` by
        # about 4x per column: the latter allocates a fixed-width unicode array
        # sized to the longest string before it converts anything.
        def f(k):
            return np.fromiter(map(float, cols[k]), dtype=np.float32, count=n)

        def b(k):
            return np.fromiter(map(int, cols[k]), dtype=np.int8, count=n).astype(bool)

        self.distance_km = f("distance_km")
        self.pickup_km = f("pickup_km")
        self.hour = f("hour")
        self.peak_intensity = f("peak_intensity")
        self.zone_congestion = f("zone_congestion_observed")
        self.matched = b("matched")
        self.accepted = b("accepted")
        self.cancelled = b("cancelled")
        self.completed = b("completed")
        self.rain = b("rain")
        self.is_weekend = b("is_weekend")
        self.late_night = b("late_night")
        self.dow = np.fromiter(map(int, cols["dow"]), dtype=np.int8, count=n)

        self.provider = _categorical(cols["provider_id"])
        self.mode = _categorical(cols["mode"])
        self.campus = _categorical(cols["campus"])
        self.campus_id = _categorical(cols["campus_id"])
        self.employee_group = _categorical(cols["employee_group"])
        self.date = _categorical(tuple(v[:10] for v in cols["ts"]))

        # Derived once, at load, because every aggregate needs them.
        rate = np.array([RATE_PER_KM.get(m, 14.0) for m in self.mode.labels],
                        dtype=np.float32)[self.mode.codes]
        base = np.array([BASE_FARE.get(m, 30.0) for m in self.mode.labels],
                        dtype=np.float32)[self.mode.codes]
        self.fare = base + rate * self.distance_km
        self.spend = np.where(self.completed, self.fare, 0.0).astype(np.float32)
        self.wasted_min = np.where(
            self.completed, 0.0,
            np.where(~self.matched, WASTE_NO_SUPPLY,
                     np.where(~self.accepted, WASTE_REJECTED, WASTE_CANCELLED))
        ).astype(np.float32)
        # A crude door-to-door estimate at an 18 km/h city average, used only
        # for the SLA count. Labelled as an estimate wherever it surfaces.
        self.door_to_door_min = (self.distance_km / 18.0 * 60.0
                                 + self.wasted_min).astype(np.float32)

    def __len__(self) -> int:
        return self.n

    @property
    def all(self) -> np.ndarray:
        return np.ones(self.n, dtype=bool)


def _categorical(values) -> Categorical:
    """Factorise a string column. np.unique does the work in C."""
    labels, codes = np.unique(np.asarray(values, dtype="U"), return_inverse=True)
    return Categorical(codes=codes.astype(np.int16),
                       labels=tuple(str(x) for x in labels))


def load_table(path: str | Path) -> BookingTable | None:
    p = Path(path)
    if not p.exists():
        log.warning("no booking history at %s — enterprise views will be empty", p)
        return None
    with open(p, newline="", encoding="utf-8") as fh:
        reader = csv.reader(fh)
        header = next(reader, None)
        if header is None:
            return None
        # zip(*rows) transposes at C speed. Reading into lists-of-strings and
        # transposing avoids ever materialising 60,000 dictionaries, which was
        # both the memory spike and most of the load time.
        rows = list(reader)

    # The bundled history predates the removal of carpool and still carries a
    # quarter of its rows against it. Reporting on a mode the product no longer
    # offers would put a provider on the scorecard that no employee can book,
    # so those rows are dropped here rather than by regenerating and retraining
    # everything downstream. The count is logged and surfaced, never silent.
    dropped = 0
    if rows and "provider_id" in header:
        pi = header.index("provider_id")
        keep = [r for r in rows if r[pi] in OFFERED_PROVIDERS]
        dropped = len(rows) - len(keep)
        rows = keep

    cols = {name: col for name, col in zip(header, zip(*rows))} if rows else {}
    del rows
    table = BookingTable(cols)
    table.excluded_rows = dropped
    if dropped:
        log.info("enterprise: loaded %d bookings (columnar); %d rows excluded "
                 "for modes JourneyMind no longer offers", table.n, dropped)
    else:
        log.info("enterprise: loaded %d bookings (columnar)", table.n)
    return table
