"""Enterprise mobility analytics.

The consumer view answers "where should I travel?". This answers "how should my
organisation manage mobility?" -- and it is the same engine pointed at a
population instead of a person.

WHAT MAKES THIS DIFFERENT FROM A REPORTING DASHBOARD
----------------------------------------------------
Any BI tool can chart historical spend. Two things here are not reporting:

1. **Failure is priced.** A cancelled booking is not a missing row; it is
   wasted minutes and a re-request at a higher fare. Conventional transport
   reporting counts completed trips and is blind to exactly the cost this
   product exists to find.
2. **Providers are ranked on what a kilometre actually costs** -- the billed
   rate divided by the share of bookings that complete. That ranking is not
   the same as the billed ranking, and the difference is the finding.

EVERY NUMBER IS LABELLED
------------------------
Aggregates come from the bundled demonstration history. Anything from the
expected-cost model is a prediction and is tagged as one. Nothing here is a
measurement of a real organisation.

PRIVACY
-------
There is no employee identifier in this pipeline, by construction -- not
hashed, not pseudonymous, absent. Groups below a minimum size are suppressed
rather than rounded, because a cell of two people is re-identifiable whatever
you do to the number.

PERFORMANCE
-----------
Everything below aggregates over NumPy masks rather than iterating rows. See
`store.py` for why that mattered: the list-of-dicts version cost 89 MB and 1.6
seconds per filter click, which is not a demo, it is a stall.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

from ..labels import label_for  # noqa: F401
from .store import BookingTable, Categorical, fare_for, load_table  # noqa: F401

log = logging.getLogger("journeymind.enterprise")

#: Below this many trips a group is suppressed, not rounded. See PRIVACY above.
MIN_COHORT = 25

#: Door-to-door minutes beyond which a commute breaches the employer's SLA.
SLA_MINUTES = 75.0

#: What a wasted minute costs the employer. A loaded-cost assumption, exposed
#: because the ROI arithmetic downstream is only as good as this number.
DEFAULT_MINUTE_COST = 6.0


#: Identifiers are for joins, not for prose. `bike_taxi` as a scorecard column,
#: and inside the sentence "bike_taxi looks cheapest but auto costs less to
#: use", is a database key that escaped into a report an operations lead reads.
#: The table lives in app/labels.py so the dashboard and the rider-facing
#: screens cannot end up calling the same vehicle two different things.


@dataclass(frozen=True)
class Filters:
    campus: str | None = None
    provider: str | None = None
    employee_group: str | None = None
    mode: str | None = None
    date_from: str | None = None
    date_to: str | None = None
    hour_from: int | None = None
    hour_to: int | None = None

    def mask(self, t: BookingTable) -> np.ndarray:
        m = t.all
        if self.campus:
            # accept either the id or the display name, so a typed filter works
            m &= (t.campus_id.mask_for(self.campus) | t.campus.mask_for(self.campus))
        if self.provider:
            m &= t.provider.mask_for(self.provider)
        if self.employee_group:
            m &= t.employee_group.mask_for(self.employee_group)
        if self.mode:
            m &= t.mode.mask_for(self.mode)
        if self.date_from or self.date_to:
            dates = np.array(t.date.labels)
            ok = np.ones(len(dates), dtype=bool)
            if self.date_from:
                ok &= dates >= self.date_from
            if self.date_to:
                ok &= dates <= self.date_to
            m &= ok[t.date.codes]
        if self.hour_from is not None:
            m &= t.hour >= self.hour_from
        if self.hour_to is not None:
            m &= t.hour < self.hour_to
        return m

    def as_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items() if v is not None}


@lru_cache(maxsize=1)
def load_bookings(path: str | None = None) -> BookingTable | None:
    """Load once, cache. Returns None when no history is bundled."""
    if path is None:
        from ..config import get_settings
        path = str(Path(get_settings().data_dir) / "mobility" / "bookings.csv")
    return load_table(path)


# --------------------------------------------------------------------------
def _rate(num, den) -> float | None:
    num, den = float(num), float(den)
    return round(num / den, 4) if den else None


def _cnt(mask: np.ndarray) -> int:
    return int(np.count_nonzero(mask))


def overview(t: BookingTable, m: np.ndarray,
             minute_cost: float = DEFAULT_MINUTE_COST) -> dict:
    """The executive numbers."""
    n = _cnt(m)
    if not n:
        return {"bookings": 0, "note": "no bookings match these filters"}
    completed = m & t.completed
    accepted = m & t.accepted
    matched = m & t.matched
    n_completed = _cnt(completed)
    spend = float(t.spend[m].sum())
    wasted = float(t.wasted_min[m].sum())
    breaches = _cnt(completed & (t.door_to_door_min > SLA_MINUTES))
    return {
        "bookings": n,
        "completed_trips": n_completed,
        "booking_success_rate": _rate(n_completed, n),
        "no_supply_rate": _rate(_cnt(m & ~t.matched), n),
        "rejection_rate": _rate(_cnt(matched & ~t.accepted), _cnt(matched)),
        "cancellation_rate": _rate(_cnt(m & t.cancelled), _cnt(accepted)),
        "total_spend": round(spend, 2),
        "mean_trip_cost": round(spend / n_completed, 2) if n_completed else None,
        "wasted_minutes": round(wasted, 1),
        "wasted_minutes_cost": round(wasted * minute_cost, 2),
        "productivity_cost_note": (
            f"Minutes lost to failed bookings, valued at ₹{minute_cost:.0f}/min. "
            "The rate is an assumption; the minutes are counted."),
        "sla_minutes": SLA_MINUTES,
        "sla_breaches": breaches,
        "sla_breach_rate": _rate(breaches, n_completed),
        "mean_distance_km": round(float(t.distance_km[m].mean()), 2),
    }


def _group(t: BookingTable, m: np.ndarray, cat: Categorical, key: str,
           minute_cost: float) -> list[dict]:
    out = []
    for code, label in enumerate(cat.labels):
        g = m & (cat.codes == code)
        n = _cnt(g)
        if not n:
            continue
        if n < MIN_COHORT:
            out.append({key: label, "bookings": n, "suppressed": True,
                        "reason": (f"fewer than {MIN_COHORT} trips — suppressed to "
                                   f"prevent re-identification")})
            continue
        completed = g & t.completed
        n_completed = _cnt(completed)
        spend = float(t.spend[g].sum())
        wasted = float(t.wasted_min[g].sum())
        out.append({
            key: label,
            "bookings": n,
            "completed": n_completed,
            "success_rate": _rate(n_completed, n),
            "cancellation_rate": _rate(_cnt(g & t.cancelled), _cnt(g & t.accepted)),
            "no_supply_rate": _rate(_cnt(g & ~t.matched), n),
            "spend": round(spend, 2),
            "mean_trip_cost": round(spend / n_completed, 2) if n_completed else None,
            "wasted_minutes": round(wasted, 1),
            "wasted_cost": round(wasted * minute_cost, 2),
            "suppressed": False,
        })
    out.sort(key=lambda d: -(d.get("spend") or 0))
    return out


def by_dimension(t: BookingTable, m: np.ndarray, key: str,
                 minute_cost: float = DEFAULT_MINUTE_COST) -> list[dict]:
    cat = {"campus": t.campus, "employee_group": t.employee_group,
           "mode": t.mode, "provider_id": t.provider}[key]
    return _group(t, m, cat, key, minute_cost)


def hourly_profile(t: BookingTable, m: np.ndarray) -> list[dict]:
    """Demand and reliability by hour — where the peaks and the pain are."""
    hours = t.hour.astype(np.int16)
    out = []
    for h in range(24):
        g = m & (hours == h)
        n = _cnt(g)
        out.append({
            "hour": h, "bookings": n,
            "success_rate": _rate(_cnt(g & t.completed), n),
            "cancellation_rate": _rate(_cnt(g & t.cancelled), _cnt(g & t.accepted)),
            "spend": round(float(t.spend[g].sum()), 2) if n else 0.0,
            "suppressed": n < MIN_COHORT,
        })
    return out


def provider_scorecard(t: BookingTable, m: np.ndarray,
                       minute_cost: float = DEFAULT_MINUTE_COST) -> list[dict]:
    """Provider performance, ranked by the number that matters to a payer.

    Reliability-adjusted cost per km: what a kilometre actually costs once the
    failures are paid for. A provider can be cheapest per km and worst on this.
    """
    out = []
    for row in _group(t, m, t.provider, "provider_id", minute_cost):
        if row.get("suppressed"):
            out.append(row)
            continue
        g = m & t.provider.mask_for(row["provider_id"])
        completed = g & t.completed
        km = float(t.distance_km[completed].sum())
        success = row["success_rate"] or 1e-6
        if not km:
            continue
        cost_per_km = row["spend"] / km
        out.append({
            **row,
            # the label the dashboard prints; the id stays for joins
            "display_name": label_for(row["provider_id"]),
            "cost_per_km": round(cost_per_km, 2),
            # every completed trip carries the cost of the attempts that failed
            "reliability_adjusted_cost_per_km": round(cost_per_km / success, 2),
            "mean_wasted_min_per_booking": round(row["wasted_minutes"] / row["bookings"], 2),
        })
    out.sort(key=lambda d: d.get("reliability_adjusted_cost_per_km") or 9e9)
    return out


def insights(t: BookingTable, m: np.ndarray,
             minute_cost: float = DEFAULT_MINUTE_COST) -> list[dict]:
    """Findings, each labelled with what kind of statement it is.

    An `observation` is arithmetic over the history. A `prediction` is the
    model extrapolating. They are never blended into one confident sentence,
    because a reader is entitled to know which is which.
    """
    out: list[dict] = []
    if _cnt(m) < MIN_COHORT * 4:
        return out

    ov = overview(t, m, minute_cost)

    # 1. the expensive hour ------------------------------------------------
    hours = [h for h in hourly_profile(t, m)
             if not h["suppressed"] and h["cancellation_rate"] is not None]
    overall_cx = ov["cancellation_rate"] or 0.0
    if hours and overall_cx:
        worst = max(hours, key=lambda h: h["cancellation_rate"])
        if worst["cancellation_rate"] > overall_cx * 1.25:
            lift = (worst["cancellation_rate"] / overall_cx - 1.0) * 100
            out.append({
                "kind": "observation", "severity": "medium",
                "title": f"Cancellations peak at {worst['hour']:02d}:00",
                "detail": (f"{worst['cancellation_rate']:.0%} of accepted bookings are "
                           f"cancelled in the {worst['hour']:02d}:00 hour, {lift:.0f}% above "
                           f"the {overall_cx:.0%} average across all hours."),
                "evidence": {"hour": worst["hour"], "bookings": worst["bookings"]},
            })

    # 2. the provider that is not what it looks like -----------------------
    cards = [c for c in provider_scorecard(t, m, minute_cost) if not c.get("suppressed")]
    if len(cards) >= 2:
        cheapest_sticker = min(cards, key=lambda c: c["cost_per_km"])
        best_real = min(cards, key=lambda c: c["reliability_adjusted_cost_per_km"])
        if cheapest_sticker["provider_id"] != best_real["provider_id"]:
            out.append({
                "kind": "observation", "severity": "high",
                "title": (f"{label_for(cheapest_sticker['provider_id'])} looks "
                          f"cheapest but {label_for(best_real['provider_id'])} "
                          f"costs less to use"),
                "detail": (
                    f"{label_for(cheapest_sticker['provider_id'])} bills ₹"
                    f"{cheapest_sticker['cost_per_km']:.2f}/km against "
                    f"{label_for(best_real['provider_id'])}'s ₹"
                    f"{best_real['cost_per_km']:.2f}/km, but "
                    f"completes only {cheapest_sticker['success_rate']:.0%} of bookings. "
                    f"Once failed attempts are paid for, it costs ₹"
                    f"{cheapest_sticker['reliability_adjusted_cost_per_km']:.2f}/km "
                    f"against ₹{best_real['reliability_adjusted_cost_per_km']:.2f}/km."),
                "evidence": {"compared": [cheapest_sticker["provider_id"],
                                          best_real["provider_id"]]},
            })

    # 3. the campus carrying the failure cost ------------------------------
    campuses = [c for c in by_dimension(t, m, "campus", minute_cost)
                if not c.get("suppressed")]
    if len(campuses) >= 2:
        worst = max(campuses, key=lambda c: c["wasted_minutes"] / max(c["bookings"], 1))
        total_bookings = sum(c["bookings"] for c in campuses)
        mean_waste = sum(c["wasted_minutes"] for c in campuses) / max(total_bookings, 1)
        per_booking = worst["wasted_minutes"] / worst["bookings"]
        if mean_waste > 0 and per_booking > mean_waste * 1.15:
            out.append({
                "kind": "observation", "severity": "medium",
                "title": f"{worst['campus']} loses the most time to failed bookings",
                "detail": (f"{per_booking:.1f} minutes wasted per booking against a "
                           f"{mean_waste:.1f}-minute average, or ₹"
                           f"{worst['wasted_cost']:,.0f} of paid time across "
                           f"{worst['bookings']:,} bookings."),
                "evidence": {"campus": worst["campus"], "bookings": worst["bookings"]},
            })

    # 4. the shift worth modelling ----------------------------------------
    if len(cards) >= 2:
        worst = max(cards, key=lambda c: c["reliability_adjusted_cost_per_km"])
        best = min(cards, key=lambda c: c["reliability_adjusted_cost_per_km"])
        if worst["provider_id"] != best["provider_id"] and worst["spend"] > 0:
            gap = (worst["reliability_adjusted_cost_per_km"]
                   - best["reliability_adjusted_cost_per_km"])
            movable = worst["spend"] * 0.25
            saving = movable * (gap / max(worst["reliability_adjusted_cost_per_km"], 1e-6))
            out.append({
                "kind": "prediction", "severity": "medium",
                "title": (f"Moving a quarter of {worst['provider_id']} trips to "
                          f"{best['provider_id']} models a ₹{saving:,.0f} saving"),
                "detail": (
                    f"MODELLED, NOT OBSERVED. Applies the reliability-adjusted cost "
                    f"gap of ₹{gap:.2f}/km to 25% of {worst['provider_id']} spend. "
                    f"Assumes the substituted trips behave like the existing "
                    f"{best['provider_id']} population, which is exactly the "
                    f"assumption a pilot would test."),
                "evidence": {"from": worst["provider_id"], "to": best["provider_id"],
                             "assumed_shift_pct": 25},
            })
    return out


def build(t: BookingTable | None, filters: Filters | None = None,
          minute_cost: float = DEFAULT_MINUTE_COST) -> dict:
    """The whole enterprise payload for one filter selection."""
    f = filters or Filters()
    if t is None or not t.n:
        return {"data_class": "SIMULATED", "filters_applied": f.as_dict(),
                "cohort_floor": MIN_COHORT, "overview": {"bookings": 0},
                "by_campus": [], "by_employee_group": [], "by_mode": [],
                "providers": [], "hourly": [], "insights": [],
                "data_note": "No booking history is bundled with this deployment."}
    m = f.mask(t)
    return {
        "data_class": "SIMULATED",
        "data_note": ("Aggregated from the bundled demonstration booking history. "
                      "Not measurements of any real organisation. Items tagged "
                      "'prediction' are model extrapolations rather than counts."),
        "filters_applied": f.as_dict(),
        "cohort_floor": MIN_COHORT,
        "overview": overview(t, m, minute_cost),
        "by_campus": by_dimension(t, m, "campus", minute_cost),
        "by_employee_group": by_dimension(t, m, "employee_group", minute_cost),
        "by_mode": by_dimension(t, m, "mode", minute_cost),
        "providers": provider_scorecard(t, m, minute_cost),
        "hourly": hourly_profile(t, m),
        "insights": insights(t, m, minute_cost),
    }


def facets(t: BookingTable | None) -> dict:
    """The filter options the dashboard offers, taken from the data itself."""
    if t is None or not t.n:
        return {"campuses": [], "providers": [], "employee_groups": [], "modes": [],
                "date_range": None}
    campuses = sorted({(t.campus_id.label_of(int(c)), t.campus.label_of(int(n)))
                       for c, n in zip(t.campus_id.codes, t.campus.codes)})
    dates = sorted(t.date.labels)
    return {
        "campuses": [{"id": cid, "name": name} for cid, name in campuses],
        "providers": sorted(t.provider.labels),
        "employee_groups": sorted(t.employee_group.labels),
        "modes": sorted(t.mode.labels),
        "date_range": {"from": dates[0], "to": dates[-1]},
    }
