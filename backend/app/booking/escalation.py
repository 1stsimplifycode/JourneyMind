"""What happens when a rider simply cannot get a ride.

A consumer app's answer to four failed bookings is a spinner. An enterprise
mobility product has a different obligation: the employee is now late for
something, that has a cost, and somebody should be told.

    attempts exhausted
      -> is the rider now at risk of missing their commitment?
      -> what should they do instead?
      -> tell the manager, with the rider's explicit consent
      -> record it as an incident the organisation can count

THE CONSENT BOUNDARY
--------------------
`NOTIFY MANAGER` is offered, never automatic. Messaging someone's manager about
their lateness is exactly the kind of outward-facing, hard-to-reverse action
that a system must not take on a person's behalf without them pressing the
button -- and it is the same boundary the MVP draws around booking and paying.

WHAT IS ACTUALLY SENT
---------------------
Nothing leaves this process. The notification is composed, recorded in the
audit trail and returned for display. Wiring it to a real mail or chat
transport is a deployment concern, and pretending it had been sent would be a
lie about an outward-facing action.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta

from ..lifecycle.states import BookingState

#: Minutes of slack below which a rider is "at risk" rather than "late".
AT_RISK_MARGIN_MIN = 10.0


@dataclass(frozen=True)
class MeetingContext:
    """What the rider is trying to get to. Optional, and shallow on purpose.

    No calendar is read and no attendee list is stored: a title and a time are
    all the arrival-risk calculation needs, and anything more would be personal
    data the product has no business holding (see V2 §72).
    """

    title: str
    starts_at: datetime
    manager: str | None = None

    def as_dict(self) -> dict:
        return {"title": self.title,
                "starts_at": self.starts_at.isoformat(timespec="minutes"),
                "manager": self.manager}


@dataclass(frozen=True)
class ArrivalRisk:
    level: str              # on_track | at_risk | late
    minutes_spare: float
    projected_arrival: datetime
    best_remaining_min: float
    best_remaining_label: str
    headline: str
    detail: str

    def as_dict(self) -> dict:
        return {
            "level": self.level,
            "minutes_spare": round(self.minutes_spare, 1),
            "projected_arrival": self.projected_arrival.isoformat(timespec="minutes"),
            "best_remaining_min": round(self.best_remaining_min, 1),
            "best_remaining_label": self.best_remaining_label,
            "headline": self.headline,
            "detail": self.detail,
        }


def assess_arrival(*, now: datetime, wasted_min: float, meeting: MeetingContext,
                   best_option: dict | None) -> ArrivalRisk:
    """Will the rider make it, given the time already burned?

    `best_option` is the fastest option still on the table, taken from the
    comparison the rider was shown -- so the projection uses the same numbers
    as the recommendation rather than a second, quietly different estimate.
    """
    remaining = float(best_option["expected_minutes"]) if best_option else 30.0
    label = best_option["display_name"] if best_option else "the next option"
    arrival = now + timedelta(minutes=remaining)
    spare = (meeting.starts_at - arrival).total_seconds() / 60.0

    if spare >= AT_RISK_MARGIN_MIN:
        level = "on_track"
        headline = f"You can still make {meeting.title}"
        detail = (f"{label} gets you there around "
                  f"{arrival:%H:%M}, about {spare:.0f} minutes early.")
    elif spare >= 0:
        level = "at_risk"
        headline = f"You are cutting it fine for {meeting.title}"
        detail = (f"{label} arrives around {arrival:%H:%M} against a "
                  f"{meeting.starts_at:%H:%M} start — {spare:.0f} minutes spare, "
                  f"and {wasted_min:.0f} minutes are already gone on failed bookings.")
    else:
        level = "late"
        headline = f"You may be late for {meeting.title}"
        detail = (f"Even on {label}, arrival is around {arrival:%H:%M} against a "
                  f"{meeting.starts_at:%H:%M} start — about {abs(spare):.0f} minutes "
                  f"late. {wasted_min:.0f} minutes have gone on bookings that fell "
                  f"through.")
    return ArrivalRisk(level=level, minutes_spare=spare, projected_arrival=arrival,
                       best_remaining_min=remaining, best_remaining_label=label,
                       headline=headline, detail=detail)


def compose_notification(*, session, meeting: MeetingContext, risk: ArrivalRisk,
                         alternative: dict | None) -> dict:
    """The message the rider may choose to send. Composed, never auto-sent."""
    failures = session.failures
    breakdown = ", ".join(
        f"{failures.count(f)}x {f.replace('_', ' ').lower()}"
        for f in dict.fromkeys(failures)) or "no completed booking"

    clock = f"{meeting.starts_at:%H:%M}"
    when = "" if clock in meeting.title else f" ({clock})"
    opener = {
        "late": f"Heads up — I am going to be late for {meeting.title}{when}.",
        "at_risk": f"Heads up — I may be late for {meeting.title}{when}.",
        "on_track": f"Heads up — I was delayed getting to {meeting.title}"
                    f"{when}, though I should still make it.",
    }[risk.level]
    n = len(session.attempts)
    tries = "once" if n == 1 else f"{n} times"
    if session.settled:
        held = f" and it took {tries} ({breakdown} before one stuck). "
    elif n == 1:
        held = f" and it did not hold ({breakdown}). "
    else:
        held = f" and none of them held ({breakdown}). "

    body = (
        f"{opener}" + chr(10) * 2 +
        f"I have tried {tries} to book a "
        f"{session.display_name} from {session.origin_label} to "
        f"{session.dest_label}"
        # "none of them held" stops being true the moment one of them does.
        # This message goes to someone's manager, which is exactly where a
        # small inaccuracy is expensive.
        + held
        + f"About {session.wasted_min:.0f} minutes have gone on that."
        + chr(10) * 2)
    if alternative and alternative.get("p_success") is not None:
        body += (f"I am switching to {alternative['display_name']}, which "
                 f"completes {alternative['p_success']:.0%} of the time, and "
                 f"expect to arrive around {risk.projected_arrival:%H:%M}.")
    elif alternative:
        # an itinerary rather than a single booking
        body += (f"I am travelling in stages instead — "
                 f"{alternative['display_name']} — and expect to arrive around "
                 f"{risk.projected_arrival:%H:%M}.")
    else:
        body += f"I expect to arrive around {risk.projected_arrival:%H:%M}."

    return {
        "to": meeting.manager or "your manager",
        "subject": f"Running late for {meeting.title}",
        "body": body,
        "delivery": "composed_not_sent",
        "delivery_note": (
            "Composed and recorded, not transmitted. This deployment has no mail "
            "or chat transport wired in, and reporting a message as sent when it "
            "was not would be a false claim about an outward-facing action."),
    }


def incident_record(*, session, meeting: MeetingContext, risk: ArrivalRisk,
                    notified: bool, minute_cost: float = 6.0) -> dict:
    """The organisation-facing view of one stranded employee.

    This is the unit the enterprise dashboard counts. It carries no employee
    identifier -- a route, a provider, a time and a cost are what an operations
    team needs to act, and a name is what they do not.
    """
    lost_cost = session.wasted_min * minute_cost
    return {
        "incident_id": f"inc_{session.session_id[3:]}",
        "opened_at": datetime.now().replace(microsecond=0).isoformat(),
        "kind": "repeated_booking_failure",
        "severity": {"late": "high", "at_risk": "medium",
                     "on_track": "low"}[risk.level],
        "route": f"{session.origin_label} → {session.dest_label}",
        "provider": session.provider_id,
        "attempts": len(session.attempts),
        "failures": session.failures,
        "minutes_lost": round(session.wasted_min, 1),
        "productivity_cost": round(lost_cost, 2),
        "productivity_cost_basis": (
            f"{session.wasted_min:.0f} min lost x ₹{minute_cost:.0f}/min loaded "
            f"cost. The minutes are counted; the rate is an assumption."),
        "arrival_risk": risk.level,
        "meeting": meeting.as_dict(),
        "manager_notified": notified,
        "data_class": "SIMULATED",
    }
