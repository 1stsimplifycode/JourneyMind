"""The study area's clock.

Every time-of-day judgement this service makes -- the congestion peak shape,
headways, whether the last train has gone -- is about local wall-clock time in
the study area. The server it runs on may be anywhere, so "now" is never
`datetime.now()` without a zone attached to it.

`tzdata` is a declared dependency because a Windows or slim-container host has
no system zone database; if it is somehow still missing, the fallback keeps the
service answering with a fixed offset rather than failing the request.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from functools import lru_cache

DEFAULT_TZ = "Asia/Kolkata"
# Only used if the zone database is unavailable. IST has no DST, so a fixed
# offset is a faithful fallback for this study area rather than a fudge.
FALLBACK_OFFSET = timezone(timedelta(hours=5, minutes=30), "IST")


@lru_cache(maxsize=8)
def city_tz(name: str | None = None):
    try:
        from zoneinfo import ZoneInfo
        return ZoneInfo(name or DEFAULT_TZ)
    except Exception:                       # no tzdata, or an unknown zone name
        return FALLBACK_OFFSET


def now_local(tz_name: str | None = None) -> datetime:
    """Right now on the study area's clock, as a naive local datetime.

    Naive on purpose: the peak-shape model, the headway tables and the
    service-hours check all reason in local wall-clock hours, and an aware
    datetime in some other zone would silently shift every one of them.
    """
    return datetime.now(city_tz(tz_name)).replace(microsecond=0, tzinfo=None)


def to_local(dt: datetime | None, tz_name: str | None = None) -> datetime:
    """Whatever the client sent, expressed on the study area's clock.

    Browsers send `new Date(...).toISOString()`, which is UTC. Reading 09:00
    IST as 09:00 would have priced the morning peak at 03:30 -- so an aware
    timestamp is converted, and a naive one is taken at face value.
    """
    if dt is None:
        return now_local(tz_name)
    if dt.tzinfo is None:
        return dt.replace(microsecond=0)
    return dt.astimezone(city_tz(tz_name)).replace(microsecond=0, tzinfo=None)
