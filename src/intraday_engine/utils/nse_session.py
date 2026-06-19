"""NSE cash-session window helpers (IST, Mon–Fri)."""

from __future__ import annotations

from datetime import datetime, time as dt_time, timedelta
from zoneinfo import ZoneInfo

IST = ZoneInfo("Asia/Kolkata")

# NSE regular session (IST)
SESSION_OPEN = dt_time(9, 15, 0)
SESSION_CLOSE = dt_time(15, 30, 0)


def now_ist() -> datetime:
    return datetime.now(IST)


def is_weekday(d: datetime) -> bool:
    return d.weekday() < 5  # Mon=0 … Fri=4


def is_nse_session_active(at: datetime | None = None) -> bool:
    """True during NSE Mon–Fri 09:15–15:30 IST (15:30 inclusive)."""
    t = at.astimezone(IST) if at else now_ist()
    if not is_weekday(t):
        return False
    ts = t.time()
    return SESSION_OPEN <= ts <= SESSION_CLOSE


def session_label() -> str:
    return "Mon–Fri 09:15–15:30 IST"


def next_session_start(after: datetime | None = None) -> datetime:
    """Next session open (09:15 IST on a weekday)."""
    t = (after or now_ist()).astimezone(IST)
    candidate = t.replace(hour=9, minute=15, second=0, microsecond=0)

    if is_weekday(t) and t.time() < SESSION_OPEN:
        return candidate

    # Move to next calendar day, then skip to Monday if needed
    candidate = (t + timedelta(days=1)).replace(
        hour=9, minute=15, second=0, microsecond=0
    )
    while not is_weekday(candidate):
        candidate += timedelta(days=1)
    return candidate


def seconds_until_next_session_start(after: datetime | None = None) -> float:
    nxt = next_session_start(after)
    t = (after or now_ist()).astimezone(IST)
    return max(1.0, (nxt - t).total_seconds())


def seconds_until_session_close(at: datetime | None = None) -> float:
    """Seconds until today's 15:30 IST; 0 if already past close."""
    t = (at or now_ist()).astimezone(IST)
    close = t.replace(hour=15, minute=30, second=0, microsecond=0)
    if t.time() > SESSION_CLOSE:
        return 0.0
    return max(0.0, (close - t).total_seconds())


def seconds_to_next_interval_within_session(interval_minutes: int, at: datetime | None = None) -> float:
    """
    Seconds until the next interval boundary, but not beyond today's session close.
    If already past close or outside session, defer to next session open.
    """
    if not is_nse_session_active(at):
        return seconds_until_next_session_start(at)

    t = (at or now_ist()).astimezone(IST)
    block = (t.minute // interval_minutes) * interval_minutes
    current = t.replace(minute=block, second=0, microsecond=0)
    nxt = current + timedelta(minutes=interval_minutes)
    close = t.replace(hour=15, minute=30, second=0, microsecond=0)

    if nxt > close:
        return max(1.0, seconds_until_session_close(t))

    return max(1.0, (nxt - t).total_seconds())
