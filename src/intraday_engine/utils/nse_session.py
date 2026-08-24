"""NSE cash-session window helpers (IST, Mon–Fri)."""

from __future__ import annotations

import logging
import time
from datetime import datetime, time as dt_time, timedelta
from typing import Callable
from zoneinfo import ZoneInfo

logger = logging.getLogger(__name__)

IST = ZoneInfo("Asia/Kolkata")

# Short sleeps avoid macOS timer coalescing / App Nap delaying long sleeps past 09:15.
WAIT_CHUNK_SECONDS = 30.0
WAIT_HEARTBEAT_SECONDS = 300.0

# NSE regular session (IST)
SESSION_OPEN = dt_time(9, 15, 0)
SESSION_CLOSE = dt_time(15, 30, 0)
MARKET_OVERVIEW_RUN = dt_time(9, 5, 0)


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


def sleep_in_chunks(
    total_seconds: float,
    *,
    chunk_seconds: float = WAIT_CHUNK_SECONDS,
    should_wake: Callable[[], bool] | None = None,
) -> None:
    """Sleep up to total_seconds in small chunks; exit early when should_wake() is true."""
    remaining = max(0.0, total_seconds)
    while remaining > 0:
        if should_wake and should_wake():
            return
        step = min(chunk_seconds, remaining)
        time.sleep(step)
        remaining -= step


def wait_until_nse_session_open(
    *,
    chunk_seconds: float = WAIT_CHUNK_SECONDS,
    heartbeat_seconds: float = WAIT_HEARTBEAT_SECONDS,
    log: logging.Logger | None = None,
    on_chunk: Callable[[], None] | None = None,
) -> None:
    """
    Block until NSE session is active.

    Uses 30s chunks so macOS does not coalesce one long sleep past the open bell.
    """
    lg = log or logger
    if is_nse_session_active():
        return

    rem = seconds_until_next_session_start()
    lg.info(
        "Outside NSE session (%s); waiting until open (~%.0f min, now %s IST)",
        session_label(),
        rem / 60,
        now_ist().strftime("%H:%M:%S"),
    )
    last_beat = time.monotonic()
    while not is_nse_session_active():
        if on_chunk:
            on_chunk()
        mono = time.monotonic()
        if mono - last_beat >= heartbeat_seconds:
            rem = seconds_until_next_session_start()
            lg.info(
                "Still waiting for NSE open — ~%.0f min left (now %s IST)",
                rem / 60,
                now_ist().strftime("%H:%M:%S"),
            )
            last_beat = mono
        rem = seconds_until_next_session_start()
        sleep_in_chunks(
            rem,
            chunk_seconds=chunk_seconds,
            should_wake=is_nse_session_active,
        )
        if on_chunk:
            on_chunk()

    lg.info(
        "NSE session open at %s IST — resuming",
        now_ist().strftime("%H:%M:%S"),
    )


def sleep_until_next_interval(
    total_seconds: float,
    *,
    chunk_seconds: float = WAIT_CHUNK_SECONDS,
    on_chunk: Callable[[], None] | None = None,
) -> None:
    """Sleep until the next scheduler tick; wake early if session ends."""
    remaining = max(0.0, total_seconds)
    while remaining > 0:
        if on_chunk:
            on_chunk()
        if not is_nse_session_active():
            return
        step = min(chunk_seconds, remaining)
        time.sleep(step)
        remaining -= step
