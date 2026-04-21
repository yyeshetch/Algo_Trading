"""Expiry day utilities for Nifty and Bank Nifty."""

from __future__ import annotations

from datetime import date


# Nifty weekly: Tuesday. Monthly: last Tuesday.
# Bank Nifty weekly: Friday. Monthly: last Wednesday.


def is_nifty_expiry_day(trade_date: date) -> bool:
    """True if trade_date is a Nifty weekly expiry day (Tuesday)."""
    from intraday_engine.core.tunables import get_int

    return trade_date.weekday() == get_int("expiry_calendar", "NIFTY_WEEKLY_EXPIRY_WEEKDAY", 1)


def is_banknifty_expiry_day(trade_date: date) -> bool:
    """True if trade_date is a Bank Nifty weekly expiry day (Friday)."""
    from intraday_engine.core.tunables import get_int

    return trade_date.weekday() == get_int("expiry_calendar", "BANKNIFTY_WEEKLY_EXPIRY_WEEKDAY", 4)


def is_expiry_day(trade_date: date, underlying: str = "NIFTY") -> bool:
    """True if trade_date is expiry day for the given underlying."""
    u = (underlying or "NIFTY").upper()
    if "BANK" in u or u == "BANKNIFTY":
        return is_banknifty_expiry_day(trade_date)
    return is_nifty_expiry_day(trade_date)
