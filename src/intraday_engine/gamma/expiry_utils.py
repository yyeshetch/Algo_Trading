"""Expiry day utilities for Nifty and Bank Nifty."""

from __future__ import annotations

from datetime import date


# Nifty weekly: Tuesday. Monthly: last Tuesday.
# Bank Nifty weekly: Friday. Monthly: last Wednesday.
NIFTY_WEEKLY_EXPIRY_WEEKDAY = 1  # Tuesday (0=Monday, 6=Sunday)
BANKNIFTY_WEEKLY_EXPIRY_WEEKDAY = 4  # Friday


def is_nifty_expiry_day(trade_date: date) -> bool:
    """True if trade_date is a Nifty weekly expiry day (Tuesday)."""
    return trade_date.weekday() == NIFTY_WEEKLY_EXPIRY_WEEKDAY


def is_banknifty_expiry_day(trade_date: date) -> bool:
    """True if trade_date is a Bank Nifty weekly expiry day (Friday)."""
    return trade_date.weekday() == BANKNIFTY_WEEKLY_EXPIRY_WEEKDAY


def is_expiry_day(trade_date: date, underlying: str = "NIFTY") -> bool:
    """True if trade_date is expiry day for the given underlying."""
    u = (underlying or "NIFTY").upper()
    if "BANK" in u or u == "BANKNIFTY":
        return is_banknifty_expiry_day(trade_date)
    return is_nifty_expiry_day(trade_date)
