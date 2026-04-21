"""Wilder RSI on close series."""

from __future__ import annotations

import pandas as pd


def rsi_series(close: pd.Series, period: int | None = None) -> pd.Series:
    """Return RSI series (NaN until enough bars)."""
    if period is None:
        from intraday_engine.core.tunables import get_int

        period = get_int("rsi", "DEFAULT_PERIOD", 14)
    close = pd.Series(close, dtype=float).reset_index(drop=True)
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, float("nan"))
    out = 100.0 - (100.0 / (1.0 + rs))
    return out.astype(float)


def rsi_last(close: pd.Series, period: int | None = None) -> float | None:
    """Last non-NaN RSI value, or None."""
    s = rsi_series(close, period)
    if s.empty:
        return None
    last = s.dropna()
    if last.empty:
        return None
    return float(last.iloc[-1])
