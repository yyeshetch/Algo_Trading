"""Hourly support/resistance from spot 60-minute candles."""

from __future__ import annotations

import logging
from datetime import date, datetime, timedelta
from typing import Any

import pandas as pd

from intraday_engine.core.config import Settings
from intraday_engine.fetch.zerodha_client import ZerodhaClient

logger = logging.getLogger(__name__)

DEFAULT_LOOKBACK_BARS = 20
NEAR_LEVEL_PCT = 0.30  # spot within this % of level counts as "at" support/resistance


def _to_hourly_df(rows: list[dict]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    if "date" not in df.columns:
        return pd.DataFrame()
    df["date"] = pd.to_datetime(df["date"])
    rename = {"open": "open", "high": "high", "low": "low", "close": "close"}
    for col in rename:
        if col not in df.columns:
            return pd.DataFrame()
    return df.sort_values("date").reset_index(drop=True)


def fetch_hourly_spot_levels(
    client: ZerodhaClient,
    settings: Settings,
    trade_date: date | None = None,
    lookback_bars: int = DEFAULT_LOOKBACK_BARS,
) -> dict[str, Any]:
    """
    Return hourly support/resistance from completed 60-minute spot candles.
    Uses min(low) / max(high) over the last `lookback_bars` completed hourly bars.
    """
    d = trade_date or date.today()
    to_dt = datetime.combine(d, datetime.min.time()).replace(hour=15, minute=30)
    from_dt = to_dt - timedelta(days=30)
    try:
        quote = client.quote([settings.spot_symbol])
        if not quote or settings.spot_symbol not in quote:
            return _empty_levels("spot quote unavailable")
        token = int(quote[settings.spot_symbol]["instrument_token"])
        rows = client.historical_data(token, from_dt, to_dt, interval="60minute")
        df = _to_hourly_df(rows)
        if df.empty:
            return _empty_levels("no hourly candles")
        return _levels_from_hourly_df(df, lookback_bars)
    except Exception as exc:
        logger.warning("Hourly levels fetch failed: %s", exc)
        return _empty_levels(str(exc))


def hourly_levels_from_snapshots(
    snapshots_df: pd.DataFrame,
    lookback_hours: int = DEFAULT_LOOKBACK_BARS,
) -> dict[str, Any]:
    """Fallback: bucket 5-min snapshots into NSE hourly bars for the session."""
    if snapshots_df.empty or "timestamp" not in snapshots_df.columns:
        return _empty_levels("no snapshots")
    df = snapshots_df.copy()
    df["ts"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["ts"])
    if df.empty:
        return _empty_levels("invalid timestamps")
    df["hour"] = df["ts"].dt.floor("h")
    open_col = "spot_open_raw" if "spot_open_raw" in df.columns else "spot_open"
    hourly = (
        df.groupby("hour", sort=True)
        .agg(
            open=(open_col, "first"),
            high=("spot_high", "max"),
            low=("spot_low", "min"),
            close=("spot_ltp", "last"),
        )
        .reset_index()
    )
    hourly = hourly.rename(columns={"hour": "date"})
    if hourly.empty:
        return _empty_levels("no hourly buckets")
    return _levels_from_hourly_df(hourly, lookback_hours)


def _levels_from_hourly_df(df: pd.DataFrame, lookback_bars: int) -> dict[str, Any]:
    completed = df.iloc[:-1] if len(df) > 1 else df
    window = completed.tail(max(lookback_bars, 3))
    support = float(window["low"].min())
    resistance = float(window["high"].max())
    return {
        "support": round(support, 2),
        "resistance": round(resistance, 2),
        "lookback_bars": int(len(window)),
        "source": "hourly",
        "last_hour": str(window.iloc[-1]["date"]) if len(window) else None,
        "near_level_pct": NEAR_LEVEL_PCT,
        "error": None,
    }


def _empty_levels(reason: str) -> dict[str, Any]:
    return {
        "support": None,
        "resistance": None,
        "lookback_bars": 0,
        "source": None,
        "last_hour": None,
        "near_level_pct": NEAR_LEVEL_PCT,
        "error": reason,
    }


def spot_near_level(spot: float, level: float | None, pct: float = NEAR_LEVEL_PCT) -> bool:
    if not spot or not level:
        return False
    dist_pct = abs(spot - level) / spot * 100
    return dist_pct <= pct
