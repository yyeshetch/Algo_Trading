"""Fetch 15-min candles with OI for F&O stock scanner."""

from __future__ import annotations

from datetime import date, datetime, timedelta
from typing import Any

import pandas as pd

from intraday_engine.core.config import Settings
from intraday_engine.core.underlyings import get_underlying_config
from intraday_engine.fetch.instrument_resolver import InstrumentResolver
from intraday_engine.fetch.zerodha_client import ZerodhaClient


def _market_window_15min(trade_date: date | None) -> tuple[datetime, datetime]:
    """Session window for completed 15-min candles only."""
    now = datetime.now()
    target_date = trade_date or now.date()
    session_start = datetime.combine(target_date, datetime.min.time()).replace(hour=9, minute=15)
    session_end = datetime.combine(target_date, datetime.min.time()).replace(hour=15, minute=30)
    if target_date < now.date():
        return session_start, session_end
    if target_date > now.date():
        return session_start, session_start - timedelta(minutes=15)
    return session_start, min(session_end, now.replace(second=0, microsecond=0))


def _to_candle_df(rows: list[dict[str, Any]], prefix: str, include_oi: bool = False) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame(columns=["timestamp"])
    raw = pd.DataFrame(rows)
    raw = raw.rename(columns={"date": "timestamp"})
    raw["timestamp"] = pd.to_datetime(raw["timestamp"]).dt.tz_localize(None)
    raw = raw.sort_values("timestamp").reset_index(drop=True)
    cols = {
        "timestamp": raw["timestamp"],
        f"{prefix}_open_raw": raw["open"].astype(float),
        f"{prefix}_high_raw": raw["high"].astype(float),
        f"{prefix}_low_raw": raw["low"].astype(float),
        f"{prefix}_close": raw["close"].astype(float),
        f"{prefix}_volume": raw["volume"].astype(float),
    }
    if include_oi and "oi" in raw.columns:
        cols[f"{prefix}_oi"] = raw["oi"].astype(float)
    return pd.DataFrame(cols)


def _drop_incomplete_candles(
    df: pd.DataFrame,
    trade_date: date | None,
    interval_minutes: int,
) -> pd.DataFrame:
    if df.empty:
        return df
    now = datetime.now()
    target_date = trade_date or now.date()
    if target_date != now.date():
        return df
    cutoff = min(
        datetime.combine(target_date, datetime.min.time()).replace(hour=15, minute=30),
        now.replace(second=0, microsecond=0),
    )
    completed_mask = df["timestamp"] + pd.to_timedelta(interval_minutes, unit="m") <= cutoff
    return df.loc[completed_mask].reset_index(drop=True)


def fetch_stock_15min_data(
    client: ZerodhaClient,
    stock_name: str,
    trade_date: date | None = None,
) -> dict[str, Any] | None:
    """
    Fetch 15-min candles for spot, future, ATM CE, ATM PE (with OI).
    Returns dict with merged df and symbols, or None if resolution fails.
    """
    uc = get_underlying_config(stock_name)
    settings = Settings.from_env(underlying=stock_name)
    resolver = InstrumentResolver(client, settings)
    from_dt, to_dt = _market_window_15min(trade_date)
    if to_dt < from_dt:
        return None
    effective_date = from_dt.date()

    spot_symbol = uc.spot_symbol
    try:
        spot_quote = client.quote([spot_symbol])
        if not spot_quote or spot_symbol not in spot_quote:
            return None
        spot_token = int(spot_quote[spot_symbol]["instrument_token"])
    except Exception:
        return None

    spot_rows = client.historical_data(spot_token, from_dt, to_dt, interval="15minute")
    spot_df = _drop_incomplete_candles(_to_candle_df(spot_rows, "spot"), trade_date, 15)
    if spot_df.empty:
        return None

    reference_spot = float(spot_df.iloc[0]["spot_open_raw"])
    try:
        symbols = resolver.resolve_for_date(reference_spot, effective_date)
    except Exception:
        return None

    deriv_quotes = client.quote([symbols.fut_symbol, symbols.ce_symbol, symbols.pe_symbol])
    fut_token = int(deriv_quotes[symbols.fut_symbol]["instrument_token"])
    ce_token = int(deriv_quotes[symbols.ce_symbol]["instrument_token"])
    pe_token = int(deriv_quotes[symbols.pe_symbol]["instrument_token"])

    fut_df = _drop_incomplete_candles(
        _to_candle_df(client.historical_data(fut_token, from_dt, to_dt, interval="15minute"), "future"),
        trade_date,
        15,
    )
    ce_df = _drop_incomplete_candles(
        _to_candle_df(
            client.historical_data(ce_token, from_dt, to_dt, interval="15minute", oi=True),
            "call",
            include_oi=True,
        ),
        trade_date,
        15,
    )
    pe_df = _drop_incomplete_candles(
        _to_candle_df(
            client.historical_data(pe_token, from_dt, to_dt, interval="15minute", oi=True),
            "put",
            include_oi=True,
        ),
        trade_date,
        15,
    )

    merged = spot_df.merge(fut_df, on="timestamp", how="inner")
    ce_cols = ["timestamp", "call_open_raw", "call_close", "call_volume"]
    if "call_oi" in ce_df.columns:
        ce_cols.append("call_oi")
    merged = merged.merge(ce_df[ce_cols], on="timestamp", how="inner")
    pe_cols = ["timestamp", "put_open_raw", "put_close", "put_volume"]
    if "put_oi" in pe_df.columns:
        pe_cols.append("put_oi")
    merged = merged.merge(pe_df[pe_cols], on="timestamp", how="inner")
    merged = merged.sort_values("timestamp").reset_index(drop=True)

    if merged.empty:
        return None

    return {
        "stock": stock_name,
        "df": merged,
        "symbols": symbols,
        "spot_open": float(merged.iloc[0]["spot_open_raw"]),
        "spot_close": float(merged.iloc[-1]["spot_close"]),
    }
