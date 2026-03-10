"""Fetch 15-min spot + future (+ optional options) for F&O stocks."""

from __future__ import annotations

from datetime import date, datetime, timedelta

import pandas as pd

from intraday_engine.core.config import Settings
from intraday_engine.core.underlyings import get_underlying_config
from intraday_engine.fetch.instrument_resolver import InstrumentResolver
from intraday_engine.fetch.zerodha_client import ZerodhaClient


def _market_window_15min(trade_date: date | None) -> tuple[datetime, datetime]:
    """Session window for 15-min candles."""
    if trade_date is not None:
        session_start = datetime.combine(trade_date, datetime.min.time()).replace(hour=9, minute=15)
        session_end = datetime.combine(trade_date, datetime.min.time()).replace(hour=15, minute=30)
        return session_start, session_end
    now = datetime.now()
    session_start = now.replace(hour=9, minute=15, second=0, microsecond=0)
    current_boundary = now.replace(minute=(now.minute // 15) * 15, second=0, microsecond=0)
    previous_completed = current_boundary - timedelta(minutes=15)
    return session_start, previous_completed


def _to_candle_df(rows: list[dict], prefix: str) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame(columns=["timestamp"])
    raw = pd.DataFrame(rows)
    raw = raw.rename(columns={"date": "timestamp"})
    raw["timestamp"] = pd.to_datetime(raw["timestamp"]).dt.tz_localize(None)
    raw = raw.sort_values("timestamp").reset_index(drop=True)
    return pd.DataFrame({
        "timestamp": raw["timestamp"],
        f"{prefix}_open_raw": raw["open"].astype(float),
        f"{prefix}_high_raw": raw["high"].astype(float),
        f"{prefix}_low_raw": raw["low"].astype(float),
        f"{prefix}_close": raw["close"].astype(float),
        f"{prefix}_volume": raw["volume"].astype(float),
    })


def _session_vwap(price: pd.Series, volume: pd.Series) -> pd.Series:
    if float(volume.fillna(0.0).sum()) <= 0.0:
        return price.expanding().mean()
    numerator = (price * volume).cumsum()
    denominator = volume.cumsum().replace(0, 1.0)
    return numerator / denominator


def fetch_stock_15min_frame(
    client: ZerodhaClient,
    stock_name: str,
    trade_date: date | None = None,
    include_options: bool = True,
) -> pd.DataFrame | None:
    """
    Fetch 15-min candles for spot + future. Optionally include ATM CE/PE.
    If include_options=True and options fail, falls back to spot+fut only.
    Returns merged frame in same format as MarketDataFetcher (spot_ltp, future_ltp, call_ltp, put_ltp, etc.).
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
    spot_df = _to_candle_df(spot_rows, "spot")
    if spot_df.empty:
        return None

    reference_spot = float(spot_df.iloc[0]["spot_open_raw"])
    try:
        symbols = resolver.resolve_for_date(reference_spot, effective_date)
    except Exception:
        return None

    fut_quote = client.quote([symbols.fut_symbol])
    fut_token = int(fut_quote[symbols.fut_symbol]["instrument_token"])
    fut_df = _to_candle_df(
        client.historical_data(fut_token, from_dt, to_dt, interval="15minute"),
        "future",
    )
    merged = spot_df.merge(fut_df, on="timestamp", how="inner")
    if merged.empty:
        return None

    spot_day_open = float(merged.iloc[0]["spot_open_raw"])
    fut_day_open = float(merged.iloc[0]["future_open_raw"])

    merged["spot_symbol"] = spot_symbol
    merged["future_symbol"] = symbols.fut_symbol
    merged["ce_symbol"] = symbols.ce_symbol
    merged["pe_symbol"] = symbols.pe_symbol
    merged["fut_symbol"] = symbols.fut_symbol
    merged["atm_strike"] = symbols.atm_strike
    merged["spot_open"] = spot_day_open
    merged["future_open"] = fut_day_open
    merged["spot_ltp"] = merged["spot_close"]
    merged["future_ltp"] = merged["future_close"]
    merged["spot_high"] = merged["spot_high_raw"]
    merged["spot_low"] = merged["spot_low_raw"]
    merged["future_high"] = merged["future_high_raw"]
    merged["future_low"] = merged["future_low_raw"]
    merged["spot_close"] = merged["spot_close"]
    merged["future_close"] = merged["future_close"]
    merged["spot_vwap"] = _session_vwap(merged["spot_close"], merged["spot_volume"])
    merged["future_vwap"] = _session_vwap(merged["future_close"], merged["future_volume"])

    if include_options:
        try:
            ce_quote = client.quote([symbols.ce_symbol])
            pe_quote = client.quote([symbols.pe_symbol])
            ce_token = int(ce_quote[symbols.ce_symbol]["instrument_token"])
            pe_token = int(pe_quote[symbols.pe_symbol]["instrument_token"])
            ce_df = _to_candle_df(
                client.historical_data(ce_token, from_dt, to_dt, interval="15minute"),
                "call",
            )
            pe_df = _to_candle_df(
                client.historical_data(pe_token, from_dt, to_dt, interval="15minute"),
                "put",
            )
            merged = merged.merge(ce_df[["timestamp", "call_close"]], on="timestamp", how="left")
            merged = merged.merge(pe_df[["timestamp", "put_close"]], on="timestamp", how="left")
            merged["call_ltp"] = merged["call_close"].fillna(0).astype(float)
            merged["put_ltp"] = merged["put_close"].fillna(0).astype(float)
            merged = merged.drop(columns=["call_close", "put_close"], errors="ignore")
        except Exception:
            merged["call_ltp"] = 0.0
            merged["put_ltp"] = 0.0
    else:
        merged["call_ltp"] = 0.0
        merged["put_ltp"] = 0.0

    merged["timestamp"] = pd.to_datetime(merged["timestamp"]).dt.strftime("%Y-%m-%dT%H:%M:%S")
    return merged
