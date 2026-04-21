"""Fetch 15-min spot + future (+ optional options) for F&O stocks."""

from __future__ import annotations

from datetime import date, datetime, timedelta

import pandas as pd

from intraday_engine.core.config import Settings
from intraday_engine.core.underlyings import get_underlying_config
from intraday_engine.fetch.instrument_resolver import InstrumentResolver
from intraday_engine.fetch.zerodha_client import ZerodhaClient


def _session_window_for_interval(
    trade_date: date | None,
    interval_minutes: int,
) -> tuple[datetime, datetime]:
    now = datetime.now()
    target_date = trade_date or now.date()
    session_start = datetime.combine(target_date, datetime.min.time()).replace(hour=9, minute=15)
    session_end = datetime.combine(target_date, datetime.min.time()).replace(hour=15, minute=30)

    if target_date < now.date():
        return session_start, session_end
    if target_date > now.date():
        return session_start, session_start - timedelta(minutes=interval_minutes)
    return session_start, min(session_end, now.replace(second=0, microsecond=0))


def _market_window_15min(trade_date: date | None) -> tuple[datetime, datetime]:
    """Session window for completed 15-min candles only."""
    return _session_window_for_interval(trade_date, 15)


def _market_window_30min(trade_date: date | None) -> tuple[datetime, datetime]:
    """Session window for completed 30-min candles only."""
    return _session_window_for_interval(trade_date, 30)


def _to_candle_df(rows: list[dict], prefix: str, include_oi: bool = False) -> pd.DataFrame:
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


def _session_vwap(price: pd.Series, volume: pd.Series) -> pd.Series:
    if float(volume.fillna(0.0).sum()) <= 0.0:
        return price.expanding().mean()
    numerator = (price * volume).cumsum()
    denominator = volume.cumsum().replace(0, 1.0)
    return numerator / denominator


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
    spot_df = _drop_incomplete_candles(_to_candle_df(spot_rows, "spot"), trade_date, 15)
    if spot_df.empty:
        return None

    reference_spot = float(spot_df.iloc[0]["spot_open_raw"])
    try:
        symbols = resolver.resolve_for_date(reference_spot, effective_date)
    except Exception:
        return None

    fut_quote = client.quote([symbols.fut_symbol])
    fut_token = int(fut_quote[symbols.fut_symbol]["instrument_token"])
    fut_df = _drop_incomplete_candles(
        _to_candle_df(
            client.historical_data(fut_token, from_dt, to_dt, interval="15minute", oi=True),
            "future",
            include_oi=True,
        ),
        trade_date,
        15,
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
            ce_merge_cols = ["timestamp", "call_close"] + (["call_oi"] if "call_oi" in ce_df.columns else [])
            pe_merge_cols = ["timestamp", "put_close"] + (["put_oi"] if "put_oi" in pe_df.columns else [])
            merged = merged.merge(ce_df[ce_merge_cols], on="timestamp", how="left")
            merged = merged.merge(pe_df[pe_merge_cols], on="timestamp", how="left")
            merged["call_ltp"] = merged["call_close"].fillna(0).astype(float)
            merged["put_ltp"] = merged["put_close"].fillna(0).astype(float)
            if "call_oi" not in merged.columns:
                merged["call_oi"] = 0.0
            if "put_oi" not in merged.columns:
                merged["put_oi"] = 0.0
            merged = merged.drop(columns=["call_close", "put_close"], errors="ignore")
        except Exception:
            merged["call_ltp"] = 0.0
            merged["put_ltp"] = 0.0
            merged["call_oi"] = 0.0
            merged["put_oi"] = 0.0
    else:
        merged["call_ltp"] = 0.0
        merged["put_ltp"] = 0.0
        merged["call_oi"] = 0.0
        merged["put_oi"] = 0.0

    merged["timestamp"] = pd.to_datetime(merged["timestamp"]).dt.strftime("%Y-%m-%dT%H:%M:%S")
    return merged


def fetch_stock_30min_frame(
    client: ZerodhaClient,
    stock_name: str,
    trade_date: date | None = None,
    include_options: bool = True,
) -> pd.DataFrame | None:
    """
    Build 30-min frame from 15-min candles (common source of truth).
    This avoids a second historical fetch path for 30m and keeps
    15m/30m analysis aligned.
    """
    m15 = fetch_stock_15min_frame(
        client,
        stock_name,
        trade_date=trade_date,
        include_options=include_options,
    )
    if m15 is None or m15.empty:
        return None
    df = m15.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    if df.empty:
        return None

    # NSE aligned 30m buckets anchored at 09:15.
    grouped = (
        df.set_index("timestamp")
        .resample("30min", origin="start_day", offset="15min")
        .agg(
            {
                "spot_open_raw": "first",
                "spot_high_raw": "max",
                "spot_low_raw": "min",
                "spot_close": "last",
                "spot_volume": "sum",
                "future_open_raw": "first",
                "future_high_raw": "max",
                "future_low_raw": "min",
                "future_close": "last",
                "future_volume": "sum",
                "future_oi": "last",
                "call_ltp": "last",
                "put_ltp": "last",
                "call_oi": "last",
                "put_oi": "last",
            }
        )
        .dropna(subset=["spot_open_raw", "spot_close", "future_open_raw", "future_close"])
        .reset_index()
    )
    if grouped.empty:
        return None

    for col in ("call_ltp", "put_ltp", "call_oi", "put_oi", "future_oi"):
        if col in grouped.columns:
            grouped[col] = grouped[col].fillna(0.0).astype(float)

    grouped["spot_day_open"] = float(grouped.iloc[0]["spot_open_raw"])
    grouped["fut_day_open"] = float(grouped.iloc[0]["future_open_raw"])
    grouped["spot_symbol"] = str(df.iloc[0].get("spot_symbol", ""))
    grouped["future_symbol"] = str(df.iloc[0].get("future_symbol", ""))
    grouped["ce_symbol"] = str(df.iloc[0].get("ce_symbol", ""))
    grouped["pe_symbol"] = str(df.iloc[0].get("pe_symbol", ""))
    grouped["fut_symbol"] = str(df.iloc[0].get("fut_symbol", ""))
    grouped["atm_strike"] = df.iloc[0].get("atm_strike")
    grouped["spot_open"] = grouped["spot_day_open"]
    grouped["future_open"] = grouped["fut_day_open"]
    grouped["spot_ltp"] = grouped["spot_close"]
    grouped["future_ltp"] = grouped["future_close"]
    grouped["spot_high"] = grouped["spot_high_raw"]
    grouped["spot_low"] = grouped["spot_low_raw"]
    grouped["future_high"] = grouped["future_high_raw"]
    grouped["future_low"] = grouped["future_low_raw"]
    grouped["spot_vwap"] = _session_vwap(grouped["spot_close"], grouped["spot_volume"])
    grouped["future_vwap"] = _session_vwap(grouped["future_close"], grouped["future_volume"])
    grouped["timestamp"] = pd.to_datetime(grouped["timestamp"]).dt.strftime("%Y-%m-%dT%H:%M:%S")

    ordered_cols = [
        "timestamp",
        "spot_open_raw",
        "spot_high_raw",
        "spot_low_raw",
        "spot_close",
        "spot_volume",
        "future_open_raw",
        "future_high_raw",
        "future_low_raw",
        "future_close",
        "future_volume",
        "future_oi",
        "spot_symbol",
        "future_symbol",
        "ce_symbol",
        "pe_symbol",
        "fut_symbol",
        "atm_strike",
        "spot_open",
        "future_open",
        "spot_ltp",
        "future_ltp",
        "spot_high",
        "spot_low",
        "future_high",
        "future_low",
        "spot_vwap",
        "future_vwap",
        "call_ltp",
        "put_ltp",
        "call_oi",
        "put_oi",
    ]
    return grouped[[c for c in ordered_cols if c in grouped.columns]]
