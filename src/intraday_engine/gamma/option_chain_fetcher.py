"""Fetch option chain data from Zerodha Kite.

Supports:
- Expiry-day only (for gamma blast): fetch_expiry_day_option_chain
- Nearest weekly expiry, 5-10 strikes (for huge move analysis): fetch_option_chain_near_spot
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import date, datetime, time as dt_time
from pathlib import Path
from typing import Any, List

import pandas as pd

from intraday_engine.core.config import Settings
from intraday_engine.fetch.zerodha_client import ZerodhaClient
from intraday_engine.storage.data_store import load_market_data
from intraday_engine.storage.layout import (
    asset_class_for_underlying,
    normalize_underlying,
    option_chain_day_path,
)
from intraday_engine.utils.nse_session import SESSION_CLOSE, SESSION_OPEN


def _parse_expiry(inst_expiry) -> date | None:
    """Parse instrument expiry to date."""
    if inst_expiry is None:
        return None
    if isinstance(inst_expiry, date):
        return inst_expiry
    if isinstance(inst_expiry, str):
        try:
            return datetime.strptime(inst_expiry[:10], "%Y-%m-%d").date()
        except (ValueError, TypeError):
            return None
    return None


def _expiry_matches(inst_expiry, trade_date: date) -> bool:
    """Compare instrument expiry (str or date) with trade_date."""
    exp = _parse_expiry(inst_expiry)
    return exp == trade_date if exp else False


@dataclass
class OptionStrikeData:
    tradingsymbol: str
    strike: int
    option_type: str  # CE or PE
    instrument_token: int
    oi: float
    volume: float
    ltp: float
    open_: float
    high: float
    low: float
    close: float


@dataclass
class OptionChainSnapshot:
    trade_date: date
    spot_price: float
    atm_strike: int
    strikes: List[OptionStrikeData]
    spot_volume: float
    timestamp: datetime
    expiry: date | None = None  # Option expiry date (for nearest-weekly fetch)


def _market_window(trade_date: date) -> tuple[datetime, datetime]:
    session_start = datetime.combine(trade_date, datetime.min.time()).replace(hour=9, minute=15)
    session_end = datetime.combine(trade_date, datetime.min.time()).replace(hour=15, minute=25)
    return session_start, session_end


# Option chain uses strikes in multiples of 100 (e.g. 23200, 23300) - not 50 (e.g. 23250)
def _oc_strike_step() -> int:
    from intraday_engine.core.tunables import get_int

    return get_int("option_chain", "OPTION_CHAIN_STRIKE_STEP", 100)


def fetch_expiry_day_option_chain(
    client: ZerodhaClient,
    settings: Settings,
    trade_date: date,
    spot_price: float,
    num_strikes_each_side: int = 3,
) -> OptionChainSnapshot | None:
    """
    Fetch option chain for expiry-day options (expiring on trade_date).
    Returns ATM ± num_strikes_each_side for both CE and PE.
    """
    instruments = client.nfo_instruments()
    underlying = settings.underlying
    step = _oc_strike_step()
    atm = int(round(spot_price / step) * step)

    # Options expiring ON trade_date
    expiry_opts = [
        r
        for r in instruments
        if r.get("name") == underlying
        and r.get("instrument_type") in ("CE", "PE")
        and _expiry_matches(r.get("expiry"), trade_date)
    ]

    if not expiry_opts:
        return None

    strikes_needed = set()
    for i in range(-num_strikes_each_side, num_strikes_each_side + 1):
        strikes_needed.add(atm + i * step)

    selected = []
    for r in expiry_opts:
        strike = int(float(r.get("strike", 0)))
        if strike in strikes_needed:
            selected.append(r)

    if not selected:
        return None

    symbols = [f"NFO:{r['tradingsymbol']}" for r in selected]
    from_dt, to_dt = _market_window(trade_date)

    quotes = client.quote(symbols)
    strike_data_list: List[OptionStrikeData] = []

    for r in selected:
        sym = f"NFO:{r['tradingsymbol']}"
        q = quotes.get(sym, {})
        strike = int(float(r.get("strike", 0)))
        opt_type = str(r.get("instrument_type", ""))
        token = int(r.get("instrument_token", 0) or q.get("instrument_token", 0))
        if token == 0:
            continue

        oi = float(q.get("oi", 0) or 0)
        volume = float(q.get("volume", 0) or 0)
        ltp = float(q.get("last_price", 0) or 0)
        o = float(q.get("ohlc", {}).get("open", 0) or 0)
        h = float(q.get("ohlc", {}).get("high", 0) or 0)
        l_ = float(q.get("ohlc", {}).get("low", 0) or 0)
        c = float(q.get("ohlc", {}).get("close", ltp) or ltp)

        strike_data_list.append(
            OptionStrikeData(
                tradingsymbol=r["tradingsymbol"],
                strike=strike,
                option_type=opt_type,
                instrument_token=token,
                oi=oi,
                volume=volume,
                ltp=ltp,
                open_=o,
                high=h,
                low=l_,
                close=c,
            )
        )

    spot_quote = client.quote([settings.spot_symbol])
    spot_vol = float(spot_quote.get(settings.spot_symbol, {}).get("volume", 0) or 0)

    return OptionChainSnapshot(
        trade_date=trade_date,
        spot_price=spot_price,
        atm_strike=atm,
        strikes=strike_data_list,
        spot_volume=spot_vol,
        timestamp=datetime.now(),
        expiry=trade_date,
    )


def fetch_expiry_day_historical_with_oi(
    client: ZerodhaClient,
    settings: Settings,
    trade_date: date,
    spot_price: float,
    interval: str = "5minute",
) -> pd.DataFrame | None:
    """
    Fetch 5-min candles with OI for ATM CE and PE on expiry day.
    Used for volume breakout and OI change analysis.
    """
    instruments = client.nfo_instruments()
    underlying = settings.underlying
    step = _oc_strike_step()
    atm = int(round(spot_price / step) * step)

    expiry_opts = [
        r
        for r in instruments
        if r.get("name") == underlying
        and r.get("instrument_type") in ("CE", "PE")
        and _expiry_matches(r.get("expiry"), trade_date)
        and int(float(r.get("strike", 0))) == atm
    ]

    if len(expiry_opts) < 2:
        return None

    from_dt, to_dt = _market_window(trade_date)
    rows_list = []

    for r in expiry_opts:
        token = int(r.get("instrument_token", 0))
        if token == 0:
            continue

        hist = client.historical_data(token, from_dt, to_dt, interval=interval, oi=True)
        opt_type = str(r.get("instrument_type", ""))
        for h in hist:
            rows_list.append({
                "timestamp": h["date"],
                "strike": atm,
                "option_type": opt_type,
                "open": h["open"],
                "high": h["high"],
                "low": h["low"],
                "close": h["close"],
                "volume": h["volume"],
                "oi": h.get("oi", 0),
            })

    if not rows_list:
        return None
    return pd.DataFrame(rows_list)


def _nearest_expiry_for_options(
    instruments: List[dict],
    underlying: str,
    trade_date: date,
) -> date | None:
    """Return nearest option expiry >= trade_date for the underlying."""
    opts = [
        r
        for r in instruments
        if r.get("name") == underlying
        and r.get("instrument_type") in ("CE", "PE")
        and _parse_expiry(r.get("expiry"))
    ]
    if not opts:
        return None
    expiries = sorted({_parse_expiry(r.get("expiry")) for r in opts if _parse_expiry(r.get("expiry"))})
    valid = [e for e in expiries if e and e >= trade_date]
    return valid[0] if valid else None


def resolve_option_chain_strike_counts(
    num_strikes: int | None = None,
) -> tuple[int, int]:
    """CE/PE strike counts from env, config.json, or legacy symmetric num_strikes."""
    from intraday_engine.core.tunables import get_int

    if num_strikes is not None:
        n = max(1, int(num_strikes))
        return n, n

    ce_env = os.getenv("OPTION_CHAIN_CE_STRIKES", "").strip()
    pe_env = os.getenv("OPTION_CHAIN_PE_STRIKES", "").strip()
    legacy_env = os.getenv("OPTION_STRIKES", "").strip()
    fallback = int(legacy_env) if legacy_env.isdigit() else 8

    ce = int(ce_env) if ce_env.isdigit() else get_int("option_chain", "OPTION_CHAIN_CE_STRIKES", fallback)
    pe = int(pe_env) if pe_env.isdigit() else get_int("option_chain", "OPTION_CHAIN_PE_STRIKES", fallback)
    return max(1, ce), max(1, pe)


def resolve_option_chain_itm_strike_counts() -> tuple[int, int]:
    """CE/PE ITM strike counts from env or tunables (default 4 each side)."""
    from intraday_engine.core.tunables import get_int

    ce_env = os.getenv("OPTION_CHAIN_CE_ITM_STRIKES", "").strip()
    pe_env = os.getenv("OPTION_CHAIN_PE_ITM_STRIKES", "").strip()
    ce = int(ce_env) if ce_env.isdigit() else get_int("option_chain", "OPTION_CHAIN_CE_ITM_STRIKES", 4)
    pe = int(pe_env) if pe_env.isdigit() else get_int("option_chain", "OPTION_CHAIN_PE_ITM_STRIKES", 4)
    return max(0, ce), max(0, pe)


def _select_expiry_options(
    instruments: List[dict],
    underlying: str,
    trade_date: date,
    *,
    use_expiry_day_only: bool,
) -> tuple[date | None, list[dict]]:
    if use_expiry_day_only:
        target_expiry = trade_date
        expiry_opts = [
            r
            for r in instruments
            if r.get("name") == underlying
            and r.get("instrument_type") in ("CE", "PE")
            and _expiry_matches(r.get("expiry"), trade_date)
        ]
    else:
        target_expiry = _nearest_expiry_for_options(instruments, underlying, trade_date)
        if not target_expiry:
            return None, []
        expiry_opts = [
            r
            for r in instruments
            if r.get("name") == underlying
            and r.get("instrument_type") in ("CE", "PE")
            and _parse_expiry(r.get("expiry")) == target_expiry
        ]
    return target_expiry, expiry_opts


def _quote_strike_rows(
    client: ZerodhaClient,
    settings: Settings,
    selected: list[dict],
    *,
    trade_date: date,
    spot_price: float,
    atm: int,
    target_expiry: date | None,
) -> OptionChainSnapshot | None:
    if not selected:
        return None

    symbols = [f"NFO:{r['tradingsymbol']}" for r in selected]
    quotes = client.quote(symbols)
    strike_data_list: List[OptionStrikeData] = []

    for r in selected:
        sym = f"NFO:{r['tradingsymbol']}"
        q = quotes.get(sym, {})
        strike = int(float(r.get("strike", 0)))
        opt_type = str(r.get("instrument_type", ""))
        token = int(r.get("instrument_token", 0) or q.get("instrument_token", 0))
        if token == 0:
            continue

        oi = float(q.get("oi", 0) or 0)
        volume = float(q.get("volume", 0) or 0)
        ltp = float(q.get("last_price", 0) or 0)
        o = float(q.get("ohlc", {}).get("open", 0) or 0)
        h = float(q.get("ohlc", {}).get("high", 0) or 0)
        l_ = float(q.get("ohlc", {}).get("low", 0) or 0)
        c = float(q.get("ohlc", {}).get("close", ltp) or ltp)

        strike_data_list.append(
            OptionStrikeData(
                tradingsymbol=r["tradingsymbol"],
                strike=strike,
                option_type=opt_type,
                instrument_token=token,
                oi=oi,
                volume=volume,
                ltp=ltp,
                open_=o,
                high=h,
                low=l_,
                close=c,
            )
        )

    if not strike_data_list:
        return None

    spot_quote = client.quote([settings.spot_symbol])
    spot_vol = float(spot_quote.get(settings.spot_symbol, {}).get("volume", 0) or 0)

    return OptionChainSnapshot(
        trade_date=trade_date,
        spot_price=spot_price,
        atm_strike=atm,
        strikes=strike_data_list,
        spot_volume=spot_vol,
        timestamp=datetime.now(),
        expiry=target_expiry,
    )


def fetch_option_chain_atm_ladder(
    client: ZerodhaClient,
    settings: Settings,
    trade_date: date,
    spot_price: float,
    *,
    num_ce_strikes: int = 8,
    num_pe_strikes: int = 8,
    num_ce_itm_strikes: int = 0,
    num_pe_itm_strikes: int = 0,
    use_expiry_day_only: bool = False,
) -> OptionChainSnapshot | None:
    """
    Fetch ATM ladder on 100-pt strikes:
      num_ce_strikes     CE going ATM → OTM (strikes above ATM)
      num_ce_itm_strikes CE going ATM → ITM (strikes below ATM, calls in-the-money)
      num_pe_strikes     PE going ATM → OTM (strikes below ATM)
      num_pe_itm_strikes PE going ATM → ITM (strikes above ATM, puts in-the-money)
    """
    instruments = client.nfo_instruments()
    underlying = settings.underlying
    step = _oc_strike_step()
    atm = int(round(spot_price / step) * step)

    target_expiry, expiry_opts = _select_expiry_options(
        instruments, underlying, trade_date, use_expiry_day_only=use_expiry_day_only,
    )
    if not expiry_opts:
        return None

    ce_otm = {atm + i * step for i in range(max(1, num_ce_strikes))}
    ce_itm = {atm - i * step for i in range(1, max(0, num_ce_itm_strikes) + 1)}
    ce_strikes = ce_otm | ce_itm
    pe_otm = {atm - i * step for i in range(max(1, num_pe_strikes))}
    pe_itm = {atm + i * step for i in range(1, max(0, num_pe_itm_strikes) + 1)}
    pe_strikes = pe_otm | pe_itm

    selected: list[dict] = []
    for r in expiry_opts:
        strike = int(float(r.get("strike", 0)))
        if strike % step != 0:
            continue
        opt = str(r.get("instrument_type", ""))
        if opt == "CE" and strike in ce_strikes:
            selected.append(r)
        elif opt == "PE" and strike in pe_strikes:
            selected.append(r)

    return _quote_strike_rows(
        client,
        settings,
        selected,
        trade_date=trade_date,
        spot_price=spot_price,
        atm=atm,
        target_expiry=target_expiry,
    )


def fetch_option_chain_near_spot(
    client: ZerodhaClient,
    settings: Settings,
    trade_date: date,
    spot_price: float,
    num_strikes_each_side: int = 5,
    use_expiry_day_only: bool = False,
) -> OptionChainSnapshot | None:
    """
    Fetch option chain for ATM ± num_strikes_each_side (5-10 strikes).
    Uses nearest weekly expiry by default; set use_expiry_day_only=True to match
    fetch_expiry_day_option_chain (expiry day only).
    """
    ce, pe = resolve_option_chain_strike_counts(num_strikes_each_side)
    ce_itm, pe_itm = resolve_option_chain_itm_strike_counts()
    return fetch_option_chain_atm_ladder(
        client,
        settings,
        trade_date,
        spot_price,
        num_ce_strikes=ce,
        num_pe_strikes=pe,
        num_ce_itm_strikes=ce_itm,
        num_pe_itm_strikes=pe_itm,
        use_expiry_day_only=use_expiry_day_only,
    )


def save_option_chain_snapshot(
    snapshot: OptionChainSnapshot,
    data_dir: Path,
    underlying: str = "NIFTY",
) -> Path:
    """
    Append flattened option-chain rows to a day-partitioned CSV file.
    """
    path = option_chain_day_path(data_dir, snapshot.trade_date)
    underlying_name = normalize_underlying(underlying)
    rows = pd.DataFrame(
        [
            {
                "timestamp": snapshot.timestamp.isoformat(),
                "trade_date": snapshot.trade_date.isoformat(),
                "underlying": underlying_name,
                "spot_price": snapshot.spot_price,
                "atm_strike": snapshot.atm_strike,
                "expiry": snapshot.expiry.isoformat() if snapshot.expiry else None,
                "spot_volume": snapshot.spot_volume,
                "tradingsymbol": s.tradingsymbol,
                "strike": s.strike,
                "option_type": s.option_type,
                "oi": s.oi,
                "volume": s.volume,
                "ltp": s.ltp,
                "open": s.open_,
                "high": s.high,
                "low": s.low,
                "close": s.close,
            }
            for s in snapshot.strikes
        ]
    )
    from intraday_engine.storage.backend import write_to_db

    if write_to_db():
        from intraday_engine.storage import db as db_store

        db_store.append_option_chain_rows(rows)
        return path
    existing = pd.read_csv(path) if path.exists() else pd.DataFrame()
    combined = pd.concat([existing, rows], ignore_index=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(path, index=False)
    return path


def remove_option_chain_snapshots(
    data_dir: Path,
    trade_date: date,
    underlying: str,
    *,
    raw_timestamps: list[str] | None = None,
    capture_prefixes: list[str] | None = None,
) -> int:
    """Drop CSV rows for exact timestamps or ISO prefixes (e.g. 2026-07-30T11:17)."""
    from intraday_engine.storage.backend import write_to_db

    if write_to_db():
        from intraday_engine.storage import db as db_store

        ts_set = {str(t) for t in raw_timestamps} if raw_timestamps else None
        return db_store.delete_option_chain_rows(
            trade_date,
            underlying,
            timestamps=ts_set,
            timestamp_prefixes=capture_prefixes,
        )

    path = option_chain_day_path(data_dir, trade_date)
    if not path.exists():
        return 0
    df = pd.read_csv(path)
    if df.empty:
        return 0
    underlying_name = normalize_underlying(underlying)
    if "underlying" in df.columns:
        mask_u = df["underlying"].astype(str).str.upper() == underlying_name
    else:
        mask_u = pd.Series(True, index=df.index)
    drop = pd.Series(False, index=df.index)
    if raw_timestamps:
        ts_set = {str(t) for t in raw_timestamps}
        drop |= mask_u & df["timestamp"].astype(str).isin(ts_set)
    if capture_prefixes:
        for prefix in capture_prefixes:
            drop |= mask_u & df["timestamp"].astype(str).str.startswith(prefix)
    removed = int(drop.sum())
    if removed:
        df.loc[~drop].to_csv(path, index=False)
    return removed


def remove_snapshots_in_bar_buckets(
    data_dir: Path,
    trade_date: date,
    underlying: str,
    bar_time_labels: list[str],
) -> tuple[int, list[str]]:
    """Drop snapshots that bucket into requested 5-min bars (e.g. 11:17 → 11:15)."""
    path = option_chain_day_path(data_dir, trade_date)
    if not path.exists():
        return 0, []
    df = pd.read_csv(path)
    if df.empty:
        return 0, []
    underlying_name = normalize_underlying(underlying)
    if "underlying" not in df.columns:
        return 0, []
    udf = df[df["underlying"].astype(str).str.upper() == underlying_name]
    if udf.empty:
        return 0, []

    bar_keys: set[str] = set()
    for label in bar_time_labels:
        hour, minute = _parse_bar_time_label(label)
        bar_open = datetime.combine(trade_date, dt_time(hour, minute))
        bar_keys.add(_snapshot_ts_key(bar_open))

    drop_ts: set[str] = set()
    for ts in udf["timestamp"].astype(str).unique():
        raw_dt = _parse_snapshot_datetime(ts, trade_date)
        floored = _floor_to_5min_bar(raw_dt, trade_date)
        if floored is not None and _snapshot_ts_key(floored) in bar_keys:
            drop_ts.add(ts)

    if not drop_ts:
        return 0, []
    drop_mask = (
        (df["underlying"].astype(str).str.upper() == underlying_name)
        & df["timestamp"].astype(str).isin(drop_ts)
    )
    removed = int(drop_mask.sum())
    df.loc[~drop_mask].to_csv(path, index=False)
    return removed, sorted(drop_ts)


def _capture_clock_for_underlying(
    existing: pd.DataFrame,
    underlying: str,
    trade_date: date,
    bar_open: datetime | None = None,
) -> tuple[int, int]:
    """Second + microsecond offset matching live scraper rows for this underlying."""
    from intraday_engine.jobs.option_chain_scraper import DEFAULT_SCRAPER_UNDERLYINGS

    underlying_name = normalize_underlying(underlying)
    udf = existing[existing["underlying"].astype(str).str.upper() == underlying_name]
    if not udf.empty:
        if bar_open is not None:
            udf = udf.copy()
            udf["_dt"] = pd.to_datetime(udf["timestamp"])
            udf["_delta"] = (udf["_dt"] - pd.Timestamp(bar_open)).abs()
            ref_ts = str(udf.nsmallest(1, "_delta").iloc[0]["timestamp"])
        else:
            ref_ts = str(udf["timestamp"].iloc[0])
        dt = _parse_snapshot_datetime(ref_ts, trade_date)
        return dt.second, dt.microsecond
    try:
        idx = list(DEFAULT_SCRAPER_UNDERLYINGS).index(underlying_name)
    except ValueError:
        idx = 0
    return idx + 1, 110766


def _capture_timestamp_for_bar(
    trade_date: date,
    bar_open: datetime,
    underlying: str,
    existing: pd.DataFrame,
) -> str:
    """Live-style capture timestamp: bar open + scraper second/microsecond offset."""
    second, micro = _capture_clock_for_underlying(existing, underlying, trade_date, bar_open)
    capture_dt = datetime.combine(
        trade_date,
        dt_time(bar_open.hour, bar_open.minute, second, micro),
    )
    return capture_dt.isoformat()


def _parse_bar_time_label(label: str) -> tuple[int, int]:
    parts = str(label).strip().split(":")
    if len(parts) != 2:
        raise ValueError(f"Invalid bar time {label!r} — expected HH:MM")
    hour, minute = int(parts[0]), int(parts[1])
    if not (0 <= hour <= 23 and 0 <= minute <= 59):
        raise ValueError(f"Invalid bar time {label!r}")
    return hour, minute


def _spot_at_bar_from_index(
    data_dir: Path,
    trade_date: date,
    underlying: str,
    bar_ts: datetime,
) -> tuple[float, int]:
    key = _snapshot_ts_key(bar_ts)
    step = _oc_strike_step()
    for snap in _index_analysis_snapshots(data_dir, trade_date, underlying):
        if snap["timestamp"] == key:
            spot = float(snap.get("spot_price") or 0)
            if spot > 0:
                return spot, int(round(spot / step) * step)
    asset_class = asset_class_for_underlying(underlying)
    df = load_market_data(data_dir, trade_date, asset_class, underlying)
    if df.empty or "timestamp" not in df.columns:
        return 0.0, 0
    target = pd.Timestamp(bar_ts)
    df = df.copy()
    df["_ts"] = pd.to_datetime(df["timestamp"])
    df["_delta"] = (df["_ts"] - target).abs()
    row = df.nsmallest(1, "_delta").iloc[0]
    spot = float(row.get("spot_ltp") or row.get("spot_close") or 0)
    if spot <= 0:
        return 0.0, 0
    return spot, int(round(spot / step) * step)


def _reference_legs_from_csv(
    df: pd.DataFrame,
    underlying: str,
    *,
    strike_min: int | None = None,
    strike_max: int | None = None,
) -> pd.DataFrame:
    underlying_name = normalize_underlying(underlying)
    udf = df[df["underlying"].astype(str).str.upper() == underlying_name].copy()
    if udf.empty:
        return udf
    if strike_min is not None:
        udf = udf[udf["strike"].astype(int) >= strike_min]
    if strike_max is not None:
        udf = udf[udf["strike"].astype(int) <= strike_max]
    if udf.empty:
        return udf
    udf = udf.sort_values("timestamp")
    return udf.drop_duplicates(subset=["tradingsymbol"], keep="last").reset_index(drop=True)


def _reference_legs_from_nfo(
    instruments: list[dict],
    underlying: str,
    trade_date: date,
    *,
    strike_min: int,
    strike_max: int,
) -> pd.DataFrame:
    """Build reference legs from NFO instrument master when CSV has no rows yet."""
    _, expiry_opts = _select_expiry_options(
        instruments, underlying, trade_date, use_expiry_day_only=False,
    )
    rows: list[dict[str, Any]] = []
    for r in expiry_opts:
        strike = int(float(r.get("strike", 0) or 0))
        if strike < strike_min or strike > strike_max:
            continue
        opt = str(r.get("instrument_type", ""))
        if opt not in ("CE", "PE"):
            continue
        sym = str(r.get("tradingsymbol", ""))
        if not sym:
            continue
        rows.append({
            "tradingsymbol": sym,
            "strike": strike,
            "option_type": opt,
            "expiry": r.get("expiry"),
        })
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).drop_duplicates(subset=["tradingsymbol"]).reset_index(drop=True)


def _infer_strike_range_from_csv(
    df: pd.DataFrame,
    underlying: str,
) -> tuple[int | None, int | None]:
    underlying_name = normalize_underlying(underlying)
    udf = df[df["underlying"].astype(str).str.upper() == underlying_name]
    if udf.empty:
        return None, None
    strikes = udf["strike"].astype(int)
    return int(strikes.min()), int(strikes.max())


def _infer_strike_range_from_index(
    data_dir: Path,
    trade_date: date,
    underlying: str,
) -> tuple[int | None, int | None]:
    """ATM ± configured ladder width from Index Analysis spot range."""
    asset_class = asset_class_for_underlying(underlying)
    df = load_market_data(data_dir, trade_date, asset_class, underlying)
    if df.empty:
        return None, None
    underlying_name = normalize_underlying(underlying)
    if "underlying" in df.columns:
        df = df[df["underlying"].astype(str).str.upper() == underlying_name]
    if df.empty:
        return None, None
    spot_col = "spot_ltp" if "spot_ltp" in df.columns else "spot_close"
    if spot_col not in df.columns:
        return None, None
    spots = pd.to_numeric(df[spot_col], errors="coerce").dropna()
    if spots.empty:
        return None, None
    step = _oc_strike_step()
    ce_count, pe_count = resolve_option_chain_strike_counts()
    lo_spot, hi_spot = float(spots.min()), float(spots.max())
    atm_lo = int(round(lo_spot / step) * step) - pe_count * step
    atm_hi = int(round(hi_spot / step) * step) + ce_count * step
    return atm_lo, atm_hi


def session_bar_time_labels() -> list[str]:
    """All NSE 5-min bar open times: 09:15 … 15:25."""
    open_min = _session_open_minutes()
    last_bar = _session_close_minutes() - 5
    labels: list[str] = []
    m = open_min
    while m <= last_bar:
        labels.append(f"{m // 60:02d}:{m % 60:02d}")
        m += 5
    return labels


def chain_bar_labels_present(
    data_dir: Path,
    trade_date: date,
    underlying: str,
) -> set[str]:
    """HH:MM labels that already have option-chain captures for this underlying."""
    path = option_chain_day_path(data_dir, trade_date)
    if not path.exists():
        return set()
    df = pd.read_csv(path)
    if df.empty or "timestamp" not in df.columns:
        return set()
    underlying_name = normalize_underlying(underlying)
    udf = df[df["underlying"].astype(str).str.upper() == underlying_name]
    if udf.empty:
        return set()
    present: set[str] = set()
    for ts in udf["timestamp"].astype(str).unique():
        raw_dt = _parse_snapshot_datetime(ts, trade_date)
        bar_ts = _floor_to_5min_bar(raw_dt, trade_date)
        if bar_ts is not None:
            present.add(bar_ts.strftime("%H:%M"))
    return present


def backfill_option_chain_full_day(
    data_dir: Path,
    trade_date: date,
    underlying: str,
    *,
    skip_existing: bool = True,
    replace_existing: bool = False,
    strike_min: int | None = None,
    strike_max: int | None = None,
    remove_capture_prefixes: list[str] | None = None,
) -> dict[str, Any]:
    """
    Backfill every 5-min session bar (09:15–15:25) from Kite historical OI.

    By default only fills bars missing from option_chain.csv. Use replace_existing=True
    to drop and rebuild all session bars for the underlying.
    """
    all_bars = session_bar_time_labels()
    present = chain_bar_labels_present(data_dir, trade_date, underlying)

    if replace_existing:
        bars_to_fill = all_bars
    elif skip_existing:
        bars_to_fill = [b for b in all_bars if b not in present]
    else:
        bars_to_fill = all_bars

    if not bars_to_fill:
        return {
            "status": "ok",
            "trade_date": trade_date.isoformat(),
            "underlying": normalize_underlying(underlying),
            "bars_filled": [],
            "rows_added": 0,
            "message": "All session bars already present in option_chain.csv",
            "session_bars_total": len(all_bars),
            "session_bars_present": len(present),
        }

    path = option_chain_day_path(data_dir, trade_date)
    existing = pd.read_csv(path) if path.exists() else pd.DataFrame()

    resolved_min = strike_min
    resolved_max = strike_max
    if resolved_min is None or resolved_max is None:
        csv_min, csv_max = _infer_strike_range_from_csv(existing, underlying)
        if resolved_min is None:
            resolved_min = csv_min
        if resolved_max is None:
            resolved_max = csv_max
    if resolved_min is None or resolved_max is None:
        idx_min, idx_max = _infer_strike_range_from_index(data_dir, trade_date, underlying)
        if resolved_min is None:
            resolved_min = idx_min
        if resolved_max is None:
            resolved_max = idx_max
    if resolved_min is None or resolved_max is None:
        return {
            "status": "failed",
            "reason": "Could not infer strike range — pass --strike-min/--strike-max",
        }

    result = backfill_option_chain_historical(
        data_dir,
        trade_date,
        underlying,
        bars_to_fill,
        strike_min=resolved_min,
        strike_max=resolved_max,
        remove_capture_prefixes=remove_capture_prefixes,
        replace_existing=replace_existing,
    )
    return {
        **result,
        "mode": "full_day",
        "session_bars_total": len(all_bars),
        "session_bars_present_before": len(present),
        "session_bars_requested": len(bars_to_fill),
        "strike_min": resolved_min,
        "strike_max": resolved_max,
        "skip_existing": skip_existing and not replace_existing,
    }


def _candle_datetime(candle: dict[str, Any]) -> datetime:
    cdt = candle.get("date")
    if isinstance(cdt, str):
        cdt = pd.to_datetime(cdt).to_pydatetime()
    if cdt.tzinfo is not None:
        cdt = cdt.replace(tzinfo=None)
    return cdt


def _fetch_spot_bars(
    client: ZerodhaClient,
    settings: Settings,
    trade_date: date,
) -> dict[str, dict[str, Any]]:
    """NIFTY 5-min candles keyed by bar open (2026-07-30T09:15:00)."""
    quote = client.quote([settings.spot_symbol])
    token = int(quote.get(settings.spot_symbol, {}).get("instrument_token", 0) or 0)
    if token <= 0:
        return {}
    from_dt, to_dt = _market_window(trade_date)
    hist = client.historical_data(token, from_dt, to_dt, interval="5minute", oi=False)
    by_bar: dict[str, dict[str, Any]] = {}
    for candle in hist:
        bar_ts = _floor_to_5min_bar(_candle_datetime(candle), trade_date)
        if bar_ts is not None:
            by_bar[_snapshot_ts_key(bar_ts)] = candle
    return by_bar


def _spot_at_bar_from_kite(
    spot_bars: dict[str, dict[str, Any]],
    bar_ts: datetime,
) -> tuple[float, int]:
    """
    Match live scheduler: spot = index last_price at ~bar open ≈ 5-min candle open.
    """
    candle = spot_bars.get(_snapshot_ts_key(bar_ts))
    if not candle:
        return 0.0, 0
    spot = float(candle.get("open") or 0)
    if spot <= 0:
        return 0.0, 0
    step = _oc_strike_step()
    return spot, int(round(spot / step) * step)


def _ladder_strikes(atm: int, num_ce: int, num_pe: int, step: int) -> list[tuple[int, str]]:
    legs: list[tuple[int, str]] = []
    for i in range(max(1, num_ce)):
        legs.append((atm + i * step, "CE"))
    for i in range(max(1, num_pe)):
        legs.append((atm - i * step, "PE"))
    return legs


def _nfo_leg_lookup(
    instruments: list[dict],
    underlying: str,
    expiry: date | None,
) -> dict[tuple[int, str], dict]:
    lookup: dict[tuple[int, str], dict] = {}
    step = _oc_strike_step()
    for r in instruments:
        if str(r.get("name", "")).upper() != normalize_underlying(underlying):
            continue
        if expiry is not None and _parse_expiry(r.get("expiry")) != expiry:
            continue
        opt = str(r.get("instrument_type", ""))
        if opt not in ("CE", "PE"):
            continue
        strike = int(float(r.get("strike", 0) or 0))
        if step > 0 and strike % step != 0:
            continue
        lookup[(strike, opt)] = r
    return lookup


def _prev_day_close(client: ZerodhaClient, token: int, trade_date: date) -> float:
    from datetime import timedelta

    hist = client.historical_data(
        token,
        trade_date - timedelta(days=10),
        trade_date,
        interval="day",
        oi=False,
    )
    prior: list[dict] = []
    for candle in hist:
        cdt = _candle_datetime(candle)
        if cdt.date() < trade_date:
            prior.append(candle)
    if not prior:
        return 0.0
    return float(prior[-1].get("close") or 0)


def _process_option_bar_series(
    hist: list[dict[str, Any]],
    trade_date: date,
    prev_close: float,
) -> dict[str, dict[str, float]]:
    """
    Per 5-min bar, build scheduler-aligned fields from Kite historical candles:
    ltp = bar open, volume = cumulative day volume, OHLC = intraday-to-bar + prev close.
    """
    ordered: list[tuple[str, dict[str, Any]]] = []
    for candle in hist:
        bar_ts = _floor_to_5min_bar(_candle_datetime(candle), trade_date)
        if bar_ts is None:
            continue
        ordered.append((_snapshot_ts_key(bar_ts), candle))
    ordered.sort(key=lambda x: x[0])

    out: dict[str, dict[str, float]] = {}
    cum_vol = 0.0
    day_open: float | None = None
    day_high = float("-inf")
    day_low = float("inf")
    for bar_key, candle in ordered:
        cum_vol += float(candle.get("volume") or 0)
        o = float(candle.get("open") or 0)
        h = float(candle.get("high") or 0)
        l_ = float(candle.get("low") or 0)
        if day_open is None:
            day_open = o
        day_high = max(day_high, h)
        day_low = min(day_low, l_)
        out[bar_key] = {
            "oi": float(candle.get("oi") or 0),
            "ltp": o,
            "volume": cum_vol,
            "open": day_open,
            "high": day_high,
            "low": day_low,
            "close": prev_close,
        }
    return out


def _scheduler_style_bar_row(
    *,
    trade_date: date,
    underlying: str,
    capture_timestamp: str,
    spot_price: float,
    atm_strike: int,
    expiry: date | None,
    tradingsymbol: str,
    strike: int,
    option_type: str,
    expiry_raw: Any,
    oi: float,
    volume: float,
    ltp: float,
    open_: float,
    high: float,
    low: float,
    close: float,
) -> dict[str, Any]:
    return {
        "timestamp": capture_timestamp,
        "trade_date": trade_date.isoformat(),
        "underlying": normalize_underlying(underlying),
        "spot_price": spot_price,
        "atm_strike": atm_strike,
        "expiry": expiry.isoformat() if expiry else expiry_raw,
        "spot_volume": 0.0,
        "tradingsymbol": tradingsymbol,
        "strike": strike,
        "option_type": option_type,
        "oi": oi,
        "volume": volume,
        "ltp": ltp,
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
    }


def _historical_bar_row(
    candle: dict[str, Any],
    *,
    trade_date: date,
    underlying: str,
    capture_timestamp: str,
    spot_price: float,
    atm_strike: int,
    expiry: date | None,
    leg: pd.Series,
) -> dict[str, Any]:
    return {
        "timestamp": capture_timestamp,
        "trade_date": trade_date.isoformat(),
        "underlying": normalize_underlying(underlying),
        "spot_price": spot_price,
        "atm_strike": atm_strike,
        "expiry": expiry.isoformat() if expiry else leg.get("expiry"),
        "spot_volume": 0.0,
        "tradingsymbol": str(leg.get("tradingsymbol", "")),
        "strike": int(leg.get("strike", 0) or 0),
        "option_type": str(leg.get("option_type", "")),
        "oi": float(candle.get("oi", 0) or 0),
        "volume": float(candle.get("volume", 0) or 0),
        "ltp": float(candle.get("close", 0) or 0),
        "open": float(candle.get("open", 0) or 0),
        "high": float(candle.get("high", 0) or 0),
        "low": float(candle.get("low", 0) or 0),
        "close": float(candle.get("close", 0) or 0),
    }


def backfill_option_chain_historical(
    data_dir: Path,
    trade_date: date,
    underlying: str,
    bar_time_labels: list[str],
    *,
    strike_min: int | None = None,
    strike_max: int | None = None,
    remove_capture_prefixes: list[str] | None = None,
    replace_existing: bool = True,
) -> dict[str, Any]:
    """
    Reconstruct 5-min option-chain snapshots from Kite historical data.

    Matches live scheduler semantics: spot = index 5-min open, ltp = option bar open,
    volume = cumulative session volume, OHLC = intraday-to-bar + prior-day close.
    """
    from intraday_engine.core.config import Settings
    from intraday_engine.fetch.zerodha_client import ZerodhaClient

    path = option_chain_day_path(data_dir, trade_date)
    existing = pd.read_csv(path) if path.exists() else pd.DataFrame()

    removed = 0
    removed_ts: list[str] = []
    if replace_existing:
        bucket_removed, bucket_ts = remove_snapshots_in_bar_buckets(
            data_dir, trade_date, underlying, bar_time_labels,
        )
        removed += bucket_removed
        removed_ts.extend(bucket_ts)
    if remove_capture_prefixes:
        removed += remove_option_chain_snapshots(
            data_dir, trade_date, underlying, capture_prefixes=remove_capture_prefixes,
        )
    existing = pd.read_csv(path) if path.exists() else pd.DataFrame()

    settings = Settings.from_env(underlying=underlying)
    client = ZerodhaClient(settings)
    instruments = client.nfo_instruments()
    sym_to_token: dict[str, int] = {}
    for r in instruments:
        sym = str(r.get("tradingsymbol", ""))
        if sym:
            sym_to_token[sym] = int(r.get("instrument_token", 0) or 0)

    ref_legs = _reference_legs_from_csv(
        existing, underlying, strike_min=strike_min, strike_max=strike_max,
    )
    expiry_raw = ref_legs.iloc[0].get("expiry") if not ref_legs.empty else None
    target_expiry = _parse_expiry(expiry_raw) if expiry_raw else None
    if target_expiry is None:
        target_expiry = _nearest_expiry_for_options(instruments, underlying, trade_date)
    if target_expiry is None:
        return {"status": "failed", "reason": "Could not resolve option expiry"}

    leg_lookup = _nfo_leg_lookup(instruments, underlying, target_expiry)
    if not leg_lookup:
        return {"status": "failed", "reason": "No NFO legs for expiry"}

    ce_count, pe_count = resolve_option_chain_strike_counts()
    step = _oc_strike_step()
    spot_bars = _fetch_spot_bars(client, settings, trade_date)
    if not spot_bars:
        return {"status": "failed", "reason": "Could not fetch index 5-min candles from Kite"}

    from_dt, to_dt = _market_window(trade_date)
    bar_times: list[datetime] = []
    capture_ts_by_bar: dict[str, str] = {}
    for label in bar_time_labels:
        hour, minute = _parse_bar_time_label(label)
        bar_ts = datetime.combine(trade_date, dt_time(hour, minute))
        if _floor_to_5min_bar(bar_ts, trade_date) is None:
            return {"status": "failed", "reason": f"{label} outside session hours"}
        bar_times.append(bar_ts)
        capture_ts_by_bar[_snapshot_ts_key(bar_ts)] = _capture_timestamp_for_bar(
            trade_date, bar_ts, underlying, existing,
        )

    def _strike_in_range(strike: int) -> bool:
        if strike_min is not None and strike < strike_min:
            return False
        if strike_max is not None and strike > strike_max:
            return False
        return True

    needed: dict[str, dict] = {}
    for bar_ts in bar_times:
        spot, atm = _spot_at_bar_from_kite(spot_bars, bar_ts)
        if spot <= 0 or atm <= 0:
            continue
        for strike, opt in _ladder_strikes(atm, ce_count, pe_count, step):
            if not _strike_in_range(strike):
                continue
            inst = leg_lookup.get((strike, opt))
            if not inst:
                continue
            sym = str(inst.get("tradingsymbol", ""))
            if sym:
                needed[sym] = inst

    if not needed:
        return {"status": "failed", "reason": "No ladder legs resolved for requested bars"}

    option_by_sym: dict[str, dict[str, dict[str, float]]] = {}
    errors: list[str] = []
    for sym, inst in needed.items():
        token = sym_to_token.get(sym, 0)
        if token <= 0:
            errors.append(f"{sym}: instrument token not found")
            continue
        try:
            hist = client.historical_data(token, from_dt, to_dt, interval="5minute", oi=True)
        except Exception as exc:
            errors.append(f"{sym}: {exc}")
            continue
        prev_close = _prev_day_close(client, token, trade_date)
        option_by_sym[sym] = _process_option_bar_series(hist, trade_date, prev_close)

    rows_out: list[dict[str, Any]] = []
    for bar_ts in bar_times:
        bar_key = _snapshot_ts_key(bar_ts)
        capture_ts = capture_ts_by_bar[bar_key]
        spot, atm = _spot_at_bar_from_kite(spot_bars, bar_ts)
        if spot <= 0 or atm <= 0:
            errors.append(f"{capture_ts}: missing Kite index candle for bar")
            continue
        for strike, opt in _ladder_strikes(atm, ce_count, pe_count, step):
            if not _strike_in_range(strike):
                continue
            inst = leg_lookup.get((strike, opt))
            if not inst:
                continue
            sym = str(inst.get("tradingsymbol", ""))
            bar_data = option_by_sym.get(sym, {}).get(bar_key)
            if not bar_data:
                errors.append(f"{sym} @ {bar_key}: no historical candle")
                continue
            rows_out.append(
                _scheduler_style_bar_row(
                    trade_date=trade_date,
                    underlying=underlying,
                    capture_timestamp=capture_ts,
                    spot_price=spot,
                    atm_strike=atm,
                    expiry=target_expiry,
                    tradingsymbol=sym,
                    strike=strike,
                    option_type=opt,
                    expiry_raw=inst.get("expiry"),
                    oi=bar_data["oi"],
                    volume=bar_data["volume"],
                    ltp=bar_data["ltp"],
                    open_=bar_data["open"],
                    high=bar_data["high"],
                    low=bar_data["low"],
                    close=bar_data["close"],
                )
            )

    if not rows_out:
        return {
            "status": "failed",
            "reason": "No rows built from historical data",
            "errors": errors,
            "removed_rows": removed,
        }

    new_df = pd.DataFrame(rows_out)
    combined = pd.concat([existing, new_df], ignore_index=True)
    combined = combined.sort_values(["timestamp", "underlying", "strike", "option_type"]).reset_index(drop=True)
    combined.to_csv(path, index=False)

    filled_labels = list(bar_time_labels)
    return {
        "status": "ok",
        "trade_date": trade_date.isoformat(),
        "underlying": normalize_underlying(underlying),
        "bars_filled": filled_labels,
        "capture_timestamps": list(capture_ts_by_bar.values()),
        "rows_added": len(rows_out),
        "legs_per_bar": len(ref_legs),
        "removed_rows": removed,
        "removed_timestamps": removed_ts,
        "errors": errors[:20],
        "path": str(path),
    }


def _snapshot_ts_minutes(dt: datetime) -> int:
    return dt.hour * 60 + dt.minute


def _session_open_minutes() -> int:
    return SESSION_OPEN.hour * 60 + SESSION_OPEN.minute


def _session_close_minutes() -> int:
    return SESSION_CLOSE.hour * 60 + SESSION_CLOSE.minute


def _parse_snapshot_datetime(ts: str | datetime, trade_date: date) -> datetime:
    if isinstance(ts, datetime):
        dt = ts
    else:
        dt = pd.to_datetime(ts).to_pydatetime()
    if dt.tzinfo is not None:
        dt = dt.replace(tzinfo=None)
    if dt.date() != trade_date:
        dt = datetime.combine(trade_date, dt.time())
    return dt


def _floor_to_5min_bar(dt: datetime, trade_date: date) -> datetime | None:
    """Map capture time to NSE 5-min bar open (09:15, 09:20, … 15:25)."""
    mins = _snapshot_ts_minutes(dt)
    open_min = _session_open_minutes()
    close_min = _session_close_minutes()
    last_bar_open = close_min - 5  # 15:25 bar covers until 15:30 close
    if mins < open_min or mins > close_min:
        return None
    capped = min(mins, last_bar_open)
    floored = open_min + ((capped - open_min) // 5) * 5
    floored = min(floored, last_bar_open)
    return datetime.combine(trade_date, dt_time(floored // 60, floored % 60))


def _snapshot_ts_key(dt: datetime) -> str:
    return dt.replace(microsecond=0).isoformat(sep="T")


def _strike_row_from_csv(row: pd.Series) -> dict[str, Any]:
    return {
        "tradingsymbol": str(row.get("tradingsymbol", "")),
        "strike": int(row.get("strike", 0) or 0),
        "option_type": str(row.get("option_type", "")),
        "oi": float(row.get("oi", 0) or 0),
        "volume": float(row.get("volume", 0) or 0),
        "ltp": float(row.get("ltp", 0) or 0),
        "open": float(row.get("open", 0) or 0),
        "high": float(row.get("high", 0) or 0),
        "low": float(row.get("low", 0) or 0),
        "close": float(row.get("close", 0) or 0),
    }


def _snapshot_from_chain_group(group: pd.DataFrame, trade_date: date, bar_ts: datetime) -> dict:
    first = group.iloc[0]
    return {
        "timestamp": _snapshot_ts_key(bar_ts),
        "trade_date": str(first.get("trade_date") or trade_date.isoformat()),
        "spot_price": float(first.get("spot_price", 0) or 0),
        "atm_strike": int(first.get("atm_strike", 0) or 0),
        "expiry": first.get("expiry"),
        "spot_volume": float(first.get("spot_volume", 0) or 0),
        "source": "option_chain",
        "strikes": [_strike_row_from_csv(row) for _, row in group.iterrows()],
    }


def _round_to_chain_strike(strike: int) -> int:
    """Align a strike to option-chain ladder step (100 for NIFTY)."""
    step = _oc_strike_step()
    if strike <= 0 or step <= 0:
        return strike
    return int(round(strike / step) * step)


def _snapshot_from_index_row(row: pd.Series, trade_date: date) -> dict | None:
    atm_raw = int(row.get("atm_strike") or 0)
    if atm_raw <= 0:
        return None
    # Index pipeline uses 50-pt ATM symbols; option chain ladder is 100-pt only.
    atm = _round_to_chain_strike(atm_raw)
    dt = _parse_snapshot_datetime(str(row["timestamp"]), trade_date)
    bar_ts = _floor_to_5min_bar(dt, trade_date)
    if bar_ts is None:
        return None
    spot = float(row.get("spot_ltp") or row.get("spot_close") or 0)
    call_oi = float(row.get("call_oi") or 0)
    put_oi = float(row.get("put_oi") or 0)
    empty_leg = {"volume": 0.0, "ltp": 0.0, "open": 0.0, "high": 0.0, "low": 0.0, "close": 0.0}
    return {
        "timestamp": _snapshot_ts_key(bar_ts),
        "trade_date": trade_date.isoformat(),
        "spot_price": spot,
        "atm_strike": atm,
        "expiry": None,
        "spot_volume": float(row.get("spot_volume") or 0),
        "source": "index_analysis",
        "strikes": [
            {"tradingsymbol": "", "strike": atm, "option_type": "CE", "oi": call_oi, **empty_leg},
            {"tradingsymbol": "", "strike": atm, "option_type": "PE", "oi": put_oi, **empty_leg},
        ],
    }


def _bucket_chain_snapshots(records: list[dict], trade_date: date) -> list[dict]:
    """Keep one snapshot per 5-min bar (latest capture wins). Session hours only."""
    buckets: dict[str, tuple[datetime, dict]] = {}
    for snap in records:
        raw_dt = _parse_snapshot_datetime(snap["timestamp"], trade_date)
        bar_ts = _floor_to_5min_bar(raw_dt, trade_date)
        if bar_ts is None:
            continue
        key = _snapshot_ts_key(bar_ts)
        prev = buckets.get(key)
        if prev is None or raw_dt >= prev[0]:
            buckets[key] = (raw_dt, snap)
    out: list[dict] = []
    for key in sorted(buckets):
        raw_dt, snap = buckets[key]
        bar_ts = _parse_snapshot_datetime(key, trade_date)
        out.append({
            **snap,
            "timestamp": key,
            "source": snap.get("source") or "option_chain",
            "_capture_time": raw_dt.isoformat(sep="T"),
        })
    return out


def fill_missing_session_bars(
    snapshots: list[dict],
    trade_date: date,
) -> tuple[list[dict], list[str]]:
    """
    Insert placeholder snapshots for every 5-min bar between first and last capture.

    Without this, a missed option-chain capture removes the column entirely (timeline
    compresses 11:15 → 11:25 with no 11:20 label).
    """
    if len(snapshots) < 2:
        return snapshots, []
    by_key = {s["timestamp"]: s for s in snapshots}
    first = _parse_snapshot_datetime(snapshots[0]["timestamp"], trade_date)
    last = _parse_snapshot_datetime(snapshots[-1]["timestamp"], trade_date)
    start_m = first.hour * 60 + first.minute
    end_m = last.hour * 60 + last.minute
    filled: list[dict] = []
    missing_labels: list[str] = []
    last_spot: float | None = None
    last_atm: int | None = None
    m = start_m
    while m <= end_m:
        bar_ts = datetime.combine(trade_date, dt_time(m // 60, m % 60))
        key = _snapshot_ts_key(bar_ts)
        if key in by_key:
            snap = by_key[key]
            filled.append(snap)
            sp = snap.get("spot_price")
            atm = snap.get("atm_strike")
            if sp is not None and float(sp) > 0:
                last_spot = float(sp)
            if atm is not None and int(atm) > 0:
                last_atm = int(atm)
        else:
            missing_labels.append(bar_ts.strftime("%H:%M"))
            filled.append({
                "timestamp": key,
                "trade_date": trade_date.isoformat(),
                "spot_price": last_spot,
                "atm_strike": last_atm,
                "expiry": None,
                "spot_volume": 0,
                "source": "missing",
                "strikes": [],
            })
        m += 5
    return filled, missing_labels


def _index_analysis_snapshots(
    data_dir: Path,
    trade_date: date,
    underlying: str,
) -> list[dict]:
    asset_class = asset_class_for_underlying(underlying)
    df = load_market_data(data_dir, trade_date, asset_class, underlying)
    if df.empty or "timestamp" not in df.columns:
        return []
    underlying_name = normalize_underlying(underlying)
    if "underlying" in df.columns:
        df = df[df["underlying"].astype(str).str.upper() == underlying_name]
    if df.empty:
        return []
    by_bar: dict[str, dict] = {}
    for _, row in df.iterrows():
        snap = _snapshot_from_index_row(row, trade_date)
        if snap:
            by_bar[snap["timestamp"]] = snap
    return [by_bar[k] for k in sorted(by_bar)]


def _option_chain_records_from_df(df: pd.DataFrame, trade_date: date, underlying: str) -> list[dict]:
    underlying_name = normalize_underlying(underlying)
    if "underlying" in df.columns:
        df = df[df["underlying"].astype(str) == underlying_name]
    if df.empty:
        return []
    records: list[dict] = []
    for ts, group in df.groupby("timestamp", sort=True):
        first = group.iloc[0]
        records.append(
            {
                "timestamp": str(ts),
                "trade_date": str(first.get("trade_date") or trade_date.isoformat()),
                "spot_price": float(first.get("spot_price", 0) or 0),
                "atm_strike": int(first.get("atm_strike", 0) or 0),
                "expiry": first.get("expiry"),
                "spot_volume": float(first.get("spot_volume", 0) or 0),
                "source": "option_chain",
                "strikes": [_strike_row_from_csv(row) for _, row in group.iterrows()],
            }
        )
    return records


def load_option_chain_snapshots(
    data_dir: Path,
    trade_date: date,
    underlying: str = "NIFTY",
) -> List[dict]:
    """Load all option chain snapshots for a date from the flattened CSV."""
    from intraday_engine.storage.backend import write_to_db

    if write_to_db():
        from intraday_engine.storage.data_cache import get_cached, set_cached
        from intraday_engine.storage import db as db_store

        cache_key = f"option_chain:{trade_date.isoformat()}:{underlying}"
        cached = get_cached(cache_key)
        if cached is not None:
            return cached
        df = db_store.load_option_chain_rows(trade_date, underlying)
        if df.empty:
            return []
        records = _option_chain_records_from_df(df, trade_date, underlying)
        set_cached(cache_key, records)
        return records

    path = option_chain_day_path(data_dir, trade_date)
    if not path.exists():
        return []
    df = pd.read_csv(path)
    return _option_chain_records_from_df(df, trade_date, underlying)


def load_session_option_chain_snapshots(
    data_dir: Path,
    trade_date: date,
    underlying: str = "NIFTY",
    *,
    backfill_from_index: bool = True,
) -> tuple[list[dict], dict[str, Any]]:
    """
    Session-scoped snapshots for OI heatmap: 5-min buckets, optional Index_Analysis
    backfill for bars missing from option_chain.csv (ATM CE/PE only; strikes rounded to chain step).
    """
    raw_chain = load_option_chain_snapshots(data_dir, trade_date, underlying)
    chain = _bucket_chain_snapshots(raw_chain, trade_date)
    chain_by_ts = {s["timestamp"]: s for s in chain}

    index_snaps = _index_analysis_snapshots(data_dir, trade_date, underlying) if backfill_from_index else []
    index_by_ts = {s["timestamp"]: s for s in index_snaps}

    all_ts = sorted(set(chain_by_ts) | set(index_by_ts))
    merged: list[dict] = []
    for ts in sorted(set(chain_by_ts) | set(index_by_ts)):
        if ts in chain_by_ts:
            merged.append(chain_by_ts[ts])
        else:
            merged.append(index_by_ts[ts])

    # Drop out-of-session bars (e.g. post-sleep captures bucketed incorrectly).
    merged = _bucket_chain_snapshots(
        [{**s, "timestamp": s.get("_capture_time") or s["timestamp"]} for s in merged],
        trade_date,
    )

    def _label(ts: str | None) -> str | None:
        if not ts:
            return None
        dt = _parse_snapshot_datetime(ts, trade_date)
        return dt.strftime("%H:%M")

    all_ts = [s["timestamp"] for s in merged]
    chain_ts_sorted = sorted(chain_by_ts)
    coverage = {
        "data_first": _label(all_ts[0]) if all_ts else None,
        "data_last": _label(all_ts[-1]) if all_ts else None,
        "chain_first": _label(chain_ts_sorted[0]) if chain_ts_sorted else None,
        "chain_last": _label(chain_ts_sorted[-1]) if chain_ts_sorted else None,
        "total_bars": len(merged),
        "chain_bars": len(chain_by_ts),
        "backfilled_bars": max(0, len(merged) - len(chain_by_ts)),
        "index_bars": len(index_by_ts),
    }
    return merged, coverage
