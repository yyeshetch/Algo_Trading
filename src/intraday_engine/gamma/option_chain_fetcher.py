"""Fetch option chain data from Zerodha Kite.

Supports:
- Expiry-day only (for gamma blast): fetch_expiry_day_option_chain
- Nearest weekly expiry, 5-10 strikes (for huge move analysis): fetch_option_chain_near_spot
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import List

import pandas as pd

from intraday_engine.core.config import Settings
from intraday_engine.fetch.zerodha_client import ZerodhaClient
from intraday_engine.storage.layout import normalize_underlying, option_chain_day_path


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
    instruments = client.nfo_instruments()
    underlying = settings.underlying
    step = _oc_strike_step()
    atm = int(round(spot_price / step) * step)

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
            return None
        expiry_opts = [
            r
            for r in instruments
            if r.get("name") == underlying
            and r.get("instrument_type") in ("CE", "PE")
            and _parse_expiry(r.get("expiry")) == target_expiry
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
        expiry=target_expiry,
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
    existing = pd.read_csv(path) if path.exists() else pd.DataFrame()
    combined = pd.concat([existing, rows], ignore_index=True)
    combined.to_csv(path, index=False)
    return path


def load_option_chain_snapshots(
    data_dir: Path,
    trade_date: date,
    underlying: str = "NIFTY",
) -> List[dict]:
    """Load all option chain snapshots for a date from the flattened CSV."""
    path = option_chain_day_path(data_dir, trade_date)
    if not path.exists():
        return []
    df = pd.read_csv(path)
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
                "strikes": [
                    {
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
                    for _, row in group.iterrows()
                ],
            }
        )
    return records
