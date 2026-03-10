"""Fetch option chain data for expiry day from Zerodha Kite."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from typing import List

import pandas as pd

from intraday_engine.core.config import Settings
from intraday_engine.fetch.zerodha_client import ZerodhaClient


def _expiry_matches(inst_expiry, trade_date: date) -> bool:
    """Compare instrument expiry (str or date) with trade_date."""
    if inst_expiry is None:
        return False
    if isinstance(inst_expiry, date):
        return inst_expiry == trade_date
    if isinstance(inst_expiry, str):
        try:
            return datetime.strptime(inst_expiry[:10], "%Y-%m-%d").date() == trade_date
        except (ValueError, TypeError):
            return False
    return False


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


def _market_window(trade_date: date) -> tuple[datetime, datetime]:
    session_start = datetime.combine(trade_date, datetime.min.time()).replace(hour=9, minute=15)
    session_end = datetime.combine(trade_date, datetime.min.time()).replace(hour=15, minute=25)
    return session_start, session_end


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
    step = settings.option_strike_step
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
    step = settings.option_strike_step
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
