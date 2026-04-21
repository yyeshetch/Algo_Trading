"""Fetch EOD (end-of-day) daily data for FnO stocks and compute market indicators."""

from __future__ import annotations

import logging
from datetime import date, datetime, timedelta
from typing import Any

from intraday_engine.core.config import Settings
from intraday_engine.core.underlyings import LIQUID_FNO_STOCKS, get_underlying_config
from intraday_engine.fetch.instrument_resolver import InstrumentResolver
from intraday_engine.fetch.zerodha_client import ZerodhaClient

# Liquid FnO symbols to scrape (indices + Chartink-selected stocks)
EOD_INDEX_SYMBOLS: tuple[str, ...] = (
    "NIFTY",
    "BANKNIFTY",
    "FINNIFTY",
    "MIDCPNIFTY",
    "NIFTYNXT50",
)
EOD_SYMBOLS: tuple[str, ...] = (
    *EOD_INDEX_SYMBOLS,
    *LIQUID_FNO_STOCKS,
)
# Override NSE spot symbols for symbols that differ from NFO name (add as needed)
EOD_SPOT_OVERRIDES: dict[str, str] = {}

logger = logging.getLogger(__name__)


def _daily_window(trade_date: date | None, days_back: int = 5) -> tuple[datetime, datetime]:
    """Return (from_dt, to_dt) for daily candles. to_dt = end of trade_date."""
    d = trade_date or date.today()
    to_dt = datetime.combine(d, datetime.min.time()).replace(hour=15, minute=30)
    from_dt = datetime.combine(d - timedelta(days=days_back), datetime.min.time()).replace(hour=9, minute=15)
    return from_dt, to_dt


def fetch_stock_eod_data(
    client: ZerodhaClient,
    stock_name: str,
    trade_date: date | None = None,
) -> dict[str, Any] | None:
    """
    Fetch daily (EOD) candles for spot, future, ATM CE, ATM PE (with OI).
    Returns dict with last day's metrics and prev day for change, or None.
    """
    uc = get_underlying_config(stock_name)
    resolver = InstrumentResolver(client, Settings.from_env(underlying=stock_name))
    from_dt, to_dt = _daily_window(trade_date)

    spot_symbol = EOD_SPOT_OVERRIDES.get(stock_name.upper(), uc.spot_symbol)
    try:
        spot_quote = client.quote([spot_symbol])
        if not spot_quote or spot_symbol not in spot_quote:
            return None
        spot_token = int(spot_quote[spot_symbol]["instrument_token"])
    except Exception:
        return None

    spot_rows = client.historical_data(spot_token, from_dt, to_dt, interval="day")
    if not spot_rows or len(spot_rows) < 1:
        return None

    last_row = spot_rows[-1]
    last_spot = float(last_row.get("close", 0) or 0)
    last_date_val = last_row.get("date")
    try:
        if isinstance(last_date_val, str):
            effective_date = datetime.strptime(last_date_val[:10], "%Y-%m-%d").date()
        elif hasattr(last_date_val, "date"):
            effective_date = getattr(last_date_val, "date")()
        else:
            effective_date = from_dt.date()
    except Exception:
        effective_date = from_dt.date()
    try:
        symbols = resolver.resolve_for_date(last_spot, effective_date)
    except Exception:
        return None

    deriv_quotes = client.quote([symbols.fut_symbol, symbols.ce_symbol, symbols.pe_symbol])
    fut_token = int(deriv_quotes[symbols.fut_symbol]["instrument_token"])
    ce_token = int(deriv_quotes[symbols.ce_symbol]["instrument_token"])
    pe_token = int(deriv_quotes[symbols.pe_symbol]["instrument_token"])

    fut_rows = client.historical_data(fut_token, from_dt, to_dt, interval="day", oi=True)
    ce_rows = client.historical_data(ce_token, from_dt, to_dt, interval="day", oi=True)
    pe_rows = client.historical_data(pe_token, from_dt, to_dt, interval="day", oi=True)

    if not fut_rows or not ce_rows or not pe_rows:
        return None

    spot_last = spot_rows[-1]
    spot_prev = spot_rows[-2] if len(spot_rows) >= 2 else spot_last
    fut_last = fut_rows[-1]
    fut_prev = fut_rows[-2] if len(fut_rows) >= 2 else fut_last
    ce_last = ce_rows[-1]
    ce_prev = ce_rows[-2] if len(ce_rows) >= 2 else ce_last
    pe_last = pe_rows[-1]
    pe_prev = pe_rows[-2] if len(pe_rows) >= 2 else pe_last

    spot_open = float(spot_last.get("open", 0) or 0)
    spot_close = float(spot_last.get("close", 0) or 0)
    spot_volume = int(float(spot_last.get("volume", 0) or 0))
    spot_prev_close = float(spot_prev.get("close", 0) or 0)

    fut_open = float(fut_last.get("open", 0) or 0)
    fut_close = float(fut_last.get("close", 0) or 0)
    fut_oi = int(float(fut_last.get("oi", 0) or 0))
    fut_oi_prev = int(float(fut_prev.get("oi", 0) or 0))
    fut_volume = int(float(fut_last.get("volume", 0) or 0))

    ce_open = float(ce_last.get("open", 0) or 0)
    ce_close = float(ce_last.get("close", 0) or 0)
    ce_oi = int(float(ce_last.get("oi", 0) or 0))
    ce_oi_prev = int(float(ce_prev.get("oi", 0) or 0))

    pe_open = float(pe_last.get("open", 0) or 0)
    pe_close = float(pe_last.get("close", 0) or 0)
    pe_oi = int(float(pe_last.get("oi", 0) or 0))
    pe_oi_prev = int(float(pe_prev.get("oi", 0) or 0))

    spot_chg = ((spot_close - spot_open) / spot_open * 100) if spot_open else 0
    spot_chg_prev = ((spot_close - spot_prev_close) / spot_prev_close * 100) if spot_prev_close else 0
    ce_prem_chg = ((ce_close - ce_open) / ce_open * 100) if ce_open else 0
    pe_prem_chg = ((pe_close - pe_open) / pe_open * 100) if pe_open else 0
    fut_chg = ((fut_close - fut_open) / fut_open * 100) if fut_open else 0

    ce_oi_chg = ((ce_oi - ce_oi_prev) / ce_oi_prev * 100) if ce_oi_prev else 0
    pe_oi_chg = ((pe_oi - pe_oi_prev) / pe_oi_prev * 100) if pe_oi_prev else 0
    fut_oi_chg = ((fut_oi - fut_oi_prev) / fut_oi_prev * 100) if fut_oi_prev else 0

    pcr = round(pe_oi / ce_oi, 2) if ce_oi else 0
    fut_vs_spot = ((fut_close - spot_close) / spot_close * 100) if spot_close else 0

    oi_buildup = _compute_oi_buildup(
        price_chg_pct=spot_chg_prev,
        oi_chg_pct=fut_oi_chg,
    )

    fno_indicator = _compute_fno_indicator(
        spot_chg=spot_chg,
        fut_oi_chg=fut_oi_chg,
        ce_oi_chg=ce_oi_chg,
        pe_oi_chg=pe_oi_chg,
        pcr=pcr,
        ce_prem_chg=ce_prem_chg,
        pe_prem_chg=pe_prem_chg,
    )

    return {
        "stock": stock_name,
        "date": spot_last.get("date", ""),
        "spot_open": round(spot_open, 2),
        "spot_close": round(spot_close, 2),
        "spot_change_pct": round(spot_chg, 2),
        "spot_change_prev_pct": round(spot_chg_prev, 2),
        "spot_volume": spot_volume,
        "fut_open": round(fut_open, 2),
        "fut_close": round(fut_close, 2),
        "fut_change_pct": round(fut_chg, 2),
        "fut_oi": fut_oi,
        "fut_oi_change_pct": round(fut_oi_chg, 2),
        "fut_volume": fut_volume,
        "fut_vs_spot_pct": round(fut_vs_spot, 2),
        "ce_oi": ce_oi,
        "ce_oi_change_pct": round(ce_oi_chg, 2),
        "pe_oi": pe_oi,
        "pe_oi_change_pct": round(pe_oi_chg, 2),
        "pcr": pcr,
        "ce_open": round(ce_open, 2),
        "ce_close": round(ce_close, 2),
        "ce_premium_change_pct": round(ce_prem_chg, 2),
        "pe_open": round(pe_open, 2),
        "pe_close": round(pe_close, 2),
        "pe_premium_change_pct": round(pe_prem_chg, 2),
        "fno_indicator": fno_indicator,
        "oi_buildup": oi_buildup,
    }


def _compute_oi_buildup(price_chg_pct: float, oi_chg_pct: float) -> str | None:
    """
    OI-based buildup indicators (Price vs Prev Close, OI vs Prev Day):
    - Long Buildup: Price UP + OI UP
    - Short Buildup: Price DOWN + OI UP
    - Long Unwinding: Price DOWN + OI DOWN
    - Short Covering: Price UP + OI DOWN
    """
    thresh = 0.3
    oi_thresh = 1.0
    price_up = price_chg_pct > thresh
    price_down = price_chg_pct < -thresh
    oi_up = oi_chg_pct > oi_thresh
    oi_down = oi_chg_pct < -oi_thresh
    if price_up and oi_up:
        return "LONG_BUILDUP"
    if price_down and oi_up:
        return "SHORT_BUILDUP"
    if price_down and oi_down:
        return "LONG_UNWINDING"
    if price_up and oi_down:
        return "SHORT_COVERING"
    return None


def _compute_fno_indicator(
    spot_chg: float,
    fut_oi_chg: float,
    ce_oi_chg: float,
    pe_oi_chg: float,
    pcr: float,
    ce_prem_chg: float,
    pe_prem_chg: float,
) -> str:
    """
    Compute FnO indicator (BULLISH/BEARISH/NEUTRAL) from spot, fut OI, CE/PE OI chg,
    PCR, CE/PE premium change.
    """
    score = 0.0
    if spot_chg > 0.3:
        score += 0.2
    elif spot_chg < -0.3:
        score -= 0.2
    if fut_oi_chg > 2:
        score += 0.1
    elif fut_oi_chg < -2:
        score -= 0.1
    if ce_oi_chg > 2:
        score += 0.15
    elif ce_oi_chg < -2:
        score -= 0.15
    if pe_oi_chg > 2:
        score -= 0.15
    elif pe_oi_chg < -2:
        score += 0.15
    if pcr < 0.7:
        score += 0.15
    elif pcr > 1.3:
        score -= 0.15
    if ce_prem_chg > 2:
        score += 0.15
    elif ce_prem_chg < -2:
        score -= 0.15
    if pe_prem_chg > 2:
        score -= 0.15
    elif pe_prem_chg < -2:
        score += 0.15
    if score >= 0.25:
        return "BULLISH"
    if score <= -0.25:
        return "BEARISH"
    return "NEUTRAL"


def run_eod_scan(
    trade_date: date | None = None,
    stock_limit: int | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    """
    Fetch EOD indicators for index underlyings plus the allowed liquid FnO stocks.
    Returns list of indicator dicts, sorted by spot_volume descending (most liquid first).
    """
    settings = Settings.from_env(underlying="NIFTY")
    client = ZerodhaClient(settings)
    stock_names = [*EOD_INDEX_SYMBOLS, *client.fno_stock_names()]
    if stock_limit:
        stock_names = stock_names[:stock_limit]

    results: list[dict[str, Any]] = []
    failed: list[dict[str, str]] = []
    for name in stock_names:
        try:
            data = fetch_stock_eod_data(client, name, trade_date)
            if data:
                results.append(data)
            else:
                failed.append({"symbol": name, "reason": "No data returned"})
        except Exception as e:
            logger.warning("EOD skip %s: %s", name, e)
            failed.append({"symbol": name, "reason": str(e)})
            continue

    if failed:
        logger.info("EOD: %d succeeded, %d failed: %s", len(results), len(failed), [f["symbol"] for f in failed])
    results.sort(key=lambda x: x.get("spot_volume", 0) or 0, reverse=True)
    return results, failed
