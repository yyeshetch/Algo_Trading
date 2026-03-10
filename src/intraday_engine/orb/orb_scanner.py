"""15-min Opening Range Breakout (ORB) and Pinbar scanner."""

from __future__ import annotations

import json
import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

from intraday_engine.core.config import Settings
from intraday_engine.fetch.zerodha_client import ZerodhaClient

logger = logging.getLogger(__name__)

ORB_VARIATION_PCT = 0.2
QUOTE_BATCH_SIZE = 500
MAX_WORKERS = 5
RATE_LIMIT = 0.25  # sec between requests (~4/sec, Kite allows 3)

_rate_lock = threading.Lock()
_rate_last = 0.0


def _rate_limit() -> None:
    global _rate_last
    with _rate_lock:
        elapsed = time.monotonic() - _rate_last
        if elapsed < RATE_LIMIT:
            time.sleep(RATE_LIMIT - elapsed)
        _rate_last = time.monotonic()


def _last_completed_15min(trade_date: date) -> datetime:
    now = datetime.now()
    if now.date() != trade_date:
        return datetime.combine(trade_date, datetime.min.time()).replace(hour=15, minute=30)
    block = (now.minute // 15) * 15
    boundary = now.replace(minute=block, second=0, microsecond=0)
    return boundary - timedelta(minutes=15)


def _orb_ranges_path(data_dir: Path, trade_date: date) -> Path:
    return data_dir / f"orb_ranges_{trade_date.isoformat()}.json"


def _load_orb_ranges(data_dir: Path, trade_date: date) -> dict[str, dict[str, float]] | None:
    path = _orb_ranges_path(data_dir, trade_date)
    if not path.exists():
        return None
    try:
        with path.open(encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.warning("Could not load OR ranges: %s", e)
        return None


def _save_orb_ranges(data_dir: Path, trade_date: date, ranges: dict[str, dict[str, float]]) -> None:
    path = _orb_ranges_path(data_dir, trade_date)
    data_dir.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(ranges, f, indent=0)


def _fetch_candles_15min(
    client: ZerodhaClient,
    token: int,
    trade_date: date,
) -> list[dict[str, Any]] | None:
    """Fetch all 15-min candles from 9:15 to last completed. One call per stock."""
    from_dt = datetime.combine(trade_date, datetime.min.time()).replace(hour=9, minute=15)
    to_dt = _last_completed_15min(trade_date)
    if to_dt < from_dt:
        return None
    _rate_limit()
    try:
        return client.historical_data(token, from_dt, to_dt, interval="15minute")
    except Exception as e:
        logger.debug("Candles fetch failed token=%s: %s", token, e)
        return None


def _fetch_all_candles_parallel(
    client: ZerodhaClient,
    symbols: list[tuple[str, str]],
    trade_date: date,
) -> dict[str, list[dict[str, Any]]]:
    """Fetch 15-min candles for all stocks in parallel. Returns {stock_name: candles}."""
    quotes = _fetch_quotes_bulk(client, [s[0] for s in symbols])
    results: dict[str, list[dict[str, Any]]] = {}

    def _fetch_one(args: tuple[str, str]) -> tuple[str, list[dict[str, Any]] | None]:
        nse_sym, stock_name = args
        if nse_sym not in quotes:
            return stock_name, None
        token = quotes[nse_sym].get("instrument_token")
        if not token:
            return stock_name, None
        candles = _fetch_candles_15min(client, int(token), trade_date)
        return stock_name, candles

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = {ex.submit(_fetch_one, (nse, name)): name for nse, name in symbols}
        for fut in as_completed(futures):
            try:
                stock_name, candles = fut.result()
                if candles:
                    results[stock_name] = candles
            except Exception as e:
                logger.debug("Worker failed: %s", e)
    return results


def _fetch_quotes_bulk(client: ZerodhaClient, symbols: list[str]) -> dict[str, dict[str, Any]]:
    quotes: dict[str, dict[str, Any]] = {}
    for i in range(0, len(symbols), QUOTE_BATCH_SIZE):
        batch = symbols[i : i + QUOTE_BATCH_SIZE]
        try:
            batch_quotes = client.quote(batch)
            quotes.update(batch_quotes)
        except Exception as e:
            logger.warning("Quote batch failed: %s", e)
    return quotes


def _is_bullish_pinbar(o: float, h: float, l: float, c: float) -> bool:
    """Long lower wick, small body, close near top. Rejection at low."""
    if h <= l:
        return False
    body = abs(c - o)
    range_ = h - l
    lower_wick = min(o, c) - l
    upper_wick = h - max(o, c)
    if range_ <= 0 or body <= 0:
        return False
    return (
        lower_wick >= 2 * body
        and body <= 0.35 * range_
        and upper_wick <= 0.5 * lower_wick
    )


def _is_bearish_pinbar(o: float, h: float, l: float, c: float) -> bool:
    """Long upper wick, small body, close near bottom. Rejection at high."""
    if h <= l:
        return False
    body = abs(c - o)
    range_ = h - l
    upper_wick = h - max(o, c)
    lower_wick = min(o, c) - l
    if range_ <= 0 or body <= 0:
        return False
    return (
        upper_wick >= 2 * body
        and body <= 0.35 * range_
        and lower_wick <= 0.5 * upper_wick
    )


def run_orb_scan(
    trade_date: date | None = None,
    stock_limit: int | None = None,
    use_cached_or: bool = True,
) -> list[dict[str, Any]]:
    """ORB scan. One historical call per stock (OR + latest close). Parallel fetch."""
    settings = Settings.from_env(underlying="NIFTY")
    client = ZerodhaClient(settings)
    data_dir = settings.data_dir
    trade_date = trade_date or date.today()

    stock_names = client.fno_stock_names()
    if stock_limit:
        stock_names = stock_names[:stock_limit]
    symbols = [(f"NSE:{n}", n) for n in stock_names]
    nse_symbols = [s[0] for s in symbols]

    # 1. OR ranges: load from cache or extract from candles
    ranges = _load_orb_ranges(data_dir, trade_date) if use_cached_or else None
    all_candles: dict[str, list[dict]] = {}

    if ranges is None or not ranges:
        logger.info("Fetching 15-min candles (parallel) for %s stocks...", len(symbols))
        all_candles = _fetch_all_candles_parallel(client, symbols, trade_date)
        ranges = {}
        for stock_name, candles in all_candles.items():
            if candles:
                first = candles[0]
                ranges[stock_name] = {
                    "or_high": float(first.get("high", 0) or 0),
                    "or_low": float(first.get("low", 0) or 0),
                }
        if ranges:
            _save_orb_ranges(data_dir, trade_date, ranges)
        else:
            return []
    else:
        logger.info("Fetching latest candles (parallel) for %s stocks...", len(symbols))
        all_candles = _fetch_all_candles_parallel(client, symbols, trade_date)

    quotes = _fetch_quotes_bulk(client, nse_symbols)
    variation = ORB_VARIATION_PCT / 100.0
    signals: list[dict[str, Any]] = []
    now_str = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")

    for stock_name, candles in all_candles.items():
        if not candles or stock_name not in ranges:
            continue
        or_high = ranges[stock_name]["or_high"]
        or_low = ranges[stock_name]["or_low"]
        last_candle = candles[-1]
        latest_close = float(last_candle.get("close", 0) or 0)
        if latest_close <= 0:
            continue

        o = float(last_candle.get("open", 0) or 0)
        h = float(last_candle.get("high", 0) or 0)
        l_ = float(last_candle.get("low", 0) or 0)
        c = latest_close

        upper_break = or_high * (1 - variation)
        lower_break = or_low * (1 + variation)
        signal = "NO_TRADE"
        if latest_close >= upper_break:
            signal = "BUY"
        elif latest_close <= lower_break:
            signal = "SELL"

        nse_sym = f"NSE:{stock_name}"
        oi = float(quotes.get(nse_sym, {}).get("oi", 0) or 0)
        oi_day_high = float(quotes.get(nse_sym, {}).get("oi_day_high", 0) or 0)

        signals.append({
            "stock": stock_name,
            "signal": signal,
            "price": round(latest_close, 2),
            "or_high": round(or_high, 2),
            "or_low": round(or_low, 2),
            "upper_break": round(upper_break, 2),
            "lower_break": round(lower_break, 2),
            "timestamp": now_str,
            "oi": oi,
            "oi_day_high": oi_day_high,
            "reasons": f"ORB: OR {or_low:.1f}-{or_high:.1f}" + (
                f" | close ≥ {upper_break:.1f}" if signal == "BUY" else
                f" | close ≤ {lower_break:.1f}" if signal == "SELL" else ""
            ),
        })

    buy = [s for s in signals if s["signal"] == "BUY"]
    sell = [s for s in signals if s["signal"] == "SELL"]
    logger.info("ORB: %s BUY, %s SELL", len(buy), len(sell))
    return signals


def run_pinbar_scan(
    trade_date: date | None = None,
    stock_limit: int | None = None,
) -> list[dict[str, Any]]:
    """Find bullish and bearish pinbars on last 15-min candle."""
    settings = Settings.from_env(underlying="NIFTY")
    client = ZerodhaClient(settings)
    trade_date = trade_date or date.today()

    stock_names = client.fno_stock_names()
    if stock_limit:
        stock_names = stock_names[:stock_limit]
    symbols = [(f"NSE:{n}", n) for n in stock_names]

    logger.info("Fetching 15-min candles for pinbar scan (%s stocks)...", len(symbols))
    all_candles = _fetch_all_candles_parallel(client, symbols, trade_date)
    quotes = _fetch_quotes_bulk(client, [s[0] for s in symbols])

    bullish: list[dict[str, Any]] = []
    bearish: list[dict[str, Any]] = []
    now_str = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")

    for stock_name, candles in all_candles.items():
        if not candles:
            continue
        last = candles[-1]
        o = float(last.get("open", 0) or 0)
        h = float(last.get("high", 0) or 0)
        l_ = float(last.get("low", 0) or 0)
        c = float(last.get("close", 0) or 0)
        if h <= l_ or c <= 0:
            continue

        nse_sym = f"NSE:{stock_name}"
        oi = float(quotes.get(nse_sym, {}).get("oi", 0) or 0)

        if _is_bullish_pinbar(o, h, l_, c):
            bullish.append({
                "stock": stock_name,
                "signal": "BUY",
                "pattern": "BULLISH_PINBAR",
                "price": round(c, 2),
                "open": round(o, 2),
                "high": round(h, 2),
                "low": round(l_, 2),
                "close": round(c, 2),
                "timestamp": now_str,
                "oi": oi,
                "reasons": "Long lower wick, small body, rejection at low",
            })
        elif _is_bearish_pinbar(o, h, l_, c):
            bearish.append({
                "stock": stock_name,
                "signal": "SELL",
                "pattern": "BEARISH_PINBAR",
                "price": round(c, 2),
                "open": round(o, 2),
                "high": round(h, 2),
                "low": round(l_, 2),
                "close": round(c, 2),
                "timestamp": now_str,
                "oi": oi,
                "reasons": "Long upper wick, small body, rejection at high",
            })

    logger.info("Pinbar: %s bullish, %s bearish", len(bullish), len(bearish))
    return bullish + bearish
