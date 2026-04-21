"""Run 15-min signal cycle for all F&O stocks."""

from __future__ import annotations

import logging
import time
from datetime import date, datetime, timedelta

import pandas as pd

from intraday_engine.core.config import Settings
from intraday_engine.core.underlyings import get_underlying_config
from intraday_engine.engine.stock_signal_engine import run_stock_cycle
from intraday_engine.fetch.stock_market_data import _drop_incomplete_candles, _market_window_15min, _to_candle_df
from intraday_engine.fetch.zerodha_client import ZerodhaClient

logger = logging.getLogger(__name__)


def run_stocks_15min_cycle(trade_date: date | None = None, stock_limit: int | None = None) -> int:
    """
    Fetch 15-min data for all F&O stocks, generate signals, store per stock.
    Returns count of stocks processed successfully.
    """
    settings = Settings.from_env(underlying="NIFTY")
    client = ZerodhaClient(settings)
    benchmark_frame = _fetch_nifty_15min_benchmark(client, trade_date)
    stock_names = client.fno_stock_names()
    if stock_limit:
        stock_names = stock_names[:stock_limit]

    success = 0
    for name in stock_names:
        try:
            payload = run_stock_cycle(
                client,
                name,
                trade_date,
                include_options=True,
                benchmark_frame=benchmark_frame,
            )
            if payload:
                success += 1
                if payload.get("signal") in ("BUY", "SELL"):
                    logger.info("%s %s @ %.2f conf=%.2f", name, payload["signal"], payload.get("entry") or 0, payload.get("confidence") or 0)
        except Exception as e:
            logger.debug("Stock %s skip: %s", name, e)
    return success


def _fetch_nifty_15min_benchmark(
    client: ZerodhaClient,
    trade_date: date | None,
) -> pd.DataFrame | None:
    try:
        from_dt, to_dt = _market_window_15min(trade_date)
        if to_dt < from_dt:
            return None
        spot_symbol = get_underlying_config("NIFTY").spot_symbol
        quote = client.quote([spot_symbol])
        if not quote or spot_symbol not in quote:
            return None
        token = int(quote[spot_symbol]["instrument_token"])
        rows = client.historical_data(token, from_dt, to_dt, interval="15minute")
        frame = _drop_incomplete_candles(_to_candle_df(rows, "spot"), trade_date, 15)
        if frame.empty:
            return None
        frame["timestamp"] = pd.to_datetime(frame["timestamp"]).dt.strftime("%Y-%m-%dT%H:%M:%S")
        return frame[["timestamp", "spot_open_raw", "spot_close"]].copy()
    except Exception as exc:
        logger.debug("NIFTY benchmark fetch skipped: %s", exc)
        return None


def run_every_15_minutes(stock_limit: int | None = 50) -> None:
    """Scheduler: run F&O stock cycle every 15 minutes."""
    logger.info("Starting F&O stocks 15-min scheduler (limit=%s).", stock_limit)
    while True:
        try:
            n = run_stocks_15min_cycle(trade_date=date.today(), stock_limit=stock_limit)
            logger.info("Stocks cycle done: %s processed.", n)
        except Exception as exc:
            logger.exception("Stocks cycle failed: %s", exc)
        sleep_seconds = _seconds_to_next_15min()
        logger.info("Sleeping %s seconds until next 15-min boundary.", sleep_seconds)
        time.sleep(sleep_seconds)


def _seconds_to_next_15min() -> int:
    now = datetime.now()
    block = (now.minute // 15) * 15
    current = now.replace(minute=block, second=0, microsecond=0)
    nxt = current + timedelta(minutes=15)
    return max(60, int((nxt - now).total_seconds()))
