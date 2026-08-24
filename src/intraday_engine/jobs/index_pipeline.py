"""Index pipeline: direction engine + options-trading scan (no dashboard required)."""

from __future__ import annotations

import logging
import os
from datetime import date, datetime

from intraday_engine.core.config import Settings
from intraday_engine.core.underlyings import list_index_underlyings
from intraday_engine.engine import DirectionEngine
from intraday_engine.fetch.instrument_resolver import InstrumentResolver
from intraday_engine.fetch.market_data import MarketDataFetcher
from intraday_engine.fetch.zerodha_client import ZerodhaClient
from intraday_engine.research.options_trading_signals import run_options_trading_scan
from intraday_engine.storage import DataStore
from intraday_engine.utils.logging_setup import setup_logging
from intraday_engine.utils.nse_session import (
    is_nse_session_active,
    seconds_to_next_interval_within_session,
    session_label,
    sleep_until_next_interval,
    wait_until_nse_session_open,
)

logger = logging.getLogger(__name__)

DEFAULT_PIPELINE_UNDERLYINGS = ("NIFTY", "BANKNIFTY")


def _pipeline_underlyings() -> tuple[str, ...]:
    raw = os.getenv("INDEX_PIPELINE_UNDERLYINGS", "").strip()
    if raw:
        return tuple(u.strip().upper() for u in raw.split(",") if u.strip())
    return DEFAULT_PIPELINE_UNDERLYINGS


def _build_engine(underlying: str) -> DirectionEngine:
    settings = Settings.from_env(underlying=underlying)
    client = ZerodhaClient(settings)
    resolver = InstrumentResolver(client, settings)
    fetcher = MarketDataFetcher(client, resolver, settings)
    store = DataStore(settings.data_dir, underlying=settings.underlying)
    return DirectionEngine(fetcher, store, settings)


def run_index_pipeline_cycle(
    *,
    underlyings: tuple[str, ...] | None = None,
    trade_date: date | None = None,
) -> dict:
    """
    One pipeline pass per index: fetch 5-min bars → Index_Analysis/Signals → options-trading scan.
    """
    td = trade_date or date.today()
    targets = underlyings or _pipeline_underlyings()
    started = datetime.now()
    results: list[dict] = []
    errors: list[str] = []

    for u in targets:
        u_key = u.strip().upper()
        if u_key not in list_index_underlyings():
            results.append({"underlying": u_key, "status": "skipped", "reason": "not an index underlying"})
            continue
        try:
            engine = _build_engine(u_key)
            payload = engine.run_cycle(trade_date=td)
            opts = run_options_trading_scan(engine.settings.data_dir, td, u_key)
            signal_count = len(opts.get("signals") or [])
            results.append({
                "underlying": u_key,
                "status": "ok",
                "trade_date": td.isoformat(),
                "latest_signal": payload.get("signal"),
                "options_trading_signals": signal_count,
            })
        except Exception as exc:
            logger.exception("Index pipeline failed for %s: %s", u_key, exc)
            errors.append(f"{u_key}: {exc}")
            results.append({"underlying": u_key, "status": "error", "reason": str(exc)})

    ok_count = sum(1 for r in results if r.get("status") == "ok")
    status = "ok" if ok_count == len(targets) else ("partial" if ok_count else "failed")
    return {
        "status": status,
        "trade_date": td.isoformat(),
        "started_at": started.isoformat(),
        "results": results,
        "errors": errors,
    }


def run_index_pipeline_scheduler_loop(
    *,
    underlyings: tuple[str, ...] | None = None,
    interval_minutes: int = 5,
) -> None:
    """Blocking loop: run index pipeline every 5 min during NSE session."""
    targets = underlyings or _pipeline_underlyings()
    settings = Settings.from_env(underlying=targets[0] if targets else "NIFTY")
    setup_logging(settings.log_level, settings.data_dir)
    logger.info(
        "Index pipeline scheduler started (%s, every %d min, underlyings=%s).",
        session_label(),
        interval_minutes,
        ", ".join(targets),
    )
    while True:
        if not is_nse_session_active():
            wait_until_nse_session_open(log=logger)
            continue

        try:
            summary = run_index_pipeline_cycle(underlyings=targets)
            logger.info(
                "Index pipeline cycle %s: %s",
                summary.get("status"),
                summary.get("results"),
            )
        except Exception as exc:
            logger.exception("Index pipeline scheduler cycle failed: %s", exc)

        sleep_s = seconds_to_next_interval_within_session(interval_minutes)
        sleep_until_next_interval(sleep_s)
