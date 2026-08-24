"""Unified intraday session pipeline: option chain + index analysis + options-trading scan."""

from __future__ import annotations

import logging
import os
from datetime import date, datetime
from pathlib import Path

from intraday_engine.core.config import Settings
from intraday_engine.jobs.index_pipeline import (
    DEFAULT_PIPELINE_UNDERLYINGS,
    run_index_pipeline_cycle,
)
from intraday_engine.jobs.option_chain_scraper import run_option_chain_scraper_job
from intraday_engine.research.market_overview import (
    load_stored_market_overview,
    run_market_overview_scan,
)
from intraday_engine.utils.logging_setup import setup_logging
from intraday_engine.utils.nse_session import (
    MARKET_OVERVIEW_RUN,
    is_nse_session_active,
    is_weekday,
    now_ist,
    seconds_to_next_interval_within_session,
    session_label,
    sleep_until_next_interval,
    wait_until_nse_session_open,
)

logger = logging.getLogger(__name__)

_market_overview_ran_date: date | None = None


def _session_underlyings() -> tuple[str, ...]:
    raw = os.getenv("SESSION_PIPELINE_UNDERLYINGS", "").strip()
    if raw:
        return tuple(u.strip().upper() for u in raw.split(",") if u.strip())
    raw = os.getenv("INDEX_PIPELINE_UNDERLYINGS", "").strip()
    if raw:
        return tuple(u.strip().upper() for u in raw.split(",") if u.strip())
    return DEFAULT_PIPELINE_UNDERLYINGS


def _market_overview_already_done(data_dir: Path, td: date) -> bool:
    global _market_overview_ran_date
    if _market_overview_ran_date == td:
        return True
    stored = load_stored_market_overview(data_dir, td)
    if stored and str(stored.get("trade_date", "")) == td.isoformat():
        _market_overview_ran_date = td
        return True
    return False


def maybe_run_market_overview(data_dir: Path, *, trade_date: date | None = None) -> dict | None:
    """
    Build the Market Overview snapshot once per weekday at/after 09:05 IST.
    Safe to call repeatedly — no-ops after the first successful run that day.
    """
    global _market_overview_ran_date
    td = trade_date or date.today()
    now = now_ist()
    if not is_weekday(now):
        return None
    if now.time() < MARKET_OVERVIEW_RUN:
        return None
    if _market_overview_already_done(data_dir, td):
        return None

    settings = Settings.from_env(underlying="NIFTY")
    logger.info("Running daily Market Overview scan (%s, ~09:05 IST window)…", td.isoformat())
    try:
        payload = run_market_overview_scan(settings=settings, trade_date=td, refresh_fii_dii=False)
        _market_overview_ran_date = td
        logger.info("Market Overview saved for %s.", td.isoformat())
        return payload
    except Exception as exc:
        logger.exception("Market Overview scan failed: %s", exc)
        return None


def run_session_cycle(
    data_dir: Path,
    *,
    underlyings: tuple[str, ...] | None = None,
    trade_date: date | None = None,
) -> dict:
    """
    One full session pass:
      1. Option chain capture (multi-strike ladder → option_chain.csv)
      2. Index pipeline per underlying (Index_Analysis/Signals → options-trading scan)
    """
    td = trade_date or date.today()
    targets = underlyings or _session_underlyings()
    started = datetime.now()

    chain = run_option_chain_scraper_job(
        data_dir,
        underlyings=targets,
        trade_date=td,
        respect_market_hours=True,
    )
    index = run_index_pipeline_cycle(underlyings=targets, trade_date=td)

    chain_ok = chain.get("status") in ("ok", "partial")
    index_ok = index.get("status") in ("ok", "partial")
    if chain_ok and index_ok:
        status = "ok"
    elif chain_ok or index_ok:
        status = "partial"
    else:
        status = "failed"

    return {
        "status": status,
        "trade_date": td.isoformat(),
        "started_at": started.isoformat(),
        "underlyings": list(targets),
        "option_chain": chain,
        "index_pipeline": index,
    }


def run_session_scheduler_loop(
    data_dir: Path,
    *,
    underlyings: tuple[str, ...] | None = None,
    interval_minutes: int = 5,
) -> None:
    """Blocking loop: full session pipeline every 5 min during NSE hours."""
    targets = underlyings or _session_underlyings()
    settings = Settings.from_env(underlying=targets[0] if targets else "NIFTY")
    setup_logging(settings.log_level, settings.data_dir)
    logger.info(
        "Session pipeline started (%s, every %d min, underlyings=%s): "
        "market overview @09:05 → option chain → index analysis → options-trading scan.",
        session_label(),
        interval_minutes,
        ", ".join(targets),
    )
    mo_hook = lambda: maybe_run_market_overview(data_dir)
    while True:
        mo_hook()
        if not is_nse_session_active():
            wait_until_nse_session_open(log=logger, on_chunk=mo_hook)
            continue

        try:
            summary = run_session_cycle(data_dir, underlyings=targets)
            logger.info("Session pipeline cycle %s", summary.get("status"))
            for r in (summary.get("index_pipeline") or {}).get("results") or []:
                if r.get("status") == "ok":
                    logger.info(
                        "  %s: %d ATM entry signal(s) (latest index signal: %s)",
                        r.get("underlying"),
                        r.get("options_trading_signals", 0),
                        r.get("latest_signal") or "—",
                    )
            logger.debug("Session pipeline detail: %s", summary)
        except Exception as exc:
            logger.exception("Session pipeline cycle failed: %s", exc)

        sleep_s = seconds_to_next_interval_within_session(interval_minutes)
        sleep_until_next_interval(sleep_s, on_chunk=mo_hook)
