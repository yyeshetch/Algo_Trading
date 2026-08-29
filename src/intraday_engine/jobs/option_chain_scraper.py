"""Option chain scraper job — capture Kite snapshots for index underlyings."""

from __future__ import annotations

import logging
from datetime import date, datetime
from pathlib import Path

import pandas as pd

from intraday_engine.core.config import Settings
from intraday_engine.core.underlyings import list_index_underlyings
from intraday_engine.fetch.zerodha_client import ZerodhaClient
from intraday_engine.gamma.huge_move_predictor import HugeMovePredictor
from intraday_engine.gamma.option_chain_fetcher import resolve_option_chain_strike_counts
from intraday_engine.jobs.registry import update_job_state
from intraday_engine.storage.layout import normalize_underlying, option_chain_day_path
from intraday_engine.utils.nse_session import (
    is_nse_session_active,
    seconds_to_next_interval_within_session,
    session_label,
    sleep_until_next_interval,
    wait_until_nse_session_open,
)

logger = logging.getLogger(__name__)

DEFAULT_SCRAPER_UNDERLYINGS = ("NIFTY", "BANKNIFTY")


def option_chain_data_available(
    data_dir: Path,
    trade_date: date,
    underlying: str,
    *,
    min_snapshots: int = 1,
) -> bool:
    """Return True when stored option-chain CSV has data for the underlying on trade_date."""
    from intraday_engine.storage.backend import write_to_db

    if write_to_db():
        from intraday_engine.storage import db as db_store

        df = db_store.load_option_chain_rows(trade_date, underlying)
        if df.empty:
            return False
        if "timestamp" not in df.columns:
            return len(df) >= min_snapshots
        return df["timestamp"].nunique() >= min_snapshots

    path = option_chain_day_path(data_dir, trade_date)
    if not path.exists():
        return False
    try:
        df = pd.read_csv(path)
    except Exception:
        return False
    if df.empty:
        return False
    u = normalize_underlying(underlying)
    if "underlying" in df.columns:
        df = df[df["underlying"].astype(str).str.upper() == u]
    if df.empty:
        return False
    if "timestamp" not in df.columns:
        return len(df) >= min_snapshots
    return df["timestamp"].nunique() >= min_snapshots


def option_chain_status(
    data_dir: Path,
    trade_date: date,
    underlying: str,
) -> dict:
    """Summary of stored option-chain data for one underlying."""
    from intraday_engine.storage.backend import write_to_db

    u = normalize_underlying(underlying)
    if write_to_db():
        from intraday_engine.storage import db as db_store

        df = db_store.load_option_chain_rows(trade_date, u)
        snapshots = int(df["timestamp"].nunique()) if not df.empty and "timestamp" in df.columns else 0
        last_ts = None
        if not df.empty and "timestamp" in df.columns:
            last_ts = str(sorted(df["timestamp"].unique())[-1])
        return {
            "available": snapshots > 0,
            "underlying": u,
            "trade_date": trade_date.isoformat(),
            "path": "mysql://option_chain_rows",
            "snapshots": snapshots,
            "last_timestamp": last_ts,
        }

    path = option_chain_day_path(data_dir, trade_date)
    u = normalize_underlying(underlying)
    if not path.exists():
        return {
            "available": False,
            "underlying": u,
            "trade_date": trade_date.isoformat(),
            "path": str(path),
            "snapshots": 0,
            "last_timestamp": None,
        }
    try:
        df = pd.read_csv(path)
        if "underlying" in df.columns:
            df = df[df["underlying"].astype(str).str.upper() == u]
        snapshots = int(df["timestamp"].nunique()) if not df.empty and "timestamp" in df.columns else 0
        last_ts = None
        if not df.empty and "timestamp" in df.columns:
            last_ts = str(sorted(df["timestamp"].unique())[-1])
        return {
            "available": snapshots > 0,
            "underlying": u,
            "trade_date": trade_date.isoformat(),
            "path": str(path),
            "snapshots": snapshots,
            "last_timestamp": last_ts,
        }
    except Exception as exc:
        return {
            "available": False,
            "underlying": u,
            "trade_date": trade_date.isoformat(),
            "path": str(path),
            "error": str(exc),
            "snapshots": 0,
            "last_timestamp": None,
        }


def capture_option_chain_for_underlying(
    underlying: str,
    trade_date: date | None = None,
    *,
    num_strikes: int | None = None,
) -> dict:
    """Fetch and store one option-chain snapshot. Returns status dict."""
    td = trade_date or date.today()
    u = normalize_underlying(underlying)
    if u not in list_index_underlyings():
        return {"underlying": u, "status": "skipped", "reason": "not an index underlying"}

    ce, pe = resolve_option_chain_strike_counts(num_strikes)
    settings = Settings.from_env(underlying=u)
    client = ZerodhaClient(settings)
    predictor = HugeMovePredictor(client, settings)
    snapshot = predictor.capture_and_store(trade_date=td, num_strikes=num_strikes)
    if snapshot is None:
        return {"underlying": u, "status": "failed", "reason": "capture returned no data"}

    return {
        "underlying": u,
        "status": "ok",
        "trade_date": td.isoformat(),
        "timestamp": snapshot.timestamp.isoformat(),
        "atm_strike": snapshot.atm_strike,
        "spot_price": snapshot.spot_price,
        "strikes": len(snapshot.strikes),
        "ce_strikes": ce,
        "pe_strikes": pe,
    }


def run_option_chain_scraper_job(
    data_dir: Path,
    *,
    underlyings: tuple[str, ...] = DEFAULT_SCRAPER_UNDERLYINGS,
    trade_date: date | None = None,
    num_strikes: int | None = None,
    respect_market_hours: bool = True,
) -> dict:
    """Run the configured option-chain scraper for all target underlyings."""
    if respect_market_hours and not is_nse_session_active():
        msg = f"Outside NSE session ({session_label()})"
        update_job_state(
            data_dir,
            "option_chain_scraper",
            last_status="skipped",
            skip_reason=msg,
        )
        return {
            "job_id": "option_chain_scraper",
            "status": "skipped",
            "message": msg,
            "schedule": session_label(),
        }

    td = trade_date or date.today()
    started = datetime.now()
    results: list[dict] = []
    errors: list[str] = []

    for u in underlyings:
        try:
            results.append(capture_option_chain_for_underlying(u, td, num_strikes=num_strikes))
        except Exception as exc:
            logger.exception("Option chain capture failed for %s: %s", u, exc)
            errors.append(f"{u}: {exc}")
            results.append({"underlying": u, "status": "error", "reason": str(exc)})

    ok_count = sum(1 for r in results if r.get("status") == "ok")
    status = "ok" if ok_count == len(underlyings) else ("partial" if ok_count else "failed")

    update_job_state(
        data_dir,
        "option_chain_scraper",
        last_run_at=started.isoformat(),
        last_status=status,
        last_results=results,
        last_error="; ".join(errors) if errors else None,
        success_count=ok_count,
        total_count=len(underlyings),
    )

    return {
        "job_id": "option_chain_scraper",
        "status": status,
        "trade_date": td.isoformat(),
        "started_at": started.isoformat(),
        "results": results,
        "errors": errors,
    }


def ensure_option_chain_data(
    data_dir: Path,
    underlying: str,
    trade_date: date,
    *,
    num_strikes: int | None = None,
) -> dict:
    """Capture option chain when missing for the given date/underlying."""
    if option_chain_data_available(data_dir, trade_date, underlying):
        st = option_chain_status(data_dir, trade_date, underlying)
        st["action"] = "none"
        return st
    cap = capture_option_chain_for_underlying(underlying, trade_date, num_strikes=num_strikes)
    st = option_chain_status(data_dir, trade_date, underlying)
    st["action"] = "captured"
    st["capture"] = cap
    return st


def run_option_chain_scheduler_loop(
    data_dir: Path,
    *,
    underlyings: tuple[str, ...] = DEFAULT_SCRAPER_UNDERLYINGS,
    interval_minutes: int = 5,
) -> None:
    """
    Blocking loop: capture option chain every 5 min during NSE session (09:15–15:30 IST).
    Run standalone — no dashboard required. Use CLI: --option-chain-scheduler
    """
    ce, pe = resolve_option_chain_strike_counts()
    logger.info(
        "Option chain scheduler started (%s, every %d min, %d CE + %d PE @ 100-pt).",
        session_label(),
        interval_minutes,
        ce,
        pe,
    )
    while True:
        if not is_nse_session_active():
            wait_until_nse_session_open(log=logger)
            continue

        try:
            run_option_chain_scraper_job(
                data_dir,
                underlyings=underlyings,
                respect_market_hours=True,
            )
        except Exception as exc:
            logger.exception("Option chain scheduler cycle failed: %s", exc)

        sleep_s = seconds_to_next_interval_within_session(interval_minutes)
        sleep_until_next_interval(sleep_s)
