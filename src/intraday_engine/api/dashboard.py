"""FastAPI dashboard for intraday signals and order execution."""

from __future__ import annotations

import asyncio
import logging
import math
import os
import sys

import pandas as pd
from contextlib import asynccontextmanager
from datetime import date, datetime, timedelta
from pathlib import Path

from fastapi import Body, FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from kiteconnect import exceptions as kite_exceptions

from intraday_engine.core.config import Settings
from intraday_engine.core.tunables import (
    apply_value_updates,
    default_config,
    ensure_config_file,
    invalidate_cache,
    load_config,
    save_config,
    validate_no_unknown_keys,
)
from intraday_engine.core.underlyings import list_index_underlyings
from intraday_engine.engine import DirectionEngine
from intraday_engine.fetch.instrument_resolver import InstrumentResolver
from intraday_engine.fetch.market_data import MarketDataFetcher
from intraday_engine.fetch.nse_market_indices import build_nse_sector_stock_map
from intraday_engine.fetch.zerodha_client import ZerodhaClient
from intraday_engine.storage import DataStore, invalidate_storage_cache
from intraday_engine.storage.backend import get_storage_backend, write_to_db
from intraday_engine.storage.data_store import _flatten_for_csv
from intraday_engine.analysis.summary_builder import build_analysis_summaries
from intraday_engine.analysis.hourly_levels import fetch_hourly_spot_levels, hourly_levels_from_snapshots
from intraday_engine.analysis.option_confluence import attach_confluence_to_summaries
from intraday_engine.analysis.realized_implied_vol import attach_volatility_compare
from intraday_engine.analysis.buyers_day import compute_buyers_day, compute_buyers_day_series
from intraday_engine.analysis.prior_session import (
    load_prior_session_summaries,
    prior_session_close,
)
from intraday_engine.analysis.price_action_volume import compute_price_action_volume
from intraday_engine.analysis.pro_print import compute_pro_print
from intraday_engine.analysis.atr_squeeze import compute_atr_squeeze
from intraday_engine.analysis.strike_oi_change import compute_strike_oi_change
from intraday_engine.analysis.futures_oi_buildup import compute_futures_oi_buildup
from intraday_engine.analysis.expiry_gamma_radar import compute_expiry_gamma_radar, expiry_window_tape
from intraday_engine.analysis.trend_ignition import compute_trend_ignition, trend_ignition_tape
from intraday_engine.analysis.intraday_trend_change import compute_intraday_trend_change
from intraday_engine.gamma.expiry_utils import is_expiry_day
from intraday_engine.eod.eod_fetcher import load_stored_eod_indicators, run_and_save_eod_scan
from intraday_engine.engine.stock_signal_engine import run_stock_analysis_30min
from intraday_engine.research.tomorrow_watchlist_scanner import (
    load_stored_tomorrow_watchlist_fresh_or_previous,
    run_tomorrow_watchlist_scan,
)
from intraday_engine.research.silent_accumulation_scanner import (
    load_stored_silent_accumulation,
    run_silent_accumulation_scan,
)
from intraday_engine.fetch.nse_public_data import FII_DII_DEFAULT_TRADING_DAYS
from intraday_engine.research.fii_dii_trends import (
    load_stored_fii_dii_trends,
    run_fii_dii_trends_scan,
)
from intraday_engine.research.relative_strength_scanner import (
    load_stored_relative_strength,
    run_relative_strength_scan,
)
from intraday_engine.research.minervini_trend_template_scanner import (
    load_stored_minervini_trend_template,
    run_minervini_trend_template_scan,
)
from intraday_engine.research.intraday_relative_strength_scanner import (
    load_stored_intraday_relative_strength,
    run_intraday_relative_strength_scan,
)
from intraday_engine.research.sector_relative_strength_scanner import (
    load_stored_sector_relative_strength,
    run_sector_relative_strength_scan,
)
from intraday_engine.research.sector_rotation_scanner import (
    load_stored_sector_rotation,
    run_sector_rotation_scan,
)
from intraday_engine.research.institutional_volume_scanner import (
    api_institutional_volume_payload,
    load_stored_institutional_volume,
    run_institutional_volume_scan,
)
from intraday_engine.research.fundamentals_screener import (
    load_latest_fundamentals_csv,
    run_fundamentals_scan,
)
from intraday_engine.research.stock_news_scanner import (
    load_latest_news_csv,
    run_news_scan,
)
from intraday_engine.orb.orb_scanner import run_orb_scan, run_pinbar_scan
from intraday_engine.research.combined_signals_scanner import run_combined_scan
from intraday_engine.research.market_overview import (
    load_stored_market_overview,
    run_market_overview_scan,
)
from intraday_engine.storage.layout import combined_signals_csv_path
from intraday_engine.jobs.registry import get_job_states
from intraday_engine.jobs.option_chain_scraper import (
    ensure_option_chain_data,
    option_chain_status,
    run_option_chain_scraper_job,
)
from intraday_engine.jobs.runner import run_configured_jobs_loop
from intraday_engine.research.options_trading_signals import (
    load_options_trading_signals,
    run_options_trading_scan,
)
from intraday_engine.storage.position_sl_store import get_sl as get_sl_record, set_sl as store_sl, update_sl_trigger as store_update_sl
from intraday_engine.storage.signal_invalidation_store import invalidate as store_invalidate_signal, load_invalidated_keys, reinstate as store_reinstate_signal
from intraday_engine.execution.auto_trade_store import get_config as get_auto_trade_config, update_config as update_auto_trade_config
from intraday_engine.execution.options_auto_executor import run_auto_trade_cycle, run_position_trail_cycle
from intraday_engine.execution.strategy_store import get_enabled_map, list_strategies, update_strategy
from intraday_engine.utils.logging_setup import setup_logging
from intraday_engine.utils.nse_session import is_nse_session_active, seconds_until_next_session_start, session_label

logger = logging.getLogger(__name__)

_jobs_loop_task: asyncio.Task | None = None
_auto_trade_task: asyncio.Task | None = None


def _skip_dashboard_option_chain_job() -> bool:
    if _read_only_dashboard():
        return True
    return os.getenv("INTRADAY_NO_OPTION_CHAIN_JOB", "").strip().lower() in ("1", "true", "yes")


def _read_only_dashboard() -> bool:
    return os.getenv("INTRADAY_READ_ONLY_DASHBOARD", "").strip().lower() in ("1", "true", "yes")


def _skip_auto_trade_loop() -> bool:
    if _read_only_dashboard():
        return True
    return os.getenv("INTRADAY_NO_AUTO_TRADE_LOOP", "").strip().lower() in ("1", "true", "yes")


async def _auto_trade_loop():
    """Every 5 min during NSE session only: refresh, scan entry signals, standard trail."""
    from intraday_engine.utils.nse_session import seconds_to_next_interval_within_session

    logger.info("Auto-trade loop started (%s).", session_label())
    loop = asyncio.get_event_loop()
    while True:
        if not is_nse_session_active():
            sleep_s = seconds_until_next_session_start()
            logger.debug("Auto-trade: outside session; sleeping %.0f min.", sleep_s / 60)
            await asyncio.sleep(min(sleep_s, 3600))
            continue
        try:
            for u in list_index_underlyings():
                try:
                    settings = _get_settings(u)
                    client = _get_client(u)
                    store = _get_store(u)
                    engine = _get_engine(u)
                    await loop.run_in_executor(None, engine.run_cycle)
                    await loop.run_in_executor(None, lambda s=settings, c=client, st=store: run_auto_trade_cycle(s, c, st))
                    await loop.run_in_executor(None, lambda s=settings, c=client, st=store: run_position_trail_cycle(s, c, st))
                except Exception as e:
                    logger.debug("Auto-trade cycle %s: %s", u, e)
        except Exception as e:
            logger.debug("Auto-trade loop: %s", e)
        sleep_s = seconds_to_next_interval_within_session(5)
        await asyncio.sleep(sleep_s)


@asynccontextmanager
async def _lifespan(app: FastAPI):
    global _jobs_loop_task, _auto_trade_task
    try:
        ensure_config_file()
    except Exception as e:
        logger.warning("Could not ensure config.json: %s", e)
    if write_to_db():
        try:
            from intraday_engine.storage.db import ensure_schema

            ensure_schema()
            logger.info("Dashboard reading from InterServer MySQL (%s).", get_storage_backend().value)
        except Exception as e:
            logger.warning("MySQL schema check failed: %s", e)
    data_dir = Settings.from_env().data_dir
    skip_jobs = frozenset({"option_chain_scraper"}) if _skip_dashboard_option_chain_job() else frozenset()
    if _read_only_dashboard():
        logger.info("Dashboard read-only mode (INTRADAY_READ_ONLY_DASHBOARD): stored data only, no background fetches.")
    elif skip_jobs:
        logger.info("Dashboard option-chain job disabled (INTRADAY_NO_OPTION_CHAIN_JOB).")
    if not _read_only_dashboard():
        _jobs_loop_task = asyncio.create_task(
            run_configured_jobs_loop(data_dir, skip_job_ids=skip_jobs),
        )
    if not _skip_auto_trade_loop():
        _auto_trade_task = asyncio.create_task(_auto_trade_loop())
    else:
        logger.info("Dashboard auto-trade / direction-engine loop disabled.")
    yield
    for task in (_jobs_loop_task, _auto_trade_task):
        if task:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass


app = FastAPI(title="Intraday Direction Engine Dashboard", lifespan=_lifespan)
_engines: dict[str, DirectionEngine] = {}
_clients: dict[str, ZerodhaClient] = {}
_settings_cache: dict[str, Settings] = {}


@app.get("/api/dashboard/mode")
async def api_dashboard_mode():
    """Whether the dashboard is read-only (stored data only, no background pipeline)."""
    return {
        "read_only": _read_only_dashboard(),
        "option_chain_job": not _skip_dashboard_option_chain_job(),
        "auto_trade_loop": not _skip_auto_trade_loop(),
        "storage_backend": get_storage_backend().value,
        "reads_from_db": write_to_db(),
    }


@app.post("/api/dashboard/reload-cache")
async def api_dashboard_reload_cache():
    """Clear in-memory DB read cache (hard reload pulls fresh rows from MySQL)."""
    invalidate_storage_cache()
    if write_to_db():
        try:
            from intraday_engine.storage.db import ensure_schema, mysql_host_label

            tables = ensure_schema()
            return {
                "status": "ok",
                "storage_backend": get_storage_backend().value,
                "reads_from_db": True,
                "mysql_host": mysql_host_label(),
                "tables": tables,
            }
        except Exception as exc:
            logger.error("MySQL reload-cache failed: %s", exc)
            raise HTTPException(status_code=503, detail=str(exc)) from exc
    return {
        "status": "ok",
        "storage_backend": get_storage_backend().value,
        "reads_from_db": write_to_db(),
    }


def _require_pipeline_writes():
    if _read_only_dashboard():
        raise HTTPException(
            status_code=403,
            detail=(
                "Dashboard is read-only. Run the session pipeline separately: "
                "python -m intraday_engine.cli.main --session-scheduler"
            ),
        )


def _underlying_key(underlying: str | None) -> str:
    u = (underlying or "").strip().upper().replace(" ", "").replace("NIFTYBANK", "BANKNIFTY")
    return u or os.getenv("UNDERLYING", "NIFTY").strip().upper()


def _get_engine(underlying: str | None = None) -> DirectionEngine:
    key = _underlying_key(underlying)
    if key not in _engines:
        settings = Settings.from_env(underlying=key)
        setup_logging(settings.log_level, settings.data_dir)
        client = ZerodhaClient(settings)
        resolver = InstrumentResolver(client, settings)
        fetcher = MarketDataFetcher(client, resolver, settings)
        store = DataStore(settings.data_dir, underlying=settings.underlying)
        _engines[key] = DirectionEngine(fetcher, store, settings)
        _clients[key] = client
        _settings_cache[key] = settings
    return _engines[key]


def _get_client(underlying: str | None = None) -> ZerodhaClient:
    key = _underlying_key(underlying)
    _get_engine(underlying)
    return _clients[key]


def _get_settings(underlying: str | None = None) -> Settings:
    key = _underlying_key(underlying)
    _get_engine(underlying)
    return _settings_cache[key]


def _get_store(underlying: str | None = None) -> DataStore:
    return _get_engine(underlying).store


def _invalidate_settings_cache(underlying: str | None = None) -> None:
    key = _underlying_key(underlying)
    _settings_cache.pop(key, None)


def _set_daily_max_loss(amount: float, underlying: str | None = None) -> float:
    """Persist daily loss cap to config.json and refresh cached settings."""
    val = max(100.0, min(100_000.0, float(amount)))
    cfg = load_config()
    sec = cfg.get("settings", {})
    node = sec.get("daily_sl_rupees")
    if isinstance(node, dict) and "value" in node:
        node["value"] = val
    else:
        sec["daily_sl_rupees"] = {"value": val, "type": "float"}
    save_config(cfg)
    invalidate_cache()
    _invalidate_settings_cache(underlying)
    return val


def _templates_dir() -> Path:
    return Path(__file__).parent / "templates"


@app.get("/", response_class=HTMLResponse)
async def dashboard():
    path = _templates_dir() / "dashboard.html"
    if not path.exists():
        raise HTTPException(status_code=500, detail="Dashboard template not found.")
    return HTMLResponse(
        path.read_text(encoding="utf-8"),
        headers={"Cache-Control": "no-store"},
    )


@app.get("/stocks", response_class=HTMLResponse)
async def stocks_dashboard():
    """F&O stocks scanner dashboard (15-min timeframe)."""
    path = _templates_dir() / "stocks_dashboard.html"
    if not path.exists():
        raise HTTPException(status_code=500, detail="Stocks dashboard template not found.")
    return HTMLResponse(
        path.read_text(encoding="utf-8"),
        headers={"Cache-Control": "no-store"},
    )


@app.get("/tunables", response_class=HTMLResponse)
async def tunables_page():
    """Edit project config.json (tunables) in the browser."""
    path = _templates_dir() / "tunables.html"
    if not path.exists():
        raise HTTPException(status_code=500, detail="Tunables template not found.")
    return HTMLResponse(path.read_text(encoding="utf-8"))


@app.get("/jobs", response_class=HTMLResponse)
async def jobs_page():
    """Configured background jobs and their last run status."""
    path = _templates_dir() / "jobs.html"
    if not path.exists():
        raise HTTPException(status_code=500, detail="Jobs template not found.")
    return HTMLResponse(path.read_text(encoding="utf-8"))


@app.get("/api/tunables")
async def api_tunables_get():
    try:
        ensure_config_file()
    except Exception as e:
        logger.warning("ensure_config_file: %s", e)
    return _sanitize_for_json(load_config(force_reload=True))


@app.put("/api/tunables")
async def api_tunables_put(payload: dict = Body(...)):
    template = default_config()
    errs = validate_no_unknown_keys(template, payload)
    if errs:
        raise HTTPException(status_code=400, detail="; ".join(errs[:12]))
    merged = load_config(force_reload=True)
    apply_value_updates(merged, payload)
    save_config(merged)
    return {"status": "ok", "message": "config.json saved. Restart long-running processes to pick up all changes."}


def _json_safe(val):
    """Convert value to JSON-serializable form (NaN -> None)."""
    if val is None:
        return None
    if hasattr(val, "item"):
        try:
            val = val.item()
        except (ValueError, AttributeError):
            pass
    if isinstance(val, float) and math.isnan(val):
        return None
    return val


def _sanitize_for_json(obj):
    """Recursively replace NaN and non-JSON values in dict/list."""
    if obj is None:
        return None
    if isinstance(obj, dict):
        return {k: _sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize_for_json(v) for v in obj]
    if hasattr(obj, "isoformat"):  # datetime
        return obj.isoformat()
    return _json_safe(obj)


@app.get("/api/stocks/orb")
async def api_stocks_orb(limit: int = 200, use_cached: bool = True):
    """15-min ORB signals (0.2% variation). Uses bulk quote for prices."""
    try:
        loop = asyncio.get_event_loop()
        results = await loop.run_in_executor(
            None,
            lambda: run_orb_scan(trade_date=date.today(), stock_limit=limit, use_cached_or=use_cached),
        )
        buy = [s for s in results if s["signal"] == "BUY"]
        sell = [s for s in results if s["signal"] == "SELL"]
        return {
            "signals": [_sanitize_for_json(s) for s in results],
            "buy": [_sanitize_for_json(s) for s in buy],
            "sell": [_sanitize_for_json(s) for s in sell],
        }
    except Exception as e:
        logger.exception("ORB scan failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/stocks/pinbar")
async def api_stocks_pinbar(limit: int = 200):
    """Bullish and bearish pinbars on last 15-min candle."""
    try:
        loop = asyncio.get_event_loop()
        results = await loop.run_in_executor(
            None,
            lambda: run_pinbar_scan(trade_date=date.today(), stock_limit=limit),
        )
        bull = [s for s in results if s.get("pattern") == "BULLISH_PINBAR"]
        bear = [s for s in results if s.get("pattern") == "BEARISH_PINBAR"]
        return {
            "signals": [_sanitize_for_json(s) for s in results],
            "bullish": [_sanitize_for_json(s) for s in bull],
            "bearish": [_sanitize_for_json(s) for s in bear],
        }
    except Exception as e:
        logger.exception("Pinbar scan failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/stocks/tomorrow-watchlist")
async def api_stocks_tomorrow_watchlist(trade_date: str | None = None):
    """Load last saved NIFTY 500 multi-timeframe watchlist JSON (per trade_date)."""
    try:
        td = date.today()
        if trade_date:
            try:
                td = datetime.strptime(trade_date, "%Y-%m-%d").date()
            except ValueError:
                pass
        settings = Settings.from_env(underlying="NIFTY")
        data, stale = load_stored_tomorrow_watchlist_fresh_or_previous(settings.data_dir, td)
        if not data:
            return {
                "trade_date": td.isoformat(),
                "picks": [],
                "message": "No saved scan for this date. Click Run scan on the Watchlist for Tomorrow tab.",
            }
        if stale:
            data = dict(data)
            data["stale"] = True
            data["requested_trade_date"] = td.isoformat()
            shown = data.get("trade_date", "?")
            data["message"] = (
                f"Showing last saved watchlist ({shown}) until you run a scan for {td.isoformat()}."
            )
        return _sanitize_for_json(data)
    except Exception as e:
        logger.exception("Tomorrow watchlist load failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/stocks/tomorrow-watchlist/refresh")
async def api_stocks_tomorrow_watchlist_refresh(
    trade_date: str | None = None,
    top_n: int = 20,
    stock_limit: int | None = None,
    max_workers: int = 4,
):
    """Run full NIFTY 500 scan (slow; many Kite historical calls)."""
    try:
        td = date.today()
        if trade_date:
            try:
                td = datetime.strptime(trade_date, "%Y-%m-%d").date()
            except ValueError:
                pass
        lim = stock_limit if stock_limit is not None and stock_limit > 0 else None
        loop = asyncio.get_event_loop()
        payload = await loop.run_in_executor(
            None,
            lambda: run_tomorrow_watchlist_scan(
                trade_date=td,
                top_n=top_n,
                max_workers=max_workers,
                stock_limit=lim,
            ),
        )
        return _sanitize_for_json(payload)
    except Exception as e:
        logger.exception("Tomorrow watchlist refresh failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/stocks/refresh-all")
async def api_stocks_refresh_all(
    limit: int | None = None,
    top_n: int = 20,
    max_workers: int = 4,
):
    """Run tomorrow watchlist scan (NIFTY 500 multi-timeframe)."""
    try:
        loop = asyncio.get_event_loop()
        td = date.today()
        lim = limit if limit is not None and limit > 0 else None
        watchlist_payload = await loop.run_in_executor(
            None,
            lambda: run_tomorrow_watchlist_scan(
                trade_date=td,
                top_n=top_n,
                max_workers=max_workers,
                stock_limit=lim,
            ),
        )

        return {
            "status": "ok",
            "message": "Tomorrow watchlist refresh complete.",
            "trade_date": td.isoformat(),
            "tomorrow_watchlist": {
                "scanned": int(watchlist_payload.get("scanned", 0) or 0),
                "picks": len(watchlist_payload.get("picks", []) or []),
                "failed": int(watchlist_payload.get("failed_count", 0) or 0),
                "data_source": watchlist_payload.get("data_source"),
            },
        }
    except Exception as e:
        logger.exception("Shared stocks refresh failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


def _parse_trade_date(s: str | None) -> date:
    if not s:
        return date.today()
    try:
        return datetime.strptime(s, "%Y-%m-%d").date()
    except ValueError:
        return date.today()


@app.get("/api/stocks/silent-accumulation")
async def api_silent_accumulation_get(trade_date: str | None = None):
    """Load last stored silent-accumulation scan (falls back to most recent file)."""
    try:
        td = _parse_trade_date(trade_date)
        settings = Settings.from_env(underlying="NIFTY")
        data = load_stored_silent_accumulation(settings.data_dir, td)
        if not data:
            return {
                "trade_date": td.isoformat(),
                "rows": [],
                "message": "No saved scan yet. Click Refresh to run.",
            }
        return _sanitize_for_json(data)
    except Exception as e:
        logger.exception("Silent accumulation load failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/stocks/silent-accumulation/refresh")
async def api_silent_accumulation_refresh(
    trade_date: str | None = None,
    top_n: int = 25,
    max_workers: int = 8,
    use_nse_data: bool = True,
):
    try:
        td = _parse_trade_date(trade_date)
        loop = asyncio.get_event_loop()
        payload = await loop.run_in_executor(
            None,
            lambda: run_silent_accumulation_scan(
                trade_date=td,
                top_n=top_n,
                max_workers=max_workers,
                use_nse_data=use_nse_data,
            ),
        )
        return _sanitize_for_json(payload)
    except Exception as e:
        logger.exception("Silent accumulation refresh failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/stocks/fii-dii-trends")
async def api_fii_dii_trends_get(trade_date: str | None = None):
    try:
        td = _parse_trade_date(trade_date)
        settings = Settings.from_env(underlying="NIFTY")
        data = load_stored_fii_dii_trends(settings.data_dir, td, hydrate_cache=True)
        if not data:
            return {
                "trade_date": td.isoformat(),
                "fii_dii": [],
                "participant_oi": [],
                "summary": {},
                "message": "No saved trends yet. Click Refresh to fetch from NSE.",
            }
        return _sanitize_for_json(data)
    except Exception as e:
        logger.exception("FII/DII trends load failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/stocks/fii-dii-trends/refresh")
async def api_fii_dii_trends_refresh(trade_date: str | None = None, days: int = FII_DII_DEFAULT_TRADING_DAYS):
    try:
        td = _parse_trade_date(trade_date)
        loop = asyncio.get_event_loop()
        payload = await loop.run_in_executor(
            None,
            lambda: run_fii_dii_trends_scan(trade_date=td, days=days),
        )
        return _sanitize_for_json(payload)
    except Exception as e:
        logger.exception("FII/DII trends refresh failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/stocks/relative-strength")
async def api_relative_strength_get(trade_date: str | None = None):
    try:
        td = _parse_trade_date(trade_date)
        settings = Settings.from_env(underlying="NIFTY")
        data = load_stored_relative_strength(settings.data_dir, td)
        if not data:
            return {
                "trade_date": td.isoformat(),
                "rows": [],
                "message": "No saved RS scan yet. Click Refresh to run.",
            }
        return _sanitize_for_json(data)
    except Exception as e:
        logger.exception("Relative strength load failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/stocks/relative-strength/refresh")
async def api_relative_strength_refresh(
    trade_date: str | None = None,
    top_n: int = 200,
    max_workers: int = 8,
    only_outperformers: bool = True,
):
    try:
        td = _parse_trade_date(trade_date)
        loop = asyncio.get_event_loop()
        payload = await loop.run_in_executor(
            None,
            lambda: run_relative_strength_scan(
                trade_date=td,
                top_n=top_n,
                max_workers=max_workers,
                only_outperformers=only_outperformers,
            ),
        )
        return _sanitize_for_json(payload)
    except Exception as e:
        logger.exception("Relative strength refresh failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/stocks/minervini-template")
async def api_minervini_template_get(trade_date: str | None = None):
    try:
        td = _parse_trade_date(trade_date)
        settings = Settings.from_env(underlying="NIFTY")
        data = load_stored_minervini_trend_template(settings.data_dir, td)
        if not data:
            return {
                "trade_date": td.isoformat(),
                "rows": [],
                "message": "No saved Minervini scan yet. Click Refresh to run.",
            }
        return _sanitize_for_json(data)
    except Exception as e:
        logger.exception("Minervini template load failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/stocks/minervini-template/refresh")
async def api_minervini_template_refresh(
    trade_date: str | None = None,
    top_n: int = 50,
    max_workers: int = 8,
    rs_min_percentile: float = 70.0,
    only_full_pass: bool = False,
    mode: str = "setup",
):
    try:
        td = _parse_trade_date(trade_date)
        scan_mode = mode if mode in ("setup", "extended") else "setup"
        loop = asyncio.get_event_loop()
        payload = await loop.run_in_executor(
            None,
            lambda: run_minervini_trend_template_scan(
                trade_date=td,
                top_n=top_n,
                max_workers=max_workers,
                rs_min_percentile=rs_min_percentile,
                only_full_pass=only_full_pass,
                mode=scan_mode,
            ),
        )
        return _sanitize_for_json(payload)
    except Exception as e:
        logger.exception("Minervini template refresh failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/stocks/intraday-rs")
async def api_intraday_rs_get(trade_date: str | None = None):
    """Load last stored intraday RS scan (NIFTY 500 vs NIFTY 50, 15-min cache)."""
    try:
        td = _parse_trade_date(trade_date)
        settings = Settings.from_env(underlying="NIFTY")
        data = load_stored_intraday_relative_strength(settings.data_dir, td)
        if not data:
            return {
                "trade_date": td.isoformat(),
                "stronger": [],
                "weaker": [],
                "nifty": None,
                "message": "No saved intraday RS yet. Click Refresh to compute from 15-min cache.",
            }
        return _sanitize_for_json(data)
    except Exception as e:
        logger.exception("Intraday RS load failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/stocks/intraday-rs/refresh")
async def api_intraday_rs_refresh(
    trade_date: str | None = None,
    max_workers: int = 8,
    top_n: int | None = None,
):
    try:
        td = _parse_trade_date(trade_date)
        loop = asyncio.get_event_loop()
        payload = await loop.run_in_executor(
            None,
            lambda: run_intraday_relative_strength_scan(
                trade_date=td,
                max_workers=max_workers,
                top_n=top_n,
            ),
        )
        return _sanitize_for_json(payload)
    except Exception as e:
        logger.exception("Intraday RS refresh failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/stocks/sector-rs")
async def api_sector_rs_get(trade_date: str | None = None):
    """Load last stored sector RS scan (sectors ranked vs NIFTY 50, with runners/laggards)."""
    try:
        td = _parse_trade_date(trade_date)
        settings = Settings.from_env(underlying="NIFTY")
        data = load_stored_sector_relative_strength(settings.data_dir, td)
        if not data:
            return {
                "trade_date": td.isoformat(),
                "sectors": [],
                "nifty": None,
                "message": "No saved sector RS yet. Click Refresh to compute from 15-min cache.",
            }
        return _sanitize_for_json(data)
    except Exception as e:
        logger.exception("Sector RS load failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/stocks/sector-rs/refresh")
async def api_sector_rs_refresh(trade_date: str | None = None):
    try:
        td = _parse_trade_date(trade_date)
        settings = Settings.from_env(underlying="NIFTY")
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            None,
            lambda: build_nse_sector_stock_map(settings.data_dir, force_refresh=True),
        )
        payload = await loop.run_in_executor(
            None,
            lambda: run_sector_relative_strength_scan(trade_date=td),
        )
        rotation = await loop.run_in_executor(
            None,
            lambda: run_sector_rotation_scan(trade_date=td),
        )
        payload = dict(payload)
        payload["sector_rotation"] = rotation
        return _sanitize_for_json(payload)
    except Exception as e:
        logger.exception("Sector RS refresh failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/stocks/sector-rotation")
async def api_sector_rotation_get(trade_date: str | None = None):
    try:
        td = _parse_trade_date(trade_date)
        settings = Settings.from_env(underlying="NIFTY")
        data = load_stored_sector_rotation(settings.data_dir, td)
        if not data:
            return {
                "trade_date": td.isoformat(),
                "rotations": [],
                "headline": "No sector rotation scan yet. Refresh Sector RS tab.",
            }
        return _sanitize_for_json(data)
    except Exception as e:
        logger.exception("Sector rotation load failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/stocks/sector-rotation/refresh")
async def api_sector_rotation_refresh(trade_date: str | None = None):
    try:
        td = _parse_trade_date(trade_date)
        loop = asyncio.get_event_loop()
        payload = await loop.run_in_executor(
            None,
            lambda: run_sector_rotation_scan(trade_date=td),
        )
        return _sanitize_for_json(payload)
    except Exception as e:
        logger.exception("Sector rotation refresh failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/trend-change")
async def get_trend_change(trade_date: str | None = None, underlying: str | None = None):
    """Intraday trend regime flips (bull ↔ bear) on index 5-min summaries."""
    store = _get_store(underlying)
    settings = _get_settings(underlying)
    date_str = trade_date or date.today().isoformat()

    snap_df = store.load_snapshots()
    sig_df = store.load_signals()
    if snap_df.empty or "timestamp" not in snap_df.columns:
        return {
            "trade_date": date_str,
            "active_side": None,
            "change_count": 0,
            "headline": "No snapshot data yet.",
            "changes": [],
            "series": [],
        }
    snap_df = snap_df[snap_df["timestamp"].astype(str).str.startswith(date_str)]
    if not sig_df.empty and "timestamp" in sig_df.columns:
        sig_df = sig_df[sig_df["timestamp"].astype(str).str.startswith(date_str)]
    summaries = [s for s in build_analysis_summaries(snap_df, sig_df, lookback=settings.lookback_bars) if s]
    return _sanitize_for_json({
        "trade_date": date_str,
        **compute_intraday_trend_change(summaries),
    })


@app.get("/api/stocks/institutional-volume")
async def api_institutional_volume_get(trade_date: str | None = None):
    """Load last stored institutional-volume scan (falls back to most recent file)."""
    try:
        td = _parse_trade_date(trade_date)
        settings = Settings.from_env(underlying="NIFTY")
        data = load_stored_institutional_volume(settings.data_dir, td)
        if not data:
            return {
                "trade_date": td.isoformat(),
                "rows": [],
                "message": "No saved scan yet. Click Refresh to scrape 1y daily bars for NIFTY 500.",
            }
        return _sanitize_for_json(api_institutional_volume_payload(data))
    except Exception as e:
        logger.exception("Institutional volume load failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/stocks/institutional-volume/refresh")
async def api_institutional_volume_refresh(
    trade_date: str | None = None,
    recent_days: int = 60,
    history_days: int = 380,
    max_workers: int = 6,
    stock_limit: int | None = None,
):
    """Fetch ~1y daily bars for NIFTY 500 and label institutional-volume candles."""
    try:
        td = _parse_trade_date(trade_date)
        lim = stock_limit if stock_limit is not None and stock_limit > 0 else None
        loop = asyncio.get_event_loop()
        payload = await loop.run_in_executor(
            None,
            lambda: run_institutional_volume_scan(
                trade_date=td,
                recent_days=recent_days,
                history_days=history_days,
                max_workers=max_workers,
                stock_limit=lim,
            ),
        )
        return _sanitize_for_json(api_institutional_volume_payload(payload))
    except Exception as e:
        logger.exception("Institutional volume refresh failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Combined Signals: Institutional Volume + Fundamentals + News
# ============================================================================


def _read_combined_csv(path: Path) -> list[dict]:
    """Read the combined CSV into a list of dicts (NaN cleansed)."""
    try:
        df = pd.read_csv(path)
    except FileNotFoundError:
        return []
    except Exception as e:
        logger.warning("Combined CSV read failed at %s: %s", path, e)
        return []
    if df.empty:
        return []
    # Replace NaN/Inf with None for clean JSON output
    df = df.where(pd.notnull(df), None)
    rows = df.to_dict(orient="records")
    for r in rows:
        for k, v in list(r.items()):
            if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
                r[k] = None
    return rows


def _latest_combined_csv(data_dir: Path) -> Path | None:
    folder = data_dir / "analysis" / "combined"
    if not folder.exists():
        return None
    files = sorted(folder.glob("combined_*.csv"))
    return files[-1] if files else None


@app.get("/api/stocks/combined-signals")
async def api_combined_signals_get(trade_date: str | None = None):
    """Load the latest cached combined (Institutional + Fundamentals + News) scan."""
    try:
        td = _parse_trade_date(trade_date)
        settings = Settings.from_env(underlying="NIFTY")
        path = combined_signals_csv_path(settings.data_dir, td)
        if not path.exists():
            path = _latest_combined_csv(settings.data_dir) or path
        rows = _read_combined_csv(path) if path.exists() else []
        return _sanitize_for_json({
            "trade_date": td.isoformat(),
            "source_csv": str(path) if path.exists() else None,
            "rows": rows,
            "count": len(rows),
            "message": (
                None if rows
                else "No combined scan yet. Click Refresh to build it from the latest cached "
                     "Institutional Volume + Fundamentals + News scans."
            ),
        })
    except Exception as e:
        logger.exception("Combined signals load failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/stocks/combined-signals/refresh")
async def api_combined_signals_refresh(
    trade_date: str | None = None,
    refresh_fundamentals: bool = False,
    refresh_news: bool = False,
    fundamentals_cache_hours: int = 168,
    news_lookback_days: int = 7,
    stock_limit: int | None = None,
    max_workers_fund: int = 4,
    max_workers_news: int = 8,
):
    """
    Refresh the combined scan. By default this just joins the three cached
    scanner outputs — cheap. Set refresh_fundamentals/refresh_news to first
    re-run those (expensive: scrapes 500 stocks per source).
    """
    try:
        td = _parse_trade_date(trade_date)
        settings = Settings.from_env(underlying="NIFTY")
        lim = stock_limit if stock_limit and stock_limit > 0 else None
        loop = asyncio.get_event_loop()
        fund_payload = news_payload = None
        if refresh_fundamentals:
            fund_payload = await loop.run_in_executor(
                None,
                lambda: run_fundamentals_scan(
                    settings=settings,
                    trade_date=td,
                    cache_max_age_hours=fundamentals_cache_hours,
                    max_workers=max_workers_fund,
                    stock_limit=lim,
                    force_refresh=False,
                ),
            )
        if refresh_news:
            news_payload = await loop.run_in_executor(
                None,
                lambda: run_news_scan(
                    settings=settings,
                    trade_date=td,
                    lookback_days=news_lookback_days,
                    max_workers=max_workers_news,
                    stock_limit=lim,
                ),
            )
        combined_payload = await loop.run_in_executor(
            None,
            lambda: run_combined_scan(settings=settings, trade_date=td),
        )
        path = Path(combined_payload.get("output_csv") or combined_signals_csv_path(settings.data_dir, td))
        rows = _read_combined_csv(path) if path.exists() else []
        return _sanitize_for_json({
            "trade_date": td.isoformat(),
            "source_csv": str(path),
            "rows": rows,
            "count": len(rows),
            "stats": {
                "fundamentals_refreshed": fund_payload,
                "news_refreshed": news_payload,
                "combined": combined_payload,
            },
        })
    except Exception as e:
        logger.exception("Combined signals refresh failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/stocks/fundamentals/refresh")
async def api_fundamentals_refresh(
    trade_date: str | None = None,
    cache_hours: int = 168,
    max_workers: int = 4,
    stock_limit: int | None = None,
    force: bool = False,
):
    """Refresh fundamentals for NIFTY 500 (uses screener.in; weekly cache by default)."""
    try:
        td = _parse_trade_date(trade_date)
        settings = Settings.from_env(underlying="NIFTY")
        lim = stock_limit if stock_limit and stock_limit > 0 else None
        loop = asyncio.get_event_loop()
        payload = await loop.run_in_executor(
            None,
            lambda: run_fundamentals_scan(
                settings=settings,
                trade_date=td,
                cache_max_age_hours=cache_hours,
                max_workers=max_workers,
                stock_limit=lim,
                force_refresh=force,
            ),
        )
        return _sanitize_for_json(payload)
    except Exception as e:
        logger.exception("Fundamentals refresh failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/stocks/news/refresh")
async def api_news_refresh(
    trade_date: str | None = None,
    lookback_days: int = 7,
    max_workers: int = 8,
    stock_limit: int | None = None,
):
    """Refresh Google News for NIFTY 500."""
    try:
        td = _parse_trade_date(trade_date)
        settings = Settings.from_env(underlying="NIFTY")
        lim = stock_limit if stock_limit and stock_limit > 0 else None
        loop = asyncio.get_event_loop()
        payload = await loop.run_in_executor(
            None,
            lambda: run_news_scan(
                settings=settings,
                trade_date=td,
                lookback_days=lookback_days,
                max_workers=max_workers,
                stock_limit=lim,
            ),
        )
        return _sanitize_for_json(payload)
    except Exception as e:
        logger.exception("News refresh failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Market Overview: GIFT Nifty, INDIA VIX, global indices, FII/DII, news, NIFTY plan
# ============================================================================


@app.get("/api/market-overview")
async def api_market_overview_get(trade_date: str | None = None):
    """Return the last cached market overview snapshot (falls back to most recent)."""
    try:
        td = _parse_trade_date(trade_date)
        settings = Settings.from_env(underlying="NIFTY")
        data = load_stored_market_overview(settings.data_dir, td)
        if not data:
            return {
                "trade_date": td.isoformat(),
                "message": "No saved overview yet. Session-scheduler builds one daily at ~09:05 IST.",
                "empty": True,
            }
        return _sanitize_for_json(data)
    except Exception as e:
        logger.exception("Market overview load failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/market-overview/refresh")
async def api_market_overview_refresh(
    trade_date: str | None = None,
    refresh_fii_dii: bool = False,
    news_lookback_hours: int = 24,
    news_per_topic: int = 6,
):
    """Rebuild the overview snapshot. Pass ``refresh_fii_dii=true`` to re-pull NSE."""
    try:
        td = _parse_trade_date(trade_date)
        settings = Settings.from_env(underlying="NIFTY")
        loop = asyncio.get_event_loop()
        payload = await loop.run_in_executor(
            None,
            lambda: run_market_overview_scan(
                settings=settings,
                trade_date=td,
                refresh_fii_dii=refresh_fii_dii,
                news_lookback_hours=news_lookback_hours,
                news_per_topic=news_per_topic,
            ),
        )
        return _sanitize_for_json(payload)
    except Exception as e:
        logger.exception("Market overview refresh failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/jobs")
async def api_jobs_list():
    """Configured background jobs and persisted run state."""
    settings = Settings.from_env()
    jobs = get_job_states(settings.data_dir)
    return {"jobs": _sanitize_for_json(jobs)}


@app.post("/api/jobs/option-chain/run")
async def api_jobs_option_chain_run(trade_date: str | None = None):
    """Manually trigger the option chain scraper job."""
    _require_pipeline_writes()
    try:
        td = _parse_trade_date(trade_date)
        settings = Settings.from_env()
        loop = asyncio.get_event_loop()
        payload = await loop.run_in_executor(
            None,
            lambda: run_option_chain_scraper_job(settings.data_dir, trade_date=td),
        )
        return _sanitize_for_json(payload)
    except Exception as e:
        logger.exception("Option chain job run failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/option-chain/status")
async def api_option_chain_status(underlying: str | None = None, trade_date: str | None = None):
    """Check whether option chain CSV exists for the date/underlying."""
    u_key = _underlying_key(underlying)
    td = _parse_trade_date(trade_date)
    settings = _get_settings(underlying)
    loop = asyncio.get_event_loop()
    status = await loop.run_in_executor(
        None,
        lambda: option_chain_status(settings.data_dir, td, u_key),
    )
    return _sanitize_for_json(status)


@app.get("/api/options-trading/signals")
async def api_options_trading_signals(underlying: str | None = None, trade_date: str | None = None):
    """Load stored options-trading signals (ATM CE/PE EMA breakout)."""
    u_key = _underlying_key(underlying)
    td = _parse_trade_date(trade_date)
    settings = _get_settings(underlying)
    store = _get_store(underlying)
    stored = load_options_trading_signals(settings.data_dir, td, u_key)
    if stored and (stored.get("analysis") is not None or stored.get("signals") is not None):
        stored["strategy_enabled"] = get_enabled_map(store.data_dir)
        return _sanitize_for_json(stored)
    if _read_only_dashboard():
        return _sanitize_for_json({
            "available": False,
            "underlying": u_key,
            "trade_date": td.isoformat(),
            "signals": [],
            "analysis": {"CE": [], "PE": []},
            "message": (
                f"No stored options-trading signals for {u_key}. "
                "Run --session-scheduler (writes signals per underlying)."
            ),
            "strategy_enabled": get_enabled_map(store.data_dir),
        })
    loop = asyncio.get_event_loop()
    payload = await loop.run_in_executor(
        None,
        lambda: run_options_trading_scan(settings.data_dir, td, u_key),
    )
    payload["strategy_enabled"] = get_enabled_map(store.data_dir)
    return _sanitize_for_json(payload)


@app.post("/api/options-trading/refresh")
async def api_options_trading_refresh(underlying: str | None = None, trade_date: str | None = None):
    """Rescan Index_Analysis data and rebuild options-trading signals."""
    _require_pipeline_writes()
    try:
        u_key = _underlying_key(underlying)
        td = _parse_trade_date(trade_date)
        settings = _get_settings(underlying)
        loop = asyncio.get_event_loop()
        payload = await loop.run_in_executor(
            None,
            lambda: run_options_trading_scan(settings.data_dir, td, u_key),
        )
        store = _get_store(underlying)
        payload["strategy_enabled"] = get_enabled_map(store.data_dir)
        return {
            "status": "ok",
            "signals": _sanitize_for_json(payload),
        }
    except Exception as e:
        logger.exception("Options trading refresh failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/underlyings")
async def get_underlyings_list():
    """Return indices only for the main dashboard dropdown. Stocks use /stocks page."""
    return {"underlyings": list_index_underlyings()}


@app.get("/api/stocks/list")
async def api_stocks_list():
    """Return F&O stock names for dropdown (excludes indices)."""
    try:
        settings = Settings.from_env(underlying="NIFTY")
        client = ZerodhaClient(settings)
        names = client.fno_stock_names()
        return {"stocks": names}
    except Exception as e:
        logger.exception("Load F&O stocks failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/stocks/eod-indicators")
async def api_stocks_eod_indicators(trade_date: str | None = None):
    """Load last saved EOD FnO scan (falls back to most recent file)."""
    try:
        td = _parse_trade_date(trade_date)
        settings = Settings.from_env(underlying="NIFTY")
        data = load_stored_eod_indicators(settings.data_dir, td)
        if not data:
            return {
                "trade_date": td.isoformat(),
                "indicators": [],
                "count": 0,
                "failed": [],
                "failed_count": 0,
                "message": "No saved EOD scan yet. Click Refresh EOD to fetch from Kite.",
            }
        return _sanitize_for_json(data)
    except Exception as e:
        logger.exception("EOD indicators load failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/stocks/eod-indicators/refresh")
async def api_stocks_eod_indicators_refresh(
    trade_date: str | None = None,
    limit: int | None = None,
):
    """Run live EOD Kite scan for all liquid FnO symbols (slow) and save to disk."""
    try:
        td = _parse_trade_date(trade_date)
        settings = Settings.from_env(underlying="NIFTY")
        lim = limit if limit is not None and limit > 0 else None
        loop = asyncio.get_event_loop()
        payload = await loop.run_in_executor(
            None,
            lambda: run_and_save_eod_scan(
                settings.data_dir,
                trade_date=td,
                stock_limit=lim,
            ),
        )
        return _sanitize_for_json(payload)
    except Exception as e:
        logger.exception("EOD indicators refresh failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/stocks/30min/signals")
async def api_stocks_30min_signals(stock: str, trade_date: str | None = None):
    """Fetch 30-min data for stock and return signals (on-the-fly, no persistence)."""
    if not stock or not stock.strip():
        raise HTTPException(status_code=400, detail="stock is required")
    stock = stock.strip().upper()
    try:
        sel_date = datetime.strptime(trade_date or date.today().isoformat(), "%Y-%m-%d").date()
    except ValueError:
        sel_date = date.today()
    try:
        loop = asyncio.get_event_loop()
        settings = Settings.from_env(underlying="NIFTY")
        client = ZerodhaClient(settings)
        merged, signals = await loop.run_in_executor(
            None,
            lambda: run_stock_analysis_30min(client, stock, sel_date, include_options=True),
        )
        if not signals:
            return {"signals": [], "latest_actionable": None}
        actionable = [s for s in signals if s.get("signal") in ("BUY", "SELL")]
        latest = actionable[-1] if actionable else None
        return {
            "signals": [_sanitize_for_json(s) for s in signals[-100:]],
            "latest_actionable": _sanitize_for_json(latest),
        }
    except Exception as e:
        logger.exception("30min signals for %s failed: %s", stock, e)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/stocks/30min/analysis-summary")
async def api_stocks_30min_analysis_summary(
    stock: str,
    trade_date: str | None = None,
    timestamp: str | None = None,
):
    """Fetch 30-min data for stock and return analysis summary (on-the-fly)."""
    if not stock or not stock.strip():
        raise HTTPException(status_code=400, detail="stock is required")
    stock = stock.strip().upper()
    try:
        sel_date = datetime.strptime(trade_date or date.today().isoformat(), "%Y-%m-%d").date()
    except ValueError:
        sel_date = date.today()
    try:
        loop = asyncio.get_event_loop()
        settings = Settings.from_env(underlying="NIFTY")
        client = ZerodhaClient(settings)
        merged, signals = await loop.run_in_executor(
            None,
            lambda: run_stock_analysis_30min(client, stock, sel_date, include_options=True),
        )
        if merged is None or merged.empty:
            return {"summaries": [], "selected": None}
        sig_df = pd.DataFrame([_flatten_for_csv(s) for s in signals]) if signals else pd.DataFrame()
        summaries = build_analysis_summaries(merged, sig_df, lookback=min(settings.lookback_bars, 10))
        summaries = [s for s in summaries if s]
        for s in summaries:
            for k, v in s.items():
                s[k] = _sanitize_for_json(v)
        if timestamp:
            selected = next((x for x in summaries if str(x.get("timestamp", "")) == timestamp), None)
            return {"summaries": summaries, "selected": selected}
        return {"summaries": summaries, "selected": summaries[-1] if summaries else None}
    except Exception as e:
        logger.exception("30min analysis for %s failed: %s", stock, e)
        raise HTTPException(status_code=500, detail=str(e))


class IndexSignalMonitorRequest(BaseModel):
    trade_date: str
    timestamp: str
    signal: str
    underlying: str | None = None
    note: str | None = None


@app.get("/api/signals")
async def get_signals(underlying: str | None = None, trade_date: str | None = None):
    store = _get_store(underlying)
    date_str = trade_date or date.today().isoformat()
    try:
        sel_date = datetime.strptime(date_str, "%Y-%m-%d").date()
    except ValueError:
        sel_date = date.today()
        date_str = sel_date.isoformat()
    df = store.load_signals(trade_date=sel_date)
    if df.empty or "signal" not in df.columns:
        return {
            "signals": [],
            "latest_actionable": None,
            "invalidation_supported": store.asset_class == "index",
        }
    df = df[df["signal"].isin(["BUY", "SELL"])]
    if df.empty:
        return {
            "signals": [],
            "latest_actionable": None,
            "invalidation_supported": store.asset_class == "index",
        }
    if "timestamp" in df.columns:
        df = df.sort_values(by="timestamp", ascending=False).head(100)
    signals = df.to_dict(orient="records")
    for s in signals:
        for k, v in s.items():
            s[k] = _json_safe(v)
    if store.asset_class == "index":
        inv = load_invalidated_keys(store.root_data_dir, store.underlying, date_str)
        for s in signals:
            key = f"{str(s.get('timestamp', '')).strip()}|{str(s.get('signal', '')).strip().upper()}"
            s["invalidated"] = key in inv
    else:
        for s in signals:
            s["invalidated"] = False
    latest = store.get_latest_actionable_signal(trade_date=sel_date)
    return {
        "signals": signals,
        "latest_actionable": _sanitize_for_json(latest),
        "invalidation_supported": store.asset_class == "index",
    }


@app.post("/api/signals/invalidate")
async def api_invalidate_index_signal(body: IndexSignalMonitorRequest):
    store = _get_store(body.underlying)
    if store.asset_class != "index":
        raise HTTPException(status_code=400, detail="Strike-out is only available for index underlyings.")
    store_invalidate_signal(
        store.root_data_dir,
        store.underlying,
        body.trade_date,
        body.timestamp,
        body.signal,
        body.note,
    )
    return {"status": "ok"}


@app.post("/api/signals/reinstate")
async def api_reinstate_index_signal(body: IndexSignalMonitorRequest):
    store = _get_store(body.underlying)
    if store.asset_class != "index":
        raise HTTPException(status_code=400, detail="Reinstate is only available for index underlyings.")
    ok = store_reinstate_signal(
        store.root_data_dir,
        store.underlying,
        body.trade_date,
        body.timestamp,
        body.signal,
    )
    if not ok:
        raise HTTPException(status_code=404, detail="No strike-out found for that signal.")
    return {"status": "ok"}


@app.get("/api/analysis-summary")
async def get_analysis_summary(timestamp: str | None = None, underlying: str | None = None, trade_date: str | None = None):
    """Return price action, futures, and option summary per timestamp. Optional ?timestamp= for specific candle, ?trade_date= for date."""
    store = _get_store(underlying)
    settings = _get_settings(underlying)
    snap_df = store.load_snapshots()
    sig_df = store.load_signals()
    if snap_df.empty:
        return {"summaries": [], "selected": None, "hourly_levels": None}
    date_str = trade_date or date.today().isoformat()
    if "timestamp" in snap_df.columns:
        snap_df = snap_df[snap_df["timestamp"].astype(str).str.startswith(date_str)]
    if snap_df.empty:
        return {"summaries": [], "selected": None, "hourly_levels": None}
    if not sig_df.empty and "timestamp" in sig_df.columns:
        sig_df = sig_df[sig_df["timestamp"].astype(str).str.startswith(date_str)]
    summaries = build_analysis_summaries(snap_df, sig_df, lookback=settings.lookback_bars)
    summaries = [s for s in summaries if s]

    trade_d = date.fromisoformat(date_str) if date_str else date.today()
    hourly_levels = hourly_levels_from_snapshots(snap_df)
    if not _read_only_dashboard():
        try:
            client = _get_client(underlying)
            live = fetch_hourly_spot_levels(client, settings, trade_d)
            if live and live.get("support") is not None:
                hourly_levels = live
        except Exception as e:
            logger.debug("Hourly levels via Kite skipped: %s", e)
    snap_records = snap_df.to_dict(orient="records") if not snap_df.empty else None
    summaries = attach_confluence_to_summaries(summaries, hourly_levels, snap_records)
    summaries = attach_volatility_compare(summaries)
    prior_summaries = load_prior_session_summaries(
        settings.data_dir,
        settings.underlying,
        trade_d,
        lookback_bars=settings.lookback_bars,
    )
    prev_close = prior_session_close(settings.data_dir, settings.underlying, trade_d)
    price_action_volume = _sanitize_for_json(compute_price_action_volume(summaries, snap_records))
    pro_print = _sanitize_for_json(compute_pro_print(summaries, snap_records))
    atr_squeeze = _sanitize_for_json(
        compute_atr_squeeze(summaries, prior_summaries=prior_summaries)
    )

    buyers_day = _sanitize_for_json(
        compute_buyers_day(summaries, prev_close=prev_close)
    )
    buyers_day["series"] = _sanitize_for_json(
        compute_buyers_day_series(summaries, prev_close=prev_close)
    )
    try:
        expiry_gamma = _sanitize_for_json(
            compute_expiry_gamma_radar(summaries, trade_date=trade_d, underlying=settings.underlying)
        )
    except Exception as e:
        logger.debug("Expiry gamma radar skipped: %s", e)
        expiry_gamma = None

    trend_change = _sanitize_for_json(compute_intraday_trend_change(summaries))

    for s in summaries:
        for k, v in s.items():
            s[k] = _sanitize_for_json(v)
    if timestamp:
        selected = next((x for x in summaries if str(x.get("timestamp", "")) == timestamp), None)
        return {
            "summaries": summaries,
            "selected": selected,
            "hourly_levels": _sanitize_for_json(hourly_levels),
            "buyers_day": buyers_day,
            "price_action_volume": price_action_volume,
            "pro_print": pro_print,
            "atr_squeeze": atr_squeeze,
            "expiry_gamma": expiry_gamma,
            "trend_change": trend_change,
        }
    return {
        "summaries": summaries,
        "selected": summaries[-1] if summaries else None,
        "hourly_levels": _sanitize_for_json(hourly_levels),
        "buyers_day": buyers_day,
        "price_action_volume": price_action_volume,
        "pro_print": pro_print,
        "atr_squeeze": atr_squeeze,
        "expiry_gamma": expiry_gamma,
        "trend_change": trend_change,
    }


@app.get("/api/expiry-gamma")
async def get_expiry_gamma(
    trade_date: str | None = None,
    underlying: str | None = None,
    window_from: str = "14:30",
    window_to: str = "15:30",
):
    """
    Expiry Gamma Radar + the 2:30-3:30 PM expiry-window candle tape.
    Only meaningful on the underlying's weekly expiry day (NIFTY = Tuesday).
    """
    store = _get_store(underlying)
    settings = _get_settings(underlying)
    date_str = trade_date or date.today().isoformat()
    try:
        trade_d = date.fromisoformat(date_str)
    except ValueError:
        trade_d = date.today()
    expiry = is_expiry_day(trade_d, settings.underlying)

    snap_df = store.load_snapshots()
    sig_df = store.load_signals()
    if snap_df.empty or "timestamp" not in snap_df.columns:
        return {
            "trade_date": date_str,
            "is_expiry_day": expiry,
            "radar": {"active": False, "is_expiry_day": expiry, "state": "OFF",
                      "headline": "No snapshot data." if expiry else "Not an expiry day — gamma radar is off."},
            "window": {"from": window_from, "to": window_to, "rows": []},
        }
    snap_df = snap_df[snap_df["timestamp"].astype(str).str.startswith(date_str)]
    if not sig_df.empty and "timestamp" in sig_df.columns:
        sig_df = sig_df[sig_df["timestamp"].astype(str).str.startswith(date_str)]
    summaries = [s for s in build_analysis_summaries(snap_df, sig_df, lookback=settings.lookback_bars) if s]

    radar = compute_expiry_gamma_radar(summaries, trade_date=trade_d, underlying=settings.underlying)
    tape = expiry_window_tape(summaries, from_hhmm=window_from, to_hhmm=window_to)
    return _sanitize_for_json({
        "trade_date": date_str,
        "is_expiry_day": expiry,
        "radar": radar,
        "window": {"from": window_from, "to": window_to, "rows": tape},
    })


@app.get("/api/trend-ignition")
async def get_trend_ignition(trade_date: str | None = None, underlying: str | None = None):
    """
    Trend Ignition Radar — directional breakout/breakdown detector for any day
    (identifies moves like the 08-Jul-2026 afternoon breakdown), plus the
    per-candle tape for the analysis table.
    """
    store = _get_store(underlying)
    settings = _get_settings(underlying)
    date_str = trade_date or date.today().isoformat()

    snap_df = store.load_snapshots()
    sig_df = store.load_signals()
    if snap_df.empty or "timestamp" not in snap_df.columns:
        return {
            "trade_date": date_str,
            "radar": {"active": False, "state": "IDLE", "headline": "No snapshot data yet."},
            "tape": [],
        }
    snap_df = snap_df[snap_df["timestamp"].astype(str).str.startswith(date_str)]
    if not sig_df.empty and "timestamp" in sig_df.columns:
        sig_df = sig_df[sig_df["timestamp"].astype(str).str.startswith(date_str)]
    summaries = [s for s in build_analysis_summaries(snap_df, sig_df, lookback=settings.lookback_bars) if s]

    radar = compute_trend_ignition(summaries)
    tape = trend_ignition_tape(summaries)
    return _sanitize_for_json({
        "trade_date": date_str,
        "radar": radar,
        "tape": tape,
    })


class AutoTradeConfigRequest(BaseModel):
    auto_trade_enabled: bool | None = None
    lots: int | None = None
    daily_max_loss: float | None = None
    underlying: str | None = None


class StrategyToggleRequest(BaseModel):
    strategy_id: str
    enabled: bool
    underlying: str | None = None


@app.get("/api/options-trading/strategies")
async def api_options_trading_strategies_get(underlying: str | None = None):
    store = _get_store(underlying)
    return _sanitize_for_json({"strategies": list_strategies(store.data_dir)})


@app.post("/api/options-trading/strategies")
async def api_options_trading_strategies_post(req: StrategyToggleRequest):
    store = _get_store(req.underlying)
    try:
        strategies = update_strategy(store.data_dir, req.strategy_id, req.enabled)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    return _sanitize_for_json({"status": "ok", "strategies": strategies})


@app.get("/api/auto-trade/config")
async def api_auto_trade_config_get(underlying: str | None = None):
    store = _get_store(underlying)
    settings = _get_settings(underlying)
    cfg = get_auto_trade_config(store.data_dir)
    cfg["daily_max_loss"] = settings.daily_sl_rupees
    day_pnl = None
    sl_reached = False
    try:
        day_pnl = _get_client(underlying).get_day_pnl()
        sl_reached = day_pnl is not None and day_pnl <= -settings.daily_sl_rupees
    except Exception:
        pass
    cfg["day_pnl"] = day_pnl
    cfg["sl_reached"] = sl_reached
    return _sanitize_for_json(cfg)


@app.post("/api/auto-trade/config")
async def api_auto_trade_config_post(req: AutoTradeConfigRequest):
    store = _get_store(req.underlying)
    updates = {k: v for k, v in req.model_dump().items() if v is not None and k != "underlying"}
    if "lots" in updates:
        updates["lots"] = max(1, min(100, int(updates["lots"])))
    if "daily_max_loss" in updates:
        updates["daily_max_loss"] = _set_daily_max_loss(updates["daily_max_loss"], req.underlying)
    # Allow toggling config anytime; live orders only fire during NSE session.
    if updates.get("auto_trade_enabled") is True and not is_nse_session_active():
        logger.info("Auto entry enabled outside session — orders will wait until %s.", session_label())
    auto_updates = {k: v for k, v in updates.items() if k != "daily_max_loss"}
    cfg = update_auto_trade_config(store.data_dir, **auto_updates) if auto_updates else get_auto_trade_config(store.data_dir)
    settings = _get_settings(req.underlying)
    cfg["daily_max_loss"] = settings.daily_sl_rupees
    return _sanitize_for_json({"status": "ok", **cfg})


@app.post("/api/auto-trade/run-once")
async def api_auto_trade_run_once(underlying: str | None = None):
    """Manual trigger: scan + execute (if enabled) + standard trail check. NSE session only."""
    if not is_nse_session_active():
        raise HTTPException(
            status_code=400,
            detail=f"Outside NSE trading hours ({session_label()}). Auto-trade runs only during the session.",
        )
    settings = _get_settings(underlying)
    client = _get_client(underlying)
    store = _get_store(underlying)
    loop = asyncio.get_event_loop()
    trade_status = await loop.run_in_executor(None, lambda: run_auto_trade_cycle(settings, client, store))
    trail_status = await loop.run_in_executor(None, lambda: run_position_trail_cycle(settings, client, store))
    return _sanitize_for_json({"trade": trade_status, "trail": trail_status})


@app.get("/api/trade-summary")
async def get_trade_summary(underlying: str | None = None):
    client = _get_client(underlying)
    settings = _get_settings(underlying)
    data = client.get_trade_summary()
    if data is None:
        return {
            "day_pnl": None,
            "day_points": None,
            "day_pnl_msg": "Unable to fetch (market may be closed)",
            "daily_sl": settings.daily_sl_rupees,
            "positions": [],
            "orders": [],
        }
    orders = data.get("orders", [])
    if isinstance(orders, list):
        orders = sorted(orders, key=lambda o: str(o.get("order_timestamp") or o.get("exchange_timestamp") or ""), reverse=True)
    positions = data.get("positions", [])
    for p in positions:
        qty = int(p.get("quantity", 0))
        sym = str(p.get("tradingsymbol", "")).replace("NFO:", "")
        if qty and sym:
            pos_underlying = _infer_underlying(sym)
            store = _get_store(pos_underlying)
            rec = get_sl_record(store.data_dir, sym, qty)
            p["sl_trigger"] = rec.get("sl_trigger") if rec else None
            p["sl_order_id"] = rec.get("sl_order_id") if rec else None
            p["entry_price"] = rec.get("entry_price") if rec else None
            p["trail_managed"] = bool(rec.get("sl_order_id")) if rec else False
    day_pnl = data.get("day_pnl", 0.0)
    sl_reached = day_pnl is not None and day_pnl <= -settings.daily_sl_rupees
    day_points = 0.0
    for p in positions:
        qty = abs(int(p.get("quantity", 0)))
        pnl = float(p.get("pnl", 0) or 0)
        if qty > 0:
            p["points"] = round(pnl / qty, 2)
            day_points += p["points"]
        else:
            p["points"] = None
    day_pts_msg = f"{day_points:+.1f} pts" if day_pnl is not None else "—"
    return {
        "day_pnl": day_pnl,
        "day_points": day_points,
        "day_pnl_msg": day_pts_msg,
        "daily_sl": settings.daily_sl_rupees,
        "sl_reached": sl_reached,
        "positions": _sanitize_for_json(positions),
        "orders": _sanitize_for_json(orders),
    }


@app.post("/api/refresh")
async def refresh(request: Request):
    _require_pipeline_writes()
    try:
        body = await request.json()
    except Exception:
        body = {}
    underlying = body.get("underlying") or None
    trade_date_raw = body.get("trade_date")
    try:
        try:
            sel_date = (
                datetime.strptime(str(trade_date_raw), "%Y-%m-%d").date()
                if trade_date_raw
                else date.today()
            )
        except ValueError:
            sel_date = date.today()
        target_date = sel_date.isoformat()
        project_root = Path(__file__).resolve().parent.parent.parent.parent
        cmd = [sys.executable, "-m", "intraday_engine.main", "--date", target_date]
        if underlying:
            cmd.extend(["--underlying", underlying])
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            cwd=project_root,
            env={**os.environ, "PYTHONPATH": "src"},
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await proc.communicate()
        if proc.returncode != 0:
            err = (stderr or stdout or b"").decode("utf-8", errors="replace")
            raise HTTPException(status_code=500, detail=f"Refresh failed: {err}")

        u_key = _underlying_key(underlying)
        options_signals = None
        if u_key in list_index_underlyings():
            try:
                settings = _get_settings(underlying)
                loop = asyncio.get_event_loop()
                options_signals = await loop.run_in_executor(
                    None,
                    lambda: run_options_trading_scan(settings.data_dir, sel_date, u_key),
                )
            except Exception as e:
                logger.warning("Options trading scan failed: %s", e)

        option_chain_info = None

        msg = (
            f"Data fetched and signals generated for {target_date}."
            if sel_date != date.today()
            else "Data fetched and signals generated."
        )
        return {
            "status": "ok",
            "message": msg,
            "trade_date": target_date,
            "option_chain": _sanitize_for_json(option_chain_info) if option_chain_info else None,
            "options_trading": _sanitize_for_json(options_signals) if options_signals else None,
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Refresh failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/strike-oi-change")
async def get_strike_oi_change(underlying: str | None = None, trade_date: str | None = None):
    """Multi-strike OI % change over time from stored option-chain snapshots."""
    u_key = _underlying_key(underlying)
    if u_key not in list_index_underlyings():
        return {"strike_oi_change": None, "message": "Strike OI tracking only for indices (NIFTY, BANKNIFTY)."}
    settings = _get_settings(underlying)
    td = date.fromisoformat(trade_date) if trade_date else date.today()
    try:
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            lambda: compute_strike_oi_change(settings.data_dir, td, u_key),
        )
        return {"strike_oi_change": _sanitize_for_json(result), "status": "ok"}
    except Exception as e:
        logger.exception("Strike OI change failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/futures-oi-buildup")
async def get_futures_oi_buildup(underlying: str | None = None, trade_date: str | None = None):
    """Bar-by-bar futures OI buildup from stored Index_Analysis.csv (5-min bars)."""
    u_key = _underlying_key(underlying)
    if u_key not in list_index_underlyings():
        return {
            "futures_oi_buildup": None,
            "message": "Futures OI buildup only for indices (NIFTY, BANKNIFTY).",
        }
    settings = _get_settings(underlying)
    td = date.fromisoformat(trade_date) if trade_date else date.today()
    try:
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            lambda: compute_futures_oi_buildup(settings.data_dir, td, u_key),
        )
        return {"futures_oi_buildup": _sanitize_for_json(result), "status": "ok"}
    except Exception as e:
        logger.exception("Futures OI buildup failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


class ExecuteRequest(BaseModel):
    lots: int = 2
    underlying: str | None = None


class UpdateSLRequest(BaseModel):
    tradingsymbol: str
    quantity: int
    sl_trigger: float
    underlying: str | None = None


class ExitRequest(BaseModel):
    tradingsymbol: str
    quantity: int
    underlying: str | None = None


class AutoTrailRequest(BaseModel):
    tradingsymbol: str
    quantity: int
    enabled: bool
    underlying: str | None = None


def _infer_underlying(tradingsymbol: str) -> str:
    """Infer underlying from option/future tradingsymbol."""
    s = str(tradingsymbol).upper()
    if s.startswith("BANKNIFTY"):
        return "BANKNIFTY"
    if s.startswith("NIFTY"):
        return "NIFTY"
    # For stocks: underlying is the part before first digit (expiry)
    for i, c in enumerate(s):
        if c.isdigit():
            return s[:i] if s[:i] else "NIFTY"
    return s or "NIFTY"


@app.post("/api/execute")
async def execute(req: ExecuteRequest):
    store = _get_store(req.underlying)
    settings = _get_settings(req.underlying)
    client = _get_client(req.underlying)
    signal = store.get_latest_actionable_signal(trade_date=date.today())
    if not signal:
        raise HTTPException(status_code=400, detail="No actionable BUY/SELL signal. Run Refresh first.")
    option_symbol = signal.get("option_symbol")

    def _valid(s):
        if s is None or s == "":
            return False
        if hasattr(s, "item") and str(s) == "nan":
            return False
        return True

    if not _valid(option_symbol):
        snap_df = store.load_snapshots()
        if snap_df.empty:
            raise HTTPException(status_code=400, detail="No snapshot data. Run Refresh first.")
        # Use snapshot matching signal timestamp (latest signal's option)
        sig_ts = str(signal.get("timestamp", ""))
        match = snap_df[snap_df["timestamp"].astype(str) == sig_ts]
        row = match.iloc[-1] if not match.empty else snap_df.iloc[-1]
        if signal.get("signal") == "BUY":
            option_symbol = row.get("ce_symbol") or row.get("call_symbol", "")
        else:
            option_symbol = row.get("pe_symbol") or row.get("put_symbol", "")
    if not _valid(option_symbol):
        raise HTTPException(status_code=400, detail="Could not resolve option symbol.")

    # Daily SL check: don't place if loss >= daily_sl_rupees
    day_pnl = client.get_day_pnl()
    if day_pnl is not None and day_pnl <= -settings.daily_sl_rupees:
        raise HTTPException(
            status_code=400,
            detail=f"Daily stop loss reached (P&L: ₹{day_pnl:.0f}). No further trades today.",
        )

    # For F&O stocks, resolve to get actual lot size from instrument
    lot_size = settings.lot_size
    try:
        spot_quote = client.quote([settings.spot_symbol])
        if spot_quote and settings.spot_symbol in spot_quote:
            spot_price = float(spot_quote[settings.spot_symbol].get("last_price", 0) or 0)
            if spot_price > 0:
                resolver = InstrumentResolver(client, settings)
                symbols = resolver.resolve(spot_price)
                if symbols.lot_size:
                    lot_size = symbols.lot_size
    except Exception as e:
        logger.debug("Could not resolve lot size, using settings: %s", e)
    quantity = req.lots * lot_size
    transaction_type = str(signal.get("signal", "BUY"))
    sym = str(option_symbol).replace("NFO:", "")
    try:
        quote = client.quote([f"NFO:{sym}"])
        ltp = 0.0
        for k, v in (quote or {}).items():
            if isinstance(v, dict) and "last_price" in v:
                ltp = float(v.get("last_price", 0) or 0)
                break
        if ltp <= 0:
            ltp = 100.0
        order_id = client.place_order(
            tradingsymbol=str(option_symbol),
            exchange="NFO",
            transaction_type=transaction_type,
            quantity=quantity,
        )
        sl_points = settings.default_sl_points
        if transaction_type == "BUY":
            sl_trigger = round(ltp - sl_points, 2)
            sl_side = "SELL"
        else:
            sl_trigger = round(ltp + sl_points, 2)
            sl_side = "BUY"
        sl_order_id = None
        try:
            sl_order_id = client.place_sl_order(
                tradingsymbol=str(option_symbol),
                exchange="NFO",
                transaction_type=sl_side,
                quantity=quantity,
                trigger_price=sl_trigger,
            )
            inst_token = client.get_instrument_token(str(option_symbol)) or 0
            store_sl(_get_store(req.underlying).data_dir, sym, quantity if transaction_type == "BUY" else -quantity, sl_order_id, sl_trigger, inst_token)
        except Exception as sl_err:
            logger.warning("SL order failed (entry placed): %s", sl_err)
        return {"status": "ok", "order_id": order_id, "sl_order_id": sl_order_id, "signal": transaction_type, "quantity": quantity}
    except kite_exceptions.InputException as e:
        msg = str(e)
        if "AMO" in msg or "After Market" in msg:
            msg = "Market is closed. Orders can only be placed during market hours (9:15–15:30)."
        logger.warning("Execute rejected: %s", e)
        raise HTTPException(status_code=400, detail=msg)
    except kite_exceptions.KiteException as e:
        logger.warning("Execute failed (Kite): %s", e)
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception("Execute failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/position/exit")
async def exit_position(req: ExitRequest):
    """Exit position at market."""
    underlying = req.underlying or _infer_underlying(req.tradingsymbol)
    client = _get_client(underlying)
    qty = abs(int(req.quantity))
    sym = str(req.tradingsymbol).replace("NFO:", "")
    side = "SELL" if req.quantity > 0 else "BUY"
    try:
        order_id = client.place_order(
            tradingsymbol=sym,
            exchange="NFO",
            transaction_type=side,
            quantity=qty,
        )
        from intraday_engine.storage.position_sl_store import remove
        remove(_get_store(underlying).data_dir, sym, req.quantity)
        return {"status": "ok", "order_id": order_id}
    except kite_exceptions.KiteException as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.put("/api/position/sl")
async def update_position_sl(req: UpdateSLRequest):
    """Update SL trigger for a position."""
    underlying = req.underlying or _infer_underlying(req.tradingsymbol)
    client = _get_client(underlying)
    store = _get_store(underlying)
    rec = get_sl_record(store.data_dir, req.tradingsymbol, req.quantity)
    if not rec or not rec.get("sl_order_id"):
        raise HTTPException(status_code=400, detail="No SL order found for this position.")
    try:
        client.modify_sl_order(rec["sl_order_id"], req.sl_trigger)
        store_update_sl(store.data_dir, req.tradingsymbol, req.quantity, req.sl_trigger)
        return {"status": "ok", "sl_trigger": req.sl_trigger}
    except kite_exceptions.KiteException as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/position/auto-trail")
async def toggle_auto_trail(req: AutoTrailRequest):
    """Trailing is always on for positions with SL. Kept for API compatibility."""
    return {"status": "ok", "auto_trail": True, "message": "Standard 5-min trail is always enabled."}
