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
    load_config,
    save_config,
    validate_no_unknown_keys,
)
from intraday_engine.core.underlyings import list_index_underlyings
from intraday_engine.engine import DirectionEngine
from intraday_engine.fetch.instrument_resolver import InstrumentResolver
from intraday_engine.fetch.market_data import MarketDataFetcher
from intraday_engine.fetch.zerodha_client import ZerodhaClient
from intraday_engine.storage import DataStore, load_signal_rows
from intraday_engine.storage.data_store import _flatten_for_csv
from intraday_engine.analysis.summary_builder import build_analysis_summaries
from intraday_engine.eod.eod_fetcher import run_eod_scan
from intraday_engine.engine.stock_cycle_runner import run_stocks_15min_cycle
from intraday_engine.engine.stock_signal_engine import run_stock_analysis_30min
from intraday_engine.orb.orb_scanner import run_orb_scan, run_pinbar_scan
from intraday_engine.scanner.fno_intraday_buy_scanner import (
    load_stored_intraday_scan,
    run_fno_intraday_scan,
)
from intraday_engine.research.tomorrow_watchlist_scanner import (
    load_stored_tomorrow_watchlist_fresh_or_previous,
    run_tomorrow_watchlist_scan,
)
from intraday_engine.gamma.huge_move_predictor import HugeMovePredictor
from intraday_engine.storage.position_sl_store import get_auto_trail_positions, get_sl as get_sl_record, set_sl as store_sl, update_sl_trigger as store_update_sl, set_auto_trail as store_auto_trail
from intraday_engine.storage.signal_invalidation_store import invalidate as store_invalidate_signal, load_invalidated_keys, reinstate as store_reinstate_signal
from intraday_engine.utils.logging_setup import setup_logging

logger = logging.getLogger(__name__)

_auto_trail_task: asyncio.Task | None = None


def _auto_trail_underlyings() -> list[str]:
    """Underlyings to check for auto-trail: indices + any with data subdirs."""
    base = list_index_underlyings()
    try:
        settings = _get_settings("NIFTY")
        data_path = settings.data_dir
        if data_path.exists():
            for d in data_path.iterdir():
                if d.is_dir() and not d.name.startswith("."):
                    u = d.name.upper()
                    if u not in base:
                        base.append(u)
    except Exception:
        pass
    return base


async def _run_auto_trail_cycle():
    """Trail SL to prev 1min candle low for positions with auto_trail."""
    try:
        now = datetime.now()
        to_dt = now.replace(second=0, microsecond=0)
        from_dt = to_dt - timedelta(minutes=5)
        loop = asyncio.get_event_loop()
        for u in _auto_trail_underlyings():
            try:
                store = _get_store(u)
                client = _get_client(u)
                positions = get_auto_trail_positions(store.data_dir)
                if not positions:
                    continue
                for rec in positions:
                    try:
                        token = rec.get("instrument_token")
                        if not token:
                            continue
                        candles = await loop.run_in_executor(
                            None,
                            lambda t=token, f=from_dt, td=to_dt: client.historical_data(int(t), f, td, interval="minute"),
                        )
                        if not candles or len(candles) < 2:
                            continue
                        prev = candles[-2]
                        low = float(prev.get("low", 0) or 0)
                        if low <= 0:
                            continue
                        sl_order_id = rec.get("sl_order_id")
                        current_sl = rec.get("sl_trigger")
                        qty = rec.get("quantity", 0)
                        is_long = qty > 0
                        if is_long and low > (current_sl or 0):
                            await loop.run_in_executor(None, lambda oid=sl_order_id, l=low: client.modify_sl_order(oid, l))
                            store_update_sl(store.data_dir, rec["tradingsymbol"], qty, low)
                            logger.info("Auto-trail: %s SL -> %.2f", rec["tradingsymbol"], low)
                        elif not is_long and low < (current_sl or float("inf")):
                            await loop.run_in_executor(None, lambda oid=sl_order_id, l=low: client.modify_sl_order(oid, l))
                            store_update_sl(store.data_dir, rec["tradingsymbol"], -qty, low)
                            logger.info("Auto-trail: %s SL -> %.2f", rec["tradingsymbol"], low)
                    except Exception as e:
                        logger.debug("Auto-trail skip %s: %s", rec.get("tradingsymbol"), e)
            except Exception as e:
                logger.debug("Auto-trail cycle %s: %s", u, e)
    except Exception as e:
        logger.debug("Auto-trail cycle: %s", e)


async def _auto_trail_loop():
    while True:
        await asyncio.sleep(60)
        await _run_auto_trail_cycle()


@asynccontextmanager
async def _lifespan(app: FastAPI):
    global _auto_trail_task
    try:
        ensure_config_file()
    except Exception as e:
        logger.warning("Could not ensure config.json: %s", e)
    _auto_trail_task = asyncio.create_task(_auto_trail_loop())
    yield
    if _auto_trail_task:
        _auto_trail_task.cancel()
        try:
            await _auto_trail_task
        except asyncio.CancelledError:
            pass


app = FastAPI(title="Intraday Direction Engine Dashboard", lifespan=_lifespan)
_engines: dict[str, DirectionEngine] = {}
_clients: dict[str, ZerodhaClient] = {}
_settings_cache: dict[str, Settings] = {}


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


def _templates_dir() -> Path:
    return Path(__file__).parent / "templates"


@app.get("/", response_class=HTMLResponse)
async def dashboard():
    path = _templates_dir() / "dashboard.html"
    if not path.exists():
        raise HTTPException(status_code=500, detail="Dashboard template not found.")
    return HTMLResponse(path.read_text(encoding="utf-8"))


@app.get("/stocks", response_class=HTMLResponse)
async def stocks_dashboard():
    """F&O stocks scanner dashboard (15-min timeframe)."""
    path = _templates_dir() / "stocks_dashboard.html"
    if not path.exists():
        raise HTTPException(status_code=500, detail="Stocks dashboard template not found.")
    return HTMLResponse(path.read_text(encoding="utf-8"))


@app.get("/tunables", response_class=HTMLResponse)
async def tunables_page():
    """Edit project config.json (tunables) in the browser."""
    path = _templates_dir() / "tunables.html"
    if not path.exists():
        raise HTTPException(status_code=500, detail="Tunables template not found.")
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


def _load_stored_stock_signals() -> list[dict]:
    """Load unique 15-min signals for today, preserving full-day history."""
    settings = Settings.from_env(underlying="NIFTY")
    today_str = date.today().isoformat()
    all_signals: list[dict] = []
    try:
        df = load_signal_rows(settings.data_dir, trade_date=date.today(), asset_class="stock")
        if df.empty or "timestamp" not in df.columns:
            return []
        df = df[df["timestamp"].astype(str).str.startswith(today_str)]
        if df.empty:
            return []
        grouped = df.groupby("underlying", sort=True)
        for underlying, rows in grouped:
            rows = rows.copy()
            if "strategy" not in rows.columns:
                rows["strategy"] = rows.apply(_infer_strategy, axis=1)
            else:
                rows["strategy"] = rows["strategy"].fillna("DIRECTIONAL").astype(str)
                rows["strategy"] = rows.apply(
                    lambda row: _infer_strategy(row) if str(row.get("strategy") or "").strip() in {"", "None", "nan"} else str(row.get("strategy")),
                    axis=1,
                )
            if "strategy_label" not in rows.columns:
                rows["strategy_label"] = rows.apply(_infer_strategy_label, axis=1)
            else:
                rows["strategy_label"] = rows.apply(
                    lambda row: _infer_strategy_label(row)
                    if str(row.get("strategy_label") or "").strip() in {"", "None", "nan"}
                    else str(row.get("strategy_label")),
                    axis=1,
                )
            rows = rows.sort_values(by="timestamp", ascending=True)
            first_by_key: dict[str, pd.Series] = {}
            latest_by_key: dict[str, pd.Series] = {}
            for _, row in rows.iterrows():
                key = _signal_identity_key(row)
                latest_by_key[key] = row
                if str(row.get("signal") or "") in {"BUY", "SELL"} and key not in first_by_key:
                    first_by_key[key] = row
            for key, first_row in first_by_key.items():
                row = first_row.to_dict()
                latest_row = latest_by_key.get(key)
                fresh_until = str(row.get("fresh_until") or "").strip()
                if fresh_until and fresh_until not in {"", "None", "nan"}:
                    try:
                        row["active_now"] = datetime.now() <= pd.to_datetime(fresh_until).to_pydatetime()
                    except Exception:
                        row["active_now"] = False
                else:
                    row["active_now"] = str(latest_row.get("signal") or "") in {"BUY", "SELL"} if latest_row is not None else False
                row["stock"] = str(underlying).upper()
                all_signals.append(row)
    except Exception:
        return []
    return all_signals


def _signal_identity_key(row: pd.Series) -> str:
    strategy = str(row.get("strategy") or _infer_strategy(row))
    strategy_label = str(row.get("strategy_label") or _infer_strategy_label(row))
    trigger_ts = str(row.get("trigger_timestamp") or "").strip()
    if strategy == "FAILED_BIAS" and trigger_ts and trigger_ts not in {"", "None", "nan"}:
        return f"{strategy}|{strategy_label}|{trigger_ts}"
    ts = str(row.get("timestamp") or "").strip()
    signal = str(row.get("signal") or "").strip()
    return f"{strategy}|{strategy_label}|{signal}|{ts}"


def _infer_strategy(row: pd.Series) -> str:
    pattern = str(row.get("pattern") or "").strip()
    bias = str(row.get("bias") or "").strip()
    if pattern.startswith("FAILED_") or bias.startswith("FAILED_") or bias == "FAILED_BIAS":
        return "FAILED_BIAS"
    return "DIRECTIONAL"


def _infer_strategy_label(row: pd.Series) -> str:
    pattern = str(row.get("pattern") or "").strip()
    if pattern:
        return pattern
    strategy = _infer_strategy(row)
    signal = str(row.get("signal") or "").strip()
    bias = str(row.get("bias") or "").strip().replace(" ", "_")
    if strategy == "FAILED_BIAS":
        return bias or "FAILED_BIAS"
    if signal in {"BUY", "SELL"} and bias:
        return f"DIRECTIONAL_{signal}_{bias}"
    return strategy


@app.get("/api/stocks/signals")
async def api_stocks_signals():
    """Return stored 15-min signals for all F&O stocks (from scheduled cycle)."""
    try:
        signals = _load_stored_stock_signals()
        buy = sorted([s for s in signals if str(s.get("signal")) == "BUY"], key=lambda x: float(x.get("confidence") or 0), reverse=True)
        sell = sorted([s for s in signals if str(s.get("signal")) == "SELL"], key=lambda x: float(x.get("confidence") or 0), reverse=True)
        no_trade = [s for s in signals if str(s.get("signal")) not in ("BUY", "SELL")]
        return {
            "signals": [_sanitize_for_json(s) for s in signals],
            "buy": [_sanitize_for_json(s) for s in buy],
            "sell": [_sanitize_for_json(s) for s in sell],
            "no_trade": [_sanitize_for_json(s) for s in no_trade],
        }
    except Exception as e:
        logger.exception("Load stock signals failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


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


@app.get("/api/stocks/intraday-signals")
async def api_stocks_intraday_signals(trade_date: str | None = None):
    """Stored combined intraday scan: watchlists from 60m/15m RSI, signals from 5m confirmations."""
    try:
        sel_date = date.today()
        if trade_date:
            try:
                sel_date = datetime.strptime(trade_date, "%Y-%m-%d").date()
            except ValueError:
                pass
        settings = Settings.from_env(underlying="NIFTY")
        data = load_stored_intraday_scan(settings.data_dir, sel_date)
        if not data:
            return {
                "trade_date": sel_date.isoformat(),
                "signals": [],
                "buy_watchlist": [],
                "sell_watchlist": [],
                "buy_signals": [],
                "sell_signals": [],
                "message": "No stored run yet. Open Intraday Signals and click Refresh.",
            }
        return _sanitize_for_json(data)
    except Exception as e:
        logger.exception("Intraday signals load failed: %s", e)
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


@app.post("/api/stocks/intraday-signals/refresh")
async def api_stocks_intraday_signals_refresh(limit: int | None = None):
    """Run a single intraday scan for watchlists and BUY/SELL signals."""
    try:
        loop = asyncio.get_event_loop()
        payload = await loop.run_in_executor(
            None,
            lambda: run_fno_intraday_scan(trade_date=date.today(), stock_limit=limit),
        )
        return _sanitize_for_json(payload)
    except Exception as e:
        logger.exception("Intraday signals refresh failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/stocks/refresh")
async def api_stocks_refresh(limit: int | None = None):
    """Run 15-min data fetch and signal generation for all F&O stocks."""
    try:
        loop = asyncio.get_event_loop()
        n = await loop.run_in_executor(
            None,
            lambda: run_stocks_15min_cycle(trade_date=date.today(), stock_limit=limit),
        )
        return {"status": "ok", "message": f"Processed {n} stocks.", "count": n}
    except Exception as e:
        logger.exception("Stocks refresh failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/stocks/refresh-all")
async def api_stocks_refresh_all(
    limit: int | None = None,
    top_n: int = 20,
    max_workers: int = 4,
):
    """
    Single refresh entrypoint for stock dashboard tabs:
    - 15-min stock signals
    - Intraday signals
    - Tomorrow watchlist (15m aggregated to 1h/1d inside scanner)
    """
    try:
        loop = asyncio.get_event_loop()
        td = date.today()

        stock_count = await loop.run_in_executor(
            None,
            lambda: run_stocks_15min_cycle(trade_date=td, stock_limit=limit),
        )
        intraday_payload = await loop.run_in_executor(
            None,
            lambda: run_fno_intraday_scan(trade_date=td, stock_limit=limit),
        )
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
            "message": "Shared refresh complete for stock tabs.",
            "trade_date": td.isoformat(),
            "stocks": {
                "processed": int(stock_count or 0),
            },
            "intraday": {
                "universe": len(intraday_payload.get("symbols_scanned", []) or []),
                "buy_watchlist": len(intraday_payload.get("buy_watchlist", []) or []),
                "sell_watchlist": len(intraday_payload.get("sell_watchlist", []) or []),
                "buy_signals": len(intraday_payload.get("buy_signals", []) or []),
                "sell_signals": len(intraday_payload.get("sell_signals", []) or []),
                "failed": len(intraday_payload.get("failed_symbols", []) or []),
            },
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
async def api_stocks_eod_indicators(trade_date: str | None = None, limit: int | None = None):
    """Return EOD FnO market indicators for all liquid FnO symbols."""
    try:
        sel_date = None
        if trade_date:
            try:
                sel_date = datetime.strptime(trade_date, "%Y-%m-%d").date()
            except ValueError:
                pass
        loop = asyncio.get_event_loop()
        results, failed = await loop.run_in_executor(
            None,
            lambda: run_eod_scan(trade_date=sel_date, stock_limit=limit),
        )
        return {
            "indicators": [_sanitize_for_json(r) for r in results],
            "count": len(results),
            "failed": [_sanitize_for_json(f) for f in failed],
            "failed_count": len(failed),
        }
    except Exception as e:
        logger.exception("EOD indicators failed: %s", e)
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
        return {"summaries": [], "selected": None}
    date_str = trade_date or date.today().isoformat()
    if "timestamp" in snap_df.columns:
        snap_df = snap_df[snap_df["timestamp"].astype(str).str.startswith(date_str)]
    if snap_df.empty:
        return {"summaries": [], "selected": None}
    if not sig_df.empty and "timestamp" in sig_df.columns:
        sig_df = sig_df[sig_df["timestamp"].astype(str).str.startswith(date_str)]
    summaries = build_analysis_summaries(snap_df, sig_df, lookback=settings.lookback_bars)
    summaries = [s for s in summaries if s]
    for s in summaries:
        for k, v in s.items():
            s[k] = _sanitize_for_json(v)
    if timestamp:
        selected = next((x for x in summaries if str(x.get("timestamp", "")) == timestamp), None)
        return {"summaries": summaries, "selected": selected}
    return {"summaries": summaries, "selected": summaries[-1] if summaries else None}


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
            p["auto_trail"] = rec.get("auto_trail", False) if rec else False
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
    try:
        body = await request.json()
    except Exception:
        body = {}
    underlying = body.get("underlying") or None
    try:
        today = date.today().isoformat()
        project_root = Path(__file__).resolve().parent.parent.parent.parent
        cmd = [sys.executable, "-m", "intraday_engine.main", "--date", today]
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

        # Run huge move prediction for index underlyings
        huge_move = None
        u_key = _underlying_key(underlying)
        if u_key in list_index_underlyings():
            try:
                loop = asyncio.get_event_loop()
                huge_move = await loop.run_in_executor(
                    None,
                    lambda: _run_huge_move_prediction(u_key),
                )
            except Exception as e:
                logger.warning("Huge move prediction failed: %s", e)

        return {
            "status": "ok",
            "message": "Data fetched and signals generated.",
            "huge_move": _sanitize_for_json(huge_move) if huge_move else None,
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Refresh failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


def _run_huge_move_prediction(underlying: str) -> dict | None:
    """Run option chain capture + huge move prediction. Returns dict for API or None."""
    settings = Settings.from_env(underlying=underlying)
    client = ZerodhaClient(settings)
    predictor = HugeMovePredictor(client, settings)
    pred = predictor.predict(num_strikes=5, use_stored=False)
    if pred is None:
        return None
    return {
        "direction": pred.direction,
        "confidence": pred.confidence,
        "prob_huge_up": pred.prob_huge_up,
        "prob_huge_down": pred.prob_huge_down,
        "prob_no_move": pred.prob_no_move,
        "pcr_oi": pred.pcr_oi,
        "pcr_volume": pred.pcr_volume,
        "max_pain": pred.max_pain,
        "spot_vs_max_pain": pred.spot_vs_max_pain,
        "suggested_strike": pred.suggested_strike,
        "reasons": pred.reasons,
    }


@app.get("/api/huge-move")
async def get_huge_move(underlying: str | None = None):
    """Fetch option chain and run huge move prediction. Returns probability of 100+ point move."""
    u_key = _underlying_key(underlying)
    if u_key not in list_index_underlyings():
        return {"huge_move": None, "message": "Huge move analysis only for indices (NIFTY, BANKNIFTY)."}
    try:
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            lambda: _run_huge_move_prediction(u_key),
        )
        return {"huge_move": _sanitize_for_json(result), "status": "ok"}
    except Exception as e:
        logger.exception("Huge move failed: %s", e)
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
    """Toggle auto-trail for a position."""
    underlying = req.underlying or _infer_underlying(req.tradingsymbol)
    store = _get_store(underlying)
    rec = get_sl_record(store.data_dir, req.tradingsymbol, req.quantity)
    if not rec:
        raise HTTPException(status_code=400, detail="No SL record for this position.")
    store_auto_trail(store.data_dir, req.tradingsymbol, req.quantity, req.enabled)
    return {"status": "ok", "auto_trail": req.enabled}
