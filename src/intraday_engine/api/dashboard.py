"""FastAPI dashboard for intraday signals and order execution."""

from __future__ import annotations

import asyncio
import logging
import math
import os
import sys
from contextlib import asynccontextmanager
from datetime import date, datetime, timedelta
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from kiteconnect import exceptions as kite_exceptions

from intraday_engine.core.config import Settings
from intraday_engine.engine import DirectionEngine
from intraday_engine.fetch.instrument_resolver import InstrumentResolver
from intraday_engine.fetch.market_data import MarketDataFetcher
from intraday_engine.fetch.zerodha_client import ZerodhaClient
from intraday_engine.storage import DataStore
from intraday_engine.analysis.summary_builder import build_analysis_summaries
from intraday_engine.storage.position_sl_store import get_auto_trail_positions, get_sl as get_sl_record, set_sl as store_sl, update_sl_trigger as store_update_sl, set_auto_trail as store_auto_trail
from intraday_engine.utils.logging_setup import setup_logging

logger = logging.getLogger(__name__)

_auto_trail_task: asyncio.Task | None = None


async def _run_auto_trail_cycle():
    """Trail SL to prev 1min candle low for positions with auto_trail."""
    try:
        _get_engine()
        store = _get_store()
        client = _get_client()
        positions = get_auto_trail_positions(store.data_dir)
        if not positions:
            return
        now = datetime.now()
        to_dt = now.replace(second=0, microsecond=0)
        from_dt = to_dt - timedelta(minutes=5)
        loop = asyncio.get_event_loop()
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
        logger.debug("Auto-trail cycle: %s", e)


async def _auto_trail_loop():
    while True:
        await asyncio.sleep(60)
        await _run_auto_trail_cycle()


@asynccontextmanager
async def _lifespan(app: FastAPI):
    global _auto_trail_task
    _auto_trail_task = asyncio.create_task(_auto_trail_loop())
    yield
    if _auto_trail_task:
        _auto_trail_task.cancel()
        try:
            await _auto_trail_task
        except asyncio.CancelledError:
            pass


app = FastAPI(title="Intraday Direction Engine Dashboard", lifespan=_lifespan)
_engine: DirectionEngine | None = None
_client: ZerodhaClient | None = None
_settings: Settings | None = None


def _get_engine() -> DirectionEngine:
    global _engine, _client, _settings
    if _engine is None:
        _settings = Settings.from_env()
        setup_logging(_settings.log_level, _settings.data_dir)
        _client = ZerodhaClient(_settings)
        resolver = InstrumentResolver(_client, _settings)
        fetcher = MarketDataFetcher(_client, resolver, _settings)
        store = DataStore(_settings.data_dir)
        _engine = DirectionEngine(fetcher, store, _settings)
    return _engine


def _get_client() -> ZerodhaClient:
    _get_engine()
    assert _client is not None
    return _client


def _get_settings() -> Settings:
    _get_engine()
    assert _settings is not None
    return _settings


def _get_store() -> DataStore:
    return _get_engine().store


def _templates_dir() -> Path:
    return Path(__file__).parent / "templates"


@app.get("/", response_class=HTMLResponse)
async def dashboard():
    path = _templates_dir() / "dashboard.html"
    if not path.exists():
        raise HTTPException(status_code=500, detail="Dashboard template not found.")
    return HTMLResponse(path.read_text(encoding="utf-8"))


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


@app.get("/api/signals")
async def get_signals():
    store = _get_store()
    df = store.load_signals()
    if df.empty:
        return {"signals": [], "latest_actionable": None}
    # Filter to current day only
    today_str = date.today().isoformat()
    if "timestamp" in df.columns:
        df = df[df["timestamp"].astype(str).str.startswith(today_str)]
    if df.empty:
        return {"signals": [], "latest_actionable": None}
    # Only BUY and SELL signals, newest first
    df = df[df["signal"].isin(["BUY", "SELL"])]
    if df.empty:
        return {"signals": [], "latest_actionable": None}
    df = df.sort_values(by="timestamp", ascending=False).head(100)
    signals = df.to_dict(orient="records")
    for s in signals:
        for k, v in s.items():
            s[k] = _json_safe(v)
    latest = store.get_latest_actionable_signal(trade_date=date.today())
    return {"signals": signals, "latest_actionable": _sanitize_for_json(latest)}


@app.get("/api/analysis-summary")
async def get_analysis_summary(timestamp: str | None = None):
    """Return price action, futures, and option summary per timestamp. Optional ?timestamp= for specific candle."""
    store = _get_store()
    settings = _get_settings()
    snap_df = store.load_snapshots()
    sig_df = store.load_signals()
    if snap_df.empty:
        return {"summaries": [], "selected": None}
    today_str = date.today().isoformat()
    if "timestamp" in snap_df.columns:
        snap_df = snap_df[snap_df["timestamp"].astype(str).str.startswith(today_str)]
    if snap_df.empty:
        return {"summaries": [], "selected": None}
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
async def get_trade_summary():
    client = _get_client()
    settings = _get_settings()
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
    store = _get_store()
    for p in positions:
        qty = int(p.get("quantity", 0))
        sym = str(p.get("tradingsymbol", "")).replace("NFO:", "")
        if qty and sym:
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
async def refresh():
    try:
        today = date.today().isoformat()
        project_root = Path(__file__).resolve().parent.parent.parent.parent
        proc = await asyncio.create_subprocess_exec(
            sys.executable, "-m", "intraday_engine.main", "--date", today,
            cwd=project_root,
            env={**os.environ, "PYTHONPATH": "src"},
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await proc.communicate()
        if proc.returncode != 0:
            err = (stderr or stdout or b"").decode("utf-8", errors="replace")
            raise HTTPException(status_code=500, detail=f"Refresh failed: {err}")
        return {"status": "ok", "message": "Data fetched and signals generated."}
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Refresh failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


class ExecuteRequest(BaseModel):
    lots: int = 2


class UpdateSLRequest(BaseModel):
    tradingsymbol: str
    quantity: int
    sl_trigger: float


class ExitRequest(BaseModel):
    tradingsymbol: str
    quantity: int


class AutoTrailRequest(BaseModel):
    tradingsymbol: str
    quantity: int
    enabled: bool


@app.post("/api/execute")
async def execute(req: ExecuteRequest):
    store = _get_store()
    settings = _get_settings()
    client = _get_client()
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

    quantity = req.lots * settings.lot_size
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
            store_sl(_get_store().data_dir, sym, quantity if transaction_type == "BUY" else -quantity, sl_order_id, sl_trigger, inst_token)
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
    client = _get_client()
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
        remove(_get_store().data_dir, sym, req.quantity)
        return {"status": "ok", "order_id": order_id}
    except kite_exceptions.KiteException as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.put("/api/position/sl")
async def update_position_sl(req: UpdateSLRequest):
    """Update SL trigger for a position."""
    client = _get_client()
    store = _get_store()
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
    store = _get_store()
    rec = get_sl_record(store.data_dir, req.tradingsymbol, req.quantity)
    if not rec:
        raise HTTPException(status_code=400, detail="No SL record for this position.")
    store_auto_trail(store.data_dir, req.tradingsymbol, req.quantity, req.enabled)
    return {"status": "ok", "auto_trail": req.enabled}
