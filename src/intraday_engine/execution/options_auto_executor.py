"""
Automated execution for Options Trading entry signals (ATM CE/PE + radar ignitions).

Runs on each 5-min candle boundary during NSE session:
  1. Rescan options-trading signals (+ gamma/trend ignitions)
  2. Place MARKET MIS entry + SL-M from the signal when auto-trade is enabled
  3. Standard position management (always on):
       - SL at signal level on entry
       - Exit if entry-candle close < signal-candle high
       - Else trail SL to previous 5-min candle low
"""

from __future__ import annotations

import logging
from datetime import date, datetime, timedelta
from typing import Any

from intraday_engine.analysis.expiry_gamma_radar import compute_expiry_gamma_radar
from intraday_engine.analysis.summary_builder import build_analysis_summaries
from intraday_engine.analysis.trend_ignition import compute_trend_ignition
from intraday_engine.core.config import Settings
from intraday_engine.execution.auto_trade_store import (
    get_config,
    is_executed,
    mark_executed,
    record_run,
)
from intraday_engine.execution.strategy_store import filter_signals_by_strategy
from intraday_engine.fetch.instrument_resolver import InstrumentResolver
from intraday_engine.fetch.zerodha_client import ZerodhaClient
from intraday_engine.research.options_trading_signals import run_options_trading_scan
from intraday_engine.storage import DataStore
from intraday_engine.storage.position_sl_store import (
    get_managed_positions,
    mark_entry_candle_checked,
    remove as remove_sl_record,
    set_sl as store_sl,
    update_sl_trigger as store_update_sl,
)
from intraday_engine.utils.nse_session import is_nse_session_active, now_ist, session_label

logger = logging.getLogger(__name__)


def signal_key(sig: dict[str, Any]) -> str:
    return f"{sig.get('timestamp')}|{sig.get('tradingsymbol')}|{sig.get('signal_type')}"


def _resolve_symbol_from_snapshots(store: DataStore, ts: str, option_type: str, strike: int) -> str:
    snap_df = store.load_snapshots()
    if snap_df.empty:
        return ""
    match = snap_df[snap_df["timestamp"].astype(str) == str(ts)]
    row = match.iloc[-1] if not match.empty else snap_df.iloc[-1]
    if option_type == "CE":
        sym = row.get("ce_symbol") or row.get("call_symbol") or ""
    else:
        sym = row.get("pe_symbol") or row.get("put_symbol") or ""
    return str(sym).replace("NFO:", "").strip()


def _normalize_radar_trigger(t: dict[str, Any], *, signal_type: str) -> dict[str, Any] | None:
    side = t.get("side")
    if not side or t.get("stage") != "IGNITION":
        return None
    strike = int(t.get("strike") or 0)
    entry = float(t.get("entry") or 0)
    sl = float(t.get("stop_ltp") or 0)
    if entry <= 0 or sl <= 0 or sl >= entry:
        return None
    return {
        "timestamp": t.get("timestamp"),
        "signal_type": signal_type,
        "option_type": side,
        "side": "BUY",
        "tradingsymbol": "",  # resolved later
        "strike": strike,
        "entry": entry,
        "stop_loss": sl,
        "risk_pct": t.get("risk_pct"),
        "candle": {"high": entry, "close": entry},
    }


def _signal_candle_fields(signal: dict[str, Any]) -> tuple[float | None, float | None]:
    """Return (signal_high, entry_candle_close) from signal payload."""
    candle = signal.get("candle") or {}
    signal_high = candle.get("high") or signal.get("high")
    entry_close = candle.get("close") or signal.get("entry")
    try:
        sh = float(signal_high) if signal_high is not None else None
    except (TypeError, ValueError):
        sh = None
    try:
        ec = float(entry_close) if entry_close is not None else None
    except (TypeError, ValueError):
        ec = None
    return sh, ec


def collect_entry_signals(
    settings: Settings,
    store: DataStore,
    trade_date: date | None = None,
) -> tuple[list[dict[str, Any]], str | None]:
    """Mirror the dashboard Entry Signals tables (rules + radar ignitions)."""
    td = trade_date or date.today()
    payload = run_options_trading_scan(settings.data_dir, td, settings.underlying)
    signals: list[dict[str, Any]] = []
    for s in payload.get("signals") or []:
        if not s:
            continue
        row = dict(s)
        row.setdefault("signal_type", row.get("signal_type") or "primary")
        row.setdefault("side", "BUY")
        signals.append(row)

    snap_df = store.load_snapshots()
    sig_df = store.load_signals()
    date_str = td.isoformat()
    if not snap_df.empty and "timestamp" in snap_df.columns:
        snap_df = snap_df[snap_df["timestamp"].astype(str).str.startswith(date_str)]
    if not sig_df.empty and "timestamp" in sig_df.columns:
        sig_df = sig_df[sig_df["timestamp"].astype(str).str.startswith(date_str)]
    summaries = [s for s in build_analysis_summaries(snap_df, sig_df, lookback=settings.lookback_bars) if s]

    latest_ts = str(summaries[-1]["timestamp"]) if summaries else None

    try:
        gamma = compute_expiry_gamma_radar(summaries, trade_date=td, underlying=settings.underlying)
        for t in gamma.get("triggers") or []:
            norm = _normalize_radar_trigger(t, signal_type="gamma_ignition")
            if norm:
                norm["tradingsymbol"] = _resolve_symbol_from_snapshots(store, norm["timestamp"], norm["option_type"], norm["strike"])
                signals.append(norm)
    except Exception as e:
        logger.debug("Gamma signals skip: %s", e)

    try:
        trend = compute_trend_ignition(summaries)
        for t in trend.get("triggers") or []:
            norm = _normalize_radar_trigger(t, signal_type="trend_ignition")
            if norm:
                norm["tradingsymbol"] = _resolve_symbol_from_snapshots(store, norm["timestamp"], norm["option_type"], norm["strike"])
                signals.append(norm)
    except Exception as e:
        logger.debug("Trend signals skip: %s", e)

    # Resolve missing tradingsymbols for rule-based signals
    for s in signals:
        sym = str(s.get("tradingsymbol") or "").strip()
        if not sym or sym == "nan":
            s["tradingsymbol"] = _resolve_symbol_from_snapshots(
                store, str(s.get("timestamp", "")), str(s.get("option_type", "CE")), int(s.get("strike") or 0),
            )

    return filter_signals_by_strategy(store.data_dir, signals), latest_ts


def _lot_size(client: ZerodhaClient, settings: Settings) -> int:
    lot_size = settings.lot_size
    try:
        quote = client.quote([settings.spot_symbol])
        if quote and settings.spot_symbol in quote:
            spot = float(quote[settings.spot_symbol].get("last_price", 0) or 0)
            if spot > 0:
                resolver = InstrumentResolver(client, settings)
                symbols = resolver.resolve(spot)
                if symbols.lot_size:
                    return int(symbols.lot_size)
    except Exception as e:
        logger.debug("Lot size resolve: %s", e)
    return int(lot_size)


def _open_position_count(client: ZerodhaClient) -> int:
    positions = client.get_positions() or []
    return sum(1 for p in positions if isinstance(p, dict) and int(p.get("quantity", 0)) != 0)


def _parse_ts(ts: str) -> datetime:
    raw = str(ts).replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(raw)
    except ValueError:
        return now_ist().replace(tzinfo=None)


def _option_candle_at_timestamp(client: ZerodhaClient, token: int, ts: str) -> dict | None:
    """Fetch the 5-min option candle at the signal/entry timestamp."""
    dt = _parse_ts(ts)
    if getattr(dt, "tzinfo", None) is not None:
        dt = dt.replace(tzinfo=None)
    from_dt = dt - timedelta(minutes=10)
    to_dt = dt + timedelta(minutes=1)
    candles = client.historical_data(int(token), from_dt, to_dt, interval="5minute")
    if not candles:
        return None
    target = dt
    best = None
    best_delta = None
    for c in candles:
        c_dt = c.get("date")
        if c_dt is None:
            continue
        if getattr(c_dt, "tzinfo", None) is not None:
            c_dt = c_dt.replace(tzinfo=None)
        delta = abs((c_dt - target).total_seconds())
        if best_delta is None or delta < best_delta:
            best_delta = delta
            best = c
    return best if best_delta is not None and best_delta <= 300 else (candles[-1] if candles else None)


def _close_position(client: ZerodhaClient, store: DataStore, rec: dict[str, Any], *, reason: str) -> dict[str, Any]:
    sym = str(rec.get("tradingsymbol") or "").replace("NFO:", "").strip()
    qty = int(rec.get("quantity") or 0)
    if not sym or qty <= 0:
        raise ValueError("Invalid position record")

    sl_order_id = rec.get("sl_order_id")
    if sl_order_id:
        try:
            client.cancel_order(str(sl_order_id))
        except Exception as e:
            logger.debug("Cancel SL before exit %s: %s", sym, e)

    order_id = client.place_order(
        tradingsymbol=sym,
        exchange="NFO",
        transaction_type="SELL",
        quantity=qty,
        product="MIS",
        order_type="MARKET",
    )
    remove_sl_record(store.data_dir, sym, qty)
    logger.info("Closed %s (%s): %s", sym, reason, order_id)
    return {"sym": sym, "action": "exit", "reason": reason, "order_id": order_id}


def _entry_candle_failed(client: ZerodhaClient, rec: dict[str, Any]) -> tuple[bool, float | None]:
    """
    True when entry-candle close < signal-candle high.
    Uses stored values, refined from live 5-min candle when available.
    """
    signal_high = float(rec.get("signal_high") or 0)
    if signal_high <= 0:
        return False, None

    entry_close = float(rec.get("entry_candle_close") or 0)
    token = rec.get("instrument_token")
    ts = rec.get("signal_timestamp")
    if token and ts:
        candle = _option_candle_at_timestamp(client, int(token), str(ts))
        if candle:
            entry_close = float(candle.get("close", 0) or entry_close)

    if entry_close <= 0:
        return False, entry_close
    return entry_close < signal_high, entry_close


def execute_entry_signal(
    client: ZerodhaClient,
    settings: Settings,
    store: DataStore,
    signal: dict[str, Any],
    *,
    lots: int = 1,
) -> dict[str, Any]:
    """Place MARKET MIS BUY + SL-M from signal entry/SL (NSE session only)."""
    if not is_nse_session_active():
        raise ValueError(f"Outside NSE trading hours ({session_label()})")

    sym = str(signal.get("tradingsymbol") or "").replace("NFO:", "").strip()
    if not sym:
        raise ValueError("Missing tradingsymbol on signal")

    entry_ref = float(signal.get("entry") or 0)
    sl_trigger = float(signal.get("stop_loss") or 0)
    if entry_ref <= 0 or sl_trigger <= 0 or sl_trigger >= entry_ref:
        raise ValueError(f"Invalid entry/SL: entry={entry_ref} sl={sl_trigger}")

    day_pnl = client.get_day_pnl()
    if day_pnl is not None and day_pnl <= -settings.daily_sl_rupees:
        raise ValueError(f"Daily stop loss reached (P&L ₹{day_pnl:.0f})")

    quantity = lots * _lot_size(client, settings)
    quote = client.quote([f"NFO:{sym}"])
    ltp = 0.0
    for v in (quote or {}).values():
        if isinstance(v, dict):
            ltp = float(v.get("last_price", 0) or 0)
            break
    if ltp <= 0:
        ltp = entry_ref

    signal_high, entry_candle_close = _signal_candle_fields(signal)
    signal_ts = str(signal.get("timestamp") or "")

    order_id = client.place_order(
        tradingsymbol=sym,
        exchange="NFO",
        transaction_type="BUY",
        quantity=quantity,
        product="MIS",
        order_type="MARKET",
    )

    sl_order_id = None
    inst_token = client.get_instrument_token(sym) or 0
    if inst_token and signal_ts and (signal_high is None or entry_candle_close is None):
        candle = _option_candle_at_timestamp(client, int(inst_token), signal_ts)
        if candle:
            if signal_high is None:
                signal_high = float(candle.get("high", 0) or 0) or None
            if entry_candle_close is None:
                entry_candle_close = float(candle.get("close", 0) or 0) or None

    try:
        sl_order_id = client.place_sl_order(
            tradingsymbol=sym,
            exchange="NFO",
            transaction_type="SELL",
            quantity=quantity,
            trigger_price=sl_trigger,
            product="MIS",
        )
        store_sl(
            store.data_dir, sym, quantity, sl_order_id, sl_trigger, inst_token,
            entry_price=ltp, initial_sl=sl_trigger,
            signal_key=signal_key(signal),
            signal_high=signal_high,
            signal_timestamp=signal_ts or None,
            entry_candle_close=entry_candle_close,
            entry_candle_checked=False,
        )
    except Exception as sl_err:
        logger.warning("Auto-trade SL failed for %s: %s", sym, sl_err)

    result: dict[str, Any] = {
        "order_id": order_id,
        "sl_order_id": sl_order_id,
        "tradingsymbol": sym,
        "quantity": quantity,
        "entry_ltp": ltp,
        "sl_trigger": sl_trigger,
        "signal_type": signal.get("signal_type"),
        "timestamp": signal.get("timestamp"),
        "signal_high": signal_high,
        "entry_candle_close": entry_candle_close,
    }

    if sl_order_id and signal_high and entry_candle_close and entry_candle_close < signal_high:
        try:
            rec = {
                "tradingsymbol": sym,
                "quantity": quantity,
                "sl_order_id": sl_order_id,
            }
            exit_info = _close_position(client, store, rec, reason="entry_close_below_signal_high")
            result["exited"] = True
            result["exit"] = exit_info
        except Exception as e:
            logger.warning("Immediate exit failed %s: %s", sym, e)

    return result


def run_auto_trade_cycle(
    settings: Settings,
    client: ZerodhaClient,
    store: DataStore,
) -> dict[str, Any]:
    """Scan + optionally execute fresh signals on the latest 5-min candle (NSE session only)."""
    status: dict[str, Any] = {
        "skipped": None,
        "scanned": 0,
        "candidates": 0,
        "executed": [],
        "errors": [],
    }

    if not is_nse_session_active():
        status["skipped"] = "market_closed"
        return status

    cfg = get_config(store.data_dir)
    if not cfg.get("auto_trade_enabled"):
        status["skipped"] = "auto_trade_disabled"
        record_run(store.data_dir, status)
        return status

    signals, latest_ts = collect_entry_signals(settings, store)
    status["scanned"] = len(signals)
    status["latest_candle"] = latest_ts

    if not latest_ts:
        status["skipped"] = "no_candles"
        record_run(store.data_dir, status)
        return status

    fresh = [s for s in signals if str(s.get("timestamp")) == latest_ts]
    status["candidates"] = len(fresh)

    lots = int(cfg.get("lots") or 1)
    max_positions = 2

    for sig in fresh:
        key = signal_key(sig)
        if is_executed(store.data_dir, key):
            continue
        sym = str(sig.get("tradingsymbol") or "").strip()
        if not sym:
            status["errors"].append({"key": key, "error": "no tradingsymbol"})
            continue
        if _open_position_count(client) >= max_positions:
            status["skipped"] = "max_positions"
            break
        try:
            result = execute_entry_signal(client, settings, store, sig, lots=lots)
            mark_executed(store.data_dir, key, {**result, "signal_key": key})
            status["executed"].append(result)
            logger.info("Auto-trade executed %s %s", sym, key)
        except Exception as e:
            logger.warning("Auto-trade failed %s: %s", key, e)
            status["errors"].append({"key": key, "error": str(e)})

    record_run(store.data_dir, status)
    return status


def _prev_5m_candle_low(client: ZerodhaClient, token: int) -> float | None:
    """Previous completed 5-min candle low for the option."""
    now = now_ist().replace(second=0, microsecond=0)
    if getattr(now, "tzinfo", None) is not None:
        now = now.replace(tzinfo=None)
    to_dt = now - timedelta(minutes=5)
    from_dt = to_dt - timedelta(minutes=30)
    candles = client.historical_data(int(token), from_dt, to_dt, interval="5minute")
    if not candles:
        return None
    prev = candles[-1]
    low = float(prev.get("low", 0) or 0)
    return low if low > 0 else None


def run_position_trail_cycle(
    settings: Settings,
    client: ZerodhaClient,
    store: DataStore,
) -> dict[str, Any]:
    """
    Standard position management (always on) every 5 min:
      1. Exit if entry-candle close < signal-candle high
      2. Else trail SL to previous 5-min candle low
    """
    if not is_nse_session_active():
        return {"skipped": "market_closed", "updated": []}

    updated: list[dict[str, Any]] = []
    for rec in get_managed_positions(store.data_dir):
        sym = rec.get("tradingsymbol")
        qty = int(rec.get("quantity") or 0)
        sl_order_id = rec.get("sl_order_id")
        current_sl = float(rec.get("sl_trigger") or 0)
        if not sym or qty <= 0 or not sl_order_id:
            continue

        try:
            if not rec.get("entry_candle_checked"):
                failed, entry_close = _entry_candle_failed(client, rec)
                mark_entry_candle_checked(store.data_dir, sym, qty)
                if failed:
                    exit_info = _close_position(
                        client, store, rec, reason="entry_close_below_signal_high",
                    )
                    updated.append({**exit_info, "entry_close": entry_close, "signal_high": rec.get("signal_high")})
                    continue

            token = rec.get("instrument_token")
            if not token:
                continue
            swing_low = _prev_5m_candle_low(client, int(token))
            if swing_low and swing_low > current_sl:
                client.modify_sl_order(str(sl_order_id), swing_low)
                store_update_sl(store.data_dir, sym, qty, swing_low)
                updated.append({"sym": sym, "action": "swing_trail", "sl": swing_low})
                logger.info("Swing trail: %s SL -> %.2f", sym, swing_low)
        except Exception as e:
            logger.warning("Trail cycle failed %s: %s", sym, e)

    return {"updated": updated}


# Backward-compatible alias
run_rr_swing_trail_cycle = run_position_trail_cycle
