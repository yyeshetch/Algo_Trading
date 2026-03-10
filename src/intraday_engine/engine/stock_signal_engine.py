"""Generate signals for a single F&O stock from 15-min data (spot + fut, optional options)."""

from __future__ import annotations

import logging
from datetime import date
from typing import Any, Dict

import pandas as pd

from intraday_engine.analysis.day_bias import probable_day_bias
from intraday_engine.analysis.momentum import momentum_direction
from intraday_engine.analysis.scoring import score_signal
from intraday_engine.analysis.sideways import is_sideways_day
from intraday_engine.analysis.support_resistance import calculate_support_resistance
from intraday_engine.analysis.trade_plan import build_trade_plan
from intraday_engine.core.config import Settings
from intraday_engine.features.feature_engineering import compute_features
from intraday_engine.fetch.stock_market_data import fetch_stock_15min_frame
from intraday_engine.fetch.zerodha_client import ZerodhaClient
from intraday_engine.storage import DataStore

logger = logging.getLogger(__name__)


def run_stock_cycle(
    client: ZerodhaClient,
    stock_name: str,
    trade_date: date | None = None,
    include_options: bool = True,
) -> Dict[str, Any] | None:
    """
    Fetch 15-min data for stock, run analysis, append signal to store.
    Returns latest signal payload or None if failed.
    """
    merged = fetch_stock_15min_frame(client, stock_name, trade_date, include_options=include_options)
    if merged is None or merged.empty or len(merged) < 3:
        return None

    settings = Settings.from_env(underlying=stock_name)
    store = DataStore(settings.data_dir, underlying=stock_name)
    store.save_snapshots(merged)
    existing_timestamps = store.load_signal_timestamps()

    latest_payload: Dict[str, Any] | None = None
    for idx in range(len(merged)):
        candle_ts = str(merged.iloc[idx]["timestamp"])
        if candle_ts in existing_timestamps:
            continue
        frame = merged.iloc[: idx + 1].reset_index(drop=True)
        payload = _analyze_frame(frame, stock_name, settings)
        if payload:
            store.append_signal(payload)
            latest_payload = payload

    if latest_payload is None:
        latest_payload = _analyze_frame(merged, stock_name, settings)
    return latest_payload


def _analyze_frame(frame: pd.DataFrame, stock_name: str, settings: Settings) -> Dict[str, Any]:
    if len(frame) < 3:
        last = frame.iloc[-1]
        return {
            "stock": stock_name,
            "timestamp": str(last["timestamp"]),
            "signal": "NO_TRADE",
            "entry": None,
            "target": None,
            "stop_loss": None,
            "rr": None,
            "confidence": 0.0,
            "bias": "NEUTRAL_DAY",
            "momentum": "NEUTRAL",
            "support": float(last["spot_low"]),
            "resistance": float(last["spot_high"]),
            "score": {"bullish": 0.0, "bearish": 0.0, "no_trade_penalty": 0.0, "final_score": 0.0, "confidence": 0.0, "reasons": ["Insufficient bars."]},
            "notes": ["Waiting for enough 15-min bars."],
        }

    features = compute_features(frame)
    support, resistance = calculate_support_resistance(frame, settings.lookback_bars)
    momentum = momentum_direction(frame)
    bias = probable_day_bias(features)

    spot = float(frame.iloc[-1]["spot_ltp"])
    is_breakout = spot > resistance
    is_breakdown = spot < support
    follow_through = abs(features["spot_change_pct"]) > 0.06
    range_size = max(resistance - support, 1e-6)
    distance_from_mid = abs(spot - (support + resistance) / 2.0)
    is_mid_range = distance_from_mid < 0.18 * range_size

    long_stop_pct = abs((spot - support) / max(spot, 1.0) * 100)
    short_stop_pct = abs((resistance - spot) / max(spot, 1.0) * 100)
    stop_too_wide = min(long_stop_pct, short_stop_pct) > settings.max_stop_pct

    range_pct = range_size / max(spot, 1.0) * 100
    sideways, sideways_reason = is_sideways_day(
        range_pct=range_pct,
        bias=bias,
        momentum=momentum,
        follow_through=follow_through,
        is_breakout=is_breakout,
        is_breakdown=is_breakdown,
        features=features,
        min_range_pct=settings.min_day_range_pct,
    )
    if sideways:
        return _no_trade(frame, stock_name, support, resistance, sideways_reason, bias, momentum)

    score = score_signal(
        features=features,
        momentum=momentum,
        bias=bias,
        is_breakout=is_breakout,
        is_breakdown=is_breakdown,
        follow_through=follow_through,
        stop_too_wide=stop_too_wide,
        is_mid_range=is_mid_range,
    )
    last_row = frame.iloc[-1]
    atm_strike = int(last_row.get("atm_strike", 0))
    call_ltp = float(last_row.get("call_ltp", 0) or 0)
    put_ltp = float(last_row.get("put_ltp", 0) or 0)
    plan = build_trade_plan(
        spot=spot,
        support=support,
        resistance=resistance,
        score=score,
        bias=bias,
        momentum=momentum,
        settings=settings,
        atm_strike=atm_strike,
        call_ltp=call_ltp,
        put_ltp=put_ltp,
    )
    payload = {"stock": stock_name, "timestamp": str(frame.iloc[-1]["timestamp"]), **plan.to_dict()}
    ce_symbol = str(last_row.get("ce_symbol", ""))
    pe_symbol = str(last_row.get("pe_symbol", ""))
    if plan.signal == "BUY" and ce_symbol:
        payload["option_symbol"] = ce_symbol
    elif plan.signal == "SELL" and pe_symbol:
        payload["option_symbol"] = pe_symbol
    else:
        payload["option_symbol"] = None
    return payload


def _no_trade(
    frame: pd.DataFrame,
    stock_name: str,
    support: float,
    resistance: float,
    reason: str,
    bias: str,
    momentum: str,
) -> Dict[str, Any]:
    last = frame.iloc[-1]
    return {
        "stock": stock_name,
        "timestamp": str(last["timestamp"]),
        "signal": "NO_TRADE",
        "entry": None,
        "target": None,
        "stop_loss": None,
        "rr": None,
        "strike_price": None,
        "option_type": None,
        "option_entry": None,
        "option_sl": None,
        "option_target": None,
        "option_symbol": None,
        "confidence": 0.0,
        "bias": bias,
        "momentum": momentum,
        "support": round(support, 2),
        "resistance": round(resistance, 2),
        "score": {"bullish": 0.0, "bearish": 0.0, "no_trade_penalty": 0.0, "final_score": 0.0, "confidence": 0.0, "reasons": [reason]},
        "notes": [reason],
    }
