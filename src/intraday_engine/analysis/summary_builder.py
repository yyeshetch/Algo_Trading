"""Build analysis summary (price action, futures, options) for a timestamp."""

from __future__ import annotations

from typing import Any

import pandas as pd


def _str_safe(v: Any, default: str = "—") -> str:
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return default
    return str(v)


def _pct_change(current: float, base: float) -> float:
    if base == 0:
        return 0.0
    return round(((current - base) / base) * 100, 2)


def _momentum(df: pd.DataFrame, col: str, bars: int) -> str:
    if len(df) < bars:
        return "NEUTRAL"
    window = df.iloc[-bars:]
    start = float(window.iloc[0][col])
    end = float(window.iloc[-1][col])
    if start == 0:
        return "NEUTRAL"
    move = ((end - start) / start) * 100
    return "UP" if move > 0.20 else ("DOWN" if move < -0.20 else "NEUTRAL")


def _volume_bias(frame: pd.DataFrame, candle_open: float, candle_close: float, candle_low: float, candle_high: float) -> dict[str, Any]:
    if frame.empty or "spot_volume" not in frame.columns or len(frame) < 2:
        return {
            "bias": "NEUTRAL",
            "volume": None,
            "volume_sma20": None,
            "volume_multiple": None,
            "liquidity_grab": None,
        }

    volume_series = frame["spot_volume"].astype(float)
    current_volume = float(volume_series.iloc[-1])
    window_size = min(20, len(volume_series))
    volume_sma20 = float(volume_series.iloc[-window_size:].mean())
    volume_multiple = (current_volume / volume_sma20) if volume_sma20 > 0 else 0.0

    bias = "NEUTRAL"
    liquidity_grab: float | None = None
    if volume_sma20 > 0 and current_volume > 1.5 * volume_sma20:
        if candle_close > candle_open:
            bias = "BULLISH_BIAS"
            liquidity_grab = candle_low
        elif candle_close < candle_open:
            bias = "BEARISH_BIAS"
            liquidity_grab = candle_high

    return {
        "bias": bias,
        "volume": round(current_volume, 2),
        "volume_sma20": round(volume_sma20, 2),
        "volume_multiple": round(volume_multiple, 2),
        "liquidity_grab": round(liquidity_grab, 2) if liquidity_grab is not None else None,
        "confirmation_timestamp": str(frame.iloc[-1].get("timestamp", "")) if bias != "NEUTRAL" else None,
        "liquidity_grab_status": "PENDING" if bias != "NEUTRAL" else "NONE",
        "liquidity_grab_hit_at": None,
    }


def _latest_volume_bias(frame: pd.DataFrame) -> dict[str, Any]:
    latest = {
        "bias": "NEUTRAL",
        "volume": None,
        "volume_sma20": None,
        "volume_multiple": None,
        "liquidity_grab": None,
        "confirmation_timestamp": None,
        "liquidity_grab_status": "NONE",
        "liquidity_grab_hit_at": None,
    }
    if frame.empty or "spot_volume" not in frame.columns or len(frame) < 2:
        return latest

    for idx in range(len(frame)):
        partial = frame.iloc[: idx + 1].reset_index(drop=True)
        row = partial.iloc[-1]
        candle_open = float(row.get("spot_open_raw", partial.iloc[-2]["spot_ltp"] if len(partial) > 1 else row.get("spot_ltp", 0)) or 0)
        bias_snapshot = _volume_bias(
            partial,
            candle_open=candle_open,
            candle_close=float(row.get("spot_ltp", 0) or 0),
            candle_low=float(row.get("spot_low", 0) or 0),
            candle_high=float(row.get("spot_high", 0) or 0),
        )
        if bias_snapshot["bias"] != "NEUTRAL":
            latest = bias_snapshot

    if latest["bias"] == "NEUTRAL" or latest["confirmation_timestamp"] is None:
        return latest

    confirmation_ts = str(latest["confirmation_timestamp"])
    confirmation_rows = frame.index[frame["timestamp"].astype(str) == confirmation_ts].tolist()
    if not confirmation_rows:
        return latest
    start_idx = confirmation_rows[-1] + 1
    if start_idx >= len(frame):
        return latest

    level = latest["liquidity_grab"]
    if level is None:
        return latest

    future = frame.iloc[start_idx:].reset_index(drop=True)
    if latest["bias"] == "BULLISH_BIAS":
        hit = future[future["spot_low"].astype(float) <= float(level)]
    else:
        hit = future[future["spot_high"].astype(float) >= float(level)]

    if not hit.empty:
        latest["liquidity_grab_status"] = "GRABBED"
        latest["liquidity_grab_hit_at"] = str(hit.iloc[0].get("timestamp", ""))
    return latest


def build_analysis_summaries(snapshots_df: pd.DataFrame, signals_df: pd.DataFrame, lookback: int = 20) -> list[dict[str, Any]]:
    """Build analysis summary for each timestamp in snapshots."""
    if snapshots_df.empty:
        return []
    summaries = []
    signals_by_ts = {}
    if not signals_df.empty and "timestamp" in signals_df.columns:
        for _, row in signals_df.iterrows():
            ts = str(row.get("timestamp", ""))
            signals_by_ts[ts] = row.to_dict()
    for idx in range(len(snapshots_df)):
        row = snapshots_df.iloc[idx]
        ts = str(row.get("timestamp", ""))
        prev = snapshots_df.iloc[idx - 1] if idx > 0 else row
        spot_ltp = float(row.get("spot_ltp", 0) or 0)
        spot_open = float(row.get("spot_open", 0) or 0)
        spot_vwap = float(row.get("spot_vwap", 0) or 0)
        fut_ltp = float(row.get("future_ltp", 0) or 0)
        call_ltp = float(row.get("call_ltp", 0) or 0)
        put_ltp = float(row.get("put_ltp", 0) or 0)
        prev_spot = float(prev.get("spot_ltp", 0) or 0)
        prev_call = float(prev.get("call_ltp", 0) or 0)
        prev_put = float(prev.get("put_ltp", 0) or 0)
        support = float(row.get("support", 0) or 0)
        resistance = float(row.get("resistance", 0) or 0)
        sig = signals_by_ts.get(ts, {})
        if "support" in sig and sig.get("support") is not None:
            support = float(sig.get("support", 0) or 0)
        elif idx > 0:
            window = snapshots_df.iloc[max(0, idx - lookback) : idx]
            support = float(window["spot_low"].min()) if "spot_low" in window else support
        if "resistance" in sig and sig.get("resistance") is not None:
            resistance = float(sig.get("resistance", 0) or 0)
        elif idx > 0:
            window = snapshots_df.iloc[max(0, idx - lookback) : idx]
            resistance = float(window["spot_high"].max()) if "spot_high" in window else resistance
        is_breakout = spot_ltp > resistance if resistance else False
        is_breakdown = spot_ltp < support if support else False
        spot_change = _pct_change(spot_ltp, prev_spot)
        follow_through = abs(spot_change) > 0.06
        fut_strength = _pct_change(fut_ltp, spot_ltp) if spot_ltp else 0
        call_change = _pct_change(call_ltp, prev_call) if prev_call else 0
        put_change = _pct_change(put_ltp, prev_put) if prev_put else 0
        frame = snapshots_df.iloc[: idx + 1]
        momentum = _momentum(frame, "spot_ltp", 4)
        momentum_3 = _momentum(frame, "spot_ltp", 3)
        momentum_5 = _momentum(frame, "spot_ltp", 5)
        fut_open = float(row.get("future_open", 0) or 0)
        fut_vwap = float(row.get("future_vwap", 0) or 0)
        fut_change = _pct_change(fut_ltp, float(prev.get("future_ltp", 0) or 0))
        fut_vs_open = _pct_change(fut_ltp, fut_open) if fut_open else 0
        fut_vs_vwap = _pct_change(fut_ltp, fut_vwap) if fut_vwap else 0
        spot_high = float(row.get("spot_high", 0) or 0)
        spot_low = float(row.get("spot_low", 0) or 0)
        candle_open = float(row.get("spot_open_raw", prev_spot) or prev_spot)
        candle_range = spot_high - spot_low if spot_high and spot_low else 0
        body = spot_ltp - candle_open
        body_pct = _pct_change(spot_ltp, candle_open) if candle_open else 0
        session_high = float(snapshots_df.iloc[: idx + 1]["spot_high"].max()) if idx >= 0 else spot_high
        session_low = float(snapshots_df.iloc[: idx + 1]["spot_low"].min()) if idx >= 0 else spot_low
        dist_from_high = _pct_change(session_high, spot_ltp) if spot_ltp else 0
        dist_from_low = _pct_change(spot_ltp, session_low) if session_low else 0
        range_size = resistance - support if resistance and support else 0
        range_pct = _pct_change(range_size, spot_ltp) if spot_ltp else 0
        put_call_ratio = round(put_ltp / call_ltp, 2) if call_ltp else 0
        spot_in_range = ((spot_ltp - support) / (resistance - support) * 100) if (resistance and support and resistance > support) else 50
        first_row = snapshots_df.iloc[0]
        fut_oi = float(row.get("future_oi", 0) or 0)
        call_oi = float(row.get("call_oi", 0) or 0)
        put_oi = float(row.get("put_oi", 0) or 0)
        fut_oi_first = float(first_row.get("future_oi", 0) or 0)
        call_oi_first = float(first_row.get("call_oi", 0) or 0)
        put_oi_first = float(first_row.get("put_oi", 0) or 0)
        fut_oi_chg = _pct_change(fut_oi, fut_oi_first) if fut_oi_first else 0
        call_oi_chg = _pct_change(call_oi, call_oi_first) if call_oi_first else 0
        put_oi_chg = _pct_change(put_oi, put_oi_first) if put_oi_first else 0
        prev_fut_oi = float(prev.get("future_oi", 0) or 0)
        fut_oi_chg_candle = _pct_change(fut_oi, prev_fut_oi) if prev_fut_oi and idx > 0 else 0
        volume_bias = _latest_volume_bias(frame)
        summaries.append({
            "timestamp": ts,
            "signal": _str_safe(sig.get("signal"), "—"),
            "price_action": {
                "spot": round(spot_ltp, 2),
                "open": round(spot_open, 2),
                "vwap": round(spot_vwap, 2),
                "high": round(spot_high, 2),
                "low": round(spot_low, 2),
                "support": round(support, 2),
                "resistance": round(resistance, 2),
                "spot_vs_open": "above" if spot_ltp > spot_open else ("below" if spot_ltp < spot_open else "flat"),
                "spot_vs_vwap": "above" if spot_ltp > spot_vwap else ("below" if spot_ltp < spot_vwap else "flat"),
                "breakout": is_breakout,
                "breakdown": is_breakdown,
                "follow_through": follow_through,
                "momentum": momentum,
                "momentum_3": momentum_3,
                "momentum_5": momentum_5,
                "spot_change_pct": spot_change,
                "candle_range": round(candle_range, 2),
                "candle_body": round(body, 2),
                "candle_body_pct": body_pct,
                "session_high": round(session_high, 2),
                "session_low": round(session_low, 2),
                "dist_from_session_high_pct": dist_from_high,
                "dist_from_session_low_pct": dist_from_low,
                "range_size": round(range_size, 2),
                "range_pct": range_pct,
                "spot_in_range_pct": round(spot_in_range, 1),
            },
            "futures": {
                "ltp": round(fut_ltp, 2),
                "open": round(fut_open, 2),
                "vwap": round(fut_vwap, 2),
                "high": round(float(row.get("future_high", 0) or 0), 2),
                "low": round(float(row.get("future_low", 0) or 0), 2),
                "vs_spot_pct": fut_strength,
                "vs_open_pct": fut_vs_open,
                "vs_vwap_pct": fut_vs_vwap,
                "change_pct": fut_change,
                "premium": "premium" if fut_ltp > spot_ltp else ("discount" if fut_ltp < spot_ltp else "flat"),
                "fut_vs_fut_vwap": "above" if fut_ltp > fut_vwap else ("below" if fut_ltp < fut_vwap else "flat"),
            },
            "options": {
                "call_ltp": round(call_ltp, 2),
                "put_ltp": round(put_ltp, 2),
                "atm_strike": int(row.get("atm_strike", 0) or 0),
                "call_change_pct": call_change,
                "put_change_pct": put_change,
                "call_expanding": call_change > 0,
                "put_decaying": put_change < 0,
                "put_expanding": put_change > 0,
                "call_decaying": call_change < 0,
                "put_call_ratio": put_call_ratio,
                "skew": "put_heavy" if put_call_ratio > 1.1 else ("call_heavy" if put_call_ratio < 0.9 else "balanced"),
            },
            "oi": {
                "future_oi": int(fut_oi),
                "call_oi": int(call_oi),
                "put_oi": int(put_oi),
                "fut_oi_change_pct": fut_oi_chg,
                "call_oi_change_pct": call_oi_chg,
                "put_oi_change_pct": put_oi_chg,
                "fut_oi_change_candle_pct": fut_oi_chg_candle,
            },
            "volume_bias": volume_bias,
            "scores": {
                "bullish": float(sig.get("bullish_score", 0) or 0),
                "bearish": float(sig.get("bearish_score", 0) or 0),
                "no_trade_penalty": float(sig.get("no_trade_penalty", 0) or 0),
                "final_score": float(sig.get("final_score", 0) or 0),
                "confidence": float(sig.get("confidence", 0) or 0),
            },
            "trade_plan": {
                "entry": sig.get("entry"),
                "target": sig.get("target"),
                "stop_loss": sig.get("stop_loss"),
                "rr": sig.get("rr"),
            },
            "bias": _str_safe(sig.get("bias"), "—"),
            "reasons": _str_safe(sig.get("reasons"), ""),
            "notes": _str_safe(sig.get("notes"), ""),
        })
    return summaries
