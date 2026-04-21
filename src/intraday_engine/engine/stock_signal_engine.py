"""Generate signals for a single F&O stock from 15-min data (spot + fut, optional options)."""

from __future__ import annotations

import logging
from datetime import date, datetime, timedelta
from typing import Any, Dict

import pandas as pd

from intraday_engine.analysis.day_bias import probable_day_bias
from intraday_engine.analysis.momentum import momentum_direction
from intraday_engine.analysis.rsi import rsi_last, rsi_series
from intraday_engine.analysis.scoring import score_signal
from intraday_engine.analysis.sideways import is_sideways_day
from intraday_engine.analysis.support_resistance import calculate_support_resistance
from intraday_engine.analysis.trade_plan import build_trade_plan
from intraday_engine.core.config import Settings
from intraday_engine.core.models import ScoreBreakdown
from intraday_engine.core.underlyings import get_underlying_config
from intraday_engine.features.feature_engineering import compute_features
from intraday_engine.fetch.stock_market_data import (
    _drop_incomplete_candles,
    _market_window_15min,
    _to_candle_df,
    fetch_stock_15min_frame,
    fetch_stock_30min_frame,
)
from intraday_engine.fetch.zerodha_client import ZerodhaClient
from intraday_engine.storage import DataStore

logger = logging.getLogger(__name__)


def _sse_f(key: str, default: float) -> float:
    from intraday_engine.core.tunables import get_float

    return get_float("stock_signal_engine", key, default)


def _sse_i(key: str, default: int) -> int:
    from intraday_engine.core.tunables import get_int

    return get_int("stock_signal_engine", key, default)


def _cp_f(key: str, default: float) -> float:
    from intraday_engine.core.tunables import get_float

    return get_float("candle_patterns", key, default)


def run_stock_cycle(
    client: ZerodhaClient,
    stock_name: str,
    trade_date: date | None = None,
    include_options: bool = True,
    benchmark_frame: pd.DataFrame | None = None,
) -> Dict[str, Any] | None:
    """
    Fetch 15-min data for stock, run analysis, append signal to store.
    Returns latest signal payload or None if failed.
    """
    merged = fetch_stock_15min_frame(client, stock_name, trade_date, include_options=include_options)
    if merged is None or merged.empty or len(merged) < 3:
        return None

    rsi_context = _fetch_spot_rsi_context(client, stock_name, trade_date)
    settings = Settings.from_env(underlying=stock_name)
    store = DataStore(settings.data_dir, underlying=stock_name)
    store.save_snapshots(merged)

    payloads: list[Dict[str, Any]] = []
    for idx in range(len(merged)):
        frame = merged.iloc[: idx + 1].reset_index(drop=True)
        frame_payloads = _analyze_signal_frames(
            frame,
            stock_name,
            settings,
            benchmark_frame=benchmark_frame,
            rsi_context=rsi_context,
        )
        payloads.extend(frame_payloads)

    if not payloads:
        return None

    store.replace_signals_for_date(trade_date or date.today(), payloads)
    actionable = [p for p in payloads if str(p.get("signal")) in {"BUY", "SELL"}]
    return actionable[-1] if actionable else payloads[-1]


def _analyze_signal_frames(
    frame: pd.DataFrame,
    stock_name: str,
    settings: Settings,
    benchmark_frame: pd.DataFrame | None = None,
    rsi_context: Dict[str, pd.DataFrame] | None = None,
) -> list[Dict[str, Any]]:
    legacy_payload = _apply_signal_side_rsi_filter(
        _decorate_signal_payload(
            _analyze_frame(frame, stock_name, settings),
            frame,
            benchmark_frame,
            rsi_context=rsi_context,
        )
    )
    failed_payload = _apply_signal_side_rsi_filter(
        _decorate_signal_payload(
            _analyze_failed_bias_frame(
                frame,
                stock_name,
                settings,
                benchmark_frame=benchmark_frame,
                rsi_context=rsi_context,
            ),
            frame,
            benchmark_frame,
            rsi_context=rsi_context,
        )
    )
    vol_ema_payload = _apply_signal_side_rsi_filter(
        _decorate_signal_payload(
            _analyze_vol_ema_bullish_frame(frame, stock_name, settings, rsi_context=rsi_context),
            frame,
            benchmark_frame,
            rsi_context=rsi_context,
        )
    )
    overbought_payload = _decorate_signal_payload(
        _analyze_overbought_rsi_frame(frame, stock_name, settings, rsi_context=rsi_context),
        frame,
        benchmark_frame,
        rsi_context=rsi_context,
    )
    return [legacy_payload, failed_payload, vol_ema_payload, overbought_payload]


def _apply_signal_side_rsi_filter(payload: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(payload)
    signal = str(out.get("signal") or "")
    label = str(out.get("strategy_label") or "")
    # Prior-bar RSI > 55 already enforced in _analyze_vol_ema_bullish_frame; current bar may dip.
    if label == "DIRECTIONAL_BUY_VOL_BULLISH_BIAS":
        return out
    rsi_15m = out.get("rsi_15m")
    if rsi_15m is None:
        return out
    try:
        rsi_value = float(rsi_15m)
    except (TypeError, ValueError):
        return out

    buy_floor = _sse_f("FAILED_BIAS_RSI_BUY_FLOOR", 45.0)
    sell_ceil = _sse_f("FAILED_BIAS_RSI_SELL_CEILING", 55.0)
    if signal == "BUY" and rsi_value < buy_floor:
        return _invalidate_signal_payload(out, f"BUY rejected: 15m RSI is below {buy_floor:.0f}.")
    if signal == "SELL" and rsi_value > sell_ceil:
        return _invalidate_signal_payload(out, f"SELL rejected: 15m RSI is above {sell_ceil:.0f}.")
    return out


def _extended_m15_closes(frame: pd.DataFrame, history_15m: pd.DataFrame | None) -> pd.DataFrame:
    """Ordered closes: optional long 15m history + session frame (for EMA10/20/200)."""
    parts: list[pd.DataFrame] = []
    if history_15m is not None and not history_15m.empty:
        h = history_15m.copy()
        h["timestamp"] = pd.to_datetime(h["timestamp"])
        parts.append(h[["timestamp", "close"]].rename(columns={"close": "c"}))
    if frame is not None and not frame.empty:
        f = frame.copy()
        f["timestamp"] = pd.to_datetime(f["timestamp"])
        parts.append(f[["timestamp", "spot_ltp"]].rename(columns={"spot_ltp": "c"}))
    if not parts:
        return pd.DataFrame(columns=["timestamp", "c"])
    comb = pd.concat(parts, ignore_index=True)
    comb = comb.sort_values("timestamp").drop_duplicates(subset=["timestamp"], keep="last")
    return comb.reset_index(drop=True)


def _pct_price_distance(a: float, b: float) -> float:
    return abs(a - b) / max(a, 1e-9) * 100.0


def _analyze_vol_ema_bullish_frame(
    frame: pd.DataFrame,
    stock_name: str,
    settings: Settings,
    rsi_context: Dict[str, pd.DataFrame] | None = None,
) -> Dict[str, Any]:
    """Large green volume candle near 10/20/200 EMA; prior 15m bar RSI > 55 (not 10-bar regime)."""
    history_15m = (rsi_context or {}).get("m15")
    support, resistance = calculate_support_resistance(frame, settings.lookback_bars)
    momentum = momentum_direction(frame)
    features = compute_features(frame)
    bias = probable_day_bias(features)
    last_row = frame.iloc[-1]
    spot = float(last_row["spot_ltp"])

    def _reject(reason: str) -> Dict[str, Any]:
        return _no_trade(
            frame,
            stock_name,
            support,
            resistance,
            reason,
            bias,
            momentum,
            strategy="DIRECTIONAL_VOL_EMA",
        )

    out_base = {
        "stock": stock_name,
        "strategy": "DIRECTIONAL_VOL_EMA",
        "strategy_label": "DIRECTIONAL_BUY_VOL_BULLISH_BIAS",
    }

    if len(frame) < _sse_i("VOL_EMA_MIN_SESSION_BARS", 3):
        p = _reject("Insufficient session bars for vol+EMA setup.")
        p.update(out_base)
        return p

    prev_rsi = _rsi_15m_value_at(
        frame,
        cutoff_ts=str(frame.iloc[-2]["timestamp"]),
        history_15m=history_15m,
    )
    buy_reg = _sse_f("FAILED_BIAS_RSI_BUY_REGIME", 55.0)
    if prev_rsi is None or prev_rsi <= buy_reg:
        p = _reject(f"Prior 15m candle RSI must be above {buy_reg:.0f} (got {prev_rsi}).")
        p.update(out_base)
        return p

    o = float(last_row.get("spot_open_raw", last_row.get("spot_ltp", 0)) or 0)
    c = spot
    if c <= o:
        p = _reject("Last candle is not green (bullish close > open).")
        p.update(out_base)
        return p

    vol = float(last_row.get("spot_volume", 0) or 0)
    prior_vol = frame["spot_volume"].iloc[:-1].astype(float)
    if len(prior_vol) < _sse_i("FAILED_BIAS_MIN_BARS", 1):
        p = _reject("Not enough prior volume history.")
        p.update(out_base)
        return p
    sma20 = float(prior_vol.tail(_sse_i("FAILED_BIAS_SMA_WINDOW", 20)).mean())
    if sma20 <= 0 or vol < _sse_f("VOL_EMA_VOLUME_MULTIPLIER", 1.5) * sma20:
        vw = _sse_i("FAILED_BIAS_SMA_WINDOW", 20)
        vm = _sse_f("VOL_EMA_VOLUME_MULTIPLIER", 1.5)
        p = _reject(f"Volume not large enough vs {vw}-bar SMA (need ≥{vm}×).")
        p.update(out_base)
        return p

    comb = _extended_m15_closes(frame, history_15m)
    if len(comb) < _sse_i("VOL_EMA_MIN_EXTENDED_CLOSES", 30):
        p = _reject("Insufficient 15m history for EMA context.")
        p.update(out_base)
        return p

    close_series = comb["c"].astype(float)
    ema10 = float(close_series.ewm(span=10, adjust=False).mean().iloc[-1])
    ema20 = float(close_series.ewm(span=20, adjust=False).mean().iloc[-1])
    dists = [_pct_price_distance(c, ema10), _pct_price_distance(c, ema20)]
    if len(close_series) >= 200:
        ema200 = float(close_series.ewm(span=200, adjust=False).mean().iloc[-1])
        dists.append(_pct_price_distance(c, ema200))
    near_pct = _sse_f("VOL_EMA_NEAR_PCT", 0.35)
    if min(dists) > near_pct:
        p = _reject(f"Close not near 10/20/200 EMA (>{near_pct:.2f}% from nearest).")
        p.update(out_base)
        return p

    score = ScoreBreakdown(
        bullish=1.0,
        bearish=0.0,
        no_trade_penalty=0.0,
        final_score=1.0,
        confidence=max(settings.min_confidence, _sse_f("VOL_EMA_SYNTHETIC_CONFIDENCE", 0.72)),
        reasons=[
            "High volume green candle at/near 10/20/200 EMA; prior 15m RSI above "
            f"{buy_reg:.0f}.",
        ],
    )
    atm_strike = int(last_row.get("atm_strike", 0))
    call_ltp = float(last_row.get("call_ltp", 0) or 0)
    put_ltp = float(last_row.get("put_ltp", 0) or 0)
    plan = build_trade_plan(
        spot=c,
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
    if plan.signal != "BUY":
        p = _reject(str(plan.notes[0]) if plan.notes else "Trade plan did not confirm BUY.")
        p.update(out_base)
        return p

    payload = {
        **out_base,
        "timestamp": str(last_row["timestamp"]),
        **plan.to_dict(),
    }
    ce_symbol = str(last_row.get("ce_symbol", ""))
    payload["option_symbol"] = ce_symbol if ce_symbol else None
    notes = list(payload.get("notes", [])) if isinstance(payload.get("notes"), list) else []
    notes.append(
        f"Vol {vol:.0f} vs SMA20 {sma20:.0f} ({vol / sma20:.2f}×); "
        f"nearest EMA distance {min(dists):.2f}%; prior RSI {prev_rsi:.1f}."
    )
    payload["notes"] = notes
    return payload


def _analyze_overbought_rsi_frame(
    frame: pd.DataFrame,
    stock_name: str,
    settings: Settings,
    rsi_context: Dict[str, pd.DataFrame] | None = None,
) -> Dict[str, Any]:
    support, resistance = calculate_support_resistance(frame, settings.lookback_bars)
    momentum = momentum_direction(frame)
    bias = probable_day_bias(compute_features(frame))
    last = frame.iloc[-1]
    cutoff = str(last["timestamp"])
    rsi = _rsi_15m(frame, rsi_context=rsi_context, cutoff_ts=cutoff)
    ob_th = _sse_f("OVERBOUGHT_RSI_THRESHOLD", 75.0)
    if rsi is None or rsi <= ob_th:
        return _no_trade(
            frame,
            stock_name,
            support,
            resistance,
            f"15m RSI not above {ob_th:.0f}.",
            bias,
            momentum,
            strategy="RSI_EXTREME",
        )
    return {
        "stock": stock_name,
        "strategy": "RSI_EXTREME",
        "strategy_label": "OVERBROUGHT_LOOK_FOR_REVERSAL",
        "timestamp": cutoff,
        "signal": "OVERBROUGHT_LOOK_FOR_REVERSAL",
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
        "score": {
            "bullish": 0.0,
            "bearish": 0.0,
            "no_trade_penalty": 0.0,
            "final_score": 0.0,
            "confidence": 0.0,
            "reasons": [f"15m RSI {rsi:.2f} — watch for mean reversion / exhaustion."],
        },
        "notes": [f"15m RSI {rsi:.2f} > {ob_th:.0f}; look for reversal setups."],
    }


def _invalidate_signal_payload(payload: Dict[str, Any], reason: str) -> Dict[str, Any]:
    out = dict(payload)
    out["signal"] = "NO_TRADE"
    for key in [
        "entry",
        "target",
        "stop_loss",
        "rr",
        "strike_price",
        "option_type",
        "option_entry",
        "option_sl",
        "option_target",
        "option_symbol",
    ]:
        if key in out:
            out[key] = None
    score = out.get("score")
    if isinstance(score, dict):
        score = dict(score)
        score["confidence"] = 0.0
        score["final_score"] = 0.0
        reasons = list(score.get("reasons", []))
        reasons.append(reason)
        score["reasons"] = reasons
        out["score"] = score
    notes = list(out.get("notes", [])) if isinstance(out.get("notes"), list) else []
    notes.append(reason)
    out["notes"] = notes
    return out


def _decorate_signal_payload(
    payload: Dict[str, Any],
    frame: pd.DataFrame,
    benchmark_frame: pd.DataFrame | None = None,
    rsi_context: Dict[str, pd.DataFrame] | None = None,
) -> Dict[str, Any]:
    out = dict(payload)
    metrics_end_idx = out.pop("_metrics_end_idx", None)
    metrics_frame = frame
    if isinstance(metrics_end_idx, int) and 0 <= metrics_end_idx < len(frame):
        metrics_frame = frame.iloc[: metrics_end_idx + 1].reset_index(drop=True)
    out.update(_common_signal_metrics(metrics_frame, benchmark_frame, rsi_context=rsi_context))
    out.update(_signal_side_columns(out.get("signal"), _orb_state(frame), _pinbar_state(frame)))
    out.setdefault("bias_volume_multiple", out.get("volume_multiple"))
    out.setdefault("trigger_volume_multiple", None)
    return out


def _analyze_frame(frame: pd.DataFrame, stock_name: str, settings: Settings) -> Dict[str, Any]:
    if len(frame) < 3:
        last = frame.iloc[-1]
        return {
            "stock": stock_name,
            "strategy": "DIRECTIONAL",
            "strategy_label": "DIRECTIONAL",
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
    follow_through = abs(features["spot_change_pct"]) > _sse_f("DIRECTIONAL_FOLLOW_THROUGH_MIN_PCT", 0.06)
    range_size = max(resistance - support, 1e-6)
    distance_from_mid = abs(spot - (support + resistance) / 2.0)
    is_mid_range = distance_from_mid < _sse_f("DIRECTIONAL_MID_RANGE_FRAC", 0.18) * range_size

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
        return _no_trade(frame, stock_name, support, resistance, sideways_reason, bias, momentum, strategy="DIRECTIONAL")

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
    strategy_label = f"DIRECTIONAL_{plan.signal}_{str(plan.bias).replace(' ', '_')}" if plan.signal in {"BUY", "SELL"} else "DIRECTIONAL"
    payload = {
        "stock": stock_name,
        "strategy": "DIRECTIONAL",
        "strategy_label": strategy_label,
        "timestamp": str(frame.iloc[-1]["timestamp"]),
        **plan.to_dict(),
    }
    ce_symbol = str(last_row.get("ce_symbol", ""))
    pe_symbol = str(last_row.get("pe_symbol", ""))
    if plan.signal == "BUY" and ce_symbol:
        payload["option_symbol"] = ce_symbol
    elif plan.signal == "SELL" and pe_symbol:
        payload["option_symbol"] = pe_symbol
    else:
        payload["option_symbol"] = None
    return payload


def _analyze_failed_bias_frame(
    frame: pd.DataFrame,
    stock_name: str,
    settings: Settings,
    benchmark_frame: pd.DataFrame | None = None,
    rsi_context: Dict[str, pd.DataFrame] | None = None,
) -> Dict[str, Any]:
    last = frame.iloc[-1]
    support, resistance = calculate_support_resistance(frame, settings.lookback_bars)
    momentum = momentum_direction(frame)
    if len(frame) < _sse_i("FAILED_BIAS_MIN_BARS", 1) + 1:
        return _no_trade(
            frame,
            stock_name,
            support,
            resistance,
            "Waiting for enough 15-min bars for failed-bias setup.",
            "FAILED_BIAS",
            momentum,
            strategy="FAILED_BIAS",
        )

    setup = _latest_failed_bias_setup(frame, history_15m=(rsi_context or {}).get("m15"))
    if not setup:
        payload = _no_trade(
            frame,
            stock_name,
            support,
            resistance,
            "No failed bullish/bearish volume-bias reclaim yet.",
            "FAILED_BIAS",
            momentum,
            strategy="FAILED_BIAS",
        )
        payload["strategy_label"] = "FAILED_BIAS"
        return payload

    trigger_idx = int(setup["trigger_idx"])
    signal_age_bars = len(frame) - 1 - trigger_idx
    if signal_age_bars > 0:
        payload = _no_trade(
            frame,
            stock_name,
            support,
            resistance,
            "Failed-bias setup already emitted on the trigger candle close.",
            setup["pattern"],
            momentum,
            strategy="FAILED_BIAS",
        )
        payload["strategy_label"] = setup["pattern"]
        return payload
    max_age = _sse_i("FAILED_BIAS_MAX_AGE_BARS", 4)
    if signal_age_bars > max_age:
        payload = _no_trade(
            frame,
            stock_name,
            support,
            resistance,
            f"Failed-bias trigger expired after {max_age} candles.",
            setup["pattern"],
            momentum,
            strategy="FAILED_BIAS",
        )
        payload["strategy_label"] = setup["pattern"]
        return payload

    minutes_left_at_trigger = _minutes_to_session_close(setup["trigger_time_close"])
    min_runway_minutes = _sse_i("FAILED_BIAS_MIN_RUNWAY_BARS", 6) * 15
    if minutes_left_at_trigger < min_runway_minutes:
        payload = _no_trade(
            frame,
            stock_name,
            support,
            resistance,
            f"Failed-bias trigger came too late ({minutes_left_at_trigger} mins left, need {min_runway_minutes}).",
            setup["pattern"],
            momentum,
            strategy="FAILED_BIAS",
        )
        payload["strategy_label"] = setup["pattern"]
        return payload

    trigger_frame = frame.iloc[: trigger_idx + 1].reset_index(drop=True)
    trigger_row = trigger_frame.iloc[-1]
    trigger_support, trigger_resistance = calculate_support_resistance(trigger_frame, settings.lookback_bars)
    trigger_momentum = momentum_direction(trigger_frame)
    score = _failed_bias_score(trigger_frame, setup, trigger_support, trigger_resistance, trigger_momentum)
    if score["confidence"] < _sse_f("FAILED_BIAS_MIN_CONFIDENCE", 0.90):
        payload = _no_trade(
            frame,
            stock_name,
            trigger_support,
            trigger_resistance,
            "Failed-bias reclaim found, but confluence is too weak.",
            setup["pattern"],
            trigger_momentum,
            strategy="FAILED_BIAS",
        )
        payload["strategy_label"] = setup["pattern"]
        return payload

    signal = setup["signal"]
    rsi_reason = _failed_bias_rsi_reason(
        signal=signal,
        frame=trigger_frame,
        history_15m=(rsi_context or {}).get("m15"),
        trigger_ts=str(trigger_row["timestamp"]),
    )
    if rsi_reason:
        payload = _no_trade(
            frame,
            stock_name,
            trigger_support,
            trigger_resistance,
            rsi_reason,
            setup["pattern"],
            trigger_momentum,
            strategy="FAILED_BIAS",
        )
        payload["strategy_label"] = setup["pattern"]
        return payload

    entry = float(trigger_row["spot_ltp"])
    trigger_low = float(trigger_row.get("spot_low", entry) or entry)
    trigger_high = float(trigger_row.get("spot_high", entry) or entry)
    if signal == "BUY":
        stop = trigger_low
        risk = max(entry - stop, 0.01)
        structure_target = max(float(setup["bias_high"]), trigger_resistance)
        fast_target = entry + (risk * _sse_f("FAILED_BIAS_TARGET_RR", 1.25))
        target = min(structure_target, fast_target) if structure_target > entry else fast_target
        option_type = "CE"
        option_entry = float(trigger_row.get("call_ltp", 0) or 0)
        option_symbol = str(trigger_row.get("ce_symbol", "")) or None
    else:
        stop = trigger_high
        risk = max(stop - entry, 0.01)
        structure_target = min(float(setup["bias_low"]), trigger_support)
        fast_target = entry - (risk * _sse_f("FAILED_BIAS_TARGET_RR", 1.25))
        target = max(structure_target, fast_target) if structure_target < entry else fast_target
        option_type = "PE"
        option_entry = float(trigger_row.get("put_ltp", 0) or 0)
        option_symbol = str(trigger_row.get("pe_symbol", "")) or None

    quality_reason = _failed_bias_quality_reason(
        signal=signal,
        entry=entry,
        risk=risk,
        minutes_left_at_trigger=minutes_left_at_trigger,
        score=score,
    )
    if quality_reason:
        payload = _no_trade(
            frame,
            stock_name,
            trigger_support,
            trigger_resistance,
            quality_reason,
            setup["pattern"],
            trigger_momentum,
            strategy="FAILED_BIAS",
        )
        payload["strategy_label"] = setup["pattern"]
        return payload

    rr = abs(target - entry) / risk if risk > 0 else 0.0
    notes = [
        f"{setup['pattern']} from {setup['bias_close_time']} with {setup['volume_multiple']:.2f}x volume.",
        f"Trigger level {setup['trigger_level']:.2f} reclaimed/failed by {setup['trigger_time_close']}.",
    ]
    payload = {
        "stock": stock_name,
        "strategy": "FAILED_BIAS",
        "strategy_label": setup["pattern"],
        "timestamp": str(last["timestamp"]),
        "signal": signal,
        "entry": round(entry, 2),
        "target": round(target, 2),
        "stop_loss": round(stop, 2),
        "rr": round(rr, 2),
        "strike_price": int(trigger_row.get("atm_strike", 0) or 0) or None,
        "option_type": option_type,
        "option_entry": round(option_entry, 2) if option_entry > 0 else None,
        "option_sl": round(option_entry * _sse_f("FAILED_BIAS_OPTION_SL_RATIO", 0.7), 2) if option_entry > 0 else None,
        "option_target": round(option_entry * _sse_f("FAILED_BIAS_OPTION_TARGET_RATIO", 1.5), 2)
        if option_entry > 0
        else None,
        "option_symbol": option_symbol,
        "confidence": score["confidence"],
        "bias": setup["pattern"],
        "momentum": trigger_momentum,
        "support": round(trigger_support, 2),
        "resistance": round(trigger_resistance, 2),
        "score": score,
        "notes": notes,
        "pattern": setup["pattern"],
        "bias_candle_timestamp": setup["bias_close_time"],
        "trigger_timestamp": setup["trigger_time_close"],
        "trigger_level": round(float(setup["trigger_level"]), 2),
        "volume_multiple": round(float(setup["volume_multiple"]), 2),
        "bias_volume_multiple": round(float(setup["volume_multiple"]), 2),
        "bias_volume": round(float(setup["bias_volume"]), 2) if setup.get("bias_volume") is not None else None,
        "bias_volume_sma20": round(float(setup["bias_volume_sma20"]), 2)
        if setup.get("bias_volume_sma20") is not None
        else None,
        "trigger_volume_multiple": round(float(setup["trigger_volume_multiple"]), 2)
        if setup.get("trigger_volume_multiple") is not None
        else None,
        "signal_age_bars": signal_age_bars,
        "fresh_until": _shift_ts(setup["trigger_time_close"], _sse_i("FAILED_BIAS_MAX_AGE_BARS", 4) * 15),
        "minutes_left_at_trigger": minutes_left_at_trigger,
        "_metrics_end_idx": trigger_idx,
    }
    payload.update(score.get("checks", {}))
    return payload


def _latest_failed_bias_setup(
    frame: pd.DataFrame,
    history_15m: pd.DataFrame | None = None,
) -> Dict[str, Any] | None:
    if len(frame) < _sse_i("FAILED_BIAS_MIN_BARS", 1) + 1:
        return None

    last_close = float(frame.iloc[-1]["spot_ltp"])
    last_ts = str(frame.iloc[-1]["timestamp"])
    for idx in range(len(frame) - 2, _sse_i("FAILED_BIAS_MIN_BARS", 1) - 2, -1):
        bias = _volume_bias_candle(frame, idx, history_15m=history_15m)
        if bias is None:
            continue
        future = frame.iloc[idx + 1 :]
        if future.empty:
            continue
        bias_rsi = _rsi_15m_value_at(frame, cutoff_ts=str(frame.iloc[idx]["timestamp"]), history_15m=history_15m)

        if bias["bias"] == "BEARISH_BIAS":
            if bias_rsi is None or bias_rsi <= _sse_f("FAILED_BIAS_RSI_BUY_FLOOR", 45.0):
                continue
            crossed = future[future["spot_ltp"].astype(float) > float(bias["bias_high"])]
            if crossed.empty or last_close <= float(bias["bias_high"]):
                continue
            trigger_row = crossed.iloc[0]
            trigger_idx = int(crossed.index[0])
            return {
                **bias,
                "bias_rsi_15m": round(float(bias_rsi), 2),
                "signal": "BUY",
                "pattern": "FAILED_BEARISH_BIAS_LONG",
                "trigger_idx": trigger_idx,
                "trigger_level": float(bias["bias_high"]),
                "trigger_time": str(trigger_row["timestamp"]),
                "trigger_time_close": _close_ts(str(trigger_row["timestamp"]), 15),
                "trigger_volume_multiple": _volume_multiple_at(frame, trigger_idx, history_15m=history_15m),
                "current_timestamp": last_ts,
            }

        if bias_rsi is None or bias_rsi >= _sse_f("FAILED_BIAS_RSI_SELL_CEILING", 55.0):
            continue
        crossed = future[future["spot_ltp"].astype(float) < float(bias["bias_low"])]
        if crossed.empty or last_close >= float(bias["bias_low"]):
            continue
        trigger_row = crossed.iloc[0]
        trigger_idx = int(crossed.index[0])
        return {
            **bias,
            "bias_rsi_15m": round(float(bias_rsi), 2),
            "signal": "SELL",
            "pattern": "FAILED_BULLISH_BIAS_SHORT",
            "trigger_idx": trigger_idx,
            "trigger_level": float(bias["bias_low"]),
            "trigger_time": str(trigger_row["timestamp"]),
            "trigger_time_close": _close_ts(str(trigger_row["timestamp"]), 15),
            "trigger_volume_multiple": _volume_multiple_at(frame, trigger_idx, history_15m=history_15m),
            "current_timestamp": last_ts,
        }
    return None


def _volume_bias_candle(
    frame: pd.DataFrame,
    idx: int,
    history_15m: pd.DataFrame | None = None,
) -> Dict[str, Any] | None:
    if idx < _sse_i("FAILED_BIAS_MIN_BARS", 1) - 1:
        return None

    row = frame.iloc[idx]
    current_volume = float(row.get("spot_volume", 0) or 0)
    if current_volume <= 0:
        return None
    volume_sma = _volume_sma_till_candle(
        current_ts=str(row["timestamp"]),
        current_idx=idx,
        frame=frame,
        history_15m=history_15m,
    )
    if volume_sma is None:
        return None
    if volume_sma <= 0 or current_volume <= _sse_f("FAILED_BIAS_VOLUME_MULTIPLIER", 1.5) * volume_sma:
        return None

    candle_open = float(row.get("spot_open_raw", row.get("spot_ltp", 0)) or 0)
    candle_close = float(row.get("spot_ltp", 0) or 0)
    if candle_close == candle_open:
        return None

    bias = "BULLISH_BIAS" if candle_close > candle_open else "BEARISH_BIAS"
    return {
        "bias": bias,
        "bias_high": float(row.get("spot_high", candle_close) or candle_close),
        "bias_low": float(row.get("spot_low", candle_close) or candle_close),
        "bias_open": candle_open,
        "bias_close": candle_close,
        "bias_timestamp": str(row["timestamp"]),
        "bias_close_time": _close_ts(str(row["timestamp"]), 15),
        "bias_volume": current_volume,
        "bias_volume_sma20": volume_sma,
        "volume_multiple": current_volume / volume_sma,
    }


def _volume_multiple_at(
    frame: pd.DataFrame,
    idx: int,
    history_15m: pd.DataFrame | None = None,
) -> float | None:
    if idx < _sse_i("FAILED_BIAS_MIN_BARS", 1) - 1 or idx >= len(frame):
        return None
    row = frame.iloc[idx]
    current_volume = float(row.get("spot_volume", 0) or 0)
    if current_volume <= 0:
        return None
    volume_sma = _volume_sma_till_candle(
        current_ts=str(row["timestamp"]),
        current_idx=idx,
        frame=frame,
        history_15m=history_15m,
    )
    if volume_sma <= 0:
        return None
    return current_volume / volume_sma


def _volume_sma_till_candle(
    current_ts: str,
    current_idx: int,
    frame: pd.DataFrame,
    history_15m: pd.DataFrame | None = None,
) -> float | None:
    if history_15m is not None and not history_15m.empty:
        history = history_15m.copy()
        history["timestamp"] = pd.to_datetime(history["timestamp"])
        upto_current = history[history["timestamp"] <= pd.to_datetime(current_ts)]
        if "volume" in upto_current.columns:
            volumes = upto_current["volume"].astype(float).tail(_sse_i("FAILED_BIAS_SMA_WINDOW", 20))
            if len(volumes) >= _sse_i("FAILED_BIAS_MIN_BARS", 1):
                return float(volumes.mean())

    start_idx = max(0, current_idx - _sse_i("FAILED_BIAS_SMA_WINDOW", 20) + 1)
    volumes = frame["spot_volume"].iloc[start_idx : current_idx + 1].astype(float)
    if len(volumes) < _sse_i("FAILED_BIAS_MIN_BARS", 1):
        return None
    return float(volumes.mean())


def _failed_bias_score(
    frame: pd.DataFrame,
    setup: Dict[str, Any],
    support: float,
    resistance: float,
    momentum: str,
) -> Dict[str, Any]:
    last = frame.iloc[-1]
    prev = frame.iloc[-2] if len(frame) > 1 else last
    spot = float(last["spot_ltp"])
    candle_open = float(last.get("spot_open_raw", prev.get("spot_ltp", spot)) or spot)
    candle_high = float(last.get("spot_high", spot) or spot)
    candle_low = float(last.get("spot_low", spot) or spot)
    candle_range = max(candle_high - candle_low, 1e-6)
    close_position = (spot - candle_low) / candle_range

    future_now = float(last.get("future_ltp", 0) or 0)
    future_prev = float(prev.get("future_ltp", 0) or 0)
    future_vwap = float(last.get("future_vwap", 0) or 0)
    call_now = float(last.get("call_ltp", 0) or 0)
    call_prev = float(prev.get("call_ltp", 0) or 0)
    put_now = float(last.get("put_ltp", 0) or 0)
    put_prev = float(prev.get("put_ltp", 0) or 0)
    call_change = ((call_now - call_prev) / call_prev * 100) if call_prev > 0 else 0.0
    put_change = ((put_now - put_prev) / put_prev * 100) if put_prev > 0 else 0.0
    bias_volume_multiple = float(setup.get("volume_multiple") or 0)
    trigger_volume_multiple = float(setup.get("trigger_volume_multiple") or 0)
    checks = {
        "bias_volume_multiple": round(bias_volume_multiple, 2),
        "momentum_up": False,
        "momentum_down": False,
        "futures_above_vwap": False,
        "futures_below_vwap": False,
        "calls_expanding": call_change > 0,
        "calls_decaying": call_change < 0,
        "puts_expanding": put_change > 0,
        "puts_decaying": put_change < 0,
        "trigger_near_high": False,
        "trigger_near_low": False,
        "break_resistance": False,
        "break_support": False,
        "bias_volume_strong": bias_volume_multiple >= 1.75,
        "trigger_volume_confirmed": trigger_volume_multiple >= 0.9,
    }

    reasons = [
        f"{setup['pattern']} triggered after reclaim/failure of {setup['trigger_level']:.2f}.",
        f"Bias candle closed at {setup['bias_close_time']} with {bias_volume_multiple:.2f}x volume.",
    ]
    confidence = 0.62
    bullish = 0.0
    bearish = 0.0
    if checks["bias_volume_strong"]:
        confidence += 0.04
        reasons.append("Bias candle volume was meaningfully above its 20-SMA.")
    if checks["trigger_volume_confirmed"]:
        confidence += 0.04
        reasons.append("Trigger candle volume confirmed the reclaim/failure.")

    if setup["signal"] == "BUY":
        bullish = 0.62
        checks["momentum_up"] = momentum == "UP"
        checks["futures_above_vwap"] = future_now > future_prev and future_now >= future_vwap > 0
        checks["trigger_near_high"] = close_position >= 0.65
        checks["break_resistance"] = spot > resistance
        if momentum == "UP":
            confidence += 0.10
            reasons.append("15-min momentum is UP.")
        if checks["futures_above_vwap"]:
            confidence += 0.08
            reasons.append("Futures are rising above VWAP.")
        if call_change > 0 and put_change < 0:
            confidence += 0.10
            reasons.append("Calls are expanding while puts are decaying.")
        if checks["trigger_near_high"]:
            confidence += 0.05
            reasons.append("Trigger candle closed near its high.")
        if checks["break_resistance"]:
            confidence += 0.05
            reasons.append("Price is breaking session resistance.")
    else:
        bearish = 0.62
        checks["momentum_down"] = momentum == "DOWN"
        checks["futures_below_vwap"] = future_now < future_prev and future_now <= future_vwap
        checks["trigger_near_low"] = close_position <= 0.35
        checks["break_support"] = spot < support
        if momentum == "DOWN":
            confidence += 0.10
            reasons.append("15-min momentum is DOWN.")
        if checks["futures_below_vwap"]:
            confidence += 0.08
            reasons.append("Futures are weakening below VWAP.")
        if put_change > 0 and call_change < 0:
            confidence += 0.10
            reasons.append("Puts are expanding while calls are decaying.")
        if checks["trigger_near_low"]:
            confidence += 0.05
            reasons.append("Trigger candle closed near its low.")
        if checks["break_support"]:
            confidence += 0.05
            reasons.append("Price is breaking session support.")

    confidence = round(max(0.0, min(0.95, confidence)), 4)
    final_score = round(confidence if setup["signal"] == "BUY" else -confidence, 4)
    return {
        "bullish": round(bullish, 4),
        "bearish": round(bearish, 4),
        "no_trade_penalty": 0.0,
        "final_score": final_score,
        "confidence": confidence,
        "reasons": reasons,
        "checks": checks,
    }


def _failed_bias_quality_reason(
    signal: str,
    entry: float,
    risk: float,
    minutes_left_at_trigger: int,
    score: Dict[str, Any],
) -> str | None:
    checks = score.get("checks", {}) if isinstance(score, dict) else {}
    stop_pct = (risk / max(entry, 1.0)) * 100.0
    minutes_from_open = 375 - minutes_left_at_trigger
    if minutes_from_open < _sse_i("FAILED_BIAS_MINUTES_FROM_OPEN", 105):
        return f"Failed-bias rejected: trigger came too early ({minutes_from_open} mins from open)."
    max_stop = _sse_f("FAILED_BIAS_MAX_STOP_PCT", 0.50)
    if stop_pct > max_stop:
        return f"Failed-bias rejected: stop is too wide ({stop_pct:.2f}% > {max_stop:.2f}%)."

    if signal == "BUY":
        if not checks.get("momentum_up"):
            return "Failed-bias BUY rejected: 15m momentum is not UP."
        if not checks.get("trigger_near_high"):
            return "Failed-bias BUY rejected: trigger candle is not strong enough at the highs."
        return None

    if not checks.get("momentum_down"):
        return "Failed-bias SELL rejected: 15m momentum is not DOWN."
    if not checks.get("trigger_near_low"):
        return "Failed-bias SELL rejected: trigger candle is not strong enough at the lows."
    return None


def _failed_bias_rsi_reason(
    signal: str,
    frame: pd.DataFrame,
    history_15m: pd.DataFrame | None,
    trigger_ts: str,
) -> str | None:
    lb = _sse_i("FAILED_BIAS_RSI_LOOKBACK_CANDLES", 10)
    buy_floor = _sse_f("FAILED_BIAS_RSI_BUY_FLOOR", 45.0)
    buy_reg = _sse_f("FAILED_BIAS_RSI_BUY_REGIME", 55.0)
    sell_ceil = _sse_f("FAILED_BIAS_RSI_SELL_CEILING", 55.0)
    sell_reg = _sse_f("FAILED_BIAS_RSI_SELL_REGIME", 45.0)
    current_rsi = _rsi_15m_value_at(frame, cutoff_ts=trigger_ts, history_15m=history_15m)
    recent_rsi = _rsi_15m_recent_values(frame, cutoff_ts=trigger_ts, history_15m=history_15m)

    if current_rsi is None:
        return "Failed-bias rejected: 15m RSI is unavailable at trigger."
    if len(recent_rsi) < lb:
        return (
            f"Failed-bias rejected: need {lb} prior 15m RSI values "
            "to validate regime."
        )

    if signal == "BUY":
        if current_rsi < buy_floor:
            return f"Failed-bias BUY rejected: trigger RSI is below {buy_floor:.0f}."
        if any(value <= buy_reg for value in recent_rsi):
            return (
                f"Failed-bias BUY rejected: last {lb} RSI readings "
                f"must stay above {buy_reg:.0f}."
            )
        return None

    if current_rsi > sell_ceil:
        return f"Failed-bias SELL rejected: trigger RSI is above {sell_ceil:.0f}."
    if any(value >= sell_reg for value in recent_rsi):
        return (
            f"Failed-bias SELL rejected: last {lb} RSI readings "
            f"must stay below {sell_reg:.0f}."
        )
    return None


def _close_ts(ts: str, minutes: int) -> str:
    return (pd.to_datetime(ts) + pd.Timedelta(minutes=minutes)).strftime("%Y-%m-%dT%H:%M:%S")


def _shift_ts(ts: str, minutes: int) -> str:
    return (pd.to_datetime(ts) + pd.Timedelta(minutes=minutes)).strftime("%Y-%m-%dT%H:%M:%S")


def _minutes_to_session_close(close_ts: str) -> int:
    ts = pd.to_datetime(close_ts)
    session_close = ts.normalize() + pd.Timedelta(hours=15, minutes=30)
    return max(0, int((session_close - ts).total_seconds() // 60))


def _common_signal_metrics(
    frame: pd.DataFrame,
    benchmark_frame: pd.DataFrame | None = None,
    rsi_context: Dict[str, pd.DataFrame] | None = None,
) -> Dict[str, Any]:
    cutoff_ts = str(frame.iloc[-1]["timestamp"]) if not frame.empty else None
    return {
        "rsi_15m": _rsi_15m(frame, rsi_context=rsi_context, cutoff_ts=cutoff_ts),
        "rsi_60m": _rsi_60m(frame, rsi_context=rsi_context, cutoff_ts=cutoff_ts),
        "future_oi_change_pct": _future_oi_change_pct(frame),
        "relative_strength_nifty": _relative_strength_vs_benchmark(frame, benchmark_frame),
    }


def _rsi_15m(
    frame: pd.DataFrame,
    rsi_context: Dict[str, pd.DataFrame] | None = None,
    cutoff_ts: str | None = None,
) -> float | None:
    history = (rsi_context or {}).get("m15")
    if history is not None and not history.empty and cutoff_ts:
        subset = history[pd.to_datetime(history["timestamp"]) <= pd.to_datetime(cutoff_ts)]
        value = rsi_last(subset["close"], 14) if not subset.empty else None
    else:
        if frame.empty:
            return None
        value = rsi_last(frame["spot_ltp"], 14)
    return round(value, 2) if value is not None else None


def _rsi_15m_value_at(
    frame: pd.DataFrame,
    cutoff_ts: str | None,
    history_15m: pd.DataFrame | None = None,
) -> float | None:
    series = _rsi_15m_series_until(frame, cutoff_ts=cutoff_ts, history_15m=history_15m)
    if series.empty:
        return None
    return round(float(series.iloc[-1]), 2)


def _rsi_15m_recent_values(
    frame: pd.DataFrame,
    cutoff_ts: str | None,
    history_15m: pd.DataFrame | None = None,
    window: int = _sse_i("FAILED_BIAS_RSI_LOOKBACK_CANDLES", 10),
) -> list[float]:
    series = _rsi_15m_series_until(frame, cutoff_ts=cutoff_ts, history_15m=history_15m)
    if series.empty:
        return []
    return [round(float(value), 2) for value in series.tail(window).tolist()]


def _rsi_15m_series_until(
    frame: pd.DataFrame,
    cutoff_ts: str | None,
    history_15m: pd.DataFrame | None = None,
) -> pd.Series:
    if history_15m is not None and not history_15m.empty:
        subset = history_15m
        if cutoff_ts:
            subset = subset[pd.to_datetime(subset["timestamp"]) <= pd.to_datetime(cutoff_ts)]
        close = subset["close"] if not subset.empty else pd.Series(dtype=float)
    else:
        if frame.empty:
            return pd.Series(dtype=float)
        subset = frame
        if cutoff_ts:
            subset = subset[pd.to_datetime(subset["timestamp"]) <= pd.to_datetime(cutoff_ts)]
        close = subset["spot_ltp"] if not subset.empty else pd.Series(dtype=float)
    if close.empty:
        return pd.Series(dtype=float)
    return rsi_series(close, 14).dropna().reset_index(drop=True)


def _rsi_60m(
    frame: pd.DataFrame,
    rsi_context: Dict[str, pd.DataFrame] | None = None,
    cutoff_ts: str | None = None,
) -> float | None:
    history = (rsi_context or {}).get("h1")
    if history is None or history.empty or not cutoff_ts:
        return None
    subset = history[pd.to_datetime(history["timestamp"]) <= pd.to_datetime(cutoff_ts)]
    if subset.empty:
        return None
    value = rsi_last(subset["close"], 14)
    return round(value, 2) if value is not None else None


def _future_oi_change_pct(frame: pd.DataFrame) -> float | None:
    if frame.empty or "future_oi" not in frame.columns:
        return None
    first = float(frame.iloc[0].get("future_oi", 0) or 0)
    last = float(frame.iloc[-1].get("future_oi", 0) or 0)
    if first <= 0:
        return None
    return round(((last - first) / first) * 100.0, 2)


def _relative_strength_vs_benchmark(
    frame: pd.DataFrame,
    benchmark_frame: pd.DataFrame | None,
) -> float | None:
    if frame.empty or benchmark_frame is None or benchmark_frame.empty:
        return None
    last_ts = str(frame.iloc[-1]["timestamp"])
    benchmark_rows = benchmark_frame[benchmark_frame["timestamp"].astype(str) <= last_ts]
    if benchmark_rows.empty:
        return None
    stock_open = float(frame.iloc[0].get("spot_open_raw", 0) or 0)
    stock_last = float(frame.iloc[-1].get("spot_ltp", 0) or 0)
    benchmark_open = float(benchmark_frame.iloc[0].get("spot_open_raw", 0) or 0)
    benchmark_last = float(benchmark_rows.iloc[-1].get("spot_close", 0) or 0)
    if stock_open <= 0 or benchmark_open <= 0:
        return None
    stock_return = ((stock_last - stock_open) / stock_open) * 100.0
    benchmark_return = ((benchmark_last - benchmark_open) / benchmark_open) * 100.0
    return round(stock_return - benchmark_return, 2)


def _fetch_spot_rsi_context(
    client: ZerodhaClient,
    stock_name: str,
    trade_date: date | None,
) -> Dict[str, pd.DataFrame]:
    context: Dict[str, pd.DataFrame] = {"m15": pd.DataFrame(), "h1": pd.DataFrame()}
    try:
        uc = get_underlying_config(stock_name)
        from_dt, to_dt = _market_window_15min(trade_date)
        if to_dt < from_dt:
            return context
        quote = client.quote([uc.spot_symbol])
        if not quote or uc.spot_symbol not in quote:
            return context
        spot_token = int(quote[uc.spot_symbol]["instrument_token"])
        trade_day = trade_date or datetime.now().date()

        m15_rows = client.historical_data(
            spot_token,
            datetime.combine(trade_day, datetime.min.time()) - timedelta(days=40),
            to_dt,
            interval="15minute",
        )
        m15 = _drop_incomplete_candles(_to_candle_df(m15_rows, "spot"), trade_date, 15)
        if not m15.empty:
            context["m15"] = m15.rename(columns={"spot_close": "close", "spot_volume": "volume"})[
                ["timestamp", "close", "volume"]
            ].copy()

        h1_rows = client.historical_data(
            spot_token,
            datetime.combine(trade_day, datetime.min.time()) - timedelta(days=120),
            to_dt,
            interval="60minute",
        )
        h1 = _drop_incomplete_candles(_to_candle_df(h1_rows, "spot"), trade_date, 60)
        if not h1.empty:
            context["h1"] = h1.rename(columns={"spot_close": "close"})[["timestamp", "close"]].copy()
    except Exception as exc:
        logger.debug("RSI history fetch skipped for %s: %s", stock_name, exc)
    return context


def _signal_side_columns(
    signal: str | None,
    orb_state: Dict[str, Any],
    pinbar_state: Dict[str, Any],
) -> Dict[str, Any]:
    orb_signal = str(orb_state.get("signal", "NO_TRADE"))
    pinbar_signal = str(pinbar_state.get("signal", "NO_TRADE"))
    return {
        "orb_signal": orb_signal,
        "orb_active": orb_signal in {"BUY", "SELL"},
        "orb_timestamp": orb_state.get("timestamp"),
        "pinbar_signal": pinbar_signal,
        "pinbar_active": pinbar_signal in {"BUY", "SELL"},
        "pinbar_pattern": pinbar_state.get("pattern"),
        "pinbar_timestamp": pinbar_state.get("timestamp"),
    }


def _orb_state(frame: pd.DataFrame) -> Dict[str, Any]:
    if frame.empty or len(frame) < 2:
        return {"signal": "NO_TRADE", "timestamp": None}
    first = frame.iloc[0]
    or_high = float(first.get("spot_high", 0) or 0)
    or_low = float(first.get("spot_low", 0) or 0)
    variation = _sse_f("ORB_VARIATION_PCT", 0.2) / 100.0
    upper_break = or_high * (1 - variation)
    lower_break = or_low * (1 + variation)
    row = frame.iloc[-1]
    close_ = float(row.get("spot_ltp", 0) or 0)
    if close_ <= 0:
        return {"signal": "NO_TRADE", "timestamp": None}
    close_ts = _close_ts(str(row["timestamp"]), 15)
    if close_ >= upper_break:
        return {"signal": "BUY", "timestamp": close_ts}
    if close_ <= lower_break:
        return {"signal": "SELL", "timestamp": close_ts}
    return {"signal": "NO_TRADE", "timestamp": None}


def _pinbar_state(frame: pd.DataFrame) -> Dict[str, Any]:
    if frame.empty:
        return {"signal": "NO_TRADE", "pattern": None, "timestamp": None}
    row = frame.iloc[-1]
    o = float(row.get("spot_open_raw", row.get("spot_ltp", 0)) or 0)
    h = float(row.get("spot_high", 0) or 0)
    l_ = float(row.get("spot_low", 0) or 0)
    c = float(row.get("spot_ltp", 0) or 0)
    ts = _close_ts(str(row["timestamp"]), 15)
    if _is_bullish_pinbar(o, h, l_, c):
        return {"signal": "BUY", "pattern": "BULLISH_PINBAR", "timestamp": ts}
    if _is_bearish_pinbar(o, h, l_, c):
        return {"signal": "SELL", "pattern": "BEARISH_PINBAR", "timestamp": ts}
    return {"signal": "NO_TRADE", "pattern": None, "timestamp": None}


def _is_bullish_pinbar(o: float, h: float, l_: float, c: float) -> bool:
    if h <= l_:
        return False
    body = abs(c - o)
    range_ = h - l_
    lower_wick = min(o, c) - l_
    upper_wick = h - max(o, c)
    if range_ <= 0 or body <= 0:
        return False
    wbr = _cp_f("PINBAR_WICK_BODY_RATIO", 2.0)
    bmr = _cp_f("PINBAR_BODY_MAX_RANGE_FRAC", 0.35)
    opr = _cp_f("PINBAR_OPP_WICK_MAX_RATIO", 0.5)
    return lower_wick >= wbr * body and body <= bmr * range_ and upper_wick <= opr * lower_wick


def _is_bearish_pinbar(o: float, h: float, l_: float, c: float) -> bool:
    if h <= l_:
        return False
    body = abs(c - o)
    range_ = h - l_
    upper_wick = h - max(o, c)
    lower_wick = min(o, c) - l_
    if range_ <= 0 or body <= 0:
        return False
    wbr = _cp_f("PINBAR_WICK_BODY_RATIO", 2.0)
    bmr = _cp_f("PINBAR_BODY_MAX_RANGE_FRAC", 0.35)
    opr = _cp_f("PINBAR_OPP_WICK_MAX_RATIO", 0.5)
    return upper_wick >= wbr * body and body <= bmr * range_ and lower_wick <= opr * upper_wick


def run_stock_analysis_30min(
    client: ZerodhaClient,
    stock_name: str,
    trade_date: date | None = None,
    include_options: bool = True,
) -> tuple[pd.DataFrame | None, list[Dict[str, Any]]]:
    """
    Fetch 30-min data for stock, run analysis in-memory (no persistence).
    Returns (merged_df, signals_list). Both can be None/empty on failure.
    """
    merged = fetch_stock_30min_frame(client, stock_name, trade_date, include_options=include_options)
    if merged is None or merged.empty or len(merged) < 2:
        return (None, [])

    settings = Settings.from_env(underlying=stock_name)
    signals: list[Dict[str, Any]] = []
    for idx in range(len(merged)):
        frame = merged.iloc[: idx + 1].reset_index(drop=True)
        payload = _analyze_frame(frame, stock_name, settings)
        if payload:
            signals.append(payload)
    return (merged, signals)


def _no_trade(
    frame: pd.DataFrame,
    stock_name: str,
    support: float,
    resistance: float,
    reason: str,
    bias: str,
    momentum: str,
    strategy: str | None = None,
) -> Dict[str, Any]:
    last = frame.iloc[-1]
    return {
        "stock": stock_name,
        "strategy": strategy,
        "strategy_label": strategy,
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
