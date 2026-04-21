from __future__ import annotations

from typing import Dict, List

from intraday_engine.core.models import ScoreBreakdown


def _w(key: str, default: float) -> float:
    from intraday_engine.core.tunables import get_nested_float

    return get_nested_float("scoring", default, "weights", key)


def _pen(key: str, default: float) -> float:
    from intraday_engine.core.tunables import get_nested_float

    return get_nested_float("scoring", default, "penalties", key)


def _thr(key: str, default: float) -> float:
    from intraday_engine.core.tunables import get_nested_float

    return get_nested_float("scoring", default, "thresholds", key)


def score_signal(
    features: Dict[str, float],
    momentum: str,
    bias: str,
    is_breakout: bool,
    is_breakdown: bool,
    follow_through: bool,
    stop_too_wide: bool,
    is_mid_range: bool,
) -> ScoreBreakdown:
    bullish = 0.0
    bearish = 0.0
    no_trade_penalty = 0.0
    reasons: List[str] = []

    if features["spot_above_open"] and features["spot_above_vwap"]:
        bullish += _w("spot_open_vwap_alignment", 0.18)
        reasons.append("Spot is above open and VWAP.")
    elif features["spot_below_open"] and features["spot_below_vwap"]:
        bearish += _w("spot_open_vwap_alignment", 0.18)
        reasons.append("Spot is below open and VWAP.")
    else:
        no_trade_penalty += _pen("chop_open_vwap", 0.10)
        reasons.append("Spot is chopping around open/VWAP.")

    if features["fut_strength_pct"] > _thr("fut_strength_pct", 0.03):
        bullish += _w("fut_strength", 0.12)
        reasons.append("Futures are stronger than spot.")
    elif features["fut_strength_pct"] < -_thr("fut_strength_pct", 0.03):
        bearish += _w("fut_strength", 0.12)
        reasons.append("Futures are weaker than spot.")

    if features.get("fut_oi_available"):
        if features.get("fut_oi_bullish"):
            bullish += _w("fut_oi", 0.10)
            reasons.append("Fut OI: longs adding or short covering.")
        elif features.get("fut_oi_bearish"):
            bearish += _w("fut_oi", 0.10)
            reasons.append("Fut OI: shorts adding or long covering.")

    if features.get("options_available", True):
        if features["call_change_pct"] > 0 and features["put_change_pct"] < 0:
            bullish += _w("options_expansion_decay", 0.15)
            reasons.append("ATM call is expanding while ATM put is decaying.")
        elif features["put_change_pct"] > 0 and features["call_change_pct"] < 0:
            bearish += _w("options_expansion_decay", 0.15)
            reasons.append("ATM put is expanding while ATM call is decaying.")
        else:
            no_trade_penalty += _pen("options_conflict", 0.08)
            reasons.append("Options behavior is conflicting.")

    if features.get("oi_available"):
        ce_oi = features.get("call_oi_change_pct", 0) or 0
        pe_oi = features.get("put_oi_change_pct", 0) or 0
        oi_abs = _thr("options_oi_change_abs", 2.0)
        if ce_oi > oi_abs and pe_oi < -oi_abs:
            bullish += _w("options_oi", 0.10)
            reasons.append("CE OI up, PE OI down (call buying).")
        elif pe_oi > oi_abs and ce_oi < -oi_abs:
            bearish += _w("options_oi", 0.10)
            reasons.append("PE OI up, CE OI down (put buying).")

    if is_breakout and follow_through:
        bullish += _w("breakout_follow_through", 0.18)
        reasons.append("Breakout with follow-through.")
    elif is_breakdown and follow_through:
        bearish += _w("breakout_follow_through", 0.18)
        reasons.append("Breakdown with follow-through.")
    else:
        no_trade_penalty += _pen("no_follow_through", 0.08)
        reasons.append("No reliable follow-through on structure break.")

    if momentum == "UP":
        bullish += _w("momentum", 0.12)
        reasons.append("Momentum direction is UP.")
    elif momentum == "DOWN":
        bearish += _w("momentum", 0.12)
        reasons.append("Momentum direction is DOWN.")
    else:
        no_trade_penalty += _pen("momentum_neutral", 0.05)
        reasons.append("Momentum is neutral.")

    if is_mid_range:
        no_trade_penalty += _w("structure_quality", 0.05)
        reasons.append("Price is in the middle of the range.")
    else:
        if bias == "BULLISH_DAY":
            bullish += _w("structure_quality", 0.05)
        elif bias == "BEARISH_DAY":
            bearish += _w("structure_quality", 0.05)

    if stop_too_wide:
        no_trade_penalty += _pen("stop_too_wide", 0.15)
        reasons.append("Stop loss is too wide for intraday risk.")

    direction_strength = abs(bullish - bearish)
    confidence = max(
        0.0,
        min(1.0, direction_strength - no_trade_penalty + _thr("confidence_offset", 0.5)),
    )
    final_score = bullish - bearish - no_trade_penalty

    return ScoreBreakdown(
        bullish=round(bullish, 4),
        bearish=round(bearish, 4),
        no_trade_penalty=round(no_trade_penalty, 4),
        final_score=round(final_score, 4),
        confidence=round(confidence, 4),
        reasons=reasons,
    )

