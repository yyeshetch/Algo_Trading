from __future__ import annotations

from typing import Dict, List

from intraday_engine.core.models import ScoreBreakdown


WEIGHTS = {
    "spot_open_vwap_alignment": 0.20,
    "fut_strength": 0.15,
    "options_expansion_decay": 0.20,
    "breakout_follow_through": 0.20,
    "momentum": 0.15,
    "structure_quality": 0.10,
}


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
        bullish += WEIGHTS["spot_open_vwap_alignment"]
        reasons.append("Spot is above open and VWAP.")
    elif features["spot_below_open"] and features["spot_below_vwap"]:
        bearish += WEIGHTS["spot_open_vwap_alignment"]
        reasons.append("Spot is below open and VWAP.")
    else:
        no_trade_penalty += 0.10
        reasons.append("Spot is chopping around open/VWAP.")

    if features["fut_strength_pct"] > 0.03:
        bullish += WEIGHTS["fut_strength"]
        reasons.append("Futures are stronger than spot.")
    elif features["fut_strength_pct"] < -0.03:
        bearish += WEIGHTS["fut_strength"]
        reasons.append("Futures are weaker than spot.")

    if features.get("options_available", True):
        if features["call_change_pct"] > 0 and features["put_change_pct"] < 0:
            bullish += WEIGHTS["options_expansion_decay"]
            reasons.append("ATM call is expanding while ATM put is decaying.")
        elif features["put_change_pct"] > 0 and features["call_change_pct"] < 0:
            bearish += WEIGHTS["options_expansion_decay"]
            reasons.append("ATM put is expanding while ATM call is decaying.")
        else:
            no_trade_penalty += 0.08
            reasons.append("Options behavior is conflicting.")

    if is_breakout and follow_through:
        bullish += WEIGHTS["breakout_follow_through"]
        reasons.append("Breakout with follow-through.")
    elif is_breakdown and follow_through:
        bearish += WEIGHTS["breakout_follow_through"]
        reasons.append("Breakdown with follow-through.")
    else:
        no_trade_penalty += 0.08
        reasons.append("No reliable follow-through on structure break.")

    if momentum == "UP":
        bullish += WEIGHTS["momentum"]
        reasons.append("Momentum direction is UP.")
    elif momentum == "DOWN":
        bearish += WEIGHTS["momentum"]
        reasons.append("Momentum direction is DOWN.")
    else:
        no_trade_penalty += 0.05
        reasons.append("Momentum is neutral.")

    if is_mid_range:
        no_trade_penalty += WEIGHTS["structure_quality"]
        reasons.append("Price is in the middle of the range.")
    else:
        if bias == "BULLISH_DAY":
            bullish += WEIGHTS["structure_quality"]
        elif bias == "BEARISH_DAY":
            bearish += WEIGHTS["structure_quality"]

    if stop_too_wide:
        no_trade_penalty += 0.15
        reasons.append("Stop loss is too wide for intraday risk.")

    direction_strength = abs(bullish - bearish)
    confidence = max(0.0, min(1.0, direction_strength - no_trade_penalty + 0.5))
    final_score = bullish - bearish - no_trade_penalty

    return ScoreBreakdown(
        bullish=round(bullish, 4),
        bearish=round(bearish, 4),
        no_trade_penalty=round(no_trade_penalty, 4),
        final_score=round(final_score, 4),
        confidence=round(confidence, 4),
        reasons=reasons,
    )

