from __future__ import annotations

from typing import Dict


def probable_day_bias(features: Dict[str, float]) -> str:
    bullish = 0
    bearish = 0

    if features["spot_above_open"] and features["spot_above_vwap"]:
        bullish += 1
    if features["spot_below_open"] and features["spot_below_vwap"]:
        bearish += 1

    if features["fut_strength_pct"] > 0.03:
        bullish += 1
    elif features["fut_strength_pct"] < -0.03:
        bearish += 1

    if features["call_change_pct"] > 0 and features["put_change_pct"] < 0:
        bullish += 1
    elif features["put_change_pct"] > 0 and features["call_change_pct"] < 0:
        bearish += 1

    if bullish > bearish:
        return "BULLISH_DAY"
    if bearish > bullish:
        return "BEARISH_DAY"
    return "NEUTRAL_DAY"

