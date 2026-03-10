"""Sideways/chop day detection to avoid trades where SL gets hit repeatedly."""

from __future__ import annotations

from typing import Dict


def is_sideways_day(
    range_pct: float,
    bias: str,
    momentum: str,
    follow_through: bool,
    is_breakout: bool,
    is_breakdown: bool,
    features: Dict[str, float],
    min_range_pct: float,
) -> tuple[bool, str]:
    """
    Returns (is_sideways, reason).
    Sideways when: narrow range, or chop + no clear structure break.
    """
    # Narrow range: session is compressed, likely to whipsaw
    if range_pct < min_range_pct:
        return True, f"Sideways day: range {range_pct:.2f}% below {min_range_pct}% threshold."

    # Chop around open/VWAP with no clear break
    chop = not (
        (features["spot_above_open"] and features["spot_above_vwap"])
        or (features["spot_below_open"] and features["spot_below_vwap"])
    )
    no_structure_break = not (is_breakout or is_breakdown)
    neutral = bias == "NEUTRAL_DAY" and momentum == "NEUTRAL"

    if chop and no_structure_break and neutral:
        return True, "Sideways day: chop around open/VWAP, no structure break, neutral bias."

    if chop and no_structure_break and not follow_through:
        return True, "Sideways day: chop with no follow-through on structure."

    return False, ""
