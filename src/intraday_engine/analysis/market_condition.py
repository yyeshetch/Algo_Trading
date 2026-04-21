"""
Market condition detection: NORMAL vs PRESSURE (bullish/bearish) vs VOLATILITY_SPIKE.

Helps identify:
- Normal choppy days (avoid aggressive trades)
- Pressure building (bearish/bullish) before huge moves
- Volatility spike (large move in progress)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List


@dataclass
class MarketCondition:
    regime: str  # NORMAL | PRESSURE_BULLISH | PRESSURE_BEARISH | VOLATILITY_SPIKE
    confidence: float
    range_pct: float
    volume_ratio: float  # vs 5-day avg
    pcr_oi: float
    pcr_volume: float
    call_premium_chg_pct: float
    put_premium_chg_pct: float
    ce_oi_chg_pct: float
    pe_oi_chg_pct: float
    reasons: List[str]


def detect_market_condition(
    range_pct: float,
    volume_ratio: float,
    pcr_oi: float,
    pcr_volume: float,
    call_premium_chg_pct: float,
    put_premium_chg_pct: float,
    ce_oi_chg_pct: float,
    pe_oi_chg_pct: float,
    avg_range_pct: float | None = None,
) -> MarketCondition:
    """
    Classify market regime from option chain and price metrics.

    Args:
        range_pct: Session range (high-low)/open * 100
        volume_ratio: Today's spot+fut volume / 5-day avg (1.0 = normal)
        pcr_oi: PE OI / CE OI
        pcr_volume: PE volume / CE volume
        call_premium_chg_pct: ATM call % change vs session open
        put_premium_chg_pct: ATM put % change vs session open
        ce_oi_chg_pct: CE OI % change (first vs last candle)
        pe_oi_chg_pct: PE OI % change (first vs last candle)
        avg_range_pct: 5-day avg range (optional, for volatility comparison)
    """
    reasons: List[str] = []
    bull_score = 0.0
    bear_score = 0.0

    # PCR OI: < 0.7 bullish, > 1.3 bearish
    if pcr_oi > 0:
        if pcr_oi < 0.6:
            bull_score += 0.25
            reasons.append(f"PCR(OI) {pcr_oi:.2f} very low (call heavy)")
        elif pcr_oi < 0.7:
            bull_score += 0.15
            reasons.append(f"PCR(OI) {pcr_oi:.2f} bullish")
        elif pcr_oi > 1.5:
            bear_score += 0.25
            reasons.append(f"PCR(OI) {pcr_oi:.2f} very high (put heavy)")
        elif pcr_oi > 1.3:
            bear_score += 0.15
            reasons.append(f"PCR(OI) {pcr_oi:.2f} bearish")

    # PCR Volume: faster signal
    if pcr_volume > 0:
        if pcr_volume < 0.5:
            bull_score += 0.2
            reasons.append(f"PCR(Vol) {pcr_volume:.2f} call buying")
        elif pcr_volume > 2.0:
            bear_score += 0.2
            reasons.append(f"PCR(Vol) {pcr_volume:.2f} put buying")

    # Premium expansion
    if call_premium_chg_pct > 15:
        bull_score += 0.15
        reasons.append(f"Call prem +{call_premium_chg_pct:.1f}%")
    elif call_premium_chg_pct < -15:
        bear_score += 0.15
        reasons.append(f"Call prem {call_premium_chg_pct:.1f}%")
    if put_premium_chg_pct > 15:
        bear_score += 0.15
        reasons.append(f"Put prem +{put_premium_chg_pct:.1f}%")
    elif put_premium_chg_pct < -15:
        bull_score += 0.15
        reasons.append(f"Put prem {put_premium_chg_pct:.1f}%")

    # OI build-up
    if ce_oi_chg_pct > 5 and pe_oi_chg_pct < -5:
        bull_score += 0.15
        reasons.append(f"CE OI +{ce_oi_chg_pct:.1f}%, PE OI {pe_oi_chg_pct:.1f}%")
    elif pe_oi_chg_pct > 5 and ce_oi_chg_pct < -5:
        bear_score += 0.15
        reasons.append(f"PE OI +{pe_oi_chg_pct:.1f}%, CE OI {ce_oi_chg_pct:.1f}%")

    # Range / volatility
    vol_spike = range_pct > 1.5 or (avg_range_pct and range_pct > avg_range_pct * 1.5)
    if vol_spike:
        reasons.append(f"Range {range_pct:.2f}% (vol spike)")
    if volume_ratio > 2.0:
        reasons.append(f"Volume {volume_ratio:.1f}x avg")

    # Classify regime
    if vol_spike and volume_ratio > 1.5:
        regime = "VOLATILITY_SPIKE"
        confidence = min(0.9, 0.5 + (range_pct / 3) * 0.2)
    elif bull_score - bear_score > 0.3:
        regime = "PRESSURE_BULLISH"
        confidence = min(0.9, 0.5 + (bull_score - bear_score) * 0.5)
    elif bear_score - bull_score > 0.3:
        regime = "PRESSURE_BEARISH"
        confidence = min(0.9, 0.5 + (bear_score - bull_score) * 0.5)
    else:
        regime = "NORMAL"
        confidence = 0.6 if range_pct < 0.5 else 0.5
        if not reasons:
            reasons.append("Balanced / choppy conditions")

    return MarketCondition(
        regime=regime,
        confidence=confidence,
        range_pct=range_pct,
        volume_ratio=volume_ratio,
        pcr_oi=pcr_oi,
        pcr_volume=pcr_volume,
        call_premium_chg_pct=call_premium_chg_pct,
        put_premium_chg_pct=put_premium_chg_pct,
        ce_oi_chg_pct=ce_oi_chg_pct,
        pe_oi_chg_pct=pe_oi_chg_pct,
        reasons=reasons,
    )
