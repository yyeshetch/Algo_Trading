"""Score and rank F&O stocks by directional probability."""

from __future__ import annotations

from typing import Any


def score_stock(stock: str, metrics: dict[str, Any], all_metrics: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Score a stock based on bullish/bearish signals.
    Bullish: spot up, call OI up, put OI down, call premium up, put premium down
    Bearish: opposite
    """
    spot_chg = metrics.get("spot_change_pct", 0) or 0
    call_oi_chg = metrics.get("call_oi_change_pct", 0) or 0
    put_oi_chg = metrics.get("put_oi_change_pct", 0) or 0
    call_prem_chg = metrics.get("call_premium_change_pct", 0) or 0
    put_prem_chg = metrics.get("put_premium_change_pct", 0) or 0
    spot_vol = metrics.get("spot_volume", 0) or 0
    fut_vol = metrics.get("futures_volume", 0) or 0

    max_spot_vol = max((m.get("spot_volume", 0) or 0 for m in all_metrics), default=1)
    max_fut_vol = max((m.get("futures_volume", 0) or 0 for m in all_metrics), default=1)
    vol_score = (spot_vol / max_spot_vol * 0.5 + fut_vol / max_fut_vol * 0.5) if (max_spot_vol and max_fut_vol) else 0

    bullish = 0.0
    bearish = 0.0
    reasons: list[str] = []

    if spot_chg > 0.3:
        bullish += 0.25
        reasons.append(f"Spot +{spot_chg:.1f}%")
    elif spot_chg < -0.3:
        bearish += 0.25
        reasons.append(f"Spot {spot_chg:.1f}%")

    if call_oi_chg > 2:
        bullish += 0.15
        reasons.append(f"CE OI +{call_oi_chg:.1f}%")
    elif call_oi_chg < -2:
        bearish += 0.15
        reasons.append(f"CE OI {call_oi_chg:.1f}%")

    if put_oi_chg > 2:
        bearish += 0.15
        reasons.append(f"PE OI +{put_oi_chg:.1f}%")
    elif put_oi_chg < -2:
        bullish += 0.15
        reasons.append(f"PE OI {put_oi_chg:.1f}%")

    if call_prem_chg > 2:
        bullish += 0.15
        reasons.append(f"CE prem +{call_prem_chg:.1f}%")
    elif call_prem_chg < -2:
        bearish += 0.15
        reasons.append(f"CE prem {call_prem_chg:.1f}%")

    if put_prem_chg > 2:
        bearish += 0.15
        reasons.append(f"PE prem +{put_prem_chg:.1f}%")
    elif put_prem_chg < -2:
        bullish += 0.15
        reasons.append(f"PE prem {put_prem_chg:.1f}%")

    bullish += vol_score * 0.15
    bearish += vol_score * 0.15

    final_score = bullish - bearish
    confidence = min(1.0, abs(final_score) + 0.3)
    direction = "BULLISH" if final_score > 0 else ("BEARISH" if final_score < 0 else "NEUTRAL")

    return {
        **metrics,
        "bullish_score": round(bullish, 3),
        "bearish_score": round(bearish, 3),
        "final_score": round(final_score, 3),
        "confidence": round(confidence, 3),
        "direction": direction,
        "reasons": reasons,
    }


def rank_stocks(scored: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Sort by |final_score| descending (highest conviction first)."""
    return sorted(scored, key=lambda x: abs(x.get("final_score", 0)), reverse=True)
