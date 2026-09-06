"""Market regime classification for swing playbook."""

from __future__ import annotations

from typing import Any

import pandas as pd

from intraday_engine.core.config import Settings
from intraday_engine.research.relative_strength_scanner import _load_nifty_daily
from intraday_engine.research.swing_playbook.common import StockMetrics


def classify_market_regime(
    settings: Settings,
    metrics: list[StockMetrics],
) -> dict[str, Any]:
    nifty = _load_nifty_daily(settings, lookback_days=280)
    if nifty.empty:
        return {"regime": "unknown", "score": 0, "reasons": ["NIFTY data unavailable"]}

    close = nifty["close"].astype(float)
    ma50 = close.rolling(50).mean()
    ma200 = close.rolling(200).mean()
    last = float(close.iloc[-1])
    m50 = float(ma50.iloc[-1])
    m200 = float(ma200.iloc[-1])
    ret_63 = (last / float(close.iloc[-64]) - 1.0) * 100.0 if len(close) > 64 else 0.0

    above_200 = last > m200
    ma50_above_200 = m50 > m200
    m200_rising = m200 > float(ma200.iloc[-21]) * 1.002 if len(ma200) > 21 else False

    breadth = 0.0
    if metrics:
        above = sum(1 for m in metrics if m.above_ma200)
        breadth = above / len(metrics) * 100.0

    score = 0
    reasons: list[str] = []
    if above_200:
        score += 25
        reasons.append("Nifty above 200 DMA")
    if ma50_above_200:
        score += 25
        reasons.append("50 > 200 DMA")
    if m200_rising:
        score += 20
        reasons.append("200 DMA rising")
    if ret_63 > 0:
        score += 15
        reasons.append(f"3M return +{ret_63:.1f}%")
    if breadth >= 55:
        score += 15
        reasons.append(f"Breadth {breadth:.0f}% above 200 DMA")
    elif breadth < 40:
        score -= 10
        reasons.append(f"Weak breadth {breadth:.0f}%")

    if score >= 70:
        regime = "bull"
    elif score >= 40:
        regime = "neutral"
    else:
        regime = "bear"

    return {
        "regime": regime,
        "score": score,
        "nifty_close": round(last, 2),
        "nifty_ma50": round(m50, 2),
        "nifty_ma200": round(m200, 2),
        "nifty_return_63d": round(ret_63, 2),
        "breadth_pct_above_200": round(breadth, 1),
        "reasons": reasons,
    }
