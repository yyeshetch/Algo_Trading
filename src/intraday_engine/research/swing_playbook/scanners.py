"""Swing playbook scanners A–E and additional strategy filters."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from intraday_engine.research.swing_playbook.common import StockMetrics


@dataclass
class ScanHit:
    stock: str
    score: float
    scanner: str
    reasons: list[str]
    metrics: dict[str, Any]


def _hit(m: StockMetrics, scanner: str, score: float, reasons: list[str]) -> ScanHit:
    from intraday_engine.research.swing_playbook.common import metrics_to_dict

    return ScanHit(
        stock=m.stock,
        score=round(score, 1),
        scanner=scanner,
        reasons=reasons,
        metrics=metrics_to_dict(m),
    )


# --- Scanner A: Momentum Leaders ---
def scan_momentum_leaders(metrics: list[StockMetrics]) -> list[ScanHit]:
    hits: list[ScanHit] = []
    for m in metrics:
        if not m.liquidity_ok:
            continue
        reasons: list[str] = []
        score = 0.0
        if not m.above_ma200:
            continue
        reasons.append("Price > 200 DMA")
        score += 15
        if not m.ma50_above_ma200:
            continue
        reasons.append("50 > 200 DMA")
        score += 15
        if not m.ma20_above_ma50:
            continue
        reasons.append("20 > 50 DMA")
        score += 10
        if m.excess_63d <= 0:
            continue
        reasons.append(f"3M excess vs Nifty +{m.excess_63d:.1f}%")
        score += min(20.0, m.excess_63d)
        if m.excess_126d <= 0:
            continue
        reasons.append(f"6M excess vs Nifty +{m.excess_126d:.1f}%")
        score += min(15.0, m.excess_126d * 0.5)
        if m.rs_slope_20d <= 0:
            continue
        reasons.append(f"RS line improving +{m.rs_slope_20d:.1f}%")
        score += min(15.0, m.rs_slope_20d)
        hits.append(_hit(m, "momentum_leaders", score, reasons))
    hits.sort(key=lambda h: h.score, reverse=True)
    return hits


# --- Scanner B: Breakout ---
def scan_breakout(metrics: list[StockMetrics]) -> list[ScanHit]:
    hits: list[ScanHit] = []
    for m in metrics:
        if not m.liquidity_ok or not m.above_ma50:
            continue
        near_20 = m.close >= m.high_20d * 0.97
        near_50 = m.close >= m.high_50d * 0.95
        near_52w = m.pct_from_52w_high >= -3.0
        if not (near_20 or near_50) or not near_52w:
            continue
        if m.volume_ratio < 1.5:
            continue
        score = 40.0
        reasons = ["Near 20/50-day or 52W high"]
        if m.volume_ratio >= 2.0:
            score += 20
            reasons.append(f"Volume {m.volume_ratio:.1f}x avg")
        else:
            score += 10
            reasons.append(f"Volume {m.volume_ratio:.1f}x avg")
        if m.close >= m.high_20d * 0.995:
            score += 15
            reasons.append("Close near HOD zone")
        if m.excess_63d > 0:
            score += min(15, m.excess_63d)
            reasons.append("Sector/index RS+")
        if m.ma20_above_ma50 and m.above_ma200:
            score += 10
            reasons.append("Above stacked MAs")
        hits.append(_hit(m, "breakout", score, reasons))
    hits.sort(key=lambda h: h.score, reverse=True)
    return hits


# --- Scanner C: VCP / Contraction ---
def scan_vcp_contraction(metrics: list[StockMetrics]) -> list[ScanHit]:
    hits: list[ScanHit] = []
    for m in metrics:
        if not m.liquidity_ok:
            continue
        if not (m.above_ma200 and m.ma50_above_ma200 and m.return_63d > 10):
            continue
        if m.atr_ratio > 0.85 or m.vol_trend_ratio > 0.95:
            continue
        if m.pct_from_52w_high < -8.0:
            continue
        if m.excess_63d < 5:
            continue
        score = 50.0
        reasons = ["Prior uptrend", "ATR/volume contracting"]
        if m.atr_ratio < 0.75:
            score += 15
            reasons.append(f"Tight ATR ratio {m.atr_ratio:.2f}")
        if m.pct_from_52w_high >= -3:
            score += 15
            reasons.append("Near resistance / 52W high")
        score += min(20, m.excess_63d)
        reasons.append(f"High RS +{m.excess_63d:.1f}%")
        hits.append(_hit(m, "vcp_contraction", score, reasons))
    hits.sort(key=lambda h: h.score, reverse=True)
    return hits


# --- Scanner D: Pullback ---
def scan_pullback(metrics: list[StockMetrics]) -> list[ScanHit]:
    hits: list[ScanHit] = []
    for m in metrics:
        if not m.liquidity_ok:
            continue
        if not (m.above_ma200 and m.ma50_above_ma200 and m.return_126d > 15):
            continue
        at_support = m.pullback_to_ma20 or m.pullback_to_ma50
        if not at_support:
            continue
        if m.vol_trend_ratio > 0.85:
            continue
        score = 45.0
        reasons = ["Bullish long-term trend"]
        if m.pullback_to_ma20:
            score += 15
            reasons.append("Retrace to 20 DMA")
        if m.pullback_to_ma50:
            score += 10
            reasons.append("Retrace to 50 DMA")
        if m.vol_trend_ratio < 0.7:
            score += 15
            reasons.append("Volume contracted")
        if m.rsi14 < 45:
            score += 10
            reasons.append(f"RSI reversal zone {m.rsi14:.0f}")
        if m.return_20d < 0:
            score += 5
            reasons.append("Short-term pullback")
        hits.append(_hit(m, "pullback", score, reasons))
    hits.sort(key=lambda h: h.score, reverse=True)
    return hits


# --- Scanner E: Earnings Acceleration (fundamentals overlay) ---
def scan_earnings_acceleration(
    metrics: list[StockMetrics],
    fundamentals: dict[str, dict[str, Any]],
) -> list[ScanHit]:
    hits: list[ScanHit] = []
    for m in metrics:
        if not m.liquidity_ok or not m.above_ma50:
            continue
        f = fundamentals.get(m.stock.upper(), {})
        pg3 = f.get("profit_growth_3y")
        sg3 = f.get("sales_growth_3y")
        pg5 = f.get("profit_growth_5y")
        if pg3 is None or sg3 is None:
            continue
        if pg3 < 15 or sg3 < 10:
            continue
        accelerating = pg5 is not None and pg3 > pg5
        score = 40.0
        reasons = [f"EPS growth 3Y {pg3:.0f}%", f"Sales growth 3Y {sg3:.0f}%"]
        if accelerating:
            score += 20
            reasons.append("Profit growth accelerating vs 5Y")
        if m.excess_63d > 0:
            score += min(20, m.excess_63d)
            reasons.append("High RS vs Nifty")
        if m.volume_ratio >= 1.2:
            score += 10
            reasons.append("Volume expansion")
        if m.above_ma200:
            score += 10
            reasons.append("Above 200 DMA")
        hits.append(_hit(m, "earnings_acceleration", score, reasons))
    hits.sort(key=lambda h: h.score, reverse=True)
    return hits


# --- Stage Analysis (Weinstein) ---
def scan_stage_analysis(metrics: list[StockMetrics], *, stage: int = 2) -> list[ScanHit]:
    hits: list[ScanHit] = []
    for m in metrics:
        if not m.liquidity_ok or m.stage != stage:
            continue
        score = 50.0
        reasons = [f"Weinstein Stage {stage}"]
        if stage == 2:
            if m.ma200_rising:
                score += 15
                reasons.append("200 DMA rising")
            if m.excess_63d > 0:
                score += min(20, m.excess_63d)
                reasons.append("RS vs index positive")
            if m.atr_ratio < 0.9:
                score += 10
                reasons.append("Volatility contracting before advance")
        hits.append(_hit(m, f"stage_{stage}", score, reasons))
    hits.sort(key=lambda h: h.score, reverse=True)
    return hits


# --- Dual Momentum ---
def scan_dual_momentum(metrics: list[StockMetrics]) -> list[ScanHit]:
    hits: list[ScanHit] = []
    for m in metrics:
        if not m.liquidity_ok:
            continue
        if m.return_126d <= 0 or m.excess_126d <= 0:
            continue
        if not m.above_ma200:
            continue
        score = 50.0 + min(25, m.return_126d * 0.3) + min(25, m.excess_126d * 0.5)
        reasons = [
            f"Absolute momentum +{m.return_126d:.1f}% (6M)",
            f"Relative momentum +{m.excess_126d:.1f}% vs Nifty",
        ]
        hits.append(_hit(m, "dual_momentum", score, reasons))
    hits.sort(key=lambda h: h.score, reverse=True)
    return hits


# --- Mean Reversion (within uptrend) ---
def scan_mean_reversion(metrics: list[StockMetrics]) -> list[ScanHit]:
    hits: list[ScanHit] = []
    for m in metrics:
        if not m.liquidity_ok:
            continue
        if not (m.above_ma200 and m.return_126d > 10):
            continue
        if m.rsi14 > 38 or m.return_20d > -3:
            continue
        at_support = m.pullback_to_ma50 or m.close <= m.ma50 * 1.02
        if not at_support:
            continue
        score = 45.0
        reasons = ["Uptrend intact", f"RSI oversold {m.rsi14:.0f}"]
        if m.pullback_to_ma50:
            score += 20
            reasons.append("Support at 50 DMA")
        if m.vol_trend_ratio < 0.75:
            score += 15
            reasons.append("Selling volume dried up")
        hits.append(_hit(m, "mean_reversion", score, reasons))
    hits.sort(key=lambda h: h.score, reverse=True)
    return hits


# --- Turtle / Volatility Breakout ---
def scan_turtle_breakout(metrics: list[StockMetrics]) -> list[ScanHit]:
    hits: list[ScanHit] = []
    for m in metrics:
        if not m.liquidity_ok:
            continue
        if m.turtle_break_55:
            score = 70.0
            reasons = ["55-day Donchian breakout"]
        elif m.turtle_break_20:
            score = 55.0
            reasons = ["20-day Donchian breakout"]
        else:
            continue
        if m.volume_ratio >= 1.3:
            score += 15
            reasons.append(f"Volume confirm {m.volume_ratio:.1f}x")
        if m.above_ma200:
            score += 10
            reasons.append("Above 200 DMA")
        hits.append(_hit(m, "turtle_breakout", score, reasons))
    hits.sort(key=lambda h: h.score, reverse=True)
    return hits


# --- Post-Earnings Momentum (price/volume proxy) ---
def scan_post_earnings_momentum(metrics: list[StockMetrics]) -> list[ScanHit]:
    hits: list[ScanHit] = []
    for m in metrics:
        if not m.liquidity_ok or not m.above_ma50:
            continue
        if m.gap_up_5d_pct < 5.0:
            continue
        if m.return_20d < 3.0:
            continue
        if m.excess_63d < 0:
            continue
        score = 40.0 + min(30, m.gap_up_5d_pct)
        reasons = [f"Gap-and-hold +{m.gap_up_5d_pct:.1f}% over 5 sessions"]
        if m.volume_ratio >= 1.5:
            score += 15
            reasons.append("Initial volume expansion")
        if m.vol_trend_ratio < 0.8:
            score += 10
            reasons.append("Volume now contracting (base forming)")
        if m.above_ma200:
            score += 5
            reasons.append("Above 200 DMA")
        hits.append(_hit(m, "post_earnings_momentum", score, reasons))
    hits.sort(key=lambda h: h.score, reverse=True)
    return hits


# --- CAN SLIM composite (doc strategy #3) ---
def scan_can_slim(
    metrics: list[StockMetrics],
    fundamentals: dict[str, dict[str, Any]],
) -> list[ScanHit]:
    hits: list[ScanHit] = []
    for m in metrics:
        if not m.liquidity_ok or not m.above_ma200:
            continue
        f = fundamentals.get(m.stock.upper(), {})
        pg = f.get("profit_growth_3y") or 0
        sg = f.get("sales_growth_3y") or 0
        if pg < 20 or sg < 15:
            continue
        if m.excess_63d < 5 or not m.ma20_above_ma50:
            continue
        score = 40.0
        reasons = ["CAN SLIM trend + growth"]
        score += min(20, pg * 0.5)
        reasons.append(f"Earnings growth {pg:.0f}%")
        score += min(20, m.excess_63d)
        reasons.append(f"RS leadership +{m.excess_63d:.1f}%")
        if m.pct_from_52w_high >= -5:
            score += 10
            reasons.append("Near new highs (N)")
        if m.volume_ratio >= 1.2:
            score += 10
            reasons.append("Volume confirmation")
        hits.append(_hit(m, "can_slim", score, reasons))
    hits.sort(key=lambda h: h.score, reverse=True)
    return hits


def hit_to_dict(h: ScanHit) -> dict[str, Any]:
    return {
        "stock": h.stock,
        "score": h.score,
        "scanner": h.scanner,
        "reasons": h.reasons,
        "metrics": h.metrics,
    }
