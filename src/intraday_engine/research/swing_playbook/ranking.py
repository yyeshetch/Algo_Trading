"""100-point composite ranking model from swing playbook doc."""

from __future__ import annotations

from typing import Any

from intraday_engine.research.swing_playbook.common import StockMetrics
from intraday_engine.research.swing_playbook.scanners import ScanHit


def _clamp(v: float, lo: float = 0.0, hi: float = 100.0) -> float:
    return max(lo, min(hi, v))


def rank_stock(
    m: StockMetrics,
    *,
    market_regime: dict[str, Any],
    fundamentals: dict[str, Any] | None = None,
    scanner_hits: list[str] | None = None,
) -> dict[str, Any]:
    """Score 0-100 across Market, Sector, Trend, Fundamentals, Setup, Risk."""
    f = fundamentals or {}
    hits = set(scanner_hits or [])

    # Market (15)
    regime = market_regime.get("regime", "neutral")
    market_pts = {"bull": 15, "neutral": 8, "bear": 3}.get(regime, 5)

    # Sector (15) — proxy via excess RS vs Nifty
    sector_pts = _clamp(m.excess_63d * 1.5, 0, 15)

    # Stock trend (20)
    trend_pts = 0.0
    if m.above_ma200:
        trend_pts += 6
    if m.ma50_above_ma200:
        trend_pts += 5
    if m.ma20_above_ma50:
        trend_pts += 4
    if m.rs_slope_20d > 0:
        trend_pts += min(5, m.rs_slope_20d * 0.3)

    # Fundamentals (20)
    fund_pts = 0.0
    pg = f.get("profit_growth_3y")
    sg = f.get("sales_growth_3y")
    roe = f.get("roe")
    if pg is not None and pg > 15:
        fund_pts += min(8, pg * 0.3)
    if sg is not None and sg > 10:
        fund_pts += min(6, sg * 0.25)
    if roe is not None and roe > 15:
        fund_pts += min(6, roe * 0.25)
    fund_pts = _clamp(fund_pts, 0, 20)

    # Setup quality (20)
    setup_pts = 0.0
    setup_map = {
        "momentum_leaders": 8,
        "breakout": 10,
        "vcp_contraction": 12,
        "pullback": 9,
        "earnings_acceleration": 11,
        "can_slim": 10,
        "turtle_breakout": 8,
        "post_earnings_momentum": 9,
    }
    for s in hits:
        setup_pts += setup_map.get(s, 4)
    setup_pts = _clamp(setup_pts, 0, 20)

    # Risk (10) — lower vol extension = better
    risk_pts = 10.0
    if m.pct_from_52w_high < -15:
        risk_pts -= 3
    if m.rsi14 > 75:
        risk_pts -= 4
    if m.atr_ratio > 1.2:
        risk_pts -= 2
    if not m.liquidity_ok:
        risk_pts -= 5
    risk_pts = _clamp(risk_pts, 0, 10)

    total = market_pts + sector_pts + trend_pts + fund_pts + setup_pts + risk_pts
    return {
        "stock": m.stock,
        "total_score": round(_clamp(total, 0, 100), 1),
        "components": {
            "market": round(market_pts, 1),
            "sector_rs": round(sector_pts, 1),
            "trend": round(trend_pts, 1),
            "fundamentals": round(fund_pts, 1),
            "setup": round(setup_pts, 1),
            "risk": round(risk_pts, 1),
        },
        "scanner_hits": sorted(hits),
        "close": m.close,
        "excess_63d": m.excess_63d,
    }


def build_rankings(
    metrics: list[StockMetrics],
    all_scans: dict[str, list[ScanHit]],
    *,
    market_regime: dict[str, Any],
    fundamentals: dict[str, dict[str, Any]],
    top_n: int = 50,
) -> list[dict[str, Any]]:
    hit_map: dict[str, list[str]] = {}
    for scanner, hits in all_scans.items():
        for h in hits:
            hit_map.setdefault(h.stock, []).append(scanner)

    ranked: list[dict[str, Any]] = []
    for m in metrics:
        if not m.liquidity_ok:
            continue
        ranked.append(
            rank_stock(
                m,
                market_regime=market_regime,
                fundamentals=fundamentals.get(m.stock.upper()),
                scanner_hits=hit_map.get(m.stock, []),
            )
        )
    ranked.sort(key=lambda r: r["total_score"], reverse=True)
    return ranked[:top_n]
