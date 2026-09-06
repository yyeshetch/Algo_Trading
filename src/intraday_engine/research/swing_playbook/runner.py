"""Orchestrate swing playbook scans and persist JSON output."""

from __future__ import annotations

import json
import logging
from datetime import date, datetime
from pathlib import Path
from typing import Any

from intraday_engine.core.config import Settings
from intraday_engine.research.fundamentals_screener import load_latest_fundamentals_csv
from intraday_engine.research.swing_playbook.common import build_universe_metrics
from intraday_engine.research.swing_playbook.market_regime import classify_market_regime
from intraday_engine.research.swing_playbook.ranking import build_rankings
from intraday_engine.research.swing_playbook.scanners import (
    hit_to_dict,
    scan_breakout,
    scan_can_slim,
    scan_dual_momentum,
    scan_earnings_acceleration,
    scan_mean_reversion,
    scan_momentum_leaders,
    scan_post_earnings_momentum,
    scan_pullback,
    scan_stage_analysis,
    scan_turtle_breakout,
    scan_vcp_contraction,
)
from intraday_engine.storage.layout import swing_playbook_path

logger = logging.getLogger(__name__)

SCANNER_KEYS = (
    "momentum_leaders",
    "breakout",
    "vcp_contraction",
    "pullback",
    "earnings_acceleration",
    "stage_2",
    "dual_momentum",
    "mean_reversion",
    "turtle_breakout",
    "post_earnings_momentum",
    "can_slim",
)


def _load_fundamentals_map(data_dir: Path) -> dict[str, dict[str, Any]]:
    df = load_latest_fundamentals_csv(data_dir)
    if df.empty:
        return {}
    out: dict[str, dict[str, Any]] = {}
    for _, row in df.iterrows():
        sym = str(row.get("stock", "")).strip().upper()
        if sym:
            out[sym] = row.to_dict()
    return out


def run_swing_playbook_scan(
    *,
    settings: Settings | None = None,
    trade_date: date | None = None,
    top_n: int = 30,
    ranking_top_n: int = 50,
    symbols_file: Path | None = None,
) -> dict[str, Any]:
    settings = settings or Settings.from_env(underlying="NIFTY")
    td = trade_date or date.today()

    metrics, meta = build_universe_metrics(settings, symbols_file=symbols_file)
    fundamentals = _load_fundamentals_map(settings.data_dir)
    regime = classify_market_regime(settings, metrics)

    scans: dict[str, list[dict[str, Any]]] = {
        "momentum_leaders": [hit_to_dict(h) for h in scan_momentum_leaders(metrics)[:top_n]],
        "breakout": [hit_to_dict(h) for h in scan_breakout(metrics)[:top_n]],
        "vcp_contraction": [hit_to_dict(h) for h in scan_vcp_contraction(metrics)[:top_n]],
        "pullback": [hit_to_dict(h) for h in scan_pullback(metrics)[:top_n]],
        "earnings_acceleration": [
            hit_to_dict(h) for h in scan_earnings_acceleration(metrics, fundamentals)[:top_n]
        ],
        "stage_2": [hit_to_dict(h) for h in scan_stage_analysis(metrics, stage=2)[:top_n]],
        "dual_momentum": [hit_to_dict(h) for h in scan_dual_momentum(metrics)[:top_n]],
        "mean_reversion": [hit_to_dict(h) for h in scan_mean_reversion(metrics)[:top_n]],
        "turtle_breakout": [hit_to_dict(h) for h in scan_turtle_breakout(metrics)[:top_n]],
        "post_earnings_momentum": [
            hit_to_dict(h) for h in scan_post_earnings_momentum(metrics)[:top_n]
        ],
        "can_slim": [hit_to_dict(h) for h in scan_can_slim(metrics, fundamentals)[:top_n]],
    }

    raw_hits = {
        "momentum_leaders": scan_momentum_leaders(metrics),
        "breakout": scan_breakout(metrics),
        "vcp_contraction": scan_vcp_contraction(metrics),
        "pullback": scan_pullback(metrics),
        "earnings_acceleration": scan_earnings_acceleration(metrics, fundamentals),
        "stage_2": scan_stage_analysis(metrics, stage=2),
        "dual_momentum": scan_dual_momentum(metrics),
        "mean_reversion": scan_mean_reversion(metrics),
        "turtle_breakout": scan_turtle_breakout(metrics),
        "post_earnings_momentum": scan_post_earnings_momentum(metrics),
        "can_slim": scan_can_slim(metrics, fundamentals),
    }
    rankings = build_rankings(
        metrics,
        raw_hits,
        market_regime=regime,
        fundamentals=fundamentals,
        top_n=ranking_top_n,
    )

    counts = {k: len(v) for k, v in scans.items()}
    payload: dict[str, Any] = {
        "trade_date": td.isoformat(),
        "generated_at": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
        "universe_meta": meta,
        "market_regime": regime,
        "fundamentals_available": len(fundamentals),
        "scanner_counts": counts,
        "scanners": scans,
        "rankings": rankings,
        "strategies": {
            "momentum_rs": "Scanner A — stacked MAs + RS vs Nifty",
            "breakout": "Scanner B — near highs + volume expansion",
            "vcp_contraction": "Scanner C — SEPA/VCP style contraction",
            "pullback": "Scanner D — trend pullback to 20/50 DMA",
            "earnings_acceleration": "Scanner E — EPS/sales growth + RS",
            "can_slim": "CAN SLIM growth + RS leadership",
            "stage_2": "Weinstein Stage 2 advancing",
            "dual_momentum": "Absolute + relative 6M momentum",
            "mean_reversion": "Uptrend RSI washout at 50 DMA",
            "turtle_breakout": "Donchian 20/55-day breakout",
            "post_earnings_momentum": "Gap-and-hold volume proxy",
        },
    }

    out_path = swing_playbook_path(settings.data_dir, td)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    logger.info(
        "Swing playbook: %d symbols, regime=%s, saved %s",
        meta.get("computed", 0),
        regime.get("regime"),
        out_path,
    )
    return payload


def load_stored_swing_playbook(data_dir: Path, trade_date: date) -> dict[str, Any] | None:
    p = swing_playbook_path(data_dir, trade_date)
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            return None
    folder = p.parent
    if not folder.exists():
        return None
    files = sorted(folder.glob("swing_playbook_*.json"))
    if not files:
        return None
    try:
        return json.loads(files[-1].read_text(encoding="utf-8"))
    except Exception:
        return None
