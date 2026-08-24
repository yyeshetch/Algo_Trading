"""
Sector rotation — rank NSE live sector indices and surface money flow shifts.

Uses https://www.nseindia.com/market-data/live-market-indices (allIndices API)
for sector index % change vs NIFTY 50. Intraday open→mid→now RRG trails are
built from snapshots taken on each dashboard refresh during the session.
"""

from __future__ import annotations

import json
import logging
from datetime import date, datetime
from pathlib import Path
from typing import Any

from intraday_engine.core.config import Settings
from intraday_engine.storage.layout import sector_rotation_path

logger = logging.getLogger(__name__)

MIN_RANK_MOVE = 2


def _rotation_events(
    checkpoints: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if len(checkpoints) < 2:
        return []

    first = {r["sector"]: r["rank"] for r in checkpoints[0]["sectors"]}
    last = {r["sector"]: r["rank"] for r in checkpoints[-1]["sectors"]}
    first_ex = {r["sector"]: r["avg_excess_pct"] for r in checkpoints[0]["sectors"]}
    last_ex = {r["sector"]: r["avg_excess_pct"] for r in checkpoints[-1]["sectors"]}

    events: list[dict[str, Any]] = []
    all_sectors = set(first) | set(last)
    for sec in all_sectors:
        r0 = first.get(sec)
        r1 = last.get(sec)
        if r0 is None or r1 is None:
            continue
        delta = r0 - r1
        if abs(delta) < MIN_RANK_MOVE:
            continue
        ex_delta = round(last_ex.get(sec, 0) - first_ex.get(sec, 0), 3)
        if delta > 0:
            events.append({
                "sector": sec,
                "direction": "into",
                "rank_from": r0,
                "rank_to": r1,
                "rank_delta": delta,
                "excess_delta_pct": ex_delta,
                "text": f"Money rotating INTO {sec} (#{r0} → #{r1}, excess {ex_delta:+.2f}%)",
            })
        else:
            events.append({
                "sector": sec,
                "direction": "out_of",
                "rank_from": r0,
                "rank_to": r1,
                "rank_delta": delta,
                "excess_delta_pct": ex_delta,
                "text": f"Money rotating OUT OF {sec} (#{r0} → #{r1}, excess {ex_delta:+.2f}%)",
            })

    events.sort(key=lambda e: abs(e["rank_delta"]), reverse=True)
    return events


def _rrg_quadrant(rs_ratio: float, rs_momentum: float, *, pivot: float = 100.0) -> str:
    if rs_ratio >= pivot and rs_momentum >= 0:
        return "leading"
    if rs_ratio >= pivot and rs_momentum < 0:
        return "weakening"
    if rs_ratio < pivot and rs_momentum < 0:
        return "lagging"
    return "improving"


def _build_rrg_points(checkpoints: list[dict[str, Any]], *, pivot: float = 100.0) -> list[dict[str, Any]]:
    if not checkpoints:
        return []

    by_label: dict[str, dict[str, float]] = {}
    for cp in checkpoints:
        label = cp.get("label", "")
        for row in cp.get("sectors") or []:
            sec = row.get("sector")
            if not sec:
                continue
            by_label.setdefault(sec, {})[label] = float(row.get("avg_excess_pct") or 0)

    points: list[dict[str, Any]] = []
    for sec, slices in by_label.items():
        open_ex = slices.get("open", slices.get("mid", 0))
        mid_ex = slices.get("mid", open_ex)
        now_ex = slices.get("now", mid_ex)
        rs_ratio = round(pivot + now_ex, 3)
        rs_momentum = round(now_ex - open_ex, 3)
        trail = []
        for label, ex in (("open", open_ex), ("mid", mid_ex), ("now", now_ex)):
            trail.append({
                "label": label,
                "rs_ratio": round(pivot + ex, 3),
                "rs_momentum": round(ex - open_ex, 3),
            })
        points.append({
            "sector": sec,
            "rs_ratio": rs_ratio,
            "rs_momentum": rs_momentum,
            "quadrant": _rrg_quadrant(rs_ratio, rs_momentum, pivot=pivot),
            "excess_now_pct": round(now_ex, 3),
            "excess_delta_pct": rs_momentum,
            "trail": trail,
        })

    points.sort(key=lambda p: abs(p["rs_momentum"]), reverse=True)
    return points


def run_sector_rotation_scan(
    *,
    settings: Settings | None = None,
    trade_date: date | None = None,
    min_bars: int = 4,
) -> dict[str, Any]:
    _ = min_bars  # kept for API compatibility with sector-rs refresh bundle
    settings = settings or Settings.from_env(underlying="NIFTY")
    requested = trade_date or date.today()
    from intraday_engine.fetch.nse_market_indices import live_sector_rotation_payload

    payload = live_sector_rotation_payload(
        settings.data_dir,
        requested,
        force_refresh=True,
    )
    _persist(payload, settings.data_dir, requested)
    return payload


def _persist(payload: dict[str, Any], data_dir: Path, trade_date: date) -> None:
    out_path = sector_rotation_path(data_dir, trade_date)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_stored_sector_rotation(data_dir: Path, trade_date: date) -> dict[str, Any] | None:
    p = sector_rotation_path(data_dir, trade_date)
    data: dict[str, Any] | None = None
    if p.exists():
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            pass
    if data is None:
        folder = p.parent
        if folder.exists():
            files = sorted(folder.glob("sector_rotation_*.json"))
            if files:
                try:
                    data = json.loads(files[-1].read_text(encoding="utf-8"))
                except Exception:
                    return None
    if not data:
        return None
    if "rrg" not in data and data.get("checkpoints"):
        data["rrg"] = {
            "pivot": 100.0,
            "x_label": "RS-Ratio (100 = NIFTY parity)",
            "y_label": "RS-Momentum (open → now excess Δ)",
            "points": _build_rrg_points(data["checkpoints"]),
        }
    return data
