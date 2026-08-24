"""
Multi-strike OI change over time — from stored option-chain snapshots.

Tracks session + interval OI change per strike, flags walls and unusual spikes,
and returns spot timeline for dashboard overlay.
"""

from __future__ import annotations

import statistics
from dataclasses import asdict, dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

from intraday_engine.gamma.option_chain_fetcher import (
    fill_missing_session_bars,
    load_session_option_chain_snapshots,
    _oc_strike_step,
)

TOP_N_PER_SIDE = 4
MIN_BASELINE_OI = 50_000
MAX_MOVERS = 12
WALL_MIN_SESSION_PCT = 5.0
SPIKE_MEDIAN_MULT = 2.5
SPIKE_MEDIAN_FLOOR = 0.35
SPOT_INTERVAL_PCT_THRESH = 0.04
OI_INTERVAL_PCT_THRESH = 0.25
EARLY_SPIKE_SPOT_PCT = 0.04
LATE_SPIKE_SPOT_PCT = 0.08

BUILDUP_LABELS: dict[str, str] = {
    "LONG_BUILD": "Long buildup",
    "SHORT_BUILD": "Short buildup",
    "LONG_UNWIND": "Long unwind",
    "SHORT_COVER": "Short covering",
    "FLAT": "Flat",
}


@dataclass
class StrikeOiChangeResult:
    available: bool
    snapshots: int
    headline: str
    spot: float | None = None
    atm_strike: int | None = None
    direction: dict[str, Any] = field(default_factory=dict)
    timestamps: list[str] = field(default_factory=list)
    spot_series: list[float] = field(default_factory=list)
    interval_median_pct: float | None = None
    time_labels: list[str] = field(default_factory=list)
    bar_sources: list[str] = field(default_factory=list)
    heatmap: list[dict[str, Any]] = field(default_factory=list)
    heatmap_max_abs: float = 1.0
    heatmap_interval_max_abs: float = 1.0
    oi_series: list[dict[str, Any]] = field(default_factory=list)
    oi_level_max: float = 1.0
    ladder: list[dict[str, Any]] = field(default_factory=list)
    series: list[dict[str, Any]] = field(default_factory=list)
    movers: list[dict[str, Any]] = field(default_factory=list)
    walls: list[dict[str, Any]] = field(default_factory=list)
    spikes: list[dict[str, Any]] = field(default_factory=list)
    reasons: list[str] = field(default_factory=list)
    coverage: dict[str, Any] = field(default_factory=dict)
    spot_interval_pct: list[float | None] = field(default_factory=list)
    buildup_thresholds: dict[str, float] = field(default_factory=dict)


def _pct_change(current: float, baseline: float) -> float:
    if baseline <= 0:
        return 0.0
    return round((current - baseline) / baseline * 100, 2)


def _pct_from_prev(current: float, previous: float) -> float:
    if previous <= 0:
        return 0.0
    return round((current - previous) / previous * 100, 2)


def _lakhs(value: float) -> float:
    return round(value / 100_000, 2)


def _spot_interval_series(spot_series: list[float]) -> list[float | None]:
    out: list[float | None] = []
    prev: float | None = None
    for s in spot_series:
        if s <= 0:
            out.append(None)
            continue
        if prev is None or prev <= 0:
            out.append(0.0)
        else:
            out.append(_pct_from_prev(s, prev))
        prev = s
    return out


def _classify_buildup(
    spot_chg_pct: float | None,
    oi_chg_pct: float | None,
    option_type: str = "CE",
) -> str:
    """Classify a bar as one of the four directional price+OI combos, else FLAT.

    Uses **option-contract-centric** labels: the label describes the position
    being built/unwound in the option contract itself (long calls, short puts,
    etc.), not the implied position in the underlying.

    For CE (calls, long call = long underlying) the mapping is direct:
      spot ↑ + OI ↑ → LONG_BUILD  (call buying)   → bullish for spot
      spot ↓ + OI ↑ → SHORT_BUILD (call writing)  → bearish for spot
      spot ↓ + OI ↓ → LONG_UNWIND (call longs out)→ bearish for spot
      spot ↑ + OI ↓ → SHORT_COVER (call writers cover) → bullish for spot

    For PE (puts, long put = short underlying) the LONG↔SHORT axis inverts:
      spot ↓ + OI ↑ → LONG_BUILD  (put buying)    → bearish for spot
      spot ↑ + OI ↑ → SHORT_BUILD (put writing)   → bullish for spot
      spot ↑ + OI ↓ → LONG_UNWIND (put longs out) → bullish for spot
      spot ↓ + OI ↓ → SHORT_COVER (put writers cover) → bearish for spot

    Quiet-spot cases (OI moves while price is flat) collapse to FLAT — no
    directional confirmation, so we can't attribute the OI change to a side.
    """
    if oi_chg_pct is None and spot_chg_pct is None:
        return "FLAT"
    spot = spot_chg_pct if spot_chg_pct is not None else 0.0
    oi = oi_chg_pct if oi_chg_pct is not None else 0.0
    price_up = spot > SPOT_INTERVAL_PCT_THRESH
    price_down = spot < -SPOT_INTERVAL_PCT_THRESH
    oi_up = oi > OI_INTERVAL_PCT_THRESH
    oi_down = oi < -OI_INTERVAL_PCT_THRESH
    is_put = str(option_type).upper() == "PE"
    if price_up and oi_up:
        return "SHORT_BUILD" if is_put else "LONG_BUILD"
    if price_down and oi_up:
        return "LONG_BUILD" if is_put else "SHORT_BUILD"
    if price_down and oi_down:
        return "SHORT_COVER" if is_put else "LONG_UNWIND"
    if price_up and oi_down:
        return "LONG_UNWIND" if is_put else "SHORT_COVER"
    return "FLAT"


def _spike_timing(spot_chg_pct: float | None, is_spike_bar: bool) -> str | None:
    if not is_spike_bar:
        return None
    if spot_chg_pct is None:
        return "MID"
    abs_spot = abs(spot_chg_pct)
    if abs_spot <= EARLY_SPIKE_SPOT_PCT:
        return "EARLY"
    if abs_spot >= LATE_SPIKE_SPOT_PCT:
        return "LATE"
    return "MID"


def _position_vs_spot(strike: int, spot: float | None) -> tuple[str, str, float | None]:
    if spot is None or spot <= 0:
        return "unknown", "—", None
    diff = strike - spot
    if diff > 0:
        return "above", "↑ above", round(diff, 1)
    if diff < 0:
        return "below", "↓ below", round(abs(diff), 1)
    return "at", "= ATM", 0.0


def _is_wall(m: dict[str, Any], rank_build: int | None) -> bool:
    if m["session_change_pct"] <= 0:
        return False
    if m["session_change_pct"] >= WALL_MIN_SESSION_PCT:
        return True
    return rank_build is not None and rank_build <= 3 and m["session_change_pct"] >= 2.0


def _leg_summary(m: dict[str, Any] | None) -> dict[str, Any] | None:
    if not m:
        return None
    pct = m["session_change_pct"]
    if pct > 0.5:
        action = "BUILD"
        arrow = "▲"
    elif pct < -0.5:
        action = "UNWIND"
        arrow = "▼"
    else:
        action = "FLAT"
        arrow = "—"
    return {
        "session_change_pct": pct,
        "session_change_lakhs": m["session_change_lakhs"],
        "interval_change_pct": m["interval_change_pct"],
        "tags": m.get("tags") or [],
        "is_wall": m.get("is_wall", False),
        "is_spike": m.get("is_spike", False),
        "action": action,
        "arrow": arrow,
    }


def _compute_direction(movers_all: list[dict[str, Any]], spot: float | None) -> dict[str, Any]:
    ce_legs = [m for m in movers_all if m["option_type"] == "CE"]
    pe_legs = [m for m in movers_all if m["option_type"] == "PE"]
    ce_net_l = round(sum(m["session_change_lakhs"] for m in ce_legs), 2)
    pe_net_l = round(sum(m["session_change_lakhs"] for m in pe_legs), 2)
    ce_build_l = round(sum(m["session_change_lakhs"] for m in ce_legs if m["session_change_pct"] > 0), 2)
    pe_build_l = round(sum(m["session_change_lakhs"] for m in pe_legs if m["session_change_pct"] > 0), 2)
    ce_unwind_l = round(sum(abs(m["session_change_lakhs"]) for m in ce_legs if m["session_change_pct"] < 0), 2)
    pe_unwind_l = round(sum(abs(m["session_change_lakhs"]) for m in pe_legs if m["session_change_pct"] < 0), 2)

    pe_wall_below = round(sum(
        m["session_change_lakhs"] for m in pe_legs
        if m.get("vs_spot") == "below" and m["session_change_pct"] > 0
    ), 2)
    ce_wall_above = round(sum(
        m["session_change_lakhs"] for m in ce_legs
        if m.get("vs_spot") == "above" and m["session_change_pct"] > 0
    ), 2)
    pe_unwind_below = round(sum(
        abs(m["session_change_lakhs"]) for m in pe_legs
        if m.get("vs_spot") == "below" and m["session_change_pct"] < 0
    ), 2)
    ce_unwind_above = round(sum(
        abs(m["session_change_lakhs"]) for m in ce_legs
        if m.get("vs_spot") == "above" and m["session_change_pct"] < 0
    ), 2)

    bear_score = pe_wall_below + ce_unwind_above + max(pe_net_l, 0)
    bull_score = ce_wall_above + pe_unwind_below + max(ce_net_l, 0)
    if bear_score > bull_score * 1.15:
        bias, arrow, color_side = "BEARISH", "↓", "PE"
        read = "Put OI building below spot / calls unwinding above — breakdown fuel"
    elif bull_score > bear_score * 1.15:
        bias, arrow, color_side = "BULLISH", "↑", "CE"
        read = "Call OI building above spot / puts unwinding below — breakout fuel"
    else:
        bias, arrow, color_side = "MIXED", "↔", "NEUTRAL"
        read = "Two-way OI rotation — wait for a wall break or spike"

    return {
        "bias": bias,
        "arrow": arrow,
        "side": color_side,
        "read": read,
        "ce_net_lakhs": ce_net_l,
        "pe_net_lakhs": pe_net_l,
        "ce_build_lakhs": ce_build_l,
        "pe_build_lakhs": pe_build_l,
        "ce_unwind_lakhs": ce_unwind_l,
        "pe_unwind_lakhs": pe_unwind_l,
        "pe_wall_below_lakhs": pe_wall_below,
        "ce_wall_above_lakhs": ce_wall_above,
    }


def _build_ladder(
    movers_all: list[dict[str, Any]],
    leaders: list[dict[str, Any]],
    walls: list[dict[str, Any]],
    spot: float | None,
) -> list[dict[str, Any]]:
    strike_ids = sorted(
        {m["strike"] for m in leaders}
        | {w["strike"] for w in walls}
        | {m["strike"] for m in movers_all if m.get("is_wall") or m.get("is_spike")},
        reverse=True,
    )
    by_key = {(m["strike"], m["option_type"]): m for m in movers_all}
    ladder: list[dict[str, Any]] = []
    for strike in strike_ids:
        pos_key, pos_label, dist_pts = _position_vs_spot(strike, spot)
        ce = by_key.get((strike, "CE"))
        pe = by_key.get((strike, "PE"))
        if not ce and not pe:
            continue
        ladder.append({
            "strike": strike,
            "vs_spot": pos_key,
            "vs_spot_label": pos_label,
            "dist_from_spot_pts": dist_pts,
            "is_near_spot": dist_pts is not None and dist_pts <= 150,
            "ce": _leg_summary(ce),
            "pe": _leg_summary(pe),
        })
    return ladder


def compute_strike_oi_change(
    data_dir: Path,
    trade_date: date,
    underlying: str = "NIFTY",
    *,
    top_n: int = TOP_N_PER_SIDE,
) -> dict[str, Any]:
    snapshots, coverage = load_session_option_chain_snapshots(
        data_dir, trade_date, underlying, backfill_from_index=False,
    )
    snapshots, missing_bars = fill_missing_session_bars(snapshots, trade_date)
    coverage = {
        **coverage,
        "missing_bars": missing_bars,
        "missing_bar_count": len(missing_bars),
    }
    if len(snapshots) < 2:
        msg = "Need 2+ session snapshots — run Refresh to capture option chain."
        return asdict(StrikeOiChangeResult(
            available=False,
            snapshots=len(snapshots),
            headline=msg,
            reasons=[msg],
            coverage=coverage,
        ))

    timestamps = [str(s["timestamp"]) for s in snapshots]
    spot_series = [round(float(s.get("spot_price") or 0), 2) for s in snapshots]
    spot_interval_pct = _spot_interval_series(spot_series)
    chain_step = _oc_strike_step()
    baseline: dict[tuple[int, str], float] = {}
    oi_by_ts: dict[tuple[int, str], dict[str, float]] = {}

    for snap in snapshots:
        ts = str(snap["timestamp"])
        for row in snap.get("strikes") or []:
            strike = int(row.get("strike") or 0)
            opt = str(row.get("option_type") or "").upper()
            if strike <= 0 or opt not in ("CE", "PE"):
                continue
            if chain_step > 0 and strike % chain_step != 0:
                continue
            key = (strike, opt)
            oi = float(row.get("oi") or 0)
            if key not in baseline:
                baseline[key] = oi
            oi_by_ts.setdefault(key, {})[ts] = oi

    points: dict[tuple[int, str], list[dict[str, Any]]] = {}
    for key, ts_map in oi_by_ts.items():
        base = baseline[key]
        prev_oi: float | None = None
        hist: list[dict[str, Any]] = []
        for ts in timestamps:
            if ts not in ts_map:
                hist.append({
                    "timestamp": ts,
                    "oi": None,
                    "session_change_pct": None,
                    "session_change_abs": None,
                    "interval_change_pct": None,
                    "interval_change_abs": None,
                })
                continue
            oi = ts_map[ts]
            interval_ref = prev_oi if prev_oi is not None else oi
            interval_abs = round(oi - interval_ref, 0) if prev_oi is not None else 0.0
            hist.append({
                "timestamp": ts,
                "oi": round(oi, 0),
                "session_change_pct": _pct_change(oi, base),
                "session_change_abs": round(oi - base, 0),
                "interval_change_pct": _pct_from_prev(oi, interval_ref) if prev_oi is not None else 0.0,
                "interval_change_abs": interval_abs,
            })
            prev_oi = oi
        points[key] = hist

    movers_all: list[dict[str, Any]] = []
    for (strike, opt), hist in points.items():
        base_oi = baseline[(strike, opt)]
        if base_oi < MIN_BASELINE_OI or not hist:
            continue
        last = next((p for p in reversed(hist) if p["oi"] is not None), None)
        if not last:
            continue
        prev = next((p for p in reversed(hist[:-1]) if p["oi"] is not None), last)
        movers_all.append({
            "strike": strike,
            "option_type": opt,
            "label": f"{strike} {opt}",
            "oi": last["oi"],
            "baseline_oi": round(base_oi, 0),
            "session_change_pct": last["session_change_pct"],
            "session_change_abs": last["session_change_abs"],
            "session_change_lakhs": _lakhs(last["session_change_abs"]),
            "interval_change_pct": last["interval_change_pct"] or 0.0,
            "interval_change_abs": last["interval_change_abs"] or 0.0,
            "interval_change_lakhs": _lakhs(last["interval_change_abs"] or 0.0),
            "history": hist,
        })

    if not movers_all:
        msg = "Option-chain rows found but OI too thin to rank strikes."
        return asdict(StrikeOiChangeResult(
            available=False,
            snapshots=len(snapshots),
            headline=msg,
            timestamps=timestamps,
            spot_series=spot_series,
            reasons=[msg],
        ))

    interval_abs_vals = [abs(m["interval_change_pct"]) for m in movers_all]
    interval_median = (
        round(statistics.median(interval_abs_vals), 2) if interval_abs_vals else None
    )
    spike_threshold = (
        max((interval_median or 0) * SPIKE_MEDIAN_MULT, SPIKE_MEDIAN_FLOOR)
        if interval_median is not None
        else SPIKE_MEDIAN_FLOOR
    )

    last_snap = snapshots[-1]
    spot = float(last_snap.get("spot_price") or 0) or None
    atm = int(last_snap.get("atm_strike") or 0) or None

    ce_build_rank = {
        m["label"]: i + 1
        for i, m in enumerate(sorted(
            [x for x in movers_all if x["session_change_pct"] > 0 and x["option_type"] == "CE"],
            key=lambda x: x["session_change_pct"],
            reverse=True,
        ))
    }
    pe_build_rank = {
        m["label"]: i + 1
        for i, m in enumerate(sorted(
            [x for x in movers_all if x["session_change_pct"] > 0 and x["option_type"] == "PE"],
            key=lambda x: x["session_change_pct"],
            reverse=True,
        ))
    }

    walls: list[dict[str, Any]] = []
    spikes: list[dict[str, Any]] = []

    for m in movers_all:
        pos_key, pos_label, dist_pts = _position_vs_spot(m["strike"], spot)
        rank = ce_build_rank.get(m["label"]) if m["option_type"] == "CE" else pe_build_rank.get(m["label"])
        is_wall = _is_wall(m, rank)
        is_spike = abs(m["interval_change_pct"]) >= spike_threshold
        tags: list[str] = []
        if is_wall:
            tags.append("WALL")
        last_hist = m["history"][-1] if m.get("history") else {}
        last_spot_i = spot_interval_pct[len(m["history"]) - 1] if m.get("history") else None
        spike_timing = _spike_timing(last_spot_i, is_spike)
        if is_spike:
            tags.append("SPIKE")
            if spike_timing == "EARLY":
                tags.append("EARLY")
            elif spike_timing == "LATE":
                tags.append("LATE")

        m.update({
            "vs_spot": pos_key,
            "vs_spot_label": pos_label,
            "dist_from_spot_pts": dist_pts,
            "tags": tags,
            "is_wall": is_wall,
            "is_spike": is_spike,
            "chart_label": f"{m['strike']} {m['option_type']} {pos_label.split()[0]}",
        })
        if is_wall:
            walls.append({
                "label": m["label"],
                "strike": m["strike"],
                "option_type": m["option_type"],
                "vs_spot_label": pos_label,
                "session_change_pct": m["session_change_pct"],
                "session_change_lakhs": m["session_change_lakhs"],
            })
        if is_spike:
            spikes.append({
                "label": m["label"],
                "strike": m["strike"],
                "option_type": m["option_type"],
                "interval_change_pct": m["interval_change_pct"],
                "interval_change_lakhs": m["interval_change_lakhs"],
                "vs_spot_label": pos_label,
                "spike_timing": spike_timing,
                "spot_interval_pct": last_spot_i,
            })

    walls.sort(key=lambda w: w["session_change_pct"], reverse=True)
    spikes.sort(key=lambda s: abs(s["interval_change_pct"]), reverse=True)

    ce_ranked = sorted(
        [m for m in movers_all if m["option_type"] == "CE"],
        key=lambda m: abs(m["session_change_pct"]),
        reverse=True,
    )[:top_n]
    pe_ranked = sorted(
        [m for m in movers_all if m["option_type"] == "PE"],
        key=lambda m: abs(m["session_change_pct"]),
        reverse=True,
    )[:top_n]
    leaders = ce_ranked + pe_ranked

    chart_series = [
        {
            "label": m["chart_label"],
            "strike": m["strike"],
            "option_type": m["option_type"],
            "vs_spot": m["vs_spot"],
            "vs_spot_label": m["vs_spot_label"],
            "is_wall": m["is_wall"],
            "is_spike": m["is_spike"],
            "values": [p["session_change_pct"] for p in m["history"]],
        }
        for m in leaders
    ]

    heatmap: list[dict[str, Any]] = []
    oi_series: list[dict[str, Any]] = []
    heatmap_vals: list[float] = []
    interval_vals: list[float] = []
    oi_level_vals: list[float] = []
    ladder_keys: set[tuple[int, str]] = set()
    if snapshots:
        for row in snapshots[-1].get("strikes") or []:
            strike = int(row.get("strike") or 0)
            opt = str(row.get("option_type") or "").upper()
            if strike <= 0 or opt not in ("CE", "PE"):
                continue
            if chain_step > 0 and strike % chain_step != 0:
                continue
            ladder_keys.add((strike, opt))

    def _on_current_ladder(m: dict[str, Any]) -> bool:
        return not ladder_keys or (m["strike"], m["option_type"]) in ladder_keys

    ce_for_viz = sorted(
        [m for m in movers_all if m["option_type"] == "CE" and _on_current_ladder(m)],
        key=lambda m: m["strike"],
        reverse=True,
    )
    pe_for_viz = sorted(
        [m for m in movers_all if m["option_type"] == "PE" and _on_current_ladder(m)],
        key=lambda m: m["strike"],
        reverse=True,
    )
    for m in ce_for_viz + pe_for_viz:
        session_vals = [p["session_change_pct"] for p in m["history"]]
        bar_vals = [p["interval_change_pct"] for p in m["history"]]
        buildup_vals: list[str | None] = []
        timing_vals: list[str | None] = []
        for idx, p in enumerate(m["history"]):
            oi_i = p.get("interval_change_pct")
            spot_i = spot_interval_pct[idx] if idx < len(spot_interval_pct) else None
            buildup_vals.append(_classify_buildup(spot_i, oi_i, m["option_type"]))
            is_spike_bar = oi_i is not None and abs(oi_i) >= spike_threshold
            timing_vals.append(_spike_timing(spot_i, is_spike_bar))
        oi_lakhs_vals = [
            _lakhs(p["oi"]) if p.get("oi") is not None else None
            for p in m["history"]
        ]
        heatmap_vals.extend(abs(v) for v in session_vals if v is not None)
        interval_vals.extend(abs(v) for v in bar_vals if v is not None)
        oi_level_vals.extend(v for v in oi_lakhs_vals if v is not None)
        strike_v = m["strike"]
        if m["option_type"] == "CE":
            moneyness = "ITM" if (spot is not None and strike_v < spot) else "OTM"
        else:
            moneyness = "ITM" if (spot is not None and strike_v > spot) else "OTM"
        heatmap.append({
            "label": m["label"],
            "option_type": m["option_type"],
            "strike": m["strike"],
            "vs_spot_label": m["vs_spot_label"],
            "moneyness": moneyness,
            "values": bar_vals,
            "session_values": session_vals,
            "oi_lakhs": oi_lakhs_vals,
            "buildup": buildup_vals,
            "spike_timing": timing_vals,
            "is_wall": m["is_wall"],
            "is_spike": m["is_spike"],
        })
        oi_series.append({
            "label": m["label"],
            "option_type": m["option_type"],
            "strike": m["strike"],
            "vs_spot_label": m["vs_spot_label"],
            "values": oi_lakhs_vals,
            "is_wall": m["is_wall"],
            "is_spike": m["is_spike"],
        })
    heatmap_max = max(heatmap_vals) if heatmap_vals else 1.0
    heatmap_interval_max = max(interval_vals) if interval_vals else 1.0
    oi_level_max = max(oi_level_vals) if oi_level_vals else 1.0
    time_labels = [ts[11:16] if len(ts) >= 16 else ts for ts in timestamps]
    bar_sources = [str(s.get("source") or "option_chain") for s in snapshots]

    movers = sorted(movers_all, key=lambda m: abs(m["session_change_pct"]), reverse=True)[:MAX_MOVERS]
    for m in movers:
        m["action"] = "BUILD" if m["session_change_pct"] > 0.5 else ("UNWIND" if m["session_change_pct"] < -0.5 else "FLAT")
        m["arrow"] = "▲" if m["action"] == "BUILD" else ("▼" if m["action"] == "UNWIND" else "—")
        m.pop("history", None)

    direction = _compute_direction(movers_all, spot)
    ladder = _build_ladder(movers_all, leaders, walls, spot)

    headline_parts: list[str] = []
    if walls:
        w = walls[0]
        headline_parts.append(
            f"{w['label']} WALL {w['session_change_pct']:+.1f}% ({w['session_change_lakhs']:+.1f}L) {w['vs_spot_label']}"
        )
    elif movers:
        top = movers[0]
        direction = "building" if top["session_change_pct"] > 0 else "unwinding"
        headline_parts.append(
            f"Largest move: {top['label']} {direction} {top['session_change_pct']:+.1f}%"
        )
    if spikes:
        s = spikes[0]
        headline_parts.append(
            f"SPIKE {s['label']} interval {s['interval_change_pct']:+.1f}% "
            f"({s['interval_change_lakhs']:+.1f}L)"
        )
    headline = " · ".join(headline_parts) if headline_parts else "Strike OI change tracked from option-chain snapshots."

    reasons: list[str] = []
    if coverage.get("missing_bars"):
        reasons.append(
            "Missing option-chain captures (empty columns): "
            + ", ".join(coverage["missing_bars"])
            + " — scheduler gap or run backfill-option-chain"
        )
    if coverage.get("chain_first") and coverage.get("chain_first") != "09:15":
        reasons.append(
            f"Option chain captures start at {coverage['chain_first']} — run scraper from 09:15 for full ladder"
        )
    if interval_median is not None:
        reasons.append(
            f"Interval median |ΔOI| = {interval_median:.2f}% · spike flag ≥ {spike_threshold:.2f}% "
            f"({SPIKE_MEDIAN_MULT}× median) · EARLY = |Δspot|≤{EARLY_SPIKE_SPOT_PCT}% · "
            f"LATE = |Δspot|≥{LATE_SPIKE_SPOT_PCT}%"
        )
    for m in movers[:5]:
        tag_txt = ", ".join(m["tags"]) if m["tags"] else "—"
        side_hint = "resistance / call wall" if m["is_wall"] and m["option_type"] == "CE" else (
            "support / put wall" if m["is_wall"] and m["option_type"] == "PE" else
            "covering / squeeze fuel" if m["session_change_pct"] < 0 else "rotation"
        )
        reasons.append(
            f"{m['label']} {m['vs_spot_label']}: session {m['session_change_pct']:+.1f}% "
            f"({m['session_change_lakhs']:+.1f}L), interval {m['interval_change_pct']:+.1f}% "
            f"[{tag_txt}] — {side_hint}"
        )

    return asdict(StrikeOiChangeResult(
        available=True,
        snapshots=len(snapshots),
        headline=headline,
        spot=spot,
        atm_strike=atm,
        direction=direction,
        timestamps=timestamps,
        spot_series=spot_series,
        spot_interval_pct=spot_interval_pct,
        interval_median_pct=interval_median,
        buildup_thresholds={
            "spot_pct": SPOT_INTERVAL_PCT_THRESH,
            "oi_pct": OI_INTERVAL_PCT_THRESH,
            "early_spot_pct": EARLY_SPIKE_SPOT_PCT,
            "late_spot_pct": LATE_SPIKE_SPOT_PCT,
            "spike_pct": round(spike_threshold, 2),
        },
        time_labels=time_labels,
        bar_sources=bar_sources,
        heatmap=heatmap,
        heatmap_max_abs=round(heatmap_max, 2),
        heatmap_interval_max_abs=round(heatmap_interval_max, 2),
        oi_series=oi_series,
        oi_level_max=round(oi_level_max, 2),
        ladder=ladder,
        series=chart_series,
        movers=movers,
        walls=walls,
        spikes=spikes,
        reasons=reasons,
        coverage=coverage,
    ))
