"""
Intraday trend change detection on index 5-min summaries.

Identifies when the session bias flips (bullish → bearish or vice versa) using
VWAP side, momentum, structure, and futures alignment. Distinct from trend
ignition (breakout start) — this tracks regime *changes* mid-session.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

BIAS_CONFIRM_BARS = 2
MIN_FLIP_GAP_BARS = 3


def _f(v: Any, default: float = 0.0) -> float:
    try:
        if v is None:
            return default
        return float(v)
    except (TypeError, ValueError):
        return default


def _short_time(ts: str) -> str:
    if "T" in ts:
        return ts.split("T", 1)[1][:5]
    return ts[-8:-3] if len(ts) >= 8 else ts


@dataclass
class TrendChangeEvent:
    timestamp: str
    from_side: str
    to_side: str
    spot: float | None
    trigger: str
    reasons: list[str] = field(default_factory=list)


@dataclass
class IntradayTrendChange:
    active_side: str | None
    change_count: int
    headline: str
    latest_change: dict[str, Any] | None
    changes: list[dict[str, Any]] = field(default_factory=list)
    series: list[dict[str, Any]] = field(default_factory=list)


def _bar_bias(summary: dict[str, Any]) -> tuple[int, list[str]]:
    """Return bias vote (-1 bear, 0 neutral, +1 bull) and reason tags."""
    pa = summary.get("price_action") or {}
    fut = summary.get("futures") or {}
    reasons: list[str] = []
    score = 0

    vwap = pa.get("spot_vs_vwap")
    if vwap == "above":
        score += 1
        reasons.append("above VWAP")
    elif vwap == "below":
        score -= 1
        reasons.append("below VWAP")

    mom = str(pa.get("momentum") or "").upper()
    if mom == "UP":
        score += 1
        reasons.append("4-bar momentum up")
    elif mom == "DOWN":
        score -= 1
        reasons.append("4-bar momentum down")

    chg = _f(pa.get("spot_change_pct"))
    if chg >= 0.04:
        score += 1
    elif chg <= -0.04:
        score -= 1

    open_side = pa.get("spot_vs_open")
    if open_side == "above":
        score += 1
    elif open_side == "below":
        score -= 1

    fut_prem = fut.get("premium")
    if fut_prem == "premium":
        score += 1
    elif fut_prem == "discount":
        score -= 1

    if score >= 2:
        return 1, reasons
    if score <= -2:
        return -1, reasons
    return 0, reasons


def _side_label(v: int) -> str:
    if v > 0:
        return "BULL"
    if v < 0:
        return "BEAR"
    return "NEUTRAL"


def compute_intraday_trend_change(summaries: list[dict[str, Any]]) -> dict[str, Any]:
    if not summaries:
        return asdict(IntradayTrendChange(
            active_side=None,
            change_count=0,
            headline="No data",
            latest_change=None,
        ))

    changes: list[TrendChangeEvent] = []
    series: list[dict[str, Any]] = []

    established: int | None = None
    pending: int | None = None
    pending_count = 0
    last_change_idx = -MIN_FLIP_GAP_BARS

    for i, s in enumerate(summaries):
        ts = str(s.get("timestamp", ""))
        pa = s.get("price_action") or {}
        bias, tags = _bar_bias(s)
        side = _side_label(bias)

        series.append({
            "timestamp": ts,
            "bias": side,
            "bias_score": bias,
            "spot": pa.get("spot"),
            "spot_vs_vwap": pa.get("spot_vs_vwap"),
            "momentum": pa.get("momentum"),
        })

        if bias == 0:
            pending = None
            pending_count = 0
            continue

        if established is None:
            if pending == bias:
                pending_count += 1
            else:
                pending = bias
                pending_count = 1
            if pending_count >= BIAS_CONFIRM_BARS:
                established = bias
            continue

        if bias == established:
            pending = None
            pending_count = 0
            continue

        if pending == bias:
            pending_count += 1
        else:
            pending = bias
            pending_count = 1

        if pending_count >= BIAS_CONFIRM_BARS and (i - last_change_idx) >= MIN_FLIP_GAP_BARS:
            evt = TrendChangeEvent(
                timestamp=ts,
                from_side=_side_label(established),
                to_side=_side_label(bias),
                spot=_f(pa.get("spot")) or None,
                trigger="regime_flip",
                reasons=[
                    f"Trend changed {_side_label(established)} → {_side_label(bias)}",
                    *tags[:4],
                ],
            )
            changes.append(evt)
            established = bias
            pending = None
            pending_count = 0
            last_change_idx = i

    active = _side_label(established) if established is not None else None
    change_dicts = [asdict(c) for c in changes]

    if changes:
        last = changes[-1]
        headline = (
            f"{len(changes)} trend change(s) — latest {_short_time(last.timestamp)}: "
            f"{last.from_side} → {last.to_side}"
        )
    elif active:
        headline = f"Intraday trend intact — {active} (no flip yet)"
    else:
        headline = "No established intraday trend yet — waiting for confirmation"

    return asdict(IntradayTrendChange(
        active_side=active if active != "NEUTRAL" else None,
        change_count=len(changes),
        headline=headline,
        latest_change=change_dicts[-1] if change_dicts else None,
        changes=change_dicts,
        series=series,
    ))
