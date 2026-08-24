"""
Pro Print detection — institutional footprint on index 5-min bars.

Fires when participation (volume), range expansion, OI, and clean spot/premium
body align on a directional candle. Wick-heavy / climax bars are classified separately.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from intraday_engine.analysis.candle_morphology import analyze_candle_morphology
from intraday_engine.analysis.price_action_volume import (
    _bar_volume,
    _f,
    _snap_map,
    _volume_multiple,
    _volume_spread,
)

VOL_MULT = 2.0
VOL_CLIMAX_MULT = 2.5
RANGE_MULT = 1.5
OI_CANDLE_PCT = 2.0
MIN_MOVE_PCT = 0.04
MIN_BODY_PCT = 0.08
RANGE_LOOKBACK = 20

# Climax candle (not pro print) — all three required
CLIMAX_VOL_MULT = 5.0       # ≥5× 20-bar volume MA (fut / spot / opt peak)
CLIMAX_RANGE_MULT = 4.0     # ≥4× 20-bar average range
CLIMAX_WICK_MIN_PCT = 30.0  # strictly >30% dominant rejection wick


def _short_time(ts: str) -> str:
    if "T" in ts:
        return ts.split("T", 1)[1][:5]
    return ts[-8:-3] if len(ts) >= 8 else ts


@dataclass
class ProPrintSession:
    count: int
    headline: str
    latest_side: str | None
    latest_timestamp: str | None
    candles: int
    climax_count: int = 0
    latest_climax_side: str | None = None
    latest_climax_timestamp: str | None = None
    triggers: list[dict[str, Any]] = field(default_factory=list)
    climax_triggers: list[dict[str, Any]] = field(default_factory=list)
    series: list[dict[str, Any]] = field(default_factory=list)
    factors: list[dict[str, Any]] = field(default_factory=list)


def _avg_range(ranges: list[float], end: int, window: int = RANGE_LOOKBACK) -> float | None:
    start = max(0, end - window + 1)
    chunk = [r for r in ranges[start : end + 1] if r > 0]
    if not chunk:
        return None
    return sum(chunk) / len(chunk)


def _price_dir(summary: dict[str, Any]) -> int:
    pa = summary.get("price_action") or {}
    spot_chg = _f(pa.get("spot_change_pct"))
    if spot_chg >= MIN_MOVE_PCT:
        return 1
    if spot_chg <= -MIN_MOVE_PCT:
        return -1
    open_ = _f(pa.get("open"))
    spot = _f(pa.get("spot"))
    if spot > open_:
        return 1
    if spot < open_:
        return -1
    fut = summary.get("futures") or {}
    fut_chg = _f(fut.get("change_pct"))
    if fut_chg >= MIN_MOVE_PCT:
        return 1
    if fut_chg <= -MIN_MOVE_PCT:
        return -1
    return 0


def _oi_confirm(oi: dict[str, Any], price_dir: int) -> tuple[bool, str | None]:
    if price_dir == 0:
        return False, None
    call_c = _f(oi.get("call_oi_change_candle_pct"))
    put_c = _f(oi.get("put_oi_change_candle_pct"))
    fut_c = _f(oi.get("fut_oi_change_candle_pct"))
    if price_dir > 0:
        if call_c >= OI_CANDLE_PCT:
            return True, f"CE OI +{call_c:.1f}% on bar"
        if put_c <= -OI_CANDLE_PCT:
            return True, f"PE OI {put_c:.1f}% unwind on bar"
        if fut_c >= OI_CANDLE_PCT:
            return True, f"Fut OI +{fut_c:.1f}% on bar"
    else:
        if put_c >= OI_CANDLE_PCT:
            return True, f"PE OI +{put_c:.1f}% on bar"
        if call_c <= -OI_CANDLE_PCT:
            return True, f"CE OI {call_c:.1f}% unwind on bar"
        if fut_c >= OI_CANDLE_PCT:
            return True, f"Fut OI +{fut_c:.1f}% (short build) on bar"
    return False, None


def _vol_ok(
    price_dir: int,
    fut_mult: float | None,
    spot_mult: float | None,
    opt_mult: float | None,
    spread_vote: int,
) -> tuple[bool, float | None, str]:
    best = max(
        (fut_mult or 0),
        (spot_mult or 0),
        (opt_mult or 0) if spread_vote == price_dir else 0,
    )
    if fut_mult and fut_mult >= VOL_MULT:
        return True, fut_mult, "future"
    if spot_mult and spot_mult >= VOL_MULT:
        return True, spot_mult, "spot"
    if opt_mult and opt_mult >= VOL_MULT and spread_vote == price_dir:
        return True, opt_mult, "options"
    return False, best if best > 0 else None, "—"


def _climax_morph_side(morph: dict[str, Any]) -> str | None:
    pattern = morph.get("pattern") or ""
    if pattern.startswith("CLIMAX"):
        return morph.get("climax_side")
    for sig in morph.get("premium_signals") or []:
        if sig.get("level") == "resistance":
            return sig.get("side")
    return None


def _climax_wick_ok(morph: dict[str, Any], side: str | None) -> bool:
    """Dominant rejection wick must be >30% of that leg's range."""
    if side == "CE":
        spot = morph.get("spot_upper_wick_pct") or 0
        prem = morph.get("ce_premium_upper_wick_pct") or 0
        return spot > CLIMAX_WICK_MIN_PCT or prem > CLIMAX_WICK_MIN_PCT
    if side == "PE":
        spot = morph.get("spot_lower_wick_pct") or 0
        prem = morph.get("pe_premium_upper_wick_pct") or 0
        return spot > CLIMAX_WICK_MIN_PCT or prem > CLIMAX_WICK_MIN_PCT
    return False


def _climax_bar_qualifies(
    morph: dict[str, Any],
    *,
    vol_peak: float | None,
    range_mult: float | None,
) -> tuple[bool, str | None, dict[str, bool]]:
    """Return (qualified, side, gate_flags) for Pro Print climax list."""
    side = _climax_morph_side(morph)
    gates = {
        "morph": side is not None,
        "vol": bool(vol_peak is not None and vol_peak >= CLIMAX_VOL_MULT),
        "range": bool(range_mult is not None and range_mult >= CLIMAX_RANGE_MULT),
        "wick": _climax_wick_ok(morph, side),
    }
    qualified = side is not None and all(gates.values())
    return qualified, side if qualified else None, gates


def _evaluate_bar(
    summary: dict[str, Any],
    *,
    idx: int,
    range_history: list[float],
    snap: dict[str, Any] | None,
    spot_vol_history: list[float],
    fut_vol_history: list[float],
    opt_vol_history: list[float],
    prev_summary: dict[str, Any] | None = None,
) -> dict[str, Any]:
    pa = summary.get("price_action") or {}
    oi = summary.get("oi") or {}
    opt = summary.get("options") or {}
    prev_opt = (prev_summary or {}).get("options") or {}
    ts = str(summary.get("timestamp", ""))
    pav = summary.get("pav") or {}
    pav_vs = pav.get("volume_spread") or {}
    pav_pa = pav.get("price_action") or {}

    candle_range = _f(pa.get("candle_range"))
    body_pct = _f(pa.get("candle_body_pct"))
    spot_chg = _f(pa.get("spot_change_pct"))
    price_dir = _price_dir(summary)
    side = "CE" if price_dir > 0 else ("PE" if price_dir < 0 else None)

    avg_rng = _avg_range(range_history, idx)
    range_mult = round(candle_range / avg_rng, 2) if avg_rng and avg_rng > 0 and candle_range > 0 else None
    range_ok = bool(range_mult and range_mult >= RANGE_MULT)

    fut_mult = pav_vs.get("future_volume_multiple")
    spot_mult = pav_vs.get("spot_volume_multiple")
    opt_mult = pav_vs.get("options_volume_multiple")
    spread_vote = int(pav_vs.get("vote") or 0)

    if fut_mult is None and snap is not None:
        vol = _bar_volume(snap)
        fut_mult = _volume_multiple(fut_vol_history, vol.get("future"))
        spot_mult = _volume_multiple(spot_vol_history, vol.get("spot"))
        opt_mult = _volume_multiple(opt_vol_history, vol.get("total_options"))
        spread_vote = _volume_spread(vol.get("call"), vol.get("put")).get("vote", 0)

    vol_ok, vol_peak, vol_source = _vol_ok(price_dir, fut_mult, spot_mult, opt_mult, spread_vote)
    oi_ok, oi_reason = _oi_confirm(oi, price_dir)
    move_ok = abs(spot_chg) >= MIN_MOVE_PCT or body_pct >= MIN_BODY_PCT

    morph = analyze_candle_morphology(
        pa,
        opt,
        prev_options=prev_opt,
        price_dir=price_dir,
        range_mult=range_mult,
        vol_mult=vol_peak,
        candle_type=str(pav_pa.get("candle_type") or ""),
    )

    def _premium_resistance(side_key: str | None) -> bool:
        if not side_key:
            return False
        return any(
            s.get("side") == side_key and s.get("level") == "resistance"
            for s in (morph.get("premium_signals") or [])
        )

    side_premium_block = _premium_resistance(side)
    morph_blocks = bool(morph.get("blocks_pro_print") or side_premium_block)
    climax_vol = bool(vol_peak and vol_peak >= VOL_CLIMAX_MULT and not morph_blocks)

    qualifies = bool(price_dir != 0 and vol_ok and range_ok and move_ok and (oi_ok or climax_vol))
    fired = bool(qualifies and not morph_blocks)

    climax_candle, climax_side, climax_gates = _climax_bar_qualifies(
        morph, vol_peak=vol_peak, range_mult=range_mult,
    )

    reasons: list[str] = []
    if climax_candle:
        wick_note = (
            f"spot ↑{morph.get('spot_upper_wick_pct')}%"
            if climax_side == "CE"
            else f"spot ↓{morph.get('spot_lower_wick_pct')}%"
        )
        reasons.append(
            f"CLIMAX {climax_side} — vol {vol_peak}× (≥{CLIMAX_VOL_MULT}×) · "
            f"range {range_mult}× (≥{CLIMAX_RANGE_MULT}×) · wick >{CLIMAX_WICK_MIN_PCT}% ({wick_note})"
        )
        for imp in morph.get("implications") or []:
            if imp not in reasons:
                reasons.append(imp)
    elif morph.get("climax_candle") or _climax_morph_side(morph):
        # Morphology hint only — did not pass climax gates
        miss = []
        if not climax_gates["vol"]:
            miss.append(f"vol {vol_peak or '—'}× (need ≥{CLIMAX_VOL_MULT}×)")
        if not climax_gates["range"]:
            miss.append(f"range {range_mult or '—'}× (need ≥{CLIMAX_RANGE_MULT}×)")
        if not climax_gates["wick"]:
            miss.append(f"wick ≤{CLIMAX_WICK_MIN_PCT}%")
        if miss:
            reasons.append(f"Near-climax ({_climax_morph_side(morph) or '—'}) — {' · '.join(miss)}")
    elif morph_blocks and morph.get("block_reason"):
        reasons.append(morph["block_reason"])
    for sig in morph.get("premium_signals") or []:
        msg = sig.get("text")
        if msg and msg not in reasons:
            reasons.append(msg)
    if fired:
        reasons.append(f"{'Bullish' if price_dir > 0 else 'Bearish'} pro print — {side}")
    if vol_ok:
        reasons.append(f"{vol_source.title()} vol {vol_peak}× avg (≥{VOL_MULT}×)")
    elif vol_peak:
        reasons.append(f"Vol {vol_peak}× — below {VOL_MULT}× threshold")
    if range_ok:
        reasons.append(f"Range {range_mult}× avg (≥{RANGE_MULT}×)")
    elif range_mult is not None:
        reasons.append(f"Range {range_mult}× — below {RANGE_MULT}× threshold")
    if oi_ok and oi_reason:
        reasons.append(oi_reason)
    elif climax_vol and not morph_blocks:
        reasons.append(f"Volume climax {vol_peak}× — OI substitute")
    elif price_dir != 0 and not climax_candle:
        reasons.append(f"OI candle change < {OI_CANDLE_PCT}% — no confirm")

    score = 0.0
    if fired:
        score = 55.0
        if vol_peak:
            score += min(20.0, (vol_peak - VOL_MULT) * 12)
        if range_mult:
            score += min(15.0, (range_mult - RANGE_MULT) * 10)
        if oi_ok:
            score += 10.0
        if body_pct >= 0.12 or pav_pa.get("candle_type") in ("MARUBOZU", "STRONG_BULL", "STRONG_BEAR"):
            score += 10.0
        if not morph.get("premium_issue") and not side_premium_block:
            score += 5.0
        score = min(100.0, round(score, 1))

    return {
        "timestamp": ts,
        "fired": fired,
        "climax_candle": climax_candle,
        "climax_side": climax_side,
        "climax_gates": climax_gates,
        "morphology": morph.get("pattern"),
        "side": side,
        "score": score,
        "price_dir": price_dir,
        "vol_ok": vol_ok,
        "range_ok": range_ok,
        "oi_ok": oi_ok,
        "move_ok": move_ok,
        "premium_ok": not side_premium_block,
        "vol_source": vol_source,
        "fut_mult": fut_mult,
        "spot_mult": spot_mult,
        "opt_mult": opt_mult,
        "vol_peak": vol_peak,
        "range_mult": range_mult,
        "spot_change_pct": round(spot_chg, 3),
        "candle_range": round(candle_range, 2) if candle_range else None,
        "spot_upper_wick_pct": morph.get("spot_upper_wick_pct"),
        "spot_lower_wick_pct": morph.get("spot_lower_wick_pct"),
        "ce_premium_upper_wick_pct": morph.get("ce_premium_upper_wick_pct"),
        "ce_premium_lower_wick_pct": morph.get("ce_premium_lower_wick_pct"),
        "pe_premium_upper_wick_pct": morph.get("pe_premium_upper_wick_pct"),
        "pe_premium_lower_wick_pct": morph.get("pe_premium_lower_wick_pct"),
        "premium_signals": morph.get("premium_signals") or [],
        "oi_reason": oi_reason,
        "wick_reason": morph.get("block_reason"),
        "implications": morph.get("implications") or [],
        "reasons": reasons,
    }


def attach_pro_print_to_summaries(
    summaries: list[dict[str, Any]],
    snapshots: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    if not summaries:
        return summaries
    smap = _snap_map(snapshots)
    range_hist: list[float] = []
    spot_hist: list[float] = []
    fut_hist: list[float] = []
    opt_hist: list[float] = []

    for i, s in enumerate(summaries):
        pa = s.get("price_action") or {}
        rng = _f(pa.get("candle_range"))
        if rng > 0:
            range_hist.append(rng)

        ts = str(s.get("timestamp", ""))
        snap = smap.get(ts)
        vol = _bar_volume(snap)
        if vol.get("spot"):
            spot_hist.append(vol["spot"])
        if vol.get("future"):
            fut_hist.append(vol["future"])
        if vol.get("total_options"):
            opt_hist.append(vol["total_options"])

        prev = summaries[i - 1] if i > 0 else None
        s["pro_print"] = _evaluate_bar(
            s,
            idx=i,
            range_history=range_hist,
            snap=snap,
            spot_vol_history=spot_hist.copy(),
            fut_vol_history=fut_hist.copy(),
            opt_vol_history=opt_hist.copy(),
            prev_summary=prev,
        )
    return summaries


def compute_pro_print(
    summaries: list[dict[str, Any]],
    snapshots: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    summaries = attach_pro_print_to_summaries(list(summaries), snapshots)
    if not summaries:
        return asdict(ProPrintSession(
            count=0,
            headline="No data",
            latest_side=None,
            latest_timestamp=None,
            candles=0,
        ))

    triggers = [s["pro_print"] for s in summaries if (s.get("pro_print") or {}).get("fired")]
    climax_triggers = [
        s["pro_print"] for s in summaries if (s.get("pro_print") or {}).get("climax_candle")
    ]
    latest = triggers[-1] if triggers else None
    latest_climax = climax_triggers[-1] if climax_triggers else None

    if latest:
        headline = (
            f"{len(triggers)} pro print(s) — latest {latest.get('side')} @ "
            f"{_short_time(str(latest.get('timestamp', '')))} (score {latest.get('score')})"
        )
    elif latest_climax:
        headline = (
            f"{len(climax_triggers)} climax candle(s) — latest {latest_climax.get('climax_side')} @ "
            f"{_short_time(str(latest_climax.get('timestamp', '')))} (not pro print — wick rejection)"
        )
    else:
        headline = (
            f"No pro prints yet — need vol ≥{VOL_MULT}×, range ≥{RANGE_MULT}× avg, OI confirm, "
            f"clean body. Climax (not pro print): vol ≥{CLIMAX_VOL_MULT}× · range ≥{CLIMAX_RANGE_MULT}× · wick >{CLIMAX_WICK_MIN_PCT}%"
        )

    factors: list[dict[str, Any]] = []
    ref = latest or latest_climax
    if ref:
        factors = [
            {
                "key": "vol",
                "label": "Volume",
                "value": f"{ref.get('vol_peak')}× ({ref.get('vol_source')})",
                "state": "bullish" if ref.get("side") == "CE" or ref.get("climax_side") == "CE" else "bearish",
                "hint": f"Fut {ref.get('fut_mult')}× · spot {ref.get('spot_mult')}× · opt {ref.get('opt_mult')}×",
            },
            {
                "key": "range",
                "label": "Range ×",
                "value": f"{ref.get('range_mult')}×",
                "state": "neutral",
                "hint": f"Candle range {ref.get('candle_range')} pts",
            },
            {
                "key": "premium",
                "label": "Premium wick",
                "value": "✓ clean" if ref.get("premium_ok", True) else "✗ wick",
                "state": "neutral" if ref.get("premium_ok", True) else "warning",
                "hint": (
                    f"CE ↑{ref.get('ce_premium_upper_wick_pct') or '—'}% ↓{ref.get('ce_premium_lower_wick_pct') or '—'}% · "
                    f"PE ↑{ref.get('pe_premium_upper_wick_pct') or '—'}% ↓{ref.get('pe_premium_lower_wick_pct') or '—'}%"
                ),
            },
            {
                "key": "oi",
                "label": "OI confirm",
                "value": "✓" if ref.get("oi_ok") else ("climax" if ref.get("climax_candle") else "—"),
                "state": "bullish" if (ref.get("side") or ref.get("climax_side")) == "CE" else "bearish",
                "hint": ref.get("oi_reason") or ref.get("wick_reason") or "—",
            },
        ]

    series = [
        {
            "timestamp": s.get("timestamp"),
            "fired": (s.get("pro_print") or {}).get("fired"),
            "climax_candle": (s.get("pro_print") or {}).get("climax_candle"),
            "climax_side": (s.get("pro_print") or {}).get("climax_side"),
            "side": (s.get("pro_print") or {}).get("side"),
            "score": (s.get("pro_print") or {}).get("score"),
        }
        for s in summaries
    ]

    result = ProPrintSession(
        count=len(triggers),
        climax_count=len(climax_triggers),
        headline=headline,
        latest_side=latest.get("side") if latest else None,
        latest_timestamp=latest.get("timestamp") if latest else None,
        latest_climax_side=latest_climax.get("climax_side") if latest_climax else None,
        latest_climax_timestamp=latest_climax.get("timestamp") if latest_climax else None,
        candles=len(summaries),
        triggers=[
            {
                "timestamp": t.get("timestamp"),
                "side": t.get("side"),
                "score": t.get("score"),
                "vol_peak": t.get("vol_peak"),
                "range_mult": t.get("range_mult"),
                "spot_change_pct": t.get("spot_change_pct"),
                "reasons": t.get("reasons", []),
            }
            for t in triggers
        ],
        climax_triggers=[
            {
                "timestamp": t.get("timestamp"),
                "climax_side": t.get("climax_side"),
                "morphology": t.get("morphology"),
                "vol_peak": t.get("vol_peak"),
                "range_mult": t.get("range_mult"),
                "spot_upper_wick_pct": t.get("spot_upper_wick_pct"),
                "spot_lower_wick_pct": t.get("spot_lower_wick_pct"),
                "ce_premium_upper_wick_pct": t.get("ce_premium_upper_wick_pct"),
                "pe_premium_upper_wick_pct": t.get("pe_premium_upper_wick_pct"),
                "climax_gates": t.get("climax_gates"),
                "reasons": t.get("reasons", []),
                "implications": t.get("implications", []),
            }
            for t in climax_triggers
        ],
        series=series,
        factors=factors,
    )
    return asdict(result)
