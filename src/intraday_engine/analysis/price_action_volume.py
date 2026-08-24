"""
Price Action + Volume Spread analysis for index intraday snapshots.

Combines candle/structure price action with volume participation (spot, futures,
ATM options) and CE vs PE volume spread (options flow skew).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from intraday_engine.analysis.candle_morphology import (
    analyze_candle_morphology,
    check_setup_confirmation,
)

VOL_SPIKE_MULT = 1.5
VOL_CLIMAX_MULT = 2.5
SPREAD_DOMINANCE = 1.25
MIN_CANDLES = 3
STRUCT_LOOKBACK = 6


@dataclass
class PavFactor:
    key: str
    label: str
    value: str
    state: str  # bullish | bearish | neutral | na
    vote: int
    hint: str


@dataclass
class PriceActionVolume:
    score: float
    verdict: str
    headline: str
    candles: int
    price_action: dict[str, Any] = field(default_factory=dict)
    volume_spread: dict[str, Any] = field(default_factory=dict)
    factors: list[dict[str, Any]] = field(default_factory=list)
    insights: list[dict[str, Any]] = field(default_factory=list)
    warnings: list[dict[str, Any]] = field(default_factory=list)


def _f(v: Any, default: float = 0.0) -> float:
    try:
        if v is None:
            return default
        return float(v)
    except (TypeError, ValueError):
        return default


def _snap_map(snapshots: list[dict[str, Any]] | None) -> dict[str, dict[str, Any]]:
    if not snapshots:
        return {}
    out: dict[str, dict[str, Any]] = {}
    for row in snapshots:
        ts = str(row.get("timestamp", ""))
        if ts:
            out[ts] = row
    return out


def _candle_type(body_pct: float, high: float, low: float, open_: float, close: float) -> str:
    rng = high - low
    if rng <= 0:
        return "FLAT"
    body = abs(close - open_)
    body_ratio = body / rng
    upper_wick = high - max(open_, close)
    lower_wick = min(open_, close) - low
    upper_ratio = upper_wick / rng
    lower_ratio = lower_wick / rng

    if body_ratio < 0.15:
        return "DOJI"
    if body_ratio > 0.75:
        return "MARUBOZU"
    if lower_ratio > 0.55 and body_ratio < 0.35 and close >= open_:
        return "HAMMER"
    if upper_ratio > 0.55 and body_ratio < 0.35 and close <= open_:
        return "SHOOTING_STAR"
    if body_pct > 0.12:
        return "STRONG_BULL" if close > open_ else "STRONG_BEAR"
    return "BULL" if close > open_ else ("BEAR" if close < open_ else "FLAT")


def _structure(closes: list[float], highs: list[float], lows: list[float]) -> tuple[str, int]:
    """Swing structure over recent bars → label + vote."""
    n = min(STRUCT_LOOKBACK, len(closes))
    if n < 3:
        return "UNDEFINED", 0
    c = closes[-n:]
    h = highs[-n:]
    lo = lows[-n:]

    hh = h[-1] > max(h[:-1]) if len(h) > 1 else False
    hl = lo[-1] > min(lo[:-1]) if len(lo) > 1 else False
    lh = h[-1] < max(h[:-1]) if len(h) > 1 else False
    ll = lo[-1] < min(lo[:-1]) if len(lo) > 1 else False

    if hh and hl:
        return "HIGHER_HIGH_HL", 1
    if lh and ll:
        return "LOWER_LOW_LH", -1
    if c[-1] > c[0] and c[-1] >= c[-2]:
        return "UPSWING", 1
    if c[-1] < c[0] and c[-1] <= c[-2]:
        return "DOWNSWING", -1
    return "RANGE", 0


def _bar_volume(snap: dict[str, Any] | None) -> dict[str, float | None]:
    if not snap:
        return {"spot": None, "future": None, "call": None, "put": None, "total_options": None}
    spot = _f(snap.get("spot_volume")) if snap.get("spot_volume") not in (None, "") else None
    fut = _f(snap.get("future_volume")) if snap.get("future_volume") not in (None, "") else None
    call = _f(snap.get("call_volume")) if snap.get("call_volume") not in (None, "") else None
    put = _f(snap.get("put_volume")) if snap.get("put_volume") not in (None, "") else None
    total_opt = (call or 0) + (put or 0)
    return {
        "spot": spot,
        "future": fut,
        "call": call,
        "put": put,
        "total_options": total_opt if total_opt > 0 else None,
    }


def _volume_spread(call_vol: float | None, put_vol: float | None) -> dict[str, Any]:
    if not call_vol and not put_vol:
        return {
            "pcr_volume": None,
            "spread": None,
            "spread_pct": None,
            "dominant_side": None,
            "vote": 0,
        }
    cv = call_vol or 0.0
    pv = put_vol or 0.0
    total = cv + pv
    pcr = round(pv / cv, 2) if cv > 0 else None
    spread = round(pv - cv, 0)
    spread_pct = round((pv - cv) / total * 100, 1) if total > 0 else None
    vote = 0
    dominant = "BALANCED"
    if cv > pv * SPREAD_DOMINANCE:
        vote = 1
        dominant = "CE"
    elif pv > cv * SPREAD_DOMINANCE:
        vote = -1
        dominant = "PE"
    return {
        "pcr_volume": pcr,
        "spread": spread,
        "spread_pct": spread_pct,
        "dominant_side": dominant,
        "vote": vote,
    }


def _fut_vs_spot_ratio(fut_vol: float | None, spot_vol: float | None) -> float | None:
    if not fut_vol or not spot_vol or spot_vol <= 0:
        return None
    return round(fut_vol / spot_vol, 2)


def _participation_vote(
    price_dir: int,
    spot_mult: float | None,
    fut_mult: float | None,
    opt_mult: float | None,
    spread_vote: int,
) -> int:
    """Volume confirms price when spot, fut, or aligned options vol spikes."""
    if price_dir == 0:
        return 0
    if spot_mult and spot_mult >= VOL_SPIKE_MULT:
        return price_dir
    if fut_mult and fut_mult >= VOL_SPIKE_MULT:
        return price_dir
    if opt_mult and opt_mult >= VOL_SPIKE_MULT and spread_vote == price_dir:
        return price_dir
    return 0


def _volume_multiple(series: list[float], current: float | None) -> float | None:
    if current is None or not series:
        return None
    window = series[-min(20, len(series)) :]
    avg = sum(window) / len(window) if window else 0
    return round(current / avg, 2) if avg > 0 else None


def _factor(key: str, label: str, value: str, vote: int, hint: str) -> dict[str, Any]:
    state = "bullish" if vote > 0 else ("bearish" if vote < 0 else "neutral")
    return asdict(PavFactor(key=key, label=label, value=value, state=state, vote=vote, hint=hint))


def _insight(
    category: str,
    tag: str,
    severity: str,
    text: str,
    timestamp: str | None = None,
) -> dict[str, str]:
    return {
        "category": category,
        "tag": tag,
        "severity": severity,
        "text": text,
        "timestamp": timestamp or "",
    }


def _warning(text: str, timestamp: str | None = None) -> dict[str, str]:
    return {"text": text, "timestamp": timestamp or ""}


def analyze_candle_pav(
    summary: dict[str, Any],
    *,
    history: list[dict[str, Any]],
    snap: dict[str, Any] | None,
    prev_snap: dict[str, Any] | None,
    spot_vol_history: list[float],
    fut_vol_history: list[float],
    opt_vol_history: list[float],
    prev_summary: dict[str, Any] | None = None,
) -> dict[str, Any]:
    pa = summary.get("price_action") or {}
    opt = summary.get("options") or {}
    prev_opt = (prev_summary or {}).get("options") or {}
    fut = summary.get("futures") or {}
    ts = summary.get("timestamp")
    spot = _f(pa.get("spot"))
    open_ = _f(pa.get("open"))
    high = _f(pa.get("high"))
    low = _f(pa.get("low"))
    vwap = _f(pa.get("vwap"))
    support = _f(pa.get("support"))
    resistance = _f(pa.get("resistance"))
    body_pct = _f(pa.get("candle_body_pct"))
    spot_chg = _f(pa.get("spot_change_pct"))

    closes = [_f(h.get("price_action", {}).get("spot")) for h in history]
    highs = [_f(h.get("price_action", {}).get("high")) for h in history]
    lows = [_f(h.get("price_action", {}).get("low")) for h in history]

    struct_label, struct_vote = _structure(closes, highs, lows)
    ctype = _candle_type(body_pct, high, low, open_, spot)

    vol = _bar_volume(snap)
    prev_vol = _bar_volume(prev_snap)
    spread = _volume_spread(vol.get("call"), vol.get("put"))
    prev_spread = _volume_spread(prev_vol.get("call"), prev_vol.get("put"))

    spot_mult = _volume_multiple(spot_vol_history, vol.get("spot"))
    fut_mult = _volume_multiple(fut_vol_history, vol.get("future"))
    opt_mult = _volume_multiple(opt_vol_history, vol.get("total_options"))
    fut_spot_ratio = _fut_vs_spot_ratio(vol.get("future"), vol.get("spot"))

    spread_delta = None
    if spread.get("spread_pct") is not None and prev_spread.get("spread_pct") is not None:
        spread_delta = round(spread["spread_pct"] - prev_spread["spread_pct"], 1)

    price_dir = 1 if spot_chg > 0.03 else (-1 if spot_chg < -0.03 else 0)
    fut_chg = _f(fut.get("change_pct"))
    if price_dir == 0:
        price_dir = 1 if fut_chg > 0.03 else (-1 if fut_chg < -0.03 else 0)
    vol_confirm_vote = _participation_vote(price_dir, spot_mult, fut_mult, opt_mult, spread["vote"])

    candle_rng = _f(pa.get("candle_range"))
    range_hist = [_f(h.get("price_action", {}).get("candle_range")) for h in history[:-1]]
    range_hist = [r for r in range_hist if r > 0]
    avg_rng = sum(range_hist[-20:]) / len(range_hist[-20:]) if range_hist else None
    range_mult = round(candle_rng / avg_rng, 2) if avg_rng and avg_rng > 0 and candle_rng > 0 else None
    vol_peak = max((spot_mult or 0), (fut_mult or 0), (opt_mult or 0))

    morph = analyze_candle_morphology(
        pa,
        opt,
        prev_options=prev_opt,
        price_dir=price_dir,
        range_mult=range_mult,
        vol_mult=vol_peak if vol_peak > 0 else None,
        candle_type=ctype,
    )
    pattern = morph.get("pattern")

    ts_str = str(ts or "")
    insights: list[dict[str, str]] = []
    warnings: list[dict[str, str]] = []

    def ins(category: str, tag: str, severity: str, text: str) -> dict[str, str]:
        return _insight(category, tag, severity, text, ts_str)

    def warn(text: str) -> dict[str, str]:
        return _warning(text, ts_str)

    # --- Price action insights ---
    if pa.get("breakout"):
        sev = "bullish" if spread["vote"] >= 0 else "warning"
        insights.append(ins("price_action", "breakout", sev,
            f"Spot broke resistance at {resistance:.0f}" + (" with CE volume leading" if spread["vote"] > 0 else " but PE volume dominates — weak breakout")))
    if pa.get("breakdown"):
        sev = "bearish" if spread["vote"] <= 0 else "warning"
        insights.append(ins("price_action", "breakdown", sev,
            f"Spot broke support at {support:.0f}" + (" with PE volume leading" if spread["vote"] < 0 else " but CE volume dominates — weak breakdown")))
    if spot > vwap and vwap > 0:
        insights.append(ins("price_action", "vwap", "bullish", f"Spot {spot:.0f} above VWAP {vwap:.0f} — buyers control session average"))
    elif spot < vwap and vwap > 0:
        insights.append(ins("price_action", "vwap", "bearish", f"Spot {spot:.0f} below VWAP {vwap:.0f} — sellers control session average"))

    dist_high = _f(pa.get("dist_from_session_high_pct"))
    dist_low = _f(pa.get("dist_from_session_low_pct"))
    if dist_high is not None and abs(dist_high) < 0.08:
        insights.append(ins("price_action", "session_extreme", "bearish",
            f"Testing session high — {abs(dist_high):.2f}% below high; watch rejection or breakout"))
    if dist_low is not None and abs(dist_low) < 0.08:
        insights.append(ins("price_action", "session_extreme", "bullish",
            f"Testing session low — {abs(dist_low):.2f}% above low; watch bounce or breakdown"))

    if ctype == "HAMMER" and pattern != "PINBAR_BULL":
        insights.append(ins("price_action", "pattern", "bullish", "Hammer candle — rejection of lows, potential reversal up"))
    elif ctype == "SHOOTING_STAR" and pattern != "PINBAR_BEAR":
        insights.append(ins("price_action", "pattern", "bearish", "Shooting star — rejection of highs, potential reversal down"))
    elif ctype == "DOJI" and pattern != "DOJI":
        insights.append(ins("price_action", "pattern", "neutral", "Doji — indecision; wait for next bar confirmation"))
    elif ctype in ("STRONG_BULL", "MARUBOZU") and spot > open_:
        insights.append(ins("price_action", "pattern", "bullish", f"Strong bullish candle ({ctype.replace('_', ' ')})"))
    elif ctype in ("STRONG_BEAR",) or (ctype == "MARUBOZU" and spot < open_):
        insights.append(ins("price_action", "pattern", "bearish", f"Strong bearish candle ({ctype.replace('_', ' ')})"))

    # --- Wick / climax / pinbar morphology ---
    if pattern == "PINBAR_BULL":
        insights.append(ins(
            "candle_setup", "pinbar_bull", "bullish",
            "Bullish pinbar — long lower wick (PE support) · support formation · reversal expected · liquidity sweep possible",
        ))
    elif pattern == "PINBAR_BEAR":
        insights.append(ins(
            "candle_setup", "pinbar_bear", "bearish",
            "Bearish pinbar — long upper wick (CE exhaustion) · resistance formation · reversal expected · liquidity sweep possible",
        ))
    elif pattern and str(pattern).startswith("CLIMAX"):
        side_l = morph.get("climax_side") or "—"
        wick_top = morph.get("spot_upper_wick_pct")
        wick_bot = morph.get("spot_lower_wick_pct")
        wick_note = (
            f"spot upper wick {wick_top}% of range"
            if wick_top and (wick_top >= (wick_bot or 0))
            else f"spot lower wick {wick_bot}% of range"
        )
        impl = " · ".join(morph.get("implications") or [])
        insights.append(ins(
            "candle_setup", "climax", "warning",
            f"Climax candle ({side_l}) — {wick_note} · {impl}",
        ))
    elif morph.get("blocks_pro_print") and morph.get("block_reason"):
        insights.append(ins("candle_setup", "wick_reject", "warning", morph["block_reason"]))

    for sig in morph.get("premium_signals") or []:
        tag = "premium_resistance" if sig.get("level") == "resistance" else "premium_support"
        sev = "warning" if sig.get("level") == "resistance" else "neutral"
        if sig.get("level") == "support" and sig.get("side") == "CE":
            sev = "bullish"
        elif sig.get("level") == "support" and sig.get("side") == "PE":
            sev = "bearish"
        insights.append(ins("candle_setup", tag, sev, sig.get("text", "")))

    candle_setup = {
        "timestamp": ts_str,
        "pattern": pattern,
        "climax_side": morph.get("climax_side"),
        "implications": morph.get("implications") or [],
        "expect_dir": morph.get("expect_dir"),
        "spot_upper_wick_pct": morph.get("spot_upper_wick_pct"),
        "spot_lower_wick_pct": morph.get("spot_lower_wick_pct"),
        "premium_signals": morph.get("premium_signals") or [],
    }

    if struct_label == "HIGHER_HIGH_HL":
        insights.append(ins("price_action", "structure", "bullish", "Higher high + higher low structure — uptrend intact"))
    elif struct_label == "LOWER_LOW_LH":
        insights.append(ins("price_action", "structure", "bearish", "Lower low + lower high structure — downtrend intact"))
    elif struct_label == "RANGE":
        insights.append(ins("price_action", "structure", "neutral", "Range-bound structure — fade extremes or wait for break"))

    in_range_pct = _f(pa.get("spot_in_range_pct"))
    if 45 <= in_range_pct <= 55:
        insights.append(ins("price_action", "levels", "neutral",
            f"Spot mid-range ({in_range_pct:.0f}% of S–R) — no edge at center"))

    # --- Volume spread insights ---
    if spread["pcr_volume"] is not None:
        pcr = spread["pcr_volume"]
        if pcr > 1.2:
            insights.append(ins("volume_spread", "pcr", "bearish",
                f"PCR(volume) {pcr:.2f} — put-side volume dominates (defensive/bearish flow)"))
        elif pcr < 0.8:
            insights.append(ins("volume_spread", "pcr", "bullish",
                f"PCR(volume) {pcr:.2f} — call-side volume dominates (aggressive/bullish flow)"))
        else:
            insights.append(ins("volume_spread", "pcr", "neutral",
                f"PCR(volume) {pcr:.2f} — balanced CE/PE participation"))

    if spread_delta is not None and abs(spread_delta) >= 8:
        if spread_delta > 0:
            insights.append(ins("volume_spread", "spread_shift", "bearish",
                f"Volume spread shifting toward puts (+{spread_delta:.1f}%) — bearish participation rising"))
        else:
            insights.append(ins("volume_spread", "spread_shift", "bullish",
                f"Volume spread shifting toward calls ({spread_delta:.1f}%) — bullish participation rising"))

    if spot_mult and spot_mult >= VOL_CLIMAX_MULT:
        insights.append(ins("volume_spread", "climax", "warning",
            f"Spot volume climax ({spot_mult:.1f}× avg) — possible exhaustion or initiation"))
    elif spot_mult and spot_mult >= VOL_SPIKE_MULT:
        insights.append(ins("volume_spread", "spike", "neutral",
            f"Elevated spot volume ({spot_mult:.1f}× avg) — participation expanding"))
    elif spot_mult and spot_mult < 0.6:
        insights.append(ins("volume_spread", "dry", "warning",
            f"Thin spot volume ({spot_mult:.1f}× avg) — move lacks conviction"))

    fut_lots = vol.get("future") or 0
    if fut_mult and fut_mult >= VOL_CLIMAX_MULT:
        sev = "bullish" if price_dir > 0 else ("bearish" if price_dir < 0 else "warning")
        insights.append(ins("volume_spread", "fut_climax", sev,
            f"Future volume climax ({fut_mult:.1f}× avg, {fut_lots:,.0f} lots) — derivatives-led participation"))
    elif fut_mult and fut_mult >= VOL_SPIKE_MULT:
        sev = "bullish" if price_dir > 0 else ("bearish" if price_dir < 0 else "neutral")
        insights.append(ins("volume_spread", "fut_spike", sev,
            f"Elevated future volume ({fut_mult:.1f}× avg, {fut_lots:,.0f} lots)"))
    elif fut_mult and fut_mult < 0.6:
        insights.append(ins("volume_spread", "fut_dry", "warning",
            f"Thin future volume ({fut_mult:.1f}× avg) — trend may stall"))

    if fut_spot_ratio is not None:
        if fut_spot_ratio >= 1.5:
            insights.append(ins("volume_spread", "fut_leads", "neutral",
                f"Fut/spot vol ratio {fut_spot_ratio:.2f}× — move led by derivatives, not cash"))
        elif fut_spot_ratio <= 0.5 and vol.get("spot"):
            insights.append(ins("volume_spread", "spot_leads", "neutral",
                f"Fut/spot vol ratio {fut_spot_ratio:.2f}× — cash market leading futures"))

    if not vol.get("spot") and vol.get("future") and fut_mult and fut_mult >= VOL_SPIKE_MULT:
        insights.append(ins("volume_spread", "fut_vol", "neutral",
            f"Future volume spike ({fut_mult:.1f}× avg) — {fut_lots:,.0f} lots (spot vol not stored)"))

    if not vol.get("spot") and vol.get("total_options"):
        if opt_mult and opt_mult >= VOL_SPIKE_MULT:
            insights.append(ins("volume_spread", "options_vol", "neutral",
                f"Options volume spike ({opt_mult:.1f}× avg) — ATM CE+PE {vol['total_options']:,.0f} (spot vol not stored)"))

    # --- Confluence / divergence ---
    vb = summary.get("volume_bias") or {}
    if vb.get("bias") == "BULLISH_BIAS":
        insights.append(ins("confluence", "liquidity", "bullish",
            f"Bullish volume bias — liquidity grab below at {vb.get('liquidity_grab')} ({vb.get('liquidity_grab_status')})"))
    elif vb.get("bias") == "BEARISH_BIAS":
        insights.append(ins("confluence", "liquidity", "bearish",
            f"Bearish volume bias — liquidity grab above at {vb.get('liquidity_grab')} ({vb.get('liquidity_grab_status')})"))

    if price_dir > 0 and spread["vote"] < 0:
        warnings.append(warn("Price up but PE volume leads — bullish move may lack follow-through"))
    elif price_dir < 0 and spread["vote"] > 0:
        warnings.append(warn("Price down but CE volume leads — bearish move may lack follow-through"))
    if pa.get("breakout") and vol_confirm_vote <= 0:
        warnings.append(warn("Breakout without volume confirmation — risk of false break"))
    if pa.get("breakdown") and vol_confirm_vote >= 0:
        warnings.append(warn("Breakdown without volume confirmation — risk of false break"))
    if price_dir > 0 and fut_mult and fut_mult >= VOL_SPIKE_MULT and spread["vote"] < 0:
        warnings.append(warn("Future volume elevated on up bar but PE options lead — mixed flow"))
    elif price_dir < 0 and fut_mult and fut_mult >= VOL_SPIKE_MULT and spread["vote"] > 0:
        warnings.append(warn("Future volume elevated on down bar but CE options lead — mixed flow"))

    pa_vote = struct_vote
    if spot > vwap:
        pa_vote += 1
    elif spot < vwap:
        pa_vote -= 1
    if ctype in ("HAMMER", "STRONG_BULL") or (ctype == "MARUBOZU" and spot > open_):
        pa_vote += 1
    if ctype in ("SHOOTING_STAR", "STRONG_BEAR") or (ctype == "MARUBOZU" and spot < open_):
        pa_vote -= 1

    fut_vote = 0
    if fut_mult and fut_mult >= VOL_SPIKE_MULT and price_dir != 0:
        fut_vote = price_dir

    votes = [pa_vote, spread["vote"], vol_confirm_vote, struct_vote, fut_vote]
    active = [v for v in votes if v != 0]
    net = sum(votes)
    score = round(net / max(len(votes), 1) * 25 + 50, 1)
    score = max(0.0, min(100.0, score))

    return {
        "timestamp": ts,
        "score": score,
        "net_vote": net,
        "price_action": {
            "structure": struct_label,
            "candle_type": ctype,
            "vs_vwap": pa.get("spot_vs_vwap"),
            "vs_open": pa.get("spot_vs_open"),
            "breakout": bool(pa.get("breakout")),
            "breakdown": bool(pa.get("breakdown")),
            "momentum": pa.get("momentum"),
        },
        "volume_spread": {
            **spread,
            "spread_delta_pct": spread_delta,
            "spot_volume": vol.get("spot"),
            "spot_volume_multiple": spot_mult,
            "future_volume": vol.get("future"),
            "future_volume_multiple": fut_mult,
            "fut_spot_volume_ratio": fut_spot_ratio,
            "options_volume": vol.get("total_options"),
            "options_volume_multiple": opt_mult,
            "call_volume": vol.get("call"),
            "put_volume": vol.get("put"),
        },
        "insights": insights,
        "warnings": warnings,
        "candle_setup": candle_setup,
    }


def _attach_setup_confirmations(summaries: list[dict[str, Any]]) -> None:
    """Add confirmation insights when a bar validates a prior pinbar/climax setup."""
    for i in range(1, len(summaries)):
        prev_pav = summaries[i - 1].get("pav") or {}
        setup = prev_pav.get("candle_setup") or {}
        if not setup.get("pattern"):
            continue
        pa = summaries[i].get("price_action") or {}
        conf = check_setup_confirmation(setup, pa)
        if not conf:
            continue
        ts_str = str(summaries[i].get("timestamp") or "")
        prev_ts = setup.get("timestamp", "")
        short_prev = prev_ts.split("T", 1)[1][:5] if "T" in prev_ts else prev_ts
        conf_insight = _insight(
            conf["category"],
            conf["tag"],
            conf["severity"],
            f"{conf['text']} (setup @ {short_prev})",
            ts_str,
        )
        pav = summaries[i].setdefault("pav", {})
        pav.setdefault("insights", []).append(conf_insight)


def attach_pav_to_summaries(
    summaries: list[dict[str, Any]],
    snapshots: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    if not summaries:
        return summaries
    smap = _snap_map(snapshots)
    spot_hist: list[float] = []
    fut_hist: list[float] = []
    opt_hist: list[float] = []

    for i, s in enumerate(summaries):
        ts = str(s.get("timestamp", ""))
        prev_ts = str(summaries[i - 1].get("timestamp", "")) if i > 0 else ""
        snap = smap.get(ts)
        prev_snap = smap.get(prev_ts) if prev_ts else None
        vol = _bar_volume(snap)
        if vol.get("spot"):
            spot_hist.append(vol["spot"])
        if vol.get("future"):
            fut_hist.append(vol["future"])
        if vol.get("total_options"):
            opt_hist.append(vol["total_options"])

        s["pav"] = analyze_candle_pav(
            s,
            history=summaries[: i + 1],
            snap=snap,
            prev_snap=prev_snap,
            spot_vol_history=spot_hist.copy(),
            fut_vol_history=fut_hist.copy(),
            opt_vol_history=opt_hist.copy(),
            prev_summary=summaries[i - 1] if i > 0 else None,
        )
    _attach_setup_confirmations(summaries)
    return summaries


def compute_price_action_volume(
    summaries: list[dict[str, Any]],
    snapshots: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    summaries = attach_pav_to_summaries(list(summaries), snapshots)
    if not summaries:
        return asdict(PriceActionVolume(
            score=50.0, verdict="NEUTRAL", headline="No data", candles=0,
        ))

    last = summaries[-1]
    last_pav = last.get("pav") or {}
    pa_block = last_pav.get("price_action") or {}
    vs_block = last_pav.get("volume_spread") or {}

    all_insights: list[dict[str, str]] = []
    all_warnings: list[dict[str, str]] = []
    for s in summaries:
        pav = s.get("pav") or {}
        all_insights.extend(pav.get("insights") or [])
        for w in pav.get("warnings") or []:
            if isinstance(w, dict):
                all_warnings.append(w)
            elif w:
                all_warnings.append(_warning(str(w), str(s.get("timestamp") or "")))

    all_insights.sort(key=lambda x: x.get("timestamp") or "")
    all_warnings.sort(key=lambda x: x.get("timestamp") or "")

    # Session factors from latest state
    pa = last.get("price_action") or {}
    struct = pa_block.get("structure", "—")
    struct_vote = 1 if "HIGH" in struct and "HH" in struct else (-1 if "LOW" in struct and "LL" in struct else 0)
    factors = [
        _factor("structure", "Market structure", struct, struct_vote,
                "Recent swing highs/lows pattern"),
        _factor("vwap", "Spot vs VWAP", str(pa.get("spot_vs_vwap", "—")),
                1 if pa.get("spot_vs_vwap") == "above" else (-1 if pa.get("spot_vs_vwap") == "below" else 0),
                "Institutional average price anchor"),
        _factor("pcr_vol", "PCR (volume)", str(vs_block.get("pcr_volume") or "—"),
                vs_block.get("vote", 0), f"Dominant side: {vs_block.get('dominant_side', '—')}"),
        _factor(
            "fut_vol",
            "Future vol ×",
            f"{vs_block.get('future_volume_multiple') or '—'}×",
            1 if (vs_block.get("future_volume_multiple") or 0) >= VOL_SPIKE_MULT else 0,
            f"Bar vol {vs_block.get('future_volume') or '—'} lots",
        ),
        _factor(
            "vol_mult",
            "Spot / opt vol ×",
            f"{vs_block.get('spot_volume_multiple') or vs_block.get('options_volume_multiple') or '—'}×",
            1 if (vs_block.get("spot_volume_multiple") or 0) >= VOL_SPIKE_MULT
            or (vs_block.get("options_volume_multiple") or 0) >= VOL_SPIKE_MULT else 0,
            "Vs 20-bar average",
        ),
        _factor("momentum", "4-bar momentum", str(pa.get("momentum", "—")),
                1 if pa.get("momentum") == "UP" else (-1 if pa.get("momentum") == "DOWN" else 0),
                "Short-term direction"),
    ]

    vb = last.get("volume_bias") or {}
    if vb.get("bias") and vb.get("bias") != "NEUTRAL":
        factors.append(_factor("vol_bias", "Liquidity bias", vb.get("bias", "—"),
                             1 if vb.get("bias") == "BULLISH_BIAS" else -1,
                             f"Grab @ {vb.get('liquidity_grab')} · {vb.get('liquidity_grab_status')}"))

    votes = [f["vote"] for f in factors if f["state"] != "na"]
    net = sum(votes)
    score = round((net / max(len(votes), 1) + 1) * 50, 1)
    score = max(0.0, min(100.0, score))

    if score >= 62:
        verdict = "BULLISH"
    elif score <= 38:
        verdict = "BEARISH"
    elif abs(net) <= 1:
        verdict = "NEUTRAL"
    else:
        verdict = "MIXED"

    headline = _headline(verdict, pa_block, vs_block, pa)

    series = [
        {
            "timestamp": s.get("timestamp"),
            "score": (s.get("pav") or {}).get("score"),
            "pcr_volume": ((s.get("pav") or {}).get("volume_spread") or {}).get("pcr_volume"),
        }
        for s in summaries
    ]

    result = PriceActionVolume(
        score=score,
        verdict=verdict,
        headline=headline,
        candles=len(summaries),
        price_action={
            "structure": struct,
            "candle_type": pa_block.get("candle_type"),
            "spot_vs_vwap": pa.get("spot_vs_vwap"),
            "spot_vs_open": pa.get("spot_vs_open"),
            "momentum": pa.get("momentum"),
            "breakout": bool(pa.get("breakout")),
            "breakdown": bool(pa.get("breakdown")),
            "session_high": pa.get("session_high"),
            "session_low": pa.get("session_low"),
            "support": pa.get("support"),
            "resistance": pa.get("resistance"),
        },
        volume_spread=vs_block,
        factors=factors,
        insights=all_insights,
        warnings=all_warnings,
    )
    out = asdict(result)
    out["series"] = series
    out["latest"] = last_pav
    return out


def _headline(verdict: str, pa_block: dict, vs_block: dict, pa: dict) -> str:
    if verdict == "BULLISH":
        lead = "Bullish price action + volume alignment"
    elif verdict == "BEARISH":
        lead = "Bearish price action + volume alignment"
    elif verdict == "MIXED":
        lead = "Mixed signals — price and volume disagree"
    else:
        lead = "Neutral — no clear edge"
    parts = []
    if pa_block.get("structure"):
        parts.append(pa_block["structure"].replace("_", " "))
    if vs_block.get("dominant_side") and vs_block["dominant_side"] != "BALANCED":
        parts.append(f"{vs_block['dominant_side']} vol leads")
    if vs_block.get("future_volume_multiple") and vs_block["future_volume_multiple"] >= VOL_SPIKE_MULT:
        parts.append(f"fut vol {vs_block['future_volume_multiple']:.1f}×")
    if pa.get("spot_vs_vwap"):
        parts.append(f"spot {pa['spot_vs_vwap']} VWAP")
    tail = " · ".join(parts[:3]) if parts else "monitor next bar"
    return f"{lead} — {tail}"
