"""Option long entry confluence checks (CE / PE) for index analysis."""

from __future__ import annotations

from typing import Any

from intraday_engine.analysis.day_bias import probable_day_bias
from intraday_engine.analysis.hourly_levels import NEAR_LEVEL_PCT, spot_near_level

OI_CANDLE_THRESHOLD = 0.5
PREM_CANDLE_THRESHOLD = 0.05
OI_SESSION_THRESHOLD = 2.0
PREM_SESSION_THRESHOLD = 1.0
FUT_STRENGTH_THRESHOLD = 0.03
DIRECTION_MIN_CHECKS = 7

CHECK_DEFS: list[dict[str, str]] = [
    {"id": "spot_open_vwap", "label": "Spot vs open + VWAP"},
    {"id": "futures_vs_spot", "label": "Futures vs spot"},
    {"id": "atm_premium", "label": "ATM premium (session)"},
    {"id": "options_flow", "label": "Options flow (OI + prem)"},
    {"id": "call_activity", "label": "Call activity"},
    {"id": "put_activity", "label": "Put activity"},
    {"id": "flow_favours", "label": "Flow favours"},
    {"id": "day_bias", "label": "Day bias"},
    {"id": "at_support_1h", "label": "At support (1H)"},
    {"id": "at_resistance_1h", "label": "At resistance (1H)"},
]


def _classify_derivative_activity(oi_chg_candle: float | None, prem_chg: float | None) -> str:
    if oi_chg_candle is None or prem_chg is None:
        return "—"
    oi_up = oi_chg_candle > OI_CANDLE_THRESHOLD
    oi_down = oi_chg_candle < -OI_CANDLE_THRESHOLD
    prem_up = prem_chg > PREM_CANDLE_THRESHOLD
    prem_down = prem_chg < -PREM_CANDLE_THRESHOLD
    if oi_up and prem_up:
        return "fresh_buying"
    if oi_up and prem_down:
        return "writing"
    if oi_down and prem_up:
        return "covering"
    if oi_down and prem_down:
        return "unwinding"
    if prem_up:
        return "premium_up"
    if prem_down:
        return "premium_down"
    return "neutral"


def _activity_label(activity: str) -> str:
    return {
        "fresh_buying": "Fresh buying",
        "writing": "Writing",
        "covering": "Covering",
        "unwinding": "Unwinding",
        "premium_up": "Premium ↑",
        "premium_down": "Premium ↓",
        "neutral": "Neutral",
        "—": "—",
    }.get(activity, activity)


def _session_features(summary: dict[str, Any], snapshots_frame: list[dict] | None) -> dict[str, Any]:
    pa = summary.get("price_action") or {}
    opt = summary.get("options") or {}
    oi = summary.get("oi") or {}
    fut = summary.get("futures") or {}

    features = {
        "spot_above_open": 1.0 if pa.get("spot_vs_open") == "above" else 0.0,
        "spot_below_open": 1.0 if pa.get("spot_vs_open") == "below" else 0.0,
        "spot_above_vwap": 1.0 if pa.get("spot_vs_vwap") == "above" else 0.0,
        "spot_below_vwap": 1.0 if pa.get("spot_vs_vwap") == "below" else 0.0,
        "fut_strength_pct": fut.get("vs_spot_pct", 0) or 0,
        "call_change_pct": opt.get("call_change_pct", 0) or 0,
        "put_change_pct": opt.get("put_change_pct", 0) or 0,
        "options_available": bool(opt.get("call_ltp") or opt.get("put_ltp")),
        "oi_available": bool(oi.get("call_oi") or oi.get("put_oi")),
        "call_oi_change_pct": oi.get("call_oi_change_pct", 0) or 0,
        "put_oi_change_pct": oi.get("put_oi_change_pct", 0) or 0,
        "fut_oi_available": False,
        "fut_oi_bullish": 0.0,
        "fut_oi_bearish": 0.0,
    }
    if snapshots_frame:
        from intraday_engine.features.feature_engineering import compute_features
        import pandas as pd

        features = compute_features(pd.DataFrame(snapshots_frame))
    return features


def _day_bias_for_summary(summary: dict[str, Any], snapshots_frame: list[dict] | None = None) -> str:
    return probable_day_bias(_session_features(summary, snapshots_frame))


def _interpret_session_flow(
    ce_oi_sess: float,
    pe_oi_sess: float,
    ce_prem_sess: float,
    pe_prem_sess: float,
) -> tuple[str, bool, bool]:
    """
    Combine session OI + premium — raw PE OI build on rallies is often put writing (bullish CE).
    Returns (label, ce_favour, pe_favour).
    """
    ce_buying = ce_oi_sess > OI_SESSION_THRESHOLD and ce_prem_sess > PREM_SESSION_THRESHOLD
    ce_writing = ce_oi_sess > OI_SESSION_THRESHOLD and ce_prem_sess < -PREM_SESSION_THRESHOLD
    pe_buying = pe_oi_sess > OI_SESSION_THRESHOLD and pe_prem_sess > PREM_SESSION_THRESHOLD
    pe_writing = pe_oi_sess > OI_SESSION_THRESHOLD and pe_prem_sess < -PREM_SESSION_THRESHOLD

    prem_bull = ce_prem_sess > PREM_SESSION_THRESHOLD and pe_prem_sess < -PREM_SESSION_THRESHOLD
    prem_bear = pe_prem_sess > PREM_SESSION_THRESHOLD and ce_prem_sess < -PREM_SESSION_THRESHOLD

    if prem_bull or pe_writing or ce_buying:
        label = "Bullish (CE)"
        if pe_writing and not prem_bull:
            label = "Put writing / CE prem ↑"
        return label, True, False
    if prem_bear or ce_writing or pe_buying:
        label = "Bearish (PE)"
        if ce_writing and not prem_bear:
            label = "Call writing / PE prem ↑"
        return label, False, True
    if ce_prem_sess > 0 and pe_prem_sess <= 0:
        return "Mild bullish", True, False
    if pe_prem_sess > 0 and ce_prem_sess <= 0:
        return "Mild bearish", False, True
    return "Mixed / flat", False, False


OI_FLOW_WEIGHT = 0.2
OI_FLOW_CAP = 40.0


def compute_directional_flow_metrics(
    ce_oi_sess: float,
    pe_oi_sess: float,
    call_sess: float,
    put_sess: float,
) -> dict[str, float]:
    """
    Premium-led flow indices for charts. OI adjusts only when paired with premium direction.
    Positive ce_flow_index = case building for CE longs; pe_flow_index for PE longs.
    flow_skew = ce_flow_index - pe_flow_index (positive → CE).
    """
    ce_idx = float(call_sess)
    pe_idx = float(put_sess)

    if pe_oi_sess > OI_SESSION_THRESHOLD and put_sess < -PREM_SESSION_THRESHOLD:
        bump = min(pe_oi_sess * OI_FLOW_WEIGHT, OI_FLOW_CAP)
        ce_idx += bump
        pe_idx -= bump * 0.5
    elif pe_oi_sess > OI_SESSION_THRESHOLD and put_sess > PREM_SESSION_THRESHOLD:
        bump = min(pe_oi_sess * OI_FLOW_WEIGHT, OI_FLOW_CAP)
        pe_idx += bump
        ce_idx -= bump * 0.5

    if ce_oi_sess > OI_SESSION_THRESHOLD and call_sess < -PREM_SESSION_THRESHOLD:
        bump = min(ce_oi_sess * OI_FLOW_WEIGHT, OI_FLOW_CAP)
        pe_idx += bump
        ce_idx -= bump * 0.5
    elif ce_oi_sess > OI_SESSION_THRESHOLD and call_sess > PREM_SESSION_THRESHOLD:
        bump = min(ce_oi_sess * OI_FLOW_WEIGHT, OI_FLOW_CAP)
        ce_idx += bump

    return {
        "ce_flow_index": round(ce_idx, 2),
        "pe_flow_index": round(pe_idx, 2),
        "flow_skew": round(ce_idx - pe_idx, 2),
        "raw_oi_skew_pp": round(ce_oi_sess - pe_oi_sess, 2),
        "premium_skew_pp": round(call_sess - put_sess, 2),
        "call_session_change_pct": round(call_sess, 2),
        "put_session_change_pct": round(put_sess, 2),
    }


def build_option_confluence(
    summary: dict[str, Any],
    hourly_levels: dict[str, Any],
    snapshots_frame: list[dict] | None = None,
) -> dict[str, Any]:
    pa = summary.get("price_action") or {}
    fut = summary.get("futures") or {}
    opt = summary.get("options") or {}
    oi = summary.get("oi") or {}

    spot = float(pa.get("spot") or 0)
    spot_above_open = pa.get("spot_vs_open") == "above"
    spot_above_vwap = pa.get("spot_vs_vwap") == "above"
    spot_below_open = pa.get("spot_vs_open") == "below"
    spot_below_vwap = pa.get("spot_vs_vwap") == "below"
    spot_bullish = spot_above_open and spot_above_vwap
    spot_bearish = spot_below_open and spot_below_vwap

    fut_strength = float(fut.get("vs_spot_pct") or 0)
    call_chg = float(opt.get("call_change_pct") or 0)
    put_chg = float(opt.get("put_change_pct") or 0)
    call_sess = float(opt.get("call_session_change_pct") or 0)
    put_sess = float(opt.get("put_session_change_pct") or 0)
    ce_oi_sess = float(oi.get("call_oi_change_pct") or 0)
    pe_oi_sess = float(oi.get("put_oi_change_pct") or 0)
    ce_oi_candle = oi.get("call_oi_change_candle_pct")
    pe_oi_candle = oi.get("put_oi_change_candle_pct")
    if ce_oi_candle is not None:
        ce_oi_candle = float(ce_oi_candle)
    if pe_oi_candle is not None:
        pe_oi_candle = float(pe_oi_candle)

    oi_skew_pp = round(ce_oi_sess - pe_oi_sess, 2)
    call_activity = _classify_derivative_activity(ce_oi_candle, call_chg)
    put_activity = _classify_derivative_activity(pe_oi_candle, put_chg)
    day_bias = _day_bias_for_summary(summary, snapshots_frame)
    flow_label, flow_ce, flow_pe = _interpret_session_flow(ce_oi_sess, pe_oi_sess, call_sess, put_sess)

    h_support = hourly_levels.get("support")
    h_resistance = hourly_levels.get("resistance")
    at_support = spot_near_level(spot, h_support)
    at_resistance = spot_near_level(spot, h_resistance)

    checks: dict[str, dict[str, Any]] = {}

    checks["spot_open_vwap"] = {
        "value": _spot_vwap_label(spot_above_open, spot_above_vwap, spot_below_open, spot_below_vwap),
        "ce": spot_bullish,
        "pe": spot_bearish,
    }

    fut_label = "Premium" if fut_strength > FUT_STRENGTH_THRESHOLD else (
        "Discount" if fut_strength < -FUT_STRENGTH_THRESHOLD else f"Flat ({fut_strength:+.2f}%)"
    )
    checks["futures_vs_spot"] = {
        "value": fut_label,
        "ce": fut_strength > FUT_STRENGTH_THRESHOLD,
        "pe": fut_strength < -FUT_STRENGTH_THRESHOLD,
    }

    prem_label = f"CE {call_sess:+.1f}% · PE {put_sess:+.1f}% (session)"
    checks["atm_premium"] = {
        "value": prem_label,
        "ce": call_sess > PREM_SESSION_THRESHOLD and put_sess < 0,
        "pe": put_sess > PREM_SESSION_THRESHOLD and call_sess < 0,
    }

    checks["options_flow"] = {
        "value": flow_label,
        "ce": flow_ce,
        "pe": flow_pe,
    }

    checks["call_activity"] = {
        "value": _activity_label(call_activity),
        "detail": call_activity,
        "ce": call_activity in ("fresh_buying", "covering") or call_sess > PREM_SESSION_THRESHOLD,
        "pe": call_activity == "writing" or (call_activity == "unwinding" and call_sess < -PREM_SESSION_THRESHOLD),
    }

    checks["put_activity"] = {
        "value": _activity_label(put_activity),
        "detail": put_activity,
        "ce": put_activity == "writing" or (put_activity in ("fresh_buying",) and put_sess < 0),
        "pe": put_activity in ("fresh_buying", "covering") or put_sess > PREM_SESSION_THRESHOLD,
    }

    checks["flow_favours"] = {
        "value": flow_label,
        "ce": flow_ce,
        "pe": flow_pe,
    }

    checks["day_bias"] = {
        "value": day_bias.replace("_DAY", ""),
        "ce": day_bias == "BULLISH_DAY",
        "pe": day_bias == "BEARISH_DAY",
    }

    sup_val = f"{h_support} ({'near' if at_support else 'away'})" if h_support else "—"
    checks["at_support_1h"] = {
        "value": sup_val,
        "ce": at_support and spot_bullish,
        "pe": False,
    }

    res_val = f"{h_resistance} ({'near' if at_resistance else 'away'})" if h_resistance else "—"
    checks["at_resistance_1h"] = {
        "value": res_val,
        "ce": False,
        "pe": at_resistance and spot_bearish,
    }

    ce_score = sum(1 for c in checks.values() if c.get("ce"))
    pe_score = sum(1 for c in checks.values() if c.get("pe"))

    if ce_score >= DIRECTION_MIN_CHECKS and ce_score > pe_score and spot_bullish:
        direction = "CE"
    elif pe_score >= DIRECTION_MIN_CHECKS and pe_score > ce_score and spot_bearish:
        direction = "PE"
    elif ce_score >= DIRECTION_MIN_CHECKS and pe_score >= DIRECTION_MIN_CHECKS:
        if spot_bullish and not spot_bearish:
            direction = "CE" if ce_score >= pe_score else "NEUTRAL"
        elif spot_bearish and not spot_bullish:
            direction = "PE" if pe_score >= ce_score else "NEUTRAL"
        else:
            direction = "CE" if ce_score >= pe_score else "PE"
    else:
        direction = "NEUTRAL"

    flow_metrics = compute_directional_flow_metrics(ce_oi_sess, pe_oi_sess, call_sess, put_sess)
    summary["flow"] = flow_metrics

    rows = []
    for spec in CHECK_DEFS:
        cid = spec["id"]
        c = checks[cid]
        rows.append({
            "id": cid,
            "label": spec["label"],
            "value": c.get("value", "—"),
            "ce": bool(c.get("ce")),
            "pe": bool(c.get("pe")),
        })

    return {
        "checks": rows,
        "ce_score": ce_score,
        "pe_score": pe_score,
        "total_checks": len(CHECK_DEFS),
        "min_for_direction": DIRECTION_MIN_CHECKS,
        "direction": direction,
        "oi_skew_pp": oi_skew_pp,
        "call_activity": call_activity,
        "put_activity": put_activity,
        "day_bias": day_bias,
        "options_flow": flow_label,
        "spot_aligned": "bullish" if spot_bullish else ("bearish" if spot_bearish else "mixed"),
        "hourly_support": h_support,
        "hourly_resistance": h_resistance,
        "near_level_pct": NEAR_LEVEL_PCT,
        "flow": flow_metrics,
    }


def _spot_vwap_label(above_o: bool, above_v: bool, below_o: bool, below_v: bool) -> str:
    if above_o and above_v:
        return "Above open & VWAP"
    if below_o and below_v:
        return "Below open & VWAP"
    parts = []
    parts.append("above open" if above_o else ("below open" if below_o else "flat vs open"))
    parts.append("above VWAP" if above_v else ("below VWAP" if below_v else "flat vs VWAP"))
    return " · ".join(parts)


def attach_confluence_to_summaries(
    summaries: list[dict[str, Any]],
    hourly_levels: dict[str, Any],
    snapshots_records: list[dict] | None = None,
) -> list[dict[str, Any]]:
    for i, summary in enumerate(summaries):
        frame = snapshots_records[: i + 1] if snapshots_records else None
        summary["confluence"] = build_option_confluence(summary, hourly_levels, frame)
    return summaries
