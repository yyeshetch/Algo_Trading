"""
Candle wick / pinbar / climax morphology — shared by Pro Print and PAV insights.
"""

from __future__ import annotations

from typing import Any

HUGE_WICK_RANGE_RATIO = 0.30
DUAL_WICK_RANGE_RATIO = 0.30
MIN_WICK_BODY_RATIO = 2.0
PINBAR_BODY_RANGE_MAX = 0.35
PINBAR_WICK_RANGE_MIN = 0.45
OPPOSITE_WICK_MAX_RATIO = 0.35
EXTENDED_RANGE_MULT = 1.5


def _f(v: Any, default: float = 0.0) -> float:
    try:
        if v is None:
            return default
        return float(v)
    except (TypeError, ValueError):
        return default


def wick_metrics_from_ohlc(open_: float, high: float, low: float, close: float) -> dict[str, float]:
    rng = high - low
    if rng <= 0:
        return {
            "range": 0.0,
            "upper_ratio": 0.0,
            "lower_ratio": 0.0,
            "body_ratio": 0.0,
            "upper": 0.0,
            "lower": 0.0,
            "body": 0.0,
        }
    body = abs(close - open_)
    upper = max(0.0, high - max(open_, close))
    lower = max(0.0, min(open_, close) - low)
    return {
        "range": rng,
        "upper_ratio": upper / rng,
        "lower_ratio": lower / rng,
        "body_ratio": body / rng,
        "upper": upper,
        "lower": lower,
        "body": body,
    }


def wick_metrics_spot(pa: dict[str, Any]) -> dict[str, float]:
    high = _f(pa.get("high"))
    low = _f(pa.get("low"))
    close = _f(pa.get("spot"))
    body_amt = _f(pa.get("candle_body"))
    open_ = close - body_amt if close else _f(pa.get("open"))
    rng = _f(pa.get("candle_range")) or max(high - low, 0.0)
    if rng <= 0:
        return wick_metrics_from_ohlc(open_, high, low, close)
    return wick_metrics_from_ohlc(open_, high, low, close)


def wick_metrics_premium(
    opt: dict[str, Any],
    side: str,
    prev_opt: dict[str, Any] | None = None,
) -> dict[str, float]:
    side = side.upper()
    if side == "CE":
        high = _f(opt.get("call_high"))
        low = _f(opt.get("call_low"))
        close = _f(opt.get("call_ltp"))
        chg = _f(opt.get("call_change_pct"))
        prev_close = _f((prev_opt or {}).get("call_ltp"))
    else:
        high = _f(opt.get("put_high"))
        low = _f(opt.get("put_low"))
        close = _f(opt.get("put_ltp"))
        chg = _f(opt.get("put_change_pct"))
        prev_close = _f((prev_opt or {}).get("put_ltp"))

    if close <= 0 or high <= low:
        return wick_metrics_from_ohlc(0, 0, 0, 0)

    if chg != 0:
        open_ = close / (1 + chg / 100.0)
    elif prev_close > 0:
        open_ = prev_close
    else:
        open_ = close
    return wick_metrics_from_ohlc(open_, high, low, close)


def _is_pinbar_bull(m: dict[str, float]) -> bool:
    if m["range"] <= 0 or m["body_ratio"] > PINBAR_BODY_RANGE_MAX:
        return False
    return (
        m["lower_ratio"] >= PINBAR_WICK_RANGE_MIN
        and m["upper_ratio"] <= OPPOSITE_WICK_MAX_RATIO
        and m["lower"] >= MIN_WICK_BODY_RATIO * max(m["body"], 1e-9)
    )


def _is_pinbar_bear(m: dict[str, float]) -> bool:
    if m["range"] <= 0 or m["body_ratio"] > PINBAR_BODY_RANGE_MAX:
        return False
    return (
        m["upper_ratio"] >= PINBAR_WICK_RANGE_MIN
        and m["lower_ratio"] <= OPPOSITE_WICK_MAX_RATIO
        and m["upper"] >= MIN_WICK_BODY_RATIO * max(m["body"], 1e-9)
    )


def _implications_upper_wick() -> list[str]:
    return ["Resistance formation", "Reversal expected", "Liquidity sweep possible"]


def _implications_lower_wick() -> list[str]:
    return ["Support formation", "Reversal expected", "Liquidity sweep possible"]


def _premium_wick_pct(m: dict[str, float], wick: str) -> float | None:
    if m["range"] <= 0:
        return None
    return round(m[f"{wick}_ratio"] * 100, 1)


def scan_premium_wicks(
    ce_prem: dict[str, float],
    pe_prem: dict[str, float],
) -> list[dict[str, Any]]:
    """
    Check ATM CE and PE premium bars for huge wicks (both directions).

    Upper wick → that option faces resistance.
    Lower wick → that option faces support.
    """
    signals: list[dict[str, Any]] = []
    for side, prem in (("CE", ce_prem), ("PE", pe_prem)):
        if prem["range"] <= 0:
            continue
        upper_pct = prem["upper_ratio"] * 100
        lower_pct = prem["lower_ratio"] * 100
        if prem["upper_ratio"] >= HUGE_WICK_RANGE_RATIO:
            signals.append({
                "side": side,
                "wick": "upper",
                "level": "resistance",
                "pct": round(upper_pct, 1),
                "text": f"{side} premium upper wick {upper_pct:.0f}% — {side} facing resistance",
                "blocks_pro_print": True,
                "expect_dir": -1 if side == "CE" else 1,
            })
        if prem["lower_ratio"] >= HUGE_WICK_RANGE_RATIO:
            signals.append({
                "side": side,
                "wick": "lower",
                "level": "support",
                "pct": round(lower_pct, 1),
                "text": f"{side} premium lower wick {lower_pct:.0f}% — {side} facing support",
                "blocks_pro_print": False,
                "expect_dir": 1 if side == "CE" else -1,
            })
    return signals


def analyze_candle_morphology(
    pa: dict[str, Any],
    options: dict[str, Any] | None = None,
    *,
    prev_options: dict[str, Any] | None = None,
    price_dir: int = 0,
    range_mult: float | None = None,
    vol_mult: float | None = None,
    candle_type: str | None = None,
) -> dict[str, Any]:
    """
    Classify spot + ATM premium morphology.

    Returns pattern, climax/pinbar flags, pro-print blockers, and implication text.
    """
    spot = wick_metrics_spot(pa)
    opt = options or {}
    ce_prem = wick_metrics_premium(opt, "CE", prev_options)
    pe_prem = wick_metrics_premium(opt, "PE", prev_options)

    ctype = (candle_type or "").upper()
    extended = bool(range_mult and range_mult >= EXTENDED_RANGE_MULT)
    vol_climax = bool(vol_mult and vol_mult >= 2.5)

    pattern: str | None = None
    climax_side: str | None = None
    implications: list[str] = []
    blocks_pro_print = False
    block_reason: str | None = None
    premium_issue: str | None = None
    premium_signals: list[dict[str, Any]] = []

    ur, lr = spot["upper_ratio"], spot["lower_ratio"]

    if _is_pinbar_bull(spot) or ctype == "HAMMER":
        pattern = "PINBAR_BULL"
        # Lower-wick rejection — PE-side support, never CE climax
        implications = _implications_lower_wick()
        blocks_pro_print = True
        block_reason = "Bullish pinbar / hammer — lower wick rejection (PE support), not pro print"
    elif _is_pinbar_bear(spot) or ctype == "SHOOTING_STAR":
        pattern = "PINBAR_BEAR"
        # Upper-wick rejection — CE-side exhaustion at highs, not a clean pro print
        implications = _implications_upper_wick()
        blocks_pro_print = True
        block_reason = "Bearish pinbar / shooting star — upper wick rejection (CE exhaustion), not pro print"
    elif ctype == "DOJI":
        pattern = "DOJI"
        implications = ["Indecision", "Wait for confirmation"]
        blocks_pro_print = True
        block_reason = "Doji — indecision, not pro print"

    huge_upper = ur >= HUGE_WICK_RANGE_RATIO
    huge_lower = lr >= HUGE_WICK_RANGE_RATIO
    dual_wick = ur >= DUAL_WICK_RANGE_RATIO and lr >= DUAL_WICK_RANGE_RATIO

    # Climax = wick-dominant rejection only (top wick → CE, bottom wick → PE)
    if not pattern and (huge_upper or huge_lower or dual_wick or (extended and (ur >= 0.22 or lr >= 0.22))):
        if dual_wick or (huge_upper and huge_lower):
            pattern = "CLIMAX_CHURN"
            climax_side = "CE" if ur >= lr else "PE"
            implications = ["Climax / churn", "Exhaustion likely", "Fade or wait for confirmation"]
        elif huge_upper and ur >= lr:
            pattern = "CLIMAX_CE"
            climax_side = "CE"
            implications = _implications_upper_wick()
        elif huge_lower and lr >= ur:
            pattern = "CLIMAX_PE"
            climax_side = "PE"
            implications = _implications_lower_wick()
        elif extended and vol_climax:
            pattern = "CLIMAX_EXTENDED"
            climax_side = "CE" if ur > lr else ("PE" if lr > ur else None)
            implications = ["Extended range + volume climax", "Exhaustion risk", "Reversal possible"]

        if pattern:
            blocks_pro_print = True
            dominant = "upper" if (ur >= lr and ur >= HUGE_WICK_RANGE_RATIO) else "lower"
            wick_pct = (ur if dominant == "upper" else lr) * 100
            block_reason = (
                f"Climax candle ({climax_side or '—'}) — "
                f"{dominant} wick {wick_pct:.0f}% of range"
            )

    # Premium wicks — both CE and PE, upper = resistance, lower = support
    premium_signals = scan_premium_wicks(ce_prem, pe_prem)
    resistance_signals = [s for s in premium_signals if s["level"] == "resistance"]

    if resistance_signals and not pattern:
        primary = resistance_signals[0]
        side = primary["side"]
        pattern = f"CLIMAX_{side}"
        climax_side = side
        implications = _implications_upper_wick()
        block_reason = primary["text"]

    if resistance_signals:
        premium_issue = resistance_signals[0]["text"]

    expect_dir = 0
    if pattern in ("CLIMAX_CE", "PINBAR_BEAR") and ur >= lr:
        expect_dir = -1
    elif pattern in ("CLIMAX_PE", "PINBAR_BULL") and lr >= ur:
        expect_dir = 1
    elif pattern == "CLIMAX_CHURN":
        expect_dir = -1 if ur >= lr else 1
    elif resistance_signals:
        expect_dir = resistance_signals[0]["expect_dir"]
    elif premium_signals:
        expect_dir = premium_signals[0]["expect_dir"]

    return {
        "pattern": pattern,
        "climax_side": climax_side,
        "climax_candle": pattern is not None and pattern.startswith("CLIMAX"),
        "pinbar": pattern in ("PINBAR_BULL", "PINBAR_BEAR"),
        "blocks_pro_print": blocks_pro_print,
        "block_reason": block_reason,
        "premium_issue": premium_issue,
        "premium_signals": premium_signals,
        "implications": implications,
        "expect_dir": expect_dir,
        "spot_upper_wick_pct": round(ur * 100, 1),
        "spot_lower_wick_pct": round(lr * 100, 1),
        "ce_premium_upper_wick_pct": _premium_wick_pct(ce_prem, "upper"),
        "ce_premium_lower_wick_pct": _premium_wick_pct(ce_prem, "lower"),
        "pe_premium_upper_wick_pct": _premium_wick_pct(pe_prem, "upper"),
        "pe_premium_lower_wick_pct": _premium_wick_pct(pe_prem, "lower"),
        "extended_range": extended,
        "vol_climax": vol_climax,
    }


def check_setup_confirmation(
    setup: dict[str, Any],
    pa: dict[str, Any],
) -> dict[str, Any] | None:
    """Return confirmation insight dict if current bar confirms a prior setup."""
    if not setup or not setup.get("pattern"):
        return None
    expect = int(setup.get("expect_dir") or 0)
    if expect == 0:
        return None

    spot_chg = _f(pa.get("spot_change_pct"))
    body_pct = _f(pa.get("candle_body_pct"))
    confirmed = False
    detail = ""

    if expect > 0:
        confirmed = spot_chg >= 0.04 or body_pct >= 0.06
        detail = "bullish follow-through"
    else:
        confirmed = spot_chg <= -0.04 or body_pct <= -0.06
        detail = "bearish follow-through"

    if not confirmed:
        return None

    pattern = setup.get("pattern", "")
    setup_ts = setup.get("timestamp", "")
    label = pattern.replace("_", " ").title()
    return {
        "category": "candle_setup",
        "tag": "confirmed",
        "severity": "bullish" if expect > 0 else "bearish",
        "text": (
            f"✓ Confirmed — {detail} after {label} "
            f"({', '.join(setup.get('implications') or [])[:80]})"
        ),
        "setup_timestamp": setup_ts,
        "pattern": pattern,
    }
