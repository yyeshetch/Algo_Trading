"""
ATR Squeeze — index intraday volatility contraction → release → breakout/breakdown.

Uses Wilder ATR on 5-min spot candles from analysis summaries:
  1. SQUEEZE  — short ATR / baseline ATR ratio below threshold (volatility compressing)
  2. ARMED    — squeeze + narrow range box (coil inside the compression)
  3. FIRED    — ATR expands and price breaks the squeeze box (breakout / breakdown)

Distinct from Trend Ignition (premium + VWAP coil); this is pure ATR/range mechanics.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

ATR_PERIOD = 14
SHORT_WINDOW = 5
BASELINE_WINDOW = 20
COIL_WINDOW = 8
SQUEEZE_RATIO = 0.82       # short ATR / baseline ATR → compression
ARMED_RATIO = 0.88         # still tight but building energy
RELEASE_RATIO = 0.94       # ATR expanding off the squeeze
MIN_SQUEEZE_BARS = 3       # consecutive squeeze bars before arming
BOX_ATR_MULT = 2.0        # coil width <= this × ATR → narrow box
BREAK_BUFFER_PCT = 0.01    # break must clear box by this % of spot
MIN_BREAK_MOVE_PCT = 0.04  # minimum spot % move on release bar
EXPANSION_MULT = 1.35      # release bar TR vs recent avg TR

MIN_CANDLES = ATR_PERIOD + 2


@dataclass
class AtrSqueezeState:
    state: str                 # IDLE | SQUEEZE | ARMED | BREAKOUT | BREAKDOWN
    side: str | None           # CE | PE | None
    headline: str
    score: float               # 0-100 squeeze tightness (higher = tighter coil)
    candles: int
    atr: float | None = None
    atr_ratio: float | None = None
    squeeze_high: float | None = None
    squeeze_low: float | None = None
    box_width: float | None = None
    factors: list[dict[str, Any]] = field(default_factory=list)
    triggers: list[dict[str, Any]] = field(default_factory=list)
    reasons: list[str] = field(default_factory=list)


def _f(v: Any, default: float = 0.0) -> float:
    try:
        if v is None:
            return default
        return float(v)
    except (TypeError, ValueError):
        return default


def _ohlc(summaries: list[dict[str, Any]]) -> tuple[list[float], list[float], list[float], list[str]]:
    highs, lows, closes, timestamps = [], [], [], []
    for s in summaries:
        pa = s.get("price_action") or {}
        highs.append(_f(pa.get("high")))
        lows.append(_f(pa.get("low")))
        closes.append(_f(pa.get("spot")))
        timestamps.append(str(s.get("timestamp", "")))
    return highs, lows, closes, timestamps


def _true_ranges(highs: list[float], lows: list[float], closes: list[float]) -> list[float]:
    trs: list[float] = []
    for i in range(len(closes)):
        if i == 0:
            trs.append(max(highs[i] - lows[i], 0.0))
        else:
            trs.append(max(
                highs[i] - lows[i],
                abs(highs[i] - closes[i - 1]),
                abs(lows[i] - closes[i - 1]),
            ))
    return trs


def _wilder_atr(trs: list[float], period: int = ATR_PERIOD) -> list[float | None]:
    out: list[float | None] = [None] * len(trs)
    if len(trs) < period:
        return out
    seed = sum(trs[:period]) / period
    out[period - 1] = seed
    prev = seed
    for i in range(period, len(trs)):
        prev = (prev * (period - 1) + trs[i]) / period
        out[i] = prev
    return out


def _avg(values: list[float | None], end: int, window: int) -> float | None:
    start = max(0, end - window + 1)
    chunk = [v for v in values[start : end + 1] if v is not None]
    if not chunk:
        return None
    return sum(chunk) / len(chunk)


def _squeeze_box(
    highs: list[float],
    lows: list[float],
    idx: int,
    window: int = COIL_WINDOW,
    *,
    include_current: bool = True,
) -> tuple[float, float, float]:
    end = idx + 1 if include_current else idx
    start = max(0, end - window)
    h = highs[start:end]
    lo = [x for x in lows[start:end] if x > 0]
    if not h or not lo:
        spot_h = highs[idx] if idx < len(highs) else 0
        spot_l = lows[idx] if idx < len(lows) else 0
        return spot_h, spot_l, max(spot_h - spot_l, 0.0)
    box_h = max(h)
    box_l = min(lo)
    return box_h, box_l, box_h - box_l


def _consecutive_squeeze(atr_ratios: list[float | None], idx: int) -> int:
    count = 0
    for j in range(idx, -1, -1):
        r = atr_ratios[j]
        if r is not None and r < SQUEEZE_RATIO:
            count += 1
        else:
            break
    return count


def _evaluate(
    idx: int,
    summaries: list[dict[str, Any]],
    *,
    trs: list[float],
    atrs: list[float | None],
    atr_ratios: list[float | None],
    highs: list[float],
    lows: list[float],
    closes: list[float],
    was_squeeze: bool,
) -> dict[str, Any]:
    pa = summaries[idx].get("price_action") or {}
    ts = summaries[idx].get("timestamp")
    spot = _f(pa.get("spot"))
    bar_high = _f(pa.get("high"))
    bar_low = _f(pa.get("low"))
    chg = _f(pa.get("spot_change_pct"))
    atr = atrs[idx]
    ratio = atr_ratios[idx]
    prev_ratio = atr_ratios[idx - 1] if idx > 0 else None

    box_h, box_l, box_w = _squeeze_box(highs, lows, idx, include_current=False)
    prior_box_h, prior_box_l, _ = box_h, box_l, box_w
    display_h, display_l, display_w = _squeeze_box(highs, lows, idx, include_current=True)
    sq_bars = _consecutive_squeeze(atr_ratios, idx)

    lo_tr = max(0, idx - SHORT_WINDOW)
    avg_tr = sum(trs[lo_tr:idx]) / max(len(trs[lo_tr:idx]), 1) if idx > 0 else trs[idx]
    tr_now = trs[idx]
    tr_expanding = avg_tr > 0 and tr_now >= avg_tr * EXPANSION_MULT

    state = "IDLE"
    side: str | None = None
    reasons: list[str] = []
    squeeze_score = 50.0

    if ratio is not None:
        squeeze_score = max(0.0, min(100.0, round((1.0 - min(ratio, 1.0)) * 100, 1)))

    in_squeeze = ratio is not None and ratio < SQUEEZE_RATIO
    near_squeeze = ratio is not None and ratio < ARMED_RATIO
    narrow_box = atr and atr > 0 and box_w <= atr * BOX_ATR_MULT
    releasing = (
        ratio is not None
        and prev_ratio is not None
        and ratio >= RELEASE_RATIO
        and prev_ratio < RELEASE_RATIO
        and was_squeeze
    )

    break_up = bar_high > prior_box_h * (1 + BREAK_BUFFER_PCT / 100) if prior_box_h else False
    break_dn = bar_low < prior_box_l * (1 - BREAK_BUFFER_PCT / 100) if prior_box_l else False
    big_move = abs(chg) >= MIN_BREAK_MOVE_PCT
    had_squeeze = was_squeeze or sq_bars >= 2 or in_squeeze or near_squeeze

    if in_squeeze and sq_bars >= MIN_SQUEEZE_BARS and narrow_box:
        state = "ARMED"
        reasons.append(
            f"ATR squeeze armed — ratio {ratio:.2f}, box {prior_box_l:.0f}–{prior_box_h:.0f} ({sq_bars} tight bars)"
        )
    elif in_squeeze or near_squeeze:
        state = "SQUEEZE"
        if ratio is not None:
            reasons.append(f"ATR contracting — short/baseline ratio {ratio:.2f}")

    fired = False
    if had_squeeze and tr_expanding and big_move:
        if break_up and chg > 0:
            state = "BREAKOUT"
            side = "CE"
            fired = True
            reasons.append(
                f"Breakout — spot {spot:.0f} cleared squeeze high {prior_box_h:.0f} on ATR release"
            )
        elif break_dn and chg < 0:
            state = "BREAKDOWN"
            side = "PE"
            fired = True
            reasons.append(
                f"Breakdown — spot {spot:.0f} broke squeeze low {prior_box_l:.0f} on ATR release"
            )
    elif releasing and not fired:
        state = "ARMED" if narrow_box else "SQUEEZE"
        reasons.append(f"ATR releasing — ratio crossed {RELEASE_RATIO:.2f}; watch box break")

    return {
        "timestamp": ts,
        "state": state,
        "side": side,
        "spot": round(spot, 2) if spot else None,
        "atr": round(atr, 2) if atr else None,
        "atr_ratio": round(ratio, 3) if ratio is not None else None,
        "squeeze_high": round(prior_box_h, 2),
        "squeeze_low": round(prior_box_l, 2),
        "box_width": round(box_w, 2),
        "display_high": round(display_h, 2),
        "display_low": round(display_l, 2),
        "squeeze_bars": sq_bars,
        "squeeze_score": squeeze_score,
        "spot_change_pct": chg,
        "tr_expanding": tr_expanding,
        "fired": fired,
        "reasons": reasons,
    }


def _build_series(
    summaries: list[dict[str, Any]],
) -> tuple[list[float], list[float], list[float], list[float], list[float | None], list[float | None]]:
    highs, lows, closes, _ = _ohlc(summaries)
    trs = _true_ranges(highs, lows, closes)
    atrs = _wilder_atr(trs, ATR_PERIOD)
    atr_ratios: list[float | None] = []
    for i in range(len(summaries)):
        short = _avg(atrs, i, SHORT_WINDOW)
        base = _avg(atrs, i, BASELINE_WINDOW)
        if short and base and base > 0:
            atr_ratios.append(round(short / base, 4))
        else:
            atr_ratios.append(None)
    return highs, lows, closes, trs, atrs, atr_ratios


def atr_squeeze_tape(
    summaries: list[dict[str, Any]],
    *,
    prior_summaries: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    combined = [s for s in (prior_summaries or []) if s] + [s for s in (summaries or []) if s]
    if len(combined) < MIN_CANDLES:
        return []
    n_prior = len([s for s in (prior_summaries or []) if s])
    highs, lows, closes, trs, atrs, atr_ratios = _build_series(combined)
    tape: list[dict[str, Any]] = []
    was_squeeze = False
    for idx in range(len(combined)):
        row = _evaluate(
            idx, combined,
            trs=trs, atrs=atrs, atr_ratios=atr_ratios,
            highs=highs, lows=lows, closes=closes,
            was_squeeze=was_squeeze,
        )
        tape.append(row)
        r = row.get("atr_ratio")
        was_squeeze = r is not None and r < ARMED_RATIO
    if n_prior <= 0:
        return tape
    return tape[n_prior:]


def compute_atr_squeeze(
    summaries: list[dict[str, Any]],
    *,
    prior_summaries: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    today = [s for s in (summaries or []) if s]
    prior = [s for s in (prior_summaries or []) if s]
    combined = prior + today
    if len(combined) < MIN_CANDLES:
        return asdict(AtrSqueezeState(
            state="IDLE",
            side=None,
            headline="Not enough candles — need ~15+ bars for ATR(14).",
            score=50.0,
            candles=len(today),
        ))

    tape = atr_squeeze_tape(today, prior_summaries=prior)
    for i, s in enumerate(today):
        if i < len(tape):
            s["atr_squeeze"] = tape[i]

    last = tape[-1] if tape else {}
    triggers = [t for t in tape if t.get("fired")]
    latest_trigger = triggers[-1] if triggers else None

    state = last.get("state", "IDLE")
    side = last.get("side")
    atr_ratio = last.get("atr_ratio")
    score = last.get("squeeze_score", 50.0)

    if latest_trigger:
        headline = (
            f"{latest_trigger['state']} @ {latest_trigger.get('squeeze_high')} / "
            f"{latest_trigger.get('squeeze_low')} — ATR ratio {latest_trigger.get('atr_ratio')}"
        )
    elif state == "ARMED":
        headline = f"Squeeze armed — box {last.get('squeeze_low')}–{last.get('squeeze_high')}, ATR ratio {atr_ratio}"
    elif state == "SQUEEZE":
        headline = f"Volatility compressing — ATR ratio {atr_ratio} (< {SQUEEZE_RATIO})"
    else:
        headline = "No active ATR squeeze — volatility normal"

    factors = []
    if atr_ratio is not None:
        if atr_ratio < SQUEEZE_RATIO:
            st, vote = "bullish" if state == "BREAKOUT" else "neutral", 0
            if state == "ARMED":
                st = "neutral"
            factors.append({
                "key": "atr_ratio", "label": "ATR ratio", "value": f"{atr_ratio:.2f}",
                "state": "bearish" if atr_ratio < SQUEEZE_RATIO and state not in ("BREAKOUT", "BREAKDOWN") else st,
                "vote": 1 if state == "BREAKOUT" else (-1 if state == "BREAKDOWN" else 0),
                "hint": f"Short {SHORT_WINDOW} / baseline {BASELINE_WINDOW} ATR",
            })
        else:
            factors.append({
                "key": "atr_ratio", "label": "ATR ratio", "value": f"{atr_ratio:.2f}",
                "state": "neutral", "vote": 0,
                "hint": "≥ squeeze threshold — expansion/normal",
            })

    box_w = last.get("box_width")
    atr_v = last.get("atr")
    if box_w and atr_v:
        tight = box_w <= atr_v * BOX_ATR_MULT
        factors.append({
            "key": "box", "label": "Squeeze box", "value": f"{last.get('squeeze_low')}–{last.get('squeeze_high')}",
            "state": "neutral" if tight else "neutral",
            "vote": 0,
            "hint": f"Width {box_w:.0f} vs ATR {atr_v:.0f} ({'narrow' if tight else 'wide'})",
        })

    factors.append({
        "key": "state", "label": "State", "value": state,
        "state": "bullish" if state == "BREAKOUT" else ("bearish" if state == "BREAKDOWN" else "neutral"),
        "vote": 1 if state == "BREAKOUT" else (-1 if state == "BREAKDOWN" else 0),
        "hint": f"{len(triggers)} release(s) today",
    })

    series = [
        {
            "timestamp": t.get("timestamp"),
            "state": t.get("state"),
            "atr_ratio": t.get("atr_ratio"),
            "squeeze_score": t.get("squeeze_score"),
            "side": t.get("side"),
        }
        for t in tape
    ]

    result = AtrSqueezeState(
        state=state,
        side=side,
        headline=headline,
        score=score,
        candles=len(today),
        atr=last.get("atr"),
        atr_ratio=atr_ratio,
        squeeze_high=last.get("squeeze_high"),
        squeeze_low=last.get("squeeze_low"),
        box_width=last.get("box_width"),
        factors=factors,
        triggers=[
            {
                "timestamp": t.get("timestamp"),
                "state": t.get("state"),
                "side": t.get("side"),
                "spot": t.get("spot"),
                "atr_ratio": t.get("atr_ratio"),
                "squeeze_high": t.get("squeeze_high"),
                "squeeze_low": t.get("squeeze_low"),
                "reasons": t.get("reasons", []),
            }
            for t in triggers
        ],
        reasons=last.get("reasons") or [],
    )
    out = asdict(result)
    out["series"] = series
    out["tape"] = tape
    out["latest"] = last
    out["latest_trigger"] = latest_trigger
    if prior:
        out["warmup_bars"] = len(prior)
    return out
