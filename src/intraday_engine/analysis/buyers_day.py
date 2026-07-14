"""
Buyer's Day Meter — decide, programmatically, whether today is a day where
buying options (long premium) has an edge, or a low-realized-vol / choppy /
theta-bleed day where buyers should stand aside or scalp only.

An option buyer is long gamma/vega and short theta: they need realized movement
to exceed premium decay. A "not a buyer's day" is measurable via a handful of
intraday factors, mostly readable within the first 30-45 minutes:

  1. range_pct        — session range vs spot (is there room to move?)
  2. efficiency_ratio — Kaufman ER (trend vs chop / whipsaw)
  3. vwap_crosses     — how often price flips across VWAP (mean reversion)
  4. adx              — Wilder ADX on the intraday candles (trend strength)
  5. straddle_decay   — session P&L of a long ATM straddle (theta bleed)
  6. straddle_momentum— is the ATM straddle expanding or contracting recently

Each factor votes -1 (hostile) / 0 (neutral) / +1 (favorable). The votes are
normalized to a 0-100 "buyer's day score" with a verdict + human reasons.

This is a de-risking FILTER, not a predictor: its best use is telling you early
to skip / reduce size on hostile days.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


# Score thresholds (favorable, hostile). Neutral is the band between them.
RANGE_PCT_GOOD = 0.9
RANGE_PCT_BAD = 0.5
ER_GOOD = 0.40
ER_BAD = 0.30
VWAP_CROSS_GOOD = 2          # <= good
VWAP_CROSS_BAD = 5           # >= bad
ADX_GOOD = 22.0
ADX_BAD = 18.0
STRADDLE_DECAY_GOOD = 0.0    # holding/expanding
STRADDLE_DECAY_BAD = -12.0   # bleeding hard
STRADDLE_MOM_GOOD = 1.0      # last-third expansion %
STRADDLE_MOM_BAD = -1.0

MIN_CANDLES = 3
ADX_PERIOD = 14


@dataclass
class Factor:
    key: str
    label: str
    value: float | None
    state: str          # good | neutral | bad | na
    vote: int           # +1 / 0 / -1
    hint: str


@dataclass
class BuyersDay:
    score: float                 # 0-100 (higher = better for buyers)
    verdict: str                 # BUYER_FRIENDLY | MIXED | AVOID
    headline: str
    candles: int
    factors: list[dict[str, Any]] = field(default_factory=list)
    reasons: list[str] = field(default_factory=list)
    info: dict[str, Any] = field(default_factory=dict)


def _f(v: Any, default: float = 0.0) -> float:
    try:
        if v is None:
            return default
        return float(v)
    except (TypeError, ValueError):
        return default


def _efficiency_ratio(closes: list[float]) -> float | None:
    """Kaufman Efficiency Ratio: |net move| / sum of absolute bar moves. 0..1."""
    if len(closes) < 2:
        return None
    net = abs(closes[-1] - closes[0])
    noise = sum(abs(closes[i] - closes[i - 1]) for i in range(1, len(closes)))
    if noise <= 0:
        return None
    return net / noise


def _vwap_crosses(spots: list[float], vwaps: list[float]) -> int | None:
    diffs = [s - v for s, v in zip(spots, vwaps) if v]
    diffs = [d for d in diffs if d != 0]
    if len(diffs) < 2:
        return None
    crosses = 0
    for i in range(1, len(diffs)):
        if (diffs[i] > 0) != (diffs[i - 1] > 0):
            crosses += 1
    return crosses


def _adx(highs: list[float], lows: list[float], closes: list[float], period: int = ADX_PERIOD) -> float | None:
    """Wilder's ADX. Returns None if not enough candles."""
    n = len(closes)
    if n < period + 1 or n < 4:
        # Fall back to a shorter period if we at least have a few candles.
        period = max(2, min(period, n - 1))
        if n < period + 1:
            return None
    plus_dm: list[float] = []
    minus_dm: list[float] = []
    tr: list[float] = []
    for i in range(1, n):
        up = highs[i] - highs[i - 1]
        down = lows[i - 1] - lows[i]
        plus_dm.append(up if (up > down and up > 0) else 0.0)
        minus_dm.append(down if (down > up and down > 0) else 0.0)
        tr.append(max(
            highs[i] - lows[i],
            abs(highs[i] - closes[i - 1]),
            abs(lows[i] - closes[i - 1]),
        ))
    if len(tr) < period:
        return None

    def _wilder_smooth(values: list[float]) -> list[float]:
        smoothed = [sum(values[:period])]
        for i in range(period, len(values)):
            smoothed.append(smoothed[-1] - smoothed[-1] / period + values[i])
        return smoothed

    tr_s = _wilder_smooth(tr)
    plus_s = _wilder_smooth(plus_dm)
    minus_s = _wilder_smooth(minus_dm)

    dx: list[float] = []
    for trv, pv, mv in zip(tr_s, plus_s, minus_s):
        if trv <= 0:
            continue
        plus_di = 100.0 * pv / trv
        minus_di = 100.0 * mv / trv
        denom = plus_di + minus_di
        if denom <= 0:
            continue
        dx.append(100.0 * abs(plus_di - minus_di) / denom)
    if not dx:
        return None
    # ADX = average of DX (use last `period` values when available).
    tail = dx[-period:] if len(dx) >= period else dx
    return sum(tail) / len(tail)


def _state_high_good(value: float, good: float, bad: float) -> tuple[str, int]:
    if value >= good:
        return "good", 1
    if value <= bad:
        return "bad", -1
    return "neutral", 0


def _state_low_good(value: float, good: float, bad: float) -> tuple[str, int]:
    if value <= good:
        return "good", 1
    if value >= bad:
        return "bad", -1
    return "neutral", 0


def compute_buyers_day(summaries: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute the Buyer's Day Meter from ordered per-candle analysis summaries."""
    summaries = [s for s in (summaries or []) if s]
    if len(summaries) < MIN_CANDLES:
        return asdict(BuyersDay(
            score=50.0,
            verdict="MIXED",
            headline="Not enough data yet — wait for a few candles.",
            candles=len(summaries),
            factors=[],
            reasons=["Need at least a few candles to assess the regime."],
            info={},
        ))

    highs = [_f(s.get("price_action", {}).get("high")) for s in summaries]
    lows = [_f(s.get("price_action", {}).get("low")) for s in summaries]
    closes = [_f(s.get("price_action", {}).get("spot")) for s in summaries]
    vwaps = [_f(s.get("price_action", {}).get("vwap")) for s in summaries]
    straddle = [
        _f(s.get("options", {}).get("call_ltp")) + _f(s.get("options", {}).get("put_ltp"))
        for s in summaries
    ]

    day_open = _f(summaries[-1].get("price_action", {}).get("open")) or closes[0]
    session_high = max(highs) if highs else 0.0
    session_low = min(lo for lo in lows if lo > 0) if any(lo > 0 for lo in lows) else 0.0
    spot_now = closes[-1]
    day_range = session_high - session_low if (session_high and session_low) else 0.0
    range_pct = (day_range / spot_now * 100.0) if spot_now else 0.0

    er = _efficiency_ratio([c for c in closes if c > 0])
    crosses = _vwap_crosses(closes, vwaps)
    adx = _adx(highs, lows, closes)

    straddle_open = next((v for v in straddle if v > 0), 0.0)
    straddle_now = straddle[-1] if straddle else 0.0
    straddle_decay = ((straddle_now / straddle_open - 1.0) * 100.0) if straddle_open > 0 else None
    # Recent momentum: last third of session vs its start.
    straddle_mom = None
    valid_straddle = [v for v in straddle if v > 0]
    if len(valid_straddle) >= 4:
        third = max(1, len(valid_straddle) // 3)
        recent_start = valid_straddle[-third - 1] if len(valid_straddle) > third else valid_straddle[0]
        if recent_start > 0:
            straddle_mom = (valid_straddle[-1] / recent_start - 1.0) * 100.0

    factors: list[Factor] = []

    # 1. Range %
    st, vote = _state_high_good(range_pct, RANGE_PCT_GOOD, RANGE_PCT_BAD)
    factors.append(Factor(
        "range_pct", "Session range", round(range_pct, 2), st, vote,
        f"{range_pct:.2f}% range — {'room to move' if vote > 0 else ('tight, low travel' if vote < 0 else 'moderate')}",
    ))

    # 2. Efficiency ratio
    if er is None:
        factors.append(Factor("efficiency_ratio", "Trend efficiency", None, "na", 0, "n/a"))
    else:
        st, vote = _state_high_good(er, ER_GOOD, ER_BAD)
        factors.append(Factor(
            "efficiency_ratio", "Trend efficiency", round(er, 2), st, vote,
            f"ER {er:.2f} — {'directional' if vote > 0 else ('choppy/whipsaw' if vote < 0 else 'mixed')}",
        ))

    # 3. VWAP crosses
    if crosses is None:
        factors.append(Factor("vwap_crosses", "VWAP crosses", None, "na", 0, "n/a"))
    else:
        st, vote = _state_low_good(crosses, VWAP_CROSS_GOOD, VWAP_CROSS_BAD)
        factors.append(Factor(
            "vwap_crosses", "VWAP crosses", crosses, st, vote,
            f"{crosses} crosses — {'one-sided' if vote > 0 else ('mean-reverting range' if vote < 0 else 'some two-way')}",
        ))

    # 4. ADX
    if adx is None:
        factors.append(Factor("adx", "ADX (trend strength)", None, "na", 0, "n/a"))
    else:
        st, vote = _state_high_good(adx, ADX_GOOD, ADX_BAD)
        factors.append(Factor(
            "adx", "ADX (trend strength)", round(adx, 1), st, vote,
            f"ADX {adx:.0f} — {'trending' if vote > 0 else ('no trend' if vote < 0 else 'building')}",
        ))

    # 5. Straddle decay (theta bleed)
    if straddle_decay is None:
        factors.append(Factor("straddle_decay", "ATM straddle P&L", None, "na", 0, "n/a"))
    else:
        st, vote = _state_high_good(straddle_decay, STRADDLE_DECAY_GOOD, STRADDLE_DECAY_BAD)
        factors.append(Factor(
            "straddle_decay", "ATM straddle P&L", round(straddle_decay, 1), st, vote,
            f"Long straddle {straddle_decay:+.1f}% — {'holding/expanding' if vote > 0 else ('theta bleeding both sides' if vote < 0 else 'flat')}",
        ))

    # 6. Straddle momentum (recent)
    if straddle_mom is None:
        factors.append(Factor("straddle_momentum", "Straddle momentum", None, "na", 0, "n/a"))
    else:
        st, vote = _state_high_good(straddle_mom, STRADDLE_MOM_GOOD, STRADDLE_MOM_BAD)
        factors.append(Factor(
            "straddle_momentum", "Straddle momentum", round(straddle_mom, 1), st, vote,
            f"Recent {straddle_mom:+.1f}% — {'premium expanding' if vote > 0 else ('premium contracting' if vote < 0 else 'flat')}",
        ))

    scored = [f for f in factors if f.state != "na"]
    n = len(scored)
    if n == 0:
        score = 50.0
    else:
        vote_sum = sum(f.vote for f in scored)
        score = (vote_sum + n) / (2 * n) * 100.0

    if score >= 60:
        verdict = "BUYER_FRIENDLY"
        headline = "Option-buyer friendly — movement is beating decay."
    elif score >= 40:
        verdict = "MIXED"
        headline = "Mixed — be selective, scalp with tight stops."
    else:
        verdict = "AVOID"
        headline = "Not an option-buyer's day — chop/theta. Stand aside or cut size."

    reasons = [f.hint for f in scored if f.vote < 0] or [
        f.hint for f in scored if f.vote > 0
    ][:2] or ["Balanced conditions."]

    info = {
        "spot": round(spot_now, 2),
        "day_open": round(day_open, 2),
        "session_high": round(session_high, 2),
        "session_low": round(session_low, 2),
        "day_range": round(day_range, 2),
        "straddle_open": round(straddle_open, 2),
        "straddle_now": round(straddle_now, 2),
        "expected_move_pts": round(straddle_open, 2),
        "range_capture": round(day_range / straddle_open, 2) if straddle_open > 0 else None,
    }

    return asdict(BuyersDay(
        score=round(score, 1),
        verdict=verdict,
        headline=headline,
        candles=len(summaries),
        factors=[asdict(f) for f in factors],
        reasons=reasons,
        info=info,
    ))


def _hhmm(ts: str) -> str:
    s = str(ts)
    if len(s) >= 16 and ("T" in s or " " in s):
        return s[11:16]
    return s


def compute_buyers_day_series(summaries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Progressive Buyer's Day score, evaluated cumulatively at each 5-min candle
    (score "as of" that time using all bars up to and including it). Lets the UI
    plot how the regime built up / decayed through the session.
    """
    summaries = [s for s in (summaries or []) if s]
    series: list[dict[str, Any]] = []
    for i in range(len(summaries)):
        if i + 1 < MIN_CANDLES:
            continue
        r = compute_buyers_day(summaries[: i + 1])
        ts = str(summaries[i].get("timestamp", ""))
        series.append({
            "timestamp": ts,
            "time": _hhmm(ts),
            "score": r["score"],
            "verdict": r["verdict"],
        })
    return series
