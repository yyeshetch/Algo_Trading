"""
Trend Ignition Radar — catch the *directional* intraday breakout/breakdown the
moment it ignites, on ANY day (not just expiry), and estimate how far it can run.

Motivation (NIFTY 08-Jul-2026): price coiled below a falling VWAP for ~2 hours,
then at 13:45 it sliced the consolidation floor and fell ~350-440 points into the
close. The ignition candle was unmistakable in the data we already store:

    13:45  spot -0.35%  (≈12x the trailing per-candle volatility)
           broke the 10-bar coil low
           ATM PE premium +44.5% / candle (fresh put buying)
           8/8 of the last candles below a declining VWAP

So the setup is programmatically identifiable. This detector formalizes it:

  REGIME   — price held one side of VWAP for a sustained run, VWAP sloping that
             way (weak = PE/down, strong = CE/up), while it coils in a tight range.
  IGNITION — a candle expands far beyond recent volatility (range expansion),
             breaks the coil extreme, and the directional option premium surges.
  ESTIMATE — size the expected move off the day's expected move (ATM straddle ≈
             1-day 1SD) plus the coil's measured move:
               SHORT  ~ coil height (scalp)          — chop / weak thrust
               MEDIUM ~ 1.0x expected move from open — clean break
               HUGE   ~ 1.4x expected move from open — trend day, full-throttle

Runs over the ordered per-candle analysis summaries, so it is fully back-testable
and never looks ahead (each candle is judged on data up to and including itself).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

# --- tunables (conservative; tuned against the 08-Jul-2026 breakdown) ---
ACTIVE_FROM_HHMM = "09:45"     # let VWAP/coil establish before hunting
VWAP_SIDE_STREAK = 5           # >= this many recent candles on one side of VWAP = biased
COIL_WINDOW = 10               # candles used to define the consolidation box
ATR_WINDOW = 10                # trailing window for per-candle volatility (abs spot %)
ATR_FLOOR_PCT = 0.03           # floor so a dead-quiet coil can't make expansion explode

MIN_MOVE_PCT = 0.15            # ignition candle must move at least this much (spot %)
EXPANSION_MULT = 4.0           # ignition move >= this x trailing volatility
PREM_SURGE_PCT = 20.0          # directional option premium jump on the ignition candle

# Continuation (adds after a confirmed ignition, same side): a lower bar.
CONT_EXPANSION_MULT = 2.0
CONT_PREM_SURGE_PCT = 12.0

# Move-size classification / targeting.
HUGE_EXP_MULT = 6.0            # strong thrust
HUGE_PREM_SURGE_PCT = 35.0
EM_MULT_MEDIUM = 1.0           # target ~1.0x expected move from open
EM_MULT_HUGE = 1.4            # target ~1.4x expected move from open


@dataclass
class TrendTrigger:
    timestamp: str
    stage: str                 # IGNITION | CONTINUATION
    side: str                  # CE | PE
    strike: int
    spot: float
    option_ltp: float
    prem_change_pct: float
    oi_change_pct: float
    expansion_x: float         # candle move / trailing volatility
    entry: float               # option LTP at signal
    stop_ltp: float            # option-premium SL (ignition candle option low)
    stop_spot: float           # underlying invalidation (coil reclaim)
    risk_pct: float
    move_class: str            # SHORT | MEDIUM | HUGE
    est_move_pts: float        # expected further move (pts) to the classified target
    target_spot: float         # classified target level on the underlying
    target_near: float         # near-term measured-move target
    confidence: float
    reasons: list[str] = field(default_factory=list)


@dataclass
class TrendIgnitionState:
    active: bool
    state: str                 # IDLE | WATCH | IGNITION | RUNNING
    side: str | None
    headline: str
    spot: float | None = None
    vwap: float | None = None
    atm_strike: int | None = None
    coil_low: float | None = None
    coil_high: float | None = None
    expected_move_pts: float | None = None
    day_open: float | None = None
    move_class: str | None = None
    est_move_pts: float | None = None
    target_spot: float | None = None
    entry: float | None = None
    stop_ltp: float | None = None
    stop_spot: float | None = None
    confidence: float = 0.0
    reasons: list[str] = field(default_factory=list)
    latest_trigger: dict[str, Any] | None = None
    triggers: list[dict[str, Any]] = field(default_factory=list)


def _f(v: Any, d: float = 0.0) -> float:
    try:
        return float(v) if v is not None else d
    except (TypeError, ValueError):
        return d


def _hhmm(ts: str) -> str:
    s = str(ts)
    if len(s) >= 16 and ("T" in s or " " in s):
        return s[11:16]
    return s


def _classify_move(exp_x: float, prem_surge: float, streak: int, streak_max: int) -> str:
    """Bucket the expected move by thrust strength + how entrenched the trend is."""
    strong_thrust = exp_x >= HUGE_EXP_MULT or prem_surge >= HUGE_PREM_SURGE_PCT
    full_trend = streak >= streak_max
    if strong_thrust and full_trend:
        return "HUGE"
    if exp_x >= EXPANSION_MULT and prem_surge >= PREM_SURGE_PCT and streak >= VWAP_SIDE_STREAK:
        return "MEDIUM"
    return "SHORT"


def _targets(side: str, spot: float, day_open: float, em_pts: float,
             coil_low: float, coil_high: float, move_class: str) -> tuple[float, float, float]:
    """Return (target_spot, near_term_target, est_move_pts) for the classified move."""
    coil_height = max(coil_high - coil_low, 0.0)
    if side == "PE":
        near = round((coil_low - coil_height), 2)                       # measured move down
        if move_class == "HUGE":
            tgt = day_open - EM_MULT_HUGE * em_pts
        elif move_class == "MEDIUM":
            tgt = day_open - EM_MULT_MEDIUM * em_pts
        else:
            tgt = near
        tgt = round(min(tgt, near), 2)
        est = round(max(spot - tgt, 0.0), 1)
    else:
        near = round((coil_high + coil_height), 2)
        if move_class == "HUGE":
            tgt = day_open + EM_MULT_HUGE * em_pts
        elif move_class == "MEDIUM":
            tgt = day_open + EM_MULT_MEDIUM * em_pts
        else:
            tgt = near
        tgt = round(max(tgt, near), 2)
        est = round(max(tgt - spot, 0.0), 1)
    return tgt, near, est


def _evaluate(idx: int, summaries: list[dict[str, Any]], em_pts: float, day_open: float,
              armed_side: str | None) -> dict[str, Any]:
    """Assess candle `idx` for a trend ignition using only data up to idx."""
    cur = summaries[idx]
    pa = cur.get("price_action", {})
    opt = cur.get("options", {})
    oi = cur.get("oi", {})

    spot = _f(pa.get("spot"))
    vwap = _f(pa.get("vwap"))
    chg = _f(pa.get("spot_change_pct"))
    atm = int(_f(opt.get("atm_strike")))
    call_ltp, put_ltp = _f(opt.get("call_ltp")), _f(opt.get("put_ltp"))
    call_low, put_low = _f(opt.get("call_low")), _f(opt.get("put_low"))
    call_chg, put_chg = _f(opt.get("call_change_pct")), _f(opt.get("put_change_pct"))
    call_oi_chg, put_oi_chg = _f(oi.get("call_oi_change_candle_pct")), _f(oi.get("put_oi_change_candle_pct"))

    lo = max(0, idx - ATR_WINDOW)
    atr = 0.0
    moves = [abs(_f(summaries[j].get("price_action", {}).get("spot_change_pct"))) for j in range(lo, idx)]
    if moves:
        atr = sum(moves) / len(moves)
    atr = max(atr, ATR_FLOOR_PCT)
    exp_x = abs(chg) / atr if atr else 0.0

    clo = max(0, idx - COIL_WINDOW)
    highs = [_f(summaries[j].get("price_action", {}).get("high")) for j in range(clo, idx)]
    lows = [_f(summaries[j].get("price_action", {}).get("low")) for j in range(clo, idx) if _f(summaries[j].get("price_action", {}).get("low")) > 0]
    coil_high = max(highs) if highs else spot
    coil_low = min(lows) if lows else spot

    below_streak = 0
    above_streak = 0
    for j in range(idx - 1, max(-1, idx - 12), -1):
        s = summaries[j].get("price_action", {}).get("spot_vs_vwap")
        if s == "below":
            if above_streak:
                break
            below_streak += 1
        elif s == "above":
            if below_streak:
                break
            above_streak += 1
        else:
            break

    # VWAP slope over the coil window.
    vwap_then = _f(summaries[clo].get("price_action", {}).get("vwap")) if clo < idx else vwap
    vwap_falling = vwap > 0 and vwap_then > 0 and vwap < vwap_then
    vwap_rising = vwap > 0 and vwap_then > 0 and vwap > vwap_then

    result: dict[str, Any] = {
        "spot": spot, "vwap": vwap, "atm": atm, "chg": chg, "exp_x": round(exp_x, 1),
        "coil_low": round(coil_low, 2), "coil_high": round(coil_high, 2),
        "below_streak": below_streak, "above_streak": above_streak,
        "call_ltp": call_ltp, "put_ltp": put_ltp,
        "call_chg": call_chg, "put_chg": put_chg,
        "call_oi_chg": call_oi_chg, "put_oi_chg": put_oi_chg,
        "stage": None, "side": None, "watch_side": None,
        "move_class": None, "confidence": 0.0, "reasons": [],
        "entry": None, "stop_ltp": None, "stop_spot": None,
        "target_spot": None, "target_near": None, "est_move_pts": None,
    }
    if spot <= 0 or atm <= 0:
        return result

    weak = below_streak >= VWAP_SIDE_STREAK and spot < vwap
    strong = above_streak >= VWAP_SIDE_STREAK and spot > vwap
    if weak and vwap_falling:
        result["watch_side"] = "PE"
    elif strong and vwap_rising:
        result["watch_side"] = "CE"

    broke_low = spot < coil_low
    broke_high = spot > coil_high
    big = abs(chg) >= MIN_MOVE_PCT and exp_x >= EXPANSION_MULT

    pe_ignite = (weak and vwap_falling and broke_low and big and chg < 0 and put_chg >= PREM_SURGE_PCT)
    ce_ignite = (strong and vwap_rising and broke_high and big and chg > 0 and call_chg >= PREM_SURGE_PCT)

    # Continuation: an ignition already fired this side; keep flagging thrusts as adds.
    pe_cont = (armed_side == "PE" and spot < vwap and chg < 0 and broke_low
               and abs(chg) >= MIN_MOVE_PCT and exp_x >= CONT_EXPANSION_MULT and put_chg >= CONT_PREM_SURGE_PCT)
    ce_cont = (armed_side == "CE" and spot > vwap and chg > 0 and broke_high
               and abs(chg) >= MIN_MOVE_PCT and exp_x >= CONT_EXPANSION_MULT and call_chg >= CONT_PREM_SURGE_PCT)

    def _finish(side: str, stage: str) -> None:
        prem_surge = put_chg if side == "PE" else call_chg
        oi_chg = put_oi_chg if side == "PE" else call_oi_chg
        streak = below_streak if side == "PE" else above_streak
        move_class = _classify_move(exp_x, prem_surge, streak, VWAP_SIDE_STREAK + 3)
        if stage == "CONTINUATION" and move_class == "HUGE":
            move_class = "MEDIUM"  # don't re-flag adds as the marquee move
        tgt, near, est = _targets(side, spot, day_open, em_pts, coil_low, coil_high, move_class)
        opt_ltp = put_ltp if side == "PE" else call_ltp
        opt_lo = put_low if side == "PE" else call_low
        stop_ltp = round(opt_lo if opt_lo > 0 else opt_ltp * 0.7, 2)
        stop_ltp = max(min(stop_ltp, opt_ltp), 0.0)
        risk_pct = round((opt_ltp - stop_ltp) / opt_ltp * 100.0, 1) if opt_ltp > 0 else 0.0
        stop_spot = round(coil_low if side == "PE" else coil_high, 2)
        arrow = "down through" if side == "PE" else "up through"
        wall = coil_low if side == "PE" else coil_high
        reasons = [
            f"Spot {spot:.0f} broke {arrow} the {wall:.0f} coil {'floor' if side == 'PE' else 'ceiling'}",
            f"Range expansion {exp_x:.0f}x trailing volatility ({chg:+.2f}%/candle)",
            f"{side} premium surging {prem_surge:+.0f}%/candle (fresh {'put' if side == 'PE' else 'call'} buying)",
            f"{streak} candles {'below' if side == 'PE' else 'above'} a {'falling' if side == 'PE' else 'rising'} VWAP",
        ]
        if (oi_chg <= -5.0):
            reasons.append(f"{side} OI unwinding {oi_chg:.0f}% (writers covering into the move)")
        base = 0.55 if stage == "IGNITION" else 0.45
        hits = sum([exp_x >= EXPANSION_MULT, prem_surge >= PREM_SURGE_PCT,
                    streak >= VWAP_SIDE_STREAK, (vwap_falling if side == "PE" else vwap_rising), exp_x >= HUGE_EXP_MULT])
        result.update(
            stage=stage, side=side, move_class=move_class,
            confidence=round(min(1.0, base + hits / 6.0), 2), reasons=reasons,
            entry=round(opt_ltp, 2), stop_ltp=stop_ltp, stop_spot=stop_spot, risk_pct=risk_pct,
            target_spot=tgt, target_near=near, est_move_pts=est,
        )

    if pe_ignite or ce_ignite:
        _finish("PE" if pe_ignite else "CE", "IGNITION")
    elif pe_cont or ce_cont:
        _finish("PE" if pe_cont else "CE", "CONTINUATION")

    return result


def trend_ignition_tape(summaries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Per-candle tape annotated with the trend-ignition state (for the analysis table)."""
    summaries = [s for s in (summaries or []) if s]
    if not summaries:
        return []
    day_open = _f(summaries[0].get("price_action", {}).get("open")) or _f(summaries[0].get("price_action", {}).get("spot"))
    em_pts = _f(summaries[0].get("options", {}).get("call_ltp")) + _f(summaries[0].get("options", {}).get("put_ltp"))

    rows: list[dict[str, Any]] = []
    armed_side: str | None = None
    for idx in range(len(summaries)):
        ts = str(summaries[idx].get("timestamp", ""))
        if _hhmm(ts) < ACTIVE_FROM_HHMM:
            a = None
        else:
            a = _evaluate(idx, summaries, em_pts, day_open, armed_side)
            if a["stage"] == "IGNITION":
                armed_side = a["side"]
        pa = summaries[idx].get("price_action", {})
        opt = summaries[idx].get("options", {})
        oi = summaries[idx].get("oi", {})
        state = ""
        side = None
        if a:
            if a["stage"] in ("IGNITION", "CONTINUATION"):
                state = a["stage"]
                side = a["side"]
            elif a["watch_side"]:
                state = "WATCH"
                side = a["watch_side"]
        rows.append({
            "time": _hhmm(ts),
            "timestamp": ts,
            "spot": round(_f(pa.get("spot")), 2),
            "vwap": round(_f(pa.get("vwap")), 2),
            "spot_vs_vwap": pa.get("spot_vs_vwap"),
            "spot_change_pct": round(_f(pa.get("spot_change_pct")), 2),
            "expansion_x": a["exp_x"] if a else None,
            "coil_low": a["coil_low"] if a else None,
            "coil_high": a["coil_high"] if a else None,
            "atm_strike": int(_f(opt.get("atm_strike"))),
            "call_change_pct": round(_f(opt.get("call_change_pct")), 1),
            "put_change_pct": round(_f(opt.get("put_change_pct")), 1),
            "call_oi_change_candle_pct": round(_f(oi.get("call_oi_change_candle_pct")), 1),
            "put_oi_change_candle_pct": round(_f(oi.get("put_oi_change_candle_pct")), 1),
            "move_class": a["move_class"] if a else None,
            "est_move_pts": a["est_move_pts"] if a else None,
            "target_spot": a["target_spot"] if a else None,
            "state": state,
            "side": side,
        })
    return rows


def compute_trend_ignition(summaries: list[dict[str, Any]]) -> dict[str, Any]:
    """Evaluate the Trend Ignition Radar over ordered per-candle summaries."""
    summaries = [s for s in (summaries or []) if s]
    if len(summaries) < COIL_WINDOW:
        return asdict(TrendIgnitionState(
            active=False, state="IDLE", side=None,
            headline="Not enough candles yet to define a coil / trend.",
        ))

    day_open = _f(summaries[0].get("price_action", {}).get("open")) or _f(summaries[0].get("price_action", {}).get("spot"))
    em_pts = _f(summaries[0].get("options", {}).get("call_ltp")) + _f(summaries[0].get("options", {}).get("put_ltp"))

    triggers: list[TrendTrigger] = []
    armed_side: str | None = None
    latest: dict[str, Any] | None = None
    for idx in range(len(summaries)):
        ts = str(summaries[idx].get("timestamp", ""))
        if _hhmm(ts) < ACTIVE_FROM_HHMM:
            continue
        a = _evaluate(idx, summaries, em_pts, day_open, armed_side)
        latest = a
        if a["stage"] in ("IGNITION", "CONTINUATION"):
            if a["stage"] == "IGNITION":
                armed_side = a["side"]
            triggers.append(TrendTrigger(
                timestamp=ts, stage=a["stage"], side=a["side"], strike=a["atm"],
                spot=round(a["spot"], 2),
                option_ltp=a["entry"], prem_change_pct=round(a["put_chg"] if a["side"] == "PE" else a["call_chg"], 1),
                oi_change_pct=round(a["put_oi_chg"] if a["side"] == "PE" else a["call_oi_chg"], 1),
                expansion_x=a["exp_x"], entry=a["entry"], stop_ltp=a["stop_ltp"], stop_spot=a["stop_spot"],
                risk_pct=a["risk_pct"], move_class=a["move_class"], est_move_pts=a["est_move_pts"],
                target_spot=a["target_spot"], target_near=a["target_near"],
                confidence=a["confidence"], reasons=a["reasons"],
            ))

    trig_dicts = [asdict(t) for t in triggers]
    first_ignition = next((t for t in trig_dicts if t["stage"] == "IGNITION"), None)

    a = latest or {}
    spot = a.get("spot")
    if triggers and a.get("side"):
        side = a["side"]
        if a.get("stage") == "IGNITION":
            state = "IGNITION"
            headline = (f"🚀 TREND IGNITION — {side} · {a['move_class']} move likely "
                        f"(~{a['est_move_pts']:.0f} pts to {a['target_spot']:.0f}).")
        else:
            state = "RUNNING"
            headline = (f"Trend running — {side} · add/hold, target ~{a.get('target_spot')}. "
                        f"First ignition {first_ignition['timestamp'][11:16] if first_ignition else ''}.")
        move_class = a.get("move_class")
        est = a.get("est_move_pts")
        target = a.get("target_spot")
        entry = a.get("entry")
        stop_ltp = a.get("stop_ltp")
        stop_spot = a.get("stop_spot")
        conf = a.get("confidence", 0.0)
        reasons = a.get("reasons", [])
    elif a.get("watch_side"):
        side = a["watch_side"]
        state = "WATCH"
        headline = (f"Watching {side}: price {'below falling' if side == 'PE' else 'above rising'} VWAP & coiling — "
                    f"break of {a['coil_low'] if side == 'PE' else a['coil_high']:.0f} on a volatility + premium surge ignites it.")
        move_class = est = target = entry = stop_ltp = stop_spot = None
        conf = 0.25
        reasons = [
            f"{a['below_streak'] if side == 'PE' else a['above_streak']} candles {'below' if side == 'PE' else 'above'} VWAP",
            f"Coiling {a['coil_low']:.0f}–{a['coil_high']:.0f}; waiting for the break candle.",
        ]
    else:
        side = None
        state = "IDLE"
        headline = "No directional coil/trend set up yet — two-way / balanced."
        move_class = est = target = entry = stop_ltp = stop_spot = None
        conf = 0.0
        reasons = ["Price not sustainably one side of VWAP, or no coil to break."]

    # If an ignition already fired earlier today but the tape has since cooled to
    # WATCH/IDLE, keep surfacing that call (side, move class, target) in the summary.
    if first_ignition and state in ("WATCH", "IDLE"):
        side = first_ignition["side"]
        move_class = first_ignition["move_class"]
        est = first_ignition["est_move_pts"]
        target = first_ignition["target_spot"]
        entry = entry or first_ignition["entry"]
        stop_ltp = stop_ltp or first_ignition["stop_ltp"]
        stop_spot = stop_spot or first_ignition["stop_spot"]
        headline = (f"🚀 {side} trend ignited {first_ignition['timestamp'][11:16]} "
                    f"({move_class} · target ~{target:.0f}). " + headline)

    return asdict(TrendIgnitionState(
        active=bool(triggers) or state in ("WATCH", "IGNITION", "RUNNING"),
        state=state, side=side, headline=headline,
        spot=round(spot, 2) if spot else None,
        vwap=round(a["vwap"], 2) if a.get("vwap") else None,
        atm_strike=a.get("atm"),
        coil_low=a.get("coil_low"), coil_high=a.get("coil_high"),
        expected_move_pts=round(em_pts, 1), day_open=round(day_open, 2),
        move_class=move_class, est_move_pts=est, target_spot=target,
        entry=entry, stop_ltp=stop_ltp, stop_spot=stop_spot,
        confidence=round(conf, 2), reasons=reasons,
        latest_trigger=trig_dicts[-1] if trig_dicts else None,
        triggers=trig_dicts,
    ))
