"""
Expiry Gamma Radar — detect the expiry-day option "explosion" setup and trigger.

Anatomy of the move it targets (validated on the 07-Jul-2026 NIFTY 24450 PE that
ran ~6x in 15 minutes near the close):

  1. ENGINE  — expiry day, final ~90 minutes: an ATM/near-ATM strike is almost
               pure time value (cheap), so near-zero DTE gamma makes a tiny spot
               move produce a huge % move in the option.
  2. SETUP   — spot pinned within a hair of a heavy-OI strike ("wall"), on the
               weak side of VWAP (PE candidate below VWAP, CE candidate above).
  3. TRIGGER — spot slices THROUGH the strike while that side's premium surges and
               its OI *unwinds* (writers short-covering / gamma squeeze), with spot
               momentum confirming. That OI-down + premium-up combo is the ignition.

This detector runs over the per-candle analysis summaries (ATM strike stream), so
it is fully back-testable on stored snapshots. It reports a state per the latest
candle: OFF (not expiry) / IDLE / ARMED / TRIGGERED, the candidate side & strike,
a confidence score, human reasons, and a timeline of trigger fires.

It is a de-risked *lottery* alert: most armed setups pin and expire worthless, so
capitalize only with tiny, defined risk (the premium) and fast exits.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import date
from typing import Any

from intraday_engine.gamma.expiry_utils import is_expiry_day

# --- tunables (sensible defaults; conservative) ---
ACTIVE_FROM_HHMM = "14:00"     # only hunt in the final stretch where gamma is huge
NEAR_STRIKE_PCT = 0.15         # spot within this % of the ATM strike = pinned to wall
CHEAP_PREMIUM_MAX = 45.0       # candidate-side premium must be "cheap lottery" when arming
PREM_SURGE_PCT = 25.0          # per-candle premium jump confirming the break (BREAK stage)
OI_UNWIND_PCT = -5.0           # per-candle OI drop on the exploding side (short covering)
SPOT_MOMENTUM_PCT = 0.05       # per-candle spot move confirming direction

# EARLY / IGNITION stage — the actionable entry BEFORE the strike breaks. Fires on the
# first candle where an armed setup shows premium expansion + roll-over while pressing
# the strike, so you enter cheap with a tight stop (the strike itself) instead of chasing
# the break candle (which needs a huge SL).
EARLY_PREM_SURGE_PCT = 30.0    # first real premium jump = fresh buying stepping in
EARLY_NEAR_PCT = 0.08          # spot pressing the strike (within ~0.08%)
IGNITION_OI_MAX = 0.5          # OI must NOT be building against you (rising OI = writers defending)
STOP_BUFFER_PCT = 0.05         # SL = strike reclaim + this buffer
IGNITION_MAX_SL_POINTS = 12.0  # ignition premium SL: below candle low, but never risk more than this


@dataclass
class GammaTrigger:
    timestamp: str
    stage: str                 # IGNITION | BREAK
    side: str                  # CE | PE
    strike: int
    spot: float
    option_ltp: float
    prem_change_pct: float
    oi_change_pct: float
    entry: float               # suggested entry (option LTP at signal)
    stop_spot: float           # invalidation level on the underlying (strike reclaim)
    stop_ltp: float            # option-premium SL = prior candle's LTP (setup fails if it retraces here)
    risk_pct: float            # (entry - stop_ltp) / entry * 100
    confidence: float
    reasons: list[str] = field(default_factory=list)


@dataclass
class ExpiryGammaState:
    active: bool               # expiry day AND inside the active window
    is_expiry_day: bool
    state: str                 # OFF | IDLE | ARMED | IGNITION | TRIGGERED
    side: str | None           # CE | PE | None
    headline: str
    spot: float | None = None
    atm_strike: int | None = None
    dist_to_strike_pct: float | None = None
    candidate_strike: int | None = None
    candidate_hint: str | None = None
    entry: float | None = None
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
    if "T" in s and len(s) >= 16:
        return s[11:16]
    if " " in s and len(s) >= 16:
        return s[11:16]
    return "00:00"


def _evaluate_candle(cur: dict[str, Any], prev: dict[str, Any] | None) -> dict[str, Any]:
    """Return per-candle assessment: armed side, trigger side, confidence, reasons."""
    pa = cur.get("price_action", {})
    opt = cur.get("options", {})
    oi = cur.get("oi", {})

    spot = _f(pa.get("spot"))
    vwap = _f(pa.get("vwap"))
    atm = int(_f(opt.get("atm_strike")))
    call_ltp = _f(opt.get("call_ltp"))
    put_ltp = _f(opt.get("put_ltp"))
    call_low = _f(opt.get("call_low"))
    put_low = _f(opt.get("put_low"))
    call_chg = _f(opt.get("call_change_pct"))
    put_chg = _f(opt.get("put_change_pct"))
    call_oi_chg = _f(oi.get("call_oi_change_candle_pct"))
    put_oi_chg = _f(oi.get("put_oi_change_candle_pct"))
    spot_chg = _f(pa.get("spot_change_pct"))
    prev_opt = (prev or {}).get("options", {})
    prev_call_ltp = _f(prev_opt.get("call_ltp"))
    prev_put_ltp = _f(prev_opt.get("put_ltp"))

    dist_pct = (abs(spot - atm) / spot * 100.0) if spot else None
    result = {
        "spot": spot, "atm": atm,
        "dist_pct": dist_pct,
        "armed_side": None, "ignition_side": None, "trigger_side": None,
        "stage": None,               # IGNITION | BREAK
        "confidence": 0.0, "reasons": [],
        "call_ltp": call_ltp, "put_ltp": put_ltp,
        "call_low": call_low, "put_low": put_low,
        "put_chg": put_chg, "call_chg": call_chg,
        "put_oi_chg": put_oi_chg, "call_oi_chg": call_oi_chg,
        "prev_call_ltp": prev_call_ltp, "prev_put_ltp": prev_put_ltp,
        "entry": None, "stop_spot": None,
    }
    if spot <= 0 or atm <= 0:
        return result

    near = dist_pct is not None and dist_pct <= NEAR_STRIKE_PCT
    pressing = dist_pct is not None and dist_pct <= EARLY_NEAR_PCT
    below_vwap = vwap > 0 and spot < vwap
    above_vwap = vwap > 0 and spot > vwap

    # --- ARMED: pinned to the wall on the weak/strong side, cheap premium ---
    if near:
        if below_vwap and put_ltp <= CHEAP_PREMIUM_MAX:
            result["armed_side"] = "PE"
        elif above_vwap and call_ltp <= CHEAP_PREMIUM_MAX:
            result["armed_side"] = "CE"

    def _stop_spot(side: str) -> float:
        # Invalidation = spot reclaims the strike (+buffer). Tight because entry is near the strike.
        buf = atm * STOP_BUFFER_PCT / 100.0
        return round(atm + buf, 2) if side == "PE" else round(atm - buf, 2)

    # --- BREAK (confirmed): spot slices through the strike with premium surge + OI unwind ---
    pe_break = spot < atm and put_chg >= PREM_SURGE_PCT and put_oi_chg <= OI_UNWIND_PCT and spot_chg <= -SPOT_MOMENTUM_PCT
    ce_break = spot > atm and call_chg >= PREM_SURGE_PCT and call_oi_chg <= OI_UNWIND_PCT and spot_chg >= SPOT_MOMENTUM_PCT

    # --- IGNITION (early entry): armed + pressing the strike + first premium expansion + rolling over,
    #     BEFORE the actual break. Cheaper entry, tighter stop (the strike itself). ---
    pe_ignite = (result["armed_side"] == "PE" and pressing and spot >= atm
                 and put_chg >= EARLY_PREM_SURGE_PCT and put_oi_chg <= IGNITION_OI_MAX and spot_chg <= 0)
    ce_ignite = (result["armed_side"] == "CE" and pressing and spot <= atm
                 and call_chg >= EARLY_PREM_SURGE_PCT and call_oi_chg <= IGNITION_OI_MAX and spot_chg >= 0)

    def _conf_reasons(side: str, stage: str) -> tuple[float, list[str]]:
        reasons: list[str] = []
        hits = 0
        chg = put_chg if side == "PE" else call_chg
        oi_chg = put_oi_chg if side == "PE" else call_oi_chg
        weak_ok = below_vwap if side == "PE" else above_vwap
        mom_ok = (spot_chg <= -SPOT_MOMENTUM_PCT) if side == "PE" else (spot_chg >= SPOT_MOMENTUM_PCT)
        arrow = "below" if side == "PE" else "above"
        if stage == "BREAK":
            reasons.append(f"Spot {spot:.0f} broke {arrow} {atm} strike")
        else:
            reasons.append(f"Spot {spot:.0f} pressing {atm} strike ({dist_pct:.2f}% away), about to break")
        surge_gate = PREM_SURGE_PCT if stage == "BREAK" else EARLY_PREM_SURGE_PCT
        if chg >= surge_gate:
            hits += 1; reasons.append(f"{side} premium expanding +{chg:.0f}%/candle (fresh buying)")
        if oi_chg <= OI_UNWIND_PCT:
            hits += 1; reasons.append(f"{side} OI unwinding {oi_chg:.0f}% (writers covering)")
        if weak_ok:
            hits += 1; reasons.append(f"Spot {'below' if side == 'PE' else 'above'} VWAP ({'weak' if side == 'PE' else 'strong'})")
        if mom_ok or (stage == "IGNITION" and ((spot_chg <= 0) if side == "PE" else (spot_chg >= 0))):
            hits += 1; reasons.append(f"Spot rolling {'down' if side == 'PE' else 'up'} {spot_chg:+.2f}%/candle")
        base = 0.5 if stage == "IGNITION" else 0.6
        return min(1.0, base + hits / 5.0), reasons

    # Priority: a confirmed break dominates; else ignition; else armed; else nothing.
    if pe_break or ce_break:
        side = "PE" if pe_break else "CE"
        conf, reasons = _conf_reasons(side, "BREAK")
        result.update(trigger_side=side, stage="BREAK", confidence=conf, reasons=reasons,
                      entry=round(put_ltp if side == "PE" else call_ltp, 2), stop_spot=_stop_spot(side))
    elif pe_ignite or ce_ignite:
        side = "PE" if pe_ignite else "CE"
        conf, reasons = _conf_reasons(side, "IGNITION")
        result.update(ignition_side=side, stage="IGNITION", confidence=conf, reasons=reasons,
                      entry=round(put_ltp if side == "PE" else call_ltp, 2), stop_spot=_stop_spot(side))
    elif result["armed_side"]:
        side = result["armed_side"]
        result["reasons"] = [
            f"Spot {spot:.0f} pinned {dist_pct:.2f}% from {atm} wall",
            f"{'Below' if side == 'PE' else 'Above'} VWAP — {side} explosion candidate if strike breaks",
            f"{side} premium cheap (₹{put_ltp if side == 'PE' else call_ltp:.1f}) = high gamma leverage",
        ]
        result["confidence"] = 0.3
        result["stop_spot"] = _stop_spot(side)

    return result


def expiry_window_tape(
    summaries: list[dict[str, Any]],
    *,
    from_hhmm: str = "14:30",
    to_hhmm: str = "15:30",
) -> list[dict[str, Any]]:
    """
    Return the per-5-min-candle tape for the expiry gamma window (default 2:30-3:30 PM),
    each row annotated with the radar state (ARMED / TRIGGERED) and side.
    """
    rows: list[dict[str, Any]] = []
    prev = None
    for s in (summaries or []):
        if not s:
            continue
        ts = str(s.get("timestamp", ""))
        hh = _hhmm(ts)
        if from_hhmm <= hh <= to_hhmm:
            a = _evaluate_candle(s, prev)
            pa = s.get("price_action", {})
            opt = s.get("options", {})
            oi = s.get("oi", {})
            fut = s.get("futures", {})
            if a["trigger_side"]:
                state = "TRIGGERED"
            elif a["ignition_side"]:
                state = "IGNITION"
            elif a["armed_side"]:
                state = "ARMED"
            else:
                state = ""
            side = a["trigger_side"] or a["ignition_side"] or a["armed_side"]
            rows.append({
                "time": hh,
                "timestamp": ts,
                "spot": round(_f(pa.get("spot")), 2),
                "vwap": round(_f(pa.get("vwap")), 2),
                "spot_vs_vwap": pa.get("spot_vs_vwap"),
                "fut": round(_f(fut.get("ltp")), 2),
                "atm_strike": int(_f(opt.get("atm_strike"))),
                "call_ltp": round(_f(opt.get("call_ltp")), 2),
                "put_ltp": round(_f(opt.get("put_ltp")), 2),
                "call_change_pct": round(_f(opt.get("call_change_pct")), 1),
                "put_change_pct": round(_f(opt.get("put_change_pct")), 1),
                "call_oi": int(_f(oi.get("call_oi"))),
                "put_oi": int(_f(oi.get("put_oi"))),
                "call_oi_change_candle_pct": round(_f(oi.get("call_oi_change_candle_pct")), 1),
                "put_oi_change_candle_pct": round(_f(oi.get("put_oi_change_candle_pct")), 1),
                "state": state,
                "side": side,
            })
        prev = s
    return rows


def compute_expiry_gamma_radar(
    summaries: list[dict[str, Any]],
    *,
    trade_date: date,
    underlying: str = "NIFTY",
    force: bool = False,
) -> dict[str, Any]:
    """
    Evaluate the Expiry Gamma Radar over the ordered per-candle summaries.

    `force=True` bypasses the expiry-day gate (used for back-testing any day).
    """
    expiry = is_expiry_day(trade_date, underlying)
    summaries = [s for s in (summaries or []) if s]

    if not (expiry or force):
        return asdict(ExpiryGammaState(
            active=False, is_expiry_day=expiry, state="OFF", side=None,
            headline="Not an expiry day — gamma radar is off.",
        ))
    if not summaries:
        return asdict(ExpiryGammaState(
            active=False, is_expiry_day=expiry, state="OFF", side=None,
            headline="No candle data yet.",
        ))

    triggers: list[GammaTrigger] = []
    latest_assess: dict[str, Any] | None = None
    latest_in_window = False

    prev = None
    for s in summaries:
        ts = str(s.get("timestamp", ""))
        in_window = _hhmm(ts) >= ACTIVE_FROM_HHMM
        assess = _evaluate_candle(s, prev)
        prev = s
        if not in_window:
            continue
        latest_assess = assess
        latest_in_window = True
        fired_side = assess["trigger_side"] or assess["ignition_side"]
        if fired_side:
            entry_ltp = round(assess["put_ltp"] if fired_side == "PE" else assess["call_ltp"], 2)
            candle_low = assess["put_low"] if fired_side == "PE" else assess["call_low"]
            # Ignition SL: park it just below the ignition candle's own low, but cap the risk at
            # IGNITION_MAX_SL_POINTS (a wide candle => use a fixed 12-pt premium stop instead).
            if assess["stage"] == "IGNITION" and candle_low > 0:
                low_risk = entry_ltp - candle_low
                if 0 < low_risk <= IGNITION_MAX_SL_POINTS:
                    stop_ltp = round(candle_low, 2)
                else:
                    stop_ltp = round(entry_ltp - IGNITION_MAX_SL_POINTS, 2)
            else:
                stop_ltp = round(assess["prev_put_ltp"] if fired_side == "PE" else assess["prev_call_ltp"], 2)
            stop_ltp = max(stop_ltp, 0.0)
            risk_pct = round((entry_ltp - stop_ltp) / entry_ltp * 100.0, 1) if entry_ltp > 0 else 0.0
            triggers.append(GammaTrigger(
                timestamp=ts,
                stage=assess["stage"],
                side=fired_side,
                strike=assess["atm"],
                spot=round(assess["spot"], 2),
                option_ltp=entry_ltp,
                prem_change_pct=round(assess["put_chg"] if fired_side == "PE" else assess["call_chg"], 1),
                oi_change_pct=round(assess["put_oi_chg"] if fired_side == "PE" else assess["call_oi_chg"], 1),
                entry=entry_ltp,
                stop_spot=assess["stop_spot"],
                stop_ltp=stop_ltp,
                risk_pct=risk_pct,
                confidence=round(assess["confidence"], 2),
                reasons=assess["reasons"],
            ))

    trig_dicts = [asdict(t) for t in triggers]
    # The best actionable entry is the FIRST ignition (earliest, cheapest, tight stop).
    first_ignition = next((t for t in trig_dicts if t["stage"] == "IGNITION"), None)

    if not latest_in_window or latest_assess is None:
        return asdict(ExpiryGammaState(
            active=False, is_expiry_day=expiry, state="IDLE", side=None,
            headline=f"Expiry day — radar arms after {ACTIVE_FROM_HHMM} (final-hour gamma window).",
            triggers=trig_dicts,
            latest_trigger=trig_dicts[-1] if trig_dicts else None,
        ))

    a = latest_assess
    candidate = None
    hint = None
    entry = a.get("entry")
    stop_spot = a.get("stop_spot")
    if a["trigger_side"]:
        state = "TRIGGERED"
        side = a["trigger_side"]
        candidate = a["atm"]
        hint = f"{a['atm']} {side} broke — momentum entry, SL spot reclaim {stop_spot}. Ride gamma, exit fast."
        headline = f"⚡ GAMMA BREAK — {a['atm']} {side} exploding. Late chase = wide SL; prefer the ignition entry."
        conf = a["confidence"]
        reasons = a["reasons"]
    elif a["ignition_side"]:
        state = "IGNITION"
        side = a["ignition_side"]
        candidate = a["atm"]
        hint = f"ENTER {a['atm']} {side} ~₹{entry} · SL spot reclaims {stop_spot} (tight). Break not required yet."
        headline = f"🎯 IGNITION — {a['atm']} {side} entry now (~₹{entry}), tight SL at the strike."
        conf = a["confidence"]
        reasons = a["reasons"]
    elif a["armed_side"]:
        state = "ARMED"
        side = a["armed_side"]
        candidate = a["atm"]
        hint = f"{a['atm']} {side} — wait for ignition (premium expansion + roll-over) or pre-position tiny."
        headline = f"Armed: {a['atm']} {side} gamma candidate — spot pinned to the wall."
        conf = a["confidence"]
        reasons = a["reasons"]
    else:
        state = "IDLE"
        side = None
        headline = "In window — no strike pin yet. Watch for spot to pin a heavy strike."
        conf = 0.0
        reasons = ["Spot not close enough to a strike to set up an expiry gamma move."]

    # If we've already passed the ignition earlier today, surface that as the recommended entry.
    if first_ignition and state in ("TRIGGERED", "IGNITION"):
        reasons = list(reasons) + [
            f"Best entry was the ignition at {first_ignition['timestamp'][11:16]} "
            f"(~₹{first_ignition['entry']}, SL {first_ignition['stop_spot']})."
        ]

    return asdict(ExpiryGammaState(
        active=True,
        is_expiry_day=expiry,
        state=state,
        side=side,
        headline=headline,
        spot=round(a["spot"], 2) if a.get("spot") else None,
        atm_strike=a.get("atm"),
        dist_to_strike_pct=round(a["dist_pct"], 3) if a.get("dist_pct") is not None else None,
        candidate_strike=candidate,
        candidate_hint=hint,
        entry=entry,
        stop_spot=stop_spot,
        confidence=round(conf, 2),
        reasons=reasons,
        latest_trigger=trig_dicts[-1] if trig_dicts else None,
        triggers=trig_dicts,
    ))
