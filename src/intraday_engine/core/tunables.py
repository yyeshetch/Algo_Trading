"""Load and save project-root config.json (tunables). See _meta in the JSON file for format notes."""

from __future__ import annotations

import copy
import json
import threading
from pathlib import Path
from typing import Any

_lock = threading.Lock()
_cache: dict[str, Any] | None = None
_cache_mtime: float | None = None


def project_root() -> Path:
    return Path(__file__).resolve().parent.parent.parent.parent


def config_path() -> Path:
    return project_root() / "config.json"


def _is_leaf(node: Any) -> bool:
    return isinstance(node, dict) and "value" in node and "type" in node


def _coerce(val: Any, typ: str) -> Any:
    if typ == "int":
        return int(val)
    if typ == "float":
        return float(val)
    if typ == "str":
        return str(val)
    return val


def _overlay_user(base: dict[str, Any], user: dict[str, Any]) -> None:
    """Apply user file on top of defaults (leaf .value only; preserve help/type/env from base)."""
    for k, uv in user.items():
        if k.startswith("_") or k == "version":
            base[k] = copy.deepcopy(uv)
            continue
        if k not in base:
            continue
        bv = base[k]
        if _is_leaf(bv):
            if _is_leaf(uv):
                try:
                    base[k] = {
                        **bv,
                        "value": _coerce(uv["value"], str(bv.get("type", "float"))),
                    }
                except (TypeError, ValueError, KeyError):
                    pass
        elif isinstance(bv, dict) and isinstance(uv, dict):
            _overlay_user(bv, uv)


def default_config() -> dict[str, Any]:
    """Full default tree; used when config.json is missing and for merge."""
    return copy.deepcopy(_DEFAULT_CONFIG)


def load_config(*, force_reload: bool = False) -> dict[str, Any]:
    global _cache, _cache_mtime
    path = config_path()
    with _lock:
        mtime = path.stat().st_mtime if path.exists() else 0.0
        if not force_reload and _cache is not None and _cache_mtime == mtime:
            return _cache
        if path.exists():
            raw = json.loads(path.read_text(encoding="utf-8"))
        else:
            raw = {}
        merged = default_config()
        _overlay_user(merged, raw)
        _cache = merged
        _cache_mtime = mtime
        return _cache


def invalidate_cache() -> None:
    global _cache, _cache_mtime
    with _lock:
        _cache = None
        _cache_mtime = None


def ensure_config_file() -> None:
    path = config_path()
    if not path.exists():
        path.write_text(json.dumps(default_config(), indent=2), encoding="utf-8")
        invalidate_cache()


def save_config(data: dict[str, Any]) -> None:
    path = config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    invalidate_cache()
    load_config(force_reload=True)


def get_leaf(section: str, key: str, default: Any) -> Any:
    cfg = load_config()
    sec = cfg.get(section, {})
    node = sec.get(key)
    if _is_leaf(node):
        try:
            return _coerce(node["value"], str(node.get("type", "float")))
        except (TypeError, ValueError, KeyError):
            return default
    return default


def get_float(section: str, key: str, default: float) -> float:
    v = get_leaf(section, key, default)
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def get_int(section: str, key: str, default: int) -> int:
    v = get_leaf(section, key, default)
    try:
        return int(v)
    except (TypeError, ValueError):
        return default


def get_str(section: str, key: str, default: str) -> str:
    v = get_leaf(section, key, default)
    return str(v) if v is not None else default


def get_nested_float(section: str, default: float, *path: str) -> float:
    cfg = load_config()
    cur: Any = cfg.get(section, {})
    for p in path[:-1]:
        cur = cur.get(p, {})
    node = cur.get(path[-1]) if path else None
    if _is_leaf(node):
        try:
            return float(_coerce(node["value"], str(node.get("type", "float"))))
        except (TypeError, ValueError, KeyError):
            return default
    return default


def apply_value_updates(current: dict[str, Any], updates: dict[str, Any]) -> None:
    """Apply user POST body: only update leaf .value fields where structure matches."""
    for uk, uv in updates.items():
        if uk.startswith("_") or uk == "version":
            continue
        if uk not in current:
            continue
        cv = current[uk]
        if _is_leaf(cv) and _is_leaf(uv):
            current[uk]["value"] = _coerce(uv["value"], str(cv.get("type", "float")))
        elif isinstance(cv, dict) and isinstance(uv, dict):
            apply_value_updates(cv, uv)


def validate_no_unknown_keys(template: dict[str, Any], incoming: dict[str, Any], prefix: str = "") -> list[str]:
    errors: list[str] = []
    for uk, uv in incoming.items():
        if uk.startswith("_") or uk == "version":
            continue
        path = f"{prefix}.{uk}" if prefix else uk
        if uk not in template:
            errors.append(f"Unknown key: {path}")
            continue
        tv = template[uk]
        if _is_leaf(tv) and isinstance(uv, dict):
            if "value" not in uv:
                errors.append(f"Missing value for {path}")
        elif isinstance(tv, dict) and isinstance(uv, dict) and not _is_leaf(tv):
            errors.extend(validate_no_unknown_keys(tv, uv, path))
        elif _is_leaf(tv) and not isinstance(uv, dict):
            errors.append(f"Expected object with value for {path}")
    return errors


# --- Default tree (help strings = in-file documentation; JSON does not support // comments) ---

_L = lambda v, t, h, env=None: ({"value": v, "type": t, "help": h, **({"env": env} if env else {})})

_DEFAULT_CONFIG: dict[str, Any] = {
    "_meta": {
        "about": "Tunables for Algo_Trading. Standard JSON has no comments; each field includes a help string.",
        "env_override": "For settings.* entries, if the listed env var is set in .env it overrides the value here.",
    },
    "version": 1,
    "settings": {
        "poll_interval_minutes": _L(5, "int", "How often the index scheduler polls (minutes).", "POLL_INTERVAL_MINUTES"),
        "lookback_bars": _L(20, "int", "Bars used for support/resistance and structure (5m index path).", "LOOKBACK_BARS"),
        "min_rr": _L(1.8, "float", "Minimum reward:risk for trade plan.", "MIN_RR"),
        "min_confidence": _L(0.70, "float", "Minimum model confidence for directional trade (non-neutral day).", "MIN_CONFIDENCE"),
        "min_confidence_neutral_day": _L(0.85, "float", "Stricter confidence when day bias is neutral.", "MIN_CONFIDENCE_NEUTRAL_DAY"),
        "max_stop_pct": _L(0.45, "float", "Max stop distance (% of spot) for longs.", "MAX_STOP_PCT"),
        "max_stop_pct_short": _L(0.65, "float", "Max stop distance (% of spot) for shorts.", "MAX_STOP_PCT_SHORT"),
        "option_stop_ratio": _L(0.35, "float", "Option premium stop as fraction of entry.", "OPTION_STOP_RATIO"),
        "min_day_range_pct": _L(0.40, "float", "Min session range (% of spot) to avoid sideways chop filter.", "MIN_DAY_RANGE_PCT"),
        "daily_sl_rupees": _L(4000, "float", "Daily loss cap (rupees) for dashboard / risk.", "DAILY_SL_RUPEES"),
        "default_sl_points": _L(10, "float", "Default SL distance in points when not specified.", "DEFAULT_SL_POINTS"),
        "log_level": _L("INFO", "str", "Logging level (DEBUG, INFO, …). Overridden by env LOG_LEVEL if set.", "LOG_LEVEL"),
    },
    "stock_signal_engine": {
        "FAILED_BIAS_VOLUME_MULTIPLIER": _L(1.5, "float", "Failed-bias candle volume vs SMA must exceed this multiple."),
        "FAILED_BIAS_MIN_BARS": _L(1, "int", "Minimum bars required for failed-bias volume logic."),
        "FAILED_BIAS_SMA_WINDOW": _L(20, "int", "Volume SMA length for bias / vol setups."),
        "FAILED_BIAS_MIN_CONFIDENCE": _L(0.90, "float", "Minimum failed-bias score confidence to allow trade."),
        "FAILED_BIAS_MAX_AGE_BARS": _L(4, "int", "Max bars after trigger before failed-bias expires."),
        "FAILED_BIAS_MIN_RUNWAY_BARS": _L(6, "int", "Min bars of session left (as bar count) for failed-bias."),
        "FAILED_BIAS_TARGET_RR": _L(1.25, "float", "Fast target RR used in failed-bias structure."),
        "FAILED_BIAS_MAX_STOP_PCT": _L(0.50, "float", "Max stop %% for failed-bias quality filter."),
        "FAILED_BIAS_MINUTES_FROM_OPEN": _L(105, "int", "Minutes from open before failed-bias allowed (intraday)."),
        "FAILED_BIAS_RSI_BUY_FLOOR": _L(45.0, "float", "RSI floor for BUY-side failed-bias and generic BUY filter."),
        "FAILED_BIAS_RSI_BUY_REGIME": _L(55.0, "float", "RSI regime: BUY needs readings above this (failed-bias 10-bar check)."),
        "FAILED_BIAS_RSI_SELL_CEILING": _L(55.0, "float", "RSI ceiling for SELL-side filters."),
        "FAILED_BIAS_RSI_SELL_REGIME": _L(45.0, "float", "RSI regime: SELL needs readings below this."),
        "FAILED_BIAS_RSI_LOOKBACK_CANDLES": _L(10, "int", "Prior RSI bars checked for failed-bias regime."),
        "ORB_VARIATION_PCT": _L(0.2, "float", "ORB breakout band: %% variation from opening range high/low."),
        "OVERBOUGHT_RSI_THRESHOLD": _L(75.0, "float", "15m RSI above this → OVERBROUGHT_LOOK_FOR_REVERSAL alert."),
        "VOL_EMA_VOLUME_MULTIPLIER": _L(1.5, "float", "Vol+EMA BUY: volume vs prior SMA multiple."),
        "VOL_EMA_NEAR_PCT": _L(0.35, "float", "Vol+EMA BUY: max %% distance from close to nearest EMA."),
        "VOL_EMA_MIN_EXTENDED_CLOSES": _L(30, "int", "Min extended 15m closes (history+session) for EMA context."),
        "VOL_EMA_MIN_SESSION_BARS": _L(3, "int", "Min session bars for vol+EMA path."),
        "VOL_EMA_SYNTHETIC_CONFIDENCE": _L(0.72, "float", "Floor for synthetic score confidence in vol+EMA BUY."),
        "DIRECTIONAL_FOLLOW_THROUGH_MIN_PCT": _L(0.06, "float", "Min |spot_change_pct| to count as follow-through."),
        "DIRECTIONAL_MID_RANGE_FRAC": _L(0.18, "float", "Distance from mid of S/R must be below this × range to be mid-range."),
        "FAILED_BIAS_OPTION_SL_RATIO": _L(0.7, "float", "Failed-bias option SL as fraction of option entry."),
        "FAILED_BIAS_OPTION_TARGET_RATIO": _L(1.5, "float", "Failed-bias option target multiple on premium."),
    },
    "candle_patterns": {
        "PINBAR_WICK_BODY_RATIO": _L(2.0, "float", "Pinbar: wick must be ≥ this × body."),
        "PINBAR_BODY_MAX_RANGE_FRAC": _L(0.35, "float", "Pinbar: body at most this fraction of range."),
        "PINBAR_OPP_WICK_MAX_RATIO": _L(0.5, "float", "Pinbar: opposite wick ≤ this × primary wick."),
    },
    "fno_intraday_buy_scanner": {
        "MAX_WORKERS": _L(5, "int", "Parallel workers for intraday FNO scan."),
        "RATE_LIMIT_SEC": _L(0.25, "float", "Delay between API calls per worker (seconds)."),
        "RSI_PERIOD": _L(14, "int", "RSI period for scanner."),
        "BUY_WATCHLIST_RSI": _L(55.0, "float", "RSI threshold for buy watchlist."),
        "SELL_WATCHLIST_RSI": _L(45.0, "float", "RSI threshold for sell watchlist."),
        "VOL_SMA_BARS": _L(20, "int", "Volume SMA length."),
        "VOL_MULTIPLIER": _L(2.0, "float", "Volume spike multiple vs SMA."),
        "SWING_LOOKBACK": _L(10, "int", "Swing lookback bars."),
        "VOLUME_BIAS_MULTIPLIER": _L(1.5, "float", "Volume bias candle multiple vs SMA."),
    },
    "orb_scanner": {
        "ORB_VARIATION_PCT": _L(0.2, "float", "ORB band %% from opening range (stock ORB scan)."),
        "QUOTE_BATCH_SIZE": _L(500, "int", "Max symbols per quote batch."),
        "MAX_WORKERS": _L(5, "int", "Parallel workers."),
        "RATE_LIMIT": _L(0.25, "float", "Seconds between requests."),
    },
    "position_monitor": {
        "TRAIL_POINTS": _L(5, "int", "Trail step in points."),
        "TRAIL_INTERVAL_MINUTES": _L(5, "int", "Trail evaluation interval."),
        "BTST_STEP_POINTS": _L(50, "int", "BTST limit price step per retry."),
        "BTST_WAIT_SECONDS": _L(2, "int", "Wait before checking fill."),
    },
    "expiry_calendar": {
        "NIFTY_WEEKLY_EXPIRY_WEEKDAY": _L(1, "int", "Python weekday: Nifty weekly expiry (0=Mon … 1=Tue)."),
        "BANKNIFTY_WEEKLY_EXPIRY_WEEKDAY": _L(4, "int", "Bank Nifty weekly expiry weekday (4=Fri)."),
    },
    "option_chain": {
        "OPTION_CHAIN_STRIKE_STEP": _L(100, "int", "Strike step when stepping chain (index strikes)."),
    },
    "feature_engineering": {
        "FUT_OI_CHANGE_THRESHOLD_PCT": _L(1.0, "float", "Futures OI change %% to flag up/down."),
    },
    "scoring": {
        "weights": {
            "spot_open_vwap_alignment": _L(0.18, "float", "Weight: spot vs open/VWAP alignment."),
            "fut_strength": _L(0.12, "float", "Weight: futures vs spot strength."),
            "fut_oi": _L(0.10, "float", "Weight: futures OI direction."),
            "options_expansion_decay": _L(0.15, "float", "Weight: ATM call/put premium expansion."),
            "options_oi": _L(0.10, "float", "Weight: CE/PE OI change."),
            "breakout_follow_through": _L(0.18, "float", "Weight: breakout with follow-through."),
            "momentum": _L(0.12, "float", "Weight: momentum direction."),
            "structure_quality": _L(0.05, "float", "Weight: structure / mid-range penalty bucket."),
        },
        "penalties": {
            "chop_open_vwap": _L(0.10, "float", "No-trade penalty: chop around open/VWAP."),
            "options_conflict": _L(0.08, "float", "Penalty: conflicting option premiums."),
            "no_follow_through": _L(0.08, "float", "Penalty: no follow-through on break."),
            "momentum_neutral": _L(0.05, "float", "Penalty: neutral momentum."),
            "stop_too_wide": _L(0.15, "float", "Penalty: stop too wide."),
        },
        "thresholds": {
            "fut_strength_pct": _L(0.03, "float", "Futures strength vs spot threshold (%%)."),
            "options_oi_change_abs": _L(2.0, "float", "CE/PE OI change threshold (%%) for directional OI signal."),
            "confidence_offset": _L(0.5, "float", "Added inside confidence = strength - penalty + offset."),
        },
    },
    "gamma_blast": {
        "PCR_BULLISH_MAX": _L(0.7, "float", "PCR below this → bullish (CE-heavy)."),
        "PCR_BEARISH_MIN": _L(1.3, "float", "PCR above this → bearish (PE-heavy)."),
        "CONFIDENCE_AFTER_1345": _L(0.7, "float", "Time factor after 13:45 on expiry day."),
        "CONFIDENCE_BEFORE_1345": _L(0.4, "float", "Time factor before 13:45."),
        "PCR_DEV_WEIGHT": _L(0.6, "float", "Weight of PCR deviation in confidence."),
        "TIME_FACTOR_WEIGHT": _L(0.4, "float", "Weight of time factor in confidence."),
    },
    "huge_move_predictor": {
        "HUGE_MOVE_POINTS": _L(100.0, "float", "Reference move size (points) for labeling."),
        "PCR_OI_EXTREME_BULL": _L(0.6, "float", "PCR OI below → extra bullish probability mass."),
        "PCR_OI_BULL": _L(0.7, "float", "PCR OI below → moderate bullish mass."),
        "PCR_OI_EXTREME_BEAR": _L(1.5, "float", "PCR OI above → extra bearish mass."),
        "PCR_OI_BEAR": _L(1.3, "float", "PCR OI above → moderate bearish mass."),
        "PCR_VOL_BULL": _L(0.5, "float", "PCR volume below → call buying."),
        "PCR_VOL_BEAR": _L(2.0, "float", "PCR volume above → put buying."),
        "PREMIUM_CHG_THRESHOLD": _L(15.0, "float", "ATM premium %% move threshold."),
        "MAX_PAIN_DEV_POINTS": _L(50.0, "float", "Spot vs max pain deviation (points) to add bias."),
        "NORM_TOTAL_OFFSET": _L(0.3, "float", "Normalization denominator offset."),
        "PROB_CAP": _L(0.7, "float", "Cap on normalized up/down probability."),
        "PROB_NO_FLOOR": _L(0.1, "float", "Floor on probability of no huge move."),
        "DIRECTION_CONF_THRESHOLD": _L(0.4, "float", "Min edge to call direction BULLISH/BEARISH."),
        "DEFAULT_NUM_STRIKES": _L(5, "int", "Strikes each side of spot for chain fetch."),
    },
    "zerodha_client": {
        "NFO_INSTRUMENTS_CACHE_HOURS": _L(6, "int", "Refresh NFO instruments cache after this many hours."),
    },
    "rsi": {
        "DEFAULT_PERIOD": _L(14, "int", "Default RSI period (Wilder)."),
    },
}
