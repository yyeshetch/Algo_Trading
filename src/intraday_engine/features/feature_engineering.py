from __future__ import annotations

from typing import Dict

import pandas as pd

from intraday_engine.core.models import MarketSnapshot


def snapshot_to_frame(snapshot: MarketSnapshot, history: pd.DataFrame) -> pd.DataFrame:
    row = pd.DataFrame([snapshot.to_record()])
    merged = pd.concat([history, row], ignore_index=True)
    return merged


def compute_features(df: pd.DataFrame) -> Dict[str, float]:
    last = df.iloc[-1]
    prev = df.iloc[-2] if len(df) > 1 else last
    first = df.iloc[0]

    spot_ltp = float(last["spot_ltp"])
    spot_open = float(last["spot_open"])
    spot_vwap = float(last["spot_vwap"])
    future_ltp = float(last["future_ltp"])

    call_ltp = float(last.get("call_ltp", 0) or 0)
    put_ltp = float(last.get("put_ltp", 0) or 0)
    prev_call = float(prev.get("call_ltp", 0) or 0)
    prev_put = float(prev.get("put_ltp", 0) or 0)
    prev_spot = float(prev["spot_ltp"])

    call_change = _safe_pct_change(call_ltp, prev_call)
    put_change = _safe_pct_change(put_ltp, prev_put)
    spot_change = _safe_pct_change(spot_ltp, prev_spot)

    options_available = call_ltp > 0 or put_ltp > 0

    # OI: session change (first vs last candle)
    fut_oi_first = float(first.get("future_oi", 0) or 0)
    fut_oi_last = float(last.get("future_oi", 0) or 0)
    call_oi_first = float(first.get("call_oi", 0) or 0)
    call_oi_last = float(last.get("call_oi", 0) or 0)
    put_oi_first = float(first.get("put_oi", 0) or 0)
    put_oi_last = float(last.get("put_oi", 0) or 0)

    fut_oi_change_pct = _safe_pct_change(fut_oi_last, fut_oi_first)
    call_oi_change_pct = _safe_pct_change(call_oi_last, call_oi_first)
    put_oi_change_pct = _safe_pct_change(put_oi_last, put_oi_first)

    # Futures OI + price interpretation: OI up + price up = longs adding (bullish)
    from intraday_engine.core.tunables import get_float

    fut_oi_available = fut_oi_first > 0 and fut_oi_last > 0
    oi_th = get_float("feature_engineering", "FUT_OI_CHANGE_THRESHOLD_PCT", 1.0)
    fut_oi_up = fut_oi_change_pct > oi_th
    fut_oi_down = fut_oi_change_pct < -oi_th
    first_fut = float(first.get("future_ltp", 0) or first.get("future_close", 0) or 0)
    fut_price_up = first_fut > 0 and future_ltp > first_fut
    fut_price_down = first_fut > 0 and future_ltp < first_fut
    fut_oi_bullish = (fut_oi_up and fut_price_up) or (fut_oi_down and fut_price_up)  # longs adding or short covering
    fut_oi_bearish = (fut_oi_up and fut_price_down) or (fut_oi_down and fut_price_down)  # shorts adding or long covering

    # Options OI: CE OI up = bullish, PE OI up = bearish
    oi_available = (call_oi_first > 0 or call_oi_last > 0) and (put_oi_first > 0 or put_oi_last > 0)

    return {
        "spot_above_open": 1.0 if spot_ltp > spot_open else 0.0,
        "spot_below_open": 1.0 if spot_ltp < spot_open else 0.0,
        "spot_above_vwap": 1.0 if spot_ltp > spot_vwap else 0.0,
        "spot_below_vwap": 1.0 if spot_ltp < spot_vwap else 0.0,
        "fut_strength_pct": _safe_pct_change(future_ltp, spot_ltp),
        "call_change_pct": call_change,
        "put_change_pct": put_change,
        "spot_change_pct": spot_change,
        "options_available": options_available,
        "fut_oi_change_pct": fut_oi_change_pct,
        "call_oi_change_pct": call_oi_change_pct,
        "put_oi_change_pct": put_oi_change_pct,
        "fut_oi_available": fut_oi_available,
        "fut_oi_bullish": 1.0 if fut_oi_available and fut_oi_bullish else 0.0,
        "fut_oi_bearish": 1.0 if fut_oi_available and fut_oi_bearish else 0.0,
        "oi_available": oi_available,
    }


def _safe_pct_change(current: float, base: float) -> float:
    if base == 0:
        return 0.0
    return ((current - base) / base) * 100

