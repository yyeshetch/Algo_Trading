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

    spot_ltp = float(last["spot_ltp"])
    spot_open = float(last["spot_open"])
    spot_vwap = float(last["spot_vwap"])
    future_ltp = float(last["future_ltp"])

    call_ltp = float(last["call_ltp"])
    put_ltp = float(last["put_ltp"])
    prev_call = float(prev["call_ltp"])
    prev_put = float(prev["put_ltp"])
    prev_spot = float(prev["spot_ltp"])

    call_change = _safe_pct_change(call_ltp, prev_call)
    put_change = _safe_pct_change(put_ltp, prev_put)
    spot_change = _safe_pct_change(spot_ltp, prev_spot)

    return {
        "spot_above_open": 1.0 if spot_ltp > spot_open else 0.0,
        "spot_below_open": 1.0 if spot_ltp < spot_open else 0.0,
        "spot_above_vwap": 1.0 if spot_ltp > spot_vwap else 0.0,
        "spot_below_vwap": 1.0 if spot_ltp < spot_vwap else 0.0,
        "fut_strength_pct": _safe_pct_change(future_ltp, spot_ltp),
        "call_change_pct": call_change,
        "put_change_pct": put_change,
        "spot_change_pct": spot_change,
    }


def _safe_pct_change(current: float, base: float) -> float:
    if base == 0:
        return 0.0
    return ((current - base) / base) * 100

