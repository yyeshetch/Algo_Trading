"""Compute scanner metrics from 15-min merged data."""

from __future__ import annotations

from typing import Any

import pandas as pd


def compute_stock_metrics(data: dict[str, Any]) -> dict[str, Any]:
    """
    Compute metrics from fetched stock data.
    data: dict with keys stock, df, spot_open, spot_close
    """
    df: pd.DataFrame = data["df"]
    if df.empty or len(df) < 2:
        return {}

    first = df.iloc[0]
    last = df.iloc[-1]

    spot_open = float(first.get("spot_open_raw", 0) or 0)
    spot_close = float(last.get("spot_close", 0) or 0)
    spot_volume = float(df["spot_volume"].sum()) if "spot_volume" in df.columns else 0
    fut_volume = float(df["future_volume"].sum()) if "future_volume" in df.columns else 0

    call_open = float(first.get("call_open_raw", 0) or first.get("call_close", 0) or 0)
    call_close = float(last.get("call_close", 0) or 0)
    put_open = float(first.get("put_open_raw", 0) or first.get("put_close", 0) or 0)
    put_close = float(last.get("put_close", 0) or 0)

    call_oi_first = float(first.get("call_oi", 0) or 0)
    call_oi_last = float(last.get("call_oi", 0) or 0)
    put_oi_first = float(first.get("put_oi", 0) or 0)
    put_oi_last = float(last.get("put_oi", 0) or 0)

    spot_change_pct = ((spot_close - spot_open) / spot_open * 100) if spot_open else 0
    call_premium_change_pct = ((call_close - call_open) / call_open * 100) if call_open else 0
    put_premium_change_pct = ((put_close - put_open) / put_open * 100) if put_open else 0

    call_oi_change = call_oi_last - call_oi_first
    put_oi_change = put_oi_last - put_oi_first
    call_oi_change_pct = ((call_oi_last - call_oi_first) / call_oi_first * 100) if call_oi_first else 0
    put_oi_change_pct = ((put_oi_last - put_oi_first) / put_oi_first * 100) if put_oi_first else 0

    return {
        "stock": data["stock"],
        "spot_open": round(spot_open, 2),
        "spot_close": round(spot_close, 2),
        "spot_change_pct": round(spot_change_pct, 2),
        "spot_volume": int(spot_volume),
        "futures_volume": int(fut_volume),
        "call_oi_change": int(call_oi_change),
        "put_oi_change": int(put_oi_change),
        "call_oi_change_pct": round(call_oi_change_pct, 2),
        "put_oi_change_pct": round(put_oi_change_pct, 2),
        "call_premium_change_pct": round(call_premium_change_pct, 2),
        "put_premium_change_pct": round(put_premium_change_pct, 2),
        "call_open": round(call_open, 2),
        "call_close": round(call_close, 2),
        "put_open": round(put_open, 2),
        "put_close": round(put_close, 2),
    }
