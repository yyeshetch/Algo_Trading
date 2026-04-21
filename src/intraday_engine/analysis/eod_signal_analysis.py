from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd

from intraday_engine.storage import load_market_data, load_signal_rows
from intraday_engine.storage.layout import signal_outcomes_day_path

OUTCOME_COLUMNS = [
    "trade_date",
    "underlying",
    "asset_class",
    "strategy",
    "strategy_label",
    "signal",
    "signal_timestamp",
    "entry",
    "target",
    "stop_loss",
    "risk_points",
    "reward_points",
    "planned_rr",
    "confidence",
    "pattern",
    "fresh_until",
    "entry_triggered",
    "entry_time",
    "entry_candle_open",
    "exit_reason",
    "outcome",
    "exit_time",
    "exit_price",
    "exit_candle_open",
    "exit_candle_high",
    "exit_candle_low",
    "bars_to_entry",
    "bars_in_trade",
    "max_favorable_excursion",
    "max_adverse_excursion",
    "mfe_rr",
    "mae_rr",
    "pnl_points",
    "realized_rr",
    "close_price_eod",
    "same_candle_conflict",
]


def analyze_signal_outcomes(data_dir: Path, trade_date: date) -> tuple[pd.DataFrame, Path]:
    """Evaluate same-day signal outcomes using 15-minute candles."""
    signals = load_signal_rows(data_dir, trade_date=trade_date)
    signals = _filter_actionable_signals(signals)

    stock_lookup = _build_market_lookup(load_market_data(data_dir, trade_date=trade_date, asset_class="stock"))
    index_lookup = _build_market_lookup(
        load_market_data(data_dir, trade_date=trade_date, asset_class="index"),
        resample_15m=True,
    )

    rows: list[dict[str, Any]] = []
    for _, signal in signals.iterrows():
        asset_class = str(signal.get("asset_class") or "")
        underlying = str(signal.get("underlying") or "")
        candles = stock_lookup.get(underlying) if asset_class == "stock" else index_lookup.get(underlying)
        rows.append(_analyze_one_signal(signal, candles))

    result = pd.DataFrame(rows, columns=OUTCOME_COLUMNS)
    path = signal_outcomes_day_path(data_dir, trade_date)
    result.to_csv(path, index=False)
    return result, path


def _filter_actionable_signals(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    out = out[out["signal"].astype(str).isin(["BUY", "SELL"])] if "signal" in out.columns else out.iloc[0:0]
    for column in ("entry", "target", "stop_loss"):
        if column in out.columns:
            out = out[out[column].notna()]
    if "timestamp" in out.columns:
        out = out.sort_values(by=["timestamp", "underlying"], ascending=True).reset_index(drop=True)
    return out


def _build_market_lookup(df: pd.DataFrame, resample_15m: bool = False) -> dict[str, pd.DataFrame]:
    if df.empty or "underlying" not in df.columns:
        return {}
    lookup: dict[str, pd.DataFrame] = {}
    for underlying, group in df.groupby("underlying", sort=True):
        candles = _prepare_candles(group, resample_15m=resample_15m)
        if not candles.empty:
            lookup[str(underlying)] = candles
    return lookup


def _prepare_candles(df: pd.DataFrame, resample_15m: bool) -> pd.DataFrame:
    if df.empty or "timestamp" not in df.columns:
        return pd.DataFrame()

    frame = df.copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"])
    frame = frame.sort_values("timestamp").drop_duplicates(subset=["timestamp"], keep="last")
    open_col = "spot_open_raw" if "spot_open_raw" in frame.columns else "spot_open"
    if open_col not in frame.columns:
        open_col = "spot_close"

    candles = frame[["timestamp", open_col, "spot_high", "spot_low", "spot_close"]].rename(
        columns={open_col: "open", "spot_high": "high", "spot_low": "low", "spot_close": "close"}
    )
    if not resample_15m:
        return candles.reset_index(drop=True)

    candles = candles.set_index("timestamp")
    resampled = pd.DataFrame(
        {
            "open": candles["open"].resample(
                "15min", label="right", closed="right", origin="start_day", offset="9h15min"
            ).first(),
            "high": candles["high"].resample(
                "15min", label="right", closed="right", origin="start_day", offset="9h15min"
            ).max(),
            "low": candles["low"].resample(
                "15min", label="right", closed="right", origin="start_day", offset="9h15min"
            ).min(),
            "close": candles["close"].resample(
                "15min", label="right", closed="right", origin="start_day", offset="9h15min"
            ).last(),
        }
    ).dropna(subset=["close"])
    return resampled.reset_index()


def _analyze_one_signal(signal: pd.Series, candles: pd.DataFrame | None) -> dict[str, Any]:
    side = str(signal.get("signal") or "")
    entry = float(signal.get("entry") or 0)
    target = float(signal.get("target") or 0)
    stop = float(signal.get("stop_loss") or 0)
    signal_ts = pd.to_datetime(signal.get("timestamp"))
    risk = abs(entry - stop)
    reward = abs(target - entry)

    result = {
        "trade_date": str(signal.get("trade_date") or str(signal_ts.date())),
        "underlying": str(signal.get("underlying") or signal.get("stock") or ""),
        "asset_class": str(signal.get("asset_class") or ""),
        "strategy": str(signal.get("strategy") or ""),
        "strategy_label": str(signal.get("strategy_label") or ""),
        "signal": side,
        "signal_timestamp": signal_ts.isoformat(),
        "entry": entry,
        "target": target,
        "stop_loss": stop,
        "risk_points": round(risk, 4),
        "reward_points": round(reward, 4),
        "planned_rr": float(signal.get("rr") or 0),
        "confidence": float(signal.get("confidence") or 0),
        "pattern": signal.get("pattern"),
        "fresh_until": signal.get("fresh_until"),
        "entry_triggered": False,
        "entry_time": None,
        "entry_candle_open": None,
        "exit_reason": "NO_DATA",
        "outcome": "NO_DATA",
        "exit_time": None,
        "exit_price": None,
        "exit_candle_open": None,
        "exit_candle_high": None,
        "exit_candle_low": None,
        "bars_to_entry": None,
        "bars_in_trade": 0,
        "max_favorable_excursion": None,
        "max_adverse_excursion": None,
        "mfe_rr": None,
        "mae_rr": None,
        "pnl_points": None,
        "realized_rr": None,
        "close_price_eod": None,
        "same_candle_conflict": False,
    }
    if candles is None or candles.empty:
        return result

    future = candles[candles["timestamp"] > signal_ts].reset_index(drop=True)
    if future.empty:
        result["exit_reason"] = "NO_FORWARD_CANDLES"
        result["outcome"] = "NO_DATA"
        return result

    entry_idx = _find_entry_index(future, entry)
    last_row = future.iloc[-1]
    result["close_price_eod"] = round(float(last_row["close"]), 4)
    if entry_idx is None:
        result["exit_reason"] = "ENTRY_NOT_TRIGGERED"
        result["outcome"] = "NO_ENTRY"
        result["exit_time"] = last_row["timestamp"].isoformat()
        result["exit_price"] = round(float(last_row["close"]), 4)
        result["exit_candle_open"] = round(float(last_row["open"]), 4)
        result["exit_candle_high"] = round(float(last_row["high"]), 4)
        result["exit_candle_low"] = round(float(last_row["low"]), 4)
        return result

    entry_row = future.iloc[entry_idx]
    active = future.iloc[entry_idx:].reset_index(drop=True)
    result["entry_triggered"] = True
    result["entry_time"] = entry_row["timestamp"].isoformat()
    result["entry_candle_open"] = round(float(entry_row["open"]), 4)
    result["bars_to_entry"] = int(entry_idx + 1)

    mfe = 0.0
    mae = 0.0
    for idx, row in active.iterrows():
        high = float(row["high"])
        low = float(row["low"])
        if side == "BUY":
            mfe = max(mfe, high - entry)
            mae = max(mae, entry - low)
            hit_stop = low <= stop
            hit_target = high >= target
        else:
            mfe = max(mfe, entry - low)
            mae = max(mae, high - entry)
            hit_stop = high >= stop
            hit_target = low <= target

        if hit_stop or hit_target:
            result["same_candle_conflict"] = bool(hit_stop and hit_target)
            result["exit_reason"] = "STOP_LOSS" if hit_stop else "TARGET"
            result["outcome"] = "SL" if hit_stop else "TGT"
            result["exit_time"] = row["timestamp"].isoformat()
            result["exit_price"] = round(stop if hit_stop else target, 4)
            result["exit_candle_open"] = round(float(row["open"]), 4)
            result["exit_candle_high"] = round(high, 4)
            result["exit_candle_low"] = round(low, 4)
            result["bars_in_trade"] = int(idx + 1)
            break

    if result["exit_time"] is None:
        exit_price = float(active.iloc[-1]["close"])
        pnl_points = (exit_price - entry) if side == "BUY" else (entry - exit_price)
        result["exit_reason"] = "DAY_CLOSE"
        result["outcome"] = "BREAKEVEN" if abs(pnl_points) <= max(risk * 0.1, 0.05) else "DAY_CLOSE"
        result["exit_time"] = active.iloc[-1]["timestamp"].isoformat()
        result["exit_price"] = round(exit_price, 4)
        result["exit_candle_open"] = round(float(active.iloc[-1]["open"]), 4)
        result["exit_candle_high"] = round(float(active.iloc[-1]["high"]), 4)
        result["exit_candle_low"] = round(float(active.iloc[-1]["low"]), 4)
        result["bars_in_trade"] = int(len(active))

    exit_price = float(result["exit_price"] or 0)
    pnl_points = (exit_price - entry) if side == "BUY" else (entry - exit_price)
    result["max_favorable_excursion"] = round(mfe, 4)
    result["max_adverse_excursion"] = round(mae, 4)
    result["mfe_rr"] = round(mfe / risk, 4) if risk > 0 else None
    result["mae_rr"] = round(mae / risk, 4) if risk > 0 else None
    result["pnl_points"] = round(pnl_points, 4)
    result["realized_rr"] = round(pnl_points / risk, 4) if risk > 0 else None
    return result


def _find_entry_index(candles: pd.DataFrame, entry: float) -> int | None:
    for idx, row in candles.iterrows():
        if float(row["low"]) <= entry <= float(row["high"]):
            return int(idx)
    return None
