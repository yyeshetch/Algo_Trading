from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Dict

import pandas as pd

from intraday_engine.storage.layout import (
    analysis_day_path,
    asset_class_for_underlying,
    normalize_underlying,
    signals_day_path,
)
from intraday_engine.storage.signal_invalidation_store import load_invalidated_keys

# FnO stock rows persisted to CSV (beyond plain BUY/SELL).
_STOCK_SIGNALS_PERSISTED = frozenset({"BUY", "SELL", "OVERBROUGHT_LOOK_FOR_REVERSAL"})


class DataStore:
    def __init__(self, data_dir: Path, underlying: str | None = None) -> None:
        self.root_data_dir = data_dir
        self.underlying = normalize_underlying(underlying)
        self.asset_class = asset_class_for_underlying(self.underlying)
        # Keep this path stable for position SL and other per-underlying utilities.
        if self.underlying == "NIFTY":
            self.data_dir = data_dir
        else:
            self.data_dir = data_dir / self.underlying.lower()
        # Per-underlying dirs (e.g. position SL) are created on first write, not on every DataStore().

    def load_snapshots(self, trade_date: date | None = None) -> pd.DataFrame:
        return load_market_data(self.root_data_dir, trade_date, self.asset_class, self.underlying)

    def save_snapshots(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df
        trade_date = _infer_trade_date_from_frame(df)
        path = analysis_day_path(self.root_data_dir, trade_date, self.asset_class)
        existing = _read_csv(path)
        prepared = _prepare_snapshot_df(df, self.underlying, self.asset_class, trade_date)
        if not existing.empty:
            existing = existing[existing["underlying"].astype(str) != self.underlying]
        combined = pd.concat([existing, prepared], ignore_index=True)
        _write_csv_or_delete(path, combined)
        return df

    def load_signals(self, trade_date: date | None = None) -> pd.DataFrame:
        return load_signal_rows(self.root_data_dir, trade_date, self.underlying)

    def get_latest_actionable_signal(self, trade_date: date | None = None) -> Dict[str, object] | None:
        df = self.load_signals(trade_date=trade_date)
        if df.empty or "signal" not in df.columns:
            return None
        actionable = df[df["signal"].isin(["BUY", "SELL"])]
        if actionable.empty:
            return None
        if self.asset_class == "index":
            td = (trade_date or date.today()).isoformat()
            inv = load_invalidated_keys(self.root_data_dir, self.underlying, td)

            def _inv_key(row: pd.Series) -> str:
                return f"{str(row.get('timestamp', '')).strip()}|{str(row.get('signal', '')).strip().upper()}"

            mask = ~actionable.apply(lambda r: _inv_key(r) in inv, axis=1)
            actionable = actionable[mask]
        if actionable.empty:
            return None
        # Latest by timestamp (descending)
        if "timestamp" in actionable.columns:
            actionable = actionable.sort_values(by="timestamp", ascending=False)
        row = actionable.iloc[0]
        out = {}
        for k, v in row.items():
            if hasattr(v, "item") and not isinstance(v, (str, bool)):
                try:
                    out[k] = v.item()
                except (ValueError, AttributeError):
                    out[k] = v
            else:
                out[k] = v
        return out

    def load_signal_timestamps(self, trade_date: date | None = None) -> set[str]:
        df = self.load_signals(trade_date=trade_date or date.today())
        if "timestamp" not in df.columns:
            return set()
        return set(df["timestamp"].astype(str).tolist())

    def append_snapshot(self, record: Dict[str, object]) -> pd.DataFrame:
        next_df = pd.concat([self.load_snapshots(), pd.DataFrame([record])], ignore_index=True)
        return self.save_snapshots(next_df)

    def append_signal(self, payload: Dict[str, object]) -> None:
        if self.asset_class == "stock" and str(payload.get("signal") or "") not in _STOCK_SIGNALS_PERSISTED:
            return
        trade_date = _infer_trade_date_from_value(payload.get("timestamp"))
        path = signals_day_path(self.root_data_dir, trade_date, self.asset_class)
        existing = _read_csv(path)
        prepared = _prepare_signal_df([payload], self.underlying, self.asset_class, trade_date)
        combined = pd.concat([existing, prepared], ignore_index=True)
        _write_csv_or_delete(path, combined)

    def replace_signals_for_date(self, trade_date: date, payloads: list[Dict[str, object]]) -> None:
        filtered_payloads = (
            [p for p in payloads if str(p.get("signal") or "") in _STOCK_SIGNALS_PERSISTED]
            if self.asset_class == "stock"
            else payloads
        )
        path = signals_day_path(self.root_data_dir, trade_date, self.asset_class)
        existing = _read_csv(path)
        if not existing.empty:
            existing = existing[existing["underlying"].astype(str) != self.underlying]
        prepared = _prepare_signal_df(filtered_payloads, self.underlying, self.asset_class, trade_date)
        combined = pd.concat([existing, prepared], ignore_index=True)
        _write_csv_or_delete(path, combined)


def load_signal_rows(
    data_dir: Path,
    trade_date: date | None = None,
    underlying: str | None = None,
    asset_class: str | None = None,
) -> pd.DataFrame:
    filenames = [f for f in ["FnO_Signals.csv", "Index_Signals.csv"] if asset_class is None or f.startswith("Index" if asset_class == "index" else "FnO")]
    frames = [
        _read_csv(path)
        for filename in filenames
        for path in _partition_paths(data_dir / "signals", filename, trade_date)
    ]
    df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    return _filter_rows(df, underlying=underlying, asset_class=asset_class)


def load_market_data(
    data_dir: Path,
    trade_date: date | None = None,
    asset_class: str | None = None,
    underlying: str | None = None,
) -> pd.DataFrame:
    cls = asset_class or asset_class_for_underlying(underlying)
    frames = [
        _read_csv(path)
        for path in _partition_paths(
            data_dir / "analysis",
            "Index_Analysis.csv" if cls == "index" else "FnO_Analysis.csv",
            trade_date,
        )
    ]
    df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    return _filter_rows(df, underlying=underlying, asset_class=cls)


def _partition_paths(root: Path, filename: str, trade_date: date | None) -> list[Path]:
    if trade_date is not None:
        path = root / f"date={trade_date.isoformat()}" / filename
        return [path] if path.exists() else []
    return sorted(root.glob(f"date=*/{filename}"))


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def _write_csv_or_delete(path: Path, df: pd.DataFrame) -> None:
    if df.empty:
        path.unlink(missing_ok=True)
        return
    df.to_csv(path, index=False)


def _prepare_snapshot_df(
    df: pd.DataFrame,
    underlying: str,
    asset_class: str,
    trade_date: date,
) -> pd.DataFrame:
    out = df.copy()
    out["underlying"] = underlying
    out["asset_class"] = asset_class
    out["trade_date"] = trade_date.isoformat()
    return out


def _prepare_signal_df(
    payloads: list[Dict[str, object]],
    underlying: str,
    asset_class: str,
    trade_date: date,
) -> pd.DataFrame:
    rows = [_flatten_for_csv(p) for p in payloads]
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df["underlying"] = underlying
    df["asset_class"] = asset_class
    df["trade_date"] = trade_date.isoformat()
    return _dedupe_signal_df(df)


def _dedupe_signal_df(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    if "timestamp" in out.columns:
        out = out.sort_values(by="timestamp", ascending=True).reset_index(drop=True)
    dedupe_key = out["trigger_timestamp"] if "trigger_timestamp" in out.columns else pd.Series([None] * len(out))
    if "timestamp" in out.columns:
        dedupe_key = dedupe_key.fillna(out["timestamp"])
    out["_dedupe_signal_key"] = dedupe_key
    subset = [col for col in ["underlying", "strategy", "strategy_label", "signal"] if col in out.columns]
    subset.append("_dedupe_signal_key")
    out = out.drop_duplicates(subset=subset, keep="first")
    return out.drop(columns=["_dedupe_signal_key"]).reset_index(drop=True)


def _infer_trade_date_from_frame(df: pd.DataFrame) -> date:
    if df.empty or "timestamp" not in df.columns:
        return date.today()
    return _infer_trade_date_from_value(df.iloc[0]["timestamp"])


def _infer_trade_date_from_value(value: object) -> date:
    text = str(value or "").strip()
    if len(text) >= 10:
        try:
            return date.fromisoformat(text[:10])
        except ValueError:
            pass
    return date.today()


def _filter_rows(
    df: pd.DataFrame,
    underlying: str | None = None,
    asset_class: str | None = None,
) -> pd.DataFrame:
    if df.empty:
        return df
    out = df
    if underlying is not None and "underlying" in out.columns:
        out = out[out["underlying"].astype(str) == normalize_underlying(underlying)]
    if asset_class is not None and "asset_class" in out.columns:
        out = out[out["asset_class"].astype(str) == asset_class]
    if "timestamp" in out.columns:
        out = out.sort_values(by="timestamp", ascending=True).reset_index(drop=True)
    return out


def _flatten_for_csv(payload: Dict[str, object]) -> Dict[str, object]:
    row = dict(payload)
    score = row.pop("score", {})
    notes = row.pop("notes", [])
    if isinstance(score, dict):
        row["bullish_score"] = score.get("bullish")
        row["bearish_score"] = score.get("bearish")
        row["no_trade_penalty"] = score.get("no_trade_penalty")
        row["final_score"] = score.get("final_score")
        row["confidence"] = score.get("confidence")
        row["reasons"] = " | ".join(score.get("reasons", []))
    row["notes"] = " | ".join(notes) if isinstance(notes, list) else str(notes)
    return row
