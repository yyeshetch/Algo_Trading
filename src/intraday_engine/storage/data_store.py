from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Dict

import pandas as pd


class DataStore:
    def __init__(self, data_dir: Path) -> None:
        self.data_dir = data_dir
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.snapshots_csv = self.data_dir / "snapshots.csv"
        self.signals_csv = self.data_dir / "signals.csv"
        self.signals_jsonl = self.data_dir / "signals.jsonl"

    def load_snapshots(self) -> pd.DataFrame:
        if self.snapshots_csv.exists():
            return pd.read_csv(self.snapshots_csv)
        return pd.DataFrame()

    def save_snapshots(self, df: pd.DataFrame) -> pd.DataFrame:
        df.to_csv(self.snapshots_csv, index=False)
        return df

    def load_signals(self) -> pd.DataFrame:
        if not self.signals_csv.exists():
            return pd.DataFrame()
        return pd.read_csv(self.signals_csv)

    def get_latest_actionable_signal(self, trade_date: date | None = None) -> Dict[str, object] | None:
        df = self.load_signals()
        if df.empty or "signal" not in df.columns:
            return None
        if trade_date is not None and "timestamp" in df.columns:
            today_str = trade_date.isoformat()
            df = df[df["timestamp"].astype(str).str.startswith(today_str)]
        if df.empty:
            return None
        actionable = df[df["signal"].isin(["BUY", "SELL"])]
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

    def load_signal_timestamps(self) -> set[str]:
        if not self.signals_csv.exists():
            return set()
        df = pd.read_csv(self.signals_csv)
        if "timestamp" not in df.columns:
            return set()
        return set(df["timestamp"].astype(str).tolist())

    def append_snapshot(self, record: Dict[str, object]) -> pd.DataFrame:
        df = self.load_snapshots()
        next_df = pd.concat([df, pd.DataFrame([record])], ignore_index=True)
        next_df.to_csv(self.snapshots_csv, index=False)
        return next_df

    def append_signal(self, payload: Dict[str, object]) -> None:
        csv_row = pd.DataFrame([_flatten_for_csv(payload)])
        if self.signals_csv.exists():
            existing = pd.read_csv(self.signals_csv)
            combined = pd.concat([existing, csv_row], ignore_index=True)
            combined.to_csv(self.signals_csv, index=False)
        else:
            csv_row.to_csv(self.signals_csv, index=False)

        with self.signals_jsonl.open("a", encoding="utf-8") as fp:
            fp.write(json.dumps(payload, ensure_ascii=True) + "\n")


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
