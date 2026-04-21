from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import pandas as pd

from intraday_engine.storage.layout import (
    analysis_day_path,
    asset_class_for_underlying,
    normalize_underlying,
    option_chain_day_path,
    signals_day_path,
)

LEGACY_RESERVED_DIRS = {
    "analysis",
    "reference",
    "research",
    "signals",
    "market_data",
    "option_chain",
}


@dataclass
class MigrationReport:
    signal_rows: int = 0
    snapshot_rows: int = 0
    option_chain_rows: int = 0
    deleted_files: int = 0
    deleted_dirs: int = 0


@dataclass
class RewriteReport:
    signal_rows: int = 0
    analysis_rows: int = 0
    deleted_files: int = 0
    deleted_dirs: int = 0


def migrate_legacy_data(data_dir: Path) -> MigrationReport:
    report = MigrationReport()

    legacy_signal_files = _legacy_files(data_dir, "signals.csv")
    legacy_snapshot_files = _legacy_files(data_dir, "snapshots.csv")
    legacy_signal_jsonl = _legacy_files(data_dir, "signals.jsonl")
    legacy_option_chain_files = _legacy_option_chain_files(data_dir)
    removable_backups = _removable_backup_files(data_dir)

    for path in legacy_signal_files:
        report.signal_rows += _migrate_legacy_signal_file(data_dir, path)
    for path in legacy_snapshot_files:
        report.snapshot_rows += _migrate_legacy_snapshot_file(data_dir, path)
    for path in legacy_option_chain_files:
        report.option_chain_rows += _migrate_legacy_option_chain_file(data_dir, path)

    files_to_delete = legacy_signal_files + legacy_snapshot_files + legacy_signal_jsonl + legacy_option_chain_files + removable_backups
    for path in files_to_delete:
        if path.exists():
            path.unlink()
            report.deleted_files += 1

    for folder in _legacy_underlying_dirs(data_dir):
        if folder.exists() and not any(folder.iterdir()):
            folder.rmdir()
            report.deleted_dirs += 1

    return report


def _legacy_files(data_dir: Path, filename: str) -> list[Path]:
    paths: list[Path] = []
    root_file = data_dir / filename
    if root_file.exists():
        paths.append(root_file)
    for folder in _legacy_underlying_dirs(data_dir):
        path = folder / filename
        if path.exists():
            paths.append(path)
    return paths


def _legacy_underlying_dirs(data_dir: Path) -> list[Path]:
    out: list[Path] = []
    for path in data_dir.iterdir():
        if not path.is_dir():
            continue
        if path.name.startswith(".") or path.name in LEGACY_RESERVED_DIRS:
            continue
        out.append(path)
    return out


def _legacy_option_chain_files(data_dir: Path) -> list[Path]:
    paths = list(data_dir.glob("option_chain_*.jsonl"))
    for folder in _legacy_underlying_dirs(data_dir):
        paths.extend(folder.glob("option_chain_*.jsonl"))
    return sorted(paths)


def _removable_backup_files(data_dir: Path) -> list[Path]:
    patterns = [
        "signals_bkp.csv",
        "fno_intraday_signals_*_bkp.json",
        "fno_intraday_signals_*_old.json",
    ]
    paths: list[Path] = []
    for pattern in patterns:
        paths.extend(sorted(data_dir.glob(pattern)))
    return paths


def _migrate_legacy_signal_file(data_dir: Path, path: Path) -> int:
    df = _safe_read_csv(path)
    if df.empty or "timestamp" not in df.columns:
        return 0
    underlying = _infer_underlying_from_path(data_dir, path)
    asset_class = asset_class_for_underlying(underlying)
    rows = 0
    df = df.copy()
    df["underlying"] = underlying
    df["asset_class"] = asset_class
    df["trade_date"] = df["timestamp"].astype(str).str.slice(0, 10)

    for trade_date_str, group in df.groupby("trade_date", sort=True):
        trade_date = date.fromisoformat(trade_date_str)
        target = signals_day_path(data_dir, trade_date, asset_class)
        existing = _safe_read_csv(target) if target.exists() else pd.DataFrame()
        combined = pd.concat([existing, group], ignore_index=True)
        subset = [col for col in ["underlying", "timestamp", "strategy", "strategy_label", "signal"] if col in combined.columns]
        if subset:
            combined = combined.drop_duplicates(subset=subset, keep="last")
        combined.to_csv(target, index=False)
        rows += len(group)
    return rows


def _migrate_legacy_snapshot_file(data_dir: Path, path: Path) -> int:
    df = _safe_read_csv(path)
    if df.empty or "timestamp" not in df.columns:
        return 0
    underlying = _infer_underlying_from_path(data_dir, path)
    asset_class = asset_class_for_underlying(underlying)
    rows = 0
    df = df.copy()
    df["underlying"] = underlying
    df["asset_class"] = asset_class
    df["trade_date"] = df["timestamp"].astype(str).str.slice(0, 10)

    for trade_date_str, group in df.groupby("trade_date", sort=True):
        trade_date = date.fromisoformat(trade_date_str)
        target = analysis_day_path(data_dir, trade_date, asset_class)
        existing = _safe_read_csv(target) if target.exists() else pd.DataFrame()
        combined = pd.concat([existing, group], ignore_index=True)
        subset = [col for col in ["underlying", "timestamp"] if col in combined.columns]
        if subset:
            combined = combined.drop_duplicates(subset=subset, keep="last")
        combined.to_csv(target, index=False)
        rows += len(group)
    return rows


def _migrate_legacy_option_chain_file(data_dir: Path, path: Path) -> int:
    underlying = _infer_underlying_from_path(data_dir, path)
    rows: list[dict] = []
    with path.open(encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            trade_date_value = str(record.get("trade_date", "")).strip()
            if not trade_date_value:
                match = re.search(r"option_chain_(\d{4}-\d{2}-\d{2})\.jsonl$", path.name)
                if not match:
                    continue
                trade_date_value = match.group(1)
            strikes = record.get("strikes", [])
            for strike in strikes:
                if not isinstance(strike, dict):
                    continue
                rows.append(
                    {
                        "timestamp": record.get("timestamp"),
                        "trade_date": trade_date_value,
                        "underlying": underlying,
                        "spot_price": record.get("spot_price"),
                        "atm_strike": record.get("atm_strike"),
                        "expiry": record.get("expiry"),
                        "spot_volume": record.get("spot_volume"),
                        "tradingsymbol": strike.get("tradingsymbol"),
                        "strike": strike.get("strike"),
                        "option_type": strike.get("option_type"),
                        "oi": strike.get("oi"),
                        "volume": strike.get("volume"),
                        "ltp": strike.get("ltp"),
                        "open": strike.get("open"),
                        "high": strike.get("high"),
                        "low": strike.get("low"),
                        "close": strike.get("close"),
                    }
                )
    if not rows:
        return 0

    df = pd.DataFrame(rows)
    migrated = 0
    for trade_date_str, group in df.groupby("trade_date", sort=True):
        trade_date = date.fromisoformat(str(trade_date_str)[:10])
        target = option_chain_day_path(data_dir, trade_date)
        existing = _safe_read_csv(target) if target.exists() else pd.DataFrame()
        combined = pd.concat([existing, group], ignore_index=True)
        subset = [col for col in ["underlying", "timestamp", "tradingsymbol"] if col in combined.columns]
        if subset:
            combined = combined.drop_duplicates(subset=subset, keep="last")
        combined.to_csv(target, index=False)
        migrated += len(group)
    return migrated


def _infer_underlying_from_path(data_dir: Path, path: Path) -> str:
    if path.parent == data_dir:
        if path.name.startswith("option_chain_"):
            return "NIFTY"
        return "NIFTY"
    return normalize_underlying(path.parent.name)


def _safe_read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, low_memory=False)


def rewrite_partitioned_storage_layout(data_dir: Path) -> RewriteReport:
    report = RewriteReport()

    for path in sorted((data_dir / "signals").glob("date=*/signals.csv")):
        df = _safe_read_csv(path)
        if not df.empty:
            report.signal_rows += _rewrite_signals_partition(data_dir, path, df)
        if path.exists():
            path.unlink()
            report.deleted_files += 1

    market_data_root = data_dir / "market_data"
    for path in sorted(market_data_root.glob("stocks/date=*/snapshots.csv")):
        df = _safe_read_csv(path)
        if not df.empty:
            report.analysis_rows += _rewrite_analysis_partition(data_dir, path, df, "stock")
        if path.exists():
            path.unlink()
            report.deleted_files += 1

    for path in sorted(market_data_root.glob("indexes/date=*/snapshots.csv")):
        df = _safe_read_csv(path)
        if not df.empty:
            report.analysis_rows += _rewrite_analysis_partition(data_dir, path, df, "index")
        if path.exists():
            path.unlink()
            report.deleted_files += 1

    for folder in sorted(market_data_root.glob("**/*"), reverse=True):
        if folder.is_dir():
            try:
                folder.rmdir()
                report.deleted_dirs += 1
            except OSError:
                pass

    return report


def _rewrite_signals_partition(data_dir: Path, path: Path, df: pd.DataFrame) -> int:
    trade_date = date.fromisoformat(path.parent.name.replace("date=", ""))
    rows = 0

    index_df = _filter_signal_partition(df, "index", keep_all=True)
    if not index_df.empty:
        target = signals_day_path(data_dir, trade_date, "index")
        existing = _safe_read_csv(target) if target.exists() else pd.DataFrame()
        combined = pd.concat([existing, index_df], ignore_index=True)
        subset = [col for col in ["underlying", "timestamp", "strategy", "strategy_label", "signal"] if col in combined.columns]
        if subset:
            combined = combined.drop_duplicates(subset=subset, keep="last")
        combined.to_csv(target, index=False)
        rows += len(index_df)

    stock_df = _filter_signal_partition(df, "stock", keep_all=False)
    if not stock_df.empty:
        target = signals_day_path(data_dir, trade_date, "stock")
        existing = _safe_read_csv(target) if target.exists() else pd.DataFrame()
        combined = pd.concat([existing, stock_df], ignore_index=True)
        subset = [col for col in ["underlying", "timestamp", "strategy", "strategy_label", "signal"] if col in combined.columns]
        if subset:
            combined = combined.drop_duplicates(subset=subset, keep="last")
        combined.to_csv(target, index=False)
        rows += len(stock_df)

    return rows


def _filter_signal_partition(df: pd.DataFrame, asset_class: str, keep_all: bool) -> pd.DataFrame:
    out = df.copy()
    if "asset_class" in out.columns:
        out = out[out["asset_class"].astype(str) == asset_class]
    elif "underlying" in out.columns:
        out = out[out["underlying"].astype(str).map(asset_class_for_underlying) == asset_class]
    if not keep_all and "signal" in out.columns:
        out = out[out["signal"].astype(str).isin(["BUY", "SELL"])]
    return out


def _rewrite_analysis_partition(data_dir: Path, path: Path, df: pd.DataFrame, asset_class: str) -> int:
    trade_date = date.fromisoformat(path.parent.name.replace("date=", ""))
    target = analysis_day_path(data_dir, trade_date, asset_class)
    existing = _safe_read_csv(target) if target.exists() else pd.DataFrame()
    combined = pd.concat([existing, df], ignore_index=True)
    subset = [col for col in ["underlying", "timestamp"] if col in combined.columns]
    if subset:
        combined = combined.drop_duplicates(subset=subset, keep="last")
    combined.to_csv(target, index=False)
    return len(df)
