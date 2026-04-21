from __future__ import annotations

from datetime import date
from pathlib import Path

from intraday_engine.core.underlyings import list_index_underlyings


def normalize_underlying(underlying: str | None) -> str:
    value = (underlying or "NIFTY").strip().upper().replace(" ", "")
    return "BANKNIFTY" if value == "NIFTYBANK" else value


def asset_class_for_underlying(underlying: str | None) -> str:
    return "index" if normalize_underlying(underlying) in set(list_index_underlyings()) else "stock"


def signals_day_path(data_dir: Path, trade_date: date, asset_class: str) -> Path:
    filename = "Index_Signals.csv" if asset_class == "index" else "FnO_Signals.csv"
    return _partition_dir(data_dir / "signals", trade_date) / filename


def analysis_day_path(data_dir: Path, trade_date: date, asset_class: str) -> Path:
    filename = "Index_Analysis.csv" if asset_class == "index" else "FnO_Analysis.csv"
    return _partition_dir(data_dir / "analysis", trade_date) / filename


def option_chain_day_path(data_dir: Path, trade_date: date) -> Path:
    return _partition_dir(data_dir / "option_chain", trade_date) / "option_chain.csv"


def signal_outcomes_day_path(data_dir: Path, trade_date: date) -> Path:
    return _partition_dir(data_dir / "analysis", trade_date) / "signal_outcomes.csv"


def watchlist_csv_path(data_dir: Path) -> Path:
    return data_dir / "reference" / "fno_watchlist.csv"


def accumulation_base_dir(data_dir: Path) -> Path:
    """Shared folder for NIFTY500 accumulation scanner (single-file OHLCV cache)."""
    p = data_dir / "analysis" / "accumulation"
    p.mkdir(parents=True, exist_ok=True)
    return p


def accumulation_ohlcv_master_path(data_dir: Path) -> Path:
    """Single incremental JSON store: symbol -> list of 1h OHLCV bars."""
    return accumulation_base_dir(data_dir) / "nifty500_1h_ohlcv_master.json"


def accumulation_partition_dir(data_dir: Path, trade_date: date) -> Path:
    """Date-partitioned outputs: analysis CSV, report text."""
    d = data_dir / "analysis" / f"date={trade_date.isoformat()}" / "accumulation"
    d.mkdir(parents=True, exist_ok=True)
    return d


def tomorrow_watchlist_json_path(data_dir: Path, trade_date: date) -> Path:
    """Stored NIFTY500 multi-TF watchlist for the stocks dashboard tab."""
    return data_dir / f"nifty500_tomorrow_watchlist_{trade_date.isoformat()}.json"


def nifty500_partition_dir(data_dir: Path, trade_date: date) -> Path:
    """Shared OHLCV export dir for incremental NIFTY500 cache."""
    _ = trade_date  # kept for backward-compatible call sites
    p = data_dir / "NIFTY500"
    p.mkdir(parents=True, exist_ok=True)
    return p


def _partition_dir(root: Path, trade_date: date) -> Path:
    directory = root / f"date={trade_date.isoformat()}"
    directory.mkdir(parents=True, exist_ok=True)
    return directory
