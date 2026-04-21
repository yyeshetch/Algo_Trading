"""Read/write per-symbol NIFTY 500 OHLCV CSVs under data/NIFTY500/."""

from __future__ import annotations

import logging
from datetime import date
from pathlib import Path

import pandas as pd

from intraday_engine.storage.layout import nifty500_partition_dir

logger = logging.getLogger(__name__)

_FILENAME_BAD = set('\\/:*?"<>|')


def nifty500_symbol_file_stem(symbol: str) -> str:
    s = str(symbol).strip().upper().replace(" ", "")
    if not s:
        return "UNKNOWN"
    return "".join("_" if c in _FILENAME_BAD else c for c in s)


def _format_date_column(s: pd.Series) -> pd.Series:
    dt = pd.to_datetime(s, errors="coerce")
    return dt.dt.strftime("%Y-%m-%d %H:%M:%S")


def _write_ohlcv_csv(df: pd.DataFrame, path: Path) -> None:
    if df.empty or "date" not in df.columns:
        return
    out = df.copy()
    cols = [c for c in ("date", "open", "high", "low", "close", "volume") if c in out.columns]
    if "date" not in cols:
        return
    out = out[cols]
    out["date"] = _format_date_column(out["date"])
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False)


def _read_ohlcv_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        df = pd.read_csv(path)
    except Exception:
        logger.warning("Could not read OHLCV CSV: %s", path)
        return pd.DataFrame()
    if "date" not in df.columns:
        return pd.DataFrame()
    out = df.copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    # Normalize mixed tz-aware / tz-naive inputs to naive IST-like timestamps.
    out["date"] = out["date"].map(
        lambda z: pd.Timestamp(z).tz_convert("Asia/Kolkata").tz_localize(None)
        if (not pd.isna(z) and getattr(pd.Timestamp(z), "tzinfo", None) is not None)
        else (pd.Timestamp(z) if not pd.isna(z) else pd.NaT)
    )
    out = out.dropna(subset=["date"]).sort_values("date")
    for c in ("open", "high", "low", "close", "volume"):
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    return out.reset_index(drop=True)


def _timeframe_suffix(timeframe: str) -> str:
    v = str(timeframe).strip().lower().replace(" ", "")
    if v in {"15m", "15min", "15minute"}:
        return "15Min"
    if v in {"1h", "60m", "60min", "60minute"}:
        return "1H"
    if v in {"1d", "day", "daily"}:
        return "1D"
    raise ValueError(f"Unsupported timeframe '{timeframe}'")


def read_nifty500_symbol_ohlcv(data_dir: Path, symbol: str, timeframe: str) -> pd.DataFrame:
    """Read cached OHLCV for one symbol/timeframe from data/NIFTY500/."""
    stem = nifty500_symbol_file_stem(symbol)
    tf = _timeframe_suffix(timeframe)
    root = data_dir / "NIFTY500"
    path = root / f"{stem}_{tf}.csv"
    df = _read_ohlcv_csv(path)
    if not df.empty:
        return df

    # Backward-compat: seed from older date-partitioned files if present.
    candidates: list[Path] = sorted(root.glob(f"date=*/{stem}_{tf}.csv"))
    if not candidates:
        return pd.DataFrame()
    return _read_ohlcv_csv(candidates[-1])


def write_nifty500_symbol_ohlcv(
    data_dir: Path,
    trade_date: date,
    symbol: str,
    m15: pd.DataFrame | None,
    h1: pd.DataFrame | None,
    d: pd.DataFrame | None,
) -> None:
    """
    Write up to three files: SYMBOL_15Min.csv, SYMBOL_1H.csv, SYMBOL_1D.csv
    under data_dir / NIFTY500 / (shared incremental cache).
    """
    stem = nifty500_symbol_file_stem(symbol)
    out_dir = nifty500_partition_dir(data_dir, trade_date)
    if m15 is not None and not m15.empty:
        _write_ohlcv_csv(m15, out_dir / f"{stem}_15Min.csv")
    if h1 is not None and not h1.empty:
        _write_ohlcv_csv(h1, out_dir / f"{stem}_1H.csv")
    if d is not None and not d.empty:
        _write_ohlcv_csv(d, out_dir / f"{stem}_1D.csv")
