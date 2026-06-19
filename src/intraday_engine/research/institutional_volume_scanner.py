"""
Institutional Volume scanner — daily timeframe lookback.

For each NIFTY 500 stock, identifies candles whose volume is the highest seen
across one or more lookback windows:

  - Week   (current + last 4 trading days  → 5 bars)
  - Month  (~21 trading days)
  - Quarter (~63 trading days)
  - 6 Months (~126 trading days)
  - Year   (~252 trading days)
  - Ever   (all cached history)

A candle can carry multiple labels (the windows nest, so a "year" candle is
also "6-month", "quarter", "month", and "week"). We surface every label that
applies so the dashboard can show the maximum reach of the print.

The scanner is "lookback only": for the candle on day ``D``, the window is
``[D - (window_bars - 1), D]`` inclusive — i.e. looks backwards including
``D`` itself.

Data flow:
  1) Read cached ``data/NIFTY500/SYMBOL_1D.csv`` if present.
  2) Incrementally extend it via Kite ``interval='day'`` so we have at least
     ~1 year of bars.
  3) Persist the merged daily CSV back to the shared cache.
  4) Label every candle in the most recent ``recent_days`` window.

Output JSON saved under ``data/analysis/institutional_volume/``.

Run via dashboard refresh button or programmatically with
``run_institutional_volume_scan``.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd

from intraday_engine.core.config import Settings
from intraday_engine.fetch.zerodha_client import ZerodhaClient
from intraday_engine.research.nifty500_accumulation_scanner import load_nifty500_symbols
from intraday_engine.storage.layout import institutional_volume_path
from intraday_engine.storage.nifty500_csv import (
    nifty500_symbol_file_stem,
    read_nifty500_symbol_ohlcv,
)

logger = logging.getLogger(__name__)


# (label, lookback_bars). ``None`` = unbounded (use all cached history).
# Order matters: ascending window size so the "biggest" label wins for display.
LOOKBACK_WINDOWS: list[tuple[str, int | None]] = [
    ("month", 21),
    ("quarter", 63),
    ("6_months", 126),
    ("year", 252),
    ("ever", None),
]

# How many trailing trading days to inspect / render in the dot strip.
DEFAULT_RECENT_DAYS = 60

# Required history for "year" label to be meaningful. We always try to fetch
# at least this many calendar days of bars when refreshing.
MIN_HISTORY_CALENDAR_DAYS = 380

# Refresh-cache thresholds: re-fetch if we have fewer than this many bars OR
# the most recent bar is older than ``REFRESH_IF_OLDER_DAYS`` calendar days.
MIN_DAILY_BARS = 60
REFRESH_IF_OLDER_DAYS = 2

# Politeness: throttle Kite historical calls.
_rate_lock = threading.Lock()
_rate_last = 0.0


def _throttle(min_interval: float = 0.28) -> None:
    global _rate_last
    with _rate_lock:
        now = time.monotonic()
        wait = min_interval - (now - _rate_last)
        if wait > 0:
            time.sleep(wait)
        _rate_last = time.monotonic()


@dataclass
class InstitutionalVolumeCandle:
    date: str
    open: float
    high: float
    low: float
    close: float
    volume: float
    avg_volume_20d: float
    volume_multiple: float
    pct_change: float       # close vs previous close (%)
    labels: list[str]
    top_label: str          # the widest window label that applies


@dataclass
class InstitutionalVolumeRow:
    stock: str
    last_close: float
    last_date: str
    bars_available: int
    candles: list[InstitutionalVolumeCandle] = field(default_factory=list)


def _format_date_column(s: pd.Series) -> pd.Series:
    dt = pd.to_datetime(s, errors="coerce")
    return dt.dt.strftime("%Y-%m-%d %H:%M:%S")


def _write_daily_cache(df: pd.DataFrame, data_dir: Path, symbol: str) -> None:
    """Persist merged daily bars back to data/NIFTY500/SYMBOL_1D.csv."""
    if df.empty or "date" not in df.columns:
        return
    out = df.copy()
    cols = [c for c in ("date", "open", "high", "low", "close", "volume") if c in out.columns]
    out = out[cols]
    out["date"] = _format_date_column(out["date"])
    stem = nifty500_symbol_file_stem(symbol)
    path = data_dir / "NIFTY500" / f"{stem}_1D.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False)


def _last_bar_dt(df: pd.DataFrame) -> datetime | None:
    if df.empty or "date" not in df.columns:
        return None
    try:
        last = pd.to_datetime(df["date"], errors="coerce").max()
        if pd.isna(last):
            return None
        ts = pd.Timestamp(last).to_pydatetime()
        if getattr(ts, "tzinfo", None) is not None:
            ts = ts.replace(tzinfo=None)
        return ts
    except Exception:
        return None


def _needs_refresh(df: pd.DataFrame, today: datetime) -> bool:
    if df.empty or len(df) < MIN_DAILY_BARS:
        return True
    last = _last_bar_dt(df)
    if last is None:
        return True
    return (today.date() - last.date()) > timedelta(days=REFRESH_IF_OLDER_DAYS)


def _merge_daily(existing: pd.DataFrame, fresh_rows: list[dict[str, Any]]) -> pd.DataFrame:
    """Merge existing cache with freshly fetched Kite rows by date."""
    if not fresh_rows and existing.empty:
        return pd.DataFrame()
    fresh_df = pd.DataFrame(fresh_rows) if fresh_rows else pd.DataFrame()
    if not fresh_df.empty:
        fresh_df["date"] = pd.to_datetime(fresh_df["date"], errors="coerce")
        for c in ("open", "high", "low", "close", "volume"):
            if c in fresh_df.columns:
                fresh_df[c] = pd.to_numeric(fresh_df[c], errors="coerce")
        fresh_df = fresh_df.dropna(subset=["date"])
        # normalize tz-aware to naive
        fresh_df["date"] = fresh_df["date"].map(
            lambda z: pd.Timestamp(z).tz_convert("Asia/Kolkata").tz_localize(None)
            if (not pd.isna(z) and getattr(pd.Timestamp(z), "tzinfo", None) is not None)
            else pd.Timestamp(z)
        )
        # normalize to date (drop time) so duplicates collapse predictably
        fresh_df["date"] = pd.to_datetime(fresh_df["date"]).dt.normalize()

    if not existing.empty:
        existing = existing.copy()
        existing["date"] = pd.to_datetime(existing["date"]).dt.normalize()

    parts = [d for d in (existing, fresh_df) if not d.empty]
    if not parts:
        return pd.DataFrame()
    merged = pd.concat(parts, ignore_index=True)
    merged = (
        merged.dropna(subset=["date"])
        .sort_values("date")
        .drop_duplicates(subset=["date"], keep="last")
        .reset_index(drop=True)
    )
    return merged


def _ensure_one_year_daily(
    client: ZerodhaClient,
    symbol: str,
    token: int,
    data_dir: Path,
    today: datetime,
    history_days: int = MIN_HISTORY_CALENDAR_DAYS,
) -> pd.DataFrame:
    """Return ~1y of daily OHLCV bars; fetch incrementally only when stale."""
    cached = read_nifty500_symbol_ohlcv(data_dir, symbol, "1D")

    if not _needs_refresh(cached, today):
        return cached

    last = _last_bar_dt(cached)
    if last is None:
        from_dt = today - timedelta(days=history_days)
    else:
        # overlap a few days to recover any late corrections
        from_dt = last - timedelta(days=5)
        # if cache is too thin, refresh full year
        if len(cached) < MIN_DAILY_BARS:
            from_dt = today - timedelta(days=history_days)

    _throttle()
    rows: list[dict[str, Any]] = []
    try:
        rows = client.historical_data(token, from_dt, today, interval="day", oi=False)
    except Exception as e:
        logger.debug("daily fetch %s err: %s", symbol, e)
        rows = []

    merged = _merge_daily(cached, rows)
    if merged.empty:
        return cached

    try:
        _write_daily_cache(merged, data_dir, symbol)
    except Exception as e:
        logger.debug("daily cache write %s err: %s", symbol, e)

    return merged


def _classify_candles(
    df: pd.DataFrame, recent_days: int = DEFAULT_RECENT_DAYS
) -> list[InstitutionalVolumeCandle]:
    """Emit one entry per trailing trading day, with applicable volume labels.

    Returns a candle for **every** trading day in the trailing ``recent_days``
    window (sorted oldest → newest), with ``labels=[]`` when the day did not
    set a new high in any lookback window. The dashboard renders these as a
    30-day dot strip per stock.

    For candle ``i`` on date ``D``:
    - Compute max(volume[i - (window_bars - 1) .. i]) for each window
    - If candle's volume equals that max (with tiny tolerance), it earns the
      label. Note: smaller windows nest inside bigger ones, so a candle may
      collect multiple labels — we keep them all.
    """
    if df.empty or "volume" not in df.columns:
        return []
    df = df.dropna(subset=["date", "open", "high", "low", "close", "volume"]).reset_index(drop=True)
    n = len(df)
    if n == 0:
        return []

    volumes = df["volume"].astype(float).values
    closes = df["close"].astype(float).values

    start = max(0, n - max(1, recent_days))
    out: list[InstitutionalVolumeCandle] = []
    for i in range(start, n):
        v = float(volumes[i])
        labels: list[str] = []
        if v > 0:
            for name, win in LOOKBACK_WINDOWS:
                if win is None:
                    lo = 0
                else:
                    lo = max(0, i - (win - 1))
                    # require at least 2 bars in window so "highest" is meaningful
                    if i - lo < 1:
                        continue
                window_max = float(volumes[lo : i + 1].max())
                if window_max <= 0:
                    continue
                # equality with a tiny epsilon for float safety
                if v + 1e-9 >= window_max:
                    labels.append(name)

        row = df.iloc[i]
        avg20 = (
            float(volumes[max(0, i - 20) : i].mean()) if i >= 1 else float(v)
        )
        mult = float(v / avg20) if avg20 > 0 else 0.0
        prev_close = float(closes[i - 1]) if i > 0 else 0.0
        pct = ((float(closes[i]) - prev_close) / prev_close * 100.0) if prev_close > 0 else 0.0

        date_val = pd.to_datetime(row["date"])
        date_str = (
            date_val.strftime("%Y-%m-%d")
            if not pd.isna(date_val)
            else str(row["date"])
        )

        top_label = labels[-1] if labels else ""

        out.append(
            InstitutionalVolumeCandle(
                date=date_str,
                open=round(float(row["open"]), 2),
                high=round(float(row["high"]), 2),
                low=round(float(row["low"]), 2),
                close=round(float(row["close"]), 2),
                volume=float(v),
                avg_volume_20d=float(round(avg20, 2)),
                volume_multiple=round(mult, 2),
                pct_change=round(pct, 2),
                labels=labels,
                top_label=top_label,
            )
        )
    return out


def _label_rank(label: str) -> int:
    order = [n for n, _ in LOOKBACK_WINDOWS]
    try:
        return order.index(label)
    except ValueError:
        return -1


def run_institutional_volume_scan(
    *,
    settings: Settings | None = None,
    symbols_file: Path | None = None,
    trade_date: date | None = None,
    recent_days: int = DEFAULT_RECENT_DAYS,
    history_days: int = MIN_HISTORY_CALENDAR_DAYS,
    max_workers: int = 6,
    stock_limit: int | None = None,
) -> dict[str, Any]:
    """Scan NIFTY 500 daily bars and emit institutional-volume labelled candles."""
    settings = settings or Settings.from_env(underlying="NIFTY")
    td = trade_date or date.today()
    today_dt = datetime.combine(td, datetime.min.time()).replace(hour=23, minute=59)

    client = ZerodhaClient(settings)
    token_map = client.nse_eq_token_map()

    symbols = load_nifty500_symbols(symbols_file, settings.data_dir)
    to_scan = [s for s in symbols if s in token_map]
    if stock_limit:
        to_scan = to_scan[: int(stock_limit)]

    rows: list[InstitutionalVolumeRow] = []
    failed: list[dict[str, str]] = []
    skipped_no_bars = 0

    def job(sym: str) -> InstitutionalVolumeRow | dict[str, str] | None:
        try:
            tok = int(token_map[sym])
        except (KeyError, ValueError, TypeError):
            return {"stock": sym, "error": "no_token"}
        try:
            df = _ensure_one_year_daily(
                client, sym, tok, settings.data_dir, today_dt, history_days
            )
        except Exception as e:
            return {"stock": sym, "error": str(e)[:200]}
        if df is None or df.empty:
            return None
        candles = _classify_candles(df, recent_days=recent_days)
        if not candles:
            return None
        # Only surface stocks that have at least one labelled day in the window
        if not any(c.labels for c in candles):
            return None
        try:
            last_row = df.iloc[-1]
            last_close = float(last_row["close"])
            last_date = pd.to_datetime(last_row["date"]).strftime("%Y-%m-%d")
        except Exception:
            last_close, last_date = 0.0, ""
        return InstitutionalVolumeRow(
            stock=sym,
            last_close=round(last_close, 2),
            last_date=last_date,
            bars_available=int(len(df)),
            candles=candles,
        )

    with ThreadPoolExecutor(max_workers=max(1, max_workers)) as ex:
        futs = {ex.submit(job, s): s for s in to_scan}
        for fut in as_completed(futs):
            try:
                res = fut.result()
            except Exception as e:
                failed.append({"stock": futs[fut], "error": str(e)[:200]})
                continue
            if res is None:
                skipped_no_bars += 1
                continue
            if isinstance(res, dict):
                failed.append(res)
                continue
            rows.append(res)

    # sort: best-label-rank desc → highest volume multiple desc → most recent date desc
    def _row_sort_key(r: InstitutionalVolumeRow) -> tuple:
        if not r.candles:
            return (-1, 0.0, "")
        best = max(r.candles, key=lambda c: (_label_rank(c.top_label), c.volume_multiple))
        return (-_label_rank(best.top_label), -best.volume_multiple, best.date)

    rows.sort(key=_row_sort_key)

    # flatten counts per label for summary
    label_counts: dict[str, int] = {name: 0 for name, _ in LOOKBACK_WINDOWS}
    for r in rows:
        for c in r.candles:
            for lab in c.labels:
                label_counts[lab] = label_counts.get(lab, 0) + 1

    payload = {
        "trade_date": td.isoformat(),
        "generated_at": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
        "universe": "NIFTY500",
        "scanned": len(to_scan),
        "passed": len(rows),
        "skipped_no_bars": skipped_no_bars,
        "failed": failed[:60],
        "failed_count": len(failed),
        "recent_days": recent_days,
        "history_days": history_days,
        "label_counts": label_counts,
        "windows": [
            {"label": name, "bars": win if win is not None else "all"}
            for name, win in LOOKBACK_WINDOWS
        ],
        "rows": [asdict(r) for r in rows],
    }

    out_path = institutional_volume_path(settings.data_dir, td)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    logger.info(
        "Institutional volume scan: %d/%d stocks labelled. Failed=%d. Saved %s",
        len(rows),
        len(to_scan),
        len(failed),
        out_path,
    )
    return payload


def api_institutional_volume_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Trim stored scan for dashboard JSON (drops unused OHLC fields per candle)."""
    if not payload:
        return payload
    out = {k: v for k, v in payload.items() if k != "rows"}
    slim_rows: list[dict[str, Any]] = []
    for row in payload.get("rows") or []:
        slim_rows.append(
            {
                "stock": row.get("stock"),
                "last_close": row.get("last_close"),
                "last_date": row.get("last_date"),
                "bars_available": row.get("bars_available"),
                "candles": [
                    {
                        "date": c.get("date"),
                        "close": c.get("close"),
                        "volume": c.get("volume"),
                        "volume_multiple": c.get("volume_multiple"),
                        "pct_change": c.get("pct_change"),
                        "labels": c.get("labels"),
                        "top_label": c.get("top_label"),
                    }
                    for c in (row.get("candles") or [])
                ],
            }
        )
    out["rows"] = slim_rows
    return out


def load_stored_institutional_volume(data_dir: Path, trade_date: date) -> dict[str, Any] | None:
    p = institutional_volume_path(data_dir, trade_date)
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            return None
    folder = p.parent
    if not folder.exists():
        return None
    files = sorted(folder.glob("institutional_volume_*.json"))
    if not files:
        return None
    try:
        return json.loads(files[-1].read_text(encoding="utf-8"))
    except Exception:
        return None
