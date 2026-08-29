"""
Relative Strength vs NIFTY 50 — daily-bar scanner over NIFTY 500 watchlist.

For each NIFTY 500 stock with cached daily bars, computes:

  • Mansfield-style RS line: (stock_close / nifty_close) normalized to 100
  • RS slope (5d / 20d / 60d) — pure rate-of-change of the RS ratio
  • % change vs NIFTY across 5d / 20d / 60d (stock_pct - nifty_pct)
  • RS rank percentile vs the NIFTY 500 universe

Outperformers are stocks with positive RS slope AND positive %change vs NIFTY.
Sorted by composite strength score; emits a small RS-line series per pick for
sparkline rendering in the dashboard.

Run:  python -m intraday_engine.main --relative-strength
"""

from __future__ import annotations

import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from intraday_engine.core.config import Settings
from intraday_engine.fetch.zerodha_client import ZerodhaClient
from intraday_engine.research.nifty500_accumulation_scanner import load_nifty500_symbols
from intraday_engine.storage.layout import (
    nifty_index_daily_path,
    relative_strength_path,
)
from intraday_engine.storage.nifty500_csv import (
    read_nifty500_symbol_ohlcv,
    write_nifty500_symbol_ohlcv,
)

logger = logging.getLogger(__name__)

NIFTY_SPOT_SYMBOL = "NSE:NIFTY 50"
RS_LOOKBACK_DAYS = 90
SPARKLINE_POINTS = 60


@dataclass
class RelativeStrengthRow:
    stock: str
    current_price: float
    rs_value: float            # latest Mansfield-style RS, 100 baseline
    rs_slope_20d_pct: float    # % change of RS over last 20d
    rs_slope_5d_pct: float
    rs_slope_60d_pct: float
    pct_change_5d: float       # stock %
    pct_change_20d: float
    pct_change_60d: float
    nifty_pct_change_5d: float
    nifty_pct_change_20d: float
    nifty_pct_change_60d: float
    excess_5d: float           # stock - nifty
    excess_20d: float
    excess_60d: float
    rs_rank_percentile: float  # vs scanned universe, 0..100
    strength_score: float      # composite ranking value
    rs_line: list[float] = field(default_factory=list)


def _load_nifty_daily(settings: Settings, lookback_days: int = 220) -> pd.DataFrame:
    """Cached daily NIFTY 50; refreshes from Kite if cache is stale or empty."""
    cache_path = nifty_index_daily_path(settings.data_dir)
    cached = pd.DataFrame()
    if cache_path.exists():
        try:
            cached = pd.read_csv(cache_path)
            cached["date"] = pd.to_datetime(cached["date"], errors="coerce")
            cached = cached.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
        except Exception:
            cached = pd.DataFrame()

    today = datetime.now()
    is_fresh = (
        not cached.empty
        and pd.Timestamp(cached["date"].iloc[-1]).normalize()
        >= pd.Timestamp(today.date() - timedelta(days=2)).normalize()
    )
    if is_fresh and len(cached) >= lookback_days:
        return cached.tail(lookback_days).reset_index(drop=True)

    try:
        client = ZerodhaClient(settings)
        quote = client.quote([NIFTY_SPOT_SYMBOL])
        token = int(quote[NIFTY_SPOT_SYMBOL]["instrument_token"])
        from_dt = today - timedelta(days=max(lookback_days * 2, 240))
        rows = client.historical_data(token, from_dt, today, interval="day", oi=False)
        if rows:
            df = pd.DataFrame(rows)
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
            for c in ("open", "high", "low", "close", "volume"):
                if c in df.columns:
                    df[c] = pd.to_numeric(df[c], errors="coerce")
            df = df.dropna(subset=["date", "close"]).sort_values("date").reset_index(drop=True)
            df = df.tail(lookback_days * 2)
            out = df[["date", "open", "high", "low", "close", "volume"]].copy()
            out["date"] = out["date"].dt.strftime("%Y-%m-%d %H:%M:%S")
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            out.to_csv(cache_path, index=False)
            df["date"] = pd.to_datetime(df["date"])
            return df.tail(lookback_days).reset_index(drop=True)
    except Exception as e:
        logger.warning("Could not refresh NIFTY 50 daily bars: %s", e)

    if not cached.empty:
        return cached.tail(lookback_days).reset_index(drop=True)
    return pd.DataFrame()


def _load_nifty_daily_with_retry(settings: Settings, lookback_days: int = 220) -> pd.DataFrame:
    """Load NIFTY 50 daily bars; retry once when Kite is cold or contended."""
    for attempt in range(2):
        df = _load_nifty_daily(settings, lookback_days=lookback_days)
        if not df.empty and len(df) >= 65:
            return df
        if attempt == 0:
            time.sleep(1.5)
    return df


def _ensure_stock_daily(
    client: ZerodhaClient,
    symbol: str,
    token: int,
    data_dir: Path,
) -> pd.DataFrame:
    """Read cached 1D bars or fetch from Kite when cache is missing or too thin."""
    df = read_nifty500_symbol_ohlcv(data_dir, symbol, "1D")
    if not df.empty and len(df) >= 65:
        return df

    today = datetime.now()
    from_dt = today - timedelta(days=240)
    try:
        rows = client.historical_data(int(token), from_dt, today, interval="day", oi=False)
    except Exception as e:
        logger.debug("RS daily fetch %s: %s", symbol, e)
        return df
    if not rows:
        return df

    fresh = pd.DataFrame(rows)
    fresh["date"] = pd.to_datetime(fresh["date"], errors="coerce")
    for col in ("open", "high", "low", "close", "volume"):
        if col in fresh.columns:
            fresh[col] = pd.to_numeric(fresh[col], errors="coerce")
    fresh = fresh.dropna(subset=["date", "close"]).sort_values("date").reset_index(drop=True)
    if df.empty:
        merged = fresh
    else:
        merged = (
            pd.concat([df, fresh], ignore_index=True)
            .dropna(subset=["date"])
            .sort_values("date")
            .drop_duplicates(subset=["date"], keep="last")
            .reset_index(drop=True)
        )
    if len(merged) >= 10:
        try:
            write_nifty500_symbol_ohlcv(data_dir, date.today(), symbol, None, None, merged)
        except Exception as e:
            logger.debug("RS cache write %s: %s", symbol, e)
    return merged


def _align_on_date(stock_df: pd.DataFrame, nifty_df: pd.DataFrame) -> pd.DataFrame:
    """Inner-join stock close + nifty close on day."""
    if stock_df.empty or nifty_df.empty:
        return pd.DataFrame()
    a = stock_df[["date", "close"]].copy()
    a["day"] = pd.to_datetime(a["date"]).dt.normalize()
    a = a.groupby("day", as_index=False)["close"].last().rename(columns={"close": "stock_close"})
    b = nifty_df[["date", "close"]].copy()
    b["day"] = pd.to_datetime(b["date"]).dt.normalize()
    b = b.groupby("day", as_index=False)["close"].last().rename(columns={"close": "nifty_close"})
    merged = a.merge(b, on="day", how="inner").sort_values("day").reset_index(drop=True)
    return merged


def _pct_change(series: pd.Series, n: int) -> float:
    if len(series) < n + 1:
        return 0.0
    older = float(series.iloc[-(n + 1)])
    newer = float(series.iloc[-1])
    if older <= 0:
        return 0.0
    return (newer / older - 1.0) * 100.0


def _compute_rs_row(symbol: str, stock_df: pd.DataFrame, nifty_df: pd.DataFrame) -> RelativeStrengthRow | None:
    aligned = _align_on_date(stock_df, nifty_df)
    if aligned.empty or len(aligned) < 65:
        return None
    rs_raw = aligned["stock_close"] / aligned["nifty_close"]
    base = float(rs_raw.iloc[-min(len(rs_raw), 60)])
    if base <= 0:
        return None
    rs = (rs_raw / base) * 100.0

    last_close = float(aligned["stock_close"].iloc[-1])

    rs_v = float(rs.iloc[-1])
    rs_slope_5 = _pct_change(rs, 5)
    rs_slope_20 = _pct_change(rs, 20)
    rs_slope_60 = _pct_change(rs, 60)

    p5 = _pct_change(aligned["stock_close"], 5)
    p20 = _pct_change(aligned["stock_close"], 20)
    p60 = _pct_change(aligned["stock_close"], 60)

    n5 = _pct_change(aligned["nifty_close"], 5)
    n20 = _pct_change(aligned["nifty_close"], 20)
    n60 = _pct_change(aligned["nifty_close"], 60)

    spark = rs.tail(SPARKLINE_POINTS).round(3).tolist()

    return RelativeStrengthRow(
        stock=symbol,
        current_price=round(last_close, 2),
        rs_value=round(rs_v, 2),
        rs_slope_5d_pct=round(rs_slope_5, 2),
        rs_slope_20d_pct=round(rs_slope_20, 2),
        rs_slope_60d_pct=round(rs_slope_60, 2),
        pct_change_5d=round(p5, 2),
        pct_change_20d=round(p20, 2),
        pct_change_60d=round(p60, 2),
        nifty_pct_change_5d=round(n5, 2),
        nifty_pct_change_20d=round(n20, 2),
        nifty_pct_change_60d=round(n60, 2),
        excess_5d=round(p5 - n5, 2),
        excess_20d=round(p20 - n20, 2),
        excess_60d=round(p60 - n60, 2),
        rs_rank_percentile=0.0,
        strength_score=0.0,
        rs_line=spark,
    )


def _compute_strength_score(row: RelativeStrengthRow) -> float:
    """Composite of multi-horizon RS slope + excess return. Higher = stronger."""
    # weights tuned to favor durable RS over single-day spikes
    return (
        0.20 * row.rs_slope_5d_pct
        + 0.40 * row.rs_slope_20d_pct
        + 0.25 * row.rs_slope_60d_pct
        + 0.10 * row.excess_20d
        + 0.05 * row.excess_60d
    )


def _stock_daily_cached(data_dir: Path, symbol: str) -> pd.DataFrame:
    return read_nifty500_symbol_ohlcv(data_dir, symbol, "1D")


def _collect_rs_rows(
    symbols: list[str],
    *,
    client: ZerodhaClient,
    token_map: dict[str, int],
    data_dir: Path,
    nifty_df: pd.DataFrame,
    max_workers: int = 8,
) -> tuple[list[RelativeStrengthRow], int]:
    """
    Build RS rows cache-first, then backfill missing symbols sequentially.

    Parallel Kite historical pulls on a cold cache often all fail (rate limits /
    non-thread-safe session). Sequential backfill after a cache-only pass fixes
    the “empty on first refresh, OK on second click” behaviour.
    """
    scored: list[RelativeStrengthRow] = []
    scored_syms: set[str] = set()
    skipped = 0
    needs_fetch: list[str] = []

    def _try_row(sym: str, df: pd.DataFrame) -> RelativeStrengthRow | None:
        if df.empty or len(df) < 65:
            return None
        return _compute_rs_row(sym, df, nifty_df)

    def cache_job(sym: str) -> tuple[str, RelativeStrengthRow | None, bool]:
        df = _stock_daily_cached(data_dir, sym)
        row = _try_row(sym, df)
        if row is not None:
            return sym, row, False
        return sym, None, len(df) < 65

    with ThreadPoolExecutor(max_workers=max(1, max_workers)) as ex:
        futs = {ex.submit(cache_job, sym): sym for sym in symbols}
        for fut in as_completed(futs):
            try:
                sym, row, fetch_needed = fut.result()
            except Exception:
                skipped += 1
                continue
            if row is not None:
                scored.append(row)
                scored_syms.add(sym)
            elif fetch_needed:
                needs_fetch.append(sym)
            else:
                skipped += 1

    for sym in needs_fetch:
        if sym in scored_syms:
            continue
        try:
            tok = token_map.get(sym)
            if tok is not None:
                df = _ensure_stock_daily(client, sym, int(tok), data_dir)
            else:
                df = _stock_daily_cached(data_dir, sym)
            row = _try_row(sym, df)
            if row is None:
                skipped += 1
            else:
                scored.append(row)
                scored_syms.add(sym)
        except Exception as e:
            logger.debug("RS fetch %s err: %s", sym, e)
            skipped += 1
        time.sleep(0.35)

    return scored, skipped


def run_relative_strength_scan(
    *,
    settings: Settings | None = None,
    symbols_file: Path | None = None,
    only_outperformers: bool = True,
    top_n: int = 50,
    max_workers: int = 8,
    trade_date: date | None = None,
) -> dict[str, Any]:
    settings = settings or Settings.from_env(underlying="NIFTY")
    td = trade_date or date.today()

    nifty_df = _load_nifty_daily_with_retry(settings)
    if nifty_df.empty:
        return {
            "trade_date": td.isoformat(),
            "generated_at": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
            "rows": [],
            "scanned": 0,
            "passed": 0,
            "message": "Could not load NIFTY 50 daily bars (Kite credentials or cache).",
        }

    nifty_period = {
        "5d": _pct_change(nifty_df["close"], 5),
        "20d": _pct_change(nifty_df["close"], 20),
        "60d": _pct_change(nifty_df["close"], 60),
    }

    symbols = load_nifty500_symbols(symbols_file, settings.data_dir)
    client = ZerodhaClient(settings)
    token_map = client.nse_eq_token_map()

    scored, skipped = _collect_rs_rows(
        symbols,
        client=client,
        token_map=token_map,
        data_dir=settings.data_dir,
        nifty_df=nifty_df,
        max_workers=max_workers,
    )

    if not scored:
        logger.info("RS scan empty after cache+fetch pass; retrying once after brief pause.")
        time.sleep(2.0)
        scored, skipped = _collect_rs_rows(
            symbols,
            client=client,
            token_map=token_map,
            data_dir=settings.data_dir,
            nifty_df=nifty_df,
            max_workers=min(max_workers, 2),
        )

    if not scored:
        return {
            "trade_date": td.isoformat(),
            "generated_at": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
            "rows": [],
            "scanned": len(symbols),
            "passed": 0,
            "skipped": skipped,
            "message": (
                "No RS rows produced after fetching daily bars. "
                "Check Kite credentials and retry."
            ),
        }

    for r in scored:
        r.strength_score = round(_compute_strength_score(r), 2)

    # rank percentile by strength_score
    sorted_rows = sorted(scored, key=lambda r: r.strength_score)
    n = len(sorted_rows)
    for i, r in enumerate(sorted_rows):
        r.rs_rank_percentile = round(((i + 1) / n) * 100.0, 1)

    final = sorted(scored, key=lambda r: r.strength_score, reverse=True)
    if only_outperformers:
        final = [
            r for r in final
            if r.rs_slope_20d_pct > 0 and r.excess_20d > 0
        ]

    top = final[:top_n]

    payload: dict[str, Any] = {
        "trade_date": td.isoformat(),
        "generated_at": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
        "scanned": len(symbols),
        "passed": len(final),
        "skipped": skipped,
        "nifty_pct_change": {k: round(v, 2) for k, v in nifty_period.items()},
        "only_outperformers": only_outperformers,
        "rows": [asdict(r) for r in top],
    }
    if not top and scored:
        payload["message"] = (
            f"0 stocks passed the outperformer filter ({len(scored)} computed). "
            "Uncheck 'Only outperformers' and refresh."
        )

    if top:
        out_path = relative_strength_path(settings.data_dir, td)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        logger.info(
            "Relative strength: %d outperformers of %d scanned -> %s",
            len(final),
            len(symbols),
            out_path,
        )
    elif scored:
        # Save unfiltered rows so GET / relative-strength works after refresh
        # even when the outperformer filter yields zero matches.
        payload["rows"] = [asdict(r) for r in sorted(scored, key=lambda r: r.strength_score, reverse=True)[:top_n]]
        payload["passed"] = len(scored)
        payload["only_outperformers"] = False
        out_path = relative_strength_path(settings.data_dir, td)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        logger.info(
            "Relative strength: saved %d computed rows (0 passed filter) -> %s",
            len(scored),
            out_path,
        )
    else:
        logger.info(
            "Relative strength: no rows to save (%d scanned, %d passed filter)",
            len(symbols),
            len(final),
        )
    return payload


def _read_rs_payload(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _rs_payload_usable(data: dict[str, Any] | None) -> bool:
    return bool(data and data.get("rows"))


def load_stored_relative_strength(data_dir: Path, trade_date: date) -> dict[str, Any] | None:
    p = relative_strength_path(data_dir, trade_date)
    if p.exists():
        data = _read_rs_payload(p)
        if _rs_payload_usable(data):
            return data

    folder = relative_strength_path(data_dir, trade_date).parent
    if not folder.exists():
        return None
    files = sorted(folder.glob("relative_strength_*.json"))
    best: dict[str, Any] | None = None
    for f in reversed(files):
        data = _read_rs_payload(f)
        if _rs_payload_usable(data):
            best = data
            break
    if not best:
        return None
    shown = best.get("trade_date") or "unknown"
    if shown != trade_date.isoformat():
        best = dict(best)
        best["stale"] = True
        best["message"] = (
            f"No saved RS scan for {trade_date.isoformat()}. "
            f"Showing last successful run ({shown})."
        )
    return best
