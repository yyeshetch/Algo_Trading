"""
Fetch maximum available 15-minute history for a symbol and rank intraday slots
by where the largest moves (range %) tend to occur.

Usage:
  python -m intraday_engine.research.intraday_move_windows --symbol RELIANCE
  python -m intraday_engine.research.intraday_move_windows --symbol "NIFTY 50" --exchange NSE
"""

from __future__ import annotations

import argparse
import logging
import time
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

from intraday_engine.core.config import Settings
from intraday_engine.fetch.zerodha_client import ZerodhaClient
from intraday_engine.utils.logging_setup import setup_logging

logger = logging.getLogger(__name__)

IST = ZoneInfo("Asia/Kolkata")

# Kite allows a large window per request for 15minute; use conservative chunks + pagination.
_CHUNK_DAYS = 180
_MAX_CHUNKS = 30
_RATE_SLEEP_SEC = 0.35


def _normalize_nse_symbol(symbol: str) -> str:
    s = symbol.strip().upper()
    if s.startswith("NSE:"):
        return s[4:].strip()
    return s


def _quote_key(exchange: str, tradingsymbol: str) -> str:
    return f"{exchange.upper()}:{tradingsymbol}"


def resolve_instrument_token(client: ZerodhaClient, exchange: str, tradingsymbol: str) -> int | None:
    """Resolve instrument_token via quote API."""
    key = _quote_key(exchange, tradingsymbol)
    try:
        q = client.quote([key])
        row = (q or {}).get(key)
        if isinstance(row, dict):
            tok = row.get("instrument_token")
            if tok is not None:
                return int(tok)
    except Exception as e:
        logger.warning("Quote failed for %s: %s", key, e)
    return None


def fetch_15min_historical_max(
    client: ZerodhaClient,
    instrument_token: int,
    *,
    to_dt: datetime | None = None,
) -> pd.DataFrame:
    """
    Pull as much 15-minute history as Kite returns, using chunked date ranges.
    """
    to_dt = to_dt or datetime.now(IST)
    rows: list[dict] = []
    seen: set[object] = set()

    for chunk_i in range(_MAX_CHUNKS):
        from_dt = to_dt - timedelta(days=_CHUNK_DAYS)
        try:
            time.sleep(_RATE_SLEEP_SEC)
            chunk = client.historical_data(
                instrument_token,
                from_dt,
                to_dt,
                interval="15minute",
            )
        except Exception as e:
            logger.warning("Historical chunk %s failed: %s", chunk_i, e)
            break

        if not chunk:
            break

        for r in chunk:
            if not isinstance(r, dict):
                continue
            d = r.get("date")
            if d is None:
                continue
            ts = pd.Timestamp(d)
            if ts.tzinfo is None:
                ts = ts.tz_localize(IST)
            else:
                ts = ts.tz_convert(IST)
            key = int(ts.timestamp())
            if key in seen:
                continue
            seen.add(key)
            rows.append(r)

        oldest_ts = None
        for r in chunk:
            d = r.get("date")
            if d is None:
                continue
            ts = pd.Timestamp(d)
            if ts.tzinfo is None:
                ts = ts.tz_localize(IST)
            else:
                ts = ts.tz_convert(IST)
            if oldest_ts is None or ts < oldest_ts:
                oldest_ts = ts

        if oldest_ts is None:
            break

        to_dt = oldest_ts.to_pydatetime() - timedelta(seconds=1)
        if len(chunk) < 10:
            break

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df = df.rename(columns={"date": "timestamp"})
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"])
    if df["timestamp"].dt.tz is None:
        df["timestamp"] = df["timestamp"].dt.tz_localize(IST)
    else:
        df["timestamp"] = df["timestamp"].dt.tz_convert(IST)
    for col in ("open", "high", "low", "close"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


def _session_slot_label(ts: pd.Timestamp) -> str:
    """15-min slot start label in IST (e.g. 09:15)."""
    t = pd.Timestamp(ts)
    if t.tzinfo is None:
        t = t.tz_localize(IST)
    else:
        t = t.tz_convert(IST)
    m = t.minute
    h = t.hour
    slot_min = (m // 15) * 15
    return f"{h:02d}:{slot_min:02d}"


def analyze_intraday_windows(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns (per_slot_stats, per_day_best_slot) for range% per 15m candle.
    """
    if df.empty or "open" not in df.columns:
        return pd.DataFrame(), pd.DataFrame()

    work = df.copy()
    work["open"] = work["open"].replace(0, pd.NA)
    work = work.dropna(subset=["open", "high", "low"])
    work["range_pct"] = (work["high"].astype(float) - work["low"].astype(float)) / work["open"].astype(float) * 100.0
    work["slot"] = work["timestamp"].apply(lambda x: _session_slot_label(pd.Timestamp(x)))

    # Regular session only (exclude pre/post if any)
    t = work["timestamp"].dt.tz_convert(IST)
    mins = t.dt.hour * 60 + t.dt.minute
    session_start = 9 * 60 + 15
    session_end = 15 * 60 + 30
    work = work[(mins >= session_start) & (mins <= session_end)]

    if work.empty:
        return pd.DataFrame(), pd.DataFrame()

    g = work.groupby("slot", sort=False)
    stats = g.agg(
        n=("range_pct", "count"),
        mean_range_pct=("range_pct", "mean"),
        median_range_pct=("range_pct", "median"),
        std_range_pct=("range_pct", "std"),
        max_range_pct=("range_pct", "max"),
    ).reset_index()
    stats = stats.sort_values("mean_range_pct", ascending=False).reset_index(drop=True)
    stats["rank_by_mean"] = range(1, len(stats) + 1)

    # Per calendar day: which slot had the largest 15m range
    work["date_only"] = work["timestamp"].dt.tz_convert(IST).dt.date
    idx = work.groupby("date_only")["range_pct"].idxmax()
    best = work.loc[idx, ["date_only", "slot", "range_pct"]].rename(
        columns={"slot": "best_slot", "range_pct": "best_slot_range_pct"}
    )
    slot_counts = best["best_slot"].value_counts().rename("days_largest_here").reset_index()
    slot_counts.columns = ["slot", "days_largest_here"]

    stats = stats.merge(slot_counts, on="slot", how="left")
    stats["days_largest_here"] = stats["days_largest_here"].fillna(0).astype(int)
    n_days = best["date_only"].nunique()
    if n_days > 0:
        stats["pct_days_largest_here"] = (100.0 * stats["days_largest_here"] / n_days).round(2)
    else:
        stats["pct_days_largest_here"] = 0.0

    return stats, best


def run_analysis(
    symbol: str,
    *,
    exchange: str = "NSE",
    data_dir: Path | None = None,
) -> None:
    settings = Settings.from_env()
    setup_logging(settings.log_level, settings.data_dir)
    client = ZerodhaClient(settings)

    sym = _normalize_nse_symbol(symbol)
    token = resolve_instrument_token(client, exchange, sym)
    if token is None:
        logger.error("Could not resolve instrument for %s:%s", exchange, sym)
        return

    logger.info("Fetching 15-minute history (chunked, max depth) for %s:%s token=%s", exchange, sym, token)
    df = fetch_15min_historical_max(client, token)
    if df.empty:
        logger.error("No 15-minute candles returned.")
        return

    logger.info("Loaded %s candles from %s to %s", len(df), df["timestamp"].min(), df["timestamp"].max())

    stats, per_day = analyze_intraday_windows(df)
    if stats.empty:
        logger.error("No rows after session filter / stats.")
        return

    out_dir = data_dir or settings.data_dir
    out_dir = Path(out_dir)
    research_dir = out_dir / "research"
    research_dir.mkdir(parents=True, exist_ok=True)
    safe = sym.replace(" ", "_")
    ts = datetime.now(IST).strftime("%Y%m%d_%H%M%S")
    csv_path = research_dir / f"move_windows_{safe}_{ts}.csv"
    stats.to_csv(csv_path, index=False)
    per_day_path = research_dir / f"move_windows_{safe}_{ts}_per_day.csv"
    per_day.to_csv(per_day_path, index=False)

    print("\n=== Intraday 15-min slots ranked by mean range % (larger = bigger moves) ===\n")
    print(stats.to_string(index=False))
    print(f"\nSaved: {csv_path}")
    print(f"Saved: {per_day_path}")
    print(
        "\nInterpretation: 'mean_range_pct' is average (high-low)/open*100 for that clock slot; "
        "'pct_days_largest_here' is how often that slot had the widest 15m bar of the day."
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze which 15-min session slots show the largest moves for a symbol.",
    )
    parser.add_argument(
        "--symbol",
        required=True,
        help="Trading symbol (e.g. RELIANCE, TCS, or NIFTY 50 for index)",
    )
    parser.add_argument(
        "--exchange",
        default="NSE",
        help="Exchange (default NSE)",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Override output directory (default: DATA_DIR from .env)",
    )
    args = parser.parse_args()
    run_analysis(
        args.symbol,
        exchange=args.exchange,
        data_dir=Path(args.data_dir).resolve() if args.data_dir else None,
    )


if __name__ == "__main__":
    main()
