"""Load prior-session analysis summaries for intraday indicator warmup."""

from __future__ import annotations

from datetime import date, timedelta
from typing import Any

from intraday_engine.analysis.summary_builder import build_analysis_summaries
from intraday_engine.storage.data_store import DataStore

# Tail of prior session used to warm ATR(14) + baseline windows at the open.
PRIOR_WARMUP_BARS = 30


def prior_trading_date(d: date) -> date:
    prev = d - timedelta(days=1)
    while prev.weekday() >= 5:
        prev -= timedelta(days=1)
    return prev


def prior_session_close(data_dir, underlying: str, trade_date: date) -> float | None:
    """Last spot print from the previous NSE session."""
    store = DataStore(data_dir, underlying=underlying)
    prev_df = store.load_snapshots(trade_date=prior_trading_date(trade_date))
    if prev_df.empty or "spot_ltp" not in prev_df.columns:
        return None
    u = underlying.upper()
    if "underlying" in prev_df.columns:
        prev_df = prev_df[prev_df["underlying"].astype(str).str.upper() == u]
    if prev_df.empty:
        return None
    if "timestamp" in prev_df.columns:
        prev_df = prev_df.sort_values("timestamp")
    spot = float(prev_df.iloc[-1]["spot_ltp"] or 0)
    return spot if spot > 0 else None


def load_prior_session_summaries(
    data_dir,
    underlying: str,
    trade_date: date,
    *,
    lookback_bars: int = 20,
    tail_bars: int = PRIOR_WARMUP_BARS,
) -> list[dict[str, Any]]:
    """Build summary dicts for the tail of the prior session (for ATR warmup)."""
    store = DataStore(data_dir, underlying=underlying)
    prev_date = prior_trading_date(trade_date)
    prev_df = store.load_snapshots(trade_date=prev_date)
    if prev_df.empty:
        return []
    u = underlying.upper()
    if "underlying" in prev_df.columns:
        prev_df = prev_df[prev_df["underlying"].astype(str).str.upper() == u]
    if prev_df.empty or "timestamp" not in prev_df.columns:
        return []
    prev_df = prev_df.sort_values("timestamp").reset_index(drop=True)
    prev_sig = store.load_signals(trade_date=prev_date)
    if not prev_sig.empty and "timestamp" in prev_sig.columns:
        if "underlying" in prev_sig.columns:
            prev_sig = prev_sig[prev_sig["underlying"].astype(str).str.upper() == u]
    summaries = [
        s for s in build_analysis_summaries(prev_df, prev_sig, lookback=lookback_bars) if s
    ]
    if len(summaries) > tail_bars:
        summaries = summaries[-tail_bars:]
    return summaries
