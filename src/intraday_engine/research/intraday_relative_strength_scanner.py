"""
Intraday Relative Strength vs NIFTY 50 — uses the cached 15-min NIFTY 500 bars
written by the regular 15-min cycle (data/NIFTY500/SYMBOL_15Min.csv).

For each stock we measure today's % move from the day's first 15-min OPEN to the
latest 15-min CLOSE, and compare it against NIFTY 50's same measure (fetched from
Kite and cached at data/reference/nifty_index/NIFTY50_15Min.csv).

  • pct_today   = (last_close / first_open - 1) * 100
  • excess_pct  = stock pct_today - nifty pct_today   (intraday RS, in % points)
  • rs_line     = per-bar intraday ratio of (stock_close/stock_open)
                  divided by (nifty_close/nifty_open), normalised to 100 baseline.
                  Values > 100 mean the stock is currently outperforming the
                  index since the 09:15 open of the session.

Outputs:
  stronger = stocks with excess_pct > 0  (descending)
  weaker   = stocks with excess_pct < 0  (ascending)

The scan only reads cached 15-min CSVs for the stock universe; no per-stock
historical_data calls are made here.
"""

from __future__ import annotations

import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd

from intraday_engine.core.config import Settings
from intraday_engine.fetch.zerodha_client import ZerodhaClient
from intraday_engine.research.nifty500_accumulation_scanner import load_nifty500_symbols
from intraday_engine.storage.layout import (
    intraday_relative_strength_path,
    nifty_index_15min_path,
)
from intraday_engine.storage.nifty500_csv import read_nifty500_symbol_ohlcv

logger = logging.getLogger(__name__)

NIFTY_SPOT_SYMBOL = "NSE:NIFTY 50"
SPARKLINE_MAX_POINTS = 26  # one full 15-min trading session = 25 bars


@dataclass
class IntradayRsRow:
    stock: str
    bars: int
    open_price: float
    last_price: float
    pct_today: float          # stock % move today (open->last)
    nifty_pct_today: float    # NIFTY 50 % move today (open->last)
    excess_pct: float         # stock - nifty (intraday RS, % points)
    high: float
    low: float
    last_bar_ts: str
    rs_line: list[float] = field(default_factory=list)
    # 100-baseline intraday RS path. >100 = leading NIFTY, <100 = lagging.


def _load_nifty_15min(settings: Settings, session_day: date) -> pd.DataFrame:
    """
    Return NIFTY 50 15-min OHLCV. Tries cache first; if last cached bar is older
    than session_day, refreshes via Kite for the last ~6 days.
    """
    cache_path = nifty_index_15min_path(settings.data_dir)
    cached = pd.DataFrame()
    if cache_path.exists():
        try:
            cached = pd.read_csv(cache_path)
            cached["date"] = pd.to_datetime(cached["date"], errors="coerce")
            cached = cached.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
        except Exception:
            cached = pd.DataFrame()

    def _has_session(df: pd.DataFrame) -> bool:
        if df is None or df.empty:
            return False
        last = pd.Timestamp(df["date"].iloc[-1]).normalize()
        return last >= pd.Timestamp(session_day)

    if _has_session(cached):
        return cached

    try:
        client = ZerodhaClient(settings)
        quote = client.quote([NIFTY_SPOT_SYMBOL])
        token = int(quote[NIFTY_SPOT_SYMBOL]["instrument_token"])
        now = datetime.now()
        from_dt = (datetime.combine(session_day, datetime.min.time()) - timedelta(days=6))
        rows = client.historical_data(token, from_dt, now, interval="15minute", oi=False)
        if rows:
            df = pd.DataFrame(rows)
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
            for c in ("open", "high", "low", "close", "volume"):
                if c in df.columns:
                    df[c] = pd.to_numeric(df[c], errors="coerce")
            df = df.dropna(subset=["date", "close"]).sort_values("date").reset_index(drop=True)
            out = df[["date", "open", "high", "low", "close", "volume"]].copy()
            out["date"] = out["date"].dt.tz_localize(None) if hasattr(out["date"].dt, "tz_localize") else out["date"]
            try:
                out["date"] = pd.to_datetime(out["date"]).dt.tz_convert("Asia/Kolkata").dt.tz_localize(None)
            except (TypeError, AttributeError):
                out["date"] = pd.to_datetime(out["date"])
            out["date"] = pd.to_datetime(out["date"]).dt.strftime("%Y-%m-%d %H:%M:%S")
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            out.to_csv(cache_path, index=False)
            df["date"] = pd.to_datetime(out["date"])
            return df
    except Exception as e:
        logger.warning("Could not refresh NIFTY 50 15-min bars: %s", e)

    return cached


def _slice_for_session(df: pd.DataFrame, session_day: date) -> pd.DataFrame:
    if df is None or df.empty or "date" not in df.columns:
        return pd.DataFrame()
    day_ts = pd.Timestamp(session_day)
    return df[pd.to_datetime(df["date"]).dt.normalize() == day_ts].reset_index(drop=True)


def _resolve_session_day(nifty_df: pd.DataFrame, requested: date) -> date:
    """Use the latest date present in NIFTY data <= requested. Falls back to requested."""
    if nifty_df is None or nifty_df.empty:
        return requested
    days = pd.to_datetime(nifty_df["date"]).dt.normalize().drop_duplicates().sort_values()
    on_or_before = days[days <= pd.Timestamp(requested)]
    if not on_or_before.empty:
        return on_or_before.iloc[-1].date()
    return requested


def _resolve_stock_aligned_session_day(
    data_dir: Path,
    symbols: list[str],
    nifty_df: pd.DataFrame,
    requested: date,
    *,
    min_bars: int = 2,
    sample_size: int = 80,
    min_fraction: float = 0.35,
) -> date:
    """
    Pick the latest session where NIFTY 15-min bars AND stock 15-min caches overlap.
    NIFTY is often refreshed live for today while NIFTY500 15-min CSVs lag until the
    intraday job runs — using NIFTY's latest day alone yields 0 stock rows.
    """
    if nifty_df is None or nifty_df.empty:
        return requested
    nifty_days = sorted(pd.to_datetime(nifty_df["date"]).dt.normalize().unique())
    candidates = [d for d in nifty_days if d <= pd.Timestamp(requested)]
    if not candidates:
        return requested

    if not symbols:
        return candidates[-1].date()

    step = max(1, len(symbols) // sample_size)
    sample = symbols[::step][:sample_size]

    for day_ts in reversed(candidates):
        session = day_ts.date()
        have = 0
        for sym in sample:
            try:
                df = read_nifty500_symbol_ohlcv(data_dir, sym, "15Min")
                today = _slice_for_session(df, session)
                if not today.empty and len(today) >= min_bars:
                    have += 1
            except Exception:
                continue
        if have / len(sample) >= min_fraction:
            return session

    return candidates[-1].date()


def _compute_row(
    symbol: str,
    stock_df: pd.DataFrame,
    nifty_today: pd.DataFrame,
    nifty_pct: float,
) -> IntradayRsRow | None:
    if stock_df.empty or nifty_today.empty:
        return None
    n_open = float(nifty_today["open"].iloc[0])
    if n_open <= 0:
        return None

    open_p = float(stock_df["open"].iloc[0])
    last_p = float(stock_df["close"].iloc[-1])
    if open_p <= 0:
        return None
    pct = (last_p / open_p - 1.0) * 100.0
    excess = pct - nifty_pct

    # Align stock and nifty bar-by-bar on timestamp for the intraday RS line.
    s = stock_df[["date", "close"]].copy()
    n = nifty_today[["date", "close"]].copy()
    s["date"] = pd.to_datetime(s["date"])
    n["date"] = pd.to_datetime(n["date"])
    merged = s.merge(n, on="date", how="inner", suffixes=("_s", "_n"))
    rs_line: list[float] = []
    if not merged.empty:
        s_norm = merged["close_s"].astype(float) / open_p
        n_norm = merged["close_n"].astype(float) / n_open
        ratio = (s_norm / n_norm) * 100.0
        rs_line = [round(float(v), 3) for v in ratio.tolist()][-SPARKLINE_MAX_POINTS:]

    last_ts = pd.to_datetime(stock_df["date"].iloc[-1]).strftime("%H:%M")

    return IntradayRsRow(
        stock=symbol,
        bars=int(len(stock_df)),
        open_price=round(open_p, 2),
        last_price=round(last_p, 2),
        pct_today=round(pct, 3),
        nifty_pct_today=round(nifty_pct, 3),
        excess_pct=round(excess, 3),
        high=round(float(stock_df["high"].max()), 2),
        low=round(float(stock_df["low"].min()), 2),
        last_bar_ts=last_ts,
        rs_line=rs_line,
    )


def run_intraday_relative_strength_scan(
    *,
    settings: Settings | None = None,
    symbols_file: Path | None = None,
    trade_date: date | None = None,
    max_workers: int = 8,
    min_bars: int = 2,
    top_n: int | None = None,
) -> dict[str, Any]:
    settings = settings or Settings.from_env(underlying="NIFTY")
    requested = trade_date or date.today()

    nifty_full = _load_nifty_15min(settings, requested)
    symbols = load_nifty500_symbols(symbols_file, settings.data_dir)
    session_day = _resolve_stock_aligned_session_day(
        settings.data_dir, symbols, nifty_full, requested, min_bars=min_bars
    )
    nifty_today = _slice_for_session(nifty_full, session_day)

    if nifty_today.empty or len(nifty_today) < min_bars:
        payload = {
            "trade_date": requested.isoformat(),
            "session_day": session_day.isoformat() if session_day else None,
            "generated_at": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
            "nifty": None,
            "stronger": [],
            "weaker": [],
            "scanned": 0,
            "message": (
                "Could not load NIFTY 50 15-min bars for the session. "
                "Ensure Kite credentials are configured (NIFTY 50 spot is fetched live)."
            ),
        }
        _persist(payload, settings.data_dir, requested)
        return payload

    n_open = float(nifty_today["open"].iloc[0])
    n_last = float(nifty_today["close"].iloc[-1])
    nifty_pct = (n_last / n_open - 1.0) * 100.0 if n_open > 0 else 0.0
    nifty_meta = {
        "open": round(n_open, 2),
        "last": round(n_last, 2),
        "pct_today": round(nifty_pct, 3),
        "bars": int(len(nifty_today)),
        "high": round(float(nifty_today["high"].max()), 2),
        "low": round(float(nifty_today["low"].min()), 2),
        "last_bar_ts": pd.to_datetime(nifty_today["date"].iloc[-1]).strftime("%H:%M"),
    }

    def job(sym: str) -> IntradayRsRow | None:
        try:
            df = read_nifty500_symbol_ohlcv(settings.data_dir, sym, "15Min")
            if df.empty:
                return None
            today = _slice_for_session(df, session_day)
            if today.empty or len(today) < min_bars:
                return None
            return _compute_row(sym, today, nifty_today, nifty_pct)
        except Exception as e:
            logger.debug("Intraday RS %s err: %s", sym, e)
            return None

    rows: list[IntradayRsRow] = []
    skipped = 0
    with ThreadPoolExecutor(max_workers=max(1, max_workers)) as ex:
        futs = {ex.submit(job, s): s for s in symbols}
        for fut in as_completed(futs):
            try:
                r = fut.result()
            except Exception:
                r = None
            if r is None:
                skipped += 1
            else:
                rows.append(r)

    stronger = sorted([r for r in rows if r.excess_pct > 0], key=lambda r: r.excess_pct, reverse=True)
    weaker = sorted([r for r in rows if r.excess_pct < 0], key=lambda r: r.excess_pct)
    if top_n is not None and top_n > 0:
        stronger = stronger[:top_n]
        weaker = weaker[:top_n]

    session_note = None
    if session_day != requested:
        session_note = (
            f"Using {session_day.isoformat()} (latest session with NIFTY500 15-min bars). "
            f"Today's stock 15-min cache is not ready yet — run 30-min Analysis during market hours."
        )
    elif not rows:
        session_note = (
            f"No NIFTY500 symbols had 15-min bars for {session_day.isoformat()}. "
            "Refresh 30-min Analysis during market hours to populate data/NIFTY500/*_15Min.csv."
        )

    payload = {
        "trade_date": requested.isoformat(),
        "session_day": session_day.isoformat(),
        "generated_at": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
        "nifty": nifty_meta,
        "stronger": [asdict(r) for r in stronger],
        "weaker": [asdict(r) for r in weaker],
        "scanned": len(symbols),
        "passed": len(rows),
        "skipped": skipped,
        "session_note": session_note,
    }
    _persist(payload, settings.data_dir, requested)
    logger.info(
        "Intraday RS (%s): %d stronger / %d weaker (NIFTY %s%%, %d bars)",
        session_day,
        len(stronger),
        len(weaker),
        f"{nifty_pct:+.2f}",
        nifty_meta["bars"],
    )
    return payload


def _persist(payload: dict[str, Any], data_dir: Path, trade_date: date) -> None:
    out_path = intraday_relative_strength_path(data_dir, trade_date)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_stored_intraday_relative_strength(data_dir: Path, trade_date: date) -> dict[str, Any] | None:
    p = intraday_relative_strength_path(data_dir, trade_date)
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            return None
    folder = p.parent
    if not folder.exists():
        return None
    files = sorted(folder.glob("intraday_rs_*.json"))
    if not files:
        return None
    try:
        return json.loads(files[-1].read_text(encoding="utf-8"))
    except Exception:
        return None
