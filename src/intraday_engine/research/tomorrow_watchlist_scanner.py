"""
NIFTY 500 — multi-timeframe scan for next-session directional candidates.

Fetches **15-minute** equity history only (chunked backward to cover enough
sessions under Kite’s per-request candle limits), then **aggregates** to 1D and
1H OHLCV locally. Raw 15m is still used for session VWAP and compression checks.
Optional front-month **future daily** OI (separate token) when FnO exists.

Run from CLI: python -m intraday_engine.main --tomorrow-watchlist
"""

from __future__ import annotations

import json
import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from intraday_engine.core.config import Settings
from intraday_engine.fetch.zerodha_client import ZerodhaClient
from intraday_engine.research.nifty500_accumulation_scanner import load_nifty500_symbols
from intraday_engine.storage.layout import tomorrow_watchlist_json_path
from intraday_engine.storage.nifty500_csv import (
    read_nifty500_symbol_ohlcv,
    write_nifty500_symbol_ohlcv,
)

logger = logging.getLogger(__name__)

_rate_lock = threading.Lock()
_rate_last = 0.0
FETCH_STALE_AFTER_DAYS = 90
INCREMENTAL_OVERLAP_DAYS = 7
SKIP_NETWORK_IF_FRESH_MINUTES = 20


def _throttle(sec: float = 0.28) -> None:
    global _rate_last
    with _rate_lock:
        now = time.monotonic()
        wait = sec - (now - _rate_last)
        if wait > 0:
            time.sleep(wait)
        _rate_last = time.monotonic()


def _rows_to_df(rows: list[dict[str, Any]]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    if "date" not in df.columns:
        return pd.DataFrame()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["date"] = df["date"].map(lambda z: _ts_naive_ist(pd.Timestamp(z)) if not pd.isna(z) else pd.NaT)
    for c in ("open", "high", "low", "close", "volume"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    if "oi" in df.columns:
        df["oi"] = pd.to_numeric(df["oi"], errors="coerce")
    return df.sort_values("date").reset_index(drop=True)


def _ts_naive_ist(ts: pd.Timestamp) -> pd.Timestamp:
    t = pd.Timestamp(ts)
    if t.tzinfo is not None:
        try:
            t = t.tz_convert("Asia/Kolkata").tz_localize(None)
        except Exception:
            t = t.replace(tzinfo=None)
    return t


def _mins_from_nse_open(ts: pd.Timestamp) -> int:
    """Minutes from same-calendar-day 09:15 (session open). Negative = before open."""
    t = _ts_naive_ist(ts)
    midnight = pd.Timestamp(t.date())
    open_t = midnight + pd.Timedelta(hours=9, minutes=15)
    return int((t - open_t) / pd.Timedelta(minutes=1))


def _nse_hour_bucket_start(ts: pd.Timestamp) -> pd.Timestamp | None:
    """Hourly bar start aligned to NSE 09:15, 10:15, … (four 15m bars per hour)."""
    t = _ts_naive_ist(ts)
    mins = _mins_from_nse_open(t)
    if mins < 0:
        return None
    slot = mins // 60
    midnight = pd.Timestamp(t.date())
    return midnight + pd.Timedelta(hours=9, minutes=15 + slot * 60)


def _aggregate_daily_from_m15(m15: pd.DataFrame) -> pd.DataFrame:
    if m15.empty:
        return pd.DataFrame()
    x = m15.dropna(subset=["date", "open", "high", "low", "close"]).copy()
    x["day"] = x["date"].apply(lambda z: _ts_naive_ist(pd.Timestamp(z)).normalize())
    g = x.groupby("day", sort=True)
    out = g.agg(
        open=("open", "first"),
        high=("high", "max"),
        low=("low", "min"),
        close=("close", "last"),
        volume=("volume", "sum"),
    ).reset_index()
    out = out.rename(columns={"day": "date"})
    return out.sort_values("date").reset_index(drop=True)


def _aggregate_hourly_from_m15(m15: pd.DataFrame) -> pd.DataFrame:
    if m15.empty:
        return pd.DataFrame()
    x = m15.dropna(subset=["date", "open", "high", "low", "close"]).copy()
    x["hb"] = x["date"].map(_nse_hour_bucket_start)
    x = x.dropna(subset=["hb"])
    if x.empty:
        return pd.DataFrame()
    g = x.groupby("hb", sort=True)
    out = g.agg(
        open=("open", "first"),
        high=("high", "max"),
        low=("low", "min"),
        close=("close", "last"),
        volume=("volume", "sum"),
    ).reset_index()
    out = out.rename(columns={"hb": "date"})
    return out.sort_values("date").reset_index(drop=True)


def _fetch_m15_chained(
    client: ZerodhaClient,
    token: int,
    to_dt: datetime,
    *,
    segment_calendar_days: int = 55,
    max_segments: int = 8,
) -> pd.DataFrame:
    """
    Walk backward with repeated 15m requests (Kite caps ~2000 candles per call)
    until max_segments or an empty/partial response.
    """
    parts: list[pd.DataFrame] = []
    end = to_dt
    for _ in range(max_segments):
        start = end - timedelta(days=segment_calendar_days)
        _throttle()
        try:
            rows = client.historical_data(token, start, end, interval="15minute", oi=False)
        except Exception:
            break
        if not rows:
            break
        df = _rows_to_df(rows)
        if df.empty:
            break
        parts.append(df)
        oldest = df["date"].min()
        if pd.isna(oldest):
            break
        oldest_dt = pd.Timestamp(oldest).to_pydatetime()
        if getattr(oldest_dt, "tzinfo", None):
            oldest_dt = oldest_dt.replace(tzinfo=None)
        end = oldest_dt - timedelta(seconds=1)
        if len(df) < 25:
            break
    if not parts:
        return pd.DataFrame()
    full = pd.concat(parts, ignore_index=True)
    full = full.sort_values("date").drop_duplicates(subset=["date"], keep="last").reset_index(drop=True)
    return full


def _incremental_fetch_from_dt(
    existing: pd.DataFrame, to_dt: datetime, full_history_days: int = 380
) -> datetime:
    if existing.empty or "date" not in existing.columns:
        return to_dt - timedelta(days=full_history_days)
    try:
        last = (
            pd.to_datetime(existing["date"], errors="coerce")
            .dropna()
            .map(lambda z: _ts_naive_ist(pd.Timestamp(z)))
            .max()
        )
        if pd.isna(last):
            return to_dt - timedelta(days=full_history_days)
        last_dt = pd.Timestamp(last).to_pydatetime()
        if getattr(last_dt, "tzinfo", None):
            last_dt = last_dt.replace(tzinfo=None)
    except Exception:
        return to_dt - timedelta(days=full_history_days)
    if to_dt - last_dt > timedelta(days=FETCH_STALE_AFTER_DAYS):
        return to_dt - timedelta(days=full_history_days)
    return last_dt - timedelta(days=INCREMENTAL_OVERLAP_DAYS)


def _should_skip_network_fetch(existing: pd.DataFrame, to_dt: datetime) -> bool:
    if existing.empty or "date" not in existing.columns:
        return False
    try:
        last = (
            pd.to_datetime(existing["date"], errors="coerce")
            .dropna()
            .map(lambda z: _ts_naive_ist(pd.Timestamp(z)))
            .max()
        )
        if pd.isna(last):
            return False
        last_dt = pd.Timestamp(last).to_pydatetime()
        if getattr(last_dt, "tzinfo", None):
            last_dt = last_dt.replace(tzinfo=None)
        return (to_dt - last_dt) <= timedelta(minutes=SKIP_NETWORK_IF_FRESH_MINUTES)
    except Exception:
        return False


def _merge_ohlcv(existing: pd.DataFrame, fresh: pd.DataFrame) -> pd.DataFrame:
    if existing.empty and fresh.empty:
        return pd.DataFrame()
    if existing.empty:
        return fresh.sort_values("date").drop_duplicates(subset=["date"], keep="last").reset_index(drop=True)
    if fresh.empty:
        return existing.sort_values("date").drop_duplicates(subset=["date"], keep="last").reset_index(drop=True)
    merged = pd.concat([existing, fresh], ignore_index=True)
    return merged.sort_values("date").drop_duplicates(subset=["date"], keep="last").reset_index(drop=True)


def _fetch_m15_incremental(client: ZerodhaClient, token: int, to_dt: datetime, cached_m15: pd.DataFrame) -> pd.DataFrame:
    if _should_skip_network_fetch(cached_m15, to_dt):
        return cached_m15

    from_dt = _incremental_fetch_from_dt(cached_m15, to_dt)
    if cached_m15.empty or (to_dt - from_dt) > timedelta(days=60):
        return _fetch_m15_chained(client, token, to_dt)

    _throttle()
    rows = client.historical_data(token, from_dt, to_dt, interval="15minute", oi=False)
    fresh = _rows_to_df(rows)
    return _merge_ohlcv(cached_m15, fresh)


def _drop_partial_last_session_if_thin(m15: pd.DataFrame, min_bars_per_session: int = 12) -> pd.DataFrame:
    """If the newest session has very few 15m bars, drop it so daily OHLC is not distorted."""
    if m15.empty or "date" not in m15.columns:
        return m15
    dates = pd.to_datetime(m15["date"])
    last_d = dates.dt.date.iloc[-1]
    n_last = int((dates.dt.date == last_d).sum())
    if n_last < min_bars_per_session:
        return m15.loc[dates.dt.date != last_d].reset_index(drop=True)
    return m15


def _ema(s: pd.Series, span: int) -> pd.Series:
    return s.ewm(span=span, adjust=False, min_periods=span).mean()


def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def _atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    h, l, c = df["high"], df["low"], df["close"]
    prev_c = c.shift(1)
    tr = pd.concat([(h - l), (h - prev_c).abs(), (l - prev_c).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()


def _nearest_fut_row(client: ZerodhaClient, stock: str, today: date) -> dict[str, Any] | None:
    futs = [
        r
        for r in client.nfo_instruments()
        if isinstance(r, dict)
        and str(r.get("name", "")).strip().upper() == stock.upper()
        and str(r.get("instrument_type", "")).upper() == "FUT"
        and r.get("expiry")
        and r["expiry"] >= today
    ]
    if not futs:
        return None
    futs.sort(key=lambda x: x["expiry"])
    return futs[0]


def _two_day_range_pct(d: pd.DataFrame) -> float:
    if len(d) < 2:
        return 0.0
    tail = d.iloc[-2:]
    lo = float(tail["low"].min())
    hi = float(tail["high"].max())
    if lo <= 0:
        return 0.0
    return (hi - lo) / lo * 100.0


def _distribution_daily(last: pd.Series, vol_ratio: float) -> bool:
    """High-volume bearish candle (avoid chasing distribution)."""
    o, h, l, c, v = (
        float(last["open"]),
        float(last["high"]),
        float(last["low"]),
        float(last["close"]),
        float(last["volume"]),
    )
    body = abs(c - o)
    rng = max(h - l, 1e-9)
    if c >= o:
        return False
    if vol_ratio < 1.35:
        return False
    return body / rng >= 0.45


def _resistance_zone(d: pd.DataFrame, look: int = 60, exclude_last: int = 2) -> tuple[float, int]:
    if len(d) < exclude_last + 25:
        return 0.0, 0
    window = d.iloc[-(look + exclude_last) : -exclude_last]
    res = float(window["high"].max())
    if res <= 0:
        return 0.0, 0
    touches = int((d.iloc[-look:]["high"] >= res * 0.99).sum())
    return res, touches


def _support_zone(d: pd.DataFrame, look: int = 60, exclude_last: int = 2) -> tuple[float, int]:
    if len(d) < exclude_last + 25:
        return 0.0, 0
    window = d.iloc[-(look + exclude_last) : -exclude_last]
    sup = float(window["low"].min())
    if sup <= 0:
        return 0.0, 0
    touches = int((d.iloc[-look:]["low"] <= sup * 1.01).sum())
    return sup, touches


def _nr7(d: pd.DataFrame) -> bool:
    if len(d) < 8:
        return False
    r = (d["high"] - d["low"]).iloc[-7:]
    last_r = float((d["high"] - d["low"]).iloc[-1])
    return last_r <= float(r.min()) * 1.001


def _session_vwap_above(m15: pd.DataFrame, session_day: pd.Timestamp) -> bool | None:
    if m15.empty or "volume" not in m15.columns:
        return None
    day = pd.to_datetime(session_day).normalize()
    sub = m15.loc[pd.to_datetime(m15["date"]).dt.normalize() == day]
    if len(sub) < 4:
        return None
    tp = (sub["high"] + sub["low"] + sub["close"]) / 3.0
    vol = sub["volume"].astype(float)
    if float(vol.sum()) <= 0:
        return None
    vwap = float((tp * vol).sum() / vol.sum())
    last_c = float(sub["close"].iloc[-1])
    return last_c >= vwap * 0.998


def _intraday_1h_flags(h1: pd.DataFrame) -> dict[str, Any]:
    out: dict[str, Any] = {"compress_1h": False, "higher_lows_1h": False, "upper_range_1h": False}
    if len(h1) < 30:
        return out
    tail = h1.iloc[-24:].reset_index(drop=True)
    rh, rl = float(tail["high"].max()), float(tail["low"].min())
    mid = (rh + rl) / 2.0
    if mid <= 0:
        return out
    width = (rh - rl) / mid
    out["compress_1h"] = width < 0.055
    lc = float(tail["close"].iloc[-1])
    out["upper_range_1h"] = lc >= rl + (rh - rl) * 0.62
    lows = tail["low"].values
    out["higher_lows_1h"] = len(lows) >= 9 and float(lows[-1]) > float(lows[-5]) > float(lows[-9])
    return out


def _m15_tight_and_vwap(m15: pd.DataFrame, last_daily_ts: pd.Timestamp) -> tuple[bool, bool | None]:
    if len(m15) < 12:
        return False, None
    tail = m15.iloc[-8:]
    ranges = (tail["high"] - tail["low"]).astype(float)
    closes = tail["close"].astype(float)
    avg_pct = float((ranges / closes.replace(0, np.nan)).mean()) * 100.0
    tight = avg_pct < 0.45
    vwap_ok = _session_vwap_above(m15, last_daily_ts)
    return tight, vwap_ok


def _fut_oi_reading(fut_df: pd.DataFrame, bullish: bool) -> str:
    if fut_df.empty or len(fut_df) < 3 or "oi" not in fut_df.columns:
        return "N/A"
    c1, c0 = fut_df.iloc[-2], fut_df.iloc[-1]
    pc = float(c0["close"]) / float(c1["close"]) - 1.0 if float(c1["close"]) > 0 else 0.0
    oi1, oi0 = float(c1.get("oi", 0) or 0), float(c0.get("oi", 0) or 0)
    doi = oi0 - oi1
    if abs(pc) < 0.0008 and abs(doi) / max(oi1, 1) < 0.003:
        return "Flat / mixed"
    if bullish:
        if pc > 0 and doi > 0:
            return "Long buildup (price↑ OI↑)"
        if pc > 0 and doi < 0:
            return "Short covering (OI↓) — use caution"
        if pc < 0 and doi > 0:
            return "Short buildup (price↓ OI↑)"
        return "Bearish / OI not confirming long"
    else:
        if pc < 0 and doi > 0:
            return "Short buildup (price↓ OI↑)"
        if pc < 0 and doi < 0:
            return "Long unwinding (OI↓)"
        if pc > 0 and doi < 0:
            return "Short covering bounce — weak fade"
        return "Bullish / OI not confirming short"


def _volume_profile(d: pd.DataFrame, vol_ratio: float) -> str:
    if len(d) < 5:
        return "Low"
    v = d["volume"].astype(float).iloc[-4:]
    rising = bool(v.iloc[-1] >= v.iloc[-2] >= v.iloc[-3])
    if vol_ratio >= 1.5:
        return "High" if rising or vol_ratio >= 2.0 else "High (spike)"
    if vol_ratio >= 1.15 and rising:
        return "Rising"
    if vol_ratio >= 1.15:
        return "Above avg"
    return "Low"


def _score_bullish(
    *,
    dist_res: float,
    res_touches: int,
    cons_width: float,
    atr_ratio: float,
    vol_ratio: float,
    vol_prof: str,
    dist_flag: dict[str, Any],
    m15_tight: bool,
    vwap_ok: bool | None,
    oi_txt: str,
    setup: str,
    distribution: bool,
) -> float:
    if distribution:
        return 0.0
    s = 22.0
    if 0 <= dist_res <= 2.0:
        s += 18.0
    elif dist_res <= 2.5:
        s += 12.0
    if res_touches >= 3:
        s += 10.0
    elif res_touches >= 2:
        s += 6.0
    if cons_width < 0.08:
        s += 8.0
    if atr_ratio < 0.92:
        s += 8.0
    elif atr_ratio < 0.98:
        s += 4.0
    if vol_ratio >= 1.5:
        s += 12.0
    elif vol_ratio >= 1.2:
        s += 6.0
    if vol_prof in ("Rising", "High", "High (spike)", "Above avg"):
        s += 5.0
    if dist_flag.get("compress_1h"):
        s += 5.0
    if dist_flag.get("higher_lows_1h"):
        s += 5.0
    if dist_flag.get("upper_range_1h"):
        s += 4.0
    if m15_tight:
        s += 4.0
    if vwap_ok is True:
        s += 5.0
    if "Long buildup" in oi_txt:
        s += 8.0
    if "Short covering" in oi_txt:
        s -= 6.0
    if setup == "Continuation":
        s += 6.0
    if setup == "Reversal":
        s += 4.0
    return max(0.0, min(100.0, s))


def _score_bearish(
    *,
    dist_sup: float,
    sup_touches: int,
    cons_width: float,
    atr_ratio: float,
    vol_ratio: float,
    vol_prof: str,
    dist_flag: dict[str, Any],
    m15_tight: bool,
    vwap_ok: bool | None,
    oi_txt: str,
    setup: str,
    distribution: bool,
) -> float:
    s = 22.0
    if distribution and vol_ratio >= 1.35:
        s += 5.0
    if 0 <= dist_sup <= 2.0:
        s += 18.0
    elif dist_sup <= 2.5:
        s += 12.0
    if sup_touches >= 3:
        s += 10.0
    elif sup_touches >= 2:
        s += 6.0
    if cons_width < 0.08:
        s += 8.0
    if atr_ratio < 0.92:
        s += 8.0
    elif atr_ratio < 0.98:
        s += 4.0
    if vol_ratio >= 1.5:
        s += 12.0
    elif vol_ratio >= 1.2:
        s += 6.0
    if vol_prof in ("Rising", "High", "High (spike)", "Above avg"):
        s += 5.0
    if dist_flag.get("compress_1h"):
        s += 5.0
    if m15_tight:
        s += 4.0
    if vwap_ok is False:
        s += 5.0
    if "Short buildup" in oi_txt:
        s += 8.0
    if setup == "Continuation":
        s += 6.0
    if setup == "Reversal":
        s += 4.0
    return max(0.0, min(100.0, s))


def _bullish_setups(d: pd.DataFrame) -> tuple[str, list[str]]:
    """Return (setup_type, reason_lines)."""
    reasons: list[str] = []
    last = d.iloc[-1]
    close, open_ = float(last["close"]), float(last["open"])
    ema20 = _ema(d["close"], 20)
    ema50 = _ema(d["close"], 50)
    rsi = _rsi(d["close"], 14)
    e20, e50 = float(ema20.iloc[-1]), float(ema50.iloc[-1])
    rv = float(rsi.iloc[-1]) if not pd.isna(rsi.iloc[-1]) else 50.0
    rv3 = float(rsi.iloc[-4]) if len(rsi) > 4 and not pd.isna(rsi.iloc[-4]) else rv

    res, rt = _resistance_zone(d)
    dist_res = (res - close) / res * 100.0 if res > 0 else 999.0
    cons = d.iloc[-18:-1]
    c_hi, c_lo = float(cons["high"].max()), float(cons["low"].min())
    cons_width = (c_hi - c_lo) / ((c_hi + c_lo) / 2) if c_hi > 0 else 1.0

    breakout = (
        res > 0
        and 0 <= dist_res <= 2.5
        and rt >= 2
        and cons_width < 0.09
        and close < res * 1.002
    )

    uptrend = close > e50 and e20 > e50 * 0.998
    pullback = float(d["low"].iloc[-5:].min()) <= float(ema20.iloc[-1]) * 1.02 and close > float(ema20.iloc[-1])
    bull_candle = close >= open_
    continuation = uptrend and pullback and bull_candle

    sup, st = _support_zone(d)
    dist_sup_pct = (close - sup) / sup * 100.0 if sup > 0 else 999.0
    body = abs(close - open_)
    rng = float(last["high"]) - float(last["low"])
    lower_wick = min(open_, close) - float(last["low"])
    hammer = rng > 0 and body / rng < 0.35 and lower_wick / rng >= 0.45
    eng = len(d) >= 2 and close > float(d.iloc[-2]["open"]) and open_ < float(d.iloc[-2]["close"]) and bull_candle
    rsi_bounce = rv3 < 40 and rv > float(rsi.iloc[-2]) if len(rsi) > 1 else False
    reversal = sup > 0 and st >= 2 and dist_sup_pct <= 3.5 and (hammer or eng or rsi_bounce)

    if breakout:
        reasons.append(f"Daily: within {dist_res:.2f}% of resistance {res:.2f} with {rt} touches; tight ~{cons_width*100:.1f}% base.")
        return "Breakout", reasons
    if continuation:
        reasons.append("Daily: uptrend (HH/HL vs 50 EMA); pullback into ~20 EMA with bullish close.")
        return "Continuation", reasons
    if reversal:
        reasons.append(f"Daily: support ~{sup:.2f} ({st} touches); reversal / RSI bounce context.")
        return "Reversal", reasons
    return "", []


def _bearish_setups(d: pd.DataFrame) -> tuple[str, list[str]]:
    reasons: list[str] = []
    last = d.iloc[-1]
    close, open_ = float(last["close"]), float(last["open"])
    ema20 = _ema(d["close"], 20)
    ema50 = _ema(d["close"], 50)
    rsi = _rsi(d["close"], 14)
    e20, e50 = float(ema20.iloc[-1]), float(ema50.iloc[-1])
    rv = float(rsi.iloc[-1]) if not pd.isna(rsi.iloc[-1]) else 50.0
    rv3 = float(rsi.iloc[-4]) if len(rsi) > 4 and not pd.isna(rsi.iloc[-4]) else rv

    sup, st = _support_zone(d)
    dist_sup = (close - sup) / sup * 100.0 if sup > 0 else 999.0
    cons = d.iloc[-18:-1]
    c_hi, c_lo = float(cons["high"].max()), float(cons["low"].min())
    cons_width = (c_hi - c_lo) / ((c_hi + c_lo) / 2) if c_hi > 0 else 1.0

    breakout = (
        sup > 0
        and 0 <= dist_sup <= 2.5
        and st >= 2
        and cons_width < 0.09
        and close > sup * 0.998
    )

    dntrend = close < e50 and e20 < e50 * 1.002
    pullback = float(d["high"].iloc[-5:].max()) >= float(ema20.iloc[-1]) * 0.98 and close < float(ema20.iloc[-1])
    bear_candle = close <= open_
    continuation = dntrend and pullback and bear_candle

    res, rt = _resistance_zone(d)
    dist_res_pct = (res - close) / res * 100.0 if res > 0 else 999.0
    body = abs(close - open_)
    rng = float(last["high"]) - float(last["low"])
    upper_wick = float(last["high"]) - max(open_, close)
    shoot = rng > 0 and body / rng < 0.35 and upper_wick / rng >= 0.45
    eng = len(d) >= 2 and close < float(d.iloc[-2]["open"]) and open_ > float(d.iloc[-2]["close"]) and bear_candle
    rsi_fail = rv3 > 60 and rv < float(rsi.iloc[-2]) if len(rsi) > 1 else False
    reversal = res > 0 and rt >= 2 and dist_res_pct <= 3.5 and (shoot or eng or rsi_fail)

    if breakout:
        reasons.append(f"Daily: within {dist_sup:.2f}% of support {sup:.2f} with {st} touches; breakdown prep.")
        return "Breakout", reasons
    if continuation:
        reasons.append("Daily: downtrend vs 50 EMA; pullback to ~20 EMA with bearish close.")
        return "Continuation", reasons
    if reversal:
        reasons.append(f"Daily: resistance ~{res:.2f} ({rt} touches); rejection / RSI rollover context.")
        return "Reversal", reasons
    return "", []


def _htf_exhaustion_bull(d: pd.DataFrame) -> bool:
    """Very tight to 120d high + bearish close — skip chasing immediate HTF supply."""
    if len(d) < 130:
        return False
    hi120 = float(d["high"].iloc[-120:].max())
    last = d.iloc[-1]
    close, open_ = float(last["close"]), float(last["open"])
    if hi120 <= 0:
        return False
    at_high = (hi120 - close) / hi120 * 100.0 < 0.35
    return at_high and close < open_


def analyze_stock_pack(
    client: ZerodhaClient,
    sym: str,
    token: int,
    trade_date: date,
    *,
    min_avg_notional: float = 12_000_000.0,
) -> dict[str, Any] | None:
    to_dt = datetime.combine(trade_date, datetime.min.time()).replace(hour=23, minute=59)
    cached_m15 = read_nifty500_symbol_ohlcv(client.settings.data_dir, sym, "15Min")

    try:
        m15 = _fetch_m15_incremental(client, token, to_dt, cached_m15)
    except Exception as e:
        return {"_error": str(e), "stock": sym}

    if m15.empty:
        return None

    # Persist the freshest cache as-is (including partial current session candles).
    # For scoring, we can still ignore a very thin last session to reduce noisy daily bars.
    m15_for_analysis = _drop_partial_last_session_if_thin(m15)
    d = _aggregate_daily_from_m15(m15_for_analysis)
    h1 = _aggregate_hourly_from_m15(m15_for_analysis)
    try:
        write_nifty500_symbol_ohlcv(client.settings.data_dir, trade_date, sym, m15, h1, d)
    except Exception as e:
        logger.warning("NIFTY500 CSV write %s: %s", sym, e)

    if len(m15) < 800:
        return None

    if len(d) < 70 or len(h1) < 40:
        return None

    last_d_ts = pd.to_datetime(d.iloc[-1]["date"])
    notional = float((d["close"].iloc[-21:-1] * d["volume"].iloc[-21:-1]).mean())
    if notional < min_avg_notional:
        return None

    if _two_day_range_pct(d) > 12.0:
        return None

    vol10 = float(d["volume"].iloc[-11:-1].mean()) if len(d) > 11 else float(d["volume"].iloc[:-1].mean())
    vol_last = float(d["volume"].iloc[-1])
    vol_ratio = vol_last / vol10 if vol10 > 0 else 1.0
    atr = _atr(d, 14)
    atr_ratio = float(atr.iloc[-5:].mean() / atr.iloc[-25:-5].mean()) if len(atr) > 25 else 1.0

    dist_flag = _intraday_1h_flags(h1)
    m15_tight, vwap_ok = _m15_tight_and_vwap(m15_for_analysis, last_d_ts)

    close = float(d.iloc[-1]["close"])
    res, rt = _resistance_zone(d)
    sup, st = _support_zone(d)
    dist_res = (res - close) / res * 100.0 if res > 0 else 999.0
    dist_sup = (close - sup) / sup * 100.0 if sup > 0 else 999.0
    cons = d.iloc[-18:-1]
    c_hi, c_lo = float(cons["high"].max()), float(cons["low"].min())
    cons_width = (c_hi - c_lo) / ((c_hi + c_lo) / 2) if c_hi > 0 else 1.0

    bull_setup, bull_reasons = _bullish_setups(d)
    bear_setup, bear_reasons = _bearish_setups(d)
    if _htf_exhaustion_bull(d) and bull_setup == "Breakout":
        bull_setup = ""
        bull_reasons = []
    if not bull_setup and not bear_setup:
        return None

    fut_df = pd.DataFrame()
    oi_bull_txt = "N/A (cash / no FnO)"
    oi_bear_txt = "N/A (cash / no FnO)"
    fut_row = _nearest_fut_row(client, sym, trade_date)
    if fut_row:
        try:
            ftok = int(fut_row["instrument_token"])
            _throttle()
            f_rows = client.historical_data(ftok, to_dt - timedelta(days=14), to_dt, interval="day", oi=True)
            fut_df = _rows_to_df(f_rows)
            oi_bull_txt = _fut_oi_reading(fut_df, True)
            oi_bear_txt = _fut_oi_reading(fut_df, False)
        except Exception:
            pass

    vol_prof = _volume_profile(d, vol_ratio)
    dist_daily = _distribution_daily(d.iloc[-1], vol_ratio)

    bull_score = (
        _score_bullish(
            dist_res=dist_res,
            res_touches=rt,
            cons_width=cons_width,
            atr_ratio=atr_ratio,
            vol_ratio=vol_ratio,
            vol_prof=vol_prof,
            dist_flag=dist_flag,
            m15_tight=m15_tight,
            vwap_ok=vwap_ok,
            oi_txt=oi_bull_txt,
            setup=bull_setup,
            distribution=dist_daily,
        )
        if bull_setup
        else 0.0
    )
    bear_score = (
        _score_bearish(
            dist_sup=dist_sup,
            sup_touches=st,
            cons_width=cons_width,
            atr_ratio=atr_ratio,
            vol_ratio=vol_ratio,
            vol_prof=vol_prof,
            dist_flag=dist_flag,
            m15_tight=m15_tight,
            vwap_ok=vwap_ok,
            oi_txt=oi_bear_txt,
            setup=bear_setup,
            distribution=dist_daily,
        )
        if bear_setup
        else 0.0
    )

    if bull_score < 38 and bear_score < 38:
        return None
    bullish_pick = bull_score >= bear_score
    if bullish_pick and not bull_setup:
        if bear_setup:
            bullish_pick = False
        else:
            return None
    if not bullish_pick and not bear_setup:
        return None

    bias = "Bullish" if bullish_pick else "Bearish"
    setup = bull_setup if bullish_pick else bear_setup
    reasons = list(bull_reasons if bullish_pick else bear_reasons)
    score = bull_score if bullish_pick else bear_score

    if atr_ratio < 0.94:
        reasons.append(f"ATR contracting vs 20d baseline (ratio {atr_ratio:.2f}).")
    if _nr7(d):
        reasons.append("Daily NR7-style narrow range — expansion watch.")
    if m15_tight:
        reasons.append("15m: recent candles compressed.")
    if vwap_ok is True and bullish_pick:
        reasons.append("Last session: spot held above session VWAP (15m).")
    if vwap_ok is False and not bullish_pick:
        reasons.append("Last session: spot below session VWAP (15m).")

    key_level = res if bullish_pick else sup
    level_type = "Resistance" if bullish_pick else "Support"
    dist_br = dist_res if bullish_pick else dist_sup
    oi_note = oi_bull_txt if bullish_pick else oi_bear_txt

    entry = close * (1.002 if bullish_pick else 0.998)
    if bullish_pick and res > 0:
        entry = max(entry, res * 1.001)
    if not bullish_pick and sup > 0:
        entry = min(entry, sup * 0.999)

    if bullish_pick:
        swing_lo = float(d["low"].iloc[-8:].min())
        stop = min(swing_lo * 0.997, close * (1 - max(0.012, min(0.04, dist_br / 100 + 0.01))))
        risk = entry - stop
        target = entry + max(risk * 2.0, entry * 0.015)
        rr = (target - entry) / risk if risk > 0 else 0.0
    else:
        swing_hi = float(d["high"].iloc[-8:].max())
        stop = max(swing_hi * 1.003, close * (1 + max(0.012, min(0.04, dist_br / 100 + 0.01))))
        risk = stop - entry
        target = entry - max(risk * 2.0, entry * 0.015)
        rr = (entry - target) / risk if risk > 0 else 0.0

    return {
        "stock": sym,
        "setup_type": setup,
        "current_price": round(close, 2),
        "key_level": round(float(key_level), 2) if key_level else None,
        "key_level_type": level_type,
        "distance_to_breakout_pct": round(float(dist_br), 3),
        "volume_profile": vol_prof,
        "volume_vs_10d": round(vol_ratio, 2),
        "oi_interpretation": oi_note,
        "next_day_bias": bias,
        "trade_plan": {
            "entry": round(entry, 2),
            "stop_loss": round(stop, 2),
            "target": round(target, 2),
            "rr": round(float(rr), 2),
        },
        "confidence_score": round(float(score), 1),
        "reason": " ".join(reasons[:3]),
        "extras": {
            "data_source": "15m_aggregated_1d_1h",
            "m15_bars_used": int(len(m15)),
            "atr_ratio": round(atr_ratio, 3),
            "consolidation_width_pct": round(cons_width * 100, 2),
            "nr7": _nr7(d),
            "h1_flags": dist_flag,
            "m15_above_vwap": vwap_ok,
        },
    }


def load_stored_tomorrow_watchlist(data_dir: Path, trade_date: date) -> dict[str, Any] | None:
    path = tomorrow_watchlist_json_path(data_dir, trade_date)
    if not path.exists():
        return None
    try:
        with path.open(encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.warning("Could not load %s: %s", path, e)
        return None


def load_stored_tomorrow_watchlist_fresh_or_previous(
    data_dir: Path, trade_date: date
) -> tuple[dict[str, Any] | None, bool]:
    """Load JSON for ``trade_date``; if missing, load newest saved file strictly before that date.

    Returns ``(payload, is_stale)`` where ``is_stale`` is True when the second branch was used
    (dashboard can show prior scan until the user refreshes for the requested day).
    """
    exact = load_stored_tomorrow_watchlist(data_dir, trade_date)
    if exact is not None:
        return exact, False

    prefix = "nifty500_tomorrow_watchlist_"
    suffix = ".json"
    candidates: list[tuple[date, Path]] = []
    for p in data_dir.glob(f"{prefix}*{suffix}"):
        stem = p.stem
        if not stem.startswith(prefix):
            continue
        ds = stem[len(prefix) :]
        try:
            d = datetime.strptime(ds, "%Y-%m-%d").date()
        except ValueError:
            continue
        if d < trade_date:
            candidates.append((d, p))

    if not candidates:
        return None, False

    _, best_path = max(candidates, key=lambda t: t[0])
    try:
        with best_path.open(encoding="utf-8") as f:
            return json.load(f), True
    except Exception as e:
        logger.warning("Could not load fallback watchlist %s: %s", best_path, e)
        return None, False


def run_tomorrow_watchlist_scan(
    *,
    settings: Settings | None = None,
    symbols_file: Path | None = None,
    trade_date: date | None = None,
    top_n: int = 20,
    min_score: float = 38.0,
    max_workers: int = 4,
    stock_limit: int | None = None,
) -> dict[str, Any]:
    settings = settings or Settings.from_env(underlying="NIFTY")
    client = ZerodhaClient(settings)
    td = trade_date or date.today()
    syms = load_nifty500_symbols(symbols_file, settings.data_dir)
    token_map = client.nse_eq_token_map()
    to_scan = [s for s in syms if s in token_map]
    if stock_limit:
        to_scan = to_scan[: int(stock_limit)]

    picks: list[dict[str, Any]] = []
    failed: list[dict[str, str]] = []
    run_at = datetime.now().isoformat(timespec="seconds")

    def job(sym: str) -> dict[str, Any] | None:
        tok = token_map[sym]
        try:
            return analyze_stock_pack(client, sym, tok, td)
        except Exception as e:
            return {"_error": str(e), "stock": sym}

    with ThreadPoolExecutor(max_workers=max(1, max_workers)) as ex:
        futs = {ex.submit(job, s): s for s in to_scan}
        for fut in as_completed(futs):
            row = fut.result()
            if not row:
                continue
            if row.get("_error"):
                failed.append({"stock": row.get("stock", ""), "error": str(row["_error"])[:200]})
                continue
            if float(row.get("confidence_score", 0)) >= min_score:
                picks.append(row)

    def sort_key(r: dict[str, Any]) -> tuple:
        volx = float(r.get("volume_vs_10d") or 0)
        ex = r.get("extras") or {}
        h1 = ex.get("h1_flags") or {}
        clean = 1.0 if h1.get("compress_1h") and h1.get("higher_lows_1h") else 0.0
        return (-float(r.get("confidence_score", 0)), -volx, -clean, str(r.get("stock", "")))

    picks.sort(key=sort_key)
    picks = picks[: max(1, top_n)]

    payload = {
        "trade_date": td.isoformat(),
        "run_at": run_at,
        "universe": "NIFTY500",
        "scanned": len(to_scan),
        "failed_count": len(failed),
        "failed_sample": failed[:30],
        "picks": picks,
    }
    if not picks and failed:
        sample = failed[0]
        payload["message"] = (
            f"Scan completed with errors for {len(failed)}/{len(to_scan)} symbols. "
            f"Example: {sample.get('stock', '')}: {sample.get('error', '')}"
        )
    out = tomorrow_watchlist_json_path(settings.data_dir, td)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=str)
    logger.info("Tomorrow watchlist: %d picks (min score %s) -> %s", len(picks), min_score, out)
    return payload
