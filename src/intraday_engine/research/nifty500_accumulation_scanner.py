"""
NIFTY 500 — 1-hour accumulation + bullish breakout readiness scanner.

Fetches the latest index constituent list (NSE archive CSV), pulls 60-minute
OHLCV from Zerodha, and ranks names by a composite breakout probability score.

Run: python -m intraday_engine.main --nifty500-accumulation
"""

from __future__ import annotations

import csv
import io
import json
import logging
import ssl
import threading
import time
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd

from intraday_engine.core.config import Settings
from intraday_engine.fetch.zerodha_client import ZerodhaClient
from intraday_engine.storage.layout import (
    accumulation_ohlcv_master_path,
    accumulation_partition_dir,
)

logger = logging.getLogger(__name__)

NIFTY500_CSV_URLS = (
    "https://nsearchives.nseindia.com/content/indices/ind_nifty500list.csv",
    "https://www.niftyindices.com/IndexConstituent/ind_nifty500list.csv",
)

MIN_RANGE_BARS = 20
MAX_RANGE_BARS = 50
MAX_RANGE_WIDTH_PCT = 10.0
TOUCH_TOLERANCE = 0.003  # 0.3% from range boundary counts as touch
MIN_TOUCHES = 3
MIN_AVG_RUPEE_TURNOVER_20H = 8_000_000.0  # ~80L avg hourly notional; tune as needed
SHARP_MOVE_LOOKBACK_H = 14  # ~2 sessions on 1h
SHARP_MOVE_PCT = 10.0
MAX_SLOPE_PCT_PER_BAR = 0.25  # trendline not too steep vs price
MIN_SWING_LOWS = 3
ATR_PERIOD = 14
RSI_PERIOD = 14
MAX_BARS_PER_SYMBOL = 2000
FETCH_STALE_AFTER_DAYS = 90
INCREMENTAL_OVERLAP_DAYS = 7
SKIP_NETWORK_IF_FRESH_MINUTES = 20

ANALYSIS_CSV_COLUMNS: tuple[str, ...] = (
    "stock",
    "passed",
    "fetch_error",
    "bars_cached",
    "last_bar_at",
    "current_price",
    "range_high",
    "range_low",
    "distance_to_breakout_pct",
    "trendline_strength",
    "volume_pattern",
    "breakout_probability_score",
    "vol_expansion_ratio",
    "suggested_entry",
    "suggested_stop",
    "rsi",
    "atr_ratio",
    "range_width_pct",
    "window_bars",
    "reason",
    "run_at",
)


@dataclass
class ScanResult:
    stock: str
    current_price: float
    range_high: float
    range_low: float
    distance_to_breakout_pct: float
    trendline_strength: str  # Weak / Moderate / Strong
    volume_pattern: str  # Dry / Increasing / Spike
    breakout_probability_score: float
    suggested_entry: float
    suggested_stop: float
    reason: str
    vol_expansion_ratio: float = 0.0
    extras: dict[str, Any] = field(default_factory=dict)


def download_nifty500_symbols(timeout: int = 45) -> list[str]:
    """Download NIFTY 500 symbols from NSE / Nifty Indices CSV."""
    ctx = ssl.create_default_context()
    headers = {"User-Agent": "Mozilla/5.0 (compatible; AlgoTrading/1.0)"}
    last_err: str | None = None
    for url in NIFTY500_CSV_URLS:
        try:
            req = Request(url, headers=headers)
            with urlopen(req, context=ctx, timeout=timeout) as resp:
                raw = resp.read().decode("utf-8-sig", errors="replace")
            reader = csv.DictReader(io.StringIO(raw))
            if not reader.fieldnames:
                continue
            # normalize header keys
            fh = {k.strip().lower(): k for k in reader.fieldnames if k}
            sym_key = fh.get("symbol")
            if not sym_key:
                continue
            syms: list[str] = []
            for row in reader:
                s = (row.get(sym_key) or "").strip().upper()
                if s and s not in syms:
                    syms.append(s)
            if len(syms) >= 400:
                return syms
        except Exception as e:
            last_err = str(e)
            logger.debug("NIFTY500 fetch %s: %s", url, e)
    if last_err:
        logger.warning("Could not download NIFTY500 list: %s", last_err)
    return []


def load_symbols_from_file(path: Path) -> list[str]:
    lines = path.read_text(encoding="utf-8").splitlines()
    out: list[str] = []
    for line in lines:
        s = line.split("#")[0].strip().upper()
        if s:
            out.append(s)
    return out


def load_nifty500_symbols(symbols_file: Path | None, data_dir: Path) -> list[str]:
    if symbols_file and symbols_file.exists():
        return load_symbols_from_file(symbols_file)
    ref = data_dir / "reference" / "nifty500_symbols.txt"
    if ref.exists():
        return load_symbols_from_file(ref)
    syms = download_nifty500_symbols()
    if syms:
        return syms
    raise RuntimeError(
        "Could not load NIFTY 500 symbols. Place one symbol per line in "
        f"{ref} or pass --nifty500-symbols-file pointing to a CSV/txt list."
    )


def _norm_ts(val: Any) -> str:
    if isinstance(val, datetime):
        return val.strftime("%Y-%m-%dT%H:%M:%S")
    return str(pd.Timestamp(val).strftime("%Y-%m-%dT%H:%M:%S"))


def _bar_from_kite(r: dict[str, Any]) -> dict[str, Any]:
    return {
        "date": _norm_ts(r.get("date")),
        "open": float(r.get("open", 0) or 0),
        "high": float(r.get("high", 0) or 0),
        "low": float(r.get("low", 0) or 0),
        "close": float(r.get("close", 0) or 0),
        "volume": float(r.get("volume", 0) or 0),
    }


def _bar_normalize_stored(r: dict[str, Any]) -> dict[str, Any]:
    return {
        "date": _norm_ts(r.get("date")),
        "open": float(r.get("open", 0) or 0),
        "high": float(r.get("high", 0) or 0),
        "low": float(r.get("low", 0) or 0),
        "close": float(r.get("close", 0) or 0),
        "volume": float(r.get("volume", 0) or 0),
    }


def merge_ohlcv(
    existing: list[dict[str, Any]] | None,
    kite_rows: list[dict[str, Any]] | None,
) -> list[dict[str, Any]]:
    m: dict[str, dict[str, Any]] = {}
    for r in existing or []:
        try:
            b = _bar_normalize_stored(r)
            m[b["date"]] = b
        except Exception:
            continue
    if kite_rows:
        for r in kite_rows:
            try:
                b = _bar_from_kite(r)
                m[b["date"]] = b
            except Exception:
                continue
    keys = sorted(m.keys())
    tail = keys[-MAX_BARS_PER_SYMBOL:] if len(keys) > MAX_BARS_PER_SYMBOL else keys
    return [m[k] for k in tail]


def load_ohlcv_master(path: Path) -> dict[str, list[dict[str, Any]]]:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            return {}
        return {str(k).strip().upper(): (v if isinstance(v, list) else []) for k, v in data.items()}
    except Exception as e:
        logger.warning("Could not load OHLCV master %s: %s", path, e)
        return {}


def save_ohlcv_master(path: Path, data: dict[str, list[dict[str, Any]]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
    tmp.replace(path)


def _records_to_df(records: list[dict[str, Any]]) -> pd.DataFrame:
    if not records:
        return pd.DataFrame()
    df = pd.DataFrame(records)
    df["date"] = pd.to_datetime(df["date"])
    for c in ("open", "high", "low", "close", "volume"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.sort_values("date").reset_index(drop=True)


def _empty_analysis_row(stock: str, run_at: str) -> dict[str, Any]:
    return {c: "" for c in ANALYSIS_CSV_COLUMNS} | {"stock": stock, "passed": "N", "run_at": run_at}


def build_analysis_row(
    stock: str,
    merged_bars: list[dict[str, Any]],
    hit: ScanResult | None,
    fetch_error: str | None,
    run_at: str,
) -> dict[str, Any]:
    row = _empty_analysis_row(stock, run_at)
    row["bars_cached"] = len(merged_bars)
    if merged_bars:
        try:
            row["last_bar_at"] = _norm_ts(merged_bars[-1].get("date"))
        except Exception:
            row["last_bar_at"] = ""
    if fetch_error:
        row["fetch_error"] = fetch_error[:500]
        row["reason"] = "analyze_failed" if str(fetch_error).startswith("analyze:") else "fetch_failed"
        if merged_bars:
            try:
                dfx = _records_to_df(merged_bars)
                if not dfx.empty:
                    row["current_price"] = round(float(dfx.iloc[-1]["close"]), 4)
            except Exception:
                pass
        return row
    if not merged_bars:
        row["reason"] = "no_bars"
        return row
    df = _records_to_df(merged_bars)
    if df.empty:
        row["reason"] = "no_bars_after_parse"
        return row
    last_px = float(df.iloc[-1]["close"])
    row["current_price"] = round(last_px, 4)
    if hit is None:
        row["reason"] = "did_not_pass_accumulation_filters"
        return row
    row["passed"] = "Y"
    row["fetch_error"] = ""
    row["range_high"] = hit.range_high
    row["range_low"] = hit.range_low
    row["distance_to_breakout_pct"] = hit.distance_to_breakout_pct
    row["trendline_strength"] = hit.trendline_strength
    row["volume_pattern"] = hit.volume_pattern
    row["breakout_probability_score"] = hit.breakout_probability_score
    row["vol_expansion_ratio"] = hit.vol_expansion_ratio
    row["suggested_entry"] = hit.suggested_entry
    row["suggested_stop"] = hit.suggested_stop
    row["reason"] = hit.reason
    ex = hit.extras or {}
    row["rsi"] = ex.get("rsi", "")
    row["atr_ratio"] = ex.get("atr_ratio", "")
    row["range_width_pct"] = ex.get("range_width_pct", "")
    row["window_bars"] = ex.get("bars", "")
    return row


def _write_analysis_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    for c in ANALYSIS_CSV_COLUMNS:
        if c not in df.columns:
            df[c] = ""
    df = df[list(ANALYSIS_CSV_COLUMNS)]
    df.to_csv(path, index=False)


def _incremental_fetch_from_dt(
    existing: list[dict[str, Any]],
    to_dt: datetime,
    full_history_days: int,
) -> datetime:
    if not existing:
        return to_dt - timedelta(days=full_history_days)
    try:
        last = max(pd.to_datetime(r["date"]) for r in existing if r.get("date"))
        last_naive = last.to_pydatetime() if hasattr(last, "to_pydatetime") else last
        if getattr(last_naive, "tzinfo", None):
            last_naive = last_naive.replace(tzinfo=None)
    except Exception:
        return to_dt - timedelta(days=full_history_days)
    if to_dt - last_naive > timedelta(days=FETCH_STALE_AFTER_DAYS):
        return to_dt - timedelta(days=full_history_days)
    return last_naive - timedelta(days=INCREMENTAL_OVERLAP_DAYS)


def _should_skip_network_fetch(existing: list[dict[str, Any]], to_dt: datetime) -> bool:
    if not existing:
        return False
    try:
        last = max(pd.to_datetime(r["date"]) for r in existing if r.get("date"))
        last_naive = last.to_pydatetime() if hasattr(last, "to_pydatetime") else last
        if getattr(last_naive, "tzinfo", None):
            last_naive = last_naive.replace(tzinfo=None)
        return (to_dt - last_naive) < timedelta(minutes=SKIP_NETWORK_IF_FRESH_MINUTES)
    except Exception:
        return False


def _rsi(close: pd.Series, period: int = RSI_PERIOD) -> pd.Series:
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def _atr(df: pd.DataFrame, period: int = ATR_PERIOD) -> pd.Series:
    h, l, c = df["high"], df["low"], df["close"]
    prev_c = c.shift(1)
    tr = pd.concat(
        [
            (h - l),
            (h - prev_c).abs(),
            (l - prev_c).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()


def _ema(s: pd.Series, span: int) -> pd.Series:
    return s.ewm(span=span, adjust=False, min_periods=span).mean()


def _swing_lows(sub: pd.DataFrame) -> list[tuple[int, float]]:
    out: list[tuple[int, float]] = []
    if len(sub) < 3:
        return out
    lows = sub["low"].values
    for i in range(1, len(sub) - 1):
        if lows[i] <= lows[i - 1] and lows[i] <= lows[i + 1]:
            out.append((i, float(lows[i])))
    return out


def _exclusion_sharp_move(df: pd.DataFrame) -> bool:
    if len(df) < SHARP_MOVE_LOOKBACK_H:
        return True
    tail = df.iloc[-SHARP_MOVE_LOOKBACK_H :]
    lo, hi = float(tail["low"].min()), float(tail["high"].max())
    if lo <= 0:
        return True
    return (hi - lo) / lo * 100.0 > SHARP_MOVE_PCT


def _exclusion_erratic_wicks(df: pd.DataFrame, n: int = 24) -> bool:
    tail = df.iloc[-n:]
    body = (tail["close"] - tail["open"]).abs().replace(0, np.nan)
    upper = tail["high"] - tail[["open", "close"]].max(axis=1)
    ratio = (upper / body).median()
    if pd.isna(ratio):
        return False
    return float(ratio) > 2.8


def _liquidity_ok(df: pd.DataFrame) -> bool:
    if len(df) < 22:
        return False
    tail = df.iloc[-20:]
    notional = (tail["close"].astype(float) * tail["volume"].astype(float)).mean()
    return notional >= MIN_AVG_RUPEE_TURNOVER_20H


def _volume_pattern(sub: pd.DataFrame, vol_expansion: float) -> str:
    n = len(sub)
    if n < 10:
        return "Dry"
    v1 = float(sub.iloc[: n // 2]["volume"].mean())
    v2 = float(sub.iloc[n // 2 :]["volume"].mean())
    if vol_expansion >= 1.35:
        return "Spike"
    if vol_expansion >= 1.08:
        return "Increasing"
    if v2 <= v1 * 1.02:
        return "Dry"
    return "Increasing"


def _pick_consolidation_window(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]] | None:
    """
    Choose the longest trailing window in [MIN_RANGE_BARS, MAX_RANGE_BARS] that satisfies
    tight range + enough touches + dry-ish volume + higher swing lows + gentle rising trendline.
    """
    n = len(df)
    if n < MAX_RANGE_BARS + 30:
        return None

    best: tuple[int, dict[str, Any]] | None = None

    for window in range(MAX_RANGE_BARS, MIN_RANGE_BARS - 1, -1):
        sub = df.iloc[-window:].reset_index(drop=True)
        rh = float(sub["high"].max())
        rl = float(sub["low"].min())
        if rh <= 0 or rl <= 0:
            continue
        mid = (rh + rl) / 2.0
        width_pct = (rh - rl) / mid * 100.0
        if width_pct > MAX_RANGE_WIDTH_PCT:
            continue

        res_touches = int((sub["high"] >= rh * (1 - TOUCH_TOLERANCE)).sum())
        sup_touches = int((sub["low"] <= rl * (1 + TOUCH_TOLERANCE)).sum())
        if res_touches < MIN_TOUCHES or sup_touches < MIN_TOUCHES:
            continue

        n_half = max(2, window // 2)
        v_first = float(sub.iloc[:n_half]["volume"].mean())
        v_second = float(sub.iloc[n_half:]["volume"].mean())
        if v_first <= 0:
            continue
        # consolidation: volume not ramping hard inside range
        if v_second > v_first * 1.25:
            continue

        swings = _swing_lows(sub)
        if len(swings) < MIN_SWING_LOWS:
            continue
        last_swings = swings[-MIN_SWING_LOWS:]
        lows_seq = [x[1] for x in last_swings]
        if not all(lows_seq[i] > lows_seq[i - 1] * 0.9995 for i in range(1, len(lows_seq))):
            continue

        use_pts = swings[-min(5, len(swings)) :]
        xs = np.array([float(p[0]) for p in use_pts], dtype=float)
        ys = np.array([float(p[1]) for p in use_pts], dtype=float)
        if len(xs) < 3 or np.std(xs) < 1e-6:
            continue
        slope, intercept = np.polyfit(xs, ys, 1)
        if slope <= 0:
            continue
        slope_pct_per_bar = (slope / mid) * 100.0
        if slope_pct_per_bar > MAX_SLOPE_PCT_PER_BAR:
            continue

        y_hat = slope * xs + intercept
        ss_res = float(np.sum((ys - y_hat) ** 2))
        ss_tot = float(np.sum((ys - ys.mean()) ** 2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 0.0

        meta = {
            "window": window,
            "rh": rh,
            "rl": rl,
            "width_pct": width_pct,
            "res_touches": res_touches,
            "sup_touches": sup_touches,
            "r2": r2,
            "slope": slope,
            "intercept": intercept,
            "swing_lows": last_swings,
        }
        # prefer longer window; tie-break by better r2
        if best is None or window > best[0]:
            best = (window, meta)

    if best is None:
        return None
    w, meta = best
    sub = df.iloc[-w:].reset_index(drop=True)
    return sub, meta


def analyze_hourly_frame(df: pd.DataFrame, symbol: str) -> ScanResult | None:
    """Return ScanResult if stock passes filters and consolidation logic."""
    if df.empty or len(df) < 80:
        return None

    work = df.copy()
    for c in ("open", "high", "low", "close", "volume"):
        if c not in work.columns:
            return None
        work[c] = pd.to_numeric(work[c], errors="coerce")
    work = work.dropna(subset=["open", "high", "low", "close", "volume"])
    if len(work) < 80:
        return None

    if _exclusion_sharp_move(work) or _exclusion_erratic_wicks(work) or not _liquidity_ok(work):
        return None

    work["ema20"] = _ema(work["close"], 20)
    work["ema50"] = _ema(work["close"], 50)
    work["rsi"] = _rsi(work["close"], RSI_PERIOD)
    work["atr"] = _atr(work, ATR_PERIOD)
    tp = (work["high"] + work["low"] + work["close"]) / 3.0
    work["vwap_roll"] = (tp * work["volume"]).rolling(30, min_periods=10).sum() / work["volume"].rolling(
        30, min_periods=10
    ).sum()

    picked = _pick_consolidation_window(work)
    if picked is None:
        return None
    sub, meta = picked
    rh, rl = meta["rh"], meta["rl"]
    last = work.iloc[-1]
    close = float(last["close"])
    if close <= 0:
        return None

    dist_pct = (rh - close) / rh * 100.0
    if dist_pct < 0 or dist_pct > 2.0:
        return None

    atr_s = work["atr"].iloc[-5:].mean()
    atr_l = work["atr"].iloc[-25:-5].mean()
    if pd.isna(atr_s) or pd.isna(atr_l) or atr_l <= 0:
        return None
    atr_ratio = float(atr_s / atr_l)
    if atr_ratio > 0.95:
        return None

    vol_tail = float(sub["volume"].iloc[-3:].mean())
    vol_base = float(sub["volume"].iloc[-25:-3].mean()) if len(sub) > 25 else float(sub["volume"].iloc[:-3].mean())
    if vol_base <= 0:
        return None
    vol_expansion = vol_tail / vol_base

    bodies = (sub["close"] - sub["open"]).abs()
    comp_short = float(bodies.iloc[-5:].mean()) if len(bodies) >= 5 else float(bodies.mean())
    comp_long = float(bodies.iloc[-20:-5].mean()) if len(bodies) >= 20 else float(bodies.iloc[:-5].mean())
    if comp_long <= 0 or comp_short >= comp_long * 0.90:
        return None

    r2 = float(meta["r2"])
    if r2 >= 0.82:
        tl_strength = "Strong"
    elif r2 >= 0.65:
        tl_strength = "Moderate"
    else:
        tl_strength = "Weak"

    # Bonus: EMAs, RSI band, VWAP
    ema20_ok = close > float(last["ema20"]) and not pd.isna(last["ema20"])
    ema50_ok = close > float(last["ema50"]) and not pd.isna(last["ema50"])
    rsi_v = float(last["rsi"]) if not pd.isna(last["rsi"]) else 50.0
    rsi_ok = 55.0 <= rsi_v <= 65.0
    vwap_v = float(last["vwap_roll"]) if not pd.isna(last["vwap_roll"]) else close
    vwap_ok = close >= vwap_v * 0.998

    start_idx = len(work) - meta["window"]
    prior = work.iloc[max(0, start_idx - 55) : max(0, start_idx)]
    room_ok = True
    if not prior.empty:
        overhead = float(prior["high"].max())
        if overhead > rh * 1.001 and (overhead - close) / close * 100.0 < 2.5:
            room_ok = False

    score = 40.0
    score += max(0.0, 15.0 * (1.0 - dist_pct / 2.0))
    score += max(0.0, 12.0 * (1.0 - meta["width_pct"] / MAX_RANGE_WIDTH_PCT))
    score += min(15.0, r2 * 15.0)
    if vol_expansion >= 1.05:
        score += min(10.0, (vol_expansion - 1.0) * 25.0)
    if atr_ratio < 0.88:
        score += 5.0
    if ema20_ok:
        score += 4.0
    if ema50_ok:
        score += 4.0
    if rsi_ok:
        score += 5.0
    if vwap_ok:
        score += 3.0
    if room_ok:
        score += 4.0
    score = max(0.0, min(100.0, score))

    vol_pat = _volume_pattern(sub, vol_expansion)

    xi = float(len(sub) - 1)
    tl_at_end = float(meta["slope"] * xi + meta["intercept"])
    suggested_entry = rh * 1.001
    suggested_stop = min(rl * 0.998, tl_at_end * 0.997)

    reasons: list[str] = []
    reasons.append(f"{meta['window']}×1h range {meta['width_pct']:.1f}% wide; {dist_pct:.2f}% below resistance.")
    reasons.append(f"Touches H/L={meta['res_touches']}/{meta['sup_touches']}; trendline R²={r2:.2f} ({tl_strength}).")
    reasons.append(f"ATR contracting (short/long={atr_ratio:.2f}); vol×{vol_expansion:.2f} vs base ({vol_pat}).")
    if ema20_ok and ema50_ok:
        reasons.append("Price above 20/50 EMA (1h).")
    if rsi_ok:
        reasons.append(f"RSI {rsi_v:.0f} in bullish band.")
    if not room_ok:
        reasons.append("Note: nearby overhead supply within ~2.5%.")

    return ScanResult(
        stock=symbol,
        current_price=round(close, 2),
        range_high=round(rh, 2),
        range_low=round(rl, 2),
        distance_to_breakout_pct=round(dist_pct, 3),
        trendline_strength=tl_strength,
        volume_pattern=vol_pat,
        breakout_probability_score=round(score, 1),
        suggested_entry=round(suggested_entry, 2),
        suggested_stop=round(suggested_stop, 2),
        reason=" ".join(reasons),
        vol_expansion_ratio=round(vol_expansion, 4),
        extras={
            "rsi": round(rsi_v, 2),
            "atr_ratio": round(atr_ratio, 4),
            "range_width_pct": round(meta["width_pct"], 3),
            "bars": meta["window"],
        },
    )


def _candles_to_df(rows: list[dict[str, Any]]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").reset_index(drop=True)
    return df


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


def run_nifty500_accumulation_scan(
    *,
    settings: Settings | None = None,
    symbols_file: Path | None = None,
    top_n: int = 15,
    max_workers: int = 4,
    history_days: int = 120,
    out_json: Path | None = None,
    trade_date: date | None = None,
) -> list[ScanResult]:
    settings = settings or Settings.from_env(underlying="NIFTY")
    client = ZerodhaClient(settings)
    symbols = load_nifty500_symbols(symbols_file, settings.data_dir)
    token_map = client.nse_eq_token_map()
    td = trade_date or date.today()
    run_at = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")

    master_path = accumulation_ohlcv_master_path(settings.data_dir)
    master = load_ohlcv_master(master_path)
    partition_dir = accumulation_partition_dir(settings.data_dir, td)

    to_scan = [s for s in symbols if s in token_map]
    missing = sorted(set(symbols) - set(to_scan))
    if missing:
        logger.info("NSE EQ tokens missing for %d symbols (sample: %s)", len(missing), missing[:8])

    to_dt = datetime.now()
    lock = threading.Lock()
    updates: dict[str, tuple[list[dict[str, Any]], ScanResult | None, str | None]] = {}

    def job(sym: str) -> tuple[str, list[dict[str, Any]], ScanResult | None, str | None]:
        existing = list(master.get(sym, []))
        fetch_err: str | None = None
        try:
            token = token_map[sym]
        except KeyError:
            return sym, existing, None, "no_nse_eq_token"

        try:
            if _should_skip_network_fetch(existing, to_dt):
                merged = merge_ohlcv(existing, None)
            else:
                from_dt = _incremental_fetch_from_dt(existing, to_dt, history_days)
                if from_dt >= to_dt - timedelta(minutes=5):
                    merged = merge_ohlcv(existing, None)
                else:
                    _throttle()
                    kite_rows = client.historical_data(
                        token, from_dt, to_dt, interval="60minute", oi=False
                    )
                    merged = merge_ohlcv(existing, kite_rows)
        except Exception as e:
            fetch_err = str(e)
            merged = merge_ohlcv(existing, None)

        hit: ScanResult | None = None
        if fetch_err is None:
            df = _records_to_df(merged)
            if not df.empty:
                try:
                    hit = analyze_hourly_frame(df, sym)
                except Exception as e2:
                    fetch_err = f"analyze:{e2}"
        return sym, merged, hit, fetch_err

    from concurrent.futures import ThreadPoolExecutor, as_completed

    with ThreadPoolExecutor(max_workers=max(1, max_workers)) as ex:
        futs = {ex.submit(job, s): s for s in to_scan}
        for fut in as_completed(futs):
            try:
                sym, merged, hit, err = fut.result()
            except Exception as e:
                sym = futs[fut]
                merged, hit, err = list(master.get(sym, [])), None, str(e)
            with lock:
                updates[sym] = (merged, hit, err)

    new_master = dict(master)
    for sym, (merged, _, _) in updates.items():
        new_master[sym] = merged
    save_ohlcv_master(master_path, new_master)

    analysis_rows: list[dict[str, Any]] = []
    for sym in missing:
        analysis_rows.append(
            build_analysis_row(sym, list(master.get(sym, [])), None, "no_nse_eq_token", run_at)
        )

    results_pass: list[ScanResult] = []
    for sym in to_scan:
        merged, hit, err = updates.get(sym, ([], None, "not_scanned"))
        analysis_rows.append(build_analysis_row(sym, merged, hit, err, run_at))
        if hit is not None and err is None:
            results_pass.append(hit)

    analysis_rows.sort(key=lambda r: str(r.get("stock", "")))

    results_pass.sort(key=lambda r: (-r.breakout_probability_score, -r.vol_expansion_ratio))
    top = results_pass[:top_n]

    csv_path = partition_dir / "nifty500_accumulation_analysis.csv"
    _write_analysis_csv(analysis_rows, csv_path)
    report_path = partition_dir / "nifty500_accumulation_report.txt"
    report_path.write_text(format_scan_report(top), encoding="utf-8")

    logger.info(
        "Accumulation scan wrote master=%s (%d symbols), analysis=%s (%d rows), report=%s",
        master_path,
        len(new_master),
        csv_path,
        len(analysis_rows),
        report_path,
    )

    if out_json:
        out_json.parent.mkdir(parents=True, exist_ok=True)
        payload = [
            {
                "stock_name": r.stock,
                "current_price": r.current_price,
                "range_high": r.range_high,
                "range_low": r.range_low,
                "distance_to_breakout_pct": r.distance_to_breakout_pct,
                "trendline_strength": r.trendline_strength,
                "volume_pattern": r.volume_pattern,
                "breakout_probability_score": r.breakout_probability_score,
                "suggested_entry_level": r.suggested_entry,
                "suggested_stop_loss": r.suggested_stop,
                "reason": r.reason,
                "vol_expansion_ratio": r.vol_expansion_ratio,
                **r.extras,
            }
            for r in top
        ]
        out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        logger.info("Wrote %d rows to %s", len(payload), out_json)

    return top


def format_scan_report(rows: list[ScanResult]) -> str:
    lines: list[str] = []
    if not rows:
        lines.append("No stocks passed accumulation + breakout-readiness filters.")
        return "\n".join(lines) + "\n"
    lines.append(
        f"\nTop {len(rows)} NIFTY 500 names (1h accumulation, bullish bias) — "
        "sorted by score, then volume expansion\n"
    )
    hdr = (
        f"{'Stock':<12} {'Px':>10} {'Range H':>10} {'Range L':>10} {'Dist%':>7} "
        f"{'TL':<9} {'VolPat':<10} {'Score':>6} {'Entry':>10} {'Stop':>10}"
    )
    lines.append(hdr)
    lines.append("-" * len(hdr))
    for r in rows:
        lines.append(
            f"{r.stock:<12} {r.current_price:>10.2f} {r.range_high:>10.2f} {r.range_low:>10.2f} "
            f"{r.distance_to_breakout_pct:>7.2f} {r.trendline_strength:<9} {r.volume_pattern:<10} "
            f"{r.breakout_probability_score:>6.1f} {r.suggested_entry:>10.2f} {r.suggested_stop:>10.2f}"
        )
    lines.append("")
    for i, r in enumerate(rows, 1):
        lines.append(f"{i}. {r.stock} — Score {r.breakout_probability_score} | {r.reason}")
        lines.append("")
    return "\n".join(lines) + "\n"


def print_scan_report(rows: list[ScanResult]) -> None:
    print(format_scan_report(rows), end="")
