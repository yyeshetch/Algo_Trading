"""
Mark Minervini-style scanners on NIFTY 500 daily bars.

Two modes:

  **setup** (default) — stocks *about to move*:
    VCP / tight base under a defined pivot, volume dry, higher lows,
    close still BELOW pivot (no breakout yet), RS improving, 200-DMA
    rising context, NOT extended above 50-DMA.

  **extended** — classic Stage-2 Trend Template (already trending):
    Full 8-point template + RS ≥ 70th percentile. Useful for confirmation,
    not early entries.

Uses cached daily bars: data/NIFTY500/SYMBOL_1D.csv

Run:  python -m intraday_engine.main --minervini-template
      python -m intraday_engine.main --minervini-template --minervini-mode extended
"""

from __future__ import annotations

import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd

from intraday_engine.core.config import Settings
from intraday_engine.research.nifty500_accumulation_scanner import load_nifty500_symbols
from intraday_engine.research.relative_strength_scanner import (
    RelativeStrengthRow,
    _compute_rs_row,
    _load_nifty_daily,
)
from intraday_engine.storage.layout import minervini_trend_template_path
from intraday_engine.storage.nifty500_csv import read_nifty500_symbol_ohlcv

logger = logging.getLogger(__name__)

ScanMode = Literal["setup", "extended"]

MIN_BARS = 260
MA50 = 50
MA150 = 150
MA200 = 200
MA200_RISE_LOOKBACK = 21
MIN_PCT_ABOVE_52W_LOW = 30.0
MAX_PCT_FROM_52W_HIGH = 25.0
DEFAULT_RS_MIN_PERCENTILE = 70.0
SETUP_RS_MIN_PERCENTILE = 50.0
WEEK52_BARS = 252

# Pre-breakout base (daily VCP)
BASE_MIN_BARS = 15
BASE_MAX_BARS = 50
BASE_MAX_WIDTH_PCT = 22.0
BASE_MIN_WIDTH_PCT = 4.0
PIVOT_MIN_DIST_PCT = 0.4
PIVOT_MAX_DIST_PCT = 8.0
MAX_PCT_ABOVE_MA50 = 18.0
MAX_RECENT_MOVE_20D_PCT = 28.0
MIN_VCP_SETUP_SCORE = 45.0
MIN_RS_SLOPE_20D_SETUP = 0.0


@dataclass
class MinerviniRow:
    stock: str
    scan_mode: str
    current_price: float
    ma50: float
    ma150: float
    ma200: float
    high_52w: float
    low_52w: float
    pct_above_52w_low: float
    pct_from_52w_high: float
    pct_above_ma50: float
    recent_move_20d_pct: float
    ma200_slope_21d_pct: float
    rs_rank_percentile: float
    rs_slope_20d_pct: float
    excess_20d: float
    criteria_passed: int
    criteria_total: int
    trend_template_pass: bool
    rs_pass: bool
    full_pass: bool
    setup_pass: bool
    breakout_pending: bool
    criteria: dict[str, bool]
    vcp_score: float
    composite_score: float
    setup_score: float
    pivot_price: float
    base_low: float
    base_width_pct: float
    base_bars: int
    dist_to_pivot_pct: float
    rs_line: list[float] = field(default_factory=list)
    reasons: list[str] = field(default_factory=list)


def _sma(series: pd.Series, period: int) -> pd.Series:
    return series.rolling(period, min_periods=period).mean()


def _pct_change(series: pd.Series, n: int) -> float:
    if len(series) < n + 1:
        return 0.0
    older = float(series.iloc[-(n + 1)])
    newer = float(series.iloc[-1])
    if older <= 0:
        return 0.0
    return (newer / older - 1.0) * 100.0


def _swing_low_indices(lows: pd.Series, order: int = 2) -> list[int]:
    idxs: list[int] = []
    arr = lows.astype(float).values
    n = len(arr)
    for i in range(order, n - order):
        if all(arr[i] <= arr[i - j] for j in range(1, order + 1)) and all(
            arr[i] <= arr[i + j] for j in range(1, order + 1)
        ):
            idxs.append(i)
    return idxs


def _find_vcp_base(work: pd.DataFrame) -> tuple[float, float, int, float] | None:
    """
    Trailing consolidation base: return (base_high, base_low, window_bars, width_pct).
    Pivot = base_high (resistance), not 52-week high.
    """
    n = len(work)
    if n < BASE_MIN_BARS + 5:
        return None

    best: tuple[int, float, float, float] | None = None

    for window in range(BASE_MAX_BARS, BASE_MIN_BARS - 1, -1):
        sub = work.iloc[-window:]
        rh = float(sub["high"].max())
        rl = float(sub["low"].min())
        if rh <= 0 or rl <= 0:
            continue
        mid = (rh + rl) / 2.0
        width_pct = (rh - rl) / mid * 100.0
        if width_pct > BASE_MAX_WIDTH_PCT or width_pct < BASE_MIN_WIDTH_PCT:
            continue

        swing_idxs = _swing_low_indices(sub["low"].reset_index(drop=True))
        if len(swing_idxs) >= 2:
            swing_vals = [float(sub["low"].iloc[i]) for i in swing_idxs[-3:]]
            if not all(swing_vals[i] >= swing_vals[i - 1] * 0.998 for i in range(1, len(swing_vals))):
                continue

        vol_first = float(sub.iloc[: window // 2]["volume"].mean())
        vol_second = float(sub.iloc[window // 2 :]["volume"].mean())
        if vol_first > 0 and vol_second > vol_first * 1.35:
            continue

        if best is None or window > best[0]:
            best = (window, rh, rl, width_pct)

    if best is None:
        return None
    window, rh, rl, width_pct = best
    return rh, rl, window, width_pct


def _vcp_score(
    df: pd.DataFrame,
    pivot: float,
    base_low: float,
    *,
    dist_pct: float | None = None,
) -> tuple[float, float]:
    """Return (vcp_score 0-100, distance to pivot % below resistance)."""
    if len(df) < BASE_MIN_BARS + 5 or pivot <= 0:
        return 0.0, 100.0
    close = float(df["close"].iloc[-1])
    if dist_pct is None:
        dist_pct = (pivot - close) / pivot * 100.0 if pivot > close else -((close - pivot) / pivot * 100.0)

    window = min(BASE_MAX_BARS, len(df) - 1)
    tail = df.tail(window)
    prior = df.iloc[-(window * 2) : -window] if len(df) >= window * 2 else df.iloc[:-window]
    if prior.empty or len(tail) < BASE_MIN_BARS:
        return 0.0, round(dist_pct, 2)

    rng_tail = (tail["high"] - tail["low"]) / tail["close"].replace(0, np.nan) * 100.0
    rng_prior = (prior["high"] - prior["low"]) / prior["close"].replace(0, np.nan) * 100.0
    w_tail = float(rng_tail.mean()) if len(rng_tail) else 0.0
    w_prior = float(rng_prior.mean()) if len(rng_prior) else w_tail
    width_contract = w_prior > 0 and w_tail < w_prior * 0.82

    vol_tail = float(tail["volume"].tail(5).mean())
    vol_base = float(prior["volume"].mean()) if len(prior) else vol_tail
    vol_dry = vol_base > 0 and vol_tail < vol_base * 0.82

    lows = tail["low"].astype(float)
    higher_lows = len(lows) >= 9 and float(lows.iloc[-1]) > float(lows.iloc[-5]) > float(lows.iloc[-9])

    tr = pd.concat(
        [
            tail["high"] - tail["low"],
            (tail["high"] - tail["close"].shift(1)).abs(),
            (tail["low"] - tail["close"].shift(1)).abs(),
        ],
        axis=1,
    ).max(axis=1)
    atr_s = float(tr.tail(5).mean())
    atr_l = float(tr.mean())
    atr_contract = atr_l > 0 and atr_s < atr_l * 0.88

    score = 0.0
    if width_contract:
        score += 28.0
    if vol_dry:
        score += 28.0
    if higher_lows:
        score += 24.0
    if atr_contract:
        score += 12.0
    if PIVOT_MIN_DIST_PCT <= dist_pct <= 5.0:
        score += 8.0
    elif dist_pct <= PIVOT_MAX_DIST_PCT:
        score += 4.0

    if base_low > 0 and close > base_low:
        score += 4.0

    return min(100.0, score), round(dist_pct, 2)


def _extended_criteria(
    c: float,
    m50: float,
    m150: float,
    m200: float,
    m200_prev: float,
    pct_above_low: float,
    pct_from_high: float,
) -> dict[str, bool]:
    return {
        "price_above_ma150": c > m150,
        "price_above_ma200": c > m200,
        "ma150_above_ma200": m150 > m200,
        "ma200_rising": m200 > m200_prev,
        "ma50_above_long_mas": m50 > m150 and m50 > m200,
        "price_above_ma50": c > m50,
        "pct_above_52w_low": pct_above_low >= MIN_PCT_ABOVE_52W_LOW,
        "near_52w_high": pct_from_high <= MAX_PCT_FROM_52W_HIGH,
    }


def _setup_criteria(
    c: float,
    m50: float,
    m200: float,
    m200_prev: float,
    pivot: float,
    dist_pivot: float,
    pct_above_ma50: float,
    recent_move_20d: float,
    vcp: float,
    rs_rank: float,
    rs_slope_20d: float,
    base_width: float,
) -> dict[str, bool]:
    return {
        "below_pivot": c < pivot * 0.999,
        "near_pivot": PIVOT_MIN_DIST_PCT <= dist_pivot <= PIVOT_MAX_DIST_PCT,
        "vcp_base": vcp >= MIN_VCP_SETUP_SCORE,
        "tight_base": BASE_MIN_WIDTH_PCT <= base_width <= BASE_MAX_WIDTH_PCT,
        "above_200dma": c > m200,
        "ma200_rising": m200 > m200_prev,
        "not_extended": pct_above_ma50 <= MAX_PCT_ABOVE_MA50,
        "recent_move_ok": recent_move_20d <= MAX_RECENT_MOVE_20D_PCT,
        "rs_improving": rs_rank >= SETUP_RS_MIN_PERCENTILE or rs_slope_20d >= MIN_RS_SLOPE_20D_SETUP,
        "above_50dma": c > m50,
    }


def _evaluate_symbol(
    df: pd.DataFrame,
    rs_row: RelativeStrengthRow | None,
    rs_rank_percentile: float,
    *,
    mode: ScanMode,
    rs_min_percentile: float,
) -> MinerviniRow | None:
    if df.empty or len(df) < MIN_BARS:
        return None

    work = df.dropna(subset=["high", "low", "close", "volume"]).reset_index(drop=True)
    if len(work) < MIN_BARS:
        return None

    close_s = work["close"].astype(float)
    high_s = work["high"].astype(float)
    low_s = work["low"].astype(float)

    ma50_s = _sma(close_s, MA50)
    ma150_s = _sma(close_s, MA150)
    ma200_s = _sma(close_s, MA200)

    c = float(close_s.iloc[-1])
    m50 = float(ma50_s.iloc[-1])
    m150 = float(ma150_s.iloc[-1])
    m200 = float(ma200_s.iloc[-1])

    if any(pd.isna(x) for x in (m50, m150, m200)):
        return None

    m200_prev = float(ma200_s.iloc[-(MA200_RISE_LOOKBACK + 1)])
    ma200_slope = ((m200 / m200_prev) - 1.0) * 100.0 if m200_prev > 0 else 0.0

    win = min(WEEK52_BARS, len(work))
    hi52 = float(high_s.tail(win).max())
    lo52 = float(low_s.tail(win).min())
    if lo52 <= 0 or hi52 <= 0:
        return None

    pct_above_low = (c - lo52) / lo52 * 100.0
    pct_from_high = (hi52 - c) / hi52 * 100.0
    pct_above_ma50 = (c - m50) / m50 * 100.0 if m50 > 0 else 0.0
    recent_move_20d = _pct_change(close_s, 20)

    base = _find_vcp_base(work)
    if base is None:
        if mode == "setup":
            return None
        pivot = hi52
        base_low = lo52
        base_bars = 0
        base_width = 0.0
    else:
        pivot, base_low, base_bars, base_width = base

    vcp, dist_pivot = _vcp_score(work, pivot, base_low)

    rs_slope_20d = rs_row.rs_slope_20d_pct if rs_row else 0.0
    excess_20d = rs_row.excess_20d if rs_row else 0.0

    ext_criteria = _extended_criteria(c, m50, m150, m200, m200_prev, pct_above_low, pct_from_high)
    setup_criteria = _setup_criteria(
        c, m50, m200, m200_prev, pivot, dist_pivot, pct_above_ma50,
        recent_move_20d, vcp, rs_rank_percentile, rs_slope_20d, base_width,
    )

    ext_passed = sum(1 for v in ext_criteria.values() if v)
    setup_passed = sum(1 for v in setup_criteria.values() if v)

    trend_pass = ext_passed == len(ext_criteria)
    rs_pass_ext = rs_rank_percentile >= rs_min_percentile
    full_pass = trend_pass and rs_pass_ext
    setup_pass = setup_passed == len(setup_criteria)
    breakout_pending = c < pivot * 0.999 and dist_pivot >= PIVOT_MIN_DIST_PCT

    if mode == "setup":
        if not setup_pass:
            return None
        criteria = setup_criteria
        passed = setup_passed
        total = len(setup_criteria)
        proximity = max(0.0, 100.0 - abs(dist_pivot - 2.5) * 12.0)
        rs_momentum = min(100.0, max(0.0, rs_rank_percentile * 0.6 + max(0.0, rs_slope_20d) * 2.0))
        not_extended = max(0.0, 100.0 - max(0.0, pct_above_ma50 - 5.0) * 5.0)
        setup_score = round(
            vcp * 0.42 + proximity * 0.22 + rs_momentum * 0.22 + not_extended * 0.14,
            1,
        )
        composite = setup_score
        reasons = [
            f"VCP base {base_bars}d, {base_width:.1f}% wide",
            f"{dist_pivot:.1f}% below pivot ₹{pivot:.2f}",
            "Breakout not triggered yet",
        ]
        if rs_slope_20d > 5:
            reasons.append(f"RS accelerating (+{rs_slope_20d:.1f}% 20d)")
    else:
        criteria = ext_criteria
        passed = ext_passed
        total = len(ext_criteria)
        setup_score = 0.0
        composite = (
            (passed / total) * 45.0
            + min(100.0, rs_rank_percentile) * 0.35
            + vcp * 0.20
        )
        if full_pass:
            composite += 10.0
        composite = round(min(100.0, composite), 1)
        reasons = []
        if full_pass:
            reasons.append("Stage-2 Trend Template + RS leadership")
        else:
            missing = [k for k, v in ext_criteria.items() if not v]
            if missing:
                reasons.append("Missing: " + ", ".join(missing[:3]))
        if vcp >= 60:
            reasons.append(f"VCP score {vcp:.0f}")

    return MinerviniRow(
        stock="",
        scan_mode=mode,
        current_price=round(c, 2),
        ma50=round(m50, 2),
        ma150=round(m150, 2),
        ma200=round(m200, 2),
        high_52w=round(hi52, 2),
        low_52w=round(lo52, 2),
        pct_above_52w_low=round(pct_above_low, 2),
        pct_from_52w_high=round(pct_from_high, 2),
        pct_above_ma50=round(pct_above_ma50, 2),
        recent_move_20d_pct=round(recent_move_20d, 2),
        ma200_slope_21d_pct=round(ma200_slope, 3),
        rs_rank_percentile=round(rs_rank_percentile, 1),
        rs_slope_20d_pct=round(rs_slope_20d, 2),
        excess_20d=round(excess_20d, 2),
        criteria_passed=passed,
        criteria_total=total,
        trend_template_pass=trend_pass,
        rs_pass=rs_pass_ext if mode == "extended" else (
            rs_rank_percentile >= SETUP_RS_MIN_PERCENTILE or rs_slope_20d >= MIN_RS_SLOPE_20D_SETUP
        ),
        full_pass=full_pass,
        setup_pass=setup_pass,
        breakout_pending=breakout_pending,
        criteria=criteria,
        vcp_score=round(vcp, 1),
        composite_score=composite,
        setup_score=setup_score,
        pivot_price=round(pivot, 2),
        base_low=round(base_low, 2),
        base_width_pct=round(base_width, 2),
        base_bars=base_bars,
        dist_to_pivot_pct=dist_pivot,
        rs_line=list(rs_row.rs_line) if rs_row else [],
        reasons=reasons,
    )


def _scan_symbol(
    symbol: str,
    data_dir: Path,
    nifty_df: pd.DataFrame,
    rs_rank: dict[str, float],
    *,
    mode: ScanMode,
    rs_min_percentile: float,
) -> MinerviniRow | None:
    df = read_nifty500_symbol_ohlcv(data_dir, symbol, "1D")
    if df.empty:
        return None
    rs_row = _compute_rs_row(symbol, df, nifty_df)
    rank = rs_rank.get(symbol, 0.0)
    row = _evaluate_symbol(
        df, rs_row, rank, mode=mode, rs_min_percentile=rs_min_percentile,
    )
    if row is None:
        return None
    row.stock = symbol
    return row


def _assign_rs_ranks(rows: list[tuple[str, RelativeStrengthRow | None]]) -> dict[str, float]:
    scored = [(sym, r.strength_score) for sym, r in rows if r is not None]
    if not scored:
        return {}
    scored.sort(key=lambda x: x[1])
    n = len(scored)
    return {sym: round(((i + 1) / n) * 100.0, 1) for i, (sym, _) in enumerate(scored)}


def _criteria_labels(mode: ScanMode) -> dict[str, str]:
    if mode == "setup":
        return {
            "below_pivot": "Close below base pivot (no breakout)",
            "near_pivot": f"{PIVOT_MIN_DIST_PCT:g}–{PIVOT_MAX_DIST_PCT:g}% below pivot",
            "vcp_base": f"VCP score ≥ {MIN_VCP_SETUP_SCORE:g}",
            "tight_base": f"Base width {BASE_MIN_WIDTH_PCT:g}–{BASE_MAX_WIDTH_PCT:g}%",
            "above_200dma": "Close > 200-DMA (uptrend context)",
            "ma200_rising": f"200-DMA rising ({MA200_RISE_LOOKBACK}d)",
            "not_extended": f"Not > {MAX_PCT_ABOVE_MA50:g}% above 50-DMA",
            "recent_move_ok": f"20d move ≤ {MAX_RECENT_MOVE_20D_PCT:g}%",
            "rs_improving": f"RS ≥ {SETUP_RS_MIN_PERCENTILE:g}th pctile or RS slope ↑",
            "above_50dma": "Close > 50-DMA",
        }
    return {
        "price_above_ma150": "Close > 150-DMA",
        "price_above_ma200": "Close > 200-DMA",
        "ma150_above_ma200": "150-DMA > 200-DMA",
        "ma200_rising": f"200-DMA rising ({MA200_RISE_LOOKBACK}d)",
        "ma50_above_long_mas": "50-DMA > 150 & 200-DMA",
        "price_above_ma50": "Close > 50-DMA",
        "pct_above_52w_low": f"≥ {MIN_PCT_ABOVE_52W_LOW:g}% above 52w low",
        "near_52w_high": f"Within {MAX_PCT_FROM_52W_HIGH:g}% of 52w high",
    }


def run_minervini_trend_template_scan(
    *,
    settings: Settings | None = None,
    symbols_file: Path | None = None,
    trade_date: date | None = None,
    top_n: int = 50,
    max_workers: int = 8,
    rs_min_percentile: float = DEFAULT_RS_MIN_PERCENTILE,
    mode: ScanMode = "setup",
    only_full_pass: bool = False,
    min_criteria: int = 8,
) -> dict[str, Any]:
    settings = settings or Settings.from_env(underlying="NIFTY")
    td = trade_date or date.today()

    nifty_df = _load_nifty_daily(settings)
    symbols = load_nifty500_symbols(symbols_file, settings.data_dir)

    if nifty_df.empty:
        payload = {
            "trade_date": td.isoformat(),
            "scan_mode": mode,
            "generated_at": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
            "rows": [],
            "scanned": 0,
            "passed": 0,
            "message": "Could not load NIFTY 50 daily bars (Kite credentials or cache).",
        }
        _save_payload(settings.data_dir, td, payload)
        return payload

    rs_pairs: list[tuple[str, RelativeStrengthRow | None]] = []
    skipped = 0

    def rs_job(sym: str) -> tuple[str, RelativeStrengthRow | None]:
        try:
            df = read_nifty500_symbol_ohlcv(settings.data_dir, sym, "1D")
            if df.empty:
                return sym, None
            return sym, _compute_rs_row(sym, df, nifty_df)
        except Exception as e:
            logger.debug("Minervini RS %s: %s", sym, e)
            return sym, None

    with ThreadPoolExecutor(max_workers=max(1, max_workers)) as ex:
        futs = {ex.submit(rs_job, s): s for s in symbols}
        for fut in as_completed(futs):
            sym, rs_row = fut.result()
            if rs_row is None:
                skipped += 1
            rs_pairs.append((sym, rs_row))

    for _, rs_row in rs_pairs:
        if rs_row is not None:
            rs_row.strength_score = round(
                0.20 * rs_row.rs_slope_5d_pct
                + 0.40 * rs_row.rs_slope_20d_pct
                + 0.25 * rs_row.rs_slope_60d_pct
                + 0.10 * rs_row.excess_20d
                + 0.05 * rs_row.excess_60d,
                2,
            )

    rs_rank = _assign_rs_ranks(rs_pairs)

    results: list[MinerviniRow] = []

    def scan_job(sym: str) -> MinerviniRow | None:
        try:
            return _scan_symbol(
                sym,
                settings.data_dir,
                nifty_df,
                rs_rank,
                mode=mode,
                rs_min_percentile=rs_min_percentile,
            )
        except Exception as e:
            logger.debug("Minervini %s %s: %s", mode, sym, e)
            return None

    with ThreadPoolExecutor(max_workers=max(1, max_workers)) as ex:
        futs = {ex.submit(scan_job, s): s for s in symbols}
        for fut in as_completed(futs):
            row = fut.result()
            if row is not None:
                results.append(row)

    if mode == "setup":
        filtered = list(results)
        filtered.sort(key=lambda r: (-r.setup_score, -r.vcp_score, r.dist_to_pivot_pct))
        passed_count = len(filtered)
    else:
        filtered = [r for r in results if r.criteria_passed >= min_criteria]
        if only_full_pass:
            filtered = [r for r in filtered if r.full_pass]
        elif min_criteria >= 8:
            filtered = [r for r in results if r.criteria_passed >= 7]
        filtered.sort(key=lambda r: (-int(r.full_pass), -r.composite_score, -r.vcp_score))
        passed_count = len([r for r in results if r.full_pass])

    payload = {
        "trade_date": td.isoformat(),
        "scan_mode": mode,
        "generated_at": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
        "scanned": len(symbols),
        "evaluated": len(results),
        "skipped_rs": skipped,
        "passed": passed_count,
        "criteria_passed_count": len(filtered),
        "rs_min_percentile": rs_min_percentile if mode == "extended" else SETUP_RS_MIN_PERCENTILE,
        "only_full_pass": only_full_pass,
        "min_criteria": min_criteria,
        "criteria_labels": _criteria_labels(mode),
        "rows": [asdict(r) for r in filtered[:top_n]],
    }

    if not payload["rows"]:
        payload["message"] = (
            "No candidates for this mode. Ensure data/NIFTY500/SYMBOL_1D.csv exists "
            "(run --tomorrow-watchlist once) and refresh."
        )

    _save_payload(settings.data_dir, td, payload)
    logger.info(
        "Minervini %s: %d pass, %d shown of %d evaluated -> %s",
        mode,
        payload["passed"],
        len(payload["rows"]),
        len(results),
        minervini_trend_template_path(settings.data_dir, td),
    )
    return payload


def _save_payload(data_dir: Path, trade_date: date, payload: dict[str, Any]) -> None:
    p = minervini_trend_template_path(data_dir, trade_date)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_stored_minervini_trend_template(data_dir: Path, trade_date: date) -> dict[str, Any] | None:
    p = minervini_trend_template_path(data_dir, trade_date)
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            pass
    folder = p.parent
    if not folder.exists():
        return None
    files = sorted(folder.glob("minervini_template_*.json"))
    if not files:
        return None
    try:
        return json.loads(files[-1].read_text(encoding="utf-8"))
    except Exception:
        return None
