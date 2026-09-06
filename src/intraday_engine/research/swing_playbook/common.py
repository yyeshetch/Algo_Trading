"""Shared daily-bar metrics for swing playbook scanners."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import date
from typing import Any

import numpy as np
import pandas as pd

from intraday_engine.core.config import Settings
from intraday_engine.research.nifty500_accumulation_scanner import load_nifty500_symbols
from intraday_engine.research.relative_strength_scanner import _align_on_date, _load_nifty_daily, _pct_change
from intraday_engine.storage.nifty500_csv import read_nifty500_symbol_ohlcv

MIN_BARS = 260
MIN_PRICE = 100.0
MIN_AVG_VALUE_CR = 5.0  # ₹ crore average daily traded value


@dataclass
class StockMetrics:
    stock: str
    close: float
    volume: float
    avg_value_cr: float
    ma20: float
    ma50: float
    ma200: float
    return_20d: float
    return_63d: float
    return_126d: float
    return_252d: float
    nifty_return_63d: float
    nifty_return_126d: float
    excess_63d: float
    excess_126d: float
    rs_slope_20d: float
    rs_value: float
    atr14: float
    atr_ratio: float
    range_20d_pct: float
    high_20d: float
    high_50d: float
    high_252d: float
    low_20d: float
    pct_from_52w_high: float
    volume_ratio: float
    vol_trend_ratio: float
    above_ma50: bool
    above_ma200: bool
    ma50_above_ma200: bool
    ma20_above_ma50: bool
    ma200_rising: bool
    stage: int
    rsi14: float
    gap_up_5d_pct: float
    pullback_to_ma20: bool
    pullback_to_ma50: bool
    turtle_break_20: bool
    turtle_break_55: bool
    liquidity_ok: bool
    components: dict[str, float] = field(default_factory=dict)


def _pct_return(series: pd.Series, bars: int) -> float:
    return _pct_change(series, bars)


def _wilder_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    h, l, c = df["high"].astype(float), df["low"].astype(float), df["close"].astype(float)
    prev_c = c.shift(1)
    tr = pd.concat([(h - l), (h - prev_c).abs(), (l - prev_c).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()


def _rsi(close: pd.Series, period: int = 14) -> float:
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    out = 100.0 - (100.0 / (1.0 + rs))
    val = out.dropna()
    return float(val.iloc[-1]) if not val.empty else 50.0


def _weinstein_stage(close: pd.Series, ma50: pd.Series, ma200: pd.Series) -> int:
    c = float(close.iloc[-1])
    m50 = float(ma50.iloc[-1])
    m200 = float(ma200.iloc[-1])
    m200_prev = float(ma200.iloc[-21]) if len(ma200) > 21 else m200
    slope = (m200 - m200_prev) / m200_prev * 100.0 if m200_prev > 0 else 0.0
    if c < m200 and slope < -0.5:
        return 4
    if c > m200 and m50 > m200 and slope > 0.3:
        return 2
    if c > m200 and abs(slope) < 0.5 and c < m50 * 1.05:
        return 3
    return 1


def _compute_metrics(df: pd.DataFrame, symbol: str, nifty_df: pd.DataFrame) -> StockMetrics | None:
    if len(df) < MIN_BARS:
        return None
    df = df.tail(280).reset_index(drop=True)
    close = df["close"].astype(float)
    volume = df["volume"].astype(float).fillna(0.0)
    last_close = float(close.iloc[-1])
    if last_close < MIN_PRICE:
        return None

    avg_val = float((close * volume).tail(20).mean())
    avg_value_cr = avg_val / 1e7
    liquidity_ok = avg_value_cr >= MIN_AVG_VALUE_CR

    ma20 = close.rolling(20, min_periods=20).mean()
    ma50 = close.rolling(50, min_periods=50).mean()
    ma200 = close.rolling(200, min_periods=200).mean()
    atr = _wilder_atr(df, 14)
    atr_last = float(atr.iloc[-1]) if not atr.dropna().empty else 0.0
    atr_base = float(atr.tail(40).mean()) if len(atr.dropna()) >= 20 else atr_last
    atr_ratio = atr_last / atr_base if atr_base > 0 else 1.0

    aligned = _align_on_date(df, nifty_df)
    if aligned.empty or len(aligned) < MIN_BARS:
        return None
    stock_close = aligned["stock_close"].astype(float)
    nifty_close = aligned["nifty_close"].astype(float)
    nifty_ret_63 = _pct_change(nifty_close, 63)
    nifty_ret_126 = _pct_change(nifty_close, 126)
    ret_20 = _pct_change(stock_close, 20)
    ret_63 = _pct_change(stock_close, 63)
    ret_126 = _pct_change(stock_close, 126)
    ret_252 = _pct_change(stock_close, 252)

    rs_raw = stock_close / nifty_close
    rs_base = float(rs_raw.iloc[-min(len(rs_raw), 60)])
    rs_line = (rs_raw / rs_base * 100.0) if rs_base > 0 else rs_raw
    rs_slope = _pct_change(rs_line, 20) if len(rs_line) > 21 else 0.0
    rs_val = float(rs_line.iloc[-1]) if len(rs_line) else 100.0

    high_20 = float(df["high"].tail(20).max())
    high_50 = float(df["high"].tail(50).max())
    high_252 = float(df["high"].tail(252).max())
    low_20 = float(df["low"].tail(20).min())
    range_20 = (high_20 - low_20) / last_close * 100.0 if last_close > 0 else 100.0
    pct_from_high = (last_close / high_252 - 1.0) * 100.0 if high_252 > 0 else 0.0

    vol_sma20 = float(volume.tail(20).mean())
    vol_ratio = float(volume.iloc[-1]) / vol_sma20 if vol_sma20 > 0 else 0.0
    vol_recent = float(volume.tail(5).mean())
    vol_prior = float(volume.tail(20).head(15).mean())
    vol_trend = vol_recent / vol_prior if vol_prior > 0 else 1.0

    m20 = float(ma20.iloc[-1])
    m50 = float(ma50.iloc[-1])
    m200 = float(ma200.iloc[-1])
    m200_prev = float(ma200.iloc[-21]) if len(ma200) > 21 else m200
    ma200_rising = m200 > m200_prev * 1.002

    gap_up = 0.0
    if len(df) >= 6:
        five_low = float(df["low"].iloc[-6:-1].min())
        if five_low > 0:
            gap_up = (last_close / five_low - 1.0) * 100.0

    highest_20 = float(close.iloc[-21:-1].max()) if len(close) > 21 else last_close
    highest_55 = float(close.iloc[-56:-1].max()) if len(close) > 56 else last_close

    return StockMetrics(
        stock=symbol,
        close=round(last_close, 2),
        volume=float(volume.iloc[-1]),
        avg_value_cr=round(avg_value_cr, 2),
        ma20=round(m20, 2),
        ma50=round(m50, 2),
        ma200=round(m200, 2),
        return_20d=round(ret_20, 2),
        return_63d=round(ret_63, 2),
        return_126d=round(ret_126, 2),
        return_252d=round(ret_252, 2),
        nifty_return_63d=round(nifty_ret_63, 2),
        nifty_return_126d=round(nifty_ret_126, 2),
        excess_63d=round(ret_63 - nifty_ret_63, 2),
        excess_126d=round(ret_126 - nifty_ret_126, 2),
        rs_slope_20d=round(rs_slope, 2),
        rs_value=round(rs_val, 2),
        atr14=round(atr_last, 4),
        atr_ratio=round(atr_ratio, 3),
        range_20d_pct=round(range_20, 2),
        high_20d=round(high_20, 2),
        high_50d=round(high_50, 2),
        high_252d=round(high_252, 2),
        low_20d=round(low_20, 2),
        pct_from_52w_high=round(pct_from_high, 2),
        volume_ratio=round(vol_ratio, 2),
        vol_trend_ratio=round(vol_trend, 2),
        above_ma50=last_close > m50,
        above_ma200=last_close > m200,
        ma50_above_ma200=m50 > m200,
        ma20_above_ma50=m20 > m50,
        ma200_rising=ma200_rising,
        stage=_weinstein_stage(close, ma50, ma200),
        rsi14=round(_rsi(close), 1),
        gap_up_5d_pct=round(gap_up, 2),
        pullback_to_ma20=abs(last_close - m20) / last_close * 100.0 <= 3.0 and last_close >= m20 * 0.98,
        pullback_to_ma50=abs(last_close - m50) / last_close * 100.0 <= 4.0 and last_close >= m50 * 0.97,
        turtle_break_20=last_close > highest_20,
        turtle_break_55=last_close > highest_55,
        liquidity_ok=liquidity_ok,
    )


def build_universe_metrics(
    settings: Settings,
    *,
    symbols: list[str] | None = None,
    symbols_file=None,
) -> tuple[list[StockMetrics], dict[str, Any]]:
    syms = symbols or load_nifty500_symbols(symbols_file, settings.data_dir)
    nifty = _load_nifty_daily(settings, lookback_days=280)
    if nifty.empty:
        raise RuntimeError("NIFTY 50 daily bars unavailable for swing playbook benchmarks.")

    metrics: list[StockMetrics] = []
    skipped = 0
    for sym in syms:
        df = read_nifty500_symbol_ohlcv(settings.data_dir, sym, "1D")
        if df.empty:
            skipped += 1
            continue
        m = _compute_metrics(df, sym, nifty)
        if m is None:
            skipped += 1
            continue
        metrics.append(m)

    meta = {
        "universe_size": len(syms),
        "computed": len(metrics),
        "skipped": skipped,
    }
    return metrics, meta


def metrics_to_dict(m: StockMetrics) -> dict[str, Any]:
    return asdict(m)
