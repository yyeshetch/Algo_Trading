"""Money flow indicators: MFI, CMF, OBV."""

from __future__ import annotations

import numpy as np
import pandas as pd


def cmf_series(
    df: pd.DataFrame,
    period: int | None = None,
    *,
    high: str = "high",
    low: str = "low",
    close: str = "close",
    volume: str = "volume",
) -> pd.Series:
    """Chaikin Money Flow over ``period`` bars."""
    if period is None:
        from intraday_engine.core.tunables import get_int

        period = get_int("money_flow", "CMF_PERIOD", 20)
    h = pd.Series(df[high], dtype=float)
    l = pd.Series(df[low], dtype=float)
    c = pd.Series(df[close], dtype=float)
    v = pd.Series(df[volume], dtype=float)
    rng = (h - l).replace(0, np.nan)
    mfm = ((c - l) - (h - c)) / rng
    mfv = (mfm * v).fillna(0.0)
    vol_sum = v.rolling(period, min_periods=period).sum()
    return mfv.rolling(period, min_periods=period).sum() / vol_sum.replace(0, np.nan)


def obv_series(close: pd.Series, volume: pd.Series) -> pd.Series:
    """On Balance Volume cumulative series."""
    close = pd.Series(close, dtype=float)
    volume = pd.Series(volume, dtype=float).fillna(0.0)
    sign = np.sign(close.diff().fillna(0.0))
    return (sign * volume).cumsum()


def mfi_series(
    df: pd.DataFrame,
    period: int | None = None,
    *,
    high: str = "high",
    low: str = "low",
    close: str = "close",
    volume: str = "volume",
) -> pd.Series:
    """Money Flow Index — volume-weighted RSI on typical price."""
    if period is None:
        from intraday_engine.core.tunables import get_int

        period = get_int("money_flow", "MFI_PERIOD", 14)
    h = pd.Series(df[high], dtype=float)
    l = pd.Series(df[low], dtype=float)
    c = pd.Series(df[close], dtype=float)
    v = pd.Series(df[volume], dtype=float).fillna(0.0)
    tp = (h + l + c) / 3.0
    rmf = tp * v
    tp_diff = tp.diff()
    pos_mf = rmf.where(tp_diff > 0, 0.0)
    neg_mf = rmf.where(tp_diff < 0, 0.0)
    pos_sum = pos_mf.rolling(period, min_periods=period).sum()
    neg_sum = neg_mf.rolling(period, min_periods=period).sum()
    mfr = pos_sum / neg_sum.replace(0, np.nan)
    out = 100.0 - (100.0 / (1.0 + mfr))
    out = out.where(neg_sum > 0, np.where(pos_sum > 0, 100.0, np.nan))
    out = out.where(pos_sum > 0, np.where(neg_sum > 0, 0.0, np.nan))
    return out.astype(float)


def _last_value(series: pd.Series) -> float | None:
    if series.empty:
        return None
    last = series.dropna()
    if last.empty:
        return None
    return float(last.iloc[-1])


def cmf_last(
    df: pd.DataFrame,
    period: int | None = None,
    *,
    high: str = "high",
    low: str = "low",
    close: str = "close",
    volume: str = "volume",
) -> float | None:
    """Last non-NaN CMF value, or None."""
    return _last_value(cmf_series(df, period, high=high, low=low, close=close, volume=volume))


def mfi_last(
    df: pd.DataFrame,
    period: int | None = None,
    *,
    high: str = "high",
    low: str = "low",
    close: str = "close",
    volume: str = "volume",
) -> float | None:
    """Last non-NaN MFI value, or None."""
    return _last_value(mfi_series(df, period, high=high, low=low, close=close, volume=volume))


def obv_slope_pct(
    df: pd.DataFrame,
    lookback: int = 20,
    *,
    close: str = "close",
    volume: str = "volume",
) -> float:
    """Net signed volume over lookback as % of total volume in the window."""
    if len(df) < lookback + 1:
        return 0.0
    tail = df.tail(lookback + 1)
    close_s = pd.Series(tail[close], dtype=float)
    vol_s = pd.Series(tail[volume], dtype=float).fillna(0.0)
    sign = np.sign(close_s.diff().fillna(0.0))
    signed_vol = (sign * vol_s).iloc[1:]
    total_vol = float(vol_s.iloc[1:].sum())
    if total_vol <= 0:
        return 0.0
    return float(signed_vol.sum() / total_vol) * 100.0


def mfi_zone(mfi: float | None) -> str:
    """Label MFI reading as overbought / oversold / neutral."""
    if mfi is None or not np.isfinite(mfi):
        return "n/a"
    from intraday_engine.core.tunables import get_float

    overbought = get_float("money_flow", "MFI_OVERBOUGHT", 80.0)
    oversold = get_float("money_flow", "MFI_OVERSOLD", 20.0)
    if mfi >= overbought:
        return "overbought"
    if mfi <= oversold:
        return "oversold"
    return "neutral"


def money_flow_bias(mfi: float | None, cmf: float | None) -> str:
    """Combined MFI + CMF bias: bullish / bearish / neutral."""
    bullish = 0
    bearish = 0
    if mfi is not None and np.isfinite(mfi):
        if mfi > 55:
            bullish += 1
        elif mfi < 45:
            bearish += 1
    if cmf is not None and np.isfinite(cmf):
        if cmf > 0.05:
            bullish += 1
        elif cmf < -0.05:
            bearish += 1
    if bullish > bearish:
        return "bullish"
    if bearish > bullish:
        return "bearish"
    return "neutral"


def spot_ohlcv_frame(
    df: pd.DataFrame,
    *,
    high: str = "spot_high",
    low: str = "spot_low",
    close: str | None = None,
    volume: str = "spot_volume",
) -> pd.DataFrame | None:
    """Map index 5-min snapshot columns to standard OHLCV names."""
    close_col = close or ("spot_close" if "spot_close" in df.columns else "spot_ltp")
    needed = (high, low, close_col, volume)
    if not all(col in df.columns for col in needed):
        return None
    return pd.DataFrame(
        {
            "high": df[high],
            "low": df[low],
            "close": df[close_col],
            "volume": df[volume],
        }
    )


def money_flow_snapshot(df: pd.DataFrame) -> dict[str, float | str | None]:
    """Latest MFI/CMF/OBV metrics for an index or OHLCV frame."""
    ohlcv = spot_ohlcv_frame(df)
    if ohlcv is None or ohlcv.empty:
        return {
            "mfi": None,
            "cmf": None,
            "obv_slope_pct": None,
            "bias": "n/a",
            "mfi_zone": "n/a",
        }
    from intraday_engine.core.tunables import get_int

    mfi = mfi_last(ohlcv)
    cmf = cmf_last(ohlcv)
    obv_lb = get_int("money_flow", "INDEX_OBV_LOOKBACK", 10)
    obv_slope = obv_slope_pct(ohlcv, lookback=obv_lb)
    bias = money_flow_bias(mfi, cmf)
    zone = mfi_zone(mfi)
    return {
        "mfi": round(mfi, 1) if mfi is not None else None,
        "cmf": round(cmf, 4) if cmf is not None else None,
        "obv_slope_pct": round(obv_slope, 2),
        "bias": bias,
        "mfi_zone": zone,
    }
