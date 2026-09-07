"""
Volume profile — POC, value area (VAH/VAL), HVN/LVN nodes.

Used for intraday session levels and EOD daily / weekly / monthly context.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal

import numpy as np
import pandas as pd

from intraday_engine.core.tunables import get_float, get_int

Period = Literal["intraday", "daily", "weekly", "monthly"]

DEFAULT_VALUE_AREA_PCT = 70.0
DEFAULT_NUM_BINS = 50
DEFAULT_HVN_TOP_N = 3
DEFAULT_LVN_TOP_N = 2


@dataclass
class VolumeProfileResult:
    period: str
    poc: float
    vah: float
    val: float
    value_area_pct: float
    total_volume: float
    bar_count: int
    range_high: float
    range_low: float
    hvn: list[float] = field(default_factory=list)
    lvn: list[float] = field(default_factory=list)
    histogram: list[dict[str, float]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _vp_int(key: str, default: int) -> int:
    return get_int("volume_profile", key, default)


def _vp_float(key: str, default: float) -> float:
    return get_float("volume_profile", key, default)


def normalize_ohlcv(
    df: pd.DataFrame,
    *,
    high: str = "high",
    low: str = "low",
    close: str = "close",
    volume: str = "volume",
    open_col: str | None = "open",
) -> pd.DataFrame:
    """Map arbitrary OHLCV columns to standard names."""
    if df.empty:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

    col_map = {
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
    }
    if open_col and open_col in df.columns:
        col_map["open"] = open_col

    out = pd.DataFrame()
    for std, src in col_map.items():
        if src in df.columns:
            out[std] = pd.to_numeric(df[src], errors="coerce")
    if "volume" not in out.columns:
        out["volume"] = 0.0
    out = out.dropna(subset=["high", "low", "close"])
    out["volume"] = out["volume"].fillna(0.0).clip(lower=0.0)
    return out.reset_index(drop=True)


def snapshots_to_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """Convert intraday snapshot frame to OHLCV for volume profile."""
    if df.empty:
        return pd.DataFrame(columns=["high", "low", "close", "volume"])
    high_col = "spot_high" if "spot_high" in df.columns else "spot_high_raw"
    low_col = "spot_low" if "spot_low" in df.columns else "spot_low_raw"
    close_col = "spot_close" if "spot_close" in df.columns else "spot_ltp"
    vol_col = "spot_volume" if "spot_volume" in df.columns else None
    if vol_col is None or df[vol_col].fillna(0).sum() <= 0:
        # Synthesize unit volume so structure levels still compute.
        tmp = df.copy()
        tmp["_unit_vol"] = 1.0
        vol_col = "_unit_vol"
    return normalize_ohlcv(
        df,
        high=high_col,
        low=low_col,
        close=close_col,
        volume=vol_col,
        open_col="spot_open_raw" if "spot_open_raw" in df.columns else None,
    )


def resample_ohlcv(df: pd.DataFrame, period: Literal["weekly", "monthly"]) -> pd.DataFrame:
    """Aggregate daily OHLCV to weekly (Fri) or calendar month."""
    if df.empty:
        return pd.DataFrame()
    work = normalize_ohlcv(df)
    if "date" in df.columns:
        work.index = pd.to_datetime(df["date"], errors="coerce")
    elif isinstance(df.index, pd.DatetimeIndex):
        work.index = df.index
    else:
        return pd.DataFrame()
    work = work.sort_index().dropna(subset=["close"])
    rule = "W-FRI" if period == "weekly" else "ME"
    agg = work.resample(rule).agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    )
    out = agg.dropna(subset=["close"]).reset_index()
    if not out.empty and out.columns[0] != "date":
        out = out.rename(columns={out.columns[0]: "date"})
    return out


def _bin_edges(low: float, high: float, *, num_bins: int, tick_size: float | None) -> np.ndarray:
    if high <= low:
        high = low * 1.001 if low > 0 else low + 1.0
    if tick_size and tick_size > 0:
        start = np.floor(low / tick_size) * tick_size
        end = np.ceil(high / tick_size) * tick_size
        n = max(2, int((end - start) / tick_size) + 1)
        return np.linspace(start, end, n)
    return np.linspace(low, high, num_bins + 1)


def _add_bar_to_hist(hist: np.ndarray, edges: np.ndarray, low: float, high: float, volume: float) -> None:
    if volume <= 0:
        return
    if high <= low:
        mid = (high + low) / 2.0
        idx = int(np.searchsorted(edges, mid, side="right")) - 1
        idx = max(0, min(len(hist) - 1, idx))
        hist[idx] += volume
        return
    i0 = int(np.searchsorted(edges, low, side="right")) - 1
    i1 = int(np.searchsorted(edges, high, side="left")) - 1
    i0 = max(0, min(len(hist) - 1, i0))
    i1 = max(0, min(len(hist) - 1, i1))
    if i0 > i1:
        i1 = i0
    share = volume / (i1 - i0 + 1)
    hist[i0 : i1 + 1] += share


def _value_area_indices(hist: np.ndarray, poc_idx: int, target_vol: float) -> tuple[int, int]:
    lo = hi = poc_idx
    acc = float(hist[poc_idx])
    total = float(hist.sum())
    if total <= 0:
        return lo, hi
    target = min(target_vol, total)
    while acc < target and (lo > 0 or hi < len(hist) - 1):
        vol_below = float(hist[lo - 1]) if lo > 0 else -1.0
        vol_above = float(hist[hi + 1]) if hi < len(hist) - 1 else -1.0
        if vol_above >= vol_below:
            if hi >= len(hist) - 1:
                break
            hi += 1
            acc += float(hist[hi])
        else:
            if lo <= 0:
                break
            lo -= 1
            acc += float(hist[lo])
    return lo, hi


def _find_nodes(hist: np.ndarray, edges: np.ndarray, *, top_hvn: int, top_lvn: int) -> tuple[list[float], list[float]]:
    if len(hist) < 3:
        return [], []
    mids = (edges[:-1] + edges[1:]) / 2.0
    med = float(np.median(hist[hist > 0])) if (hist > 0).any() else 0.0
    hvn_idx: list[int] = []
    lvn_idx: list[int] = []
    for i in range(1, len(hist) - 1):
        if hist[i] >= hist[i - 1] and hist[i] >= hist[i + 1] and hist[i] >= med:
            hvn_idx.append(i)
        if hist[i] <= hist[i - 1] and hist[i] <= hist[i + 1] and hist[i] <= med * 0.5:
            lvn_idx.append(i)
    hvn_idx.sort(key=lambda i: hist[i], reverse=True)
    lvn_idx.sort(key=lambda i: hist[i])
    hvn = [round(float(mids[i]), 2) for i in hvn_idx[:top_hvn]]
    lvn = [round(float(mids[i]), 2) for i in lvn_idx[:top_lvn]]
    return hvn, lvn


def compute_volume_profile(
    df: pd.DataFrame,
    *,
    period: str = "intraday",
    value_area_pct: float | None = None,
    num_bins: int | None = None,
    tick_size: float | None = None,
    hvn_top_n: int | None = None,
    lvn_top_n: int | None = None,
    include_histogram: bool = False,
) -> VolumeProfileResult | None:
    """Build volume-at-price histogram and derive POC / VAH / VAL / nodes."""
    ohlcv = normalize_ohlcv(df) if "high" in df.columns else snapshots_to_ohlcv(df)
    if ohlcv.empty or len(ohlcv) < 1:
        return None

    va_pct = value_area_pct if value_area_pct is not None else _vp_float("VALUE_AREA_PCT", DEFAULT_VALUE_AREA_PCT)
    bins = num_bins if num_bins is not None else _vp_int("NUM_BINS", DEFAULT_NUM_BINS)
    tick = tick_size if tick_size is not None else _vp_float("TICK_SIZE", 0.0) or None
    top_hvn = hvn_top_n if hvn_top_n is not None else _vp_int("HVN_TOP_N", DEFAULT_HVN_TOP_N)
    top_lvn = lvn_top_n if lvn_top_n is not None else _vp_int("LVN_TOP_N", DEFAULT_LVN_TOP_N)

    lo = float(ohlcv["low"].min())
    hi = float(ohlcv["high"].max())
    edges = _bin_edges(lo, hi, num_bins=bins, tick_size=tick)
    hist = np.zeros(len(edges) - 1, dtype=float)

    for _, row in ohlcv.iterrows():
        _add_bar_to_hist(hist, edges, float(row["low"]), float(row["high"]), float(row["volume"]))

    total_vol = float(hist.sum())
    if total_vol <= 0:
        return None

    poc_idx = int(np.argmax(hist))
    target = total_vol * (va_pct / 100.0)
    va_lo, va_hi = _value_area_indices(hist, poc_idx, target)
    poc = round(float((edges[poc_idx] + edges[poc_idx + 1]) / 2.0), 2)
    val = round(float(edges[va_lo]), 2)
    vah = round(float(edges[va_hi + 1]), 2)
    hvn, lvn = _find_nodes(hist, edges, top_hvn=top_hvn, top_lvn=top_lvn)

    histogram: list[dict[str, float]] = []
    if include_histogram:
        mids = (edges[:-1] + edges[1:]) / 2.0
        for i, mid in enumerate(mids):
            if hist[i] <= 0:
                continue
            histogram.append(
                {
                    "price": round(float(mid), 2),
                    "volume": round(float(hist[i]), 2),
                    "pct": round(float(hist[i] / total_vol * 100.0), 2),
                }
            )

    return VolumeProfileResult(
        period=period,
        poc=poc,
        vah=vah,
        val=val,
        value_area_pct=va_pct,
        total_volume=round(total_vol, 2),
        bar_count=len(ohlcv),
        range_high=round(hi, 2),
        range_low=round(lo, 2),
        hvn=hvn,
        lvn=lvn,
        histogram=histogram,
    )


def _distance_pct(spot: float, level: float) -> float:
    if spot <= 0 or level <= 0:
        return 0.0
    return round((spot / level - 1.0) * 100.0, 2)


def profile_context(result: VolumeProfileResult | None, spot: float) -> dict[str, Any]:
    """Annotate profile with spot-relative context."""
    if result is None or spot <= 0:
        return {}
    inside_va = result.val <= spot <= result.vah
    if spot > result.vah:
        position = "above_value_area"
    elif spot < result.val:
        position = "below_value_area"
    elif abs(spot - result.poc) / spot * 100.0 <= 0.15:
        position = "at_poc"
    else:
        position = "inside_value_area"
    return {
        **result.to_dict(),
        "spot": round(spot, 2),
        "inside_value_area": inside_va,
        "position": position,
        "dist_poc_pct": _distance_pct(spot, result.poc),
        "dist_vah_pct": _distance_pct(spot, result.vah),
        "dist_val_pct": _distance_pct(spot, result.val),
    }


def important_levels_from_profile(result: VolumeProfileResult | None) -> list[dict[str, Any]]:
    """Rank key prices for trading context."""
    if result is None:
        return []
    levels: dict[float, str] = {}
    for price, label in (
        (result.poc, "POC"),
        (result.vah, "VAH"),
        (result.val, "VAL"),
        (result.range_high, "RANGE_HIGH"),
        (result.range_low, "RANGE_LOW"),
    ):
        levels[round(price, 2)] = label
    for p in result.hvn:
        levels.setdefault(round(p, 2), "HVN")
    for p in result.lvn:
        levels.setdefault(round(p, 2), "LVN")
    out = [{"price": p, "label": levels[p]} for p in sorted(levels.keys(), reverse=True)]
    return out


def volume_profile_snapshot(
    df: pd.DataFrame,
    spot: float,
    *,
    period: str = "intraday",
) -> dict[str, Any]:
    """Single-period profile + spot context + important levels."""
    result = compute_volume_profile(df, period=period)
    ctx = profile_context(result, spot)
    if not ctx:
        return {}
    ctx["important_levels"] = important_levels_from_profile(result)
    return ctx


def slice_daily_bars(df: pd.DataFrame, bars: int) -> pd.DataFrame:
    if df.empty:
        return df
    return df.tail(max(1, bars)).reset_index(drop=True)


def multi_period_volume_profiles(
    *,
    intraday_df: pd.DataFrame | None = None,
    daily_df: pd.DataFrame | None = None,
    spot: float,
    daily_lookback: int | None = None,
    weekly_lookback: int | None = None,
    monthly_lookback: int | None = None,
) -> dict[str, Any]:
    """
    Compute intraday session + daily / weekly / monthly profiles.

    daily_df expects columns: date, open, high, low, close, volume (1D bars).
    """
    d_lb = daily_lookback or _vp_int("DAILY_LOOKBACK_BARS", 1)
    w_lb = weekly_lookback or _vp_int("WEEKLY_LOOKBACK_BARS", 5)
    m_lb = monthly_lookback or _vp_int("MONTHLY_LOOKBACK_BARS", 22)

    out: dict[str, Any] = {"spot": round(spot, 2), "periods": {}}

    if intraday_df is not None and not intraday_df.empty and spot > 0:
        snap = volume_profile_snapshot(intraday_df, spot, period="intraday")
        if snap:
            out["periods"]["intraday"] = snap

    if daily_df is not None and not daily_df.empty:
        work = daily_df.copy()
        if "date" in work.columns:
            work = work.sort_values("date")

        # Daily profile — last N sessions
        daily_slice = slice_daily_bars(work, d_lb)
        daily_ctx = profile_context(compute_volume_profile(daily_slice, period="daily"), spot)
        if daily_ctx:
            daily_ctx["important_levels"] = important_levels_from_profile(
                compute_volume_profile(daily_slice, period="daily")
            )
            out["periods"]["daily"] = daily_ctx

        # Weekly — aggregate last N daily bars into weekly candles, then profile last weeks
        w_slice = slice_daily_bars(work, max(w_lb, 10))
        if len(w_slice) >= 5 and "date" in w_slice.columns:
            w_bars = resample_ohlcv(w_slice, "weekly")
            if not w_bars.empty:
                w_ctx = profile_context(compute_volume_profile(w_bars, period="weekly"), spot)
                if w_ctx:
                    w_ctx["important_levels"] = important_levels_from_profile(
                        compute_volume_profile(w_bars, period="weekly")
                    )
                    out["periods"]["weekly"] = w_ctx

        # Monthly — last N daily bars aggregated
        m_slice = slice_daily_bars(work, m_lb)
        if len(m_slice) >= 10 and "date" in m_slice.columns:
            m_bars = resample_ohlcv(m_slice, "monthly")
            if m_bars.empty and len(m_slice) >= 5:
                m_bars = m_slice
            if not m_bars.empty:
                m_ctx = profile_context(compute_volume_profile(m_bars, period="monthly"), spot)
                if m_ctx:
                    m_ctx["important_levels"] = important_levels_from_profile(
                        compute_volume_profile(m_bars, period="monthly")
                    )
                    out["periods"]["monthly"] = m_ctx

    # Flat merged level list (deduped, nearest first)
    merged: dict[float, set[str]] = {}
    for pdata in out.get("periods", {}).values():
        for lvl in pdata.get("important_levels") or []:
            p = round(float(lvl["price"]), 2)
            merged.setdefault(p, set()).add(str(lvl.get("label", "")))
    if spot > 0 and merged:
        ranked = sorted(merged.keys(), key=lambda p: abs(p - spot))
        out["merged_levels"] = [
            {"price": p, "labels": sorted(merged[p]), "dist_pct": _distance_pct(spot, p)}
            for p in ranked[:12]
        ]

    return out


def compact_volume_profile(vp: dict[str, Any] | None) -> dict[str, Any]:
    """Trim profile payload for signals / API responses."""
    if not vp:
        return {}
    periods: dict[str, Any] = {}
    for key, pdata in (vp.get("periods") or {}).items():
        if not pdata:
            continue
        periods[key] = {
            "poc": pdata.get("poc"),
            "vah": pdata.get("vah"),
            "val": pdata.get("val"),
            "position": pdata.get("position"),
            "inside_value_area": pdata.get("inside_value_area"),
            "dist_poc_pct": pdata.get("dist_poc_pct"),
            "hvn": pdata.get("hvn") or [],
            "lvn": pdata.get("lvn") or [],
        }
    return {
        "spot": vp.get("spot"),
        "periods": periods,
        "merged_levels": (vp.get("merged_levels") or [])[:8],
    }
