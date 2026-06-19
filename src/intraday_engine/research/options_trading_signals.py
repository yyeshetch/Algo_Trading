"""
Options Trading signals from Index_Analysis.csv (direction-engine 5-min ATM CE/PE bars).

Primary rules (ATM CE / PE separately):
- 13 EMA on HIGH
- Entry when candle closes above EMA within 5% of close, top wick < 4% of premium (high)
- Prior 5 bars: red OPEN / green CLOSE within signal candle [low, close]
- Close above the high of the prior 4 candles
- SL at 75% of candle range below close; skip if risk > 20% of premium

Alternative — doji → strong candle:
- Prior bar is a doji (body <= 15% of range)
- Current bar is strong bullish; skip if close > 5% above EMA13(H)
- Same SL / max-risk filters

Alternative — cluster range breakout + volume:
- Prior 5 bars form a tight range cluster (width ≤ 10%), volume > EMA20(volume)
- Current bar closes above cluster high (bullish)
- Same SL / max-risk filters

Alternative — sideways cluster breakout:
- Consecutive sideways / barcode candles (variable length, min 5)
- First expansion candle closes above cluster high (bullish)
- Same SL / max-risk filters (no volume filter)
"""

from __future__ import annotations

import json
import logging
from datetime import date, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from intraday_engine.storage.data_store import load_market_data
from intraday_engine.storage.layout import (
    analysis_day_path,
    asset_class_for_underlying,
    normalize_underlying,
    options_trading_signals_path,
)

logger = logging.getLogger(__name__)

EMA_PERIOD = 13
VOLUME_EMA_PERIOD = 20
MAX_RISK_PCT = 20.0
TOP_WICK_MAX_PCT_OF_HIGH = 3.0
PRIOR_CONSOLIDATION_BARS = 5
PRIOR_HIGH_BARS = 4
DOJI_MAX_BODY_PCT_OF_RANGE = 15.0
STRONG_MIN_BODY_PCT_OF_RANGE = 60.0
STRONG_MAX_UPPER_WICK_PCT_OF_HIGH = 5.0
SL_RANGE_FRACTION = 0.75
MAX_EMA_DISTANCE_PCT_OF_CLOSE = 5.0
CLUSTER_RANGE_BARS = 5
CLUSTER_MAX_WIDTH_PCT = 10.0
CLUSTER_BAR_MAX_RANGE_PCT = 25.0
MIN_SIDEWAYS_CLUSTER_BARS = 5
MAX_SIDEWAYS_CLUSTER_LOOKBACK = 50
SIDEWAYS_BARCODE_MAX_RANGE_PCT = 9.0
SIDEWAYS_CHOP_MAX_RANGE_PCT = 12.0
SIDEWAYS_CHOP_MAX_BODY_PCT = 45.0
SIDEWAYS_CLUSTER_MAX_DIRECTIONAL_BODY_PCT = 52.0
BREAKOUT_MIN_BODY_PCT = 55.0
SIDEWAYS_BREAKOUT_RANGE_MULTIPLIER = 1.5

SIGNAL_TYPE_PRIORITY = ("primary", "cluster_breakout", "sideways_breakout", "doji_strong")


def _common_risk_checks(close: float, sl: float, risk_pct: float | None) -> dict[str, bool]:
    return {
        "entry_above_sl": close > sl,
        "risk_ok": risk_pct is not None and risk_pct <= MAX_RISK_PCT,
    }


def options_signal_definitions() -> list[dict[str, Any]]:
    """Rules for each signal_type with toggleable condition ids (shown on dashboard)."""
    return [
        {
            "id": "primary",
            "label": "Primary",
            "conditions": [
                {"id": "close_above_ema", "label": f"Close above EMA{EMA_PERIOD}(high)"},
                {"id": "close_near_ema", "label": f"Close within {MAX_EMA_DISTANCE_PCT_OF_CLOSE:g}% of EMA"},
                {"id": "no_top_wick", "label": f"Top wick < {TOP_WICK_MAX_PCT_OF_HIGH:g}% of high"},
                {
                    "id": "prior_consolidation",
                    "label": f"Prior {PRIOR_CONSOLIDATION_BARS} bars in close–low band",
                },
                {"id": "close_above_prior4", "label": f"Close above prior {PRIOR_HIGH_BARS} highs"},
                {"id": "entry_above_sl", "label": f"Entry above SL ({SL_RANGE_FRACTION:.0%} of range below close)"},
                {"id": "risk_ok", "label": f"Risk ≤ {MAX_RISK_PCT:g}% of premium"},
            ],
        },
        {
            "id": "doji_strong",
            "label": "Doji→Strong",
            "conditions": [
                {"id": "prior_doji", "label": f"Prior bar is doji (body ≤ {DOJI_MAX_BODY_PCT_OF_RANGE:g}% of range)"},
                {
                    "id": "strong_bullish",
                    "label": f"Strong bullish (body ≥ {STRONG_MIN_BODY_PCT_OF_RANGE:g}% of range)",
                },
                {"id": "ema_not_extended", "label": f"Close not > {MAX_EMA_DISTANCE_PCT_OF_CLOSE:g}% above EMA13(H)"},
                {"id": "entry_above_sl", "label": f"Entry above SL ({SL_RANGE_FRACTION:.0%} of range below close)"},
                {"id": "risk_ok", "label": f"Risk ≤ {MAX_RISK_PCT:g}% of premium"},
            ],
        },
        {
            "id": "cluster_breakout",
            "label": "Cluster+Vol",
            "conditions": [
                {
                    "id": "range_cluster",
                    "label": f"Prior {CLUSTER_RANGE_BARS} range candles (width ≤ {CLUSTER_MAX_WIDTH_PCT:g}%)",
                },
                {"id": "clears_cluster", "label": "Bullish close above cluster high"},
                {"id": "volume_above_ema20", "label": f"Volume > EMA{VOLUME_EMA_PERIOD}(volume)"},
                {"id": "entry_above_sl", "label": f"Entry above SL ({SL_RANGE_FRACTION:.0%} of range below close)"},
                {"id": "risk_ok", "label": f"Risk ≤ {MAX_RISK_PCT:g}% of premium"},
            ],
        },
        {
            "id": "sideways_breakout",
            "label": "Sideways Breakout",
            "conditions": [
                {
                    "id": "sideways_cluster",
                    "label": (
                        f"≥ {MIN_SIDEWAYS_CLUSTER_BARS} consecutive sideways/barcode bars "
                        f"(range ≤ {SIDEWAYS_BARCODE_MAX_RANGE_PCT:g}% or chop ≤ {SIDEWAYS_CHOP_MAX_RANGE_PCT:g}%)"
                    ),
                },
                {
                    "id": "breakout_expansion",
                    "label": f"Expansion breakout bar (body ≥ {BREAKOUT_MIN_BODY_PCT:g}% of range)",
                },
                {"id": "clears_sideways_cluster", "label": "Bullish close above sideways cluster high"},
                {"id": "entry_above_sl", "label": f"Entry above SL ({SL_RANGE_FRACTION:.0%} of range below close)"},
                {"id": "risk_ok", "label": f"Risk ≤ {MAX_RISK_PCT:g}% of premium"},
            ],
        },
    ]


def _load_index_analysis(
    data_dir: Path,
    trade_date: date,
    underlying: str,
) -> pd.DataFrame:
    u = normalize_underlying(underlying)
    df = load_market_data(
        data_dir,
        trade_date=trade_date,
        asset_class=asset_class_for_underlying(u),
        underlying=u,
    )
    if df.empty:
        return df
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df.sort_values("timestamp").reset_index(drop=True)


def _has_option_ohlc(df: pd.DataFrame, prefix: str) -> bool:
    cols = (f"{prefix}_open", f"{prefix}_high", f"{prefix}_low", f"{prefix}_close")
    if not all(c in df.columns for c in cols):
        return False
    close = pd.to_numeric(df[f"{prefix}_close"], errors="coerce")
    return close.notna().any() and (close > 0).any()


def _candles_from_csv_ohlc(df: pd.DataFrame, option_type: str) -> pd.DataFrame:
    opt = option_type.upper()
    prefix = "call" if opt == "CE" else "put"
    sym_col = "ce_symbol" if opt == "CE" else "pe_symbol"
    vol_col = f"{prefix}_volume"

    out = pd.DataFrame(
        {
            "timestamp": df["timestamp"],
            "open": pd.to_numeric(df[f"{prefix}_open"], errors="coerce"),
            "high": pd.to_numeric(df[f"{prefix}_high"], errors="coerce"),
            "low": pd.to_numeric(df[f"{prefix}_low"], errors="coerce"),
            "close": pd.to_numeric(df[f"{prefix}_close"], errors="coerce"),
            "volume": pd.to_numeric(df[vol_col], errors="coerce")
            if vol_col in df.columns
            else 0.0,
            "tradingsymbol": df[sym_col].astype(str) if sym_col in df.columns else "",
            "atm_strike": pd.to_numeric(df.get("atm_strike", 0), errors="coerce"),
            "spot_price": pd.to_numeric(df.get("spot_ltp", 0), errors="coerce"),
        }
    )
    out = out.dropna(subset=["close"])
    out = out[out["close"] > 0]
    return out.reset_index(drop=True)


def _fetch_option_ohlc_kite(
    trade_date: date,
    underlying: str,
    option_type: str,
    symbol: str,
) -> pd.DataFrame:
    """Fetch real 5-min option OHLC from Kite for EMA calculation."""
    from intraday_engine.core.config import Settings
    from intraday_engine.fetch.market_data import _drop_incomplete_candles, _market_window, _to_candle_df
    from intraday_engine.fetch.zerodha_client import ZerodhaClient

    symbol = str(symbol or "").strip()
    if not symbol or symbol.lower() == "nan":
        return pd.DataFrame()

    prefix = "call" if option_type.upper() == "CE" else "put"
    settings = Settings.from_env(underlying=underlying)
    client = ZerodhaClient(settings)
    from_dt, to_dt = _market_window(trade_date)

    quote = client.quote([symbol])
    token = int(quote[symbol]["instrument_token"])
    raw = client.historical_data(token, from_dt, to_dt, interval="5minute", oi=True)
    candles = _drop_incomplete_candles(_to_candle_df(raw, prefix, include_oi=True), trade_date, 5)
    if candles.empty:
        return pd.DataFrame()

    tradingsymbol = symbol.split(":", 1)[-1] if ":" in symbol else symbol
    vol_col = f"{prefix}_volume"
    return pd.DataFrame(
        {
            "timestamp": candles["timestamp"],
            "open": candles[f"{prefix}_open_raw"],
            "high": candles[f"{prefix}_high_raw"],
            "low": candles[f"{prefix}_low_raw"],
            "close": candles[f"{prefix}_close"],
            "volume": candles[vol_col] if vol_col in candles.columns else 0.0,
            "tradingsymbol": tradingsymbol,
        }
    )


def _merge_analysis_metadata(candles: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
    if candles.empty:
        return candles

    meta = df[["timestamp", "atm_strike", "spot_ltp"]].copy()
    meta["timestamp"] = pd.to_datetime(meta["timestamp"])
    out = candles.merge(meta, on="timestamp", how="left")
    out["spot_price"] = pd.to_numeric(out["spot_ltp"], errors="coerce").fillna(0)
    out["atm_strike"] = pd.to_numeric(out["atm_strike"], errors="coerce").fillna(0).astype(int)
    return out.drop(columns=["spot_ltp"], errors="ignore").reset_index(drop=True)


def _merge_kite_volume(candles: pd.DataFrame, kite: pd.DataFrame) -> pd.DataFrame:
    """Attach Kite volume to CSV OHLC rows matched by timestamp."""
    if candles.empty or kite.empty or "volume" not in kite.columns:
        return candles
    out = candles.copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"])
    kv = kite[["timestamp", "volume"]].copy()
    kv["timestamp"] = pd.to_datetime(kv["timestamp"])
    merged = out.merge(kv, on="timestamp", how="left", suffixes=("", "_kite"))
    if "volume_kite" in merged.columns:
        merged["volume"] = pd.to_numeric(merged["volume_kite"], errors="coerce").fillna(
            pd.to_numeric(merged.get("volume", 0), errors="coerce").fillna(0.0)
        )
        merged = merged.drop(columns=["volume_kite"])
    return merged.reset_index(drop=True)


def _option_candles_from_analysis(
    df: pd.DataFrame,
    option_type: str,
    *,
    trade_date: date,
    underlying: str,
) -> tuple[pd.DataFrame, str]:
    """Build 5-min OHLC for ATM CE or PE. Returns (candles, source)."""
    if df.empty:
        return pd.DataFrame(), "none"

    opt = option_type.upper()
    prefix = "call" if opt == "CE" else "put"
    sym_col = "ce_symbol" if opt == "CE" else "pe_symbol"

    if _has_option_ohlc(df, prefix):
        candles = _candles_from_csv_ohlc(df, opt)
        vol = pd.to_numeric(candles.get("volume", 0), errors="coerce").fillna(0.0)
        if (vol > 0).any():
            return candles, "csv"
        symbol = ""
        if sym_col in df.columns:
            symbols = df[sym_col].dropna().astype(str)
            symbols = symbols[symbols.str.len() > 0]
            if not symbols.empty:
                symbol = symbols.mode().iloc[0]
        if symbol:
            try:
                kite_candles = _fetch_option_ohlc_kite(trade_date, underlying, opt, symbol)
                if not kite_candles.empty and (pd.to_numeric(kite_candles["volume"], errors="coerce").fillna(0) > 0).any():
                    return _merge_kite_volume(candles, kite_candles), "csv+kite_vol"
            except Exception as exc:
                logger.warning("Kite volume merge failed for %s %s: %s", underlying, opt, exc)
        return candles, "csv"

    symbol = ""
    if sym_col in df.columns:
        symbols = df[sym_col].dropna().astype(str)
        symbols = symbols[symbols.str.len() > 0]
        if not symbols.empty:
            symbol = symbols.mode().iloc[0]

    if symbol:
        try:
            kite_candles = _fetch_option_ohlc_kite(trade_date, underlying, opt, symbol)
            if not kite_candles.empty:
                return _merge_analysis_metadata(kite_candles, df), "kite"
        except Exception as exc:
            logger.warning("Kite OHLC fetch failed for %s %s: %s", underlying, opt, exc)

    logger.warning(
        "No real OHLC for %s %s on %s — refresh index data or check Kite credentials",
        underlying,
        opt,
        trade_date.isoformat(),
    )
    return pd.DataFrame(), "none"


def _prepare_indicator_df(candles: pd.DataFrame) -> pd.DataFrame:
    df = candles.copy()
    df["ema13_high"] = df["high"].ewm(span=EMA_PERIOD, adjust=False).mean()
    df["volume"] = pd.to_numeric(df.get("volume", 0), errors="coerce").fillna(0.0)
    if (df["volume"] > 0).any():
        df["ema20_volume"] = df["volume"].ewm(span=VOLUME_EMA_PERIOD, adjust=False).mean()
    else:
        df["ema20_volume"] = float("nan")
    return df


def _no_top_wick_bullish(row: pd.Series, max_pct: float = TOP_WICK_MAX_PCT_OF_HIGH) -> bool:
    o, h, _, c = float(row["open"]), float(row["high"]), float(row["low"]), float(row["close"])
    if c < o or h <= 0:
        return False
    upper_wick = max(0.0, h - max(o, c))
    return upper_wick < (max_pct / 100.0) * h


def _body_pct_of_range(row: pd.Series) -> float | None:
    o, h, l, c = float(row["open"]), float(row["high"]), float(row["low"]), float(row["close"])
    rng = h - l
    if rng <= 0:
        return None
    return abs(c - o) / rng * 100.0


def _range_pct_of_close(row: pd.Series) -> float | None:
    h, l, c = float(row["high"]), float(row["low"]), float(row["close"])
    if c <= 0:
        return None
    return (h - l) / c * 100.0


def _is_doji(row: pd.Series) -> bool:
    body_pct = _body_pct_of_range(row)
    return body_pct is not None and body_pct <= DOJI_MAX_BODY_PCT_OF_RANGE


def _is_strong_bullish(row: pd.Series) -> bool:
    o, h, l, c = float(row["open"]), float(row["high"]), float(row["low"]), float(row["close"])
    if c <= o or h <= 0:
        return False
    body_pct = _body_pct_of_range(row)
    if body_pct is None or body_pct < STRONG_MIN_BODY_PCT_OF_RANGE:
        return False
    upper_wick = max(0.0, h - c)
    return upper_wick < (STRONG_MAX_UPPER_WICK_PCT_OF_HIGH / 100.0) * h


def _is_consolidation_bar(row: pd.Series) -> bool:
    """Sideways / barcode candle — tight range or indecision chop, not a directional leg."""
    rng_pct = _range_pct_of_close(row)
    body_pct = _body_pct_of_range(row)
    if rng_pct is None:
        return False
    if body_pct is not None and body_pct >= SIDEWAYS_CLUSTER_MAX_DIRECTIONAL_BODY_PCT:
        return False
    if rng_pct <= SIDEWAYS_BARCODE_MAX_RANGE_PCT:
        return True
    if (
        rng_pct <= SIDEWAYS_CHOP_MAX_RANGE_PCT
        and body_pct is not None
        and body_pct <= SIDEWAYS_CHOP_MAX_BODY_PCT
    ):
        return True
    return False


def _is_breakout_expansion_bar(row: pd.Series) -> bool:
    """First expansion candle after a sideways cluster — not another consolidation bar."""
    o, c = float(row["open"]), float(row["close"])
    if c <= o:
        return False
    if _is_consolidation_bar(row):
        return False
    body_pct = _body_pct_of_range(row)
    rng_pct = _range_pct_of_close(row)
    if body_pct is not None and body_pct >= BREAKOUT_MIN_BODY_PCT:
        return True
    if rng_pct is not None and rng_pct >= SIDEWAYS_BARCODE_MAX_RANGE_PCT * SIDEWAYS_BREAKOUT_RANGE_MULTIPLIER:
        return True
    return False


def _sideways_cluster_before(
    df: pd.DataFrame,
    idx: int,
) -> tuple[bool, float | None, float | None, int]:
    """Longest consecutive sideways/barcode run immediately before idx."""
    if idx < 1:
        return False, None, None, 0
    count = 0
    j = idx - 1
    while j >= 0 and count < MAX_SIDEWAYS_CLUSTER_LOOKBACK:
        if not _is_consolidation_bar(df.iloc[j]):
            break
        count += 1
        j -= 1
    if count < MIN_SIDEWAYS_CLUSTER_BARS:
        return False, None, None, count
    cluster = df.iloc[idx - count : idx]
    c_high = float(cluster["high"].max())
    c_low = float(cluster["low"].min())
    return True, c_high, c_low, count


def _is_range_candle(row: pd.Series) -> bool:
    """Small-range / indecision bar suitable for a consolidation cluster."""
    rng_pct = _range_pct_of_close(row)
    body_pct = _body_pct_of_range(row)
    if rng_pct is None:
        return False
    if rng_pct <= CLUSTER_BAR_MAX_RANGE_PCT:
        return True
    return body_pct is not None and body_pct <= 35.0


def _stop_loss_and_risk(h: float, l: float, c: float) -> tuple[float, float | None]:
    rng = h - l
    if rng <= 0 or c <= 0:
        return c, None
    sl = c - SL_RANGE_FRACTION * rng
    risk_pct = (c - sl) / c * 100.0 if c > sl else None
    return sl, risk_pct


def _ema_distance_pct(close: float, ema: float) -> float | None:
    if close <= 0 or not pd.notna(ema):
        return None
    return abs(close - ema) / close * 100.0


def _close_near_ema13_high(close: float, ema: float) -> tuple[bool, float | None]:
    if close <= 0 or not pd.notna(ema):
        return False, None
    if close <= ema:
        return False, _ema_distance_pct(close, ema)
    dist = (close - ema) / close * 100.0
    return dist <= MAX_EMA_DISTANCE_PCT_OF_CLOSE, dist


def _close_not_extended_above_ema13_high(close: float, ema: float) -> tuple[bool, float | None]:
    """True when close is at or below EMA, or within MAX % above EMA (doji→strong filter)."""
    if close <= 0 or not pd.notna(ema):
        return True, None
    if close <= ema:
        return True, _ema_distance_pct(close, ema)
    dist = (close - ema) / close * 100.0
    return dist <= MAX_EMA_DISTANCE_PCT_OF_CLOSE, dist


def _volume_above_ema20(row: pd.Series) -> tuple[bool, float | None, float | None]:
    vol = float(row.get("volume", 0) or 0)
    ema_vol = row.get("ema20_volume")
    if vol <= 0 or not pd.notna(ema_vol) or float(ema_vol) <= 0:
        return False, vol if vol > 0 else None, float(ema_vol) if pd.notna(ema_vol) else None
    ema_f = float(ema_vol)
    return vol > ema_f, vol, ema_f


def _cluster_range_before(
    df: pd.DataFrame,
    idx: int,
    n_bars: int,
    *,
    max_width_pct: float | None = None,
) -> tuple[bool, float | None, float | None]:
    """Prior n_bars are range candles; optional max cluster width as % of midpoint."""
    if idx < n_bars:
        return False, None, None
    cluster = df.iloc[idx - n_bars : idx]
    for j in range(len(cluster)):
        if not _is_range_candle(cluster.iloc[j]):
            return False, None, None
    c_high = float(cluster["high"].max())
    c_low = float(cluster["low"].min())
    mid = (c_high + c_low) / 2.0
    if mid <= 0:
        return False, None, None
    if max_width_pct is not None:
        width_pct = (c_high - c_low) / mid * 100.0
        if width_pct > max_width_pct:
            return False, None, None
    return True, c_high, c_low


def _consolidation_reference_price(open_: float, close: float) -> float:
    return open_ if close < open_ else close


def _prior_bars_in_signal_close_low_band(df: pd.DataFrame, idx: int) -> bool:
    if idx < PRIOR_CONSOLIDATION_BARS:
        return False

    sig_low = float(df.iloc[idx]["low"])
    sig_close = float(df.iloc[idx]["close"])
    band_lo = min(sig_low, sig_close)
    band_hi = max(sig_low, sig_close)

    for j in range(idx - PRIOR_CONSOLIDATION_BARS, idx):
        prev = df.iloc[j]
        ref = _consolidation_reference_price(float(prev["open"]), float(prev["close"]))
        if ref < band_lo or ref > band_hi:
            return False
    return True


def _close_above_prior_highs(df: pd.DataFrame, idx: int) -> tuple[bool, float | None]:
    if idx < PRIOR_HIGH_BARS:
        return False, None
    prior_max_high = float(df.iloc[idx - PRIOR_HIGH_BARS : idx]["high"].max())
    close = float(df.iloc[idx]["close"])
    return close > prior_max_high, prior_max_high


def _evaluate_primary_entry(df: pd.DataFrame, idx: int) -> dict[str, Any]:
    row = df.iloc[idx]
    o, h, l, c = float(row["open"]), float(row["high"]), float(row["low"]), float(row["close"])
    ema = float(row["ema13_high"]) if "ema13_high" in row.index else float("nan")

    close_above = c > ema if pd.notna(ema) else False
    close_near_ema, ema_dist_pct = _close_near_ema13_high(c, ema) if pd.notna(ema) else (False, None)
    no_top_wick = _no_top_wick_bullish(row)
    prior_consolidation = _prior_bars_in_signal_close_low_band(df, idx)
    close_above_prior4, prior4_high = _close_above_prior_highs(df, idx)
    sl, risk_pct = _stop_loss_and_risk(h, l, c)

    skip: list[str] = []
    if idx < EMA_PERIOD:
        skip.append(f"EMA{EMA_PERIOD} warming")
    elif not close_above:
        skip.append("close ≤ EMA")
    elif not close_near_ema:
        skip.append(f"EMA gap {ema_dist_pct:.1f}% > {MAX_EMA_DISTANCE_PCT_OF_CLOSE:g}%")
    if idx < PRIOR_CONSOLIDATION_BARS:
        skip.append(f"need {PRIOR_CONSOLIDATION_BARS} prior bars")
    elif not prior_consolidation:
        skip.append("prior 5 not in close–low band")
    if idx < PRIOR_HIGH_BARS:
        skip.append(f"need {PRIOR_HIGH_BARS} prior highs")
    elif not close_above_prior4:
        skip.append("close ≤ prior 4 high")
    if not no_top_wick:
        skip.append("top wick too large")
    if c <= sl:
        skip.append("entry ≤ SL")
    if risk_pct is not None and risk_pct > MAX_RISK_PCT:
        skip.append(f"risk {risk_pct:.1f}%")

    is_signal = (
        idx >= EMA_PERIOD
        and idx >= PRIOR_CONSOLIDATION_BARS
        and idx >= PRIOR_HIGH_BARS
        and close_above
        and close_near_ema
        and no_top_wick
        and prior_consolidation
        and close_above_prior4
        and c > sl
        and risk_pct is not None
        and risk_pct <= MAX_RISK_PCT
    )

    checks = {
        "close_above_ema": close_above if idx >= EMA_PERIOD else False,
        "close_near_ema": close_near_ema if idx >= EMA_PERIOD else False,
        "no_top_wick": no_top_wick,
        "prior_consolidation": prior_consolidation if idx >= PRIOR_CONSOLIDATION_BARS else False,
        "close_above_prior4": close_above_prior4 if idx >= PRIOR_HIGH_BARS else False,
        **_common_risk_checks(c, sl, risk_pct),
    }

    return {
        "open": o,
        "high": h,
        "low": l,
        "close": c,
        "ema13_high": ema if pd.notna(ema) else None,
        "close_above_ema": close_above if idx >= EMA_PERIOD else None,
        "ema_distance_pct": round(ema_dist_pct, 2) if ema_dist_pct is not None else None,
        "close_near_ema": close_near_ema if idx >= EMA_PERIOD else None,
        "no_top_wick": no_top_wick,
        "prior_consolidation": prior_consolidation if idx >= PRIOR_CONSOLIDATION_BARS else None,
        "prior4_high": prior4_high,
        "close_above_prior4": close_above_prior4 if idx >= PRIOR_HIGH_BARS else None,
        "risk_pct": risk_pct,
        "stop_loss": sl,
        "checks": checks,
        "signal": is_signal,
        "skip": skip,
        "status": "SIGNAL (primary)" if is_signal else (" · ".join(skip) if skip else "—"),
    }


def _evaluate_doji_strong_entry(df: pd.DataFrame, idx: int) -> dict[str, Any]:
    row = df.iloc[idx]
    o, h, l, c = float(row["open"]), float(row["high"]), float(row["low"]), float(row["close"])
    ema = float(row["ema13_high"]) if "ema13_high" in row.index else float("nan")
    prior_doji = idx >= 1 and _is_doji(df.iloc[idx - 1])
    strong = _is_strong_bullish(row)
    ema_ok, ema_dist_pct = _close_not_extended_above_ema13_high(c, ema) if pd.notna(ema) else (True, None)
    sl, risk_pct = _stop_loss_and_risk(h, l, c)

    skip: list[str] = []
    if idx < 1:
        skip.append("need prior bar")
    elif not prior_doji:
        skip.append("prior bar not doji")
    if idx >= EMA_PERIOD and pd.notna(ema) and not ema_ok:
        skip.append(f"close > {MAX_EMA_DISTANCE_PCT_OF_CLOSE:g}% above EMA")
    if not strong:
        skip.append("not strong bullish")
    if c <= sl:
        skip.append("entry ≤ SL")
    if risk_pct is not None and risk_pct > MAX_RISK_PCT:
        skip.append(f"risk {risk_pct:.1f}%")

    is_signal = (
        idx >= 1
        and prior_doji
        and strong
        and ema_ok
        and c > sl
        and risk_pct is not None
        and risk_pct <= MAX_RISK_PCT
    )

    checks = {
        "prior_doji": prior_doji if idx >= 1 else False,
        "strong_bullish": strong,
        "ema_not_extended": ema_ok,
        **_common_risk_checks(c, sl, risk_pct),
    }

    return {
        "open": o,
        "high": h,
        "low": l,
        "close": c,
        "prior_doji": prior_doji if idx >= 1 else None,
        "is_doji": _is_doji(row),
        "is_strong": strong,
        "ema_distance_pct": round(ema_dist_pct, 2) if ema_dist_pct is not None else None,
        "ema_not_extended": ema_ok if idx >= EMA_PERIOD and pd.notna(ema) else None,
        "risk_pct": risk_pct,
        "stop_loss": sl,
        "checks": checks,
        "signal": is_signal,
        "skip": skip,
        "status": "SIGNAL (doji→strong)" if is_signal else (" · ".join(skip) if skip else "—"),
    }


def _evaluate_cluster_breakout_entry(df: pd.DataFrame, idx: int) -> dict[str, Any]:
    row = df.iloc[idx]
    o, h, l, c = float(row["open"]), float(row["high"]), float(row["low"]), float(row["close"])
    has_cluster, cluster_high, cluster_low = _cluster_range_before(
        df, idx, CLUSTER_RANGE_BARS, max_width_pct=CLUSTER_MAX_WIDTH_PCT
    )
    clears_cluster = has_cluster and cluster_high is not None and c > cluster_high
    bullish = c > o
    vol_ok, vol, ema_vol = _volume_above_ema20(row)
    sl, risk_pct = _stop_loss_and_risk(h, l, c)

    skip: list[str] = []
    if idx < CLUSTER_RANGE_BARS:
        skip.append(f"need {CLUSTER_RANGE_BARS} prior bars")
    elif not has_cluster:
        skip.append("no range cluster")
    elif not clears_cluster:
        skip.append("close ≤ cluster high")
    if idx < VOLUME_EMA_PERIOD:
        skip.append(f"vol EMA{VOLUME_EMA_PERIOD} warming")
    elif not vol_ok:
        skip.append("volume ≤ EMA20")
    if not bullish:
        skip.append("not bullish")
    if c <= sl:
        skip.append("entry ≤ SL")
    if risk_pct is not None and risk_pct > MAX_RISK_PCT:
        skip.append(f"risk {risk_pct:.1f}%")

    is_signal = (
        idx >= CLUSTER_RANGE_BARS
        and idx >= VOLUME_EMA_PERIOD
        and has_cluster
        and clears_cluster
        and bullish
        and vol_ok
        and c > sl
        and risk_pct is not None
        and risk_pct <= MAX_RISK_PCT
    )

    clears_ok = clears_cluster and bullish
    checks = {
        "range_cluster": has_cluster if idx >= CLUSTER_RANGE_BARS else False,
        "clears_cluster": clears_ok if idx >= CLUSTER_RANGE_BARS else False,
        "volume_above_ema20": vol_ok if idx >= VOLUME_EMA_PERIOD else False,
        **_common_risk_checks(c, sl, risk_pct),
    }

    return {
        "open": o,
        "high": h,
        "low": l,
        "close": c,
        "range_cluster": has_cluster if idx >= CLUSTER_RANGE_BARS else None,
        "cluster_high": round(cluster_high, 2) if cluster_high is not None else None,
        "cluster_low": round(cluster_low, 2) if cluster_low is not None else None,
        "clears_cluster": clears_cluster if idx >= CLUSTER_RANGE_BARS else None,
        "volume": round(vol, 0) if vol is not None else None,
        "ema20_volume": round(ema_vol, 0) if ema_vol is not None else None,
        "volume_above_ema20": vol_ok if idx >= VOLUME_EMA_PERIOD else None,
        "risk_pct": risk_pct,
        "stop_loss": sl,
        "checks": checks,
        "signal": is_signal,
        "skip": skip,
        "status": "SIGNAL (cluster+vol)" if is_signal else (" · ".join(skip) if skip else "—"),
    }


def _evaluate_sideways_breakout_entry(df: pd.DataFrame, idx: int) -> dict[str, Any]:
    row = df.iloc[idx]
    o, h, l, c = float(row["open"]), float(row["high"]), float(row["low"]), float(row["close"])
    has_cluster, cluster_high, cluster_low, cluster_bars = _sideways_cluster_before(df, idx)
    breakout = _is_breakout_expansion_bar(row)
    clears_cluster = has_cluster and cluster_high is not None and c > cluster_high and c > o
    sl, risk_pct = _stop_loss_and_risk(h, l, c)

    skip: list[str] = []
    if idx < MIN_SIDEWAYS_CLUSTER_BARS:
        skip.append(f"need {MIN_SIDEWAYS_CLUSTER_BARS}+ prior bars")
    elif not has_cluster:
        skip.append(f"no sideways cluster ({cluster_bars} bars)")
    elif not breakout:
        skip.append("not expansion breakout")
    elif not clears_cluster:
        skip.append("close ≤ cluster high")
    if c <= sl:
        skip.append("entry ≤ SL")
    if risk_pct is not None and risk_pct > MAX_RISK_PCT:
        skip.append(f"risk {risk_pct:.1f}%")

    is_signal = (
        has_cluster
        and breakout
        and clears_cluster
        and c > sl
        and risk_pct is not None
        and risk_pct <= MAX_RISK_PCT
    )

    checks = {
        "sideways_cluster": has_cluster,
        "breakout_expansion": breakout,
        "clears_sideways_cluster": clears_cluster if has_cluster else False,
        **_common_risk_checks(c, sl, risk_pct),
    }

    return {
        "open": o,
        "high": h,
        "low": l,
        "close": c,
        "sideways_cluster": has_cluster if idx >= 1 else None,
        "sideways_cluster_bars": cluster_bars if idx >= 1 else None,
        "sideways_cluster_high": round(cluster_high, 2) if cluster_high is not None else None,
        "sideways_cluster_low": round(cluster_low, 2) if cluster_low is not None else None,
        "breakout_expansion": breakout,
        "clears_sideways_cluster": clears_cluster if has_cluster else None,
        "risk_pct": risk_pct,
        "stop_loss": sl,
        "checks": checks,
        "signal": is_signal,
        "skip": skip,
        "status": (
            f"SIGNAL (sideways breakout, {cluster_bars} bars)"
            if is_signal
            else (" · ".join(skip) if skip else "—")
        ),
    }


def _pick_signal_type(
    primary: dict,
    doji: dict,
    cluster: dict,
    sideways: dict,
) -> tuple[bool, str | None, str, dict]:
    if primary["signal"]:
        return True, "primary", primary["status"], primary
    if cluster["signal"]:
        return True, "cluster_breakout", cluster["status"], cluster
    if sideways["signal"]:
        return True, "sideways_breakout", sideways["status"], sideways
    if doji["signal"]:
        return True, "doji_strong", doji["status"], doji
    status = primary["status"] if primary["skip"] else (
        cluster["status"] if cluster["skip"] else (
            sideways["status"] if sideways["skip"] else doji["status"]
        )
    )
    return False, None, status, primary


def _evaluate_entry_bar(df: pd.DataFrame, idx: int) -> dict[str, Any]:
    primary = _evaluate_primary_entry(df, idx)
    doji = _evaluate_doji_strong_entry(df, idx)
    cluster = _evaluate_cluster_breakout_entry(df, idx)
    sideways = _evaluate_sideways_breakout_entry(df, idx)

    is_signal, signal_type, status, active = _pick_signal_type(primary, doji, cluster, sideways)

    return {
        **primary,
        "prior_doji": doji.get("prior_doji"),
        "is_doji": doji.get("is_doji"),
        "is_strong": doji.get("is_strong"),
        "ema_not_extended": doji.get("ema_not_extended"),
        "range_cluster": cluster.get("range_cluster"),
        "cluster_high": cluster.get("cluster_high"),
        "cluster_low": cluster.get("cluster_low"),
        "clears_cluster": cluster.get("clears_cluster"),
        "volume": cluster.get("volume"),
        "ema20_volume": cluster.get("ema20_volume"),
        "volume_above_ema20": cluster.get("volume_above_ema20"),
        "sideways_cluster": sideways.get("sideways_cluster"),
        "sideways_cluster_bars": sideways.get("sideways_cluster_bars"),
        "sideways_cluster_high": sideways.get("sideways_cluster_high"),
        "breakout_expansion": sideways.get("breakout_expansion"),
        "clears_sideways_cluster": sideways.get("clears_sideways_cluster"),
        "signal": is_signal,
        "signal_type": signal_type,
        "primary_signal": primary["signal"],
        "doji_strong_signal": doji["signal"],
        "cluster_breakout_signal": cluster["signal"],
        "sideways_breakout_signal": sideways["signal"],
        "condition_checks": {
            "primary": primary.get("checks", {}),
            "doji_strong": doji.get("checks", {}),
            "cluster_breakout": cluster.get("checks", {}),
            "sideways_breakout": sideways.get("checks", {}),
        },
        "stop_loss": active["stop_loss"],
        "risk_pct": active["risk_pct"],
        "status": status,
    }


def _analyze_candles(candles: pd.DataFrame, option_type: str) -> list[dict[str, Any]]:
    if candles.empty:
        return []

    df = _prepare_indicator_df(candles)
    rows: list[dict[str, Any]] = []

    for i in range(len(df)):
        row = df.iloc[i]
        ts = row["timestamp"]
        ts_str = ts.isoformat() if hasattr(ts, "isoformat") else str(ts)
        ev = _evaluate_entry_bar(df, i)

        rows.append(
            {
                "timestamp": ts_str,
                "option_type": option_type,
                "tradingsymbol": str(row.get("tradingsymbol", "")),
                "strike": int(row.get("atm_strike", 0) or 0),
                "open": round(ev["open"], 2),
                "high": round(ev["high"], 2),
                "low": round(ev["low"], 2),
                "close": round(ev["close"], 2),
                "ema13_high": round(ev["ema13_high"], 2) if ev["ema13_high"] is not None else None,
                "close_above_ema": ev["close_above_ema"],
                "ema_distance_pct": ev.get("ema_distance_pct"),
                "close_near_ema": ev.get("close_near_ema"),
                "no_top_wick": ev["no_top_wick"],
                "prior_consolidation": ev["prior_consolidation"],
                "prior4_high": round(ev["prior4_high"], 2) if ev["prior4_high"] is not None else None,
                "close_above_prior4": ev["close_above_prior4"],
                "prior_doji": ev["prior_doji"],
                "is_doji": ev["is_doji"],
                "is_strong": ev["is_strong"],
                "range_cluster": ev.get("range_cluster"),
                "clears_cluster": ev.get("clears_cluster"),
                "cluster_high": ev.get("cluster_high"),
                "volume_above_ema20": ev.get("volume_above_ema20"),
                "sideways_cluster": ev.get("sideways_cluster"),
                "sideways_cluster_bars": ev.get("sideways_cluster_bars"),
                "sideways_cluster_high": ev.get("sideways_cluster_high"),
                "breakout_expansion": ev.get("breakout_expansion"),
                "clears_sideways_cluster": ev.get("clears_sideways_cluster"),
                "ema_not_extended": ev.get("ema_not_extended"),
                "marubozu": ev["no_top_wick"],
                "risk_pct": round(ev["risk_pct"], 2) if ev["risk_pct"] is not None else None,
                "signal": ev["signal"],
                "signal_type": ev.get("signal_type"),
                "primary_signal": ev.get("primary_signal"),
                "doji_strong_signal": ev.get("doji_strong_signal"),
                "cluster_breakout_signal": ev.get("cluster_breakout_signal"),
                "sideways_breakout_signal": ev.get("sideways_breakout_signal"),
                "condition_checks": ev.get("condition_checks", {}),
                "stop_loss": round(ev["stop_loss"], 2) if ev.get("stop_loss") is not None else None,
                "status": ev["status"],
            }
        )

    return list(reversed(rows))


def _scan_candles(candles: pd.DataFrame, option_type: str, underlying: str) -> list[dict[str, Any]]:
    if len(candles) < 2:
        return []

    df = _prepare_indicator_df(candles)
    signals: list[dict[str, Any]] = []

    evaluators: list[tuple[str, Any, str, int]] = [
        (
            "primary",
            _evaluate_primary_entry,
            (
                f"{option_type} 5m close above EMA{EMA_PERIOD}(high) within "
                f"{MAX_EMA_DISTANCE_PCT_OF_CLOSE:g}% of close, top wick < {TOP_WICK_MAX_PCT_OF_HIGH:g}%, "
                f"prior {PRIOR_CONSOLIDATION_BARS} in close–low band, close above prior {PRIOR_HIGH_BARS} highs, "
                f"SL at {SL_RANGE_FRACTION:.0%} of range below close"
            ),
            EMA_PERIOD,
        ),
        (
            "doji_strong",
            _evaluate_doji_strong_entry,
            (
                f"{option_type} 5m doji then strong bullish candle; "
                f"skip if close > {MAX_EMA_DISTANCE_PCT_OF_CLOSE:g}% above EMA{EMA_PERIOD}(high); "
                f"SL at {SL_RANGE_FRACTION:.0%} of range below close"
            ),
            1,
        ),
        (
            "cluster_breakout",
            _evaluate_cluster_breakout_entry,
            (
                f"{option_type} 5m clears {CLUSTER_RANGE_BARS}-bar range cluster "
                f"(width ≤ {CLUSTER_MAX_WIDTH_PCT:g}%) on volume > EMA{VOLUME_EMA_PERIOD}, "
                f"SL at {SL_RANGE_FRACTION:.0%} of range below close"
            ),
            VOLUME_EMA_PERIOD,
        ),
        (
            "sideways_breakout",
            _evaluate_sideways_breakout_entry,
            (
                f"{option_type} 5m first expansion candle after ≥{MIN_SIDEWAYS_CLUSTER_BARS} "
                f"sideways/barcode bars, clears cluster high, "
                f"SL at {SL_RANGE_FRACTION:.0%} of range below close"
            ),
            MIN_SIDEWAYS_CLUSTER_BARS,
        ),
    ]

    for i in range(1, len(df)):
        row = df.iloc[i]
        for signal_type, fn, reason, min_idx in evaluators:
            if i < min_idx:
                continue
            ev = fn(df, i)
            if not ev["signal"]:
                continue
            close = ev["close"]
            ema_raw = row.get("ema13_high")
            ema_val = float(ema_raw) if pd.notna(ema_raw) else None
            ts = row["timestamp"]
            ts_str = ts.isoformat() if hasattr(ts, "isoformat") else str(ts)
            signals.append(
                {
                    "underlying": underlying,
                    "option_type": option_type,
                    "signal_type": signal_type,
                    "side": "BUY",
                    "tradingsymbol": str(row.get("tradingsymbol", "")),
                    "strike": int(row.get("atm_strike", 0) or 0),
                    "spot_price": float(row.get("spot_price", 0) or 0),
                    "timestamp": ts_str,
                    "entry": round(close, 2),
                    "stop_loss": round(ev["stop_loss"], 2),
                    "risk_pct": round(ev["risk_pct"], 2),
                    "ema13_high": round(ema_val, 2) if ema_val is not None else None,
                    "cluster_high": ev.get("cluster_high") or ev.get("sideways_cluster_high"),
                    "volume": ev.get("volume"),
                    "ema20_volume": ev.get("ema20_volume"),
                    "candle": {
                        "open": round(ev["open"], 2),
                        "high": round(ev["high"], 2),
                        "low": round(ev["low"], 2),
                        "close": round(close, 2),
                    },
                    "reason": reason,
                }
            )
    return signals


def scan_options_trading_signals(
    data_dir: Path,
    trade_date: date,
    underlying: str = "NIFTY",
) -> dict[str, Any]:
    u = normalize_underlying(underlying)
    analysis_path = analysis_day_path(data_dir, trade_date, asset_class_for_underlying(u))
    df = _load_index_analysis(data_dir, trade_date, u)

    if df.empty:
        return {
            "underlying": u,
            "trade_date": trade_date.isoformat(),
            "signals": [],
            "message": (
                f"No Index_Analysis data for {u} on {trade_date.isoformat()}. "
                "Click Refresh on the index dashboard for that date."
            ),
            "bars": 0,
            "data_source": str(analysis_path),
        }

    bars = len(df)
    all_signals: list[dict[str, Any]] = []
    ohlc_sources: dict[str, str] = {}
    analysis: dict[str, list[dict[str, Any]]] = {"CE": [], "PE": []}

    for opt_type in ("CE", "PE"):
        candles, source = _option_candles_from_analysis(
            df,
            opt_type,
            trade_date=trade_date,
            underlying=u,
        )
        ohlc_sources[opt_type] = source
        analysis[opt_type] = _analyze_candles(candles, opt_type)
        all_signals.extend(_scan_candles(candles, opt_type, u))

    all_signals.sort(key=lambda s: s.get("timestamp", ""), reverse=True)

    message = None
    if not all_signals:
        if all(src == "none" for src in ohlc_sources.values()):
            message = (
                "No option OHLC available. Click Refresh on the index dashboard "
                "to export call/put OHLC, or ensure Kite credentials are valid."
            )
        else:
            message = "No entry signals yet for current rules."

    return {
        "underlying": u,
        "trade_date": trade_date.isoformat(),
        "bars": bars,
        "data_source": str(analysis_path),
        "ohlc_sources": ohlc_sources,
        "analysis": analysis,
        "signals": all_signals,
        "signal_types": options_signal_definitions(),
        "latest": all_signals[0] if all_signals else None,
        "run_at": datetime.now().isoformat(),
        "message": message,
    }


def save_options_trading_signals(data_dir: Path, payload: dict[str, Any]) -> Path:
    td = datetime.strptime(str(payload["trade_date"]), "%Y-%m-%d").date()
    path = options_trading_signals_path(data_dir, td)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    return path


def load_options_trading_signals(data_dir: Path, trade_date: date, underlying: str) -> dict[str, Any] | None:
    path = options_trading_signals_path(data_dir, trade_date)
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("underlying") != normalize_underlying(underlying):
            return None
        return payload
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("Could not load options trading signals: %s", exc)
        return None


def run_options_trading_scan(
    data_dir: Path,
    trade_date: date | None = None,
    underlying: str = "NIFTY",
) -> dict[str, Any]:
    td = trade_date or date.today()
    payload = scan_options_trading_signals(data_dir, td, underlying)
    save_options_trading_signals(data_dir, payload)
    return payload
