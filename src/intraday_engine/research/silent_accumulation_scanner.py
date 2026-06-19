"""
Silent institutional accumulation scanner (daily bars).

Detects the OHLCV + delivery footprint of stealthy buying inside flat/range-bound
price action — the classic Wyckoff Phase B/C signature. Per-stock data only;
real-time per-symbol FII/DII flow is not public, so we infer from:

  • OBV slope while price is flat   (money flowing in without price up)
  • Chaikin Money Flow (CMF) > 0    (close in upper part of bar on volume)
  • Up/down volume ratio > 1.3      (effort vs result)
  • Close-in-bar position           (closes in upper third more often)
  • Higher-lows + same ceiling      (springs absorbed)
  • Range tightness                 (compression)
  • Rising delivery %               (real holding, not jobbing) [optional]
  • Bulk-deal buyer = FII/MF/Insurance, no equivalent seller    [optional]

Each signal contributes to a weighted Silent Accumulation Score (0-100) with an
explainable per-component breakdown. Uses the existing daily-bar cache under
data/NIFTY500/SYMBOL_1D.csv (written by the tomorrow-watchlist pipeline).

Run:  python -m intraday_engine.main --silent-accumulation
"""

from __future__ import annotations

import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from intraday_engine.core.config import Settings
from intraday_engine.fetch.nse_public_data import (
    fetch_delivery_bhavcopy,
    get_recent_bulk_deals,
)
from intraday_engine.research.nifty500_accumulation_scanner import load_nifty500_symbols
from intraday_engine.storage.layout import silent_accumulation_path
from intraday_engine.storage.nifty500_csv import read_nifty500_symbol_ohlcv

logger = logging.getLogger(__name__)


LOOKBACK_DAYS = 40
MIN_BARS = 60
RANGE_MAX_WIDTH_PCT = 12.0
MAX_RECENT_MOVE_PCT = 8.0  # avoid stocks already running

INSTITUTIONAL_KEYWORDS = (
    "FII", "FPI", "MUTUAL FUND", "MF", "INSURANCE", "LIC",
    "GOLDMAN", "MORGAN", "BLACKROCK", "VANGUARD", "NORGES",
    "GOVERNMENT", "PENSION", "SOCIETE", "CITIGROUP", "BNP", "JP MORGAN",
    "ABU DHABI", "GMO", "TROWE", "FIDELITY", "FRANKLIN",
    "HDFC MUTUAL", "SBI MUTUAL", "ICICI PRUDENTIAL", "NIPPON",
    "AXIS MUTUAL", "KOTAK MAHINDRA MF", "DSP", "ADITYA BIRLA SUN",
    "INVESCO", "MIRAE", "TATA MUTUAL", "UTI",
)


@dataclass
class SilentAccumulationRow:
    stock: str
    score: float
    current_price: float
    range_high: float
    range_low: float
    range_width_pct: float
    obv_slope_pct: float
    cmf: float
    up_down_vol_ratio: float
    close_upper_third_pct: float
    higher_lows: bool
    range_tight: bool
    avg_delivery_pct: float | None
    delivery_trend: str  # "rising" / "flat" / "falling" / "n/a"
    institutional_bulk_buys: int
    institutional_bulk_sells: int
    components: dict[str, float] = field(default_factory=dict)
    reasons: list[str] = field(default_factory=list)


def _load_daily(data_dir: Path, symbol: str) -> pd.DataFrame:
    df = read_nifty500_symbol_ohlcv(data_dir, symbol, "1D")
    if df.empty:
        return df
    if "date" in df.columns:
        df = df.dropna(subset=["date", "open", "high", "low", "close", "volume"])
    return df.tail(220).reset_index(drop=True)


def _obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    sign = np.sign(close.diff().fillna(0.0))
    return (sign * volume.fillna(0.0)).cumsum()


def _cmf(df: pd.DataFrame, period: int = 20) -> pd.Series:
    h, l, c, v = df["high"], df["low"], df["close"], df["volume"]
    rng = (h - l).replace(0, np.nan)
    mfm = ((c - l) - (h - c)) / rng
    mfv = (mfm * v).fillna(0.0)
    return mfv.rolling(period, min_periods=period).sum() / v.rolling(period, min_periods=period).sum()


def _up_down_volume_ratio(df: pd.DataFrame, lookback: int = 20) -> float:
    tail = df.tail(lookback)
    if len(tail) < lookback:
        return float("nan")
    diff = tail["close"].diff()
    up_v = float(tail.loc[diff > 0, "volume"].sum())
    dn_v = float(tail.loc[diff < 0, "volume"].sum())
    if dn_v <= 0:
        return float("inf") if up_v > 0 else 0.0
    return up_v / dn_v


def _close_position_pct(df: pd.DataFrame, lookback: int = 20, threshold: float = 2 / 3) -> float:
    """% of last N bars where close is in upper third of the bar."""
    tail = df.tail(lookback)
    rng = (tail["high"] - tail["low"]).replace(0, np.nan)
    pos = ((tail["close"] - tail["low"]) / rng).fillna(0.5)
    return float((pos >= threshold).mean()) * 100.0


def _higher_lows(df: pd.DataFrame, lookback: int = 30) -> bool:
    tail = df.tail(lookback).reset_index(drop=True)
    if len(tail) < 12:
        return False
    swings: list[float] = []
    lows = tail["low"].values
    for i in range(2, len(lows) - 2):
        if lows[i] <= lows[i - 1] and lows[i] <= lows[i - 2] and lows[i] <= lows[i + 1] and lows[i] <= lows[i + 2]:
            swings.append(float(lows[i]))
    if len(swings) < 3:
        return False
    last3 = swings[-3:]
    return last3[0] < last3[1] < last3[2] or (last3[2] > last3[0] * 1.001)


def _range_metrics(df: pd.DataFrame, lookback: int = 30) -> tuple[float, float, float]:
    tail = df.tail(lookback)
    rh = float(tail["high"].max())
    rl = float(tail["low"].min())
    mid = (rh + rl) / 2.0
    width_pct = (rh - rl) / mid * 100.0 if mid > 0 else 100.0
    return rh, rl, width_pct


def _obv_slope_pct(df: pd.DataFrame, lookback: int = 20) -> float:
    """Net OBV change over the lookback as a percentage of total volume traded in the
    same window. Range roughly [-100, +100]. Survives OBV sign-flips because the
    denominator is total volume (always positive), not OBV itself."""
    if len(df) < lookback + 1:
        return 0.0
    tail = df.tail(lookback + 1).copy()
    sign = np.sign(tail["close"].diff().fillna(0.0))
    signed_vol = (sign * tail["volume"]).iloc[1:]
    total_vol = float(tail["volume"].iloc[1:].sum())
    if total_vol <= 0:
        return 0.0
    return float(signed_vol.sum() / total_vol) * 100.0


def _delivery_trend(deliv_series: list[float]) -> str:
    if not deliv_series or len(deliv_series) < 6:
        return "n/a"
    arr = np.array(deliv_series, dtype=float)
    arr = arr[~np.isnan(arr)]
    if len(arr) < 6:
        return "n/a"
    half = len(arr) // 2
    first = float(np.mean(arr[:half]))
    second = float(np.mean(arr[half:]))
    if first <= 0:
        return "n/a"
    delta = (second - first) / first
    if delta > 0.05:
        return "rising"
    if delta < -0.05:
        return "falling"
    return "flat"


def _is_institutional(client_name: str) -> bool:
    s = (client_name or "").upper()
    return any(k in s for k in INSTITUTIONAL_KEYWORDS)


def _score_row(
    df: pd.DataFrame,
    delivery_per_history: list[float],
    institutional_buy_count: int,
    institutional_sell_count: int,
    symbol: str,
) -> SilentAccumulationRow | None:
    if len(df) < MIN_BARS:
        return None

    rh, rl, width_pct = _range_metrics(df, lookback=LOOKBACK_DAYS)
    if width_pct > RANGE_MAX_WIDTH_PCT:
        return None

    last_close = float(df["close"].iloc[-1])
    if last_close <= 0:
        return None

    # exclude names already running
    five_close_pct = (last_close / float(df["close"].iloc[-min(11, len(df))]) - 1.0) * 100.0
    if abs(five_close_pct) > MAX_RECENT_MOVE_PCT:
        return None

    obv_slope = _obv_slope_pct(df, lookback=LOOKBACK_DAYS)
    cmf_series = _cmf(df, period=20).dropna()
    cmf_val = float(cmf_series.iloc[-1]) if not cmf_series.empty else 0.0
    udr = _up_down_volume_ratio(df, lookback=20)
    upper_pct = _close_position_pct(df, lookback=20)
    hl = _higher_lows(df, lookback=LOOKBACK_DAYS)
    range_tight = width_pct <= 7.0

    avg_deliv = float(np.nanmean(delivery_per_history)) if delivery_per_history else None
    deliv_trend = _delivery_trend(delivery_per_history)

    components: dict[str, float] = {}
    reasons: list[str] = []

    # 1) OBV slope: 0 → +25 if slope >= 5%
    obv_pts = max(0.0, min(25.0, obv_slope * 5.0))
    components["obv_slope"] = round(obv_pts, 2)
    if obv_slope > 1.0:
        reasons.append(f"OBV +{obv_slope:.1f}% over {LOOKBACK_DAYS}d while price flat")

    # 2) CMF: positive 0..0.2 maps 0..20
    cmf_pts = max(0.0, min(20.0, cmf_val * 100.0))
    components["cmf"] = round(cmf_pts, 2)
    if cmf_val > 0.05:
        reasons.append(f"CMF +{cmf_val:.2f} (volume on closes near high)")

    # 3) Up/down volume ratio: 1.0..2.5 → 0..15
    if np.isfinite(udr):
        udr_pts = max(0.0, min(15.0, (udr - 1.0) * 10.0))
    else:
        udr_pts = 15.0
    components["up_down_vol"] = round(udr_pts, 2)
    if udr > 1.3:
        reasons.append(f"Up-day vol / down-day vol = {udr:.2f}x")

    # 4) Close in upper third %: 0..10
    upper_pts = max(0.0, min(10.0, (upper_pct - 50.0) / 5.0))
    components["close_upper_third"] = round(upper_pts, 2)
    if upper_pct >= 60.0:
        reasons.append(f"Closes in upper third {upper_pct:.0f}% of last 20 days")

    # 5) Higher lows + range tight: 0/5/10
    structure_pts = (5.0 if hl else 0.0) + (5.0 if range_tight else 0.0)
    components["structure"] = structure_pts
    if hl:
        reasons.append("Higher swing lows under same ceiling (springs absorbed)")
    if range_tight:
        reasons.append(f"Range width {width_pct:.1f}% — compression")

    # 6) Delivery %: 0..15
    delivery_pts = 0.0
    if avg_deliv is not None and not np.isnan(avg_deliv):
        delivery_pts = max(0.0, min(10.0, (avg_deliv - 40.0) / 4.0))
        if deliv_trend == "rising":
            delivery_pts += 5.0
        delivery_pts = min(15.0, delivery_pts)
        reasons.append(
            f"Avg delivery {avg_deliv:.0f}%"
            + (f" — {deliv_trend}" if deliv_trend != "n/a" else "")
        )
    components["delivery"] = round(delivery_pts, 2)

    # 7) Institutional bulk-deal buys (vs sells): 0..15
    inst_pts = 0.0
    if institutional_buy_count > 0 and institutional_sell_count == 0:
        inst_pts = 15.0
        reasons.append(f"{institutional_buy_count} bulk-deal buy(s) by FII/MF, no offsetting sells")
    elif institutional_buy_count > institutional_sell_count:
        inst_pts = 8.0
        reasons.append(
            f"Bulk deals net positive: {institutional_buy_count} buys vs {institutional_sell_count} sells"
        )
    components["institutional_deals"] = inst_pts

    score = sum(components.values())

    return SilentAccumulationRow(
        stock=symbol,
        score=round(score, 1),
        current_price=round(last_close, 2),
        range_high=round(rh, 2),
        range_low=round(rl, 2),
        range_width_pct=round(width_pct, 2),
        obv_slope_pct=round(obv_slope, 2),
        cmf=round(cmf_val, 4),
        up_down_vol_ratio=round(udr, 2) if np.isfinite(udr) else 0.0,
        close_upper_third_pct=round(upper_pct, 1),
        higher_lows=hl,
        range_tight=range_tight,
        avg_delivery_pct=round(avg_deliv, 1) if avg_deliv is not None and not np.isnan(avg_deliv) else None,
        delivery_trend=deliv_trend,
        institutional_bulk_buys=institutional_buy_count,
        institutional_bulk_sells=institutional_sell_count,
        components=components,
        reasons=reasons,
    )


def _build_delivery_history(data_dir: Path, days: int = 30) -> dict[str, list[float]]:
    """Pull last N business days of delivery bhavcopy and pivot to {SYMBOL: [DELIV_PER..]}.
    Returns empty dict on total failure (NSE block etc.)."""
    from datetime import timedelta as _td

    today = date.today()
    out: dict[str, list[float]] = {}
    fetched = 0
    look = 0
    while fetched < days and look < days * 2 + 6:
        d = today - _td(days=look)
        look += 1
        if d.weekday() >= 5:
            continue
        df = fetch_delivery_bhavcopy(data_dir, d)
        if df.empty or "DELIV_PER" not in df.columns or "SYMBOL" not in df.columns:
            continue
        for _, r in df.iterrows():
            s = str(r.get("SYMBOL", "")).strip().upper()
            if not s:
                continue
            out.setdefault(s, []).append(float(r.get("DELIV_PER")) if pd.notna(r.get("DELIV_PER")) else float("nan"))
        fetched += 1
    return out


def _build_bulk_deal_index(deals: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    """Returns {SYMBOL: {'inst_buy': int, 'inst_sell': int}}."""
    out: dict[str, dict[str, int]] = {}
    for d in deals or []:
        if not isinstance(d, dict):
            continue
        sym_raw = d.get("symbol") or d.get("BD_SYMBOL") or d.get("Symbol") or ""
        client_raw = d.get("clientName") or d.get("BD_CLIENT_NAME") or d.get("Client Name") or ""
        side_raw = (d.get("buySell") or d.get("BD_BUY_SELL") or d.get("Buy/Sell") or "").upper()
        sym = str(sym_raw).strip().upper()
        if not sym:
            continue
        if not _is_institutional(str(client_raw)):
            continue
        rec = out.setdefault(sym, {"inst_buy": 0, "inst_sell": 0})
        if side_raw.startswith("B"):
            rec["inst_buy"] += 1
        elif side_raw.startswith("S"):
            rec["inst_sell"] += 1
    return out


def run_silent_accumulation_scan(
    *,
    settings: Settings | None = None,
    symbols_file: Path | None = None,
    top_n: int = 25,
    max_workers: int = 8,
    use_nse_data: bool = True,
    trade_date: date | None = None,
) -> dict[str, Any]:
    """Run the scan against the existing data/NIFTY500/SYMBOL_1D.csv cache.
    Outputs a JSON ranked list under data/analysis/silent_accumulation/."""
    settings = settings or Settings.from_env(underlying="NIFTY")
    td = trade_date or date.today()

    delivery_index: dict[str, list[float]] = {}
    bulk_index: dict[str, dict[str, int]] = {}
    nse_status = "skipped"
    if use_nse_data:
        try:
            delivery_index = _build_delivery_history(settings.data_dir, days=20)
            deals = get_recent_bulk_deals(settings.data_dir, days=45)
            bulk_index = _build_bulk_deal_index(deals)
            if delivery_index or bulk_index:
                nse_status = "ok"
            else:
                nse_status = "blocked"
        except Exception as e:
            logger.warning("NSE public data unavailable: %s", e)
            nse_status = "error"

    symbols = load_nifty500_symbols(symbols_file, settings.data_dir)

    scored: list[SilentAccumulationRow] = []
    skipped_no_bars = 0
    failed = 0

    def job(sym: str) -> SilentAccumulationRow | None:
        try:
            df = _load_daily(settings.data_dir, sym)
            if df.empty:
                return None
            deliv = delivery_index.get(sym, [])
            inst = bulk_index.get(sym, {"inst_buy": 0, "inst_sell": 0})
            return _score_row(df, deliv, inst.get("inst_buy", 0), inst.get("inst_sell", 0), sym)
        except Exception as e:
            logger.debug("Silent accumulation %s err: %s", sym, e)
            return None

    with ThreadPoolExecutor(max_workers=max(1, max_workers)) as ex:
        futs = {ex.submit(job, s): s for s in symbols}
        for fut in as_completed(futs):
            try:
                row = fut.result()
            except Exception:
                failed += 1
                continue
            if row is None:
                skipped_no_bars += 1
                continue
            scored.append(row)

    scored.sort(key=lambda r: r.score, reverse=True)
    top = scored[:top_n]

    payload = {
        "trade_date": td.isoformat(),
        "generated_at": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
        "scanned": len(symbols),
        "passed": len(scored),
        "skipped_no_bars": skipped_no_bars,
        "failed": failed,
        "nse_data_status": nse_status,
        "rows": [asdict(r) for r in top],
    }

    out_path = silent_accumulation_path(settings.data_dir, td)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    logger.info("Silent accumulation: %d passed of %d. Saved %s", len(scored), len(symbols), out_path)
    return payload


def load_stored_silent_accumulation(data_dir: Path, trade_date: date) -> dict[str, Any] | None:
    p = silent_accumulation_path(data_dir, trade_date)
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            return None
    # fall back to most recent file
    folder = p.parent
    if not folder.exists():
        return None
    files = sorted(folder.glob("silent_accumulation_*.json"))
    if not files:
        return None
    try:
        return json.loads(files[-1].read_text(encoding="utf-8"))
    except Exception:
        return None
