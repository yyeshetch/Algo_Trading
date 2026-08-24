"""Futures OI buildup analyzer — bar-by-bar Long/Short buildup classification.

Reads persisted 5-min index futures OHLC + OI from
``data/analysis/date=<td>/Index_Analysis.csv`` (already populated by the session
scheduler via ``MarketDataFetcher`` with ``oi=True``), classifies each 5-min
bar's price+OI move using the standard futures convention, and returns a
per-bar timeline + session summary suitable for a dashboard strip renderer.

Classification (labels reflect the position *on the underlying* — the same
convention every Indian broker uses for futures):

    Price ↑ + OI ↑  → LONG_BUILD   (fresh longs, strongest bullish continuation)
    Price ↓ + OI ↑  → SHORT_BUILD  (fresh shorts, strongest bearish continuation)
    Price ↓ + OI ↓  → LONG_UNWIND  (longs booking / cutting; weak bearish)
    Price ↑ + OI ↓  → SHORT_COVER  (shorts squeezed; weak bullish)
    else            → FLAT
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd

from intraday_engine.storage.data_store import load_market_data

# Bar-over-bar thresholds tuned for index futures 5-min bars:
# - Price 0.04% is the same as the options spot threshold — filters micro-noise.
# - OI 0.10% catches meaningful positioning while ignoring end-of-bar rounding
#   and clearing-side reconciliations. On a NIFTY nearest-expiry contract with
#   ~1.2 Cr OI, 0.10% ≈ 12k contracts moved in 5 min — the noise floor.
PRICE_INTERVAL_PCT_THRESH = 0.04
OI_INTERVAL_PCT_THRESH = 0.10

BUILDUP_LABELS: dict[str, str] = {
    "LONG_BUILD": "Long buildup",
    "SHORT_BUILD": "Short buildup",
    "LONG_UNWIND": "Long unwinding",
    "SHORT_COVER": "Short covering",
    "FLAT": "Flat",
}

BUILDUP_BIAS: dict[str, str] = {
    "LONG_BUILD": "bullish",
    "SHORT_BUILD": "bearish",
    "LONG_UNWIND": "bearish",
    "SHORT_COVER": "bullish",
    "FLAT": "neutral",
}


@dataclass
class FuturesOiBuildupResult:
    available: bool
    underlying: str
    trade_date: str
    future_symbol: str | None
    bars_count: int
    headline: str
    reasons: list[str] = field(default_factory=list)
    timestamps: list[str] = field(default_factory=list)
    bars: list[dict[str, Any]] = field(default_factory=list)
    session: dict[str, Any] = field(default_factory=dict)
    thresholds: dict[str, float] = field(default_factory=dict)
    buildup_labels: dict[str, str] = field(default_factory=dict)


def _pct_change(cur: float, prev: float) -> float | None:
    if prev is None or cur is None or prev == 0:
        return None
    return round((cur / prev - 1.0) * 100.0, 3)


def _classify_bar(
    price_chg_pct: float | None,
    oi_chg_pct: float | None,
) -> str:
    if price_chg_pct is None or oi_chg_pct is None:
        return "FLAT"
    price_up = price_chg_pct > PRICE_INTERVAL_PCT_THRESH
    price_down = price_chg_pct < -PRICE_INTERVAL_PCT_THRESH
    oi_up = oi_chg_pct > OI_INTERVAL_PCT_THRESH
    oi_down = oi_chg_pct < -OI_INTERVAL_PCT_THRESH
    if price_up and oi_up:
        return "LONG_BUILD"
    if price_down and oi_up:
        return "SHORT_BUILD"
    if price_down and oi_down:
        return "LONG_UNWIND"
    if price_up and oi_down:
        return "SHORT_COVER"
    return "FLAT"


def _hhmm(ts: str) -> str:
    if not ts:
        return "—"
    if "T" in ts:
        try:
            return ts.split("T", 1)[1][:5]
        except Exception:
            return ts[:5]
    return ts[:5]


def _empty_result(
    underlying: str,
    trade_date: date,
    message: str,
) -> dict[str, Any]:
    return asdict(FuturesOiBuildupResult(
        available=False,
        underlying=underlying,
        trade_date=trade_date.isoformat(),
        future_symbol=None,
        bars_count=0,
        headline=message,
        reasons=[message],
        thresholds={
            "price_pct": PRICE_INTERVAL_PCT_THRESH,
            "oi_pct": OI_INTERVAL_PCT_THRESH,
        },
        buildup_labels=BUILDUP_LABELS,
    ))


def compute_futures_oi_buildup(
    data_dir: Path,
    trade_date: date | None = None,
    underlying: str = "NIFTY",
) -> dict[str, Any]:
    """Compute per-5-min futures OI buildup timeline for one underlying."""
    td = trade_date or date.today()
    u = (underlying or "NIFTY").upper()
    df = load_market_data(data_dir, td, underlying=u)
    if df is None or df.empty:
        return _empty_result(u, td, f"No Index_Analysis.csv rows for {u} on {td.isoformat()}.")

    needed = {"timestamp", "future_close", "future_oi"}
    missing = needed - set(df.columns)
    if missing:
        return _empty_result(u, td, f"Missing columns in Index_Analysis.csv: {sorted(missing)}")

    df = df.sort_values("timestamp").reset_index(drop=True)
    df = df.dropna(subset=["future_close", "future_oi"])
    df = df[df["future_oi"] > 0]
    if len(df) < 2:
        return _empty_result(u, td, "Need 2+ 5-min bars with future OI to classify buildup.")

    fut_symbol_series = df.get("future_symbol") if "future_symbol" in df.columns else None
    future_symbol = None
    if fut_symbol_series is not None and not fut_symbol_series.dropna().empty:
        future_symbol = str(fut_symbol_series.dropna().iloc[-1])

    # If the futures contract changed mid-session (expiry rollover), analyze
    # only the tail with the latest contract — OI levels across contracts are
    # not comparable.
    if future_symbol and "future_symbol" in df.columns:
        df = df[df["future_symbol"] == future_symbol].reset_index(drop=True)

    if len(df) < 2:
        return _empty_result(u, td, "Insufficient bars on current futures contract.")

    price_open = float(df["future_close"].iloc[0])
    price_last = float(df["future_close"].iloc[-1])
    oi_open = float(df["future_oi"].iloc[0])
    oi_last = float(df["future_oi"].iloc[-1])

    timestamps: list[str] = []
    bars: list[dict[str, Any]] = []
    regime_counts: dict[str, int] = {k: 0 for k in BUILDUP_LABELS}
    prev_price: float | None = None
    prev_oi: float | None = None

    for _, row in df.iterrows():
        ts_raw = str(row["timestamp"])
        ts = _hhmm(ts_raw)
        price = float(row["future_close"])
        oi = float(row["future_oi"])
        price_chg_pct = _pct_change(price, prev_price) if prev_price is not None else None
        oi_chg_pct = _pct_change(oi, prev_oi) if prev_oi is not None else None
        buildup = _classify_bar(price_chg_pct, oi_chg_pct) if price_chg_pct is not None else None
        if buildup:
            regime_counts[buildup] = regime_counts.get(buildup, 0) + 1
        session_price_pct = _pct_change(price, price_open)
        session_oi_pct = _pct_change(oi, oi_open)
        bars.append({
            "timestamp": ts,
            "timestamp_raw": ts_raw,
            "price": round(price, 2),
            "oi": round(oi, 0),
            "price_change_pct": price_chg_pct,
            "oi_change_pct": oi_chg_pct,
            "price_session_pct": session_price_pct,
            "oi_session_pct": session_oi_pct,
            "buildup": buildup,
        })
        timestamps.append(ts)
        prev_price = price
        prev_oi = oi

    # Dominant regime = most-frequent non-FLAT classification;
    # ties broken by directional bias against session move (fall back to FLAT).
    non_flat = {k: v for k, v in regime_counts.items() if k != "FLAT" and v > 0}
    if non_flat:
        dominant = max(non_flat.items(), key=lambda kv: kv[1])[0]
    else:
        dominant = "FLAT"

    current_regime = bars[-1].get("buildup") or "FLAT" if bars else "FLAT"
    last_active_regime = next(
        (b["buildup"] for b in reversed(bars) if b.get("buildup") and b["buildup"] != "FLAT"),
        "FLAT",
    )
    last_active_ts = next(
        (b["timestamp"] for b in reversed(bars) if b.get("buildup") and b["buildup"] != "FLAT"),
        None,
    )

    price_change_pct = round((price_last / price_open - 1.0) * 100.0, 3) if price_open else 0.0
    price_change_abs = round(price_last - price_open, 2)
    oi_change_pct = round((oi_last / oi_open - 1.0) * 100.0, 3) if oi_open else 0.0
    oi_change_abs = round(oi_last - oi_open, 0)
    oi_change_lakhs = round((oi_last - oi_open) / 1e5, 2)

    session = {
        "price_open": round(price_open, 2),
        "price_last": round(price_last, 2),
        "price_change_pct": price_change_pct,
        "price_change_abs": price_change_abs,
        "oi_open_lakhs": round(oi_open / 1e5, 2),
        "oi_last_lakhs": round(oi_last / 1e5, 2),
        "oi_change_pct": oi_change_pct,
        "oi_change_abs_lakhs": oi_change_lakhs,
        "dominant_regime": dominant,
        "current_regime": current_regime,
        "last_active_regime": last_active_regime,
        "last_active_at": last_active_ts,
        "regime_counts": regime_counts,
        "bars_classified": sum(regime_counts.values()),
    }

    headline = _build_headline(u, session, future_symbol)
    reasons = _build_reasons(session, regime_counts, future_symbol)

    return asdict(FuturesOiBuildupResult(
        available=True,
        underlying=u,
        trade_date=td.isoformat(),
        future_symbol=future_symbol,
        bars_count=len(bars),
        headline=headline,
        reasons=reasons,
        timestamps=timestamps,
        bars=bars,
        session=session,
        thresholds={
            "price_pct": PRICE_INTERVAL_PCT_THRESH,
            "oi_pct": OI_INTERVAL_PCT_THRESH,
        },
        buildup_labels=BUILDUP_LABELS,
    ))


def _build_headline(underlying: str, session: dict[str, Any], future_symbol: str | None) -> str:
    dom = session.get("dominant_regime") or "FLAT"
    cur = session.get("current_regime") or "FLAT"
    last_active = session.get("last_active_regime") or "FLAT"
    last_active_at = session.get("last_active_at")
    price_pct = session.get("price_change_pct") or 0.0
    oi_pct = session.get("oi_change_pct") or 0.0
    dom_label = BUILDUP_LABELS.get(dom, dom)
    dom_bias = BUILDUP_BIAS.get(dom, "neutral")
    parts = [f"{underlying} futures session {price_pct:+.2f}% · OI {oi_pct:+.2f}%"]
    if dom == "FLAT":
        parts.append("no dominant regime — mostly flat/rotation.")
    else:
        parts.append(f"dominant: {dom_label} ({dom_bias}).")
    if cur != "FLAT":
        parts.append(f"Current bar: {BUILDUP_LABELS.get(cur, cur)}.")
    elif last_active != "FLAT" and last_active_at:
        parts.append(f"Last active: {BUILDUP_LABELS.get(last_active, last_active)} @ {last_active_at}.")
    if future_symbol:
        parts.append(f"({future_symbol})")
    return " · ".join(parts)


def _build_reasons(
    session: dict[str, Any],
    counts: dict[str, int],
    future_symbol: str | None,
) -> list[str]:
    reasons: list[str] = []
    total = sum(counts.values()) or 1
    top3 = sorted(
        [(k, v) for k, v in counts.items() if k != "FLAT" and v > 0],
        key=lambda kv: kv[1],
        reverse=True,
    )[:3]
    if top3:
        breakdown = ", ".join(
            f"{BUILDUP_LABELS[k]} {v} ({v * 100 // total}%)" for k, v in top3
        )
        reasons.append(f"Regime mix: {breakdown} · Flat {counts.get('FLAT', 0)} bars.")
    else:
        reasons.append("All bars flat — spot micro-moves without OI conviction.")
    price_pct = session.get("price_change_pct") or 0.0
    oi_pct = session.get("oi_change_pct") or 0.0
    oi_lakhs = session.get("oi_change_abs_lakhs") or 0.0
    reasons.append(
        f"Session: price {price_pct:+.2f}%, OI {oi_pct:+.2f}% ({oi_lakhs:+.2f}L contracts)."
    )
    dom = session.get("dominant_regime") or "FLAT"
    if dom == "LONG_BUILD":
        reasons.append("Fresh longs dominate — trend has fuel; failed breakdowns likely bought.")
    elif dom == "SHORT_BUILD":
        reasons.append("Fresh shorts dominate — trend has fuel; failed breakouts likely sold.")
    elif dom == "LONG_UNWIND":
        reasons.append("Longs unwinding without fresh shorts — weak hands exiting; watch for exhaustion.")
    elif dom == "SHORT_COVER":
        reasons.append("Shorts covering without fresh longs — squeeze move; often exhausts once OI drops out.")
    if future_symbol:
        reasons.append(f"Contract: {future_symbol}")
    reasons.append(
        f"Thresholds: |Δprice| ≥ {PRICE_INTERVAL_PCT_THRESH}% and |ΔOI| ≥ "
        f"{OI_INTERVAL_PCT_THRESH}% per 5-min bar for a non-FLAT classification."
    )
    return reasons
