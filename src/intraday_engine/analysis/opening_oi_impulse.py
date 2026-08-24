"""Opening OI Impulse Scanner — FnO stocks with big futures OI moves in the
first N minutes of the session.

Signal thesis: when a stock's futures OI jumps > threshold% in the opening
5-15 minutes AND the price moves in a coherent direction, that's fresh
institutional positioning happening in real time. Classifying by our standard
futures matrix (long buildup / short buildup / long unwind / short cover) gives
you a direct read on which side is committing.

Universe: NSE 'most active F&O underlyings' (top-N by turnover), filtered to
stocks only (excludes NIFTY/BANKNIFTY/etc.). Falls back to the liquid FnO
watchlist when NSE returns empty (weekends, pre-open, NSE API hiccup).

Data source: Kite `historical_data(fut_token, interval='minute', oi=True)` from
09:15 to 09:15 + window_minutes. First bar's OI is the baseline; last bar's OI
is current. Parallel over ~30-50 symbols via ThreadPoolExecutor.
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, time as dtime
from pathlib import Path
from typing import Any, Iterable

from intraday_engine.core.underlyings import list_liquid_fno_stocks
from intraday_engine.fetch.instrument_resolver import InstrumentResolver
from intraday_engine.fetch.nse_public_data import fetch_most_active_underlyings
from intraday_engine.fetch.zerodha_client import ZerodhaClient

logger = logging.getLogger(__name__)


INDEX_UNDERLYINGS = frozenset({
    "NIFTY", "BANKNIFTY", "FINNIFTY", "MIDCPNIFTY", "MIDCAP",
    "BANKEX", "SENSEX", "NIFTYNXT50",
})

FUT_OI_PCT_DEFAULT = 7.0
WINDOW_MIN_DEFAULT = 15
MAX_SYMBOLS_DEFAULT = 60
MIN_TURNOVER_CR_DEFAULT = 5.0  # NSE-reported total turnover threshold for the universe
MARKET_OPEN = dtime(9, 15)
MARKET_CLOSE = dtime(15, 30)

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

# Buildup strength — bright (buildup) vs faded (unwind/cover) — mirrors the
# dashboard's futures buildup colors so the panel is consistent.
BUILDUP_STRENGTH: dict[str, str] = {
    "LONG_BUILD": "strong",
    "SHORT_BUILD": "strong",
    "LONG_UNWIND": "weak",
    "SHORT_COVER": "weak",
    "FLAT": "flat",
}


@dataclass
class OpeningImpulseResult:
    available: bool
    trade_date: str
    scan_time: str
    window_minutes: int
    threshold_pct: float
    min_turnover_cr: float
    universe_source: str
    universe_size: int
    scanned_count: int
    hits: list[dict[str, Any]] = field(default_factory=list)
    skipped: list[dict[str, Any]] = field(default_factory=list)
    summary: dict[str, Any] = field(default_factory=dict)
    message: str | None = None


def _classify_bar(price_chg_pct: float | None, oi_chg_pct: float | None) -> str:
    """Standard futures matrix (labels reflect position on the underlying)."""
    if price_chg_pct is None or oi_chg_pct is None:
        return "FLAT"
    price_up = price_chg_pct > 0.02
    price_down = price_chg_pct < -0.02
    oi_up = oi_chg_pct > 0.5
    oi_down = oi_chg_pct < -0.5
    if price_up and oi_up:
        return "LONG_BUILD"
    if price_down and oi_up:
        return "SHORT_BUILD"
    if price_down and oi_down:
        return "LONG_UNWIND"
    if price_up and oi_down:
        return "SHORT_COVER"
    return "FLAT"


def _suggested_side(buildup: str) -> str | None:
    if buildup == "LONG_BUILD":
        return "CE"
    if buildup == "SHORT_BUILD":
        return "PE"
    if buildup == "LONG_UNWIND":
        return "PE"  # weak bearish; smaller size
    if buildup == "SHORT_COVER":
        return "CE"  # weak bullish; smaller size
    return None


def _build_universe(
    data_dir: Path,
    max_symbols: int,
    min_turnover_cr: float,
) -> tuple[list[str], list[dict[str, Any]], str]:
    """Return (symbol_list, ordered_meta_rows, source_tag). Falls back to
    liquid FnO watchlist if NSE most-active is empty."""
    rows = fetch_most_active_underlyings(data_dir)
    stock_rows = [
        r for r in rows
        if r.get("symbol") not in INDEX_UNDERLYINGS
        and (r.get("total_turnover_cr") or 0) >= min_turnover_cr
    ]
    if stock_rows:
        stock_rows.sort(key=lambda x: (x.get("total_turnover_cr") or 0), reverse=True)
        stock_rows = stock_rows[:max_symbols]
        return [r["symbol"] for r in stock_rows], stock_rows, "nse_most_active"
    # Fallback: liquid FnO watchlist with no per-symbol metadata
    fallback = list(list_liquid_fno_stocks(data_dir))[:max_symbols]
    fallback_meta = [{"symbol": s} for s in fallback]
    return fallback, fallback_meta, "liquid_fno_fallback"


def _fetch_symbol_impulse(
    client: ZerodhaClient,
    nfo_instruments: list[dict[str, Any]],
    symbol: str,
    trade_date: date,
    from_dt: datetime,
    to_dt: datetime,
) -> tuple[str, dict[str, Any] | None, str | None]:
    """Fetch minute bars for one symbol's front-month future and return the
    computed impulse row (or a skip reason)."""
    try:
        fut = InstrumentResolver._nearest_future(nfo_instruments, symbol, trade_date)
        if not fut:
            return symbol, None, "no_front_month_future"
        fut_token = int(fut.get("instrument_token") or 0)
        if fut_token <= 0:
            return symbol, None, "no_fut_token"
        fut_symbol = f"NFO:{fut.get('tradingsymbol')}"
        bars = client.historical_data(fut_token, from_dt, to_dt, interval="minute", oi=True)
        if not bars or len(bars) < 2:
            return symbol, None, "insufficient_bars"
        # Filter to bars within our window (Kite occasionally returns extras)
        bars = [b for b in bars if b.get("oi") is not None and b.get("open") is not None]
        if len(bars) < 2:
            return symbol, None, "no_oi_data"
        first = bars[0]
        last = bars[-1]
        baseline_oi = float(first.get("oi") or 0)
        current_oi = float(last.get("oi") or 0)
        if baseline_oi <= 0 or current_oi <= 0:
            return symbol, None, "zero_oi"
        baseline_price = float(first.get("open") or 0)
        current_price = float(last.get("close") or 0)
        if baseline_price <= 0 or current_price <= 0:
            return symbol, None, "zero_price"
        oi_change_pct = (current_oi / baseline_oi - 1.0) * 100.0
        price_change_pct = (current_price / baseline_price - 1.0) * 100.0
        cum_volume = sum(float(b.get("volume") or 0) for b in bars)
        # Approximate turnover in ₹cr — volume × avg price / 1e7
        avg_price = (baseline_price + current_price) / 2
        turnover_cr = (cum_volume * avg_price) / 1e7 if cum_volume else 0.0
        return symbol, {
            "symbol": symbol,
            "fut_symbol": fut_symbol,
            "fut_expiry": fut.get("expiry").isoformat() if hasattr(fut.get("expiry"), "isoformat") else str(fut.get("expiry", "")),
            "fut_lot_size": int(fut.get("lot_size", 0) or 0) or None,
            "baseline_oi": round(baseline_oi, 0),
            "current_oi": round(current_oi, 0),
            "oi_change_abs": round(current_oi - baseline_oi, 0),
            "oi_change_pct": round(oi_change_pct, 3),
            "baseline_price": round(baseline_price, 2),
            "current_price": round(current_price, 2),
            "price_change_abs": round(current_price - baseline_price, 2),
            "price_change_pct": round(price_change_pct, 3),
            "fut_cum_volume": int(cum_volume),
            "fut_turnover_cr": round(turnover_cr, 2),
            "bars_used": len(bars),
        }, None
    except Exception as e:
        logger.debug("Impulse fetch failed for %s: %s", symbol, e)
        return symbol, None, f"error:{type(e).__name__}"


def _enrich_atm(
    client: ZerodhaClient,
    nfo_instruments: list[dict[str, Any]],
    row: dict[str, Any],
    trade_date: date,
) -> None:
    """Add atm_strike, atm_ce_symbol/ltp, atm_pe_symbol/ltp, suggested_option_*."""
    try:
        ce, pe, atm = InstrumentResolver._nearest_option_pair(
            nfo_instruments, row["symbol"], trade_date, row["current_price"]
        )
        if not ce or not pe or atm <= 0:
            return
        ce_symbol = f"NFO:{ce.get('tradingsymbol')}"
        pe_symbol = f"NFO:{pe.get('tradingsymbol')}"
        row["atm_strike"] = int(atm)
        row["atm_ce_symbol"] = ce_symbol
        row["atm_pe_symbol"] = pe_symbol
        try:
            quotes = client.quote([ce_symbol, pe_symbol])
            row["atm_ce_ltp"] = round(float(quotes.get(ce_symbol, {}).get("last_price") or 0), 2) or None
            row["atm_pe_ltp"] = round(float(quotes.get(pe_symbol, {}).get("last_price") or 0), 2) or None
        except Exception as e:
            logger.debug("Quote failed for %s: %s", row["symbol"], e)
        side = _suggested_side(row.get("buildup"))
        if side == "CE" and row.get("atm_ce_ltp") is not None:
            row["suggested_side"] = "BUY_CE"
            row["suggested_option_symbol"] = ce_symbol
            row["suggested_option_ltp"] = row["atm_ce_ltp"]
        elif side == "PE" and row.get("atm_pe_ltp") is not None:
            row["suggested_side"] = "BUY_PE"
            row["suggested_option_symbol"] = pe_symbol
            row["suggested_option_ltp"] = row["atm_pe_ltp"]
    except Exception as e:
        logger.debug("ATM enrich failed for %s: %s", row.get("symbol"), e)


def compute_opening_oi_impulse(
    client: ZerodhaClient,
    data_dir: Path,
    trade_date: date | None = None,
    *,
    window_minutes: int = WINDOW_MIN_DEFAULT,
    oi_threshold_pct: float = FUT_OI_PCT_DEFAULT,
    min_turnover_cr: float = MIN_TURNOVER_CR_DEFAULT,
    max_symbols: int = MAX_SYMBOLS_DEFAULT,
    fetch_atm_ltp: bool = True,
    max_workers: int = 6,
) -> dict[str, Any]:
    """Scan FnO stocks for outsized opening futures-OI moves.

    Returns a dict shaped like `OpeningImpulseResult` (see dataclass above)."""
    td = trade_date or date.today()
    scan_time_now = datetime.now()

    # Data window: 09:15 to min(09:15 + window_minutes, now/market_close)
    from_dt = datetime.combine(td, MARKET_OPEN)
    window_end_dt = datetime.combine(td, dtime(9, 15 + int(window_minutes))) if window_minutes < 375 \
        else datetime.combine(td, MARKET_CLOSE)
    # Live day cap: don't ask Kite for future bars
    if td == date.today():
        window_end_dt = min(window_end_dt, scan_time_now)
    to_dt = window_end_dt

    if to_dt <= from_dt:
        return asdict(OpeningImpulseResult(
            available=False,
            trade_date=td.isoformat(),
            scan_time=scan_time_now.isoformat(timespec="seconds"),
            window_minutes=window_minutes,
            threshold_pct=oi_threshold_pct,
            min_turnover_cr=min_turnover_cr,
            universe_source="unknown",
            universe_size=0,
            scanned_count=0,
            message="Market has not opened yet — no bars in the requested window.",
        ))

    symbols, meta_rows, source = _build_universe(data_dir, max_symbols, min_turnover_cr)
    if not symbols:
        return asdict(OpeningImpulseResult(
            available=False,
            trade_date=td.isoformat(),
            scan_time=scan_time_now.isoformat(timespec="seconds"),
            window_minutes=window_minutes,
            threshold_pct=oi_threshold_pct,
            min_turnover_cr=min_turnover_cr,
            universe_source=source,
            universe_size=0,
            scanned_count=0,
            message="Universe empty — NSE most-active returned nothing and fallback watchlist is empty.",
        ))

    meta_by_symbol = {r["symbol"]: r for r in meta_rows}
    nfo_instruments = client.nfo_instruments()

    rows: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {
            ex.submit(_fetch_symbol_impulse, client, nfo_instruments, sym, td, from_dt, to_dt): sym
            for sym in symbols
        }
        for fut in as_completed(futures):
            sym, row, err = fut.result()
            if row is None:
                skipped.append({"symbol": sym, "reason": err or "unknown"})
                continue
            # Merge NSE-reported metadata (total turnover, opt/fut volume from NSE)
            meta = meta_by_symbol.get(sym) or {}
            row["nse_total_turnover_cr"] = meta.get("total_turnover_cr")
            row["nse_fut_turnover_cr"] = meta.get("fut_turnover_cr")
            row["nse_opt_volume"] = meta.get("opt_volume")
            row["spot_ltp_hint"] = meta.get("ltp")
            row["buildup"] = _classify_bar(row["price_change_pct"], row["oi_change_pct"])
            row["buildup_label"] = BUILDUP_LABELS[row["buildup"]]
            row["bias"] = BUILDUP_BIAS[row["buildup"]]
            row["strength"] = BUILDUP_STRENGTH[row["buildup"]]
            rows.append(row)

    # Filter by OI threshold (both directions)
    hits = [r for r in rows if abs(r["oi_change_pct"]) >= oi_threshold_pct]

    # Enrich hits with ATM options info + suggested trade side
    if fetch_atm_ltp and hits:
        with ThreadPoolExecutor(max_workers=min(4, max_workers)) as ex:
            list(ex.map(lambda r: _enrich_atm(client, nfo_instruments, r, td), hits))

    # Sort by absolute OI change % desc; tiebreak on turnover
    hits.sort(key=lambda r: (abs(r["oi_change_pct"]), r.get("fut_turnover_cr") or 0), reverse=True)

    summary = {
        "hits_long_build": sum(1 for r in hits if r["buildup"] == "LONG_BUILD"),
        "hits_short_build": sum(1 for r in hits if r["buildup"] == "SHORT_BUILD"),
        "hits_long_unwind": sum(1 for r in hits if r["buildup"] == "LONG_UNWIND"),
        "hits_short_cover": sum(1 for r in hits if r["buildup"] == "SHORT_COVER"),
        "hits_flat": sum(1 for r in hits if r["buildup"] == "FLAT"),
        "bullish_count": sum(1 for r in hits if r["bias"] == "bullish"),
        "bearish_count": sum(1 for r in hits if r["bias"] == "bearish"),
        "scanned_ok": len(rows),
        "skipped": len(skipped),
    }

    return asdict(OpeningImpulseResult(
        available=True,
        trade_date=td.isoformat(),
        scan_time=scan_time_now.isoformat(timespec="seconds"),
        window_minutes=int((to_dt - from_dt).total_seconds() // 60),
        threshold_pct=oi_threshold_pct,
        min_turnover_cr=min_turnover_cr,
        universe_source=source,
        universe_size=len(symbols),
        scanned_count=len(rows),
        hits=hits,
        skipped=skipped,
        summary=summary,
    ))


def opening_impulse_cache_path(data_dir: Path, trade_date: date) -> Path:
    return data_dir / "analysis" / "opening_impulse" / f"opening_impulse_{trade_date.isoformat()}.json"


def save_opening_impulse(payload: dict[str, Any], data_dir: Path, trade_date: date) -> Path:
    import json
    path = opening_impulse_cache_path(data_dir, trade_date)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def load_opening_impulse(data_dir: Path, trade_date: date) -> dict[str, Any] | None:
    import json
    path = opening_impulse_cache_path(data_dir, trade_date)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
