"""
NSE public data downloader (delivery %, bulk/block deals, FII/DII, participant OI).

Internally uses `jugaad-data` (https://pypi.org/project/jugaad-data/) for the
heavy-lifting bits — its `NSEArchives` and `NSELive` classes carry a maintained
cookie + header configuration that survives most NSE redesigns. For endpoints
jugaad doesn't expose first-class (FII/DII JSON, archived participant-OI CSV,
block deals JSON), we lift jugaad's `requests.Session` and aim it at those URLs
ourselves so the same anti-bot warmup is reused.

Every successful response is cached forever under `data/reference/nse/`. All
public functions are fail-soft: they return an empty DataFrame / list on
failure and never raise.
"""

from __future__ import annotations

import io
import json
import logging
import threading
import time
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd

from intraday_engine.storage.layout import nse_public_data_dir
from intraday_engine.utils.nse_trading_calendar import (
    is_nse_trading_day,
    load_nse_holidays,
    recent_trading_days,
    refresh_nse_holidays,
)

logger = logging.getLogger(__name__)


# ---------- jugaad-data session singletons ----------

_lock = threading.Lock()
_archives: Any = None
_live: Any = None
_live_warmed = False


def _get_archives():
    """Lazy-loaded jugaad-data NSEArchives singleton (used for bhavcopy + bulk deals)."""
    global _archives
    if _archives is not None:
        return _archives
    with _lock:
        if _archives is None:
            try:
                from jugaad_data.nse import NSEArchives
                inst = NSEArchives()
                inst.timeout = 25  # default 4s is too aggressive for slow endpoints
                _archives = inst
            except ImportError:
                logger.warning("jugaad-data not installed; NSE downloads will be skipped.")
                _archives = False  # sentinel: unavailable
    return _archives or None


def _get_live():
    """Lazy-loaded jugaad-data NSELive singleton, with one-shot warmup against the
    reports page so /api/fiidiiTradeReact returns 200 instead of timing out."""
    global _live, _live_warmed
    if _live is not None and _live_warmed:
        return _live
    with _lock:
        if _live is None:
            try:
                from jugaad_data.nse import NSELive
                _live = NSELive()
            except ImportError:
                logger.warning("jugaad-data not installed; FII/DII fetch will be skipped.")
                _live = False
                return None
        if not _live_warmed and _live:
            try:
                _live.s.get("https://www.nseindia.com/reports/fii-dii", timeout=20)
                # touch market_status to ensure cookies fully bake
                try:
                    _live.get("market_status")
                except Exception:
                    pass
                _live_warmed = True
            except Exception as e:
                logger.debug("NSELive warmup failed: %s", e)
                # mark warmed anyway so we don't loop forever; subsequent fetches still try
                _live_warmed = True
    return _live or None


# ---------- cache helpers ----------

def _cache_path(data_dir: Path, name: str) -> Path:
    return nse_public_data_dir(data_dir) / name


def _save_cache(data_dir: Path, name: str, payload: bytes | str) -> Path:
    p = _cache_path(data_dir, name)
    if isinstance(payload, str):
        p.write_text(payload, encoding="utf-8")
    else:
        p.write_bytes(payload)
    return p


def _load_cache_text(data_dir: Path, name: str) -> str | None:
    p = _cache_path(data_dir, name)
    if not p.exists():
        return None
    try:
        return p.read_text(encoding="utf-8")
    except Exception:
        return None


def _ddmmyyyy(d: date) -> str:
    return d.strftime("%d%m%Y")


# ---------- Security-wise delivery bhavcopy ----------

def fetch_delivery_bhavcopy(data_dir: Path, d: date) -> pd.DataFrame:
    """Returns a DataFrame with columns SYMBOL, SERIES, DELIV_QTY, DELIV_PER, etc.
    Cached forever once obtained."""
    name = f"sec_bhavdata_full_{_ddmmyyyy(d)}.csv"
    raw = _load_cache_text(data_dir, name)
    if raw is None:
        archives = _get_archives()
        if archives is not None:
            try:
                raw = archives.full_bhavcopy_raw(d)
                if raw and len(raw) > 1024:
                    _save_cache(data_dir, name, raw)
                else:
                    raw = None
            except Exception as e:
                logger.debug("delivery bhavcopy %s err: %s", d, e)
                raw = None
    if not raw:
        return pd.DataFrame()
    try:
        df = pd.read_csv(io.StringIO(raw))
        df.columns = [str(c).strip() for c in df.columns]
        for c in ("SYMBOL", "SERIES"):
            if c in df.columns:
                df[c] = df[c].astype(str).str.strip()
        for c in ("DELIV_QTY", "DELIV_PER", "TTL_TRD_QNTY", "AVG_PRICE", "CLOSE_PRICE"):
            if c in df.columns:
                df[c] = pd.to_numeric(
                    df[c].astype(str).str.replace(",", "").str.strip(), errors="coerce"
                )
        if "SERIES" in df.columns:
            df = df[df["SERIES"].isin(["EQ", "BE", "BZ"])].copy()
        return df.reset_index(drop=True)
    except Exception as e:
        logger.debug("Parse delivery bhavcopy %s failed: %s", d, e)
        return pd.DataFrame()


def get_delivery_history(data_dir: Path, days: int = 30) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    today = date.today()
    fetched = 0
    look = 0
    while fetched < days and look < days * 2 + 6:
        d = today - timedelta(days=look)
        look += 1
        if d.weekday() >= 5:
            continue
        df = fetch_delivery_bhavcopy(data_dir, d)
        if df.empty:
            continue
        df = df.copy()
        df["TRADE_DATE"] = d.isoformat()
        rows.append(df)
        fetched += 1
        time.sleep(0.4)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


# ---------- Bulk / Block deals ----------
#
# jugaad-data exposes only the "current" bulk deals CSV (last ~14 days). To
# build a 30-day rolling history we accumulate that snapshot into a single
# JSON cache (`bulk_deals_history.json`) keyed by (date, symbol, client, side, qty).
# Each refresh reads the CSV, dedupes against the cache, and grows the history.

_BULK_CACHE = "bulk_deals_history.json"
_BLOCK_CACHE = "block_deals_history.json"
_BLOCK_URL = "https://www.nseindia.com/api/historical/cm/block"


def _load_history_json(data_dir: Path, name: str) -> list[dict[str, Any]]:
    p = _cache_path(data_dir, name)
    if not p.exists():
        return []
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        return list(data) if isinstance(data, list) else []
    except Exception:
        return []


def _save_history_json(data_dir: Path, name: str, rows: list[dict[str, Any]]) -> None:
    p = _cache_path(data_dir, name)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(rows, indent=2), encoding="utf-8")


def _deal_key(d: dict[str, Any]) -> str:
    """Stable identity of a single deal across refreshes."""
    return "|".join(
        str(d.get(k, "")).strip()
        for k in ("date", "symbol", "clientName", "buySell", "quantity", "price")
    )


def _normalize_deal_row(row: dict[str, Any]) -> dict[str, Any] | None:
    """Map the jugaad bulk-deals CSV row (or the block-deals JSON row) to a
    canonical dict our scanners consume."""
    sym = (
        row.get("Symbol")
        or row.get("symbol")
        or row.get("BD_SYMBOL")
        or ""
    )
    sym = str(sym).strip().upper()
    if not sym:
        return None
    client = (
        row.get("Client Name")
        or row.get("clientName")
        or row.get("BD_CLIENT_NAME")
        or ""
    )
    side = str(
        row.get("Buy/Sell")
        or row.get("buySell")
        or row.get("BD_BUY_SELL")
        or ""
    ).strip().upper()
    qty_raw = (
        row.get("Quantity Traded")
        or row.get("quantity")
        or row.get("BD_QTY_TRD")
        or 0
    )
    price_raw = (
        row.get("Trade Price / Wght. Avg. Price")
        or row.get("price")
        or row.get("BD_TP_WATP")
        or 0
    )
    date_raw = (
        row.get("Date")
        or row.get("date")
        or row.get("BD_DT_DATE")
        or ""
    )
    try:
        qty = int(float(str(qty_raw).replace(",", "").strip() or 0))
    except Exception:
        qty = 0
    try:
        price = float(str(price_raw).replace(",", "").strip() or 0)
    except Exception:
        price = 0.0
    iso = _normalize_iso_date(date_raw)
    return {
        "date": iso,
        "symbol": sym,
        "clientName": str(client).strip(),
        "buySell": side,
        "quantity": qty,
        "price": price,
    }


def _normalize_iso_date(val: Any) -> str:
    s = str(val or "").strip()
    if not s:
        return ""
    for fmt in ("%d-%b-%Y", "%d-%B-%Y", "%Y-%m-%d", "%d-%m-%Y", "%d %b %Y"):
        try:
            return datetime.strptime(s, fmt).date().isoformat()
        except Exception:
            continue
    try:
        return pd.to_datetime(s).date().isoformat()
    except Exception:
        return s


def _refresh_bulk_deals(data_dir: Path) -> list[dict[str, Any]]:
    """Pull jugaad's bulk-deals CSV (last ~14 days) and merge into rolling history."""
    archives = _get_archives()
    if archives is None:
        return _load_history_json(data_dir, _BULK_CACHE)
    try:
        raw_csv = archives.bulk_deals_raw()
    except Exception as e:
        logger.debug("bulk_deals fetch err: %s", e)
        return _load_history_json(data_dir, _BULK_CACHE)
    if not raw_csv or len(raw_csv) < 64:
        return _load_history_json(data_dir, _BULK_CACHE)

    try:
        df = pd.read_csv(io.StringIO(raw_csv))
    except Exception:
        return _load_history_json(data_dir, _BULK_CACHE)

    fresh: list[dict[str, Any]] = []
    for _, row in df.iterrows():
        d = _normalize_deal_row(row.to_dict())
        if d:
            fresh.append(d)

    history = _load_history_json(data_dir, _BULK_CACHE)
    seen = {_deal_key(d) for d in history}
    for d in fresh:
        if _deal_key(d) not in seen:
            history.append(d)
            seen.add(_deal_key(d))
    history.sort(key=lambda d: (d.get("date", ""), d.get("symbol", "")), reverse=True)
    _save_history_json(data_dir, _BULK_CACHE, history)
    return history


def _refresh_block_deals(data_dir: Path) -> list[dict[str, Any]]:
    """jugaad doesn't expose block deals; reuse its NSELive session for the JSON API."""
    live = _get_live()
    if live is None:
        return _load_history_json(data_dir, _BLOCK_CACHE)
    today = date.today()
    frm = today - timedelta(days=45)
    fmt = "%d-%m-%Y"
    url = f"{_BLOCK_URL}?from={frm.strftime(fmt)}&to={today.strftime(fmt)}"
    try:
        r = live.s.get(
            url,
            headers={"Referer": "https://www.nseindia.com/market-data/large-deals"},
            timeout=25,
        )
        if r.status_code != 200:
            return _load_history_json(data_dir, _BLOCK_CACHE)
        payload = r.json()
    except Exception as e:
        logger.debug("block_deals err: %s", e)
        return _load_history_json(data_dir, _BLOCK_CACHE)

    fresh = []
    for item in (payload.get("data") or []):
        if not isinstance(item, dict):
            continue
        d = _normalize_deal_row(item)
        if d:
            fresh.append(d)

    history = _load_history_json(data_dir, _BLOCK_CACHE)
    seen = {_deal_key(d) for d in history}
    for d in fresh:
        if _deal_key(d) not in seen:
            history.append(d)
            seen.add(_deal_key(d))
    history.sort(key=lambda d: (d.get("date", ""), d.get("symbol", "")), reverse=True)
    _save_history_json(data_dir, _BLOCK_CACHE, history)
    return history


def get_recent_bulk_deals(data_dir: Path, days: int = 30) -> list[dict[str, Any]]:
    rows = _refresh_bulk_deals(data_dir)
    return _filter_deals_by_days(rows, days)


def get_recent_block_deals(data_dir: Path, days: int = 30) -> list[dict[str, Any]]:
    rows = _refresh_block_deals(data_dir)
    return _filter_deals_by_days(rows, days)


def _filter_deals_by_days(rows: list[dict[str, Any]], days: int) -> list[dict[str, Any]]:
    if not rows:
        return []
    cutoff = (date.today() - timedelta(days=days)).isoformat()
    return [r for r in rows if str(r.get("date", "")) >= cutoff]


# ---------- Daily FII/DII (cash) ----------

_FII_DII_URL = "https://www.nseindia.com/api/fiidiiTradeReact"
_FIIDII_TRADE_PREFIX = "fiidii_trade_"


def _fiidii_trade_cache_name(trade_date: date) -> str:
    return f"{_FIIDII_TRADE_PREFIX}{trade_date.isoformat()}.json"


def _save_fiidii_trade_day(data_dir: Path, trade_date_iso: str, items: list[dict[str, Any]]) -> None:
    if not trade_date_iso or not items:
        return
    _save_cache(data_dir, _fiidii_trade_cache_name(date.fromisoformat(trade_date_iso)), json.dumps(items))


def _cached_fiidii_trade_dates(data_dir: Path) -> set[str]:
    base = nse_public_data_dir(data_dir)
    dates: set[str] = set()
    for p in base.glob(f"{_FIIDII_TRADE_PREFIX}*.json"):
        iso = p.stem.replace(_FIIDII_TRADE_PREFIX, "", 1)
        if iso:
            dates.add(iso)
    return dates


def _reindex_legacy_fiidii_caches(data_dir: Path) -> int:
    """Map legacy fiidii_YYYY-MM-DD.json (fetch-day snapshots) to trade-date files."""
    base = nse_public_data_dir(data_dir)
    written = 0
    for p in base.glob("fiidii_*.json"):
        if p.name.startswith(_FIIDII_TRADE_PREFIX):
            continue
        try:
            payload = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(payload, list):
            continue
        by_day: dict[str, list[dict[str, Any]]] = {}
        for item in payload:
            if not isinstance(item, dict):
                continue
            d_iso = _normalize_iso_date(item.get("date"))
            if not d_iso:
                continue
            by_day.setdefault(d_iso, []).append(item)
        for d_iso, items in by_day.items():
            target = _cache_path(data_dir, _fiidii_trade_cache_name(date.fromisoformat(d_iso)))
            if not target.exists():
                _save_fiidii_trade_day(data_dir, d_iso, items)
                written += 1
    return written


def fetch_fii_dii_today(data_dir: Path) -> list[dict[str, Any]]:
    """Provisional FII/DII numbers for the latest published trading day."""
    cache_name = f"fiidii_{date.today().isoformat()}.json"
    live = _get_live()
    if live is None:
        cached = _load_cache_text(data_dir, cache_name)
        if cached:
            try:
                payload = json.loads(cached)
                if isinstance(payload, list):
                    for item in payload:
                        d_iso = _normalize_iso_date(item.get("date"))
                        if d_iso:
                            day_items = [
                                x
                                for x in payload
                                if isinstance(x, dict) and _normalize_iso_date(x.get("date")) == d_iso
                            ]
                            _save_fiidii_trade_day(data_dir, d_iso, day_items)
                return payload
            except Exception:
                return []
        return []
    try:
        r = live.s.get(
            _FII_DII_URL,
            headers={"Referer": "https://www.nseindia.com/reports/fii-dii"},
            timeout=30,
        )
        if r.status_code == 200 and r.text and r.text.strip().startswith("["):
            _save_cache(data_dir, cache_name, r.text)
            payload = r.json()
            by_day: dict[str, list[dict[str, Any]]] = {}
            for item in payload:
                if not isinstance(item, dict):
                    continue
                d_iso = _normalize_iso_date(item.get("date"))
                if not d_iso:
                    continue
                by_day.setdefault(d_iso, []).append(item)
            for d_iso, items in by_day.items():
                _save_fiidii_trade_day(data_dir, d_iso, items)
            return payload
    except Exception as e:
        logger.debug("fiidii fetch err: %s", e)
    cached = _load_cache_text(data_dir, cache_name)
    if cached:
        try:
            return json.loads(cached)
        except Exception:
            return []
    return []


def _fiidii_rows_from_trade_cache(data_dir: Path, trade_days: list[date]) -> list[dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for td in trade_days:
        raw = _load_cache_text(data_dir, _fiidii_trade_cache_name(td))
        if not raw:
            continue
        try:
            payload = json.loads(raw)
        except Exception:
            continue
        if not isinstance(payload, list):
            continue
        for item in payload:
            if not isinstance(item, dict):
                continue
            cat_raw = str(item.get("category") or "").upper()
            d_iso = _normalize_iso_date(item.get("date")) or td.isoformat()
            if "FII" in cat_raw or "FPI" in cat_raw:
                prefix = "fii"
            elif "DII" in cat_raw:
                prefix = "dii"
            else:
                continue
            row = out.setdefault(d_iso, {"date": d_iso})
            row[f"{prefix}_buy"] = _to_float(item.get("buyValue"))
            row[f"{prefix}_sell"] = _to_float(item.get("sellValue"))
            row[f"{prefix}_net"] = _to_float(item.get("netValue"))
    return [out[d.isoformat()] for d in trade_days if d.isoformat() in out]


def get_fii_dii_cached_history(
    data_dir: Path,
    *,
    within_trading_days: int = 30,
) -> list[dict[str, Any]]:
    """All trade-date keyed FII/DII rows in cache (newest first)."""
    _reindex_legacy_fiidii_caches(data_dir)
    cached_isos = sorted(_cached_fiidii_trade_dates(data_dir))
    if within_trading_days > 0:
        window = {
            d.isoformat()
            for d in recent_trading_days(within_trading_days, date.today(), data_dir)
        }
        cached_isos = [d for d in cached_isos if d in window]
    trade_days = [date.fromisoformat(d) for d in cached_isos]
    rows = _fiidii_rows_from_trade_cache(data_dir, trade_days)
    return list(reversed(rows))


def get_fii_dii_30d_history(data_dir: Path, days: int = 30) -> list[dict[str, Any]]:
    """Last N trading sessions from trade-date keyed cache."""
    return get_fii_dii_cached_history(data_dir, within_trading_days=days)


def get_fii_dii_trading_window(
    data_dir: Path,
    trading_days: int,
    *,
    as_of: date | None = None,
) -> list[dict[str, Any]]:
    """
    One row per trading session in the last ``trading_days`` window (newest first).
    Sessions without cached FII/DII still appear with ``date`` only (null nets).
    """
    _reindex_legacy_fiidii_caches(data_dir)
    end = as_of or date.today()
    sessions = recent_trading_days(trading_days, end, data_dir)
    cached_rows = _fiidii_rows_from_trade_cache(data_dir, sessions)
    by_date = {r["date"]: r for r in cached_rows}
    return [
        by_date.get(td.isoformat(), {"date": td.isoformat()})
        for td in reversed(sessions)
    ]


def fii_dii_cache_coverage(data_dir: Path, trading_days: int = 30) -> dict[str, Any]:
    """Lightweight FII/DII cache counts (no network)."""
    _reindex_legacy_fiidii_caches(data_dir)
    holidays = load_nse_holidays(data_dir)
    targets = recent_trading_days(trading_days, date.today(), data_dir, holidays=holidays)
    target_isos = [d.isoformat() for d in targets]
    fii_cached = _cached_fiidii_trade_dates(data_dir)
    return {
        "trading_days_requested": trading_days,
        "fii_dii_have": sum(1 for iso in target_isos if iso in fii_cached),
        "fii_dii_missing": [iso for iso in target_isos if iso not in fii_cached],
        "fii_dii_cached_dates": sorted(fii_cached),
    }


def ensure_fii_dii_oi_history(data_dir: Path, trading_days: int = 30) -> dict[str, Any]:
    """
    Ensure last ``trading_days`` NSE sessions are cached (weekends/holidays skipped).
    Fetches missing participant-OI archives; refreshes today's FII/DII snapshot.
    """
    holidays = refresh_nse_holidays(data_dir)
    targets = recent_trading_days(trading_days, date.today(), data_dir, holidays=holidays)
    target_isos = [d.isoformat() for d in targets]

    reindexed = _reindex_legacy_fiidii_caches(data_dir)
    fetch_fii_dii_today(data_dir)

    oi_fetched = 0
    oi_missing: list[str] = []
    for td in targets:
        cache_name = f"fao_participant_oi_{_ddmmyyyy(td)}.csv"
        if _load_cache_text(data_dir, cache_name) is not None:
            continue
        if not is_nse_trading_day(td, holidays):
            continue
        df = fetch_participant_oi(data_dir, td)
        if df.empty:
            oi_missing.append(td.isoformat())
        else:
            oi_fetched += 1
        time.sleep(0.4)

    fii_cached = _cached_fiidii_trade_dates(data_dir)
    fii_missing = [iso for iso in target_isos if iso not in fii_cached]
    oi_have = sum(
        1
        for td in targets
        if _load_cache_text(data_dir, f"fao_participant_oi_{_ddmmyyyy(td)}.csv") is not None
    )
    fii_have = sum(1 for iso in target_isos if iso in fii_cached)

    return {
        "trading_days_requested": trading_days,
        "trading_days_targets": target_isos,
        "holidays_cached": len(holidays),
        "fiidii_reindexed": reindexed,
        "participant_oi_fetched": oi_fetched,
        "participant_oi_have": oi_have,
        "participant_oi_missing": oi_missing,
        "fii_dii_have": fii_have,
        "fii_dii_missing": fii_missing,
        "complete": oi_have >= trading_days and fii_have >= trading_days,
    }


def _to_float(v: Any) -> float | None:
    if v is None:
        return None
    try:
        return float(str(v).replace(",", "").strip())
    except Exception:
        return None


# ---------- Participant-wise F&O OI ----------

_PARTICIPANT_OI_URL = (
    "https://nsearchives.nseindia.com/content/nsccl/fao_participant_oi_{dd}.csv"
)


def _parse_participant_oi_csv(raw: str, d: date) -> pd.DataFrame:
    body: list[str] = []
    started = False
    for line in raw.splitlines():
        s = line.strip()
        if not s:
            continue
        if not started and s.lower().startswith("client type"):
            started = True
        if started:
            body.append(line)
    if not body:
        return pd.DataFrame()
    try:
        df = pd.read_csv(io.StringIO("\n".join(body)))
        df.columns = [str(c).strip() for c in df.columns]
        df.iloc[:, 0] = df.iloc[:, 0].astype(str).str.strip()
        for c in df.columns[1:]:
            df[c] = pd.to_numeric(
                df[c].astype(str).str.replace(",", "").str.strip(), errors="coerce"
            )
        df["TRADE_DATE"] = d.isoformat()
        return df.reset_index(drop=True)
    except Exception as e:
        logger.debug("Participant OI parse %s failed: %s", d, e)
        return pd.DataFrame()


def fetch_participant_oi(data_dir: Path, d: date) -> pd.DataFrame:
    if not is_nse_trading_day(d, load_nse_holidays(data_dir)):
        return pd.DataFrame()

    name = f"fao_participant_oi_{_ddmmyyyy(d)}.csv"
    raw = _load_cache_text(data_dir, name)
    if raw is None:
        archives = _get_archives()
        if archives is not None:
            url = _PARTICIPANT_OI_URL.format(dd=_ddmmyyyy(d))
            try:
                r = archives.s.get(url, timeout=25)
                if r.status_code == 200 and r.text and len(r.text) > 256:
                    raw = r.text
                    _save_cache(data_dir, name, raw)
            except Exception as e:
                logger.debug("participant_oi %s err: %s", d, e)
    if not raw:
        return pd.DataFrame()
    return _parse_participant_oi_csv(raw, d)


def get_participant_oi_history(data_dir: Path, days: int = 30) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for td in recent_trading_days(days, date.today(), data_dir):
        name = f"fao_participant_oi_{_ddmmyyyy(td)}.csv"
        raw = _load_cache_text(data_dir, name)
        if not raw:
            continue
        df = _parse_participant_oi_csv(raw, td)
        if not df.empty:
            rows.append(df)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def participant_oi_long_short(df: pd.DataFrame) -> list[dict[str, Any]]:
    """Convert raw participant OI to slim per-day long/short rows for charting."""
    if df.empty:
        return []
    cols = {c.lower(): c for c in df.columns}
    client_col = df.columns[0]
    out: list[dict[str, Any]] = []
    for _, r in df.iterrows():
        row = {
            "date": str(r.get("TRADE_DATE", "")),
            "client_type": str(r[client_col]).strip().upper(),
        }
        for key, lookup in (
            ("idx_fut_long", "future index long"),
            ("idx_fut_short", "future index short"),
            ("stk_fut_long", "future stock long"),
            ("stk_fut_short", "future stock short"),
            ("idx_call_long", "option index call long"),
            ("idx_call_short", "option index call short"),
            ("idx_put_long", "option index put long"),
            ("idx_put_short", "option index put short"),
            ("stk_call_long", "option stock call long"),
            ("stk_call_short", "option stock call short"),
            ("stk_put_long", "option stock put long"),
            ("stk_put_short", "option stock put short"),
        ):
            col = cols.get(lookup)
            row[key] = float(r[col]) if col is not None and pd.notna(r[col]) else 0.0
        out.append(row)
    return out
