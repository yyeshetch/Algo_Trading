"""
NSE live market indices — sector index performance and constituent mapping.

Source page: https://www.nseindia.com/market-data/live-market-indices

APIs (via jugaad-data NSELive session):
  • /api/allIndices — live sector index % change vs open
  • /api/equity-stock-indices?index=… — index constituents (use indexSymbol when
    it differs from the display name, e.g. NIFTY PVT BANK)
"""

from __future__ import annotations

import json
import logging
import time
from datetime import date, datetime
from pathlib import Path
from typing import Any

from intraday_engine.fetch.nse_public_data import _get_live, _load_cache_text, _save_cache
from intraday_engine.storage.layout import (
    nse_live_all_indices_cache_path,
    nse_sector_rotation_snapshots_path,
    nse_sector_stock_map_path,
)

logger = logging.getLogger(__name__)

LIVE_MARKET_INDICES_URL = "https://www.nseindia.com/market-data/live-market-indices"
LIVE_MARKET_INDICES_REFERER = LIVE_MARKET_INDICES_URL
ALL_INDICES_API = "https://www.nseindia.com/api/allIndices"
EQUITY_STOCK_INDICES_API = "https://www.nseindia.com/api/equity-stock-indices"

SECTORAL_KEY = "SECTORAL INDICES"
# NIFTY BANK lives under derivatives on the live page but is the primary bank sector index.
EXTRA_ROTATION_INDICES = ("NIFTY BANK",)
MAP_CACHE_MAX_AGE_SEC = 24 * 3600
INDICES_CACHE_MAX_AGE_SEC = 120
CONSTITUENT_FETCH_DELAY_SEC = 0.32


def _warm_live_session(live: Any) -> None:
    live.s.get(LIVE_MARKET_INDICES_URL, timeout=20)


def _api_get(live: Any, url: str, *, params: dict[str, str] | None = None) -> dict[str, Any] | None:
    try:
        r = live.s.get(
            url,
            params=params or {},
            headers={"Referer": LIVE_MARKET_INDICES_REFERER},
            timeout=25,
        )
        if r.status_code != 200:
            logger.debug("NSE API %s -> %s", url, r.status_code)
            return None
        return r.json()
    except Exception as e:
        logger.debug("NSE API %s failed: %s", url, e)
        return None


def _index_query_name(row: dict[str, Any]) -> str:
    return (row.get("indexSymbol") or row.get("index") or "").strip()


def _display_name(row: dict[str, Any]) -> str:
    return (row.get("index") or row.get("indexSymbol") or "").strip()


def _parse_constituent_symbols(payload: dict[str, Any] | None) -> list[str]:
    if not payload:
        return []
    out: list[str] = []
    for row in payload.get("data") or []:
        if int(row.get("priority") or 0) != 0:
            continue
        sym = (row.get("symbol") or "").strip().upper()
        if not sym or sym.startswith("NIFTY"):
            continue
        out.append(sym)
    return out


def fetch_all_indices_live(
    data_dir: Path,
    *,
    force_refresh: bool = False,
    max_age_sec: int = INDICES_CACHE_MAX_AGE_SEC,
) -> dict[str, Any] | None:
    """Fetch /api/allIndices with a short-lived cache."""
    cache_path = nse_live_all_indices_cache_path(data_dir)
    if not force_refresh and cache_path.exists():
        try:
            cached = json.loads(cache_path.read_text(encoding="utf-8"))
            fetched_at = cached.get("fetched_at_epoch") or 0
            if time.time() - float(fetched_at) < max_age_sec:
                return cached
        except Exception:
            pass

    live = _get_live()
    if not live:
        return None
    _warm_live_session(live)
    payload = _api_get(live, ALL_INDICES_API)
    if not payload:
        stale = _load_cache_text(data_dir, "live_all_indices.json")
        if stale:
            try:
                return json.loads(stale)
            except Exception:
                pass
        return None

    wrapped = {
        "fetched_at": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
        "fetched_at_epoch": time.time(),
        "source_url": LIVE_MARKET_INDICES_URL,
        **payload,
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(wrapped, indent=2), encoding="utf-8")
    return wrapped


def sectoral_index_rows(all_indices: dict[str, Any]) -> list[dict[str, Any]]:
    rows = [
        row for row in (all_indices.get("data") or [])
        if (row.get("key") or "") == SECTORAL_KEY
    ]
    seen = {_display_name(r) for r in rows}
    for name in EXTRA_ROTATION_INDICES:
        if name in seen:
            continue
        for row in all_indices.get("data") or []:
            if _display_name(row) == name:
                rows.append(row)
                break
    return rows


def fetch_index_constituents(live: Any, index_row: dict[str, Any]) -> list[str]:
    query = _index_query_name(index_row)
    if not query:
        return []
    payload = _api_get(live, EQUITY_STOCK_INDICES_API, params={"index": query})
    syms = _parse_constituent_symbols(payload)
    if syms:
        return syms
    # Retry with display name when indexSymbol differs.
    alt = _display_name(index_row)
    if alt and alt != query:
        payload = _api_get(live, EQUITY_STOCK_INDICES_API, params={"index": alt})
        return _parse_constituent_symbols(payload)
    return []


def _map_build_order(index_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Broad indices first; specific indices overwrite on collision."""
    broad_keywords = (
        "FINANCIAL SERVICES",
        "MIDSMALL",
        "NIFTY500",
        "REITS",
        "25/50",
    )

    def sort_key(row: dict[str, Any]) -> tuple[int, str]:
        name = _display_name(row).upper()
        is_broad = any(k in name for k in broad_keywords)
        return (0 if is_broad else 1, name)

    return sorted(index_rows, key=sort_key)


def build_nse_sector_stock_map(
    data_dir: Path,
    *,
    force_refresh: bool = False,
) -> dict[str, str]:
    """
    Build {SYMBOL: NSE sector index name} from live sector index constituent pages.
    Cached under data/reference/nse/nse_sector_stock_map.json.
    """
    out_path = nse_sector_stock_map_path(data_dir)
    if not force_refresh and out_path.exists():
        try:
            cached = json.loads(out_path.read_text(encoding="utf-8"))
            fetched_at = cached.get("fetched_at_epoch") or 0
            mapping = cached.get("map") or {}
            if mapping and time.time() - float(fetched_at) < MAP_CACHE_MAX_AGE_SEC:
                return {k.upper(): v for k, v in mapping.items()}
        except Exception:
            pass

    all_indices = fetch_all_indices_live(data_dir, force_refresh=True)
    if not all_indices:
        logger.warning("Could not fetch NSE allIndices for sector map")
        return _load_sector_map_file(out_path)

    live = _get_live()
    if not live:
        return _load_sector_map_file(out_path)

    index_rows = sectoral_index_rows(all_indices)
    ordered = _map_build_order(index_rows)

    symbol_to_sector: dict[str, str] = {}
    sector_members: dict[str, list[str]] = {}

    for row in ordered:
        sector = _display_name(row)
        if not sector:
            continue
        time.sleep(CONSTITUENT_FETCH_DELAY_SEC)
        members = fetch_index_constituents(live, row)
        sector_members[sector] = members
        for sym in members:
            symbol_to_sector[sym] = sector

    payload = {
        "fetched_at": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
        "fetched_at_epoch": time.time(),
        "source_url": LIVE_MARKET_INDICES_URL,
        "sector_count": len(sector_members),
        "mapped_symbols": len(symbol_to_sector),
        "sectors": {
            sec: {
                "members": len(sector_members.get(sec, [])),
                "index_query": next(
                    (_index_query_name(r) for r in index_rows if _display_name(r) == sec),
                    "",
                ),
            }
            for sec in sector_members
        },
        "map": symbol_to_sector,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    logger.info(
        "NSE sector map: %d symbols across %d sector indices",
        len(symbol_to_sector),
        len(sector_members),
    )
    return symbol_to_sector


def _load_sector_map_file(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    try:
        cached = json.loads(path.read_text(encoding="utf-8"))
        return {k.upper(): v for k, v in (cached.get("map") or {}).items()}
    except Exception:
        return {}


def load_nse_sector_stock_map(data_dir: Path, *, refresh: bool = False) -> dict[str, str]:
    if refresh:
        return build_nse_sector_stock_map(data_dir, force_refresh=True)
    out_path = nse_sector_stock_map_path(data_dir)
    mapping = _load_sector_map_file(out_path)
    if mapping:
        return mapping
    return build_nse_sector_stock_map(data_dir, force_refresh=True)


def _sector_live_rows(all_indices: dict[str, Any]) -> list[dict[str, Any]]:
    nifty_pct = None
    for row in all_indices.get("data") or []:
        if _display_name(row) == "NIFTY 50":
            nifty_pct = float(row.get("percentChange") or 0)
            break
    if nifty_pct is None:
        return []

    out: list[dict[str, Any]] = []
    for row in sectoral_index_rows(all_indices):
        sector = _display_name(row)
        pct = float(row.get("percentChange") or 0)
        out.append({
            "sector": sector,
            "index_symbol": _index_query_name(row),
            "pct_today": round(pct, 3),
            "excess_pct": round(pct - nifty_pct, 3),
            "last": float(row.get("last") or 0),
            "open": float(row.get("open") or 0),
            "variation": float(row.get("variation") or 0),
        })
    out.sort(key=lambda r: r["excess_pct"], reverse=True)
    for i, row in enumerate(out):
        row["rank"] = i + 1
    return out


def _load_snapshots(data_dir: Path, trade_date: date) -> dict[str, Any]:
    p = nse_sector_rotation_snapshots_path(data_dir, trade_date)
    if not p.exists():
        return {"trade_date": trade_date.isoformat(), "snapshots": []}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {"trade_date": trade_date.isoformat(), "snapshots": []}


def _save_snapshots(data_dir: Path, trade_date: date, doc: dict[str, Any]) -> None:
    p = nse_sector_rotation_snapshots_path(data_dir, trade_date)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(doc, indent=2), encoding="utf-8")


def append_sector_rotation_snapshot(
    data_dir: Path,
    trade_date: date,
    sectors: list[dict[str, Any]],
    *,
    nifty_pct: float,
) -> dict[str, Any]:
    doc = _load_snapshots(data_dir, trade_date)
    ts = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
    snapshots = doc.setdefault("snapshots", [])
    if snapshots:
        last = snapshots[-1]
        if last.get("nifty_pct") == round(nifty_pct, 3):
            last_sectors = {s["sector"]: s.get("excess_pct") for s in last.get("sectors") or []}
            cur_sectors = {s["sector"]: s.get("excess_pct") for s in sectors}
            if last_sectors == cur_sectors:
                return doc

    snapshots.append({
        "ts": ts,
        "nifty_pct": round(nifty_pct, 3),
        "sectors": sectors,
    })
    _save_snapshots(data_dir, trade_date, doc)
    return doc


def _snapshot_to_checkpoint(label: str, snap: dict[str, Any]) -> dict[str, Any]:
    ts = snap.get("ts") or ""
    bar_time = ts[11:16] if len(ts) >= 16 else "—"
    ranked = []
    for row in sorted(snap.get("sectors") or [], key=lambda r: r.get("rank") or 999):
        ranked.append({
            "sector": row["sector"],
            "rank": row.get("rank"),
            "avg_excess_pct": row.get("excess_pct", 0),
            "pct_today": row.get("pct_today", 0),
        })
    return {"label": label, "bar_time": bar_time, "sectors": ranked}


def checkpoints_from_snapshots(snapshots: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not snapshots:
        return []
    if len(snapshots) == 1:
        return [_snapshot_to_checkpoint("now", snapshots[0])]
    open_snap = snapshots[0]
    mid_snap = snapshots[len(snapshots) // 2]
    now_snap = snapshots[-1]
    out = [_snapshot_to_checkpoint("open", open_snap)]
    if mid_snap is not open_snap:
        out.append(_snapshot_to_checkpoint("mid", mid_snap))
    if now_snap is not mid_snap:
        out.append(_snapshot_to_checkpoint("now", now_snap))
    elif out[-1]["label"] != "now":
        out.append(_snapshot_to_checkpoint("now", now_snap))
    return out


def live_sector_rotation_payload(
    data_dir: Path,
    trade_date: date,
    *,
    force_refresh: bool = False,
) -> dict[str, Any]:
    """Build sector rotation + RRG inputs from NSE live sector indices."""
    all_indices = fetch_all_indices_live(data_dir, force_refresh=force_refresh)
    if not all_indices:
        return {
            "trade_date": trade_date.isoformat(),
            "generated_at": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
            "source": "nse_live_market_indices",
            "source_url": LIVE_MARKET_INDICES_URL,
            "checkpoints": [],
            "rotations": [],
            "headline": "Could not fetch NSE live sector indices",
            "message": "NSE /api/allIndices unavailable — retry during market hours.",
        }

    sectors = _sector_live_rows(all_indices)
    nifty_pct = 0.0
    nifty_last = None
    for row in all_indices.get("data") or []:
        if _display_name(row) == "NIFTY 50":
            nifty_pct = float(row.get("percentChange") or 0)
            nifty_last = float(row.get("last") or 0)
            break

    snap_doc = append_sector_rotation_snapshot(
        data_dir, trade_date, sectors, nifty_pct=nifty_pct
    )
    checkpoints = checkpoints_from_snapshots(snap_doc.get("snapshots") or [])

    # Import here to avoid circular import at module load.
    from intraday_engine.research.sector_rotation_scanner import (
        _build_rrg_points,
        _rotation_events,
    )

    rotations = _rotation_events(checkpoints)
    into = [r for r in rotations if r["direction"] == "into"]
    out = [r for r in rotations if r["direction"] == "out_of"]
    rrg_points = _build_rrg_points(checkpoints)

    n_snaps = len(snap_doc.get("snapshots") or [])
    if into:
        headline = (
            f"Rotation active — {into[0]['sector']} gaining "
            f"(#{into[0]['rank_from']}→#{into[0]['rank_to']}, NSE live)"
        )
    elif out:
        headline = (
            f"Sectors weakening — {out[0]['sector']} falling "
            f"(#{out[0]['rank_from']}→#{out[0]['rank_to']}, NSE live)"
        )
    elif sectors:
        headline = f"Stable leadership — leading: {sectors[0]['sector']} ({sectors[0]['excess_pct']:+.2f}% vs NIFTY)"
    else:
        headline = "No NSE sector index data"

    trail_note = (
        f"{n_snaps} intraday snapshot(s) — refresh during session for open→mid→now trail"
        if n_snaps < 3
        else f"{n_snaps} intraday snapshots — trail uses first / mid / latest refresh"
    )

    return {
        "trade_date": trade_date.isoformat(),
        "session_day": trade_date.isoformat(),
        "generated_at": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
        "source": "nse_live_market_indices",
        "source_url": LIVE_MARKET_INDICES_URL,
        "indices_fetched_at": all_indices.get("fetched_at"),
        "nifty": {
            "pct_today": round(nifty_pct, 3),
            "last": nifty_last,
        },
        "live_sectors": sectors,
        "snapshot_count": n_snaps,
        "trail_note": trail_note,
        "checkpoints": checkpoints,
        "rotations": rotations,
        "rotating_into": into[:8],
        "rotating_out_of": out[:8],
        "headline": headline,
        "rrg": {
            "pivot": 100.0,
            "x_label": "RS-Ratio (100 = NIFTY parity, NSE live %)",
            "y_label": "RS-Momentum (excess change open→now snapshot Δ)",
            "points": rrg_points,
        },
    }
