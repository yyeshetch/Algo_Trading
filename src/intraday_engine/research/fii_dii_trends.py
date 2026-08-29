"""
30-day FII/DII flow + participant-wise OI trends.

Aggregates the cached NSE daily downloads (see fetch.nse_public_data) into a
single JSON ready for charting:

  • fii_dii: list of last-30-day rows with cash buy/sell/net per category
  • participant_oi: list of {date, client_type, idx_fut_long, idx_fut_short, ...}
                    for Client/FII/DII/Pro
  • summary: simple net-flow rollups (5d/10d/30d, current participant tilt)
"""

from __future__ import annotations

import json
import logging
from datetime import date, datetime
from pathlib import Path
from typing import Any

from intraday_engine.core.config import Settings
from intraday_engine.fetch.nse_public_data import (
    FII_DII_DEFAULT_TRADING_DAYS,
    ensure_fii_dii_oi_history,
    fii_dii_cache_coverage,
    get_fii_dii_30d_history,
    get_fii_dii_trading_window,
    get_participant_oi_history,
    participant_oi_long_short,
)
from intraday_engine.storage.layout import fii_dii_trends_path

logger = logging.getLogger(__name__)

FII_DII_CHART_TRADING_DAYS = 6
FII_DII_HISTORY_TRADING_DAYS = FII_DII_DEFAULT_TRADING_DAYS


def _net_sum(rows: list[dict[str, Any]], key: str, n: int) -> float:
    total = 0.0
    cnt = 0
    for r in rows[:n]:
        v = r.get(key)
        if v is None:
            continue
        try:
            total += float(v)
            cnt += 1
        except Exception:
            continue
    return round(total, 2) if cnt else 0.0


def _participant_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """For each client_type return latest snapshot net-long stance for index/stock futures."""
    if not rows:
        return {}
    by_client: dict[str, dict[str, Any]] = {}
    for r in rows:
        ct = str(r.get("client_type", "")).strip().upper()
        if not ct:
            continue
        # keep most recent (rows are ordered newest first when fed)
        by_client.setdefault(ct, r)
    out: dict[str, Any] = {}
    for ct, r in by_client.items():
        idx_net_fut = float(r.get("idx_fut_long", 0.0)) - float(r.get("idx_fut_short", 0.0))
        stk_net_fut = float(r.get("stk_fut_long", 0.0)) - float(r.get("stk_fut_short", 0.0))
        idx_net_call = float(r.get("idx_call_long", 0.0)) - float(r.get("idx_call_short", 0.0))
        idx_net_put = float(r.get("idx_put_long", 0.0)) - float(r.get("idx_put_short", 0.0))
        out[ct] = {
            "as_of": r.get("date"),
            "index_fut_net": round(idx_net_fut, 0),
            "stock_fut_net": round(stk_net_fut, 0),
            "index_call_net": round(idx_net_call, 0),
            "index_put_net": round(idx_net_put, 0),
            "tilt": "long" if idx_net_fut + stk_net_fut > 0 else "short",
        }
    return out


def _fii_dii_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "fii_net_5d": _net_sum(rows, "fii_net", 5),
        "fii_net_10d": _net_sum(rows, "fii_net", 10),
        "fii_net_30d": _net_sum(rows, "fii_net", 30),
        "fii_net_45d": _net_sum(rows, "fii_net", 45),
        "dii_net_5d": _net_sum(rows, "dii_net", 5),
        "dii_net_10d": _net_sum(rows, "dii_net", 10),
        "dii_net_30d": _net_sum(rows, "dii_net", 30),
        "dii_net_45d": _net_sum(rows, "dii_net", 45),
    }


def hydrate_fii_dii_from_cache(
    data_dir: Path,
    payload: dict[str, Any],
    *,
    days: int = FII_DII_HISTORY_TRADING_DAYS,
    chart_trading_days: int = FII_DII_CHART_TRADING_DAYS,
    as_of: date | None = None,
) -> dict[str, Any]:
    """Rebuild cash FII/DII rows from trade-date cache (no NSE fetch)."""
    fii_dii_rows = get_fii_dii_30d_history(data_dir, days=days, as_of=as_of or date.today())
    fii_dii_chart = get_fii_dii_trading_window(
        data_dir, chart_trading_days, as_of=as_of or date.today()
    )
    out = dict(payload)
    out["fii_dii"] = fii_dii_rows
    out["fii_dii_chart"] = fii_dii_chart

    summary = dict(out.get("summary") or {})
    summary.update(_fii_dii_summary(fii_dii_rows))
    out["summary"] = summary

    data_status = dict(out.get("data_status") or {})
    data_status["fii_dii_rows"] = len(fii_dii_rows)
    data_status["fii_dii_chart_days"] = chart_trading_days
    coverage = dict(data_status.get("coverage") or {})
    coverage.update(fii_dii_cache_coverage(data_dir, trading_days=days))
    data_status["coverage"] = coverage
    out["data_status"] = data_status
    return out


def run_fii_dii_trends_scan(
    *,
    settings: Settings | None = None,
    days: int = FII_DII_HISTORY_TRADING_DAYS,
    trade_date: date | None = None,
) -> dict[str, Any]:
    settings = settings or Settings.from_env(underlying="NIFTY")
    td = trade_date or date.today()

    coverage = ensure_fii_dii_oi_history(settings.data_dir, trading_days=days)
    fii_dii_rows = get_fii_dii_30d_history(settings.data_dir, days=days, as_of=td)
    fii_dii_chart = get_fii_dii_trading_window(
        settings.data_dir, FII_DII_CHART_TRADING_DAYS, as_of=td
    )

    raw_oi = get_participant_oi_history(settings.data_dir, days=days)
    long_short: list[dict[str, Any]] = []
    if not raw_oi.empty:
        long_short = participant_oi_long_short(raw_oi)
    long_short_sorted = sorted(long_short, key=lambda r: str(r.get("date", "")), reverse=True)

    summary = {
        **_fii_dii_summary(fii_dii_rows),
        "participants": _participant_summary(long_short_sorted),
    }

    payload = {
        "trade_date": td.isoformat(),
        "generated_at": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
        "fii_dii": fii_dii_rows,
        "fii_dii_chart": fii_dii_chart,
        "participant_oi": long_short_sorted,
        "summary": summary,
        "data_status": {
            "fii_dii_rows": len(fii_dii_rows),
            "fii_dii_chart_days": FII_DII_CHART_TRADING_DAYS,
            "participant_oi_rows": len(long_short_sorted),
            "coverage": coverage,
        },
    }

    out_path = fii_dii_trends_path(settings.data_dir, td)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    logger.info("FII/DII trends: %d FII/DII rows, %d OI rows -> %s", len(fii_dii_rows), len(long_short_sorted), out_path)
    return payload


def load_stored_fii_dii_trends(
    data_dir: Path,
    trade_date: date,
    *,
    hydrate_cache: bool = True,
    days: int = FII_DII_HISTORY_TRADING_DAYS,
) -> dict[str, Any] | None:
    p = fii_dii_trends_path(data_dir, trade_date)
    payload: dict[str, Any] | None = None
    if p.exists():
        try:
            payload = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            payload = None
    if payload is None:
        folder = p.parent
        if folder.exists():
            files = sorted(folder.glob("fii_dii_trends_*.json"))
            if files:
                try:
                    payload = json.loads(files[-1].read_text(encoding="utf-8"))
                except Exception:
                    payload = None
    if payload is None:
        fii_dii_rows = get_fii_dii_30d_history(data_dir, days=days, as_of=trade_date)
        if not fii_dii_rows:
            return None
        payload = {
            "trade_date": trade_date.isoformat(),
            "fii_dii": fii_dii_rows,
            "participant_oi": [],
            "summary": _fii_dii_summary(fii_dii_rows),
            "data_status": {"fii_dii_rows": len(fii_dii_rows)},
        }
    if hydrate_cache:
        payload = hydrate_fii_dii_from_cache(
            data_dir, payload, days=days, as_of=trade_date
        )
    return payload
