"""
Sector Relative Strength vs NIFTY 50 — group NIFTY 500 stocks by sector
(Industry column of ind_nifty500list.csv) and rank sectors by how they are
performing against NIFTY 50 intraday, then surface the strong "runners" and
weak "laggards" inside each sector so you can ride the leaders.

Reuses the intraday RS engine (data/NIFTY500/*_15Min.csv + cached NIFTY 50
15-min bars). No per-stock historical_data calls are made here.

Per stock:
  • pct_today  = (last_close / first_open - 1) * 100
  • excess_pct = stock pct_today - nifty pct_today   (intraday RS, % points)

Per sector (aggregated over members with data):
  • avg_excess_pct   = mean member excess vs NIFTY
  • median_excess_pct
  • breadth_pct       = % of members beating NIFTY (advancers / members)
  • rs_line           = average member intraday RS path (100 baseline)
  • runners           = strongest members (excess desc)
  • laggards          = weakest members (excess asc)

Outputs sectors sorted strongest → weakest by avg_excess_pct.
"""

from __future__ import annotations

import csv
import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from intraday_engine.core.config import Settings
from intraday_engine.research.intraday_relative_strength_scanner import (
    _compute_row,
    _load_nifty_15min,
    _resolve_stock_aligned_session_day,
    _slice_for_session,
)
from intraday_engine.research.nifty500_accumulation_scanner import load_nifty500_symbols
from intraday_engine.storage.layout import sector_relative_strength_path
from intraday_engine.storage.nifty500_csv import read_nifty500_symbol_ohlcv

logger = logging.getLogger(__name__)

UNKNOWN_SECTOR = "Unclassified"
SECTOR_RUNNERS_TOP_N = 5
RS_LINE_MAX_POINTS = 26


@dataclass
class SectorMemberRow:
    stock: str
    pct_today: float
    excess_pct: float
    last_price: float
    rs_line: list[float] = field(default_factory=list)


@dataclass
class SectorRsRow:
    sector: str
    members_with_data: int
    members_total: int
    advancers: int
    decliners: int
    breadth_pct: float
    avg_excess_pct: float
    median_excess_pct: float
    avg_pct_today: float
    leaders_excess: float           # best member excess (the top runner)
    laggards_excess: float          # worst member excess
    rs_line: list[float] = field(default_factory=list)
    runners: list[dict[str, Any]] = field(default_factory=list)
    laggards: list[dict[str, Any]] = field(default_factory=list)


def load_sector_map(data_dir: Path) -> dict[str, str]:
    """
    Return {SYMBOL: NSE sector index name} from live NSE sector constituent pages
    (https://www.nseindia.com/market-data/live-market-indices).

    Falls back to ind_nifty500list.csv Industry column when NSE map is unavailable.
    """
    try:
        from intraday_engine.fetch.nse_market_indices import load_nse_sector_stock_map

        nse_map = load_nse_sector_stock_map(data_dir)
        if nse_map:
            return nse_map
    except Exception as e:
        logger.warning("NSE sector map unavailable: %s", e)

    csv_ref = data_dir / "reference" / "ind_nifty500list.csv"
    mapping: dict[str, str] = {}
    if not csv_ref.exists():
        logger.warning("Sector map source not found: %s", csv_ref)
        return mapping
    try:
        with csv_ref.open("r", encoding="utf-8-sig", errors="replace") as fh:
            reader = csv.DictReader(fh)
            if not reader.fieldnames:
                return mapping
            fh_map = {k.strip().lower(): k for k in reader.fieldnames if k}
            sym_key = fh_map.get("symbol")
            ind_key = fh_map.get("industry") or fh_map.get("sector")
            if not sym_key:
                return mapping
            for row in reader:
                sym = (row.get(sym_key) or "").strip().upper()
                if not sym:
                    continue
                sector = (row.get(ind_key) or "").strip() if ind_key else ""
                mapping[sym] = sector or UNKNOWN_SECTOR
    except Exception as e:
        logger.warning("Could not parse sector map: %s", e)
    return mapping


def _avg_rs_line(member_lines: list[list[float]]) -> list[float]:
    """Average member RS lines aligned from the END (latest bars align)."""
    lines = [ln for ln in member_lines if ln]
    if not lines:
        return []
    max_len = min(RS_LINE_MAX_POINTS, max(len(ln) for ln in lines))
    sums = [0.0] * max_len
    counts = [0] * max_len
    for ln in lines:
        tail = ln[-max_len:]
        offset = max_len - len(tail)
        for i, v in enumerate(tail):
            sums[offset + i] += v
            counts[offset + i] += 1
    return [round(sums[i] / counts[i], 3) for i in range(max_len) if counts[i] > 0]


def _median(values: list[float]) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    n = len(s)
    mid = n // 2
    if n % 2:
        return s[mid]
    return (s[mid - 1] + s[mid]) / 2.0


def run_sector_relative_strength_scan(
    *,
    settings: Settings | None = None,
    symbols_file: Path | None = None,
    trade_date: date | None = None,
    min_bars: int = 2,
    runners_top_n: int = SECTOR_RUNNERS_TOP_N,
) -> dict[str, Any]:
    settings = settings or Settings.from_env(underlying="NIFTY")
    requested = trade_date or date.today()

    nifty_full = _load_nifty_15min(settings, requested)
    symbols = load_nifty500_symbols(symbols_file, settings.data_dir)
    sector_map = load_sector_map(settings.data_dir)
    session_day = _resolve_stock_aligned_session_day(
        settings.data_dir, symbols, nifty_full, requested, min_bars=min_bars
    )
    nifty_today = _slice_for_session(nifty_full, session_day)

    if nifty_today.empty or len(nifty_today) < min_bars:
        payload = {
            "trade_date": requested.isoformat(),
            "session_day": session_day.isoformat() if session_day else None,
            "generated_at": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
            "nifty": None,
            "sectors": [],
            "scanned": 0,
            "message": (
                "Could not load NIFTY 50 15-min bars for the session. "
                "Ensure Kite credentials are configured."
            ),
        }
        _persist(payload, settings.data_dir, requested)
        return payload

    n_open = float(nifty_today["open"].iloc[0])
    n_last = float(nifty_today["close"].iloc[-1])
    nifty_pct = (n_last / n_open - 1.0) * 100.0 if n_open > 0 else 0.0
    nifty_meta = {
        "open": round(n_open, 2),
        "last": round(n_last, 2),
        "pct_today": round(nifty_pct, 3),
        "bars": int(len(nifty_today)),
        "last_bar_ts": pd.to_datetime(nifty_today["date"].iloc[-1]).strftime("%H:%M"),
    }

    # Compute per-stock RS, grouped by sector.
    sector_members: dict[str, list[SectorMemberRow]] = {}
    sector_totals: dict[str, int] = {}
    scanned = 0
    passed = 0

    for sym in symbols:
        sector = sector_map.get(sym, UNKNOWN_SECTOR)
        sector_totals[sector] = sector_totals.get(sector, 0) + 1
        scanned += 1
        try:
            df = read_nifty500_symbol_ohlcv(settings.data_dir, sym, "15Min")
            if df.empty:
                continue
            today = _slice_for_session(df, session_day)
            if today.empty or len(today) < min_bars:
                continue
            row = _compute_row(sym, today, nifty_today, nifty_pct)
            if row is None:
                continue
            passed += 1
            sector_members.setdefault(sector, []).append(
                SectorMemberRow(
                    stock=sym,
                    pct_today=row.pct_today,
                    excess_pct=row.excess_pct,
                    last_price=row.last_price,
                    rs_line=row.rs_line,
                )
            )
        except Exception as e:
            logger.debug("Sector RS %s err: %s", sym, e)
            continue

    sector_rows: list[SectorRsRow] = []
    for sector, members in sector_members.items():
        if not members:
            continue
        excesses = [m.excess_pct for m in members]
        advancers = sum(1 for e in excesses if e > 0)
        decliners = sum(1 for e in excesses if e < 0)
        sorted_by_excess = sorted(members, key=lambda m: m.excess_pct, reverse=True)
        runners = sorted_by_excess[:runners_top_n]
        laggards = sorted_by_excess[-runners_top_n:][::-1] if len(sorted_by_excess) > runners_top_n else []
        sector_rows.append(
            SectorRsRow(
                sector=sector,
                members_with_data=len(members),
                members_total=sector_totals.get(sector, len(members)),
                advancers=advancers,
                decliners=decliners,
                breadth_pct=round(advancers / len(members) * 100.0, 1),
                avg_excess_pct=round(sum(excesses) / len(excesses), 3),
                median_excess_pct=round(_median(excesses), 3),
                avg_pct_today=round(sum(m.pct_today for m in members) / len(members), 3),
                leaders_excess=round(sorted_by_excess[0].excess_pct, 3),
                laggards_excess=round(sorted_by_excess[-1].excess_pct, 3),
                rs_line=_avg_rs_line([m.rs_line for m in members]),
                runners=[asdict(m) for m in runners],
                laggards=[asdict(m) for m in laggards],
            )
        )

    # Strongest sector first.
    sector_rows.sort(key=lambda s: s.avg_excess_pct, reverse=True)

    session_note = None
    if session_day != requested:
        session_note = (
            f"Using {session_day.isoformat()} (latest session with NIFTY500 15-min bars). "
            f"Today's stock 15-min cache is not ready yet — run 30-min Analysis during market hours."
        )

    payload = {
        "trade_date": requested.isoformat(),
        "session_day": session_day.isoformat(),
        "generated_at": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
        "nifty": nifty_meta,
        "sectors": [asdict(s) for s in sector_rows],
        "sector_count": len(sector_rows),
        "scanned": scanned,
        "passed": passed,
        "session_note": session_note,
    }
    _persist(payload, settings.data_dir, requested)
    logger.info(
        "Sector RS (%s): %d sectors ranked (NIFTY %s%%), %d/%d stocks with data",
        session_day,
        len(sector_rows),
        f"{nifty_pct:+.2f}",
        passed,
        scanned,
    )
    return payload


def _persist(payload: dict[str, Any], data_dir: Path, trade_date: date) -> None:
    out_path = sector_relative_strength_path(data_dir, trade_date)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_stored_sector_relative_strength(data_dir: Path, trade_date: date) -> dict[str, Any] | None:
    p = sector_relative_strength_path(data_dir, trade_date)
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            return None
    folder = p.parent
    if not folder.exists():
        return None
    files = sorted(folder.glob("sector_rs_*.json"))
    if not files:
        return None
    try:
        return json.loads(files[-1].read_text(encoding="utf-8"))
    except Exception:
        return None
