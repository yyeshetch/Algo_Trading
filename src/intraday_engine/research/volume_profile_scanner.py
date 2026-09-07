"""EOD volume profile scanner — daily, weekly, monthly important levels for NIFTY 500."""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from intraday_engine.analysis.volume_profile import multi_period_volume_profiles
from intraday_engine.core.config import Settings
from intraday_engine.core.tunables import get_int
from intraday_engine.research.nifty500_accumulation_scanner import load_nifty500_symbols
from intraday_engine.storage.layout import volume_profile_path
from intraday_engine.storage.nifty500_csv import read_nifty500_symbol_ohlcv

logger = logging.getLogger(__name__)

MIN_DAILY_BARS = 60


@dataclass
class VolumeProfileRow:
    stock: str
    close: float
    inside_daily_va: bool = False
    near_poc_pct: float = 0.0
    score: float = 0.0
    daily: dict[str, Any] = field(default_factory=dict)
    weekly: dict[str, Any] = field(default_factory=dict)
    monthly: dict[str, Any] = field(default_factory=dict)
    important_levels: list[dict[str, Any]] = field(default_factory=list)


def _score_row(profiles: dict[str, Any]) -> float:
    score = 0.0
    periods = profiles.get("periods") or {}
    for key, weight in (("daily", 30), ("weekly", 35), ("monthly", 35)):
        p = periods.get(key) or {}
        if not p:
            continue
        pos = str(p.get("position", ""))
        if pos == "at_poc":
            score += weight * 0.9
        elif pos == "inside_value_area":
            score += weight * 0.7
        elif p.get("inside_value_area"):
            score += weight * 0.5
        dist = abs(float(p.get("dist_poc_pct") or 99))
        if dist <= 1.0:
            score += weight * 0.2
        elif dist <= 2.5:
            score += weight * 0.1
    return round(min(100.0, score), 1)


def _analyze_symbol(df: pd.DataFrame, symbol: str) -> VolumeProfileRow | None:
    if len(df) < MIN_DAILY_BARS:
        return None
    close = float(df["close"].iloc[-1])
    if close <= 0:
        return None

    profiles = multi_period_volume_profiles(daily_df=df, spot=close)
    periods = profiles.get("periods") or {}
    daily = periods.get("daily") or {}
    weekly = periods.get("weekly") or {}
    monthly = periods.get("monthly") or {}

    near_poc = abs(float(daily.get("dist_poc_pct") or 999))
    return VolumeProfileRow(
        stock=symbol,
        close=round(close, 2),
        inside_daily_va=bool(daily.get("inside_value_area")),
        near_poc_pct=round(near_poc, 2),
        score=_score_row(profiles),
        daily=daily,
        weekly=weekly,
        monthly=monthly,
        important_levels=profiles.get("merged_levels") or [],
    )


def run_volume_profile_scan(
    *,
    settings: Settings | None = None,
    symbols_file: Path | None = None,
    trade_date: date | None = None,
    top_n: int = 50,
    symbol_limit: int | None = None,
) -> dict[str, Any]:
    settings = settings or Settings.from_env(underlying="NIFTY")
    td = trade_date or date.today()
    symbols = load_nifty500_symbols(symbols_file, settings.data_dir)
    if symbol_limit:
        symbols = symbols[:symbol_limit]

    rows: list[VolumeProfileRow] = []
    skipped = 0
    for sym in symbols:
        df = read_nifty500_symbol_ohlcv(settings.data_dir, sym, "1D")
        if df.empty:
            skipped += 1
            continue
        try:
            row = _analyze_symbol(df, sym)
        except Exception as exc:
            logger.debug("Volume profile %s: %s", sym, exc)
            skipped += 1
            continue
        if row is None:
            skipped += 1
            continue
        rows.append(row)

    rows.sort(key=lambda r: r.score, reverse=True)
    top = rows[:top_n]

    payload = {
        "trade_date": td.isoformat(),
        "generated_at": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
        "scanned": len(symbols),
        "passed": len(rows),
        "skipped": skipped,
        "periods": ["daily", "weekly", "monthly"],
        "rows": [asdict(r) for r in top],
    }
    out = volume_profile_path(settings.data_dir, td)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    logger.info("Volume profile: %d passed / %d scanned → %s", len(rows), len(symbols), out)
    return payload


def load_stored_volume_profile(data_dir: Path, trade_date: date) -> dict[str, Any] | None:
    p = volume_profile_path(data_dir, trade_date)
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            return None
    folder = p.parent
    if not folder.exists():
        return None
    files = sorted(folder.glob("volume_profile_*.json"))
    if not files:
        return None
    try:
        return json.loads(files[-1].read_text(encoding="utf-8"))
    except Exception:
        return None


def volume_profile_for_symbol(
    data_dir: Path,
    symbol: str,
    *,
    spot: float | None = None,
    intraday_df: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """On-demand multi-period profile for one symbol (API / stock detail)."""
    daily = read_nifty500_symbol_ohlcv(data_dir, symbol, "1D")
    if daily.empty:
        return {}
    last = spot if spot and spot > 0 else float(daily["close"].iloc[-1])
    return multi_period_volume_profiles(
        intraday_df=intraday_df,
        daily_df=daily,
        spot=last,
    )
