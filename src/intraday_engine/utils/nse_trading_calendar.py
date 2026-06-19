"""NSE trading-day calendar with persisted holiday list (weekends + NSE holidays)."""

from __future__ import annotations

import json
import logging
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

from intraday_engine.storage.layout import nse_holidays_path

logger = logging.getLogger(__name__)

_HOLIDAY_API = "https://www.nseindia.com/api/holiday-master?type=trading"
_HOLIDAY_REFERER = "https://www.nseindia.com/resources/exchange-communication-holidays"
# F&O segment holidays — relevant for participant OI / FII-DII session counts.
_HOLIDAY_SEGMENT = "FO"
_CACHE_MAX_AGE_DAYS = 30


def _parse_holiday_date(raw: str) -> str | None:
    s = str(raw or "").strip()
    if not s:
        return None
    for fmt in ("%d-%b-%Y", "%d-%B-%Y"):
        try:
            return datetime.strptime(s, fmt).date().isoformat()
        except ValueError:
            continue
    return None


def load_nse_holidays(data_dir: Path) -> set[str]:
    """Return ISO holiday dates from the local cache (may be empty)."""
    path = nse_holidays_path(data_dir)
    if not path.exists():
        return set()
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return set(str(d) for d in payload.get("holidays", []) if d)
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("Could not read NSE holidays cache: %s", exc)
        return set()


def refresh_nse_holidays(data_dir: Path, *, force: bool = False) -> set[str]:
    """
    Pull trading holidays from NSE and persist under data/reference/nse/.
    Uses on-disk cache when fresh unless ``force`` is True.
    """
    path = nse_holidays_path(data_dir)
    path.parent.mkdir(parents=True, exist_ok=True)

    if not force and path.exists():
        try:
            cached = json.loads(path.read_text(encoding="utf-8"))
            updated = str(cached.get("updated_at", ""))
            if updated:
                upd = datetime.strptime(updated[:19], "%Y-%m-%dT%H:%M:%S").date()
                if (date.today() - upd).days < _CACHE_MAX_AGE_DAYS:
                    return set(str(d) for d in cached.get("holidays", []) if d)
        except (json.JSONDecodeError, OSError, ValueError):
            pass

    holidays: set[str] = set()
    segments: dict[str, list[dict[str, Any]]] = {}
    try:
        from intraday_engine.fetch.nse_public_data import _get_live

        live = _get_live()
        if live is not None:
            r = live.s.get(
                _HOLIDAY_API,
                headers={"Referer": _HOLIDAY_REFERER},
                timeout=30,
            )
            if r.status_code == 200 and r.text:
                data = r.json()
                if isinstance(data, dict):
                    for seg, rows in data.items():
                        if not isinstance(rows, list):
                            continue
                        segments[seg] = rows
                        for row in rows:
                            if not isinstance(row, dict):
                                continue
                            iso = _parse_holiday_date(row.get("tradingDate"))
                            if iso:
                                holidays.add(iso)
    except Exception as exc:
        logger.warning("NSE holiday fetch failed: %s", exc)

    if not holidays:
        return load_nse_holidays(data_dir)

    fo_holidays = sorted(
        {
            _parse_holiday_date(r.get("tradingDate"))
            for r in segments.get(_HOLIDAY_SEGMENT, [])
            if isinstance(r, dict)
        }
        - {None}
    )
    payload = {
        "updated_at": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
        "source": "nse_holiday_master",
        "segment_primary": _HOLIDAY_SEGMENT,
        "holidays": fo_holidays or sorted(holidays),
        "segments": {
            k: [
                {
                    "date": _parse_holiday_date(r.get("tradingDate")),
                    "description": r.get("description"),
                }
                for r in v
                if isinstance(r, dict)
            ]
            for k, v in segments.items()
        },
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    logger.info("Saved %d NSE trading holidays -> %s", len(payload["holidays"]), path)
    return set(payload["holidays"])


def is_nse_trading_day(d: date, holidays: set[str] | None = None, *, data_dir: Path | None = None) -> bool:
    if d.weekday() >= 5:
        return False
    if holidays is None:
        holidays = load_nse_holidays(data_dir) if data_dir else set()
    return d.isoformat() not in holidays


def recent_trading_days(
    count: int,
    as_of: date | None = None,
    data_dir: Path | None = None,
    *,
    holidays: set[str] | None = None,
) -> list[date]:
    """Return the last ``count`` NSE trading sessions on or before ``as_of`` (newest last)."""
    if count <= 0:
        return []
    if holidays is None and data_dir is not None:
        holidays = load_nse_holidays(data_dir)
    holidays = holidays or set()

    end = as_of or date.today()
    out: list[date] = []
    cursor = end
    guard = 0
    while len(out) < count and guard < count * 4 + 60:
        if is_nse_trading_day(cursor, holidays):
            out.append(cursor)
        cursor -= timedelta(days=1)
        guard += 1
    return list(reversed(out))
