"""Manual strike-out of index BUY/SELL signals (monitoring) without waiting for SL."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from intraday_engine.storage.layout import normalize_underlying


def _path(root_data_dir: Path) -> Path:
    return root_data_dir / "reference" / "index_signal_invalidations.json"


def _normalize_key(timestamp: str, signal: str) -> str:
    return f"{str(timestamp).strip()}|{str(signal).strip().upper()}"


def load_all(root_data_dir: Path) -> dict[str, Any]:
    p = _path(root_data_dir)
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _save(root_data_dir: Path, data: dict[str, Any]) -> None:
    p = _path(root_data_dir)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, indent=2), encoding="utf-8")


def load_invalidated_keys(root_data_dir: Path, underlying: str | None, trade_date_iso: str) -> set[str]:
    u = normalize_underlying(underlying)
    root = load_all(root_data_dir)
    day = root.get(u, {}).get(trade_date_iso, {})
    if not isinstance(day, dict):
        return set()
    return {str(k) for k in day.keys()}


def is_invalidated(root_data_dir: Path, underlying: str | None, trade_date_iso: str, timestamp: str, signal: str) -> bool:
    key = _normalize_key(timestamp, signal)
    return key in load_invalidated_keys(root_data_dir, underlying, trade_date_iso)


def invalidate(
    root_data_dir: Path,
    underlying: str | None,
    trade_date_iso: str,
    timestamp: str,
    signal: str,
    note: str | None = None,
) -> str:
    u = normalize_underlying(underlying)
    key = _normalize_key(timestamp, signal)
    data = load_all(root_data_dir)
    if u not in data:
        data[u] = {}
    if trade_date_iso not in data[u] or not isinstance(data[u][trade_date_iso], dict):
        data[u][trade_date_iso] = {}
    entry: dict[str, Any] = {
        "at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    if note:
        entry["note"] = note[:500]
    data[u][trade_date_iso][key] = entry
    _save(root_data_dir, data)
    return key


def reinstate(root_data_dir: Path, underlying: str | None, trade_date_iso: str, timestamp: str, signal: str) -> bool:
    u = normalize_underlying(underlying)
    key = _normalize_key(timestamp, signal)
    data = load_all(root_data_dir)
    day = data.get(u, {}).get(trade_date_iso)
    if not isinstance(day, dict) or key not in day:
        return False
    del day[key]
    if not day:
        data[u].pop(trade_date_iso, None)
    if data.get(u) == {}:
        data.pop(u, None)
    _save(root_data_dir, data)
    return True
