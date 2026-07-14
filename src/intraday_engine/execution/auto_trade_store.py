"""Persist auto-trade toggles, defaults, and executed-signal dedup keys."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Any


def _path(data_dir: Path) -> Path:
    return data_dir / "auto_trade_config.json"


def _default() -> dict[str, Any]:
    return {
        "auto_trade_enabled": True,
        "rr_trail_enabled": True,
        "lots": 1,
        "executed_keys": [],
        "last_run_at": None,
        "last_run_status": None,
        "last_executions": [],
        "trade_date": None,
    }


def load(data_dir: Path) -> dict[str, Any]:
    p = _path(data_dir)
    if not p.exists():
        return _default()
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        base = _default()
        base.update(data)
        return base
    except Exception:
        return _default()


def save(data_dir: Path, data: dict[str, Any]) -> None:
    p = _path(data_dir)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, indent=2), encoding="utf-8")


def get_config(data_dir: Path) -> dict[str, Any]:
    return load(data_dir)


def update_config(data_dir: Path, **kwargs: Any) -> dict[str, Any]:
    data = load(data_dir)
    for k, v in kwargs.items():
        if k in _default():
            data[k] = v
    save(data_dir, data)
    return data


def _roll_trade_date(data: dict[str, Any], today: str) -> None:
    if data.get("trade_date") != today:
        data["trade_date"] = today
        data["executed_keys"] = []
        data["last_executions"] = []


def mark_executed(data_dir: Path, key: str, record: dict[str, Any]) -> None:
    data = load(data_dir)
    today = date.today().isoformat()
    _roll_trade_date(data, today)
    keys = list(data.get("executed_keys") or [])
    if key not in keys:
        keys.append(key)
    data["executed_keys"] = keys[-200:]
    execs = list(data.get("last_executions") or [])
    execs.insert(0, record)
    data["last_executions"] = execs[:20]
    save(data_dir, data)


def is_executed(data_dir: Path, key: str) -> bool:
    data = load(data_dir)
    today = date.today().isoformat()
    if data.get("trade_date") != today:
        return False
    return key in (data.get("executed_keys") or [])


def record_run(data_dir: Path, status: dict[str, Any]) -> None:
    data = load(data_dir)
    today = date.today().isoformat()
    _roll_trade_date(data, today)
    from datetime import datetime
    data["last_run_at"] = datetime.now().isoformat()
    data["last_run_status"] = status
    save(data_dir, data)
