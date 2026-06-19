"""Configured dashboard jobs and persisted run state."""

from __future__ import annotations

import json
import threading
from datetime import datetime
from pathlib import Path
from typing import Any

_lock = threading.Lock()
_state_cache: dict[str, dict[str, Any]] | None = None

CONFIGURED_JOBS: list[dict[str, Any]] = [
    {
        "id": "option_chain_scraper",
        "name": "Option chain data scraper",
        "description": (
            "Fetches ATM ± strikes from Kite every 5 minutes (Mon–Fri 09:15–15:30 IST only) "
            "and appends snapshots to data/option_chain/date=YYYY-MM-DD/option_chain.csv"
        ),
        "interval_minutes": 5,
        "schedule": "Mon–Fri 09:15–15:30 IST",
        "enabled": True,
        "underlyings": ["NIFTY", "BANKNIFTY"],
        "storage_path": "data/option_chain/date={trade_date}/option_chain.csv",
    },
]


def _state_path(data_dir: Path) -> Path:
    p = data_dir / "jobs" / "job_state.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def _load_state(data_dir: Path) -> dict[str, dict[str, Any]]:
    global _state_cache
    path = _state_path(data_dir)
    if path.exists():
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(raw, dict):
                _state_cache = raw
                return raw
        except (json.JSONDecodeError, OSError):
            pass
    _state_cache = {}
    return _state_cache


def _save_state(data_dir: Path, state: dict[str, dict[str, Any]]) -> None:
    path = _state_path(data_dir)
    path.write_text(json.dumps(state, indent=2, default=str), encoding="utf-8")


def update_job_state(data_dir: Path, job_id: str, **fields: Any) -> dict[str, Any]:
    with _lock:
        state = _load_state(data_dir)
        entry = dict(state.get(job_id, {}))
        entry.update(fields)
        entry["updated_at"] = datetime.now().isoformat()
        state[job_id] = entry
        _save_state(data_dir, state)
        return entry


def get_job_states(data_dir: Path) -> list[dict[str, Any]]:
    state = _load_state(data_dir)
    rows: list[dict[str, Any]] = []
    for job in CONFIGURED_JOBS:
        st = dict(state.get(job["id"], {}))
        rows.append({**job, "state": st})
    return rows
