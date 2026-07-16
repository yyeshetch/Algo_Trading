"""Persist per-strategy ON/OFF toggles for dashboard display and auto-trade."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from intraday_engine.research.options_trading_signals import options_signal_definitions

RADAR_STRATEGIES: list[dict[str, str]] = [
    {"id": "gamma_ignition", "label": "Expiry Gamma Ignition"},
    {"id": "trend_ignition", "label": "Trend Ignition"},
]


def strategy_catalog() -> list[dict[str, str]]:
    out = [{"id": d["id"], "label": d["label"]} for d in options_signal_definitions()]
    out.extend(RADAR_STRATEGIES)
    return out


def _default_enabled() -> dict[str, bool]:
    return {s["id"]: True for s in strategy_catalog()}


def _path(data_dir: Path) -> Path:
    return data_dir / "strategy_config.json"


def load(data_dir: Path) -> dict[str, Any]:
    defaults = _default_enabled()
    p = _path(data_dir)
    if not p.exists():
        return {"strategies": defaults}
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        strategies = defaults.copy()
        strategies.update(data.get("strategies") or {})
        return {"strategies": strategies}
    except Exception:
        return {"strategies": defaults}


def save(data_dir: Path, data: dict[str, Any]) -> None:
    p = _path(data_dir)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, indent=2), encoding="utf-8")


def get_enabled_map(data_dir: Path) -> dict[str, bool]:
    return load(data_dir)["strategies"]


def is_enabled(data_dir: Path, strategy_id: str) -> bool:
    return get_enabled_map(data_dir).get(strategy_id, True)


def list_strategies(data_dir: Path) -> list[dict[str, Any]]:
    enabled = get_enabled_map(data_dir)
    return [
        {"id": s["id"], "label": s["label"], "enabled": enabled.get(s["id"], True)}
        for s in strategy_catalog()
    ]


def update_strategy(data_dir: Path, strategy_id: str, enabled: bool) -> list[dict[str, Any]]:
    ids = {s["id"] for s in strategy_catalog()}
    if strategy_id not in ids:
        raise ValueError(f"Unknown strategy: {strategy_id}")
    data = load(data_dir)
    data["strategies"][strategy_id] = bool(enabled)
    save(data_dir, data)
    return list_strategies(data_dir)


def filter_signals_by_strategy(data_dir: Path, signals: list[dict[str, Any]]) -> list[dict[str, Any]]:
    enabled = get_enabled_map(data_dir)
    out: list[dict[str, Any]] = []
    for sig in signals:
        st = str(sig.get("signal_type") or "primary")
        if st == "trend_continuation":
            if enabled.get("trend_ignition", True):
                out.append(sig)
            continue
        if enabled.get(st, True):
            out.append(sig)
    return out
