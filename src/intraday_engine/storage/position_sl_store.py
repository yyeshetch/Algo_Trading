"""Store position SL order mappings and auto-trail state."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _path(data_dir: Path) -> Path:
    return data_dir / "position_sl.json"


def load(data_dir: Path) -> dict[str, Any]:
    """Load position SL data. Key: tradingsymbol, value: {sl_order_id, sl_trigger, auto_trail, ...}."""
    p = _path(data_dir)
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}


def save(data_dir: Path, data: dict[str, Any]) -> None:
    p = _path(data_dir)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, indent=2), encoding="utf-8")


def get_position_key(tradingsymbol: str, quantity: int) -> str:
    """Unique key for a position (symbol + direction)."""
    direction = "long" if quantity > 0 else "short"
    sym = str(tradingsymbol).replace("NFO:", "").strip()
    return f"{sym}_{direction}"


def set_sl(data_dir: Path, tradingsymbol: str, quantity: int, sl_order_id: str, sl_trigger: float, instrument_token: int,
           *, entry_price: float | None = None, initial_sl: float | None = None, risk_pts: float | None = None,
           rr_locked: bool = False, rr_trail: bool = False, signal_key: str | None = None) -> None:
    data = load(data_dir)
    key = get_position_key(tradingsymbol, quantity)
    prev = data.get(key, {})
    data[key] = {
        "tradingsymbol": str(tradingsymbol).replace("NFO:", ""),
        "quantity": abs(int(quantity)),
        "sl_order_id": str(sl_order_id),
        "sl_trigger": float(sl_trigger),
        "instrument_token": int(instrument_token),
        "auto_trail": prev.get("auto_trail", False),
        "entry_price": float(entry_price) if entry_price is not None else prev.get("entry_price"),
        "initial_sl": float(initial_sl) if initial_sl is not None else prev.get("initial_sl"),
        "risk_pts": float(risk_pts) if risk_pts is not None else prev.get("risk_pts"),
        "rr_locked": bool(rr_locked) if rr_locked else prev.get("rr_locked", False),
        "rr_trail": bool(rr_trail) if rr_trail else prev.get("rr_trail", False),
        "signal_key": signal_key or prev.get("signal_key"),
    }
    save(data_dir, data)


def get_sl(data_dir: Path, tradingsymbol: str, quantity: int) -> dict | None:
    key = get_position_key(tradingsymbol, quantity)
    return load(data_dir).get(key)


def get_sl_for_position(data_dir: Path, tradingsymbol: str, quantity: int) -> tuple[str | None, float | None]:
    """Return (sl_order_id, sl_trigger) or (None, None)."""
    rec = get_sl(data_dir, tradingsymbol, quantity)
    if not rec:
        return None, None
    return rec.get("sl_order_id"), rec.get("sl_trigger")


def update_sl_trigger(data_dir: Path, tradingsymbol: str, quantity: int, sl_trigger: float) -> None:
    data = load(data_dir)
    key = get_position_key(tradingsymbol, quantity)
    if key in data:
        data[key]["sl_trigger"] = float(sl_trigger)
        save(data_dir, data)


def set_auto_trail(data_dir: Path, tradingsymbol: str, quantity: int, enabled: bool) -> None:
    data = load(data_dir)
    key = get_position_key(tradingsymbol, quantity)
    if key in data:
        data[key]["auto_trail"] = bool(enabled)
        save(data_dir, data)


def set_rr_locked(data_dir: Path, tradingsymbol: str, quantity: int, *, rr_locked: bool, sl_trigger: float | None = None) -> None:
    data = load(data_dir)
    key = get_position_key(tradingsymbol, quantity)
    if key not in data:
        return
    data[key]["rr_locked"] = bool(rr_locked)
    if sl_trigger is not None:
        data[key]["sl_trigger"] = float(sl_trigger)
    save(data_dir, data)


def set_rr_trail(data_dir: Path, tradingsymbol: str, quantity: int, enabled: bool) -> None:
    data = load(data_dir)
    key = get_position_key(tradingsymbol, quantity)
    if key in data:
        data[key]["rr_trail"] = bool(enabled)
        save(data_dir, data)


def get_rr_trail_positions(data_dir: Path) -> list[dict]:
    data = load(data_dir)
    return [v for v in data.values() if v.get("entry_price") is not None]


def get_auto_trail_positions(data_dir: Path) -> list[dict]:
    """Return list of positions with auto_trail enabled."""
    data = load(data_dir)
    return [v for v in data.values() if v.get("auto_trail")]


def remove(data_dir: Path, tradingsymbol: str, quantity: int) -> None:
    """Remove SL record when position is closed."""
    data = load(data_dir)
    key = get_position_key(tradingsymbol, quantity)
    data.pop(key, None)
    save(data_dir, data)
