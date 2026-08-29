"""In-memory cache for DB-backed dashboard reads (cleared on hard reload)."""

from __future__ import annotations

import threading
from typing import Any

_lock = threading.Lock()
_cache: dict[str, Any] = {}


def get_cached(key: str) -> Any | None:
    with _lock:
        return _cache.get(key)


def set_cached(key: str, value: Any) -> None:
    with _lock:
        _cache[key] = value


def invalidate_data_cache(prefix: str | None = None) -> None:
    with _lock:
        if prefix is None:
            _cache.clear()
            return
        drop = [k for k in _cache if k.startswith(prefix)]
        for key in drop:
            _cache.pop(key, None)
