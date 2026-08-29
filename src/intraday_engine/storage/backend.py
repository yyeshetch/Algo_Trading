"""Storage backend selection: local CSV/JSON files vs InterServer MySQL."""

from __future__ import annotations

import os
from enum import Enum


class StorageBackend(str, Enum):
    FILES = "write_to_files"
    DB = "write_to_db"


def get_storage_backend() -> StorageBackend:
    raw = os.getenv("STORAGE_BACKEND", "write_to_files").strip().lower()
    if raw in {"write_to_db", "db", "mysql"}:
        return StorageBackend.DB
    return StorageBackend.FILES


def write_to_db() -> bool:
    return get_storage_backend() == StorageBackend.DB


def write_to_files() -> bool:
    return not write_to_db()


def set_storage_backend(value: str) -> None:
    normalized = value.strip().lower()
    if normalized not in {StorageBackend.FILES.value, StorageBackend.DB.value}:
        raise ValueError(f"Invalid storage backend: {value!r}. Use write_to_files or write_to_db.")
    os.environ["STORAGE_BACKEND"] = normalized
