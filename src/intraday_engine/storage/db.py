"""MySQL persistence for session-scheduler output (InterServer)."""

from __future__ import annotations

import json
import logging
import os
from datetime import date
from typing import Any
from urllib.parse import quote_plus

import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from sqlalchemy.exc import SQLAlchemyError

from intraday_engine.storage.data_cache import invalidate_data_cache
from intraday_engine.storage.layout import normalize_underlying

logger = logging.getLogger(__name__)

_engine: Engine | None = None
_schema_ready = False

# bar_timestamp avoids MySQL reserved word `timestamp`
_SCHEMA_STATEMENTS: tuple[str, ...] = (
    """
    CREATE TABLE IF NOT EXISTS signals (
      id BIGINT AUTO_INCREMENT PRIMARY KEY,
      trade_date DATE NOT NULL,
      bar_timestamp VARCHAR(32) NOT NULL,
      underlying VARCHAR(32) NOT NULL,
      asset_class VARCHAR(16) NOT NULL,
      row_json JSON NOT NULL,
      KEY idx_signals_lookup (trade_date, underlying, asset_class),
      KEY idx_signals_ts (trade_date, bar_timestamp)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
    """,
    """
    CREATE TABLE IF NOT EXISTS market_snapshots (
      id BIGINT AUTO_INCREMENT PRIMARY KEY,
      trade_date DATE NOT NULL,
      bar_timestamp VARCHAR(32) NOT NULL,
      underlying VARCHAR(32) NOT NULL,
      asset_class VARCHAR(16) NOT NULL,
      row_json JSON NOT NULL,
      KEY idx_snapshots_lookup (trade_date, underlying, asset_class),
      KEY idx_snapshots_ts (trade_date, bar_timestamp)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
    """,
    """
    CREATE TABLE IF NOT EXISTS option_chain_rows (
      id BIGINT AUTO_INCREMENT PRIMARY KEY,
      trade_date DATE NOT NULL,
      bar_timestamp VARCHAR(32) NOT NULL,
      underlying VARCHAR(32) NOT NULL,
      row_json JSON NOT NULL,
      KEY idx_option_chain_lookup (trade_date, underlying, bar_timestamp)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
    """,
    """
    CREATE TABLE IF NOT EXISTS json_artifacts (
      artifact_type VARCHAR(64) NOT NULL,
      trade_date DATE NOT NULL,
      underlying VARCHAR(32) NOT NULL DEFAULT '',
      payload JSON NOT NULL,
      updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
      PRIMARY KEY (artifact_type, trade_date, underlying)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
    """,
)


def database_url() -> str:
    explicit = os.getenv("DATABASE_URL", "").strip()
    if explicit:
        return explicit
    host = (os.getenv("MYSQL_HOST") or os.getenv("IP") or "localhost").strip()
    port = (os.getenv("MYSQL_PORT") or "3306").strip()
    database = (os.getenv("MYSQL_DATABASE") or os.getenv("DATABASE_NAME") or "").strip()
    user = (os.getenv("MYSQL_USER") or os.getenv("USERNAME") or "").strip()
    password = (os.getenv("MYSQL_PASSWORD") or os.getenv("DATABASE_PASSWORD") or "").strip()
    if not all([database, user, password]):
        raise ValueError(
            "MySQL credentials missing. Set MYSQL_DATABASE, MYSQL_USER, MYSQL_PASSWORD in .env"
        )
    return (
        f"mysql+pymysql://{quote_plus(user)}:{quote_plus(password)}@"
        f"{host}:{port}/{quote_plus(database)}?charset=utf8mb4"
    )


def mysql_host_label() -> str:
    url = database_url()
    return url.split("@", 1)[-1].split("?", 1)[0]


def get_engine() -> Engine:
    global _engine
    if _engine is None:
        connect_timeout = int(os.getenv("MYSQL_CONNECT_TIMEOUT", "15"))
        _engine = create_engine(
            database_url(),
            pool_pre_ping=True,
            pool_recycle=3600,
            connect_args={"connect_timeout": connect_timeout},
        )
    return _engine


def reset_engine() -> None:
    global _engine, _schema_ready
    if _engine is not None:
        _engine.dispose()
    _engine = None
    _schema_ready = False


def test_connection() -> dict[str, Any]:
    """Ping MySQL and return host + table list (raises on failure)."""
    engine = get_engine()
    with engine.connect() as conn:
        conn.execute(text("SELECT 1"))
        tables = [row[0] for row in conn.execute(text("SHOW TABLES")).fetchall()]
    return {"host": mysql_host_label(), "tables": tables}


def ensure_schema(*, force: bool = False) -> list[str]:
    """Create tables if missing. Returns table names present after init."""
    global _schema_ready
    if _schema_ready and not force:
        return _existing_tables()

    try:
        engine = get_engine()
        with engine.begin() as conn:
            for stmt in _SCHEMA_STATEMENTS:
                conn.execute(text(stmt))
        _schema_ready = True
        tables = _existing_tables()
        logger.info("MySQL schema ready on %s: %s", mysql_host_label(), ", ".join(tables) or "(none)")
        return tables
    except SQLAlchemyError as exc:
        _schema_ready = False
        host = mysql_host_label()
        hint = (
            "InterServer blocks remote MySQL by default. "
            "Run scripts/init_mysql_schema.sql in phpMyAdmin, or set MYSQL_HOST=localhost on the VPS."
        )
        if "timed out" in str(exc).lower() or "2003" in str(exc):
            raise ConnectionError(f"Cannot reach MySQL at {host}. {hint}") from exc
        raise


def _existing_tables() -> list[str]:
    expected = {"signals", "market_snapshots", "option_chain_rows", "json_artifacts"}
    with get_engine().connect() as conn:
        found = {row[0] for row in conn.execute(text("SHOW TABLES")).fetchall()}
    return sorted(expected & found)


def _json_safe(value: Any) -> Any:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    if hasattr(value, "item") and not isinstance(value, (str, bytes, bool)):
        try:
            return value.item()
        except (ValueError, AttributeError):
            pass
    return value


def _row_to_json(row: pd.Series | dict[str, Any]) -> str:
    data = row.to_dict() if isinstance(row, pd.Series) else dict(row)
    cleaned = {str(k): _json_safe(v) for k, v in data.items()}
    return json.dumps(cleaned, default=str)


def _rows_to_dataframe(rows: list[dict[str, Any]]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame()
    records: list[dict[str, Any]] = []
    for row in rows:
        payload = row.get("row_json") or row.get("payload")
        if isinstance(payload, str):
            payload = json.loads(payload)
        if isinstance(payload, dict):
            records.append(payload)
    if not records:
        return pd.DataFrame()
    return pd.DataFrame(records)


def _insert_rows(table: str, rows: pd.DataFrame) -> None:
    if rows.empty:
        return
    ensure_schema()
    params = []
    for _, row in rows.iterrows():
        params.append(
            {
                "trade_date": str(row.get("trade_date") or "")[:10] or None,
                "bar_timestamp": str(row.get("timestamp") or ""),
                "underlying": normalize_underlying(str(row.get("underlying") or "")),
                "asset_class": str(row.get("asset_class") or ""),
                "row_json": _row_to_json(row),
            }
        )
    sql = text(
        f"""
        INSERT INTO {table} (trade_date, bar_timestamp, underlying, asset_class, row_json)
        VALUES (:trade_date, :bar_timestamp, :underlying, :asset_class, CAST(:row_json AS JSON))
        """
    )
    with get_engine().begin() as conn:
        conn.execute(sql, params)


def _insert_option_chain_rows(rows: pd.DataFrame) -> None:
    if rows.empty:
        return
    ensure_schema()
    params = []
    for _, row in rows.iterrows():
        params.append(
            {
                "trade_date": str(row.get("trade_date") or "")[:10] or None,
                "bar_timestamp": str(row.get("timestamp") or ""),
                "underlying": normalize_underlying(str(row.get("underlying") or "")),
                "row_json": _row_to_json(row),
            }
        )
    sql = text(
        """
        INSERT INTO option_chain_rows (trade_date, bar_timestamp, underlying, row_json)
        VALUES (:trade_date, :bar_timestamp, :underlying, CAST(:row_json AS JSON))
        """
    )
    with get_engine().begin() as conn:
        conn.execute(sql, params)


def _replace_underlying_rows(
    table: str,
    *,
    trade_date: date,
    asset_class: str,
    underlying: str,
    rows: pd.DataFrame,
) -> None:
    ensure_schema()
    u = normalize_underlying(underlying)
    with get_engine().begin() as conn:
        conn.execute(
            text(
                f"""
                DELETE FROM {table}
                WHERE trade_date = :trade_date
                  AND asset_class = :asset_class
                  AND underlying = :underlying
                """
            ),
            {"trade_date": trade_date.isoformat(), "asset_class": asset_class, "underlying": u},
        )
    if not rows.empty:
        _insert_rows(table, rows)
    invalidate_data_cache(f"{table}:")


def load_table_rows(
    table: str,
    *,
    trade_date: date | None = None,
    underlying: str | None = None,
    asset_class: str | None = None,
) -> pd.DataFrame:
    ensure_schema()
    clauses = ["1=1"]
    params: dict[str, Any] = {}
    if trade_date is not None:
        clauses.append("trade_date = :trade_date")
        params["trade_date"] = trade_date.isoformat()
    if underlying is not None:
        clauses.append("underlying = :underlying")
        params["underlying"] = normalize_underlying(underlying)
    if asset_class is not None:
        clauses.append("asset_class = :asset_class")
        params["asset_class"] = asset_class
    sql = text(
        f"""
        SELECT row_json
        FROM {table}
        WHERE {' AND '.join(clauses)}
        ORDER BY bar_timestamp ASC
        """
    )
    with get_engine().connect() as conn:
        result = conn.execute(sql, params)
        raw = [{"row_json": row[0]} for row in result.fetchall()]
    return _rows_to_dataframe(raw)


def replace_signals_rows(
    *,
    trade_date: date,
    asset_class: str,
    underlying: str,
    rows: pd.DataFrame,
) -> None:
    _replace_underlying_rows(
        "signals",
        trade_date=trade_date,
        asset_class=asset_class,
        underlying=underlying,
        rows=rows,
    )


def replace_snapshot_rows(
    *,
    trade_date: date,
    asset_class: str,
    underlying: str,
    rows: pd.DataFrame,
) -> None:
    _replace_underlying_rows(
        "market_snapshots",
        trade_date=trade_date,
        asset_class=asset_class,
        underlying=underlying,
        rows=rows,
    )


def append_option_chain_rows(rows: pd.DataFrame) -> None:
    if rows.empty:
        return
    _insert_option_chain_rows(rows)
    invalidate_data_cache("option_chain:")


def load_option_chain_rows(
    trade_date: date,
    underlying: str | None = None,
) -> pd.DataFrame:
    ensure_schema()
    clauses = ["trade_date = :trade_date"]
    params: dict[str, Any] = {"trade_date": trade_date.isoformat()}
    if underlying is not None:
        clauses.append("underlying = :underlying")
        params["underlying"] = normalize_underlying(underlying)
    sql = text(
        f"""
        SELECT row_json
        FROM option_chain_rows
        WHERE {' AND '.join(clauses)}
        ORDER BY bar_timestamp ASC
        """
    )
    with get_engine().connect() as conn:
        result = conn.execute(sql, params)
        raw = [{"row_json": row[0]} for row in result.fetchall()]
    return _rows_to_dataframe(raw)


def delete_option_chain_rows(
    trade_date: date,
    underlying: str,
    *,
    timestamps: set[str] | None = None,
    timestamp_prefixes: list[str] | None = None,
) -> int:
    df = load_option_chain_rows(trade_date, underlying)
    if df.empty:
        return 0
    u = normalize_underlying(underlying)
    drop = pd.Series(False, index=df.index)
    if timestamps:
        drop |= df["timestamp"].astype(str).isin({str(t) for t in timestamps})
    if timestamp_prefixes:
        for prefix in timestamp_prefixes:
            drop |= df["timestamp"].astype(str).str.startswith(prefix)
    kept = df.loc[~drop]
    removed = int(drop.sum())
    ensure_schema()
    with get_engine().begin() as conn:
        conn.execute(
            text(
                """
                DELETE FROM option_chain_rows
                WHERE trade_date = :trade_date AND underlying = :underlying
                """
            ),
            {"trade_date": trade_date.isoformat(), "underlying": u},
        )
    if not kept.empty:
        append_option_chain_rows(kept)
    else:
        invalidate_data_cache("option_chain:")
    return removed


def save_json_artifact(
    artifact_type: str,
    trade_date: date,
    payload: dict[str, Any],
    *,
    underlying: str = "",
) -> None:
    ensure_schema()
    u = normalize_underlying(underlying) if underlying else ""
    body = json.dumps(payload, default=str)
    with get_engine().begin() as conn:
        conn.execute(
            text(
                """
                INSERT INTO json_artifacts (artifact_type, trade_date, underlying, payload)
                VALUES (:artifact_type, :trade_date, :underlying, CAST(:payload AS JSON))
                ON DUPLICATE KEY UPDATE payload = VALUES(payload)
                """
            ),
            {
                "artifact_type": artifact_type,
                "trade_date": trade_date.isoformat(),
                "underlying": u,
                "payload": body,
            },
        )
    invalidate_data_cache(f"artifact:{artifact_type}:")


def load_json_artifact(
    artifact_type: str,
    trade_date: date,
    *,
    underlying: str = "",
) -> dict[str, Any] | None:
    ensure_schema()
    u = normalize_underlying(underlying) if underlying else ""
    sql = text(
        """
        SELECT payload
        FROM json_artifacts
        WHERE artifact_type = :artifact_type
          AND trade_date = :trade_date
          AND underlying = :underlying
        LIMIT 1
        """
    )
    with get_engine().connect() as conn:
        row = conn.execute(
            sql,
            {
                "artifact_type": artifact_type,
                "trade_date": trade_date.isoformat(),
                "underlying": u,
            },
        ).first()
    if row is None:
        return None
    payload = row[0]
    if isinstance(payload, str):
        return json.loads(payload)
    return dict(payload)


def load_latest_json_artifact(
    artifact_type: str,
    *,
    underlying: str = "",
) -> dict[str, Any] | None:
    ensure_schema()
    u = normalize_underlying(underlying) if underlying else ""
    sql = text(
        """
        SELECT payload
        FROM json_artifacts
        WHERE artifact_type = :artifact_type
          AND underlying = :underlying
        ORDER BY trade_date DESC
        LIMIT 1
        """
    )
    with get_engine().connect() as conn:
        row = conn.execute(sql, {"artifact_type": artifact_type, "underlying": u}).first()
    if row is None:
        return None
    payload = row[0]
    if isinstance(payload, str):
        return json.loads(payload)
    return dict(payload)
