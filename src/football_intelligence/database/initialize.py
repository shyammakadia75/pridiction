"""Repeatable database initialization.

``initialize_database`` creates the schema directory, all tables and indexes,
and records the application schema version. It is **idempotent**: running it
again leaves existing rows untouched and never fabricates data.
"""
from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from .. import get_logger
from ..config import settings as _app_settings
from ..exceptions import DatabaseError
from .connection import database_connection, resolve_db_path
from .schema import ALL_TABLES, INDEX_STATEMENTS, MIGRATIONS, SCHEMA_VERSION

logger = get_logger("database.initialize")


def _utcnow() -> str:
    """Return the current UTC time as an ISO-8601 string."""
    return datetime.now(timezone.utc).isoformat()


def _current_version(conn: sqlite3.Connection) -> int:
    """Return the highest recorded schema version (0 if none yet)."""
    row = conn.execute(
        "SELECT MAX(version) AS v FROM schema_version;"
    ).fetchone()
    return int(row["v"]) if row and row["v"] is not None else 0


def initialize_database(db_path=None) -> int:
    """Create the database schema and indexes, applying migrations as needed.

    Args:
        db_path: Optional path to the SQLite file. Defaults to the configured
            database path. The parent directory is created if missing.

    Returns:
        The schema version now in effect.

    Raises:
        DatabaseError: If initialization fails.
    """
    path = resolve_db_path(db_path)
    # Ensure the parent directory exists before SQLite tries to open the file.
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise DatabaseError(
            f"Cannot create database directory '{path.parent}': {exc}"
        ) from exc

    try:
        with database_connection(path) as conn:
            for stmt in ALL_TABLES:
                conn.execute(stmt)
            for stmt in INDEX_STATEMENTS:
                conn.execute(stmt)

            current = _current_version(conn)
            target = SCHEMA_VERSION
            # Apply every migration newer than what is currently recorded.
            for version in range(current + 1, target + 1):
                for stmt in MIGRATIONS.get(version, ()):
                    conn.execute(stmt)
                conn.execute(
                    "INSERT OR REPLACE INTO schema_version "
                    "(version, applied_at) VALUES (?, ?);",
                    (version, _utcnow()),
                )

            final_version = _current_version(conn)
    except DatabaseError:
        raise
    except Exception as exc:  # pragma: no cover - defensive
        raise DatabaseError(f"Database initialization failed: {exc}") from exc

    logger.info(
        "Database initialized to schema version %d at '%s'.",
        final_version,
        path,
    )
    return final_version


def get_database_version(db_path=None) -> int:
    """Return the recorded schema version, or 0 if uninitialized."""
    path = resolve_db_path(db_path)
    try:
        with database_connection(path) as conn:
            return _current_version(conn)
    except sqlite3.Error as exc:
        raise DatabaseError(
            f"Cannot read schema version from '{path}': {exc}"
        ) from exc
