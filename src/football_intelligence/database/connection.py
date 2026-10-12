"""SQLite connection management.

Foreign-key enforcement is enabled on every connection immediately after it
is opened, because SQLite ships with foreign keys disabled by default. Two
access patterns are provided:

* :func:`get_connection` -- a raw, configured connection (caller manages the
  transaction and cleanup).
* :func:`database_connection` -- a context manager that commits on a clean
  exit, rolls back on error, and always closes the connection.
"""
from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Generator, Optional, Union

from ..config import settings as _settings
from ..exceptions import DatabaseError


def resolve_db_path(db_path: Optional[Union[str, Path]] = None) -> Path:
    """Resolve a database path, defaulting to the configured database path."""
    if db_path is None:
        return _settings.database_path
    path = Path(db_path)
    return path if path.is_absolute() else path.resolve()


def get_connection(
    db_path: Optional[Union[str, Path]] = None,
) -> sqlite3.Connection:
    """Open a SQLite connection with foreign keys enabled and Row factory set."""
    path = resolve_db_path(db_path)
    try:
        conn = sqlite3.connect(str(path))
    except sqlite3.Error as exc:  # pragma: no cover - defensive
        raise DatabaseError(f"Cannot open database at '{path}': {exc}") from exc
    conn.row_factory = sqlite3.Row
    # foreign_keys is a per-connection pragma; set it before any DML.
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn


@contextmanager
def database_connection(
    db_path: Optional[Union[str, Path]] = None,
) -> Generator[sqlite3.Connection, None, None]:
    """Context manager yielding a connection.

    Commits on a clean exit, rolls back on exception, and always closes the
    connection. Foreign keys are enabled for the yielded connection.
    """
    conn = get_connection(db_path)
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()
