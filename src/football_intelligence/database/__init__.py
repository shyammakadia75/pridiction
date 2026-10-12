"""Database subpackage: connection management, schema, and initialization."""
from __future__ import annotations

from .connection import database_connection, get_connection, resolve_db_path
from .initialize import initialize_database
from .schema import MIGRATIONS, SCHEMA_VERSION

__all__ = [
    "database_connection",
    "get_connection",
    "resolve_db_path",
    "initialize_database",
    "SCHEMA_VERSION",
    "MIGRATIONS",
]
