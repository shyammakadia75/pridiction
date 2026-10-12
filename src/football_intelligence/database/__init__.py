"""Database subpackage: connection management, schema, and initialization."""
from __future__ import annotations

from .connection import database_connection, get_connection, resolve_db_path
from .initialize import get_database_version, initialize_database
from .schema import MIGRATIONS, SCHEMA_VERSION

__all__ = [
    "database_connection",
    "get_connection",
    "resolve_db_path",
    "get_database_version",
    "initialize_database",
    "SCHEMA_VERSION",
    "MIGRATIONS",
]
