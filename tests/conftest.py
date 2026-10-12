"""Shared pytest fixtures and path bootstrap for the test suite.

The src-layout package is made importable here so the tests can run even
before an editable install (defensive; the canonical install is still
``pip install -e .``). Tests must never depend on paid APIs, internet
connectivity, real credentials, or a production database.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Make the src/ layout importable without requiring an editable install.
_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from football_intelligence.logging_config import reset_logging  # noqa: E402


@pytest.fixture(autouse=True)
def _reset_logging() -> None:
    """Reset the package logger around every test to avoid handler pollution."""
    reset_logging()
    yield
    reset_logging()


@pytest.fixture()
def db_path(tmp_path: Path) -> Path:
    """A fresh, disposable SQLite database path for each test."""
    return tmp_path / "test.db"
