"""Centralized application configuration.

All runtime settings are derived from environment variables (optionally
loaded from a ``.env`` file). No machine-specific absolute paths are hard
coded in source; every path is resolved relative to the project root.

Environment variables use the ``FI_`` prefix (Football Intelligence) so they
do not collide with unrelated tools. Critical values are validated on load.

The module-level ``settings`` singleton is built once at import. Call
:func:`load_settings` explicitly (e.g. in tests after ``monkeypatch.setenv``)
to rebuild settings from a changed environment.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from dotenv import load_dotenv

from .exceptions import ConfigError

# ---------------------------------------------------------------------------
# Project layout.
# config.py is at  src/football_intelligence/config.py
# => PACKAGE_DIR        = src/football_intelligence
# => SRC_DIR            = src
# => PROJECT_ROOT       = repository root
# ---------------------------------------------------------------------------
PACKAGE_DIR: Final[Path] = Path(__file__).resolve().parent
SRC_DIR: Final[Path] = PACKAGE_DIR.parent.parent  # src/
PROJECT_ROOT: Final[Path] = SRC_DIR.parent

DATA_DIR: Final[Path] = PROJECT_ROOT / "data"
RAW_DATA_DIR: Final[Path] = DATA_DIR / "raw"
PROCESSED_DATA_DIR: Final[Path] = DATA_DIR / "processed"
EXTERNAL_DATA_DIR: Final[Path] = DATA_DIR / "external"
MODELS_DIR: Final[Path] = PROJECT_ROOT / "models"
NOTEBOOKS_DIR: Final[Path] = PROJECT_ROOT / "notebooks"
LOGS_DIR: Final[Path] = PROJECT_ROOT / "logs"
CONFIG_DIR: Final[Path] = PROJECT_ROOT / "config"
SCRIPTS_DIR: Final[Path] = PROJECT_ROOT / "scripts"

ENV_PREFIX: Final[str] = "FI_"
VALID_LOG_LEVELS: Final[frozenset[str]] = frozenset(
    {"DEBUG", "INFO", "WARNING", "WARN", "ERROR", "CRITICAL", "NOTSET"}
)
VALID_ENVIRONMENTS: Final[frozenset[str]] = frozenset(
    {"development", "testing", "production"}
)


def _env_str(name: str, default: str) -> str:
    """Return a string setting from the ``FI_``-prefixed environment."""
    return os.getenv(f"{ENV_PREFIX}{name}", default)


def _env_path(name: str, default: Path) -> Path:
    """Return a path setting, resolving relative values against the project root."""
    raw = os.getenv(f"{ENV_PREFIX}{name}")
    if raw:
        path = Path(raw)
        return path if path.is_absolute() else (PROJECT_ROOT / path)
    return default


def load_dotenv_file() -> None:
    """Load variables from ``.env`` without overriding real env vars.

    ``override=False`` means variables already present in the real environment
    (for example set in the shell or by the test runner) always win.
    """
    env_path = PROJECT_ROOT / ".env"
    if env_path.exists():
        load_dotenv(env_path, override=False)


# Load .env once when the module is imported.
load_dotenv_file()


@dataclass(frozen=True)
class Settings:
    """Immutable application settings sourced from the environment."""

    environment: str
    database_path: Path
    log_level: str
    log_dir: Path
    log_file: str
    log_format: str

    def __post_init__(self) -> None:
        self._validate()

    def _validate(self) -> None:
        if self.log_level.upper() not in VALID_LOG_LEVELS:
            raise ConfigError(
                f"Invalid FI_LOG_LEVEL '{self.log_level}'. "
                f"Must be one of: {sorted(VALID_LOG_LEVELS)}"
            )
        if self.environment not in VALID_ENVIRONMENTS:
            raise ConfigError(
                f"Invalid FI_ENVIRONMENT '{self.environment}'. "
                f"Must be one of: {sorted(VALID_ENVIRONMENTS)}"
            )
        db_parent = self.database_path.parent
        if db_parent.exists() and not db_parent.is_dir():
            raise ConfigError(
                f"Database directory '{db_parent}' exists but is not a directory."
            )


def load_settings() -> Settings:
    """Build :class:`Settings` from the current environment.

    Call this whenever the environment may have changed -- for example inside
    tests after ``monkeypatch.setenv``.
    """
    return Settings(
        environment=_env_str("ENVIRONMENT", "development"),
        database_path=_env_path(
            "DATABASE_PATH", DATA_DIR / "football_intelligence.db"
        ),
        log_level=_env_str("LOG_LEVEL", "INFO"),
        log_dir=_env_path("LOG_DIR", LOGS_DIR),
        log_file=_env_str("LOG_FILE", "football_intelligence.log"),
        log_format=_env_str(
            "LOG_FORMAT",
            "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        ),
    )


# Module-level singleton used throughout the application.
settings: Settings = load_settings()
