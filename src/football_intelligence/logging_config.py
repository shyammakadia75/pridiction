"""Reusable logging configuration.

Provides a single configured logger ("football_intelligence") with console
and optional file handlers. Repeated initialization is safe: existing
handlers are reused so the same message is never written twice to the same
destination. Secrets must never be logged via this logger.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Union

from .config import settings as _default_settings

LOGGER_NAME: str = "football_intelligence"

_LEVEL_MAP: dict[str, int] = {
    "DEBUG": logging.DEBUG,
    "INFO": logging.INFO,
    "WARNING": logging.WARNING,
    "WARN": logging.WARNING,
    "ERROR": logging.ERROR,
    "CRITICAL": logging.CRITICAL,
    "NOTSET": logging.NOTSET,
}


def _coerce_level(level: str) -> int:
    """Map a level name string to a numeric logging level."""
    value = _LEVEL_MAP.get(level.upper(), logging.INFO)
    if value is logging.INFO and level.upper() not in _LEVEL_MAP:
        # Unknown level: fall back to INFO but let config validation catch it.
        return logging.INFO
    return value


def setup_logging(
    log_level: Optional[str] = None,
    log_dir: Optional[Union[str, Path]] = None,
    log_file: Optional[str] = None,
    log_format: Optional[str] = None,
) -> logging.Logger:
    """Configure and return the package logger.

    Any parameter left ``None`` falls back to the loaded application settings.
    The log directory is created (with parents) if it does not exist.
    """
    log_level = log_level or _default_settings.log_level
    log_dir = Path(log_dir) if log_dir is not None else _default_settings.log_dir
    log_file = log_file or _default_settings.log_file
    log_format = log_format or _default_settings.log_format

    logger = logging.getLogger(LOGGER_NAME)
    level = _coerce_level(log_level)
    logger.setLevel(level)

    formatter = logging.Formatter(log_format)

    # --- Console handler: always present, never duplicated. ---
    has_console = any(
        isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler)
        for h in logger.handlers
    )
    if not has_console:
        console = logging.StreamHandler()
        console.setFormatter(formatter)
        console.setLevel(level)
        logger.addHandler(console)

    # --- File handler: created safely, never duplicated. ---
    if log_file:
        log_dir.mkdir(parents=True, exist_ok=True)
        target_file = log_dir / log_file
        has_file = any(
            isinstance(h, logging.FileHandler)
            and Path(h.stream.name) == target_file
            for h in logger.handlers
        )
        if not has_file:
            file_handler = logging.FileHandler(target_file, encoding="utf-8")
            file_handler.setFormatter(formatter)
            file_handler.setLevel(level)
            logger.addHandler(file_handler)

    # Do not propagate to the root logger -> avoids duplicate console output when
    # the library is embedded in an application that also configures root.
    logger.propagate = False
    return logger


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """Return a child logger under the package logger.

    Usage inside the package: ``get_logger("database")`` ->
    ``football_intelligence.database``.
    """
    if name:
        return logging.getLogger(LOGGER_NAME).getChild(name)
    return logging.getLogger(LOGGER_NAME)


def reset_logging() -> None:
    """Remove all handlers from the package logger (useful for tests)."""
    logger = logging.getLogger(LOGGER_NAME)
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()
