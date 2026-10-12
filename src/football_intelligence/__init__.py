"""football-intelligence: a modular football analytics foundation.

This package currently provides the project's engineering foundation only:
configuration, logging, database, ingestion interfaces, and validation.
Machine-learning models, betting logic, and dashboards are intentionally
NOT implemented in this phase.
"""
from __future__ import annotations

from .exceptions import (
    ConfigError,
    DatabaseError,
    FootballIntelligenceError,
    IngestionError,
    ValidationError,
)

__version__ = "0.1.0"

__all__ = [
    "FootballIntelligenceError",
    "ConfigError",
    "DatabaseError",
    "IngestionError",
    "ValidationError",
    "__version__",
]
