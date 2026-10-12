"""Exception hierarchy for the football-intelligence package.

A single, well-defined hierarchy keeps error handling readable and prevents
accidental swallowing of bugs via bare ``except Exception`` blocks.
"""
from __future__ import annotations


class FootballIntelligenceError(Exception):
    """Base class for all errors raised by this package."""


class ConfigError(FootballIntelligenceError):
    """Raised when configuration values are missing or invalid."""


class DatabaseError(FootballIntelligenceError):
    """Raised for database connection or schema errors."""


class IngestionError(FootballIntelligenceError):
    """Raised when a data ingestion step fails."""


class ValidationError(FootballIntelligenceError):
    """Raised when collected validation errors must abort an operation."""
