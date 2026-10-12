"""Data ingestion subpackage."""
from __future__ import annotations

from .base import BaseIngestor, DataSourceRef, IngestionResult
from .odds import OddsIngestor, list_leagues
from .statsbomb import StatsBombIngestor, list_competitions, record_ingestion_run

__all__ = [
    "BaseIngestor",
    "DataSourceRef",
    "IngestionResult",
    "OddsIngestor",
    "StatsBombIngestor",
    "list_competitions",
    "list_leagues",
    "record_ingestion_run",
]
