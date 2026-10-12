"""Data ingestion subpackage."""
from __future__ import annotations

from .base import BaseIngestor, DataSourceRef, IngestionResult
from .statsbomb import StatsBombIngestor, list_competitions, record_ingestion_run

__all__ = [
    "BaseIngestor",
    "DataSourceRef",
    "IngestionResult",
    "StatsBombIngestor",
    "list_competitions",
    "record_ingestion_run",
]
