"""Provider-agnostic ingestion interface.

Defines an abstract ingestor contract so that concrete providers (StatsBomb
Open Data, Football-Data.co.uk, ...) can be added later without changing the
surrounding infrastructure. The interface is intentionally provider-neutral;
no real network call is performed here.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional

import pandas as pd

from ...exceptions import IngestionError
from ..validation.validators import ValidationResult


@dataclass
class DataSourceRef:
    """Provenance metadata describing where a dataset came from."""

    provider: str
    dataset_name: str
    source_reference: Optional[str] = None
    data_version: Optional[str] = None
    metadata: Optional[dict[str, Any]] = None


@dataclass
class IngestionResult:
    """Outcome of an :meth:`BaseIngestor.ingest` run."""

    provider: str
    dataset_name: str
    records_read: int = 0
    records_inserted: int = 0
    records_updated: int = 0
    records_skipped: int = 0
    records_matched: int = 0
    records_unmatched: int = 0
    records_ambiguous: int = 0
    records_invalid: int = 0
    status: str = "success"
    error_message: Optional[str] = None
    started_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    completed_at: Optional[str] = None

    @property
    def is_success(self) -> bool:
        """True when the run completed without error."""
        return self.status == "success"


class BaseIngestor(ABC):
    """Abstract base class for all data ingestors.

    Subclasses implement :meth:`retrieve`, :meth:`validate`, and
    :meth:`load`. :meth:`ingest` orchestrates them and records an
    :class:`IngestionResult`.
    """

    @property
    @abstractmethod
    def provider(self) -> str:
        """Identifier of the data provider (e.g. ``"statsbomb"``)."""

    @property
    @abstractmethod
    def dataset_name(self) -> str:
        """Identifier of the dataset being ingested (e.g. ``"matches"``)."""

    @property
    def source_reference(self) -> Optional[str]:  # pragma: no cover - trivial
        """Optional stable reference (URL, file path, or API endpoint)."""
        return None

    @abstractmethod
    def retrieve(self) -> pd.DataFrame:
        """Load raw data from the provider into a DataFrame."""

    @abstractmethod
    def validate(self, data: pd.DataFrame) -> ValidationResult:
        """Validate ``data`` before loading it into the database."""

    @abstractmethod
    def load(self, data: pd.DataFrame) -> IngestionResult:
        """Persist validated data and return an :class:`IngestionResult`."""

    def ingest(self) -> IngestionResult:
        """Run the full retrieve -> validate -> load pipeline."""
        started_at = datetime.now(timezone.utc).isoformat()
        result = IngestionResult(
            provider=self.provider,
            dataset_name=self.dataset_name,
            started_at=started_at,
        )
        try:
            data = self.retrieve()
            result.records_read = len(data)
            validation = self.validate(data)
            if validation.has_errors:
                raise IngestionError(
                    f"Validation failed for {self.provider}/{self.dataset_name}: "
                    f"{len(validation.issues)} error(s)"
                )
            load_result = self.load(data)
            result.records_inserted = load_result.records_inserted
            result.records_updated = load_result.records_updated
            result.records_skipped = load_result.records_skipped
            result.records_matched = load_result.records_matched
            result.records_unmatched = load_result.records_unmatched
            result.records_ambiguous = load_result.records_ambiguous
            result.records_invalid = load_result.records_invalid
        except IngestionError:
            result.status = "failed"
            result.error_message = "Validation failed (see logs)."
            result.completed_at = datetime.now(timezone.utc).isoformat()
            raise
        except Exception as exc:
            result.status = "failed"
            result.error_message = str(exc)
            result.completed_at = datetime.now(timezone.utc).isoformat()
            raise IngestionError(
                f"Ingestion of {self.provider}/{self.dataset_name} failed: {exc}"
            ) from exc

        result.completed_at = datetime.now(timezone.utc).isoformat()
        return result
