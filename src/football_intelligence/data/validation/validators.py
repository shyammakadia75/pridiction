"""Reusable data validation framework.

Validators operate on pandas DataFrames and return a structured
:class:`ValidationResult` containing :class:`ValidationIssue` objects tagged
as ERROR or WARNING. Invalid data is reported, never silently discarded; each
issue is attributed to a field and, where relevant, a 0-based row position so
the offending record can be located.

Distinguish three situations: *invalid data* (ERROR), *missing-but-optional
data* (WARNING), and *valid data with warnings* (WARNING). Validators degrade
gracefully when expected columns are missing: they record a warning and return
rather than raising, so callers can run the full suite across providers that
rename or omit fields.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Sequence

import numpy as np
import pandas as pd

from ...exceptions import ValidationError

__all__ = [
    "Severity",
    "ValidationIssue",
    "ValidationResult",
    "validate_required_columns",
    "validate_no_missing_values",
    "validate_non_negative_goals",
    "validate_no_duplicate_provider_ids",
    "validate_probability_ranges",
    "validate_probability_sum",
    "validate_positive_odds",
    "validate_date_fields",
    "validate_home_away_different",
    "validate_referential_integrity",
    "validate_team_ids",
]


class Severity(str, Enum):
    """Severity of a validation issue."""

    ERROR = "error"
    WARNING = "warning"


@dataclass
class ValidationIssue:
    """A single problem found while validating a dataset."""

    field_name: str
    severity: Severity
    message: str
    record_index: Optional[int] = None

    def __str__(self) -> str:
        loc = (
            f" (row {self.record_index})"
            if self.record_index is not None
            else ""
        )
        return f"[{self.severity.value.upper()}] {self.field_name}{loc}: {self.message}"


@dataclass
class ValidationResult:
    """Accumulator for validation issues returned by every validator."""

    issues: list[ValidationIssue] = field(default_factory=list)
    row_count: Optional[int] = None
    summary: dict = field(default_factory=dict)

    @property
    def is_valid(self) -> bool:
        return not any(i.severity == Severity.ERROR for i in self.issues)

    @property
    def has_errors(self) -> bool:
        return any(i.severity == Severity.ERROR for i in self.issues)

    @property
    def has_warnings(self) -> bool:
        return any(i.severity == Severity.WARNING for i in self.issues)

    @property
    def error_count(self) -> int:
        return sum(1 for i in self.issues if i.severity == Severity.ERROR)

    @property
    def warning_count(self) -> int:
        return sum(1 for i in self.issues if i.severity == Severity.WARNING)

    def add_error(
        self, field_name: str, message: str, record_index: Optional[int] = None
    ) -> None:
        self.issues.append(
            ValidationIssue(field_name, Severity.ERROR, message, record_index)
        )

    def add_warning(
        self, field_name: str, message: str, record_index: Optional[int] = None
    ) -> None:
        self.issues.append(
            ValidationIssue(field_name, Severity.WARNING, message, record_index)
        )

    def merge(self, other: "ValidationResult") -> "ValidationResult":
        self.issues.extend(other.issues)
        if self.row_count is None:
            self.row_count = other.row_count
        self.summary.update(other.summary)
        return self

    def raise_if_errors(self) -> None:
        if self.has_errors:
            errors = [str(i) for i in self.issues if i.severity == Severity.ERROR]
            raise ValidationError(
                f"{self.error_count} validation error(s):\n  - "
                + "\n  - ".join(errors)
            )

    def __str__(self) -> str:
        if not self.issues:
            return f"ValidationResult(valid={self.is_valid}, rows={self.row_count})"
        lines = [
            f"ValidationResult(valid={self.is_valid}, rows={self.row_count}, "
            f"errors={self.error_count}, warnings={self.warning_count})"
        ]
        for issue in self.issues:
            lines.append(f"  - {issue}")
        return "\n".join(lines)


def _positions(mask: pd.Series) -> list[int]:
    """Return 0-based row positions where ``mask`` is True."""
    return [int(i) for i in np.where(mask.to_numpy())[0]]


def validate_required_columns(
    df: pd.DataFrame, required_columns: Sequence[str]
) -> ValidationResult:
    """Ensure every required column name is present in ``df``."""
    result = ValidationResult(row_count=len(df))
    missing = [c for c in required_columns if c not in df.columns]
    if missing:
        result.add_error("columns", f"Missing required column(s): {list(missing)}")
    return result


def validate_no_missing_values(
    df: pd.DataFrame, required_columns: Sequence[str]
) -> ValidationResult:
    """Report nulls/NaN in the requested columns (which must be present)."""
    result = ValidationResult(row_count=len(df))
    for col in required_columns:
        if col not in df.columns:
            result.add_warning(col, "Column absent; cannot check for missing values.")
            continue
        null_count = int(df[col].isna().sum())
        if null_count:
            result.add_error(col, f"{null_count} missing value(s) in required column.")
    return result


def validate_non_negative_goals(
    df: pd.DataFrame,
    home_col: str = "home_goals",
    away_col: str = "away_goals",
) -> ValidationResult:
    """Goals, when present, must be non-negative."""
    result = ValidationResult(row_count=len(df))
    for col in (home_col, away_col):
        if col not in df.columns:
            result.add_warning(col, "Column absent; skipping goal-sign check (goals optional).")
            continue
        numeric = pd.to_numeric(df[col], errors="coerce")
        bad_mask = numeric.notna() & (numeric < 0)
        positions = _positions(bad_mask)
        if positions:
            values = numeric.to_numpy()[bad_mask.to_numpy()]
            for pos, val in zip(positions[:5], values[:5]):
                result.add_error(col, f"Negative goal count {val}.", record_index=pos)
            result.add_error(col, f"{len(positions)} negative goal value(s).")
    return result


def validate_no_duplicate_provider_ids(
    df: pd.DataFrame,
    provider_col: str = "provider",
    id_col: str = "provider_id",
) -> ValidationResult:
    """Detect duplicate (provider, provider_id) pairs within a dataset."""
    result = ValidationResult(row_count=len(df))
    needed = [c for c in (provider_col, id_col) if c not in df.columns]
    if needed:
        result.add_warning(id_col, f"Column(s) {needed} absent; skipping duplicate check.")
        return result
    dup_mask = df.duplicated(subset=[provider_col, id_col], keep=False)
    positions = _positions(dup_mask)
    if positions:
        dup_values = (
            df.loc[dup_mask, [provider_col, id_col]].drop_duplicates().head(10).values.tolist()
        )
        detail = "; ".join(f"{p}/{i}" for p, i in dup_values)
        result.add_error(
            id_col,
            f"{len(positions)} duplicate provider-id row(s). Examples: {detail[:200]}",
        )
    return result


def validate_probability_ranges(
    df: pd.DataFrame, columns: Sequence[str]
) -> ValidationResult:
    """Each probability column must lie within [0, 1] when present."""
    result = ValidationResult(row_count=len(df))
    for col in columns:
        if col not in df.columns:
            result.add_warning(col, "Column absent; skipping probability-range check.")
            continue
        numeric = pd.to_numeric(df[col], errors="coerce")
        bad_mask = numeric.notna() & ((numeric < 0) | (numeric > 1))
        positions = _positions(bad_mask)
        if positions:
            values = numeric.to_numpy()[bad_mask.to_numpy()]
            for pos, val in zip(positions[:5], values[:5]):
                result.add_error(col, f"Probability {val} outside [0, 1].", record_index=pos)
            result.add_error(col, f"{len(positions)} value(s) outside [0, 1].")
    return result


def validate_probability_sum(
    df: pd.DataFrame,
    columns: Sequence[str],
    tolerance: float = 1e-6,
) -> ValidationResult:
    """When all probability columns are present, their sum must be ~1."""
    result = ValidationResult(row_count=len(df))
    present = [c for c in columns if c in df.columns]
    missing = [c for c in columns if c not in df.columns]
    if missing:
        result.add_warning("probability_sum", f"Column(s) {missing} absent; skipping sum check.")
        if not present:
            return result
    numeric = pd.DataFrame({c: pd.to_numeric(df[c], errors="coerce") for c in present})
    row_sum = numeric.sum(axis=1)
    fully_present = numeric.notna().all(axis=1)
    bad_mask = fully_present & ((row_sum - 1.0).abs() > tolerance)
    positions = _positions(bad_mask)
    if positions:
        sums = row_sum.to_numpy()
        for pos in positions[:5]:
            result.add_error(
                "probability_sum",
                f"Probability sum {float(sums[pos]):.6f} != 1.0 (tolerance {tolerance}).",
                record_index=pos,
            )
        result.add_error("probability_sum", f"{len(positions)} row(s) deviate from sum=1.")
    return result
def validate_positive_odds(
    df: pd.DataFrame, odds_col: str = "decimal_odds"
) -> ValidationResult:
    """Decimal odds, when present, must be strictly positive."""
    result = ValidationResult(row_count=len(df))
    if odds_col not in df.columns:
        result.add_warning(odds_col, "Column absent; skipping odds-range check.")
        return result
    numeric = pd.to_numeric(df[odds_col], errors="coerce")
    bad_mask = numeric.notna() & (numeric <= 0)
    positions = _positions(bad_mask)
    if positions:
        values = numeric.to_numpy()[bad_mask.to_numpy()]
        for pos, val in zip(positions[:5], values[:5]):
            result.add_error(odds_col, f"Non-positive decimal odds {val}.", record_index=pos)
        result.add_error(odds_col, f"{len(positions)} non-positive odds value(s).")
    return result


def validate_date_fields(
    df: pd.DataFrame, columns: Sequence[str]
) -> ValidationResult:
    """Each date/timestamp column must parse as a datetime when present."""
    result = ValidationResult(row_count=len(df))
    for col in columns:
        if col not in df.columns:
            result.add_warning(col, "Column absent; skipping date validation.")
            continue
        series = df[col]
        if series.isna().all():
            result.add_warning(col, "Column is entirely null.")
            continue
        try:
            parsed = pd.to_datetime(series, errors="coerce")
        except (ValueError, TypeError) as exc:
            result.add_error(col, f"Could not parse dates: {exc}")
            continue
        bad_mask = parsed.isna() & series.notna()
        positions = _positions(bad_mask)
        if positions:
            for pos in positions[:5]:
                result.add_error(
                    col, f"Unparseable date value {series.iloc[pos]!r}.", record_index=pos
                )
            result.add_error(col, f"{len(positions)} unparseable date value(s).")
    return result


def validate_home_away_different(
    df: pd.DataFrame,
    home_col: str = "home_team_id",
    away_col: str = "away_team_id",
) -> ValidationResult:
    """A match may not list the same team as both home and away."""
    result = ValidationResult(row_count=len(df))
    missing = [c for c in (home_col, away_col) if c not in df.columns]
    if missing:
        result.add_warning(
            "teams", f"Column(s) {missing} absent; skipping home/away-difference check."
        )
        return result
    home = df[home_col].astype(object)
    away = df[away_col].astype(object)
    bad_mask = (home == away) & home.notna()
    positions = _positions(bad_mask)
    if positions:
        for pos in positions[:5]:
            result.add_error(
                "teams", f"Home and away team identical ({home.iloc[pos]}).", record_index=pos
            )
        result.add_error("teams", f"{len(positions)} row(s) with home == away team.")
    return result


def validate_referential_integrity(
    df: pd.DataFrame, column: str, valid_ids: Sequence, label: Optional[str] = None
) -> ValidationResult:
    """Every non-null value in ``column`` must appear in ``valid_ids``."""
    result = ValidationResult(row_count=len(df))
    field = label or column
    if column not in df.columns:
        result.add_warning(field, "Column absent; skipping referential-integrity check.")
        return result
    valid = set(valid_ids)
    series = df[column].astype(object)
    bad_mask = series.notna() & ~series.isin(valid)
    positions = _positions(bad_mask)
    if positions:
        values = series.to_numpy()
        for pos in positions[:5]:
            result.add_error(
                field,
                f"Unknown {field} identifier {values[pos]!r} not in reference set.",
                record_index=pos,
            )
        result.add_error(field, f"{len(positions)} row(s) reference unknown {field}.")
    return result


def validate_team_ids(
    df: pd.DataFrame, columns: Sequence[str], valid_ids: Optional[Sequence] = None
) -> ValidationResult:
    """Validate team/match identifier columns.

    Checks that each requested column is present and non-null. When
    ``valid_ids`` is supplied, additionally enforces that every value belongs
    to that reference set (referential integrity).
    """
    result = ValidationResult(row_count=len(df))
    result.merge(validate_required_columns(df, list(columns)))
    for col in columns:
        if col not in df.columns:
            continue
        null_count = int(df[col].isna().sum())
        if null_count:
            result.add_error(col, f"{null_count} missing team/match identifier(s).")
    if valid_ids is not None:
        for col in columns:
            if col in df.columns:
                result.merge(
                    validate_referential_integrity(df, col, valid_ids, label=col)
                )
    return result
