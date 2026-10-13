"""Data-quality checks for the football-intelligence database.

This module provides reusable, deterministic checks that inspect match,
team, and competition data loaded from the database.  Checks distinguish
**errors** (data that is invalid and should block downstream processing)
from **warnings** (data that is suspicious but may still be usable).

Design principles
-----------------
* Reuse the existing :mod:`validators` framework where applicable.
* Never silently discard records; every issue is attributed to a
  specific field and, where possible, a row identifier.
* Never modify historical records -- this module is strictly read-only.
* Degradation is graceful: when a column is absent from the dataset, a
  warning is recorded and the corresponding check is skipped rather than
  raising.

Limitations
-----------
* Kickoff time is **not** stored in the schema, so same-day fixture
  ambiguity cannot be resolved here.  See :mod:`features.engineering`
  for the conservative handling policy.
* A match is considered "complete" (usable for feature history) when
  both ``home_goals`` and ``away_goals`` are non-NULL.  This is
  intentionally simpler than relying on ``match_status``, which varies
  between providers.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Sequence

import pandas as pd

from ..database import database_connection
from ..logging_config import get_logger
from .validation.validators import (
    Severity,
    ValidationIssue,
    ValidationResult,
    validate_date_fields,
    validate_home_away_different,
    validate_non_negative_goals,
    validate_no_duplicate_provider_ids,
    validate_no_missing_values,
    validate_required_columns,
)

logger = get_logger("data.quality")

__all__ = [
    "DataQualityReport",
    "check_match_data_quality",
    "check_match_data_quality_from_df",
        "load_matches_for_quality",
]


# ---------------------------------------------------------------------------
# Report dataclass
# ---------------------------------------------------------------------------


@dataclass
class DataQualityReport:
    """Aggregated report from running all data-quality checks.

    A report is **valid** when it contains no ERROR-severity issues.
    Warnings are informational and do not block downstream processing.
    """

    validation: ValidationResult = field(default_factory=ValidationResult)

    @property
    def is_valid(self) -> bool:
        return self.validation.is_valid

    @property
    def has_errors(self) -> bool:
        return self.validation.has_errors

    @property
    def has_warnings(self) -> bool:
        return self.validation.has_warnings

    @property
    def error_count(self) -> int:
        return self.validation.error_count

    @property
    def warning_count(self) -> int:
        return self.validation.warning_count

    @property
    def issue_count(self) -> int:
        return len(self.validation.issues)

    def summary(self) -> dict[str, Any]:
        """Return a machine-readable summary of the report."""
        return {
            "is_valid": self.is_valid,
            "rows_checked": self.validation.row_count,
            "total_issues": len(self.validation.issues),
            "errors": self.error_count,
            "warnings": self.warning_count,
                        "issues": [str(i) for i in self.validation.issues],
        }


# ---------------------------------------------------------------------------
# Database loading
# ---------------------------------------------------------------------------


def load_matches_for_quality(db_path: Optional[str | Path] = None) -> pd.DataFrame:
    """Load all matches with team and competition names for quality checks.

    Returns a DataFrame with columns::
        match_id, provider, provider_match_id, competition_id,
        competition_name, season_name, match_date,
        home_team_id, home_team_name, away_team_id, away_team_name,
        home_goals, away_goals, match_status

    The DataFrame is **not** sorted; callers that need chronological order
    must sort explicitly.
    """
    query = """
        SELECT
            m.id              AS match_id,
            m.provider        AS provider,
            m.provider_match_id AS provider_match_id,
            m.competition_id  AS competition_id,
            c.competition_name AS competition_name,
            m.season_name     AS season_name,
            m.match_date      AS match_date,
            m.home_team_id    AS home_team_id,
            ht.team_name      AS home_team_name,
            m.away_team_id    AS away_team_id,
            at.team_name      AS away_team_name,
            m.home_goals      AS home_goals,
            m.away_goals      AS away_goals,
            m.match_status    AS match_status
        FROM matches m
        LEFT JOIN competitions c  ON m.competition_id = c.id
        LEFT JOIN teams ht        ON m.home_team_id   = ht.id
        LEFT JOIN teams at        ON m.away_team_id   = at.id
    """
    with database_connection(db_path) as conn:
        df = pd.read_sql_query(query, conn)
    logger.info("Loaded %d match row(s) for quality analysis.", len(df))
    return df


# ---------------------------------------------------------------------------
# Individual check helpers
# ---------------------------------------------------------------------------


def _check_missing_match_identifiers(df: pd.DataFrame) -> ValidationResult:
    """Check for missing required match identifiers (provider_match_id)."""
    result = ValidationResult(row_count=len(df))
    required = ["provider_match_id", "provider"]
    result.merge(validate_required_columns(df, required))
    if all(c in df.columns for c in required):
        result.merge(validate_no_missing_values(df, required))
    return result


def _check_missing_team_identifiers(df: pd.DataFrame) -> ValidationResult:
    """Check for missing home_team_id or away_team_id."""
    result = ValidationResult(row_count=len(df))
    required = ["home_team_id", "away_team_id"]
    result.merge(validate_required_columns(df, required))
    if all(c in df.columns for c in required):
        null_home = int(df["home_team_id"].isna().sum())
        null_away = int(df["away_team_id"].isna().sum())
        if null_home:
            result.add_error("home_team_id", f"{null_home} missing home team identifier(s).")
        if null_away:
            result.add_error("away_team_id", f"{null_away} missing away team identifier(s).")
    return result


def _check_home_away_different(df: pd.DataFrame) -> ValidationResult:
    """Check that home_team_id != away_team_id (no team vs itself)."""
    return validate_home_away_different(df, "home_team_id", "away_team_id")


def _check_match_dates(df: pd.DataFrame) -> ValidationResult:
    """Check for invalid or missing match dates."""
    result = ValidationResult(row_count=len(df))
    if "match_date" not in df.columns:
        result.add_warning("match_date", "Column absent; skipping date validation.")
        return result
    null_dates = int(df["match_date"].isna().sum())
    if null_dates:
        result.add_error("match_date", f"{null_dates} missing match date(s).")
    result.merge(validate_date_fields(df, ["match_date"]))
    return result


def _check_duplicate_provider_ids(df: pd.DataFrame) -> ValidationResult:
    """Check for duplicate (provider, provider_match_id) pairs."""
    return validate_no_duplicate_provider_ids(df, "provider", "provider_match_id")


def _check_duplicate_logical_matches(df: pd.DataFrame) -> ValidationResult:
    """Check for duplicate logical matches (same date + home + away team).

    A duplicate logical match is two rows with the same ``match_date``,
    ``home_team_id``, and ``away_team_id`` that are **not** the same
    provider match (i.e. they have different provider_match_id values or
    different providers).  Exact duplicates (same provider + provider_id)
    are caught by :func:`_check_duplicate_provider_ids` and are not
    re-reported here.
    """
    result = ValidationResult(row_count=len(df))
    needed = ["match_date", "home_team_id", "away_team_id"]
    missing_cols = [c for c in needed if c not in df.columns]
    if missing_cols:
        result.add_warning(
            "logical_match",
            f"Column(s) {missing_cols} absent; skipping logical-duplicate check.",
        )
        return result

    subset = df.dropna(subset=needed)
    if len(subset) == 0:
        return result

    dup_mask = subset.duplicated(subset=needed, keep=False)
    if not dup_mask.any():
        return result

    dup_rows = subset[dup_mask]
    if "provider_match_id" in dup_rows.columns and "provider" in dup_rows.columns:
        exact_dups = dup_rows.duplicated(
            subset=["provider", "provider_match_id"], keep=False
        )
        logical_only = dup_rows[~exact_dups]
        if len(logical_only) > 0:
            sample = logical_only[needed + ["match_id"]].head(10).to_dict("records")
            result.add_error(
                "logical_match",
                f"{len(logical_only)} duplicate logical match(es) found "
                f"(same date + teams but different provider_match_id). "
                f"Samples: {sample}",
            )
    else:
        result.add_error(
            "logical_match",
            f"{len(dup_rows)} duplicate logical match(es) found "
                    f"(same date + teams).",
        )
    return result


def _check_referential_ids(
    df: pd.DataFrame,
    column: str,
    valid_ids: Sequence,
    label: str,
) -> ValidationResult:
    """Helper: every non-null value in *column* must be in *valid_ids*."""
    result = ValidationResult(row_count=len(df))
    if column not in df.columns:
        return result
    if not valid_ids:
        result.add_warning(
            column,
            f"Reference set for {label} is empty; cannot verify referential integrity.",
        )
        return result
    valid = set(valid_ids)
    series = df[column]
    bad_mask = series.notna() & ~series.astype(object).isin(valid)
    bad_positions = [i for i in range(len(df)) if bad_mask.iloc[i]]
    if bad_positions:
        values = series.iloc[bad_positions].to_numpy()
        for pos, val in zip(bad_positions[:5], values[:5]):
            result.add_error(
                column,
                f"Unknown {label} identifier {val!r} not in reference set.",
                record_index=pos,
            )
    return result


def _check_provider_consistency(df: pd.DataFrame) -> ValidationResult:
    """Check for inconsistent provider identifiers within a provider_match_id."""
    result = ValidationResult(row_count=len(df))
    cols = ["provider", "provider_match_id"]
    missing = [c for c in cols if c not in df.columns]
    if missing:
        result.add_warning(
            "provider_consistency",
            f"Column(s) {missing} absent; skipping provider consistency check.",
        )
        return result

    subset = df.dropna(subset=cols)
    if len(subset) == 0:
        return result

    grouped = subset.groupby("provider_match_id")["provider"].nunique()
    inconsistent = grouped[grouped > 1]
    if len(inconsistent) > 0:
        examples = inconsistent.head(5).index.tolist()
        result.add_error(
            "provider_consistency",
            f"{len(inconsistent)} provider_match_id(s) map to multiple providers. "
            f"Examples: {examples}",
        )
    return result


def _check_completeness_for_features(df: pd.DataFrame) -> ValidationResult:
    """Check for records that cannot safely be used for feature engineering."""
    result = ValidationResult(row_count=len(df))
    required_for_features = [
        "provider_match_id",
        "home_team_id",
        "away_team_id",
        "match_date",
    ]
    missing_cols = [c for c in required_for_features if c not in df.columns]
    if missing_cols:
        result.add_error(
            "completeness",
            f"Column(s) {missing_cols} absent; feature usability cannot be determined.",
        )
        return result
    for col in required_for_features:
        null_count = int(df[col].isna().sum())
        if null_count:
            result.add_error(
                col,
                f"{null_count} record(s) missing {col}; "
                f"cannot be used for feature engineering.",
            )
    return result


# ---------------------------------------------------------------------------
# Combined check entry points
# ---------------------------------------------------------------------------


def check_match_data_quality_from_df(
    df: pd.DataFrame,
    valid_team_ids: Optional[Sequence] = None,
    valid_comp_ids: Optional[Sequence] = None,
) -> DataQualityReport:
    """Run all data-quality checks on an in-memory DataFrame."""
    validation = ValidationResult(row_count=len(df))

    validation.merge(_check_missing_match_identifiers(df))
    validation.merge(_check_missing_team_identifiers(df))
    validation.merge(_check_home_away_different(df))
    validation.merge(_check_match_dates(df))
    validation.merge(_check_duplicate_provider_ids(df))
    validation.merge(_check_duplicate_logical_matches(df))
    validation.merge(_check_invalid_scores(df))
    if valid_team_ids is not None or valid_comp_ids is not None:
        validation.merge(
            _check_referential_integrity(
                df,
                valid_team_ids or [],
                valid_comp_ids or [],
            )
        )
    validation.merge(_check_provider_consistency(df))
    validation.merge(_check_completeness_for_features(df))

    return DataQualityReport(validation=validation)


def check_match_data_quality(
    db_path: Optional[str | Path] = None,
) -> DataQualityReport:
    """Load matches from *db_path* and run all quality checks."""
    df = load_matches_for_quality(db_path)

    valid_team_ids: list[int] = []
    valid_comp_ids: list[int] = []
    if len(df) > 0:
        # Build reference sets from both home_team_id and away_team_id,
        # since either can appear as a foreign key reference.
        for col in ("home_team_id", "away_team_id"):
            if col in df.columns:
                valid_team_ids.extend(
                    df.loc[df[col].notna(), col]
                    .drop_duplicates()
                    .astype(int)
                    .tolist()
                )
        valid_team_ids = sorted(set(valid_team_ids))
        if "competition_id" in df.columns:
            valid_comp_ids = sorted(
                df.loc[df["competition_id"].notna(), "competition_id"]
                .drop_duplicates()
                .astype(int)
                .tolist()
            )

    return check_match_data_quality_from_df(df, valid_team_ids, valid_comp_ids)


def _check_invalid_scores(df: pd.DataFrame) -> ValidationResult:
    """Check for invalid scores (negative goals).

    NULL goals are treated as 'not yet played' and generate a warning,
    not an error, because some matches may legitimately be scheduled.
    """
    result = ValidationResult(row_count=len(df))
    result.merge(validate_non_negative_goals(df, "home_goals", "away_goals"))

    for col in ("home_goals", "away_goals"):
        if col not in df.columns:
            continue
        null_count = int(df[col].isna().sum())
        if null_count:
            result.add_warning(
                col,
                f"{null_count} match record(s) have NULL {col} (incomplete).",
            )
    return result


def _check_referential_integrity(
    df: pd.DataFrame,
    valid_team_ids: Sequence,
    valid_comp_ids: Sequence,
) -> ValidationResult:
    """Check that team and competition references point to existing records."""
    result = ValidationResult(row_count=len(df))
    result.merge(_check_referential_ids(df, "home_team_id", valid_team_ids, "team"))
    result.merge(_check_referential_ids(df, "away_team_id", valid_team_ids, "team"))
    result.merge(
        _check_referential_ids(df, "competition_id", valid_comp_ids, "competition")
    )
    return result
