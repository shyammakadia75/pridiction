"""Tests for the data validation framework.

Tests cover valid records, invalid records, probability/odds validation,
missing-but-optional data, and duplicate provider identifiers. Validation
rules are never weakened to force a pass.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from football_intelligence.data.validation import (
    Severity,
    ValidationIssue,
    ValidationResult,
    validate_date_fields,
    validate_home_away_different,
    validate_no_duplicate_provider_ids,
    validate_non_negative_goals,
    validate_no_missing_values,
    validate_positive_odds,
    validate_probability_ranges,
    validate_probability_sum,
    validate_referential_integrity,
    validate_required_columns,
    validate_team_ids,
)


# ---------------------------------------------------------------------------
# Core ValidationResult behaviour
# ---------------------------------------------------------------------------
def test_empty_result_is_valid():
    result = ValidationResult()
    assert result.is_valid is True
    assert result.has_errors is False
    assert result.error_count == 0
    assert result.warning_count == 0


def test_add_error_marks_invalid():
    result = ValidationResult()
    result.add_error("goals", "negative")
    assert result.is_valid is False
    assert result.has_errors is True
    assert result.error_count == 1
    assert result.warning_count == 0


def test_add_warning_keeps_valid():
    result = ValidationResult()
    result.add_warning("col", "absent")
    assert result.is_valid is True
    assert result.has_warnings is True
    assert result.warning_count == 1


def test_raise_if_errors_raises_only_on_errors():
    ok = ValidationResult()
    ok.add_warning("c", "warn")
    ok.raise_if_errors()  # should not raise

    bad = ValidationResult()
    bad.add_error("c", "err")
    with pytest.raises(Exception):
        bad.raise_if_errors()


def test_merge_combines_issues():
    a = ValidationResult()
    a.add_error("x", "e1")
    b = ValidationResult()
    b.add_warning("y", "w1")
    a.merge(b)
    assert a.error_count == 1
    assert a.warning_count == 1
    assert a.has_errors is True


def test_required_columns_present_and_missing():
    df = pd.DataFrame({"a": [1], "b": [2]})
    assert validate_required_columns(df, ["a", "b"]).is_valid
    result = validate_required_columns(df, ["a", "c"])
    assert not result.is_valid
    assert result.error_count == 1


def test_no_missing_values_clean_and_dirty():
    clean = pd.DataFrame({"x": [1, 2, 3]})
    assert validate_no_missing_values(clean, ["x"]).is_valid

    dirty = pd.DataFrame({"x": [1, None, 3]})
    result = validate_no_missing_values(dirty, ["x"])
    assert not result.is_valid
    assert result.error_count == 1


def test_valid_match_records_pass():
    df = pd.DataFrame(
        {
            "provider": ["s", "s"],
            "provider_match_id": ["m1", "m2"],
            "home_team_id": [10, 20],
            "away_team_id": [11, 21],
            "home_goals": [1, 2],
            "away_goals": [0, 2],
        }
    )
    assert validate_no_missing_values(
        df, ["provider", "provider_match_id", "home_team_id", "away_team_id"]
    ).is_valid
    assert validate_non_negative_goals(df).is_valid
    assert validate_home_away_different(df).is_valid


def test_negative_goals_invalid():
    df = pd.DataFrame({"home_goals": [1, -1], "away_goals": [0, 2]})
    result = validate_non_negative_goals(df)
    assert not result.is_valid
    assert result.error_count == 1
    assert any(i.record_index == 1 for i in result.issues)


def test_home_equals_away_invalid():
    df = pd.DataFrame({"home_team_id": [1, 2], "away_team_id": [1, 3]})
    result = validate_home_away_different(df)
    assert not result.is_valid
    assert result.error_count == 1
    assert any(i.record_index == 0 for i in result.issues)


def test_home_equals_away_with_nulls_ignored():
    df = pd.DataFrame({"home_team_id": [1, np.nan], "away_team_id": [1, np.nan]})
    result = validate_home_away_different(df)
    assert result.error_count == 1  # row 1 (both null) is ignored


def test_probability_ranges_valid_and_invalid():
    valid = pd.DataFrame(
        {"home": [0.5, 0.3], "draw": [0.3, 0.3], "away": [0.2, 0.4]}
    )
    assert validate_probability_ranges(valid, ["home", "draw", "away"]).is_valid

    invalid = pd.DataFrame({"home": [0.5, 1.5], "draw": [0.3, 0.3], "away": [0.2, -0.8]})
    result = validate_probability_ranges(invalid, ["home", "draw", "away"])
    assert not result.is_valid
    assert result.error_count == 2


def test_probability_sum_valid_and_invalid():
    valid = pd.DataFrame({"home": [0.5, 0.4], "draw": [0.3, 0.3], "away": [0.2, 0.3]})
    assert validate_probability_sum(valid, ["home", "draw", "away"]).is_valid

    invalid = pd.DataFrame({"home": [0.5, 0.4], "draw": [0.3, 0.3], "away": [0.2, 0.2]})
    result = validate_probability_sum(invalid, ["home", "draw", "away"])
    assert not result.is_valid
    assert result.error_count == 1  # only row 1 (sum 0.9)


def test_probability_sum_missing_value_skipped():
    df = pd.DataFrame({"home": [0.5, np.nan], "draw": [0.3, 0.3], "away": [0.2, 0.7]})
    result = validate_probability_sum(df, ["home", "draw", "away"])
    assert result.is_valid  # row 1 has a null so it is not fully present


def test_positive_odds_valid_and_invalid():
    valid = pd.DataFrame({"decimal_odds": [1.5, 2.0, 3.25]})
    assert validate_positive_odds(valid).is_valid

    invalid = pd.DataFrame({"decimal_odds": [1.5, 0.0, -2.0]})
    result = validate_positive_odds(invalid)
    assert not result.is_valid
    assert result.error_count == 2


def test_date_fields_valid_and_invalid():
    valid = pd.DataFrame({"date": ["2024-08-12", "2024-08-13T15:30:00+00:00"]})
    assert validate_date_fields(valid, ["date"]).is_valid

    invalid = pd.DataFrame({"date": ["2024-08-12", "not-a-date"]})
    result = validate_date_fields(invalid, ["date"])
    assert not result.is_valid
    assert result.error_count == 1


def test_referential_integrity_unknown_value():
    df = pd.DataFrame({"team_id": [1, 2, 99]})
    result = validate_referential_integrity(df, "team_id", [1, 2, 3])
    assert not result.is_valid
    assert result.error_count == 1


def test_referential_integrity_nulls_ignored():
    df = pd.DataFrame({"team_id": [1, None, np.nan]})
    result = validate_referential_integrity(df, "team_id", [1, 2, 3])
    assert result.is_valid  # nulls are not "unknown" references


def test_no_duplicate_provider_ids_detected():
    df = pd.DataFrame({"provider": ["a", "a"], "provider_id": ["x", "x"]})
    result = validate_no_duplicate_provider_ids(df)
    assert not result.is_valid
    assert result.error_count == 1

    unique = pd.DataFrame({"provider": ["a", "b"], "provider_id": ["x", "y"]})
    assert validate_no_duplicate_provider_ids(unique).is_valid


def test_missing_optional_data_is_warning_not_error():
    df = pd.DataFrame({"home": [0.5, 0.3]})
    # draw/away absent -> warning, still valid
    result = validate_probability_ranges(df, ["draw", "away"])
    assert result.is_valid
    assert result.warning_count == 2
    assert result.error_count == 0


def test_validate_team_ids_with_reference_set():
    df = pd.DataFrame({"team_id": [1, 2, 99]})
    result = validate_team_ids(df, ["team_id"], valid_ids=[1, 2, 3])
    assert not result.is_valid  # 99 is unknown

    clean = pd.DataFrame({"team_id": [1, 2]})
    assert validate_team_ids(clean, ["team_id"], valid_ids=[1, 2, 3]).is_valid


def test_validators_do_not_silently_discard_invalid_records():
    # An invalid record must be reported, not dropped, and row_count preserved.
    df = pd.DataFrame(
        {
            "home_goals": [1, -1, 3],
            "away_goals": [0, 2, 0],
            "home_team_id": [1, 1, 1],
            "away_team_id": [2, 2, 2],
        }
    )
    result = validate_non_negative_goals(df)
    assert result.row_count == 3
    assert result.error_count == 1
    assert any(i.record_index == 1 for i in result.issues)
    assert len(df) == 3  # original data untouched

