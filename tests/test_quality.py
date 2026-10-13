"""Tests for the data-quality module.

All tests use synthetic fixtures in disposable databases.  No real data
is read or written.  Test data is clearly labelled as synthetic.
"""
from __future__ import annotations

import pandas as pd

from football_intelligence.data.quality import (
    check_match_data_quality,
    check_match_data_quality_from_df,
)
from football_intelligence.data.validation.validators import Severity

from _fixtures import (
    _insert_match,
    make_seeded_db,
    make_initialized_db,
    _connect,
)

NOW = "2024-01-01T00:00:00+00:00"



# ---------------------------------------------------------------------------
# 1. Valid historical matches
# ---------------------------------------------------------------------------

def test_valid_matches_produce_no_errors(tmp_path):
    db = make_seeded_db(tmp_path)
    with _connect(db) as conn:
        _insert_match(conn, provider_match_id="valid-1",
                      home_team_id=1, away_team_id=2,
                      match_date="2024-08-12T15:00:00+00:00",
                      home_goals=2, away_goals=1)
        _insert_match(conn, provider_match_id="valid-2",
                      home_team_id=3, away_team_id=4,
                      match_date="2024-08-13T15:00:00+00:00",
                      home_goals=1, away_goals=1)
    report = check_match_data_quality(db)
    assert report.has_errors is False
    assert report.error_count == 0


# ---------------------------------------------------------------------------
# 2. Missing required match identifiers
# ---------------------------------------------------------------------------

def test_missing_provider_match_id_is_error():
    """Provider-match-id is NOT NULL at the DB level, so test at DataFrame level."""
    df = pd.DataFrame({
        "match_id": [1],
        "provider": ["test"],
        "provider_match_id": [None],
        "competition_id": [1],
        "home_team_id": [1],
        "away_team_id": [2],
        "match_date": ["2024-08-12T15:00:00+00:00"],
        "home_goals": [1],
        "away_goals": [0],
    })
    report = check_match_data_quality_from_df(df)
    assert report.has_errors is True
    fields = [i.field_name for i in report.validation.issues
              if i.severity == Severity.ERROR]
    assert "provider_match_id" in fields


def test_missing_provider_is_error():
    """Provider is NOT NULL at the DB level, so test at DataFrame level."""
    df = pd.DataFrame({
        "match_id": [1],
        "provider": [None],
        "provider_match_id": ["m1"],
        "competition_id": [1],
        "home_team_id": [1],
        "away_team_id": [2],
        "match_date": ["2024-08-12T15:00:00+00:00"],
        "home_goals": [1],
        "away_goals": [0],
    })
    report = check_match_data_quality_from_df(df)
    assert report.has_errors is True


# ---------------------------------------------------------------------------
# 3. Duplicate provider match identifiers
# ---------------------------------------------------------------------------

def test_duplicate_provider_match_id_detected():
    """Schema enforces UNIQUE on (provider, provider_match_id), so test at DF level."""
    df = pd.DataFrame({
        "match_id": [1, 2],
        "provider": ["test", "test"],
        "provider_match_id": ["dup-1", "dup-1"],
        "competition_id": [1, 1],
        "home_team_id": [1, 1],
        "away_team_id": [2, 2],
        "match_date": ["2024-08-12T15:00:00+00:00", "2024-08-13T15:00:00+00:00"],
        "home_goals": [1, 2],
        "away_goals": [0, 1],
    })
    report = check_match_data_quality_from_df(df)
    assert report.has_errors is True
    dup_issues = [i for i in report.validation.issues
                  if "duplicate" in i.message.lower()]
    assert len(dup_issues) >= 1
    assert all(i.severity == Severity.ERROR for i in dup_issues)

# ---------------------------------------------------------------------------
# 4. Invalid scores (negative goals)
# ---------------------------------------------------------------------------

def test_negative_goals_detected_in_dataframe():
    df = pd.DataFrame({
        "match_id": [1],
        "provider": ["test"],
        "provider_match_id": ["m1"],
        "home_team_id": [1],
        "away_team_id": [2],
        "match_date": ["2024-08-12T15:00:00+00:00"],
        "home_goals": [-1],
        "away_goals": [0],
    })
    report = check_match_data_quality_from_df(df)
    assert report.has_errors is True
    neg_issues = [i for i in report.validation.issues if "negative" in i.message.lower()]
    assert len(neg_issues) >= 1


def test_negative_away_goals_detected():
    df = pd.DataFrame({
        "match_id": [1],
        "provider": ["test"],
        "provider_match_id": ["m1"],
        "home_team_id": [1],
        "away_team_id": [2],
        "match_date": ["2024-08-12T15:00:00+00:00"],
        "home_goals": [0],
                "away_goals": [-3],
    })
    report = check_match_data_quality_from_df(df)
    assert report.has_errors is True


# ---------------------------------------------------------------------------
# 5. Home team equals away team
# ---------------------------------------------------------------------------

def test_home_team_equals_away_team_detected():
    df = pd.DataFrame({
        "match_id": [1],
        "provider": ["test"],
        "provider_match_id": ["m1"],
        "home_team_id": [1],
        "away_team_id": [1],
        "match_date": ["2024-08-12T15:00:00+00:00"],
        "home_goals": [1],
        "away_goals": [0],
    })
    report = check_match_data_quality_from_df(df)
    assert report.has_errors is True
    same_team = [i for i in report.validation.issues if "identical" in i.message.lower()]
    assert len(same_team) >= 1


# ---------------------------------------------------------------------------
# 6. Missing match dates
# ---------------------------------------------------------------------------

def test_missing_match_date_is_error(tmp_path):
    db = make_seeded_db(tmp_path)
    with _connect(db) as conn:
        row = {
            "provider": "test", "provider_match_id": "m1",
            "competition_id": 1, "season_name": "2024/25",
            "match_date": None, "home_team_id": 1, "away_team_id": 2,
            "home_goals": 1, "away_goals": 0, "match_status": "available",
            "created_at": NOW, "updated_at": NOW,
        }
        cols = list(row)
        ph = ",".join("?" * len(cols))
        conn.execute(
            f"INSERT INTO matches ({','.join(cols)}) VALUES ({ph})",
            [row[c] for c in cols],
        )
    report = check_match_data_quality(db)
    assert report.has_errors is True
    date_issues = [i for i in report.validation.issues
                   if i.field_name == "match_date" and i.severity == Severity.ERROR]
    assert len(date_issues) >= 1


def test_invalid_match_date_format_detected():
    df = pd.DataFrame({
        "match_id": [1],
        "provider": ["test"],
        "provider_match_id": ["m1"],
        "home_team_id": [1],
        "away_team_id": [2],
        "match_date": ["not-a-date"],
        "home_goals": [1],
        "away_goals": [0],
    })
    report = check_match_data_quality_from_df(df)
    assert report.has_errors is True
    date_issues = [i for i in report.validation.issues if i.field_name == "match_date"]
    assert len(date_issues) >= 1


# ---------------------------------------------------------------------------
# 7. Referential integrity (nonexistent teams/competitions)
# ---------------------------------------------------------------------------

def test_nonexistent_team_reference_detected():
    df = pd.DataFrame({
        "match_id": [1],
        "provider": ["test"],
        "provider_match_id": ["m1"],
        "competition_id": [1],
        "home_team_id": [999],
        "away_team_id": [2],
        "match_date": ["2024-08-12T15:00:00+00:00"],
        "home_goals": [1],
        "away_goals": [0],
    })
    report = check_match_data_quality_from_df(
        df, valid_team_ids=[1, 2], valid_comp_ids=[1]
    )
    assert report.has_errors is True
    ref_issues = [i for i in report.validation.issues if "Unknown" in i.message]
    assert len(ref_issues) >= 1


def test_nonexistent_competition_reference_detected():
    df = pd.DataFrame({
        "match_id": [1],
        "provider": ["test"],
        "provider_match_id": ["m1"],
        "competition_id": [999],
        "home_team_id": [1],
        "away_team_id": [2],
        "match_date": ["2024-08-12T15:00:00+00:00"],
        "home_goals": [1],
        "away_goals": [0],
    })
    report = check_match_data_quality_from_df(
        df, valid_team_ids=[1, 2], valid_comp_ids=[1]
    )
    assert report.has_errors is True


# ---------------------------------------------------------------------------
# 8. Inconsistent provider identifiers
# ---------------------------------------------------------------------------

def test_inconsistent_provider_for_same_match_id_detected():
    df = pd.DataFrame({
        "match_id": [1, 2],
        "provider": ["statsbomb", "football-data"],
        "provider_match_id": ["22912", "22912"],
        "competition_id": [1, 1],
        "home_team_id": [1, 1],
        "away_team_id": [2, 2],
        "match_date": ["2024-08-12T15:00:00+00:00", "2024-08-12T15:00:00+00:00"],
        "home_goals": [1, 1],
        "away_goals": [0, 0],
    })
    report = check_match_data_quality_from_df(df)
    assert report.has_errors is True
    prov_issues = [i for i in report.validation.issues if "multiple providers" in i.message]
    assert len(prov_issues) >= 1


# ---------------------------------------------------------------------------
# 9. Duplicate logical matches (same date + teams, different provider_id)
# ---------------------------------------------------------------------------

def test_duplicate_logical_matches_detected():
    df = pd.DataFrame({
        "match_id": [1, 2],
        "provider": ["test", "test"],
        "provider_match_id": ["m1", "m2"],
        "competition_id": [1, 1],
        "home_team_id": [1, 1],
        "away_team_id": [2, 2],
        "match_date": ["2024-08-12T15:00:00+00:00", "2024-08-12T15:00:00+00:00"],
        "home_goals": [1, 1],
        "away_goals": [0, 0],
    })
    report = check_match_data_quality_from_df(df)
    assert report.has_errors is True
    logical_issues = [i for i in report.validation.issues if "logical" in i.message.lower()]
    assert len(logical_issues) >= 1


# ---------------------------------------------------------------------------
# 10. Warnings do not block validity
# ---------------------------------------------------------------------------

def test_null_goals_generate_warning_not_error(tmp_path):
    db = make_seeded_db(tmp_path)
    with _connect(db) as conn:
        _insert_match(conn, provider_match_id="scheduled-1",
                      home_team_id=1, away_team_id=2,
                      match_date="2024-08-12T15:00:00+00:00",
                      home_goals=None, away_goals=None,
                      match_status="scheduled")
    report = check_match_data_quality(db)
    assert report.is_valid is True
    assert report.has_warnings is True


# ---------------------------------------------------------------------------
# 11. Empty database
# ---------------------------------------------------------------------------

def test_empty_database_valid(tmp_path):
    db = make_initialized_db(tmp_path)
    report = check_match_data_quality(db)
    assert report.has_errors is False
    assert report.is_valid is True


# ---------------------------------------------------------------------------
# 12. Report summary is machine-readable
# ---------------------------------------------------------------------------

def test_report_summary_structure():
    df = pd.DataFrame({
        "match_id": [1],
        "provider": ["test"],
        "provider_match_id": ["m1"],
        "competition_id": [1],
        "home_team_id": [1],
        "away_team_id": [1],
        "match_date": ["invalid"],
        "home_goals": [1],
        "away_goals": [0],
    })
    report = check_match_data_quality_from_df(df)
    summary = report.summary()
    assert "is_valid" in summary
    assert "errors" in summary
    assert "warnings" in summary
    assert "rows_checked" in summary
    assert "issues" in summary
    assert isinstance(summary["issues"], list)
    assert summary["is_valid"] is False


# ---------------------------------------------------------------------------
# 13. DataQualityReport properties
# ---------------------------------------------------------------------------

def test_report_properties():
    df = pd.DataFrame({
        "match_id": [1],
        "provider": ["test"],
        "provider_match_id": ["m1"],
        "competition_id": [1],
        "home_team_id": [999],
        "away_team_id": [1],
        "match_date": [None],
        "home_goals": [1],
        "away_goals": [0],
    })
    report = check_match_data_quality_from_df(
        df, valid_team_ids=[1, 2], valid_comp_ids=[1]
    )
    assert report.has_errors is True
    assert report.is_valid is False
    assert report.error_count >= 3  # missing date, missing team ref, same team