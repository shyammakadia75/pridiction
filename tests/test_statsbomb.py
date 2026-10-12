"""Tests for the StatsBomb Open Data ingestion adapter.

All network access is mocked; no real HTTP calls are made.  The fixtures
below are **synthetic** match/competition dictionaries forged to exercise
specific code paths -- they are NOT real StatsBomb production data and must
never be used as such.
"""
from __future__ import annotations

import argparse
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

from football_intelligence.cli import cmd_statsbomb_import, cmd_statsbomb_list
from football_intelligence.data.ingestion.base import IngestionResult
from football_intelligence.data.ingestion.statsbomb import (
    COMPETITIONS_URL,
    PROVIDER,
    STATSBOMB_BASE_URL,
    StatsBombIngestor,
    _normalize_match,
    _matches_to_dataframe,
    list_competitions,
    record_ingestion_run,
)
from football_intelligence.data.validation import (
    ValidationResult,
    validate_non_negative_goals,
)
from football_intelligence.database import initialize_database
from football_intelligence.database.connection import get_connection
from football_intelligence.exceptions import DatabaseError, IngestionError

# ===========================================================================
# SYNTHETIC TEST FIXTURES  -- NOT real StatsBomb data
# Forged locally to exercise specific code paths; DO NOT use in production.
# ===========================================================================

SYNTHETIC_COMPETITIONS = [
    {
        "competition_id": 7,
        "season_id": 3,
        "country_name": "International",
        "competition_name": "FIFA World Cup",
        "competition_gender": "male",
        "season_name": "2022",
        "match_updated": "2024-01-15T10:00:00.000",
        "match_available": True,
    },
    {
        "competition_id": 12,
        "season_id": 4,
        "country_name": "England",
        "competition_name": "Test League",
        "competition_gender": "male",
        "season_name": "2023/24",
        "match_updated": "2024-01-15T10:00:00.000",
        "match_available": True,
    },
]


def _make_match(
    match_id: int = 1001,
    competition_id: int = 7,
    season_id: int = 3,
    home_id: int = 5001,
    away_id: int = 5002,
    home_name: str = "Team Alpha",
    away_name: str = "Team Beta",
    home_score=None,
    away_score=None,
    match_status: str = "available",
    home_country: str = "Alphaland",
    away_country: str = "Betaland",
    data_version: str = "1.1.0",
    match_date: str = "2022-11-20",
    kick_off: str = "16:00:00.000",
    competition_name: str = "FIFA World Cup",
    season_name: str = "2022",
    last_updated: str = "2024-01-15T10:30:00.000",
) -> dict:
    """Build a single synthetic StatsBomb match dict."""
    return {
        "match_id": match_id,
        "match_date": match_date,
        "kick_off": kick_off,
        "competition": {
            "competition_id": competition_id,
            "country_name": "International",
            "competition_name": competition_name,
        },
        "season": {"season_id": season_id, "season_name": season_name},
        "home_team": {
            "home_team_id": home_id,
            "home_team_name": home_name,
            "home_team_gender": "male",
            "home_team_group": None,
            "country": {"id": 1, "name": home_country},
            "managers": [],
        },
        "away_team": {
            "away_team_id": away_id,
            "away_team_name": away_name,
            "away_team_gender": "male",
            "away_team_group": None,
            "country": {"id": 2, "name": away_country},
            "managers": [],
        },
        "home_score": home_score,
        "away_score": away_score,
        "match_status": match_status,
        "last_updated": last_updated,
        "metadata": {"data_version": data_version},
        "competition_stage": {"id": 1, "name": "Final"},
        "match_week": 1,
    }


SYNTHETIC_MATCHES = [
    _make_match(
        match_id=1001, home_id=5001, away_id=5002,
        home_score=1, away_score=2,
    ),
    _make_match(
        match_id=1002, home_id=5003, away_id=5004,
        home_score=0, away_score=0,
        home_name="Team Gamma", away_name="Team Delta",
    ),
]

VALID_SOURCE_URL = "http://mock.test/matches/7/3.json"
DEFAULT_MATCHES_URL = f"{STATSBOMB_BASE_URL}/matches/7/3.json"


# ===========================================================================
# Fixtures
# ===========================================================================

@pytest.fixture
def initialized_db(tmp_path: Path) -> Path:
    """A freshly initialized test database."""
    db = tmp_path / "test.db"
    initialize_database(db)
    return db


@pytest.fixture
def http_mock(monkeypatch):
    """Patch ``_http_get`` so tests never hit the network.

    Populate ``responses`` with ``url -> bytes`` mappings.
    Inspect ``calls`` to verify which URLs were requested.
    """
    from football_intelligence.data.ingestion import statsbomb as sb_mod

    responses: dict[str, bytes] = {}
    calls: list[str] = []

    def fake_get(url: str, timeout: int = 30) -> bytes:
        calls.append(url)
        if url not in responses:
            raise IngestionError(f"No mock response for: {url}")
        return responses[url]

    monkeypatch.setattr(sb_mod, "_http_get", fake_get)
    return {"responses": responses, "calls": calls}


@pytest.fixture
def ingestor(tmp_path: Path) -> StatsBombIngestor:
    """A StatsBombIngestor pointed at a mock URL and temp cache/db."""
    return StatsBombIngestor(
        competition_id=7,
        season_id=3,
        db_path=tmp_path / "test.db",
        cache_dir=tmp_path / "cache",
        source_url=VALID_SOURCE_URL,
    )

# ===========================================================================
# Normalization tests
# ===========================================================================

def test_normalize_match_flattened_correctly():
    match = _make_match(home_score=1, away_score=2, match_status="available")
    rec = _normalize_match(match)
    assert rec["provider"] == "statsbomb"
    assert rec["provider_match_id"] == "1001"
    assert rec["provider_competition_id"] == "7"
    assert rec["competition_name"] == "FIFA World Cup"
    assert rec["country_name"] == "International"
    assert rec["season_id"] == "3"
    assert rec["season_name"] == "2022"
    assert rec["match_date"] == "2022-11-20"
    assert rec["home_team_id"] == "5001"
    assert rec["home_team_name"] == "Team Alpha"
    assert rec["home_team_country"] == "Alphaland"
    assert rec["away_team_id"] == "5002"
    assert rec["away_team_name"] == "Team Beta"
    assert rec["away_team_country"] == "Betaland"
    assert rec["home_score"] == 1
    assert rec["away_score"] == 2
    assert rec["match_status"] == "available"
    assert rec["data_version"] == "1.1.0"
    assert rec["competition_stage"] == "Final"
    assert rec["last_updated"] == "2024-01-15T10:30:00.000"


def test_normalize_match_scheduled_score_is_none():
    """A scheduled match has null scores -- these must survive normalization."""
    match = _make_match(
        home_score=None, away_score=None,
        match_status="scheduled",
    )
    rec = _normalize_match(match)
    assert rec["home_score"] is None
    assert rec["away_score"] is None
    assert rec["match_status"] == "scheduled"


def test_normalize_match_handles_missing_nested_dict():
    """A match with no competition / season / country still normalizes."""
    rec = _normalize_match({})
    assert rec["provider_match_id"] is None
    assert rec["provider_competition_id"] is None
    assert rec["home_team_id"] is None
    assert rec["away_team_id"] is None
    assert rec["match_date"] is None


def test_matches_to_dataframe_empty_list_has_columns():
    df = _matches_to_dataframe([])
    assert len(df) == 0
    assert "provider_match_id" in df.columns
    assert "home_team_id" in df.columns
    assert "away_team_id" in df.columns


def test_matches_to_dataframe_populated():
    df = _matches_to_dataframe(SYNTHETIC_MATCHES)
    assert len(df) == 2
    assert df.iloc[0]["provider_match_id"] == "1001"
    assert df.iloc[1]["provider_match_id"] == "1002"

# ===========================================================================
# list_competitions tests
# ===========================================================================

def test_list_competitions_returns_dataframe(http_mock):
    http_mock["responses"][COMPETITIONS_URL] = (
        json.dumps(SYNTHETIC_COMPETITIONS).encode("utf-8")
    )
    df = list_competitions(refresh=True, source_url=COMPETITIONS_URL)
    assert len(df) == 2
    assert list(df.columns) >= ["competition_id", "season_id", "competition_name"]
    assert df.iloc[0]["competition_name"] == "FIFA World Cup"


def test_list_competitions_uses_cache(tmp_path, http_mock):
    cache = tmp_path / "cache"
    cache.mkdir()
    cache_path = cache / "competitions.json"
    cache_path.write_text(json.dumps(SYNTHETIC_COMPETITIONS), encoding="utf-8")
    df = list_competitions(cache_dir=cache)
    assert len(df) == 2
    assert http_mock["calls"] == []  # no network


def test_list_competitions_refresh_bypasses_cache(tmp_path, http_mock):
    cache = tmp_path / "cache"
    cache.mkdir()
    cache_path = cache / "competitions.json"
    cache_path.write_text(json.dumps(SYNTHETIC_COMPETITIONS), encoding="utf-8")
    http_mock["responses"][COMPETITIONS_URL] = (
        json.dumps(SYNTHETIC_COMPETITIONS).encode("utf-8")
    )
    df = list_competitions(cache_dir=cache, refresh=True, source_url=COMPETITIONS_URL)
    assert len(df) == 2
    assert COMPETITIONS_URL in http_mock["calls"]


def test_list_competitions_creates_cache_dir(tmp_path, http_mock):
    cache = tmp_path / "new_cache"
    http_mock["responses"][COMPETITIONS_URL] = (
        json.dumps(SYNTHETIC_COMPETITIONS).encode("utf-8")
    )
    list_competitions(cache_dir=cache, source_url=COMPETITIONS_URL)
    assert (cache / "competitions.json").exists()


def test_list_competitions_http_error(tmp_path):
    cache = tmp_path / "cache"
    with patch(
        "football_intelligence.data.ingestion.statsbomb._http_get"
    ) as mock_get:
        mock_get.side_effect = IngestionError("HTTP 404: Not Found")
        with pytest.raises(IngestionError, match="404"):
            list_competitions(
                cache_dir=cache, source_url=COMPETITIONS_URL,
            )


def test_list_competitions_network_timeout(tmp_path):
    cache = tmp_path / "cache"
    with patch(
        "football_intelligence.data.ingestion.statsbomb._http_get"
    ) as mock_get:
        mock_get.side_effect = IngestionError(
            "Timeout (30s) fetching competitions.json"
        )
        with pytest.raises(IngestionError, match="Timeout"):
            list_competitions(
                cache_dir=cache, source_url=COMPETITIONS_URL,
            )


def test_list_competitions_malformed_json(tmp_path, http_mock):
    http_mock["responses"][COMPETITIONS_URL] = b"<html>404 Not Found</html>"
    with pytest.raises(json.JSONDecodeError):
        list_competitions(
            cache_dir=tmp_path / "cache",
            source_url=COMPETITIONS_URL,
            refresh=True,
        )

# ===========================================================================
# retrieve tests
# ===========================================================================

def test_retrieve_valid_data(ingestor, http_mock):
    http_mock["responses"][VALID_SOURCE_URL] = (
        json.dumps(SYNTHETIC_MATCHES).encode("utf-8")
    )
    df = ingestor.retrieve()
    assert len(df) == 2
    assert df.iloc[0]["provider_match_id"] == "1001"
    assert df.iloc[0]["provider"] == "statsbomb"
    assert ingestor.was_cached is False
    assert ingestor.retrieval_timestamp is not None


def test_retrieve_uses_cache_when_available(tmp_path, http_mock):
    cache = tmp_path / "cache"
    cache.mkdir()
    cache_file = cache / "matches" / "7" / "3.json"
    cache_file.parent.mkdir(parents=True)
    cache_file.write_text(
        json.dumps(SYNTHETIC_MATCHES), encoding="utf-8"
    )
    ing = StatsBombIngestor(
        7, 3, db_path=tmp_path / "test.db",
        cache_dir=cache, source_url=VALID_SOURCE_URL,
    )
    df = ing.retrieve()
    assert len(df) == 2
    assert ing.was_cached is True
    assert http_mock["calls"] == []  # no network call


def test_retrieve_refresh_bypasses_cache(tmp_path, http_mock):
    cache = tmp_path / "cache"
    cache.mkdir()
    cache_file = cache / "matches" / "7" / "3.json"
    cache_file.parent.mkdir(parents=True)
    cache_file.write_text(
        json.dumps(SYNTHETIC_MATCHES), encoding="utf-8"
    )
    http_mock["responses"][VALID_SOURCE_URL] = (
        json.dumps(SYNTHETIC_MATCHES).encode("utf-8")
    )
    ing = StatsBombIngestor(
        7, 3, db_path=tmp_path / "test.db",
        cache_dir=cache, source_url=VALID_SOURCE_URL,
        refresh=True,
    )
    df = ing.retrieve()
    assert len(df) == 2
    assert ing.was_cached is False
    assert VALID_SOURCE_URL in http_mock["calls"]


def test_retrieve_writes_cache_file(tmp_path, http_mock):
    cache = tmp_path / "cache"
    http_mock["responses"][VALID_SOURCE_URL] = (
        json.dumps(SYNTHETIC_MATCHES).encode("utf-8")
    )
    ing = StatsBombIngestor(
        7, 3, cache_dir=cache, source_url=VALID_SOURCE_URL,
    )
    ing.retrieve()
    cache_file = cache / "matches" / "7" / "3.json"
    assert cache_file.exists()
    cached = json.loads(cache_file.read_text(encoding="utf-8"))
    assert cached == SYNTHETIC_MATCHES


def test_retrieve_empty_matches_file(tmp_path, http_mock):
    cache = tmp_path / "cache"
    http_mock["responses"][VALID_SOURCE_URL] = b"[]"
    ing = StatsBombIngestor(
        7, 3, cache_dir=cache, source_url=VALID_SOURCE_URL,
    )
    df = ing.retrieve()
    assert len(df) == 0
    assert "provider_match_id" in df.columns


def test_retrieve_non_list_response_raises(tmp_path, http_mock):
    cache = tmp_path / "cache"
    http_mock["responses"][VALID_SOURCE_URL] = (
        json.dumps({"not": "an array"}).encode("utf-8")
    )
    ing = StatsBombIngestor(
        7, 3, cache_dir=cache, source_url=VALID_SOURCE_URL,
    )
    with pytest.raises(IngestionError, match="expected a list"):
        ing.retrieve()

# ===========================================================================
# validate tests
# ===========================================================================

def test_validate_valid_records(tmp_path):
    df = _matches_to_dataframe(SYNTHETIC_MATCHES)
    ing = StatsBombIngestor(7, 3, cache_dir=tmp_path / "c")
    result = ing.validate(df)
    assert result.is_valid
    assert result.error_count == 0


def test_validate_empty_dataframe_is_valid():
    df = _matches_to_dataframe([])
    ing = StatsBombIngestor(7, 3)
    result = ing.validate(df)
    assert result.is_valid
    assert result.error_count == 0
    assert result.row_count == 0


def test_validate_missing_required_value(tmp_path):
    df = _matches_to_dataframe(SYNTHETIC_MATCHES).copy()
    df.loc[0, "provider_match_id"] = None
    ing = StatsBombIngestor(7, 3, cache_dir=tmp_path / "c")
    result = ing.validate(df)
    assert not result.is_valid
    assert result.error_count >= 1


def test_validate_invalid_date(tmp_path):
    df = _matches_to_dataframe(SYNTHETIC_MATCHES).copy()
    df.loc[0, "match_date"] = "not-a-date"
    ing = StatsBombIngestor(7, 3, cache_dir=tmp_path / "c")
    result = ing.validate(df)
    assert not result.is_valid


def test_validate_negative_score(tmp_path):
    df = _matches_to_dataframe(SYNTHETIC_MATCHES).copy()
    df.loc[0, "home_score"] = -1
    ing = StatsBombIngestor(7, 3, cache_dir=tmp_path / "c")
    result = ing.validate(df)
    assert not result.is_valid


def test_validate_same_home_and_away(tmp_path):
    df = _matches_to_dataframe(SYNTHETIC_MATCHES).copy()
    df.loc[0, "away_team_id"] = df.loc[0, "home_team_id"]
    ing = StatsBombIngestor(7, 3, cache_dir=tmp_path / "c")
    result = ing.validate(df)
    assert not result.is_valid


def test_validate_duplicate_match_ids(tmp_path):
    matches = [SYNTHETIC_MATCHES[0], SYNTHETIC_MATCHES[0].copy()]
    df = _matches_to_dataframe(matches)
    ing = StatsBombIngestor(7, 3, cache_dir=tmp_path / "c")
    result = ing.validate(df)
    assert not result.is_valid


def test_validate_null_score_is_ok(tmp_path):
    """Scheduled matches have null scores -- validation must not flag them."""
    matches = [
        _make_match(
            match_id=2001, home_id=6001, away_id=6002,
            home_score=None, away_score=None,
            match_status="scheduled",
        )
    ]
    df = _matches_to_dataframe(matches)
    ing = StatsBombIngestor(7, 3, cache_dir=tmp_path / "c")
    result = ing.validate(df)
    assert result.is_valid

# ===========================================================================
# load tests
# ===========================================================================

def test_load_inserts_records(initialized_db, ingestor):
    df = _matches_to_dataframe(SYNTHETIC_MATCHES)
    result = ingestor.load(df)
    assert result.status == "success"
    assert result.records_inserted == 2
    assert result.records_updated == 0
    conn = get_connection(initialized_db)
    comps = conn.execute(
        "SELECT * FROM competitions WHERE provider='statsbomb'"
    ).fetchall()
    assert len(comps) == 1
    teams = conn.execute(
        "SELECT * FROM teams WHERE provider='statsbomb'"
    ).fetchall()
    assert len(teams) == 4  # 2 home + 2 away
    matches = conn.execute(
        "SELECT * FROM matches WHERE provider='statsbomb'"
    ).fetchall()
    assert len(matches) == 2
    conn.close()


def test_load_idempotent_reimport(initialized_db, ingestor):
    df = _matches_to_dataframe(SYNTHETIC_MATCHES)
    r1 = ingestor.load(df)
    assert r1.records_inserted == 2
    assert r1.records_updated == 0
    r2 = ingestor.load(df)
    assert r2.records_inserted == 0
    assert r2.records_updated == 2
    conn = get_connection(initialized_db)
    count = conn.execute(
        "SELECT COUNT(*) FROM matches WHERE provider='statsbomb'"
    ).fetchone()[0]
    assert count == 2  # no duplicates
    conn.close()


def test_load_data_source_recorded(initialized_db, ingestor):
    df = _matches_to_dataframe(SYNTHETIC_MATCHES)
    ingestor.load(df)
    conn = get_connection(initialized_db)
    sources = conn.execute(
        "SELECT * FROM data_sources WHERE provider='statsbomb'"
    ).fetchall()
    assert len(sources) == 1
    src = sources[0]
    assert src["source_reference"] == VALID_SOURCE_URL
    assert src["dataset_name"] == "matches"
    assert src["provider"] == "statsbomb"
    meta = json.loads(src["metadata"])
    assert meta["competition_id"] == 7
    assert meta["season_id"] == 3
    assert meta["record_count"] == 2
    assert src["data_version"] == "1.1.0"
    conn.close()


def test_load_null_scores_stored_as_null(initialized_db, ingestor):
    matches = [
        _make_match(
            match_id=2001, home_id=6001, away_id=6002,
            home_score=None, away_score=None,
            match_status="scheduled",
        )
    ]
    df = _matches_to_dataframe(matches)
    ingestor.load(df)
    conn = get_connection(initialized_db)
    row = conn.execute(
        "SELECT home_goals, away_goals, match_status FROM matches "
        "WHERE provider_match_id='2001'"
    ).fetchone()
    assert row["home_goals"] is None
    assert row["away_goals"] is None
    assert row["match_status"] == "scheduled"
    conn.close()


def test_load_requires_initialized_db(tmp_path):
    db = tmp_path / "uninitialized.db"
    ing = StatsBombIngestor(
        7, 3, db_path=db, cache_dir=tmp_path / "cache",
    )
    df = _matches_to_dataframe(SYNTHETIC_MATCHES)
    with pytest.raises(DatabaseError, match="not initialized"):
        ing.load(df)


def test_load_transaction_rolls_back_on_failure(initialized_db, ingestor):
    """If a row fails to insert, the entire transaction is rolled back."""
    df = _matches_to_dataframe(SYNTHETIC_MATCHES).copy()
    # Negative score violates the DB CHECK constraint (home_goals >= 0),
    # causing sqlite3.IntegrityError inside the transaction.
    df.loc[0, "home_score"] = -1
    with pytest.raises(sqlite3.IntegrityError):
        ingestor.load(df)
    # No data should have been committed (transaction rolled back).
    conn = get_connection(initialized_db)
    matches = conn.execute(
        "SELECT COUNT(*) FROM matches WHERE provider='statsbomb'"
    ).fetchone()[0]
    comps = conn.execute(
        "SELECT COUNT(*) FROM competitions WHERE provider='statsbomb'"
    ).fetchone()[0]
    assert matches == 0
    assert comps == 0
    conn.close()

# ===========================================================================
# End-to-end ingest tests
# ===========================================================================

def test_ingest_full_pipeline(initialized_db, ingestor, http_mock):
    """retrieve -> validate -> load in a single ingest() call."""
    http_mock["responses"][VALID_SOURCE_URL] = (
        json.dumps(SYNTHETIC_MATCHES).encode("utf-8")
    )
    result = ingestor.ingest()
    assert result.status == "success"
    assert result.records_read == 2
    assert result.records_inserted == 2
    assert result.records_updated == 0
    assert ingestor.was_cached is False
    conn = get_connection(initialized_db)
    count = conn.execute(
        "SELECT COUNT(*) FROM matches WHERE provider='statsbomb'"
    ).fetchone()[0]
    assert count == 2
    conn.close()


def test_ingest_full_pipeline_cached(initialized_db, http_mock, tmp_path):
    """When the cache file exists, no HTTP call is made."""
    cache = tmp_path / "cache"
    cache_file = cache / "matches" / "7" / "3.json"
    cache_file.parent.mkdir(parents=True)
    cache_file.write_text(
        json.dumps(SYNTHETIC_MATCHES), encoding="utf-8"
    )
    ing = StatsBombIngestor(
        7, 3, db_path=initialized_db,
        cache_dir=cache, source_url=VALID_SOURCE_URL,
        refresh=False,
    )
    result = ing.ingest()
    assert result.status == "success"
    assert result.records_inserted == 2
    assert ing.was_cached is True
    assert http_mock["calls"] == []


def test_ingest_validation_failure(initialized_db, http_mock, tmp_path):
    """Invalid data aborts before load(); no records are persisted."""
    bad_match = _make_match(
        home_id=5001, away_id=5001,  # same team
        home_score=1, away_score=0,
    )
    http_mock["responses"][VALID_SOURCE_URL] = (
        json.dumps([bad_match]).encode("utf-8")
    )
    ing = StatsBombIngestor(
        7, 3, db_path=initialized_db,
        cache_dir=tmp_path / "cache",
        source_url=VALID_SOURCE_URL,
        refresh=True,
    )
    with pytest.raises(IngestionError, match="Validation failed"):
        ing.ingest()
    conn = get_connection(initialized_db)
    assert conn.execute(
        "SELECT COUNT(*) FROM matches"
    ).fetchone()[0] == 0
    conn.close()


def test_ingest_http_failure(initialized_db, http_mock, tmp_path):
    """HTTP failure during retrieve aborts ingest with IngestionError."""
    with patch(
        "football_intelligence.data.ingestion.statsbomb._http_get"
    ) as mock_get:
        mock_get.side_effect = IngestionError("HTTP 404: Not Found")
        ing = StatsBombIngestor(
            7, 3, db_path=initialized_db,
            cache_dir=tmp_path / "cache",
            source_url=VALID_SOURCE_URL,
            refresh=True,
        )
        with pytest.raises(IngestionError, match="404"):
            ing.ingest()

# ===========================================================================
# Constructor and edge-case tests
# ===========================================================================

def test_constructor_rejects_non_integer_ids():
    with pytest.raises(IngestionError, match="must be integers"):
        StatsBombIngestor("abc", 3)


def test_constructor_rejects_negative_ids():
    with pytest.raises(IngestionError, match="must be positive"):
        StatsBombIngestor(-1, 3)


def test_constructor_rejects_zero_season():
    with pytest.raises(IngestionError, match="must be positive"):
        StatsBombIngestor(7, 0)


def test_constructor_properties():
    ing = StatsBombIngestor(7, 3, source_url="http://example.com/m.json")
    assert ing.provider == "statsbomb"
    assert ing.dataset_name == "matches"
    assert ing.source_url == "http://example.com/m.json"
    assert ing.source_reference == "http://example.com/m.json"
    assert ing.competition_id == 7
    assert ing.season_id == 3
    assert ing.was_cached is False
    assert ing.retrieval_timestamp is None


def test_constructor_default_url():
    ing = StatsBombIngestor(7, 3)
    assert ing.source_url == (
        "https://raw.githubusercontent.com/statsbomb/open-data/master/data/"
        "matches/7/3.json"
    )


# ===========================================================================
# record_ingestion_run tests
# ===========================================================================

def test_record_ingestion_run_success(initialized_db):
    result = IngestionResult(
        provider="statsbomb",
        dataset_name="matches",
        records_read=5,
        records_inserted=3,
        records_updated=2,
        status="success",
        started_at="2024-01-01T00:00:00+00:00",
        completed_at="2024-01-01T00:01:00+00:00",
    )
    record_ingestion_run(initialized_db, result)
    conn = get_connection(initialized_db)
    rows = conn.execute(
        "SELECT * FROM ingestion_runs WHERE provider='statsbomb'"
    ).fetchall()
    assert len(rows) == 1
    row = rows[0]
    assert row["dataset_name"] == "matches"
    assert row["status"] == "success"
    assert row["records_read"] == 5
    assert row["records_inserted"] == 3
    assert row["records_updated"] == 2
    assert row["error_message"] is None
    conn.close()


def test_record_ingestion_run_failed(initialized_db):
    result = IngestionResult(
        provider="statsbomb",
        dataset_name="matches",
        status="failed",
        error_message="something went wrong",
        started_at="2024-01-01T00:00:00+00:00",
        completed_at="2024-01-01T00:01:00+00:00",
    )
    record_ingestion_run(initialized_db, result)
    conn = get_connection(initialized_db)
    row = conn.execute(
        "SELECT * FROM ingestion_runs WHERE provider='statsbomb'"
    ).fetchone()
    assert row["status"] == "failed"
    assert "something went wrong" in row["error_message"]
    conn.close()

# ===========================================================================
# CLI tests
# ===========================================================================

def test_cli_list_success(http_mock, tmp_path):
    http_mock["responses"][COMPETITIONS_URL] = (
        json.dumps(SYNTHETIC_COMPETITIONS).encode("utf-8")
    )
    args = argparse.Namespace(
        command="statsbomb", action="list",
        refresh=True, cache_dir=str(tmp_path / "cache"), timeout=30,
    )
    ret = cmd_statsbomb_list(args)
    assert ret == 0
    assert COMPETITIONS_URL in http_mock["calls"]


def test_cli_list_empty(http_mock, tmp_path):
    http_mock["responses"][COMPETITIONS_URL] = b"[]"
    args = argparse.Namespace(
        command="statsbomb", action="list",
        refresh=True, cache_dir=str(tmp_path / "cache"), timeout=30,
    )
    ret = cmd_statsbomb_list(args)
    assert ret == 0


def test_cli_list_http_error(tmp_path):
    with patch(
        "football_intelligence.data.ingestion.statsbomb._http_get"
    ) as mock_get:
        mock_get.side_effect = IngestionError("HTTP 404: Not Found")
        args = argparse.Namespace(
            command="statsbomb", action="list",
            refresh=True, cache_dir=str(tmp_path / "cache"), timeout=30,
        )
        ret = cmd_statsbomb_list(args)
    assert ret == 1


def test_cli_import_success(initialized_db, http_mock, tmp_path):
    http_mock["responses"][DEFAULT_MATCHES_URL] = (
        json.dumps(SYNTHETIC_MATCHES).encode("utf-8")
    )
    args = argparse.Namespace(
        command="statsbomb", action="import",
        competition_id=7, season_id=3,
        db_path=str(initialized_db),
        cache_dir=str(tmp_path / "cache"),
        refresh=True, timeout=30,
    )
    ret = cmd_statsbomb_import(args)
    assert ret == 0
    conn = get_connection(initialized_db)
    matches = conn.execute(
        "SELECT * FROM matches WHERE provider='statsbomb'"
    ).fetchall()
    assert len(matches) == 2
    runs = conn.execute(
        "SELECT * FROM ingestion_runs WHERE provider='statsbomb'"
    ).fetchall()
    assert len(runs) == 1
    assert runs[0]["status"] == "success"
    assert runs[0]["records_inserted"] == 2
    conn.close()


def test_cli_import_failure_records_run(initialized_db, http_mock, tmp_path):
    with patch(
        "football_intelligence.data.ingestion.statsbomb._http_get"
    ) as mock_get:
        mock_get.side_effect = IngestionError("HTTP 404: Not Found")
        args = argparse.Namespace(
            command="statsbomb", action="import",
            competition_id=7, season_id=3,
            db_path=str(initialized_db),
            cache_dir=str(tmp_path / "cache"),
            refresh=True, timeout=30,
        )
        ret = cmd_statsbomb_import(args)
    assert ret == 1
    conn = get_connection(initialized_db)
    runs = conn.execute(
        "SELECT * FROM ingestion_runs WHERE provider='statsbomb'"
    ).fetchall()
    assert len(runs) == 1
    assert runs[0]["status"] == "failed"
    assert "404" in runs[0]["error_message"]
    conn.close()


def test_cli_import_validation_failure(initialized_db, http_mock, tmp_path):
    bad_match = _make_match(
        home_id=5001, away_id=5001,  # same team -> validation error
        home_score=1, away_score=0,
    )
    http_mock["responses"][DEFAULT_MATCHES_URL] = (
        json.dumps([bad_match]).encode("utf-8")
    )
    args = argparse.Namespace(
        command="statsbomb", action="import",
        competition_id=7, season_id=3,
        db_path=str(initialized_db),
        cache_dir=str(tmp_path / "cache"),
        refresh=True, timeout=30,
    )
    ret = cmd_statsbomb_import(args)
    assert ret == 1
    conn = get_connection(initialized_db)
    runs = conn.execute(
        "SELECT * FROM ingestion_runs WHERE provider='statsbomb'"
    ).fetchall()
    assert len(runs) == 1
    assert runs[0]["status"] == "failed"
    conn.close()

# ===========================================================================
# Path traversal / security tests
# ===========================================================================

def test_safe_cache_path_rejects_traversal(tmp_path):
    from football_intelligence.data.ingestion.statsbomb import _safe_cache_path
    base = tmp_path / "cache"
    base.mkdir()
    # A relative path with .. that would escape base is rejected.
    with pytest.raises(IngestionError, match="Unsafe cache path"):
        _safe_cache_path(base, "matches", "..", "..", "evil.json")


def test_safe_cache_path_allows_legitimate(tmp_path):
    from football_intelligence.data.ingestion.statsbomb import _safe_cache_path
    base = tmp_path / "cache"
    base.mkdir()
    path = _safe_cache_path(base, "matches", "7", "3.json")
    assert path.is_relative_to(base.resolve())


def test_constructor_prevents_path_traversal_via_ids():
    """Non-integer IDs are rejected before they reach the cache path."""
    with pytest.raises(IngestionError):
        StatsBombIngestor("7/../../etc/passwd", 3)


# ===========================================================================
# Constants / module-level tests
# ===========================================================================

def test_provider_constant():
    assert PROVIDER == "statsbomb"


def test_required_columns_include_key_fields():
    from football_intelligence.data.ingestion.statsbomb import REQUIRED_COLUMNS
    for col in [
        "provider_match_id", "provider_competition_id",
        "home_team_id", "away_team_id", "match_date",
    ]:
        assert col in REQUIRED_COLUMNS


def test_competitions_url_format():
    assert COMPETITIONS_URL.endswith("competitions.json")
    assert "raw.githubusercontent.com" in COMPETITIONS_URL









