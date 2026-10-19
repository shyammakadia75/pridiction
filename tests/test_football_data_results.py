"""Tests for the Football-Data.co.uk results adapter.

All network access is mocked and every database write targets a disposable
temp DB.  Tests never touch the production database.
"""
from __future__ import annotations

import io
import json
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

from football_intelligence.cli import build_parser, main
from football_intelligence.data.ingestion.base import IngestionResult
from football_intelligence.data.ingestion.football_data_results import (
    DATASET_NAME,
    FootballDataResultsIngestor,
    _compose_provider_match_id,
    _league_display,
    _normalize_results_dataframe,
    _season_code_to_name,
    list_result_leagues,
    validate_outcome_score_consistency,
)
from football_intelligence.data.ingestion.odds import (
    PROVIDER,
    _parse_csv,
)
from football_intelligence.database import initialize_database
from football_intelligence.database.connection import database_connection
from football_intelligence.exceptions import IngestionError

# ---------------------------------------------------------------------------
# Synthetic CSV fixtures  -- NOT real Football-Data production data.
# ---------------------------------------------------------------------------

VALID_CSV = (
    "Date,HomeTeam,AwayTeam,FTHG,FTAG,FTR\n"
    "13/08/2016,Arsenal,Burnley,1,0,H\n"
    "13/08/2016,Man United,Hull City,2,2,D\n"
    "20/08/2016,Chelsea,Tottenham,0,1,A\n"
)

ISO_CSV = (
    "Date,HomeTeam,AwayTeam,FTHG,FTAG,FTR\n"
    "2016-09-10,Arsenal,Burnley,1,0,H\n"
)

VALID_URL = "http://mock/football-data/E0/1617.csv"
DEFAULT_URL = "https://www.football-data.co.uk/mmz4281/1617/E0.csv"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def db_path(tmp_path: Path) -> Path:
    """A freshly initialized, disposable SQLite database."""
    db = tmp_path / "test.db"
    initialize_database(db)
    return db


@pytest.fixture
def cache_dir(tmp_path: Path) -> Path:
    return tmp_path / "cache"


@pytest.fixture
def ingestor(tmp_path: Path, cache_dir: Path) -> FootballDataResultsIngestor:
    """An ingestor pointed at a mock URL with an initialized temp DB."""
    return FootballDataResultsIngestor(
        league_code="E0",
        season_code="1617",
        db_path=tmp_path / "test.db",
        cache_dir=cache_dir,
        source_url=VALID_URL,
    )


@pytest.fixture
def http_mock(monkeypatch):
    """Patch the canonical ``_http_get`` so tests never hit the network.

    Populate ``responses`` with ``url -> bytes`` mappings (used to exercise the
    real download/cache pipeline incl. UTF-8 BOM handling).
    """
    responses: dict[str, bytes] = {}
    calls: list[str] = []

    def fake_get(url: str, timeout: int = 30) -> bytes:
        calls.append(url)
        if url not in responses:
            raise IngestionError(f"No mock response for: {url}")
        return responses[url]

    monkeypatch.setattr(
        "football_intelligence.data.ingestion.odds._http_get", fake_get
    )
    return {"responses": responses, "calls": calls}


@pytest.fixture
def csv_mock(monkeypatch):
    """Patch the in-memory ``_download_csv`` used by :meth:`retrieve`.

    Populate ``responses`` with ``url -> csv_text`` mappings.
    """
    responses: dict[str, str] = {}
    calls: list[str] = []

    def fake_download(url, cache_path, timeout, refresh):
        calls.append(url)
        if url not in responses:
            raise IngestionError(f"No mock response for: {url}")
        return responses[url], False

    monkeypatch.setattr(
        "football_intelligence.data.ingestion.football_data_results._download_csv",
        fake_download,
    )
    return {"responses": responses, "calls": calls}


def _seed_statsbomb_fixture(
    db_path: Path, date_iso: str, home: str, away: str,
    home_goals: int = 1, away_goals: int = 0, match_id: str = "sb_cross",
):
    """Insert a *StatsBomb* fixture (different provider) for cross-source tests."""
    with database_connection(db_path) as conn:
        cid = conn.execute(
            "INSERT INTO competitions (provider, provider_competition_id, "
            "competition_name, country, created_at, updated_at) "
            "VALUES ('statsbomb', '2', 'Premier League', 'England', "
            "datetime('now'), datetime('now'))"
        ).lastrowid
        ht = conn.execute(
            "INSERT INTO teams (provider, provider_team_id, team_name, country, "
            "created_at, updated_at) VALUES "
            "('statsbomb', ?, ?, 'England', datetime('now'), datetime('now'))",
            (home.lower(), home),
        ).lastrowid
        at = conn.execute(
            "INSERT INTO teams (provider, provider_team_id, team_name, country, "
            "created_at, updated_at) VALUES "
            "('statsbomb', ?, ?, 'England', datetime('now'), datetime('now'))",
            (away.lower(), away),
        ).lastrowid
        conn.execute(
            "INSERT INTO matches (provider, provider_match_id, competition_id, "
            "season_name, match_date, home_team_id, away_team_id, home_goals, "
            "away_goals, match_status, created_at, updated_at) "
            "VALUES ('statsbomb', ?, ?, ?, ?, ?, ?, ?, ?, "
            "'available', datetime('now'), datetime('now'))",
            (match_id, cid, "2016", date_iso, ht, at, home_goals, away_goals),
        )


# ===========================================================================
# Pure-function / no-DB normalization tests
# ===========================================================================


def test_season_code_to_name():
    assert _season_code_to_name("1617") == "2016/2017"
    assert _season_code_to_name("1516") == "2015/2016"
    assert _season_code_to_name("1819") == "2018/2019"


@pytest.mark.parametrize("bad", ["", "161", "abc", "16170", " 161 ", None])
def test_season_code_invalid_raises(bad):
    with pytest.raises(IngestionError):
        _season_code_to_name(bad)


def test_league_display_splits_country():
    competition, country = _league_display("E0")
    assert competition == "Premier League"
    assert country == "England"
    competition, country = _league_display("SC0")
    assert competition == "Premiership"
    assert country == "Scotland"


def test_league_display_invalid_raises():
    with pytest.raises(IngestionError):
        _league_display("ZZ")


def test_compose_provider_match_id_stable_and_marked():
    key = _compose_provider_match_id("1617", "2016-08-13", "arsenal", "burnley")
    assert key == "football-data|1617|2016-08-13|arsenal|burnley"
    # Clearly marked as project-derived, never presented as source-issued.
    assert key.startswith("football-data|")
    assert _compose_provider_match_id("1617", "2016-08-13", "arsenal", "burnley") == key


def test_normalize_results_dataframe_field_mapping(csv_mock):
    csv_mock["responses"][VALID_URL] = VALID_CSV
    ing = FootballDataResultsIngestor(
        league_code="E0", season_code="1617",
        db_path=Path("/tmp/unused.db"), source_url=VALID_URL,
        cache_dir=Path("/tmp/unused_cache"),
    )
    # retrieve() returns the normalized frame (download is mocked).
    df = ing.retrieve()
    assert len(df) == 3
    row = df.iloc[0]
    assert row["provider"] == "football-data"
    assert row["provider_competition_id"] == "E0"
    assert row["competition_name"] == "Premier League"
    assert row["country_name"] == "England"
    assert row["season_name"] == "2016/2017"
    assert row["match_date"] == "2016-08-13"
    assert row["home_team_id"] == "arsenal"
    assert row["home_team_name"] == "Arsenal"
    assert row["away_team_id"] == "burnley"
    assert row["home_score"] == 1
    assert row["away_score"] == 0
    assert row["ftr"] == "H"
    assert row["match_status"] == "available"
    assert row["provider_match_id"].startswith("football-data|1617|2016-08-13|")
    assert row["ftr"] == "H"  # FTR retained on the frame for validation only


def test_normalize_results_dataframe_parses_date_formats():
    raw = _parse_csv(VALID_CSV)
    df = _normalize_results_dataframe(raw, "E0", "1617")
    # 13/08/2016 (DD/MM/YYYY) and 2016-08-13 (ISO) both normalize correctly.
    assert df.iloc[0]["match_date"] == "2016-08-13"
    assert df.iloc[1]["match_date"] == "2016-08-13"
    assert df.iloc[2]["match_date"] == "2016-08-20"


def test_normalize_team_alias_man_united():
    df = _normalize_results_dataframe(_parse_csv(VALID_CSV), "E0", "1617")
    assert df.iloc[1]["home_team_id"] == "manchester united"  # "Man United"
    assert df.iloc[2]["away_team_id"] == "tottenham hotspur"  # "Tottenham"


@pytest.mark.parametrize(
    "ftr,home,away",
    [("H", 1, 0), ("A", 0, 2), ("D", 1, 1), ("H", 3, 0)],
)
def test_validate_outcome_consistency_ok(ftr, home, away):
    df = pd.DataFrame([{"ftr": ftr, "home_score": home, "away_score": away}])
    assert not validate_outcome_score_consistency(df).has_errors


def test_validate_outcome_score_consistency_mismatch():
    df = pd.DataFrame([{"ftr": "D", "home_score": 1, "away_score": 0}])
    report = validate_outcome_score_consistency(df)
    assert report.has_errors


def test_validate_outcome_score_consistency_invalid_char():
    df = pd.DataFrame([{"ftr": "X", "home_score": 1, "away_score": 0}])
    report = validate_outcome_score_consistency(df)
    assert report.has_errors


# ===========================================================================
# Database integration tests (temp DB only)
# ===========================================================================


def _result_for(db_path: Path, ing: FootballDataResultsIngestor, csv_text: str) -> IngestionResult:
    with patch(
        "football_intelligence.data.ingestion.football_data_results._download_csv",
        return_value=(csv_text, False),
        ):
        return ing.ingest()



def test_ingest_full_pipeline(db_path, cache_dir, csv_mock):
    csv_mock["responses"][VALID_URL] = VALID_CSV
    ing = FootballDataResultsIngestor(
        league_code="E0", season_code="1617", db_path=db_path,
        cache_dir=cache_dir, source_url=VALID_URL,
    )
    result = ing.ingest()
    assert result.status == "success"
    assert result.records_read == 3
    assert result.records_inserted == 3
    assert result.records_updated == 0
    assert result.records_skipped == 0
    assert result.records_ambiguous == 0
    assert result.dataset_name == "results"
    assert result.provider == "football-data"


def test_ingest_persists_competition_teams_matches(db_path, cache_dir, csv_mock):
    csv_mock["responses"][VALID_URL] = VALID_CSV
    ing = FootballDataResultsIngestor(
        league_code="E0", season_code="1617", db_path=db_path,
        cache_dir=cache_dir, source_url=VALID_URL,
    )
    ing.ingest()
    with database_connection(db_path) as c:
        comps = [dict(r) for r in c.execute(
            "SELECT provider, provider_competition_id, competition_name, country "
            "FROM competitions WHERE provider='football-data'"
        )]
        assert len(comps) == 1
        assert comps[0]["competition_name"] == "Premier League"
        assert comps[0]["country"] == "England"
        teams = [dict(r) for r in c.execute(
            "SELECT provider_team_id, team_name, country FROM teams "
            "WHERE provider='football-data' ORDER BY provider_team_id"
        )]
        ids = {t["provider_team_id"] for t in teams}
        assert {"arsenal", "burnley", "manchester united", "tottenham hotspur"} <= ids
        arsenal = next(t for t in teams if t["provider_team_id"] == "arsenal")
        assert arsenal["team_name"] == "Arsenal"
        assert arsenal["country"] == "England"
        matches = [dict(r) for r in c.execute(
            "SELECT provider, provider_match_id, match_date, home_goals, "
            "away_goals, match_status FROM matches ORDER BY match_date"
        )]
        assert len(matches) == 3
        first = matches[0]
        assert first["provider"] == "football-data"
        assert first["provider_match_id"].startswith("football-data|1617|2016-08-13|")
        assert first["home_goals"] == 1
        assert first["away_goals"] == 0
        assert first["match_status"] == "available"


def test_ingest_records_data_source_provenance(db_path, cache_dir, csv_mock):
    csv_mock["responses"][VALID_URL] = VALID_CSV
    ing = FootballDataResultsIngestor(
        league_code="E0", season_code="1617", db_path=db_path,
        cache_dir=cache_dir, source_url=VALID_URL,
    )
    ing.ingest()
    with database_connection(db_path) as c:
        rows = [dict(r) for r in c.execute(
            "SELECT provider, dataset_name, source_reference, data_version, metadata "
            "FROM data_sources WHERE provider='football-data'"
        )]
        assert len(rows) == 1
        row = rows[0]
        assert row["dataset_name"] == "results"
        assert row["source_reference"] == VALID_URL
        meta = json.loads(row["metadata"])
        assert meta["league_code"] == "E0"
        assert meta["season_code"] == "1617"
        assert meta["season_name"] == "2016/2017"
        assert "composite" in meta["provider_match_id_strategy"]
    assert meta["ftr_usage"] == "validation only; not stored in matches"


def test_ingest_does_not_touch_odds_table(db_path, cache_dir, csv_mock):
    csv_mock["responses"][VALID_URL] = VALID_CSV
    ing = FootballDataResultsIngestor(
        league_code="E0", season_code="1617", db_path=db_path,
        cache_dir=cache_dir, source_url=VALID_URL,
    )
    ing.ingest()
    with database_connection(db_path) as c:
        assert c.execute("SELECT COUNT(*) FROM odds").fetchone()[0] == 0


def test_ingest_idempotent_rerun_updates(db_path, cache_dir, csv_mock):
    csv_mock["responses"][VALID_URL] = VALID_CSV
    ing = FootballDataResultsIngestor(
        league_code="E0", season_code="1617", db_path=db_path,
        cache_dir=cache_dir, source_url=VALID_URL,
    )
    r1 = ing.ingest()
    assert r1.records_inserted == 3
    r2 = ing.ingest()
    assert r2.records_inserted == 0
    assert r2.records_updated == 3
    with database_connection(db_path) as c:
        assert c.execute("SELECT COUNT(*) FROM matches").fetchone()[0] == 3


def test_ingest_parses_iso_and_dmy_dates(csv_mock):
    csv_mock["responses"][VALID_URL] = ISO_CSV
    ing = FootballDataResultsIngestor(
        league_code="E0", season_code="1617",
        db_path=Path("/tmp/unused.db"), cache_dir=Path("/tmp/unused_cache"),
        source_url=VALID_URL,
    )
    df = ing.retrieve()
    assert df.iloc[0]["match_date"] == "2016-09-10"
    assert df.iloc[0]["ftr"] == "H"


def test_validate_missing_required_columns():
    df = pd.DataFrame({"Date": ["x"], "HomeTeam": ["a"]})
    ing = FootballDataResultsIngestor(
        league_code="E0", season_code="1617", source_url=VALID_URL,
        cache_dir=Path("/tmp/x"), db_path=Path("/tmp/x.db"),
    )
    assert ing.validate(df).has_errors


def test_validate_rejects_negative_goals():
    df = pd.DataFrame([dict(
        provider="football-data", provider_match_id="k", provider_competition_id="E0",
        competition_name="Premier League", match_date="2016-08-13",
        home_team_id="a", away_team_id="b", home_score=-1, away_score=0,
        ftr="H", match_status="available",
    )])
    ing = FootballDataResultsIngestor(
        league_code="E0", season_code="1617", source_url=VALID_URL,
        cache_dir=Path("/tmp/x"), db_path=Path("/tmp/x.db"),
    )
    assert ing.validate(df).has_errors


def test_validate_rejects_same_team_home_away():
    df = pd.DataFrame([dict(
        provider="football-data", provider_match_id="k", provider_competition_id="E0",
        competition_name="Premier League", match_date="2016-08-13",
        home_team_id="t", away_team_id="t", home_score=1, away_score=0,
        ftr="H", match_status="available",
    )])
    ing = FootballDataResultsIngestor(
        league_code="E0", season_code="1617", source_url=VALID_URL,
        cache_dir=Path("/tmp/x"), db_path=Path("/tmp/x.db"),
    )
    assert ing.validate(df).has_errors


def test_ingest_transaction_rollback_on_validation_error(db_path, cache_dir, csv_mock):
    bad = VALID_CSV + "25/12/2016,Arsenal,Burnley,-1,0,H\n"  # negative goal
    csv_mock["responses"][VALID_URL] = bad
    ing = FootballDataResultsIngestor(
        league_code="E0", season_code="1617", db_path=db_path,
        cache_dir=cache_dir, source_url=VALID_URL,
    )
    with pytest.raises(IngestionError):
        ing.ingest()
    with database_connection(db_path) as c:
        assert c.execute("SELECT COUNT(*) FROM matches").fetchone()[0] == 0
        assert c.execute("SELECT COUNT(*) FROM competitions "
                         "WHERE provider='football-data'").fetchone()[0] == 0
        assert c.execute("SELECT COUNT(*) FROM data_sources "
                         "WHERE provider='football-data'").fetchone()[0] == 0


def test_ingest_invalid_file_logs_error(csv_mock):
    csv_mock["responses"][VALID_URL] = "garbage,not,results\n1,2,3\n"
    ing = FootballDataResultsIngestor(
        league_code="E0", season_code="1617",
        db_path=Path("/tmp/unused.db"), cache_dir=Path("/tmp/unused_cache"),
        source_url=VALID_URL,
    )
    with pytest.raises(IngestionError):
        ing.ingest()


def test_ingest_empty_csv_rejected(csv_mock):
    csv_mock["responses"][VALID_URL] = "Date,HomeTeam,AwayTeam,FTHG,FTAG,FTR\n"
    ing = FootballDataResultsIngestor(
        league_code="E0", season_code="1617",
        db_path=Path("/tmp/unused.db"), cache_dir=Path("/tmp/unused_cache"),
        source_url=VALID_URL,
    )
    with pytest.raises(IngestionError):
        ing.ingest()


def test_ingest_uses_default_url_for_league_season():
    ing = FootballDataResultsIngestor(league_code="E0", season_code="1617")
    assert ing.source_url == DEFAULT_URL
    assert ing.season_name == "2016/2017"
    assert ing.league_code == "E0"


def test_invalid_league_code_raises():
    with pytest.raises(IngestionError):
        FootballDataResultsIngestor(league_code="ZZ", season_code="1617")


def test_invalid_season_code_raises():
    with pytest.raises(IngestionError):
        FootballDataResultsIngestor(league_code="E0", season_code="invalid")


def test_caching_reuses_file(db_path, cache_dir, http_mock):
    http_mock["responses"][VALID_URL] = VALID_CSV.encode("utf-8")
    ing = FootballDataResultsIngestor(
        league_code="E0", season_code="1617", db_path=db_path,
        cache_dir=cache_dir, source_url=VALID_URL,
    )
    ing.retrieve()
    assert ing.was_cached is False
    # Second retrieve hits the on-disk cache written by the first call.
    ing2 = FootballDataResultsIngestor(
        league_code="E0", season_code="1617", db_path=db_path,
        cache_dir=cache_dir, source_url=VALID_URL,
    )
    df2 = ing2.retrieve()
    assert ing2.was_cached is True
    assert len(df2) == 3


def test_cross_source_duplicate_skipped(db_path, cache_dir, csv_mock):
    csv_mock["responses"][VALID_URL] = VALID_CSV
    _seed_statsbomb_fixture(db_path, "2016-08-13", "Arsenal", "Burnley",
                            home_goals=1, away_goals=0, match_id="sb_ars_burn")
    ing = FootballDataResultsIngestor(
        league_code="E0", season_code="1617", db_path=db_path,
        cache_dir=cache_dir, source_url=VALID_URL,
    )
    result = ing.ingest()
    assert result.records_inserted == 2
    assert result.records_skipped == 1
    with database_connection(db_path) as c:
        rows = [dict(r) for r in c.execute(
            "SELECT provider, provider_match_id FROM matches "
            "WHERE match_date='2016-08-13' ORDER BY provider"
        )]
        fd_ars_burn = [r for r in rows if r["provider"] == "football-data"
                       and "arsenal" in r["provider_match_id"]]
        assert not fd_ars_burn
        sb = [r for r in rows if r["provider"] == "statsbomb"]
        assert len(sb) == 1


def test_cross_source_duplicate_idempotent_after_seed(db_path, cache_dir, csv_mock):
    csv_mock["responses"][VALID_URL] = VALID_CSV
    _seed_statsbomb_fixture(db_path, "2016-08-13", "Arsenal", "Burnley",
                            home_goals=1, away_goals=0, match_id="sb_ars_burn")
    ing = FootballDataResultsIngestor(
        league_code="E0", season_code="1617", db_path=db_path,
        cache_dir=cache_dir, source_url=VALID_URL,
    )
    r1 = ing.ingest()
    r2 = ing.ingest()
    assert r1.records_skipped == 1
    assert r1.records_inserted == 2
    # Re-running never inserts or double-counts: the seeded Arsenal-Burnley
    # fixture is skipped again, and the other two rows update in place.
    assert r2.records_skipped == 1
    assert r2.records_inserted == 0
    assert r2.records_updated == 2
    with database_connection(db_path) as c:
        assert c.execute("SELECT COUNT(*) FROM matches").fetchone()[0] == 3


def test_multiple_seasons_distinct(db_path, cache_dir, csv_mock):
    csv_mock["responses"]["http://mock/1516.csv"] = VALID_CSV
    csv_mock["responses"][VALID_URL] = ISO_CSV
    ing1 = FootballDataResultsIngestor(
        league_code="E0", season_code="1516", db_path=db_path,
        cache_dir=cache_dir, source_url="http://mock/1516.csv",
    )
    ing2 = FootballDataResultsIngestor(
        league_code="E0", season_code="1617", db_path=db_path,
        cache_dir=cache_dir, source_url=VALID_URL,
    )
    ing1.ingest()
    ing2.ingest()
    with database_connection(db_path) as c:
        seasons = sorted(r[0] for r in c.execute(
            "SELECT DISTINCT season_name FROM matches WHERE provider='football-data'"
                ))
        assert seasons == ["2015/2016", "2016/2017"]
        assert c.execute(
            "SELECT COUNT(*) FROM matches WHERE provider='football-data'"
        ).fetchone()[0] == 4


def test_teams_are_provider_scoped(db_path, cache_dir, csv_mock):
    csv_mock["responses"][VALID_URL] = VALID_CSV
    ing = FootballDataResultsIngestor(
        league_code="E0", season_code="1617", db_path=db_path,
        cache_dir=cache_dir, source_url=VALID_URL,
    )
    ing.ingest()
    with database_connection(db_path) as c:
        rows = [dict(r) for r in c.execute("SELECT provider, provider_team_id FROM teams")]
        fd = [r for r in rows if r["provider"] == "football-data"]
        assert len(fd) == 6
        assert all(r["provider"] for r in rows)


def test_list_result_leagues_includes_known_codes():
    leagues = list_result_leagues()
    codes = dict(leagues)
    assert {"E0", "E1", "D1", "I1", "SP1"} <= set(codes)
    assert leagues == sorted(leagues)


# ===========================================================================
# CLI tests
# ===========================================================================


def test_cli_import_subcommand_exists():
    parser = build_parser()
    args = parser.parse_args(
        ["football-data", "results", "import", "--league", "E0", "--season", "1617"]
    )
    assert args.league_code == "E0"
    assert args.season_code == "1617"


def test_cli_list_leagues_subcommand_exists():
    parser = build_parser()
    args = parser.parse_args(["football-data", "results", "list-leagues"])
    assert vars(args)


def test_cli_import_end_to_end(db_path, cache_dir, csv_mock):
    csv_mock["responses"][VALID_URL] = VALID_CSV
    rc = main(
        ["football-data", "results", "import",
         "--league", "E0", "--season", "1617",
         "--db", str(db_path), "--cache-dir", str(cache_dir),
         "--source-url", VALID_URL]
    )
    assert rc == 0
    with database_connection(db_path) as c:
        assert c.execute(
            "SELECT COUNT(*) FROM matches WHERE provider='football-data'"
        ).fetchone()[0] == 3


def test_cli_import_invalid_league_nonzero(db_path, cache_dir):
    rc = main(
        ["football-data", "results", "import",
         "--league", "ZZ", "--season", "1617",
         "--db", str(db_path), "--cache-dir", str(cache_dir),
         "--source-url", VALID_URL]
    )
    assert rc != 0







# ===========================================================================
# Phase 15: provenance repair (Repair A + B) and transactional rollback
# All tests hermetic: temp DBs, mocked network, no production access.
# ===========================================================================

import sqlite3  # noqa: E402

import football_intelligence.cli as cli_mod  # noqa: E402
from football_intelligence.cli import _safe_record_run  # noqa: E402
from football_intelligence.data.ingestion import (  # noqa: E402
    football_data_results as fd_results,
)
from football_intelligence.data.ingestion.odds import (  # noqa: E402
    record_ingestion_run as odds_record_ingestion_run,
)
from football_intelligence.exceptions import DatabaseError  # noqa: E402


def _run_result(provider, dataset_name, status, **kw):
    base = dict(
        provider=provider, dataset_name=dataset_name, status=status,
        started_at="2026-10-18T00:00:00+00:00",
        completed_at="2026-10-18T00:01:00+00:00",
    )
    base.update(kw)
    return IngestionResult(**base)


def test_record_ingestion_run_odds_aligned_with_schema(db_path):
    """Repair A: odds.record_ingestion_run persists a row to the 11-column
    ingestion_runs schema. Previously its 15-column INSERT raised
    'table ingestion_runs has no column named records_skipped'."""
    result = _run_result(
        PROVIDER, "odds", "success",
        records_read=7, records_inserted=3, records_updated=1,
        records_skipped=1, records_matched=0, records_unmatched=1,
        records_ambiguous=0, records_invalid=0,
    )
    odds_record_ingestion_run(db_path, result)  # must not raise
    with database_connection(db_path) as c:
        row = c.execute(
            "SELECT provider, dataset_name, status, records_read, "
            "records_inserted, records_updated, error_message "
            "FROM ingestion_runs WHERE provider=? AND dataset_name=?",
            (PROVIDER, "odds"),
        ).fetchone()
    assert row is not None
    assert row["provider"] == PROVIDER
    assert row["dataset_name"] == "odds"
    assert row["status"] == "success"
    assert row["records_read"] == 7
    assert row["records_inserted"] == 3
    assert row["records_updated"] == 1
    assert row["error_message"] is None


def test_safe_record_run_surfaces_provenance_failure(db_path, monkeypatch):
    """Repair B: a provenance-recording failure is surfaced, not swallowed."""
    def boom(db_path, result):
        raise sqlite3.OperationalError("no such table: ingestion_runs")

    monkeypatch.setattr(cli_mod, "record_ingestion_run", boom)
    result = _run_result(PROVIDER, "results", "failed", error_message="boom")
    with pytest.raises(DatabaseError) as ei:
        _safe_record_run(db_path, result)
    assert isinstance(ei.value.__cause__, sqlite3.OperationalError)
    assert "Failed to record ingestion run" in str(ei.value)


def test_cli_results_import_records_run_row(db_path, cache_dir, csv_mock):
    """End-to-end: a successful results import writes an ingestion_runs audit
    row under provider=football-data / dataset_name=results."""
    csv_mock["responses"][VALID_URL] = VALID_CSV
    rc = main(
        ["football-data", "results", "import",
         "--league", "E0", "--season", "1617",
         "--db", str(db_path), "--cache-dir", str(cache_dir),
         "--source-url", VALID_URL]
    )
    assert rc == 0
    with database_connection(db_path) as c:
        row = c.execute(
            "SELECT provider, dataset_name, status, records_read, "
            "records_inserted, records_updated, error_message "
            "FROM ingestion_runs WHERE provider=? AND dataset_name=? "
            "ORDER BY id DESC LIMIT 1",
            (PROVIDER, DATASET_NAME),
        ).fetchone()
        assert row is not None
        assert row["dataset_name"] == DATASET_NAME
        assert row["status"] == "success"
        assert row["records_read"] == 3
        assert row["records_inserted"] == 3
        # records_updated is a persisted column (idempotency re-runs update it);
        # records_skipped is intentionally NOT persisted (Repair A alignment).
        assert row["records_updated"] is not None
        assert row["error_message"] is None
        assert c.execute("SELECT COUNT(*) FROM matches WHERE provider=?",
                         (PROVIDER,)).fetchone()[0] == 3



def test_cli_results_import_success_but_audit_failure_is_distinct(
    db_path, cache_dir, csv_mock, monkeypatch, capsys
):
    """A successful import whose audit-row write fails must surface a distinct
    non-zero exit (data committed, audit record lost) -- never silently 0."""
    def boom(db_path, result):
        raise sqlite3.OperationalError("no such table: ingestion_runs")

    monkeypatch.setattr(cli_mod, "record_ingestion_run", boom)
    csv_mock["responses"][VALID_URL] = VALID_CSV
    rc = main(
        ["football-data", "results", "import",
         "--league", "E0", "--season", "1617",
         "--db", str(db_path), "--cache-dir", str(cache_dir),
         "--source-url", VALID_URL]
    )
    assert rc != 0
    captured = capsys.readouterr()
    assert "ERROR:" in captured.err
    assert "Failed to record ingestion run" in captured.err
    # Data was committed by load(); only the audit record was lost.
    with database_connection(db_path) as c:
        assert c.execute("SELECT COUNT(*) FROM matches WHERE provider=?",
                         (PROVIDER,)).fetchone()[0] == 3
    with database_connection(db_path) as c:
        assert c.execute("SELECT COUNT(*) FROM ingestion_runs").fetchone()[0] == 0


def test_cli_results_import_failure_preserves_original_error(
    db_path, cache_dir, csv_mock, monkeypatch, capsys
):
    """On ingestion failure: the ORIGINAL error is preserved AND a
    provenance-recording failure is surfaced distinctly (never silently)."""
    def boom(db_path, result):
        raise sqlite3.OperationalError("no such table: ingestion_runs")

    monkeypatch.setattr(cli_mod, "record_ingestion_run", boom)
    csv_mock["responses"][VALID_URL] = VALID_CSV  # league below is invalid
    rc = main(
        ["football-data", "results", "import",
         "--league", "ZZ", "--season", "1617",
         "--db", str(db_path), "--cache-dir", str(cache_dir),
         "--source-url", VALID_URL]
    )
    assert rc != 0
    captured = capsys.readouterr()
    # Provenance failure surfaced distinctly.
    assert "failed to record audit trail" in captured.err
    # Original ingestion error preserved verbatim.
    assert "ERROR: Unknown league code 'ZZ'" in captured.err


def test_ingest_rollback_on_mid_load_failure(
    db_path, cache_dir, csv_mock, monkeypatch
):
    """Transactional integrity: a mid-load failure rolls back ALL writes of the
    run (competition/team/match/data_sources) -- never a partial import."""
    orig = fd_results._upsert_match
    state = {"n": 0}

    def spy(conn, row, competition_id, home_team_id, away_team_id, now, provider=PROVIDER):
        action = orig(conn, row, competition_id, home_team_id, away_team_id, now, provider)
        state["n"] += 1
        if state["n"] >= 2:
            raise sqlite3.OperationalError("injected mid-load failure")
        return action

    monkeypatch.setattr(fd_results, "_upsert_match", spy)
    csv_mock["responses"][VALID_URL] = VALID_CSV
    ing = FootballDataResultsIngestor(
        league_code="E0", season_code="1617",
        db_path=db_path, cache_dir=cache_dir, source_url=VALID_URL,
    )
    with pytest.raises(IngestionError):
        ing.ingest()
    # Two rows attempted before the injected failure on the second.
    assert state["n"] == 2
    with database_connection(db_path) as c:
        assert c.execute("SELECT COUNT(*) FROM matches").fetchone()[0] == 0
        assert c.execute("SELECT COUNT(*) FROM teams").fetchone()[0] == 0
        assert c.execute("SELECT COUNT(*) FROM competitions").fetchone()[0] == 0
        assert c.execute("SELECT COUNT(*) FROM data_sources").fetchone()[0] == 0
        assert c.execute("PRAGMA foreign_key_check").fetchall() == []
        assert c.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
