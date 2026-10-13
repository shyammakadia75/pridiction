"""Tests for the Football-Data.co.uk historical odds ingestion adapter.

All network access is mocked; no real HTTP calls are made.  The CSV fixtures
below are **synthetic** -- forged locally to exercise specific code paths; they
are NOT real Football-Data production data and must never be used as such.
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

from football_intelligence.data.ingestion.odds import (
    DEFAULT_TIMEOUT,
    ODDS_BASE_URL,
    PROVIDER,
    OddsIngestor,
    _download_csv,
    _extract_odds_records,
    _normalize_match_date,
    _normalize_team_name,
    _parse_csv,
    _upsert_odds,
    list_leagues,
)
from football_intelligence.database import initialize_database
from football_intelligence.database.connection import database_connection
from football_intelligence.exceptions import DatabaseError, IngestionError


SAMPLE_CSV = (
    "Date,HomeTeam,AwayTeam,B365H,B365D,B365A,B365>2.5,B365<2.5,"
    "B365CH,B365CD,B365CA,VCCH,VCA\n"
    "01/08/2023,Arsenal,Burnley,1.50,4.00,7.00,1.80,2.00,1.55,4.20,6.50,1.60,3.80,7.50\n"
    "05/08/2023,Chelsea,Arsenal,2.20,3.40,3.20,1.90,1.95,2.25,3.30,3.10,2.30,3.50,3.00\n"
    "12/08/2023,Liverpool,Man United,1.80,3.60,4.50,1.70,1.10,1.85,3.70,4.30,1.85,3.60,4.20\n"
)


def _make_csv_bytes(text: str = SAMPLE_CSV) -> bytes:
    return b"\xef\xbb\xbf" + text.encode("utf-8")


@pytest.fixture()
def initialized_db(tmp_path: Path) -> Path:
    """A freshly initialized test database."""
    db = tmp_path / "test.db"
    initialize_database(db)
    return db


@pytest.fixture()
def http_mock(monkeypatch):
    """Patch _http_get so tests never hit the network."""
    responses: dict[str, bytes] = {}
    calls: list[str] = []
    def fake_get(url: str, timeout: int = DEFAULT_TIMEOUT) -> bytes:
        calls.append(url)
        if url not in responses:
            raise IngestionError(f"HTTP 404: Not Found for {url}")
        return responses[url]
    monkeypatch.setattr(
        "football_intelligence.data.ingestion.odds._http_get", fake_get
    )
    return {"responses": responses, "calls": calls}


@pytest.fixture()
def seeded_db(initialized_db: Path) -> Path:
    """An initialized DB with a match row pre-seeded for odds reconciliation."""
    with database_connection(initialized_db) as conn:
        conn.execute(
            "INSERT INTO teams (provider, provider_team_id, team_name, country, "
            "created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?)",
            ("football-data", "1", "Arsenal", "England", "2024-01-01", "2024-01-01"),
        )
        conn.execute(
            "INSERT INTO teams (provider, provider_team_id, team_name, country, "
            "created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?)",
            ("football-data", "2", "Burnley", "England", "2024-01-01", "2024-01-01"),
        )
        conn.execute(
            "INSERT INTO competitions (provider, provider_competition_id, "
            "competition_name, country, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            ("football-data", "E0", "Premier League", "England", "2024-01-01", "2024-01-01"),
        )
        conn.execute(
            "INSERT INTO matches (provider, provider_match_id, competition_id, "
            "season_name, match_date, home_team_id, away_team_id, home_goals, "
            "away_goals, match_status, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            ("football-data", "M001", 1, "2023/24", "2023-08-01", 1, 2, 1, 0, "available",
             "2024-01-01", "2024-01-01"),
        )
    return initialized_db


# ---------------------------------------------------------------------------
# Constants / module-level tests
# ---------------------------------------------------------------------------

class TestProviderAndConstants:
    def test_provider_constant(self):
        assert PROVIDER == "football-data"

    def test_odds_base_url(self):
        assert ODDS_BASE_URL == "https://www.football-data.co.uk/mmz4281"

    def test_list_leagues_returns_sorted_tuples(self):
        leagues = list_leagues()
        assert isinstance(leagues, list)
        assert all(isinstance(item, tuple) and len(item) == 2 for item in leagues)
        codes = [code for code, _ in leagues]
        assert codes == sorted(codes)

    def test_list_leagues_contains_key_codes(self):
        leagues = dict(list_leagues())
        assert "E0" in leagues
        assert "E1" in leagues
        assert "SP1" in leagues
        assert "D1" in leagues
        assert "I1" in leagues

    def test_league_codes_count(self):
        assert len(list_leagues()) > 10


# ---------------------------------------------------------------------------
# Constructor / league + season validation
# ---------------------------------------------------------------------------

class TestConstructorValidation:
    def test_valid_league_code_uppercase(self):
        ing = OddsIngestor("E0", "2324")
        assert ing.league_code == "E0"

    def test_valid_league_code_lowercase_normalized(self):
        ing = OddsIngestor("e0", "2324")
        assert ing.league_code == "E0"

    def test_valid_league_code_sp1(self):
        ing = OddsIngestor("SP1", "2324")
        assert ing.league_code == "SP1"

    def test_invalid_league_code_raises(self):
        with pytest.raises(IngestionError, match="Unknown league code"):
            OddsIngestor("XX", "2324")

    def test_invalid_league_code_lowercase_raises(self):
        with pytest.raises(IngestionError, match="Unknown league code"):
            OddsIngestor("zz", "2324")

    def test_invalid_season_code_short_raises(self):
        with pytest.raises(IngestionError, match="Invalid season code"):
            OddsIngestor("E0", "23")

    def test_invalid_season_code_long_raises(self):
        with pytest.raises(IngestionError, match="Invalid season code"):
            OddsIngestor("E0", "23244")

    def test_invalid_season_code_non_digit_raises(self):
        with pytest.raises(IngestionError, match="Invalid season code"):
            OddsIngestor("E0", "abcd")

    def test_valid_season_code_accepted(self):
        ing = OddsIngestor("E0", "2324")
        assert ing.season_code == "2324"

    def test_season_code_stripped(self):
        ing = OddsIngestor("E0", "  2324  ")
        assert ing.season_code == "2324"

    def test_source_url_built_from_league_and_season(self):
        ing = OddsIngestor("E0", "2324")
        assert ing.source_url == f"{ODDS_BASE_URL}/2324/E0.csv"

    def test_custom_source_url(self):
        ing = OddsIngestor("E0", "2324", source_url="https://example.com/custom.csv")
        assert ing.source_url == "https://example.com/custom.csv"

    def test_default_properties(self):
        ing = OddsIngestor("E0", "2324")
        assert ing.provider == PROVIDER
        assert ing.dataset_name == "odds"
        assert ing.source_reference is not None
        assert ing.was_cached is False
        assert ing.retrieval_timestamp is None


class TestNormalizeTeamName:
    def test_basic_lowercase(self):
        assert _normalize_team_name("ARS") == "ars"

    def test_strips_whitespace(self):
        assert _normalize_team_name("  Arsenal  ") == "arsenal"

    def test_removes_fc_suffix(self):
        assert _normalize_team_name("Arsenal FC") == "arsenal"

    def test_removes_f_c_dot_suffix(self):
        assert _normalize_team_name("Arsenal F.C.") == "arsenal"

    def test_alias_man_united(self):
        assert _normalize_team_name("Man United") == "manchester united"

    def test_alias_man_city(self):
        assert _normalize_team_name("Man City") == "manchester city"

    def test_alias_spurs(self):
        assert _normalize_team_name("Spurs") == "tottenham hotspur"

    def test_alias_wolves(self):
        assert _normalize_team_name("Wolves") == "wolverhampton wanderers"

    def test_alias_newcastle(self):
        assert _normalize_team_name("Newcastle") == "newcastle united"

    def test_empty_string(self):
        assert _normalize_team_name("") == ""

    def test_removes_apostrophe(self):
        assert _normalize_team_name("Nott'm Forest") == "nottm forest"


class TestNormalizeMatchDate:
    def test_dd_mm_yyyy(self):
        assert _normalize_match_date("01/08/2023") == "2023-08-01"

    def test_dd_mm_yyyy_no_padding(self):
        assert _normalize_match_date("1/8/2023") == "2023-08-01"

    def test_iso_date(self):
        assert _normalize_match_date("2023-08-01") == "2023-08-01"

    def test_empty_string_returns_none(self):
        assert _normalize_match_date("") is None

    def test_unparseable_returns_none(self):
        assert _normalize_match_date("not a date") is None


class TestExtractOddsRecords:
    def _make_row(self, **kwargs):
        base = {"Date": "01/08/2023", "HomeTeam": "Arsenal", "AwayTeam": "Burnley"}
        base.update(kwargs)
        return base

    def test_extracts_basic_win_draw_away(self):
        row = self._make_row(B365H="1.50", B365D="4.00", B365A="7.00")
        records = _extract_odds_records(row)
        assert len(records) == 3
        for rec in records:
            assert rec["match_date"] == "2023-08-01"
            assert rec["home_team"] == "Arsenal"
            assert rec["away_team"] == "Burnley"
        selections = {rec["selection"] for rec in records}
        assert selections == {"home", "draw", "away"}

    def test_extracts_closing_odds(self):
        row = self._make_row(B365H="1.50", B365CH="1.55")
        records = _extract_odds_records(row)
        assert len(records) == 2
        odds_types = {rec["odds_type"] for rec in records}
        assert odds_types == {"opening", "closing"}

    def test_skips_empty_odds_values(self):
        row = self._make_row(B365H="", B365D="4.00", B365A="")
        records = _extract_odds_records(row)
        assert len(records) == 1

    def test_skips_none_odds_values(self):
        row = self._make_row(B365H=None, B365D="4.00", B365A=None)
        records = _extract_odds_records(row)
        assert len(records) == 1

    def test_skips_non_numeric_odds(self):
        row = self._make_row(B365H="N/A", B365D="4.00")
        records = _extract_odds_records(row)
        assert len(records) == 1

    def test_skips_zero_odds(self):
        row = self._make_row(B365H="0.00", B365D="4.00")
        records = _extract_odds_records(row)
        assert len(records) == 1

    def test_no_odds_columns(self):
        row = self._make_row()
        records = _extract_odds_records(row)
        assert records == []

    def test_records_have_required_fields(self):
        row = self._make_row(B365H="1.50")
        records = _extract_odds_records(row)
        assert len(records) == 1
        rec = records[0]
        for key in ("match_date", "home_team", "away_team", "bookmaker",
                     "market", "selection", "odds_type", "decimal_odds"):
            assert key in rec
        assert rec["bookmaker"] == "bet365"


class TestParseCsv:
    def test_parses_basic_csv(self):
        df = _parse_csv(SAMPLE_CSV)
        assert len(df) == 3
        assert "Date" in df.columns

    def test_empty_csv_returns_empty_df(self):
        df = _parse_csv("Date,HomeTeam,AwayTeam\n")
        assert len(df) == 0

    def test_skips_rows_with_empty_date(self):
        csv_text = "Date,HomeTeam,B365H\n01/08/2023,Arsenal,1.50\n,,,\n"
        df = _parse_csv(csv_text)
        assert len(df) == 1


class TestDownloadCsv:
    def test_returns_cached_file(self, tmp_path, monkeypatch):
        cache_file = tmp_path / "cached.csv"
        cache_file.write_text("Date,HomeTeam,B365H\n01/08/2023,Arsenal,1.50\n")
        calls = []

        def fake_get(url, timeout=DEFAULT_TIMEOUT):
            calls.append(url)
            raise AssertionError("Should not hit network when cached")

        monkeypatch.setattr(
            "football_intelligence.data.ingestion.odds._http_get", fake_get
        )
        text, was_cached = _download_csv(
            "http://example.com/test.csv", cache_file, DEFAULT_TIMEOUT, refresh=False
        )
        assert was_cached is True
        assert "Arsenal" in text
        assert len(calls) == 0

    def test_fetches_and_caches_when_no_cache(self, tmp_path, monkeypatch):
        cache_file = tmp_path / "cached.csv"

        def fake_get(url, timeout=DEFAULT_TIMEOUT):
            return b"Date,HomeTeam\n01/08/2023,Arsenal\n"

        monkeypatch.setattr(
            "football_intelligence.data.ingestion.odds._http_get", fake_get
        )
        text, was_cached = _download_csv(
            "http://example.com/test.csv", cache_file, DEFAULT_TIMEOUT, refresh=False
        )
        assert was_cached is False
        assert cache_file.exists()
        assert "Arsenal" in text

    def test_refresh_forces_network(self, tmp_path, monkeypatch):
        cache_file = tmp_path / "cached.csv"
        cache_file.write_text("OLD DATA")
        calls = []

        def fake_get(url, timeout=DEFAULT_TIMEOUT):
            calls.append(url)
            return b"NEW DATA"

        monkeypatch.setattr(
            "football_intelligence.data.ingestion.odds._http_get", fake_get
        )
        text, was_cached = _download_csv(
            "http://example.com/test.csv", cache_file, DEFAULT_TIMEOUT, refresh=True
        )
        assert was_cached is False
        assert text == "NEW DATA"
        assert len(calls) == 1


class TestRetrieve:
    def test_retrieve_with_mocked_http(self, tmp_path, http_mock):
        url = "https://www.football-data.co.uk/mmz4281/2324/E0.csv"
        http_mock["responses"][url] = _make_csv_bytes()
        ing = OddsIngestor("E0", "2324", cache_dir=tmp_path / "cache")
        df = ing.retrieve()
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0
        for col in ("match_date", "home_team", "away_team", "bookmaker",
                     "market", "selection", "odds_type", "decimal_odds"):
            assert col in df.columns
        assert ing.was_cached is False
        assert ing.retrieval_timestamp is not None

    def test_retrieve_uses_cache_on_second_call(self, tmp_path, http_mock):
        url = "https://www.football-data.co.uk/mmz4281/2324/E0.csv"
        http_mock["responses"][url] = _make_csv_bytes()
        ing = OddsIngestor("E0", "2324", cache_dir=tmp_path / "cache")
        ing.retrieve()
        assert ing.was_cached is False
        ing.retrieve()
        assert ing.was_cached is True

    def test_retrieve_http_error_raises(self, tmp_path, http_mock):
        ing = OddsIngestor(
            "E0", "2324", cache_dir=tmp_path / "cache",
            source_url="http://nonexistent.example.com/missing.csv",
        )
        with pytest.raises(IngestionError, match="404"):
            ing.retrieve()

    def test_retrieve_returns_empty_df_for_empty_csv(self, tmp_path, http_mock):
        url = "https://www.football-data.co.uk/mmz4281/2324/E0.csv"
        http_mock["responses"][url] = _make_csv_bytes("Date,HomeTeam,AwayTeam,B365H\n")
        ing = OddsIngestor("E0", "2324", cache_dir=tmp_path / "cache")
        df = ing.retrieve()
        assert len(df) == 0


class TestLoad:
    def test_load_inserts_odds_for_matched_match(self, seeded_db, tmp_path):
        ing = OddsIngestor(
            "E0", "2324", db_path=seeded_db, cache_dir=tmp_path / "cache"
        )
        with patch(
            "football_intelligence.data.ingestion.odds._download_csv",
            return_value=(SAMPLE_CSV, False),
        ):
            data = ing.retrieve()
        result = ing.load(data)
        assert result.status == "success"
        assert result.records_matched > 0
        assert result.records_read > 0

    def test_load_unmatched_match_increments_unmatched(self, initialized_db, tmp_path):
        ing = OddsIngestor(
            "E0", "2324", db_path=initialized_db, cache_dir=tmp_path / "cache"
        )
        with patch(
            "football_intelligence.data.ingestion.odds._download_csv",
            return_value=(SAMPLE_CSV, False),
        ):
            data = ing.retrieve()
        result = ing.load(data)
        assert result.status == "success"
        assert result.records_unmatched > 0
        assert result.records_matched == 0

    def test_load_raises_on_uninitialized_db(self, tmp_path):
        empty_db = tmp_path / "uninit.db"
        ing = OddsIngestor(
            "E0", "2324", db_path=empty_db, cache_dir=tmp_path / "cache"
        )
        with patch(
            "football_intelligence.data.ingestion.odds._download_csv",
            return_value=(SAMPLE_CSV, False),
        ):
            data = ing.retrieve()
        with pytest.raises(DatabaseError, match="not initialized"):
            ing.load(data)

    def test_load_is_idempotent(self, seeded_db, tmp_path):
        ing = OddsIngestor(
            "E0", "2324", db_path=seeded_db, cache_dir=tmp_path / "cache"
        )
        with patch(
            "football_intelligence.data.ingestion.odds._download_csv",
            return_value=(SAMPLE_CSV, False),
        ):
            data = ing.retrieve()
        ing.load(data)
        result2 = ing.load(data)
        assert result2.records_inserted == 0
        assert result2.records_updated > 0

    def test_odds_table_populated_after_load(self, seeded_db, tmp_path):
        ing = OddsIngestor(
            "E0", "2324", db_path=seeded_db, cache_dir=tmp_path / "cache"
        )
        with patch(
            "football_intelligence.data.ingestion.odds._download_csv",
            return_value=(SAMPLE_CSV, False),
        ):
            data = ing.retrieve()
        ing.load(data)
        with database_connection(seeded_db) as conn:
            count = conn.execute("SELECT COUNT(*) FROM odds").fetchone()
            assert count[0] > 0
            cols = [
                row["name"]
                for row in conn.execute("PRAGMA table_info(odds)").fetchall()
            ]
            for col in (
                "match_id", "provider", "bookmaker", "market",
                "selection", "odds_type", "decimal_odds",
                "observed_at", "source_reference",
            ):
                assert col in cols


class TestUpsertOdds:
    def test_insert_new_record(self, initialized_db):
        with database_connection(initialized_db) as conn:
            conn.execute(
                "INSERT INTO teams (provider, provider_team_id, team_name, "
                "country, created_at, updated_at) "
                "VALUES ('ft-data', '1', 'Alpha', 'X', '2024', '2024')"
            )
            conn.execute(
                "INSERT INTO teams (provider, provider_team_id, team_name, "
                "country, created_at, updated_at) "
                "VALUES ('ft-data', '2', 'Beta', 'X', '2024', '2024')"
            )
            conn.execute(
                "INSERT INTO competitions (provider, provider_competition_id, "
                "competition_name, country, created_at, updated_at) "
                "VALUES ('ft-data', 'C1', 'Test', 'X', '2024', '2024')"
            )
            conn.execute(
                "INSERT INTO matches (provider, provider_match_id, competition_id, "
                "match_date, home_team_id, away_team_id, created_at, updated_at) "
                "VALUES ('ft-data', 'M1', 1, '2023-08-01', 1, 2, '2024', '2024')"
            )
            match_id = conn.execute(
                "SELECT id FROM matches WHERE provider_match_id='M1'"
            ).fetchone()["id"]
            action = _upsert_odds(
                conn,
                match_id=match_id,
                provider="football-data",
                bookmaker="bet365",
                market="match_odds",
                selection="home",
                odds_type="opening",
                decimal_odds=1.50,
                observed_at="2023-08-01T10:00:00",
                source_reference="http://example.com/odds.csv",
                now="2024-01-01",
            )
            assert action == "inserted"

    def test_update_existing_record(self, initialized_db):
        with database_connection(initialized_db) as conn:
            conn.execute(
                "INSERT INTO teams (provider, provider_team_id, team_name, "
                "country, created_at, updated_at) "
                "VALUES ('ft-data', '1', 'Alpha', 'X', '2024', '2024')"
            )
            conn.execute(
                "INSERT INTO teams (provider, provider_team_id, team_name, "
                "country, created_at, updated_at) "
                "VALUES ('ft-data', '2', 'Beta', 'X', '2024', '2024')"
            )
            conn.execute(
                "INSERT INTO competitions (provider, provider_competition_id, "
                "competition_name, country, created_at, updated_at) "
                "VALUES ('ft-data', 'C1', 'Test', 'X', '2024', '2024')"
            )
            conn.execute(
                "INSERT INTO matches (provider, provider_match_id, competition_id, "
                "match_date, home_team_id, away_team_id, created_at, updated_at) "
                "VALUES ('ft-data', 'M1', 1, '2023-08-01', 1, 2, '2024', '2024')"
            )
            match_id = conn.execute(
                "SELECT id FROM matches WHERE provider_match_id='M1'"
            ).fetchone()["id"]
            _upsert_odds(
                conn, match_id=match_id, provider="football-data",
                bookmaker="bet365", market="match_odds", selection="home",
                odds_type="opening", decimal_odds=1.50,
                observed_at="2023-08-01T10:00:00",
                source_reference="http://example.com/odds.csv", now="2024-01-01",
            )
            action = _upsert_odds(
                conn, match_id=match_id, provider="football-data",
                bookmaker="bet365", market="match_odds", selection="home",
                odds_type="opening", decimal_odds=1.60,
                observed_at="2023-08-02T10:00:00",
                source_reference="http://example.com/odds.csv", now="2024-01-02",
            )
            assert action == "updated"
            row = conn.execute(
                "SELECT decimal_odds FROM odds WHERE match_id=?",
                (match_id,)
            ).fetchone()
            assert row["decimal_odds"] == 1.60


class TestIngestPipeline:
    def test_ingest_full_pipeline(self, initialized_db, tmp_path, http_mock):
        url = "https://www.football-data.co.uk/mmz4281/2324/E0.csv"
        http_mock["responses"][url] = _make_csv_bytes()
        ing = OddsIngestor(
            "E0", "2324", db_path=initialized_db, cache_dir=tmp_path / "cache"
        )
        result = ing.ingest()
        assert result.status == "success"
        assert result.records_read > 0
        assert result.provider == PROVIDER
        assert result.dataset_name == "odds"

    def test_ingest_raises_on_http_failure(self, tmp_path, http_mock):
        ing = OddsIngestor(
            "E0", "2324", db_path=tmp_path / "test.db", cache_dir=tmp_path / "cache",
            source_url="http://nonexistent.example.com/missing.csv",
        )
        with pytest.raises(IngestionError, match="HTTP 404"):
            ing.ingest()
