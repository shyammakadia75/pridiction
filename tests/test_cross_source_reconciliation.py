"""Symmetric cross-source reconciliation tests (Phase 17.2).

A fixture already supplied by one provider must be skipped -- never double-counted
and never have its score silently overwritten -- when the other provider imports,
in either import order. All DB writes target a disposable DB; production is never
touched. Synthetic fixtures; no real provider data.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from football_intelligence.data.ingestion.football_data_results import (
    FootballDataResultsIngestor,
)
from football_intelligence.data.ingestion.statsbomb import (
    StatsBombIngestor,
    _matches_to_dataframe,
    _normalize_match,
    record_ingestion_run,
)
from football_intelligence.database import initialize_database
from football_intelligence.database.connection import database_connection

FD, SB = "football-data", "statsbomb"
URL = "http://mock/E0/1617.csv"
CSV = ("Date,HomeTeam,AwayTeam,FTHG,FTAG,FTR\n13/08/2016,Arsenal,Burnley,1,0,H\n"
       "13/08/2016,Man United,Hull City,2,2,D\n20/08/2016,Chelsea,Tottenham,0,1,A\n")


@pytest.fixture
def db_path(tmp_path: Path) -> Path:
    db = tmp_path / "t.db"
    initialize_database(db)
    return db


@pytest.fixture
def cache_dir(tmp_path: Path) -> Path:
    return tmp_path / "c"


@pytest.fixture
def csv_mock(monkeypatch):
    resp: dict[str, str] = {}
    monkeypatch.setattr(
        "football_intelligence.data.ingestion.football_data_results._download_csv",
        lambda url, cache_path, timeout, refresh: (resp[url], False),
    )
    return resp


def _seed(db, prov, pid, date, home, away, hg, ag):
    with database_connection(db) as c:
        cid = c.execute(
            "INSERT INTO competitions (provider, provider_competition_id, "
            "competition_name, country, created_at, updated_at) "
            "VALUES (?,?,?,?, datetime('now'), datetime('now'))",
            (prov, "2" if prov == SB else "E0", "Premier League", "England"),
        ).lastrowid
        ht = c.execute(
            "INSERT INTO teams (provider, provider_team_id, team_name, country, "
            "created_at, updated_at) VALUES (?,?,?,?, datetime('now'), datetime('now'))",
            (prov, home.lower(), home, "England"),
        ).lastrowid
        at = c.execute(
            "INSERT INTO teams (provider, provider_team_id, team_name, country, "
            "created_at, updated_at) VALUES (?,?,?,?, datetime('now'), datetime('now'))",
            (prov, away.lower(), away, "England"),
        ).lastrowid
        c.execute(
            "INSERT INTO matches (provider, provider_match_id, competition_id, "
            "season_name, match_date, home_team_id, away_team_id, home_goals, "
            "away_goals, match_status, created_at, updated_at) "
            "VALUES (?,?,?,?,?,?,?,?,?,'available', datetime('now'), datetime('now'))",
            (prov, pid, cid, "2016/2017", date, ht, at, hg, ag),
        )


def _sb_df(home="Arsenal", away="Burnley", date="2016-08-13", mid="9001",
           hg=2, ag=1, hid="5001", aid="5002"):
    raw = {
        "match_id": mid, "match_date": date, "kick_off": "16:00:00.000",
        "competition": {"competition_id": 7, "season_id": 3,
                        "competition_name": "Premier League", "country_name": "England",
                        "season_name": "2016/2017"},
        "season": {"season_id": 3, "season_name": "2016/2017"},
        "home_team": {"home_team_id": hid, "home_team_name": home,
                      "country": {"name": "England"}},
        "away_team": {"away_team_id": aid, "away_team_name": away,
                      "country": {"name": "England"}},
        "home_score": hg, "away_score": ag, "match_status": "available",
        "competition_stage": {"name": "League"},
        "metadata": {"data_version": "1.1.0"}, "match_week": 1,
        "last_updated": "2024-01-15T10:30:00.000",
    }
    return _matches_to_dataframe([raw])


def _cnt(db, prov=None):
    with database_connection(db) as c:
        if prov:
            return c.execute("SELECT COUNT(*) FROM matches WHERE provider=?", (prov,)).fetchone()[0]
        return c.execute("SELECT COUNT(*) FROM matches").fetchone()[0]


def _sb(db, cd):
    return StatsBombIngestor(7, 3, db_path=db, cache_dir=cd, source_url="http://m")


def _fd(db, cd):
    return FootballDataResultsIngestor(league_code="E0", season_code="1617",
                                       db_path=db, cache_dir=cd, source_url=URL)


def test_sb_after_fd_skips_and_preserves_score(db_path, cache_dir):
    _seed(db_path, FD, "fd1", "2016-08-13", "Arsenal", "Burnley", 1, 0)
    r = _sb(db_path, cache_dir).load(_sb_df(hg=2, ag=1))
    assert r.records_skipped == 1 and r.records_inserted == 0
    with database_connection(db_path) as c:
        m = c.execute("SELECT provider, home_goals, away_goals FROM matches "
                      "WHERE match_date='2016-08-13'").fetchone()
        assert m["provider"] == FD and (m["home_goals"], m["away_goals"]) == (1, 0)


def test_fd_after_sb_skips(db_path, cache_dir, csv_mock):
    _seed(db_path, SB, "sb1", "2016-08-13", "Arsenal", "Burnley", 1, 0)
    csv_mock[URL] = CSV
    r = _fd(db_path, cache_dir).ingest()
    assert r.records_skipped == 1 and r.records_inserted == 2
    assert _cnt(db_path) == 3


def test_exact_duplicate_same_provider_updates(db_path, cache_dir):
    a = _sb(db_path, cache_dir).load(_sb_df(hg=2, ag=1))
    b = _sb(db_path, cache_dir).load(_sb_df(hg=3, ag=0))
    assert a.records_inserted == 1 and b.records_updated == 1 and b.records_skipped == 0
    assert _cnt(db_path, SB) == 1


def test_ambiguous_same_date_same_names(db_path, cache_dir):
    _seed(db_path, FD, "fd1", "2016-08-13", "Arsenal", "Burnley", 1, 0)
    _seed(db_path, SB, "sb1", "2016-08-13", "Arsenal", "Burnley", 1, 0)
    r = _sb(db_path, cache_dir).load(_sb_df(mid="sb_amb"))
    assert r.records_ambiguous == 1 and r.records_inserted == 0
    assert _cnt(db_path) == 2  # FD fixture + SB fixture, no extra insert


def test_different_date_no_false_match(db_path, cache_dir):
    _seed(db_path, FD, "fd1", "2016-08-13", "Arsenal", "Burnley", 1, 0)
    r = _sb(db_path, cache_dir).load(_sb_df(date="2016-08-20", hg=2, ag=1))
    assert r.records_inserted == 1 and r.records_skipped == 0
    assert _cnt(db_path) == 2


def test_home_away_reversal_not_merged(db_path, cache_dir):
    _seed(db_path, FD, "fd1", "2016-08-13", "Arsenal", "Burnley", 1, 0)
    r = _sb(db_path, cache_dir).load(_sb_df(home="Burnley", away="Arsenal"))
    assert r.records_inserted == 1 and r.records_skipped == 0
    assert _cnt(db_path) == 2


def test_repeated_import_idempotent(db_path, cache_dir):
    _seed(db_path, FD, "fd1", "2016-08-13", "Arsenal", "Burnley", 1, 0)
    a = _sb(db_path, cache_dir).load(_sb_df())
    b = _sb(db_path, cache_dir).load(_sb_df())
    assert a.records_skipped == 1 and b.records_skipped == 1
    assert _cnt(db_path) == 1


def test_rollback_on_failure(db_path, cache_dir):
    ok = _sb_df(home="Arsenal", away="Burnley", mid="r1")
    bad = _sb_df(home="X", away="Y", mid="r2", hid="9", aid="9")
    with pytest.raises(Exception):
        _sb(db_path, cache_dir).load(pd.concat([ok, bad], ignore_index=True))
    assert _cnt(db_path) == 0


def test_audit_row_recorded(db_path, cache_dir):
    r = _sb(db_path, cache_dir).load(_sb_df())
    record_ingestion_run(db_path, r)
    with database_connection(db_path) as c:
        assert c.execute("SELECT COUNT(*) FROM ingestion_runs").fetchone()[0] == 1


def test_sb_after_fd_different_scores_keeps_fd(db_path, cache_dir):
    _seed(db_path, FD, "fd1", "2016-08-13", "Arsenal", "Burnley", 4, 2)
    _sb(db_path, cache_dir).load(_sb_df(hg=0, ag=0))
    with database_connection(db_path) as c:
        m = c.execute("SELECT home_goals, away_goals FROM matches WHERE provider=?", (FD,)).fetchone()
        assert (m["home_goals"], m["away_goals"]) == (4, 2)

