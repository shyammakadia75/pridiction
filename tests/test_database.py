"""Tests for database initialization, constraints, and foreign-key enforcement.

Every test uses a throwaway database in pytest's ``tmp_path`` so no real or
production database is ever touched.
"""
from __future__ import annotations

import sqlite3

import pytest

from football_intelligence.database import (
    SCHEMA_VERSION,
    get_database_version,
    initialize_database,
)
from football_intelligence.database.connection import get_connection
from football_intelligence.database.schema import TABLE_NAMES

NOW = "2024-01-01T00:00:00+00:00"


def _connect(db_path):
    return get_connection(db_path)


def _seed_match_prereqs(conn):
    conn.execute(
        "INSERT INTO competitions (provider, provider_competition_id, "
        "competition_name, created_at, updated_at) VALUES (?, ?, ?, ?, ?)",
        ("s", "c1", "C1", NOW, NOW),
    )
    conn.execute(
        "INSERT INTO teams (provider, provider_team_id, team_name, "
        "created_at, updated_at) VALUES (?, ?, ?, ?, ?)",
        ("s", "t1", "Home", NOW, NOW),
    )
    conn.execute(
        "INSERT INTO teams (provider, provider_team_id, team_name, "
        "created_at, updated_at) VALUES (?, ?, ?, ?, ?)",
        ("s", "t2", "Away", NOW, NOW),
    )
    conn.commit()
    comp = conn.execute(
        "SELECT id FROM competitions WHERE provider_competition_id=?", ("c1",)
    ).fetchone()[0]
    home = conn.execute(
        "SELECT id FROM teams WHERE provider_team_id=?", ("t1",)
    ).fetchone()[0]
    away = conn.execute(
        "SELECT id FROM teams WHERE provider_team_id=?", ("t2",)
    ).fetchone()[0]
    return comp, home, away


MATCH_DEFAULTS = {
    "provider": "s",
    "provider_match_id": "m1",
    "season_name": "2023/24",
    "match_date": "2024-08-12T15:00:00+00:00",
    "competition_id": 1,
    "home_team_id": 1,
    "away_team_id": 2,
    "home_goals": 1,
    "away_goals": 0,
    "match_status": "scheduled",
    "created_at": NOW,
    "updated_at": NOW,
}


def _insert_match(conn, **overrides):
    row = dict(MATCH_DEFAULTS)
    row.update(overrides)
    cols = list(row)
    ph = ",".join("?" * len(cols))
    conn.execute(
        f"INSERT INTO matches ({','.join(cols)}) VALUES ({ph})",
        [row[c] for c in cols],
    )
    return conn.execute(
        "SELECT id FROM matches WHERE provider_match_id=?", (row["provider_match_id"],)
    ).fetchone()[0]


def test_database_directory_and_schema_created(db_path):
    initialize_database(db_path)
    assert db_path.exists()
    conn = _connect(db_path)
    tables = {
        r[0]
        for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table';"
        ).fetchall()
    }
    conn.close()
    for name in TABLE_NAMES:
        assert name in tables


def test_schema_version_recorded(db_path):
    initialize_database(db_path)
    assert get_database_version(db_path) == SCHEMA_VERSION


def test_repeated_init_does_not_destroy_data(db_path):
    initialize_database(db_path)
    conn = _connect(db_path)
    _seed_match_prereqs(conn)
    mid = _insert_match(conn, provider_match_id="keep-me")
    conn.commit()
    conn.close()

    # Re-running initialization must be safe and leave rows intact.
    initialize_database(db_path)
    initialize_database(db_path)

    conn = _connect(db_path)
    count = conn.execute("SELECT COUNT(*) AS n FROM matches").fetchone()["n"]
    row = conn.execute(
        "SELECT id FROM matches WHERE provider_match_id=?", ("keep-me",)
    ).fetchone()
    conn.close()
    assert count == 1
    assert row is not None
    assert row[0] == mid


def test_repeated_init_keeps_schema_version(db_path):
    initialize_database(db_path)
    version_first = get_database_version(db_path)
    initialize_database(db_path)
    version_second = get_database_version(db_path)
    assert version_first == version_second == SCHEMA_VERSION


def test_foreign_keys_enabled_by_default(db_path):
    initialize_database(db_path)
    conn = _connect(db_path)
    fk = conn.execute("PRAGMA foreign_keys;").fetchone()[0]
    conn.close()
    assert fk == 1


def test_foreign_key_violation_on_missing_competition(db_path):
    initialize_database(db_path)
    conn = _connect(db_path)
    _, home, away = _seed_match_prereqs(conn)
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute(
            "INSERT INTO matches (provider, provider_match_id, competition_id, "
            "home_team_id, away_team_id, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
            ("s", "m_bad_fk", 999, home, away, NOW, NOW),
        )
    conn.close()


def test_foreign_key_violation_on_missing_team(db_path):
    initialize_database(db_path)
    conn = _connect(db_path)
    comp, _, _ = _seed_match_prereqs(conn)
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute(
            "INSERT INTO matches (provider, provider_match_id, competition_id, "
            "home_team_id, away_team_id, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
            ("s", "m_bad_team", comp, 1, 999, NOW, NOW),
        )
    conn.close()


def test_duplicate_provider_team_id_rejected(db_path):
    initialize_database(db_path)
    conn = _connect(db_path)
    conn.execute(
        "INSERT INTO teams (provider, provider_team_id, team_name, created_at, updated_at) "
        "VALUES (?, ?, ?, ?, ?)", ("s", "t1", "A", NOW, NOW),
    )
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute(
            "INSERT INTO teams (provider, provider_team_id, team_name, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?)", ("s", "t1", "B", NOW, NOW),
        )
    conn.close()


def test_negative_goals_rejected(db_path):
    initialize_database(db_path)
    conn = _connect(db_path)
    _seed_match_prereqs(conn)
    with pytest.raises(sqlite3.IntegrityError):
        _insert_match(conn, provider_match_id="neg", home_goals=-1, away_goals=0)
    conn.close()


def test_same_home_and_away_team_rejected(db_path):
    initialize_database(db_path)
    conn = _connect(db_path)
    _, home, _ = _seed_match_prereqs(conn)
    with pytest.raises(sqlite3.IntegrityError):
        _insert_match(conn, provider_match_id="same", home_team_id=home, away_team_id=home)
    conn.close()


def test_prediction_probability_out_of_range_rejected(db_path):
    initialize_database(db_path)
    conn = _connect(db_path)
    _seed_match_prereqs(conn)
    mid = _insert_match(conn, provider_match_id="p1")
    conn.execute(
        "INSERT INTO model_versions (model_name, version, created_at, configuration_json) "
        "VALUES (?, ?, ?, ?)", ("demo", "1", NOW, "{}"),
    )
    conn.commit()
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute(
            "INSERT INTO predictions (match_id, model_version_id, prediction_timestamp, "
            "home_win_probability, draw_probability, away_win_probability, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (mid, 1, NOW, 1.5, 0.0, 0.0, NOW),
        )
    conn.close()


def test_odds_must_be_positive(db_path):
    initialize_database(db_path)
    conn = _connect(db_path)
    _seed_match_prereqs(conn)
    mid = _insert_match(conn, provider_match_id="o1")
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute(
            "INSERT INTO odds (match_id, decimal_odds, observed_at, created_at) "
            "VALUES (?, ?, ?, ?)", (mid, 0.0, NOW, NOW),
        )
    conn.close()


def test_valid_match_inserts_succeed(db_path):
    initialize_database(db_path)
    conn = _connect(db_path)
    _seed_match_prereqs(conn)
    mid = _insert_match(conn, provider_match_id="ok1", home_goals=2, away_goals=1)
    conn.commit()
    row = conn.execute("SELECT home_goals, away_goals FROM matches WHERE id=?", (mid,)).fetchone()
    conn.close()
    assert row[0] == 2
    assert row[1] == 1
