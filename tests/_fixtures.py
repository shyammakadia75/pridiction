"""Shared test helpers for Part 4 tests (data quality and features).

These helpers create synthetic data in disposable databases.  All test
fixtures are clearly labelled as test data and never touch the real
``data/football_intelligence.db``.
"""
from __future__ import annotations

from pathlib import Path

from football_intelligence.database import initialize_database
from football_intelligence.database.connection import get_connection

NOW = "2024-01-01T00:00:00+00:00"


def _connect(db_path):
    return get_connection(db_path)


def _seed_teams(conn, team_specs):
    """Insert teams.  team_specs is a list of (provider, provider_team_id, team_name).

    Returns dict mapping (provider, provider_team_id) -> team_id.
    """
    ids = {}
    for provider, pid, name in team_specs:
        conn.execute(
            "INSERT INTO teams (provider, provider_team_id, team_name, "
            "country, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?)",
            (provider, pid, name, "TestLand", NOW, NOW),
        )
        tid = conn.execute(
            "SELECT id FROM teams WHERE provider=? AND provider_team_id=?",
            (provider, pid),
        ).fetchone()[0]
        ids[(provider, pid)] = tid
    conn.commit()
    return ids


def _seed_competitions(conn, comp_specs):
    """Insert competitions.  comp_specs is a list of (provider, provider_comp_id, name).

    Returns dict mapping (provider, provider_comp_id) -> comp_id.
    """
    ids = {}
    for provider, pid, name in comp_specs:
        conn.execute(
            "INSERT INTO competitions (provider, provider_competition_id, "
            "competition_name, country, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (provider, pid, name, "TestLand", NOW, NOW),
        )
        cid = conn.execute(
            "SELECT id FROM competitions WHERE provider=? AND provider_competition_id=?",
            (provider, pid),
        ).fetchone()[0]
        ids[(provider, pid)] = cid
    conn.commit()
    return ids


def _insert_match(conn, **overrides):
    """Insert a match into the database. Override any field.

    Default values: provider='test', provider_match_id='m1', season='2024/25',
    match_date='2024-08-12T15:00:00+00:00', competition_id=1, home_team_id=1,
    away_team_id=2, home_goals=1, away_goals=0, match_status='available'.
    """
    row = {
        "provider": "test",
        "provider_match_id": "m1",
        "season_name": "2024/25",
        "match_date": "2024-08-12T15:00:00+00:00",
        "competition_id": 1,
        "home_team_id": 1,
        "away_team_id": 2,
        "home_goals": 1,
        "away_goals": 0,
        "match_status": "available",
        "created_at": NOW,
        "updated_at": NOW,
    }
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


def make_initialized_db(tmp_path: Path) -> Path:
    """Create a fresh initialized test database at tmp_path/test.db."""
    db = tmp_path / "test.db"
    initialize_database(db)
    return db


def make_seeded_db(tmp_path: Path) -> Path:
    """Create an initialized DB with four teams and one competition.

    Teams:
    - team_id 1: 'Home Team'  (provider='test', provider_team_id='HT')
    - team_id 2: 'Away Team'  (provider='test', provider_team_id='AT')
    - team_id 3: 'Team Three'  (provider='test', provider_team_id='T3')
    - team_id 4: 'Team Four'   (provider='test', provider_team_id='T4')
    Competition:
    - comp_id 1: 'Test League' (provider='test', provider_comp_id='CL')
    """
    db = make_initialized_db(tmp_path)
    with _connect(db) as conn:
        _seed_teams(conn, [
            ("test", "HT", "Home Team"),
            ("test", "AT", "Away Team"),
            ("test", "T3", "Team Three"),
            ("test", "T4", "Team Four"),
        ])
        _seed_competitions(conn, [
            ("test", "CL", "Test League"),
        ])
    return db
