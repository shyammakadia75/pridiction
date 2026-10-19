"""Shared synthetic-database builder for Phase-18 evaluation tests.

All data is SYNTHETIC and lives in a disposable database under pytest tmp_path;
the production database is never read or written here.
"""
from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

from _fixtures import _connect, _insert_match, _seed_competitions, _seed_teams, make_initialized_db

STATS = "statsbomb"


def build_two_season_db(tmp_path: Path) -> Path:
    """Disposable DB: two Premier-League seasons (40 matches) + 2 CL matches.

    Seasons 2015/16 and 2016/17 each have 20 matches with a deterministic goal
    pattern that guarantees all three outcome classes (home/draw/away).
    """
    db = make_initialized_db(tmp_path)
    with _connect(db) as conn:
        comps = _seed_competitions(conn, [
            (STATS, "2", "Premier League"),
            (STATS, "7", "Champions League"),
        ])
        teams = _seed_teams(conn, [(STATS, f"T{n}", f"Team{n}") for n in range(1, 7)])
    epl, cl = comps[(STATS, "2")], comps[(STATS, "7")]
    tids = [teams[(STATS, f"T{n}")] for n in ("1", "2", "3", "4", "5", "6")]
    base = datetime(2016, 8, 13, 15, 0, 0)
    rows = []
    for season, off in (("2015/16", 0), ("2016/17", 365)):
        for j in range(20):
            rows.append((f"epl-{season}-{j}", epl, season,
                         (base + timedelta(days=off + j)).isoformat(),
                         tids[j % 6], tids[(j + 1) % 6], j % 4, (j + 1) % 3))
    rows.append(("cl1", cl, "2018/19", (base + timedelta(days=700)).isoformat(),
                 tids[0], tids[1], 1, 2))
    rows.append(("cl2", cl, "2018/19", (base + timedelta(days=701)).isoformat(),
                 tids[2], tids[3], 0, 0))
    with _connect(db) as conn:
        for pmid, comp, season, date, h, a, hg, ag in rows:
            _insert_match(conn, provider=STATS, provider_match_id=pmid,
                          competition_id=comp, season_name=season, match_date=date,
                          home_team_id=h, away_team_id=a, home_goals=hg, away_goals=ag,
                          match_status="available")
    return db
