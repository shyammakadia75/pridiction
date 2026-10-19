"""CLI tests for `football-intelligence ratings elo` (Phase 19)."""
import json

from _phase18_db import build_two_season_db
from football_intelligence.cli import build_parser, main


def _run(argv):
    return main(list(argv))


def test_ratings_elo_parser():
    parser = build_parser()
    a = parser.parse_args(["ratings", "elo", "--db-path", "x.db"])
    assert a.func.__name__ == "cmd_ratings_elo"
    assert a.competition_id == "2"
    assert a.provider == "statsbomb"
    assert a.K == 20.0
    assert a.home_advantage == 50.0
    assert a.initial_rating == 1500.0
    assert a.season is None


def test_ratings_elo_smoke_all_seasons(tmp_path):
    db = build_two_season_db(tmp_path)
    out = tmp_path / "reports"
    rc = _run(["ratings", "elo", "--db-path", str(db),
               "--competition-id", "2", "--output-dir", str(out)])
    assert rc == 0
    j = sorted(out.glob("phase19_elo_2_all_*.json"))
    assert len(j) == 1
    data = json.loads(j[0].read_text(encoding="utf-8"))
    assert data["dataset"]["n_matches"] == 40
    assert len(data["ratings"]) == 6
    assert data["production_db_unchanged"] is True
    txt = out / (j[0].name[:-5] + ".txt")
    assert txt.exists()
    body = txt.read_text(encoding="utf-8")
    assert "Phase-19 Elo Ratings Report" in body
    assert "top 10 by Elo" not in body  # that's the CLI stdout line, not the TXT


def test_ratings_elo_single_season(tmp_path):
    db = build_two_season_db(tmp_path)
    out = tmp_path / "reports"
    rc = _run(["ratings", "elo", "--db-path", str(db),
               "--competition-id", "2", "--season", "2015/16",
               "--output-dir", str(out)])
    assert rc == 0
    j = sorted(out.glob("phase19_elo_2_2015-16_*.json"))
    assert len(j) == 1


def test_ratings_elo_missing_db_returns_nonzero(tmp_path):
    out = tmp_path / "reports"
    missing = tmp_path / "nope.db"
    rc = _run(["ratings", "elo", "--db-path", str(missing),
               "--competition-id", "2", "--output-dir", str(out)])
    assert rc == 1
    assert not list(out.glob("*.json"))


def test_ratings_elo_invalid_competition_returns_nonzero(tmp_path):
    db = build_two_season_db(tmp_path)
    out = tmp_path / "reports"
    rc = _run(["ratings", "elo", "--db-path", str(db),
               "--competition-id", "9999", "--output-dir", str(out)])
    assert rc == 1
