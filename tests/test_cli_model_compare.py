"""CLI tests for `football-intelligence model compare` (Phase 21)."""
import json
from pathlib import Path

import pytest

from _phase18_db import build_two_season_db
from football_intelligence.cli import build_parser, main


def _run(argv):
    return main(list(argv))


def _artifact(out, name_glob):
    hits = list(sorted(out.rglob(name_glob)))
    assert len(hits) == 1, f"expected one artifact matching {name_glob}, got {[p.name for p in hits]}"
    return hits[0]


def test_parser_registers_model_compare():
    parser = build_parser()
    args = parser.parse_args(["model", "compare", "--db-path", "x.db"])
    assert args.func.__name__ == "cmd_model_compare"
    assert args.competition_id == "2"
    assert args.provider == "statsbomb"
    assert args.train_frac == pytest.approx(0.60)
    assert args.val_frac == pytest.approx(0.20)
    assert args.season is None
    # Phase 21 additions (Elo / Poisson / bootstrap) are exposed on the parser.
    assert args.elo_k_factor == 20.0
    assert args.elo_home_advantage == 50.0
    assert args.elo_initial_rating == 1500.0
    assert args.max_goals == 8
    assert args.min_history == 5
    assert args.prior_strength == 1.0
    assert args.n_bootstrap == 2000
    assert args.bootstrap_seed == 42


def test_cli_compare_all_seasons(tmp_path):
    db = build_two_season_db(tmp_path)
    out = tmp_path / "reports"
    rc = _run(["model", "compare", "--db-path", str(db),
               "--competition-id", "2", "--output-dir", str(out)])
    assert rc == 0
    j = _artifact(out, "phase21_comparison_2_all.json")
    data = json.loads(j.read_text(encoding="utf-8"))
    assert {m for m in data["models"]} == {"logistic_regression", "poisson", "elo"}
    assert data["dataset"]["n_matches"] == 40
    assert (data["split"]["n_train"], data["split"]["n_val"], data["split"]["n_test"]) == (24, 8, 8)
    assert data["production_db_unchanged"] is True
    assert data["production_db_sha256"]
    assert len(data["paired"]) == 3
    # TXT companion with the matching stem.
    txt = out / (j.name[:-5] + ".txt")
    assert txt.exists()
    body = txt.read_text(encoding="utf-8")
    assert "Phase-21 Model Comparison Report" in body
    assert "McNemar" in body and "bootstrap" in body


def test_cli_compare_single_season(tmp_path):
    db = build_two_season_db(tmp_path)
    out = tmp_path / "reports"
    rc = _run(["model", "compare", "--db-path", str(db),
               "--competition-id", "2", "--season", "2015/16",
               "--output-dir", str(out)])
    assert rc == 0
    j = _artifact(out, "phase21_comparison_2_2015_16.json")
    data = json.loads(j.read_text(encoding="utf-8"))
    assert data["dataset"]["season_name"] == "2015/16"
    total = data["dataset"]["n_matches"]
    assert 1 <= total <= 20
    assert data["split"]["n_train"] + data["split"]["n_val"] + data["split"]["n_test"] == total
    assert data["split"]["n_train"] >= data["split"]["n_val"]
    assert data["split"]["n_train"] >= data["split"]["n_test"]
    assert data["production_db_unchanged"] is True


def test_cli_compare_multi_season_writes_one_artifact_each(tmp_path):
    db = build_two_season_db(tmp_path)
    out = tmp_path / "reports"
    rc = _run(["model", "compare", "--db-path", str(db),
               "--competition-id", "2",
               "--season", "2015/16", "--season", "2016/17",
               "--output-dir", str(out)])
    assert rc == 0
    jsons = sorted(out.glob("phase21_comparison_2_*.json"))
    assert len(jsons) == 2
    stems = {j.name for j in jsons}
    assert any(s.startswith("phase21_comparison_2_2015_16") for s in stems)
    assert any(s.startswith("phase21_comparison_2_2016_17") for s in stems)


def test_cli_compare_invalid_competition_returns_nonzero(tmp_path):
    db = build_two_season_db(tmp_path)
    out = tmp_path / "reports"
    rc = _run(["model", "compare", "--db-path", str(db),
               "--competition-id", "9999", "--output-dir", str(out)])
    assert rc == 1
    assert not list(out.rglob("*.json"))


def test_cli_compare_missing_db_returns_nonzero(tmp_path):
    out = tmp_path / "reports"
    missing = tmp_path / "does_not_exist.db"
    rc = _run(["model", "compare", "--db-path", str(missing),
               "--competition-id", "2", "--output-dir", str(out)])
    assert rc == 1
    assert not list(out.rglob("*.json"))
