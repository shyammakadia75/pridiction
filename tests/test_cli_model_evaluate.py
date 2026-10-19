"""CLI tests for `football-intelligence model evaluate` (Phase 18, Task A)."""
import json

import pytest

from _phase18_db import build_two_season_db
from football_intelligence.cli import build_parser, main


def _run(argv):
    return main(list(argv))


def _artifact(out, name_glob):
    hits = list(sorted(out.rglob(name_glob)))
    assert len(hits) == 1, f"expected exactly one artifact matching {name_glob}, got {[p.name for p in hits]}"
    return hits[0]


def test_parser_registers_model_evaluate():
    parser = build_parser()
    args = parser.parse_args(["model", "evaluate", "--db-path", "x.db"])
    assert args.func.__name__ == "cmd_model_evaluate"
    assert args.competition_id == "2"
    assert args.provider == "statsbomb"
    assert args.train_frac == pytest.approx(0.60)
    assert args.val_frac == pytest.approx(0.20)
    assert args.season is None  # all seasons when omitted


def test_cli_evaluate_all_seasons(tmp_path):
    db = build_two_season_db(tmp_path)
    out = tmp_path / "reports"
    rc = _run(["model", "evaluate", "--db-path", str(db),
               "--competition-id", "2", "--output-dir", str(out)])
    assert rc == 0

    j = _artifact(out, "phase18_evaluation_2_all_*.json")
    data = json.loads(j.read_text(encoding="utf-8"))
    assert {m["name"] for m in data["models"]} == {"frequency_baseline", "logistic_regression"}
    # 40 valid matches -> chronological 60/20/20 -> 24/8/8 (matches the
    # synthetic DB contract asserted elsewhere).
    rc_counts = data["splits"]["raw_counts"]
    assert (rc_counts["total"], rc_counts["train"], rc_counts["val"], rc_counts["test"]) == (40, 24, 8, 8)
    assert data["production_db_unchanged"] is True
    assert data["production_db_sha256"]

    # TXT companion exists with the same timestamp stem.
    stem = j.name[: -len(".json")]
    txt = out / f"{stem}.txt"
    assert txt.exists()
    txt_body = txt.read_text(encoding="utf-8")
    assert "frequency_baseline" in txt_body and "logistic_regression" in txt_body
    assert "split counts" in txt_body


def test_cli_evaluate_single_season(tmp_path):
    db = build_two_season_db(tmp_path)
    out = tmp_path / "reports"
    rc = _run(["model", "evaluate", "--db-path", str(db),
               "--competition-id", "2", "--season", "2015/16",
               "--output-dir", str(out)])
    assert rc == 0
    j = _artifact(out, "phase18_evaluation_2_2015-16_*.json")
    data = json.loads(j.read_text(encoding="utf-8"))
    rc_counts = data["splits"]["raw_counts"]
    # A single 20-match season: split sizes must be internally consistent.
    total = rc_counts["total"]
    assert 1 <= total <= 20
    assert rc_counts["train"] + rc_counts["val"] + rc_counts["test"] == total
    assert rc_counts["train"] >= rc_counts["val"]
    assert rc_counts["train"] >= rc_counts["test"]
    # season filter is honored in the dataset metadata
    assert data["dataset"]["season_name"] == "2015/16"


def test_cli_evaluate_multi_season_writes_one_artifact_each(tmp_path):
    db = build_two_season_db(tmp_path)
    out = tmp_path / "reports"
    rc = _run(["model", "evaluate", "--db-path", str(db),
               "--competition-id", "2",
               "--season", "2015/16", "--season", "2016/17",
               "--output-dir", str(out)])
    assert rc == 0
    jsons = sorted(out.rglob("phase18_evaluation_2_*_*.json"))
    assert len(jsons) == 2
    names = {j.name for j in jsons}
    assert any(n.startswith("phase18_evaluation_2_2015-16_") for n in names)
    assert any(n.startswith("phase18_evaluation_2_2016-17_") for n in names)


def test_cli_evaluate_invalid_competition_returns_nonzero(tmp_path):
    db = build_two_season_db(tmp_path)
    out = tmp_path / "reports"
    rc = _run(["model", "evaluate", "--db-path", str(db),
               "--competition-id", "9999", "--output-dir", str(out)])
    assert rc == 1
    # No artifacts on a failed (no-data) run.
    assert not list(out.rglob("*.json"))


def test_cli_evaluate_missing_db_returns_nonzero(tmp_path):
    out = tmp_path / "reports"
    missing = tmp_path / "does_not_exist.db"
    rc = _run(["model", "evaluate", "--db-path", str(missing),
               "--competition-id", "2", "--output-dir", str(out)])
    assert rc == 1
