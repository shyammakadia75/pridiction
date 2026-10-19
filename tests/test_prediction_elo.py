"""Phase 19 tests: Elo ratings (chronological, read-only, deterministic).

All tests use a disposable synthetic DB (tests/_phase18_db.py); the production
database is never read or written here.
"""
import json
from pathlib import Path

import pytest

from _phase18_db import STATS, build_two_season_db
from football_intelligence.prediction.elo import (
    EloModel,
    EloReport,
    run_elo_evaluation,
    save_elo_report,
    load_elo_matches,
)


def _sha(path: Path) -> str:
    import hashlib
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _build_db(tmp_path):
    return build_two_season_db(tmp_path)


def test_elo_deterministic_and_readonly(tmp_path):
    db = _build_db(tmp_path)
    before = _sha(db)
    r1 = run_elo_evaluation(db, provider=STATS, provider_competition_id="2")
    r2 = run_elo_evaluation(db, provider=STATS, provider_competition_id="2")
    after = _sha(db)
    assert r1.ratings == r2.ratings, "Elo must be deterministic across runs"
    assert before == after, "production DB must be byte-identical (read-only)"
    assert r1.production_db_unchanged is True
    assert r1.production_db_sha256 == before


def test_elo_dataset_and_ratings_shape(tmp_path):
    db = _build_db(tmp_path)
    rep = run_elo_evaluation(db, provider=STATS, provider_competition_id="2")
    assert isinstance(rep, EloReport)
    assert rep.dataset["n_matches"] == 40          # 2 seasons x 20
    assert rep.dataset["n_teams"] == 6             # 6 synthetic teams
    assert rep.dataset["season_name"] is None      # all seasons for comp
    assert len(rep.ratings) == 6
    for val in rep.ratings.values():
        assert isinstance(val, float)
        assert 1000.0 <= val <= 2000.0               # sane Elo band


def test_elo_eval_metrics_shape(tmp_path):
    db = _build_db(tmp_path)
    rep = run_elo_evaluation(db, provider=STATS, provider_competition_id="2")
    m = rep.eval_metrics
    assert m["n"] == 40
    assert 0.0 <= m["accuracy"] <= 1.0
    assert 0.0 <= m["macro_f1"] <= 1.0
    assert 0.0 <= m["brier_expected_score"] <= 1.0
    assert len(m["confusion_matrix"]) == 3
    assert len(m["confusion_matrix"][0]) == 3
    counts = m["class_distribution"]["counts"]
    assert sum(counts.values()) == m["n"]
    assert set(counts) == {"home_win", "draw", "away_win"}


def test_elo_leakage_first_match_uses_initial_ratings(tmp_path):
    db = _build_db(tmp_path)
    df = load_elo_matches(db, provider=STATS, provider_competition_id="2")
    model = EloModel()
    model.fit(df)
    first = model.history_[0]
    # Both teams start at the initial 1500 -> expected home score is the
    # canonical value for a 50-point home advantage.
    assert first.home_rating == pytest.approx(1500.0)
    assert first.away_rating == pytest.approx(1500.0)
    assert first.expected_home == pytest.approx(1.0 / (1.0 + 10 ** (-50.0 / 400.0)))
    # Ratings have actually moved away from 1500 after learning.
    assert any(abs(v - model.initial_rating) > 1e-9 for v in model.final_ratings().values())


def test_elo_predict_home_win_prob_monotonic():
    model = EloModel()
    equal = model.predict_home_win_prob(1500, 1500)
    dominant_home = model.predict_home_win_prob(1900, 1100)
    dominant_away = model.predict_home_win_prob(1100, 1900)
    assert 0.0 <= equal <= 1.0
    assert dominant_home > equal > dominant_away
    # Elo expected-score is NOT complementary under a swapped call because the
    # home advantage is applied to the first argument in both directions;
    # complementarity requires E_away = 1 - E_home (different call shape).
    assert dominant_home > 0.9
    assert dominant_away < 0.1
    assert dominant_home > equal > dominant_away


def test_run_elo_invalid_competition_raises(tmp_path):
    db = _build_db(tmp_path)
    with pytest.raises(ValueError):
        run_elo_evaluation(db, provider=STATS, provider_competition_id="9999")


def test_save_elo_report_writes_both_artifacts(tmp_path):
    db = _build_db(tmp_path)
    rep = run_elo_evaluation(db, provider=STATS, provider_competition_id="2")
    out = tmp_path / "reports"
    art = save_elo_report(rep, str(out), name="test_elo")
    json_path = Path(art["json"])
    txt_path = Path(art["txt"])
    assert json_path.exists() and txt_path.exists()
    data = json.loads(json_path.read_text(encoding="utf-8"))
    assert data["dataset"]["n_matches"] == 40
    assert data["production_db_unchanged"] is True
    body = txt_path.read_text(encoding="utf-8")
    assert "Phase-19 Elo Ratings Report" in body
    assert "Final ratings" in body
    # at least one team name appears in the rendered table
    assert any(name in body for name in rep.ratings)
