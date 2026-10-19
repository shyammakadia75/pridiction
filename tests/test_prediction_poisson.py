"""Phase 20 tests: Poisson goal model (chronological, read-only, deterministic).

All tests use a disposable synthetic DB (tests/_phase18_db.py); the production
database is never read or written here.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.stats import poisson as scipy_poisson

from _phase18_db import STATS, build_two_season_db
from football_intelligence.prediction.poisson import (
    DEFAULT_MAX_GOALS,
    DEFAULT_MIN_HISTORY,
    DEFAULT_PRIOR_STRENGTH,
    DEFAULT_RATE_FLOOR,
    OUTCOME_LABELS,
    PoissonFixturePrediction,
    PoissonModel,
    both_teams_to_score_probability,
    home_draw_away_probabilities,
    over_under_probability,
    poisson_pmf,
    run_poisson_chronological,
    run_poisson_evaluation,
    save_poisson_report,
    score_probability_grid,
)
from football_intelligence.cli import build_parser, main


def _sha(path):
    import hashlib
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture()
def db(tmp_path):
    return build_two_season_db(tmp_path)


# ---------------------------------------------------------------------------
# Poisson pmf correctness
# ---------------------------------------------------------------------------
def test_poisson_pmf_matches_scipy():
    lam = 1.7
    ks = [0, 1, 2, 3, 4, 5]
    got = poisson_pmf(ks, lam)
    expected = scipy_poisson.pmf(ks, lam)
    assert np.allclose(got, expected, atol=1e-10)


def test_poisson_pmf_negative_mean_clamped_to_floor():
    out = poisson_pmf([0, 1, 2], -3.0)
    expected = scipy_poisson.pmf([0, 1, 2], DEFAULT_RATE_FLOOR)
    assert np.allclose(out, expected, atol=1e-12)


def test_poisson_pmf_nan_mean_clamped_to_floor():
    out = poisson_pmf([0, 1], float("nan"))
    expected = scipy_poisson.pmf([0, 1], DEFAULT_RATE_FLOOR)
    assert np.allclose(out, expected, atol=1e-12)


# ---------------------------------------------------------------------------
# Score grid + outcome aggregation
# ---------------------------------------------------------------------------
def test_score_grid_sums_to_one_and_shape():
    g = score_probability_grid(1.5, 1.0, max_goals=8)
    assert g.shape == (9, 9)
    assert abs(float(g.sum()) - 1.0) < 1e-12


def test_score_grid_sums_to_one_for_small_max_goals():
    g = score_probability_grid(2.0, 2.0, max_goals=3)
    assert g.shape == (4, 4)
    assert abs(float(g.sum()) - 1.0) < 1e-12


def test_home_draw_away_sum_to_one():
    hw, dr, aw = home_draw_away_probabilities(1.5, 1.0, max_goals=8)
    assert abs((hw + dr + aw) - 1.0) < 1e-12


def test_home_win_dominant_when_home_rate_high():
    hw, dr, aw = home_draw_away_probabilities(3.0, 0.5, max_goals=8)
    assert hw > dr and dr > aw
    assert abs((hw + dr + aw) - 1.0) < 1e-12


def test_away_win_dominant_when_away_rate_high():
    hw, dr, aw = home_draw_away_probabilities(0.5, 3.0, max_goals=8)
    assert aw > dr and dr > hw


def test_zero_rates_fall_back_to_floor():
    hw, dr, aw = home_draw_away_probabilities(0.0, 0.0, max_goals=8)
    assert abs((hw + dr + aw) - 1.0) < 1e-12
    assert all(0.0 <= p <= 1.0 for p in (hw, dr, aw))


# ---------------------------------------------------------------------------
# Over/under and BTTS
# ---------------------------------------------------------------------------
def test_over_under_sum_to_one():
    over, under = over_under_probability(1.5, 1.0, line=2.5)
    assert abs((over + under) - 1.0) < 1e-12


def test_over_2_5_high_when_rates_high():
    over_hi, _ = over_under_probability(3.0, 3.0, line=2.5)
    over_lo, _ = over_under_probability(0.5, 0.5, line=2.5)
    assert over_hi > over_lo


def test_over_2_5_matches_poisson_cdf():
    lam = 1.5 + 1.0
    over_expected = 1.0 - scipy_poisson.cdf(2.5, lam)
    over_got, _ = over_under_probability(1.5, 1.0, line=2.5)
    assert abs(over_got - over_expected) < 1e-10


def test_both_teams_to_score_sum_to_one():
    btts, not_btts = both_teams_to_score_probability(1.5, 1.0)
    assert abs((btts + not_btts) - 1.0) < 1e-12


def test_btts_matches_independence_formula():
    lh, la = 1.5, 1.0
    expected = scipy_poisson.sf(0, lh) * scipy_poisson.sf(0, la)
    got, _ = both_teams_to_score_probability(lh, la)
    assert abs(got - expected) < 1e-10


def test_btts_zero_when_one_rate_is_zero():
    btts, _ = both_teams_to_score_probability(0.0, 3.0)
    assert btts < 1e-3


# ---------------------------------------------------------------------------
# Model: determinism, ordering, fallback
# ---------------------------------------------------------------------------
def _make_history_frame():
    rows = []
    for i in range(40):
        rows.append({
            "home_team_id": 1,
            "away_team_id": 2,
            "home_goals": 2,
            "away_goals": 1,
            "match_date": f"2020-01-{i+1:02d}",
        })
    return pd.DataFrame(rows)


def test_model_deterministic_for_identical_inputs():
    hist = _make_history_frame()
    m = PoissonModel()
    p1 = m.predict_fixture(1, 2, hist)
    p2 = m.predict_fixture(1, 2, hist)
    assert p1.probabilities == p2.probabilities
    assert p1.home_rate == p2.home_rate
    assert p1.away_rate == p2.away_rate


def test_model_home_away_ordering_matters():
    hist = _make_history_frame()
    m = PoissonModel()
    p1 = m.predict_fixture(1, 2, hist)
    p2 = m.predict_fixture(2, 1, hist)
    assert isinstance(p1, PoissonFixturePrediction)
    total = p1.probabilities["home_win"] + p1.probabilities["draw"] + p1.probabilities["away_win"]
    assert total == pytest.approx(1.0)


def test_model_unknown_teams_uses_league_prior():
    m = PoissonModel()
    empty = pd.DataFrame(columns=["home_team_id", "away_team_id", "home_goals",
                                  "away_goals", "match_date"])
    pred = m.predict_fixture(99, 100, empty)
    probs = pred.probabilities
    assert abs(sum(probs.values()) - 1.0) < 1e-12
    assert "No eligible pre-match history" in " ".join(pred.notes)


def test_model_insufficient_history_uses_prior():
    hist = pd.DataFrame([
        {"home_team_id": 1, "away_team_id": 2, "home_goals": 1,
         "away_goals": 0, "match_date": "2020-01-01"},
    ])
    m = PoissonModel(min_history=5)
    pred = m.predict_fixture(1, 2, hist)
    probs = pred.probabilities
    assert abs(sum(probs.values()) - 1.0) < 1e-12
    assert any("prior strength" in n for n in pred.notes)


# ---------------------------------------------------------------------------
# Chronological leakage prevention
# ---------------------------------------------------------------------------
def test_first_fixture_has_history_size_zero():
    df = pd.DataFrame([
        {"home_team_id": 1, "away_team_id": 2, "home_goals": 1,
         "away_goals": 0, "match_date": "2020-01-01", "match_status": "available"},
        {"home_team_id": 2, "away_team_id": 1, "home_goals": 0,
         "away_goals": 2, "match_date": "2020-01-02", "match_status": "available"},
        {"home_team_id": 1, "away_team_id": 2, "home_goals": 1,
         "away_goals": 1, "match_date": "2020-01-03", "match_status": "available"},
    ])
    m = PoissonModel()
    preds, _ = run_poisson_chronological(df, model=m)
    assert len(preds) == 3
    assert preds[0]["history_size"] == 0
    assert preds[1]["history_size"] == 1
    assert preds[2]["history_size"] == 2


def test_same_day_results_do_not_leak_into_history():
    # Two fixtures on the SAME date: the second must NOT see the first's result.
    df = pd.DataFrame([
        {"home_team_id": 1, "away_team_id": 2, "home_goals": 5,
         "away_goals": 0, "match_date": "2020-01-01", "match_status": "available"},
        {"home_team_id": 3, "away_team_id": 4, "home_goals": 0,
         "away_goals": 5, "match_date": "2020-01-01", "match_status": "available"},
    ])
    m = PoissonModel()
    preds, _ = run_poisson_chronological(df, model=m)
    # Both share the same date -> neither may use the other's result.
    assert preds[0]["history_size"] == 0
    assert preds[1]["history_size"] == 0
    # With empty history both fixtures fall back to the league prior (rates clamped
    # to the floor, symmetric -> home==away win prob, draw dominant).
    assert preds[0]["home_rate"] == pytest.approx(preds[1]["home_rate"])
    assert preds[0]["probs"]["draw"] > preds[0]["probs"]["home_win"]


def test_no_lookahead_first_match_uses_prior(db):
    from football_intelligence.prediction.poisson import _load_poisson_matches
    matches = _load_poisson_matches(db, provider=STATS, provider_competition_id="2")
    m = PoissonModel()
    preds, _ = run_poisson_chronological(matches, model=m)
    assert preds[0]["history_size"] == 0
    assert preds[0]["home_rate"] > 0


def test_predictions_are_deterministic_end_to_end():
    m = PoissonModel()
    matches = pd.DataFrame([
        {"home_team_id": 1, "away_team_id": 2, "home_goals": i % 4,
         "away_goals": (i + 1) % 3, "match_date": f"2021-01-{i+1:02d}",
         "match_status": "available"} for i in range(20)
    ])
    p1, _ = run_poisson_chronological(matches, model=m)
    p2, _ = run_poisson_chronological(matches, model=m)
    assert [r["probs"] for r in p1] == [r["probs"] for r in p2]


def test_model_handles_missing_goal_columns():
    hist = pd.DataFrame([
        {"home_team_id": 1, "away_team_id": 2, "match_date": "2020-01-01"},
    ])
    m = PoissonModel()
    pred = m.predict_fixture(1, 2, hist)
    probs = pred.probabilities
    assert abs(sum(probs.values()) - 1.0) < 1e-12



# ---------------------------------------------------------------------------
# End-to-end evaluation against synthetic DB
# ---------------------------------------------------------------------------
def test_run_poisson_evaluation_readonly_and_scored(db):
    before = _sha(db)
    rep = run_poisson_evaluation(db, provider=STATS, provider_competition_id="2")
    after = _sha(db)
    assert before == after
    assert rep.production_db_unchanged is True
    assert rep.production_db_sha256 == before
    assert rep.dataset["n_matches"] == 40
    assert rep.dataset["n_teams"] == 6
    assert rep.eval_metrics["n_scored"] == 40
    assert 0.0 <= rep.eval_metrics["accuracy"] <= 1.0
    assert 0.0 <= rep.eval_metrics["brier_score"] <= 1.0
    assert 0.0 <= rep.eval_metrics["log_loss"]
    assert len(rep.eval_metrics["confusion_matrix"]) == 3
    counts = rep.eval_metrics["class_distribution"]["counts"]
    assert sum(counts.values()) == 40
    assert set(counts) == {"home_win", "draw", "away_win"}
    gm = rep.eval_metrics["goal_metrics"]
    assert "expected_total_goals_mae" in gm
    assert "over_2_5_brier" in gm
    assert "btts_brier" in gm


def test_run_poisson_evaluation_deterministic(db):
    r1 = run_poisson_evaluation(db, provider=STATS, provider_competition_id="2")
    r2 = run_poisson_evaluation(db, provider=STATS, provider_competition_id="2")
    assert r1.eval_metrics["accuracy"] == r2.eval_metrics["accuracy"]
    assert r1.eval_metrics["brier_score"] == r2.eval_metrics["brier_score"]
    assert r1.eval_metrics["log_loss"] == r2.eval_metrics["log_loss"]
    assert r1.eval_metrics["goal_metrics"] == r2.eval_metrics["goal_metrics"]
    assert [p["probs"] for p in r1.predictions] == [p["probs"] for p in r2.predictions]


def test_run_poisson_evaluation_invalid_competition_raises(tmp_path):
    db = build_two_season_db(tmp_path)
    with pytest.raises(ValueError):
        run_poisson_evaluation(db, provider=STATS, provider_competition_id="9999")


def test_save_poisson_report_writes_both_artifacts(db, tmp_path):
    rep = run_poisson_evaluation(db, provider=STATS, provider_competition_id="2")
    out = tmp_path / "reports"
    art = save_poisson_report(rep, str(out), name="test_poisson")
    json_path = Path(art["json"])
    txt_path = Path(art["txt"])
    assert json_path.exists() and txt_path.exists()
    data = json.loads(json_path.read_text(encoding="utf-8"))
    assert data["dataset"]["n_matches"] == 40
    assert data["production_db_unchanged"] is True
    body = txt_path.read_text(encoding="utf-8")
    assert "Phase-20 Poisson Goal Model Report" in body
    assert "Goal metrics" in body


# ---------------------------------------------------------------------------
# CLI integration
# ---------------------------------------------------------------------------
def test_cli_poisson_parser():
    parser = build_parser()
    a = parser.parse_args(["ratings", "poisson", "--db-path", "x.db"])
    assert a.func.__name__ == "cmd_ratings_poisson"
    assert a.competition_id == "2"
    assert a.provider == "statsbomb"
    assert a.max_goals == 8
    assert a.min_history == 5
    assert a.prior_strength == 1.0
    assert a.season is None


def test_cli_poisson_smoke_all_seasons(db, tmp_path):
    out = tmp_path / "reports"
    rc = main(["ratings", "poisson", "--db-path", str(db),
               "--competition-id", "2", "--output-dir", str(out)])
    assert rc == 0
    j = sorted(out.glob("phase20_poisson_2_all_*.json"))
    assert len(j) == 1
    data = json.loads(j[0].read_text(encoding="utf-8"))
    assert data["dataset"]["n_matches"] == 40
    assert data["production_db_unchanged"] is True
    txt = out / (j[0].name[:-5] + ".txt")
    assert txt.exists()
    assert "Phase-20 Poisson Goal Model Report" in txt.read_text(encoding="utf-8")


def test_cli_poisson_single_season(db, tmp_path):
    out = tmp_path / "reports"
    rc = main(["ratings", "poisson", "--db-path", str(db),
               "--competition-id", "2", "--season", "2015/16",
               "--output-dir", str(out)])
    assert rc == 0
    j = sorted(out.glob("phase20_poisson_2_2015-16_*.json"))
    assert len(j) == 1


def test_cli_poisson_missing_db_returns_nonzero(tmp_path):
    out = tmp_path / "reports"
    missing = tmp_path / "nope.db"
    rc = main(["ratings", "poisson", "--db-path", str(missing),
               "--competition-id", "2", "--output-dir", str(out)])
    assert rc == 1
    assert not list(out.glob("*.json"))


def test_cli_poisson_invalid_competition_returns_nonzero(db, tmp_path):
    out = tmp_path / "reports"
    rc = main(["ratings", "poisson", "--db-path", str(db),
               "--competition-id", "9999", "--output-dir", str(out)])
    assert rc == 1


def test_cli_poisson_negative_max_goals_clamped(db, tmp_path):
    out = tmp_path / "reports"
    rc = main(["ratings", "poisson", "--db-path", str(db),
               "--competition-id", "2", "--max-goals", "-1",
               "--output-dir", str(out)])
    assert rc == 0
    j = sorted(out.glob("phase20_poisson_2_all_*.json"))
    assert len(j) == 1
    data = json.loads(j[0].read_text(encoding="utf-8"))
    assert data["config"]["max_goals"] == 0

