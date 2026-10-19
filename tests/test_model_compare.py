"""Phase 21 tests: LR vs. Elo vs. Poisson comparison harness.

All tests use a disposable synthetic DB (tests/_phase18_db.py and tests/_fixtures.py);
the production database is never read or written here.
"""
import json
from pathlib import Path

import numpy as np
import pytest

from _fixtures import make_initialized_db, _connect, _insert_match, _seed_competitions, _seed_teams
from _phase18_db import STATS, build_two_season_db
from football_intelligence.features.engineering import FeatureEngineer, load_matches_for_features
from football_intelligence.prediction import (
    ComparisonReport,
    chronological_split,
    feature_columns,
    prepare_training_data,
)
from football_intelligence.prediction.comparison import (
    mcp_mcnemar_exact,
    paired_bootstrap,
    run_model_comparison,
    save_comparison_report,
)
from football_intelligence.prediction.evaluation import _json_default
from football_intelligence.prediction.elo import EloModel
from football_intelligence.prediction.poisson import (
    DEFAULT_MAX_GOALS,
    DEFAULT_MIN_HISTORY,
    DEFAULT_PRIOR_STRENGTH,
    OUTCOME_LABELS,
    PoissonModel,
    run_poisson_chronological,
)

EL0 = 1.0 / (1.0 + 10.0 ** (-50.0 / 400.0))  # expected home score @ 1500 vs 1500, home adv=50


def _build_same_day_db(tmp_path):
    """Two fixtures on the SAME date (same-day leakage check).

    Fixture A: T1 (home) vs T2 (away), 2-0 -> home win
    Fixture B: T3 (home) vs T1 (away), 0-1 -> away win (T1)
    T1 appears in both fixtures on the same date, so B's pre-date baseline for T1
    must NOT reflect A's just-applied rating update (the leak-fixed behaviour).
    """
    db = make_initialized_db(tmp_path)
    with _connect(db) as conn:
        comps = _seed_competitions(conn, [(STATS, "2", "Premier League")])
        teams = _seed_teams(conn, [
            (STATS, "T1", "Team1"), (STATS, "T2", "Team2"), (STATS, "T3", "Team3"),
        ])
    cid = comps[(STATS, "2")]
    t1 = teams[(STATS, "T1")]
    t2 = teams[(STATS, "T2")]
    t3 = teams[(STATS, "T3")]
    same_day = "2024-08-12T15:00:00+00:00"
    with _connect(db) as conn:
        _insert_match(conn, provider=STATS, provider_match_id="sd1",
                      competition_id=cid, season_name="2024/25", match_date=same_day,
                      home_team_id=t1, away_team_id=t2, home_goals=2, away_goals=0,
                      match_status="available")
        _insert_match(conn, provider=STATS, provider_match_id="sd2",
                      competition_id=cid, season_name="2024/25", match_date=same_day,
                      home_team_id=t3, away_team_id=t1, home_goals=0, away_goals=1,
                      match_status="available")
    return db, t1, t2, t3


def _sha(path):
    import hashlib
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_compare_matches_evaluation_split_contract(tmp_path):
    db = build_two_season_db(tmp_path)
    rep = run_model_comparison(db, provider=STATS, provider_competition_id="2")
    assert (rep.dataset["n_matches"], rep.dataset["n_train"],
            rep.dataset["n_val"], rep.dataset["n_test"]) == (40, 24, 8, 8)


def test_compare_deterministic_and_read_only(tmp_path):
    db = build_two_season_db(tmp_path)
    before = _sha(db)
    r1 = run_model_comparison(db, provider=STATS, provider_competition_id="2", n_boot=50)
    r2 = run_model_comparison(db, provider=STATS, provider_competition_id="2", n_boot=50)
    after = _sha(db)
    assert before == after, "comparison must not write the production DB"
    assert r1.production_db_unchanged is True
    assert r1.production_db_sha256 == before
    # Deterministic metrics + aligned per-fixture predictions (ignore produced_at).
    _j = lambda r: json.dumps(r, default=_json_default, sort_keys=True)
    assert _j(r1.models) == _j(r2.models)
    assert r1.predictions == r2.predictions
    assert r1.paired == r2.paired
    assert len(r1.predictions) == r1.dataset["n_test"]


def test_compare_three_models_and_metric_shape(tmp_path):
    db = build_two_season_db(tmp_path)
    rep = run_model_comparison(db, provider=STATS, provider_competition_id="2", n_boot=50)
    assert isinstance(rep, ComparisonReport)
    assert set(rep.models) == {"logistic_regression", "poisson", "elo"}
    for nm, m in rep.models.items():
        assert 0.0 <= m["accuracy"] <= 1.0
        assert 0.0 <= m["macro_f1"] <= 1.0
        cm = m["confusion_matrix"]
        assert len(cm) == 3 and len(cm[0]) == 3
        assert sum(m["class_distribution"]["counts"].values()) == m["n_samples"]
    lr = rep.models["logistic_regression"]
    assert lr["brier_score"] is not None and lr["log_loss"] is not None
    assert lr["brier_expected_score"] is None
    pois = rep.models["poisson"]
    assert pois["brier_score"] is not None and pois["log_loss"] is not None
    em = rep.models["elo"]
    assert em["brier_score"] is None and em["log_loss"] is None
    assert em["brier_expected_score"] is not None



def test_compare_predictions_aligned_to_test_fixtures(tmp_path):
    db = build_two_season_db(tmp_path)
    rep = run_model_comparison(db, provider=STATS, provider_competition_id="2", n_boot=50)
    raw = load_matches_for_features(db, provider=STATS, provider_competition_id="2")
    raw = raw.sort_values(["match_date", "home_team_id", "away_team_id"]).reset_index(drop=True)
    eng, _X, _y = prepare_training_data(raw, engineer=FeatureEngineer())
    feats = feature_columns(eng)  # noqa: F841 (documents the feature gate used by LR)
    _, _, test_df = chronological_split(eng, train_frac=0.6, val_frac=0.2, date_col="target_match_date")
    elo = EloModel().fit(raw)
    by_key = {(str(m.match_date), int(m.home_team_id), int(m.away_team_id)): m for m in elo.history_}
    pois = PoissonModel()
    preds, _ = run_poisson_chronological(raw, model=pois)
    pby = {(str(r["match_date"]), int(r["home_team_id"]), int(r["away_team_id"])): r for r in preds}
    assert len(rep.predictions) == len(test_df)
    for rec, row in zip(rep.predictions, test_df.itertuples(index=False)):
        k = (str(row.target_match_date), int(row.home_team_id), int(row.away_team_id))
        assert rec["match_date"] == k[0]
        assert rec["home_team_id"] == k[1] and rec["away_team_id"] == k[2]
        e = by_key[k]
        assert rec["elo_expected_home"] == pytest.approx(e.expected_home)
        pr = pby[k]
        assert rec["poisson_proba"][0] == pytest.approx(pr["probs"][OUTCOME_LABELS[0]])
        assert rec["poisson_proba"][2] == pytest.approx(pr["probs"][OUTCOME_LABELS[2]])
        assert rec["logistic_regression_pred"] in (0, 1, 2)
        assert rec["poisson_pred"] in (0, 1, 2)
        assert rec["elo_pred"] in (0, 1, 2)


def test_mcnemar_exact_sanity():
    y = [0, 0, 0, 0, 0, 0, 0, 0]
    pi = [0, 0, 0, 0, 1, 1, 1, 1]  # correct first 4, wrong last 4
    pj = [0, 0, 0, 0, 0, 0, 0, 0]  # all correct
    res = mcp_mcnemar_exact(y, pi, pj)
    assert res["n01"] == 4          # i wrong, j right
    assert res["n10"] == 0         # i never right when j wrong
    assert res["n_discordant"] == 4
    assert 0.0 <= res["p_value"] <= 1.0
    assert res["p_value"] == pytest.approx(0.125)  # 2 * P(X<=0 | Binom(4, .5))


def test_mcnemar_all_agree_returns_p_one():
    y = [0, 1, 2, 0, 1]
    res = mcp_mcnemar_exact(y, y, y)
    assert res["n_discordant"] == 0
    assert res["p_value"] == 1.0
    assert res["statistic"] == 0.0


def test_paired_bootstrap_sanity():
    a = np.array([1, 1, 1, 1, 0, 0, 0, 0], dtype=float)   # mean 0.5
    b = np.array([1, 1, 1, 1, 1, 1, 0, 0], dtype=float)   # mean 0.75
    res = paired_bootstrap(a, b, n_boot=1000, seed=0)
    assert res["n_samples"] == 8
    assert res["mean_diff"] == pytest.approx(0.5 - 0.75)
    assert res["ci_low_95"] <= res["ci_high_95"]
    assert 0.0 <= res["p_value"] <= 1.0
    assert res["n_boot"] == 1000


def test_paired_bootstrap_identical_losses():
    a = np.array([0, 1, 0, 1], dtype=float)
    res = paired_bootstrap(a, a, n_boot=200, seed=1)
    assert res["mean_diff"] == 0.0
    assert res["ci_low_95"] <= 0.0 <= res["ci_high_95"]


def test_save_comparison_report_writes_both_artifacts(tmp_path):
    db = build_two_season_db(tmp_path)
    rep = run_model_comparison(db, provider=STATS, provider_competition_id="2", n_boot=50)
    out = tmp_path / "reports"
    arts = save_comparison_report(rep, str(out), name="phase21_comparison_2_all")
    json_path = Path(arts["json"])
    txt_path = Path(arts["txt"])
    assert json_path.exists() and txt_path.exists()
    data = json.loads(json_path.read_text(encoding="utf-8"))
    assert set(data["models"]) == {"logistic_regression", "poisson", "elo"}
    assert data["dataset"]["n_matches"] == 40
    assert data["production_db_unchanged"] is True
    assert data["production_db_sha256"]
    assert len(data["paired"]) == 3
    body = txt_path.read_text(encoding="utf-8")
    assert "Phase-21 Model Comparison Report" in body
    assert "logistic_regression" in body and "poisson" in body and "elo" in body
    assert "McNemar" in body


def test_elo_same_day_leakage_regression(tmp_path):
    """Two fixtures on one date must both score against the pre-date baseline."""
    db, t1, t2, t3 = _build_same_day_db(tmp_path)
    raw = load_matches_for_features(db, provider=STATS, provider_competition_id="2")
    model = EloModel().fit(raw)
    hist = model.history_
    assert len(hist) == 2
    assert hist[0].match_date == hist[1].match_date
    A = next(r for r in hist if r.home_team_id == t1 and r.away_team_id == t2)
    B = next(r for r in hist if r.home_team_id == t3 and r.away_team_id == t1)
    # Shared team T1 (home in A, away in B) must read its INITIAL 1500 rating in
    # BOTH records -- i.e. B cannot see A's intra-day update (the leak fix).
    assert A.home_rating == 1500.0
    assert B.away_rating == 1500.0
    assert A.expected_home == pytest.approx(EL0)
    assert B.expected_home == pytest.approx(EL0)
    assert A.expected_home == pytest.approx(B.expected_home)
    final = model.final_ratings()
    assert final[t1] > 1500.0          # T1 won both fixtures
    assert final[t2] < 1500.0          # T2 lost
    assert final[t3] < 1500.0          # T3 lost
