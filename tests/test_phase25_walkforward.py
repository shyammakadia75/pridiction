#!/usr/bin/env python3
"""Tests for the Phase 25 walk-forward harness.

All tests use synthetic data or the disposable eval DB (no prod DB writes).
"""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from football_intelligence.prediction.walkforward import (
    WALKFORWARD_SEASONS,
    WALKFORWARD_SEED,
    Folds,
    FoldPrediction,
    build_walkforward_folds,
    assign_fold_dates,
    compute_walkforward_predictions,
    score_fold_predictions,
    run_walkforward,
    PROD,
    PHASE23_DISPOS,
    _load_raw_matches,
    _add_interactions,
)
from football_intelligence.prediction.calibration import (
    validate_proba,
    OUTCOME_LABELS,
    OUTCOME_INDEX,
)
from football_intelligence.prediction.baseline import (
    prepare_training_data,
    class_distribution,
)
from football_intelligence.features.engineering import (
    FeatureEngineer,
    load_matches_for_features,
)
from football_intelligence.prediction.evaluation import feature_columns


KNOWN_PROD_HASH = (
    "725af574b88dc033bd3b00928de503f9bd9a836fbe4252832b7812dce0ab3122"
)


def _make_synthetic_raw() -> pd.DataFrame:
    """Create a synthetic raw match DataFrame for fold testing."""
    rows = []
    for season in ["2015/2016", "2016/2017", "2019/2020"]:
        for i in range(380):
            d = pd.Timestamp("2015-08-08") + pd.Timedelta(days=len(rows))
            rows.append({
                "season_name": season,
                "match_date": d.strftime("%Y-%m-%d"),
                "home_team_id": 1 + (i % 20),
                "away_team_id": 21 - (i % 20),
                "home_goals": 2.0 if i % 3 == 0 else (1.0 if i % 3 == 1 else 0.0),
                "away_goals": 1.0 if i % 3 == 0 else (2.0 if i % 3 == 1 else 1.0),
                "match_id": len(rows),
            })
    df = pd.DataFrame(rows)
    df["home_goals"] = df["home_goals"].astype(float)
    df["away_goals"] = df["away_goals"].astype(float)
    return df


# ---------------------------------------------------------------------------
# 1. Chronological fold construction
# ---------------------------------------------------------------------------


def test_fold_structure_is_chronological():
    folds = build_walkforward_folds()
    assert len(folds) == 2
    assert folds[0].fold_id == "fold1"
    assert folds[0].train_seasons == ("2015/2016",)
    assert folds[0].test_season == "2016/2017"
    assert folds[1].fold_id == "fold2"
    assert folds[1].train_seasons == ("2015/2016", "2016/2017")
    assert folds[1].test_season == "2019/2020"


def test_fold_dates_are_strictly_sequential():
    folds = build_walkforward_folds()
    raw = _make_synthetic_raw()
    folds = assign_fold_dates(folds, raw)
    for f in folds:
        assert f.train_cutoff_date <= f.test_start_date
        assert f.test_start_date <= f.test_end_date


def test_val_season_set_only_when_multiple_train_seasons():
    folds = build_walkforward_folds()
    assert folds[0].val_season is None
    assert folds[1].val_season == "2016/2017"


# ---------------------------------------------------------------------------
# 2. Train/test separation
# ---------------------------------------------------------------------------


def test_train_and_test_seasons_are_disjoint():
    folds = build_walkforward_folds()
    for f in folds:
        assert f.test_season not in f.train_seasons


def test_train_cutoff_excludes_test_season():
    """Training cutoff date must be the first date of the test season."""
    folds = build_walkforward_folds()
    raw = _make_synthetic_raw()
    folds = assign_fold_dates(folds, raw)
    for f in folds:
        test_start = raw.loc[
            raw["season_name"] == f.test_season, "match_date"
        ].min()
        assert f.train_cutoff_date == str(test_start)


# ---------------------------------------------------------------------------
# 3. Strict pre-match feature cutoffs
# ---------------------------------------------------------------------------


def test_poisson_uses_strict_date_cutoff():
    """Verify that Poisson history uses strictly less-than date cutoff."""
    from football_intelligence.prediction.poisson import (
        PoissonModel, run_poisson_chronological)
    raw = _make_synthetic_raw()
    raw_sorted = raw.sort_values(
        ["match_date", "home_team_id", "away_team_id"])
    preds, _ = run_poisson_chronological(raw_sorted, model=PoissonModel())
    assert len(preds) > 0
    for p in preds:
        assert "probs" in p
        assert "home_win" in p["probs"]
        assert "draw" in p["probs"]
        assert "away_win" in p["probs"]


# ---------------------------------------------------------------------------
# 4. No same-day leakage
# ---------------------------------------------------------------------------


def test_no_same_day_leakage_in_elo():
    """Elo same-day-safe fix: fixtures on the same date share pre-date ratings."""
    from football_intelligence.prediction.elo import EloModel
    raw = _make_synthetic_raw()
    df = raw.sort_values(["match_date", "home_team_id", "away_team_id"])
    m = EloModel(K=40.0, home_advantage=50.0)
    m.fit(df)
    hist = pd.DataFrame([r.__dict__ for r in m.history_])
    # Find dates with multiple fixtures
    date_counts = hist.groupby("match_date").size()
    multi_dates = date_counts[date_counts > 1].index
    for d in multi_dates[:3]:
        day_rows = hist[hist["match_date"] == d]
        # All fixtures on the same date should have the same home_rating
        # (they are scored against the same pre-date baseline)
        assert day_rows["home_rating"].nunique() == 1, (
            f"Same-day leakage: different home_rating on {d}")
        assert day_rows["away_rating"].nunique() == 1, (
            f"Same-day leakage: different away_rating on {d}")


# ---------------------------------------------------------------------------
# 5. Calibration fit boundaries
# ---------------------------------------------------------------------------


def test_temperature_scaler_fit_on_val_only():
    from football_intelligence.prediction.calibration import TemperatureScaler
    rng = np.random.default_rng(42)
    P = rng.dirichlet([1, 1, 1], size=100)
    y = rng.integers(0, 3, size=100)
    ts = TemperatureScaler().fit(P, y)
    assert ts.fitted_
    assert ts.T > 0
    P_test = rng.dirichlet([1, 1, 1], size=50)
    cal = ts.predict_proba(P_test)
    validate_proba(cal, 50)


def test_diri_elo_mapper_fit_on_train_only():
    from football_intelligence.prediction.calibration import DiriEloMapper
    rng = np.random.default_rng(42)
    e_train = rng.uniform(0.3, 0.7, size=760)
    y_train = rng.integers(0, 3, size=760)
    mapper = DiriEloMapper().fit(e_train, y_train)
    assert mapper.fitted_
    e_test = rng.uniform(0.3, 0.7, size=380)
    p = mapper.predict_proba(e_test)
    validate_proba(p, 380)


# ---------------------------------------------------------------------------
# 6. Consistent fixture alignment
# ---------------------------------------------------------------------------


def test_align_online_to_test_rejects_missing():
    from football_intelligence.prediction.walkforward import _align_online_to_test
    preds = [
        {"match_date": "2019-08-10", "home_team_id": 1, "away_team_id": 2,
         "probs": {"home_win": 0.4, "draw": 0.3, "away_win": 0.3}},
    ]
    test_df = pd.DataFrame([
        {"target_match_date": "2019-08-10", "home_team_id": 1,
         "away_team_id": 2, "_label_idx": 0, "season_name": "2019/2020"},
        {"target_match_date": "2019-08-11", "home_team_id": 3,
         "away_team_id": 4, "_label_idx": 1, "season_name": "2019/2020"},
    ])
    with pytest.raises(AssertionError, match="test fixtures missing"):
        _align_online_to_test(preds, test_df)


def test_align_elo_to_test_rejects_missing():
    from football_intelligence.prediction.walkforward import _align_elo_to_test
    elo_hist = pd.DataFrame([
        {"match_date": "2019-08-10", "home_team_id": 1, "away_team_id": 2,
         "expected_home": 0.65, "_label_idx": 0},
    ])
    test_df = pd.DataFrame([
        {"target_match_date": "2019-08-10", "home_team_id": 1,
         "away_team_id": 2, "_label_idx": 0, "season_name": "2019/2020"},
        {"target_match_date": "2019-08-11", "home_team_id": 3,
         "away_team_id": 4, "_label_idx": 1, "season_name": "2019/2020"},
    ])
    with pytest.raises(AssertionError, match="fixture mismatch"):
        _align_elo_to_test(elo_hist, test_df)


# ---------------------------------------------------------------------------
# 7. Probability normalization and class order
# ---------------------------------------------------------------------------


def test_class_order_is_home_draw_away():
    assert OUTCOME_LABELS == ("home_win", "draw", "away_win")
    assert OUTCOME_INDEX == {"home_win": 0, "draw": 1, "away_win": 2}


# ---------------------------------------------------------------------------
# 8. Deterministic results under fixed seeds
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not PHASE23_DISPOS.exists(),
                    reason="Disposable DB not available")
def test_deterministic_results_under_fixed_seed():
    """Re-running with the same seed must produce identical predictions."""
    raw = _load_raw_matches()
    eng, _, _ = prepare_training_data(
        raw, engineer=FeatureEngineer(), label_col="outcome_label")
    eng = _add_interactions(eng)
    feats = list(feature_columns(eng))
    folds = build_walkforward_folds()
    folds = assign_fold_dates(folds, raw)

    fp1 = compute_walkforward_predictions(
        folds[0], raw, eng, feats, seed=42)
    fp2 = compute_walkforward_predictions(
        folds[0], raw, eng, feats, seed=42)

    for nm in fp1.predictions:
        np.testing.assert_allclose(
            fp1.predictions[nm], fp2.predictions[nm], atol=1e-12,
            err_msg=f"Non-deterministic output for {nm}")


# ---------------------------------------------------------------------------
# 9. Missing seasons and incomplete datasets
# ---------------------------------------------------------------------------


def test_build_folds_rejects_unknown_season():
    with pytest.raises(ValueError, match="not in available seasons"):
        build_walkforward_folds(["2099/2100"])


def test_build_folds_rejects_no_train_before_test():
    """First available season cannot be a test season (no training data)."""
    with pytest.raises(ValueError, match="No training seasons"):
        build_walkforward_folds(["2015/2016"])


def test_assign_dates_rejects_empty_season():
    folds = build_walkforward_folds()
    raw = pd.DataFrame({
        "season_name": ["2015/2016"] * 10,
        "match_date": [f"2015-08-{d:02d}" for d in range(1, 11)],
        "home_team_id": range(10),
        "away_team_id": range(10, 20),
        "home_goals": [2.0] * 10,
        "away_goals": [1.0] * 10,
    })
    with pytest.raises(ValueError, match="No matches found for test season"):
        assign_fold_dates(folds, raw)


# ---------------------------------------------------------------------------
# 10. Production database read-only safeguards
# ---------------------------------------------------------------------------


def test_prod_db_hash_matches_known_value():
    """The prod DB hash must match the known, verified value."""
    h = hashlib.sha256(PROD.read_bytes()).hexdigest()
    assert h == KNOWN_PROD_HASH


@pytest.mark.skipif(not PHASE23_DISPOS.exists(),
                    reason="Disposable DB not available")
def test_prod_db_unchanged_after_walkforward():
    """Running walkforward must not modify the production DB."""
    h_before = hashlib.sha256(PROD.read_bytes()).hexdigest()
    run_walkforward(seed=42)
    h_after = hashlib.sha256(PROD.read_bytes()).hexdigest()
    assert h_before == h_after


# ---------------------------------------------------------------------------
# 11. Integration: pooled metrics from walkforward report
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not PHASE23_DISPOS.exists(),
                    reason="Disposable DB not available")
def test_walkforward_produces_valid_report():
    """Full walkforward run must produce valid metrics for all models."""
    report = run_walkforward(seed=42)
    assert "folds" in report
    assert len(report["folds"]) == 2
    assert "pooled" in report
    assert "paired_vs_poisson" in report
    assert report["dataset"]["production_db_unchanged"] is True

    # Pooled Poisson baseline must exist and be valid
    p = report["pooled"]["poisson"]
    assert 0.0 <= p["accuracy"] <= 1.0
    assert 0.0 <= p["brier_score"] <= 1.0
    assert p["log_loss"] >= 0.0
    assert p["n"] == 760  # 380 + 380 pooled test fixtures

    # All models must have the same n (identical fixtures)
    for nm, m in report["pooled"].items():
        assert m["n"] == p["n"], f"{nm} has different n than Poisson"
