"""Tests for the Phase-24 calibration + ensemble module.

Coverage (all use synthetic data -- no real DB reads, no prod DB writes):

  1. Probability validation: shape, finiteness, normalization, class order.
  2. Class ordering: OUTCOME_LABELS == (home_win, draw, away_win).
  3. TemperatureScaler: deterministic, preserves normalization, requires fit.
  4. IsotonicCalibrator: identity fallback when a class has < 10 positives.
  5. Ensembles: equal-weight + optimal-weight are convex & normalized.
  6. DiriEloMapper: valid 3-class probabilities; requires fit.
  7. evaluate_proba returns standard metrics.
  8. Leakage: alignment rejects missing fixtures.
  9. reliability_bins structure.
  10. Split boundaries are chronological.
"""
from __future__ import annotations

import numpy as np
import pytest

from football_intelligence.prediction.calibration import (
    OUTCOME_LABELS,
    TRAIN_MAX,
    VAL_MAX,
    DiriEloMapper,
    IsotonicCalibrator,
    TemperatureScaler,
    ensemble_equal_weight,
    ensemble_optimal_weights,
    evaluate_proba,
    reliability_bins,
    validate_proba,
    LABEL_IDX,
)


# 1. Probability validation
def test_validate_proba_accepts_valid_vector():
    P = np.array([[0.3, 0.4, 0.3], [0.5, 0.2, 0.3]])
    validate_proba(P, 2)


def test_validate_proba_rejects_wrong_shape():
    P = np.array([[0.3, 0.4, 0.3], [0.5, 0.2, 0.3]])
    with pytest.raises(AssertionError):
        validate_proba(P, 3)


def test_validate_proba_rejects_non_finite():
    P = np.array([[0.3, np.nan, 0.3], [0.5, 0.2, 0.3]])
    with pytest.raises(AssertionError):
        validate_proba(P, 2)


def test_validate_proba_rejects_unnormalized():
    P = np.array([[0.3, 0.3, 0.3], [0.5, 0.2, 0.3]])
    with pytest.raises(AssertionError):
        validate_proba(P, 2)


def test_validate_proba_rejects_negative():
    P = np.array([[-0.1, 0.5, 0.6], [0.5, 0.2, 0.3]])
    with pytest.raises(AssertionError):
        validate_proba(P, 2)


# 2. Class ordering
def test_class_order_is_home_draw_away():
    assert OUTCOME_LABELS == ("home_win", "draw", "away_win")


def test_label_idx_matches_class_order():
    assert LABEL_IDX == {"home_win": 0, "draw": 1, "away_win": 2}


# 3. TemperatureScaler
def test_temperature_scaler_preserves_normalization():
    rng = np.random.default_rng(0)
    P = rng.dirichlet([1, 1, 1], size=50)
    y = rng.integers(0, 3, size=50)
    ts = TemperatureScaler().fit(P, y)
    cal = ts.predict_proba(P)
    validate_proba(cal, 50)


def test_temperature_scaler_deterministic():
    P = np.tile([0.4, 0.3, 0.3], (20, 1))
    y = np.array([0] * 10 + [1] * 10)
    a = TemperatureScaler().fit(P, y).predict_proba(P)
    b = TemperatureScaler().fit(P, y).predict_proba(P)
    np.testing.assert_allclose(a, b)


def test_temperature_scaler_requires_fit():
    ts = TemperatureScaler()
    with pytest.raises(AssertionError):
        ts.predict_proba(np.array([[0.4, 0.3, 0.3]]))


# 4. IsotonicCalibrator
def test_isotonic_falls_back_to_identity_for_rare_class():
    P = np.array([[0.5, 0.3, 0.2], [0.4, 0.4, 0.2], [0.6, 0.2, 0.2]])
    y = np.array([0, 0, 0])  # classes 1 and 2 have < 10 positives
    iso = IsotonicCalibrator().fit(P, y)
    cal = iso.predict_proba(P)
    validate_proba(cal, 3)
    np.testing.assert_allclose(cal, P, atol=1e-6)


def test_isotonic_normalizes_output():
    rng = np.random.default_rng(1)
    P = rng.dirichlet([2, 2, 2], size=100)
    y = rng.integers(0, 3, size=100)
    iso = IsotonicCalibrator().fit(P, y)
    cal = iso.predict_proba(P)
    validate_proba(cal, 100)


# 5. Ensembles
def test_equal_weight_ensemble_normalized():
    P1 = np.full((10, 3), 0.333333333)
    P2 = np.array([[0.5, 0.3, 0.2]] * 10)
    out = ensemble_equal_weight([P1, P2])
    validate_proba(out, 10)
    s = out.sum(axis=1)
    assert np.allclose(s, 1.0, atol=1e-9)


def test_optimal_weights_two_models():
    rng = np.random.default_rng(2)
    P1 = rng.dirichlet([2, 2, 2], size=40)
    P2 = rng.dirichlet([2, 2, 2], size=40)
    y = rng.integers(0, 3, size=40)
    pt, w, _ = ensemble_optimal_weights([P1, P2], y, [P1, P2])
    validate_proba(pt, 40)
    assert len(w) == 2
    assert abs(sum(w) - 1.0) < 1e-6
    assert all(wi >= -1e-9 for wi in w)


def test_optimal_weights_three_models_convex():
    rng = np.random.default_rng(3)
    P1 = rng.dirichlet([2, 2, 2], size=40)
    P2 = rng.dirichlet([2, 2, 2], size=40)
    P3 = rng.dirichlet([2, 2, 2], size=40)
    y = rng.integers(0, 3, size=40)
    pt, w, _ = ensemble_optimal_weights([P1, P2, P3], y, [P1, P2, P3])
    validate_proba(pt, 40)
    assert len(w) == 3
    assert all(wi >= -1e-9 for wi in w)
    assert abs(sum(w) - 1.0) < 1e-6


# 6. DiriEloMapper
def test_diri_elo_mapper_valid_probs():
    e = np.array([0.4, 0.5, 0.6, 0.45, 0.55, 0.65])
    y = np.array([0, 1, 2, 1, 0, 2])
    m = DiriEloMapper().fit(e, y)
    p = m.predict_proba(np.array([0.5, 0.6]))
    validate_proba(p, 2)
    assert m.config()["type"] == "dirichlet_elo_mapping"


def test_diri_elo_mapper_requires_fit():
    m = DiriEloMapper()
    with pytest.raises(AssertionError):
        m.predict_proba(np.array([0.5]))


# 7. evaluate_proba
def test_evaluate_proba_returns_standard_metrics():
    rng = np.random.default_rng(4)
    P = rng.dirichlet([2, 2, 2], size=60)
    y = rng.integers(0, 3, size=60)
    m = evaluate_proba(y, P, name="t")
    for k in ["n", "accuracy", "macro_f1", "log_loss", "brier_score",
              "per_class_precision_recall", "confusion_matrix",
              "class_distribution", "calibration"]:
        assert k in m
    assert m["n"] == 60


# 8. Leakage -- alignment rejects missing fixtures
def test_align_by_keys_rejects_missing_fixtures():
    from football_intelligence.prediction.calibration import _align_by_keys
    import pandas as pd
    preds = [
        {"match_date": "2019-08-10", "home_team_id": 1, "away_team_id": 2,
         "probs": {"home_win": 0.4, "draw": 0.3, "away_win": 0.3}},
    ]
    test_df = pd.DataFrame([
        {"target_match_date": "2019-08-10", "home_team_id": 1,
         "away_team_id": 2},
        {"target_match_date": "2019-08-11", "home_team_id": 3,
         "away_team_id": 4},
    ])
    with pytest.raises(AssertionError):
        _align_by_keys(preds, test_df)


# 9. reliability_bins structure
def test_reliability_bins_structure():
    P = np.array([[0.5, 0.3, 0.2], [0.2, 0.5, 0.3]])
    y = np.array([0, 1])
    b = reliability_bins(y, P, n_bins=5)
    assert set(b.keys()) == set(OUTCOME_LABELS)
    for c in OUTCOME_LABELS:
        assert len(b[c]["confidence"]) == 5
        assert len(b[c]["count"]) == 5


# 10. Split boundaries are chronological
def test_split_boundaries_are_chronological():
    assert TRAIN_MAX == "2016-07-31"
    assert VAL_MAX == "2019-07-31"
    assert TRAIN_MAX < VAL_MAX

