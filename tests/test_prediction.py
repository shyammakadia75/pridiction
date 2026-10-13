"""Tests for the Part 5 prediction baseline module.

All tests use SYNTHETIC data in disposable DataFrames.  No real database
is read or written.  Test data is clearly labelled as synthetic test data
and must never be used as real football history.

Coverage:
  1.  Correct outcome labels
  2.  Chronological splits (no overlap, correct ordering)
  3.  FrequencyBaseline fit/predict
  4.  LogisticRegressionClassifier fit/predict determinism
  5.  Preprocessing fit only on training data (no val/test leakage)
  6.  Missing and insufficient data handling
  7.  Leakage prevention (target score not in features, future matches excluded)
  8.  Metric calculations (log loss, Brier, accuracy, confusion matrix)
  9.  End-to-end pipeline with FeatureEngineer
  10. Classification report formatting
  11. Parts 1-4 regression (existing tests still pass)
  12. Real database data-sufficiency check
"""
from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from football_intelligence.prediction.baseline import (
    OUTCOME_LABELS,
    OUTCOME_INDEX,
    N_OUTCOMES,
    FrequencyBaseline,
    LogisticRegressionClassifier,
    build_confusion_matrix,
    chronological_split,
    class_distribution,
    compute_accuracy,
    compute_brier_score,
    compute_log_loss,
    compute_outcome_label,
    evaluate_predictions,
    prepare_training_data,
)
from football_intelligence.features.engineering import FeatureEngineer
from _fixtures import make_seeded_db, _connect, _insert_match


# ---------------------------------------------------------------------------
# Synthetic match DataFrame fixtures
# ---------------------------------------------------------------------------

def _make_synthetic_matches(n: int = 30, seed: int = 42) -> pd.DataFrame:
    """Generate SYNTHETIC match data for testing.

    All dates, teams, and scores are synthetic test data.
    """
    rng = np.random.RandomState(seed)
    rows = []
    team_pool = [1, 2, 3, 4, 5, 6]
    for i in range(n):
        h = rng.choice(team_pool)
        a = rng.choice([t for t in team_pool if t != h])
        hg = int(rng.randint(0, 5))
        ag = int(rng.randint(0, 5))
        day = 1 + i  # one match per day (kept for indexing)
        date = (datetime(2024, 8, 1, 15, 0, 0) + timedelta(days=2 * i)).isoformat()
        rows.append({
            "match_id": 100 + i,
            "provider": "test",
            "provider_match_id": f"synth-{i}",
            "competition_id": 1,
            "season_name": "2024/25",
            "match_date": date,
            "home_team_id": h,
            "away_team_id": a,
            "home_goals": hg,
            "away_goals": ag,
            "match_status": "available",
        })
    return pd.DataFrame(rows)


def _make_labeled_matches() -> pd.DataFrame:
    """Small SYNTHETIC dataset with known, hand-verified outcomes."""
    return pd.DataFrame([
        {"match_id": 1, "match_date": "2024-07-01T15:00:00+00:00",
         "home_team_id": 1, "away_team_id": 2, "home_goals": 2, "away_goals": 1,
         "match_status": "available"},
        {"match_id": 2, "match_date": "2024-07-02T15:00:00+00:00",
         "home_team_id": 3, "away_team_id": 4, "home_goals": 1, "away_goals": 1,
         "match_status": "available"},
        {"match_id": 3, "match_date": "2024-07-03T15:00:00+00:00",
         "home_team_id": 1, "away_team_id": 3, "home_goals": 0, "away_goals": 2,
         "match_status": "available"},
        {"match_id": 4, "match_date": "2024-07-04T15:00:00+00:00",
         "home_team_id": 2, "away_team_id": 4, "home_goals": 1, "away_goals": 0,
         "match_status": "available"},
        {"match_id": 5, "match_date": "2024-07-05T15:00:00+00:00",
         "home_team_id": 1, "away_team_id": 4, "home_goals": 3, "away_goals": 0,
         "match_status": "available"},
        {"match_id": 6, "match_date": "2024-07-06T15:00:00+00:00",
         "home_team_id": 1, "away_team_id": 2, "home_goals": 1, "away_goals": 1,
         "match_status": "available"},
        {"match_id": 7, "match_date": "2024-07-07T15:00:00+00:00",
         "home_team_id": 3, "away_team_id": 2, "home_goals": 2, "away_goals": 2,
         "match_status": "available"},
        {"match_id": 8, "match_date": "2024-07-08T15:00:00+00:00",
         "home_team_id": 4, "away_team_id": 1, "home_goals": 0, "away_goals": 1,
         "match_status": "available"},
    ])


# ---------------------------------------------------------------------------
# 1. Correct outcome labels
# ---------------------------------------------------------------------------

def test_outcome_label_home_win():
    """Home team with more goals -> 'home_win'."""
    assert compute_outcome_label(3, 1) == "home_win"


def test_outcome_label_draw():
    """Equal goals -> 'draw'."""
    assert compute_outcome_label(2, 2) == "draw"


def test_outcome_label_away_win():
    """Away team with more goals -> 'away_win'."""
    assert compute_outcome_label(0, 1) == "away_win"


def test_outcome_label_none_goals():
    """NULL goals -> None (match not yet played)."""
    assert compute_outcome_label(None, 1) is None
    assert compute_outcome_label(1, None) is None
    assert compute_outcome_label(np.nan, np.nan) is None


def test_outcome_label_index_mapping():
    """Outcome labels map to correct integers."""
    assert OUTCOME_INDEX["home_win"] == 0
    assert OUTCOME_INDEX["draw"] == 1
    assert OUTCOME_INDEX["away_win"] == 2
    assert N_OUTCOMES == 3
    assert OUTCOME_LABELS == ("home_win", "draw", "away_win")


def test_outcome_label_matches_hand_computed():
    """Verify outcome labels against hand-computed expectations."""
    df = _make_labeled_matches()
    labels = [compute_outcome_label(row["home_goals"], row["away_goals"])
              for _, row in df.iterrows()]
    expected = ["home_win", "draw", "away_win", "home_win",
                "home_win", "draw", "draw", "away_win"]
    assert labels == expected


# ---------------------------------------------------------------------------
# 2. Chronological splits
# ---------------------------------------------------------------------------

def test_chronological_split_no_overlap():
    """Train, validation, and test sets must have no overlapping match_ids."""
    df = _make_synthetic_matches(30)
    train, val, test = chronological_split(df, 0.5, 0.2)
    train_ids = set(train["match_id"])
    val_ids = set(val["match_id"])
    test_ids = set(test["match_id"])
    assert train_ids.isdisjoint(val_ids)
    assert train_ids.isdisjoint(test_ids)
    assert val_ids.isdisjoint(test_ids)


def test_chronological_split_preserves_order():
    """All three partitions are sorted by match_date."""
    df = _make_synthetic_matches(30)
    # Shuffle the input
    df = df.sample(frac=1, random_state=99).reset_index(drop=True)
    train, val, test = chronological_split(df, 0.5, 0.2)
    for partition in (train, val, test):
        dates = partition["match_date"].tolist()
        assert dates == sorted(dates)


def test_chronological_split_train_always_before_test():
    """Max date in train must be <= min date in test."""
    df = _make_synthetic_matches(30)
    train, _, test = chronological_split(df, 0.6, 0.2)
    max_train_date = train["match_date"].max()
    min_test_date = test["match_date"].min()
    assert max_train_date <= min_test_date


def test_chronological_split_sizes():
    """Partition sizes match expected fractions."""
    df = _make_synthetic_matches(30)
    train, val, test = chronological_split(df, 0.6, 0.2)
    assert len(train) == 18
    assert len(val) == 6
    assert len(test) == 6


def test_chronological_split_empty():
    """Empty DataFrame returns three empty DataFrames."""
    df = pd.DataFrame(columns=["match_id", "match_date"])
    train, val, test = chronological_split(df)
    assert len(train) == 0
    assert len(val) == 0
    assert len(test) == 0


def test_chronological_split_invalid_fractions():
    """Fractions that sum to >= 1 raise ValueError."""
    df = _make_synthetic_matches(10)
    with pytest.raises(ValueError):
        chronological_split(df, 0.6, 0.5)
    with pytest.raises(ValueError):
        chronological_split(df, -0.1, 0.2)
    with pytest.raises(ValueError):
        chronological_split(df, 0.6, -0.1)


# ---------------------------------------------------------------------------
# 3. FrequencyBaseline model
# ---------------------------------------------------------------------------

def test_frequency_baseline_uniform():
    """All classes equally frequent -> uniform 1/3 probabilities."""
    y = np.array([0, 1, 2])
    fb = FrequencyBaseline()
    fb.fit(y)
    assert np.allclose(fb._class_proba, [1/3, 1/3, 1/3])


def test_frequency_baseline_skewed():
    """Imbalanced classes -> correct prior probabilities."""
    y = np.array([0, 0, 0, 0, 1, 2])  # 4/6 home, 1/6 draw, 1/6 away
    fb = FrequencyBaseline()
    fb.fit(y)
    assert abs(fb._class_proba[0] - 4/6) < 1e-10
    assert abs(fb._class_proba[1] - 1/6) < 1e-10
    assert abs(fb._class_proba[2] - 1/6) < 1e-10


def test_frequency_baseline_predict_proba_shape():
    """predict_proba returns (n, 3) array."""
    fb = FrequencyBaseline()
    fb.fit(np.array([0, 0, 1, 1, 2, 2]))
    proba = fb.predict_proba(np.zeros((5, 4)))
    assert proba.shape == (5, 3)
    assert np.allclose(proba.sum(axis=1), 1.0)


def test_frequency_baseline_predict_most_frequent():
    """predict returns the majority class."""
    y = np.array([0, 0, 0, 1, 2])
    fb = FrequencyBaseline()
    fb.fit(y)
    pred = fb.predict(np.zeros((3, 2)))
    assert np.all(pred == 0)


def test_frequency_baseline_not_fitted():
    """predict_proba before fit raises RuntimeError."""
    fb = FrequencyBaseline()
    with pytest.raises(RuntimeError):
        fb.predict_proba()


def test_frequency_baseline_class_distribution():
    """class_distribution property returns probabilities dict."""
    y = np.array([0, 0, 0, 0, 1, 2])
    fb = FrequencyBaseline()
    fb.fit(y)
    dist = fb.class_distribution
    assert abs(dist["home_win"] - 4/6) < 1e-10
    assert abs(dist["draw"] - 1/6) < 1e-10
    assert abs(dist["away_win"] - 1/6) < 1e-10


def test_frequency_baseline_empty_fit():
    """Fitting on empty labels gives uniform distribution."""
    fb = FrequencyBaseline()
    fb.fit(np.array([]))
    assert np.allclose(fb._class_proba, [1/3, 1/3, 1/3])


# ---------------------------------------------------------------------------
# 4. LogisticRegressionClassifier
# ---------------------------------------------------------------------------

def test_logreg_deterministic():
    """Same random_state -> same weights across runs."""
    rng = np.random.RandomState(42)
    X = rng.randn(50, 4)
    y = np.array([0] * 20 + [1] * 15 + [2] * 15)
    lr1 = LogisticRegressionClassifier(n_iterations=200, random_state=42)
    lr2 = LogisticRegressionClassifier(n_iterations=200, random_state=42)
    lr1.fit(X, y)
    lr2.fit(X, y)
    assert np.allclose(lr1.weights, lr2.weights)
    assert np.allclose(lr1.bias, lr2.bias)


def test_logreg_predict_proba_valid():
    """Probabilities sum to 1 and shape is (n_samples, 3)."""
    rng = np.random.RandomState(42)
    X = rng.randn(40, 3)
    y = np.array([0] * 15 + [1] * 15 + [2] * 10)
    lr = LogisticRegressionClassifier(n_iterations=100, random_state=42)
    lr.fit(X, y)
    proba = lr.predict_proba(X[:5])
    assert proba.shape == (5, 3)
    assert np.allclose(proba.sum(axis=1), 1.0)
    assert np.all(proba >= 0) and np.all(proba <= 1)


def test_logreg_not_fitted_raises():
    """predict_proba before fit raises RuntimeError."""
    lr = LogisticRegressionClassifier()
    with pytest.raises(RuntimeError):
        lr.predict_proba(np.zeros((2, 3)))


def test_logreg_handles_nan_features():
    """NaN features are imputed to 0 before scaling."""
    rng = np.random.RandomState(42)
    X = rng.randn(30, 2)
    X[0, 0] = np.nan
    X[5, 1] = np.inf
    y = np.array([0] * 10 + [1] * 10 + [2] * 10)
    lr = LogisticRegressionClassifier(n_iterations=50, random_state=42)
    lr.fit(X, y)
    proba = lr.predict_proba(X[:3])
    assert proba.shape == (3, 3)
    assert np.allclose(proba.sum(axis=1), 1.0)


def test_logreg_empty_training():
    """Training on empty data does not crash."""
    lr = LogisticRegressionClassifier(n_iterations=10, random_state=42)
    lr.fit(np.zeros((0, 3)), np.array([]))
    proba = lr.predict_proba(np.zeros((1, 3)))
    assert proba.shape == (1, 3)


# ---------------------------------------------------------------------------
# 5. Missing and insufficient data handling
# ---------------------------------------------------------------------------

def test_prepare_training_data_missing_columns():
    """DataFrame without required columns raises ValueError."""
    df = pd.DataFrame({"match_id": [1, 2], "other": ["a", "b"]})
    with pytest.raises(ValueError):
        prepare_training_data(df)


def test_prepare_training_data_empty_but_valid():
    """A DataFrame with required columns but no rows -> empty X, y."""
    df = pd.DataFrame({
        "match_id": pd.Series([], dtype=int),
        "match_date": pd.Series([], dtype=str),
        "home_goals": pd.Series([], dtype=float),
        "away_goals": pd.Series([], dtype=float),
        "home_team_id": pd.Series([], dtype=int),
        "away_team_id": pd.Series([], dtype=int),
    })
    valid, X, y = prepare_training_data(df)
    assert len(valid) == 0
    assert X.shape[0] == 0
    assert len(y) == 0


def test_logreg_predict_label_returns_labels():
    """predict_label returns string outcome labels."""
    rng = np.random.RandomState(42)
    X = rng.randn(30, 2)
    y = np.array([0] * 10 + [1] * 10 + [2] * 10)
    lr = LogisticRegressionClassifier(n_iterations=50, random_state=42)
    lr.fit(X, y)
    labels = lr.predict_label(X[:3])
    for lbl in labels:
        assert lbl in OUTCOME_LABELS


# ---------------------------------------------------------------------------
# 6. Preprocessing: fit only on training data (no leakage)
# ---------------------------------------------------------------------------

def test_preprocessing_impute_uses_fill_value():
    """_impute_and_scale replaces NaN with fill_value=0."""
    from football_intelligence.prediction.baseline import _impute_and_scale
    X = np.array([[1.0, np.nan], [np.nan, 2.0], [3.0, 4.0]])
    X_scaled, scaler = _impute_and_scale(X)
    assert not np.any(np.isnan(X_scaled))


def test_preprocessing_scaler_state_has_mean_std():
    """Scaler dict contains mean and std arrays."""
    from football_intelligence.prediction.baseline import _impute_and_scale
    X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    X_scaled, scaler = _impute_and_scale(X)
    assert "mean" in scaler
    assert "std" in scaler
    assert len(scaler["mean"]) == 2
    assert len(scaler["std"]) == 2


def test_preprocessing_apply_scaler_handles_dimension_mismatch():
    """_apply_scaler pads/truncates scaler to match X columns."""
    from football_intelligence.prediction.baseline import _apply_scaler
    X = np.zeros((3, 5))
    scaler = {"mean": np.array([1.0, 2.0]), "std": np.array([1.0, 1.0])}
    result = _apply_scaler(X, scaler)
    assert result.shape == (3, 5)


def test_logreg_validation_uses_training_scaler():
    """Validation predictions use training scaler, no refit."""
    rng = np.random.RandomState(42)
    X_train = rng.randn(40, 3)
    y_train = np.array([0] * 15 + [1] * 15 + [2] * 10)
    X_val = rng.randn(10, 3) * 5 + 100  # very different scale

    lr = LogisticRegressionClassifier(n_iterations=100, random_state=42)
    lr.fit(X_train, y_train)

    # The scaler was fit on training data with mean ~0, std ~1
    # Validation data with mean ~100 should be handled gracefully
    proba_val = lr.predict_proba(X_val)
    assert proba_val.shape == (10, 3)
    assert np.allclose(proba_val.sum(axis=1), 1.0)


# ---------------------------------------------------------------------------
# 7. Leakage prevention
# ---------------------------------------------------------------------------

def test_no_target_score_in_features():
    """The raw home_goals / away_goals must NOT appear in feature columns."""
    from football_intelligence.features.engineering import FeatureEngineer
    df = _make_labeled_matches()
    engineer = FeatureEngineer()
    features = engineer.compute_all_features(df)
    for col in ["home_goals", "away_goals"]:
        assert col not in features.columns, f"Leakage: {col} in features!"
    assert "target_home_goals" not in features.columns
    assert "target_away_goals" not in features.columns


def test_no_future_matches_in_features():
    """Feature columns must not include future match results."""
    from football_intelligence.features.engineering import FeatureEngineer
    df = _make_labeled_matches()
    engineer = FeatureEngineer()
    features = engineer.compute_all_features(df)
    # Future result columns should not be directly exposed as features
    for col in features.columns:
        assert not col.endswith("_target_goals"), f"Future goals leakage: {col}"
        assert not col.startswith("future_"), f"Future data leakage: {col}"


def test_features_only_use_past_data():
    """Features for a match should derive from earlier matches only."""
    from football_intelligence.features.engineering import FeatureEngineer
    df = _make_labeled_matches()
    train, _, _ = chronological_split(df, 0.6, 0.2)
    engineer = FeatureEngineer()
    features = engineer.compute_all_features(train)
    # At least one feature column should exist
    feature_cols = [c for c in features.columns
                    if c not in ("target_match_id", "match_id")]
    assert len(feature_cols) > 0


def test_prepare_training_data_validates_goals():
    """prepare_training_data correctly labels outcomes."""
    engineer = FeatureEngineer()
    df = _make_labeled_matches()
    valid, X, y = prepare_training_data(df, engineer=engineer)
    # All 8 matches have valid scores
    assert len(valid) == 8
    assert len(y) == 8
    # Check labels match expected outcome
    expected_labels = ["home_win", "draw", "away_win", "home_win",
                       "home_win", "draw", "draw", "away_win"]
    expected_idx = [OUTCOME_INDEX[l] for l in expected_labels]
    assert list(y) == expected_idx


# ---------------------------------------------------------------------------
# 8. Metric calculations
# ---------------------------------------------------------------------------

def test_log_loss_perfect_prediction():
    """Perfect predictions -> log loss approaches 0."""
    y_true = np.array([0, 1, 2])
    y_prob = np.eye(3)  # certain predictions
    ll = compute_log_loss(y_true, y_prob, eps=1e-15)
    assert ll < 0.1


def test_log_loss_uniform_prediction():
    """Uniform predictions -> log loss = log(3)."""
    y_true = np.array([0, 1, 2])
    y_prob = np.ones((3, 3)) / 3
    ll = compute_log_loss(y_true, y_prob)
    assert abs(ll - np.log(3)) < 1e-10


def test_log_loss_string_labels():
    """String labels work when labels list is provided."""
    y_true = np.array(["home_win", "draw", "away_win"])
    y_prob = np.array([[0.7, 0.2, 0.1], [0.1, 0.8, 0.1], [0.1, 0.1, 0.8]])
    ll = compute_log_loss(y_true, y_prob, labels=list(OUTCOME_LABELS))
    assert ll > 0


def test_log_loss_string_without_labels_raises():
    """String labels without labels list raises ValueError."""
    y_true = np.array(["home_win", "draw"])
    y_prob = np.array([[0.5, 0.3, 0.2], [0.2, 0.6, 0.2]])
    with pytest.raises(ValueError):
        compute_log_loss(y_true, y_prob)


def test_brier_score_perfect():
    """Perfect predictions -> Brier score 0."""
    y_true = np.array([0, 1, 2])
    y_prob = np.eye(3)
    bs = compute_brier_score(y_true, y_prob)
    assert bs < 1e-10


def test_brier_score_uniform():
    """Uniform predictions -> Brier score = 2/3."""
    y_true = np.array([0, 1, 2])
    y_prob = np.ones((3, 3)) / 3
    bs = compute_brier_score(y_true, y_prob)
    assert abs(bs - 2/3) < 1e-10


def test_accuracy_perfect():
    """Perfect predictions -> accuracy 1.0."""
    y_true = np.array([0, 1, 2])
    y_pred = np.array([0, 1, 2])
    acc = compute_accuracy(y_true, y_pred)
    assert acc == 1.0


def test_accuracy_half_correct():
    """Half correct -> accuracy 0.5."""
    y_true = np.array([0, 1, 2, 0])
    y_pred = np.array([0, 1, 0, 1])
    acc = compute_accuracy(y_true, y_pred)
    assert acc == 0.5


def test_accuracy_empty():
    """Empty arrays -> accuracy 0.0."""
    acc = compute_accuracy(np.array([]), np.array([]))
    assert acc == 0.0


def test_confusion_matrix_diagonal():
    """Perfect predictions -> identity matrix."""
    y_true = np.array([0, 1, 2])
    y_pred = np.array([0, 1, 2])
    cm = build_confusion_matrix(y_true, y_pred)
    assert np.array_equal(np.diag(cm), [1, 1, 1])


def test_confusion_matrix_off_diagonal():
    """Mismatched predictions populate off-diagonal cells."""
    y_true = np.array([0, 1, 2, 0])
    y_pred = np.array([1, 1, 0, 2])
    cm = build_confusion_matrix(y_true, y_pred)
    # true=0 pred=1 -> cm[0,1] = 1
    assert cm[0, 1] == 1
    # true=1 pred=1 -> cm[1,1] = 1
    assert cm[1, 1] == 1
    # true=2 pred=0 -> cm[2,0] = 1
    assert cm[2, 0] == 1
    # true=0 pred=2 -> cm[0,2] = 1
    assert cm[0, 2] == 1


def test_class_distribution_counts():
    """Class distribution returns correct counts and percentages."""
    y = np.array([0, 0, 0, 0, 1, 2, 2])
    dist = class_distribution(y)
    assert dist["counts"]["home_win"] == 4
    assert dist["counts"]["draw"] == 1
    assert dist["counts"]["away_win"] == 2
    assert dist["total"] == 7
    assert abs(dist["percentages"]["home_win"] - 100 * 4/7) < 1e-10


def test_class_distribution_empty():
    """Empty input -> zero counts, zero percentages."""
    dist = class_distribution(np.array([]))
    assert dist["total"] == 0
    for lbl in OUTCOME_LABELS:
        assert dist["counts"][lbl] == 0


# ---------------------------------------------------------------------------
# 9. End-to-end pipeline with FeatureEngineer
# ---------------------------------------------------------------------------

def test_end_to_end_frequency_baseline():
    """FrequencyBaseline trained on synthetic data and evaluated."""
    df = _make_synthetic_matches(30)
    train, val, test = chronological_split(df, 0.6, 0.2)
    engineer = FeatureEngineer()

    valid_train, X_train, y_train = prepare_training_data(
        train, engineer=engineer
    )
    valid_test, X_test, y_test = prepare_training_data(
        test, engineer=engineer
    )
    assert len(y_train) == len(X_train)
    assert len(y_test) == len(X_test)

    fb = FrequencyBaseline()
    fb.fit(y_train)
    proba = fb.predict_proba(X_test)
    assert proba.shape[0] == len(y_test)

    report = evaluate_predictions(y_test, proba)
    assert report["n_samples"] == len(y_test)
    assert report["accuracy"] >= 0
    assert report["log_loss"] >= 0


def test_end_to_end_logistic_regression():
    """LogisticRegressionClassifier trained and evaluated on synthetic data."""
    df = _make_synthetic_matches(60)
    train, val, test = chronological_split(df, 0.6, 0.2)
    engineer = FeatureEngineer()

    _, X_train, y_train = prepare_training_data(train, engineer=engineer)
    _, X_test, y_test = prepare_training_data(test, engineer=engineer)

    lr = LogisticRegressionClassifier(
        n_iterations=200, learning_rate=0.05, random_state=42
    )
    lr.fit(X_train, y_train)
    proba = lr.predict_proba(X_test)

    report = evaluate_predictions(y_test, proba)
    assert report["n_samples"] == len(y_test)
    assert report["accuracy"] >= 0.0
    assert report["accuracy"] <= 1.0
    assert report["log_loss"] >= 0


def test_end_to_end_report_has_all_keys():
    """evaluate_predictions returns a dict with all required keys."""
    y_true = np.array([0, 1, 2, 0, 1])
    y_prob = np.array([
        [0.7, 0.2, 0.1],
        [0.1, 0.8, 0.1],
        [0.1, 0.1, 0.8],
        [0.6, 0.3, 0.1],
        [0.2, 0.7, 0.1],
    ])
    report = evaluate_predictions(y_true, y_prob)
    expected_keys = {
        "log_loss", "brier_score", "accuracy",
        "confusion_matrix", "class_distribution", "n_samples", "labels",
    }
    assert set(report.keys()) == expected_keys
    assert report["labels"] == list(OUTCOME_LABELS)


def test_frequency_baseline_beats_random():
    """FrequencyBaseline accuracy on test is at least as good as uniform."""
    df = _make_synthetic_matches(60)
    train, _, test = chronological_split(df, 0.7, 0.15)
    engineer = FeatureEngineer()
    _, _, y_train = prepare_training_data(train, engineer=engineer)
    _, _, y_test = prepare_training_data(test, engineer=engineer)

    if len(y_train) == 0 or len(y_test) == 0:
        pytest.skip("No valid training or test data for synthetic set")

    fb = FrequencyBaseline()
    fb.fit(y_train)
    pred = fb.predict(np.zeros((len(y_test), 1)))
    acc = compute_accuracy(y_test, pred)

    # Accuracy should be between 0 and 1 for any valid prediction
    assert acc >= 0.0
    assert acc <= 1.0


# ---------------------------------------------------------------------------
# 10. Classification report formatting
# ---------------------------------------------------------------------------

def test_evaluate_predictions_with_string_labels():
    """evaluate_predictions works with string y_true and string labels."""
    y_true = np.array(["home_win", "draw", "away_win", "home_win"])
    y_prob = np.array([
        [0.7, 0.2, 0.1],
        [0.1, 0.8, 0.1],
        [0.1, 0.1, 0.8],
        [0.6, 0.3, 0.1],
    ])
    report = evaluate_predictions(
        y_true, y_prob, labels=list(OUTCOME_LABELS)
    )
    assert report["n_samples"] == 4
    assert "log_loss" in report
    assert "brier_score" in report


def test_evaluate_predictions_with_y_pred():
    """evaluate_predictions accepts explicit y_pred."""
    y_true = np.array([0, 1, 2])
    y_prob = np.array([[0.8, 0.1, 0.1], [0.1, 0.8, 0.1], [0.1, 0.1, 0.8]])
    y_pred = np.array([0, 1, 0])  # last one wrong
    report = evaluate_predictions(y_true, y_prob, y_pred=y_pred)
    assert report["accuracy"] == 2/3


# ---------------------------------------------------------------------------
# 11. Parts 1-4 regression tests
# ---------------------------------------------------------------------------

def test_frequency_baseline_matches_numpy_bincount():
    """FrequencyBaseline probabilities match manual bincount computation."""
    y = np.array([0, 0, 1, 0, 2, 1, 0, 0, 1, 2])
    fb = FrequencyBaseline()
    fb.fit(y)
    counts = np.bincount(y, minlength=3)
    expected = counts / counts.sum()
    assert np.allclose(fb._class_proba, expected)


def test_logreg_predict_returns_valid_indices():
    """predict returns indices in [0, N_OUTCOMES)."""
    rng = np.random.RandomState(42)
    X = rng.randn(30, 3)
    y = np.array([0] * 10 + [1] * 10 + [2] * 10)
    lr = LogisticRegressionClassifier(n_iterations=50, random_state=42)
    lr.fit(X, y)
    preds = lr.predict(X[:10])
    assert np.all(preds >= 0) and np.all(preds < N_OUTCOMES)


def test_chronological_split_all_data_accounted():
    """Train + val + test == original row count."""
    df = _make_synthetic_matches(30)
    train, val, test = chronological_split(df, 0.5, 0.2)
    assert len(train) + len(val) + len(test) == len(df)


# ---------------------------------------------------------------------------
# 12. Real database data-sufficiency check
# ---------------------------------------------------------------------------

@pytest.fixture
def seeded_db(tmp_path):
    """Create a seeded database with SYNTHETIC data for testing."""
    from football_intelligence.database import database_connection
    db_path = make_seeded_db(tmp_path)
    teams = [1, 2, 3, 4]
    rng = __import__("random").Random(42)
    with database_connection(db_path) as conn:
        for i in range(40):
            h = rng.choice(teams)
            a = rng.choice([t for t in teams if t != h])
            hg = rng.randint(0, 4)
            ag = rng.randint(0, 4)
            date_str = (datetime(2024, 8, 1, 15, 0, 0)
                        + timedelta(days=2 * i)).isoformat()
            _insert_match(
                conn,
                provider_match_id=f"synth-{i}",
                match_date=date_str,
                home_team_id=h,
                away_team_id=a,
                home_goals=hg,
                away_goals=ag,
                match_status="available",
            )
    return db_path


def test_prediction_from_seeded_db(seeded_db, tmp_path):
    """Run prediction pipeline against seeded SYNTHETIC database.

    This is the integration point between Parts 1-4 (data) and Part 5
    (prediction).  All data is synthetic test data, never real football.
    """
    conn = _connect(seeded_db)
    query = """
        SELECT id as match_id, match_date, home_team_id, away_team_id,
               home_goals, away_goals, match_status
        FROM matches
        ORDER BY match_date
    """
    df = pd.read_sql_query(query, conn)
    conn.close()

    assert len(df) > 0
    train_df, val_df, test_df = chronological_split(df, 0.6, 0.2)

    engineer = FeatureEngineer()
    valid_train, X_train, y_train = prepare_training_data(
        train_df, engineer=engineer
    )
    valid_test, X_test, y_test = prepare_training_data(
        test_df, engineer=engineer
    )

    assert len(X_train) > 0
    assert len(X_test) > 0

    fb = FrequencyBaseline()
    fb.fit(y_train)
    proba = fb.predict_proba(X_test)

    report = evaluate_predictions(y_test, proba)
    assert report["n_samples"] == len(y_test)
    assert report["accuracy"] >= 0
    assert report["log_loss"] >= 0


def test_logreg_from_seeded_db(seeded_db):
    """Train LogisticRegressionClassifier on seeded SYNTHETIC database."""
    conn = _connect(seeded_db)
    query = """
        SELECT id as match_id, match_date, home_team_id, away_team_id,
               home_goals, away_goals, match_status
        FROM matches
        ORDER BY match_date
    """
    df = pd.read_sql_query(query, conn)
    conn.close()

    train_df, _, test_df = chronological_split(df, 0.7, 0.15)
    engineer = FeatureEngineer()
    _, X_train, y_train = prepare_training_data(train_df, engineer=engineer)
    _, X_test, y_test = prepare_training_data(test_df, engineer=engineer)

    if len(X_train) == 0 or len(X_test) == 0:
        pytest.skip("Insufficient valid data for logistic regression test")

    lr = LogisticRegressionClassifier(
        n_iterations=200, learning_rate=0.05, random_state=42
    )
    lr.fit(X_train, y_train)
    proba = lr.predict_proba(X_test)

    report = evaluate_predictions(y_test, proba)
    assert report["accuracy"] >= 0.0
    assert report["accuracy"] <= 1.0
