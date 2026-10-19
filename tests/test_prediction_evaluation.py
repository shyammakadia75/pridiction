"""Phase-18 tests: pure-numpy metric correctness + dataset leakage safety."""
import math

import numpy as np
import pytest

from football_intelligence.features.engineering import FeatureEngineer
from football_intelligence.prediction.baseline import (
    N_OUTCOMES,
    OUTCOME_LABELS,
    chronological_split,
    compute_log_loss,
    prepare_training_data,
)
from football_intelligence.prediction.evaluation import (
    LEAKAGE_COLS,
    build_feature_dataset,
    classification_report,
    compute_brier_score,
    compute_macro_f1,
    compute_precision_recall,
    feature_columns,
)
from _phase18_db import STATS, build_two_season_db


# --- 1. Metric correctness on tiny hand-verified arrays ---------------------

def test_macro_f1_known_values():
    yt = [0, 0, 1, 2, 2]
    yp = [0, 1, 1, 2, 0]
    # class0: P=R=0.5,F1=0.5 ; class1: P=0.5,R=1,F1=2/3 ; class2: P=1,R=0.5,F1=2/3
    assert compute_macro_f1(yt, yp) == pytest.approx((0.5 + 2 / 3 + 2 / 3) / 3)
    pc = compute_precision_recall(yt, yp)
    assert pc["home_win"] == {"precision": 0.5, "recall": 0.5, "f1": 0.5,
                              "support": 2, "tp": 1, "fp": 1, "fn": 1}
    assert pc["draw"]["precision"] == 0.5 and pc["draw"]["recall"] == 1.0
    assert pc["away_win"]["precision"] == 1.0 and pc["away_win"]["recall"] == 0.5


def test_brier_and_logloss_uniform():
    y = np.array([0, 1, 2, 0, 1, 2])
    p = np.full((6, 3), 1 / 3)
    assert compute_brier_score(y, p, labels=list(OUTCOME_LABELS)) == pytest.approx(2 / 3)
    assert compute_log_loss(y, p, labels=list(OUTCOME_LABELS)) == pytest.approx(math.log(3))


def test_brier_perfect_is_zero():
    y = np.array([0, 1, 2])
    p = np.eye(3)
    assert compute_brier_score(y, p, labels=list(OUTCOME_LABELS)) == pytest.approx(0.0, abs=1e-9)
    assert compute_log_loss(y, p, labels=list(OUTCOME_LABELS)) == pytest.approx(0.0, abs=1e-9)


def test_classification_report_structure_and_ranges():
    rng = np.random.RandomState(0)
    y = rng.randint(0, 3, size=60)
    p = np.abs(rng.randn(60, 3)) + 0.1
    p = p / p.sum(axis=1, keepdims=True)
    rep = classification_report(y, p)
    assert rep["n_samples"] == 60
    assert rep["labels"] == list(OUTCOME_LABELS)
    assert 0.0 <= rep["accuracy"] <= 1.0
    assert 0.0 <= rep["macro_f1"] <= 1.0
    assert rep["log_loss"] >= 0.0
    assert rep["brier_score"] >= 0.0
    assert np.asarray(rep["confusion_matrix"]).shape == (N_OUTCOMES, N_OUTCOMES)
    assert set(rep["per_class_precision_recall"]) == set(OUTCOME_LABELS)
    assert set(rep["calibration"]) == set(OUTCOME_LABELS)


def test_empty_classification_report_is_safe():
    rep = classification_report(np.array([], dtype=int), np.zeros((0, 3)))
    assert rep["n_samples"] == 0
    assert rep["macro_f1"] == 0.0 and rep["accuracy"] == 0.0
    assert rep["log_loss"] == 0.0 and rep["brier_score"] == 0.0
    assert np.asarray(rep["confusion_matrix"]).shape == (N_OUTCOMES, N_OUTCOMES)
    assert all(rep["per_class_precision_recall"][lb]["support"] == 0 for lb in OUTCOME_LABELS)


# --- 2. Dataset leakage containment ----------------------------------------

def test_feature_columns_exclude_leakage(tmp_path):
    df = build_feature_dataset(build_two_season_db(tmp_path),
                               provider=STATS, provider_competition_id="2")
    feats = feature_columns(df)
    for col in LEAKAGE_COLS:
        assert col not in feats
    for col in ("home_goals", "away_goals", "match_date", "competition_id",
                "season_name", "provider_match_id"):
        if col in df.columns:
            assert col not in feats, col
    assert "_label_idx" not in feats and "outcome_label" not in feats


def test_build_feature_dataset_filters_competition(tmp_path):
    db = build_two_season_db(tmp_path)
    assert len(build_feature_dataset(db, provider=STATS, provider_competition_id="2")) == 40
    assert len(build_feature_dataset(db, provider=STATS, provider_competition_id="7")) == 2


def test_prepare_training_data_uses_only_past_history(tmp_path):
    # Build features from the RAW (unprepared) match frame; the engine must
    # retain all labelled rows and sort chronologically by target_match_date.
    from football_intelligence.features.engineering import load_matches_for_features
    db = build_two_season_db(tmp_path)
    raw = load_matches_for_features(db, provider=STATS, provider_competition_id="2")
    valid, X, y = prepare_training_data(raw, engineer=FeatureEngineer())
    assert len(X) == len(y) == len(valid) == len(raw)
    assert set(y.tolist()).issubset({0, 1, 2})
    tr, va, te = chronological_split(valid, train_frac=0.6, val_frac=0.2,
                                     date_col="target_match_date")
    assert tr["target_match_date"].max() <= va["target_match_date"].min()
