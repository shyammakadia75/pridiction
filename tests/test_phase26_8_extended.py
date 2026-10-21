"""Regression tests for Phase-26.8 extended walk-forward."""
import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from football_intelligence.prediction.walkforward_extended import (
    INTRA_SEASON_CUTOFFS,
    EXTENDED_FOLD_COUNT,
    FAST_MODELS,
    build_extended_folds,
    compute_fast_predictions,
    brier_decomposition,
    run_walkforward_extended,
)
from football_intelligence.prediction import walkforward as wf
from football_intelligence.prediction.calibration import validate_proba
from football_intelligence.prediction.baseline import prepare_training_data
from football_intelligence.features.engineering import FeatureEngineer
from football_intelligence.prediction.evaluation import feature_columns

KNOWN_PROD_HASH = (
    "725af574b88dc033bd3b00928de503f9bd9a836fbe4252832b7812dce0ab3122"
)
HAS_DISPOS = wf.PHASE23_DISPOS.exists()
HAS_PROD = wf.PROD.exists()


# --- 1. Fold construction ---

def test_intra_season_cutoffs():
    assert "fold3_mids" in INTRA_SEASON_CUTOFFS
    assert "fold4_covid" in INTRA_SEASON_CUTOFFS
    assert INTRA_SEASON_CUTOFFS["fold3_mids"] == "2019-12-29"
    assert INTRA_SEASON_CUTOFFS["fold4_covid"] == "2020-03-09"


@pytest.mark.skipif(not HAS_DISPOS, reason="DB not available")
def test_extended_fold_count():
    raw = wf._load_raw_matches()
    folds = build_extended_folds(raw)
    assert len(folds) == EXTENDED_FOLD_COUNT == 4


@pytest.mark.skipif(not HAS_DISPOS, reason="DB not available")
def test_fold_ids_and_seasons():
    raw = wf._load_raw_matches()
    folds = build_extended_folds(raw)
    assert [f.fold_id for f in folds] == ["fold1", "fold2", "fold3", "fold4"]
    assert folds[0].train_seasons == ("2015/2016",)
    assert folds[0].test_season == "2016/2017"
    assert folds[1].train_seasons == ("2015/2016", "2016/2017")
    assert folds[1].test_season == "2019/2020"
    for f in folds[2:]:
        assert f.test_season == "2019/2020"
        assert f.train_seasons == ("2015/2016", "2016/2017")
        assert f.val_season is None


@pytest.mark.skipif(not HAS_DISPOS, reason="DB not available")
def test_fold3_and_fold4_disjoint():
    raw = wf._load_raw_matches()
    folds = build_extended_folds(raw)
    f3 = next(f for f in folds if f.fold_id == "fold3")
    f4 = next(f for f in folds if f.fold_id == "fold4")
    assert f3.test_end_date < f4.test_start_date
    assert f3.train_cutoff_date == INTRA_SEASON_CUTOFFS["fold3_mids"]
    assert f4.train_cutoff_date == INTRA_SEASON_CUTOFFS["fold4_covid"]


@pytest.mark.skipif(not HAS_DISPOS, reason="DB not available")
def test_build_folds_no_intra_season():
    raw = wf._load_raw_matches()
    folds = build_extended_folds(raw, include_intra_season=False)
    assert len(folds) == 2
    assert [f.fold_id for f in folds] == ["fold1", "fold2"]


# --- 2. Brier decomposition identity (no DB) ---

def test_brier_decomposition_identity():
    rng = np.random.default_rng(12345)
    y = rng.integers(0, 3, size=500)
    proba = rng.dirichlet([1, 1, 1], size=500)
    dec = brier_decomposition(y, proba)
    assert abs(dec["check"] - dec["brier"]) < 0.01


def test_brier_decomposition_uniform_probs():
    y = np.random.default_rng(99).integers(0, 3, size=300)
    proba = np.full((300, 3), 1.0 / 3.0)
    dec = brier_decomposition(y, proba)
    assert dec["reliability"] < 0.01
    assert abs(dec["check"] - dec["brier"]) < 0.01


def test_brier_decomposition_empty():
    dec = brier_decomposition(np.array([], dtype=int), np.empty((0, 3)))
    assert dec["brier"] == 0.0
    assert dec["n"] == 0


# --- 3. compute_fast_predictions ---

@pytest.mark.skipif(not HAS_DISPOS, reason="DB not available")
def test_fast_predictions_all_models():
    raw = wf._load_raw_matches()
    eng, _, _ = prepare_training_data(
        raw, engineer=FeatureEngineer(), label_col="outcome_label")
    eng = wf._add_interactions(eng)
    feats = list(feature_columns(eng))
    folds = build_extended_folds(raw)
    fp = compute_fast_predictions(folds[0], raw, eng, feats, seed=42)
    for nm in FAST_MODELS:
        assert nm in fp.predictions
        p = fp.predictions[nm]
        assert p.shape == (fp.n_test, 3)
        validate_proba(p, fp.n_test)


@pytest.mark.skipif(not HAS_DISPOS, reason="DB not available")
def test_fast_predictions_row_counts():
    raw = wf._load_raw_matches()
    eng, _, _ = prepare_training_data(
        raw, engineer=FeatureEngineer(), label_col="outcome_label")
    eng = wf._add_interactions(eng)
    feats = list(feature_columns(eng))
    folds = build_extended_folds(raw)
    expected = {"fold1": 380, "fold2": 380, "fold3": 91, "fold4": 93}
    for fold in folds:
        fp = compute_fast_predictions(fold, raw, eng, feats, seed=42)
        assert fp.n_test == expected[fold.fold_id]


@pytest.mark.skipif(not HAS_DISPOS, reason="DB not available")
def test_fast_predictions_deterministic():
    raw = wf._load_raw_matches()
    eng, _, _ = prepare_training_data(
        raw, engineer=FeatureEngineer(), label_col="outcome_label")
    eng = wf._add_interactions(eng)
    feats = list(feature_columns(eng))
    folds = build_extended_folds(raw)
    fp1 = compute_fast_predictions(folds[0], raw, eng, feats, seed=42)
    fp2 = compute_fast_predictions(folds[0], raw, eng, feats, seed=42)
    for nm in FAST_MODELS:
        np.testing.assert_allclose(
            fp1.predictions[nm], fp2.predictions[nm], atol=1e-12)


# --- 4. Full integration tests ---

@pytest.mark.skipif(not HAS_DISPOS, reason="DB not available")
def test_report_structure():
    report = run_walkforward_extended(seed=42, n_boot=50, fast=True)
    required = {"phase", "title", "produced_at", "config", "dataset",
        "folds", "pooled", "pooled_seasonal", "pooled_intra",
        "brier_decomposition", "paired_vs_poisson",
        "pooled_y_true_class_distribution", "notes",
        "ranking_by_brier_pooled"}
    assert required.issubset(report.keys())
    assert report["phase"] == 26.8
    assert len(report["folds"]) == 4
    assert report["config"]["fast_mode"] is True


@pytest.mark.skipif(not HAS_DISPOS, reason="DB not available")
def test_pooled_n_values():
    report = run_walkforward_extended(seed=42, n_boot=50, fast=True)
    assert report["pooled_seasonal"]["poisson"]["n"] == 760
    assert report["pooled_intra"]["poisson"]["n"] == 564


@pytest.mark.skipif(not HAS_DISPOS, reason="DB not available")
def test_poisson_best_by_brier():
    report = run_walkforward_extended(seed=42, n_boot=50, fast=True)
    pb = report["pooled"]["poisson"]["brier_score"]
    for nm, m in report["pooled"].items():
        assert m["brier_score"] >= pb - 1e-9


@pytest.mark.skipif(not HAS_DISPOS, reason="DB not available")
def test_poisson_metrics_range():
    report = run_walkforward_extended(seed=42, n_boot=50, fast=True)
    p = report["pooled"]["poisson"]
    assert 0.54 <= p["accuracy"] <= 0.56
    assert 0.57 <= p["brier_score"] <= 0.60
    assert 0.96 <= p["log_loss"] <= 1.00


@pytest.mark.skipif(not HAS_DISPOS, reason="DB not available")
def test_brier_decomp_identity_in_report():
    report = run_walkforward_extended(seed=42, n_boot=50, fast=True)
    for pool_name, dec in report["brier_decomposition"].items():
        d = dec["poisson"]
        assert abs(d["check"] - d["brier"]) < 0.01, pool_name


# --- 5. Production DB & paired significance ---

@pytest.mark.skipif(not HAS_PROD, reason="Prod DB not available")
def test_prod_db_hash_matches():
    h = hashlib.sha256(wf.PROD.read_bytes()).hexdigest()
    assert h == KNOWN_PROD_HASH


@pytest.mark.skipif(not HAS_DISPOS, reason="DB not available")
def test_prod_db_unchanged():
    h_before = hashlib.sha256(wf.PROD.read_bytes()).hexdigest()
    run_walkforward_extended(seed=42, n_boot=20, fast=True)
    h_after = hashlib.sha256(wf.PROD.read_bytes()).hexdigest()
    assert h_before == h_after


@pytest.mark.skipif(not HAS_DISPOS, reason="DB not available")
def test_paired_significance_both_baselines():
    report = run_walkforward_extended(seed=42, n_boot=50, fast=True)
    pairs = [p["pair"][1] for p in report["paired_vs_poisson"]]
    assert "frequency_baseline" in pairs
    assert "logistic_regression" in pairs
    assert len(pairs) == 2


@pytest.mark.skipif(not HAS_DISPOS, reason="DB not available")
def test_paired_significance_poisson_superior():
    report = run_walkforward_extended(seed=42, n_boot=50, fast=True)
    for t in report["paired_vs_poisson"]:
        assert t["mcnemar"]["p_value"] < 0.001
        assert t["bootstrap"]["p_value"] < 0.01


@pytest.mark.skipif(not HAS_DISPOS, reason="DB not available")
def test_fast_mode_only_fast_models():
    report = run_walkforward_extended(seed=42, n_boot=20, fast=True)
    for nm in report["config"]["models_evaluated"]:
        assert nm in FAST_MODELS
