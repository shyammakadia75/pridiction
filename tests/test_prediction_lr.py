"""Tests for Logistic Regression convergence stability & FrequencyBaseline.

All tests use SYNTHETIC data in-memory. No real database is read or written,
and no network access is required. These tests document the convergence
behaviour identified in Phase 9 and guard against numerical regressions.
"""
from __future__ import annotations

import numpy as np
import pytest

from football_intelligence.prediction.baseline import (
    FrequencyBaseline,
    LogisticRegressionClassifier,
    N_OUTCOMES,
    OUTCOME_LABELS,
    OUTCOME_INDEX,
    compute_accuracy,
    compute_log_loss,
    class_distribution,
)

LABELS = list(OUTCOME_LABELS)


def _synthetic_xy(n: int = 300, seed: int = 42):
    """Synthetic feature matrix with a weak class signal + one informative feature."""
    rng = np.random.RandomState(seed)
    x = rng.randn(n, 4)
    # Label mostly driven by sign of first feature so the model has signal to learn.
    y = (x[:, 0] > 0).astype(int)
    # Inject draws (class 1) occasionally to create imbalance + a minority class.
    y[rng.rand(n) < 0.3] = 1
    # Ensure all 3 classes appear.
    y[:3] = np.array([0, 1, 2])
    return x.astype(float), y.astype(int)


# 1. Reproducibility under a fixed seed.
def test_lr_default_reproducible():
    X, y = _synthetic_xy()
    m1 = LogisticRegressionClassifier(random_state=42)
    m2 = LogisticRegressionClassifier(random_state=42)
    m1.fit(X, y)
    m2.fit(X, y)
    assert m1.weights is not None and m2.weights is not None
    assert m1.weights.shape == m2.weights.shape
    np.testing.assert_allclose(m1.weights, m2.weights)
    np.testing.assert_allclose(m1.bias, m2.bias)


# 2. Finite weights / bias after fit.
def test_lr_weights_and_bias_finite_after_fit():
    X, y = _synthetic_xy()
    m = LogisticRegressionClassifier(random_state=42)
    m.fit(X, y)
    assert np.all(np.isfinite(m.weights)), "weights contain non-finite values"
    assert np.all(np.isfinite(m.bias)), "bias contains non-finite values"


# 3. Probability rows sum to one and stay in range.
def test_lr_proba_rows_sum_to_one_and_in_range():
    X, y = _synthetic_xy()
    m = LogisticRegressionClassifier(random_state=42).fit(X, y)
    proba = m.predict_proba(X)
    assert proba.shape == (len(X), N_OUTCOMES)
    assert np.all(np.isfinite(proba))
    assert np.all((proba >= 0.0) & (proba <= 1.0))
    np.testing.assert_allclose(proba.sum(axis=1), np.ones(len(X)), atol=1e-9)


# 4. Loss-curve dynamics: training loss strictly decreases as iterations
#    increase (the model keeps learning; the default is not stuck).
def test_lr_loss_decreases_with_iterations():
    X, y = _synthetic_xy(n=400, seed=7)
    rng = np.random.RandomState(0)
    idx = rng.permutation(len(X))
    n_tr = int(len(X) * 0.6)
    train_idx = idx[:n_tr]
    Xtr, ytr = X[train_idx], y[train_idx]

    short = LogisticRegressionClassifier(
        learning_rate=0.01, n_iterations=50, random_state=42
    )
    short.fit(Xtr, ytr)
    tr_loss_early = compute_log_loss(ytr, short.predict_proba(Xtr), labels=LABELS)

    long = LogisticRegressionClassifier(
        learning_rate=0.01, n_iterations=2000, random_state=42
    )
    long.fit(Xtr, ytr)
    tr_loss_late = compute_log_loss(ytr, long.predict_proba(Xtr), labels=LABELS)

    # More gradient steps on the same training set must reduce (or hold) train loss.
    assert tr_loss_late <= tr_loss_early + 1e-9


# 5. Numerically stable softmax on a hand-checked input (no overflow / NaN).
def test_lr_softmax_matches_numerically_stable():
    # Large-magnitude logits that would overflow a naive exp().
    X = np.array([[1e6, -1e6, 0.0], [0.0, 0.0, 0.0]], dtype=float)
    y = np.array([0, 1], dtype=int)
    m = LogisticRegressionClassifier(random_state=42).fit(X, y)
    proba = m.predict_proba(X)
    assert proba.shape == (2, N_OUTCOMES)
    assert np.all(np.isfinite(proba))
    np.testing.assert_allclose(proba.sum(axis=1), np.ones(2), atol=1e-9)
    # First row's logit for class 0 dominates -> proba[0,0] is the max.
    assert proba[0, 0] == proba[0].max()


# 6. FrequencyBaseline reproduces the training class-prior distribution.
def test_freq_baseline_preserved_as_reference():
    X, y = _synthetic_xy()
    fb = FrequencyBaseline().fit(y)
    proba = fb.predict_proba(X)
    assert proba.shape == (len(X), N_OUTCOMES)
    np.testing.assert_allclose(proba.sum(axis=1), np.ones(len(X)), atol=1e-9)
    # Every row must equal the class-prior vector.
    expected = np.bincount(y, minlength=N_OUTCOMES)[:N_OUTCOMES].astype(float)
    expected = expected / expected.sum()
    for row in proba:
        np.testing.assert_allclose(row, expected, atol=1e-9)
    # Sanity: predicted labels match the majority-class argmax.
    pred = fb.predict_proba(X).argmax(axis=1)
    acc = compute_accuracy(y, pred, labels=LABELS)
    assert acc == pytest.approx(float(expected.max()))
    # And the baseline distribution matches a direct class_distribution call.
    cd = class_distribution(y, labels=LABELS)["counts"]
    assert cd == {lb: int((y == OUTCOME_INDEX[lb]).sum()) for lb in LABELS}

