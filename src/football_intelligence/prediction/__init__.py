"""Prediction subpackage: baseline models and evaluation for match prediction.

This package provides leakage-safe, interpretable baseline models for predicting
football match outcomes (home win / draw / away win).  It integrates with the
existing :mod:`features.engineering` pipeline and supports chronological
train/validation/test splitting to prevent temporal leakage.

Public API
----------
:class:`FrequencyBaseline`
    A no-skill baseline that predicts class-prior probabilities.

:class:`LogisticRegressionClassifier`
    A multinomial logistic-regression classifier implemented with numpy.

:data:`OUTCOME_LABELS`
    Canonical outcome labels: ``("home_win", "draw", "away_win")``.

:data:`OUTCOME_INDEX`
    Mapping from label to integer index (0, 1, 2).

:func:`compute_outcome_label`
    Extract the outcome label from a match row's goal counts.

:func:`chronological_split`
    Split a sorted DataFrame into train/val/test by chronological index.

:func:`compute_log_loss`
    Multiclass cross-entropy loss.

:func:`compute_brier_score`
    Multiclass Brier score (mean squared error of probabilities).

:func:`compute_accuracy`
    Classification accuracy.

:func:`build_confusion_matrix`
    Confusion matrix for 3-class outcomes.

:func:`class_distribution`
    Count and percentage of each outcome class.
"""
from __future__ import annotations

from .baseline import (
    FrequencyBaseline,
    LogisticRegressionClassifier,
    OUTCOME_LABELS,
    OUTCOME_INDEX,
    compute_outcome_label,
    chronological_split,
    compute_log_loss,
    compute_brier_score,
    compute_accuracy,
    build_confusion_matrix,
    class_distribution,
    evaluate_predictions,
    prepare_training_data,
)

__all__ = [
    "FrequencyBaseline",
    "LogisticRegressionClassifier",
    "OUTCOME_LABELS",
    "OUTCOME_INDEX",
    "compute_outcome_label",
    "chronological_split",
    "compute_log_loss",
    "compute_brier_score",
    "compute_accuracy",
    "build_confusion_matrix",
    "class_distribution",
    "evaluate_predictions",
    "prepare_training_data",
]
