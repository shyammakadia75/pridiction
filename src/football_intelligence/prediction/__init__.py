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


from .evaluation import (
    EvaluationReport,
    LEAKAGE_COLS,
    build_feature_dataset,
    classification_report,
    compute_brier_score,
    compute_calibration,
    compute_macro_f1,
    compute_precision_recall,
    feature_columns,
    run_prediction_evaluation,
    save_report,
)

from .elo import (
    EloModel,
    EloReport,
    run_elo_evaluation,
    save_elo_report,
)

from .poisson import (
    PoissonModel,
    PoissonFixturePrediction,
    PoissonReport,
    poisson_pmf,
    score_probability_grid,
    home_draw_away_probabilities,
    over_under_probability,
    both_teams_to_score_probability,
    run_poisson_chronological,
    run_poisson_evaluation,
    save_poisson_report,
    DEFAULT_MAX_GOALS,
    DEFAULT_MIN_HISTORY,
    DEFAULT_PRIOR_STRENGTH,
    DEFAULT_RATE_FLOOR,
)

from .service import (
    PHASE25_POOLED_METRICS,
    PROD_DB_PATH,
    SUPPORTED_MODELS,
    DEFAULT_MODEL,
    KNOWN_PROD_HASH,
    MatchPrediction,
    PredictionExplanation,
    PredictionRequest,
    build_explanation,
    build_pre_match_history,
    double_chance_probabilities,
    over_under_probabilities,
    predict_match,
    resolve_competition,
    resolve_fixture_date,
    resolve_team,
    team_lookup,
    validate_probability_vector,
)
from .comparison import (
    ComparisonReport,
    mcp_mcnemar_exact,
    paired_bootstrap,
    run_model_comparison,
    save_comparison_report,
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
    "EvaluationReport",
    "LEAKAGE_COLS",
    "build_feature_dataset",
    "classification_report",
    "compute_brier_score",
    "compute_calibration",
    "compute_macro_f1",
    "compute_precision_recall",
    "feature_columns",
    "run_prediction_evaluation",
    "save_report",
    "EloModel",
    "EloReport",
    "run_elo_evaluation",
    "save_elo_report",
    "PoissonModel",
    "PoissonFixturePrediction",
    "PoissonReport",
    "poisson_pmf",
    "score_probability_grid",
    "home_draw_away_probabilities",
    "over_under_probability",
    "both_teams_to_score_probability",
    "run_poisson_chronological",
    "run_poisson_evaluation",
    "save_poisson_report",
    "DEFAULT_MAX_GOALS",
    "DEFAULT_MIN_HISTORY",
    "DEFAULT_PRIOR_STRENGTH",
    "DEFAULT_RATE_FLOOR",
    "ComparisonReport",
    "mcp_mcnemar_exact",
    "paired_bootstrap",
    "run_model_comparison",
    "save_comparison_report",
    "predict_match",
    "PredictionRequest",
    "MatchPrediction",
    "PredictionExplanation",
    "build_explanation",
    "build_pre_match_history",
    "double_chance_probabilities",
    "over_under_probabilities",
    "validate_probability_vector",
    "team_lookup",
    "resolve_team",
    "resolve_competition",
    "resolve_fixture_date",
    "SUPPORTED_MODELS",
    "DEFAULT_MODEL",
    "KNOWN_PROD_HASH",
    "PROD_DB_PATH",
    "PHASE25_POOLED_METRICS",
]
