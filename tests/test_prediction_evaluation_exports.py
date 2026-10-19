"""Phase-18 test: the prediction package re-exports the evaluation API."""
from football_intelligence.prediction import evaluation as ev
from football_intelligence.prediction import (
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


def test_package_reexports_evaluation_api():
    # Same objects as the implementation module -> no shadowing/typo.
    assert feature_columns is ev.feature_columns
    assert classification_report is ev.classification_report
    assert run_prediction_evaluation is ev.run_prediction_evaluation
    assert save_report is ev.save_report
    assert EvaluationReport is ev.EvaluationReport
    assert build_feature_dataset is ev.build_feature_dataset
    assert compute_macro_f1 is ev.compute_macro_f1
    assert compute_precision_recall is ev.compute_precision_recall
    assert compute_calibration is ev.compute_calibration
    assert compute_brier_score is ev.compute_brier_score
    assert LEAKAGE_COLS is ev.LEAKAGE_COLS
