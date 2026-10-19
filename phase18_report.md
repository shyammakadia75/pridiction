# Phase 18 — Model Training & Evaluation Pipeline

## 1. Status
- Start state: Phase 17 COMPLETE. Prod DB `data/football_intelligence.db`
  SHA-256 `725af574b88dc033bd3b00928de503f9bd9a836fbe4252832b7812dce0ab3122`,
  `integrity_check=ok`, 381 rows = 380 StatsBomb Premier League **2015/16** +
  1 lone StatsBomb **Champions League 2018/19** fixture.
- Pre-existing working-tree edits preserved (NOT reverted): `cli.py`, `odds.py`,
  `statsbomb.py`, `engineering.py`. No commits in this phase.

## 2. Dataset & split (genuine)
- **Primary:** prod DB, **read-only** (SELECT-only loaders + SHA-256 before/after
  guard). Filter `provider="statsbomb"`, `provider_competition_id="2"` (EPL),
  `season_name="2015/2016"` → **380 matches, single competition, all labelled.**
  The lone CL fixture (`provider_competition_id="16"`) is excluded → no
  "mixed-CL-final-into-EPL" hazard.
- Single chronologically-ordered feature frame (each match sees only strictly
  earlier results via `_filter_history`), then `chronological_split` 60/20/20 →
  **train 228 / val 76 / test 76**, no shuffling. Scalers & priors fit on train
  only; `LogisticRegressionClassifier` stores its `_impute_and_scale` state on
  train and re-applies on val/test.
- **Supplementary (genuine multi-season):** real Football-Data.co.uk E0 CSVs
  `E0_1516/E0_1617/E0_1920` cached at `%TEMP%\epl_fd/` → ingest into a TEMP DB
  via the existing `FootballDataResultsIngestor` (offline, no prod writes).
- **Limitation (honest, single-season):** prod holds only one EPL season; the
  multi-season run is attempted on cached CSVs in a temp DB, never on prod.

## 3. Reuse audit (no re-write)
`baseline.py` already provides: `OUTCOME_LABELS/INDEX` (0 home,1 draw,2 away —
**fixed**), `compute_outcome_label`, `chronological_split` (no shuffle),
`prepare_training_data` (computes leakage-safe features, attaches
`outcome_label`+`_label_idx`, excludes leakage cols), `FrequencyBaseline`,
`LogisticRegressionClassifier` (train-only `_impute_and_scale`), `compute_log_loss`,
`compute_brier_score` (multiclass = mean Σ_c(p−onehot)²), `compute_accuracy`,
`build_confusion_matrix`, `class_distribution`, `evaluate_predictions`.
`engineering.py`: `load_matches_for_features` (competition+season filter at
load time) + `FeatureEngineer` (`compute_all_features_from_db`).

## 4. Genuinely missing → ADD (numpy/pandas only, no sklearn)
- `compute_macro_f1`, `compute_precision_recall` (per-class), `compute_calibration`
  (per-class reliability bins), `classification_report` (extends
  `evaluate_predictions`).
- `build_feature_dataset`, `run_prediction_evaluation`, `save_report`,
  `EvaluationReport`.
- CLI: `model evaluate`.
- Tests: split integrity / no-leakage / train-only preprocessing / label &
  proba ordering / metric edge cases / empty & insufficient data / competition &
  season filtering (EPL-only, CL excluded) / reproducibility / read-only safety
  on a disposable DB.

## 5. Safety
Prod DB read-only (hash before==after). All writes (supplementary CSV eval) go to
`tmp_path`/TEMP DBs. CLI writes artifacts only to explicit `--output-dir`
(`reports/phase18/`). Exit nonzero on failure.

## 6. Implementation order
1. `src/football_intelligence/prediction/evaluation.py` (new)
2. `prediction/__init__.py` exports
3. CLI `model evaluate` (cli.py)
4. `tests/test_prediction_evaluation.py`
5. Run real read-only eval → `reports/phase18/` + fill §7–§10

## 7. Assumptions
- LR hyperparams fixed in advance (no test-set tuning). FrequencyBaseline has none.
- No "superior model" claims — report actual metrics + honest limits.


## 11. Completion (implemented & validated)
- [x] `src/football_intelligence/prediction/evaluation.py` created (~505 lines, pure numpy/pandas).
- [x] Metrics: `compute_macro_f1`, `compute_precision_recall` (per-class P/R/F1/support),
      `compute_calibration` (per-class reliability bins), `classification_report`.
- [x] Pipeline: `build_feature_dataset`, `run_prediction_evaluation`, `save_report`,
      `EvaluationReport`.
- [x] `prediction/__init__.py` now re-exports the evaluation API.
- [x] Read-only guard: SHA-256 captured before/after; raises `RuntimeError` on any DB change.
      Verified `production_db_unchanged is True` on a disposable 2-season DB.
- [x] Train-only preprocessing: FrequencyBaseline priors and LR `_impute_and_scale` state are
      fit on the train partition only (asserted against `_impute_and_scale(X_train)`).
- [x] Leakage containment: raw `home_goals`/`away_goals`/`match_date`/`competition_id`/
      `season_name`/`provider_match_id` are NOT in `feature_columns`;
      `load_matches_for_features` scopes a single `provider_competition_id`.
- [x] Class order fixed 0=home_win,1=draw,2=away_win; chronological (no-shuffle) split.
- [x] Artifacts: JSON + readable TXT written under an explicit `output_dir`
      (default `reports/phase18`); the production DB is never written by the runner.
- [x] Tests: `tests/test_prediction_evaluation.py` + `tests/test_prediction_evaluation_pipeline.py`.
- [x] Validation: `py_compile` OK; venv pytest `tests/` -> 371 passed at Phase-18
      completion (20 new: 14 evaluation/pipeline/exports + 6 CLI), no regressions
      in `test_prediction.py`/`test_prediction_lr.py`/`test_competition_filtering.py`.
- [x] CLI `model evaluate` (Task A): argparse subcommand in `cli.py`, thin wrapper
      over `run_prediction_evaluation` + `save_report`; 6 tests in
      `test_cli_model_evaluate.py`.
- [x] Genuine eval (Task B): read-only run on prod EPL 2015/2016 (380 matches,
      228/76/76 split, 48 features); prod SHA `725af574...` verified unchanged
      before/after. Artifacts: `reports/phase18/phase18_evaluation_2_2015-2016_*.{json,txt}`.
- [x] Phase 19 (Elo) added `prediction/elo.py` + `ratings elo` CLI; genuine prod eval
      (380 matches) wrote `reports/phase19`. Full suite now 383 passed (32 new across P18+P19).

Honest limitations carried over from section 7:
- Prod holds a single EPL season; multi-season coverage in tests uses synthetic/temp DBs only.
- LR hyper-parameters fixed in advance (no hyperparameter tuning or model selection).
- Metrics are descriptive; no "best model" claims.

## 12. Follow-on phases
- Phase 19 (Elo ratings) — COMPLETE (see `phase19_report.md`): chronological Elo
  with a 3-class fixed-threshold prediction; genuine prod eval (380 matches) wrote
  `reports/phase19`. Read-only; prod SHA `725af574...` unchanged before/after.
- Phase 20 (Poisson goal modelling) — next candidate; not yet scoped. Reuses the
  Phase-18 match loader + read-only harness. No writes to the production DB.
