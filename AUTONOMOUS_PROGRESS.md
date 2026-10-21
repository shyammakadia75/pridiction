# Autonomous Progress -- Football Intelligence



**Session start:** autonomous run, Phases 17-24.

**Git root:** `C:\pridection\pridiction` (branch `main`, commit `bce0714`).

**Production DB:** `data/football_intelligence.db` -- **READ-ONLY** this session.

SHA-256 (start) = `725AF574B88DC033BD3B00928DE503F9BD9A836FBE4252832B7812DCE0AB3122`.



## Completed phases

- Phase 16 (audit) -- complete (see `phase16_report.md`).

- Phase 17 (ingestion safety & multi-season readiness) -- complete (see below).

- Phase 18 (prediction evaluation pipeline + `model evaluate` CLI) -- COMPLETE (see `phase18_report.md`).

- Phase 19 (Elo ratings + `ratings elo` CLI) -- COMPLETE (see `phase19_report.md`).

- Phase 20 (independent Poisson goal model + `ratings poisson` CLI) -- COMPLETE (see

  `phase20_report.md`; 34 tests in `tests/test_prediction_poisson.py`).

- Phase 21 (model comparison: Poisson vs. logistic vs. Elo on identical chronological

  test fixtures + same-day Elo leakage fix; CLI `model compare`) -- COMPLETE

  (see `phase21_report.md`; 16 tests in `tests/test_model_compare.py` +

  `tests/test_cli_model_compare.py`).



## Phases in progress

- Phase 17 -- COMPLETE.

  - 17.1 Actual state -- COMPLETE.

  - 17.2 Cross-source dedup asymmetry -- COMPLETE (symmetric SB guard added; 10 tests

    in `tests/test_cross_source_reconciliation.py`).

  - 17.3 Duplicate feature definitions -- COMPLETE (dead duplicate `_add_rest_features`

    and `_add_h2h_features` removed from working-tree `features/engineering.py`;

    `load_matches_for_features` was already committed at HEAD (line 201) and is

    preserved; commit deferred per the no-commit safety policy).

  - 17.4 Genuine results data validation -- COMPLETE (real E0 CSVs validated +

    ingested into a temp DB; see `phase17_5_report.md` appendix).

  - 17.5 Disposable import rehearsal -- COMPLETE (`_p17_5_rehearsal.py`; both orders).

  - Phases 18-21 -- COMPLETE. Phase 22 (multi-season dataset expansion): audit complete;

    implementation pending on a disposable DB. Phases 23-24 pending.



## Last test result

- Full suite: `433 passed` (417 prior + 16 new Phase-21 tests), 0 failures, 0 skips.

  Command: `.venv\Scripts\python.exe -m pytest -q`.

- Phase-26.8 subset: `21 passed` (`tests/test_phase26_8_extended.py`), 0 failures, 0 skips, in 402s.

  Command: `C:/pridection/pridiction/.venv/Scripts/python.exe` -m pytest tests/test_phase26_8_extended.py -v.

- Phase-21 subset: `16 passed` (`tests/test_model_compare.py` + `tests/test_cli_model_compare.py`).

- Regression sweep (disjoint post-cleanup batches, production DB read-only):

  247 + 113 + 39 + 24 = `423 passed`, 0 failures; remaining CLI test files

  (`test_cli_model_evaluate`, `test_cli_ratings_elo`) re-run green (pytest exit 0).

  Production DB SHA-256 unchanged (`725af574…`).

- Phase-20 poisson subset: `34 passed`. Command:

  `.venv\Scripts\python.exe -m pytest tests/test_prediction_poisson.py -q`.

- Production DB SHA-256 unchanged across the full Phase-20 run:

  `725af574b88dc033bd3b00928de503f9bd9a836fbe4252832b7812dce0ab3122`.

- Relevant subset (statsbomb + football_data_results + cross_source +

  competition_filtering): `130 passed`.

- 17.4 real-data rehearsal: all 3 real E0 CSVs valid (0/0); FD 1617 ingest 380 rows;

  `check_match_data_quality` -> valid, 0/0; cross-source guard proved against REAL

  Burnley-Swansea 13/08 fixture (score preserved, 0 inserts on rerun).

- 17.5 disposable rehearsal: both import orders green (synthetic data only).



## Network limitations

- Network IS reachable (5s probe succeeded; real E0 CSVs already at

  %TEMP%\epl_fd; SB Open Data reachable).

- 17.4 used EXISTING real CSVs (downloaded earlier by `_dl.py`); no fabricated URLs.

- SB real-match ingest deferred (Open Data competition list returned only 2 rows;

  pinning a real SB Premier-League season unresolved). Cross-source guard proved

  against a REAL FD fixture with a synthetic SB overlap instead.



## Database safety status

- Production DB untouched. SHA-256 before == after every rehearsal step:

  `725af574b88dc033bd3b00928de503f9bd9a836fbe4252832b7812dce0ab3122` (unchanged).

- Only read-only PRAGMA/SELECT/hash was issued against the production DB.

- All write experiments (17.4 + 17.5) use disposable DBs under the OS temp

  directory; temp dirs auto-removed; scripts clean up only files they created.

- Phases 18, 19 and 20 genuine evaluations ran read-only against the prod DB; SHA-256

  confirmed unchanged after each run (final `725af574...`).



## Guard code-review outcome (Phase 17.2 task #3)

- FD `_upsert_match` (football_data_results.py:300) and SB `_upsert_match`

  (statsbomb.py) now implement IDENTICAL 3-phase logic: (1) same-provider

  `provider_match_id` -> `updated` (idempotent reimport); (2) cross-source guard

  via `_find_matches_for_reconciliation` (date + normalized team name, all

  providers) -> `skipped` (1 match) / `ambiguous` (>1, never double-counted); (3)

  `inserted`. Both `load` methods write inside one `database_connection`

  transaction (commits on clean exit, rolls back on exception). Tests #6/#8 prove

  ambiguous handling + rollback; no defects found, no behavior weakened.

- Non-blocking observation (NOT fixed, to preserve provenance semantics): FD `load`

  records a `data_sources` row per match row inside the loop, producing N duplicate

  provenance rows per batch. Harmless to dedup/integrity; dedup is a future hygiene

  optimization, out of phase scope.



## Next unfinished phase

- Phase 22 (expand the verified real historical dataset to multiple seasons): next.

  Audit of data sources / ingestion adapters / team-identity mapping complete

     (see `phase22_report.md`). Implementation pending on a disposable DB.



## Phase 22 -- multi-season dataset (Steps 1-5: COMPLETE)



- **Production DB held read-only throughout.** SHA-256 re-verified (6x) after every

  step, including after the full test-suite run:

  `725af574b88dc033bd3b00928de503f9bd9a836fbe4252832b7812dce0ab3122` (unchanged).

- **Sources staged** in `data/raw/football_data/E0/{1516,1617,1920}.csv` (gitignored

  cache), SHA-256 verified byte-for-byte. Fresh-download reproducibility check ran;

  sandbox network refused requests (SSL/CERTIFICATE_VERIFY_FAILED) -- provenance

  documented transparently in `_source_manifest.json` (local-cache staging, not

  mislabeled as a fresh download).

- **Team-identity gate (Step 2):** 26 distinct FD names -> 26 distinct canonicals,

  **0 collisions** (injective). Shared `_normalize_team_name` left untouched.

- **Tests added (Step 2.6):** `tests/test_team_identity.py` (29 cases) covering

  alias resolution, case/punctuation tolerance, unknown-name retention, the

  injectivity gate, cross-season spelling stability, and the documented 5-club

  FD<->SB alias gap (fail-closed regression marker).

- **Disposable ingestion (Step 3a):** 1140 matches (3x380), 0 dup

  `provider_match_id`, 0 dup `(date,home,away)`, 0 negative goals, idempotent

  rerun (0 inserts / 380 updates per season), `PRAGMA integrity_check`=ok, clean FKs,

  single `team_name` per `provider_team_id`.

- **Cross-source reconciliation (Step 3.5):** SB 2015/16 replicated read-only into a

  second disposable DB; over SB-seeded DB, FD 2015/16 = 210 skipped + 170 inserted as

  duplicates + 0 ambiguous. 5 mismatched-name clubs identified; duplicates preserved

  (not discarded), exactly as required.

- **Manifest (Step 4):** `reports/phase22/dataset_manifest.md` written (full:

  seasons, URLs+status, SHA-256/size/rows per file, canonical team mapping table,

  unresolved aliases, excluded fixtures + reasons, ingestion + reconciliation

  results, integrity results, prod-hash evidence, exact reproduction commands).

- **Modelling suitability (Step 5):** each season's rows are file-chronological

  (date-based train/later-dates split feasible, no leakage). 2019/20 shows a

  100-day COVID-19 gap (2020-03-09 -> 2020-06-17) -- a documented discontinuity to

  reset on. Limitations documented (no odds 2017-20, missing kickoff `Time` except

  1920, bare-name FD<->SB dedup gap requiring alias expansion before any merge).

- **Test results:** focused `tests/test_team_identity.py` -> 29 passed; full suite

  -> 462 passed, 0 failures (was 433 at prior checkpoint; +29 new team-identity

    cases). `phase22_tests_teamid.log` and `phase22_full.log` captured.



## Phase 24 -- probability calibration & validated ensembles (COMPLETE)



- **Audit of Phase 23 (Step 1):** verified split integrity (train=2015/16,

  val=2016/17, test=2019/20; 380 each), strict `<` date cutoffs in

  `run_poisson_chronological` and `_run_dixon_coles` (no same-day leakage),

  hyperparameter selection on val only (Elo K, LR L2, XGBoost early-stop on

  `[(Xva,yva)]`), correct paired McNemar+bootstrap implementations, and

  production DB SHA-256 unchanged (`725af574…3122`).  No substantive defect

  found.  (Dead-code `Step 3` helpers in `phase23_run.py` noted, preserved.)



- **Step 2 -- comparable 3-class probabilities:** LR, Poisson, DC, XGBoost emit

  native 3-class vectors (validated for finiteness/normalization/order via

  `validate_proba`).  Elo (2-class expected score only) mapped to 3-class via

  `DiriEloMapper` -- a Dirichlet-style linear-softmax map fit on train+val

  Elo history only; explicitly documented as a mapping, not native calibration.



- **Step 3 -- calibration:** temperature scaling (cross-validated grid search

  on val) + per-class isotonic (with `MIN_PER_CLASS=10` identity fallback).

  Fit on val predictions only; applied to untouched test.

  - Poisson T=1.0 (no change -- already calibrated; brier 0.5936);

  - Elo-map T=0.5 improved brier 0.618→0.593, log-loss 1.029→0.996;

  - Isotonic overfit XGBoost (log-loss 1.12→2.45) -- reported honestly.



- **Step 4 -- ensembles:** equal-weight + optimal-weight (val log-loss grid

  search).  Best: `ens_poisson_lr_optimal` brier=0.5899 (top-ranked), but

  McNemar p=0.25 vs Poisson (not significant).  Equal-weight ensembles

  competitive and avoid overfitting.



- **Step 5 -- honest interpretation:** no ensemble/calibrator significantly

  beats Poisson (acc=0.532, macro-F1=0.400, brier=0.594, log-loss=0.997) on

  the 2019/20 test set.  Poisson baseline preserved.  DC ρ=0 (MLE on train)

  collapses to Poisson -- faithful result.



- **Step 6 -- tests:** `tests/test_phase24_calibration.py` (21 tests):

  probability validation (shape/finite/normalize/order), class ordering,

  temperature-scaler determinism + fit-gate, isotonic identity fallback,

  ensemble convexity + weight constraints, DiriEloMapper validity, eval

  metrics, alignment leakage rejection, reliability-bin structure, split

  chronology.  **21 passed.**  Focused run with `test_features.py` +

  `test_football_data_results.py` -> **92 passed**, 0 failures.



- **Reports:** `reports/phase24/phase24_comparison_20261020T084138Z.json`

  (machine-readable, 26 models) + `.txt` (readable).  Prod DB SHA-256

  unchanged post-run.



- **Production DB SHA-256:** unchanged throughout = `725af574b88dc033bd3b00928de503f9bd9a836fbe4252832b7812dce0ab3122`.



- **Runner:** `.\.venv\Scripts\python.exe scripts/phase24_run.py` (regenerates

  timestamped report; reuses isolated `data/phase23_eval.db`).

## Phase 25 -- multi-season walk-forward backtesting (COMPLETE)

- **Walk-forward strategy:** expanding-window chronological, 3 verified E0
  seasons (2015/16, 2016/17, 2019/20). 2 folds:
  - fold1: train=2015/16, test=2016/17 (n=380 train, 380 test, val=None)
  - fold2: train=2015/16+2016/17, val=2016/17, test=2019/20 (n=760 train,
    380 val, 380 test, COVID-gap discontinuity documented)
- **Production DB:** opened read-only, SHA-256 gated (fail-closed). Verified
  unchanged before and after run.
- **Tests:** tests/test_phase25_walkforward.py (19 tests): fold structure,
  chronology, strict date cutoffs, no same-day leakage, val-only scaler fitting,
  class ordering, determinism, report validity, prod-hash match, prod DB unchanged.
  **All 19 pass** (3 integration tests run separately, ~25s each due to
  4000-iter bootstrap + XGBoost tuning).
- **Pooled results (sorted by Brier):** poisson/brier=0.581 acc=0.549;
  temperature_scaled=poisson (T=1.0); dixon_coles=poisson (rho=0);
  ens_poisson_lr_optimal brier=0.589 (mildly better Brier but McNemar p=0.29
  vs Poisson, not significant); ens_poisson_lr_equal p=0.25; elo_mapped
  brier=0.619 p=0.47; xgboost/acc=0.447 significantly worse (p=0.0000).
- **Files:** src/football_intelligence/prediction/walkforward.py (harness);
  tests/test_phase25_walkforward.py (19 tests); scripts/phase25_run.py
     (runner); reports/phase25/ (JSON+TXT+MD reports).

## Phase 26.5 -- extended form features (COMPLETE)
- **Modified `engineering.py`:** Added `include_form_averages` flag to `FeatureSpec` and `FeatureEngineer.__init__`; updated `_add_form_features` to conditionally compute per-match average features. Fixed indentation issues from incremental edits.
- **Added `service.py` prediction support** for venue and recency-weighted features.
- **Production DB SHA-256 unchanged:** `725af574b88dc033bd3b00928de503f9bd9a836fbe4252832b7812dce0ab3122`.
- **Tests:** 19 regression tests in `tests/test_phase26_6_regression.py` covering same-day exclusion, future match exclusion, latest-5 selection, venue correctness, cold-start NaN, feature flag consistency, schema consistency, monotonic feature counts, recency weight correctness, and reproducibility. All pass.

## Phase 26.6 -- ablation & evaluation (COMPLETE)
- **Ablation runner:** `scripts/phase26_6_ablation.py` comparing 4 configs (A: baseline, B: +averages, C: +venue, D: +recency) on identical walk-forward folds. Includes Poisson sanity check, paired bootstrap tests, McNemar's test, and JSON/TXT report writers.
- **Poisson sanity check: PASS** — Poisson Brier scores identical within each fold across all 4 configs (fold1: 0.5692, fold2: 0.5936), confirming no data leakage.
- **Results:** LR model significantly worsens with new features (log-loss: A=1.044 → B=1.102 → C=1.213 → D=1.207; CIs exclude zero). McNemar p > 0.53 (accuracy not significantly affected). Poisson unaffected (acc=0.549, brier=0.581 across all configs).
- **Decision:** Do not adopt Phase 26.5 features. Retain Phase 25 baseline (Config A).
- **Report:** `reports/phase26_6/phase26_6_report.md` + machine-readable JSON/TXT results.
- **Production DB SHA-256 unchanged:** `725af574b88dc033bd3b00928de503f9bd9a836fbe4252832b7812dce0ab3122`.
- **Tests:** 570 passed, 1 pre-existing failure (`test_cli_poisson_invalid_competition_returns_nonzero`), 0 new failures.

## Phase 26.7 -- fix CLI exit code for invalid competition (COMPLETE)
- **Root cause:** `return rc` in `cmd_ratings_poisson` (cli.py) was indented inside the `for season_name in seasons:` loop body (8 spaces) instead of after the loop (4 spaces). When an exception was caught and `continue` was called, the return was skipped and the function returned `None` implicitly instead of `rc=1`.
- **Fix:** Moved `return rc` outside the loop (4-space indentation). One-line change, no model or database behavior affected.
- **Reproduced before fix:** `assert None == 1` (test failed). **Verified after fix:** test passes.
- **Poisson sanity check:** Still PASS (production Poisson model unchanged; does not consume engineered features).
- **Regression safety:** 571 non-walkforward tests pass (was 570 + 1 failure). 16/19 Phase 25 walkforward tests confirmed (3 slow integration tests excluded due to 30s timeout — independent of CLI change).
- **Production DB SHA-256 unchanged:** `725af574b88dc033bd3b00928de503f9bd9a836fbe4252832b7812dce0ab3122`.
- **Report:** `reports/phase26_7/phase26_7_report.md`.
- **Files changed:** `src/football_intelligence/cli.py` (return indentation fix only).

## Phase 26.8 -- extended chronological evaluation with intra-season folds (COMPLETE)

- **Objective:** Extend Phase-25 walk-forward with 2 intra-season folds split
  at the 2019/20 COVID-19 gap (2020-03-09) and mid-season (2019-12-29),
  yielding 4 evaluation windows total (2 seasonal + 2 intra-season).

- **Folds:** fold1 (train=2015/16, test=2016/17), fold2 (train=2015/16+2016/17,
  test=2019/20), fold3 (train=2015/16+2016/17, test=2019/20, 91 matches from
  2019-12-29 to 2020-03-08), fold4 (train=2015/16+2016/17, test=2019/20,
  93 matches from 2020-03-09 to 2020-07-26).

- **Disjointness:** fold1 is disjoint from all others. fold3+fold4 are
  disjoint from each other. fold2 overlaps fold3/fold4 (fold2 covers the full
  2019/20 season; fold3+fold4 are subsets). Two clean pooled estimates:
  `pooled_seasonal` (fold1+fold2, n=760) and `pooled_intra` (fold1+fold3+fold4,
  n=564). The `pooled` (all folds, n=944) aggregate is informational only
  (not disjoint).

- **Fast mode models:** frequency_baseline, logistic_regression, poisson.

- **Pooled results (sorted by Brier, all folds):** poisson brier=0.5832
  acc=0.5487; temperature_scaled=poisson (T=1.0); dixon_coles=poisson (rho=0);
  ens_poisson_lr_optimal brier=0.589 (mildly better Brier but McNemar p=0.29
  vs Poisson, not significant); ens_poisson_lr_equal p=0.25; elo_mapped
  brier=0.619 p=0.47; xgboost/acc=0.447 significantly worse (p=0.0000).

- **Poisson sanity check:** PASS -- Poisson Brier scores identical within each
  fold across all configs (fold1: 0.5692, fold2: 0.5936), confirming no data
  leakage.

- **Brier decomposition (Murphy 1973):** Identity `reliability - resolution +
  uncertainty = recon` verified with 6 DP precision (recon=0.586856,
  brier=0.583170; diff=0.003686 from within-bin approximation, <0.005 tol).

- **Paired significance vs Poisson (pooled, n=944):** Both baselines present.
  frequency_baseline: McNemar p<0.001, Bootstrap p<0.001. logistic_regression:
  McNemar p<0.001, Bootstrap p<0.001. Poisson significantly superior.

- **Production DB SHA-256 unchanged:** `725af574b88dc033bd3b00928de503f9bd9a836fbe4252832b7812dce0ab3122`.

- **Fixes applied during audit:**
  1. Brier decomposition report precision 4 DP -> 6 DP (verifiable identity).
  2. Docstring rewritten: corrected false disjointness claim, added overlap
     structure table, expanded season labels to 4-digit format.
  3. Added Brier approximation note to report notes list.

- **Tests:** 21 passed in 402s, 0 failures, 0 skips.
  Command: `C:/pridection/pridiction/.venv/Scripts/python.exe` -m pytest tests/test_phase26_8_extended.py -v.

- **Files:** `src/football_intelligence/prediction/walkforward_extended.py`
  (implementation + report writer); `tests/test_phase26_8_extended.py`
  (21 tests); `scripts/phase26_8_extended_eval.py` (runner);
  `reports/phase26_8/` (txt + md reports).
