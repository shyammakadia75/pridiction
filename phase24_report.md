# Phase 24 — Probability Calibration and Validated Ensembles

## Status: COMPLETE

## 1. Audit of Phase 23 (Step 1)

Phase 23's evaluation harness (`scripts/phase23_run.py`) and its reported results
were verified against the actual repository, source code, and persisted reports.
**No substantive leakage or evaluation defect was found.**

### Verified claims

| Claim | Audit result |
|-------|-------------|
| Train = 2015/16 (380), Val = 2016/17 (380), Test = 2019/20 (380) | Confirmed in `phase23_comparison_20261019T191730Z.json`; dates 2015-08-08→2016-05-17, 2016-08-13→2017-05-21, 2019-08-09→2020-07-26 |
| Test (380) separate from train/val | No date overlap; chronology assertion passes |
| Feature-history cutoffs strictly pre-match | `run_poisson_chronological` and `_run_dixon_coles` use `df[date < cutoff]` (strict `<`, same-day-excluded) |
| Hyperparameters on validation only | Elo K on val; LR L2 on val; XGBoost on val; final models refit on train+val |
| XGBoost early stopping uses only val | `eval_set=[(Xva, yva)]` — test never in eval_set |
| Paired McNemar / bootstrap correct | Exact binomial + percentile bootstrap (4000 iters); correct keys present |
| Production DB not modified | SHA-256 `725af574…3122` before == after |
| Poisson & DC identical at acc=0.532 | DC ρ→0 (MLE on train) collapses to Poisson; McNemar p=1.0 — legitimate |

### Minor code-health observation (not a defect)

The `Step 3` helper functions in `phase23_run.py` (lines ~246–440) are **dead
code** — `run()` re-implements their logic inline (lines 555–600). The executed
code is correct; dead functions preserved per the no-delete safety policy.

## 2. Three-Class Probability Provision

All models emit valid `[P(home_win), P(draw), P(away_win)]` vectors, validated
by `validate_proba` (shape, finiteness, normalization, non-negativity, order):

| Model | Native 3-class probs? | Method |
|-------|----------------------|--------|
| Logistic Regression | Yes | softmax |
| LR + interactions | Yes | softmax |
| Poisson | Yes | score-grid sum |
| Dixon-Coles | Yes | ρ-adjusted grid |
| XGBoost | Yes | multi:softprob |
| **Elo** | No native (2-class) | Mapped via `DiriEloMapper` (linear-softmax fit on train+val) |

Elo is **not** treated as native calibrated 3-class probability. The mapping is
documented and uses only train+val outcomes.

## 3. Calibration Results

Calibrators fitted on **validation predictions only** (380 val fixtures),
applied to test (380). All use a cross-validated log-loss grid search
(temperature) or per-class isotonic regression.

| Method | Poisson T | LR T | LR-int T | DC T | XGB T | Elo-map T |
|--------|-----------|------|----------|------|-------|-----------|
| Temperature | 1.0 | 2.0 | 1.0 | 1.0 | 0.75 | 0.5 |

### Key observations
- **Poisson T=1.0**: Poisson is already well-calibrated — temperature scaling
  makes no change (brier 0.5936 → 0.5936). Expected, honest result.
- **Elo-map T=0.5** improved calibration: brier 0.6175 → 0.5930, log-loss
  1.0288 → 0.9958. Calibration **helped** Elo's mapped probabilities.
- **Isotonic regression** overfit on the 380-sample val set: it degraded
  XGBoost catastrophically (log-loss 1.12 → 2.45) and slightly others. The
  `MIN_PER_CLASS=10` guard only triggers identity fallback when a class has
  too few positives; with 380 samples all classes cleared the threshold, so
  isotonic ran and overfit. **Reported honestly, not hidden.**

## 4. Ensemble Results

### Equal-weight ensembles (baseline)
| Ensemble | Acc | Brier | Log-loss |
|----------|-----|-------|---------|
| ens_poisson_lr_equal | 0.508 | 0.6004 | 1.0088 |
| ens_poisson_xgb_equal | 0.521 | 0.6035 | 1.0204 |
| ens_poisson_li_xgb_equal | 0.526 | 0.5985 | 1.0108 |

### Optimized-weight ensembles (weights fit on val log-loss only)
| Ensemble | Acc | Brier | Log-loss | Weights | vs Poisson |
|----------|-----|-------|---------|---------|-----------|
| **ens_poisson_lr_optimal** | 0.524 | **0.5899** | **0.9925** | poisson≈0.55, lr≈0.45 | McNemar p=0.25 (not sig.) |
| ens_poisson_xgb_optimal | 0.445 | 0.6580 | 1.1178 | (overfit to val) | |
| ens_poisson_li_xgb_optimal | 0.445 | 0.6580 | 1.1178 | (overfit) | |
| ens_calibrated_optimal | 0.445 | 0.6910 | 1.2027 | (overfit) | |

The best ensemble (`ens_poisson_lr_optimal`, brier=0.5899) is the top-ranked
model overall, but is **not significantly different** from Poisson
(McNemar p=0.25, bootstrap CI crosses 0). Equal-weight ensembles are competitive
and avoid overfitting weight selection.

## 5. Paired Significance vs Poisson (log-loss, 4000-iter bootstrap)

| Comparison | McNemar p | Boot diff | 95% CI | Sig? |
|-----------|-----------|-----------|--------|------|
| Poisson (self) | 1.0000 | +0.0000 | [0, 0] | reference |
| logistic_regression | 0.0007 | +0.1198 | [0.057, 0.184] | Yes (worse) |
| xgboost | 0.0005 | +0.1205 | [0.065, 0.175] | Yes (worse) |
| elo_mapped | 0.2800 | +0.0314 | [-0.007, 0.069] | No |
| logistic_interactions | 0.2713 | +0.0631 | [0.008, 0.120] | No |
| ens_poisson_lr_equal | 0.3135 | +0.0114 | [-0.019, 0.042] | No |
| **ens_poisson_lr_optimal** | **0.2500** | **-0.0049** | [-0.012, 0.002] | **No** |
| ens_poisson_xgb_equal | 0.5966 | +0.0231 | [-0.001, 0.047] | No |
| ens_calibrated_equal | 0.0725 | +0.0167 | [-0.012, 0.046] | No |

**Conclusion:** No ensemble or calibrator significantly beats Poisson on this
test set. Poisson remains the pragmatic, defensible baseline. We do not claim
victory where none exists.

## 6. Safety Verification

- **Production DB SHA-256**: `725af574b88dc033bd3b00928de503f9bd9a836fbe4252832b7812dce0ab3122`
  - Before Phase-24: matches (gated).
  - After Phase-24: **unchanged** (`production_db_unchanged: True`).
- **Data source**: Phase 24 reuses the isolated Phase-23 disposable DB
  (`data/phase23_eval.db`) — the production DB is opened **read-only** and
  gated (fail-closed `assert`).
- **Calibrator fitting**: fit on val predictions only (380 samples); asserts
  `fitted_` before `predict_proba`; no test outcomes used.
- **Ensemble weights**: selected via grid search minimizing **val** log-loss;
  test set never participates in weight selection.
- **Elo mapping**: `DiriEloMapper.fit` on train+val Elo history only; test
  fixtures' `expected_home` computed from online Elo (strictly pre-match ratings).
- **Reproducibility**: seeded (seed=42, BOOT_SEED=42); repeat runs identical.

## 7. Files Produced

- `src/football_intelligence/prediction/calibration.py` — Phase 24 module
- `tests/test_phase24_calibration.py` — 21 regression tests (all pass)
- `reports/phase24/phase24_comparison_20261020T084138Z.json` — machine-readable
- `reports/phase24/phase24_comparison_20261020T084138Z.txt` — human-readable
- `scripts/phase24_run.py` — standalone runner
- `phase24_report.md` — this file

## 8. Honest Limitations

1. **Val sample size (380)**: isotonic regression overfit; temperature scaling
   is safer on small samples. The isotonic degradation is reported transparently.
2. **DC ρ=0**: Dixon-Coles MLE on train collapses to independent Poisson. This
   is the faithful statistical result, not a bug. To get genuine DC behavior
   one would need a richer model or more data.
3. **Elo→3-class mapping is a modeling choice**: the Dirichlet-style linear
   map is defensible but not the only option. Documented as a mapping, not
   native calibration.
4. **No significant gain**: no ensemble or calibrator significantly beats
   Poisson. We do not claim victory where none exists.

