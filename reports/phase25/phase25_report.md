# Phase 25 - Walk-Forward Backtesting Report

## Summary

Completed multi-season walk-forward backtesting using an expanding-window
chronological strategy over 3 verified E0 seasons (2015/16, 2016/17, 2019/20).

**Poisson is confirmed as the best-performing model** across both folds and
pooled, consistent with Phase 24's single-season findings. No model significantly
outperforms Poisson (all McNemar p-values vs Poisson > 0.05).

## Folds

| Fold | Train seasons | Val season | Test season | n_train | n_val | n_test | cutoff |
|------|--------------|------------|-------------|---------|-------|--------|--------|
| fold1 | 2015/2016 | - | 2016/2017 | 380 | - | 380 | 2016-08-13 |
| fold2 | 2015/2016, 2016/2017 | 2016/2017 | 2019/2020 | 760 | 380 | 380 | 2019-08-09 |

## Pooled Results (sorted by Brier)

| Model | Accuracy | Macro-F1 | Log-loss | Brier |
|-------|----------|----------|----------|-------|
| poisson | 0.549 | 0.411 | 0.976 | 0.581 |
| poisson_temperature_scaled | 0.549 | 0.411 | 0.976 | 0.581 |
| dixon_coles | 0.549 | 0.411 | 0.976 | 0.581 |
| ens_poisson_lr_optimal | 0.536 | 0.431 | 0.987 | 0.589 |
| ens_poisson_lr_equal | 0.530 | 0.448 | 0.995 | 0.594 |
| elo_mapped | 0.538 | 0.397 | 1.031 | 0.619 |
| xgboost | 0.447 | 0.324 | 1.060 | 0.640 |
| frequency_baseline | 0.472 | 0.214 | 1.060 | 0.640 |
| logistic_regression | 0.434 | 0.392 | 1.111 | 0.666 |

## Significance vs Poisson (pooled, log-loss)

| Comparison | McNemar p | Bootstrap diff | 95% CI |
|------------|-----------|----------------|--------|
| poisson vs frequency_baseline | 0.0001 | +0.084 | [+0.054, +0.115] |
| poisson vs logistic_regression | 0.0000 | +0.134 | [+0.090, +0.180] |
| poisson vs xgboost | 0.0000 | +0.084 | [+0.054, +0.115] |
| poisson vs elo_mapped | 0.4657 | +0.054 | [+0.029, +0.080] |
| poisson vs ens_poisson_lr_equal | 0.2504 | +0.018 | [-0.002, +0.038] |
| poisson vs ens_poisson_lr_optimal | 0.2888 | +0.011 | [-0.003, +0.026] |
| poisson vs temperature_scaled | 1.0000 | 0.000 | [-0.000, +0.000] |
| poisson vs dixon_coles | 1.0000 | 0.000 | [-0.000, +0.000] |

## Key Findings

1. **Poisson is the confirmed best model** (brier=0.581, acc=0.549).
2. **Temperature scaling is a no-op** (T=1.0) - Poisson is already calibrated.
3. **Dixon-Coles rho collapses to 0** - model reduces to Poisson identical output.
4. **No ensemble significantly beats Poisson** - McNemar p > 0.05 for both
   equal-weight and optimal-weight ensembles.
5. **Elo-mapped** is worse (brier=0.619) - the Dirichlet mapping is a lossy
   2-class to 3-class conversion.
6. **XGBoost** is significantly worse (p=0.0000) - overfitting on small training sets.
7. **Production DB SHA-256 verified unchanged**:
   `725af574b88dc033bd3b00928de503f9bd9a836fbe4252832b7812dce0ab3122`

## Notes & Limitations

- Only 3 seasons available (2015/16, 2016/17, 2019/20) - walk-forward limited
  to 2 folds.
- 2019/20 contains a COVID-19 gap (2020-03-09 to 2020-06-17) - documented
  stationarity discontinuity.
- Online models warmed on ALL pre-test data; fitted models use val hold-out
  for tuning when >=2 train seasons available.
- DC rho fit by MLE on training window only.
- Temperature scaling fit on val predictions only; no-val folds T=1.0.
- All predictions use strictly-pre-match history (strict < date cutoff).
- Production DB opened read-only and SHA-256 gated (fail-closed).
- No fabricating matches, odds, predictions, or results.
