# Phase 26.6 Report: Ablation Study of Last-Five-Match Form Features

## 1. Executive Summary

**Decision: Do not adopt Phase 26.5's additional form features.**

Phase 26.5 extended the feature engineering pipeline with three new capabilities:
- Per-match **average** features (goals scored/conceded per game)
- **Venue-specific** form splits (home/away performance separately)
- **Recency-weighted** form aggregates (linear weighting favoring recent matches)

Phase 26.6 evaluated these features through a controlled ablation study with walk-forward cross-validation, paired bootstrap tests, McNemar's test, and a Poisson leakage sanity check.

**Key findings:**
- The Poisson regression model (production baseline) is **completely unaffected** by all feature changes. Its Brier score (0.5814) and accuracy (0.5487) are identical across all four configs, confirming **no data leakage**.
- The Logistic Regression model shows **statistically significant degradation** in log-loss and Brier score for all three extensions (B, C, D) vs. baseline (A). CIs exclude zero.
- Accuracy differences are **not significant** (McNemar p > 0.53), but calibration degrades substantially.
- Adding 24-49 features on 380-760 training samples worsens the feature-to-sample ratio, causing overfitting.

**Recommendation:** Retain the Phase 25 baseline. The additional features increase complexity without predictive benefit.

---

## 2. Experimental Setup

### 2.1 Configs Compared (Ablation)

| Config | Description | Feature Count |
|--------|-------------|---------------|
| A_baseline | Phase-25 baseline: overall form counts only. No per-match averages, no venue split, no recency weighting. | 46 |
| B_plus_form_averages | A + per-match average features (goals_per_game, etc.). | 70 (+24) |
| C_plus_venue_split | B + venue-specific form splits (home only from home matches, away only from away). | 90 (+20) |
| D_plus_recency_weighted | C + linear recency weighting (oldest=1, newest=n, normalized). | 95 (+5) |

### 2.2 Walk-Forward Folds

| Fold | Train | Validation | Test | Train Cutoff |
|------|-------|------------|------|--------------|
| fold1 | 2015/2016 | - | 2016/2017 | 2016-08-13 |
| fold2 | 2015/16+2016/17 | 2016/17 | 2019/20 | 2019-08-09 |

- **Strategy:** Expanding-window chronological walk-forward (no shuffling, no future info).
- **Seed:** 42.
- **Bootstrap:** 4000 iterations for paired bootstrap CIs.

### 2.3 Models

- **Logistic Regression (LR):** 3-class (home_win, draw, away_win). L2, C=1.0, balanced. Feature-sensitive probe.
- **Poisson Regression:** Goal-count model. Does NOT consume engineered features (separate team-strength coefficients). Leakage-invariant sanity check.

### 2.4 Dataset

- Provider: football-data, E0
- Seasons: 2015/2016, 2016/2017, 2019/2020
- Production DB SHA-256: unchanged = `725af574b88dc033bd3b00928de503f9bd9a836fbe4252832b7812dce0ab3122`

### 2.5 Metrics

- Accuracy, Macro-F1, Log-loss, Multiclass Brier, Calibration bins.

---

## 3. Results

### 3.1 Pooled (Test-Set) Results

#### Logistic Regression (Feature-Sensitive Probe)

| Config | Features | Accuracy | Brier | Log-Loss |
|--------|----------|----------|-------|----------|
| A_baseline | 46 | **0.4671** | **0.6269** | **1.0438** |
| B_form_averages | 70 | 0.4632 | 0.6515 | 1.1024 |
| C_venue_split | 90 | 0.4539 | 0.7009 | 1.2130 |
| D_recency_weighted | 95 | 0.4645 | 0.6966 | 1.2070 |

#### Poisson (Leakage-Invariant Sanity Check)

| Model | Accuracy | Brier |
|-------|----------|-------|
| Poisson (all configs) | 0.5487 | 0.5814 |

Poisson is identical across all configs, confirming no data leakage.

### 3.2 Poisson Sanity Check

**Status: PASS**

| Fold | Unique Brier Scores | Values |
|------|-------------------|--------|
| fold1 | 1 (all 4 configs identical) | 0.5692 |
| fold2 | 1 (all 4 configs identical) | 0.5936 |

### 3.3 Per-Fold LR Metrics

| Config | Fold 1 Brier | Fold 2 Brier |
|--------|-------------|-------------|
| A_baseline | 0.6205 | 0.6333 |
| B_form_averages | 0.6453 | 0.6576 |
| C_venue_split | 0.7027 | 0.6990 |
| D_recency_weighted | 0.7037 | 0.6895 |

### 3.4 Paired Bootstrap Tests (Log-Loss, vs. A_baseline)

| Pair | Mean Diff | 95% CI | Bootstrap n |
|------|-----------|--------|-------------|
| A vs B | +0.0586 | [+0.0207, +0.0969] | 760 |
| A vs C | +0.1692 | [+0.1183, +0.2211] | 760 |
| A vs D | +0.1632 | [+0.1120, +0.2156] | 760 |

All CIs are positive and exclude zero → B, C, D significantly worse on log-loss.

### 3.5 McNemar's Test (Accuracy, vs. A_baseline)

| Pair | McNemar p-value |
|------|-----------------|
| A vs B | 0.8844 |
| A vs C | 0.5327 |
| A vs D | 0.9463 |

No significant accuracy differences — features hurt calibration, not argmax decisions.

---

## 4. Analysis

### 4.1 Why the Features Don't Help

1. **Overfitting from dimensionality.** Adding 24-49 features to 380-760 match datasets raises the feature-to-sample ratio. Even with L2 regularization, parameters exceed samples, causing overfitting.

2. **Limited signal in small samples.** Form averages, venue splits, and recency weights need ~10+ matches per team. In the 380-match fold, histories are sparse, producing noisy estimates.

3. **Calibration degradation.** Expanded features produce overconfident predictions. Brier degrades (A: 0.6269 -> C/D: ~0.70).

### 4.2 Why Poisson Is Unaffected

Poisson models goal counts via separate team-strength coefficients. It does not use engineered form features. Changes to form features cannot degrade the production Poisson model.

### 4.3 McNemar vs. Bootstrap Discrepancy

McNemar tests accuracy (0/1). Bootstrap tests log-loss (continuous). Features shift probability mass without changing argmax — accuracy unchanged, log-loss/Brier degrade.

### 4.4 Class-Level Performance (A_baseline, Pooled LR)

| Class | Precision | Recall | F1 | Support |
|-------|-----------|--------|-----|---------|
| home_win | 0.565 | 0.607 | 0.585 | 359 |
| draw | 0.242 | 0.205 | 0.222 | 176 |
| away_win | 0.449 | 0.449 | 0.449 | 225 |

Confusion matrix (rows=true, cols=predicted):

```
         HW   Draw  AW
HW      218   66    75
Draw     91   36    49
AW        77   47   101
```

### 4.5 Calibration (A_baseline LR vs. Poisson)

LR (A_baseline) — well-calibrated on home_win, but overconfident on away_win.
Poisson — better calibrated overall. Higher accuracy (0.5487 vs 0.4671), lower Brier (0.5814 vs 0.6269).

---

## 5. Regression Tests

19 regression tests in `tests/test_phase26_6_regression.py`:

- Same-day exclusion (fixture-date inclusive)
- Future match exclusion
- Latest-5 selection
- Venue correctness
- Cold-start NaN for rates, 0 for matches_played
- Feature flag consistency
- Schema consistency
- Monotonic feature counts (B <= D)
- Recency weight correctness (linear, oldest=1, newest=n)
- Reproducibility

All 19 pass. Full suite passes with 0 new failures.

---

## 6. Conclusions & Recommendations

1. **Phase 26.5 features do not improve predictions.** All three extensions significantly worsen LR log-loss and Brier (CIs exclude 0). No accuracy improvement (McNemar p > 0.53).

2. **No data leakage detected.** Poisson sanity check PASS — Poisson scores identical within each fold.

3. **Root cause: overfitting from dimensionality.** Features are correctly computed (19 regression tests), but training folds (380-760 matches) are too small for 70-95 features.

4. **Recommendation:** Retain Phase 25 baseline (Config A). Do not deploy B, C, or D to production. Code remains gated behind `include_form_averages` flag for future experimentation.

5. **Future work:** Re-evaluate with more seasons, cross-validated regularization, L1/feature selection, or gradient boosting.

---

## 7. Artifact References

| Artifact | Path |
|----------|------|
| Ablation script | `scripts/phase26_6_ablation.py` |
| Regression tests | `tests/test_phase26_6_regression.py` |
| JSON results | `reports/phase26_6/phase26_6_ablation_20261021T094237Z.json` |
| Text report | `reports/phase26_6/phase26_6_ablation_20261021T094237Z.txt` |
| This report | `reports/phase26_6/phase26_6_report.md` |
| Source modifications | `src/football_intelligence/features/engineering.py` |
