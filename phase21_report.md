# Phase 21 — Model Comparison (LR vs. Elo vs. Poisson)

## 1. Status
- Phase 20 COMPLETE (Poisson; see `phase20_report.md`).
- Phase 21 = a strictly read-only, deterministic comparison of the Phase-18 multinomial
  logistic regression, Phase-19 Elo and Phase-20 Poisson on the **same** chronological
  3-class outcome task over **one identical test split**, with a same-day Elo leakage
  fix. New module: `src/football_intelligence/prediction/comparison.py`. CLI:
  `football-intelligence model compare`. Working tree preserved; production DB
  **read-only** this phase.

## 2. Dataset (genuine, read-only)
- Prod DB `data/football_intelligence.db`, SHA-256 (before == after):
  `725af574b88dc033bd3b00928de503f9bd9a836fbe4252832b7812dce0ab3122`
  (checked around `load_matches_for_features`; `run_model_comparison` raises if it
  changes — leakage contract).
- Filter: provider=`statsbomb`, provider_competition_id=`2` (Premier League).
- **n_matches = 380, n_teams = 20**, date range `2015-08-08 -> 2016-05-17`.
- Identical chronological split (60/20/20): **train 228 / val 76 / test 76**;
  test window `2016-04-02 -> 2016-05-17`.
- Elo/Poisson are warmed online over all 380 pre-test matches; LR is fit on the
  train split (228) and predicts the 76-test split only.

## 3. Elo same-day leakage fix
- `EloModel.fit` now batches per `match_date`: all same-date fixtures read the same
  pre-date baseline; zero-sum deltas accumulate and are committed only at the next
  date boundary (or flushed at the end). A fixture can no longer observe a
  concurrently-played fixture's just-applied update.
- Regression: `test_elo_same_day_leakage_regression` builds two same-date fixtures
  sharing team T1 and asserts both read T1's initial 1500 rating (B cannot see A's
  intra-day update) while final ratings still reflect the batched day-end update.
- Materiality on prod PL: **99 distinct match dates, 70 of them multi-fixture**, max
  **10 fixtures on one date** — material on real data (the synthetic DB has one
  fixture per date, where the fix is correctness-hygiene with ~zero metric impact).

## 4. Comparison design (identical fixtures, honest metrics)
- Alignment: fixtures sorted by `(match_date, home_team_id, away_team_id)`; Elo
  history and Poisson predictions are keyed by the same `_key` and looked up per
  test row, so all three models are scored on the **exact same 76 fixtures**.
- LR & Poisson emit proper 3-class probabilities → comparable Brier + log-loss.
- Elo emits an expected home **score** (2-class Elo quantity, not a calibrated
  3-class probability); it reports native `brier_expected_score` and is compared
  with LR/Poisson on accuracy / macro-F1 / confusion / P-R only. Documented in
  `ComparisonReport.notes`.
- Paired statistics on identical fixtures:
  - `mcp_mcnemar_exact` — exact two-sided binomial McNemar on accuracy discordance.
  - `paired_bootstrap` — percentile bootstrap of per-sample loss difference with
  a 95% CI (2000 resamples default; this prod artifact used 500 for speed).

## 5. Results (prod test window, 76 fixtures)

| model                | accuracy | macro-F1 | Brier(3) | log-loss | Brier exp(2) |
|----------------------|---------:|---------:|---------:|---------:|-------------:|
| logistic_regression  | 0.4605   | 0.3391   | 0.7081   | 1.2699   | n/a          |
| poisson              | 0.5132   | 0.3879   | 0.5968   | 0.9952   | n/a          |
| elo                  | 0.5132   | 0.4771   | n/a      | n/a      | 0.1312       |

Confusion (rows=true home/draw/away, cols=pred):
- LR `[[25,8,4],[12,10,2],[6,9,0]]`
- Poisson `[[28,0,9],[14,0,10],[4,0,11]]` (never predicts draw — expected from the
  grid-max heuristic)
- Elo `[[21,15,1],[8,14,2],[3,8,4]]`

Paired significance (identical test fixtures):

| pair            | McNemar p | Δloss mean (i−j) | 95% CI (boot)     | boot p | Δbrier (boot)                       |
|-----------------|----------:|-----------------:|-------------------|-------:|-------------------------------------|
| LR vs Poisson   | 0.5847    | 0.0526           | [-0.0789, 0.1842] | 0.4360 | 0.1114 [-0.0029, 0.2326], p=0.0600  |
| LR vs Elo       | 0.5413    | 0.0526           | [-0.0526, 0.1711] | 0.4360 | n/a (Elo 2-class)                   |
| Elo vs Poisson  | 1.0000    | 0.0000           | [-0.1316, 0.1385] | 1.0000 | n/a (Elo 2-class)                   |

`mean_diff` is the per-sample loss difference (i − j); positive ⇒ model `i` has
higher loss (lower accuracy). Poisson and Elo tie on accuracy here (0.5132); LR is
5.26pp behind. No pair reaches significance on 76 test fixtures.

## 6. Tests
- `tests/test_model_compare.py` (11): split-contract, determinism + read-only,
  3-model metric shape, fixture alignment, McNemar sanity (p=0.125 hand-checked) +
  edge cases, paired bootstrap sanity + identical-losses, artifact round-trip,
  Elo same-day leakage regression.
- `tests/test_cli_model_compare.py` (6): parser registration, all-season /
  single-season / multi-season artifact emission, invalid-competition → exit 1,
  missing-DB → exit 1.
- `tests/test_prediction_elo.py` (7) still green (same-day fix retained).
- `python -m py_compile` on `comparison.py`, `__init__.py`, `elo.py`, `cli.py`: OK.

## 7. Artifacts
- `reports/phase21/phase21_comparison_2_all.json` (full per-fixture predictions,
  per-class P/R/F1, confusion, calibration, config, paired stats).
- `reports/phase21/phase21_comparison_2_all.txt` (readable summary).
- Production DB SHA-256 recorded in both artifacts (`production_db_unchanged: true`).

## 8. Reproduction
```
.venv\Scripts\python -m pytest tests/test_model_compare.py tests/test_cli_model_compare.py -q   # 16 passed
.venv\Scripts\python -m pytest tests/test_prediction_elo.py -q                                   # 7 passed
football-intelligence model compare \
    --db-path data/football_intelligence.db \
    --competition-id 2 --output-dir reports/phase21 --n-bootstrap 500
```

## 9. Limitations / honest caveats
- Single competition, single two-season window (2015/16 + 2016/17 in prod). No
  multi-season walk-forward validation yet — see Phase 22.
- LR vs Poisson Brier/log-loss are directly comparable; Elo Brier is structurally
  `n/a` (2-class expected score, not 3-class probability).
- 76 test fixtures is a hard sample for paired inference (none significant).
- No model is ranked "superior"; metrics are descriptive.

