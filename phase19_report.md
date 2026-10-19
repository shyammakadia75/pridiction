# Phase 19 — Elo Ratings

## 1. Status
- Phase 18 COMPLETE (prediction evaluation pipeline + `model evaluate` CLI;
  see `phase18_report.md`).
- Phase 19 = chronological Elo ratings, read-only and deterministic. New module:
  `src/football_intelligence/prediction/elo.py`. CLI: `ratings elo`.
- Working tree preserved; production DB read-only this phase.

## 2. Dataset (genuine, read-only)
- Prod DB `data/football_intelligence.db`, SHA-256 (before == after):
  `725af574b88dc033bd3b00928de503f9bd9a836fbe4252832b7812dce0ab3122`
  (verified both externally and via the internal `_sha256` guard).
- Filter applied at load time via `load_matches_for_features`: provider=`statsbomb`,
  provider_competition_id=`2` (Premier League), season_name=`2015/2016`. The lone
  Champions-League 2018/19 fixture (competition_id `16`) is excluded by the
  competition filter — no cross-competition contamination.
- n_matches = 380 (all labelled, no NULL goals). n_teams = 20. Date range:
  2015-08-08 -> 2016-05-17.
- Matches sorted ascending by `match_date`; each match is scored from ratings
  learned from strictly-earlier results only (online, no leakage of same-day/future
  results).

## 3. Model
Classic Elo (home advantage, 3-outcome update win=1/draw=0.5/loss=0):

    E_home = 1 / (1 + 10^((R_away - R_home - H) / 400))
    update:  R_home += K*(S - E_home);  R_away -= K*(S - E_home)   (zero-sum)

Fixed hyper-parameters (NOT tuned on any partition):
- K = 20.0, home_advantage H = 50.0, initial_rating = 1500.0.
- 3-class prediction rule (documented, fixed thresholds): if E_home >= 0.60 ->
  `home_win`; if E_home <= 0.40 -> `away_win`; else `draw`.
- OUTCOME_LABELS order unchanged from Phase 18: 0=home_win, 1=draw, 2=away_win.

The expected score E_home is the canonical Elo quantity; the 3-class rule is a
descriptive mapping, not a calibrated probability. See Limitations.

## 4. Reuse audit (no re-write)
- `compute_outcome_label`, `OUTCOME_LABELS` (label order) — reused from
  `prediction/baseline.py`.
- `compute_macro_f1`, `build_confusion_matrix`, `class_distribution` — reused from
  `prediction/evaluation.py`.
- `load_matches_for_features` (load-time competition+season filter) + `resolve_db_path`
  — reused; the read-only SHA-256 guard is mirrored from `evaluation.py`.
- `prediction/elo.py` contains only Elo-specific logic (no duplicated metric code).

## 5. Genuine-data evaluation (prod, read-only)
Command:
```
football-intelligence ratings elo \
    --db-path data/football_intelligence.db \
    --competition-id 2 --season 2015/2016 \
    --output-dir reports/phase19
```
Online evaluation (each of the 380 matches predicted from prior ratings):
- n = 380, accuracy = 0.4079, macro-F1 = 0.3527, brier_expected_score = 0.1636.
- Class distribution (true): home_win=157, draw=107, away_win=116.
- Confusion matrix (rows=true [home/draw/away], cols=pred):
  `[[76, 80, 1], [30, 72, 5], [22, 87, 7]]` -> accuracy (76+72+7)/380 = 0.4079.
- Final ratings (top 10): Leicester City 1627.0, Tottenham 1575.7, Arsenal 1575.1,
  Southampton 1561.1, Manchester United 1554.8, West Ham 1550.1, Manchester City
  1547.9, Liverpool 1538.6, Chelsea 1513.3, Swansea City 1495.7.
- Artifacts: `reports/phase19/phase19_elo_2_2015-2016_<ts>.{json,txt}`.

## 6. Comparison context (honest, descriptive only — no ranking claims)
Phase-18 logistic regression (test set, n=76): accuracy 0.4737, Brier 0.6799,
macro-F1 0.3529. Elo (all 380, online) accuracy 0.4079. Elo under-initializes
(1500 start) on a single season and, with K=20, drifts slowly; both are weak
baselines on a single season. No "superior model" claim is made and no parameter
was tuned against any partition.

## 7. Safety
Prod DB write-checked: SHA-256 before == after (`725af574...`); internal guard
`production_db_unchanged == True`. All writes go to `reports/phase19`. CLI returns
exit code 1 on missing DB / invalid competition / evaluation failure.

## 8. Tests
- `tests/test_prediction_elo.py` (7): determinism, dataset/ratings shape, eval-metric
  shape, first-match expected-score + pre-update ratings (leakage sanity), monotonicity
  of `predict_home_win_prob`, invalid-competition raises, artifact round-trip.
- `tests/test_cli_ratings_elo.py` (5): parser registration, all-seasons smoke
  (40-match synthetic -> 6 teams + artifacts), single-season, missing-DB -> exit 1,
  invalid competition -> exit 1.
- `python -m py_compile` on `elo.py`, `prediction/__init__.py`, `cli.py`: OK.

## 9. Limitations / honest caveats
- Single genuine season (2015/16) in prod — no multi-season prod validation
  (multi-season coverage is synthetic-only in tests).
- K=20 / H=50 are conventional defaults, not tuned; metrics are descriptive.
- Elo expected score is not a calibrated 3-class probability; the fixed-threshold
  (0.60/0.40) 3-class rule is a heuristic and tends to under-predict draws here
  (it predicts draws in ~56% of matches and never separates the away-win class).
- No player/injury/home-rest features; ratings start all teams at 1500 (a known
  weakness for under-funded clubs whose early results are not yet reflected).
- Poisson goal modelling (Phase 20) would yield proper 3-class draw probabilities
  and is the natural next step.
