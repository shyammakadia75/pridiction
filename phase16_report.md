# Phase 16 Report — Leakage & Multi-Season Readiness Audit

**Status:** COMPLETE. Verified against the actual workspace and the production
database (read-only). Produced the audit findings that Phase 17 builds on.

## Environment
- Git root: `C:\pridection\pridiction`, branch `main`, commit `bce0714`.
- Interpreter: `.venv` Python 3.13; `pytest 9.1.1`.
- Production DB: `data/football_intelligence.db`, SHA-256
  `725AF574B88DC033BD3B00928DE503F9BD9A836FBE4252832B7812DCE0AB3122`.

## Findings

### Schema & provenance — PASS
- `ingestion_runs` = 11 columns (`id` + `provider, dataset_name, started_at,
  completed_at, status, records_read, records_inserted, records_updated,
  error_message, created_at`); **no** `records_skipped` column. `IngestionResult`
  keeps richer in-memory counts (`records_skipped/matched/unmatched/ambiguous/
  invalid`) but does not invent schema. `odds.record_ingestion_run` and
  `statsbomb.record_ingestion_run` are aligned (10 persisted columns).
- `cli._safe_record_run` surfaces provenance-write failures as `DatabaseError`
  (never swallowed); failed runs still record an audit row.
- Production: `integrity_check=ok`, `schema_version=1`, 381 matches (all
  `statsbomb`), 20 teams, 2 competitions, 4 ingestion_runs, 4 data_sources, 0
  cross-provider duplicate fixtures. No Football-Data results imported yet.

### Temporal leakage — PASS
- `_filter_history` (`engineering.py`) uses **strict `<` target_date**, excludes
  the target match and all same-date fixtures, keeps only non-null-goal matches,
  sorts by date. It is invoked on every feature path
  (`compute_features` → `compute_all_features` → `compute_all_features_from_db`).
- The target match's own score never reaches `X`: `compute_features` builds a
  fresh dict that omits `home_goals`/`away_goals`; `prepare_training_data`
  additionally excludes `target_match_id, target_match_date, home_team_id,
  away_team_id, features_valid, outcome_label, _label_idx`.

### Scaler & splits — PASS
- `_impute_and_scale` computes mean/std from the **input (train) matrix only**;
  `LogisticRegressionClassifier.fit` fits the scaler on training `X`;
  `predict_proba` applies it via `_apply_scaler`. `chronological_split` is an
  index cut on a date-sorted frame (no shuffle); test set is used only after the
  procedure is frozen. No `train_and_evaluate`/`evaluate_model` orchestrator
  exists in `src` — evaluation is composed in scripts/tests.

### Competition & season filtering — PASS (external change to PRESERVE)
- `load_matches_for_features` + `_resolve_competition_id` + `_ensure_competition_exists`
  apply competition/season filters at **load time** (mutually-exclusive params,
  `ValueError` on contradictions/unknown competitions, backward-compatible). No
  cross-competition contamination; `_filter_history` then enforces the date rule.

### Cross-source — PASS with one gap
- Football-Data stores `provider_team_id = normalized_name` (provider
  `football-data`); StatsBomb stores `provider_team_id = str(team_id)`
  (provider `statsbomb`). Per `teams` UNIQUE(`provider, provider_team_id`),
  the same real team is **two different `teams.id` rows** per provider — no
  canonical team id → provider histories are not merged (acceptable because a
  competition is sourced from one provider).
- Football-Data `_upsert_match` cross-source guard (`_find_matches_for_reconciliation`
  by `match_date[:10]` + normalized team names) skips/ambiguates duplicates.
  StatsBomb `_upsert_match` has **no reverse guard** → ingest order matters
  (FD-after-SB safe; SB-after-FD can double-insert). Flagged for Phase 17.

### Code-quality observation (non-blocking)
- `engineering.py` has duplicate definitions: `_add_rest_features` (568 & 639)
  and `_add_h2h_features` (592 & 663); the later identical defs shadow the
  earlier. Functionally inert (tests pass). Phase 17.3 removes the dead copies.

## Tests
- Full suite: `341 passed` (22.5s).
- Leakage/competition/prediction subset: `101 passed`.

## Conclusion
No feature-to-label leakage. Competition/season filtering is intact and must be
preserved. The StatsBomb-side cross-source deduplication gap and the duplicate
feature definitions are the two non-blocking items handed off to Phase 17.
