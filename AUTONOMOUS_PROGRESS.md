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
  - Phases 18-19 -- COMPLETE. Phase 20 (Poisson) -- COMPLETE; Phases 21-24 pending.

## Last test result
- Full suite: `417 passed` (383 prior suite + 34 new Phase-20 poisson tests), 0 failures,
  0 skips. Command: `.venv\Scripts\python.exe -m pytest -q`.
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
- Phase 21 (model comparison: Poisson vs logistic vs Elo on identical chronological
  test fixtures): next. Read-only harness reused from Phases 18-20.
