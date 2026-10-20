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
