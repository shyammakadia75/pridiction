# Phase 14 — Audit Report: Football-Data.co.uk Results Adapter

**Mode:** read-only audit + validation. **No production import performed.**
**Repository root:** `C:\pridection\pridiction` (git repo, branch `main`).
**Date of audit:** 2026-10-18.
**Report location:** `C:\pridection\pridiction\phase14_report.md`

---

## 1. Files inspected and changed

### State of the working tree (confirmed via `git status` / `git diff --stat`)
- **Branch:** `main`, HEAD = `bce0714d` ("2342", 2026-10-14).
- **Tracked, modified (working tree vs HEAD):**
  - `src/football_intelligence/cli.py` — 115 insertions (lines).
  - `src/football_intelligence/features/engineering.py` — +189/-22 (189 lines).
- **Untracked / new (Phase 13 deliverables, not yet committed):**
  - `src/football_intelligence/data/ingestion/football_data_results.py` (new adapter).
  - `tests/test_football_data_results.py` (new test module).
  - `tests/test_competition_filtering.py`, `tests/test_prediction_lr.py` (pre-existing untracked test files).
- Pre-existing untracked scratch files `_smoke.py` and `check.py` were **left untouched**.

### Phase 13 changes (within scope of this audit)
- `cli.py`:
  - Two indentation regressions repaired (8->4 spaces): the `print(f"Status: {result.status}")` block inside `cmd_results_import` and the analogous block in `cmd_odds_import`.
  - `cmd_results_import` now constructs `FootballDataResultsIngestor(...)` **inside** `try/except IngestionError`, so an invalid league/season code raises `IngestionError` from `__init__` (instead of crashing) and the CLI returns exit code `1` with a clean `ERROR:` message.
  - New `football-data results` subcommand group with `list-leagues` and `import` (--league/--league-code, --season/--season-code, --db-path, --cache-dir, --refresh, --timeout, --source-url).
- `football_data_results.py` (new): dedicated results adapter.
- `test_football_data_results.py` (new): 44 collected tests.

### Engineering.py — unrelated, preserved
`engineering.py` was modified (tracked) to add competition/season scoping to `FeatureEngineer.compute_all_features_from_db` / `load_matches_for_features` and a new `_resolve_competition_id` helper (prevents cross-competition contamination and same-day/future leakage in feature engineering). Inspected and confirmed **unrelated** to results ingestion. **Not reverted, not overwritten**, as instructed.

### AST / compile check (all 3 files)
```
OK  src/football_intelligence/cli.py
OK  src/football_intelligence/data/ingestion/football_data_results.py
OK  tests/test_football_data_results.py
```
`ast.parse` + `py_compile.compile(..., doraise=True)` succeed for all three — no syntax/indentation errors remain.

## 2. Test commands, counts, and results

| Command | Result |
|---|---|
| `.\.venv\Scripts\python -m pytest tests/test_football_data_results.py -p no:cacheprovider -q -p no:logging --color=no` | **44 passed** in 5.43s, exit 0 |
| `.\.venv\Scripts\python -m pytest -p no:cacheprovider -q -p no:logging --color=no` (full suite) | **335 passed** in 34.56s, exit 0 |

All 44 adapter tests run with `tmp_path`/`monkeypatch` fixtures only — every DB write targets a disposable temp DB (`db_path` fixture). No test touched the production database. Confirmed post-run that the production DB SHA-256 is unchanged (see Section 6).

The 44 adapter tests cover (by name): season-code mapping & validation, league display, provider-match-id composition/marked-internal, normalized field mapping, date formats, team aliases (`man united`->`manchester united`), FTR/score outcome consistency (mismatch + illegal char + parametrized ok cases), full pipeline persistence, competition/team/match counts, `data_sources` provenance, odds-table isolation, idempotent re-run, ISO/DMY date parsing, missing required columns, negative goals, same home/away team, transaction rollback on validation error, invalid file, empty CSV, default URL, invalid league/season codes, cache reuse (real on-disk cache), cross-source duplicate skipped, cross-source duplicate idempotent after seed, multiple seasons distinct, provider-scoped teams, `list_result_leagues`, and CLI subcommand parsing + end-to-end import + invalid-league exit code.

---

## 3. Audit findings (Step 3)

All behaviors below were verified against the code and the passing tests.

| Audit item | Verdict | Evidence |
|---|---|---|
| Supported league & season validation | OK | `__init__` checks `league_code in LEAGUE_CODES` and `season_code` is 4 digits, raising `IngestionError`; tests `test_invalid_league_code_raises`, `test_invalid_season_code_raises`, `test_season_code_invalid_raises` (6 params), `test_league_display_invalid_raises`; CLI `test_cli_import_invalid_league_nonzero` (rc!=0). |
| CSV parsing / date formats / scores / FTR | OK | `_parse_csv` (DictReader, skips blank `Date` rows); `_normalize_match_date` handles `DD/MM/YYYY`, `D/M/YYYY`, `YYYY-MM-DD`; `validate_outcome_score_consistency` checks FTR in {H,D,A} and agreement with scores. Real seasons: 0 invalid dates/scores/outcomes (see Section 4). |
| Missing/malformed columns & rows | OK Mostly | `test_validate_missing_required_columns`, `test_ingest_invalid_file_logs_error`, `test_ingest_empty_csv_rejected`. Note: column-level absence is surfaced as missing-value errors (the normalizer uses `row.get()`->null) rather than "column missing", which still aborts ingestion. |
| Team aliases & unknown team names | OK | `TEAM_ALIASES` + `_normalize_team_name` (lowercase, strip `f.c./fc` suffixes, alias map). Unknown names are retained normalized (lowercase) - no failure. `test_teams_are_provider_scoped`, `test_normalize_results_dataframe_field_mapping`. |
| Provider-scoped team/match identity | OK | `UNIQUE(provider, provider_team_id)` on teams; `UNIQUE(provider, provider_match_id)` on matches; `provider_match_id` is the project-derived composite `football-data|{season}|{date}|{home}|{away}` (clearly marked internal, never a source id). |
| Idempotent repeated imports (same provider) | OK | `_upsert_match` (1) updates existing `(provider, provider_match_id)` rows; `test_ingest_idempotent_rerun_updates` (2nd run -> 0 inserted, 3 updated, count stays 3). |
| Cross-source dedup incl. seeded StatsBomb | OK | `_upsert_match` (2) consults `_find_matches_for_reconciliation` (any provider) by `(date, norm_home, norm_away)`; `test_cross_source_duplicate_skipped` (1 skip, 0 football-data rows for the fixture), `test_cross_source_duplicate_idempotent_after_seed`. |
| Duplicate encountered repeatedly | OK | Covered by the idempotent-after-seed test (2nd ingest re-skips, never re-inserts). |
| Same teams on different dates | OK | `test_multiple_seasons_distinct` imports 1516 + 1617 (Arsenal-Burnley on different dates) -> 4 matches, 2 seasons, both inserted. Different `match_date` => no false dedup. |
| Provenance: `data_sources` | OK | `load()` -> `_record_data_source` writes one `data_sources` row per import with `dataset_name='results'`, full metadata. `test_ingest_records_data_source_provenance`. |
| Provenance: `ingestion_runs` | WARNING Pre-existing (see notes) | CLI's `_safe_record_run` calls `record_ingestion_run`, which INSERTs 15 columns into the 11-column `ingestion_runs` table -> `OperationalError: table ingestion_runs has no column named records_skipped` -> swallowed -> 0 rows written. Reproduced in isolation. The 4 prod rows are pre-existing leftovers. |
| Transaction rollback on failure | OK Mechanism / WARNING Test gap | `load()` wraps all writes (incl. `_record_data_source`) in `with database_connection(...)` which commits on clean exit, rolls back on exception. `test_ingest_transaction_rollback_on_validation_error` confirms a validation failure yields 0 rows. No test exercises a mid-load DB error after some rows insert (the validation gate covers the realistic failure modes). |
| Download caching & cache reuse | OK | `_download_csv` writes raw bytes to `cache_dir/{league}/{season}.csv`; reads cached (utf-8-sig) on subsequent calls. Unit test + a real-network demo (Section 4) confirm 2nd `retrieve()` -> `was_cached=True`, identical frame. |
| CLI invalid-argument behavior & exit codes | OK | `test_cli_import_subcommand_exists`, `test_cli_import_end_to_end` (rc==0, 3 matches), `test_cli_import_invalid_league_nonzero` (rc!=0). Invalid league no longer crashes (`IngestionError` from `__init__` caught). |
| Separation: results vs odds ingestion | OK | Results adapter writes only `matches/teams/competitions/data_sources`; never the `odds` table (`test_ingest_does_not_touch_odds_table` -> 0 odds rows). Separate CLI sub-parsers; `DATASET_NAME='results'` vs `'odds'`. |

### Section 3 notes / limitations
#### Cross-source dedup key
- **Same provider (idempotency):** exact lookup by `(provider, provider_match_id)` where `provider_match_id` = `football-data|{season}|{date}|{home_norm}|{away_norm}`. Deterministic -> safe re-import updates in place, never double-counts.
- **Cross provider:** `_find_matches_for_reconciliation` scans all matches by `(date, normalized home, normalized away)`. 1 existing -> `skipped`; >1 -> `ambiguous` (logged, **not** inserted, counted separately). Because a league team plays at most one fixture per day, this tuple uniquely identifies a real fixture.

#### Limitations / risks
1. **Name-only matching:** relies on `_normalize_team_name` (lowercase, strip `f.c./fc` suffixes, `TEAM_ALIASES` map). An atypical team name not in the alias map could miss a real fixture (-> inserted as new -> potential duplicate across sources).
2. **Home/away ordered:** requires home==home, away==away (no swapped-side matching). Fine for league fixtures with fixed home venues.
3. **Date granularity `YYYY-MM-DD`:** two-legged ties on the same calendar date become `ambiguous` -> skipped (safe but lossy); impossible for a single team's league fixtures.
4. **O(n) scan:** no composite index on `(match_date, home_team_id, away_team_id)`; acceptable for EPL-scale, a performance concern on the full StatsBomb corpus.
5. **Ambiguous matches (>1 existing) are never inserted** -> double-counting is structurally impossible.

#### Other pre-existing findings
- `record_ingestion_run` / `ingestion_runs` drift is pre-existing (modules `odds.py`, `schema.py`, `base.py` are original). Phase 13 made the CLI robust to it via `_safe_record_run` (best-effort, swallows the error). Not a Phase 13 regression.
- Production DB at `schema_version=1` vs code `SCHEMA_VERSION=2` (v2 migration adds only `odds` columns; never applied). Out of scope (no migrations run). Does not affect results ingestion.

---

## 4. Per-season validation results (Step 4) — real Football-Data CSVs

A temporary, non-production cache dir and throwaway DB were used. `retrieve()` + `validate()` were run against the real URLs (live network, no mocks). No production database was written (`retrieve`/`validate` never open the DB).

| Season | URL tail | Source | Rows | Accepted | Null dates | Null teams | Teams | Dates | Dup ids | FTR (H/D/A) | Validation |
|---|---|---|---|---|---|---|---|---|---|---|---|
| EPL 2015/16 | `.../1516/E0.csv` | network download | 380 | 380 | 0 | 0 | 20 | 99 | 0 | 157/107/116 | 0 err / 0 warn |
| EPL 2016/17 | `.../1617/E0.csv` | network download | 380 | 380 | 0 | 0 | 20 | 107 | 0 | 187/84/109 | 0 err / 0 warn |
| EPL 2019/20 | `.../1920/E0.csv` | network download | 380 | 380 | 0 | 0 | 20 | 115 | 0 | 172/92/116 | 0 err / 0 warn |

- All FTR distributions sum exactly to 380 (no missing/illegal outcomes).
- All dates parsed as `DD/MM/YYYY`; ISO dates also parse (unit tests confirm).
- **Cache-reuse demo** (real network, same temp cache dir, season 1617): 1st `retrieve()` -> `was_cached=False`, cache file written; 2nd `retrieve()` -> `was_cached=True`, **identical** 380-row frame, validation clean. Proves the real on-disk cache path (not just the mocked one).
- Sample internal key: `football-data|1516|2015-08-08|afc bournemouth|aston villa` (deterministic, marked internal).
- Normalization assumptions: names lowercased; `FC`/`F.C.` suffixes stripped; common short names aliased (`Man United`->`manchester united`, `Spurs`->`tottenham hotspur`, `Leicester`->`leicester city`). Unknown names are kept as their lowercased form (no data loss).

## 5. Cross-source deduplication design & limitations

(Consolidated; verification matrix is in Section 3.)

- **Same provider (idempotency):** exact lookup by `(provider, provider_match_id)` where `provider_match_id` = `football-data|{season}|{date}|{home_norm}|{away_norm}`. Deterministic -> re-import updates in place, never double-counts.
- **Cross provider:** `_find_matches_for_reconciliation` scans all matches by `(match_date[:10], _normalize_team_name(DB home), _normalize_team_name(DB away))` vs normalized input names. 1 existing -> `skipped`; >1 -> `ambiguous` (logged, **never inserted**, counted separately). Because a league team plays at most one fixture per day, this tuple uniquely identifies a real fixture.
- **Risks/limitations:** name-only matching (finite alias table, could miss an atypical name -> potential missed dedup); home/away-ordered (no swapped-side matching); day-granular date (same-date two-leg ties become `ambiguous`->skipped, safe but lossy); O(n) scan with no composite index on `(match_date, home_team_id, away_team_id)`. Ambiguous (>1) matches are **never** inserted, so double-counting is structurally impossible.
- For the three validated real seasons: 0 ambiguous cases, 0 duplicate ids.

---

## 6. Production database safety verification (Step 5)

- **Path:** `C:\pridection\pridiction\data\football_intelligence.db`.
- **SHA-256:** `725af574b88dc033bd3b00928de503f9bd9a836fbe4252832b7812dce0ab3122` — captured at audit start **and** re-captured after every audit action; **identical** (size 200,704 bytes both times). All Phase 14 activity used temp DBs / mocks only.
- **Baseline caveat:** no trusted pre-Phase-13 hash was available (Phase 13 left the repo uncommitted). Immutability is demonstrated by before/after-this-phase hash equality plus the fact that no football-data data was ever written to prod (see counts below). The DB was not altered to obtain a baseline.
- **integrity_check:** ok. **foreign_key_check:** no rows (no FK violations). (`foreign_keys` pragma reports 0 only on the read-only inspection connection; the app enables it per-connection in `connection.py`, and `foreign_key_check` is pragma-independent and clean.)
- **Schema version:** `schema_version` row = 1. Code constant `SCHEMA_VERSION = 2`; migration v2 only adds columns to the `odds` table; never applied to this DB (prod `odds` table is v1 baseline). Pre-existing, out of scope (no migrations run). Does not affect results ingestion.
- **Counts:** matches 381 (380 StatsBomb 2015/16 + 1 StatsBomb 2018/19 UCL), teams 20, competitions 2 (StatsBomb PL id=2, CL id=16), `data_sources` 4 (all statsbomb/matches), `ingestion_runs` 4 (all statsbomb/matches, pre-existing), `odds` 0, predictions 0.
- **Football-Data presence in prod:** 0 teams / 0 competitions / 0 matches / 0 `data_sources(dataset_name='results')` / 0 `ingestion_runs(dataset_name='results')`. **A results import has never been run against production.**

> Import-approval note: prod already holds a full StatsBomb 2015/16 Premier League season (380 matches). A Football-Data 2015/16 results import against this DB would report 0 inserts, 380 cross-source skips. Football-Data imports would add only seasons absent from StatsBomb here (e.g. StatsBomb has no 2016/17 or 2019/20 data, so those would insert).

## 7. Verdict

### **PASS WITH WARNINGS**

The Phase 13 results adapter is functionally correct, well-engineered, and well-tested:
- 335/335 tests pass (incl. 44 adapter tests); AST clean; reported CLI indentation/error-handling fixes present and correct.
- 3 real Football-Data seasons validate cleanly (380 rows each, 0 nulls, 0 invalid outcomes, 0 duplicate ids, 20 teams each); caching/cache-reuse confirmed end-to-end on live network.
- Cross-source deduplication is structurally safe (ambiguous => skip, never double-count).
- Production DB confirmed untouched.

**Warnings (all pre-existing, not regressions from Phase 13):**
1. `record_ingestion_run` INSERTs 15 columns into the 11-column `ingestion_runs` table -> `OperationalError` -> swallowed by `_safe_record_run` -> CLI imports do not record an `ingestion_runs` audit row. Match/team/competition/`data_sources` provenance is unaffected and transactionally correct. Recommend aligning `record_ingestion_run` with the `ingestion_runs` schema (or adding the missing columns via a migration) before relying on `ingestion_runs` for audit.
2. Production DB at `schema_version=1` vs code `SCHEMA_VERSION=2` (v2 migration never applied; only affects `odds`). Out of scope (no migrations run).
3. No test exercises a mid-`load` DB failure (only validation-gated rollback is tested). The rollback mechanism (`database_connection` context manager) is sound.

---

## 8. Readiness for a separate import-approval phase

**Ready - conditionally.** The adapter is safe to import behind an approval gate, provided the import-approval phase:
1. Snapshots the production DB SHA-256 before and after and confirms equality.
2. Runs imports against a **copy** of (or migrates) the production DB first; do not import directly into the live prod DB until the `ingestion_runs` column drift above is reconciled (else the audit trail is silently omitted; match/team/competition/`data_sources` rows are still written correctly and transactionally).
3. Applies the v2 migration separately if odds ingestion is also desired (no action needed for results ingestion, which is schema-v1 compatible).
4. Understands the cross-source dedup semantics (an already-StatsBomb season like 2015/16 EPL will report 380 `skipped`, 0 `inserted`; other seasons will insert).

**No production import is performed in this phase.** Stopping here per the stop condition; awaiting approval to proceed to an import-approval phase.




