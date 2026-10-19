# Phase 15 — Safe Import Dry-Run & Provenance Repair

**Scope:** Diagnose and repair the `ingestion_runs` provenance drift surfaced in
Phase 14, make provenance failures loud, add hermetic regression tests, and
perform a safe (disposable-copy) dry-run of the Football-Data.co.uk results
adapter at the real 380-fixture scale.

**Outcome:** Production database is byte-for-byte unchanged (SHA256 verified
before/after). Full unit suite green — **341 passed**. No production writes.

---

## 1. Root-cause diagnosis (Step 2)

Two provenance defects were found:

### 1.1 `ingestion_runs` INSERT / schema drift — `odds.py`
`src/football_intelligence/database/schema.py` declares `ingestion_runs` with
**11 columns** (`INGESTION_RUNS_SQL`): `id, provider, dataset_name, started_at,
completed_at, status, records_read, records_inserted, records_updated,
error_message, created_at`.

- The **canonical** recorder — `statsbomb.record_ingestion_run` — inserts
  exactly these 10 data columns (+ `created_at`). It is imported by `cli.py`
  and `__init__.py`, and it is the one that seeded production's 4
  `ingestion_runs` rows (all `provider='statsbomb'`).
- `odds.record_ingestion_run` (odds.py:514) is a **second, divergent, dead**
  copy that INSERTs **15 columns**, referencing 5 columns
  (`records_skipped, records_matched, records_unmatched, records_ambiguous,
  records_invalid`) that **no schema version declares**. On any call it raises
  `sqlite3.OperationalError: table ingestion_runs has no column named
  records_skipped`.

### 1.2 Swallowed provenance failures — `cli._safe_record_run` (cli.py:188)
The CLI records every audit row via `_safe_record_run`, which did:
```python
except Exception as exc:  # noqa: BLE001 -- best-effort audit trail
    logger.warning("Could not record ingestion run: %s", exc)
```
This swallowed **every** provenance failure at `warning` level — the exact
mechanism that hid defect 1.1 in Phase 14 (a missing audit record on a failed
import was silently turned into a warning, masking the real error).

---

## 2. Repair decision: align to the current schema (no migration)

Options considered: (1) align the INSERT to the 11-column schema; (2) add a V3
migration adding the 5 undeclared columns (and update `statsbomb.py` to persist
them, backfill production).

**Decision: (1) align.** Evidence-based:
- The declared table is 11 columns; no schema version or migration in the
  entire codebase ever extends `ingestion_runs` (the only V2 migration alters
  the `odds` table).
- The **production-exercised** recorder (`statsbomb.record_ingestion_run`)
  already persists the 10-column subset, and production's 4 `ingestion_runs`
  rows were written by it. Aligning `odds.py` makes the two recorders
  consistent with each other and with real-world behavior.
- The 5 "extra" fields exist on `IngestionResult` **in memory** (used by CLI
  summaries) but were **never persisted** by any production code. Adding them
  via a new migration would **invent schema** — exactly the "guess which
  columns to add" the task forbids; it would also diverge from the canonical
  recorder, ballooning scope.
- Smallest justified repair: one INSERT literal, no migration, no production DB
  change, no backfill.

The 5 fields remain on `IngestionResult` for human-readable CLI output; they
are intentionally not written to `ingestion_runs`, matching the schema and
production.

---

## 3. Changes made

### Repair A — `src/football_intelligence/data/ingestion/odds.py`
`record_ingestion_run` now INSERTs the 10 schema columns (mirrors
`statsbomb.record_ingestion_run`). Docstring + inline comment document the
alignment and why the extra `IngestionResult` fields are not persisted.

### Repair B — `src/football_intelligence/cli.py`
- `_safe_record_run` logs at **ERROR** and **re-raises `DatabaseError`**
  (chained via `raise ... from exc`), so a missing audit record is never
  silently swallowed. No credentials are logged.
- The 6 call sites (3 commands × success/failure path) now catch `DatabaseError`:
  - **Success path** ("import succeeded, audit lost"): distinct `ERROR:` to
    stderr, exit code 1 (data committed, audit trail not recorded).
  - **Failure path** ("import failed"): the original `IngestionError` is still
    printed/logged verbatim, **and** the provenance failure is reported as a
    separate `ERROR: failed to record audit trail: …` line — original cause
    preserved, provenance failure surfaced.

`DatabaseError` was already imported in `cli.py`.

### Tests — `tests/test_football_data_results.py`
Six hermetic tests (temp DB, mocked network — never touch production):

| Test | Proves |
|---|---|
| `test_record_ingestion_run_odds_aligned_with_schema` | Repair A: odds recorder writes an `ingestion_runs` row (previously raised). |
| `test_safe_record_run_surfaces_provenance_failure` | Repair B: recording failure re-raises `DatabaseError` chained to the cause. |
| `test_cli_results_import_records_run_row` | End-to-end: a successful import writes `provider=football-data/dataset_name=results` audit row. |
| `test_cli_results_import_success_but_audit_failure_is_distinct` | Data committed but audit lost → rc≠0, distinct stderr, no silent 0. |
| `test_cli_results_import_failure_preserves_original_error` | On failure, original error preserved *and* audit failure surfaced distinctly. |
| `test_ingest_rollback_on_mid_load_failure` | Mid-load failure rolls back ALL writes; FK + integrity intact. |
## 4. Safe dry-run & validation

### Live network (Step 4/5 as specified) — **unavailable in this sandbox**
The environment intercepts HTTPS with a self-signed (MITM) certificate and
returns **HTTP 403 Forbidden** for `www.football-data.co.uk`, even with an
unverified SSL context. The *live* 1617/1516 download therefore cannot run here.
This is an environment limitation, not a code defect.

### Hermetic real-scale substitute (real fixture identities, disposable copies)
Because the live CSV can't be fetched, `_p15_dryrun_hermetic.py` reads the
**380 real StatsBomb 2015/16 fixtures** (dates + canonical team names + scores)
out of a **copy** of production, synthesizes Football-Data CSVs from them, and
imports against **disposable copies** of production (the adapter downloader is
patched in-process only; separate from pytest). Results:

```
prod counts: {competitions:2, teams:20, matches:381, data_sources:4, ingestion_runs:4}
statsbomb 1516 fixtures read: 380 | goal-sum: 1026

STEP 4 — 1617 (no statsbomb 1617 counterpart)
  first import : read=380 inserted=380 updated=0 skipped=0
  ingestion_runs row: (football-data, results, success, r=380, ins=380, upd=0)
  idempotent   : inserted=0 updated=380              -> True
  ingestion_runs(results) rows after: 2
  -> 380 inserted; idempotent re-run updates 380/0; 2 audit rows recorded. PASS

STEP 5 — 1516 (cross-source against real StatsBomb 2015/16)
  import       : read=380 inserted=0 updated=0 skipped=380
  football-data 1516 rows: 0    (all skipped by cross-source guard)
  statsbomb 1516 rows   : 380    (untouched)
  statsbomb 1516 goal-sum: 1026 -> 1026 (unchanged)
  ingestion_runs latest: (success, r=380, ins=0, upd=0)
  -> cross-source dedup engaged (380 skipped); StatsBomb rows unchanged. PASS

PROD sha256 before == after: 725af574… == 725af574…  -> UNCHANGED: True
```

**Evidence notes:**
- `records_skipped` is **not** persisted in `ingestion_runs` (Repair A
  alignment); it lives on `IngestionResult` in memory. The 1516 audit row
  therefore shows `read=380, inserted=0, updated=0` / `status=success` — the
  380 skips are a run-level result, not a persisted column, consistent with the
  11-column schema and production.
- `ingest()` does **not** write the `ingestion_runs` row — the **CLI** does, via
  `_safe_record_run`/`record_ingestion_run`. The dry-run calls
  `record_ingestion_run` after each `ingest()` to exercise that CLI provenance
  step at scale (exactly what `main([...])` does).
- The cross-source guard normalizes **both** incoming and stored team names
  (`_find_matches_for_reconciliation`), so verbatim StatsBomb team names
  engage a clean 380/380 match.

### Unit suite
```
tests/test_football_data_results.py ...........  50 passed
full suite (341 tests) .........................  341 passed
```
No existing test regressed from the shared `_safe_record_run` change: it only
diverts on *genuine* recording failures, which no test triggers on an
initialized temp DB.

---

## 5. Files changed

**Authored by this phase (tracked `M` vs HEAD):**
```
 M src/football_intelligence/cli.py                       (Repair B: surface failures)
 M src/football_intelligence/data/ingestion/odds.py       (Repair A: 11-col INSERT)
```

**New / appended (untracked `??`):**
```
?? phase15_report.md
?? tests/test_football_data_results.py                    (+6 hermetic tests; pre-existed untracked)
```
`src/football_intelligence/data/ingestion/football_data_results.py` and the
other `tests/test_*.py` files are **untracked and untouched** by this phase.

**External / pre-existing (NOT authored here):** `src/football_intelligence/features/engineering.py`
shows as modified (`+167/-22` vs HEAD) but its diff adds competition/season
filters to `FeatureEngineer.compute_all_features_from_db` — unrelated to
provenance/odds/ingestion and referenced by no Phase 15 change. It is left
unmodified by this phase; the full suite (incl. `test_competition_filtering`,
`test_prediction_lr`) passes with it as-is.

All disposable scratch scripts created during this phase (`_p15_inspect.py`,
`_p15_record_check.py`, `_p15_apply_edits.py`, `_p15_apply_edits_cli.py`,
`_p15_append_tests_{1,2}.py`, `_p15_fix_test.py`, `_p15_edit_dryrun.py`,
`_p15_dryrun.py`, `_p15_dryrun2.py`, `_p15_dryrun_hermetic.py`,
`_p15_verify.py`) have been removed before completion; disposable copy DBs
lived in the OS temp dir and are covered by `.gitignore` (`*.db`).

---

## 6. Assumptions & limitations
- **No migration added.** The repair aligns the orphaned `odds.py` recorder to
  the declared 11-column schema rather than inventing a V3 migration.
  Persisting skipped/ambiguous/invalid counts would be a deliberate schema
  decision (new migration + backfill + updating both recorders) — out of scope
  for a "smallest justified repair."
- **Live network blocked (403/MITM)** in this sandbox; the real-scale claim is
  validated hermetically against real production fixture identities in
  disposable copies.
- **Rollback guarantee** rests on `database_connection` committing on clean
  exit and rolling back on exception
  (`src/football_intelligence/database/connection.py`), which `load()`'s single
  transaction exercises.

