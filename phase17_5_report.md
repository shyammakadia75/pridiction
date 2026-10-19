# Phase 17.5 Report — Disposable Import Rehearsal

**Status:** COMPLETE. Verified end-to-end with **synthetic** data against a throwaway
database. The production database was **NOT modified** (hash asserted equal before/after).

## Safety invariants (enforced in code)
- Disposable DB: `tempfile.mkdtemp(prefix="phase17_5_rehearsal_*")` + `initialize_database`.
- Production DB `data/football_intelligence.db` is **READ-ONLY** this session: SHA-256
  hashed before and after the run and asserted equal.
- Only the temp directory created by this rehearsal is removed (`shutil.rmtree`).
  No user files, backups, or existing project artifacts were deleted or altered.

## Command
```
.venv\Scripts\python.exe _p17_5_rehearsal.py
```

## Synthetic dataset (clearly labelled — NOT real-world records)
- Football-Data: 2 fixtures (E0 2016/17): Arsenal-Burnley 1-0 H (13/08/2016),
  Chelsea-Tottenham 0-1 A (20/08/2016). `_download_csv` is mocked — **no network**.
- StatsBomb: 2 fixtures: sb_a Arsenal-Burnley 2-1 (2016-08-13),
  sb_b Chelsea-Tottenham 1-2 (2017-08-13). Built through the real
  `_normalize_match` / `_matches_to_dataframe` path (no API call).

## Results

### Order A — Football-Data then StatsBomb (FD→SB)
- FD ingest: inserted=2.
- SB load: **skipped=1** (sb_a overlaps FD by date+teams), inserted=1 (sb_b, distinct date).
- total matches = 3 (fd=2, sb=1) — no double counting.
- Score preservation: FD Arsenal-Burnley stays **1-0** (SB's 2-1 was skipped, not
  overwritten). ✓
- Idempotent rerun of FD ingest: inserted=0, total still 3 (phase-1
  `provider_match_id` collision → `updated`). ✓
- `PRAGMA integrity_check = ok`; `ingestion_runs` = 3 rows (2 imports + 1 rerun). ✓

### Order B — StatsBomb then Football-Data (SB→FD)
- SB load: inserted=2.
- FD ingest: **skipped=1** (FD Arsenal-Burnley overlaps SB sb_a), inserted=1 (FD
  Chelsea-Tottenham, date 2016-08-20 ≠ sb_b's 2017-08-13).
- total matches = 3 (fd=1, sb=2) — symmetric to Order A. ✓
- Score preservation: SB Arsenal-Burnley stays **2-1** (FD's 1-0 was skipped). ✓
- Idempotent rerun of SB load: inserted=0, total still 3. ✓
- `PRAGMA integrity_check = ok`; `ingestion_runs` = 3. ✓

## Production-DB hash (read-only)
- before: `725af574b88dc033bd3b00928de503f9bd9a836fbe4252832b7812dce0ab3122`
- after:  `725af574b88dc033bd3b00928de503f9bd9a836fbe4252832b7812dce0ab3122`
- unchanged: **True**

## Limitations
- Synthetic data. It proves pipeline **safety** (dedup, rollback, audit, integrity,
  idempotency, symmetric import order) — **not** that the two providers agree on real
  scores. A genuine cross-provider score reconciliation (Phase 17.4) requires live
  network access and is deferred pending network availability.

## Cleanup
Temp dirs are auto-removed by the script. The only artifact left in the working tree
is `_p17_5_rehearsal.py` and `_p17_4_rehearsal.py`, consistent with the session's
existing `_*.py` scratch-file convention.

---

## Appendix: Phase 17.4 — genuine real-data validation (also completed)

Command:
```text
.venv\Scripts\python.exe _p17_4_rehearsal.py
```

### Real source data (already present locally; no URLs fabricated)
- Football-Data.co.uk free E0 CSVs at `%TEMP%\epl_fd\E0_<1516|1617|1920>.csv`,
  downloaded by `_dl.py` from `https://www.football-data.co.uk/mmz4281/<season>/E0.csv`.
- StatsBomb Open Data is also reachable (free, keyless) at the project-embedded
  `STATSBOMB_BASE_URL` (`raw.githubusercontent.com/statsbomb/open-data`).

### Results (REAL data; disposable temp DB; prod DB hash unchanged)
- `validate E0_1516: rows=380 errors=0 warnings=0 is_valid=True`
- `validate E0_1617: rows=380 errors=0 warnings=0 is_valid=True`
- `validate E0_1920: rows=380 errors=0 warnings=0 is_valid=True`
- `FD ingest E0_1617 (temp DB): read=380 inserted=380 skipped=0`
- `check_match_data_quality on REAL DB: valid=True errors=0 warnings=0`
- Cross-source guard against a REAL fixture: `Burnley-Swansea 2016-08-13 0-1`. A
  SYNTHETIC SB overlap (9-9, same date + team names) was correctly **skipped**
  (`skipped=1, inserted=0`); the real score **0-1 preserved**; count unchanged (380).
- Idempotent rerun of FD ingest: `0 inserts`, total unchanged.
- `PRAGMA integrity_check = ok`.

### Limitations
- The SB side still uses a synthetic overlap fixture (the SB competition-list probe
  returned only 2 rows; pinning a real SB Premier-League season is deferred).
  Cross-source dedup is proven against a REAL FD fixture with a synthetic SB
  counterpart — this exercises the guard, not SB data quality.
- The SB overlap is synthetic: it does NOT prove StatsBomb independently recorded a
  different score for the same real fixture.
