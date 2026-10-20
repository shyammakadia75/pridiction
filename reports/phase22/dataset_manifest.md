# Phase 22 — Dataset Manifest: Multi-season Premier League (Read-Only)

> **Status:** Steps 1-5 complete (verified green). **No production writes performed.**
> Production DB `data/football_intelligence.db` left strictly read-only throughout.
>
> **PROD_SHA256 =** `725af574b88dc033bd3b00928de503f9bd9a836fbe4252832b7812dce0ab3122`
> (verified unchanged before/after every step — confirmed **6x**, including after
> the full test-suite run.)

## 1. Purpose

Build a reproducible, validated multi-season Premier League dataset in a
**disposable** SQLite DB, strictly read-only against the **production** DB, using
Football-Data `E0` CSVs (2015/16, 2016/17, 2019/20) as the FD source and a
read-only replica of the StatsBomb 2015/16 fixture set as the SB reference.

## 2. Exact reproduction commands

All commands run from the repo root `C:\pridection\pridiction` using `.venv`.

```powershell
# 0. Re-confirm prod DB is read-only / hash gate
$env:PYTHONIOENCODING='utf-8'
.\.venv\Scripts\python.exe scripts/phase22_prod_probe.py | Tee-Object phase22_prod_probe_recheck.log

# 1. Source cache + SHA-256 verification (reproducibility check runs too)
.\.venv\Scripts\python.exe scripts/phase22_verify.py   2>&1 | Tee-Object phase22_step1_verify.log
.\.venv\Scripts\python.exe scripts/phase22_stage.py    2>&1 | Tee-Object phase22_step1_stage.log

# 2. Disposable, FD-only ingestion (idempotency + integrity assertions)
.\.venv\Scripts\python.py scripts/phase22_ingest_disposable.py  2>&1 | Tee-Object phase22_step3a.log

# 3. Cross-source reconciliation (SB-seeded disposable DB + FD overlap)
.\.venv\Scripts\python.py scripts/phase22_cross_source.py    2>&1 | Tee-Object phase22_step3b.log

# 4. Tests (focused + full suite)
.\.venv\Scripts\python.py -m pytest tests/test_team_identity.py -v
.\.venv\Scripts\python.py -m pytest -q
```

> **NB:** Step 1's fresh-download path fails inside the sandbox
> (`SSL CERTIFICATE_VERIFY_FAILED` / connection refused). Sources are therefore
> staged from a verified **local cache** (`%TEMP%\epl_fd`) with provenance
> documented transparently in `data/raw/football_data/_source_manifest.json` —
> not mislabeled as a fresh download.

## 3. Source files

Canonical URLs + acquisition status (per `_source_manifest.json`):

| Season | Canonical URL | Download status | Local cache SHA-256 | Bytes | Rows | Staged path |
|--------|---------------|-----------------|---------------------|-------|------|-------------|
| 2015/16 (1516) | `https://www.football-data.co.uk/mmz4281/1516/E0.csv` | FAILED (SSL/conn refused in sandbox; staged from local cache) | `bd3502a18c38a1597fd9af62e2366b4015006d3528dd4d18b311bd6237bbc085` | 100656 | 380 | `data/raw/football_data/E0/1516.csv` |
| 2016/17 (1617) | `https://www.football-data.co.uk/mmz4281/1617/E0.csv` | FAILED (SSL/conn refused in sandbox; staged from local cache) | `9625a7652b5f98fbd3e2e4d378c851fc246693f3343e34a72428d5b6e864d3e0` | 99990 | 380 | `data/raw/football_data/E0/1617.csv` |
| 2019/20 (1920) | `https://www.football-data.co.uk/mmz4281/1920/E0.csv` | FAILED (SSL/conn refused in sandbox; staged from local cache) | `100037618b94f94057400bb02bf6bac4ef74ddaa58cde4b38370839c39caee61` | 175475 | 380 | `data/raw/football_data/E0/1920.csv` |

**Staged SHA-256 verified** via `Get-FileHash` (matches manifest `local_sha256`,
case-insensitive) before any ingestion.

## 4. Per-file verification (Step 1)

For each staged CSV (380 rows × 20 teams per season):

| Check | 1516 | 1617 | 1920 |
|-------|------|------|------|
| Distinct teams | 20 | 20 | 20 |
| Duplicate `(Date,Home,Away)` fixtures | 0 | 0 | 0 |
| Score errors (FTR mismatch / invalid digits) | 0 | 0 | 0 |
| Goal sanity (non-negative ints, 0-9) | 0 errors | 0 errors | 0 errors |
| Unparseable dates | 0 | 0 | 0 |

`_normalize_match_date` handled the date-format drift across the three files:
`dd/mm/yy` (1516/1617) -> ISO; `dd/mm/YYYY` (1920) -> ISO; the extra `Time` column
in `E0_1920` is harmlessly ignored by `DictReader`.

## 5. Team-identity mapping (Step 2)

Canonical name normalization lives in `data/ingestion/odds.py::_normalize_team_name`
(`PROVIDER="football-data"`; `TEAM_ALIASES` dict), imported by both FD and SB
adapters. FD short names fold to SB-style lowercase canonicals via aliases.

**Injectivity gate:** of the 26 distinct raw team names across the 3 seasons,
**zero normalized collisions** (mapping is 1:1/injective); idempotent reruns
confirmed. `normalize_team_name` was left **untouched** (no real collision found).

### Canonical team mapping table

| FD short name | Normalized canonical | SB canonical | Alias covered? |
|---------------|----------------------|--------------|----------------|
| Arsenal | arsenal | Arsenal | bare |
| Aston Villa | aston villa | Aston Villa | bare |
| Bournemouth | afc bournemouth | AFC Bournemouth | yes |
| Brighton | brighton and hove albion | Brighton & Hove Albion | yes |
| Burnley | burnley | Burnley | bare |
| Chelsea | chelsea | Chelsea | bare |
| Crystal Palace | crystal palace | Crystal Palace | bare |
| Everton | everton | Everton | bare |
| Hull | hull city | Hull City | bare (FD-only; no SB overlap) |
| Leicester | leicester | Leicester City | **no - gap** |
| Liverpool | liverpool | Liverpool | bare |
| Man City | manchester city | Manchester City | yes |
| Man United | manchester united | Manchester United | yes |
| Middlesbrough | middlesbrough | Middlesbrough | bare |
| Newcastle | newcastle united | Newcastle United | yes |
| Norwich | norwich | Norwich City | **no - gap** |
| Sheffield United | sheffield united | Sheffield United | yes (1920 only) |
| Southampton | southampton | Southampton | bare |
| Stoke | stoke | Stoke City | **no - gap** |
| Sunderland | sunderland | Sunderland | bare |
| Swansea | swansea | Swansea City | **no - gap** |
| Tottenham | tottenham hotspur | Tottenham Hotspur | yes |
| Watford | watford | Watford | bare |
| West Brom | west brom | West Bromwich Albion | **no - gap** |
| West Ham | west ham united | West Ham United | yes |
| Wolves | wolverhampton wanderers | Wolverhampton Wanderers | yes (1920 only) |

14 "bare" names need no alias (FD spelling already normalises to a stable
canonical distinct from all other FD names; FD-only stable canonical).

### Unresolved aliases / excluded fixtures

- **5 clubs** lack an FD->SB alias, so they do NOT reconcile by
  `(date, canonical home, canonical away)` against StatsBomb:
  **Leicester, Norwich, Stoke, Swansea, West Brom**
  (FD bare e.g. `leicester` vs SB `leicester city`).
- **No fixtures were excluded.** Per the fail-closed rule, ambiguous/mismatched
  fixtures are **preserved as duplicate rows, not discarded**.

## 6. Disposable ingestion results (FD-only, Step 3a)

1140 matches (3 seasons x 380) ingested into a fresh disposable SQLite DB.

| Metric | Value |
|--------|-------|
| Matches inserted | 1140 |
| Duplicate `provider_match_id` | 0 |
| Duplicate `(date, home, away)` fixtures | 0 |
| Negative goals | 0 |
| Distinct logical fixtures | 1140 |
| Idempotent rerun | 0 inserts / 380 updates per season |
| `PRAGMA integrity_check` | ok |
| Foreign keys | clean (every home/away -> teams row) |
| `provider_team_id` -> `team_name` | 1:1 (single name per team) |

Upsert is a 3-phase transaction per season:
1. same `provider + fixture_id` -> update;
2. existing `provider_match_id` -> update;
3. cross-source `(date, home_norm, away_norm)` exists -> skip;
4. else insert.

## 7. Cross-source reconciliation (Step 3.5)

Replicated prod's StatsBomb 2015/16 set (read-only SELECT on
`data/football_intelligence.db`) into a **second** disposable DB (SB-seeded).
Then ingested Football-Data 2015/16 *over* that SB-seeded DB.

| Outcome | Count |
|---------|-------|
| FD 2015/16 fixtures matching an SB fixture (skip/dedup) | 210 |
| FD 2015/16 fixtures inserted as duplicate rows | 170 |
| Ambiguous (same date, identical names) | 0 |
| Total FD 2015/16 fixtures | 380 |

SB 2015/16 reference (prod, read-only): 20 teams, 380 matches,
`2015-08-08` -> `2016-05-17` (full season). SB team list: AFC Bournemouth,
Arsenal, Aston Villa, Chelsea, Crystal Palace, Everton, **Leicester City**,
Liverpool, Manchester City, Manchester United, Newcastle United, **Norwich City**,
Southampton, **Stoke City**, Sunderland, **Swansea City**, Tottenham Hotspur,
Watford, **West Bromwich Albion**, West Ham United.

The 170 inserted-as-duplicate rows are exactly those involving the
**5 mismatched-name clubs** (Section 5): FD `leicester` != SB `leicester city`,
etc. They are **preserved as duplicates** (fail-closed), not silently discarded.
The shared `_normalize_team_name` was left untouched (Section 5 gate = no real
collision).

**Final match counts (disposable DB):** 380 x3 seasons = **1140 matches**
(1516 + 1617 + 1920); 20 distinct teams per season.

## 8. Integrity results

- Disposable DB `PRAGMA integrity_check` = `ok`.
- FK enforcement on; every `home_team_id`/`away_team_id` references `teams.id`.
- `provider_team_id` -> `team_name` is 1:1 within each provider (no team renamed
  mid-table).
- No duplicate logical fixture keys anywhere in the 1140-row set.

## 9. Production-read-only evidence (prod-hash gate)

- `PROD_DB = data/football_intelligence.db`
- `PROD_SHA256 = 725af574b88dc033bd3b00928de503f9bd9a836fbe4252832b7812dce0ab3122`
- Verified **unchanged before/after every step** (6 checks): Step-1 verify,
  Step-1 stage, prod-probe, Step-3a ingest, Step-3b cross-source, and after the
  full test-suite run. Confirmed via `Get-FileHash -Algorithm SHA256`.
- Prod connection opened `file:{db}?mode=ro` (read-only URI) in all scripts.

## 10. Modelling suitability (Step 5 — chronological-split sanity)

Per-season date analysis of the 3 staged FD files:

| Season | File chronological? | Bad dates | Date range | Notes |
|--------|---------------------|-----------|------------|-------|
| 1516 | yes (non-decreasing) | 0 | 2015-08-08 -> 2016-05-17 | normal ~13-day matchweek gaps |
| 1617 | yes (non-decreasing) | 0 | 2016-08-13 -> 2017-05-21 | normal ~13-day matchweek gaps |
| 1920 | yes (non-decreasing) | 0 | 2019-08-09 -> 2020-07-26 | **100-day COVID gap** 2020-03-09 -> 2020-06-17 |

- **Feasible:** each season's rows are already chronological; a date-based train
  vs later-dates test split is valid (strictly-earlier-dates-only, no leakage).
- **COVID-19 discontinuity (1920):** a 100-day gap mid-season (last match
  2020-03-09, resumed 2020-06-17). Pre/post-gap form is non-contiguous — any
  model must treat the break as a reset (e.g. drop the gap window, reset
  rolling features).
- "Duplicate dates" per season are matchweek pairs (Sat+Sun) / midweek doubles —
  not collisions; distinct `fixture_id` keys still disambiguate.

### Limitations / blocking issues for any FD<->SB merge

1. **No odds for 2017-2020 windows** in this cache; `odds.py` coverage limited.
2. **Missing kick-off `Time`** except in `E0_1920` (harmlessly ignored).
3. **Bare-name cross-source dedup gap:** the 5 clubs in Section 5 must receive
   reviewed aliases before any FD<->SB merged dataset is trusted for modelling.
4. **Sandbox network** refused fresh `football-data.co.uk` downloads; reproduction
   depends on the staged local cache (provenance in
   `data/raw/football_data/_source_manifest.json`). Re-run source acquisition
   locally where network is available to refresh.


