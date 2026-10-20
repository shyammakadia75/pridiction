# Phase 22 — Expand the Verified Real Historical Dataset (Audit)

## 1. Objective
Add **multiple genuine Premier-League seasons** to the modelling dataset so the
Phase-21 comparison (and Phases 23-24 models) can be evaluated with a real
**multi-season walk-forward** split instead of a single 380-match / 76-test window.
Constraints from the master plan: reproducible data provenance, leakage-safe
evaluation, **disposable databases for all ingestion experiments**, and the
production DB held strictly read-only.

## 2. Data sources available
| source | what's available | location | reproducible? |
|---|---|---|---|
| StatsBomb Open Data | 1 PL season pair (2015/16+2016/17, 380 matches) | `data/football_intelligence.db` | yes (committed DB) |
| Football-Data.co.uk E0 CSVs | `E0_1516.csv`, `E0_1617.csv`, `E0_1920.csv` (3 seasons) | `%TEMP%\epl_fd\` (ephemeral) | **no** — OS temp |
| Football-Data.co.uk archive | E0 seasons 2015/16 → 2024/25 | `https://www.football-data.co.uk` (network reachable) | yes (downloadable) |
| StatsBomb open-data | `competitions.json` + `27.json` | `data/raw/statsbomb/` | partial (27.json cached) |

Prod DB currently holds **only the StatsBomb PL pair (2015/16, 2016/17)** — exactly
the two seasons Phase-21 evaluates on. There is **no multi-season real data in prod**.

## 3. Ingestion adapters (present, tested, idempotent)
- `src/football_intelligence/data/ingestion/football_data_results.py` —
  `FootballDataResultsIngestor(league_code='E0', season_code=<...>)`. Maps
  `Date/HomeTeam/AwayTeam/FTHG/FTAG/FTR` → normalized rows. Downloads to a
  deterministic cache under `data/raw/football_data/` (`RAW_DATA_DIR/football_data`).
- `_compose_provider_match_id` builds a deterministic idempotent composite key
  `football-data|{season_code}|{date}|{home_norm}|{away_norm}`.
- 3-phase upsert (`_upsert_match` + `_find_matches_for_reconciliation`): (1) same
  provider+fixture_id → update, (2) cross-source `(date, canonical home, canonical away)`
  existing → skip, (3) else insert; one transaction per `load`.
- `statsbomb.py` — `StatsBombIngestor` (cache/network), same 3-phase upsert.
- Per-season ingest is already exposed via CLI flags; expansion is mechanical **once**
  reproducible CSV sources are staged.

## 4. Team-identity mapping
- Canonical team-name normalization lives in **`data/ingestion/odds.py:: _normalize_team_name`**
  (imported by both adapters) — the single team-identity bridge.
- Football-Data uses short names ("Man United", "Man City", "Spurs"); StatsBomb uses
  canonical names ("Manchester United", "Manchester City", "Tottenham Hotspur").
  Cross-source dedup compares **normalized** names across providers by
  `(date, home_norm, away_norm)`.
- **RISK (untested on these CSVs):** team naming drift across seasons
  (e.g. "Man United" vs "Man Utd") and club renames/relocations. The cross-source
  guard is proven in `test_cross_source_reconciliation.py` (synthetic); stability of
  real FD short names across seasons must be validated before trusting dedup counts.

## 5. Existing test coverage (audit)
- `tests/test_football_data_results.py` — FD normalization, FTR/score consistency,
  fixture-key synthesis, idempotent upsert counts.
- `tests/test_statsbomb.py` — SB JSON parsing, team/competition upsert.
- `tests/test_cross_source_reconciliation.py` (10) — cross-provider skip/ambiguous
  dedup by normalized team name + date.
- `tests/test_competition_filtering.py` — season/competition filtering at load.
- `tests/test_odds.py`, `tests/test_quality.py`, `tests/test_validation.py`.
- Phase 21 added `tests/test_model_compare.py` + `tests/test_cli_model_compare.py` (16).

## 6. Gaps / risks
1. **Reproducible source location**: the 3 real E0 CSVs currently live only in
   `%TEMP%\epl_fd` (ephemeral). The adapter cache `data/raw/football_data/` is empty.
   Ingestion experiments must stage CSVs in a stable, gitignored cache first.
2. **Name-dictionary verification**: before multi-season merge, audit the set of
   `HomeTeam/AwayTeam` short names across E0_1516/E0_1617/E0_1920 and confirm
   `normalize_team_name` maps each club to exactly one canonical identity per season.
3. **Prod contamination**: do NOT ingest E0 CSVs into `data/football_intelligence.db`.
   All expansion must target a **disposable** DB (a fresh sqlite file under temp),
   and re-run the Phase-21 comparison there.
4. **Walk-forward leakage**: any expanded DB must keep the Phase-20/21 protocol
   (Poisson/Elo "strictly earlier dates only"; LR fixed train split).

## 7. Proposed implementation plan (pending go-ahead; not started)
1. **Stage sources**: copy E0_1516/E0_1617/E0_1920 into `data/raw/football_data/`
   (gitignored cache) and download 2017/18–2022/23 so the adapter skips re-download.
2. **Team-name dictionary**: build the FD short-name → canonical set from staged CSVs;
   assert each PL club maps to exactly one canonical name per season; fix
   `normalize_team_name` only if a real collision is found.
3. **Disposable multi-season DB**: create a temp DB, ingest staged E0 seasons via
   `FootballDataResultsIngestor`, assert counts + dedup integrity + prod SHA
   unchanged.
4. **Walk-forward evidence**: re-run `run_model_comparison` on the expanded
   disposable DB over a later-season test window (e.g. 2018/19+) as out-of-sample
   evidence of stability; persist to `reports/phase22/`.
5. Only after 1–4 are green is the expanded dataset suitable for Phase-23 model work
   / Phase-24 ensembles.

**This audit is read-only; no ingestion was performed and the production DB was not
touched (SHA `725af574…` verified unchanged).**

---

## 9. Implementation status (Steps 1-5 complete)

**Status: Steps 1-5 VERIFIED GREEN — do NOT begin Phase 23 (stronger models)
until explicit approval; no production DB writes planned.**

- **Step 1 (source cache):** staged `data/raw/football_data/E0/{1516,1617,1920}.csv`,
  SHA-256 verified byte-for-byte. Each: 380 rows x 20 teams, 0 dupes/score errors/
  date failures. `_source_manifest.json` documents canonical URLs + acquisition
  method (fresh-download reproducibility check ran; sandbox network refused requests
  -> staged from verified local cache transparently documented, not mislabeled).
- **Step 2 (team-identity gate):** 26 distinct FD names -> 26 distinct canonicals,
  **0 collisions** (injective). No prod normalizer edit needed. 14 bare names
  (FD-only stable canonicals); cross-source FD<->SB gap documented but no real
  collision.
- **Step 3a (FD-only disposable ingestion):** 1140 matches; 0 dup
  `provider_match_id`; 0 dup `(date,home,away)`; 0 negative goals; 1140 distinct
  fixtures; idempotent rerun (0 inserts / 380 updates per season); `PRAGMA
  integrity_check`=ok; clean FKs; single `team_name` per `provider_team_id`.
- **Step 3.5 (cross-source reconciliation):** SB 2015/16 replicated read-only into a
  SECOND disposable DB; over SB-seeded DB, FD 2015/16 = 210 skipped + 170 inserted as
  duplicates + 0 ambiguous. 5 mismatched-name clubs (Leicester/Norwich/Stoke/Swansea/
  West Brom). Duplicates preserved, not discarded. Shared `_normalize_team_name`
  left untouched (no real collision per gate).
- **Step 2.6 (tests):** added `tests/test_team_identity.py` (29 cases) covering alias
  resolution, case/punctuation, unknown-name retention, injectivity gate,
  cross-season name stability, and the 5-club cross-source gap (fail-closed
  regression marker). New tests: 29 passed. Full suite: 462 passed, 0 failures.
- **Step 4 (manifest):** written to `reports/phase22/dataset_manifest.md` (full:
  seasons, URLs+status, SHA-256/size/rows per file, pre/post row counts, canonical
  team mapping table, unresolved aliases, excluded fixtures + reasons,
  reconciliation results, final match counts, integrity results, exact
  reproduction commands, prod-hash evidence).
- **Step 5 (modelling suitability):** per-season rows are file-chronological (date-based
  train/later-dates split feasible, no leakage). 2019/20 shows a 100-day COVID-19 gap
  (2020-03-09 -> 2020-06-17). Limitations: no odds for 2017-2020, missing kickoff
  `Time` except 1920, bare-name FD<->SB dedup gap requiring alias expansion before any
  merge. Full analysis in the manifest (Section 10).

**Open question requiring approval:** sandbox network refused fresh
`football-data.co.uk` downloads -- sources are staged from a verified local cache
with provenance documented in `_source_manifest.json`. If the user wants to retry
network acquisition, that is the only blocker to full network reproducibility.

