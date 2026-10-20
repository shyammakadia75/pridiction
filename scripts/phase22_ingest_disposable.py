#!/usr/bin/env python3
"""Phase 22 Step 3a: ingest 3 verified FD E0 seasons into a DISPOSABLE db.
Read-only against production."""
from __future__ import annotations
import hashlib, sqlite3, sys, tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from football_intelligence.database import initialize_database  # noqa: E402
from football_intelligence.data.ingestion.football_data_results import (  # noqa: E402
    FootballDataResultsIngestor,
)

PROD = REPO / "data" / "football_intelligence.db"
STAGE_CACHE = REPO / "data" / "raw" / "football_data"
SEASONS = [("1516", "2015/16"), ("1617", "2016/17"), ("1920", "2019/20")]

def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(65536), b""): h.update(c)
    return h.hexdigest()

def q(con, sql):
    return con.execute(sql).fetchone()

print("=== prod SHA before ===")
before = sha(PROD)
print("PROD_SHA256_BEFORE =", before)

disposable = Path(tempfile.mkdtemp(prefix="p22_disp_")) / "phase22_fd.db"
initialize_database(disposable)
con = sqlite3.connect(disposable)
con.row_factory = sqlite3.Row

for sc, label in SEASONS:
    ing = FootballDataResultsIngestor(
        league_code="E0", season_code=sc, db_path=disposable, cache_dir=STAGE_CACHE,
        source_url=f"https://www.football-data.co.uk/mmz4281/{sc}/E0.csv",
    )
    r = ing.ingest()
    print(f"  {label}: read={r.records_read} inserted={r.records_inserted} "
          f"updated={r.records_updated} skipped={r.records_skipped} "
          f"ambiguous={r.records_ambiguous} status={r.status} cached={ing._was_cached}")
    assert r.status == "success"
    assert r.records_read == 380
    assert r.records_inserted == 380 and r.records_skipped == 0

print("\n=== aggregate counts ===")
print("  matches (FD):", q(con, "SELECT COUNT(*) FROM matches WHERE provider='football-data'"))
print("  teams (FD):", q(con, "SELECT COUNT(*) FROM teams WHERE provider='football-data'"))
print("  competitions:", q(con, "SELECT COUNT(*) FROM competitions"))
print("  data_sources:", q(con, "SELECT COUNT(*) FROM data_sources"))
for row in con.execute(
    "SELECT season_name, COUNT(*) c, MIN(match_date) mn, MAX(match_date) mx FROM matches "
    "WHERE provider='football-data' GROUP BY season_name ORDER BY season_name"
):
    print("   ", dict(row))

total = q(con, "SELECT COUNT(*) FROM matches WHERE provider='football-data'")[0]
assert total == 1140, f"expected 1140, got {total}"
dups = q(con, "SELECT COUNT(*) - COUNT(DISTINCT provider_match_id) FROM matches WHERE provider='football-data'")[0]
assert dups == 0
dupfix = q(con, "SELECT COUNT(*) FROM (SELECT match_date, home_team_id, away_team_id, COUNT(*) c FROM matches WHERE provider='football-data' GROUP BY match_date,home_team_id,away_team_id HAVING c>1)")[0]
assert dupfix == 0
neg = q(con, "SELECT COUNT(*) FROM matches WHERE home_goals<0 OR away_goals<0")[0]
assert neg == 0
ld = q(con, "SELECT COUNT(DISTINCT match_date||'|'||home_team_id||'|'||away_team_id) FROM matches WHERE provider='football-data'")[0]
assert ld == 1140
print("[OK] 1140 matches; 0 dup provider_match_id; 0 dup fixtures; 0 neg goals; 1140 distinct logical fixtures")

print("\n=== idempotency: re-ingest 2016/17 (cache hit) ===")
r2 = FootballDataResultsIngestor(
    league_code="E0", season_code="1617", db_path=disposable, cache_dir=STAGE_CACHE,
    source_url="https://www.football-data.co.uk/mmz4281/1617/E0.csv",
).ingest()
print(f"  inserted={r2.records_inserted} updated={r2.records_updated} skipped={r2.records_skipped}")
assert r2.records_inserted == 0 and r2.records_updated == 380
assert total == q(con, "SELECT COUNT(*) FROM matches WHERE provider='football-data'")[0]
print("[OK] idempotent: 0 inserts / 380 updates / totals unchanged")

print("\n=== integrity ===")
ic = con.execute("PRAGMA integrity_check").fetchone()[0]
fk = con.execute("PRAGMA foreign_key_check").fetchall()
bad = con.execute(
    "SELECT provider_team_id, COUNT(DISTINCT team_name) FROM teams WHERE provider='football-data' "
    "GROUP BY provider_team_id HAVING COUNT(DISTINCT team_name)>1"
).fetchall()
print("  integrity_check:", ic, "| fk violations:", len(fk), "| multi-name team ids:", len(bad))
assert ic == "ok" and len(fk) == 0 and len(bad) == 0
print("[OK] integrity + foreign keys + single-name-per-team clean")

after = sha(PROD)
print("\nPROD_SHA256_AFTER =", after)
assert after == before, "PROD DB CHANGED -- BLOCKER"
print("[OK] production DB SHA-256 unchanged")
con.close()
print("\nSTEP 3a DONE")
