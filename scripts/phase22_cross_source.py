#!/usr/bin/env python3
"""Phase 22 Step 3.5: explicit cross-source reconciliation report.
Replicates prod's SB 2015/16 fixtures (read-only SELECT from prod) into a DISPOSABLE
db, then ingests FD 2015/16 and reports skip/insert/ambiguous outcomes. No prod writes."""
from __future__ import annotations
import hashlib, sqlite3, sys, tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from football_intelligence.database import initialize_database  # noqa: E402
from football_intelligence.data.ingestion.football_data_results import (  # noqa: E402
    FootballDataResultsIngestor,
)
from football_intelligence.data.ingestion.odds import _normalize_team_name  # noqa: E402

PROD = REPO / "data" / "football_intelligence.db"
STAGE_CACHE = REPO / "data" / "raw" / "football_data"

def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(65536), b""): h.update(c)
    return h.hexdigest()

before = sha(PROD)
print("PROD_SHA256_BEFORE =", before)

# --- read SB 2015/16 rows from prod (READ ONLY) ---
prodr = sqlite3.connect(f"file:{PROD}?mode=ro", uri=True)
prodr.row_factory = sqlite3.Row
sb_comp = prodr.execute(
    "SELECT id,provider,provider_competition_id,competition_name,country,created_at,updated_at "
    "FROM competitions WHERE provider='statsbomb'").fetchall()
sb_team = prodr.execute(
    "SELECT id,provider,provider_team_id,team_name,country,created_at,updated_at "
    "FROM teams WHERE provider='statsbomb'").fetchall()
sb_match = prodr.execute(
    "SELECT id,provider,provider_match_id,competition_id,season_name,match_date,home_team_id,"
    "away_team_id,home_goals,away_goals,match_status,created_at,updated_at FROM matches "
    "WHERE provider='statsbomb' AND season_name='2015/2016'").fetchall()
prodr.close()
print(f"read-only from prod: {len(sb_comp)} comps, {len(sb_team)} teams, {len(sb_match)} SB 2015/16 matches")

# --- new DISPOSABLE db; replicate SB 2015/16 rows verbatim ---
disp = Path(tempfile.mkdtemp(prefix="p22_xs_")) / "xs.db"
initialize_database(disp)
con = sqlite3.connect(disp)
con.row_factory = sqlite3.Row
_insert = {
    "competitions": ["id","provider","provider_competition_id","competition_name","country","created_at","updated_at"],
    "teams": ["id","provider","provider_team_id","team_name","country","created_at","updated_at"],
    "matches": ["id","provider","provider_match_id","competition_id","season_name","match_date","home_team_id","away_team_id","home_goals","away_goals","match_status","created_at","updated_at"],
}
with con:
    for t, cols, rows in [("competitions", _insert["competitions"], sb_comp),
                          ("teams", _insert["teams"], sb_team),
                          ("matches", _insert["matches"], sb_match)]:
        for r in rows:
            con.execute(f"INSERT INTO {t} ({','.join(cols)}) VALUES ({','.join('?'*len(cols))})",
                        [r[c] for c in cols])
print(f"replicated SB 2015/16 into disposable: {con.execute('SELECT COUNT(*) FROM matches').fetchone()[0]} matches")

sb_names = {r["team_name"] for r in sb_team}
fd_short_for = {
    "AFC Bournemouth":"Bournemouth","Manchester City":"Man City","Manchester United":"Man United",
    "Newcastle United":"Newcastle","West Ham United":"West Ham","Tottenham Hotspur":"Tottenham",
    "Leicester City":"Leicester","Norwich City":"Norwich","Stoke City":"Stoke",
    "Swansea City":"Swansea","West Bromwich Albion":"West Brom","Sunderland":"Sunderland","Watford":"Watford",
}
mismatch = []
for sb, fd in fd_short_for.items():
    if sb not in sb_names: continue
    c_sb = _normalize_team_name(sb); c_fd = _normalize_team_name(fd)
    if c_sb != c_fd: mismatch.append((fd, sb, c_fd, c_sb))
print(f"  mismatched-name clubs ({len(mismatch)}):")
for fd, sb, cfd, csb in mismatch:
    print(f"    FD {fd!r}->{cfd!r}  vs SB {sb!r}->{csb!r}")

print("\n=== ingest FD 2015/16 over SB-seeded disposable db ===")
r = FootballDataResultsIngestor(
    league_code="E0", season_code="1516", db_path=disp, cache_dir=STAGE_CACHE,
    source_url="https://www.football-data.co.uk/mmz4281/1516/E0.csv",
).ingest()
print(f"  inserted={r.records_inserted} updated={r.records_updated} skipped={r.records_skipped} "
      f"ambiguous={r.records_ambiguous} read={r.records_read}")

print("\n=== sample FD fixtures that DUPLICATED an existing SB fixture (name gap) ===")
fd_dup = con.execute(
    "SELECT m.match_date, ht.team_name AS home, at.team_name AS away, m.home_goals, m.away_goals "
    "FROM matches m JOIN teams ht ON m.home_team_id=ht.id JOIN teams at ON m.away_team_id=at.id "
    "WHERE m.provider='football-data' AND m.match_date IN "
    "(SELECT match_date FROM matches WHERE provider='statsbomb')"
    " ORDER BY m.match_date LIMIT 6").fetchall()
for d in fd_dup:
    print("   ", dict(d))

tot = con.execute("SELECT COUNT(*) FROM matches").fetchone()[0]
fdm = con.execute("SELECT COUNT(*) FROM matches WHERE provider='football-data'").fetchone()[0]
sbm = con.execute("SELECT COUNT(*) FROM matches WHERE provider='statsbomb'").fetchone()[0]
print("\n=== reconciliation summary ===")
print(f"  total matches in disposable: {tot} (SB={sbm}, FD={fdm})")
print(f"  FD 2015/16 read={r.records_read} inserted={r.records_inserted} "
      f"skipped={r.records_skipped} ambiguous={r.records_ambiguous}")
print("  -> inserted>0 means name-dict gap produced duplicates that were")
print("     preserved (NOT silently discarded), per Step 3.5 requirement.")
after = sha(PROD)
assert after == before, "PROD DB CHANGED -- BLOCKER"
print("\nPROD_SHA256_AFTER =", after, "(unchanged)")
con.close()
print("STEP 3.5 DONE")

