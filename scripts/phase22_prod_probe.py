#!/usr/bin/env python3
"""Phase 22: read-only probe of the production DB (NO writes).
Captures prod SHA-256 + SB fixture/team inventory for cross-source reconciliation."""
from __future__ import annotations
import hashlib, sqlite3
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
prod = REPO / "data" / "football_intelligence.db"

def sha(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(65536), b""): h.update(c)
    return h.hexdigest()

print("PROD_DB=", prod.resolve())
print("PROD_SHA256_BEFORE=", sha(prod))
print("PROD_SIZE=", prod.stat().st_size)

con = sqlite3.connect(f"file:{prod}?mode=ro", uri=True)
con.row_factory = sqlite3.Row
print("=== columns ===")
for t in ["competitions", "teams", "matches", "data_sources"]:
    cols = con.execute(f"PRAGMA table_info({t})").fetchall()
    print(t, "->", [c["name"] for c in cols])

print("=== matches by provider+season ===")
for r in con.execute(
    "SELECT provider, season_name, COUNT(*) c, MIN(match_date) mn, MAX(match_date) mx "
    "FROM matches GROUP BY provider, season_name ORDER BY season_name"
):
    print(dict(r))

print("=== teams by provider ===")
for r in con.execute(
    "SELECT provider, COUNT(DISTINCT id) teams, COUNT(DISTINCT provider_team_id) pids FROM teams GROUP BY provider"
):
    print(dict(r))

print("=== SB team_name list (prod, read-only) ===")
for r in con.execute(
    "SELECT DISTINCT team_name FROM teams WHERE provider='statsbomb' ORDER BY team_name"
):
    print("  ", r["team_name"])

print("=== SB match_date range per season + home/away sample ===")
for r in con.execute(
    "SELECT season_name, MIN(match_date) mn, MAX(match_date) mx, COUNT(*) n FROM matches "
    "WHERE provider='statsbomb' GROUP BY season_name ORDER BY season_name"
):
    print("  ", dict(r))

print("=== SB sample fixture (team_name, date) for overlap check ===")
for r in con.execute(
    "SELECT m.season_name, m.match_date, ht.team_name AS home, at.team_name AS away, "
    "m.home_goals, m.away_goals FROM matches m "
    "JOIN teams ht ON m.home_team_id=ht.id JOIN teams at ON m.away_team_id=at.id "
    "WHERE m.provider='statsbomb' AND m.season_name='2015/2016' LIMIT 5"
):
    print("  ", dict(r))
con.close()
