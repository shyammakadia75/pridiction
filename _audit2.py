"""Phase 27.1 audit: inspect production DB data content."""
import sqlite3

conn = sqlite3.connect("data/football_intelligence.db")
conn.row_factory = sqlite3.Row
c = conn.cursor()

print("Schema version:", c.execute("SELECT * FROM schema_version").fetchone()["version"])
print()

print("=== Competitions ===")
for r in c.execute("SELECT * FROM competitions"):
    print(f"  id={r['id']} provider={r['provider']} comp_id={r['provider_competition_id']} name={r['competition_name']} country={r['country']}")

print()
print("=== Teams (provider, team_name) ===")
for r in c.execute("SELECT provider, team_name FROM teams ORDER BY provider, team_name"):
    print(f"  {r['provider']}: {r['team_name']}")

print()
print("=== Matches by provider/season ===")
for r in c.execute("SELECT provider, season_name, COUNT(*) as cnt FROM matches GROUP BY provider, season_name ORDER BY provider, season_name"):
    print(f"  {r['provider']} / {r['season_name']}: {r['cnt']}")

print()
print("=== Matches: providers present ===")
for r in c.execute("SELECT provider, COUNT(*) as cnt FROM matches GROUP BY provider"):
    print(f"  {r['provider']}: {r['cnt']}")

print()
print("=== Sample matches (football-data) ===")
for r in c.execute("SELECT provider, provider_match_id, season_name, match_date, home_team_id, away_team_id, home_goals, away_goals, match_status FROM matches WHERE provider='football-data' LIMIT 5"):
    print(f"  id={r['provider_match_id']} season={r['season_name']} date={r['match_date']} home={r['home_team_id']} away={r['away_team_id']} H={r['home_goals']} A={r['away_goals']} status={r['match_status']}")

print()
print("=== Sample matches (statsBomb) ===")
for r in c.execute("SELECT provider, provider_match_id, season_name, match_date, home_team_id, away_team_id, home_goals, away_goals, match_status FROM matches WHERE provider='statsBomb' LIMIT 5"):
    print(f"  id={r['provider_match_id']} season={r['season_name']} date={r['match_date']} home={r['home_team_id']} away={r['away_team_id']} H={r['home_goals']} A={r['away_goals']} status={r['match_status']}")

print()
print("=== Data sources ===")
for r in c.execute("SELECT provider, dataset_name, source_reference, data_version, metadata FROM data_sources"):
    print(f"  provider={r['provider']} dataset={r['dataset_name']} version={r['data_version']}")
    if r['metadata']:
        print(f"    metadata: {r['metadata'][:200]}")

print()
print("=== Odds table columns (production DB) ===")
for col in c.execute("PRAGMA table_info(odds)"):
    print(f"  {col['name']} {col['type']} pk={col['pk']} nn={col['notnull']} dflt={col['dflt_value']}")

print()
print("=== Odds constraints ===")
for r in c.execute("PRAGMA table_info(odds)"):
    print(f"  {r['name']}")

for r in c.execute("SELECT sql FROM sqlite_master WHERE type='table' AND name='odds'"):
    print(f"Full SQL: {r['sql']}")

conn.close()
