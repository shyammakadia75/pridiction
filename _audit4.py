"""Phase 27.1 audit: inspect phase23_eval.db data."""
import sqlite3

conn = sqlite3.connect("data/phase23_eval.db")
conn.row_factory = sqlite3.Row
c = conn.cursor()

print("Schema version:", c.execute("SELECT * FROM schema_version").fetchall())
print()

print("=== Providers in matches ===")
for r in c.execute("SELECT provider, COUNT(*) as cnt FROM matches GROUP BY provider"):
    print(f"  {r['provider']}: {r['cnt']}")

print()
print("=== Seasons ===")
for r in c.execute("SELECT provider, season_name, COUNT(*) as cnt FROM matches GROUP BY provider, season_name ORDER BY provider, season_name"):
    print(f"  {r['provider']} / {r['season_name']}: {r['cnt']}")

print()
print("=== Competitions ===")
for r in c.execute("SELECT id, provider, provider_competition_id, competition_name FROM competitions"):
    print(f"  {dict(r)}")

print()
print("=== Teams ===")
for r in c.execute("SELECT provider, team_name FROM teams ORDER BY provider, team_name"):
    print(f"  {r['provider']}: {r['team_name']}")

print()
print("=== Sample matches ===")
for r in c.execute("SELECT provider, provider_match_id, season_name, match_date FROM matches LIMIT 5"):
    print(f"  {r['provider']}/{r['provider_match_id']} season={r['season_name']} date={r['match_date']}")

conn.close()
