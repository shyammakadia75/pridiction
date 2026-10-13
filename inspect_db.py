"""Temporary script to inspect the real database (read-only)."""
import sqlite3, os

DB = "data/football_intelligence.db"

print(f"Database exists: {os.path.exists(DB)}")
print(f"Database size: {os.path.getsize(DB) if os.path.exists(DB) else 0} bytes")
print()

conn = sqlite3.connect(DB)
conn.row_factory = sqlite3.Row
cursor = conn.cursor()

tables = [t[0] for t in cursor.execute(
    "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
).fetchall()]
print("Tables:", tables)
print()

for t in ["matches", "teams", "competitions", "odds", "predictions",
          "model_versions", "prediction_errors", "model_performance"]:
    try:
        cnt = cursor.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
        print(f"  {t}: {cnt} rows")
    except Exception as e:
        print(f"  {t}: ERROR - {e}")

print()
print("--- Matches: columns ---")
cols = [c[1] for c in cursor.execute("PRAGMA table_info(matches)").fetchall()]
print("Columns:", cols)

print()
print("--- Match status distribution ---")
for r in cursor.execute(
    "SELECT match_status, COUNT(*) as cnt FROM matches GROUP BY match_status"
).fetchall():
    print(f"  {r['match_status']}: {r['cnt']}")

print()
print("--- Matches with results (non-null goals) ---")
cnt = cursor.execute(
    "SELECT COUNT(*) FROM matches WHERE home_goals IS NOT NULL AND away_goals IS NOT NULL"
).fetchone()[0]
print(f"  Matches with results: {cnt}")
total = cursor.execute("SELECT COUNT(*) FROM matches").fetchone()[0]
print(f"  Total matches: {total}")
pct = 100.0 * cnt / total if total else 0
print(f"  % with results: {pct:.1f}%")

print()
print("--- Date range ---")
try:
    dr = cursor.execute(
        "SELECT MIN(match_date) as min_d, MAX(match_date) as max_d FROM matches"
    ).fetchone()
    print(f"  min: {dr['min_d']}, max: {dr['max_d']}")
except Exception as e:
    print(f"  Error: {e}")

print()
print("--- Sample matches (first 5 and last 3 by date) ---")
for r in cursor.execute(
    "SELECT id, provider, provider_match_id, match_date, home_team_id, away_team_id, home_goals, away_goals, match_status FROM matches ORDER BY match_date LIMIT 5"
).fetchall():
    print(f"  {dict(r)}")

print("  ...")
for r in cursor.execute(
    "SELECT id, provider, provider_match_id, match_date, home_team_id, away_team_id, home_goals, away_goals, match_status FROM matches ORDER BY match_date DESC LIMIT 3"
).fetchall():
    print(f"  {dict(r)}")

print()
print("--- Teams ---")
teams_list = cursor.execute(
    "SELECT provider, provider_team_id, team_name FROM teams"
).fetchall()
print(f"  Total teams: {len(teams_list)}")
for r in teams_list[:10]:
    print(f"  {r['provider']} / {r['provider_team_id']}: {r['team_name']}")

print()
print("--- Competitions ---")
comps = cursor.execute(
    "SELECT provider, provider_competition_id, competition_name, season_name FROM competitions"
).fetchall()
print(f"  Total competitions: {len(comps)}")
for r in comps[:5]:
    print(f"  {r['provider']} / {r['provider_competition_id']}: {r['competition_name']} ({r['season_name']})")

print()
print("--- Odds ---")
odds_list = cursor.execute(
    "SELECT o.match_id, o.bookmaker, o.market, o.selection, o.decimal_odds "
    "FROM odds o LIMIT 5"
).fetchall()
print(f"  Sample odds ({len(odds_list)} shown):")
for r in odds_list:
    print(f"  match={r['match_id']}, {r['bookmaker']}, {r['market']}, {r['selection']}, {r['decimal_odds']}")

print()
print("--- Odds coverage: how many matches have odds? ---")
cnt_with_odds = cursor.execute(
    "SELECT COUNT(DISTINCT match_id) FROM odds"
).fetchone()[0]
print(f"  Matches with odds: {cnt_with_odds}")
cnt_completed_with_odds = cursor.execute(
    "SELECT COUNT(DISTINCT m.id) FROM matches m "
    "JOIN odds o ON o.match_id = m.id "
    "WHERE m.home_goals IS NOT NULL AND m.away_goals IS NOT NULL"
).fetchone()[0]
print(f"  Completed matches with odds: {cnt_completed_with_odds}")

print()
print("--- Distinct match_date values (year-month) ---")
for r in cursor.execute(
    "SELECT substr(match_date, 1, 7) as ym, COUNT(*) as cnt FROM matches "
    "GROUP BY ym ORDER BY ym"
).fetchall():
    print(f"  {r['ym']}: {r['cnt']}")

conn.close()
print("\n=== Inspection complete ===")
