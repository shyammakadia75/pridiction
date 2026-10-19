"""One-off read-only probe of the production DB schema and contents."""
import sqlite3

db = r"data\football_intelligence.db"
con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)

tables = [r[0] for r in con.execute(
    "SELECT name FROM sqlite_master WHERE type='table'").fetchall()]
print("tables:", tables)

cols = [r[1] for r in con.execute("PRAGMA table_info(matches)").fetchall()]
print("matches cols:", cols)

comp_cols = [r[1] for r in con.execute("PRAGMA table_info(competitions)").fetchall()]
print("competitions cols:", comp_cols)
# Resolve provider_competition_id='2' (statsbomb PL) to local competition id.
cid = con.execute(
    "SELECT id FROM competitions WHERE provider='statsbomb' AND provider_competition_id='2'"
).fetchone()
print("SB PL competition id:", cid)
if cid:
    cid = cid[0]
    print("PL matches count:", con.execute(
        "SELECT COUNT(*) FROM matches WHERE provider='statsbomb' AND competition_id=?", (cid,)
    ).fetchone()[0])
    print("seasons:", con.execute(
        "SELECT DISTINCT season_name FROM matches WHERE provider='statsbomb' AND competition_id=?", (cid,)
    ).fetchall())
    print("date range:", con.execute(
        "SELECT MIN(match_date), MAX(match_date) FROM matches WHERE provider='statsbomb' AND competition_id=?", (cid,)
    ).fetchone())
    print("NULL goals:", con.execute(
        "SELECT COUNT(*) FROM matches WHERE provider='statsbomb' AND competition_id=? AND (home_goals IS NULL OR away_goals IS NULL)", (cid,)
    ).fetchone()[0])
    print("distinct teams in PL matches:", con.execute(
        "SELECT COUNT(DISTINCT home_team_id) FROM matches WHERE provider='statsbomb' AND competition_id=?", (cid,)
    ).fetchone()[0])

print("teams count:", con.execute("SELECT COUNT(*) FROM teams").fetchone()[0])
print("teams cols:", [r[1] for r in con.execute("PRAGMA table_info(teams)").fetchall()])
con.close()
