"""Phase 27.1 audit: inspect DB schema and existing data."""
import sqlite3
import os

DB_PATH = "data/football_intelligence.db"

conn = sqlite3.connect(DB_PATH)
cur = conn.cursor()

cur.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
tables = [r[0] for r in cur.fetchall()]
print("=== Tables ({}) ===".format(len(tables)))
for t in tables:
    cur.execute("SELECT sql FROM sqlite_master WHERE name=?", (t,))
    schema = cur.fetchone()[0]
    cur.execute("SELECT COUNT(*) FROM [{}]".format(t))
    count = cur.fetchone()[0]
    print("\n--- {} ({} rows) ---".format(t, count))
    print(schema[:800])
conn.close()
