"""Phase 27.1 audit: precise FD-SB match overlap."""
import csv
import sqlite3
import re
from datetime import datetime

conn = sqlite3.connect("data/football_intelligence.db")
conn.row_factory = sqlite3.Row
c = conn.cursor()

def norm_team(name):
    if not name:
        return ""
    name = name.strip().lower()
    for suffix in (" f.c.", ".f.c.", " f.c", " fc", " a.f.c.", " a.f.c"):
        if name.endswith(suffix):
            name = name[:-len(suffix)].strip()
    name = re.sub(r"[.']", "", name)
    name = re.sub(r"\s+", " ", name).strip()
    aliases = {
        "man united": "manchester united",
        "man city": "manchester city",
        "spurs": "tottenham hotspur",
        "tottenham": "tottenham hotspur",
        "lei": "leicester city",
        "bournemouth": "afc bournemouth",
        "brighton": "brighton and hove albion",
        "newcastle": "newcastle united",
        "west ham": "west ham united",
        "sheffield utd": "sheffield united",
        "wolves": "wolverhampton wanderers",
        "wigan": "wigan athletic",
        "leeds": "leeds united",
    }
    return aliases.get(name, name)

# Show raw SB dates
print("=== Raw SB match_date values (first 3) ===")
for r in c.execute("SELECT id, match_date, season_name FROM matches WHERE provider='statsbomb' AND season_name='2015/2016' LIMIT 3"):
    print(f"  id={r['id']} match_date={r['match_date']!r} season={r['season_name']!r}")

# Show the SB match with team names
print("\n=== SB 1516 match with teams (first 3) ===")
for r in c.execute(
    "SELECT m.id, m.match_date, ht.team_name as home, at.team_name as away "
    "FROM matches m JOIN teams ht ON m.home_team_id = ht.id JOIN teams at ON m.away_team_id = at.id "
    "WHERE m.provider='statsbomb' AND m.season_name='2015/2016' LIMIT 3"
):
    print(f"  id={r['id']} date={r['match_date']!r} home={r['home']!r} away={r['away']!r}")
    date_norm = r["match_date"][:10] if r["match_date"] else ""
    print(f"    date_norm={date_norm!r} home_norm={norm_team(r['home'])!r} away_norm={norm_team(r['away'])!r}")

# Now check FD 1516 first 3 rows
print("\n=== FD 1516 CSV first 3 rows ===")
with open("data/raw/football_data/E0/1516.csv", "r", encoding="utf-8-sig") as f:
    reader = csv.DictReader(f)
    rows = list(reader)
for row in rows[:3]:
    date = row.get("Date", "")
    home = row.get("HomeTeam", "")
    away = row.get("AwayTeam", "")
    try:
        date_norm = datetime.strptime(date.strip(), "%d/%m/%Y").strftime("%Y-%m-%d")
    except ValueError:
        date_norm = date
    print(f"  date={date!r} home={home!r} away={away!r}")
    print(f"    date_norm={date_norm!r} home_norm={norm_team(home)!r} away_norm={norm_team(away)!r}")

# Now do the full comparison
sb_keys = set()
for r in c.execute(
    "SELECT m.match_date, ht.team_name as home, at.team_name as away "
    "FROM matches m JOIN teams ht ON m.home_team_id = ht.id JOIN teams at ON m.away_team_id = at.id "
    "WHERE m.provider='statsbomb' AND m.season_name='2015/2016'"
):
    date_norm = r["match_date"][:10] if r["match_date"] else ""
    key = (date_norm, norm_team(r["home"]), norm_team(r["away"]))
    sb_keys.add(key)

print(f"\nSB 1516 keys: {len(sb_keys)}")

fd_keys = set()
for row in rows:
    date = row.get("Date", "")
    home = row.get("HomeTeam", "")
    away = row.get("AwayTeam", "")
    try:
        date_norm = datetime.strptime(date.strip(), "%d/%m/%Y").strftime("%Y-%m-%d")
    except ValueError:
        date_norm = date
    key = (date_norm, norm_team(home), norm_team(away))
    fd_keys.add(key)

print(f"FD 1516 keys: {len(fd_keys)}")
print(f"Overlap: {len(sb_keys & fd_keys)}")

# Show some non-matching SB keys
print("\n=== SB keys NOT in FD (first 5) ===")
for key in sorted(sb_keys - fd_keys)[:5]:
    print(f"  {key}")

# Show some non-matching FD keys
print("\n=== FD keys NOT in SB (first 5) ===")
for key in sorted(fd_keys - sb_keys)[:5]:
    print(f"  {key}")

conn.close()



