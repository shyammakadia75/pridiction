"""Probe: does EloModel.fit leak same-day results? (Phase 21 bug check)"""
import sqlite3
from datetime import datetime
from pathlib import Path
import pandas as pd
from football_intelligence.prediction.elo import EloModel, load_elo_matches

PROD = r"data/football_intelligence.db"

# 1) Date multiplicities on the real prod DB (statsbomb, comp 2).
con = sqlite3.connect(f"file:{PROD}?mode=ro", uri=True)
try:
    df = pd.read_sql_query(
        "SELECT match_date, home_team_id, away_team_id, home_goals, away_goals "
        "FROM matches m JOIN competitions c ON m.competition_id=c.id "
        "WHERE m.provider='statsbomb' AND c.provider_competition_id='2' "
        "ORDER BY match_date", con)
finally:
    con.close()
print("prod comp2 rows:", len(df))
vc = df["match_date"].value_counts()
print("dates with >1 fixture:", int((vc > 1).sum()), "max per date:", int(vc.max()))

# 2) Construct a same-day scenario: 3 fixtures on the SAME date, distinct teams.
same_day = pd.DataFrame({
    "match_date": ["2020-01-01", "2020-01-01", "2020-01-01"],
    "home_team_id": [1, 3, 5],
    "away_team_id": [2, 4, 6],
    "home_goals": [1, 1, 1],   # all home wins
    "away_goals": [0, 0, 0],
})
m = EloModel(initial_rating=1500.0, K=20.0, home_advantage=50.0)
m.fit(same_day)
h = m.history_
print("\nSame-day Elo (3 fixtures, same date):")
for i, e in enumerate(h):
    print(f"  fixture {i}: exp_home={e.expected_home:.6f} home_rating={e.home_rating:.2f} away_rating={e.away_rating:.2f}")
# If SAME-DAY LEAKAGE: fixture 1 & 2 ratings should differ from fixture 0's pre-update ratings.
r0 = h[0].home_rating
r1 = h[1].home_rating
print("\nLEAKAGE?", "fixture1 home_rating changed from fixture0 (same-date update seen)" if r1 != r0 else "no same-date change")
print("distinct ratings across same-date fixtures:",
      len({round(e.home_rating, 6) for e in h}) > 1 or len({round(e.away_rating, 6) for e in h}) > 1)
