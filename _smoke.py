import pathlib, sqlite3, sys, tempfile
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0, str(pathlib.Path("src").resolve()))
from football_intelligence.database import initialize_database
from football_intelligence.database.connection import database_connection
from football_intelligence.data.ingestion.football_data_results import FootballDataResultsIngestor

CSV = ("Date,HomeTeam,AwayTeam,FTHG,FTAG,FTR\n"
       "13/08/2016,Arsenal,Burnley,1,0,H\n"
       "13/08/2016,Man United,Bookings FC,2,2,D\n"  # alias 'Man United' -> manchester united
       "2016-08-20,Chelsea,Spurs,0,1,A\n")  # ISO date + 'Spurs' alias

tmp = Path(tempfile.mkdtemp())
db = tmp / "test.db"
initialize_database(db)
cache = tmp / "cache"

def fake_download(url, cache_path, timeout, refresh):
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(CSV, encoding="utf-8")
    return CSV, False

ing = FootballDataResultsIngestor(league_code="E0", season_code="1617",
                                  db_path=db, cache_dir=cache, source_url="http://mock/E0_1617.csv")
with patch("football_intelligence.data.ingestion.football_data_results._download_csv", side_effect=fake_download):
    res = ing.ingest()
print("RUN1:", res.status, "read", res.records_read, "ins", res.records_inserted, "upd", res.records_updated, "skip", res.records_skipped, "amb", res.records_ambiguous)

with database_connection(db) as c:
    teams = [dict(r) for r in c.execute("SELECT provider, provider_team_id, team_name, country FROM teams ORDER BY provider_team_id")]
    comps = [dict(r) for r in c.execute("SELECT provider, provider_competition_id, competition_name, country FROM competitions")]
    matches = [dict(r) for r in c.execute("SELECT provider, provider_match_id, match_date, home_goals, away_goals, match_status FROM matches ORDER BY match_date")]
    ds = [dict(r) for r in c.execute("SELECT provider, dataset_name, source_reference, data_version, metadata FROM data_sources")]
    odds_cnt = c.execute("SELECT COUNT(*) FROM odds").fetchone()[0]
print("teams:", teams)
print("comps:", comps)
print("matches:", matches)
print("data_sources:", ds)
print("odds rows (must be 0):", odds_cnt)

# Idempotent re-import -> all updated, no new inserts
with patch("football_intelligence.data.ingestion.football_data_results._download_csv", side_effect=fake_download):
    res2 = ing.ingest()
print("RUN2 (idempotent):", res2.status, "ins", res2.records_inserted, "upd", res2.records_updated, "skip", res2.records_skipped)

with database_connection(db) as c:
    print("match count after rerun:", c.execute("SELECT COUNT(*) FROM matches").fetchone()[0])

# Cross-source: fresh DB with a STATSBROMB Arsenal-Burnley fixture pre-seeded
# BEFORE the football-data import -> that football-data fixture must be SKIPPED.
db2 = tmp / "test2.db"
initialize_database(db2)
cache2 = tmp / "cache2"
with database_connection(db2) as c:
    cid = c.execute("INSERT INTO competitions (provider, provider_competition_id, competition_name, country, created_at, updated_at) VALUES ('statsbomb','2','Premier League','England',datetime('now'),datetime('now'))").lastrowid
    ht = c.execute("INSERT INTO teams (provider, provider_team_id, team_name, country, created_at, updated_at) VALUES ('statsbomb','9001','Arsenal','England',datetime('now'),datetime('now'))").lastrowid
    at = c.execute("INSERT INTO teams (provider, provider_team_id, team_name, country, created_at, updated_at) VALUES ('statsbomb','9002','Burnley','England',datetime('now'),datetime('now'))").lastrowid
    c.execute("INSERT INTO matches (provider, provider_match_id, competition_id, season_name, match_date, home_team_id, away_team_id, home_goals, away_goals, match_status, created_at, updated_at) VALUES ('statsbomb','sb_2016_ars_burn',%d,'2016/17','2016-08-13',%d,%d,1,0,'available',datetime('now'),datetime('now'))" % (cid, ht, at))
ing2 = FootballDataResultsIngestor(league_code="E0", season_code="1617", db_path=db2, cache_dir=cache2, source_url="http://mock2/E0_1617.csv")
with patch("football_intelligence.data.ingestion.football_data_results._download_csv", side_effect=lambda *a, **k: (CSV, False)):
    res3 = ing2.ingest()
print("RUN3 (cross-source, fresh):", res3.status, "ins", res3.records_inserted, "upd", res3.records_updated, "skip", res3.records_skipped)
with database_connection(db2) as c:
    rows = [dict(r) for r in c.execute("SELECT provider, provider_match_id, home_goals FROM matches WHERE match_date='2016-08-13' ORDER BY provider")]
    print("2016-08-13 matches (expect exactly 1, the statsbomb one):", rows)
    print("total matches (expect 2: skip Arsenal-Burnley, insert 2 others):", c.execute("SELECT COUNT(*) FROM matches").fetchone()[0])

