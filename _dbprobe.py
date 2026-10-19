import sqlite3
c=sqlite3.connect('data/football_intelligence.db'); c.row_factory=sqlite3.Row; cur=c.cursor()
print('providers:', [tuple(r) for r in cur.execute('SELECT provider, COUNT(*) FROM matches GROUP BY provider')])
print('SB match_date sample:', [r[0] for r in cur.execute("SELECT match_date FROM matches WHERE provider='statsbomb' LIMIT 5").fetchall()])
print('SB date substr(1,10) sample:', [r[0] for r in cur.execute("SELECT DISTINCT substr(match_date,1,10) FROM matches WHERE provider='statsbomb' ORDER BY match_date LIMIT 5").fetchall()])
print('competition/season sample:', [tuple(r) for r in cur.execute('SELECT DISTINCT competition_id, season_name FROM matches LIMIT 6').fetchall()])
print('teams sample (provider,pteam_id,name):', [tuple(r) for r in cur.execute('SELECT provider, provider_team_id, team_name FROM teams LIMIT 8').fetchall()])
print("FD team count:", cur.execute("SELECT COUNT(*) FROM teams WHERE provider='football-data'").fetchone()[0])
