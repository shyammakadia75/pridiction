import urllib.request, os, csv, io
OUT=os.path.join(os.environ['TEMP'], 'epl_fd')
os.makedirs(OUT, exist_ok=True)
seasons = {'1516': '1516/E0.csv', '1617': '1617/E0.csv', '1920': '1920/E0.csv'}
base = 'https://www.football-data.co.uk/mmz4281/'
results = {}
for k, path in seasons.items():
    url = base + path
    fn = os.path.join(OUT, 'E0_' + k + '.csv')
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0 (FootballIntelligence/0.1)'})
        data = urllib.request.urlopen(req, timeout=30).read()
    except Exception as e:
        results[k] = ('FAILED', type(e).__name__, str(e)[:160])
        print(k, 'DOWNLOAD FAILED:', type(e).__name__, str(e)[:160])
        continue
    open(fn, 'wb').write(data)
    text = data.decode('utf-8', 'replace')
    rows = list(csv.DictReader(io.StringIO(text)))
    cols = list(rows[0].keys()) if rows else []
    # validation
    homecol = 'HomeTeam' if 'HomeTeam' in cols else ('Home' if 'Home' in cols else None)
    awaycol = 'AwayTeam' if 'AwayTeam' in cols else ('Away' if 'Away' in cols else None)
    dcol = 'Date' if 'Date' in cols else ('date' if 'date' in cols else None)
    fcol = 'FTR' if 'FTR' in cols else ('FullTimeResult' if 'FullTimeResult' in cols else None)
    hcol = 'FTHG' if 'FTHG' in cols else ('HomeGoals' if 'HomeGoals' in cols else None)
    acol = 'FTAG' if 'FTAG' in cols else ('AwayGoals' if 'AwayGoals' in cols else None)
    teams = sorted(set(r.get(homecol,'') for r in rows if homecol) | set(r.get(awaycol,'') for r in rows if awaycol))
    dupes = len(rows) - len(set((r.get(dcol), r.get(homecol), r.get(awaycol)) for r in rows if dcol and (r.get(dcol) or r.get(homecol) or r.get(awaycol)) and (r.get(hcol) is not None and r.get(acol) is not None)))
    dates = [r.get(dcol) for r in rows[:2] + rows[-2:]] if dcol else []
    results[k] = ('OK', len(data), len(rows), cols[:8], teams[:3], dates)
    print(k, 'OK bytes=%d rows=%d cols=%s teams_sample=%s dates_sample=%s dupes=%d' % (len(data), len(rows), cols[:8], teams[:3], dates, dupes))
print('SUMMARY:', results)
