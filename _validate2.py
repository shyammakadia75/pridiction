import csv, os, collections, datetime
OUT=os.path.join(os.environ['TEMP'],'epl_fd')
def pd_(s):
    for f in ('%d/%m/%y','%d/%m/%Y','%Y-%m-%d'):
        try: return datetime.datetime.strptime(str(s).strip(),f).date()
        except Exception: pass
    return None
for k in ['1516','1617','1920']:
    fn=os.path.join(OUT,'E0_'+k+'.csv')
    if not os.path.exists(fn):
        print(k,'FILE MISSING'); continue
    rows=list(csv.DictReader(open(fn,encoding='utf-8')))
    cols=list(rows[0].keys()) if rows else []
    dates=[pd_(r.get('Date','')) for r in rows]
    teams=sorted(set(r.get('HomeTeam','') for r in rows)|set(r.get('AwayTeam','') for r in rows))
    ftr=collections.Counter(r.get('FTR','') for r in rows)
    bad=[r for r in rows if r.get('FTHG') in ('','None') or r.get('FTAG') in ('','None')]
    selfplay=[r for r in rows if r.get('HomeTeam')==r.get('AwayTeam')]
    fdup=len(rows)-len({(r.get('Date'),r.get('HomeTeam'),r.get('AwayTeam')) for r in rows})
    gd=[d for d in dates if d]
    print('=== %s ==='%k)
    print(' rows=%d cols=%s'%(len(rows),cols))
    print(' distinct_teams=%d list=%s'%(len(teams),teams))
    print(' date_range=%s .. %s'%(min(gd) if gd else None, max(gd) if gd else None))
    print(' FTR=%s'%(dict(ftr)))
    print(' null_or_missing_scores=%d  self_play=%d  fixture_dupes=%d'%(len(bad),len(selfplay),fdup))
