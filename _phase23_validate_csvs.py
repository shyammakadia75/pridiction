"""Phase 23 Step 1: strict re-validation of the 3 staged FD E0 CSVs
against the Phase-22 manifest. Read-only; no DB writes.
"""
import csv
import hashlib
from pathlib import Path
from datetime import datetime

BASE = Path("data/raw/football_data/E0")
EXPECT = {
    "1516": ("bd3502a18c38a1597fd9af62e2366b4015006d3528dd4d18b311bd6237bbc085", 100656, 20),
    "1617": ("9625a7652b5f98fbd3e2e4d378c851fc246693f3343e34a72428d5b6e864d3e0", 99990, 20),
    "1920": ("100037618b94f94057400bb02bf6bac4ef74ddaa58cde4b38370839c39caee61", 175475, 20),
}
SEASON_MAP = {"1516": "2015/16", "1617": "2016/17", "1920": "2019/20"}


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_date(s):
    s = (s or "").strip()
    for fmt in ("%d/%m/%Y", "%d/%m/%y"):
        try:
            return datetime.strptime(s, fmt).date()
        except ValueError:
            pass
    return None


all_ok = True
total = 0
for code, (exp_sha, exp_bytes, exp_teams) in EXPECT.items():
    p = BASE / f"{code}.csv"
    if not p.exists():
        print(f"[FAIL] {p} missing"); all_ok = False; continue
    actual_sha = sha(p)
    actual_bytes = p.stat().st_size
    rows = list(csv.DictReader(p.open(encoding="utf-8")))
    teams = set()
    fixtures = set()
    bad_dates = 0
    score_errs = 0
    neg = 0
    dup_fixtures = 0
    parsed = []
    for r in rows:
        teams.update([r["HomeTeam"], r["AwayTeam"]])
        key = (r["Date"], r["HomeTeam"], r["AwayTeam"])
        if key in fixtures:
            dup_fixtures += 1
        fixtures.add(key)
        d = parse_date(r.get("Date"))
        if d is None:
            bad_dates += 1
        else:
            parsed.append(d)
        try:
            hg = int(r["FTHG"]); ag = int(r["FTAG"])
        except (ValueError, KeyError):
            score_errs += 1; continue
        if hg < 0 or ag < 0:
            neg += 1
        ftr = r.get("FTR", "").strip().upper()
        # FTR consistency
        if ftr not in ("H", "D", "A"):
            score_errs += 1
        elif ftr == "H" and hg <= ag:
            score_errs += 1
        elif ftr == "A" and ag <= hg:
            score_errs += 1
        elif ftr == "D" and hg != ag:
            score_errs += 1
    parsed_sorted = all(parsed[i] <= parsed[i+1] for i in range(len(parsed)-1))
    total += len(rows)
    sha_ok = actual_sha == exp_sha
    bytes_ok = actual_bytes == exp_bytes
    teams_ok = len(teams) == exp_teams
    status = "OK" if (sha_ok and bytes_ok and teams_ok and not bad_dates and not score_errs and not neg and not dup_fixtures and parsed_sorted) else "FAIL"
    if status != "OK":
        all_ok = False
    print(f"=== E0_{code} ({SEASON_MAP[code]}) :: {status} ===")
    print(f"  rows={len(rows)}  distinct_teams={len(teams)}  dupe_fixtures={dup_fixtures}")
    print(f"  sha_match={sha_ok}  size_match={bytes_ok} (exp {exp_bytes}, got {actual_bytes})")
    print(f"  bad_dates={bad_dates}  score_errors={score_errs}  negative_goals={neg}  chronological={parsed_sorted}")
    print(f"  date_range={min(parsed)} -> {max(parsed)}")
    print(f"  sha256={actual_sha}")
print()
print(f"TOTAL rows across 3 seasons = {total}")
print(f"ALL CHECKS PASSED = {all_ok}")
