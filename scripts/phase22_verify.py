#!/usr/bin/env python3
"""Phase 22 Step 1.3 + Step 2: verify E0 source CSVs and build the team-name inventory.
Read-only against production. No DB writes. Stages nothing yet."""
from __future__ import annotations
import csv, hashlib, os
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
import sys; sys.path.insert(0, str(REPO / "src"))
from football_intelligence.data.ingestion.odds import (  # noqa: E402
    _normalize_team_name, _normalize_match_date, TEAM_ALIASES,
)
from football_intelligence.data.ingestion.football_data_results import (  # noqa: E402
    _season_code_to_name, _league_display,
)

TEMP_CACHE = Path(os.path.expandvars("%TEMP%\\epl_fd"))
SEASONS = [("1516","2015/16"), ("1617","2016/17"), ("1920","2019/20")]

def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(65536), b""): h.update(c)
    return h.hexdigest()

def verify(src):
    with open(src, encoding="utf-8-sig", newline="") as fh:
        reader = csv.DictReader(fh)
        header = reader.fieldnames or []
        rows = list(reader)
    n = len(rows)
    missing = [c for c in ["Date","HomeTeam","AwayTeam","FTHG","FTAG","FTR"] if c not in header]
    bad_ftr = bad_score = bad_date = dup = 0
    names = set(); iso_dates = []; keys = set()
    for r in rows:
        fthg, ftag, ftr = int(r["FTHG"]), int(r["FTAG"]), r["FTR"].strip()
        if ftr not in {"H","D","A"}: bad_ftr += 1
        exp = "H" if fthg>ftag else ("A" if fthg<ftag else "D")
        if ftr != exp: bad_score += 1
        iso = _normalize_match_date(r["Date"])
        if iso is None: bad_date += 1; iso_dates.append(r["Date"])
        else: iso_dates.append(iso)
        names.update([r["HomeTeam"], r["AwayTeam"]])
        keys.add((r["Date"], r["HomeTeam"], r["AwayTeam"]))
    dup = n - len(keys)
    return dict(n=n, n_teams=len(names), n_cols=len(header), missing_cols=missing,
        date_range_iso=(min(iso_dates), max(iso_dates)), date_parse_failures=bad_date,
        dup_logical=dup, bad_ftr=bad_ftr, ftr_score_mismatch=bad_score,
        sha256=sha256(src), size=src.stat().st_size)

print("=== STEP 1.3: per-file verification ===")
by_season = {}
for sc, label in SEASONS:
    src = TEMP_CACHE / f"E0_{sc}.csv"
    v = verify(src)
    by_season[sc] = src
    rows = list(csv.DictReader(open(src, encoding="utf-8-sig", newline="")))
    print(f"--- E0_{sc}.csv  ({_season_code_to_name(sc)}  |  {_league_display('E0')}) ---")
    for k,val in v.items(): print(f"  {k}: {val}")

print("\n=== STEP 2: team-identity inventory ===")
all_raw = set()
by_name = {}
collisions = {}
for sc, label in SEASONS:
    src = by_season[sc]
    rows = list(csv.DictReader(open(src, encoding="utf-8-sig", newline="")))
    names = sorted({r["HomeTeam"] for r in rows} | {r["AwayTeam"] for r in rows})
    by_name[sc] = names
    all_raw |= set(names)
    for nm in names:
        collisions.setdefault(_normalize_team_name(nm), []).append(nm)
print(f"  distinct raw team names across all seasons: {len(all_raw)}")
print(f"  per-season team counts: { {sc:len(ns) for sc,ns in by_name.items()} }")
multi = {k:v for k,v in collisions.items() if len(set(v))>1}
print(f"  normalized collisions (canonical <- >1 raw): {multi if multi else 'NONE (injective)'}")
print("  alias coverage (raw short name -> canonical where raw differs from canonical):")
for nm in sorted(all_raw):
    canon = _normalize_team_name(nm)
    if nm.lower() != canon:
        print(f"    {nm!r:24s} -> {canon!r:26s}  alias={TEAM_ALIASES.get(nm.lower(),'<none>')}")
bare = sorted(n for n in all_raw if " " not in n and n.lower() not in TEAM_ALIASES
              and _normalize_team_name(n)==n.lower())
print(f"  bare names w/o alias ({len(bare)}): {bare}")
print("  -> these are FD-only stable canonicals; cross-source SB names like 'Leicester City'")
print("     won't match them (documented limitation, FD-only ingestion unaffected).")
print("\n=== cross-source name-gap audit (FD canonicals vs SB canonicals) ===")
# Map FD canonical -> does a likely SB form differ?
for nm in sorted(all_raw):
    canon = _normalize_team_name(nm)
    sb_like = canon
    # known SB adds 'city'/'united' suffixes
    print(f"    {nm!r:24s} FD-canon={canon!r}")
print("\nSTEP 1+2 verify DONE")
