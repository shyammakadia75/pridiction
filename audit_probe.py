import os, sys
from pathlib import Path

print("=== data/ (project) contents ===")
for p in sorted((Path("data")).iterdir()):
    if p.is_file():
        print(f"  {p.name:24s} {p.stat().st_size}")

print("=== raw cache dirs (data/raw/*) ===")
raw = Path("data") / "raw"
if raw.exists():
    for sub in sorted(raw.iterdir()):
        files = list(sub.glob("*.csv")) if sub.is_dir() else []
        print(f"  {sub.name}: {len(files)} csv", [f.name for f in files[:6]])
else:
    print("  data/raw does NOT exist")

print("=== %TEMP% epl_fd ===")
tmp = Path(os.environ["TEMP"]) / "epl_fd"
if tmp.exists():
    for p in sorted(tmp.glob("*.csv")):
        print(f"  {p.name:20s} {p.stat().st_size}")
else:
    print("  missing")

print("=== football_data_results.py 148-340 ===")
lines = Path("src/football_intelligence/data/ingestion/football_data_results.py").read_text().splitlines()
for i in range(147, min(340, len(lines))):
    print(f"{i+1:4d}|{lines[i]}")

print("=== odds.py normalize + reconciliation ===")
olines = Path("src/football_intelligence/data/ingestion/odds.py").read_text().splitlines()
for i, l in enumerate(olines):
    if "def _normalize_team_name" in l or "def _find_matches_for_reconciliation" in l:
        for j in range(i, min(i+70, len(olines))):
            print(f"{j+1:4d}|{olines[j]}")
        print("---")
