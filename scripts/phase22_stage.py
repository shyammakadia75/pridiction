#!/usr/bin/env python3
"""Phase 22 Step 1.4 + 1.6/1.7: fresh-download reproducibility check AND
byte-exact staging into data/raw/football_data/E0/ using the canonical
football-data.co.uk URL. Read-only against production; no DB writes."""
from __future__ import annotations
import hashlib, json, os, shutil, sys, tempfile, urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from football_intelligence.data.ingestion.odds import ODDS_BASE_URL  # noqa: E402

TEMP_CACHE = Path(os.path.expandvars("%TEMP%\\epl_fd"))
STAGE = REPO / "data" / "raw" / "football_data" / "E0"
SEASONS = [("1516","2015/16"), ("1617","2016/17"), ("1920","2019/20")]
UA = "football-intelligence/0.1.0 (+https://github.com/shyammakadia75/pridiction)"

def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(65536), b""): h.update(c)
    return h.hexdigest()

STAGE.mkdir(parents=True, exist_ok=True)
probe = Path(tempfile.mkdtemp(prefix="p22_dl_"))
manifest = {"staged_dir": str(STAGE), "files": []}
print("=== freshness probe (fresh download vs local %TEMP% cache) ===")
for sc, label in SEASONS:
    url = f"{ODDS_BASE_URL}/{sc}/E0.csv"
    src = TEMP_CACHE / f"E0_{sc}.csv"
    t_sha = sha256(src); t_size = src.stat().st_size
    rec = {"season_code": sc, "season_name": label, "canonical_url": url,
           "local_cache": str(src), "local_sha256": t_sha, "local_bytes": t_size,
           "acquisition_method": "staged from local %TEMP% cache (NOT a fresh download)"}
    try:
        req = urllib.request.Request(url, headers={"User-Agent": UA})
        with urllib.request.urlopen(req, timeout=25) as r, open(probe / f"E0_{sc}.csv", "wb") as out:
            out.write(r.read())
        d_sha = sha256(probe / f"E0_{sc}.csv"); d_size = (probe / f"E0_{sc}.csv").stat().st_size
        rec["download_sha256"] = d_sha; rec["download_bytes"] = d_size
        rec["provenance"] = "CONFIRMED (download matches local cache)" if d_sha == t_sha else "DIFFERS -- see notes"
        print(f"  {label}: download {d_size}B sha={d_sha[:16]}..  local {t_size}B sha={t_sha[:16]}..  -> {rec['provenance']}")
    except Exception as e:
        rec["download_status"] = f"FAILED {type(e).__name__}: {e}"
        rec["provenance"] = "local cache used; network download failed (network reachable per docs but refused/timed out)"
        print(f"  {label}: download FAILED ({type(e).__name__}); using verified local cache.")
    # stage byte-exact copy
    dest = STAGE / f"{sc}.csv"
    shutil.copy2(src, dest)
    assert sha256(dest) == t_sha
    rec["staged_path"] = str(dest.relative_to(REPO))
    manifest["files"].append(rec)
    print(f"    staged {dest.relative_to(REPO)} (sha256 verified)")
manifest_path = STAGE.parent / "_source_manifest.json"
manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
print(f"\nWrote manifest: {manifest_path.relative_to(REPO)}")
print("STAGE DONE")
