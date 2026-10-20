#!/usr/bin/env python3
"""Phase 23 (Step 2): build an isolated, DISPOSABLE multi-season database from
the *verified* staged Football-Data E0 CSVs (2015/16, 2016/17, 2019/20).

This script performs NO writes against the production DB
(``data/football_intelligence.db``); it only reads it for a SHA-256
tamper-gate before/after building the disposable copy.  All ingested rows go
into a fresh, gitignored SQLite file under ``data/`` (``*.db`` is gitignored).

Run:
    .venv\\Scripts\\python.exe scripts/phase23_build_db.py
"""
from __future__ import annotations

import hashlib
import sqlite3
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from football_intelligence.database import initialize_database  # noqa: E402
from football_intelligence.data.ingestion.football_data_results import (  # noqa: E402
    FootballDataResultsIngestor,
)

PROD = REPO / "data" / "football_intelligence.db"
STAGE_CACHE = REPO / "data" / "raw" / "football_data"
SEASONS = [("1516", "2015/16"), ("1617", "2016/17"), ("1920", "2019/20")]

# Verified staged-source manifest (Phase-22 evidence).
EXPECTED = {
    "1516": {
        "sha256": "bd3502a18c38a1597fd9af62e2366b4015006d3528dd4d18b311bd6237bbc085",
        "bytes": 100656,
        "rows": 380,
    },
    "1617": {
        "sha256": "9625a7652b5f98fbd3e2e4d378c851fc246693f3343e34a72428d5b6e864d3e0",
        "bytes": 99990,
        "rows": 380,
    },
    "1920": {
        "sha256": "100037618b94f94057400bb02bf6bac4ef74ddaa58cde4b38370839c39caee61",
        "bytes": 175475,
        "rows": 380,
    },
}
PROD_SHA256 = "725af574b88dc033bd3b00928de503f9bd9a836fbe4252832b7812dce0ab3122"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_staged_sources() -> None:
    """Abort (fail-closed) if any staged CSV is missing or hash-mismatched."""
    for code, exp in EXPECTED.items():
        p = STAGE_CACHE / "E0" / f"{code}.csv"
        if not p.exists():
            raise FileNotFoundError(f"Staged source missing (fail-closed): {p}")
        actual = sha256_file(p)
        if actual != exp["sha256"]:
            raise RuntimeError(
                f"SHA-256 mismatch for E0_{code} (fail-closed): "
                f"expected {exp['sha256']}, got {actual}"
            )
        if p.stat().st_size != exp["bytes"]:
            raise RuntimeError(
                f"Size mismatch for E0_{code}: expected {exp['bytes']}, "
                f"got {p.stat().st_size}"
            )
    print("[OK] staged FD E0 sources verified (sha256 + size):",
          [SEASONS[i][0] for i in range(3)])


def build_disposable_db(db_path: Path) -> dict:
    """Create a fresh disposable DB from the 3 staged FD seasons; prod untouched."""
    prod_before = sha256_file(PROD)
    if prod_before != PROD_SHA256:
        raise RuntimeError(
            "Production DB hash changed before this run (fail-closed): "
            f"{prod_before}"
        )

    db_path = Path(db_path)
    if db_path.exists():
        db_path.unlink()
    initialize_database(db_path)
    con = sqlite3.connect(str(db_path))
    con.row_factory = sqlite3.Row

    inserted = 0
    for code, label in SEASONS:
        ing = FootballDataResultsIngestor(
            league_code="E0", season_code=code, db_path=db_path,
            cache_dir=STAGE_CACHE,
            source_url=f"https://www.football-data.co.uk/mmz4281/{code}/E0.csv",
        )
        r = ing.ingest()
        print(f"  {label}: read={r.records_read} inserted={r.records_inserted} "
              f"updated={r.records_updated} skipped={r.records_skipped} "
              f"ambiguous={r.records_ambiguous} cached={ing._was_cached}")
        assert r.status == "success", f"ingest failed for {code}: {r.status}"
        assert r.records_read == EXPECTED[code]["rows"], (
            f"{code}: read {r.records_read} != {EXPECTED[code]['rows']}"
        )
        assert r.records_inserted == EXPECTED[code]["rows"], (
            f"{code}: inserted {r.records_inserted} (expected {EXPECTED[code]['rows']})"
        )
        inserted += r.records_inserted

    con.commit()
    total = con.execute(
        "SELECT COUNT(*) FROM matches WHERE provider='football-data'"
    ).fetchone()[0]
    teams = con.execute(
        "SELECT COUNT(DISTINCT provider_team_id) FROM teams WHERE provider='football-data'"
    ).fetchone()[0]
    ic = con.execute("PRAGMA integrity_check").fetchone()[0]
    fk = con.execute("PRAGMA foreign_key_check").fetchall()
    con.close()

    prod_after = sha256_file(PROD)
    assert prod_after == prod_before, (
        f"PRODUCTION DB CHANGED (fail-closed): {prod_before} -> {prod_after}"
    )

    summary = {
        "db_path": str(db_path),
        "seasons": [s[0] for s in SEASONS],
        "total_matches": total,
        "distinct_teams": teams,
        "integrity_check": ic,
        "fk_violations": len(fk),
        "prod_sha256_before": prod_before,
        "prod_sha256_after": prod_after,
        "prod_unchanged": prod_after == prod_before,
    }
    assert total == 1140, f"expected 1140 matches, got {total}"
    assert ic == "ok" and len(fk) == 0
    assert summary["prod_unchanged"]
    return summary


if __name__ == "__main__":
    verify_staged_sources()
    dispos = REPO / "data" / "phase23_eval.db"
    s = build_disposable_db(dispos)
    print("\n=== Phase 23 disposable DB built ===")
    for k, v in s.items():
        print(f"  {k}: {v}")
    print("\nBUILD OK; production DB SHA-256 unchanged.")
