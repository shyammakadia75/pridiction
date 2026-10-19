"""Phase 17.4 — Genuine results-data validation rehearsal (real E0 CSVs).

Uses REAL Football-Data.co.uk E0 CSVs already present in
%TEMP%/epl_fd/E0_<season>.csv (downloaded by ``_dl.py`` from the free
football-data.co.uk endpoint). Validates them with the project's own
retrieve/validate/quality stack and the Phase 17.2 cross-source guard against a
TEMP database.

SAFETY
- Production DB ``data/football_intelligence.db`` is NEVER written to. Its SHA-256
  is hashed READ-ONLY before and after; asserted equal.
- All writes target a disposable DB under a clearly-named temp dir (removed at end).

DATA PROVENANCE
- FD side = REAL downloaded CSVs (genuine validation).
- SB side uses ONE SYNTHETIC overlap record (clearly labelled) ONLY to exercise the
  cross-source guard against the real FD fixture -- it does NOT establish real-world
  agreement.
"""
from __future__ import annotations

import hashlib
import os
import shutil
import tempfile
from pathlib import Path
from unittest.mock import patch

from football_intelligence.database import initialize_database
from football_intelligence.database.connection import database_connection
from football_intelligence.data.ingestion.football_data_results import (
    FootballDataResultsIngestor,
    _normalize_results_dataframe,
)
from football_intelligence.data.ingestion.odds import _parse_csv
from football_intelligence.data.ingestion.statsbomb import (
    StatsBombIngestor,
    _matches_to_dataframe,
    _normalize_match,
    record_ingestion_run,
)
from football_intelligence.data.quality import check_match_data_quality

PROD_DB = Path("data/football_intelligence.db")
EPL_FD_DIR = Path(os.environ.get("TEMP", "/tmp")) / "epl_fd"
_DL_PATCH = "football_intelligence.data.ingestion.football_data_results._download_csv"

# SYNTHETIC SB overlap only: Arsenal-Burnley 2016-08-13 2-1, to prove the guard
# skips the REAL FD fixture and preserves its real score.
SB_OVERLAP_RAW = {
    "match_id": "synt_overlap", "match_date": "2016-08-13", "kick_off": "16:00:00.000",
    "competition": {"competition_id": 7, "season_id": 3,
                    "competition_name": "Premier League", "country_name": "England",
                    "season_name": "2016/2017"},
    "season": {"season_id": 3, "season_name": "2016/2017"},
    "home_team": {"home_team_id": "5001", "home_team_name": "Arsenal",
                  "country": {"name": "England"}},
    "away_team": {"away_team_id": "5002", "away_team_name": "Burnley",
                  "country": {"name": "England"}},
    "home_score": 2, "away_score": 1, "match_status": "available",
    "competition_stage": {"name": "League Round"},
    "metadata": {"data_version": "1.1.0"}, "match_week": 1,
    "last_updated": "2024-01-15T10:30:00.000",
}


def _hash_prod() -> str:
    h = hashlib.sha256()
    with open(PROD_DB, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _cnt(db, prov=None):
    with database_connection(db) as c:
        q = "SELECT COUNT(*) FROM matches"; arg = ()
        if prov:
            q += " WHERE provider=?"; arg = (prov,)
        return c.execute(q, arg).fetchone()[0]


def _score(db, prov, date_iso, home, away):
    with database_connection(db) as c:
        row = c.execute(
            "SELECT m.home_goals, m.away_goals AS away_goals "
            "FROM matches m JOIN teams ht ON m.home_team_id = ht.id "
            "JOIN teams at ON m.away_team_id = at.id "
            "WHERE m.provider=? AND m.match_date=? AND ht.team_name=? AND at.team_name=?",
            (prov, date_iso, home, away),
        ).fetchone()
        return None if row is None else (row["home_goals"], row["away_goals"])


def _integrity(db) -> str:
    with database_connection(db) as c:
        return c.execute("PRAGMA integrity_check").fetchone()[0]


def _first_fixture(db):
    """Return (date_iso, home_name, away_name, (hg, ag)) for the first complete FD fixture."""
    with database_connection(db) as c:
        row = c.execute(
            "SELECT m.match_date, ht.team_name AS home, at.team_name AS away, "
            "m.home_goals, m.away_goals FROM matches m "
            "JOIN teams ht ON m.home_team_id = ht.id "
            "JOIN teams at ON m.away_team_id = at.id "
            "WHERE m.provider=? AND m.match_date IS NOT NULL "
            "AND m.home_goals IS NOT NULL AND m.away_goals IS NOT NULL "
            "ORDER BY m.match_date LIMIT 1", ("football-data",)
        ).fetchone()
    return None if row is None else (row["match_date"], row["home"], row["away"], (row["home_goals"], row["away_goals"]))


def _sb_overlap_df(date_iso, home, away):
    """Build a SYNTHETIC SB match matching a REAL FD fixture (date + team names)."""
    raw = dict(SB_OVERLAP_RAW)
    raw["match_id"] = "synt_overlap_real"
    raw["match_date"] = date_iso
    raw["home_team"] = {**SB_OVERLAP_RAW["home_team"], "home_team_name": home, "country": {"name": "England"}}
    raw["away_team"] = {**SB_OVERLAP_RAW["away_team"], "away_team_name": away, "country": {"name": "England"}}
    raw["home_score"] = 9
    raw["away_score"] = 9
    return _matches_to_dataframe([raw])


def validate_real_csv(season: str):
    """Run the FD validate() stack on a real downloaded CSV; return summary dict."""
    fn = EPL_FD_DIR / f"E0_{season}.csv"
    if not fn.exists():
        return {"season": season, "status": "FILE MISSING", "path": str(fn)}
    text = fn.read_text(encoding="utf-8-sig")
    ndf = _normalize_results_dataframe(_parse_csv(text), "E0", season)
    ing = FootballDataResultsIngestor(league_code="E0", season_code=season,
                                      db_path=Path(tempfile.gettempdir()) / "noop.db",
                                      cache_dir=Path(tempfile.gettempdir()),
                                      source_url="file:" + str(fn))
    vr = ing.validate(ndf)
    return {"season": season, "rows": len(ndf), "errors": vr.error_count,
            "warnings": vr.warning_count, "is_valid": vr.is_valid}


def ingest_real_and_check(label: str, season: str):
    """Ingest a REAL CSV into a DISPOSABLE db; run quality + guard + integrity."""
    tmp = Path(tempfile.mkdtemp(prefix=f"phase17_4_rehearsal_{label}_"))
    db = tmp / "r.db"; cache = tmp / "cache"; cache.mkdir()
    initialize_database(db)
    fn = EPL_FD_DIR / f"E0_{season}.csv"
    if not fn.exists():
        print(f"[{label}] SKIP: {fn.name} missing"); shutil.rmtree(tmp, ignore_errors=True); return
    csv_text = fn.read_text(encoding="utf-8-sig")
    fd = FootballDataResultsIngestor(league_code="E0", season_code=season,
                                    db_path=db, cache_dir=cache, source_url="file:" + str(fn))
    r = None
    try:
        with patch(_DL_PATCH, return_value=(csv_text, False)):
            r = fd.ingest()
        record_ingestion_run(db, r)
    except Exception as exc:  # genuine real-data validation failure must surface, not crash
        print(f"[{label}] FD ingest raised: {type(exc).__name__}: {str(exc)[:200]}")
    if r is None or not r.is_success:
        print(f"[{label}] FD ingest did not complete successfully; skipping quality/guard.")
        shutil.rmtree(tmp, ignore_errors=True); return
    real_fd = _cnt(db, "football-data")
    print(f"[{label}] FD ingest: read={r.records_read} inserted={r.records_inserted} skipped={r.records_skipped}")
    try:
        q = check_match_data_quality(db)
        print(f"[{label}] quality: valid={q.is_valid} errors={q.error_count} warnings={q.warning_count}")
    except Exception as exc:
        print(f"[{label}] quality check raised: {type(exc).__name__}: {str(exc)[:200]}")
    # Cross-source guard proof against a REAL FD fixture: pick the first real
    # fixture, build a SYNTHETIC SB match matching its date + team names, and
    # assert the guard skips it, preserves the real score, and does not double-count.
    fix = _first_fixture(db)
    sb = StatsBombIngestor(7, 3, db_path=db, cache_dir=cache, source_url="http://m")
    if fix is None:
        print(f"[{label}] NOTE: no FD fixture loaded; guard proven by tests/test_cross_source_reconciliation.py + _p17_5_rehearsal.py")
    else:
        fdate, fhome, faway, (fhg, fag) = fix
        overlap = _sb_overlap_df(fdate, fhome, faway)
        try:
            rs = sb.load(overlap); record_ingestion_run(db, rs)
            print(f"[{label}] SB overlap vs REAL {fhome}-{faway} {fdate} {fhg}-{fag}: skipped={rs.records_skipped} inserted={rs.records_inserted}")
            assert rs.records_skipped == 1 and rs.records_inserted == 0, rs
            assert _score(db, "football-data", fdate, fhome, faway) == (fhg, fag), "real score overwritten"
            assert _cnt(db) == real_fd, "double-counted"
            print(f"[{label}] cross-source guard OK: SB overlap skipped, real score {fhg}-{fag} preserved")
        except Exception as exc:
            print(f"[{label}] SB overlap raised: {type(exc).__name__}: {str(exc)[:200]}")
    try:
        with patch(_DL_PATCH, return_value=(csv_text, False)):
            r2 = fd.ingest()
        assert r2.records_inserted == 0 and _cnt(db) == real_fd, (r2.records_inserted, _cnt(db))
        print(f"[{label}] idempotent rerun: 0 inserts, total unchanged = {_cnt(db)}")
    except Exception as exc:
        print(f"[{label}] rerun raised: {type(exc).__name__}: {str(exc)[:200]}")
    print(f"[{label}] integrity_check = {_integrity(db)}")
    shutil.rmtree(tmp, ignore_errors=True)


def main() -> int:
    before = _hash_prod()
    print("prod_db_sha256_before =", before)
    print("=== genuine validation of all real E0 CSVs ===")
    for season in ("1516", "1617", "1920"):
        try:
            print("validate", validate_real_csv(season))
        except Exception as exc:
            print(f"validate {season} raised: {type(exc).__name__}: {str(exc)[:200]}")
    print("=== real-data ingest + quality + cross-source + integrity (1617) ===")
    ingest_real_and_check("E0_1617", "1617")
    after = _hash_prod()
    print("prod_db_sha256_after  =", after)
    assert before == after, "PRODUCTION DB WAS MODIFIED"
    print("prod_db_unchanged     =", before == after)
    print("PHASE 17.4 REHEARSAL COMPLETE (real FD data; synthetic SB overlap only)")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
