"""Phase 17.5 — Disposable Import Rehearsal.

End-to-end rehearsal of the Football-Data + StatsBomb ingestion pipeline with the
Phase 17.2 symmetric cross-source deduplication guard, against a THROWAWAY
database created under a clearly-named temporary directory.

SAFETY
- Production DB: ``data/football_intelligence.db`` is NEVER written to. Its SHA-256
  is hashed READ-ONLY before and after the run and asserted equal.
- Only files created by this rehearsal (the temp dir) are removed at the end.

DATA PROVENANCE
- SYNTHETIC records only (clearly labelled). They validate pipeline safety
  (no double-counting, score preservation, idempotency, rollback, audit rows,
  integrity) -- NOT independent real-world score agreement.
"""
from __future__ import annotations

import hashlib
import shutil
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

from football_intelligence.database import initialize_database
from football_intelligence.database.connection import database_connection
from football_intelligence.data.ingestion.football_data_results import (
    FootballDataResultsIngestor,
)
from football_intelligence.data.ingestion.statsbomb import (
    StatsBombIngestor,
    _matches_to_dataframe,
    _normalize_match,
    record_ingestion_run,
)

PROD_DB = Path("data/football_intelligence.db")

# SYNTHETIC Football-Data CSV (E0 2016/17). FTR is validation-only, not stored.
FD_CSV = (
    "Date,HomeTeam,AwayTeam,FTHG,FTAG,FTR\n"
    "13/08/2016,Arsenal,Burnley,1,0,H\n"      # overlaps SB match 'sb_a' (date+teams)
    "20/08/2016,Chelsea,Tottenham,0,1,A\n"    # distinct date vs SB 'sb_b'
)

# SYNTHETIC StatsBomb match records (raw, normalized via _normalize_match).
SB_RAW = [
    {
        "match_id": "sb_a", "match_date": "2016-08-13", "kick_off": "16:00:00.000",
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
    },
    {
        "match_id": "sb_b", "match_date": "2017-08-13", "kick_off": "15:00:00.000",
        "competition": {"competition_id": 7, "season_id": 4,
                        "competition_name": "Premier League", "country_name": "England",
                        "season_name": "2017/2018"},
        "season": {"season_id": 4, "season_name": "2017/2018"},
        "home_team": {"home_team_id": "5003", "home_team_name": "Chelsea",
                      "country": {"name": "England"}},
        "away_team": {"away_team_id": "5004", "away_team_name": "Tottenham",
                      "country": {"name": "England"}},
        "home_score": 1, "away_score": 2, "match_status": "available",
        "competition_stage": {"name": "League Round"},
        "metadata": {"data_version": "1.1.0"}, "match_week": 1,
        "last_updated": "2024-01-15T10:30:00.000",
    },
]


def _sb_df():
    return _matches_to_dataframe(list(SB_RAW))


def _hash_prod() -> str:
    h = hashlib.sha256()
    with open(PROD_DB, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _cnt(db, prov=None):
    with database_connection(db) as c:
        if prov:
            return c.execute("SELECT COUNT(*) FROM matches WHERE provider=?", (prov,)).fetchone()[0]
        return c.execute("SELECT COUNT(*) FROM matches").fetchone()[0]


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


def _audit_count(db) -> int:
    with database_connection(db) as c:
        return c.execute("SELECT COUNT(*) FROM ingestion_runs").fetchone()[0]


_FD_DL_PATCH = "football_intelligence.data.ingestion.football_data_results._download_csv"


def run_order(label: str, first: str):
    """first='fd' -> Football-Data then StatsBomb; first='sb' -> reverse."""
    tmp = Path(tempfile.mkdtemp(prefix=f"phase17_5_rehearsal_{label}_"))
    db = tmp / "rehearsal.db"
    cache = tmp / "cache"
    cache.mkdir()
    initialize_database(db)

    fd = FootballDataResultsIngestor(league_code="E0", season_code="1617",
                                     db_path=db, cache_dir=cache,
                                     source_url="http://mock/E0/1617.csv")
    sb = StatsBombIngestor(7, 3, db_path=db, cache_dir=cache, source_url="http://m")
    sb_df = _sb_df()

    def _fd_ingest():
        with patch(_FD_DL_PATCH, return_value=(FD_CSV, False)):
            r = fd.ingest()
        record_ingestion_run(db, r)
        return r

    def _sb_load():
        r = sb.load(sb_df)
        record_ingestion_run(db, r)
        return r

    if first == "fd":
        r1 = _fd_ingest(); r2 = _sb_load()
        assert r1.records_inserted == 2 and r1.records_skipped == 0, (label, r1)
        assert r2.records_skipped == 1 and r2.records_inserted == 1, (label, r2)
        assert _cnt(db) == 3, (label, _cnt(db))
        assert _cnt(db, "football-data") == 2 and _cnt(db, "statsbomb") == 1, label
        assert _score(db, "football-data", "2016-08-13", "Arsenal", "Burnley") == (1, 0), label
        r3 = _fd_ingest()
        assert r3.records_inserted == 0, (label, r3)
        assert _cnt(db) == 3, (label, _cnt(db))
    else:
        r2 = _sb_load(); r1 = _fd_ingest()
        assert r2.records_inserted == 2 and r2.records_skipped == 0, (label, r2)
        assert r1.records_inserted == 1 and r1.records_skipped == 1, (label, r1)
        assert _cnt(db) == 3, (label, _cnt(db))
        assert _cnt(db, "football-data") == 1 and _cnt(db, "statsbomb") == 2, label
        assert _score(db, "statsbomb", "2016-08-13", "Arsenal", "Burnley") == (2, 1), label
        r3 = _sb_load()
        assert r3.records_inserted == 0, (label, r3)
        assert _cnt(db) == 3, (label, _cnt(db))

    assert _integrity(db) == "ok", (label, "integrity")
    assert _audit_count(db) == 3, (label, "audits", _audit_count(db))
    print(f"[{label}] OK  total={_cnt(db)} integrity={_integrity(db)} "
          f"audits={_audit_count(db)} fd={_cnt(db,'football-data')} sb={_cnt(db,'statsbomb')}")
    shutil.rmtree(tmp, ignore_errors=True)


def main() -> int:
    if not PROD_DB.exists():
        print(f"ABORT: production DB not found at {PROD_DB} (expected read-only).")
        return 2
    before = _hash_prod()
    print(f"prod_db_sha256_before = {before}")
    run_order("FD_then_SB", "fd")
    run_order("SB_then_FD", "sb")
    after = _hash_prod()
    print(f"prod_db_sha256_after  = {after}")
    assert before == after, "PRODUCTION DB WAS MODIFIED - ABORT."
    print("prod_db_unchanged     =", before == after)
    print("PHASE 17.5 REHEARSAL: ALL CHECKS PASSED (synthetic data only)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
