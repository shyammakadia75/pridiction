"""Football-Data.co.uk historical odds ingestion adapter.

Implements a :class:`BaseIngestor` sub-class that downloads (or reads from a
deterministic cache) CSV files from the Football-Data.co.uk archive
(https://www.football-data.co.uk), normalizes the wide-format odds columns
into a long-format DataFrame, reconciles each match against the existing
``matches`` table, and upserts odds records with provenance tracking.

Football-Data.co.uk CSV format
------------------------------
* URL pattern: ``https://www.football-data.co.uk/mmz4281/{season}/{league}.csv``
* Season code is a 4-digit string, e.g. ``2324`` for the 2023/24 season.
* League codes include ``E0`` (Premier League), ``E1`` (Championship),
  ``SP1`` (La Liga), ``D1`` (Bundesliga), ``I1`` (Serie A), etc.
* Opening odds use column suffixes ``H`` (home), ``D`` (draw), ``A`` (away),
  ``>2.5``, ``<2.5``, ``AHH``, ``AHA``.
* Closing odds insert a ``C`` before the selection: ``CH``, ``CD``, ``CA``,
  ``C>2.5``, ``C<2.5``, ``CAHH``, ``CAHA``.
* Files are UTF-8 with BOM and contain trailing blank lines.

Licensing / attribution
-----------------------
Football-Data.co.uk distributes results and odds from the betting industry.
When publishing analysis based on this data, credit Football-Data.co.uk as
the source. Files are cached under ``data/raw/odds`` and are never committed
to version control.
"""
from __future__ import annotations

import csv
import io
import json
import re
import sqlite3
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, Tuple

import pandas as pd

from ...config import RAW_DATA_DIR
from ...database import database_connection, resolve_db_path
from ...exceptions import DatabaseError, IngestionError
from ...logging_config import get_logger
from ..ingestion.base import BaseIngestor, IngestionResult
from ..ingestion.statsbomb import _safe_cache_path, _utcnow
from ..validation.validators import (
    ValidationResult,
    validate_no_missing_values,
    validate_positive_odds,
    validate_required_columns,
)

logger = get_logger("data.ingestion.odds")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Provider identifier stored alongside every record.
PROVIDER = "football-data"

#: Base URL for the Football-Data.co.uk CSV archive.
ODDS_BASE_URL = "https://www.football-data.co.uk/mmz4281"

#: Default HTTP timeout in seconds.
DEFAULT_TIMEOUT = 30

#: HTTP User-Agent identifying this project.
HTTP_USER_AGENT = (
    "football-intelligence/0.1.0 "
    "(+https://github.com/shyammakadia75/pridiction)"
)

#: Mapping of Football-Data league codes to human-readable descriptions.
LEAGUE_CODES: dict[str, str] = {
    "E0": "England - Premier League",
    "E1": "England - Championship",
    "E2": "England - League One",
    "E3": "England - League Two",
    "EC": "England - Conference",
    "SC0": "Scotland - Premiership",
    "SC1": "Scotland - Championship",
    "SC2": "Scotland - League One",
    "SC3": "Scotland - League Two",
    "D1": "Germany - Bundesliga",
    "D2": "Germany - 2. Bundesliga",
    "D3": "Germany - 3. Liga",
    "I1": "Italy - Serie A",
    "I2": "Italy - Serie B",
    "SP1": "Spain - La Liga",
    "SP2": "Spain - Segunda Division",
    "F1": "France - Ligue 1",
    "F2": "France - Ligue 2",
    "N1": "Netherlands - Eredivisie",
    "B1": "Belgium - Pro League",
    "P1": "Portugal - Primeira Liga",
    "T1": "Turkey - Super Lig",
}

#: Columns expected in every Football-Data CSV (minimum required).
REQUIRED_CSV_COLUMNS = ["Date", "HomeTeam", "AwayTeam"]

#: -----------------------------------------------------------------------
#: Bookmaker column mapping
#: Each key is a Football-Data CSV column name; each value is a
#: (bookmaker, market, selection, odds_type) tuple where
#: *odds_type* is ``"opening"`` or ``"closing"``.
#: -----------------------------------------------------------------------
ODDS_COLUMN_MAP: dict[str, tuple[str, str, str, str]] = {
    # -- Bet365: match odds
    "B365H":  ("bet365",      "match_odds",     "home",  "opening"),
    "B365D":  ("bet365",      "match_odds",     "draw",  "opening"),
    "B365A":  ("bet365",      "match_odds",     "away",  "opening"),
    "B365CH": ("bet365",      "match_odds",     "home",  "closing"),
    "B365CD": ("bet365",      "match_odds",     "draw",  "closing"),
    "B365CA": ("bet365",      "match_odds",     "away",  "closing"),
    # -- Bet365: over / under 2.5
    "B365>2.5":  ("bet365",       "over_under_2_5", "over",  "opening"),
    "B365<2.5":  ("bet365",       "over_under_2_5", "under", "opening"),
    "B365C>2.5": ("bet365",       "over_under_2_5", "over",  "closing"),
    "B365C<2.5": ("bet365",       "over_under_2_5", "under", "closing"),
    # -- Bet365: asian handicap
    "B365AHH":  ("bet365",      "asian_handicap", "home",  "opening"),
    "B365AHA":  ("bet365",      "asian_handicap", "away",  "opening"),
    "B365CAHH": ("bet365",      "asian_handicap", "home",  "closing"),
    "B365CAHA": ("bet365",      "asian_handicap", "away",  "closing"),
    # -- Pinnacle (PS prefix for 1X2, P prefix for O/U & Asian)
    "PSH":  ("pinnacle",      "match_odds",     "home",  "opening"),
    "PSD":  ("pinnacle",      "match_odds",     "draw",  "opening"),
    "PSA":  ("pinnacle",      "match_odds",     "away",  "opening"),
    "PSCH": ("pinnacle",      "match_odds",     "home",  "closing"),
    "PSCD": ("pinnacle",      "match_odds",     "draw",  "closing"),
    "PSCA": ("pinnacle",      "match_odds",     "away",  "closing"),
    "P>2.5":   ("pinnacle",       "over_under_2_5", "over",  "opening"),
    "P<2.5":   ("pinnacle",       "over_under_2_5", "under", "opening"),
    "PC>2.5":  ("pinnacle",       "over_under_2_5", "over",  "closing"),
    "PC<2.5":  ("pinnacle",       "over_under_2_5", "under", "closing"),
    "PAHH":  ("pinnacle",      "asian_handicap", "home",  "opening"),
    "PAHA":  ("pinnacle",      "asian_handicap", "away",  "opening"),
    "PCAHH": ("pinnacle",      "asian_handicap", "home",  "closing"),
    "PCAHA": ("pinnacle",      "asian_handicap", "away",  "closing"),
    # -- Bet&Win
    "BWH":  ("betandwin",     "match_odds",     "home",  "opening"),
    "BWD":  ("betandwin",     "match_odds",     "draw",  "opening"),
    "BWA":  ("betandwin",     "match_odds",     "away",  "opening"),
    "BWCH": ("betandwin",     "match_odds",     "home",  "closing"),
    "BWCD": ("betandwin",     "match_odds",     "draw",  "closing"),
    "BWCA": ("betandwin",     "match_odds",     "away",  "closing"),
    # -- Interwetten
    "IWH":  ("interwetten",   "match_odds",     "home",  "opening"),
    "IWD":  ("interwetten",   "match_odds",     "draw",  "opening"),
    "IWA":  ("interwetten",   "match_odds",     "away",  "opening"),
    "IWCH": ("interwetten",   "match_odds",     "home",  "closing"),
    "IWCD": ("interwetten",   "match_odds",     "draw",  "closing"),
    "IWCA": ("interwetten",   "match_odds",     "away",  "closing"),
    # -- William Hill
    "WHH":  ("williamhill",   "match_odds",     "home",  "opening"),
    "WHD":  ("williamhill",   "match_odds",     "draw",  "opening"),
    "WHA":  ("williamhill",   "match_odds",     "away",  "opening"),
    "WHCH": ("williamhill",   "match_odds",     "home",  "closing"),
    "WHCD": ("williamhill",   "match_odds",     "draw",  "closing"),
    "WHCA": ("williamhill",   "match_odds",     "away",  "closing"),
    # -- VC Bet
    "VCH":  ("vcbet",         "match_odds",     "home",  "opening"),
    "VCD":  ("vcbet",         "match_odds",     "draw",  "opening"),
    "VCA":  ("vcbet",         "match_odds",     "away",  "opening"),
    "VCCH": ("vcbet",         "match_odds",     "home",  "closing"),
    "VCCD": ("vcbet",         "match_odds",     "draw",  "closing"),
    "VCCA": ("vcbet",         "match_odds",     "away",  "closing"),
    # -- Market max / avg
    "MaxH":  ("market_max", "match_odds",     "home",  "opening"),
    "MaxD":  ("market_max", "match_odds",     "draw",  "opening"),
    "MaxA":  ("market_max", "match_odds",     "away",  "opening"),
    "AvgH":  ("market_avg", "match_odds",     "home",  "opening"),
    "AvgD":  ("market_avg", "match_odds",     "draw",  "opening"),
    "AvgA":  ("market_avg", "match_odds",     "away",  "opening"),
    "MaxCH": ("market_max", "match_odds",     "home",  "closing"),
    "MaxCD": ("market_max", "match_odds",     "draw",  "closing"),
    "MaxCA": ("market_max", "match_odds",     "away",  "closing"),
    "AvgCH": ("market_avg", "match_odds",     "home",  "closing"),
    "AvgCD": ("market_avg", "match_odds",     "draw",  "closing"),
    "AvgCA": ("market_avg", "match_odds",     "away",  "closing"),
    "Max>2.5":  ("market_max", "over_under_2_5", "over",  "opening"),
    "Max<2.5":  ("market_max", "over_under_2_5", "under", "opening"),
    "Avg>2.5":  ("market_avg", "over_under_2_5", "over",  "opening"),
    "Avg<2.5":  ("market_avg", "over_under_2_5", "under", "opening"),
    "MaxC>2.5": ("market_max", "over_under_2_5", "over",  "closing"),
    "MaxC<2.5": ("market_max", "over_under_2_5", "under", "closing"),
    "AvgC>2.5": ("market_avg", "over_under_2_5", "over",  "closing"),
    "AvgC<2.5": ("market_avg", "over_under_2_5", "under", "closing"),
}

# Columns that carry the opening/closing *handicap line* (not odds).
# These are intentionally excluded from the odds mapping.
_HC_LINE_COLUMNS = frozenset({"AHh", "AHCh"})


def _default_cache_dir() -> Path:
    """Default cache directory for Football-Data raw files."""
    return RAW_DATA_DIR / "odds"


def _http_get(url: str, timeout: int = DEFAULT_TIMEOUT) -> bytes:
    """Fetch raw bytes from *url* with a timeout.

    Raises :class:`IngestionError` on any HTTP or network failure so that
    callers never receive partial / error-page responses.
    """
    req = urllib.request.Request(url, headers={"User-Agent": HTTP_USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.read()
    except urllib.error.HTTPError as exc:
        raise IngestionError(
            f"HTTP {exc.code} fetching {url}: {exc.reason}"
        ) from exc
    except urllib.error.URLError as exc:
        reason = exc.reason
        if "timed out" in str(reason).lower():
            raise IngestionError(
                f"Timeout ({timeout}s) fetching {url}"
            ) from exc
        raise IngestionError(
            f"Network error fetching {url}: {reason}"
        ) from exc
    except TimeoutError as exc:
        raise IngestionError(
            f"Timeout ({timeout}s) fetching {url}"
        ) from exc


def _download_csv(
    url: str,
    cache_path: Path,
    timeout: int,
    refresh: bool,
) -> Tuple[str, bool]:
    """Fetch CSV text from *url*, caching the raw bytes to *cache_path*.

    Returns ``(csv_text, was_from_cache)``.  When the cache file exists
    and *refresh* is ``False``, the cached copy is read instead of hitting
    the network.
    """
    cache_path = Path(cache_path)
    if cache_path.exists() and not refresh:
        logger.info("Using cached file: %s", cache_path)
        try:
            text = cache_path.read_text(encoding="utf-8-sig")
        except OSError as exc:
            raise IngestionError(
                f"Cache read failed for {cache_path}: {exc}"
            ) from exc
        return text, True
    logger.info("Downloading: %s", url)
    raw = _http_get(url, timeout)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_bytes(raw)
    return raw.decode("utf-8-sig"), False


def _parse_csv(csv_text: str) -> pd.DataFrame:
    """Parse Football-Data CSV text into a DataFrame.

    Handles UTF-8 BOM (already stripped by the caller via ``utf-8-sig``),
    trailing blank lines, and the ``Date`` field.
    """
    reader = csv.DictReader(io.StringIO(csv_text))
    rows: list[dict[str, str]] = []
    for row in reader:
        # Skip rows where Date is empty (trailing blank lines).
        if not row.get("Date"):
            continue
        rows.append(row)
    if not rows:
        return pd.DataFrame(columns=REQUIRED_CSV_COLUMNS)
    df = pd.DataFrame(rows)
    return df


def _normalize_match_date(date_str: str) -> Optional[str]:
    """Convert a Football-Data date string to ISO ``YYYY-MM-DD``.

    Football-Data uses ``DD/MM/YYYY`` (sometimes ``D/M/YYYY``).
    Returns ``None`` if the string cannot be parsed.
    """
    if not date_str:
        return None
    date_str = date_str.strip()
    for fmt in ("%d/%m/%Y", "%d/%m/%y", "%Y-%m-%d"):
        try:
            return datetime.strptime(date_str, fmt).strftime("%Y-%m-%d")
        except ValueError:
            continue
    return None


# ---------------------------------------------------------------------------
# Team name normalization for match reconciliation
# ---------------------------------------------------------------------------

#: Explicit alias mapping from a *normalized* Football-Data team name to a
#: *normalized* canonical name used in the database.  Keys are lower-case
#: and stripped; values are the canonical lower-case form.
TEAM_ALIASES: dict[str, str] = {
    "man united": "manchester united",
    "man city": "manchester city",
    "spurs": "tottenham hotspur",
    "tottenham": "tottenham hotspur",
    "lei": "leicester city",
    "bournemouth": "afc bournemouth",
    "brighton": "brighton and hove albion",
    "newcastle": "newcastle united",
    "west ham": "west ham united",
    "sheffield utd": "sheffield united",
    "sheffield wed": "sheffield wednesday",
    "wolves": "wolverhampton wanderers",
    "wigan": "wigan athletic",
    "leeds": "leeds united",
}


def _normalize_team_name(name: str) -> str:
    """Normalize a team name for fuzzy matching.

    Lowercases, strips whitespace, removes common suffixes and punctuation,
    and applies :data:`TEAM_ALIASES` for known mismatches between
    Football-Data short names and database canonical names.
    """
    if not name:
        return ""
    name = name.strip().lower()
    for suffix in (" f.c.", ".f.c.", " f.c", " fc", " a.f.c.", " a.f.c"):
        if name.endswith(suffix):
            name = name[:-len(suffix)].strip()
    name = re.sub(r"[.']", "", name)
    name = re.sub(r"\s+", " ", name).strip()
    name = TEAM_ALIASES.get(name, name)
    return name


# ---------------------------------------------------------------------------
# Odds extraction from wide CSV to long-format records
# ---------------------------------------------------------------------------

def _extract_odds_records(row: dict[str, str]) -> list[dict[str, Any]]:
    """Extract all odds from a single CSV row into long-format records.

    Each returned dict has keys: ``match_date``, ``home_team``,
    ``away_team``, ``bookmaker``, ``market``, ``selection``,
    ``odds_type``, ``decimal_odds``.
    """
    date = _normalize_match_date(row.get("Date", "")) or row.get("Date", "")
    home = (row.get("HomeTeam") or "").strip()
    away = (row.get("AwayTeam") or "").strip()
    records: list[dict[str, Any]] = []
    for col, (bookmaker, market, selection, odds_type) in ODDS_COLUMN_MAP.items():
        if col in _HC_LINE_COLUMNS:
            continue
        raw_val = row.get(col)
        if raw_val is None or raw_val.strip() == "":
            continue
        try:
            odds = float(raw_val)
        except (ValueError, TypeError):
            continue
        if odds <= 0 or odds != odds:  # NaN check
            continue
        records.append({
            "match_date": date,
            "home_team": home,
            "away_team": away,
            "bookmaker": bookmaker,
            "market": market,
            "selection": selection,
            "odds_type": odds_type,
            "decimal_odds": odds,
        })
    return records


# ---------------------------------------------------------------------------
# Database helpers
# ---------------------------------------------------------------------------

def _ensure_initialized(db_path: Optional[Path], conn) -> None:
    """Verify the schema_version table has been created."""
    try:
        row = conn.execute(
            "SELECT MAX(version) FROM schema_version"
        ).fetchone()
    except sqlite3.OperationalError as exc:
        if "schema_version" in str(exc):
            raise DatabaseError(
                f"Database '{db_path}' is not initialized. "
                "Run 'football-intelligence init-db' first."
            ) from exc
        raise
    if row is None or row[0] is None:
        raise DatabaseError(
            f"Database '{db_path}' is not initialized. "
            "Run 'football-intelligence init-db' first."
        )


def _find_matches_for_reconciliation(
    conn, date_iso: str, home_team: str, away_team: str
) -> list[int]:
    """Look up match IDs in the database by date + team names.

    Returns a list of matching match IDs (0, 1, or more).  Uses normalized
    team name comparison so that Football-Data short names can match
    canonical names from the StatsBomb ingestion.
    """
    matches = conn.execute(
        "SELECT m.id AS match_id, match_date, ht.team_name AS home_name, "
        "at.team_name AS away_name "
        "FROM matches m "
        "JOIN teams ht ON m.home_team_id = ht.id "
        "JOIN teams at ON m.away_team_id = at.id "
        "WHERE m.match_date IS NOT NULL"
    ).fetchall()
    home_norm = _normalize_team_name(home_team)
    away_norm = _normalize_team_name(away_team)
    found: list[int] = []
    for m in matches:
        match_date = m["match_date"] or ""
        if match_date[:10] != date_iso:
            continue
        db_home = _normalize_team_name(m["home_name"] or "")
        db_away = _normalize_team_name(m["away_name"] or "")
        if home_norm == db_home and away_norm == db_away:
            found.append(int(m["match_id"]))
    return found


def _upsert_odds(
    conn,
    match_id: int,
    provider: str,
    bookmaker: str,
    market: str,
    selection: str,
    odds_type: str,
    decimal_odds: float,
    observed_at: str,
    source_reference: str,
    now: str,
) -> str:
    """Insert or update a single odds record.

    Returns ``"inserted"`` or ``"updated"``.
    """
    row = conn.execute(
        "SELECT id FROM odds WHERE match_id=? AND provider=? "
        "AND bookmaker=? AND market=? AND selection=? "
        "AND odds_type=? AND source_reference=?",
        (match_id, provider, bookmaker, market, selection, odds_type, source_reference),
    ).fetchone()
    if row:
        conn.execute(
            "UPDATE odds SET decimal_odds=?, observed_at=?, updated_at=? "
            "WHERE id=?",
            (decimal_odds, observed_at, now, int(row["id"])),
        )
        return "updated"
    conn.execute(
        "INSERT INTO odds "
        "(match_id, provider, bookmaker, market, selection, odds_type, "
        "decimal_odds, observed_at, source_reference, created_at, updated_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (match_id, provider, bookmaker, market, selection, odds_type,
         decimal_odds, observed_at, source_reference, now, now),
    )
    return "inserted"


def _record_data_source(
    conn,
    source_ref: str,
    retrieval_ts: str,
    metadata_json: str,
    now: str,
) -> None:
    """Record provenance metadata for a downloaded source file."""
    conn.execute(
        "INSERT INTO data_sources "
        "(provider, dataset_name, source_reference, retrieval_timestamp, "
        "data_version, metadata, created_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        (
            PROVIDER,
            "odds",
            source_ref,
            retrieval_ts,
            None,  # football-data CSVs have no embedded version
            metadata_json,
            now,
        ),
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def list_leagues() -> list[tuple[str, str]]:
    """Return a sorted list of ``(league_code, description)`` tuples."""
    return sorted(LEAGUE_CODES.items())


def record_ingestion_run(db_path: Optional[Path], result: IngestionResult) -> None:
    """Persist an :class:`IngestionResult` as a row in ``ingestion_runs``.

    Always creates a new row (audit trail).  Safe to call for both successful
    and failed runs.  Only the columns declared by :data:`INGESTION_RUNS_SQL`
    are written; :class:`IngestionResult` carries additional in-memory summary
    fields (``records_skipped``/``records_matched``/``records_unmatched``/
    ``records_ambiguous``/``records_invalid``) that are intentionally not
    persisted -- no schema version or migration in this codebase declares them,
    so persisting them here would invent schema.  Mirrors
    :func:`statsbomb.record_ingestion_run`.
    """
    path = resolve_db_path(db_path)
    now = _utcnow()
    with database_connection(path) as conn:
        conn.execute(
            # Aligned to ``INGESTION_RUNS_SQL`` (schema.py). The previous
            # 15-column form referenced columns no schema version declares
            # (records_skipped/matched/unmatched/ambiguous/invalid) and raised
            # "table ingestion_runs has no column named ...". Those fields
            # remain on IngestionResult for CLI summaries but are intentionally
            # not persisted. Mirrors statsbomb.record_ingestion_run.
            "INSERT INTO ingestion_runs "
            "(provider, dataset_name, started_at, completed_at, status, "
            "records_read, records_inserted, records_updated, error_message, "
            "created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                result.provider,
                result.dataset_name,
                result.started_at,
                result.completed_at,
                result.status,
                result.records_read,
                result.records_inserted,
                result.records_updated,
                result.error_message,
                now,
            ),
        )
        logger.info(
            "Recorded ingestion run: %s/%s status=%s read=%d inserted=%d updated=%d",
            result.provider,
            result.dataset_name,
            result.status,
            result.records_read,
            result.records_inserted,
            result.records_updated,
        )

# ---------------------------------------------------------------------------
# Ingestor
# ---------------------------------------------------------------------------

class OddsIngestor(BaseIngestor):
    """Ingestor for Football-Data.co.uk historical odds.

    Downloads (or reads from cache) a CSV file for a given ``league_code``
    and ``season_code``, normalizes the wide-format odds columns into
    long-format records, reconciles each match against the existing
    ``matches`` table, and upserts odds records with provenance tracking.

    Args:
        league_code: Football-Data league code (e.g. ``E0``).
        season_code: 4-digit season code (e.g. ``2324`` for 2023/24).
        db_path: Target SQLite database path (defaults to configured path).
        cache_dir: Override for the raw-file cache directory.
        refresh: When ``True``, force re-download even if cached.
        source_url: Override the download URL (testing).
        timeout: HTTP timeout in seconds.
    """

    def __init__(
        self,
        league_code: str,
        season_code: str,
        db_path: Optional[Path] = None,
        cache_dir: Optional[Path] = None,
        refresh: bool = False,
        source_url: Optional[str] = None,
        timeout: int = DEFAULT_TIMEOUT,
    ) -> None:
        league_code = league_code.strip().upper()
        season_code = season_code.strip()
        if league_code not in LEAGUE_CODES:
            raise IngestionError(
                f"Unknown league code '{league_code}'. "
                f"Valid codes: {sorted(LEAGUE_CODES.keys())}"
            )
        if not re.match(r"^\d{4}$", season_code):
            raise IngestionError(
                f"Invalid season code '{season_code}'. "
                "Must be 4 digits, e.g. '2324' for the 2023/24 season."
            )
        self._league_code = league_code
        self._season_code = season_code
        self._db_path = db_path
        self._cache_dir = Path(cache_dir) if cache_dir else _default_cache_dir()
        self._refresh = refresh
        self._timeout = timeout
        self._source_url = source_url or (
            f"{ODDS_BASE_URL}/{season_code}/{league_code}.csv"
        )
        self._was_cached: bool = False
        self._retrieval_ts: Optional[str] = None

    # -- properties -------------------------------------------------------

    @property
    def provider(self) -> str:
        return PROVIDER

    @property
    def dataset_name(self) -> str:
        return "odds"

    @property
    def source_reference(self) -> Optional[str]:
        return self._source_url

    @property
    def league_code(self) -> str:
        return self._league_code

    @property
    def season_code(self) -> str:
        return self._season_code

    @property
    def was_cached(self) -> bool:
        """True if the last ``retrieve()`` used a cached file."""
        return self._was_cached

    @property
    def retrieval_timestamp(self) -> Optional[str]:
        return self._retrieval_ts

    @property
    def source_url(self) -> str:
        return self._source_url

    # -- cache path -------------------------------------------------------

    def _cache_path(self) -> Path:
        return _safe_cache_path(
            self._cache_dir, self._league_code, f"{self._season_code}.csv"
        )

    # -- pipeline ---------------------------------------------------------

    def retrieve(self) -> pd.DataFrame:
        """Download (or read cached) the CSV, return long-format odds rows."""
        url = self._source_url
        cache_path = self._cache_path()
        self._retrieval_ts = _utcnow()
        csv_text, self._was_cached = _download_csv(
            url, cache_path, self._timeout, self._refresh,
        )
        df = _parse_csv(csv_text)
        logger.info(
            "Retrieved %d CSV row(s) for %s season %s (source: %s)",
            len(df),
            self._league_code,
            self._season_code,
            "cache" if self._was_cached else "network",
        )
        # Normalize all odds columns into long format.
        all_records: list[dict[str, Any]] = []
        for _, row in df.iterrows():
            all_records.extend(_extract_odds_records(row.to_dict()))
        self._odds_df = pd.DataFrame(all_records)
        return self._odds_df

    def validate(self, data: pd.DataFrame) -> ValidationResult:
        """Validate the long-format odds DataFrame before loading."""
        result = ValidationResult(row_count=len(data))
        result.merge(validate_required_columns(
            data, ["match_date", "home_team", "away_team", "decimal_odds"]
        ))
        result.merge(validate_no_missing_values(
            data, ["decimal_odds", "match_date", "home_team", "away_team"]
        ))
        result.merge(validate_positive_odds(data, "decimal_odds"))
        return result

    def load(self, data: pd.DataFrame) -> IngestionResult:
        """Persist validated odds, returning insert / update / unmatched counts.

        All odds records are written inside a single transaction.  Re-running
        with the same data is idempotent: existing rows are updated, no
        duplicates are created.
        """
        db_path = resolve_db_path(self._db_path)
        now = _utcnow()
        inserted = 0
        updated = 0
        matched_rows = 0
        unmatched_rows = 0
        ambiguous_rows = 0

        match_keys = ["match_date", "home_team", "away_team"]
        if not data.empty:
            unique_matches = data[match_keys].drop_duplicates()
        else:
            unique_matches = pd.DataFrame(columns=match_keys)

        metadata_json = json.dumps(
            {
                "league_code": self._league_code,
                "season_code": self._season_code,
                "record_count": len(data),
                "retrieval_source": "cache" if self._was_cached else "network",
            }
        )

        with database_connection(db_path) as conn:
            _ensure_initialized(db_path, conn)
            _record_data_source(
                conn,
                self._source_url,
                self._retrieval_ts or now,
                metadata_json,
                now,
            )
            for _, mrow in unique_matches.iterrows():
                match_date = mrow["match_date"]
                home_team = mrow["home_team"]
                away_team = mrow["away_team"]
                found_ids = _find_matches_for_reconciliation(
                    conn, match_date, home_team, away_team,
                )
                if not found_ids:
                    unmatched_rows += 1
                    logger.warning(
                        "No match found for %s %s vs %s; skipping odds.",
                        match_date, home_team, away_team,
                    )
                    continue
                if len(found_ids) > 1:
                    ambiguous_rows += 1
                    logger.warning(
                        "Multiple matches found for %s %s vs %s "
                        "(ids=%s); skipping.",
                        match_date, home_team, away_team, found_ids,
                    )
                    continue
                match_id = int(found_ids[0])
                matched_rows += 1
                match_odds = data[
                    (data["match_date"] == match_date)
                    & (data["home_team"] == home_team)
                    & (data["away_team"] == away_team)
                ]
                for _, orow in match_odds.iterrows():
                    action = _upsert_odds(
                        conn,
                        match_id=match_id,
                        provider=PROVIDER,
                        bookmaker=orow["bookmaker"],
                        market=orow["market"],
                        selection=orow["selection"],
                        odds_type=orow["odds_type"],
                        decimal_odds=float(orow["decimal_odds"]),
                        observed_at=self._retrieval_ts or now,
                        source_reference=self._source_url,
                        now=now,
                    )
                    if action == "inserted":
                        inserted += 1
                    else:
                        updated += 1

        total_matches = matched_rows + unmatched_rows + ambiguous_rows
        logger.info(
            "Loaded %d odds record(s): %d inserted, %d updated. "
            "Matches: %d matched, %d unmatched, %d ambiguous (of %d unique).",
            len(data), inserted, updated,
            matched_rows, unmatched_rows, ambiguous_rows, total_matches,
        )
        return IngestionResult(
            provider=PROVIDER,
            dataset_name=self.dataset_name,
            records_read=len(data),
            records_inserted=inserted,
            records_updated=updated,
            records_matched=matched_rows,
            records_unmatched=unmatched_rows,
            records_ambiguous=ambiguous_rows,
            records_skipped=unmatched_rows + ambiguous_rows,
            status="success",
        )
