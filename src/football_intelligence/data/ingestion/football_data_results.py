"""Football-Data.co.uk historical match-**results** ingestion adapter.

A dedicated adapter is required because :mod:`.odds` only ingests the *odds*
columns of the same CSVs.  CSV columns ``Date, HomeTeam, AwayTeam, FTHG, TAG,
FTR`` map to ``match_date``, teams, ``home_goals``/``away_goals``; ``FTR`` is
used only for score/outcome *validation* (it is never stored -- ``matches``
has no outcome column).

Attribution: credit Football-Data.co.uk when publishing analysis based on
this data.  Raw CSVs are cached under ``data/raw/football_data`` and never
committed to version control.

A Football-Data CSV row carries **no fixture id**, so a deterministic,
project-derived composite key is synthesized and clearly marked:
``football-data|{season}|{date}|{home_norm}|{away_norm}`` (NOT a source-issued
id).  Re-imports update in place via ``UNIQUE(provider, provider_match_id)``.
Teams have no source ids either; the canonical normalized name is used as
``provider_team_id`` with the source-as-published name kept in ``team_name``.

Cross-source duplicate prevention: before inserting a fixture, :func:`_upsert_match`
queries existing matches by ``(date, canonical home, canonical away)`` across
ALL providers; a match already present under another provider is ``skipped``
(or ``ambiguous`` when >1), so a real fixture is never counted twice.  Because
a team plays at most one fixture per day in these leagues, (date, home, away)
uniquely identifies a real fixture regardless of competition-name wording.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd

from ...config import RAW_DATA_DIR
from ...database import database_connection, resolve_db_path
from ...exceptions import IngestionError
from ...logging_config import get_logger
from ..ingestion.base import BaseIngestor, IngestionResult
from ..ingestion.odds import (
    DEFAULT_TIMEOUT,
    LEAGUE_CODES,
    ODDS_BASE_URL,
    PROVIDER,
    _download_csv,
    _ensure_initialized,
    _find_matches_for_reconciliation,
    _normalize_match_date,
    _normalize_team_name,
    _parse_csv,
)
from ..ingestion.statsbomb import (_or_none, _safe_cache_path, _to_int_or_none, _utcnow)
from ..validation.validators import (
    ValidationResult,
    validate_date_fields,
    validate_home_away_different,
    validate_non_negative_goals,
    validate_no_missing_values,
    validate_required_columns,
)

logger = get_logger("data.ingestion.football_data_results")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Base URL for the Football-Data.co.uk CSV archive (shared with the odds module).
BASE_URL: str = ODDS_BASE_URL

#: Dataset identifier written to ``data_sources`` / ``ingestion_runs``.
DATASET_NAME: str = "results"

#: Default league imported when none is specified on the CLI.
DEFAULT_LEAGUE_CODE: str = "E0"

#: Columns the source CSV must contain for results ingestion.
RESULT_CSV_COLUMNS: List[str] = ["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR"]

#: Outcome codes written by Football-Data in the ``FTR`` column.
VALID_FTR = frozenset({"H", "D", "A"})

#: Normalized columns that must be present in the in-memory DataFrame.
RESULT_REQUIRED_COLUMNS: List[str] = [
    "provider", "provider_match_id", "provider_competition_id",
    "competition_name", "match_date", "home_team_id", "away_team_id",
    "home_score", "away_score", "ftr", "match_status",
]

#: Normalized columns that must carry a non-null value.
RESULT_REQUIRED_NON_NULL: List[str] = [
    "provider_match_id", "match_date", "home_team_id", "away_team_id",
    "home_score", "away_score", "ftr",
]

#: Full, ordered column set of the normalized results DataFrame.
RESULT_NORMALIZED_COLUMNS: List[str] = [
    "provider", "provider_match_id", "provider_competition_id",
    "competition_name", "country_name", "season_name", "match_date",
    "home_team_id", "home_team_name", "home_team_country",
    "away_team_id", "away_team_name", "away_team_country",
    "home_score", "away_score", "match_status", "ftr",
]


# ---------------------------------------------------------------------------
# Small public / reused helpers
# ---------------------------------------------------------------------------


def _default_cache_dir() -> Path:
    """Default cache directory for Football-Data results CSVs."""
    return RAW_DATA_DIR / "football_data"


def _season_code_to_name(season_code: str) -> str:
    """Map a Football-Data 4-digit season code to a ``20YY/20YY+1`` name.

    Football-Data encodes seasons as ``YYXX`` where the first two digits are the
    August year (e.g. ``1617`` == 2016/17).
    """
    if not season_code or not (season_code.isdigit() and len(season_code) == 4):
        raise IngestionError(
            f"Invalid season code '{season_code}': expected 4 digits, "
            "e.g. '1617' for the 2016/17 season."
        )
    yy_start, yy_end = season_code[:2], season_code[2:]
    return f"20{yy_start}/20{yy_end}"


def _league_display(league_code: str) -> Tuple[str, str]:
    """Return ``(competition_name, country)`` for a league code.

    ``LEAGUE_CODES`` values have the shape ``"Country - Competition"`` (e.g.
    ``"England - Premier League"``); this splits them so country is stored
    separately, matching the StatsBomb convention.
    """
    if league_code not in LEAGUE_CODES:
        raise IngestionError(
            f"Unknown league code '{league_code}'. "
            f"Valid codes: {sorted(LEAGUE_CODES.keys())}"
        )
    description = LEAGUE_CODES[league_code]
    if " - " in description:
        country, _, competition = description.partition(" - ")
    else:
        country, competition = "Unknown", description
    return competition, country


def _compose_provider_match_id(
    season_code: str, match_date: Optional[str], home_norm: str, away_norm: str
) -> str:
    """Synthesize a deterministic, clearly-marked internal fixture key.

    Football-Data CSVs carry no fixture id, so this MUST NOT be presented as a
    source-issued identifier.  It is derived from stable, observable source
    fields (season + date + the two canonical team names) so it is idempotent.
    """
    return f"{PROVIDER}|{season_code}|{match_date}|{home_norm}|{away_norm}"


def _normalize_results_dataframe(
    df: pd.DataFrame, league_code: str, season_code: str
) -> pd.DataFrame:
    """Convert raw Football-Data CSV rows into normalized match records.

    Each output row carries the fields consumed by the upsert helpers plus the
    source ``FTR`` (retained only so :meth:`validate` can cross-check it
    against the scores).
    """
    competition_name, country = _league_display(league_code)
    season_name = _season_code_to_name(season_code)
    records: List[dict] = []
    for _, row in df.iterrows():
        date_iso = _normalize_match_date(str(row.get("Date") or ""))
        home_raw = str(row.get("HomeTeam") or "").strip()
        away_raw = str(row.get("AwayTeam") or "").strip()
        home_norm = _normalize_team_name(home_raw)
        away_norm = _normalize_team_name(away_raw)
        records.append({
            "provider": PROVIDER,
            "provider_match_id": _compose_provider_match_id(
                season_code, date_iso, home_norm, away_norm
            ),
            "provider_competition_id": league_code,
            "competition_name": competition_name,
            "country_name": country,
            "season_name": season_name,
            "match_date": date_iso,
            "home_team_id": home_norm or None,
            "home_team_name": home_raw or None,
            "home_team_country": country,
            "away_team_id": away_norm or None,
            "away_team_name": away_raw or None,
            "away_team_country": country,
            "home_score": _to_int_or_none(row.get("FTHG")),
            "away_score": _to_int_or_none(row.get("FTAG")),
            "match_status": "available",
            "ftr": (str(row.get("FTR") or "").strip() or None),
        })
    return pd.DataFrame(records, columns=RESULT_NORMALIZED_COLUMNS)


def validate_outcome_score_consistency(df: pd.DataFrame) -> ValidationResult:
    """Cross-check that ``FTR`` agrees with ``home_score``/``away_score``.

    Football-Data's ``FTR`` (H/D/A) must be consistent with the recorded
    scores.  Missing/null FTR is not reported here (the generic missing-value
    validator handles it); only *present-but-wrong* or *illegal* values do.
    """
    result = ValidationResult(row_count=len(df))
    if "ftr" not in df.columns:
        return result
    positions = list(range(len(df)))
    for i in positions:
        ftr = _or_none(df.iloc[i].get("ftr"))
        if ftr is None:
            continue
        if ftr not in VALID_FTR:
            result.add_error(
                "ftr", f"Invalid FTR value {ftr!r}.", record_index=i
            )
            continue
        home = _to_int_or_none(df.iloc[i].get("home_score"))
        away = _to_int_or_none(df.iloc[i].get("away_score"))
        if home is None or away is None:
            continue
        expected = "H" if home > away else ("D" if home == away else "A")
        if ftr != expected:
            result.add_error(
                "ftr",
                f"FTR {ftr} inconsistent with score {home}-{away}.",
                record_index=i,
            )
    return result


# ---------------------------------------------------------------------------
# Database helpers
# ---------------------------------------------------------------------------


def _upsert_competition(
    conn, provider_comp_id: str, name: Optional[str],
    country: Optional[str], now: str, provider: str = PROVIDER,
) -> int:
    """Insert or update a competition, returning the internal row id."""
    row = conn.execute(
        "SELECT id FROM competitions WHERE provider=? AND provider_competition_id=?",
        (provider, provider_comp_id),
    ).fetchone()
    if row:
        conn.execute(
            "UPDATE competitions SET competition_name=?, country=?, updated_at=? "
            "WHERE id=?",
            (name, country, now, int(row["id"])),
        )
        return int(row["id"])
    cur = conn.execute(
        "INSERT INTO competitions (provider, provider_competition_id, "
        "competition_name, country, created_at, updated_at) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (provider, provider_comp_id, name, country, now, now),
    )
    return int(cur.lastrowid)


def _upsert_team(
    conn, provider_team_id: str, name: Optional[str],
    country: Optional[str], now: str, provider: str = PROVIDER,
) -> int:
    """Insert or update a team, returning the internal row id.

    Football-Data identifies teams only by name, so the canonical normalized
    name is used as ``provider_team_id`` and the source-as-published name is
    retained in ``team_name``.
    """
    if not provider_team_id:
        raise IngestionError("Cannot upsert a team with no provider_team_id.")
    name = name if (name and str(name).strip()) else provider_team_id
    row = conn.execute(
        "SELECT id FROM teams WHERE provider=? AND provider_team_id=?",
        (provider, provider_team_id),
    ).fetchone()
    if row:
        conn.execute(
            "UPDATE teams SET team_name=?, country=?, updated_at=? WHERE id=?",
            (name, country, now, int(row["id"])),
        )
        return int(row["id"])
    cur = conn.execute(
        "INSERT INTO teams (provider, provider_team_id, team_name, country, "
        "created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?)",
        (provider, provider_team_id, name, country, now, now),
    )
    return int(cur.lastrowid)


def _upsert_match(
    conn, row: pd.Series, competition_id: int, home_team_id: int,
    away_team_id: int, now: str, provider: str = PROVIDER,
) -> str:
    """Insert or update a match.

    Returns one of ``"inserted"``, ``"updated"``, ``"skipped"`` or
    ``"ambiguous"``.  The cross-source duplicate guard consults
    :func:`_find_matches_for_reconciliation` so a fixture already represented
    under a *different* provider is not double-counted.
    """
    pid = _or_none(row.get("provider_match_id"))
    assert pid is not None  # validate() guarantees this
    match_date = _or_none(row.get("match_date"))
    home_name = _or_none(row.get("home_team_name"))
    away_name = _or_none(row.get("away_team_name"))

    # (1) Idempotent update of an existing same-provider fixture.
    exists = conn.execute(
        "SELECT id FROM matches WHERE provider=? AND provider_match_id=?",
        (provider, pid),
    ).fetchone()
    if exists:
        conn.execute(
            "UPDATE matches SET competition_id=?, season_name=?, match_date=?, "
            "home_team_id=?, away_team_id=?, home_goals=?, away_goals=?, "
            "match_status=?, updated_at=? WHERE id=?",
            (
                competition_id, _or_none(row.get("season_name")), match_date,
                home_team_id, away_team_id,
                _to_int_or_none(row.get("home_score")),
                _to_int_or_none(row.get("away_score")),
                _or_none(row.get("match_status")), now, int(exists["id"]),
            ),
        )
        return "updated"

    # (2) Cross-source duplicate guard: refuse to double-count a real fixture
    #     already supplied by another provider (e.g. StatsBomb).
    found = _find_matches_for_reconciliation(conn, match_date, home_name, away_name)
    if found:
        if len(found) == 1:
            logger.info(
                "Skipping cross-source fixture %s (already present, "
                "match_id=%s)", pid, found[0],
            )
            return "skipped"
        logger.warning(
            "Ambiguous fixture %s (multiple existing matches): %s", pid, found
        )
        return "ambiguous"

    # (3) Genuinely new fixture -> insert.
    conn.execute(
        "INSERT INTO matches (provider, provider_match_id, competition_id, "
        "season_name, match_date, home_team_id, away_team_id, home_goals, "
        "away_goals, match_status, created_at, updated_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            provider, pid, competition_id,
            _or_none(row.get("season_name")), match_date,
            home_team_id, away_team_id,
            _to_int_or_none(row.get("home_score")),
            _to_int_or_none(row.get("away_score")),
            _or_none(row.get("match_status")), now, now,
        ),
    )
    return "inserted"


def _record_data_source(
    conn, dataset_name: str, source_ref: str,
    retrieval_ts: str, data_version: Optional[str],
    metadata_json: str, now: str,
) -> None:
    """Record provenance metadata for a downloaded source file.

    Unlike the odds/statsbomb helpers this takes ``dataset_name`` explicitly so
    results provenance is recorded under ``'results'`` (never ``'odds'``).
    """
    conn.execute(
        "INSERT INTO data_sources (provider, dataset_name, source_reference, "
        "retrieval_timestamp, data_version, metadata, created_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        (
            PROVIDER, dataset_name, source_ref,
            retrieval_ts, data_version, metadata_json, now,
        ),
    )


# ---------------------------------------------------------------------------
# Ingestor
# ---------------------------------------------------------------------------


class FootballDataResultsIngestor(BaseIngestor):
    """Ingestor for Football-Data.co.uk historical **match results**.

    Downloads (or reads from a deterministic cache) a results CSV for a given
    ``league_code`` and ``season_code``, normalizes each row to the project
    schema, validates scores/outcome consistency, synthesizes a deterministic
    internal fixture key (the source CSV carries none), and upserts
    competition -> team -> match rows inside a single transaction with a
    cross-source duplicate-prevention guard.  Provenance is recorded under
    ``dataset_name='results'``; the ``odds`` table is never touched.

    Args:
        league_code: Football-Data league code (e.g. ``E0``).
        season_code: 4-digit season code (e.g. ``1617`` for 2016/17).
        db_path: Target SQLite database path (defaults to configured path).
        cache_dir: Override for the raw-file cache directory.
        refresh: When ``True``, force re-download even if cached.
        source_url: Override the download URL (testing).
        timeout: HTTP timeout in seconds.
    """

    def __init__(
        self,
        league_code: str = DEFAULT_LEAGUE_CODE,
        season_code: str = "",
        db_path: Optional[Path] = None,
        cache_dir: Optional[Path] = None,
        refresh: bool = False,
        source_url: Optional[str] = None,
        timeout: int = DEFAULT_TIMEOUT,
    ) -> None:
        league_code = str(league_code).strip().upper()
        season_code = str(season_code).strip()
        if league_code not in LEAGUE_CODES:
            raise IngestionError(
                f"Unknown league code '{league_code}'. "
                f"Valid codes: {sorted(LEAGUE_CODES.keys())}"
            )
        if not (season_code.isdigit() and len(season_code) == 4):
            raise IngestionError(
                f"Invalid season code '{season_code}': expected 4 digits, "
                "e.g. '1617' for the 2016/17 season."
            )
        self._league_code = league_code
        self._season_code = season_code
        self._db_path = db_path
        self._cache_dir = Path(cache_dir) if cache_dir else _default_cache_dir()
        self._refresh = refresh
        self._timeout = timeout
        self._source_url = source_url or (
            f"{BASE_URL}/{self._season_code}/{self._league_code}.csv"
        )
        self._was_cached: bool = False
        self._retrieval_ts: Optional[str] = None

    # -- properties -------------------------------------------------------

    @property
    def provider(self) -> str:
        return PROVIDER

    @property
    def dataset_name(self) -> str:
        return DATASET_NAME

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
    def season_name(self) -> str:
        return _season_code_to_name(self._season_code)

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
        """Download (or read cached) the results CSV; return normalized rows."""
        url = self._source_url
        cache_path = self._cache_path()
        self._retrieval_ts = _utcnow()
        csv_text, self._was_cached = _download_csv(
            url, cache_path, self._timeout, self._refresh,
        )
        df = _parse_csv(csv_text)
        normalized = _normalize_results_dataframe(
            df, self._league_code, self._season_code
        )
        logger.info(
            "Retrieved %d results row(s) for %s season %s (source: %s)",
            len(normalized), self._league_code, self._season_code,
            "cache" if self._was_cached else "network",
        )
        return normalized

    def validate(self, data: pd.DataFrame) -> ValidationResult:
        """Validate the normalized results DataFrame before loading."""
        result = ValidationResult(row_count=len(data))
        result.merge(validate_required_columns(data, RESULT_REQUIRED_COLUMNS))
        result.merge(validate_no_missing_values(data, RESULT_REQUIRED_NON_NULL))
        result.merge(validate_date_fields(data, ["match_date"]))
        result.merge(
            validate_non_negative_goals(data, "home_score", "away_score")
        )
        result.merge(
            validate_home_away_different(data, "home_team_id", "away_team_id")
        )
        result.merge(validate_outcome_score_consistency(data))
        return result

    def load(self, data: pd.DataFrame) -> IngestionResult:
        """Persist validated results, enforcing the cross-source guard.

        Competition -> team -> match rows and provenance are written inside a
        single transaction so a failure rolls everything back.  Re-running with
        the same data is idempotent (existing rows update in place) and never
        double-counts a fixture already supplied by another provider.  The
        ``odds`` table is never touched.
        """
        db_path = resolve_db_path(self._db_path)
        now = _utcnow()
        inserted = 0
        updated = 0
        skipped = 0
        ambiguous = 0

        metadata_json = json.dumps(
            {
                "league_code": self._league_code,
                "season_code": self._season_code,
                "season_name": self.season_name,
                "record_count": len(data),
                "retrieval_source": "cache" if self._was_cached else "network",
                "provider_match_id_strategy": (
                    "project-derived composite (no fixture id in source CSV)"
                ),
                "ftr_usage": "validation only; not stored in matches",
            }
        )

        with database_connection(db_path) as conn:
            _ensure_initialized(db_path, conn)
            for _, row in data.iterrows():
                country = _or_none(row.get("country_name"))
                comp_id = _upsert_competition(
                    conn,
                    str(_or_none(row.get("provider_competition_id")) or ""),
                    _or_none(row.get("competition_name")), country, now,
                )
                home_id = _upsert_team(
                    conn,
                    str(_or_none(row.get("home_team_id")) or ""),
                    _or_none(row.get("home_team_name")), country, now,
                )
                away_id = _upsert_team(
                    conn,
                    str(_or_none(row.get("away_team_id")) or ""),
                    _or_none(row.get("away_team_name")), country, now,
                )
                action = _upsert_match(conn, row, comp_id, home_id, away_id, now)
                if action == "inserted":
                    inserted += 1
                elif action == "updated":
                    updated += 1
                elif action == "skipped":
                    skipped += 1
                elif action == "ambiguous":
                    ambiguous += 1
            _record_data_source(
                conn, DATASET_NAME, self._source_url,
                self._retrieval_ts or now, None, metadata_json, now,
            )

        logger.info(
            "Loaded %d results row(s): %d inserted, %d updated, %d skipped, "
            "%d ambiguous",
            len(data), inserted, updated, skipped, ambiguous,
        )
        return IngestionResult(
            provider=PROVIDER,
            dataset_name=DATASET_NAME,
            records_read=len(data),
            records_inserted=inserted,
            records_updated=updated,
                        records_skipped=skipped,
            records_ambiguous=ambiguous,
            status="success",
        )


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------


def list_result_leagues() -> List[Tuple[str, str]]:
    """Return a sorted list of ``(league_code, description)`` tuples."""
    return sorted(LEAGUE_CODES.items())


__all__ = [
    "FootballDataResultsIngestor",
    "DATASET_NAME",
    "list_result_leagues",
]





