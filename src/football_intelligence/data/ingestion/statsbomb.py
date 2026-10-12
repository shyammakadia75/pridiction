"""StatsBomb Open Data ingestion adapter.

Implements a :class:`BaseIngestor` sub-class that downloads (or reads from a
deterministic cache) match metadata from the official StatsBomb Open Data
repository (``https://github.com/statsbomb/open-data``) and normalizes it into
the project's existing SQLite schema.

The adapter intentionally ingests **match metadata only** -- competition, team,
and match records.  Event-level and lineup-level files are never downloaded in
this phase (see project roadmap).

Licensing / attribution
-----------------------
StatsBomb Open Data is made freely available for research and genuine interest
in football analytics.  Per the repository's README, anyone who publishes,
shares, or distributes analysis based on this data must state the source as
StatsBomb and use their logo from the `StatsBomb Media Pack
<https://statsbomb.com/media-pack/>`_.  No API key is required.

Files are cached under ``data/raw/statsbomb`` and are never committed to
version control.  The cache is deterministic: re-running the same import
without ``--refresh`` reuses cached files instead of re-downloading.
"""
from __future__ import annotations

import json
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
from ..validation.validators import (
    ValidationResult,
    validate_date_fields,
    validate_home_away_different,
    validate_no_duplicate_provider_ids,
    validate_no_missing_values,
    validate_non_negative_goals,
    validate_required_columns,
)

logger = get_logger("data.ingestion.statsbomb")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Provider identifier stored alongside every record.
PROVIDER = "statsbomb"

#: Base URL for the raw StatsBomb Open Data repository on GitHub.
STATSBOMB_BASE_URL = (
    "https://raw.githubusercontent.com/statsbomb/open-data/master/data"
)

#: URL to the canonical ``competitions.json`` index.
COMPETITIONS_URL = f"{STATSBOMB_BASE_URL}/competitions.json"

#: Default HTTP timeout in seconds.
DEFAULT_TIMEOUT = 30

#: HTTP User-Agent identifying this project.
HTTP_USER_AGENT = (
    "football-intelligence/0.1.0 "
    "(+https://github.com/shyammakadia75/pridiction)"
)

#: Columns the normalized match DataFrame must contain.
REQUIRED_COLUMNS = [
    "provider_match_id",
    "provider_competition_id",
    "home_team_id",
    "away_team_id",
    "match_date",
    "home_score",
    "away_score",
    "match_status",
    "home_team_name",
    "away_team_name",
]

#: Fields that must not be null in a valid match record.
REQUIRED_NON_NULL = [
    "provider_match_id",
    "provider_competition_id",
    "home_team_id",
    "away_team_id",
    "match_date",
]

#: Columns in the normalized match DataFrame used for DB persistence.
MATCH_COLUMNS = [
    "provider",
    "provider_match_id",
    "competition_id",
    "season_name",
    "match_date",
    "home_team_id",
    "away_team_id",
    "home_goals",
    "away_goals",
    "match_status",
    "created_at",
    "updated_at",
]


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def _utcnow() -> str:
    """Return the current UTC time as an ISO-8601 string."""
    return datetime.now(timezone.utc).isoformat()


def _default_cache_dir() -> Path:
    """Default cache directory for StatsBomb raw files."""
    return RAW_DATA_DIR / "statsbomb"


def _or_none(value: Any) -> Optional[Any]:
    """Return *value* or ``None`` if it is pandas-NA / NaN."""
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return value


def _to_int_or_none(value: Any) -> Optional[int]:
    """Coerce a score to ``int`` or ``None``."""
    val = _or_none(value)
    if val is None:
        return None
    return int(val)

# ---------------------------------------------------------------------------
# HTTP / caching
# ---------------------------------------------------------------------------

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


def _safe_cache_path(cache_dir: Path, *parts: str) -> Path:
    """Build a cache path under *cache_dir*, preventing path traversal.

    All *parts* must be relative; the resolved target must stay inside
    *cache_dir*.
    """
    base = Path(cache_dir).resolve()
    target = base.joinpath(*parts).resolve()
    if not target.is_relative_to(base):
        raise IngestionError(
            f"Unsafe cache path rejected: {target} escapes {base}"
        )
    return target


def _download_json(
    url: str,
    cache_path: Path,
    timeout: int,
    refresh: bool,
) -> Tuple[Any, bool]:
    """Fetch JSON from *url*, caching the raw bytes to *cache_path*.

    Returns ``(parsed_data, was_from_cache)``.  When the cache file exists
    and *refresh* is ``False``, the cached copy is read instead of hitting
    the network.
    """
    cache_path = Path(cache_path)
    if cache_path.exists() and not refresh:
        logger.info("Using cached file: %s", cache_path)
        try:
            text = cache_path.read_text(encoding="utf-8")
            return json.loads(text), True
        except (json.JSONDecodeError, OSError) as exc:
            raise IngestionError(
                f"Cache read failed for {cache_path}: {exc}"
            ) from exc
    logger.info("Downloading: %s", url)
    raw = _http_get(url, timeout)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(raw.decode("utf-8"), encoding="utf-8")
    return json.loads(raw.decode("utf-8")), False

# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------

def _normalize_match(match: dict[str, Any]) -> dict[str, Any]:
    """Flatten one StatsBomb match record into the project's normalized schema.

    All identifier fields are preserved as strings for stable round-tripping;
    scores are left as-is (they may be ``None`` for scheduled, not-yet-played
    matches -- the database schema allows ``NULL`` for goals).
    """
    comp = match.get("competition") or {}
    season = match.get("season") or {}
    home = match.get("home_team") or {}
    away = match.get("away_team") or {}
    home_country = home.get("country") or {}
    away_country = away.get("country") or {}
    meta = match.get("metadata") or {}
    stage = match.get("competition_stage") or {}

    def _s(val: Any) -> Optional[str]:
        val = _or_none(val)
        return str(val) if val is not None else None

    return {
        "provider": PROVIDER,
        "provider_match_id": _s(match.get("match_id")),
        "provider_competition_id": _s(comp.get("competition_id")),
        "competition_name": comp.get("competition_name"),
        "country_name": comp.get("country_name"),
        "season_id": _s(season.get("season_id")),
        "season_name": season.get("season_name"),
        "match_date": match.get("match_date"),
        "kick_off": match.get("kick_off"),
        "home_team_id": _s(home.get("home_team_id")),
        "home_team_name": home.get("home_team_name"),
        "home_team_country": home_country.get("name"),
        "away_team_id": _s(away.get("away_team_id")),
        "away_team_name": away.get("away_team_name"),
        "away_team_country": away_country.get("name"),
        "home_score": match.get("home_score"),
        "away_score": match.get("away_score"),
        "match_status": match.get("match_status"),
        "competition_stage": stage.get("name"),
        "match_week": match.get("match_week"),
        "data_version": meta.get("data_version"),
        "last_updated": match.get("last_updated"),
    }


def _matches_to_dataframe(matches: list[dict[str, Any]]) -> pd.DataFrame:
    """Convert a list of StatsBomb match dicts into a normalized DataFrame."""
    if not matches:
        return pd.DataFrame(columns=list(_normalize_match({}).keys()))
    return pd.DataFrame([_normalize_match(m) for m in matches])


# ---------------------------------------------------------------------------
# Competition listing (CLI helper)
# ---------------------------------------------------------------------------

def list_competitions(
    cache_dir: Optional[Path] = None,
    refresh: bool = False,
    source_url: Optional[str] = None,
    timeout: int = DEFAULT_TIMEOUT,
) -> pd.DataFrame:
    """Fetch StatsBomb competition metadata and return it as a DataFrame.

    Retrieves ``competitions.json`` from the official registry (or an
    override URL), caching the raw file under *cache_dir*.
    """
    url = source_url or COMPETITIONS_URL
    cache = Path(cache_dir) if cache_dir else _default_cache_dir()
    cache_path = _safe_cache_path(cache, "competitions.json")
    data, _was_cached = _download_json(url, cache_path, timeout, refresh)
    if not isinstance(data, list):
        raise IngestionError(
            f"Unexpected competitions.json structure "
            f"(expected a list, got {type(data).__name__})"
        )
    return pd.DataFrame(data)

# ---------------------------------------------------------------------------
# Database helpers
# ---------------------------------------------------------------------------

def _ensure_initialized(db_path: Optional[Path], conn) -> None:
    """Verify the schema_version table has been created.

    Raises :class:`DatabaseError` if the database has not been initialized
    with ``init-db``.
    """
    row = conn.execute("SELECT MAX(version) FROM schema_version").fetchone()
    if row is None or row[0] is None:
        raise DatabaseError(
            f"Database '{db_path}' is not initialized. "
            "Run 'football-intelligence init-db' first."
        )


def _upsert_competition(
    conn,
    provider: str,
    provider_comp_id: str,
    name: Optional[str],
    country: Optional[str],
    now: str,
) -> int:
    """Insert or update a competition, returning the internal row id."""
    row = conn.execute(
        "SELECT id FROM competitions "
        "WHERE provider=? AND provider_competition_id=?",
        (provider, provider_comp_id),
    ).fetchone()
    if row:
        conn.execute(
            "UPDATE competitions "
            "SET competition_name=?, country=?, updated_at=? WHERE id=?",
            (name, country, now, int(row["id"])),
        )
        return int(row["id"])
    cur = conn.execute(
        "INSERT INTO competitions "
        "(provider, provider_competition_id, competition_name, country, "
        "created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?)",
        (provider, provider_comp_id, name, country, now, now),
    )
    return int(cur.lastrowid)


def _upsert_team(
    conn,
    provider: str,
    provider_team_id: str,
    name: Optional[str],
    country: Optional[str],
    now: str,
) -> int:
    """Insert or update a team, returning the internal row id."""
    row = conn.execute(
        "SELECT id FROM teams "
        "WHERE provider=? AND provider_team_id=?",
        (provider, provider_team_id),
    ).fetchone()
    if row:
        conn.execute(
            "UPDATE teams "
            "SET team_name=?, country=?, updated_at=? WHERE id=?",
            (name, country, now, int(row["id"])),
        )
        return int(row["id"])
    cur = conn.execute(
        "INSERT INTO teams "
        "(provider, provider_team_id, team_name, country, "
        "created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?)",
        (provider, provider_team_id, name, country, now, now),
    )
    return int(cur.lastrowid)

# CHUNK_END



