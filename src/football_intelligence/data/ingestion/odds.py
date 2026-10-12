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


