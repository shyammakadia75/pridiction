"""Phase 22 (Step 2.6) - team-identity mapping rules.

These tests codify the *verified* Football-Data E0 inventory discovered during
the Phase-22 multi-season expansion (E0_1516/E0_1617/E0_1920). The name sets
below are the audit-recorded spellings, embedded as constants so the suite stays
self-contained (no dependency on the gitignored raw cache or the production DB).

Covers: alias coverage, case/punctuation tolerance, unknown-name retention,
the FD-internal 1:1 injectivity gate, cross-season spelling stability, and the
documented cross-source (FD<->StatsBomb) alias gap.
"""
from __future__ import annotations

import pytest

from football_intelligence.data.ingestion.odds import (
    PROVIDER,
    TEAM_ALIASES,
    _normalize_team_name,
)

# --- Verified inventory of FD E0 short names observed across 1516/1617/1920 ---
SEASON_NAMES = {
    "1516": {"Arsenal", "Aston Villa", "Bournemouth", "Chelsea", "Crystal Palace",
             "Everton", "Leicester", "Liverpool", "Man City", "Man United", "Newcastle",
             "Norwich", "Southampton", "Stoke", "Sunderland", "Swansea", "Tottenham",
             "Watford", "West Brom", "West Ham"},
    "1617": {"Arsenal", "Bournemouth", "Burnley", "Chelsea", "Crystal Palace",
             "Everton", "Hull", "Leicester", "Liverpool", "Man City", "Man United",
             "Middlesbrough", "Southampton", "Stoke", "Sunderland", "Swansea",
             "Tottenham", "Watford", "West Brom", "West Ham"},
    "1920": {"Arsenal", "Aston Villa", "Bournemouth", "Brighton", "Burnley",
             "Chelsea", "Crystal Palace", "Everton", "Leicester", "Liverpool",
             "Man City", "Man United", "Newcastle", "Norwich", "Sheffield United",
             "Southampton", "Tottenham", "Watford", "West Ham", "Wolves"},
}
ALL_FD_NAMES = set()
for _names in SEASON_NAMES.values():
    ALL_FD_NAMES |= _names

# Clubs appearing in >=2 of the verified seasons (must keep identical spelling).
MULTI_SEASON_CLUBS = {
    "Arsenal", "Bournemouth", "Chelsea", "Crystal Palace", "Everton", "Leicester",
    "Liverpool", "Man City", "Man United", "Southampton", "Stoke", "Sunderland",
    "Swansea", "Tottenham", "Watford", "West Brom", "West Ham",
}

# (FD short name, expected canonical) for every alias the dict is expected to cover.
ALIAS_EXPECTED = [
    ("Man United", "manchester united"), ("Man City", "manchester city"),
    ("Tottenham", "tottenham hotspur"), ("Spurs", "tottenham hotspur"),
    ("Newcastle", "newcastle united"), ("West Ham", "west ham united"),
    ("Bournemouth", "afc bournemouth"), ("Brighton", "brighton and hove albion"),
    ("Wolves", "wolverhampton wanderers"), ("Leeds", "leeds united"),
    ("Sheffield utd", "sheffield united"), ("Sheffield wed", "sheffield wednesday"),
    ("Wigan", "wigan athletic"),
]

# The 5 clubs whose FD bare short name does NOT (yet) alias to the StatsBomb
# canonical form. Documented gap; resolves to a bare canonical today.
CROSS_SOURCE_GAPS = {
    "Leicester": "leicester",      # SB: "Leicester City"
    "Norwich": "norwich",          # SB: "Norwich City"
    "Stoke": "stoke",              # SB: "Stoke City"
    "Swansea": "swansea",          # SB: "Swansea City"
        "West Brom": "west brom",      # SB: "West Bromwich Albion"
}


# ---------------------------------------------------------------------------
# 1. Alias coverage
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("raw,expected", ALIAS_EXPECTED)
def test_known_fd_aliases_resolve(raw, expected):
    assert _normalize_team_name(raw) == expected


# ---------------------------------------------------------------------------
# 2. Case / punctuation tolerance
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "raw,expected",
    [
        ("man united", "manchester united"),
        ("MAN UNITED", "manchester united"),
        ("  Man United  ", "manchester united"),
        ("Man.Utd", "manutd"),            # '.' removed (no space inserted)
        ("Arsenal FC", "arsenal"),
        ("Arsenal F.C.", "arsenal"),
        ("Arsenal f.c.", "arsenal"),
        ("Arsenal  F.C.", "arsenal"),       # double space + suffix
        ("Nott'm Forest", "nottm forest"),  # apostrophe stripped
    ],
)
def test_case_and_punctuation_tolerance(raw, expected):
    assert _normalize_team_name(raw) == expected


# ---------------------------------------------------------------------------
# 3. Unknown names are retained (lowercased) - never silently fail
# ---------------------------------------------------------------------------

def test_unknown_name_retained_normalized():
    assert _normalize_team_name("Some New Club") == "some new club"
    assert _normalize_team_name("Wrexham") == "wrexham"


def test_empty_name_returns_empty():
    assert _normalize_team_name("") == ""
    assert _normalize_team_name("   ") == ""


# ---------------------------------------------------------------------------
# 4. Injectivity gate: no two distinct FD names collapse to one canonical
#    (the Step-2 "each club maps 1:1 to one canonical identity" rule).
# ---------------------------------------------------------------------------

def test_fd_teamname_set_is_injective():
    canonicals = {_normalize_team_name(n) for n in ALL_FD_NAMES}
    assert len(canonicals) == len(ALL_FD_NAMES), (
        f"not injective: {len(ALL_FD_NAMES)} names -> {len(canonicals)} canonicals"
    )


def test_each_fd_name_has_single_canonical():
    """Determinism: normalize(n) is stable regardless of input case."""
    for n in ALL_FD_NAMES:
        assert _normalize_team_name(n) == _normalize_team_name(n.lower())


# ---------------------------------------------------------------------------
# 5. Consistency across seasons: spelling (and thus canonical) is stable
# ---------------------------------------------------------------------------

def test_multi_season_club_spelling_is_stable():
    """Every club in >=2 seasons must appear with an identical spelling in each."""
    for club in MULTI_SEASON_CLUBS:
        present = {s for s, names in SEASON_NAMES.items() if club in names}
        assert len(present) >= 2, f"{club} expected in >=2 seasons"
        assert _normalize_team_name(club) == _normalize_team_name(club.lower())


# ---------------------------------------------------------------------------
# 6. Cross-source alias gap (documented, not silently merged)
# ---------------------------------------------------------------------------

def test_cross_source_gap_documented():
    """5 clubs below normalise to a bare canonical that does NOT match the
    StatsBomb canonical (e.g. 'leicester' vs 'leicester city'). Verified gap
    (170 duplicate fixtures on ingest over an SB-seeded DB). Fail-closed:
    remains as-is rather than auto-aliased from a fuzzy guess; a documented
    precondition for any FD<->SB merged dataset.
    """
    for raw, expected in CROSS_SOURCE_GAPS.items():
        assert _normalize_team_name(raw) == expected, raw
    for raw in CROSS_SOURCE_GAPS:
        assert raw.lower() not in TEAM_ALIASES


def test_provider_is_football_data():
    assert PROVIDER == "football-data"
