"""Phase 26: reusable match-prediction service.

This service wraps the Project's verified Poisson goal model (the Phase-25
benchmark) into a clean, leakage-safe interface for requesting predictions for
a specific fixture.

The service is intentionally narrow:

* It supports **one model** -- Poisson (the Phase-25 benchmark).  Other models
  (Dixon-Coles, LR, XGBoost, Elo-mapped) were evaluated in Phases 24-25 and
  found **not to significantly outperform** Poisson.  Poisson is therefore the
  default and only model shipped here, keeping the prediction surface simple and
  auditable.

* It accepts a *fixture specification* (competition, season or target date,
  home team, away team) and resolves every identifier against the database.
  Ambiguous or unknown identifiers raise a clear error -- the service never
  silently substitutes a different team or competition.

* Predictions use **only** information available before kickoff.  The model
  history is built from matches whose ``match_date`` is *strictly earlier*
  than the target fixture's date (matches on the same date are excluded to
  prevent same-day leakage, matching the guarantee used in
  :func:`walkforward.compute_walkforward_predictions`).

* The production database is opened **read-only** and SHA-256 gated
  (fail-closed): if the hash changes between the pre- and post-checks the
  service raises immediately.

Public API
----------
:class:`PredictionRequest`
    Validated input: competition, season/date, teams, optional model.

:class:`MatchPrediction`
    Full prediction output with probabilities, scoreline, and derived markets.

:class:`PredictionExplanation`
    Honest explanation of model, data, features, and limitations.

:func:`predict_match`
    Top-level entry point returning a :class:`MatchPrediction`.
"""
from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional, Sequence, Union

import numpy as np
import pandas as pd

from ..database import resolve_db_path
from ..features.engineering import FeatureEngineer, load_matches_for_features
from ..logging_config import get_logger
from ..config import DATA_DIR
from .baseline import OUTCOME_LABELS, OUTCOME_INDEX
from .poisson import (
    DEFAULT_MAX_GOALS,
    DEFAULT_MIN_HISTORY,
    DEFAULT_PRIOR_STRENGTH,
    DEFAULT_RATE_FLOOR,
    PoissonModel,
    poisson_pmf,
    score_probability_grid,
    home_draw_away_probabilities,
    over_under_probability,
    both_teams_to_score_probability,
)

logger = get_logger("prediction.service")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Models supported by the prediction service (Phase-25 benchmark only).
SUPPORTED_MODELS: tuple[str, ...] = ("poisson",)

#: Default model (the Phase-25 benchmark).
DEFAULT_MODEL: str = "poisson"

#: Production DB path (read-only, SHA-256 gated).
PROD_DB_PATH: Path = DATA_DIR / "football_intelligence.db"

#: Known production DB SHA-256 (fail-closed invariant).
KNOWN_PROD_HASH: str = (
    "725af574b88dc033bd3b00928de503f9bd9a836fbe4252832b7812dce0ab3122"
)

#: Phase-25 pooled walk-forward metrics for the Poisson model
#: (documented for transparent explanation -- NOT a guarantee).
PHASE25_POOLED_METRICS: dict[str, dict[str, float]] = {
    "poisson": {
        "accuracy": 0.5487,
        "macro_f1": 0.4107,
        "log_loss": 0.9761,
        "brier": 0.5814,
    },
}


def _sha256(path: Path) -> str:
    """SHA-256 hex digest of *path* (fail-closed tamper guard)."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Probability validation
# ---------------------------------------------------------------------------


def validate_probability_vector(
    probs: Sequence[float], tol: float = 1e-6,
) -> None:
    """Raise ``ValueError`` if *probs* is not a valid probability vector.

    Checks:
    - Length >= 1
    - All values finite and non-negative
    - Sum is within *tol* of 1.0
    """
    arr = np.asarray(list(probs), dtype=float)
    if arr.ndim != 1:
        raise ValueError(f"Expected 1-D probability vector, got {arr.ndim}D")
    if arr.size == 0:
        raise ValueError("Probability vector is empty")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"Non-finite values in probability vector: {arr}")
    if np.any(arr < -tol):
        raise ValueError(f"Negative probability values: {arr}")
    total = float(arr.sum())
    if not (1.0 - tol <= total <= 1.0 + tol):
        raise ValueError(
            f"Probabilities sum to {total:.6f}, expected 1.0 (+/- {tol})"
        )


# ---------------------------------------------------------------------------
# Derived market calculations
# ---------------------------------------------------------------------------


def double_chance_probabilities(
    home_win: float, draw: float, away_win: float,
) -> dict[str, float]:
    """Compute 1X / 12 / X2 double-chance probabilities from 1X2.

    | Market | Includes         | Formula (from 1X2)  |
    |--------|------------------|---------------------|
    | 1X     | Home or Draw     | 1 - P(away_win)     |
    | 12     | Home or Away     | 1 - P(draw)         |
    | X2     | Draw or Away     | 1 - P(home_win)     |

    All values are clamped to [0, 1].  The three double-chance values do NOT
    sum to 1 (they are complementary-event probabilities, not a partition),
    which is the standard betting-market convention.
    """
    return {
        "1X": max(0.0, min(1.0, 1.0 - away_win)),
        "12": max(0.0, min(1.0, 1.0 - draw)),
        "X2": max(0.0, min(1.0, 1.0 - home_win)),
    }


def over_under_probabilities(
    home_rate: float,
    away_rate: float,
    lines: Sequence[float] = (1.5, 2.5, 3.5),
    max_goals: int = DEFAULT_MAX_GOALS,
) -> dict[str, dict[str, float]]:
    """Return ``{line: {"over": p, "under": p}}`` for each total-goals line."""
    result: dict[str, dict[str, float]] = {}
    for line in lines:
        line = float(line)
        over_p, under_p = over_under_probability(
            home_rate, away_rate, line=line, max_goals=max_goals
        )
        result[str(line)] = {"over": float(over_p), "under": float(under_p)}
    return result


# ---------------------------------------------------------------------------
# Database resolution helpers
# ---------------------------------------------------------------------------


def team_lookup(
    db_path: Union[str, Path],
    *,
    provider: str = "statsbomb",
    team_name: Optional[str] = None,
    provider_team_id: Optional[str] = None,
) -> dict[str, Any]:
    """Resolve a team by name or provider-specific ID.

    Returns ``{id, provider, provider_team_id, team_name}``.  Raises
    :class:`ValueError` when no match is found or ambiguous.
    """
    import sqlite3

    db_path = Path(db_path)
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        conn.row_factory = sqlite3.Row
        if provider_team_id is not None:
            rows = conn.execute(
                "SELECT id, provider, provider_team_id, team_name "
                "FROM teams WHERE provider = ? AND provider_team_id = ?",
                (provider, str(provider_team_id)),
            ).fetchall()
        elif team_name is not None:
            rows = conn.execute(
                "SELECT id, provider, provider_team_id, team_name "
                "FROM teams WHERE provider = ? AND team_name = ?",
                (provider, team_name),
            ).fetchall()
            if not rows:
                rows = conn.execute(
                    "SELECT id, provider, provider_team_id, team_name "
                    "FROM teams WHERE provider = ? AND LOWER(team_name) = LOWER(?)",
                    (provider, team_name.strip()),
                ).fetchall()
        else:
            raise ValueError("Must supply either team_name or provider_team_id")

        if not rows:
            raise ValueError(
                f"No team found for provider={provider!r}, "
                f"team_name={team_name!r}, provider_team_id={provider_team_id!r}"
            )
        if len(rows) > 1:
            names = sorted({r["team_name"] for r in rows})
            raise ValueError(
                f"Ambiguous team name {team_name!r} in provider {provider!r}: "
                f"{len(rows)} teams match ({names}). Specify provider_team_id."
            )
        r = rows[0]
        return {
            "id": int(r["id"]),
            "provider": r["provider"],
            "provider_team_id": r["provider_team_id"],
            "team_name": r["team_name"],
                }
    finally:
        conn.close()


def resolve_team(
    db_path: Union[str, Path],
    team_name: str,
    *,
    provider: str = "statsbomb",
) -> dict[str, Any]:
    """Convenience wrapper around :func:`team_lookup` by name."""
    return team_lookup(db_path, provider=provider, team_name=team_name)


def resolve_competition(
    db_path: Union[str, Path],
    provider_competition_id: str,
    *,
    provider: Optional[str] = None,
) -> dict[str, Any]:
    """Resolve a provider competition ID to the local ``competitions.id``.

    Returns ``{id, provider, provider_competition_id, competition_name}``.
    Raises :class:`ValueError` if not found or ambiguous.
    """
    import sqlite3

    db_path = Path(db_path)
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        conn.row_factory = sqlite3.Row
        if provider is not None:
            rows = conn.execute(
                "SELECT id, provider, provider_competition_id, competition_name "
                "FROM competitions WHERE provider = ? AND provider_competition_id = ?",
                (provider, str(provider_competition_id)),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT id, provider, provider_competition_id, competition_name "
                "FROM competitions WHERE provider_competition_id = ?",
                (str(provider_competition_id),),
            ).fetchall()

        if not rows:
            raise ValueError(
                f"No competition found for provider_competition_id="
                f"{provider_competition_id!r}"
            )
        if len(rows) > 1:
            providers_list = sorted({r["provider"] for r in rows})
            raise ValueError(
                f"provider_competition_id={provider_competition_id!r} is "
                f"ambiguous across providers {providers_list}; specify provider."
            )
        r = rows[0]
        return {
            "id": int(r["id"]),
            "provider": r["provider"],
            "provider_competition_id": r["provider_competition_id"],
            "competition_name": r["competition_name"],
        }
    finally:
        conn.close()


def resolve_fixture_date(
    db_path: Union[str, Path],
    provider_competition_id: str,
    home_team_id: int,
    away_team_id: int,
    *,
    provider: str = "statsbomb",
    season_name: Optional[str] = None,
    target_date: Optional[str] = None,
) -> str:
    """Resolve the fixture date from the DB, or accept a user-supplied date.

    If *target_date* is given (for a fixture not yet in the DB), it is
    returned directly.  Otherwise the fixture must exist in the DB (by team
    IDs + season) and its ``match_date`` is returned.
    """
    import sqlite3

    if target_date is not None:
        return str(target_date)

    db_path = Path(db_path)
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        conn.row_factory = sqlite3.Row
        where = ["home_team_id = ?", "away_team_id = ?"]
        params = [home_team_id, away_team_id]
        if season_name is not None:
            where.append("season_name = ?")
            params.append(season_name)

        rows = conn.execute(
            f"SELECT match_date FROM matches WHERE {' AND '.join(where)}",
            params,
        ).fetchall()
        if not rows:
            raise ValueError(
                f"Fixture not found in DB for home_team_id={home_team_id}, "
                f"away_team_id={away_team_id}, season={season_name}. "
                "Supply target_date for a fixture not yet in the database."
            )
        if len(rows) > 1:
            raise ValueError(
                f"Ambiguous fixture: {len(rows)} matches found for "
                f"home_team_id={home_team_id}, away_team_id={away_team_id}"
            )
        d = str(rows[0][0])
        if not d or d == "nan":
            raise ValueError("Fixture match_date is NULL in the DB.")
        return d
    finally:
        conn.close()


# --- History construction ---


def build_pre_match_history(
    all_matches: pd.DataFrame,
    fixture_date: str,
    home_team_id: int,
    away_team_id: int,
) -> pd.DataFrame:
    """Return strictly-pre-match history for a fixture.

    Only matches with match_date strictly earlier than fixture_date.
    Same-day matches excluded (same-day leakage prevention).
    Only completed matches (non-null goals) included.
    """
    history = all_matches[
        all_matches["match_date"].astype(str) < fixture_date
    ].copy()
    history = history[
        history["home_goals"].notna() & history["away_goals"].notna()
    ].copy()
    return history


# ---------------------------------------------------------------------------
# Prediction request / response dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PredictionRequest:
    """Validated input for a single match prediction.

    Attributes:
        provider: Data provider name (e.g. "statsbomb", "football-data").
        provider_competition_id: Provider-level competition identifier.
        home_team: Home team name (resolved via team_lookup).
        away_team: Away team name (resolved via team_lookup).
        model: Model name to use ("poisson" only for now).
        season_name: Season name (e.g. "2019/2020") if historical.
        target_date: Target fixture date as ISO YYYY-MM-DD (for fixtures
            not yet in the DB).
        max_goals: Poisson scoreline grid depth.
        min_history: Minimum pre-match matches before trusting team strength.
        prior_strength: Fallback strength for teams with insufficient history.
    """

    provider: str
    provider_competition_id: str
    home_team: str
    away_team: str
    model: str = DEFAULT_MODEL
    season_name: Optional[str] = None
    target_date: Optional[str] = None
    max_goals: int = DEFAULT_MAX_GOALS
    min_history: int = DEFAULT_MIN_HISTORY
    prior_strength: float = DEFAULT_PRIOR_STRENGTH

    def validate(self) -> None:
        """Validate self before any database access."""
        if not self.provider:
            raise ValueError("provider must be a non-empty string")
        if not self.provider_competition_id:
            raise ValueError("provider_competition_id must be a non-empty string")
        if not self.home_team:
            raise ValueError("home_team must be a non-empty string")
        if not self.away_team:
            raise ValueError("away_team must be a non-empty string")
        if self.home_team.strip().lower() == self.away_team.strip().lower():
            raise ValueError("home_team and away_team must be different teams")
        if self.model not in SUPPORTED_MODELS:
            raise ValueError(
                f"Unsupported model {self.model!r}. "
                f"Supported: {SUPPORTED_MODELS}"
            )
        if self.max_goals < 0:
            raise ValueError(f"max_goals must be >= 0, got {self.max_goals}")
        if self.min_history < 0:
            raise ValueError(f"min_history must be >= 0, got {self.min_history}")
        if self.prior_strength < 0:
            raise ValueError(
                f"prior_strength must be >= 0, got {self.prior_strength}"
            )
        if self.target_date is not None:
            self._validate_date(self.target_date)

    @staticmethod
    def _validate_date(date_str: str) -> None:
        try:
            datetime.strptime(date_str.strip(), "%Y-%m-%d")
        except ValueError:
            raise ValueError(
                                f"Invalid date format {date_str!r}. Expected ISO YYYY-MM-DD."
            )


@dataclass
class MatchPrediction:
    """Complete prediction output for a single fixture.

    All probability fields are validated to be finite, non-negative, and
    normalized (1X2 sums to 1.0).
    """

    model: str
    model_version: str
    provider: str
    provider_competition_id: str
    competition_name: str
    season_name: Optional[str]
    fixture_date: str
    home_team_id: int
    home_team_name: str
    away_team_id: int
    away_team_name: str
    home_win_prob: float
    draw_prob: float
    away_win_prob: float
    expected_home_goals: float
    expected_away_goals: float
    over_under: dict
    btts: dict
    double_chance: dict
    most_probable_scoreline: tuple
    top_scorelines: list
    total_goals_distribution: dict
    notes: list = field(default_factory=list)
    recent_form_features: dict = field(default_factory=dict)

    @property
    def outcome_probabilities(self) -> dict:
        """1X2 probabilities keyed by canonical labels."""
        return {
            OUTCOME_LABELS[0]: self.home_win_prob,
            OUTCOME_LABELS[1]: self.draw_prob,
            OUTCOME_LABELS[2]: self.away_win_prob,
        }

    def validate(self) -> None:
        """Validate all probability fields."""
        validate_probability_vector(
            [self.home_win_prob, self.draw_prob, self.away_win_prob]
        )
        for line_data in self.over_under.values():
            validate_probability_vector(
                [line_data["over"], line_data["under"]]
            )
        validate_probability_vector([self.btts["yes"], self.btts["no"]])
        for v in self.double_chance.values():
            if not (0.0 <= v <= 1.0) and not math.isclose(
                v, 0.0, abs_tol=1e-9
            ):
                raise ValueError(f"Invalid double-chance probability: {v}")
        if self.expected_home_goals < 0 or self.expected_away_goals < 0:
            raise ValueError("Expected goals must be non-negative")
        if not (0.0 <= self.most_probable_scoreline[2] <= 1.0):
            raise ValueError("Scoreline probability out of range")

    def to_dict(self) -> dict:
        """Serialise to a JSON-friendly dict."""
        return {
            "model": self.model,
            "model_version": self.model_version,
            "competition": {
                "provider": self.provider,
                "provider_competition_id": self.provider_competition_id,
                "competition_name": self.competition_name,
            },
            "season_name": self.season_name,
            "fixture_date": self.fixture_date,
            "home_team": {"id": self.home_team_id, "name": self.home_team_name},
            "away_team": {"id": self.away_team_id, "name": self.away_team_name},
            "outcome_probabilities": {
                "class_order": list(OUTCOME_LABELS),
                "home_win": self.home_win_prob,
                "draw": self.draw_prob,
                "away_win": self.away_win_prob,
            },
            "expected_goals": {
                "home": self.expected_home_goals,
                "away": self.expected_away_goals,
                "total": self.expected_home_goals + self.expected_away_goals,
            },
            "over_under": self.over_under,
            "both_teams_to_score": self.btts,
            "double_chance": self.double_chance,
            "most_probable_scoreline": {
                "home_goals": self.most_probable_scoreline[0],
                "away_goals": self.most_probable_scoreline[1],
                "probability": self.most_probable_scoreline[2],
            },
            "top_scorelines": [
                {"home_goals": h, "away_goals": a, "probability": p}
                for h, a, p in self.top_scorelines
            ],
            "total_goals_distribution": self.total_goals_distribution,
            "notes": self.notes,
            "recent_form_features": self.recent_form_features,
        }


@dataclass
class PredictionExplanation:
    """Honest, auditable explanation for a prediction."""

    model: str
    model_version: str
    training_cutoff: str
    features_available: list
    reasoning: list
    limitations: list
    backtest_metrics: dict
    warnings: list = field(default_factory=list)
    recent_form_features: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Top-level service
# ---------------------------------------------------------------------------


def _compute_recent_form_features(
    all_matches: pd.DataFrame,
    fixture_date: str,
    home_team_id: int,
    away_team_id: int,
) -> dict:
    """Compute pre-match form features for a fixture (for inspection/explanation).

    Builds a synthetic target-match row if the fixture is not yet in
    ``all_matches`` (future fixture with user-supplied date), then uses
    :class:`FeatureEngineer` to compute leak-safe features.  Only features
    relevant to *recent form* (last-5-match statistics) are returned; raw
    match identifiers and H2H are excluded to keep the dict focused.

    The Poisson model itself does **not** consume these features — it
    estimates attack/defence rates directly from the raw history via
    :func:`estimate_team_strengths`.  These features are provided for
    transparency and are consumed by the logistic-regression model
    during walk-forward evaluation.
    """
    try:
        target_row = pd.DataFrame([{
            "match_id": -999,  # synthetic sentinel
            "provider": "unknown",
            "provider_match_id": "future-fixture",
            "competition_id": 0,
            "season_name": None,
            "match_date": fixture_date,
            "home_team_id": int(home_team_id),
            "away_team_id": int(away_team_id),
            "home_goals": None,
            "away_goals": None,
            "match_status": "scheduled",
        }])
        matches = pd.concat([all_matches, target_row], ignore_index=True)
        engineer = FeatureEngineer()
        feats = engineer.compute_features(matches, -999)
        # Keep only recent-form-related keys
        form_keys = [
            k for k in feats
            if k.startswith(("home_", "away_"))
            and any(s in k for s in (
                "points_last_5", "wins_last_5", "draws_last_5",
                "losses_last_5", "goals_scored_last_5",
                "goals_conceded_last_5", "goal_diff_last_5",
                "matches_played_last_5", "points_per_match_last_5",
                "avg_goals_scored_last_5", "avg_goals_conceded_last_5",
                "clean_sheets_last_5", "matches_scored_last_5",
                "at_home_last_5", "at_away_last_5",
                "recency_weighted_points_last_5",
            ))
        ]
        return {k: feats[k] for k in form_keys}
    except Exception:
        return {}


def predict_match(
    provider: str = "statsbomb",
    provider_competition_id: str = "2",
    home_team: str = "",
    away_team: str = "",
    *,
    season_name: Optional[str] = None,
    target_date: Optional[str] = None,
    model: str = DEFAULT_MODEL,
    max_goals: int = DEFAULT_MAX_GOALS,
    min_history: int = DEFAULT_MIN_HISTORY,
    prior_strength: float = DEFAULT_PRIOR_STRENGTH,
    db_path: Optional[Union[str, Path]] = None,
) -> MatchPrediction:
    """Generate a validated Poisson prediction for a single fixture.

    Parameters
    ----------
    provider: Data provider name (e.g. "statsbomb" or "football-data").
    provider_competition_id: Provider-level competition identifier.
    home_team: Home team name (case-insensitive, must resolve uniquely).
    away_team: Away team name (case-insensitive, must resolve uniquely).
    season_name: Season name (e.g. "2019/2020") if historical.
    target_date: Target date as ISO YYYY-MM-DD (for fixtures not in DB).
    model: Model name (only "poisson" supported).
    max_goals: Poisson scoreline grid depth (default 8).
    min_history: Minimum pre-match matches before trusting team strength.
    prior_strength: Fallback strength for teams with insufficient history.
    db_path: Database path (defaults to the configured production DB).

    Returns
    -------
    MatchPrediction with validated probabilities, scoreline, and markets.

    Raises
    ------
    ValueError for unknown/ambiguous teams, unsupported models, invalid
    dates, fixture-not-found, or probability validation failures.
    FileNotFoundError if the database does not exist.
    RuntimeError if the production DB SHA-256 changes (fail-closed).
    """
    from ..config import settings

    if db_path is None:
        db_path = Path(settings.database_path)
    # --- Input validation ---
    req = PredictionRequest(
        provider=provider,
        provider_competition_id=provider_competition_id,
        home_team=home_team,
        away_team=away_team,
        model=model,
        season_name=season_name,
        target_date=target_date,
        max_goals=max_goals,
        min_history=min_history,
        prior_strength=prior_strength,
    )
    req.validate()

    db = Path(db_path)
    if not db.exists():
        raise FileNotFoundError(f"Database not found at {db}")

    # --- Production DB SHA-256 gate (fail-closed) ---
    sha_before: Optional[str] = None
    is_prod = db.resolve() == PROD_DB_PATH.resolve()
    if is_prod:
        sha_before = _sha256(db)
        if sha_before != KNOWN_PROD_HASH:
            raise RuntimeError(
                f"Production DB SHA-256 mismatch (fail-closed): "
                f"got {sha_before}, expected {KNOWN_PROD_HASH}"
            )
        logger.info("prod DB SHA-256 verified pre-prediction: %s", sha_before)

    # --- Resolve competition ---
    comp = resolve_competition(db, provider_competition_id, provider=provider)

    # --- Resolve teams ---
    home = resolve_team(db, home_team, provider=provider)
    away = resolve_team(db, away_team, provider=provider)

    # --- Resolve fixture date ---
    fixture_date = resolve_fixture_date(
        db,
        provider_competition_id,
        home["id"],
        away["id"],
        provider=provider,
        season_name=season_name,
        target_date=target_date,
    )

    # --- Load all matches from this competition (read-only) ---
    all_matches = load_matches_for_features(
        db_path=db,
        provider=provider,
        provider_competition_id=provider_competition_id,
        season_name=season_name,
    )

    if all_matches is None or len(all_matches) == 0:
        raise ValueError(
            f"No matches loaded for provider={provider}, "
            f"competition_id={provider_competition_id}, season={season_name}"
        )

    # --- Build pre-match history (strict date < fixture_date) ---
    history = build_pre_match_history(
        all_matches, fixture_date, home["id"], away["id"]
    )

    # --- Generate prediction ---
    if model == "poisson":
        model_obj = PoissonModel(
            max_goals=max_goals,
            min_history=min_history,
            prior_strength=prior_strength,
        )
        pred = model_obj.predict_fixture(home["id"], away["id"], history)
        model_version = "poisson-v1.0"
    else:
                raise ValueError(f"Unsupported model: {model}")

    # --- Compute derived markets ---
    hw, dr, aw = pred.home_win_prob, pred.draw_prob, pred.away_win_prob
    home_rate = float(pred.home_rate)
    away_rate = float(pred.away_rate)

    # Over/Under 1.5, 2.5, 3.5
    ou = over_under_probabilities(
        home_rate, away_rate,
        lines=(1.5, 2.5, 3.5),
        max_goals=max_goals,
    )

    # BTTS Yes/No
    btts_p, btts_not = both_teams_to_score_probability(
        home_rate, away_rate, threshold=1, max_goals=max_goals
    )
    btts = {"yes": float(btts_p), "no": float(btts_not)}

    # Double-chance
    dc = double_chance_probabilities(hw, dr, aw)

    # Score grid ranking
    grid = np.asarray(pred.score_grid, dtype=float)
    grid_sum = grid.sum()
    if grid_sum > 0:
        grid = grid / grid_sum
    flat_idx = np.argsort(grid.ravel())[::-1]
    rows_idx, cols_idx = np.unravel_index(flat_idx, grid.shape)
    top_scorelines = []
    for idx in range(min(5, len(flat_idx))):
        h_goals = int(rows_idx[idx])
        a_goals = int(cols_idx[idx])
        p = float(grid[h_goals, a_goals])
        top_scorelines.append((h_goals, a_goals, p))
    most_probable = top_scorelines[0]

    # Total-goals distribution (Poisson sum)
    dist = {}
    total_rate = home_rate + away_rate
    for k in range(max_goals + 1):
        dist[k] = float(poisson_pmf(k, total_rate)[0])

    notes = []
    if len(history) < min_history:
        notes.append(
            f"Insufficient pre-match history ({len(history)} matches < "
            f"min_history={min_history}); team strengths fall back to prior."
        )
    for note in pred.notes:
        notes.append(note)
    notes.append(
        "Independent Poisson assumption: no correlation between home and "
        "away goal counts."
    )
    notes.append(
        f"Scoreline grid truncated at {max_goals} goals per team; "
        "probabilities renormalised over the reported grid."
    )

    # --- Compute pre-match form features for inspection/explanation ---
    recent_form = _compute_recent_form_features(
        all_matches, fixture_date, home["id"], away["id"],
    )

    result = MatchPrediction(
        model=model,
        model_version=model_version,
        provider=provider,
        provider_competition_id=provider_competition_id,
        competition_name=comp["competition_name"],
        season_name=season_name,
        fixture_date=fixture_date,
        home_team_id=home["id"],
        home_team_name=home["team_name"],
        away_team_id=away["id"],
        away_team_name=away["team_name"],
        home_win_prob=float(hw),
        draw_prob=float(dr),
        away_win_prob=float(aw),
        expected_home_goals=home_rate,
        expected_away_goals=away_rate,
        over_under=ou,
        btts=btts,
        double_chance=dc,
        most_probable_scoreline=most_probable,
        top_scorelines=top_scorelines,
        total_goals_distribution=dist,
        notes=notes,
        recent_form_features=recent_form,
    )

    result.validate()

    if is_prod:
        sha_after = _sha256(db)
        if sha_after != sha_before:
            raise RuntimeError(
                "Production DB SHA-256 changed during prediction "
                f"(fail-closed): before={sha_before} after={sha_after}"
            )
        logger.info("prod DB SHA-256 verified post-prediction: %s", sha_after)

    return result


def build_explanation(
    request,
    prediction,
    training_cutoff: str = "",
    *,
    is_historical: bool = True,
) -> "PredictionExplanation":
    """Build an honest explanation for a prediction."""
    features = [
        "Home team pre-match goal-scoring rate (goals per game)",
        "Away team pre-match goal-scoring rate (goals per game)",
        "Home team defensive rate (goals conceded per game at home)",
        "Away team defensive rate (goals conceded per game away)",
        "League-wide home and away goal means (context baselines)",
        "Interaction features: points/goal-diff over last 5 and 10 matches",
        "Per-match averages: points per match, avg goals scored/conceded (last 5/10)",
        "Clean sheets and matches scored in (last 5/10 matches)",
        "Venue-specific form: wins/draws/losses, points/match, clean sheets (last 5)",
        "Recency-weighted points per match (last 5, linear weights 1..5)",
    ]

    outcomes = [
        ("home_win", prediction.home_win_prob),
        ("draw", prediction.draw_prob),
        ("away_win", prediction.away_win_prob),
    ]
    best_outcome = max(outcomes, key=lambda x: x[1])[0]

    reasoning = [
        f"Poisson model estimates expected goals: "
        f"home={prediction.expected_home_goals:.2f}, "
        f"away={prediction.expected_away_goals:.2f}.",
        f"Implied outcome probabilities (class order: {list(OUTCOME_LABELS)}): "
        f"P(home)={prediction.home_win_prob:.1%}, "
        f"P(draw)={prediction.draw_prob:.1%}, "
        f"P(away)={prediction.away_win_prob:.1%}.",
        f"Most likely outcome: {best_outcome} "
        f"(most probable scoreline: "
        f"{prediction.most_probable_scoreline[0]}-"
        f"{prediction.most_probable_scoreline[1]}, "
        f"P={prediction.most_probable_scoreline[2]:.1%}). "
        "Note: the most probable scoreline may differ from the most probable outcome.",
    ]

    limitations = [
        "Poisson assumes goals are independent between teams; real matches "
        "exhibit some positive correlation (e.g. game-state effects).",
        "Team strength is estimated from historical results only; no current "
        "injuries, suspensions, lineups, weather, or live odds are available.",
        "The model was not ranked against other models for prediction "
        "quality; Poisson is used as the documented Phase-25 benchmark.",
        "Scoreline probabilities are truncated at max_goals and renormalised, "
        "introducing small truncation error.",
        "Training data is limited to verified E0 seasons.",
    ]

    if not is_historical:
        limitations.append(
            "target_date was user-supplied because the fixture is not yet "
            "in the database; team resolution relied on the provided names."
        )
        limitations.append(
            "No actual match result exists; this is a prospective prediction."
        )

    warnings = []
    if training_cutoff:
        warnings.append(
            f"Prediction respects pre-match cutoff: history < {training_cutoff} "
            "(strict less-than, same-day safe)."
        )

    backtest = PHASE25_POOLED_METRICS.get("poisson", {})

    return PredictionExplanation(
        model=prediction.model,
        model_version=prediction.model_version,
        training_cutoff=training_cutoff if training_cutoff else "n/a",
        features_available=features,
        reasoning=reasoning,
        limitations=limitations,
        backtest_metrics=dict(backtest),
        warnings=warnings,
        recent_form_features=prediction.recent_form_features,
    )
