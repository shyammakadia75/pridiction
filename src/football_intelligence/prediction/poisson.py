"""Poisson goal model for match-outcome probabilities (Phase 20).

A classic, transparent independent-Poisson goal model.  Each team is assigned a
single expected-goals rate and the two teams' goal counts are modelled as
independent Poisson random variables.  All match probabilities are derived
from the product of the two marginal Poisson distributions.

The implementation is self-contained (numpy only -- reusing the project's
existing numerical stack) and makes every modelling assumption explicit.

Model
-----
For a fixture between home team H and away team A:

    home_goals ~ Poisson(lambda_h)
    away_goals ~ Poisson(lambda_a)

    P(H wins)   = sum over grid cells with i > j
    P(draw)     = sum over grid cells with i == j
    P(A wins)   = sum over grid cells with i < j
    P(over 2.5) = 1 - P(total <= 2) via Poisson CDF of (lambda_h + lambda_a)
    P(BTTS)     = P(home>=1) * P(away>=1)   (independence)

The scoreline grid is truncated at ``max_goals``; outcome probabilities are
re-normalised over the reported grid so they sum to 1.

Expected goals estimation
-------------------------
Rates are estimated from *eligible* history only -- matches strictly before the
target fixture's date (the caller enforces the chronological cutoff; this
module never looks at the target fixture itself).

    home_attack  = home goals scored per game / league home mean
    away_attack  = away goals scored per game / league away mean
    home_defence = away goals conceded by home team (at home) / league away mean
    away_defence = home goals conceded by away team (away) / league home mean
    lambda_home  = league_home_mean * home_attack * away_defence
    lambda_away  = league_away_mean * away_attack * home_defence

Teams with fewer than ``min_history`` matches fall back to ``prior_strength``
(default 1.0 = league average).  When no eligible history exists the league-wide
rates are used with strength 1.0 for both teams, guaranteeing a deterministic,
defensive prediction that never looks at future matches.  All rates are clamped
to a small positive floor so the Poisson mean is strictly positive.
"""
from __future__ import annotations

import math
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd
from scipy.stats import poisson

from ..database import resolve_db_path
from ..features.engineering import load_matches_for_features

# Outcome-label convention mirrors football_intelligence.prediction.baseline.
#   0 = "home_win", 1 = "draw", 2 = "away_win".
__all__ = [
    "OUTCOME_LABELS",
    "OUTCOME_INDEX",
    "poisson_pmf",
    "score_probability_grid",
    "home_draw_away_probabilities",
    "over_under_probability",
    "both_teams_to_score_probability",
    "PoissonFixturePrediction",
    "PoissonModel",
    "PoissonReport",
    "run_poisson_chronological",
    "run_poisson_evaluation",
    "save_poisson_report",
    "DEFAULT_MAX_GOALS",
    "DEFAULT_MIN_HISTORY",
    "DEFAULT_PRIOR_STRENGTH",
    "DEFAULT_RATE_FLOOR",
]

OUTCOME_LABELS: tuple[str, str, str] = ("home_win", "draw", "away_win")
OUTCOME_INDEX: dict[str, int] = {lbl: i for i, lbl in enumerate(OUTCOME_LABELS)}

# Configurable defaults
DEFAULT_MAX_GOALS: int = 8
DEFAULT_MIN_HISTORY: int = 5
DEFAULT_PRIOR_STRENGTH: float = 1.0
DEFAULT_RATE_FLOOR: float = 1e-4


# ---------------------------------------------------------------------------
# Core Poisson primitives
# ---------------------------------------------------------------------------
def poisson_pmf(k, lam: float) -> np.ndarray:
    """Poisson probability mass function ``P(X = k)`` for mean ``lam``.

    ``lam`` is clamped to a small positive floor so the mean is never exactly
    zero (see module docstring).  The result is identical to
    ``scipy.stats.poisson.pmf`` for positive ``lam``.
    """
    lam_safe = float(lam) if lam is not None else 0.0
    if lam_safe < 0 or not math.isfinite(lam_safe):
        lam_safe = 0.0
    if lam_safe < DEFAULT_RATE_FLOOR:
        lam_safe = DEFAULT_RATE_FLOOR
    ks = np.atleast_1d(np.asarray(k, dtype=int))
    return np.asarray(poisson.pmf(ks, lam_safe), dtype=float)


def score_probability_grid(home_rate: float, away_rate: float,
                           max_goals: int = DEFAULT_MAX_GOALS) -> np.ndarray:
    """Return the ``(max_goals+1, max_goals+1)`` scoreline probability matrix.

    Entry ``[i, j]`` is ``P(home scores i AND away scores j)`` under the
    independent-Poisson assumption.  The matrix is re-normalised over the grid
    so its total mass is 1 (i.e. probability assigned to scores above
    ``max_goals`` is redistributed across the reported grid).
    """
    if max_goals < 0:
        max_goals = DEFAULT_MAX_GOALS
    max_goals = int(max_goals)
    home_safe = max(float(home_rate), DEFAULT_RATE_FLOOR)
    away_safe = max(float(away_rate), DEFAULT_RATE_FLOOR)
    ks = np.arange(max_goals + 1, dtype=float)
    home_pmf = poisson.pmf(ks, home_safe)
    away_pmf = poisson.pmf(ks, away_safe)
    grid = np.outer(home_pmf, away_pmf)
    s = grid.sum()
    if not np.isfinite(s) or s <= 0.0:
        grid = np.full_like(grid, 1.0 / grid.size)
    else:
        grid = grid / s
    return grid


def home_draw_away_probabilities(home_rate: float, away_rate: float,
                                 max_goals: int = DEFAULT_MAX_GOALS) -> tuple[float, float, float]:
    """Return ``(home_win, draw, away_win)`` probabilities from the grid.

    Computed by summing the appropriate cells of :func:`score_probability_grid`
    (which is already renormalised over the reported grid), so the three
        returned probabilities sum to 1.
    """
    grid = score_probability_grid(home_rate, away_rate, max_goals)
    home_win = float(np.tril(grid, k=-1).sum())    # i > j -> home > away
    draw = float(np.trace(grid))                   # i == j
    away_win = float(np.triu(grid, k=1).sum())     # i < j -> away > home
    total = home_win + draw + away_win
    if not np.isfinite(total) or total <= 0.0:
        base = 1.0 / 3.0
        return base, base, base
    return home_win / total, draw / total, away_win / total


def over_under_probability(home_rate: float, away_rate: float,
                           line: float = 2.5,
                           max_goals: int = DEFAULT_MAX_GOALS) -> tuple[float, float]:
    """Return ``(P(over line), P(under line))`` for total goals.

    ``line=2.5`` is the conventional "over/under 2.5 goals" market.  Computed
    from the Poisson CDF of the sum of two independent Poisson variables
    (``Poisson(lambda_home + lambda_away)``), so the result is exact and
    independent of the ``max_goals`` grid truncation.
    """
    home_safe = max(float(home_rate), DEFAULT_RATE_FLOOR)
    away_safe = max(float(away_rate), DEFAULT_RATE_FLOOR)
    over_p = float(1.0 - poisson.cdf(line, home_safe + away_safe))
    over_p = max(0.0, min(1.0, over_p))
    under_p = 1.0 - over_p
    return over_p, under_p


def both_teams_to_score_probability(home_rate: float, away_rate: float,
                                    threshold: int = 1,
                                    max_goals: int = DEFAULT_MAX_GOALS) -> tuple[float, float]:
    """Return ``(P(both>=thr), P(not))`` for "both teams score >= threshold".

    ``threshold=1`` is the standard BTTS market (both teams score at least 1
    goal).  Uses independence: ``P(both >= t) = P(home>=t) * P(away>=t)``,
    computed from the Poisson survival function (exact, grid-independent).
    """
    home_safe = max(float(home_rate), DEFAULT_RATE_FLOOR)
    away_safe = max(float(away_rate), DEFAULT_RATE_FLOOR)
    p_home = float(poisson.sf(threshold - 1, home_safe))
    p_away = float(poisson.sf(threshold - 1, away_safe))
    btts = max(0.0, min(1.0, p_home * p_away))
    return btts, 1.0 - btts




# ---------------------------------------------------------------------------
# Parameter estimation
# ---------------------------------------------------------------------------
@dataclass
class TeamStrengths:
    """Container for estimated Poisson model parameters (single fixture)."""

    home_rate: float = 0.0
    away_rate: float = 0.0
    league_home_mean: float = 0.0
    league_away_mean: float = 0.0
    home_attack: float = DEFAULT_PRIOR_STRENGTH
    home_defence: float = DEFAULT_PRIOR_STRENGTH
    away_attack: float = DEFAULT_PRIOR_STRENGTH
    away_defence: float = DEFAULT_PRIOR_STRENGTH
    home_history: int = 0
    away_history: int = 0
    notes: list = field(default_factory=list)


def _safe_mean(values: Sequence[float]) -> float:
    arr = np.asarray(list(values), dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return 0.0
    return float(arr.mean())




def estimate_team_strengths(history: pd.DataFrame,
                            home_team_id, away_team_id,
                            min_history: int = DEFAULT_MIN_HISTORY,
                            prior_strength: float = DEFAULT_PRIOR_STRENGTH) -> TeamStrengths:
    """Estimate Poisson attack/defence strengths for a fixture from history.

    ``history`` must contain ``home_team_id``, ``away_team_id``,
    ``home_goals``, ``away_goals`` and be restricted to matches strictly
    earlier than the target fixture (leakage-safe).  Teams with fewer than
    ``min_history`` matches fall back to ``prior_strength``.
        """
    s = TeamStrengths(notes=[])
    if len(history) == 0:
        s.notes.append("No eligible pre-match history; using league-wide prior.")
        return s
    # Defensive: missing goal/team columns -> no usable signal.
    need = {"home_team_id", "away_team_id", "home_goals", "away_goals"}
    if not need.issubset(history.columns):
        s.notes.append(
            "Goal/team columns missing from history; using league-wide prior."
        )
        return s


    home_goals = pd.to_numeric(history["home_goals"], errors="coerce")
    away_goals = pd.to_numeric(history["away_goals"], errors="coerce")
    valid = home_goals.notna() & away_goals.notna()
    h_valid = home_goals[valid]
    a_valid = away_goals[valid]
    if len(h_valid) == 0:
        s.notes.append("No history with valid goal counts; using league-wide prior.")
        return s

    league_home_mean = _safe_mean(h_valid)
    league_away_mean = _safe_mean(a_valid)
    s.league_home_mean = league_home_mean
    s.league_away_mean = league_away_mean
    lm_h = max(league_home_mean, DEFAULT_RATE_FLOOR)
    lm_a = max(league_away_mean, DEFAULT_RATE_FLOOR)

    home_attack = prior_strength
    home_defence = prior_strength
    away_attack = prior_strength
    away_defence = prior_strength

    home_hist = history[history["home_team_id"] == home_team_id]
    away_hist = history[history["away_team_id"] == away_team_id]
    h_valid_rows = home_hist[
        pd.to_numeric(home_hist["home_goals"], errors="coerce").notna()
        & pd.to_numeric(home_hist["away_goals"], errors="coerce").notna()
    ]
    a_valid_rows = away_hist[
        pd.to_numeric(away_hist["away_goals"], errors="coerce").notna()
        & pd.to_numeric(away_hist["home_goals"], errors="coerce").notna()
    ]
    h_count = len(h_valid_rows)
    a_count = len(a_valid_rows)
    s.home_history = h_count
    s.away_history = a_count

    if h_count >= min_history:
        home_attack = float(h_valid_rows["home_goals"].astype(float).mean()) / lm_h
        home_defence = float(h_valid_rows["away_goals"].astype(float).mean()) / lm_a
        s.notes.append(f"Home strength from {h_count} pre-match matches.")
    else:
        s.notes.append(
            f"Home team has {h_count} pre-match matches "
            f"(< min_history={min_history}); using prior strength."
        )
    if a_count >= min_history:
        away_attack = float(a_valid_rows["away_goals"].astype(float).mean()) / lm_a
        away_defence = float(a_valid_rows["home_goals"].astype(float).mean()) / lm_h
        s.notes.append(f"Away strength from {a_count} pre-match matches.")
    else:
        s.notes.append(
            f"Away team has {a_count} pre-match matches "
            f"(< min_history={min_history}); using prior strength."
        )

    s.home_attack = home_attack
    s.home_defence = home_defence
    s.away_attack = away_attack
    s.away_defence = away_defence

    if league_home_mean > 0 and league_away_mean > 0:
        s.home_rate = league_home_mean * home_attack * away_defence
        s.away_rate = league_away_mean * away_attack * home_defence
    else:
        s.home_rate = lm_h
        s.away_rate = lm_a
        s.notes.append("League means unavailable; fallback to rate-floor means.")

    s.home_rate = max(float(s.home_rate), DEFAULT_RATE_FLOOR)
    s.away_rate = max(float(s.away_rate), DEFAULT_RATE_FLOOR)
    return s



class PoissonModel:
    """Independent Poisson goal model with home/away attack-defence Strengths."""

    def __init__(self, max_goals: int = DEFAULT_MAX_GOALS,
                 min_history: int = DEFAULT_MIN_HISTORY,
                 prior_strength: float = DEFAULT_PRIOR_STRENGTH) -> None:
        self.max_goals = max(int(max_goals), 0)
        self.min_history = int(min_history)
        self.prior_strength = float(prior_strength)

    def predict_fixture(self, home_team_id, away_team_id,
                        history: pd.DataFrame) -> "PoissonFixturePrediction":
        """Predict one fixture from its eligible pre-match ``history``.

        ``history`` must contain only matches strictly earlier than the target
        fixture (leakage-safe).  Deterministic for identical inputs.
        """
        return self._build(home_team_id, away_team_id, history)

    def _build(self, home_team_id, away_team_id,
               history: pd.DataFrame) -> "PoissonFixturePrediction":
        strengths = estimate_team_strengths(
            history, home_team_id, away_team_id,
            min_history=self.min_history,
            prior_strength=self.prior_strength,
        )
        home_rate = max(float(strengths.home_rate), DEFAULT_RATE_FLOOR)
        away_rate = max(float(strengths.away_rate), DEFAULT_RATE_FLOOR)
        grid = score_probability_grid(home_rate, away_rate, self.max_goals)
        hw, dr, aw = home_draw_away_probabilities(home_rate, away_rate, self.max_goals)
        over, under = over_under_probability(
            home_rate, away_rate, line=2.5, max_goals=self.max_goals)
        btts, _not_btts = both_teams_to_score_probability(
            home_rate, away_rate, threshold=1, max_goals=self.max_goals)
        idx = np.unravel_index(np.argmax(grid), grid.shape)
        most_probable = (int(idx[0]), int(idx[1]), float(grid[idx]))
        return PoissonFixturePrediction(
            home_team_id=home_team_id,
            away_team_id=away_team_id,
            home_rate=home_rate,
            away_rate=away_rate,
            league_home_mean=strengths.league_home_mean,
            league_away_mean=strengths.league_away_mean,
            home_attack=strengths.home_attack,
            home_defence=strengths.home_defence,
            away_attack=strengths.away_attack,
            away_defence=strengths.away_defence,
            home_win_prob=hw,
            draw_prob=dr,
            away_win_prob=aw,
            over_2_5=over,
            under_2_5=under,
            both_teams_to_score=btts,
            score_grid=grid,
            most_probable_scoreline=most_probable,
            notes=strengths.notes,
        )


@dataclass
class PoissonFixturePrediction:
    """Full Poisson prediction for a single fixture."""

    home_team_id: object
    away_team_id: object
    home_rate: float
    away_rate: float
    league_home_mean: float
    league_away_mean: float
    home_attack: float
    home_defence: float
    away_attack: float
    away_defence: float
    home_win_prob: float
    draw_prob: float
    away_win_prob: float
    over_2_5: float
    under_2_5: float
    both_teams_to_score: float
    score_grid: np.ndarray
    most_probable_scoreline: tuple
    notes: list = field(default_factory=list)

    @property
    def probabilities(self) -> dict:
        """3-class outcome probabilities keyed by the project label convention."""
        return {
            OUTCOME_LABELS[0]: float(self.home_win_prob),
            OUTCOME_LABELS[1]: float(self.draw_prob),
            OUTCOME_LABELS[2]: float(self.away_win_prob),
        }

    def to_dict(self) -> dict:
        return {
            "home_team_id": self.home_team_id,
            "away_team_id": self.away_team_id,
            "home_rate": self.home_rate,
            "away_rate": self.away_rate,
            "league_home_mean": self.league_home_mean,
            "league_away_mean": self.league_away_mean,
            "home_attack": self.home_attack,
            "home_defence": self.home_defence,
            "away_attack": self.away_attack,
            "away_defence": self.away_defence,
            "home_win_prob": self.home_win_prob,
            "draw_prob": self.draw_prob,
            "away_win_prob": self.away_win_prob,
            "over_2_5": self.over_2_5,
            "under_2_5": self.under_2_5,
            "both_teams_to_score_prob": self.both_teams_to_score,
            "most_probable_scoreline": {
                "home_goals": self.most_probable_scoreline[0],
                "away_goals": self.most_probable_scoreline[1],
                "probability": self.most_probable_scoreline[2],
            },
            "max_goals": int(self.score_grid.shape[0] - 1),
            "notes": list(self.notes),
        }




# ---------------------------------------------------------------------------
# JSON + SHA helpers (local mirrors of evaluation._json_default / _sha256)
# ---------------------------------------------------------------------------
def _json_default(o):
    """JSON fallback for numpy/pandas objects (mirrors evaluation._json_default)."""
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        f = float(o)
        return None if (math.isnan(f) or math.isinf(f)) else f
    if isinstance(o, float):
        return None if (math.isnan(o) or math.isinf(o)) else o
    if isinstance(o, (set, frozenset)):
        return list(o)
    import datetime as _dt
    if isinstance(o, (_dt.datetime, _dt.date, pd.Timestamp)):
        return o.isoformat()
    if o is pd.NaT:
        return None
    return str(o)


def _sha256(path) -> Optional[str]:
    """SHA-256 hex digest of a file (read-only tamper guard)."""
    import hashlib
    p = Path(path)
    if not p.exists():
        return None
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_poisson_matches(db_path=None, *, provider="statsbomb",
                          provider_competition_id="2", season_name=None) -> pd.DataFrame:
    """Load + sort matches for a single competition (mirrors elo.load_elo_matches)."""
    db = resolve_db_path(db_path)
    raw = load_matches_for_features(
        db, provider=provider,
        provider_competition_id=str(provider_competition_id),
        season_name=season_name,
    )
    if raw is None or len(raw) == 0:
        raise ValueError(
            f"No matches for provider={provider}, "
            f"competition_id={provider_competition_id}, season={season_name}"
        )
    return raw.sort_values("match_date").reset_index(drop=True)


def _outcome_label(home_goals, away_goals) -> Optional[int]:
    """Map a played scoreline to the 0/1/2 outcome index; None if invalid."""
    try:
        hg = int(round(float(home_goals)))
        ag = int(round(float(away_goals)))
    except (TypeError, ValueError):
        return None
    if hg < 0 or ag < 0:
        return None
    if hg > ag:
        return OUTCOME_INDEX["home_win"]
    if hg == ag:
        return OUTCOME_INDEX["draw"]
    return OUTCOME_INDEX["away_win"]


def run_poisson_chronological(matches: pd.DataFrame,
                              model: Optional[PoissonModel] = None) -> tuple:
    """Generate leak-free Poisson predictions for every fixture, in order.

    For each fixture ``m`` (chronological order):

    1. Eligible history = matches strictly earlier than ``m.match_date``.
    2. Estimate model parameters from that history.
    3. Generate the fixture's probabilities **before** reading its result.
    4. Only then read ``m.home_goals`` / ``m.away_goals`` for scoring.

    Returns ``(predictions, summary)`` where ``predictions`` is a list of
    per-fixture dicts (with ``history_size`` and ``cutoff_date``) and
    ``summary`` holds dataset counts.
    """
    if model is None:
        model = PoissonModel()
    predictions: list = []
    df = matches.sort_values("match_date").reset_index(drop=True)
    for i, row in df.iterrows():
        cutoff_date = str(row["match_date"])
        # Strictly earlier dates ONLY: same-day results must never leak into a
        # fixture's parameter estimation (multiple fixtures share a match_date).
        history = df[df["match_date"].astype(str) < cutoff_date]
        pred = model.predict_fixture(row["home_team_id"], row["away_team_id"], history)
        probs = pred.probabilities
        outcome = _outcome_label(row["home_goals"], row["away_goals"])
        rec = {
            "match_index": int(i),
            "match_date": cutoff_date,
            "cutoff_date": cutoff_date,
            "home_team_id": row["home_team_id"],
            "away_team_id": row["away_team_id"],
            "home_goals": row["home_goals"],
            "away_goals": row["away_goals"],
            "history_size": int(len(history)),
            "home_rate": float(pred.home_rate),
            "away_rate": float(pred.away_rate),
            "probs": {k: float(v) for k, v in probs.items()},
            "over_2_5": float(pred.over_2_5),
            "under_2_5": float(pred.under_2_5),
            "both_teams_to_score": float(pred.both_teams_to_score),
            "most_probable_scoreline": {
                "home_goals": pred.most_probable_scoreline[0],
                "away_goals": pred.most_probable_scoreline[1],
                "probability": pred.most_probable_scoreline[2],
            },
            "outcome_label": OUTCOME_LABELS[outcome] if outcome is not None else None,
            "outcome_idx": outcome,
        }
        predictions.append(rec)
    summary = {
        "n_matches": int(len(df)),
        "n_trainable": int(sum(1 for p in predictions if p["history_size"] > 0)),
    }
    return predictions, summary




# ---------------------------------------------------------------------------
# Chronological evaluation
# ---------------------------------------------------------------------------
@dataclass
class PoissonReport:
    """Evaluation artifact for the Poisson model (mirrors EloReport shape)."""

    dataset: dict
    predictions: list = field(default_factory=list)
    eval_metrics: dict = field(default_factory=dict)
    config: dict = field(default_factory=dict)
    produced_at: str = ""
    notes: list = field(default_factory=list)
    production_db_sha256: Optional[str] = None
    production_db_unchanged: Optional[bool] = None

    def to_dict(self) -> dict:
        return {
            "dataset": self.dataset,
            "predictions": self.predictions,
            "eval_metrics": self.eval_metrics,
            "config": self.config,
            "produced_at": self.produced_at,
            "notes": self.notes,
            "production_db_sha256": self.production_db_sha256,
            "production_db_unchanged": self.production_db_unchanged,
        }


def _score_predictions(scored: list, metrics: dict) -> None:
    """Populate ``metrics`` in place from scored prediction records."""
    from .baseline import (
        compute_accuracy, compute_log_loss, compute_brier_score,
        build_confusion_matrix, class_distribution,
    )
    from .evaluation import (
        compute_macro_f1, compute_precision_recall, compute_calibration,
    )
    y_true = np.array([p["outcome_idx"] for p in scored], dtype=int)
    proba = np.array(
        [[p["probs"][lbl] for lbl in OUTCOME_LABELS] for p in scored], dtype=float)
    y_pred = proba.argmax(axis=1)
    metrics["n_scored"] = int(len(scored))
    metrics["accuracy"] = compute_accuracy(y_true, y_pred, OUTCOME_LABELS)
    metrics["macro_f1"] = compute_macro_f1(y_true, y_pred, OUTCOME_LABELS)
    metrics["brier_score"] = compute_brier_score(y_true, proba, OUTCOME_LABELS)
    metrics["log_loss"] = compute_log_loss(y_true, proba, OUTCOME_LABELS)
    metrics["per_class_precision_recall"] = compute_precision_recall(
        y_true, y_pred, OUTCOME_LABELS)
    metrics["calibration"] = compute_calibration(
        y_true, proba, OUTCOME_LABELS, n_bins=5)
    metrics["confusion_matrix"] = build_confusion_matrix(
        y_true, y_pred, OUTCOME_LABELS)
    metrics["class_distribution"] = class_distribution(y_true, OUTCOME_LABELS)

    exp_total = np.array([p["home_rate"] + p["away_rate"] for p in scored],
                         dtype=float)
    act_total = np.array(
        [float(p["home_goals"]) + float(p["away_goals"]) for p in scored],
        dtype=float)
    mask = np.isfinite(exp_total) & np.isfinite(act_total)
    if mask.any():
        gm = metrics["goal_metrics"]
        gm["expected_total_goals_mae"] = float(
            np.mean(np.abs(exp_total[mask] - act_total[mask])))
        gm["mean_actual_total_goals"] = float(np.mean(act_total[mask]))
        gm["mean_predicted_total_goals"] = float(np.mean(exp_total[mask]))
    over_actual = act_total > 2.5
    over_pred = np.array([p["over_2_5"] for p in scored], dtype=float)
    if over_pred.size and mask.any():
        metrics["goal_metrics"]["over_2_5_brier"] = float(
            np.mean((over_pred - over_actual.astype(float)) ** 2))
    btts_actual = np.array(
        [bool(float(p["home_goals"]) >= 1) and bool(float(p["away_goals"]) >= 1)
         for p in scored], dtype=bool)
    btts_pred = np.array([p["both_teams_to_score"] for p in scored], dtype=float)
    if btts_pred.size and mask.any():
        metrics["goal_metrics"]["btts_brier"] = float(
            np.mean((btts_pred - btts_actual.astype(float)) ** 2))



def run_poisson_evaluation(db_path=None, *, provider: str = "statsbomb",
                           provider_competition_id: str = "2",
                           season_name: Optional[str] = None,
                           max_goals: int = DEFAULT_MAX_GOALS,
                           min_history: int = DEFAULT_MIN_HISTORY,
                           prior_strength: float = DEFAULT_PRIOR_STRENGTH) -> PoissonReport:
    """Run a leak-free chronological Poisson evaluation.

    The database SHA-256 is captured before and after loading; any change raises
    RuntimeError (no writes to the production DB).  Predictions are generated
    strictly from pre-match history per fixture.
    """
    db = resolve_db_path(db_path)
    sha_before = _sha256(db)
    df = _load_poisson_matches(db, provider=provider,
                               provider_competition_id=provider_competition_id,
                               season_name=season_name)
    sha_after = _sha256(db)
    if sha_before is not None and sha_before != sha_after:
        raise RuntimeError(
            "Database SHA-256 changed during Poisson evaluation -- aborting "
            "(writes to a read-only DB violate the leakage contract)."
        )

    model = PoissonModel(max_goals=max_goals, min_history=min_history,
                         prior_strength=prior_strength)
    predictions, summary = run_poisson_chronological(df, model=model)

    n = len(predictions)
    metrics: dict = {
        "n": n, "accuracy": 0.0, "macro_f1": 0.0, "brier_score": 0.0,
        "log_loss": 0.0, "per_class_precision_recall": {},
        "calibration": {lbl: [] for lbl in OUTCOME_LABELS},
        "confusion_matrix": [], "class_distribution": {},
        "labels": list(OUTCOME_LABELS), "goal_metrics": {},
    }
    scored = [p for p in predictions if p["outcome_idx"] is not None]
    if scored:
        _score_predictions(scored, metrics)

    teams = set(df["home_team_id"]).union(df["away_team_id"])
    report = PoissonReport(
        dataset={
            "provider": provider,
            "provider_competition_id": str(provider_competition_id),
            "season_name": season_name,
            "n_matches": int(len(df)),
            "n_teams": int(len(teams)),
            "date_min": str(df["match_date"].min()) if len(df) else None,
            "date_max": str(df["match_date"].max()) if len(df) else None,
            "n_scored": metrics.get("n_scored", 0),
            "evaluation_protocol": (
                "Chronological online prediction: fixture i is scored from "
                "history = matches[0..i-1] (strictly earlier, no same-day, "
                "no future). Parameters re-estimated per fixture."
            ),
        },
        predictions=predictions,
        eval_metrics=metrics,
                config={
            "max_goals": max(int(max_goals), 0),
            "min_history": max(int(min_history), 0),
            "prior_strength": float(prior_strength),
            "rate_floor": DEFAULT_RATE_FLOOR,
            "over_under_line": 2.5,
            "btts_threshold": 1,
        },
        produced_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        notes=[
            "Read-only evaluation: SELECT-only loaders; SHA-256 re-checked after "
            "load; any modification aborts.",
            "No cross-competition leakage: load-time provider_competition_id filter "
            "isolates a single competition.",
            "No same-day / future leakage: each fixture's history is the strict "
            "prefix of earlier-dated matches.",
            "No model selection: hyperparameters are fixed; metrics are "
            "descriptive, not ranked as superiority claims.",
            "Goal model is independent Poisson (no correlated-score correction).",
            "Over/under and BTTS metrics use Brier score only.",
        ],
        production_db_sha256=sha_after,
        production_db_unchanged=(True if sha_before is None else sha_before == sha_after),
    )
    return report



def save_poisson_report(report: PoissonReport, output_dir: str,
                        name: str = "phase20_poisson") -> dict:
    """Write JSON + readable TXT artifacts. NEVER writes to the production DB."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    base = out / f"{name}_{ts}"
    json_path = base.with_suffix(".json")
    txt_path = base.with_suffix(".txt")
    json_path.write_text(
        json.dumps(report.to_dict(), indent=2, default=_json_default),
        encoding="utf-8",
    )
    _write_poisson_txt(report, txt_path)
    return {"json": str(json_path), "txt": str(txt_path), "dir": str(out)}


def _write_poisson_txt(report: PoissonReport, path) -> None:
    L = []
    L.append("Phase-20 Poisson Goal Model Report")
    L.append("===================================")
    L.append(f"Produced: {report.produced_at}")
    d = report.dataset
    L.append(f"provider={d.get('provider')} competition_id={d.get('provider_competition_id')} season={d.get('season_name')}")
    L.append(f"n_matches={d.get('n_matches')} n_teams={d.get('n_teams')} date_min={d.get('date_min')} date_max={d.get('date_max')}")
    L.append(f"production DB sha256: {report.production_db_sha256}")
    L.append(f"production DB unchanged: {report.production_db_unchanged}")
    L.append("")
    L.append("Config")
    L.append("------")
    for k, v in report.config.items():
        L.append(f"  {k}: {v}")
    L.append("")
    L.append("Outcome metrics (online, chronological)")
    L.append("--------------------------------------")
    m = report.eval_metrics
    L.append(f"  n={m.get('n')} n_scored={m.get('n_scored', m.get('n'))}")
    L.append(f"  accuracy: {m.get('accuracy'):.4f}")
    L.append(f"  macro_f1: {m.get('macro_f1'):.4f}")
    L.append(f"  brier_score: {m.get('brier_score'):.4f}")
    L.append(f"  log_loss: {m.get('log_loss'):.4f}")
    pc = m.get("per_class_precision_recall", {})
    if pc:
        L.append("  per-class P/R/F1:")
        for lbl in OUTCOME_LABELS:
            c = pc.get(lbl, {})
            L.append(f"    {lbl:9s} P={c.get('precision', 0.0):.4f} "
                     f"R={c.get('recall', 0.0):.4f} F1={c.get('f1', 0.0):.4f} N={c.get('support', 0)}")
    cd = m.get("class_distribution") or {}
    if cd:
        L.append(f"  class distribution counts: {cd.get('counts')}")
    gm = m.get("goal_metrics", {}) or {}
    if gm:
        L.append("")
        L.append("Goal metrics")
        L.append("------------")
        for k, v in gm.items():
            if isinstance(v, float):
                L.append(f"  {k}: {v:.6f}")
            else:
                L.append(f"  {k}: {v}")
    L.append("")
    L.append("Notes")
    L.append("-----")
    for note in report.notes:
        L.append(f"- {note}")
    Path(path).write_text("\n".join(L), encoding="utf-8")

