"""Phase 23: Dixon-Coles low-score correction on top of the independent Poisson model.

The classic Dixon & Coles (1989) modification relaxes the independence assumption
only for low-scoring outcomes (0-0, 1-0, 0-1, 1-1), applying a single scalar
parameter ``rho`` that adjusts the joint probability of those four cells while
leaving all higher-score cells identical to the independent-Poisson product.

This module is a thin, leakage-safe wrapper over the Phase-20 Poisson engine:
team attack/defence strengths (and hence the marginal Poisson rates) are
re-estimated from **strictly pre-match** history per fixture via
:func:`football_intelligence.prediction.poisson.estimate_team_strengths`, and a
single global ``rho`` is fit by maximum likelihood on the training window only.
At prediction time the fitted ``rho`` (never re-tuned on test) is applied to the
same independence grid, then re-normalised -- exactly the "low-score correction
to the independent Poisson" requested for Phase 23 improvement #3.

Bivariate modification (Dixon & Coles, 1989), applied multiplicatively to the
independent-Poisson cell probability ``p_lambda(i) * p_mu(j)``::

    p(0,0) *= (1 - rho * lambda * mu)
    p(1,0) *= (1 + rho * mu)
    p(0,1) *= (1 + rho * lambda)
    p(1,1) *= (1 - rho)

All other cells ((i,j) not in the set above) are unchanged.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.stats import poisson as _scipy_poisson

from .poisson import (
    DEFAULT_MAX_GOALS,
    DEFAULT_MIN_HISTORY,
    DEFAULT_PRIOR_STRENGTH,
    DEFAULT_RATE_FLOOR,
    OUTCOME_LABELS,
    OUTCOME_INDEX,  # noqa: F401  (re-export parity with poisson)
    estimate_team_strengths,
    over_under_probability,
    score_probability_grid,
)

__all__ = [
    "DEFAULT_RHO",
    "DixonColesFixturePrediction",
    "DixonColesModel",
]

#: Neutral default rho (reduces to independent Poisson at rho -> 0).
DEFAULT_RHO: float = 0.0


def _dc_grid(home_rate: float, away_rate: float, rho: float,
             max_goals: int = DEFAULT_MAX_GOALS) -> np.ndarray:
    """Return the Dixon-Coles-adjusted (max_goals+1, max_goals+1) scoreline grid.

    The grid is renormalised over the truncated support so that it sums to 1
    (matching the convention used by ``score_probability_grid``).
    """
    lam = max(float(home_rate), DEFAULT_RATE_FLOOR)
    mu = max(float(away_rate), DEFAULT_RATE_FLOOR)
    rho = float(rho)
    grid = score_probability_grid(lam, mu, max_goals).copy()
    base = score_probability_grid(lam, mu, max_goals)
    factors = {
        (0, 0): 1.0 - rho * lam * mu,
        (1, 0): 1.0 + rho * mu,
        (0, 1): 1.0 + rho * lam,
        (1, 1): 1.0 - rho,
    }
    for (i, j), f in factors.items():
        if i <= max_goals and j <= max_goals:
            grid[i, j] = base[i, j] * f
    s = grid.sum()
    if s > 0:
        grid = grid / s
    return grid


def _dc_cell_logpmf(i: int, j: int, home_rate: float, away_rate: float,
                    rho: float) -> float:
    """Log probability of observed scoreline (i,j) under Dixon-Coles."""
    lam = max(float(home_rate), DEFAULT_RATE_FLOOR)
    mu = max(float(away_rate), DEFAULT_RATE_FLOOR)
    base_log = _scipy_poisson.logpmf(i, lam) + _scipy_poisson.logpmf(j, mu)
    if (i, j) == (0, 0):
        f = 1.0 - rho * lam * mu
    elif (i, j) == (1, 0):
        f = 1.0 + rho * mu
    elif (i, j) == (0, 1):
        f = 1.0 + rho * lam
    elif (i, j) == (1, 1):
        f = 1.0 - rho
    else:
        f = 1.0
    f = max(f, 1e-12)
    return float(base_log + np.log(f))


def _dc_neg_loglik(rho: float, fixtures: list) -> float:
    """Negative log-likelihood of observed (home_goals, away_goals) fixtures."""
    total = 0.0
    for (hg, ag, lam, mu) in fixtures:
        lp = _dc_cell_logpmf(hg, ag, lam, mu, rho)
        if not np.isfinite(lp):
            return 1e12
        total += lp
    return -total


def _safe_int_goals(v) -> Optional[int]:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    if f != f:  # NaN
        return None
        return int(round(f))


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


class DixonColesModel:
    """Dixon-Coles corrected independent-Poisson model.

    Parameters
    ----------
    rho :
        Fixed rho at construction turns the model into a pure online predictor
        (no fitting).  ``None`` requires an explicit ``fit_rho`` call first.
    rho_bounds :
        Optimisation bounds for the global ``rho`` when fitting on the training
        window (|rho| is bounded to keep the bivariate cells non-negative).
    max_goals, min_history, prior_strength :
        Forwarded to Poisson strength estimation.
    """

    def __init__(
        self,
        rho: Optional[float] = None,
        rho_bounds: tuple = (-0.9, 0.9),
        max_goals: int = DEFAULT_MAX_GOALS,
        min_history: int = DEFAULT_MIN_HISTORY,
        prior_strength: float = DEFAULT_PRIOR_STRENGTH,
    ) -> None:
        self._rho_fixed = rho
        self.rho_bounds = rho_bounds
        self.max_goals = int(max_goals)
        self.min_history = int(min_history)
        self.prior_strength = float(prior_strength)
        self.rho_: float = float(rho) if rho is not None else 0.0
        self._is_fitted: bool = rho is not None

    # ------------------------------------------------------------------ fit
    def fit_rho(self, matches: pd.DataFrame, date_col: str = "match_date") -> "DixonColesModel":
        """Fit the global ``rho`` by MLE on a training window.

        Each fixture's (lambda_h, mu_a) rates are estimated from matches
        strictly earlier than that fixture's date (leakage-safe, same-day-safe).
        ``rho`` is a single scalar shared across all fixtures -- fit on the
        training window only and never re-tuned at prediction time.
        """
        df = matches.sort_values(date_col).reset_index(drop=True)
        fixtures: list = []
        for _, row in df.iterrows():
            target_date = str(row[date_col])
            history = df[df[date_col].astype(str) < target_date]
            hg = _safe_int_goals(row["home_goals"])
            ag = _safe_int_goals(row["away_goals"])
            if hg is None or ag is None:
                continue
            strengths = estimate_team_strengths(
                history,
                home_team_id=row["home_team_id"],
                away_team_id=row["away_team_id"],
                min_history=self.min_history,
                prior_strength=self.prior_strength,
            )
            fixtures.append((hg, ag, float(strengths.home_rate), float(strengths.away_rate)))
        if not fixtures:
            self.rho_ = 0.0
            self._is_fitted = True
            return self
        res = minimize_scalar(
            _dc_neg_loglik,
            bounds=self.rho_bounds,
            method="bounded",
            args=(fixtures,),
        )
        self.rho_ = float(res.x) if np.isfinite(res.x) else 0.0
        self._is_fitted = True
        return self

    # ----------------------------------------------------------- predict
    def predict_fixture(
        self,
        home_team_id,
        away_team_id,
        history: pd.DataFrame,
        rho: Optional[float] = None,
    ) -> "DixonColesFixturePrediction":
        """Predict one fixture from strictly-pre-match ``history``.

        ``rho`` defaults to the fitted ``self.rho_`` (fixed, non-leaking) and is
        never re-estimated from the target fixture or its result.
        """
        if not self._is_fitted and rho is None:
            raise RuntimeError("Model must be fitted before calling predict_fixture")
        strengths = estimate_team_strengths(
            history,
            home_team_id=home_team_id,
            away_team_id=away_team_id,
            min_history=self.min_history,
            prior_strength=self.prior_strength,
        )
        lam = max(float(strengths.home_rate), DEFAULT_RATE_FLOOR)
        mu = max(float(strengths.away_rate), DEFAULT_RATE_FLOOR)
        used_rho = self.rho_ if rho is None else float(rho)
        grid = _dc_grid(lam, mu, used_rho, self.max_goals)
        hw = float(np.sum(np.tril(grid, -1)))   # home wins: i > j
        dr = float(np.trace(grid))              # draws: i == j
        aw = float(np.sum(np.triu(grid, 1)))    # away wins: i < j
        norm = hw + dr + aw
        if norm > 0:
            hw, dr, aw = hw / norm, dr / norm, aw / norm
        over, under = over_under_probability(lam, mu, line=2.5, max_goals=self.max_goals)
        idx = np.unravel_index(np.argmax(grid), grid.shape)
        most_probable = (int(idx[0]), int(idx[1]), float(grid[idx]))
        return DixonColesFixturePrediction(
            home_team_id=home_team_id,
            away_team_id=away_team_id,
            home_rate=lam,
            away_rate=mu,
            rho=used_rho,
            home_win_prob=hw,
            draw_prob=dr,
            away_win_prob=aw,
            over_2_5=over,
            under_2_5=under,
            score_grid=grid,
                        most_probable_scoreline=most_probable,
            notes=strengths.notes,
        )


@dataclass
class DixonColesFixturePrediction:
    """Full Dixon-Coles prediction for a single fixture."""

    home_team_id: object
    away_team_id: object
    home_rate: float
    away_rate: float
    rho: float
    home_win_prob: float
    draw_prob: float
    away_win_prob: float
    over_2_5: float
    under_2_5: float
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
            "home_rate": float(self.home_rate),
            "away_rate": float(self.away_rate),
            "rho": float(self.rho),
            "home_win_prob": float(self.home_win_prob),
            "draw_prob": float(self.draw_prob),
            "away_win_prob": float(self.away_win_prob),
            "over_2_5": float(self.over_2_5),
            "under_2_5": float(self.under_2_5),
            "most_probable_scoreline": {
                "home_goals": self.most_probable_scoreline[0],
                "away_goals": self.most_probable_scoreline[1],
                "probability": self.most_probable_scoreline[2],
            },
        }


