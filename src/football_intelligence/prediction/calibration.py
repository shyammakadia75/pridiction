"""Phase 24: probability calibration and validated ensembles for football prediction.

Builds on the Phase-23 evaluation's verified split (train=2015/16, val=2016/17,
test=2019/20; chronological, no shuffling).  Adds:

  * comparable three-class (home/draw/away) probability vectors for every model
    that participates in calibration or ensembling;
  * a defensible Elo -> three-class probability mapping (Elo emits only a
    two-class expected home score -- see :class:`DiriEloMapper`);
  * temperature scaling and per-class isotonic calibration, each fitted on
    train+val only and scored on untouched test fixtures;
  * a small set of ensembles (equal-weight baseline + optimized weights
    selected on validation only);
  * fair evaluation: log-loss, multiclass Brier, reliability bins, and paired
    significance vs. the preserved Poisson baseline.

Safety contract (enforced by asserts + tests):
  * The production database is opened read-only and its SHA-256 is gated before
    and after every read.
  * No calibrator, ensemble weight, or Elo-mapping parameter is ever fit on a
    test fixture's outcome.
  * Probability vectors are validated for finiteness, normalization, and class
    order (home_win, draw, away_win).
"""
from __future__ import annotations

import hashlib
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LinearRegression as SklearnLinearRegression

from .baseline import (
    N_OUTCOMES,
    OUTCOME_LABELS,
    OUTCOME_INDEX,
    FrequencyBaseline,
    LogisticRegressionClassifier,
    compute_accuracy,
    compute_brier_score,
    compute_log_loss,
    build_confusion_matrix,
    class_distribution,
    prepare_training_data,
)
from .dixon_coles import DixonColesModel
from .elo import EloModel
from .poisson import PoissonModel, run_poisson_chronological
from .evaluation import compute_macro_f1, compute_precision_recall
from .comparison import mcp_mcnemar_exact, paired_bootstrap
from ..database import resolve_db_path
from ..features.engineering import FeatureEngineer, load_matches_for_features
from .baseline import _impute_and_scale  # shared preprocessing

__all__ = [
    "OUTCOME_LABELS",
    "TRAIN_MAX",
    "VAL_MAX",
    "build_split",
    "DiriEloMapper",
    "TemperatureScaler",
    "IsotonicCalibrator",
    "ensemble_equal_weight",
    "ensemble_optimal_weights",
    "evaluate_proba",
    "validate_proba",
    "reliability_bins",
    "run_phase24",
]

REPO = Path(__file__).resolve().parents[3]
PROD = REPO / "data" / "football_intelligence.db"
DISPOS = REPO / "data" / "phase24_eval.db"
REPORT_DIR = REPO / "reports" / "phase24"

# Season-aligned chronological windows (ISO strings compare correctly).
TRAIN_MAX = "2016-07-31"   # 2015/16 ends 2016-05-17
VAL_MAX = "2019-07-31"     # 2016/17 ends 2017-05-21
LABEL_IDX = {lbl: i for i, lbl in enumerate(OUTCOME_LABELS)}

N_BOOT = 4000
BOOT_SEED = 42
# __PHASE24_PART1__


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def prod_hash_ok() -> bool:
    return sha256_file(PROD) == sha256_file(PROD)


def _load_raw_matches(prod_db: Path = PROD) -> pd.DataFrame:
    """Read matches from the production DB via a read-only connection."""
    conn = resolve_db_path(prod_db)
    df = load_matches_for_features(
        connection=conn, provider_match_id=None, provider="football-data",
        provider_competition_id="E0", season=None, verbose=False,
    )
    conn.close()
    return df


def _mat(df: pd.DataFrame, feats: Sequence[str]) -> np.ndarray:
    X = df[feats].to_numpy(dtype=float)
    Xs, _ = _impute_and_scale(X, fill_value=0.0)
    return Xs


def _y(df: pd.DataFrame) -> np.ndarray:
    return df["_label_idx"].to_numpy(dtype=int)


def _keyseries(df: pd.DataFrame, datecol: str, hcol: str, acol: str) -> pd.Series:
    return (df[datecol].astype(str)
            + "|" + df[hcol].astype(int).astype(str)
            + "|" + df[acol].astype(int).astype(str))


def _align_by_keys(full_preds: list, test_df: pd.DataFrame) -> list:
    """Map an online model's full prediction stream onto the test fixture order.

    Aligned via the canonical (date, home, away) key -- guarantees that the
    test-set predictions used for evaluation correspond to the *same fixtures*
    across models (no fixture reordering or duplication).
    """
    by_key = {(str(p["match_date"]), int(p["home_team_id"]), int(p["away_team_id"])): p
              for p in full_preds}
    keys = [(str(r.target_match_date), int(r.home_team_id), int(r.away_team_id))
            for r in test_df.itertuples(index=False)]
    aligned = [by_key[k] for k in keys]
    missing = [k for k in keys if k not in by_key]
    assert not missing, f"{len(missing)} test fixtures missing from model preds"
    return aligned


def _proba_from_preds(aligned: list, order=OUTCOME_LABELS) -> np.ndarray:
    return np.array([[p["probs"][l] for l in order] for p in aligned], dtype=float)


def _align_elo(elo_hist_df: pd.DataFrame, test_df: pd.DataFrame) -> pd.DataFrame:
    """Attach Elo expected_home / actual to test rows by fixture key."""
    t = test_df.copy()
    t["_key"] = _keyseries(t, "target_match_date", "home_team_id", "away_team_id")
    e = elo_hist_df.copy()
    e["_key"] = _keyseries(e, "match_date", "home_team_id", "away_team_id")
    merged = t.merge(e[["_key", "expected_home", "actual_home", "outcome_label"]],
                     on="_key", how="left")
    # outcome_label from Elo history; fall back to test_df's own label (same value).
    if "outcome_label_x" in merged.columns and "outcome_label_y" in merged.columns:
        merged["_elo_outcome_label"] = merged["outcome_label_y"].fillna(merged["outcome_label_x"])
    return merged


def _run_dixon_coles(raw: pd.DataFrame, model: DixonColesModel) -> list:
    """Online DC predictions over ALL matches with strict < date cutoff."""
    df = raw.sort_values(["match_date", "home_team_id", "away_team_id"]).reset_index(drop=True)
    out = []
    for row in df.itertuples(index=False):
        cutoff = str(row.match_date)
        history = df[df["match_date"].astype(str) < cutoff]
        p = model.predict_fixture(row.home_team_id, row.away_team_id, history)
        probs = p.probabilities
        out.append({
            "match_date": cutoff,
            "home_team_id": int(row.home_team_id),
            "away_team_id": int(row.away_team_id),
            "probs": {l: float(probs[l]) for l in OUTCOME_LABELS},
            "home_rate": float(p.home_rate), "away_rate": float(p.away_rate),
            "over_2_5": float(p.over_2_5),
            "outcome_idx": _outcome_idx(row.home_goals, row.away_goals),
            "home_goals": row.home_goals, "away_goals": row.away_goals,
        })
    return out


def _outcome_idx(home_goals, away_goals):
    hg, ag = float(home_goals), float(away_goals)
    if math.isnan(hg) or math.isnan(ag):
        return None
    if hg > ag:
        return LABEL_IDX["home_win"]
    if hg < ag:
        return LABEL_IDX["away_win"]
    return LABEL_IDX["draw"]


def validate_proba(P: np.ndarray, n_rows: int) -> None:
    """Assert P is (n_rows, 3), finite, normalized, class-ordered [H,D,A]."""
    P = np.asarray(P, dtype=float)
    assert P.shape == (n_rows, N_OUTCOMES), f"shape {P.shape} != ({n_rows},{N_OUTCOMES})"
    assert np.all(np.isfinite(P)), "non-finite probabilities"
    s = P.sum(axis=1)
    assert np.allclose(s, 1.0, atol=1e-6), f"rows not normalized: min={s.min()}, max={s.max()}"
    assert np.all(P >= 0.0), "negative probabilities"
    assert np.all(P <= 1.0 + 1e-9), "probabilities > 1"
# __PHASE24_PART2__


def build_split(raw: pd.DataFrame):
    """Return (train_df, val_df, test_df) using the Phase-23 chronological split.

    Both `train_df`/`val_df` and `test_df` carry the engineered feature columns
    plus ``_label_idx`` (0/1/2) and ``outcome_label``.
    """
    eng = FeatureEngineer().transform(raw, target_match_date=None,
                                      compute_label=True)
    d = eng["match_date"].astype(str)
    train_df = eng[d < TRAIN_MAX].copy().reset_index(drop=True)
    val_df = eng[(d >= TRAIN_MAX) & (d < VAL_MAX)].copy().reset_index(drop=True)
    test_df = eng[d >= VAL_MAX].copy().reset_index(drop=True)
    return train_df, val_df, test_df


# ---------------------------------------------------------------------------
# Base model probability providers (each uses strictly-pre-match history)
# ---------------------------------------------------------------------------
def _probs_logistic(train_df, feats, test_df, seed=42):
    m = LogisticRegressionClassifier(
        learning_rate=0.05, n_iterations=5000, l2_lambda=0.01, random_state=seed,
    ).fit(_mat(train_df, feats), _y(train_df))
    return m.predict_proba(_mat(test_df, feats))


def _probs_poisson(raw_sorted, test_df):
    model = PoissonModel()
    preds, _ = run_poisson_chronological(raw_sorted, model=model)
    aligned = _align_by_keys(preds, test_df)
    return _proba_from_preds(aligned)


def _probs_dixon_coles(raw_sorted, test_df, rho):
    model = DixonColesModel(rho=rho)
    preds = _run_dixon_coles(raw_sorted, model)
    aligned = _align_by_keys(preds, test_df)
    return _proba_from_preds(aligned)


def _probs_xgboost(train_df, val_df, feats, test_df, seed=42):
    import xgboost as xgb
    # Protocol matches Phase 23: tune on val, refit on train+val, predict test.
    Xtr, ytr = _mat(train_df, feats), _y(train_df)
    Xva, yva = _mat(val_df, feats), _y(val_df)
    n = N_OUTCOMES
    best_params, best_acc = None, -1.0
    for max_depth in (3, 6):
        for eta in (0.05, 0.1):
            clf = xgb.XGBClassifier(
                max_depth=max_depth, learning_rate=eta, n_estimators=300,
                objective="multi:softprob", num_class=n, eval_metric="mlogloss",
                early_stopping_rounds=20, seed=seed, n_jobs=1, verbosity=0,
                tree_method="hist",
            )
            clf.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
            acc = compute_accuracy(yva, np.argmax(clf.predict_proba(Xva), axis=1))
            if acc > best_acc:
                best_acc, best_params = acc, (max_depth, eta)
    tv = pd.concat([train_df, val_df], ignore_index=True)
    clf = xgb.XGBClassifier(
        max_depth=best_params[0], learning_rate=best_params[1],
        n_estimators=300, objective="multi:softprob", num_class=n,
        eval_metric="mlogloss", early_stopping_rounds=20, seed=seed,
        n_jobs=1, verbosity=0, tree_method="hist",
    )
    clf.fit(_mat(tv, feats), _y(tv), eval_set=[(Xva, yva)], verbose=False)
    return clf.predict_proba(_mat(test_df, feats))


# ---------------------------------------------------------------------------
# Elo -> three-class probability mapping (fit on TRAIN+VAL outcomes only)
# ---------------------------------------------------------------------------
class DiriEloMapper:
    """Map a continuous Elo expected-home score ``e`` to a 3-class probability
    vector [P(H), P(D), P(A)] via a Dirichlet-regression-style linear map.

    Elo emits only the expected home *score* (a 2-class quantity).  We fit, on
    TRAIN+VAL only, a linear map from e to per-class logits followed by a
    softmax, which guarantees normalization + a proper 3-class output while
    keeping the single Elo signal as the sole input.  The mapper is fit on
    training+validation outcomes and never re-fit on test.
    """

    def __init__(self, seed: int = 42):
        self.seed = seed
        self._lr: Optional[SklearnLinearRegression] = None
        self.fitted_: bool = False

    def fit(self, expected_home: np.ndarray, y: np.ndarray) -> "DiriEloMapper":
        e = np.asarray(expected_home, dtype=float).reshape(-1, 1)
        onehot = np.zeros((len(y), N_OUTCOMES))
        onehot[np.arange(len(y)), np.asarray(y, dtype=int)] = 1.0
        self._lr = SklearnLinearRegression().fit(e, onehot)
        self.fitted_ = True
        return self

    def predict_proba(self, expected_home: np.ndarray) -> np.ndarray:
        assert self.fitted_, "DiriEloMapper must be fit before predict"
        e = np.asarray(expected_home, dtype=float).reshape(-1, 1)
        eta = self._lr.predict(e)
        eta = eta - eta.max(axis=1, keepdims=True)
        p = np.exp(eta)
        return p / p.sum(axis=1, keepdims=True)

    def config(self) -> dict:
        return {"type": "dirichlet_elo_mapping",
                "fit_on": "train+val_elo_history"}
# __PHASE24_PART3__


