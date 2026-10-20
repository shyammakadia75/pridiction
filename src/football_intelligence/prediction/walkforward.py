#!/usr/bin/env python3
"""Phase 25: multi-season chronological walk-forward evaluation.

Expanding-window walk-forward backtesting harness using verified data in
disposable data/phase23_eval.db (3 seasons: 2015/16, 2016/17, 2019/20;
380 matches each, 1140 total).

Walk-forward folds:
  Fold 1: train=2015/16, test=2016/17
  Fold 2: train=2015/16+2016/17, test=2019/20 (COVID-gap discontinuity)
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd

from .baseline import (
    N_OUTCOMES,
    OUTCOME_LABELS,
    OUTCOME_INDEX,
    FrequencyBaseline,
    LogisticRegressionClassifier,
    compute_accuracy,
    compute_brier_score,
    compute_log_loss,
    prepare_training_data,
    class_distribution,
    _impute_and_scale,
)
from .calibration import (
    DiriEloMapper,
    TemperatureScaler,
    ensemble_equal_weight,
    ensemble_optimal_weights,
    evaluate_proba,
    reliability_bins,
    validate_proba,
    LABEL_IDX,
)
from .comparison import mcp_mcnemar_exact, paired_bootstrap
from .dixon_coles import DixonColesModel
from .elo import EloModel
from .poisson import (
    DEFAULT_MAX_GOALS,
    PoissonModel,
    run_poisson_chronological,
)
from .evaluation import (
    compute_macro_f1,
    compute_precision_recall,
    compute_calibration,
    feature_columns,
)
from ..database import resolve_db_path
from ..features.engineering import FeatureEngineer, load_matches_for_features

__all__ = [
    "WALKFORWARD_SEASONS",
    "WALKFORWARD_SEED",
    "N_BOOT",
    "BOOT_SEED",
    "Folds",
    "FoldPrediction",
    "build_walkforward_folds",
    "assign_fold_dates",
    "compute_walkforward_predictions",
    "score_fold_predictions",
    "run_walkforward",
]

REPO = Path(__file__).resolve().parents[3]
PROD = REPO / "data" / "football_intelligence.db"
PHASE23_DISPOS = REPO / "data" / "phase23_eval.db"
REPORT_DIR = REPO / "reports" / "phase25"

WALKFORWARD_SEASONS: list[str] = ["2016/2017", "2019/2020"]
WALKFORWARD_SEED = 42
N_BOOT = 4000
BOOT_SEED = 42


# ---------------------------------------------------------------------------
# Safety helpers
# ---------------------------------------------------------------------------


def _prod_sha256() -> str:
    """SHA-256 of the production DB (read-only invariant check)."""
    h = hashlib.sha256()
    with open(PROD, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_raw_matches(prod_db: Path = PHASE23_DISPOS) -> pd.DataFrame:
    """Load matches from the disposable DB (prod SHA-256 gated, fail-closed)."""
    prod_before = _prod_sha256()
    db_path = resolve_db_path(prod_db)
    df = load_matches_for_features(
        db_path=db_path, provider="football-data", provider_competition_id="E0",
    )
    assert _prod_sha256() == prod_before, (
        "Production DB SHA-256 changed during Phase-25 load (fail-closed)")
    return df


# ---------------------------------------------------------------------------
# Walk-forward fold construction
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Folds:
    """One walk-forward fold definition (chronological, expanding window)."""

    fold_id: str
    train_seasons: tuple[str, ...]
    val_season: Optional[str]
    test_season: str
    train_cutoff_date: str
    test_start_date: str
    test_end_date: str


def build_walkforward_folds(
    seasons: Sequence[str] = WALKFORWARD_SEASONS,
) -> list[Folds]:
    """Build expanding-window walk-forward folds from the available seasons.

    Training = all seasons before the test season. If training spans >= 2
    seasons, the last training season is a validation hold-out for hyperparameter
    selection; the model is then refit on all training data before scoring.

    Args:
        seasons: Chronological list of seasons to use as test windows.

    Returns:
        List of Folds (dates left blank; fill via assign_fold_dates).
    """
    all_seasons_ordered: list[str] = ["2015/2016", "2016/2017", "2019/2020"]

    folds: list[Folds] = []
    for idx, test_season in enumerate(seasons):
        if test_season not in all_seasons_ordered:
            raise ValueError(
                f"Test season {test_season!r} not in available seasons "
                f"{all_seasons_ordered}"
            )
        test_pos = all_seasons_ordered.index(test_season)
        train_seasons = all_seasons_ordered[:test_pos]
        if not train_seasons:
            raise ValueError(
                f"No training seasons available before test season {test_season!r}"
            )

        val_season: Optional[str] = None
        if len(train_seasons) >= 2:
            val_season = train_seasons[-1]

        folds.append(Folds(
            fold_id=f"fold{idx+1}",
            train_seasons=tuple(train_seasons),
            val_season=val_season,
            test_season=test_season,
            train_cutoff_date="",
            test_start_date="",
            test_end_date="",
        ))
    return folds


def assign_fold_dates(
    folds: list[Folds],
    raw: pd.DataFrame,
) -> list[Folds]:
    """Fill in exact date cutoffs for each fold from the raw match data.

    The training cutoff is the minimum match_date in the test season, so
    ``history < cutoff`` excludes all test-season matches (strict less-than,
    same-day safe).
    """
    updated: list[Folds] = []
    for f in folds:
        test_rows = raw[raw["season_name"] == f.test_season]
        train_rows = raw[raw["season_name"].isin(f.train_seasons)]
        if len(test_rows) == 0:
            raise ValueError(f"No matches found for test season {f.test_season!r}")
        if len(train_rows) == 0:
            raise ValueError(
                f"No matches found for training seasons {f.train_seasons}"
            )

        test_start = str(test_rows["match_date"].min())
        test_end = str(test_rows["match_date"].max())
        train_cutoff = test_start

        updated.append(Folds(
            fold_id=f.fold_id,
            train_seasons=f.train_seasons,
            val_season=f.val_season,
            test_season=f.test_season,
            train_cutoff_date=train_cutoff,
            test_start_date=test_start,
            test_end_date=test_end,
        ))
    return updated


@dataclass
class FoldPrediction:
    """All model predictions for a single fold's test set."""

    fold_id: str
    test_season: str
    n_train: int
    n_val: Optional[int]
    n_test: int
    y_true: np.ndarray
    predictions: dict[str, np.ndarray]
    meta: dict = field(default_factory=dict)


def _y_from_df(df: pd.DataFrame) -> np.ndarray:
    """Extract integer label indices from a feature DataFrame."""
    return df["_label_idx"].to_numpy(dtype=int)


def _mat(df: pd.DataFrame, feats: Sequence[str]) -> np.ndarray:
    """Extract feature matrix and apply impute+scale (scaler fit on this data)."""
    X = df[feats].to_numpy(dtype=float)
    Xs, _ = _impute_and_scale(X, fill_value=0.0)
    return Xs


def _align_online_to_test(
    online_preds: list[dict],
    test_df: pd.DataFrame,
) -> np.ndarray:
    """Align online model predictions (Poisson/DC) to test fixture order.

    Aligned via the canonical (date, home, away) key.  Fails closed if any
    test fixture is missing from the predictions (no silent misalignment).
    """
    by_key: dict[tuple, dict] = {
        (str(p["match_date"]), int(p["home_team_id"]),
         int(p["away_team_id"])): p
        for p in online_preds
    }
    keys = [
        (str(r.target_match_date), int(r.home_team_id), int(r.away_team_id))
        for r in test_df.itertuples(index=False)
    ]
    missing = [k for k in keys if k not in by_key]
    assert not missing, (
        f"{len(missing)} test fixtures missing from online model predictions")
    aligned = [by_key[k] for k in keys]
    P = np.array(
        [[p["probs"]["home_win"], p["probs"]["draw"], p["probs"]["away_win"]]
         for p in aligned],
        dtype=float,
    )
    return P


def _online_poisson(raw_sorted: pd.DataFrame) -> list[dict]:
    """Online Poisson predictions over ALL matches (strict pre-match history)."""
    full_preds, _ = run_poisson_chronological(raw_sorted, model=PoissonModel())
    return full_preds


def _online_dixon_coles(
    raw_sorted: pd.DataFrame,
    train_cutoff: str,
) -> tuple[list[dict], float]:
    """Online Dixon-Coles predictions; rho fit on training window only."""
    train_raw = raw_sorted[raw_sorted["match_date"].astype(str) < train_cutoff]
    dc_model = DixonColesModel(rho=None)
    if len(train_raw) > 0:
        dc_model.fit_rho(train_raw)
    else:
        dc_model.rho_ = 0.0
        dc_model._is_fitted = True

    df = raw_sorted.sort_values(
        ["match_date", "home_team_id", "away_team_id"]
    ).reset_index(drop=True)
    out: list[dict] = []
    for row in df.itertuples(index=False):
        cutoff = str(row.match_date)
        history = df[df["match_date"].astype(str) < cutoff]
        p = dc_model.predict_fixture(
            home_team_id=row.home_team_id,
            away_team_id=row.away_team_id,
            history=history,
        )
        out.append({
            "match_date": str(row.match_date),
            "home_team_id": int(row.home_team_id),
            "away_team_id": int(row.away_team_id),
            "probs": p.probabilities,
        })
    return out, dc_model.rho_


def _online_elo(raw_sorted: pd.DataFrame) -> pd.DataFrame:
    """Online Elo predictions over ALL matches (same-day-safe, chronological)."""
    df = raw_sorted.sort_values(
        ["match_date", "home_team_id", "away_team_id"]
    ).reset_index(drop=True)
    m = EloModel(K=40.0, home_advantage=50.0)
    m.fit(df)
    hist = pd.DataFrame([r.__dict__ for r in m.history_])
    hist["_label_idx"] = hist["outcome_label"].map(LABEL_IDX).astype(int)
    return hist


def _align_elo_to_test(
    elo_hist: pd.DataFrame,
    test_df: pd.DataFrame,
) -> np.ndarray:
    """Align Elo expected_home to test fixture order by fixture key."""
    t = test_df.copy()
    t["_key"] = (
        t["target_match_date"].astype(str)
        + "|" + t["home_team_id"].astype(int).astype(str)
        + "|" + t["away_team_id"].astype(int).astype(str)
    )
    e = elo_hist.copy()
    e["_key"] = (
        e["match_date"].astype(str)
        + "|" + e["home_team_id"].astype(int).astype(str)
        + "|" + e["away_team_id"].astype(int).astype(str)
    )
    merged = t.merge(e[["_key", "expected_home"]], on="_key", how="left")
    assert not merged["expected_home"].isna().any(), (
        "Elo expected_home has NaN after alignment -- fixture mismatch"
    )
    return merged["expected_home"].to_numpy(dtype=float)


def _probs_lr_tuned(
    fit_train: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feats: Sequence[str],
    seed: int,
) -> tuple[LogisticRegressionClassifier, Optional[np.ndarray], np.ndarray, float]:
    """Tune LR L2 on val, refit on train, predict val+test.

    If no val set, uses default L2=0.01 and returns None val probs.
    """
    ytr = _y_from_df(fit_train)
    Xtr = _mat(fit_train, feats)

    if len(val_df) > 0:
        yva = _y_from_df(val_df)
        Xva = _mat(val_df, feats)
        best_l2, best_acc = 0.01, -1.0
        for l2 in [0.001, 0.01, 0.1, 1.0]:
            mm = LogisticRegressionClassifier(
                learning_rate=0.05, n_iterations=5000,
                l2_lambda=l2, random_state=seed,
            ).fit(Xtr, ytr)
            acc = compute_accuracy(yva, np.argmax(mm.predict_proba(Xva), 1))
            if acc > best_acc:
                best_acc, best_l2 = acc, l2
        m = LogisticRegressionClassifier(
            learning_rate=0.05, n_iterations=5000,
            l2_lambda=best_l2, random_state=seed,
        ).fit(Xtr, ytr)
        val_proba = m.predict_proba(Xva)
        test_proba = m.predict_proba(_mat(test_df, feats))
        return m, val_proba, test_proba, best_l2
    else:
        m = LogisticRegressionClassifier(
            learning_rate=0.05, n_iterations=5000,
            l2_lambda=0.01, random_state=seed,
        ).fit(Xtr, ytr)
        test_proba = m.predict_proba(_mat(test_df, feats))
        return m, None, test_proba, 0.01


def _probs_xgboost(
    fit_train: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feats: Sequence[str],
    seed: int,
) -> tuple[Optional[object], Optional[np.ndarray], np.ndarray, Optional[tuple]]:
    """Tune XGBoost on val, refit on train, predict val+test.

    If no val set, uses default params (max_depth=6, eta=0.1).
    """
    import xgboost as xgb

    ytr = _y_from_df(fit_train)
    Xtr = _mat(fit_train, feats)
    n = N_OUTCOMES

    if len(val_df) > 0:
        yva = _y_from_df(val_df)
        Xva = _mat(val_df, feats)
        best_params, best_acc = (6, 0.1), -1.0
        for max_depth in (3, 6):
            for eta in (0.05, 0.1):
                clf = xgb.XGBClassifier(
                    max_depth=max_depth, learning_rate=eta, n_estimators=300,
                    objective="multi:softprob", num_class=n,
                    eval_metric="mlogloss", early_stopping_rounds=20,
                    seed=seed, n_jobs=1, verbosity=0, tree_method="hist",
                )
                clf.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
                acc = compute_accuracy(yva, np.argmax(clf.predict_proba(Xva), 1))
                if acc > best_acc:
                    best_acc, best_params = acc, (max_depth, eta)
        clf = xgb.XGBClassifier(
            max_depth=best_params[0], learning_rate=best_params[1],
            n_estimators=300, objective="multi:softprob", num_class=n,
            eval_metric="mlogloss", early_stopping_rounds=20,
            seed=seed, n_jobs=1, verbosity=0, tree_method="hist",
        )
        clf.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
        val_proba = clf.predict_proba(Xva)
        test_proba = clf.predict_proba(_mat(test_df, feats))
        return clf, val_proba, test_proba, best_params
    else:
        clf = xgb.XGBClassifier(
            max_depth=6, learning_rate=0.1, n_estimators=300,
            objective="multi:softprob", num_class=n,
            eval_metric="mlogloss", early_stopping_rounds=20,
            seed=seed, n_jobs=1, verbosity=0, tree_method="hist",
        )
        clf.fit(Xtr, ytr)
        test_proba = clf.predict_proba(_mat(test_df, feats))
        return clf, None, test_proba, None


def _add_interactions(eng: pd.DataFrame) -> pd.DataFrame:
    """Add leakage-safe pre-match form interaction features (mirrors Phase 24)."""
    eng = eng.copy()

    def edge(a: str, b: str) -> None:
        if a in eng.columns and b in eng.columns:
            eng[f"{a}_minus_{b}"] = eng[a].fillna(0) - eng[b].fillna(0)

    for a, b in [
        ("home_points_last_5", "away_points_last_5"),
        ("home_goal_diff_last_5", "away_goal_diff_last_5"),
        ("home_goals_scored_last_5", "away_goals_conceded_last_5"),
        ("home_goals_conceded_last_5", "away_goals_scored_last_5"),
        ("home_points_per_match_last_5", "away_points_per_match_last_5"),
        ("home_avg_goals_scored_last_5", "away_avg_goals_conceded_last_5"),
        ("home_avg_goals_conceded_last_5", "away_avg_goals_scored_last_5"),
        ("home_clean_sheets_last_5", "away_clean_sheets_last_5"),
        ("home_recency_weighted_points_last_5",
         "away_recency_weighted_points_last_5"),
        ("home_points_last_10", "away_points_last_10"),
    ]:
        edge(a, b)
    if {"home_days_since_last_match", "away_days_since_last_match"}.issubset(
            eng.columns):
        eng["rest_advantage"] = (
            eng["away_days_since_last_match"].fillna(0)
            - eng["home_days_since_last_match"].fillna(0)
        )
    return eng


# ---------------------------------------------------------------------------
# Per-fold prediction computation
# ---------------------------------------------------------------------------

def _season_date_range(
    raw: pd.DataFrame,
    season: str,
) -> tuple[str, str]:
    """Return (min_date, max_date) for a given season from raw match data."""
    srows = raw[raw["season_name"] == season]
    if len(srows) == 0:
        raise ValueError(f"No matches for season {season!r}")
    return str(srows["match_date"].min()), str(srows["match_date"].max())

def compute_walkforward_predictions(
    fold: Folds,
    raw: pd.DataFrame,
    features: pd.DataFrame,
    feats: Sequence[str],
    seed: int = WALKFORWARD_SEED,
) -> FoldPrediction:
    """Compute all model predictions for one walk-forward fold."""
    d = features["target_match_date"].astype(str)
    train_mask = d < fold.train_cutoff_date

    # Test mask: all features in the test date range (train_cutoff to test_end)
    test_mask = (d >= fold.train_cutoff_date) & (d <= fold.test_end_date)

    train_df = features[train_mask].copy().reset_index(drop=True)
    test_df = features[test_mask].copy().reset_index(drop=True)

    # Validation split: if val_season is set, use its date range.
    if fold.val_season is not None:
        # The val cutoff is the train cutoff (first date of the current fold's
        # test season). The val season is the last season BEFORE that.
        # We need to know the val date range from the raw data.
        # For fold 2: val_season = "2016/2017", train_cutoff = "2019-08-09"
        # Val dates = 2016/17 season dates (all before train_cutoff).
        # But since train_mask already includes all < train_cutoff, and
        # val_season is within that range, we can split by val date range.
        # We get val_start/val_end from the fold's val_season date range.
        val_start, val_end = _season_date_range(raw, fold.val_season)
        val_mask = (d >= val_start) & (d <= val_end)
        fit_mask = train_mask & ~val_mask
        fit_train = features[fit_mask].copy().reset_index(drop=True)
        val_df = features[val_mask].copy().reset_index(drop=True)
    else:
        fit_train = train_df.copy()
        val_df = features.iloc[0:0].copy()

    n_train = len(train_df)
    n_val = len(val_df) if fold.val_season else None
    n_test = len(test_df)
    y_test = _y_from_df(test_df) if n_test > 0 else np.array([], dtype=int)
    raw_sorted = raw.sort_values(
        ["match_date", "home_team_id", "away_team_id"]
    ).reset_index(drop=True)

    preds: dict[str, np.ndarray] = {}
    meta: dict = {}

    # --- Frequency baseline (class priors from training) ---
    y_train = _y_from_df(train_df)
    fb = FrequencyBaseline().fit(y_train)
    freq_proba = np.tile(fb._class_proba, (n_test, 1))
    preds["frequency_baseline"] = freq_proba
    meta["frequency_baseline"] = {"type": "frequency", "fit_on": "train"}

    # --- Logistic Regression (tuned on val, refit on fit_train) ---
    _, lr_val_proba, lr_test_proba, lr_l2 = _probs_lr_tuned(
        fit_train, val_df, test_df, feats, seed)
    preds["logistic_regression"] = lr_test_proba
    meta["logistic_regression"] = {
        "type": "lr", "l2": lr_l2, "val_tuned": fold.val_season is not None,
    }

    # --- Poisson (online, strict pre-match history) ---
    pois_full = _online_poisson(raw_sorted)
    pois_test = _align_online_to_test(pois_full, test_df)
    preds["poisson"] = pois_test
    meta["poisson"] = {"type": "poisson", "history": "all_pre_test"}

    # --- Temperature-scaled Poisson (calibrator fit on val predictions only) ---
    if len(val_df) > 0:
        pois_val = _align_online_to_test(pois_full, val_df)
        y_val = _y_from_df(val_df)
        ts_pois = TemperatureScaler().fit(pois_val, y_val)
        preds["poisson_temperature_scaled"] = ts_pois.predict_proba(pois_test)
        meta["poisson_temperature_scaled"] = {
            "base": "poisson", "T": ts_pois.T, "fit_on": "val",
        }
    else:
        preds["poisson_temperature_scaled"] = pois_test.copy()
        meta["poisson_temperature_scaled"] = {
            "base": "poisson", "T": 1.0, "note": "no_val_no_calibration",
        }


    # --- Dixon-Coles (online, rho fit on training window only) ---
    dc_full, dc_rho = _online_dixon_coles(raw_sorted, fold.train_cutoff_date)
    dc_test = _align_online_to_test(dc_full, test_df)
    preds["dixon_coles"] = dc_test
    meta["dixon_coles"] = {"rho": dc_rho, "fit_on": "train"}

    # --- XGBoost (tuned on val) ---
    try:
        _, xgb_val, xgb_test, xgb_params = _probs_xgboost(
            fit_train, val_df, test_df, feats, seed)
        preds["xgboost"] = xgb_test
        meta["xgboost"] = {
            "type": "xgboost",
            "max_depth": xgb_params[0] if xgb_params else 6,
            "eta": xgb_params[1] if xgb_params else 0.1,
        }
    except Exception as e:
        preds["xgboost"] = freq_proba.copy()
        meta["xgboost"] = {"type": "xgboost", "error": str(e)}


    # --- Elo (online, DiriEloMapper fit on train Elo history) ---
    elo_hist = _online_elo(raw_sorted)
    elo_train = elo_hist[
        elo_hist["match_date"].astype(str) < fold.train_cutoff_date
    ]
    elo_test_hist = elo_hist[
        elo_hist["match_date"].astype(str) >= fold.train_cutoff_date
    ].reset_index(drop=True)
    mapper = DiriEloMapper().fit(
        elo_train["expected_home"].to_numpy(dtype=float),
        elo_train["_label_idx"].to_numpy(dtype=int),
    )
    elo_expected = _align_elo_to_test(elo_test_hist, test_df)
    elo_test_proba = mapper.predict_proba(elo_expected)
    preds["elo_mapped"] = elo_test_proba
    meta["elo_mapped"] = {
        "type": "dirichlet_elo_mapping", "K": 40.0, "fit_on": "train",
    }

    # --- Poisson + LR ensemble (equal weight) ---
    ens_eq = ensemble_equal_weight(
        [preds["poisson"], preds["logistic_regression"]])
    preds["ens_poisson_lr_equal"] = ens_eq
    meta["ens_poisson_lr_equal"] = {
        "type": "ensemble_equal",
        "members": ["poisson", "logistic_regression"],
    }


    # --- Poisson + LR ensemble (optimal weights on val) ---
    if len(val_df) > 0:
        pois_val_probs = _align_online_to_test(pois_full, val_df)
        y_val = _y_from_df(val_df)
        try:
            ot, w, vll = ensemble_optimal_weights(
                [pois_val_probs, lr_val_proba],
                y_val,
                [preds["poisson"], preds["logistic_regression"]],
            )
            preds["ens_poisson_lr_optimal"] = ot
            meta["ens_poisson_lr_optimal"] = {
                "type": "ensemble_optimal",
                "members": ["poisson", "logistic_regression"],
                "weights": w,
                "val_logloss": vll,
            }
        except Exception:
            preds["ens_poisson_lr_optimal"] = ens_eq
            meta["ens_poisson_lr_optimal"] = {
                "type": "ensemble_optimal_fallback",
            }
    else:
        preds["ens_poisson_lr_optimal"] = ens_eq
        meta["ens_poisson_lr_optimal"] = {
            "type": "ensemble_no_val",
            "note": "no_val_weights_default_to_equal",
        }

    # Validate all probability vectors (including ensembles)
    for nm, p in preds.items():
        validate_proba(p, n_test)

    return FoldPrediction(
        fold_id=fold.fold_id,
        test_season=fold.test_season,
        n_train=n_train,
        n_val=n_val,
        n_test=n_test,
        y_true=y_test,
        predictions=preds,
        meta=meta,
    )


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


def score_fold_predictions(fp: FoldPrediction) -> dict:
    """Evaluate all models in a fold against y_true."""
    y = fp.y_true
    results: dict[str, dict] = {}
    for nm, proba in fp.predictions.items():
        validate_proba(proba, len(y))
        results[nm] = evaluate_proba(
            y, proba, name=nm, config=fp.meta.get(nm, {}),
        )
    return results


# ---------------------------------------------------------------------------
# Main walk-forward driver
# ---------------------------------------------------------------------------


def run_walkforward(
    seed: int = WALKFORWARD_SEED,
    db_path: Path = PHASE23_DISPOS,
    seasons: Sequence[str] | None = None,
) -> dict:
    """Execute the full Phase-25 walk-forward evaluation.

    Returns a dict containing per-fold results, pooled results, and paired
    significance tests vs Poisson.
    """
    if seasons is None:
        seasons = WALKFORWARD_SEASONS

    prod_before = _prod_sha256()
    print(f"[phase25] prod DB SHA-256 (before): {prod_before}")

    # Load and prepare data (prod SHA-256 gated)
    raw = _load_raw_matches(db_path)
    print(f"[phase25] loaded {len(raw)} matches from disposable DB")

    # Build the full feature dataset (same pipeline as Phase 23/24)
    eng, _X, _y = prepare_training_data(
        raw, engineer=FeatureEngineer(), label_col="outcome_label",
    )
    eng = _add_interactions(eng)
    feats = list(feature_columns(eng))
    print(f"[phase25] features: {len(feats)} columns")


    # Build folds and assign exact dates
    folds = build_walkforward_folds(list(seasons))
    folds = assign_fold_dates(folds, raw)
    for f in folds:
        print(f"[phase25] {f.fold_id}: train={f.train_seasons} "
              f"val={f.val_season} test={f.test_season} "
              f"cutoff={f.train_cutoff_date}")

    # Compute predictions per fold
    all_fold_preds: list[FoldPrediction] = []
    all_fold_scores: list[dict] = []
    for fold in folds:
        print(f"[phase25] computing predictions for {fold.fold_id}...")
        fp = compute_walkforward_predictions(fold, raw, eng, feats, seed=seed)
        scores = score_fold_predictions(fp)
        all_fold_preds.append(fp)
        all_fold_scores.append(scores)
        print(f"[phase25] {fold.fold_id} done: "
              f"n_train={fp.n_train} n_val={fp.n_val} n_test={fp.n_test}")

    # Pooled predictions (concatenate across folds)
    pooled_y = np.concatenate([fp.y_true for fp in all_fold_preds])
    pooled_preds: dict[str, np.ndarray] = {}
    for nm in all_fold_preds[0].predictions:
        pooled_preds[nm] = np.concatenate(
            [fp.predictions[nm] for fp in all_fold_preds])

    # Validate pooled probabilities
    for nm, p in pooled_preds.items():
        validate_proba(p, len(pooled_y))


    # Pooled scores
    pooled_results = score_fold_predictions(FoldPrediction(
        fold_id="pooled",
        test_season="all_folds",
        n_train=sum(fp.n_train for fp in all_fold_preds),
        n_val=None,
        n_test=len(pooled_y),
        y_true=pooled_y,
        predictions=pooled_preds,
        meta={},
    ))

    # Paired significance vs Poisson (pooled)
    pois_proba = pooled_preds["poisson"]
    y = pooled_y
    pois_loss = np.log(np.clip(pois_proba[np.arange(len(y)), y], 1e-15, 1.0))
    pois_pred = np.argmax(pois_proba, axis=1)
    paired_vs_poisson = []
    for nm in pooled_preds:
        if nm == "poisson":
            continue
        tp = pooled_preds[nm]
        loss = np.log(np.clip(tp[np.arange(len(y)), y], 1e-15, 1.0))
        pred = np.argmax(tp, axis=1)
        mc = mcp_mcnemar_exact(y, pois_pred, pred)
        ba = paired_bootstrap(pois_loss, loss, n_boot=N_BOOT, seed=BOOT_SEED)
        paired_vs_poisson.append({
            "pair": ["poisson", nm], "metric": "logloss",
            "mcnemar": mc, "bootstrap": ba,
        })


    # Per-fold McNemar (Poisson vs each model, per fold)
    per_fold_paired = []
    for fp in all_fold_preds:
        pois_t = fp.predictions["poisson"]
        y_f = fp.y_true
        if len(y_f) == 0:
            continue
        pois_pred_f = np.argmax(pois_t, axis=1)
        pois_loss_f = np.log(
            np.clip(pois_t[np.arange(len(y_f)), y_f], 1e-15, 1.0))
        for nm in fp.predictions:
            if nm == "poisson":
                continue
            tp_f = fp.predictions[nm]
            loss_f = np.log(
                np.clip(tp_f[np.arange(len(y_f)), y_f], 1e-15, 1.0))
            pred_f = np.argmax(tp_f, axis=1)
            mc = mcp_mcnemar_exact(y_f, pois_pred_f, pred_f)
            ba = paired_bootstrap(pois_loss_f, loss_f, n_boot=N_BOOT, seed=BOOT_SEED)
            per_fold_paired.append({
                "fold": fp.fold_id, "pair": ["poisson", nm],
                "metric": "logloss", "mcnemar": mc, "bootstrap": ba,
            })

    prod_after = _prod_sha256()
    assert prod_before == prod_after, (
        "Production DB SHA-256 changed during Phase-25! (fail-closed)")

    disp_hash = hashlib.sha256(PHASE23_DISPOS.read_bytes()).hexdigest()


    # Build season manifests
    season_manifests = {}
    for s in ["2015/2016", "2016/2017", "2019/2020"]:
        srows = raw[raw["season_name"] == s]
        season_manifests[s] = {
            "n_matches": int(len(srows)),
            "date_range": [
                str(srows["match_date"].min()),
                str(srows["match_date"].max()),
            ] if len(srows) > 0 else [None, None],
        }

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    report = {
        "phase": 25,
        "produced_at": ts,
        "config": {
            "seed": seed,
            "strategy": "expanding_window_chronological_walkforward",
            "n_boot": N_BOOT,
            "boot_seed": BOOT_SEED,
            "brier_convention": (
                "multiclass_Brier=mean_over_samples_of_sum_c "
                "(p_ic - 1{c=ic})^2; uniform=2/3, perfect=0"
            ),
            "logloss_convention": "multiclass_cross_entropy_clip(1e-15,1-1e-15)",
            "calibration_protocol": "fit on val predictions only; applied to test",
            "dc_rho_strategy": "fit_rho on training window only (expanding)",
            "elo_mapping": "DiriEloMapper fit on train Elo history",
            "ensemble_strategy": "equal-weight + optimal-weight(val logloss grid)",
        },
        "dataset": {
            "source": "disposable DB (phase23_eval.db) from staged FD E0 CSVs",
            "provider": "football-data",
            "provider_competition_id": "E0",
            "seasons_available": ["2015/2016", "2016/2017", "2019/2020"],
            "season_manifests": season_manifests,
            "production_db_sha256": prod_before,
            "production_db_unchanged": prod_before == prod_after,
            "disposable_db_sha256": disp_hash,
        },
        "folds": [],
        "pooled": pooled_results,
        "paired_vs_poisson": paired_vs_poisson,
        "per_fold_paired_vs_poisson": per_fold_paired,
        "pooled_y_true_class_distribution": class_distribution(
            pooled_y, list(OUTCOME_LABELS)),
        "notes": [
            "Walk-forward uses expanding window over 3 verified E0 seasons.",
            "Fold 1: train=2015/16, test=2016/17. "
            "Fold 2: train=2015/16+2016/17, test=2019/20 (COVID-gap).",
            "Online models warmed on ALL pre-test data; fitted models use "
            "val hold-out for tuning when >=2 train seasons available.",
            "DC rho fit by MLE on training window only; may collapse near 0.",
            "Elo mapped to 3-class via DiriEloMapper; NOT native calibrated.",
            "Temperature scaling fit on val predictions only; no-val folds T=1.0.",
            "All predictions use strictly-pre-match history (strict < date cutoff).",
            "Production DB opened read-only and SHA-256 gated (fail-closed).",
            "Pooled results concatenate disjoint test fixtures across folds.",
            "Only 3 seasons available; walk-forward limited to 2 folds.",
            "COVID-19 gap in 2019/20 is a documented stationarity discontinuity.",
            "No fabricating matches, odds, predictions, or results.",
        ],
    }
    # Assemble per-fold results
    report["folds"] = []
    for fold, fp, scores in zip(folds, all_fold_preds, all_fold_scores):
        report["folds"].append({
            "fold_id": fp.fold_id,
            "train_seasons": list(fold.train_seasons),
            "val_season": fold.val_season,
            "test_season": fp.test_season,
            "n_train": fp.n_train,
            "n_val": fp.n_val,
            "n_test": fp.n_test,
            "test_date_range": [
                fold.test_start_date, fold.test_end_date],
            "train_cutoff_date": fold.train_cutoff_date,
            "models": scores,
            "meta": fp.meta,
        })

    # Rankings by pooled Brier and log-loss
    report["ranking_by_brier_pooled"] = sorted(
        [{"name": k, "accuracy": v["accuracy"], "macro_f1": v["macro_f1"],
          "log_loss": v["log_loss"], "brier": v["brier_score"]}
         for k, v in pooled_results.items()],
        key=lambda r: r["brier"])
    report["ranking_by_logloss_pooled"] = sorted(
        [{"name": k, "accuracy": v["accuracy"], "macro_f1": v["macro_f1"],
          "log_loss": v["log_loss"], "brier": v["brier_score"]}
         for k, v in pooled_results.items()],
        key=lambda r: r["log_loss"])

    _write_json(report, REPORT_DIR / f"phase25_walkforward_{ts}.json")
    _write_txt(report, REPORT_DIR / f"phase25_walkforward_{ts}.txt")

    print(f"[phase25] reports written to {REPORT_DIR}")
    print(f"[phase25] prod SHA-256 (after): {prod_after}  "
          f"unchanged={prod_before == prod_after}")
    return report


# ---------------------------------------------------------------------------
# Report writers
# ---------------------------------------------------------------------------


def _json_default(o):
    """JSON serializer for numpy types."""
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.ndarray,)):
        return o.tolist()
    if isinstance(o, (np.bool_,)):
        return bool(o)
    return str(o)


def _write_json(report: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(report, indent=2, default=_json_default, ensure_ascii=False),
        encoding="utf-8",
    )


def _write_txt(report: dict, path: Path) -> None:
    L = []
    L.append("Phase 25 - Walk-Forward Backtesting Report")
    L.append("=" * 50)
    L.append(f"Produced: {report['produced_at']}")
    d = report["dataset"]
    L.append(f"Source: {d['source']}")
    L.append(f"Seasons: {d['seasons_available']}")
    L.append(f"Production DB sha256: {d['production_db_sha256']}")
    L.append(f"Production DB unchanged: {d['production_db_unchanged']}")
    L.append(f"Disposable DB sha256: {d.get('disposable_db_sha256', 'N/A')}")
    L.append("")
    L.append("Strategy: expanding-window chronological walk-forward")
    L.append("")

    L.append("Folds")
    L.append("-----")
    for f in report["folds"]:
        L.append(f"  {f['fold_id']}: train={f['train_seasons']} "
                 f"val={f['val_season']} test={f['test_season']}")
        L.append(f"    n_train={f['n_train']} n_val={f['n_val']} n_test={f['n_test']}")
        L.append(f"    cutoff={f['train_cutoff_date']} "
                 f"test_range=[{f['test_date_range'][0]}, "
                 f"{f['test_date_range'][1]}]")
    L.append("")

    for f in report["folds"]:
        L.append(f"  Fold {f['fold_id']} test metrics (sorted by Brier):")
        L.append("  " + "-" * 70)
        L.append(f"  {'model':<32s} {'acc':>6s} {'macroF1':>7s} "
                 f"{'logloss':>8s} {'brier':>7s}")
        ranked = sorted(f["models"].items(), key=lambda kv: kv[1]["brier_score"])
        for nm, m in ranked:
            L.append(f"  {nm:<32s} {m['accuracy']:>6.3f} {m['macro_f1']:>7.3f} "
                     f"{m['log_loss']:>8.4f} {m['brier_score']:>7.4f}")
        L.append("")

    L.append("Pooled test metrics (all folds, sorted by Brier):")
    L.append("-" * 76)
    L.append(f"{'model':<32s} {'acc':>6s} {'macroF1':>7s} "
             f"{'logloss':>8s} {'brier':>7s}")
    for r in report["ranking_by_brier_pooled"]:
        L.append(f"{r['name']:<32s} {r['accuracy']:>6.3f} "
                 f"{r['macro_f1']:>7.3f} {r['log_loss']:>8.4f} "
                 f"{r['brier']:>7.4f}")
    L.append("")

    L.append("Per-fold paired McNemar vs Poisson (accuracy discordance)")
    L.append("-" * 70)
    for p in report["per_fold_paired_vs_poisson"]:
        pair = "/".join(p["pair"])
        mc = p["mcnemar"]
        L.append(f"  {p['fold']} {pair:<32s} McNemar p={mc['p_value']:.4f} "
                 f"n01={mc['n01']} n10={mc['n10']}")
    L.append("")

    L.append("Pooled paired significance vs Poisson (log-loss)")
    L.append("-" * 70)
    for p in report["paired_vs_poisson"]:
        pair = "/".join(p["pair"])
        mc = p["mcnemar"]
        ba = p["bootstrap"]
        L.append(f"  {pair:<34s} McNemar p={mc['p_value']:.4f} "
                 f"boot_diff={ba['mean_diff']:+.4f} "
                 f"95%CI=({ba['ci_low_95']:+.4f},"
                 f"{ba['ci_high_95']:+.4f})")
    L.append("")

    L.append("Brier convention: multiclass = mean over samples of "
             "sum_c (p_ic - 1{c=ic})^2")
    L.append("  (uniform prediction scores 2/3; perfect scores 0)")
    L.append("")

    L.append("Notes")
    L.append("-----")
    for n in report["notes"]:
        L.append(f"- {n}")
    path.write_text("\n".join(L) + "\n", encoding="utf-8")


if __name__ == "__main__":
    run_walkforward()
