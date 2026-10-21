#!/usr/bin/env python3
"""Phase 26.8: Extended chronological evaluation with intra-season folds.

Extends the Phase-25 walk-forward harness with two additional intra-season folds
split at the 2019/20 COVID-19 gap (2020-03-09) and the mid-season date
(2019-12-29).  These folds reuse the *same* verified 3-season Football-Data E0
dataset (3 x 380 = 1140 matches) but provide finer-grained chronological
evaluation windows.

Key additions:
1. Two extra chronological folds (fold3/fold4) with date-based cutoffs.
2. Fast evaluation mode (Poisson + LR + frequency only).
3. Brier-score decomposition (Murphy: reliability/resolution/uncertainty).
4. Pooled comparison of seasonal vs intra-season evaluation windows.
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
    class_distribution,
    compute_brier_score,
    compute_log_loss,
    _impute_and_scale,
    prepare_training_data,
)
from .calibration import (
    TemperatureScaler,
    ensemble_equal_weight,
    ensemble_optimal_weights,
    evaluate_proba,
    validate_proba,
    LABEL_IDX,
)
from .comparison import mcp_mcnemar_exact, paired_bootstrap
from .dixon_coles import DixonColesModel
from .elo import EloModel
from .poisson import PoissonModel, run_poisson_chronological
from .evaluation import (
    compute_macro_f1,
    compute_precision_recall,
    compute_calibration,
    feature_columns,
)
from ..database import resolve_db_path
from ..features.engineering import FeatureEngineer, load_matches_for_features

from . import walkforward as wf

__all__ = [
    "INTRA_SEASON_CUTOFFS",
    "EXTENDED_FOLD_COUNT",
    "build_extended_folds",
    "compute_fast_predictions",
    "brier_decomposition",
    "run_walkforward_extended",
    "write_extended_report_txt",
    "FAST_MODELS",
    "ALL_MODELS",
]

# Date cutoffs for intra-season folds within the 2019/20 season.
# COVID-19 created a natural 100-day gap (2020-03-09 -> 2020-06-17).
INTRA_SEASON_CUTOFFS: dict[str, str] = {
    "fold3_mids": "2019-12-29",   # mid-season rolling origin
    "fold4_covid": "2020-03-09",   # COVID-gap boundary
}

# Model subsets
FAST_MODELS = ("frequency_baseline", "logistic_regression", "poisson")
ALL_MODELS = (
    "frequency_baseline", "logistic_regression", "poisson",
    "poisson_temperature_scaled", "dixon_coles", "xgboost",
    "elo_mapped", "ens_poisson_lr_equal", "ens_poisson_lr_optimal",
)

EXTENDED_FOLD_COUNT = 4



# ---------------------------------------------------------------------------
# Extended fold construction
# ---------------------------------------------------------------------------

def build_extended_folds(
    raw: pd.DataFrame,
    include_intra_season: bool = True,
) -> list[wf.Folds]:
    """Build the Phase-25 seasonal folds plus intra-season folds from 2019/20.

    Seasonal folds (identical to Phase 25):
        fold1: train=2015/16,          test=2016/17
        fold2: train=2015/16+2016/17,  test=2019/20

    Intra-season folds (Phase 26.8 addition):
        fold3: train=2015/16+2016/17, test=2019/20 (matches from 2019-12-29 onward)
        fold4: train=2015/16+2016/17, test=2019/20 (matches from 2020-03-09 onward)

    All four test sets are pairwise disjoint, enabling valid pooled estimates.

    Args:
        raw: Raw match DataFrame with columns season_name, match_date, etc.
        include_intra_season: If False, only the 2 original seasonal folds.

    Returns:
        List of Folds with dates assigned (no need for assign_fold_dates).
    """
    seasonal = wf.build_walkforward_folds()
    seasonal = wf.assign_fold_dates(seasonal, raw)

    if not include_intra_season:
        return seasonal

    season_2019 = raw[raw["season_name"] == "2019/2020"]
    extra_folds = []

    # Compute disjoint test_end for fold3 (ends before fold4 cutoff)
    covid_cutoff = INTRA_SEASON_CUTOFFS["fold4_covid"]
    pre_covid = season_2019[season_2019["match_date"].astype(str) < covid_cutoff]
    fold3_test_end = str(pre_covid["match_date"].max()) if len(pre_covid) else covid_cutoff

    fold_specs = [
        ("fold3", INTRA_SEASON_CUTOFFS["fold3_mids"], fold3_test_end),
        ("fold4", INTRA_SEASON_CUTOFFS["fold4_covid"],
         str(season_2019["match_date"].max())),
    ]
    for fold_id, cutoff, test_end in fold_specs:
        test_dates = season_2019[
            (season_2019["match_date"].astype(str) >= cutoff) &
            (season_2019["match_date"].astype(str) <= test_end)
        ]
        if len(test_dates) == 0:
            continue
        extra_folds.append(wf.Folds(
            fold_id=fold_id,
            train_seasons=("2015/2016", "2016/2017"),
            val_season=None,
            test_season="2019/2020",
            train_cutoff_date=cutoff,
            test_start_date=str(test_dates["match_date"].min()),
            test_end_date=test_end,
        ))

    return seasonal + extra_folds


# ---------------------------------------------------------------------------
# Fast prediction computation (Poisson + LR + frequency only)
# ---------------------------------------------------------------------------

def _y_from_df(df: pd.DataFrame) -> np.ndarray:
    return df["_label_idx"].to_numpy(dtype=int)


def _mat(df: pd.DataFrame, feats: Sequence[str]) -> np.ndarray:
    X = df[feats].to_numpy(dtype=float)
    Xs, _ = _impute_and_scale(X, fill_value=0.0)
    return Xs


def compute_fast_predictions(
    fold: wf.Folds,
    raw: pd.DataFrame,
    features: pd.DataFrame,
    feats: Sequence[str],
    seed: int = 42,
) -> wf.FoldPrediction:
    """Compute predictions for only fast models (Poisson, LR, frequency).

    Avoids XGBoost hyperparameter search and Dixon-Coles online fitting,
    reducing per-fold time from ~15 s to ~1 s while still evaluating the
    production Poisson model and its direct comparison baseline.
    """
    d = features["target_match_date"].astype(str)
    train_mask = d < fold.train_cutoff_date
    test_mask = (d >= fold.train_cutoff_date) & (d <= fold.test_end_date)

    train_df = features[train_mask].copy().reset_index(drop=True)
    test_df = features[test_mask].copy().reset_index(drop=True)

    n_train = len(train_df)
    n_val = None
    n_test = len(test_df)
    y_test = _y_from_df(test_df) if n_test > 0 else np.array([], dtype=int)
    raw_sorted = raw.sort_values(
        ["match_date", "home_team_id", "away_team_id"]
    ).reset_index(drop=True)

    preds: dict[str, np.ndarray] = {}
    meta: dict = {}

    # --- Frequency baseline ---
    y_train = _y_from_df(train_df)
    fb = FrequencyBaseline().fit(y_train)
    freq_proba = np.tile(fb._class_proba, (n_test, 1))
    preds["frequency_baseline"] = freq_proba
    meta["frequency_baseline"] = {"type": "frequency", "fit_on": "train"}

    # --- Logistic Regression (default params, no val tuning) ---
    m = LogisticRegressionClassifier(
        learning_rate=0.05, n_iterations=1000,
        l2_lambda=0.01, random_state=seed,
    ).fit(_mat(train_df, feats), y_train)
    preds["logistic_regression"] = m.predict_proba(_mat(test_df, feats))
    meta["logistic_regression"] = {"type": "lr", "l2": 0.01, "val_tuned": False}

    # --- Poisson (online, strict pre-match history) ---
    pois_full = wf._online_poisson(raw_sorted)
    pois_test = wf._align_online_to_test(pois_full, test_df)
    preds["poisson"] = pois_test
    meta["poisson"] = {"type": "poisson", "history": "all_pre_test"}

    # --- Temperature-scaled Poisson (no val -> T=1.0) ---
    preds["poisson_temperature_scaled"] = pois_test.copy()
    meta["poisson_temperature_scaled"] = {
        "base": "poisson", "T": 1.0, "note": "no_val_no_calibration",
    }

    # Validate all probability vectors
    for nm, p in preds.items():
        validate_proba(p, n_test)

    return wf.FoldPrediction(
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
# Brier score decomposition (Murphy, 1973)
# ---------------------------------------------------------------------------

def brier_decomposition(y_true: np.ndarray, proba: np.ndarray) -> dict:
    """Decompose multiclass Brier score (Murphy, 1973 per-class).

    For each class c, the binary Brier component is decomposed:
        Brier_c = Reliability_c - Resolution_c + Uncertainty_c
    Total Brier = sum_c Brier_c.

    Args:
        y_true: Integer class labels (0, 1, 2).
        proba: Probability matrix of shape (n_samples, 3).

    Returns:
        Dict with brier, reliability, resolution, uncertainty, n, n_bins, check.
    """
    y = np.asarray(y_true, dtype=int)
    P = np.asarray(proba, dtype=float)
    n = len(y)
    if n == 0:
        return {"brier": 0.0, "reliability": 0.0, "resolution": 0.0,
                "uncertainty": 0.0, "n": 0, "n_bins": 0, "check": 0.0}

    brier = compute_brier_score(y, P)

    n_bins = min(10, n)
    if n_bins < 2:
        n_bins = 2
    edges = np.linspace(0.0, 1.0, n_bins + 1)

    total_reliability = 0.0
    total_resolution = 0.0
    total_uncertainty = 0.0

    for c in range(N_OUTCOMES):
        y_c = (y == c).astype(float)
        p_c = P[:, c]
        o_c = float(np.mean(y_c))
        total_uncertainty += o_c * (1.0 - o_c)

        bin_idx = np.digitize(p_c, edges[1:-1])
        for b in range(n_bins):
            mask = bin_idx == b
            cnt = int(mask.sum())
            if cnt == 0:
                continue
            p_mean = float(np.mean(p_c[mask]))
            o_mean = float(np.mean(y_c[mask]))
            total_reliability += cnt * (p_mean - o_mean) ** 2 / n
            total_resolution += cnt * (o_mean - o_c) ** 2 / n

    return {
        "brier": float(brier),
        "reliability": float(total_reliability),
        "resolution": float(total_resolution),
        "uncertainty": float(total_uncertainty),
        "n": int(n),
        "n_bins": int(n_bins),
        "check": float(total_reliability - total_resolution + total_uncertainty),
    }


# ---------------------------------------------------------------------------
# Pooling and significance helpers
# ---------------------------------------------------------------------------

def _pool_fold_predictions(
    fold_preds: list[wf.FoldPrediction],
    model_names: Sequence[str] = FAST_MODELS,
) -> tuple[np.ndarray, dict[str, np.ndarray], list[str]]:
    """Concatenate test predictions across folds for specified models."""
    pooled_y = np.concatenate(
        [fp.y_true for fp in fold_preds if fp.n_test > 0]
    )
    pooled_preds: dict[str, np.ndarray] = {}
    available: list[str] = []
    n_test_folds = len([fp for fp in fold_preds if fp.n_test > 0])
    for nm in model_names:
        arrs = [
            fp.predictions[nm] for fp in fold_preds
            if nm in fp.predictions and fp.n_test > 0
        ]
        if len(arrs) == n_test_folds:
            pooled_preds[nm] = np.concatenate(arrs)
            available.append(nm)
    return pooled_y, pooled_preds, available


def _paired_significance(
    y: np.ndarray,
    pooled_preds: dict[str, np.ndarray],
    base_model: str = "poisson",
    n_boot: int = 200,
    seed: int = 42,
) -> list[dict]:
    """Compute paired McNemar + bootstrap tests vs base_model."""
    results: list[dict] = []
    base_loss = np.log(
        np.clip(pooled_preds[base_model][np.arange(len(y)), y], 1e-15, 1.0)
    )
    base_pred = np.argmax(pooled_preds[base_model], axis=1)
    for nm in pooled_preds:
        if nm == base_model:
            continue
        tp = pooled_preds[nm]
        loss = np.log(np.clip(tp[np.arange(len(y)), y], 1e-15, 1.0))
        pred = np.argmax(tp, axis=1)
        mc = mcp_mcnemar_exact(y, base_pred, pred)
        ba = paired_bootstrap(base_loss, loss, n_boot=n_boot, seed=seed)
        results.append({
            "pair": [base_model, nm], "metric": "logloss",
            "mcnemar": mc, "bootstrap": ba,
        })
    return results


# ---------------------------------------------------------------------------
# Main driver
# ---------------------------------------------------------------------------

def run_walkforward_extended(
    seed: int = 42,
    n_boot: int = 200,
    db_path: Path = wf.PHASE23_DISPOS,
    include_intra_season: bool = True,
    fast: bool = True,
) -> dict:
    """Run the extended Phase-26.8 walk-forward evaluation.

    Args:
        seed: Random seed for reproducibility.
        n_boot: Bootstrap iterations for paired significance tests.
        db_path: Path to the disposable eval DB.
        include_intra_season: If True, adds fold3/fold4 intra-season splits.
        fast: If True, only evaluates Poisson + LR + frequency (skips XGBoost/DC/Elo).

    Returns:
        Enhanced report dict with per-fold and pooled metrics, Brier
        decomposition, and paired significance tests.
    """
    prod_before = wf._prod_sha256()

    raw = wf._load_raw_matches(db_path)
    eng, _, _ = prepare_training_data(
        raw, engineer=FeatureEngineer(), label_col="outcome_label",
    )
    eng = wf._add_interactions(eng)
    feats = list(feature_columns(eng))

    folds = build_extended_folds(raw, include_intra_season=include_intra_season)
    for f in folds:
        print(f"[phase26_8] {f.fold_id}: train={f.train_seasons} "
              f"val={f.val_season} test={f.test_season} "
              f"cutoff={f.train_cutoff_date}")

    compute_fn = compute_fast_predictions if fast else wf.compute_walkforward_predictions
    all_fold_preds: list[wf.FoldPrediction] = []
    all_fold_scores: list[dict] = []
    for fold in folds:
        fp = compute_fn(fold, raw, eng, feats, seed=seed)
        scores = wf.score_fold_predictions(fp)
        all_fold_preds.append(fp)
        all_fold_scores.append(scores)
        print(f"[phase26_8] {fold.fold_id} done: "
              f"n_train={fp.n_train} n_val={fp.n_val} n_test={fp.n_test}")

    # Pooled across ALL folds (disjoint test sets)
    model_names = FAST_MODELS if fast else ALL_MODELS
    pooled_y, pooled_preds, _ = _pool_fold_predictions(
        all_fold_preds, model_names=model_names,
    )
    for nm, p in pooled_preds.items():
        validate_proba(p, len(pooled_y))

    pooled_results = wf.score_fold_predictions(
        wf.FoldPrediction(
            fold_id="pooled", test_season="all_folds",
            n_train=sum(fp.n_train for fp in all_fold_preds),
            n_val=None, n_test=len(pooled_y),
            y_true=pooled_y, predictions=pooled_preds, meta={},
        )
    )

    # Pooled A: fold1+fold2 (season-level, original Phase 25 pool)
    pooled_a_y, pooled_a_preds, _ = _pool_fold_predictions(
        all_fold_preds[:2], model_names=model_names,
    )
    pooled_a_results = wf.score_fold_predictions(
        wf.FoldPrediction(
            fold_id="pooled_seasonal", test_season="seasonal_folds",
            n_train=0, n_val=None, n_test=len(pooled_a_y),
            y_true=pooled_a_y, predictions=pooled_a_preds, meta={},
        )
    )

    # Pooled B: fold1+fold3+fold4 (intra-season, disjoint from fold2)
    intra_indices = [i for i, f in enumerate(folds)
                     if f.fold_id in ("fold1", "fold3", "fold4")]
    pooled_b_y, pooled_b_preds, _ = _pool_fold_predictions(
        [all_fold_preds[i] for i in intra_indices], model_names=model_names,
    )
    pooled_b_results = wf.score_fold_predictions(
        wf.FoldPrediction(
            fold_id="pooled_intra", test_season="intra_folds",
                        n_train=0, n_val=None, n_test=len(pooled_b_y),
            y_true=pooled_b_y, predictions=pooled_b_preds, meta={},
        )
    )

    # Brier decomposition for Poisson in each pool
    brier_decomp = {}
    for pool_name, (pv_y, pv_preds) in [
        ("pooled", (pooled_y, pooled_preds)),
        ("pooled_seasonal", (pooled_a_y, pooled_a_preds)),
        ("pooled_intra", (pooled_b_y, pooled_b_preds)),
    ]:
        if "poisson" in pv_preds:
            brier_decomp[pool_name] = {
                "poisson": brier_decomposition(pv_y, pv_preds["poisson"]),
            }

    # Paired significance tests vs Poisson (pooled)
    paired_tests = _paired_significance(
        pooled_y, pooled_preds, base_model="poisson",
        n_boot=n_boot, seed=seed,
    )

    prod_after = wf._prod_sha256()
    assert prod_before == prod_after, (
        "Production DB SHA-256 changed during Phase-26.8! (fail-closed)"
    )

    disp_hash = hashlib.sha256(wf.PHASE23_DISPOS.read_bytes()).hexdigest()

    season_manifests = {}
    for s in ["2015/2016", "2016/2017", "2019/2020"]:
        srows = raw[raw["season_name"] == s]
        season_manifests[s] = {
            "n_matches": int(len(srows)),
            "date_range": [str(srows["match_date"].min()),
                           str(srows["match_date"].max())] if len(srows) else [None, None],
        }

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    report = {
        "phase": 26.8,
        "title": "Extended chronological evaluation with intra-season folds",
        "produced_at": ts,
        "config": {
            "seed": seed,
            "n_boot": n_boot,
            "fast_mode": fast,
            "strategy": "expanding_window_walkforward + intra-season COVID split",
            "intra_season_cutoffs": INTRA_SEASON_CUTOFFS,
            "models_evaluated": list(pooled_preds.keys()),
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
        "pooled_seasonal": pooled_a_results,
        "pooled_intra": pooled_b_results,
        "brier_decomposition": brier_decomp,
        "paired_vs_poisson": paired_tests,
        "pooled_y_true_class_distribution": class_distribution(
            pooled_y, list(OUTCOME_LABELS)),
        "notes": [
            "Phase 26.8 extends Phase-25 with 2 intra-season folds split at the "
            "2019/20 COVID-19 gap (2020-03-09) and mid-season (2019-12-29).",
            "The seasonal pool (fold1+fold2) and intra-season pool "
            "(fold1+fold3+fold4) use pairwise-disjoint test sets.  The "
            "\"pooled (all folds)\" aggregate includes fold2 (full 2019/20) "
            "which overlaps with fold3+fold4 subsets, so it is not a strictly "
            "disjoint pool -- use pooled_seasonal or pooled_intra for clean "
            "estimates.",
            "COVID-19 gap (2020-03-09 to 2020-06-17, 100 days) is a documented "
            "stationarity discontinuity.",
            "No additional historical data was available locally; intra-season "
            "folds reuse the same verified 3-season E0 dataset.",
            "Fast mode evaluates only Poisson, LR, and frequency baseline.",
            "Brier decomposition follows Murphy (1973) formulation.",
            "Production DB opened read-only and SHA-256 gated (fail-closed).",
            "Only verified local data used -- no fabrication of matches, odds, "
            "or predictions.",
        ],
    }

    for fold, fp, scores in zip(folds, all_fold_preds, all_fold_scores):
        report["folds"].append({
            "fold_id": fp.fold_id,
            "train_seasons": list(fold.train_seasons),
            "val_season": fold.val_season,
            "test_season": fp.test_season,
            "n_train": fp.n_train,
            "n_val": fp.n_val,
            "n_test": fp.n_test,
            "test_date_range": [fold.test_start_date, fold.test_end_date],
            "train_cutoff_date": fold.train_cutoff_date,
            "models": scores,
            "meta": fp.meta,
        })

    report["ranking_by_brier_pooled"] = sorted(
        [{"name": k, "accuracy": v["accuracy"], "macro_f1": v["macro_f1"],
          "log_loss": v["log_loss"], "brier_score": v["brier_score"]}
         for k, v in pooled_results.items()],
        key=lambda r: r["brier_score"])

    report_dir = wf.REPORT_DIR.parent / "phase26_8"
    report_dir.mkdir(parents=True, exist_ok=True)
    json_path = report_dir / f"phase26_8_extended_{ts}.json"
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, default=str)

    print(f"[phase26_8] Report written to {json_path}")
    print(f"[phase26_8] prod SHA-256 (after): {prod_after}  "
          f"unchanged={prod_before == prod_after}")

    return report


# ---------------------------------------------------------------------------
# Enhanced text report writer
# ---------------------------------------------------------------------------

def write_extended_report_txt(report: dict, path: Path) -> None:
    """Write a human-readable text report from the extended walkforward results."""
    L: list[str] = []
    L.append("=" * 72)
    L.append("Phase 26.8: Extended Walk-Forward Evaluation Report")
    L.append(f"  Produced: {report['produced_at']}")
    L.append(f"  Seed: {report['config']['seed']}")
    L.append(f"  Bootstrap iterations: {report['config']['n_boot']}")
    L.append(f"  Fast mode: {report['config']['fast_mode']}")
    L.append(f"  Models: {', '.join(report['config']['models_evaluated'])}")
    L.append("=" * 72)

    L.append("")
    L.append("## Dataset")
    ds = report["dataset"]
    L.append(f"  Provider: {ds['provider']} / {ds['provider_competition_id']}")
    L.append(f"  Seasons: {ds['seasons_available']}")
    L.append(f"  Production DB unchanged: {ds['production_db_unchanged']}")
    for s, m in ds["season_manifests"].items():
        L.append(f"  {s}: {m['n_matches']} matches, "
                 f"{m['date_range'][0]} to {m['date_range'][1]}")

    L.append("")
    L.append("## Folds (4 chronological evaluation windows)")
    for f in report["folds"]:
        L.append(f"\n  {f['fold_id']}: train={f['train_seasons']} "
                 f"test={f['test_season']}")
        L.append(f"    train_cutoff: {f['train_cutoff_date']}")
        L.append(f"    test_range: {f['test_date_range'][0]} to "
                 f"{f['test_date_range'][1]}")
        L.append(f"    n_train={f['n_train']} n_val={f['n_val']} "
                 f"n_test={f['n_test']}")
        for nm, m in sorted(f["models"].items()):
            L.append(f"    {nm}: acc={m['accuracy']:.4f} "
                     f"f1={m['macro_f1']:.4f} "
                     f"logloss={m['log_loss']:.4f} "
                     f"brier={m['brier_score']:.4f}")

    L.append("")
    L.append("## Pooled Results - All Folds")
    for nm, m in sorted(report["pooled"].items()):
        L.append(f"  {nm}: acc={m['accuracy']:.4f} f1={m['macro_f1']:.4f} "
                 f"logloss={m['log_loss']:.4f} brier={m['brier_score']:.4f} "
                 f"n={m['n']}")

    L.append("")
    L.append("## Pooled - Seasonal Folds Only (fold1+fold2)")
    for nm, m in sorted(report["pooled_seasonal"].items()):
        L.append(f"  {nm}: acc={m['accuracy']:.4f} f1={m['macro_f1']:.4f} "
                 f"logloss={m['log_loss']:.4f} brier={m['brier_score']:.4f} "
                 f"n={m['n']}")

    L.append("")
    L.append("## Pooled - Intra-Season Folds Only (fold1+fold3+fold4)")
    for nm, m in sorted(report["pooled_intra"].items()):
        L.append(f"  {nm}: acc={m['accuracy']:.4f} f1={m['macro_f1']:.4f} "
                 f"logloss={m['log_loss']:.4f} brier={m['brier_score']:.4f} "
                 f"n={m['n']}")

    L.append("")
    L.append("## Brier Decomposition (Murphy)")
    for pool_name, dec in report["brier_decomposition"].items():
        d = dec.get("poisson", {})
        L.append(f"  {pool_name}:")
        L.append(f"    brier={d.get('brier', 0):.4f} "
                 f"reliability={d.get('reliability', 0):.4f} "
                 f"resolution={d.get('resolution', 0):.4f} "
                 f"uncertainty={d.get('uncertainty', 0):.4f} "
                 f"recon={d.get('check', 0):.4f}")

    L.append("")
    L.append("## Paired Significance vs Poisson (pooled)")
    for p in report["paired_vs_poisson"]:
        mc = p["mcnemar"]
        ba = p["bootstrap"]
        L.append(f"  {p['pair'][1]}:")
        L.append(f"    McNemar: n01={mc['n01']} n10={mc['n10']} "
                 f"stat={mc['statistic']:.4f} p={mc['p_value']:.4f}")
        L.append(f"    Bootstrap: mean_diff={ba['mean_diff']:.4f} "
                 f"CI95=[{ba['ci_low_95']:.4f}, {ba['ci_high_95']:.4f}] "
                 f"p={ba['p_value']:.4f}")

    L.append("")
    L.append("## Notes")
    for note in report["notes"]:
        L.append(f"  - {note}")

    Path(path).write_text("\n".join(L), encoding="utf-8")






