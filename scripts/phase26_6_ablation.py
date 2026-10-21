#!/usr/bin/env python3
"""Phase 26.6: controlled ablation study of last-five-match form features.

Compares the logistic-regression model across four feature configurations
on identical walk-forward folds, using the same split dates, training
protocol, and eligible fixtures:

  A. Phase-25 baseline  (overall form counts only)
  B. + overall averages  (points_per_match, avg_goals_scored/conceded,
                          clean_sheets, matches_scored)
  C. + venue split       (also venue-specific at_home/at_away form)
  D. + recency-weighted  (also recency_weighted_points, recency_matches)

The Poisson model does NOT consume engineered features, so its predictions
must be identical across all four configurations (sanity check).

Production DB is opened strictly read-only and SHA-256 gated.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd

from football_intelligence.features.engineering import (
    FeatureEngineer,
    load_matches_for_features,
)
from football_intelligence.prediction.calibration import evaluate_proba
from football_intelligence.prediction.baseline import (
    _impute_and_scale,
    compute_brier_score,
    compute_log_loss,
    prepare_training_data,
    class_distribution,
    N_OUTCOMES,
)
from football_intelligence.prediction.walkforward import (
    Folds,
    PHASE23_DISPOS,
    PROD,
    WALKFORWARD_SEASONS,
    WALKFORWARD_SEED,
    N_BOOT,
    BOOT_SEED,
    _add_interactions,
    _y_from_df,
    _probs_lr_tuned,
    _online_poisson,
    _align_online_to_test,
    _season_date_range,
    build_walkforward_folds,
    assign_fold_dates,
    feature_columns,
    validate_proba,
    mcp_mcnemar_exact,
    paired_bootstrap,
    ensemble_equal_weight,
    ensemble_optimal_weights,
    TemperatureScaler,
)
from football_intelligence.database import resolve_db_path

REPORT_DIR = Path("reports/phase26_6")


@dataclass(frozen=True)
class AblationConfig:
    """A single ablation configuration."""
    name: str
    feature_engineer: FeatureEngineer
    description: str


def make_configs() -> list[AblationConfig]:
    """Return the four ablation configurations A, B, C, D."""
    return [
        AblationConfig(
            name="A_baseline",
            feature_engineer=FeatureEngineer(
                include_form_averages=False,
                include_home_away_split=False,
                include_recency_weighted=False,
            ),
            description=(
                "Phase-25 baseline: overall form counts only (points, "
                "wins, draws, losses, goals_scored/conceded, goal_diff, "
                "matches_played). No per-match averages, no venue split, "
                "no recency weighting."
            ),
        ),
        AblationConfig(
            name="B_plus_form_averages",
            feature_engineer=FeatureEngineer(
                include_form_averages=True,
                include_home_away_split=False,
                include_recency_weighted=False,
            ),
            description=(
                "Config A + per-match average features: points_per_match, "
                "avg_goals_scored, avg_goals_conceded, clean_sheets, "
                "matches_scored for last-5 and last-10 windows."
            ),
        ),
        AblationConfig(
            name="C_plus_venue_split",
            feature_engineer=FeatureEngineer(
                include_form_averages=True,
                include_home_away_split=True,
                include_recency_weighted=False,
            ),
            description=(
                "Config B + venue-specific form features: *_at_home_last_5 "
                "and *_at_away_last_5 for both teams."
            ),
        ),
        AblationConfig(
            name="D_plus_recency_weighted",
            feature_engineer=FeatureEngineer(
                include_form_averages=True,
                include_home_away_split=True,
                include_recency_weighted=True,
            ),
                        description=(
                "Config C + recency-weighted points: linear weights "
                "(oldest=1, newest=n) normalised, for last-5 window."
            ),
        ),
    ]


# ---------------------------------------------------------------------------
# Data loading and feature building
# ---------------------------------------------------------------------------

def _prod_sha256() -> str:
    """SHA-256 of the production DB (read-only invariant check)."""
    h = hashlib.sha256()
    with open(PROD, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_raw_matches(db_path: Path = PHASE23_DISPOS) -> pd.DataFrame:
    """Load matches from disposable DB (prod SHA-256 gated, fail-closed)."""
    prod_before = _prod_sha256()
    db = resolve_db_path(db_path)
    df = load_matches_for_features(
        db_path=db, provider="football-data", provider_competition_id="E0",
    )
    assert prod_before == _prod_sha256(), (
        "Production DB SHA-256 changed during load (fail-closed)"
    )
    return df


def _build_feature_dataset(
    raw: pd.DataFrame,
    engineer: FeatureEngineer,
) -> tuple[pd.DataFrame, list[str]]:
    """Build features, add interactions, return (df, feature_cols)."""
    eng, _X, _y = prepare_training_data(
        raw, engineer=engineer, label_col="outcome_label",
    )
    eng = _add_interactions(eng)
    feats = list(feature_columns(eng))
    return eng, feats


# ---------------------------------------------------------------------------
# Single-fold ablation computation
# ---------------------------------------------------------------------------

def compute_ablation_fold(
    fold: Folds,
    raw: pd.DataFrame,
    features: pd.DataFrame,
    feats: Sequence[str],
    seed: int = WALKFORWARD_SEED,
) -> dict:
    """Compute LR + Poisson + ensemble predictions for one fold.

    Mirrors the model-training logic in walkforward.compute_walkforward_predictions
    but only for LR, Poisson, and the Poisson+LR ensembles.
    """
    d = features["target_match_date"].astype(str)
    train_mask = d < fold.train_cutoff_date
    test_mask = (d >= fold.train_cutoff_date) & (d <= fold.test_end_date)

    train_df = features[train_mask].copy().reset_index(drop=True)
    test_df = features[test_mask].copy().reset_index(drop=True)

    # Validation split
    if fold.val_season is not None:
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

    models: dict[str, dict] = {}

    # --- Logistic Regression (uses features) ---
    _lr, lr_val_proba, lr_test_proba, lr_l2 = _probs_lr_tuned(
        fit_train, val_df, test_df, feats, seed
    )
    validate_proba(lr_test_proba, n_test)
    models["logistic_regression"] = evaluate_proba(
        y_test, lr_test_proba, name="lr",
        config={"l2": lr_l2, "n_features": len(feats),
                "tuned_on_val": fold.val_season is not None},
    )

    # --- Poisson (does NOT use features — sanity check) ---
    pois_full = _online_poisson(raw_sorted)
    pois_test = _align_online_to_test(pois_full, test_df)
    validate_proba(pois_test, n_test)
    models["poisson"] = evaluate_proba(
        y_test, pois_test, name="poisson",
        config={"type": "poisson", "history": "all_pre_test"},
    )

    # --- Poisson + LR equal-weight ensemble ---
    ens_eq = ensemble_equal_weight([pois_test, lr_test_proba])
    models["ens_poisson_lr_equal"] = evaluate_proba(
        y_test, ens_eq, name="ens_eq",
        config={"members": ["poisson", "logistic_regression"]},
    )

    # --- Poisson + LR optimal-weight ensemble (val-tuned) ---
    ens_opt = ens_eq  # default fallback
    if len(val_df) > 0:
        pois_val_probs = _align_online_to_test(pois_full, val_df)
        y_val = _y_from_df(val_df)
        try:
            ens_opt, w, vll = ensemble_optimal_weights(
                [pois_val_probs, lr_val_proba], y_val,
                [pois_test, lr_test_proba],
            )
            models["ens_poisson_lr_optimal"] = evaluate_proba(
                y_test, ens_opt, name="ens_opt",
                config={"weights": list(w), "val_logloss": vll,
                        "members": ["poisson", "logistic_regression"]},
            )
        except Exception:
            models["ens_poisson_lr_optimal"] = evaluate_proba(
                y_test, ens_opt, name="ens_opt_fallback",
            )
    else:
        models["ens_poisson_lr_optimal"] = evaluate_proba(
            y_test, ens_opt, name="ens_opt_noval",
            config={"note": "no_val_weights_default_to_equal"},
        )

    return {
        "fold_id": fold.fold_id,
        "n_train": n_train,
        "n_val": n_val,
        "n_test": n_test,
        "n_features": len(feats),
        "models": models,
        "meta": {"train_cutoff": fold.train_cutoff_date,
                 "test_season": fold.test_season,
                 "test_date_range": [fold.test_start_date,
                                     fold.test_end_date]},
        "predictions": {
            "logistic_regression": lr_test_proba,
            "poisson": pois_test,
            "ens_poisson_lr_equal": ens_eq,
            "ens_poisson_lr_optimal": ens_opt,
        },
        "y_true": y_test,
    }


# ---------------------------------------------------------------------------
# Pooled metrics and significance testing
# ---------------------------------------------------------------------------

def _pool_predictions(fold_results: list[dict]) -> dict[str, np.ndarray]:
    """Concatenate predictions and labels across folds."""
    pooled_y = np.concatenate([fr["y_true"] for fr in fold_results
                               if fr["y_true"].size > 0])
    pooled_preds: dict[str, np.ndarray] = {}
    for model_name in ["logistic_regression", "poisson",
                       "ens_poisson_lr_equal", "ens_poisson_lr_optimal"]:
        arrs = [fr["predictions"][model_name] for fr in fold_results
                if fr["predictions"][model_name].size > 0]
        if arrs:
            pooled_preds[model_name] = np.concatenate(arrs)
    return {**pooled_preds, "_y_true": pooled_y}


def _score_pooled(y_true: np.ndarray, proba: np.ndarray,
                   name: str = "", config: dict | None = None) -> dict:
    """Compute all metrics on pooled predictions."""
    if y_true.size == 0 or proba.size == 0:
        return {"name": name, "n": 0, "accuracy": 0.0,
                "macro_f1": 0.0, "log_loss": 0.0, "brier_score": 0.0}
    return evaluate_proba(y_true, proba, name=name, config=config)


def _paired_logloss_test(
    y_true: np.ndarray,
    proba_a: np.ndarray,
    proba_b: np.ndarray,
    n_boot: int = N_BOOT,
    seed: int = BOOT_SEED,
) -> dict:
    """Paired bootstrap test on log-loss difference (B - A).

    Positive mean_diff means config B has worse log loss (higher is worse).
    """
    loss_a = np.log(np.clip(proba_a[np.arange(len(y_true)), y_true], 1e-15, 1.0))
    loss_b = np.log(np.clip(proba_b[np.arange(len(y_true)), y_true], 1e-15, 1.0))
    diff = loss_a - loss_b  # positive => B better (lower loss)
    rng = np.random.RandomState(seed)
    boot_means = np.array([
        np.mean(diff[rng.randint(0, len(diff), size=len(diff))])
        for _ in range(n_boot)
    ])
    mean_diff = float(np.mean(diff))
    ci_lo = float(np.percentile(boot_means, 2.5))
    ci_hi = float(np.percentile(boot_means, 97.5))
    # Proportion of bootstrap samples where B is better (diff > 0)
    p_better = float(np.mean(boot_means > 0))
    return {
        "mean_diff": mean_diff,
        "ci_95": [ci_lo, ci_hi],
        "p_better_b": p_better,
        "n": int(len(diff)),
        "n_boot": n_boot,
    }


def _mcnemar_pooled(
    y_true: np.ndarray,
    proba_a: np.ndarray,
    proba_b: np.ndarray,
) -> dict:
    """McNemar's test on prediction accuracy (B vs A)."""
    pred_a = np.argmax(proba_a, axis=1)
    pred_b = np.argmax(proba_b, axis=1)
    return mcp_mcnemar_exact(y_true, pred_a, pred_b)


def _verify_poisson_sanity(config_results: dict, folds: list) -> dict:
    """Verify Poisson predictions are identical across all feature configs.

    Poisson uses raw match data only — engineered features must not affect it.
    Checks that for each fold, the Poisson brier score is the same across
    all configs (different folds legitimately have different scores).
    """
    # per-fold_id -> list of brier scores across configs
    per_fold: dict[str, list[float]] = {}
    for cfg_name, res in config_results.items():
        for fr in res["folds"]:
            fid = fr["fold_id"]
            brier = fr["models"]["poisson"]["brier_score"]
            per_fold.setdefault(fid, []).append(brier)

    all_pass = True
    fold_details = {}
    for fid, briers in per_fold.items():
        unique = set(round(b, 12) for b in briers)
        ok = len(unique) == 1
        if not ok:
            all_pass = False
        fold_details[fid] = {
            "n_configs": len(briers),
            "n_unique_brier": len(unique),
            "brier_values": briers,
            "pass": ok,
        }

    return {
        "status": "PASS" if all_pass else "FAIL",
        "n_folds": len(fold_details),
        "fold_details": fold_details,
        "message": (
            "Poisson predictions identical across all configs for every "
            "fold (Poisson does not consume engineered features)."
            if all_pass else
            "Poisson predictions DIFFER across configs for some fold "
            "— potential leakage!"
        ),
    }


# ---------------------------------------------------------------------------
# Main ablation runner
# ---------------------------------------------------------------------------

def run_phase26_6_ablation(
    db_path: Path = PHASE23_DISPOS,
    seed: int = WALKFORWARD_SEED,
    seasons: Sequence[str] | None = None,
) -> dict:
    """Execute the Phase-26.6 ablation study.

    Returns a dict with per-config, per-fold results, pooled metrics,
    paired significance tests, and Poisson sanity check.
    """
    if seasons is None:
        seasons = WALKFORWARD_SEASONS

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    prod_before = _prod_sha256()
    print(f"[phase26_6] prod DB SHA-256 (before): {prod_before}")

    raw = _load_raw_matches(db_path)
    print(f"[phase26_6] loaded {len(raw)} matches from disposable DB")

    # Build folds ONCE (identical for all configs)
    folds = build_walkforward_folds(list(seasons))
    folds = assign_fold_dates(folds, raw)
    for f in folds:
        print(f"[phase26_6] {f.fold_id}: train={f.train_seasons} "
              f"val={f.val_season} test={f.test_season} "
              f"cutoff={f.train_cutoff_date}")

    configs = make_configs()
    config_results: dict[str, dict] = {}

    for cfg in configs:
        print(f"\n[phase26_6] === Config {cfg.name} ===")
        print(f"[phase26_6] {cfg.description}")

        eng, feats = _build_feature_dataset(raw, cfg.feature_engineer)
        print(f"[phase26_6] features: {len(feats)} columns")

        fold_results = []
        for fold in folds:
            print(f"[phase26_6] fold {fold.fold_id}...")
            fr = compute_ablation_fold(
                fold, raw, eng, feats, seed=seed
            )
            fold_results.append(fr)
            print(f"[phase26_6]   lr_brier={fr['models']['logistic_regression']['brier_score']:.4f}"
                  f" pois_brier={fr['models']['poisson']['brier_score']:.4f}")

        # Pooled metrics
        pooled = _pool_predictions(fold_results)
        pooled_y = pooled["_y_true"]
        pooled_metrics: dict[str, dict] = {}
        for model_name in ["logistic_regression", "poisson",
                           "ens_poisson_lr_equal", "ens_poisson_lr_optimal"]:
            if model_name in pooled and pooled[model_name].size > 0:
                pooled_metrics[model_name] = _score_pooled(
                    pooled_y, pooled[model_name], name=model_name,
                    config={"pooled": True},
                )

        config_results[cfg.name] = {
            "description": cfg.description,
            "n_features": len(feats),
            "feature_names": list(feats),
            "folds": fold_results,
            "pooled": pooled_metrics,
            "pooled_predictions": pooled,
        }

    # Poisson sanity check
    poisson_check = _verify_poisson_sanity(config_results, folds)
    print(f"\n[phase26_6] Poisson sanity check: {poisson_check['status']}")

    # Paired comparisons vs baseline A (pooled)
    baseline_name = "A_baseline"
    paired_tests: list[dict] = []
    if baseline_name in config_results:
        baseline_pred = config_results[baseline_name]["pooled_predictions"]
        baseline_y = baseline_pred["_y_true"]
        for cfg_name, res in config_results.items():
            if cfg_name == baseline_name:
                continue
            cfg_pred = res["pooled_predictions"]
            if ("logistic_regression" in baseline_pred
                    and "logistic_regression" in cfg_pred):
                test = _paired_logloss_test(
                    baseline_y,
                    baseline_pred["logistic_regression"],
                    cfg_pred["logistic_regression"],
                )
                mcnemar = _mcnemar_pooled(
                    baseline_y,
                    baseline_pred["logistic_regression"],
                    cfg_pred["logistic_regression"],
                )
                paired_tests.append({
                    "pair": [baseline_name, cfg_name],
                    "metric": "logloss",
                    "bootstrap": test,
                    "mcnemar": mcnemar,
                })

    prod_after = _prod_sha256()
    assert prod_before == prod_after, (
        "Production DB SHA-256 changed during Phase-26.6! (fail-closed)"
    )
    disp_hash = hashlib.sha256(PHASE23_DISPOS.read_bytes()).hexdigest()

    # Build report
    report = {
        "phase": 26,
        "subphase": 6,
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
            "eligible_seasons": list(seasons),
        },
        "dataset": {
            "source": "disposable DB from staged FD E0 CSVs",
            "provider": "football-data",
            "provider_competition_id": "E0",
            "seasons_available": ["2015/2016", "2016/2017", "2019/2020"],
            "production_db_sha256": prod_before,
            "production_db_unchanged": prod_before == prod_after,
            "disposable_db_sha256": disp_hash,
        },
        "configs": {
            name: {
                "description": res["description"],
                "n_features": res["n_features"],
                "pooled": res["pooled"],
            }
            for name, res in config_results.items()
        },
        "folds_def": [
            {
                "fold_id": f.fold_id,
                "train_seasons": list(f.train_seasons),
                "val_season": f.val_season,
                "test_season": f.test_season,
                "train_cutoff_date": f.train_cutoff_date,
                "test_start_date": f.test_start_date,
                "test_end_date": f.test_end_date,
            }
            for f in folds
        ],
        "per_fold_results": {
            name: [
                {
                    "fold_id": fr["fold_id"],
                    "n_train": fr["n_train"],
                    "n_val": fr["n_val"],
                    "n_test": fr["n_test"],
                    "n_features": fr["n_features"],
                    "models": fr["models"],
                    "meta": fr["meta"],
                }
                for fr in res["folds"]
            ]
            for name, res in config_results.items()
        },
        "paired_tests": paired_tests,
        "poisson_sanity_check": poisson_check,
        "notes": [
            "Ablation compares logistic-regression model across 4 feature "
            "configurations on identical walk-forward folds.",
            "Config A = Phase-25 baseline (no form averages, no venue "
            "split, no recency weighting).",
            "Config B = A + per-match average features.",
            "Config C = B + venue-specific form features.",
            "Config D = C + recency-weighted points.",
            "Poisson model does NOT consume engineered features; its "
            "predictions must be identical across all configs (sanity "
            "check enforced).",
            "Same-day exclusions: fixture date inclusive (matches on the "
            "fixture date are excluded from history).",
            "Cold-start: NaN for rate features when no matches exist; "
            "matches_played = 0.",
            "Production DB opened read-only and SHA-256 gated.",
            "Only 3 seasons available; walk-forward limited to 2 folds.",
            "COVID-19 gap in 2019/20 is a documented stationarity "
            "discontinuity.",
            "No fabricating matches, odds, predictions, or results.",
        ],
    }

    _write_json(report, REPORT_DIR / f"phase26_6_ablation_{ts}.json")
    _write_txt(report, REPORT_DIR / f"phase26_6_ablation_{ts}.txt")
    print(f"\n[phase26_6] reports written to {REPORT_DIR}")
    print(f"[phase26_6] prod SHA-256 (after): {prod_after}  "
          f"unchanged={prod_before == prod_after}")
    return report


# ---------------------------------------------------------------------------
# Report writers
# ---------------------------------------------------------------------------

def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.ndarray,)):
        return o.tolist()
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, pd.DataFrame):
        return o.to_dict(orient="records")
    return str(o)


def _write_json(report: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(report, indent=2, default=_json_default,
                    ensure_ascii=False),
        encoding="utf-8",
    )


def _write_txt(report: dict, path: Path) -> None:
    L = []
    L.append("Phase 26.6 - Feature Ablation Study")
    L.append("=" * 50)
    L.append(f"Produced: {report['produced_at']}")
    L.append(f"Strategy: {report['config']['strategy']}")
    L.append(f"Seed: {report['config']['seed']}")
    L.append("")
    L.append("Dataset")
    L.append("-------")
    ds = report["dataset"]
    L.append(f"Source: {ds['source']}")
    L.append(f"Production DB SHA-256: {ds['production_db_sha256']}")
    L.append(f"Production DB unchanged: {ds['production_db_unchanged']}")
    L.append(f"Disposable DB SHA-256: {ds['disposable_db_sha256']}")
    L.append("")

    ps = report["poisson_sanity_check"]
    L.append("Poisson Sanity Check")
    L.append("--------------------")
    L.append(f"Status: {ps['status']}")
    L.append(f"Message: {ps['message']}")
    L.append("")

    # Config summary table
    L.append("Pooled Metric Summary (LR model)")
    L.append("=" * 70)
    L.append(f"{'Config':<24} {'#Feats':>7} {'Accuracy':>9} "
             f"{'Macro-F1':>9} {'LogLoss':>9} {'Brier':>9}")
    L.append("-" * 70)
    for cfg_name, res in report["configs"].items():
        lr = res["pooled"].get("logistic_regression", {})
        L.append(
            f"{cfg_name:<24} {res['n_features']:>7d} "
            f"{lr.get('accuracy', 0):>9.4f} "
            f"{lr.get('macro_f1', 0):>9.4f} "
            f"{lr.get('log_loss', 0):>9.4f} "
            f"{lr.get('brier_score', 0):>9.4f}"
        )
    L.append("")

    # Poisson pooled (should be same for all)
    L.append("Pooled Metric Summary (Poisson - invariant)")
    L.append("-" * 70)
    for cfg_name, res in report["configs"].items():
        pois = res["pooled"].get("poisson", {})
        L.append(
            f"{cfg_name:<24} "
            f"acc={pois.get('accuracy', 0):.4f} "
            f"brier={pois.get('brier_score', 0):.4f}"
        )
    L.append("")

    # Paired tests
    L.append("Paired Tests (Config B/C/D vs Config A - LR log-loss)")
    L.append("-" * 70)
    for pt in report["paired_tests"]:
        pair = pt["pair"]
        bt = pt["bootstrap"]
        mc = pt["mcnemar"]
        direction = "B better" if bt["mean_diff"] > 0 else "B worse"
        if bt["ci_95"][0] > 0 or bt["ci_95"][1] < 0:
            sig = "significant"
        else:
            sig = "not significant"
        L.append(
            f"  {pair[0]} -> {pair[1]}: mean_diff={bt['mean_diff']:+.6f}, "
            f"95%CI=[{bt['ci_95'][0]:+.6f}, {bt['ci_95'][1]:+.6f}], "
            f"p_better_b={bt['p_better_b']:.3f} ({direction}, {sig})"
        )
        L.append(
            f"    McNemar: p={mc.get('p_value', 'N/A')}, "
            f"significant={mc.get('significant', 'N/A')}"
        )
    L.append("")

    # Per-fold detail
    L.append("Per-Fold Details")
    L.append("----------------")
    for cfg_name in report["per_fold_results"]:
        for fr in report["per_fold_results"][cfg_name]:
            L.append(
                f"\n  Config {cfg_name} | {fr['fold_id']} | "
                f"n_train={fr['n_train']} n_test={fr['n_test']} "
                f"n_feats={fr['n_features']}"
            )
            for model_name in ["logistic_regression", "poisson",
                               "ens_poisson_lr_equal",
                               "ens_poisson_lr_optimal"]:
                if model_name in fr["models"]:
                    m = fr["models"][model_name]
                    L.append(
                        f"    {model_name}: "
                        f"acc={m['accuracy']:.4f} "
                        f"macroF1={m['macro_f1']:.4f} "
                        f"logloss={m['log_loss']:.4f} "
                        f"brier={m['brier_score']:.4f}"
                    )

    L.append("")
    L.append("Notes")
    L.append("-----")
    for note in report["notes"]:
        L.append(f"- {note}")

    Path(path).write_text("\n".join(L), encoding="utf-8")


if __name__ == "__main__":
    run_phase26_6_ablation()
