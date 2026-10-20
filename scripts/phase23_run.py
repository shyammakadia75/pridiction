#!/usr/bin/env python3
"""Phase 23 (Steps 2-4): isolated multi-season evaluation of football prediction
models on strictly-verified real data, production DB read-only.

Pipeline (all on a DISPOSABLE database -- `data/phase23_eval.db`, gitignored):
  1. Verify the 3 staged FD E0 CSVs (sha256/size) -- fail-closed.
  2. Build/refresh the disposable DB from the staged CSVs (reuse ingestor).
  3. Load matches (prod SHA-256 gated before/after every prod-touching op).
  4. Compute leakage-safe pre-match features (FeatureEngineer: history is
     strictly older than the target fixture's date; same-day matches excluded).
  5. Season-aligned CHRONOLOGICAL split (no shuffling):
        train = 2015/16 (380)   |  dates 2015-08-08 .. 2016-05-17
        val   = 2016/17 (380)   |  dates 2016-08-13 .. 2017-05-21   (model selection only)
        test  = 2019/20 (380)   |  dates 2019-08-09 .. 2020-07-26   (includes COVID gap)
     train < val < test by date; test is untouched during model selection.
  6. Baselines: Frequency, multinomial Logistic Regression, Elo, independent
     Poisson.
  7. Improvements: K-tuned Elo, regularized LR + form-interaction features,
     Dixon-Coles low-score correction, XGBoost.
  8. Fair evaluation (accuracy, macro-F1, per-class P/R, log-loss, Brier,
     calibration, goal metrics) + paired McNemar / bootstrap significance.
  9. Save JSON + readable TXT to reports/phase23/ ; re-verify prod hash.

Run:  .venv\\Scripts\\python.exe scripts/phase23_run.py
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from football_intelligence.database import resolve_db_path  # noqa: E402
from football_intelligence.features.engineering import (  # noqa: E402
    FeatureEngineer,
    load_matches_for_features,
)
from football_intelligence.prediction.baseline import (  # noqa: E402
    N_OUTCOMES,
    OUTCOME_LABELS,
    OUTCOME_INDEX,
    FrequencyBaseline,
    LogisticRegressionClassifier,
    chronological_split,
    compute_accuracy,
    compute_brier_score,
    compute_log_loss,
    prepare_training_data,
)
from football_intelligence.prediction.evaluation import (  # noqa: E402
    classification_report,
    compute_calibration,
    compute_macro_f1,
    compute_precision_recall,
    feature_columns,
)
from football_intelligence.prediction.elo import EloModel  # noqa: E402
from football_intelligence.prediction.poisson import (  # noqa: E402
    DEFAULT_MAX_GOALS,
    PoissonModel,
    run_poisson_chronological,
)
from football_intelligence.prediction.dixon_coles import DixonColesModel  # noqa: E402

PROD = REPO / "data" / "football_intelligence.db"
DISPOS = REPO / "data" / "phase23_eval.db"
REPORT_DIR = REPO / "reports" / "phase23"
PROD_SHA256 = "725af574b88dc033bd3b00928de503f9bd9a836fbe4252832b7812dce0ab3122"

# Season-aligned date windows (YYYY-MM-DD string comparison on ISO dates).
TRAIN_MAX = "2016-07-31"   # 2015/16 ends 2016-05-17
VAL_MAX = "2019-07-31"     # 2016/17 ends 2017-05-21; gap to 2019/20 start
LABEL_IDX = {lbl: i for i, lbl in enumerate(OUTCOME_LABELS)}


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def prod_hash_ok() -> bool:
    if not PROD.exists():
        return False
    return sha256_file(PROD) == PROD_SHA256


# ---- model-eval helpers ----------------------------------------------------

def proba_metrics(y_true_idx, probs) -> dict:
    """Full 3-class report for models emitting proper probabilities."""
    y_true_idx = np.asarray(y_true_idx, dtype=int)
    probs = np.asarray(probs, dtype=float)
    y_pred = np.argmax(probs, axis=1).astype(int)
    base = classification_report(y_true_idx, probs, y_pred=y_pred)
    return {
        "n": int(len(y_true_idx)),
        "accuracy": float(compute_accuracy(y_true_idx, y_pred)),
        "macro_f1": float(compute_macro_f1(y_true_idx, y_pred)),
        "log_loss": float(compute_log_loss(y_true_idx, probs)),
        "brier_score": float(compute_brier_score(y_true_idx, probs)),
        "per_class_precision_recall": compute_precision_recall(y_true_idx, y_pred),
        "confusion_matrix": base["confusion_matrix"],
        "class_distribution": base["class_distribution"],
        "calibration": compute_calibration(y_true_idx, probs, n_bins=5),
    }


# ===========================================================================
# Step 2: data load + leakage-safe features + chronological split
# ===========================================================================


def load_raw_matches() -> pd.DataFrame:
    """Load all FD matches from the DISPOSABLE db (production DB NOT touched)."""
    assert prod_hash_ok(), "prod DB hash mismatch before load"
    raw = load_matches_for_features(
        resolve_db_path(str(DISPOS)),
        provider="football-data",
        provider_competition_id="E0",
    )
    assert prod_hash_ok(), "prod DB hash changed during load (fail-closed)"
    raw = raw.sort_values(["match_date", "home_team_id", "away_team_id"])
    raw = raw.reset_index(drop=True)
    return raw


def _add_interactions(eng: pd.DataFrame) -> pd.DataFrame:
    """Add pre-match form interaction features.

    Deterministic functions of already-computed pre-match features, so
    leakage-safety is preserved (history is already strictly-before)."""
    eng = eng.copy()

    def edge(a: str, b: str) -> None:
        if a in eng.columns and b in eng.columns:
            eng[f"{a}_minus_{b}"] = eng[a].fillna(0) - eng[b].fillna(0)

    for a, b in [
        ("home_points_last_5", "away_points_last_5"),
        ("home_goal_diff_last_5", "away_goal_diff_last_5"),
        ("home_goals_scored_last_5", "away_goals_conceded_last_5"),
        ("home_goals_conceded_last_5", "away_goals_scored_last_5"),
        ("home_points_last_10", "away_points_last_10"),
    ]:
        edge(a, b)
    if {"home_days_since_last_match", "away_days_since_last_match"}.issubset(eng.columns):
        eng["rest_advantage"] = (
            eng["away_days_since_last_match"].fillna(0)
            - eng["home_days_since_last_match"].fillna(0)
        )
    return eng


def build_feature_dataset(raw: pd.DataFrame, use_interactions: bool = False):
    """Leakage-safe feature matrix + labels -> (eng, feats, (train,val,test))."""
    engineer = FeatureEngineer()
    eng, _X, _y = prepare_training_data(raw, engineer=engineer)
    if use_interactions:
        eng = _add_interactions(eng)
        feats = feature_columns(eng)

    d = eng["target_match_date"].astype(str)
    train_df = eng[d < TRAIN_MAX].copy().reset_index(drop=True)
    val_df = eng[(d >= TRAIN_MAX) & (d < VAL_MAX)].copy().reset_index(drop=True)
    test_df = eng[d >= VAL_MAX].copy().reset_index(drop=True)
    total = len(train_df) + len(val_df) + len(test_df)
    assert total == len(eng), f"split lost rows: {total} != {len(eng)}"
    tm = str(train_df["target_match_date"].max()) if len(train_df) else "0"
    tmin = str(val_df["target_match_date"].min()) if len(val_df) else "9"
    tmax = str(val_df["target_match_date"].max()) if len(val_df) else "9"
    te_min = str(test_df["target_match_date"].min()) if len(test_df) else "9"
    assert tm <= tmin, f"train/val chronology broken: {tm} > {tmin}"
    assert tmax <= te_min, f"val/test chronology broken: {tmax} > {te_min}"
    return eng, feats, (train_df, val_df, test_df)


# ===========================================================================
# Outcome + metric helpers
# ===========================================================================


def _outcome_idx(home_goals, away_goals):
    try:
        hg = int(round(float(home_goals)))
        ag = int(round(float(away_goals)))
    except (TypeError, ValueError):
        return None
    if hg > ag:
        return OUTCOME_INDEX["home_win"]
    if hg == ag:
        return OUTCOME_INDEX["draw"]
    return OUTCOME_INDEX["away_win"]


def _mat(df, feats):
    return df[feats].to_numpy(dtype=float)


def _y(df):
    return df["_label_idx"].to_numpy(dtype=int)


from football_intelligence.prediction.baseline import (  # noqa: E402
    build_confusion_matrix,
    class_distribution,
)


def eval_proba(y_true_idx, proba) -> dict:
    """Full 3-class report for models emitting proper probabilities."""
    y_true_idx = np.asarray(y_true_idx, dtype=int)
    proba = np.asarray(proba, dtype=float)
    y_pred = np.argmax(proba, axis=1).astype(int)
    return {
        "n": int(len(y_true_idx)),
        "accuracy": float(compute_accuracy(y_true_idx, y_pred)),
        "macro_f1": float(compute_macro_f1(y_true_idx, y_pred)),
        "log_loss": float(compute_log_loss(y_true_idx, proba)),
        "brier_score": float(compute_brier_score(y_true_idx, proba)),
        "per_class_precision_recall": compute_precision_recall(y_true_idx, y_pred),
        "confusion_matrix": build_confusion_matrix(y_true_idx, y_pred, OUTCOME_LABELS).tolist(),
        "class_distribution": class_distribution(y_true_idx, OUTCOME_LABELS),
        "calibration": compute_calibration(y_true_idx, proba, n_bins=5),
    }


def baseline_frequency(train_df, test_df) -> dict:
    fb = FrequencyBaseline().fit(_y(train_df))
    proba = fb.predict_proba(np.zeros((len(test_df), 1)))
    return eval_proba(_y(test_df), proba)


# ===========================================================================
# Step 3: baselines + improvements
# ===========================================================================


def baseline_logistic(train_df, feats, seed=42) -> dict:
    """Fixed-config multinomial LR (Phase-21 convention: fit on train only)."""
    lr = LogisticRegressionClassifier(
        learning_rate=0.05, n_iterations=5000, l2_lambda=0.01, random_state=seed,
    ).fit(_mat(train_df, feats), _y(train_df))
    cfg = {"type": "logistic_regression", "l2_lambda": 0.01,
           "learning_rate": 0.05, "n_iterations": 5000, "random_state": seed,
           "form_interaction": False}
    return {"model": lr, "config": cfg}


def improved_logistic(train_df, val_df, feats, seed=42) -> dict:
    """Regularized LR + form-interaction features; L2 tuned on val.

    Hyperparameters are selected on the validation window only; the final
    model is refit on train+val and scored exclusively on test.
    """
    grid = [0.001, 0.01, 0.1, 1.0]
    best_l2, best_acc = grid[0], -1.0
    for l2 in grid:
        m = LogisticRegressionClassifier(
            learning_rate=0.05, n_iterations=5000, l2_lambda=l2, random_state=seed,
        ).fit(_mat(train_df, feats), _y(train_df))
        vp = m.predict_proba(_mat(val_df, feats))
        acc = compute_accuracy(_y(val_df), np.argmax(vp, axis=1))
        if acc > best_acc:
            best_acc, best_l2 = float(acc), l2
    tv = pd.concat([train_df, val_df], ignore_index=True)
    m = LogisticRegressionClassifier(
        learning_rate=0.05, n_iterations=5000, l2_lambda=best_l2, random_state=seed,
    ).fit(_mat(tv, feats), _y(tv))
    cfg = {"type": "logistic_regression", "l2_lambda": best_l2,
           "learning_rate": 0.05, "n_iterations": 5000, "random_state": seed,
           "form_interaction": True, "l2_grid": grid,
           "val_accuracy": float(best_acc)}
    return {"model": m, "config": cfg}


def _elo_history_df(m: EloModel) -> pd.DataFrame:
    rows = []
    for em in m.history_:
        rows.append({
            "date": str(em.match_date),
            "home_team_id": int(em.home_team_id),
            "away_team_id": int(em.away_team_id),
            "expected_home": float(em.expected_home),
            "actual_home": float(em.actual_home) if em.actual_home is not None else np.nan,
            "outcome_label": em.outcome_label,
        })
    return pd.DataFrame(rows)


def baseline_elo(raw: pd.DataFrame, date_lo: str, date_hi: str, K=20.0) -> dict:
    """Online same-day-safe Elo (Phase-21 fix); scored on [date_lo, date_hi]."""
    m = EloModel(K=K, home_advantage=50.0)
    m.fit(raw.sort_values(["match_date", "home_team_id", "away_team_id"])
          .reset_index(drop=True))
    h = _elo_history_df(m)
    sub = h[(h["date"] >= date_lo) & (h["date"] <= date_hi)]
    exp = sub["expected_home"].to_numpy(dtype=float)
    act = sub["actual_home"].to_numpy(dtype=float)
    valid = np.isfinite(act)
    exp_v, act_v = exp[valid], act[valid]
    yt = np.array([LABEL_IDX[l] for l in sub["outcome_label"][valid]])
    yp = np.where(exp_v >= 0.60, 0, np.where(exp_v <= 0.40, 2, 1))
    return {
        "metrics": {
            "n": int(len(sub)),
            "accuracy": float(compute_accuracy(yt, yp)),
            "macro_f1": float(compute_macro_f1(yt, yp)),
            "brier_expected_score": float(np.mean((exp_v - act_v) ** 2)),
            "per_class_precision_recall": compute_precision_recall(yt, yp),
            "confusion_matrix": build_confusion_matrix(yt, yp, OUTCOME_LABELS).tolist(),
            "class_distribution": class_distribution(yt, OUTCOME_LABELS),
        },
        "goal_metrics": {},
        "config": {"type": "elo_expected_score_thresholded", "K": float(K),
                   "home_advantage": 50.0,
                   "home_win_threshold": 0.60, "away_win_threshold": 0.40},
    }


def elo_tune_K(raw_for_tune: pd.DataFrame, val_lo: str, val_hi: str) -> dict:
    """Select Elo K on validation via macro-F1; returns best K + per-K scores."""
    grid = [10.0, 20.0, 40.0, 60.0]
    scores = {}
    for K in grid:
        m = EloModel(K=K, home_advantage=50.0)
        m.fit(raw_for_tune)
        scores[K] = baseline_elo(raw_for_tune, val_lo, val_hi, K=K)["metrics"]
    best = max(scores, key=lambda k: scores[k]["macro_f1"])
    return {"best_K": float(best), "per_K_val": {str(k): scores[k] for k in scores}}


def _goal_metrics(preds: list) -> dict:
    exp_total = np.array([p["home_rate"] + p["away_rate"] for p in preds], dtype=float)
    act_total = np.array([float(p["home_goals"]) + float(p["away_goals"]) for p in preds])
    over_pred = np.array([p.get("over_2_5", 0.0) for p in preds], dtype=float)
    over_actual = (act_total > 2.5).astype(float)
    return {
        "expected_total_goals_mae": float(np.mean(np.abs(exp_total - act_total))),
        "mean_actual_total_goals": float(np.mean(act_total)),
        "mean_predicted_total_goals": float(np.mean(exp_total)),
        "over_2_5_brier": float(np.mean((over_pred - over_actual) ** 2)),
    }


def baseline_poisson(raw: pd.DataFrame, date_lo: str, date_hi: str) -> dict:
    """Independent-Poisson baseline (Phase-20); scored on [date_lo, date_hi]."""
    model = PoissonModel()
    preds, _summary = run_poisson_chronological(raw, model=model)
    sub = [p for p in preds if date_lo <= p["match_date"] <= date_hi]
    proba = np.array([[p["probs"][lbl] for lbl in OUTCOME_LABELS] for p in sub])
    yt = np.array([p["outcome_idx"] for p in sub], dtype=int)
    m = eval_proba(yt, proba)
    gm = _goal_metrics(sub)
    return {"metrics": m, "goal_metrics": gm,
            "config": {"type": "independent_poisson",
                       "max_goals": model.max_goals,
                       "min_history": model.min_history,
                       "prior_strength": model.prior_strength}}

def _run_dixon_coles(raw: pd.DataFrame, model: DixonColesModel) -> list:
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


def improved_dixon_coles(raw_full: pd.DataFrame, rho: float,
                         date_lo: str, date_hi: str) -> dict:
    """Dixon-Coles with rho fit on TRAIN; predict scored on [date_lo, date_hi]."""
    model = DixonColesModel(rho=rho)
    preds = _run_dixon_coles(raw_full, model)
    sub = [p for p in preds if date_lo <= p["match_date"] <= date_hi]
    proba = np.array([[p["probs"][lbl] for lbl in OUTCOME_LABELS] for p in sub])
    yt = np.array([p["outcome_idx"] for p in sub], dtype=int)
    m = eval_proba(yt, proba)
    gm = _goal_metrics(sub)
    return {"metrics": m, "goal_metrics": gm,
            "config": {"type": "dixon_coles", "rho": float(rho),
                       "max_goals": model.max_goals}}


def improved_xgboost(train_df, val_df, feats, seed=42) -> dict:
    """XGBoost (xgboost 3.4.1 installed); early-stopped, proba-calibrated class."""
    import xgboost as xgb
    Xtr, ytr = _mat(train_df, feats), _y(train_df)
    Xva, yva = _mat(val_df, feats), _y(val_df)
    n = len(np.unique(np.concatenate([ytr, yva])))
    best, best_params, best_acc = None, None, -1.0
    for max_depth in (3, 6):
        for eta in (0.05, 0.1):
            clf = xgb.XGBClassifier(
                max_depth=max_depth, learning_rate=eta, n_estimators=300,
                objective="multi:softprob", num_class=n, eval_metric="mlogloss",
                early_stopping_rounds=20, seed=seed, n_jobs=1, verbosity=0,
                tree_method="hist",
            )
            clf.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
            vp = clf.predict_proba(Xva)
            acc = compute_accuracy(yva, np.argmax(vp, axis=1))
            if acc > best_acc:
                best, best_acc, best_params = clf, acc, {"max_depth": max_depth, "eta": eta}
    tv = pd.concat([train_df, val_df], ignore_index=True)
    clf = xgb.XGBClassifier(
        max_depth=best_params["max_depth"], learning_rate=best_params["eta"],
        n_estimators=300, objective="multi:softprob", num_class=n,
        eval_metric="mlogloss", early_stopping_rounds=20, seed=seed,
        n_jobs=1, verbosity=0, tree_method="hist",
    )
    clf.fit(_mat(tv, feats), _y(tv), eval_set=[(Xva, yva)], verbose=False)
    cfg = {"type": "xgboost", "max_depth": best_params["max_depth"],
           "eta": best_params["eta"], "n_estimators": 300, "seed": seed,
           "early_stopping_rounds": 20, "val_accuracy": float(best_acc)}
    return {"model": clf, "config": cfg}

# ===========================================================================
# Step 4-5: orchestrate + fair evaluation + significance
# ===========================================================================
from football_intelligence.prediction.comparison import (  # noqa: E402
    mcp_mcnemar_exact,
    paired_bootstrap,
)

N_BOOT = 2000
BOOT_SEED = 42
LABELS = list(OUTCOME_LABELS)
# COVID-19 English football suspension: 13 Mar 2020 -> 17 Jun 2020.
COVID_PAUSE_START = "2020-03-10"
COVID_RESTART = "2020-06-17"
TEST_LO, TEST_HI = "2019-08-09", "2020-07-26"
VAL_LO, VAL_HI = "2016-08-13", "2017-05-21"


def _save_json(report, path):
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, default=_json_serial)
    return str(path)


def _json_serial(o):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        f = float(o)
        return None if (np.isnan(f) or np.isinf(f)) else f
    if isinstance(o, (pd.Timestamp,)):
        return o.isoformat()
    if isinstance(o, float) and (np.isnan(o) or np.isinf(o)):
        return None
        return str(o)


def _keyseries(df, datecol, hcol, acol):
    return (df[datecol].astype(str)
            + "|" + df[hcol].astype(int).astype(str)
            + "|" + df[acol].astype(int).astype(str))


def _align_by_keys(full_preds, test_df):
    """Map an online model's full prediction stream onto the test fixture order
    using the canonical (date, home, away) key -> leakage-safe alignment."""
    by_key = {(str(p["match_date"]), int(p["home_team_id"]), int(p["away_team_id"])): p
              for p in full_preds}
    keys = [(str(r.target_match_date), int(r.home_team_id), int(r.away_team_id))
            for r in test_df.itertuples(index=False)]
    aligned = [by_key[k] for k in keys]
    missing = [k for k in keys if k not in by_key]
    assert not missing, f"{len(missing)} test fixtures missing from model preds: {missing[:3]}"
    return aligned


def _proba_from_preds(aligned, order=OUTCOME_LABELS):
    return np.array([[p["probs"][l] for l in order] for p in aligned], dtype=float)


def _elo_aligned(elo_df, test_df):
    """Attach Elo expected_home/actual to test rows by fixture key."""
    t = test_df.copy()
    t["_key"] = _keyseries(t, "target_match_date", "home_team_id", "away_team_id")
    e = elo_df.copy()
    e["_key"] = _keyseries(e, "date", "home_team_id", "away_team_id")
    merged = t.merge(e[["_key", "expected_home", "actual_home", "outcome_label"]],
                     on="_key", how="left")
    return merged


def run(dispos_path: Path = DISPOS, seed: int = 42) -> dict:
    """End-to-end Phase-23 evaluation. Returns the report dict; also saved."""
    assert prod_hash_ok(), "prod DB hash mismatch at run start (fail-closed)"
    sha_before = sha256_file(PROD)
    print("[1/7] load raw matches (prod read-only)...")
    raw = load_raw_matches()
    assert prod_hash_ok(), "prod DB changed during load (fail-closed)"

    eng, feats, (train_df, val_df, test_df) = build_feature_dataset(raw, use_interactions=True)
    test_df = test_df.sort_values(["target_match_date", "home_team_id", "away_team_id"])
    test_df = test_df.reset_index(drop=True)
    yt = _y(test_df)
    test_dates = test_df["target_match_date"].astype(str).to_numpy()
    print(f"      train={len(train_df)} val={len(val_df)} test={len(test_df)} "
          f"features={len(feats)} prod-sha={sha_before[:12]}")

    print("[2/7] fit Dixon-Coles rho on TRAIN (MLE)...")
    train_raw = raw[raw["match_date"].astype(str) < TRAIN_MAX].copy()
    train_raw = train_raw.sort_values(["match_date", "home_team_id", "away_team_id"])
    train_raw = train_raw.reset_index(drop=True)
    dc_model = DixonColesModel(rho=None).fit_rho(train_raw)

    print("[3/7] select Elo K on validation (macro-F1)...")
    raw_train_val = raw[raw["match_date"].astype(str) < VAL_MAX].copy()
    raw_train_val = raw_train_val.sort_values(
        ["match_date", "home_team_id", "away_team_id"]).reset_index(drop=True)
    elo_sel = elo_tune_K(raw_train_val, VAL_LO, VAL_HI)
    best_K = elo_sel["best_K"]

    # Containers: per-model aligned TEST arrays.
    proba: dict = {}   # name -> (n_test, 3) float array (None for Elo-only)
    pred_arr: dict = {}  # name -> (n_test,) int
    goal_metrics: dict = {}
    print("[4/7] score baselines (Frequency, LR, Poisson, Elo)...")
    fb = FrequencyBaseline().fit(_y(train_df))
    fp = np.tile(fb._class_proba, (len(test_df), 1))
    proba["frequency_baseline"] = fp; pred_arr["frequency_baseline"] = fp.argmax(1)

    lr_res = baseline_logistic(train_df, feats, seed=seed)
    lr = lr_res["model"]
    lp = lr.predict_proba(_mat(test_df, feats))
    proba["logistic_regression"] = lp; pred_arr["logistic_regression"] = lp.argmax(1)

    raw_sorted = raw.sort_values(["match_date", "home_team_id", "away_team_id"])
    pois_model = PoissonModel()
    pois_full, _ = run_poisson_chronological(raw_sorted, model=pois_model)
    pois_cfg = {"type": "independent_poisson",
                "max_goals": pois_model.max_goals,
                "min_history": pois_model.min_history,
                "prior_strength": pois_model.prior_strength}
    pa = _align_by_keys(pois_full, test_df)
    pp = _proba_from_preds(pa)
    proba["poisson"] = pp; pred_arr["poisson"] = pp.argmax(1)
    goal_metrics["poisson"] = _goal_metrics(pa)

    elo_b_full = _elo_history_df(
        EloModel(K=20.0, home_advantage=50.0).fit(raw)
    )
    elo_b_merged = _elo_aligned(elo_b_full, test_df)
    elo_i_full = _elo_history_df(
        EloModel(K=best_K, home_advantage=50.0).fit(raw)
    )
    elo_i_merged = _elo_aligned(elo_i_full, test_df)

    print("[5/7] score improvements (LR+interaction, DC, Elo-tuned, XGBoost)...")
    # Logistic + form-interaction (regularized, L2 tuned on val)
    lri_res = improved_logistic(train_df, val_df, feats, seed=seed)
    lri = lri_res["model"]
    lpp = lri.predict_proba(_mat(test_df, feats))
    proba["logistic_interactions"] = lpp
    pred_arr["logistic_interactions"] = lpp.argmax(1)

    # Dixon-Coles (rho from TRAIN only)
    dc_full = _run_dixon_coles(raw, dc_model)
    da = _align_by_keys(dc_full, test_df)
    dcp = _proba_from_preds(da)
    proba["dixon_coles"] = dcp; pred_arr["dixon_coles"] = dcp.argmax(1)
    goal_metrics["dixon_coles"] = _goal_metrics(da)

    # XGBoost (tuned on val)
    xgb_res = improved_xgboost(train_df, val_df, feats, seed=seed)
    xgb_r = xgb_res["model"]
    xp = xgb_r.predict_proba(_mat(test_df, feats))
    proba["xgboost"] = xp; pred_arr["xgboost"] = xp.argmax(1)

    # ---- Elo test metrics (online; thresholded from expected_home score) ----
    def _elo_test_metrics(merged):
        exp = merged["expected_home"].to_numpy(dtype=float)
        act = merged["actual_home"].to_numpy(dtype=float)
        ok = np.isfinite(act)
        e, a = exp[ok], act[ok]
        yt_e = merged["_label_idx"].to_numpy()[ok].astype(int)
        yp = np.where(e >= 0.60, 0, np.where(e <= 0.40, 2, 1))
        return {
            "n": int(len(merged)),
            "accuracy": float(compute_accuracy(yt_e, yp)),
            "macro_f1": float(compute_macro_f1(yt_e, yp)),
            "brier_expected_score": float(np.mean((e - a) ** 2)) if len(e) else 0.0,
            "per_class_precision_recall": compute_precision_recall(yt_e, yp),
            "confusion_matrix": build_confusion_matrix(
                yt_e, yp, OUTCOME_LABELS).tolist(),
            "class_distribution": class_distribution(yt_e, OUTCOME_LABELS),
        }

    elo_b_metrics = _elo_test_metrics(elo_b_merged)
    elo_i_metrics = _elo_test_metrics(elo_i_merged)
    # =====================================================================
    # Per-model TEST metrics + COVID sub-windows
    # =====================================================================
    PROBA_MODELS = ["frequency_baseline", "logistic_regression",
                    "logistic_interactions", "poisson", "dixon_coles", "xgboost"]
    model_metrics: dict = {}
    sub_windows: dict = {}

    pre_mask = test_dates < COVID_PAUSE_START
    post_mask = test_dates >= COVID_RESTART

    def _sub_acc(y, p, mask):
        n = int(mask.sum())
        if n == 0:
            return {"n": 0}
        return {"n": n, "accuracy": float(compute_accuracy(y[mask], np.argmax(p, axis=1)[mask]))}

    def _elo_sub(merged, mask):
        n = int(mask.sum())
        if n == 0:
            return {"n": 0}
        exp_all = merged["expected_home"].to_numpy(dtype=float)[mask]
        act_all = merged["actual_home"].to_numpy(dtype=float)[mask]
        lbl_all = merged["_label_idx"].to_numpy()[mask].astype(int)
        ok = np.isfinite(act_all)
        e, a = exp_all[ok], act_all[ok]
        yt_e = lbl_all[ok]
        yp = np.where(e >= 0.60, 0, np.where(e <= 0.40, 2, 1))
        return {"n": n,
                "accuracy": float(compute_accuracy(yt_e, yp)) if len(yt_e) else 0.0,
                "brier_expected_score": float(np.mean((e - a) ** 2)) if len(e) else 0.0}

    def _elo_pred(merged):
        exp = merged["expected_home"].to_numpy(dtype=float)
        return np.where(exp >= 0.60, 0, np.where(exp <= 0.40, 2, 1)).astype(int)

    for name in PROBA_MODELS:
        m = proba_metrics(yt, proba[name])
        model_metrics[name] = m
        sub_windows[name] = {
            "pre_COVID_pause": _sub_acc(yt, proba[name], pre_mask),
            "post_COVID_restart": _sub_acc(yt, proba[name], post_mask),
        }
    model_metrics["elo"] = elo_b_metrics
    model_metrics["elo_tuned"] = elo_i_metrics
    sub_windows["elo"] = {"pre": _elo_sub(elo_b_merged, pre_mask),
                          "post": _elo_sub(elo_b_merged, post_mask)}
    sub_windows["elo_tuned"] = {"pre": _elo_sub(elo_i_merged, pre_mask),
                                "post": _elo_sub(elo_i_merged, post_mask)}

    for name in PROBA_MODELS:
        gm = goal_metrics.get(name)
        if gm:
            model_metrics[name]["goal_metrics"] = gm

    # =====================================================================
    # Paired significance (identical test fixtures)
    # =====================================================================
    print("[6/7] paired McNemar + bootstrap significance...")
    onehot = np.zeros((len(yt), 3)); onehot[np.arange(len(yt)), yt] = 1.0
    brier_per_sample = {n: np.sum((proba[n] - onehot) ** 2, axis=1) for n in PROBA_MODELS}
    loss_per_sample = {n: (np.argmax(proba[n], axis=1) != yt).astype(float) for n in PROBA_MODELS}

    proba_pairs = [
        ("logistic_regression", "logistic_interactions"),
        ("poisson", "dixon_coles"),
        ("logistic_regression", "poisson"),
        ("logistic_regression", "dixon_coles"),
        ("logistic_regression", "xgboost"),
        ("poisson", "xgboost"),
        ("dixon_coles", "xgboost"),
    ]
    paired = []
    for a, b in proba_pairs:
        pa = mcp_mcnemar_exact(yt, pred_arr[a], pred_arr[b], LABELS)
        pb_acc = paired_bootstrap(loss_per_sample[a], loss_per_sample[b],
                                  n_boot=N_BOOT, seed=BOOT_SEED)
        pb_brier = paired_bootstrap(brier_per_sample[a], brier_per_sample[b],
                                    n_boot=N_BOOT, seed=BOOT_SEED)
        paired.append({"pair": [a, b], "metric": "accuracy",
                       "mcnemar": pa, "bootstrap_accuracy": pb_acc,
                       "bootstrap_brier": pb_brier})
    # Elo vs LR (accuracy only; Elo has no calibrated 3-class proba)
    for a, b in [("logistic_regression", "elo"), ("elo", "elo_tuned")]:
        ya = pred_arr[a] if a in pred_arr else _elo_pred(elo_b_merged if a == "elo"
                                                         else elo_i_merged)
        yb = pred_arr[b] if b in pred_arr else _elo_pred(elo_b_merged if b == "elo"
                                                         else elo_i_merged)
        paired.append({"pair": [a, b], "metric": "accuracy",
                       "mcnemar": mcp_mcnemar_exact(yt, ya, yb, LABELS),
                       "bootstrap_accuracy": paired_bootstrap(
                           (ya != yt).astype(float), (yb != yt).astype(float),
                           n_boot=N_BOOT, seed=BOOT_SEED),
                       "bootstrap_brier": None})
    print("[7/7] assembling + saving report...")
    sha_after = sha256_file(PROD)
    prod_ok = sha_after == sha_before

    ranking = sorted(PROBA_MODELS,
                     key=lambda n: model_metrics[n]["macro_f1"], reverse=True)
    elo_b_cfg = {"type": "elo_expected_score_thresholded", "K": 20.0,
                 "home_advantage": 50.0, "home_win_threshold": 0.60,
                 "away_win_threshold": 0.40}
    elo_i_cfg = {"type": "elo_expected_score_thresholded", "K_selected": float(best_K),
                 "home_advantage": 50.0, "home_win_threshold": 0.60,
                 "away_win_threshold": 0.40, "selected_on": "val"}
    dc_cfg = {"type": "dixon_coles", "rho": float(dc_model.rho_),
              "rho_fit_on": "train_mle", "max_goals": dc_model.max_goals}

    models_summary = [
        {"name": "frequency_baseline", "kind": "baseline",
         "config": {"type": "frequency",
                    "class_priors": [float(v) for v in fb._class_proba]},
         "metrics": model_metrics["frequency_baseline"]},
        {"name": "logistic_regression", "kind": "baseline",
         "config": lr_res["config"], "metrics": model_metrics["logistic_regression"]},
        {"name": "poisson", "kind": "baseline", "config": pois_cfg,
         "metrics": model_metrics["poisson"], "goal_metrics": goal_metrics["poisson"]},
        {"name": "elo", "kind": "baseline", "config": elo_b_cfg,
         "metrics": elo_b_metrics},
        {"name": "logistic_interactions", "kind": "improvement",
         "config": lri_res["config"],
         "metrics": model_metrics["logistic_interactions"]},
        {"name": "dixon_coles", "kind": "improvement", "config": dc_cfg,
         "metrics": model_metrics["dixon_coles"], "goal_metrics": goal_metrics["dixon_coles"]},
        {"name": "elo_tuned", "kind": "improvement", "config": elo_i_cfg,
         "metrics": elo_i_metrics},
        {"name": "xgboost", "kind": "improvement", "config": xgb_res["config"],
         "metrics": model_metrics["xgboost"]},
    ]

    report = {
        "phase": 23,
        "produced_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "config": {"seed": seed,
                   "split_rule": "season-aligned chronological (train=2015/16, "
                                 "val=2016/17, test=2019/20). No shuffling.",
                   "train_date_max": TRAIN_MAX, "val_date_max": VAL_MAX,
                   "covid_pause_start": COVID_PAUSE_START, "covid_restart": COVID_RESTART,
                   "elo_home_advantage": 50.0, "elo_thresholds": [0.60, 0.40],
                   "n_boot": N_BOOT, "boot_seed": BOOT_SEED, "xgb_available": True},
        "dataset": {"provider": "football-data", "provider_competition_id": "E0",
                    "seasons": ["2015/16", "2016/17", "2019/20"],
                    "n_matches": int(len(raw)),
                    "n_train": int(len(train_df)), "n_val": int(len(val_df)),
                    "n_test": int(len(test_df)), "n_features": int(len(feats)),
                    "train_date_range": [_s(train_df["target_match_date"].min()),
                                         _s(train_df["target_match_date"].max())],
                    "val_date_range": [_s(val_df["target_match_date"].min()),
                                       _s(val_df["target_match_date"].max())],
                    "test_date_range": [_s(test_df["target_match_date"].min()),
                                        _s(test_df["target_match_date"].max())],
                    "disposable_db": str(DISPOS),
                    "production_db_sha256": sha_before,
                    "production_db_unchanged": bool(prod_ok)},
        "models": models_summary,
        "sub_windows": sub_windows,
        "tuning": {"elo_K": elo_sel, "dixon_coles_rho": float(dc_model.rho_)},
        "paired": paired,
        "ranking_macro_f1": ranking,
        "class_labels": list(LABELS),
        "covid": {"pause_start": COVID_PAUSE_START, "restart": COVID_RESTART,
                  "note": "2019/20 test season spans the COVID-19 suspension "
                          "(13 Mar 2020 -> 17 Jun 2020); pre/post sub-windows "
                          "reported separately in sub_windows."},
        "notes": _REPORT_NOTES,
    }
    assert prod_ok, "PRODUCTION DB CHANGED during run (fail-closed)"
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    jp = REPORT_DIR / f"phase23_comparison_{ts}.json"
    tp = REPORT_DIR / f"phase23_comparison_{ts}.txt"
    _save_json(report, jp)
    _write_txt(report, tp)
    report["_artifact"] = {"json": str(jp), "txt": str(tp)}
    print(f"\nREPORT SAVED: {jp}")
    print(f"TXT SAVED:    {tp}")
    print(f"prod SHA-256 unchanged: {prod_ok}")
    return report


_REPORT_NOTES = [
    "Disposable DB (data/phase23_eval.db) built from 3 verified staged FD E0 "
    "CSV (2015/16, 2016/17, 2019/20); production DB is never written and its "
    "SHA-256 is gated before/after every touch.",
    "Chronological, season-aligned split: train=2015/16, val=2016/17, "
    "test=2019/20. No shuffling; train < val < test by date.",
    "Val is used ONLY for hyperparameter selection (Elo K, LR L2, XGBoost); "
    "final models are refit on train+val and scored on the untouched test set.",
    "Same-day / future leakage prevented everywhere: FeatureEngineer history is "
    "strictly older than the target date; Elo uses same-date batching; Poisson/DC "
    "use a strict less-than date cutoff.",
    "Elo reports its native brier_expected_score (2-class expected home score "
    "vs realised result) and is compared only on accuracy/macro-F1; it is NOT "
    "mixed with the proper 3-class Brier/log-loss used by LR/Poisson/DC/XGB.",
    "Dixon-Coles rho is a single global parameter fit by MLE on TRAIN only; "
    "team Poisson rates are re-estimated per fixture from strictly-earlier history.",
    "XGBoost is included because xgboost 3.4.1 is installed in the .venv; "
    "LightGBM/sklearn are not installed, so they are excluded as unjustified.",
]


def _s(x) -> str:
    if x is None:
        return ""
    return str(x)


def _write_txt(report: dict, path) -> None:
    L, R = [], report
    L.append("Phase 23 — Multi-Season Prediction Evaluation")
    L.append("============================================")
    L.append(f"Produced: {R['produced_at']}")
    d = R["dataset"]
    L.append("")
    L.append("Dataset / split")
    L.append("---------------")
    L.append(f"provider={d['provider']} competition={d['provider_competition_id']} "
             f"seasons={d['seasons']}")
    L.append(f"n_matches={d['n_matches']} train={d['n_train']} val={d['n_val']} "
             f"test={d['n_test']} n_features={d['n_features']}")
    L.append(f"train dates: {d['train_date_range']}")
    L.append(f"val   dates: {d['val_date_range']}")
    L.append(f"test  dates: {d['test_date_range']}")
    L.append(f"production DB sha256: {d['production_db_sha256']}")
    L.append(f"production DB unchanged: {d['production_db_unchanged']}")
    L.append(f"disposable DB: {d['disposable_db']}")
    L.append("")
    L.append("Tuning")
    L.append("------")
    t = R["tuning"]
    L.append(f"Dixon-Coles rho (fit on TRAIN): {t['dixon_coles_rho']}")
    ek = t["elo_K"]
    L.append(f"Elo K selected on val: {ek['best_K']}")
    for k, v in ek["per_K_val"].items():
        L.append(f"  K={k:>4s}: val_accuracy={v['accuracy']:.4f} "
                 f"macro_f1={v['macro_f1']:.4f} brier_exp={v['brier_expected_score']:.4f}")
    L.append("")
    L.append("Test-set results (sorted by macro-F1)")
    L.append("-------------------------------------")
    L.append(f"{'model':<28s} {'kind':<11s} {'n':>4s} {'acc':>6s} {'macroF1':>7s} "
             f"{'logloss':>8s} {'brier':>7s}")
    for m in R["models"]:
        mm = m["metrics"]
        kind = m["kind"]
        n = mm.get("n", mm.get("n_samples", 0))
        acc = mm.get("accuracy", 0.0) or 0.0
        if "log_loss" in mm:
            f1 = mm["macro_f1"]; ll = mm.get("log_loss", 0) or 0; br = mm.get("brier_score", 0) or 0
            L.append(f"{m['name']:<28s} {kind:<11s} {n:>4d} {acc:>6.3f} "
                     f"{f1:>7.3f} {ll:>8.3f} {br:>7.3f}")
        else:
            f1 = mm.get("macro_f1", 0); be = mm.get("brier_expected_score", 0)
            L.append(f"{m['name']:<28s} {kind:<11s} {n:>4d} {acc:>6.3f} "
                     f"{f1:>7.3f} {'n/a':>8s} {'n/a':>7s}   (brier_exp={be:.4f})")
    L.append("")
    L.append("Goal metrics (Poisson / Dixon-Coles)")
    L.append("-----------------------------------")
    for m in R["models"]:
        gm = m.get("goal_metrics") or {}
        if gm:
            L.append(f"  {m['name']}: MAE(exp,act)={gm.get('expected_total_goals_mae','?'):.3f} "
                     f"mean_actual={gm.get('mean_actual_total_goals',0):.2f} "
                     f"mean_pred={gm.get('mean_predicted_total_goals',0):.2f} "
                     f"over2.5_brier={gm.get('over_2_5_brier',0):.4f}")
    L.append("")
    L.append("COVID-19 sub-windows (test season)")
    L.append("-----------------------------------")
    sw = R["sub_windows"]
    for name, sub in sw.items():
        pre = sub.get("pre_COVID_pause") or sub.get("pre", {})
        post = sub.get("post_COVID_restart") or sub.get("post", {})
        pa = pre.get("accuracy", pre.get("n", 0))
        poa = post.get("accuracy", post.get("n", 0))
        L.append(f"  {name:<24s} pre(={pre.get('n',0)}): acc={pre.get('accuracy','?')}  "
                 f"post(={post.get('n',0)}): acc={post.get('accuracy','?')}")
    L.append("")
    L.append("Paired significance (accuracy; Elo has no 3-class Brier)")
    L.append("-------------------------------------------------------")
    for pr in R["paired"]:
        pair = "/".join(pr["pair"])
        mc = pr["mcnemar"]
        ba = pr["bootstrap_accuracy"]
        L.append(f"  {pair:<32s} McNemar p={mc.get('p_value',0):.4f} "
                 f"boot_diff={ba.get('mean_diff',0):.4f} "
                 f"95%CI=({ba.get('ci_low_95',0):.4f},{ba.get('ci_high_95',0):.4f})")
    L.append("")
    L.append("Notes")
    L.append("-----")
    for n in R["notes"]:
        L.append(f"- {n}")
    Path(path).write_text("\n".join(L) + "\n", encoding="utf-8")


if __name__ == "__main__":
    run()





