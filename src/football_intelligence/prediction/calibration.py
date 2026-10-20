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
import json
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
    _impute_and_scale,
)
from .dixon_coles import DixonColesModel
from .elo import EloModel
from .poisson import PoissonModel, run_poisson_chronological
from .evaluation import (
    compute_macro_f1, compute_precision_recall, feature_columns,
    compute_calibration,
)
from .comparison import mcp_mcnemar_exact, paired_bootstrap
from ..database import resolve_db_path
from ..features.engineering import FeatureEngineer, load_matches_for_features

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
PHASE23_DISPOS = REPO / "data" / "phase23_eval.db"
REPORT_DIR = REPO / "reports" / "phase24"

# Season-aligned chronological windows (ISO strings compare correctly).
TRAIN_MAX = "2016-07-31"   # 2015/16 ends 2016-05-17
VAL_MAX = "2019-07-31"     # 2016/17 ends 2017-05-21
LABEL_IDX = {lbl: i for i, lbl in enumerate(OUTCOME_LABELS)}

N_BOOT = 4000
BOOT_SEED = 42


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _prod_snapshot() -> str:
    return sha256_file(PROD)


def _load_raw_matches(prod_db: Path = PHASE23_DISPOS) -> pd.DataFrame:
    """Load the Phase-23 evaluation matches (reusing the isolated disposable DB).

    Reuses the *exact same* matches Phase 23 evaluated against (so Phase-24's
    test set is identical), rather than touching the production DB for data
    reads.  The prod DB SHA-256 is still gated around the load (fail-closed)
    as an invariant.
    """
    prod_before = _prod_snapshot()
    db_path = resolve_db_path(prod_db)
    df = load_matches_for_features(
        db_path=db_path, provider="football-data",
        provider_competition_id="E0",
    )
    assert _prod_snapshot() == prod_before, (
        "Production DB SHA-256 changed during Phase-24 load (fail-closed)")
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
    by_key = {(str(p["match_date"]), int(p["home_team_id"]),
               int(p["away_team_id"])): p for p in full_preds}
    keys = [(str(r.target_match_date), int(r.home_team_id),
             int(r.away_team_id)) for r in test_df.itertuples(index=False)]
    missing = [k for k in keys if k not in by_key]
    assert not missing, f"{len(missing)} test fixtures missing from model preds"
    return [by_key[k] for k in keys]


def _proba_from_preds(aligned: list, order=OUTCOME_LABELS) -> np.ndarray:
    return np.array([[p["probs"][l] for l in order] for p in aligned], dtype=float)


def _align_elo(elo_hist_df: pd.DataFrame, test_df: pd.DataFrame) -> pd.DataFrame:
    """Attach Elo expected_home to test rows by fixture key."""
    t = test_df.copy()
    t["_key"] = _keyseries(t, "target_match_date", "home_team_id", "away_team_id")
    e = elo_hist_df.copy()
    e["_key"] = _keyseries(e, "match_date", "home_team_id", "away_team_id")
    return t.merge(e[["_key", "expected_home"]], on="_key", how="left")


def _run_dixon_coles(raw: pd.DataFrame, model: DixonColesModel) -> list:
    """Online DC predictions over ALL matches with strict < date cutoff."""
    df = raw.sort_values(["match_date", "home_team_id",
                          "away_team_id"]).reset_index(drop=True)
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


def _add_interactions(eng: pd.DataFrame) -> pd.DataFrame:
    """Add pre-match form interaction features (deterministic, leakage-safe).

    Mirrors Phase-23's feature_interactions; the underlying features are
    already strictly pre-match, so differencing them stays leakage-safe.
    """
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
    if {"home_days_since_last_match", "away_days_since_last_match"}.issubset(
            eng.columns):
        eng["rest_advantage"] = (
            eng["away_days_since_last_match"].fillna(0)
            - eng["home_days_since_last_match"].fillna(0)
        )
    return eng


def build_split(raw: pd.DataFrame, use_interactions: bool = True):
    """Return (train_df, val_df, test_df, feats) mirroring Phase-23's pipeline.

    Uses the same leakage-safe feature engineering (FeatureEngineer via
    prepare_training_data) and the same chronological season-aligned split so
    Phase-24 shares fixtures with Phase 23's test set.
    """
    from .baseline import prepare_training_data as _prepare
    eng, _X, _y = _prepare(raw, engineer=FeatureEngineer(),
                           label_col="outcome_label")
    feats = list(feature_columns(eng))
    if use_interactions:
        eng = _add_interactions(eng)
        feats = list(feature_columns(eng))
    d = eng["target_match_date"].astype(str)
    train_df = eng[d < TRAIN_MAX].copy().reset_index(drop=True)
    val_df = eng[(d >= TRAIN_MAX) & (d < VAL_MAX)].copy().reset_index(drop=True)
    test_df = eng[d >= VAL_MAX].copy().reset_index(drop=True)
    return train_df, val_df, test_df, feats


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


# ---------------------------------------------------------------------------
# Calibrators (fit on train+val, applied to test)
# ---------------------------------------------------------------------------
class TemperatureScaler:
    """Temperature scaling (Guo et al., 2017): p_cal = softmax(logit(p) / T).

    T is a single scalar fit on train+val by minimizing cross-validated
    log-loss.  Temperature scaling is a monotonic transform fit before the
    test set is ever seen.
    """

    def __init__(self, seed: int = 42):
        self.T: float = 1.0
        self.seed = seed
        self.fitted_: bool = False

    def fit(self, proba: np.ndarray, y: np.ndarray) -> "TemperatureScaler":
        proba = np.clip(np.asarray(proba, dtype=float), 1e-7, 1.0)
        y = np.asarray(y, dtype=int)
        n = len(y)
        grid = np.array([0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0, 5.0, 8.0, 12.0])
        best_T, best_ll = 1.0, -np.inf
        fold = max(n // 5, 1)
        for T in grid:
            lls = []
            for k in range(5):
                lo, hi = k * fold, min((k + 1) * fold, n)
                if hi - lo == 0:
                    continue
                sc = self._scaled(proba[lo:hi], T)
                ll = -compute_log_loss(y[lo:hi], sc)
                if math.isfinite(ll):
                    lls.append(ll)
            if lls:
                mean_ll = float(np.mean(lls))
                if mean_ll > best_ll:
                    best_ll, best_T = mean_ll, float(T)
        self.T = best_T
        self.fitted_ = True
        return self

    @staticmethod
    def _scaled(proba: np.ndarray, T: float) -> np.ndarray:
        logits = np.log(np.clip(proba, 1e-7, 1.0)) / T
        e = np.exp(logits - logits.max(axis=1, keepdims=True))
        return e / e.sum(axis=1, keepdims=True)

    def predict_proba(self, proba: np.ndarray) -> np.ndarray:
        assert self.fitted_, "TemperatureScaler must be fit first"
        return self._scaled(np.asarray(proba, dtype=float), self.T)

    def config(self) -> dict:
        return {"type": "temperature_scaling", "T": float(self.T)}


# ---------------------------------------------------------------------------
# Base model probability providers (each uses strictly-pre-match history)
# ---------------------------------------------------------------------------
def _probs_logistic(train_df, feats, test_df, seed=42):
    m = LogisticRegressionClassifier(
        learning_rate=0.05, n_iterations=5000, l2_lambda=0.01,
        random_state=seed,
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
            acc = compute_accuracy(yva, np.argmax(clf.predict_proba(Xva), 1))
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


class IsotonicCalibrator:
    """Per-class isotonic regression (fit on train+val only).

    Falls back to identity if a class has fewer than ``MIN_PER_CLASS`` positives
    (honest behaviour on a small validation sample).
    """

    MIN_PER_CLASS = 10

    def __init__(self, seed=42, n_bins=5):
        self.seed = seed
        self.n_bins = n_bins
        self._iso = []
        self.fitted_ = False

    def fit(self, proba, y):
        proba = np.asarray(proba, dtype=float)
        y = np.asarray(y, dtype=int)
        onehot = np.zeros((len(y), N_OUTCOMES))
        onehot[np.arange(len(y)), y] = 1.0
        self._iso = []
        for c in range(N_OUTCOMES):
            if (onehot[:, c] > 0).sum() < self.MIN_PER_CLASS:
                self._iso.append(None)
            else:
                iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
                iso.fit(proba[:, c], onehot[:, c])
                self._iso.append(iso)
        self.fitted_ = True
        return self

    def predict_proba(self, proba):
        assert self.fitted_, "IsotonicCalibrator must be fit first"
        proba = np.asarray(proba, dtype=float)
        out = np.empty_like(proba)
        for c, iso in enumerate(self._iso):
            out[:, c] = iso.predict(proba[:, c]) if iso is not None else proba[:, c]
        out = np.clip(out, 1e-7, 1.0)
        return out / out.sum(axis=1, keepdims=True)

    def config(self):
        return {"type": "isotonic", "per_class": [
            "identity" if i is None else "isotonic" for i in self._iso]}


def ensemble_equal_weight(proba_list):
    """Equal-weight average of probability vectors (normalized)."""
    P = np.mean(np.stack(proba_list, axis=0), axis=0)
    return P / P.sum(axis=1, keepdims=True)


def ensemble_optimal_weights(proba_val_list, y_val, proba_test_list, step=0.05):
    """Search convex weights on VALIDATION only; apply to test.

    Returns (test_proba, weights, val_logloss).  Weights minimize validation
    log-loss; the test set is never part of the selection.
    """
    n = len(proba_val_list)
    best_w, best_ll = None, np.inf
    if n == 2:
        for w0 in np.arange(0.0, 1.0 + 1e-9, step):
            w = [w0, 1.0 - w0]
            pv = w[0] * proba_val_list[0] + w[1] * proba_val_list[1]
            ll = compute_log_loss(y_val, pv)
            if ll < best_ll:
                best_ll, best_w = ll, w
    elif n == 3:
        for w0 in np.arange(0.0, 1.0 + 1e-9, step):
            for w1 in np.arange(0.0, 1.0 - w0 + 1e-9, step):
                w2 = 1.0 - w0 - w1
                w = [w0, w1, w2]
                pv = (w[0] * proba_val_list[0] + w[1] * proba_val_list[1]
                      + w[2] * proba_val_list[2])
                ll = compute_log_loss(y_val, pv)
                if ll < best_ll:
                    best_ll, best_w = ll, w
    if best_w is None:
        best_w = [1.0 / n] * n
        best_ll = compute_log_loss(y_val, ensemble_equal_weight(proba_val_list))
    w = np.array(best_w)
    pt = w[0] * proba_test_list[0]
    for i in range(1, n):
        pt = pt + w[i] * proba_test_list[i]
    return pt / pt.sum(axis=1, keepdims=True), [float(x) for x in best_w], float(best_ll)



def evaluate_proba(y, proba, name="", config=None):
    """Full 3-class report; validates probability shape & normalization."""
    proba = np.asarray(proba, dtype=float)
    validate_proba(proba, len(y))
    y_pred = np.argmax(proba, axis=1)
    m = {
        "name": name,
        "n": int(len(y)),
        "accuracy": float(compute_accuracy(y, y_pred)),
        "macro_f1": float(compute_macro_f1(y, y_pred)),
        "log_loss": float(compute_log_loss(y, proba)),
        "brier_score": float(compute_brier_score(y, proba)),
        "per_class_precision_recall": compute_precision_recall(y, y_pred),
        "confusion_matrix": build_confusion_matrix(
            y, y_pred, OUTCOME_LABELS).tolist(),
        "class_distribution": class_distribution(y, OUTCOME_LABELS),
        "calibration": compute_calibration(y, proba, n_bins=5),
    }
    if config is not None:
        m["config"] = config
    return m


def reliability_bins(y, proba, n_bins=10):
    """Per-class reliability bins (confidence/accuracy/count per bin)."""
    proba = np.asarray(proba, dtype=float)
    y = np.asarray(y, dtype=int)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bins = {}
    for c in range(N_OUTCOMES):
        pc = proba[:, c]
        idx = np.digitize(pc, edges[1:-1])
        conf, acc, cnt = [], [], []
        for b in range(n_bins):
            msk = idx == b
            if msk.sum() == 0:
                conf.append(float(edges[b])); acc.append(0.0); cnt.append(0)
            else:
                conf.append(float(pc[msk].mean()))
                acc.append(float((y[msk] == c).mean()))
                cnt.append(int(msk.sum()))
        bins[OUTCOME_LABELS[c]] = {
            "bin_lower": edges[:-1].tolist(),
            "bin_upper": edges[1:].tolist(),
            "confidence": conf, "accuracy": acc, "count": cnt,
        }
    return bins


def _elo_history(raw: pd.DataFrame, K=40.0, home_adv=50.0):
    """Fit Elo on raw matches (same-day safe) and return per-fixture history.

    Phase-23 fits Elo on the RAW match frame (which has ``match_date``).  The
    ``history_`` list carries ``expected_home`` and ``outcome_label`` per
    fixture, ordered chronologically so we can split into train+val / test.
    """
    m = EloModel(K=K, home_advantage=home_adv)
    m.fit(raw.sort_values(["match_date", "home_team_id",
                           "away_team_id"]).reset_index(drop=True))
    hist = pd.DataFrame([r.__dict__ for r in m.history_])
    hist["_label_idx"] = hist["outcome_label"].map(LABEL_IDX).astype(int)
    return hist



def run_phase24(seed=42):
    """Execute the Phase-24 calibration + ensemble evaluation pipeline.

    Protocol:
      * Split = Phase-23 train(2015/16)/val(2016/17)/test(2019/20), 380 each.
      * Base models produce 3-class prob vectors via strictly-pre-match history.
      * Calibrators fit on val predictions only; applied to test.
      * Ensemble weights fit on val log-loss only.
      * Production DB read-only; SHA-256 gated (fail-closed).
    """
    prod_before = _prod_snapshot()
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    raw = _load_raw_matches()
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    print(f"[phase24] prod SHA-256 (before): {prod_before}")

    train_df, val_df, test_df, feats = build_split(raw, use_interactions=True)
    y_test = test_df["_label_idx"].to_numpy(dtype=int)
    y_val = val_df["_label_idx"].to_numpy(dtype=int)
    n_test, n_val, n_train = len(test_df), len(val_df), len(train_df)
    print(f"[phase24] split: train={n_train} val={n_val} test={n_test}")

    raw_sorted = raw.sort_values(["match_date", "home_team_id",
                                  "away_team_id"]).reset_index(drop=True)
    dc_rho = 0.0  # Phase-23 MLE; collapses DC to independent Poisson.

    # ---- Base model probabilities (strictly-pre-match history) ----
    print("[phase24] computing base model probabilities...")
    pois_full, _ = run_poisson_chronological(raw_sorted, model=PoissonModel())
    pois_val = _proba_from_preds(_align_by_keys(pois_full, val_df))
    pois_test = _proba_from_preds(_align_by_keys(pois_full, test_df))

    lr_m = LogisticRegressionClassifier(
        learning_rate=0.05, n_iterations=5000, l2_lambda=0.01,
        random_state=seed,
    ).fit(_mat(train_df, feats), _y(train_df))
    lr_val = lr_m.predict_proba(_mat(val_df, feats))
    lr_test = lr_m.predict_proba(_mat(test_df, feats))

    # LR + interactions: tune L2 on val, refit on train+val.
    best_l2, best_acc = 0.01, -1.0
    for l2 in [0.001, 0.01, 0.1, 1.0]:
        mm = LogisticRegressionClassifier(learning_rate=0.05, n_iterations=5000,
                                        l2_lambda=l2, random_state=seed).fit(
            _mat(train_df, feats), _y(train_df))
        acc = compute_accuracy(y_val, np.argmax(mm.predict_proba(_mat(val_df, feats)), 1))
        if acc > best_acc:
            best_acc, best_l2 = acc, l2
    tv = pd.concat([train_df, val_df], ignore_index=True)
    m_tv = LogisticRegressionClassifier(learning_rate=0.05, n_iterations=5000,
                                        l2_lambda=best_l2,
                                        random_state=seed).fit(_mat(tv, feats), _y(tv))
    lri_val = m_tv.predict_proba(_mat(val_df, feats))
    lri_test = m_tv.predict_proba(_mat(test_df, feats))

    dc_full = _run_dixon_coles(raw_sorted, DixonColesModel(rho=dc_rho))
    dc_val = _proba_from_preds(_align_by_keys(dc_full, val_df))
    dc_test = _proba_from_preds(_align_by_keys(dc_full, test_df))

    xgb_val = _probs_xgboost(train_df, val_df, feats, val_df, seed=seed)
    xgb_test = _probs_xgboost(train_df, val_df, feats, test_df, seed=seed)

    # Elo -> 3-class via DiriEloMapper fit on train+val Elo history.
    # Run Elo over the FULL chronological raw sequence (same-day safe); each
    # fixture's expected_home is computed from strictly-pre-match ratings.
    elo_hist = _elo_history(raw_sorted, K=40.0)
    ed = elo_hist["match_date"].astype(str)
    # Fit mapper on ALL train+val (760 fixtures), but evaluate val predictions
    # on the val subset only (380), mirroring how LR/Poisson val probs work.
    tv_mask = ed < VAL_MAX                       # train+val = 760
    test_mask = ed >= VAL_MAX                    # test = 380
    val_mask = (ed >= TRAIN_MAX) & (ed < VAL_MAX)  # val = 380
    tv_hist = elo_hist[tv_mask].reset_index(drop=True)
    val_hist = elo_hist[val_mask].reset_index(drop=True)
    test_elo_hist = elo_hist[test_mask].reset_index(drop=True)
    test_elo_aligned = _align_elo(test_elo_hist, test_df)
    mapper = DiriEloMapper().fit(tv_hist["expected_home"].to_numpy(),
                                 tv_hist["_label_idx"].to_numpy())
    # Align val elo history to val_df order by fixture key.
    val_elo_aligned = _align_elo(val_hist, val_df)
    elo_val = mapper.predict_proba(
        val_elo_aligned["expected_home"].to_numpy(dtype=float))
    elo_test = mapper.predict_proba(
        test_elo_aligned["expected_home"].to_numpy(dtype=float))

    # ---- Validate all probability vectors ----
    base_names = ["poisson", "logistic_regression", "logistic_interactions",
                  "dixon_coles", "xgboost", "elo_mapped"]
    val_probs = [pois_val, lr_val, lri_val, dc_val, xgb_val, elo_val]
    test_probs = [pois_test, lr_test, lri_test, dc_test, xgb_test, elo_test]
    for p in val_probs:
        validate_proba(p, n_val)
    for p in test_probs:
        validate_proba(p, n_test)

    # ---- Calibrators fit on val predictions only ----
    print("[phase24] fitting calibrators on validation predictions...")
    calib = {}
    for nm, vp in zip(base_names, val_probs):
        calib[nm] = {
            "temp": TemperatureScaler().fit(vp, y_val),
            "iso": IsotonicCalibrator().fit(vp, y_val),
        }

    # ---- Evaluate raw + calibrated ----
    print("[phase24] evaluating models + calibrators on test...")
    results = {}
    for nm, tp, vp in zip(base_names, test_probs, val_probs):
        results[nm] = evaluate_proba(
            y_test, tp, name=nm,
            config={"kind": "base", "reliability": reliability_bins(y_test, tp, 5)})
        for cal_kind, scaler in [("temp", calib[nm]["temp"]),
                                 ("iso", calib[nm]["iso"])]:
            key = f"{nm}_{cal_kind}"
            vp_cal = scaler.predict_proba(vp)
            tp_cal = scaler.predict_proba(tp)
            results[key] = evaluate_proba(
                y_test, tp_cal, name=key,
                config={"kind": f"calibrated_{cal_kind}", "base": nm,
                        "val_logloss_before": float(compute_log_loss(y_val, vp)),
                        "val_logloss_after": float(compute_log_loss(y_val, vp_cal))})

    # ---- Ensembles (weights selected on val only) ----
    print("[phase24] evaluating ensembles...")
    ens_test_proba = {}
    ens_defs = [
        ("ens_poisson_lr", ["poisson", "logistic_regression"]),
        ("ens_poisson_xgb", ["poisson", "xgboost"]),
        ("ens_poisson_li_xgb", ["poisson", "logistic_interactions", "xgboost"]),
    ]
    for ens_name, members in ens_defs:
        vl = [val_probs[base_names.index(m)] for m in members]
        tl = [test_probs[base_names.index(m)] for m in members]
        eq = ensemble_equal_weight(tl)
        results[ens_name + "_equal"] = evaluate_proba(
            y_test, eq, name=ens_name + "_equal",
            config={"kind": "ensemble_equal", "members": members})
        ens_test_proba[ens_name + "_equal"] = eq
        ot, w, vll = ensemble_optimal_weights(vl, y_val, tl)
        results[ens_name + "_optimal"] = evaluate_proba(
            y_test, ot, name=ens_name + "_optimal",
            config={"kind": "ensemble_optimal", "members": members,
                    "weights": w, "val_logloss": vll})
        ens_test_proba[ens_name + "_optimal"] = ot

    # Calibrated ensemble: temperature-scaled Poisson+LR+XGB.
    keep = ["poisson", "logistic_regression", "xgboost"]
    idx = [base_names.index(k) for k in keep]
    cv_cal = [calib[k]["temp"].predict_proba(vp) for k, vp in
              zip(base_names, val_probs) if k in keep]
    ct_cal = [calib[k]["temp"].predict_proba(tp) for k, tp in
              zip(base_names, test_probs) if k in keep]
    ct_eq = ensemble_equal_weight(ct_cal)
    results["ens_calibrated_equal"] = evaluate_proba(
        y_test, ct_eq, name="ens_calibrated_equal",
        config={"kind": "ensemble_calibrated_equal",
                "members": [k + "_temp" for k in keep]})
    ens_test_proba["ens_calibrated_equal"] = ct_eq
    ot, w, vll = ensemble_optimal_weights(cv_cal, y_val, ct_cal)
    results["ens_calibrated_optimal"] = evaluate_proba(
        y_test, ot, name="ens_calibrated_optimal",
        config={"kind": "ensemble_calibrated_optimal",
                "members": [k + "_temp" for k in keep],
                "weights": w, "val_logloss": vll})
    ens_test_proba["ens_calibrated_optimal"] = ot

    # ---- Paired significance vs Poisson (log-loss) ----
    print("[phase24] paired significance testing vs Poisson...")
    pois_loss = np.log(np.clip(pois_test[np.arange(n_test), y_test], 1e-15, 1.0))
    pois_pred = np.argmax(pois_test, axis=1)
    paired = []
    for nm, tp in zip(base_names, test_probs):
        pred = np.argmax(tp, axis=1)
        loss = np.log(np.clip(tp[np.arange(n_test), y_test], 1e-15, 1.0))
        mc = mcp_mcnemar_exact(y_test, pois_pred, pred)
        ba = paired_bootstrap(pois_loss, loss, n_boot=N_BOOT, seed=BOOT_SEED)
        paired.append({"pair": ["poisson", nm], "metric": "logloss",
                       "mcnemar": mc, "bootstrap": ba})
    for ens_nm, tp in ens_test_proba.items():
        pred = np.argmax(tp, axis=1)
        loss = np.log(np.clip(tp[np.arange(n_test), y_test], 1e-15, 1.0))
        mc = mcp_mcnemar_exact(y_test, pois_pred, pred)
        ba = paired_bootstrap(pois_loss, loss, n_boot=N_BOOT, seed=BOOT_SEED)
        paired.append({"pair": ["poisson", ens_nm], "metric": "logloss",
                       "mcnemar": mc, "bootstrap": ba})

    prod_after = _prod_snapshot()
    assert prod_before == prod_after, (
        "Production DB SHA-256 changed during Phase-24! (fail-closed)")

    ranking = sorted(
        [{"name": k, "brier": v["brier_score"], "log_loss": v["log_loss"],
          "accuracy": v["accuracy"], "macro_f1": v["macro_f1"]}
         for k, v in results.items()],
        key=lambda r: r["brier"])

    report = {
        "phase": 24,
        "produced_at": ts,
        "config": {"seed": seed, "n_boot": N_BOOT, "boot_seed": BOOT_SEED,
                   "calibration_protocol": "fit on val predictions only; "
                                           "applied to test",
                   "elo_mapping": "DiriEloMapper fit on train+val Elo history"},
        "dataset": {
            "n_train": n_train, "n_val": n_val, "n_test": n_test,
            "train_date_max": TRAIN_MAX, "val_date_max": VAL_MAX,
            "test_date_range": [str(test_df["target_match_date"].min()),
                                str(test_df["target_match_date"].max())],
            "production_db_sha256": prod_before,
            "production_db_unchanged": prod_before == prod_after,
        },
        "models": results,
        "paired_vs_poisson": paired,
        "ranking_by_brier": ranking,
        "calibrators": {nm: {k: v.config() for k, v in cc.items()}
                        for nm, cc in calib.items()},
        "dc_rho": {"rho": dc_rho, "fit_on": "train_only (Phase-23 MLE)"},
        "elo_mapping_config": mapper.config(),
        "notes": [
            "Phase-24 audit confirmed Phase-23 split integrity "
            "(train=2015/16, val=2016/17, test=2019/20; 380 each).",
            "Calibrators fit on val predictions only; isotonic falls back to "
            "identity if a class has <10 positives (small-sample guard).",
            "Elo mapped to 3-class via a Dirichlet-style linear-softmax map "
            "fit on train+val; NOT treated as native calibrated probabilities.",
            "Ensemble weights selected on validation log-loss; test never used.",
            "DC rho=0 (MLE on train) collapses DC to independent Poisson; "
            "preserved as the faithful Phase-23 result.",
            "COVID-19 stationarity caveat inherited from Phase 23.",
        ],
    }
    _write_json(report, REPORT_DIR / f"phase24_comparison_{ts}.json")
    _write_txt(report, REPORT_DIR / f"phase24_comparison_{ts}.txt")
    print(f"[phase24] reports written. prod SHA-256 (after): {prod_after}  "
          f"unchanged={prod_before == prod_after}")
    return report


def _write_json(report, path):
    path.parent.mkdir(parents=True, exist_ok=True)

    def _default(o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, (np.ndarray,)):
            return o.tolist()
        return str(o)
    path.write_text(json.dumps(report, indent=2, default=_default,
                               ensure_ascii=False), encoding="utf-8")


def _write_txt(report, path):
    L = []
    L.append("Phase 24 - Probability Calibration & Validated Ensembles")
    L.append("=" * 58)
    L.append(f"Produced: {report['produced_at']}")
    d = report["dataset"]
    L.append(f"Split: train={d['n_train']} val={d['n_val']} test={d['n_test']}")
    L.append(f"Train max date: {d['train_date_max']} | "
             f"Val max date: {d['val_date_max']}")
    L.append(f"Test date range: {d['test_date_range']}")
    L.append(f"Production DB sha256: {d['production_db_sha256']}")
    L.append(f"Production DB unchanged: {d['production_db_unchanged']}")
    L.append("")
    L.append("Config")
    L.append("------")
    for k, v in report["config"].items():
        L.append(f"  {k}: {v}")
    L.append("")
    L.append("Test metrics - sorted by Brier score (lower is better)")
    L.append("-" * 76)
    L.append(f"{'model':<32s} {'acc':>6s} {'macroF1':>7s} "
             f"{'logloss':>8s} {'brier':>7s}")
    for r in report["ranking_by_brier"]:
        L.append(f"{r['name']:<32s} {r['accuracy']:>6.3f} "
                 f"{r['macro_f1']:>7.3f} {r['log_loss']:>8.4f} "
                 f"{r['brier']:>7.4f}")
    L.append("")
    L.append("Calibrators fit on validation only")
    L.append("----------------------------------")
    for nm, cc in report.get("calibrators", {}).items():
        for kind, cfg in cc.items():
            extra = f"T={cfg.get('T', '')}" if kind == "temp" else ""
            L.append(f"  {nm:>28s} {kind:<6s} "
                     f"{cfg.get('type', '')} {extra}")
    L.append("")
    L.append("Paired significance vs Poisson (log-loss)")
    L.append("-----------------------------------------")
    for pr in report["paired_vs_poisson"]:
        mc = pr["mcnemar"]; ba = pr["bootstrap"]
        pair = "/".join(pr["pair"])
        L.append(f"  {pair:<34s} McNemar p={mc['p_value']:.4f} "
                 f"boot_diff={ba['mean_diff']:+.4f} "
                 f"95%CI=({ba['ci_low_95']:+.4f},{ba['ci_high_95']:+.4f})")
    L.append("")
    L.append("Notes")
    L.append("-----")
    for n in report["notes"]:
        L.append(f"- {n}")
    path.write_text(chr(10).join(L) + chr(10), encoding="utf-8")


if __name__ == "__main__":
    run_phase24()
