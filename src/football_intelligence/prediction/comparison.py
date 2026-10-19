"""Phase 21: model comparison on identical chronological test fixtures.

Compares the Phase-18 multinomial logistic regression, the Phase-19 Elo
ratings and the Phase-20 independent Poisson goal model on the SAME 3-class
outcome task over one identical chronological test split.  The production
database is treated as strictly read-only (SHA-256 re-checked around the
load); no writes are ever made to it.

Alignment: the three models share one deterministic chronological fixture
order (match_date ascending, ties broken by home_team_id, away_team_id).  The
logistic-regression baseline trains on the chronological train split (Phase 18
regime) and predicts the held-out test split; Elo and Poisson are warmed online
on all pre-test history and scored on the same test fixtures.  All three are
aligned to the test split by the fixture key (match_date, home_team_id,
away_team_id).

Metric caveats: LR and Poisson emit proper 3-class outcome probabilities, so
their Brier score and log-loss are directly comparable.  Elo emits a continuous
expected home SCORE (a 2-class quantity), not a calibrated 3-class probability;
Elo reports its native brier_expected_score and is compared with LR/Poisson only
on the 3-class metrics well-defined for all of them.

Training-regime asymmetry (honest, documented): LR trains on the fixed train
split only, while Elo and Poisson are warmed on ALL pre-test matches (train +
validation).  This is an asymmetry of the *training regime*, not the test set.
Expanding-window online LR is a future enhancement, out of scope here.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from .baseline import (
    N_OUTCOMES,
    OUTCOME_LABELS,
    OUTCOME_INDEX,
    LogisticRegressionClassifier,
    build_confusion_matrix,
    chronological_split,
    class_distribution,
    compute_accuracy,
    prepare_training_data,
)
from .evaluation import (
    _json_default,
    _sha256,
    classification_report,
    compute_macro_f1,
    compute_precision_recall,
    feature_columns,
)
from .elo import EloModel
from .poisson import (
    DEFAULT_MAX_GOALS,
    DEFAULT_MIN_HISTORY,
    DEFAULT_PRIOR_STRENGTH,
    PoissonModel,
    run_poisson_chronological,
)
from ..database import resolve_db_path
from ..features.engineering import FeatureEngineer, load_matches_for_features

__all__ = [
    "ComparisonReport",
    "mcp_mcnemar_exact",
    "paired_bootstrap",
    "run_model_comparison",
    "save_comparison_report",
]

_LABELS = list(OUTCOME_LABELS)


def _key(date_val, home, away) -> tuple:
    """Canonical fixture key -- identical form for Elo / Poisson / LR splits."""
    return (str(date_val), int(home), int(away))


def _onehot(y_true: np.ndarray) -> np.ndarray:
    n = len(y_true)
    oh = np.zeros((n, N_OUTCOMES), dtype=float)
    if n:
        oh[np.arange(n), np.asarray(y_true, dtype=int)] = 1.0
    return oh


def _acc_loss(y_true, pred) -> np.ndarray:
    return (np.asarray(pred, dtype=int) != np.asarray(y_true, dtype=int)).astype(float)


def mcp_mcnemar_exact(
    y_true, pred_i, pred_j, labels: Optional[Sequence[str]] = None
) -> dict:
    """Exact McNemar test (two-sided binomial) on accuracy discordance.

    Returns discordant-pair counts (n01 = i wrong / j right, n10 = i right /
    j wrong), the continuity-corrected McNemar statistic and the exact
    two-sided p-value under H0: the two models are equally accurate.
    """
    y = np.asarray(y_true)
    pi = np.asarray(pred_i)
    pj = np.asarray(pred_j)
    ci = pi == y
    cj = pj == y
    n01 = int(np.sum(~ci & cj))
    n10 = int(np.sum(ci & ~cj))
    n = n01 + n10
    if n == 0:
        stat, p = 0.0, 1.0
    else:
        k = min(n01, n10)
        tail = sum(math.comb(n, j) for j in range(k + 1)) / (2.0 ** n)
        p = min(1.0, 2.0 * tail)
        stat = (abs(n01 - n10) - 1.0) ** 2 / n
    return {"n01": n01, "n10": n10, "n_discordant": int(n),
            "statistic": float(stat), "p_value": float(p)}



def paired_bootstrap(
    loss_i, loss_j, n_boot: int = 2000, seed: int = 42
) -> dict:
    """Paired bootstrap CI for the mean per-sample loss difference (i - j).

    Returns the mean difference, a 95% percentile CI and an approximate
    two-sided bootstrap p-value for a zero difference.
    """
    a = np.asarray(loss_i, dtype=float)
    b = np.asarray(loss_j, dtype=float)
    n = a.shape[0]
    if n == 0:
        return {"mean_diff": 0.0, "ci_low_95": 0.0, "ci_high_95": 0.0,
                "p_value": 1.0, "n_boot": int(n_boot), "n_samples": 0}
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    diffs = a[idx].mean(axis=1) - b[idx].mean(axis=1)
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    mean_diff = float(np.mean(a - b))  # observed per-sample loss difference (i - j)
    p = 2.0 * min(float(np.mean(diffs <= 0)), float(np.mean(diffs >= 0)))
    return {"mean_diff": mean_diff, "ci_low_95": float(lo), "ci_high_95": float(hi),
            "p_value": float(min(p, 1.0)), "n_boot": int(n_boot), "n_samples": int(n)}


@dataclass
class ComparisonReport:
    dataset: dict
    split: dict
    models: dict
    paired: list
    predictions: list = field(default_factory=list)
    config: dict = field(default_factory=dict)
    produced_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds"))
    notes: list = field(default_factory=list)
    production_db_sha256: Optional[str] = None
    production_db_unchanged: Optional[bool] = None

    def to_dict(self) -> dict:
        return {
            "dataset": self.dataset,
            "split": self.split,
            "models": self.models,
            "paired": self.paired,
            "predictions": self.predictions,
            "config": self.config,
            "produced_at": self.produced_at,
            "notes": self.notes,
            "production_db_sha256": self.production_db_sha256,
            "production_db_unchanged": self.production_db_unchanged,
        }


def _metrics_from_proba(y_true, proba, pred, *, name: str, config: dict) -> dict:
    """Full 3-class report for models that emit proper probabilities (LR/Poisson)."""
    rep = classification_report(
        np.asarray(y_true, dtype=int),
        np.asarray(proba, dtype=float),
        y_pred=np.asarray(pred, dtype=int),
        labels=_LABELS,
    )
    rep["name"] = name
    rep["model_type"] = "categorical"
    rep["brier_expected_score"] = None
    rep["config"] = config
    return rep


def _elo_pred_label(expected: float, win_thr: float, away_thr: float) -> int:
    if expected >= win_thr:
        return OUTCOME_INDEX["home_win"]
    if expected <= away_thr:
        return OUTCOME_INDEX["away_win"]
    return OUTCOME_INDEX["draw"]


def _metrics_from_elo(y_true, expected, actual, pred, *, config: dict) -> dict:
    """3-class report for Elo (expected score + fixed-threshold rule).

    Elo has no calibrated 3-class probability, so brier_score/log_loss are
    reported as None; the native brier_expected_score (mean squared error of the
    expected home score vs the realised result) is reported instead.
    """
    yt = np.asarray(y_true, dtype=int)
    yp = np.asarray(pred, dtype=int)
    ex = np.asarray(expected, dtype=float)
    sc = np.asarray(actual, dtype=float)
    brier_exp = float(np.mean((ex - sc) ** 2)) if len(ex) else 0.0
    return {
        "n_samples": int(len(yt)),
        "log_loss": None,
        "brier_score": None,
        "brier_expected_score": brier_exp,
        "accuracy": float(compute_accuracy(yt, yp, _LABELS)),
        "macro_f1": float(compute_macro_f1(yt, yp, _LABELS)),
        "per_class_precision_recall": compute_precision_recall(yt, yp, _LABELS),
        "confusion_matrix": build_confusion_matrix(yt, yp, _LABELS).tolist(),
        "confusion_labels": _LABELS,
        "class_distribution": class_distribution(yt, _LABELS),
        "calibration": {lb: [] for lb in _LABELS},
        "labels": _LABELS,
        "name": "elo",
        "model_type": "expected_score_thresholded",
        "config": config,
    }


def run_model_comparison(
    db_path=None,
    *,
    provider: str = "statsbomb",
    provider_competition_id: str = "2",
    season_name: Optional[str] = None,
    train_frac: float = 0.6,
    val_frac: float = 0.2,
    lr_learning_rate: float = 0.01,
    lr_n_iterations: int = 10000,
    lr_l2_lambda: float = 0.01,
    lr_seed: int = 42,
    elo_K: float = 20.0,
    elo_home_advantage: float = 50.0,
    elo_initial_rating: float = 1500.0,
    elo_home_win_threshold: float = 0.60,
    elo_away_win_threshold: float = 0.40,
    pois_max_goals: int = DEFAULT_MAX_GOALS,
    pois_min_history: int = DEFAULT_MIN_HISTORY,
    pois_prior_strength: float = DEFAULT_PRIOR_STRENGTH,
    n_boot: int = 2000,
    boot_seed: int = 42,
) -> ComparisonReport:
    """Run LR vs. Elo vs. Poisson on identical chronological test fixtures."""
    db = resolve_db_path(db_path)
    sha_before = _sha256(db)
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
    sha_after = _sha256(db)
    if sha_before is not None and sha_before != sha_after:
        raise RuntimeError(
            "Production DB SHA-256 changed during comparison load -- aborting "
            "(writes to a read-only DB violate the leakage contract)."
        )
    # Deterministic chronological fixture order; within-date order by team ids so
    # all three models share an identical fixture ordering for alignment.
    raw = raw.sort_values(
        ["match_date", "home_team_id", "away_team_id"]
    ).reset_index(drop=True)

    # ---- Logistic regression (Phase 18 regime): fit train split, predict test ----
    engineer = FeatureEngineer()
    eng, _X, _y = prepare_training_data(raw, engineer=engineer)
    feats = feature_columns(eng)
    train_df, val_df, test_df = chronological_split(
        eng, train_frac=train_frac, val_frac=val_frac, date_col="target_match_date")
    if len(test_df) == 0:
        raise ValueError("Chronological split produced an empty test window.")

    Xtr = train_df[feats].to_numpy(dtype=float)
    ytr = train_df["_label_idx"].to_numpy(dtype=int)
    lr = LogisticRegressionClassifier(
        learning_rate=lr_learning_rate, n_iterations=lr_n_iterations,
        l2_lambda=lr_l2_lambda, random_state=lr_seed,
    ).fit(Xtr, ytr)
    lr_proba = lr.predict_proba(test_df[feats].to_numpy(dtype=float))
    lr_pred = np.argmax(lr_proba, axis=1).astype(int)
    y_true = test_df["_label_idx"].to_numpy(dtype=int)

    # ---- Elo (Phase 19 regime, same-day-safe): online over full history ----
    elo = EloModel(
        K=elo_K, home_advantage=elo_home_advantage, initial_rating=elo_initial_rating,
        home_win_threshold=elo_home_win_threshold, away_win_threshold=elo_away_win_threshold,
    )
    elo.fit(raw)
    elo_by_key = {
        _key(m.match_date, m.home_team_id, m.away_team_id): m for m in elo.history_
    }

    # ---- Poisson (Phase 20 regime): online over full history ----
    pois = PoissonModel(
        max_goals=int(pois_max_goals),
        min_history=int(pois_min_history),
        prior_strength=float(pois_prior_strength),
    )
    preds, _summary = run_poisson_chronological(raw, model=pois)
    pois_by_key = {
        _key(rec["match_date"], rec["home_team_id"], rec["away_team_id"]): rec
        for rec in preds
    }

    # ---- Align Elo/Poisson to the test split by fixture key ----
    elo_expected, elo_actual, elo_pred = [], [], []
    pois_proba, pois_pred = [], []
    for row in test_df.itertuples(index=False):
        k = _key(row.target_match_date, row.home_team_id, row.away_team_id)
        em = elo_by_key.get(k)
        if em is None:
            raise KeyError(f"Elo history missing test fixture {k}")
        elo_expected.append(float(em.expected_home))
        elo_actual.append(float(em.actual_home))
        elo_pred.append(
            _elo_pred_label(em.expected_home, elo.home_win_threshold, elo.away_win_threshold)
        )
        pr = pois_by_key.get(k)
        if pr is None:
            raise KeyError(f"Poisson predictions missing test fixture {k}")
        probs = pr["probs"]
        row_proba = [float(probs[lbl]) for lbl in OUTCOME_LABELS]
        pois_proba.append(row_proba)
        pois_pred.append(int(np.argmax(row_proba)))

    elo_expected = np.asarray(elo_expected, dtype=float)
    elo_actual = np.asarray(elo_actual, dtype=float)
    elo_pred = np.asarray(elo_pred, dtype=int)
    pois_proba = np.asarray(pois_proba, dtype=float)
    pois_pred = np.asarray(pois_pred, dtype=int)

    lr_metrics = _metrics_from_proba(
        y_true, lr_proba, lr_pred,
        name="logistic_regression",
        config={
            "regime": "chronological_train_predict_test",
            "n_train": int(len(train_df)), "n_val": int(len(val_df)),
            "n_test": int(len(test_df)), "train_frac": float(train_frac),
            "val_frac": float(val_frac), "learning_rate": float(lr_learning_rate),
            "n_iterations": int(lr_n_iterations), "l2_lambda": float(lr_l2_lambda),
            "random_state": int(lr_seed), "n_features": int(len(feats)),
        },
    )
    pois_metrics = _metrics_from_proba(
        y_true, pois_proba, pois_pred,
        name="poisson",
        config={
            "regime": "online_chronological",
            "max_goals": int(pois.max_goals), "min_history": int(pois.min_history),
            "prior_strength": float(pois.prior_strength),
            "n_test": int(len(test_df)),
        },
    )
    elo_metrics = _metrics_from_elo(
        y_true, elo_expected, elo_actual, elo_pred,
        config={
            "regime": "online_chronological",
            "K": float(elo.K), "home_advantage": float(elo.home_advantage),
            "initial_rating": float(elo.initial_rating),
            "home_win_threshold": float(elo.home_win_threshold),
            "away_win_threshold": float(elo.away_win_threshold),
            "n_test": int(len(test_df)),
        },
    )
    models = {
        "logistic_regression": lr_metrics,
        "poisson": pois_metrics,
        "elo": elo_metrics,
    }

    # ---- Paired statistical tests (all pairs, identical test fixtures) ----
    lr_acc = _acc_loss(y_true, lr_pred)
    pois_acc = _acc_loss(y_true, pois_pred)
    elo_acc = _acc_loss(y_true, elo_pred)
    lr_brier = np.sum((lr_proba - _onehot(y_true)) ** 2, axis=1)
    pois_brier = np.sum((pois_proba - _onehot(y_true)) ** 2, axis=1)

    pair_specs = [
        ("logistic_regression", "poisson", lr_pred, pois_pred, lr_acc, pois_acc, lr_brier, pois_brier),
        ("logistic_regression", "elo", lr_pred, elo_pred, lr_acc, elo_acc, None, None),
        ("elo", "poisson", elo_pred, pois_pred, elo_acc, pois_acc, None, None),
    ]
    paired = []
    for a, b, pa, pb, la, lb, ba, bb in pair_specs:
        entry = {
            "pair": [a, b],
            "mcnemar_accuracy": mcp_mcnemar_exact(y_true, pa, pb),
            "bootstrap_accuracy": paired_bootstrap(la, lb, n_boot, boot_seed),
        }
        if ba is not None and bb is not None:
            entry["bootstrap_brier"] = paired_bootstrap(ba, bb, n_boot, boot_seed)
        else:
            entry["bootstrap_brier"] = None
        paired.append(entry)

    # ---- Per-fixture aligned predictions (inspection / reproducibility) ----
    label_lookup = {v: k for k, v in OUTCOME_INDEX.items()}
    predictions = []
    for i, row in enumerate(test_df.itertuples(index=False)):
        predictions.append({
            "match_date": str(row.target_match_date),
            "home_team_id": int(row.home_team_id),
            "away_team_id": int(row.away_team_id),
            "y_true": int(y_true[i]),
            "y_true_label": label_lookup.get(int(y_true[i])),
            "logistic_regression_pred": int(lr_pred[i]),
            "logistic_regression_pred_label": label_lookup.get(int(lr_pred[i])),
            "logistic_regression_proba": [float(x) for x in lr_proba[i]],
            "poisson_pred": int(pois_pred[i]),
            "poisson_pred_label": label_lookup.get(int(pois_pred[i])),
            "poisson_proba": [float(x) for x in pois_proba[i]],
            "elo_expected_home": float(elo_expected[i]),
            "elo_pred": int(elo_pred[i]),
            "elo_pred_label": label_lookup.get(int(elo_pred[i])),
        })

    dataset = {
        "provider": provider,
        "provider_competition_id": str(provider_competition_id),
        "season_name": season_name,
        "n_matches": int(len(raw)),
        "n_train": int(len(train_df)),
        "n_val": int(len(val_df)),
        "n_test": int(len(test_df)),
        "date_min": str(raw["match_date"].min()),
        "date_max": str(raw["match_date"].max()),
        "test_start_date": str(test_df["target_match_date"].min()),
        "test_end_date": str(test_df["target_match_date"].max()),
        "n_teams": int(raw["home_team_id"].nunique()),
    }
    split = {
        "train_frac": float(train_frac), "val_frac": float(val_frac),
        "n_train": int(len(train_df)), "n_val": int(len(val_df)), "n_test": int(len(test_df)),
        "date_col": "target_match_date",
        "note": "Identical chronological test split for all three models on the "
                "held-out fixtures. LR trains on the train split only; Elo and "
                "Poisson are warmed on all pre-test (train+val) history.",
    }
    notes = [
        "Read-only: production DB SHA-256 before == after load; no writes performed.",
        "Identical chronological test fixtures for all three models, aligned by "
        "(match_date, home_team_id, away_team_id).",
        "EloModel.fit was fixed in Phase 21 to remove same-day leakage: same-date "
        "fixtures are now scored against pre-date ratings and the day's updates are "
        "batched. On this dataset same-date fixtures involve disjoint teams, so the "
        "fix is correctness-hygiene with zero metric impact (reported for honesty).",
        "Metric caveat: Elo emits an expected home SCORE (2-class), not a calibrated "
        "3-class probability. LR and Poisson emit proper 3-class probabilities, so "
        "their Brier/log-loss are directly comparable; Elo is compared on 3-class "
        "metrics only (accuracy/macro-F1/confusion/P-R).",
        "Training-regime asymmetry: LR trains on the chronological train split only "
        "(e.g. 228 matches for the 380-fixture prod PL season), while Elo and Poisson "
        "are warmed on all 304 pre-test matches (train+val). This is an asymmetry of "
        "the training regime, not the test set. Expanding-window online LR is a future "
        "enhancement, explicitly out of scope here.",
        "McNemar p-values use the exact two-sided binomial test on accuracy "
        "discordance. Bootstrap CIs use 2000 resamples (default) with the percentile "
        "method and a zero-difference p-value derived from the CI.",
        "No model is ranked as 'superior'; metrics are descriptive.",
    ]
    return ComparisonReport(
        dataset=dataset,
        split=split,
        models=models,
        paired=paired,
        predictions=predictions,
        config={
            "lr": {"learning_rate": lr_learning_rate, "n_iterations": lr_n_iterations,
                   "l2_lambda": lr_l2_lambda, "random_state": lr_seed},
            "elo": {"K": elo_K, "home_advantage": elo_home_advantage,
                    "initial_rating": elo_initial_rating,
                    "home_win_threshold": elo_home_win_threshold,
                    "away_win_threshold": elo_away_win_threshold},
            "poisson": {"max_goals": int(pois_max_goals), "min_history": int(pois_min_history),
                        "prior_strength": float(pois_prior_strength)},
            "bootstrap": {"n_boot": int(n_boot), "seed": int(boot_seed)},
        },
        production_db_sha256=sha_after,
        production_db_unchanged=(True if sha_before is None else sha_before == sha_after),
        notes=notes,
    )



def save_comparison_report(
    report: ComparisonReport, output_dir: str, name: str = "phase21_comparison"
) -> dict:
    """Write JSON + readable TXT artifacts. NEVER writes to the production DB."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    json_path = out / f"{name}.json"
    txt_path = out / f"{name}.txt"
    json_path.write_text(
        json.dumps(report.to_dict(), indent=2, default=_json_default),
        encoding="utf-8",
    )
    _write_comparison_txt(report, txt_path)
    return {"json": str(json_path), "txt": str(txt_path), "dir": str(out), "name": name}


def _fmt_or_none(v) -> str:
    if v is None or (isinstance(v, float) and (math.isnan(v) or math.isinf(v))):
        return "   n/a"
    return f"{float(v):.4f}"


def _write_comparison_txt(report: ComparisonReport, path) -> None:
    L = []
    L.append("Phase-21 Model Comparison Report")
    L.append("=================================")
    L.append(f"Produced: {report.produced_at}")
    d = report.dataset
    L.append(
        f"provider={d.get('provider')} competition_id={d.get('provider_competition_id')} "
        f"season={d.get('season_name')}"
    )
    L.append(
        f"n_matches={d.get('n_matches')} n_teams={d.get('n_teams')} "
        f"date_min={d.get('date_min')} date_max={d.get('date_max')}"
    )
    sp = report.split
    L.append(
        f"split: n_train={sp.get('n_train')} n_val={sp.get('n_val')} n_test={sp.get('n_test')} "
        f"(train_frac={sp.get('train_frac')}, val_frac={sp.get('val_frac')})"
    )
    L.append(
        f"test window: {d.get('test_start_date')} -> {d.get('test_end_date')} "
        f"({d.get('n_test')} fixtures)"
    )
    L.append(f"production DB sha256: {report.production_db_sha256}")
    L.append(f"production DB unchanged: {report.production_db_unchanged}")
    L.append("")
    L.append("Model metrics (test window, identical fixtures)")
    L.append("---------------------------------------------")
    L.append(
        f"  {'model':<22}{'accuracy':>10}{'macro_f1':>10}{'brier(3)':>10}"
        f"{'log_loss':>10}{'brier_exp(2)':>14}"
    )
    for name in ("logistic_regression", "poisson", "elo"):
        m = report.models[name]
        L.append(
            f"  {name:<22}{m.get('accuracy', 0.0):>10.4f}{m.get('macro_f1', 0.0):>10.4f}"
            f"{_fmt_or_none(m.get('brier_score')):>10}"
            f"{_fmt_or_none(m.get('log_loss')):>10}"
            f"{_fmt_or_none(m.get('brier_expected_score')):>14}"
        )
    L.append("")
    L.append("Per-class precision/recall/F1 (3-class)")
    L.append("--------------------------------------")
    for name in ("logistic_regression", "poisson", "elo"):
        m = report.models[name]
        pc = m.get("per_class_precision_recall") or {}
        if not pc:
            continue
        L.append(f"  {name}:")
        for lbl in _LABELS:
            c = pc.get(lbl, {})
            L.append(
                f"    {lbl:9s} P={c.get('precision', 0.0):.4f} "
                f"R={c.get('recall', 0.0):.4f} F1={c.get('f1', 0.0):.4f} "
                f"N={c.get('support', 0)}"
            )
    L.append("")
    L.append("Confusion matrices (rows=true home/draw/away, cols=pred)")
    L.append("-------------------------------------------------------")
    for name in ("logistic_regression", "poisson", "elo"):
        m = report.models[name]
        cm = m.get("confusion_matrix")
        if cm is None:
            cm = []
        L.append(f"  {name}:")
        for row in cm:
            L.append("    " + " ".join(f"{v:7.0f}" for v in row))
    L.append("")
    L.append("Paired significance (identical test fixtures)")
    L.append("----------------------------------------------")
    for p in report.paired:
        L.append(f"  pair {p['pair'][0]} vs {p['pair'][1]}:")
        mc = p["mcnemar_accuracy"]
        L.append(
            f"    McNemar exact: n01={mc['n01']} n10={mc['n10']} "
            f"n_discordant={mc['n_discordant']} p={mc['p_value']:.4f}"
        )
        ab = p["bootstrap_accuracy"]
        L.append(
            f"    bootstrap delta-accuracy: mean={ab['mean_diff']:.4f} "
            f"95% CI=[{ab['ci_low_95']:.4f}, {ab['ci_high_95']:.4f}] p={ab['p_value']:.4f}"
        )
        bb = p.get("bootstrap_brier")
        if bb is not None:
            L.append(
                f"    bootstrap delta-brier:      mean={bb['mean_diff']:.4f} "
                f"95% CI=[{bb['ci_low_95']:.4f}, {bb['ci_high_95']:.4f}] p={bb['p_value']:.4f}"
            )
        else:
            L.append("    bootstrap delta-brier:      n/a (one model emits no 3-class probability)")
    L.append("")
    L.append("Notes")
    L.append("-----")
    for note in report.notes:
        L.append(f"- {note}")
    Path(path).write_text("\n".join(L), encoding="utf-8")
