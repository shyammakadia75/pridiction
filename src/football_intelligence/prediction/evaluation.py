"""Extended evaluation metrics + leakage-safe model-evaluation pipeline.

Pure numpy/pandas only (no sklearn).  Reuses the project's existing baseline
models, feature-engineering pipeline and metric primitives from
:mod:`football_intelligence.prediction.baseline`.

Label convention (FIXED -- never change): 0 = "home_win", 1 = "draw",
2 = "away_win".

Brier convention: multiclass Brier = mean over samples of sum_c (p-ic - onehot-ic)^2
(a uniform 3-class prediction scores 2/3; a perfect prediction scores 0).
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
    OUTCOME_LABELS,
    OUTCOME_INDEX,
    N_OUTCOMES,
    FrequencyBaseline,
    LogisticRegressionClassifier,
    build_confusion_matrix,
    chronological_split,
    class_distribution,
    compute_brier_score,
    compute_log_loss,
    evaluate_predictions,
    prepare_training_data,
)
from ..features.engineering import FeatureEngineer, load_matches_for_features
from ..database import resolve_db_path

__all__ = [
    "LEAKAGE_COLS",
    "feature_columns",
    "compute_macro_f1",
    "compute_precision_recall",
    "compute_calibration",
    "classification_report",
    "build_feature_dataset",
    "run_prediction_evaluation",
    "save_report",
    "EvaluationReport",
]

# Columns that must NEVER enter model features (label + match identifiers).
LEAKAGE_COLS: frozenset[str] = frozenset({
    "target_match_id",
    "target_match_date",
    "home_team_id",
    "away_team_id",
    "features_valid",
    "outcome_label",
    "_label_idx",
})


def _labels_arg(labels: Optional[Sequence[str]]) -> list[str]:
    return list(labels) if labels is not None else list(OUTCOME_LABELS)


def _to_index(y, labels: list[str]) -> np.ndarray:
    """Map string/int/idx labels to integer class indices aligned to ``labels``."""
    y = np.asarray(y)
    if y.size == 0:
        return np.zeros(0, dtype=int)
    if y.dtype == object or y.dtype.kind in ("U", "S"):
        l2i = {lbl: i for i, lbl in enumerate(labels)}
        return np.array([l2i[l] for l in y], dtype=int)
    return y.astype(int)


def compute_macro_f1(y_true, y_pred, labels=None) -> float:
    """Macro-averaged F1 across all classes (zero-division -> 0.0)."""
    labels = _labels_arg(labels)
    yt = _to_index(y_true, labels)
    yp = _to_index(y_pred, labels)
    if len(yt) == 0:
        return 0.0
    f1s = []
    for i in range(len(labels)):
        tp = int(np.sum((yt == i) & (yp == i)))
        fp = int(np.sum((yt != i) & (yp == i)))
        fn = int(np.sum((yt == i) & (yp != i)))
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        f1s.append(f1)
    return float(np.mean(f1s))


def compute_precision_recall(y_true, y_pred, labels=None) -> dict:
    """Per-class precision, recall, F1, support (zero-division -> 0.0)."""
    labels = _labels_arg(labels)
    yt = _to_index(y_true, labels)
    yp = _to_index(y_pred, labels)
    out: dict[str, dict] = {}
    for i, lbl in enumerate(labels):
        tp = int(np.sum((yt == i) & (yp == i)))
        fp = int(np.sum((yt != i) & (yp == i)))
        fn = int(np.sum((yt == i) & (yp != i)))
        support = int(np.sum(yt == i))
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        out[lbl] = {
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "support": support,
            "tp": tp,
            "fp": fp,
            "fn": fn,
        }
    return out


def compute_calibration(y_true, y_prob, labels=None, n_bins: int = 5) -> dict:
    """Per-class reliability-diagram data (probability calibration diagnostic).

    For each class c the predicted-probability column ``y_prob[:, c]`` is split
    into ``n_bins`` equal-width bins over [0, 1].  Each non-empty bin reports
    ``mean_predicted``, ``observed_frequency`` (fraction of the bin whose TRUE
    label equals c) and ``count``.  Empty bins are omitted.

    Descriptive only: never used to tune thresholds, select features, or pick
    hyperparameters (Phase-18 leakage contract).
    """
    labels = _labels_arg(labels)
    yt = _to_index(y_true, labels)
    y_prob = np.asarray(y_prob, dtype=float)
    if y_prob.ndim != 2 or y_prob.shape[1] != len(labels):
        return {lbl: [] for lbl in labels}
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    result: dict[str, list] = {}
    for i, lbl in enumerate(labels):
        col = y_prob[:, i]
        bins = []
        for b in range(n_bins):
            lo, hi = edges[b], edges[b + 1]
            if b == n_bins - 1:
                mask = (col >= lo) & (col <= hi)
            else:
                mask = (col >= lo) & (col < hi)
            cnt = int(np.sum(mask))
            if cnt == 0:
                continue
            bins.append({
                "bin_lower": float(lo),
                "bin_upper": float(hi),
                "count": cnt,
                "mean_predicted": float(col[mask].mean()),
                "observed_frequency": float(np.mean(yt[mask] == i)),
            })
        result[lbl] = bins
    return result



# ---------------------------------------------------------------------------
# Leakage-safe evaluation pipeline
# ---------------------------------------------------------------------------


def _sha256(path) -> Optional[str]:
    """SHA-256 hex digest of a file (read-only tamper guard)."""
    path = Path(path)
    if not path.exists():
        return None
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def build_feature_dataset(
    db_path=None,
    *,
    provider: str = "statsbomb",
    provider_competition_id: str = "2",
    season_name: Optional[str] = None,
) -> pd.DataFrame:
    """Load + feature-engineer a single competition/season, leakage-safe.

    Competition + season filtering is applied at load time via
    load_matches_for_features, so only same-competition matches enter the frame
    (no cross-competition leakage).  prepare_training_data then attaches the
    outcome label, drops leakage columns and enforces strict match_date <
    target_date history (no same-day / future leakage) before sorting the frame
    ascending by target_match_date.
    """
    raw = load_matches_for_features(
        db_path,
        provider=provider,
        provider_competition_id=str(provider_competition_id),
        season_name=season_name,
    )
    if raw is None or len(raw) == 0:
        raise ValueError(
            f"No matches for provider={provider}, "
            f"competition_id={provider_competition_id}, season={season_name}"
        )
    engineer = FeatureEngineer()
    sorted_df, _X, _y = prepare_training_data(raw, engineer=engineer)
    return sorted_df


@dataclass
class EvaluationReport:
    """Full Phase-18 evaluation artifact (JSON-serializable via to_dict)."""
    dataset: dict
    splits: dict
    models: list
    produced_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds"))
    config: dict = field(default_factory=dict)
    notes: list = field(default_factory=list)
    production_db_sha256: Optional[str] = None
    production_db_unchanged: Optional[bool] = None

    def to_dict(self) -> dict:
        return {
            "dataset": self.dataset,
            "splits": self.splits,
            "models": self.models,
            "produced_at": self.produced_at,
            "config": self.config,
            "notes": self.notes,
            "production_db_sha256": self.production_db_sha256,
            "production_db_unchanged": self.production_db_unchanged,
        }


def _split_summary(df: pd.DataFrame, name: str) -> dict:
    if len(df) == 0:
        return {"name": name, "size": 0, "date_min": None,
                "date_max": None, "class_distribution": {}}
    d = df["target_match_date"]
    return {
        "name": name,
        "size": int(len(df)),
        "date_min": str(d.min()),
        "date_max": str(d.max()),
        "class_distribution": {
            OUTCOME_LABELS[c]: float(np.mean(df["_label_idx"].to_numpy() == c))
            for c in range(N_OUTCOMES)
        },
    }


def _empty_report() -> dict:
    return {
        "n_samples": 0,
        "log_loss": 0.0,
        "brier_score": 0.0,
        "accuracy": 0.0,
        "macro_f1": 0.0,
        "per_class_precision_recall": {},
        "calibration": {lbl: [] for lbl in OUTCOME_LABELS},
        "confusion_matrix": [],
        "confusion_labels": list(OUTCOME_LABELS),
        "class_distribution": {},
        "labels": list(OUTCOME_LABELS),
    }


def _model_config(model) -> dict:
    if isinstance(model, FrequencyBaseline):
        return {
            "type": "frequency",
            "class_priors": [float(v) for v in np.asarray(model._class_proba).ravel()],
        }
    if isinstance(model, LogisticRegressionClassifier):
        scaler = getattr(model, "_scaler", {}) or {}
        mean = np.asarray(scaler.get("mean", [])).ravel()
        std = np.asarray(scaler.get("std", [])).ravel()
        return {
            "type": "logistic_regression",
            "learning_rate": float(model.learning_rate),
            "n_iterations": int(model.n_iterations),
            "l2_lambda": float(model.l2_lambda),
            "random_state": model.random_state,
            "fitted_on_train_only": True,
            "scaler_mean_train": [float(x) for x in mean],
            "scaler_std_train": [float(x) for x in std],
        }
    return {"type": type(model).__name__}



def run_prediction_evaluation(
    db_path=None,
    *,
    provider: str = "statsbomb",
    provider_competition_id: str = "2",
    season_name: Optional[str] = None,
    train_frac: float = 0.60,
    val_frac: float = 0.20,
    lr_learning_rate: float = 0.05,
    lr_n_iterations: int = 1000,
    lr_l2_lambda: float = 0.01,
    lr_seed: int = 42,
) -> EvaluationReport:
    """Run a leakage-safe, read-only model comparison (FrequencyBaseline vs LR).

    The database SHA-256 is captured before and after loading; any change raises
    RuntimeError (no writes to the production DB).  Models are compared on a
    chronological train/val/test split; LR is fit on training rows only.
    """
    db = resolve_db_path(db_path)
    sha_before = _sha256(db)
    df = build_feature_dataset(
        db, provider=provider,
        provider_competition_id=str(provider_competition_id),
        season_name=season_name,
    )
    sha_after = _sha256(db)
    if sha_before is not None and sha_before != sha_after:
        raise RuntimeError(
            "Database SHA-256 changed during evaluation -- aborting (writes to a "
            "read-only DB violate the Phase-18 leakage contract)."
        )

    feats = feature_columns(df)
    train_df, val_df, test_df = chronological_split(
        df, train_frac=train_frac, val_frac=val_frac, date_col="target_match_date"
    )
    Xtr = train_df[feats].to_numpy(dtype=float)
    ytr = train_df["_label_idx"].to_numpy(dtype=int)
    Xva = val_df[feats].to_numpy(dtype=float)
    yva = val_df["_label_idx"].to_numpy(dtype=int)
    Xte = test_df[feats].to_numpy(dtype=float)
    yte = test_df["_label_idx"].to_numpy(dtype=int)

    fb = FrequencyBaseline().fit(ytr)
    lr = LogisticRegressionClassifier(
        learning_rate=lr_learning_rate, n_iterations=lr_n_iterations,
        l2_lambda=lr_l2_lambda, random_state=lr_seed,
    ).fit(Xtr, ytr)

    def _eval(model) -> dict:
        splits = {}
        for sp, (X, y) in (
            ("train", (Xtr, ytr)), ("val", (Xva, yva)), ("test", (Xte, yte))
        ):
            if len(y) == 0:
                splits[sp] = _empty_report()
                continue
            proba = model.predict_proba(X)
            pred = proba.argmax(axis=1).astype(int)
            splits[sp] = classification_report(y, proba, y_pred=pred)
        return {"config": _model_config(model), "n_train": int(len(ytr)),
                "n_val": int(len(yva)), "n_test": int(len(yte)), "splits": splits}

    models = [_eval(fb), _eval(lr)]
    for m, nm in zip(models, ["frequency_baseline", "logistic_regression"]):
        m["name"] = nm

    return EvaluationReport(
        dataset={
            "provider": provider,
            "provider_competition_id": str(provider_competition_id),
            "season_name": season_name,
            "n_matches": int(len(df)),
            "n_features": int(len(feats)),
            "feature_cols": list(feats),
            "split_rule": "chronological (no shuffle), ascending by target_match_date",
            "preprocessing_fit_on": "train only",
            "excluded_leakage_columns": sorted(LEAKAGE_COLS),
        },
        splits={
            "train": _split_summary(train_df, "train"),
            "val": _split_summary(val_df, "val"),
            "test": _split_summary(test_df, "test"),
            "raw_counts": {"total": int(len(df)), "train": int(len(train_df)),
                           "val": int(len(val_df)), "test": int(len(test_df))},
        },
        models=models,
        config={"train_frac": train_frac, "val_frac": val_frac,
                "lr": {"learning_rate": lr_learning_rate, "n_iterations": lr_n_iterations,
                       "l2_lambda": lr_l2_lambda, "random_state": lr_seed}},
        notes=[
            "Read-only evaluation: SELECT-only loaders; SHA-256 re-checked after "
            "load; any modification aborts.",
            "No cross-competition leakage: load-time provider_competition_id filter "
            "isolates a single competition.",
            "No same-day / future leakage: feature history for match m uses only "
            "matches strictly older than m.",
            "Preprocessing fit on TRAIN ONLY: class priors and the LR imputer/scal"
            "er are fit on train and re-applied to val/test via the model object.",
            "No model selection: LR hyperparameters are fixed; metrics are "
            "descriptive, not ranked.",
        ],
        production_db_sha256=sha_after,
        production_db_unchanged=(True if sha_before is None else (sha_before == sha_after)),
    )



def _json_default(o):
    """JSON fallback for numpy / pandas / datetime objects."""
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        f = float(o)
        return None if (np.isnan(f) or np.isinf(f)) else f
    if isinstance(o, float):
        return None if (np.isnan(o) or np.isinf(o)) else o
    if isinstance(o, (datetime, pd.Timestamp)):
        return o.isoformat()
    if o is pd.NaT:
        return None
    if isinstance(o, (set, frozenset)):
        return list(o)
    return str(o)


def save_report(report: EvaluationReport, output_dir: str, name: str = "phase18_evaluation") -> dict:
    """Write JSON + readable TXT artifacts. NEVER writes to the production DB.

    Returns {"json": ..., "txt": ..., "dir": ...}.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    base = out / f"{name}_{ts}"
    json_path = base.with_suffix(".json")
    txt_path = base.with_suffix(".txt")
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(report.to_dict(), fh, indent=2, default=_json_default)
    _write_txt(report, txt_path)
    return {"json": str(json_path), "txt": str(txt_path), "dir": str(out)}


def _write_txt(report: EvaluationReport, path) -> None:
    L = []
    L.append("Phase-18 Prediction Evaluation Report")
    L.append("======================================")
    L.append(f"Produced: {report.produced_at}")
    d = report.dataset
    L.append("")
    L.append("Dataset")
    L.append("-------")
    L.append(f"provider={d.get('provider')} competition_id={d.get('provider_competition_id')} season={d.get('season_name')}")
    L.append(f"n_matches={d.get('n_matches')} n_features={d.get('n_features')}")
    rc = report.splits.get("raw_counts", {})
    L.append(f"split counts: {rc}")
    L.append(f"production DB sha256: {report.production_db_sha256}")
    L.append(f"production DB unchanged: {report.production_db_unchanged}")
    L.append("")
    for m in report.models:
        cfg = m["config"]
        L.append(f"Model: {m.get('name')} (type={cfg.get('type')})")
        L.append(f"  train={m['n_train']} val={m['n_val']} test={m['n_test']}")
        for sp in ("train", "val", "test"):
            r = m["splits"].get(sp, {})
            L.append(f"  [{sp}] n={r.get('n_samples', '?')} acc={r.get('accuracy', '?')} "
                     f"brier={r.get('brier_score', '?')} logloss={r.get('log_loss', '?')} "
                     f"macroF1={r.get('macro_f1', '?')}")
            pc = r.get("per_class_precision_recall", {})
            if pc:
                L.append("    per-class P/R/F1:")
                for lbl in OUTCOME_LABELS:
                    c = pc.get(lbl, {})
                    prec = c.get("precision", 0.0)
                    rec = c.get("recall", 0.0)
                    f1v = c.get("f1", 0.0)
                    sup = c.get("support", 0)
                    L.append(f"      {lbl:9s} P={prec:.4f} R={rec:.4f} F1={f1v:.4f} N={sup}")
            cm = r.get("confusion_matrix")
            if cm is not None and len(cm) > 0:
                L.append("    confusion (rows=true, cols=pred, home/draw/away):")
                for row in cm:
                    L.append("      " + " ".join(f"{v:7.0f}" for v in row))
        L.append("")
    L.append("Notes")
    L.append("-----")
    for note in report.notes:
        L.append(f"- {note}")
    Path(path).write_text("\n".join(L), encoding="utf-8")



def classification_report(
    y_true,
    y_prob,
    y_pred=None,
    labels=None,
) -> dict:
    """Full evaluation: base metrics + macro-F1 + per-class P/R + calibration."""
    labels = _labels_arg(labels)
    y_prob = np.asarray(y_prob, dtype=float)
    n = y_prob.shape[0] if y_prob.ndim == 2 else 0
    if n == 0:
        z = np.zeros(0, dtype=int)
        return {
            "log_loss": 0.0, "brier_score": 0.0, "accuracy": 0.0, "macro_f1": 0.0,
            "per_class_precision_recall": {
                lb: {"precision": 0.0, "recall": 0.0, "f1": 0.0, "support": 0}
                for lb in labels
            },
            "calibration": {lb: [] for lb in labels},
            "confusion_matrix": build_confusion_matrix(z, z, labels),
            "confusion_labels": labels,
            "class_distribution": class_distribution(z, labels),
            "n_samples": 0, "labels": labels,
        }
    base = evaluate_predictions(y_true, y_prob, y_pred=y_pred, labels=labels)
    yt = _to_index(y_true, labels)
    yp = np.argmax(y_prob, axis=1) if y_pred is None else _to_index(y_pred, labels)
    return {
        "log_loss": base["log_loss"],
        "brier_score": base["brier_score"],
        "accuracy": base["accuracy"],
        "macro_f1": compute_macro_f1(yt, yp, labels),
        "per_class_precision_recall": compute_precision_recall(yt, yp, labels),
        "calibration": compute_calibration(yt, y_prob, labels),
        "confusion_matrix": base["confusion_matrix"],
        "confusion_labels": labels,
        "class_distribution": base["class_distribution"],
        "n_samples": base["n_samples"],
        "labels": labels,
    }


def feature_columns(df: pd.DataFrame) -> list:
    """Return feature column names (exclude label + match-identifier columns)."""
    return [c for c in df.columns if c not in LEAKAGE_COLS]
