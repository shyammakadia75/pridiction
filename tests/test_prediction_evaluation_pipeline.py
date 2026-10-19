"""Phase-18 tests: end-to-end runner, chronological split, train-only
preprocessing, reproducibility, read-only DB guard, and report artifacts.

All data is SYNTHETIC in disposable databases; the production DB is never
read or written.
"""
import hashlib
import json
from pathlib import Path

import numpy as np

from football_intelligence.prediction.baseline import (
    FrequencyBaseline,
    LogisticRegressionClassifier,
    _impute_and_scale,
    chronological_split,
)
from football_intelligence.prediction.evaluation import (
    EvaluationReport,
    build_feature_dataset,
    feature_columns,
    run_prediction_evaluation,
    save_report,
)
from _phase18_db import STATS, build_two_season_db

LR_KW = dict(learning_rate=0.05, n_iterations=200, l2_lambda=0.01, random_state=42)
BUILD_KW = dict(provider=STATS, provider_competition_id="2")
RUN_KW = dict(provider=STATS, provider_competition_id="2", lr_n_iterations=200)


def test_models_and_readonly_guard(tmp_path):
    db = build_two_season_db(tmp_path)
    rep = run_prediction_evaluation(db_path=db, **RUN_KW)
    assert isinstance(rep, EvaluationReport)
    assert [m["name"] for m in rep.models] == ["frequency_baseline", "logistic_regression"]
    rc = rep.splits["raw_counts"]
    assert rc["total"] == 40 and rc["train"] + rc["val"] + rc["test"] == 40
    # Read-only DB guard: SHA-256 captured before/after; unchanged here.
    assert rep.production_db_unchanged is True
    assert rep.production_db_sha256 == hashlib.sha256(Path(db).read_bytes()).hexdigest()
    for m in rep.models:
        for sp in ("val", "test"):
            r = m["splits"][sp]
            assert r["n_samples"] > 0
            assert np.isfinite(r["accuracy"])
            assert np.isfinite(r["brier_score"])
            assert np.isfinite(r["log_loss"])
            assert np.isfinite(r["macro_f1"])


def test_chronological_split_disjoint_and_ordered(tmp_path):
    df = build_feature_dataset(build_two_season_db(tmp_path), **BUILD_KW)
    tr, va, te = chronological_split(df, train_frac=0.6, val_frac=0.2,
                                     date_col="target_match_date")
    assert len(tr) + len(va) + len(te) == 40
    assert tr["target_match_date"].max() <= va["target_match_date"].min()
    assert va["target_match_date"].max() <= te["target_match_date"].max()
    ids = lambda d: set(d["target_match_id"].tolist())
    assert ids(tr).isdisjoint(ids(va)) and ids(va).isdisjoint(ids(te))


def test_train_only_preprocessing(tmp_path):
    df = build_feature_dataset(build_two_season_db(tmp_path), **BUILD_KW)
    feats = feature_columns(df)
    tr, va, te = chronological_split(df, train_frac=0.6, val_frac=0.2,
                                     date_col="target_match_date")
    Xtr = tr[feats].to_numpy(dtype=float)
    ytr = tr["_label_idx"].to_numpy(dtype=int)
    _, scaler = _impute_and_scale(Xtr)
    lr = LogisticRegressionClassifier(**LR_KW).fit(Xtr, ytr)
    # Scaler must be derived from the TRAIN partition only and reused on val/test.
    assert np.allclose(lr._scaler["mean"], scaler["mean"])
    assert np.allclose(lr._scaler["std"], scaler["std"])
    fb = FrequencyBaseline().fit(ytr)
    exp = np.bincount(ytr, minlength=3).astype(float)
    exp = exp / exp.sum()
    assert np.allclose(fb._class_proba, exp)


def test_reproducible_across_runs(tmp_path):
    db = build_two_season_db(tmp_path)
    r1 = run_prediction_evaluation(db_path=db, **RUN_KW)
    r2 = run_prediction_evaluation(db_path=db, **RUN_KW)
    for m1, m2 in zip(r1.models, r2.models):
        for sp in ("val", "test"):
            assert m1["splits"][sp]["accuracy"] == m2["splits"][sp]["accuracy"]
            assert m1["splits"][sp]["brier_score"] == m2["splits"][sp]["brier_score"]
            assert m1["splits"][sp]["macro_f1"] == m2["splits"][sp]["macro_f1"]


def test_save_report_json_round_trip(tmp_path):
    db = build_two_season_db(tmp_path)
    rep = run_prediction_evaluation(db_path=db, **RUN_KW)
    out = save_report(rep, str(tmp_path / "out"))
    j = json.loads(Path(out["json"]).read_text(encoding="utf-8"))
    assert {"dataset", "splits", "models", "production_db_sha256",
            "production_db_unchanged"}.issubset(j.keys())
    assert len(j["models"]) == 2
    assert j["production_db_unchanged"] is True
    txt = Path(out["txt"]).read_text(encoding="utf-8")
    assert "Phase-18" in txt
    assert "logistic_regression" in txt
    assert "confusion" in txt
