"""Phase 19: Elo ratings (chronological, read-only, deterministic baseline).

Classic Elo with a home advantage and the standard 3-outcome update
(win = 1, draw = 0.5, loss = 0). Ratings are updated strictly chronologically,
so a match is scored from ratings learned only from strictly-earlier results
(no same-day / future leakage). K, home advantage and prediction thresholds are
fixed in advance; no hyperparameter tuning is performed against any held-out
partition. Elo naturally yields an *expected-score* per match (home team expected
points in [0, 1]); this module also emits a fixed-rule 3-class prediction using
documented thresholds. Descriptive baseline, not a "best model".
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from sqlite3 import connect as _sqlite3_connect
from typing import Optional

import numpy as np
import pandas as pd

from .baseline import (
    OUTCOME_LABELS,
    build_confusion_matrix,
    class_distribution,
    compute_outcome_label,
)
from ..database import resolve_db_path
from .evaluation import _json_default, compute_macro_f1
from ..features.engineering import load_matches_for_features

__all__ = ["EloModel", "EloReport", "run_elo_evaluation", "save_elo_report"]


def _sha256(path) -> Optional[str]:
    path = Path(path)
    if not path.exists():
        return None
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _team_name_map(db_path) -> dict:
    """Read-only id -> team_name for pretty reporting (SELECT only)."""
    path = Path(db_path)
    mapping: dict = {}
    if not path.exists():
        return mapping
    con = _sqlite3_connect(f"file:{path}?mode=ro", uri=True)
    try:
        for tid, name in con.execute(
            "SELECT id, team_name FROM teams WHERE team_name IS NOT NULL"
        ):
            mapping[tid] = name
    finally:
        con.close()
    return mapping


@dataclass
class EloMatch:
    match_date: str
    home_team_id: int
    away_team_id: int
    home_goals: Optional[float]
    away_goals: Optional[float]
    home_rating: float
    away_rating: float
    expected_home: float
    actual_home: Optional[float]
    outcome_label: Optional[str]


class EloModel:
    """Classic Elo. ``K`` and ``home_advantage`` are in Elo points."""

    def __init__(
        self,
        K: float = 20.0,
        home_advantage: float = 50.0,
        initial_rating: float = 1500.0,
        home_win_threshold: float = 0.60,
        away_win_threshold: float = 0.40,
    ) -> None:
        self.K = float(K)
        self.home_advantage = float(home_advantage)
        self.initial_rating = float(initial_rating)
        self.home_win_threshold = float(home_win_threshold)
        self.away_win_threshold = float(away_win_threshold)
        self.ratings_: dict = {}
        self.history_: list[EloMatch] = []

    @staticmethod
    def _score(hg, ag) -> Optional[float]:
        if hg is None or ag is None:
            return None
        try:
            hg = float(hg)
            ag = float(ag)
        except (TypeError, ValueError):
            return None
        if pd.isna(hg) or pd.isna(ag):
            return None
        if hg > ag:
            return 1.0
        if hg < ag:
            return 0.0
        return 0.5

    def fit(self, df: pd.DataFrame, date_col: str = "match_date") -> "EloModel":
        """Same-day-safe Elo fit (Phase 21 leakage fix).

        All fixtures sharing a ``match_date`` are scored against the SAME
        pre-date ratings; the zero-sum updates of each date are accumulated and
        committed only after every same-date fixture has been scored, so a
        fixture never observes the just-applied rating update of a
        concurrently-played fixture -- matching the strict less-than-date
        guarantee used by the Poisson model.
        """
        ordered = df.sort_values(date_col).reset_index(drop=True)
        ratings: dict = {}
        history: list[EloMatch] = []
        # Phase-21 same-day-leakage fix: all fixtures sharing a ``match_date``
        # are scored against the SAME pre-date ratings; the daily zero-sum
        # updates are accumulated and committed only AFTER every same-date
        # fixture has been scored.  This guarantees a fixture never observes
        # the just-applied rating update of a concurrently-played fixture.
        cur_date = None
        baseline: dict = {}
        deltas: dict = {}
        for row in ordered.itertuples(index=False):
            d = str(row.match_date)
            if d != cur_date:
                # Commit the previous date's batched updates, then open a fresh
                # date with a clean pre-date baseline for every involved team.
                for tid, base in baseline.items():
                    ratings[tid] = base + deltas.get(tid, 0.0)
                cur_date = d
                baseline = {}
                deltas = {}
            h = int(row.home_team_id)
            a = int(row.away_team_id)
            # Baseline rating for this date -- captured once per date, so same-date
            # opponents cannot influence one another on the same date.
            baseline.setdefault(h, ratings.get(h, self.initial_rating))
            baseline.setdefault(a, ratings.get(a, self.initial_rating))
            rh = baseline[h]
            ra = baseline[a]
            exp = 1.0 / (1.0 + 10.0 ** ((ra - rh - self.home_advantage) / 400.0))
            s = self._score(row.home_goals, row.away_goals)
            history.append(EloMatch(
                match_date=d,
                home_team_id=h, away_team_id=a,
                home_goals=row.home_goals, away_goals=row.away_goals,
                home_rating=rh, away_rating=ra,
                expected_home=exp, actual_home=s,
                outcome_label=compute_outcome_label(row.home_goals, row.away_goals),
            ))
            if s is not None:
                # Accumulate this date's zero-sum update; applied in batch at the
                # next date boundary (or at flush) so it cannot leak intra-day.
                deltas[h] = deltas.get(h, 0.0) + self.K * (s - exp)
                deltas[a] = deltas.get(a, 0.0) - self.K * (s - exp)
        # Flush the final date's updates.
        for tid, base in baseline.items():
            ratings[tid] = base + deltas.get(tid, 0.0)
        self.ratings_ = ratings
        self.history_ = history
        return self

    def final_ratings(self) -> dict:
        return dict(self.ratings_)

    def rating_history(self) -> pd.DataFrame:
        return pd.DataFrame([m.__dict__ for m in self.history_])

    def predict_home_win_prob(self, home_rating: float, away_rating: float) -> float:
        """Expected home SCORE = standard Elo expected score (not a 3-class prob)."""
        return 1.0 / (1.0 + 10.0 ** ((away_rating - home_rating - self.home_advantage) / 400.0))

    def evaluate(self) -> dict:
        """Online, leak-free evaluation: each match vs. its pre-match ratings."""
        valid = [m for m in self.history_ if m.actual_home is not None]
        if not valid:
            return {"n": 0, "brier_expected_score": 0.0, "accuracy": 0.0,
                    "macro_f1": 0.0, "confusion_matrix": [],
                    "confusion_labels": list(OUTCOME_LABELS),
                    "class_distribution": {},
                    "home_win_threshold": self.home_win_threshold,
                    "away_win_threshold": self.away_win_threshold}
        labels = list(OUTCOME_LABELS)
        idx = {"home_win": 0, "draw": 1, "away_win": 2}
        y_true = np.array([idx[m.outcome_label] for m in valid], dtype=int)
        preds = []
        exps = []
        scores = []
        for m in valid:
            exps.append(m.expected_home)
            scores.append(m.actual_home)
            if m.expected_home >= self.home_win_threshold:
                preds.append(0)
            elif m.expected_home <= self.away_win_threshold:
                preds.append(2)
            else:
                preds.append(1)
        y_pred = np.array(preds, dtype=int)
        exps = np.array(exps, dtype=float)
        scores = np.array(scores, dtype=float)
        brier = float(np.mean((exps - scores) ** 2))
        return {
            "n": int(len(valid)),
            "brier_expected_score": brier,
            "accuracy": float(np.mean(y_true == y_pred)),
            "macro_f1": float(compute_macro_f1(y_true, y_pred, labels)),
            "confusion_matrix": build_confusion_matrix(y_true, y_pred, labels).tolist(),
            "confusion_labels": labels,
            "class_distribution": class_distribution(y_true, labels),
            "home_win_threshold": self.home_win_threshold,
            "away_win_threshold": self.away_win_threshold,
        }


@dataclass
class EloReport:
    dataset: dict
    ratings: dict
    eval_metrics: dict
    config: dict
    produced_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds"))
    notes: list = field(default_factory=list)
    production_db_sha256: Optional[str] = None
    production_db_unchanged: Optional[bool] = None

    def to_dict(self) -> dict:
        return {
            "dataset": self.dataset,
            "ratings": self.ratings,
            "eval_metrics": self.eval_metrics,
            "config": self.config,
            "produced_at": self.produced_at,
            "notes": self.notes,
            "production_db_sha256": self.production_db_sha256,
            "production_db_unchanged": self.production_db_unchanged,
        }


def load_elo_matches(
    db_path=None,
    *,
    provider: str = "statsbomb",
    provider_competition_id: str = "2",
    season_name: Optional[str] = None,
) -> pd.DataFrame:
    db = resolve_db_path(db_path)
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
    return raw.sort_values("match_date").reset_index(drop=True)


def run_elo_evaluation(
    db_path=None,
    *,
    provider: str = "statsbomb",
    provider_competition_id: str = "2",
    season_name: Optional[str] = None,
    K: float = 20.0,
    home_advantage: float = 50.0,
    initial_rating: float = 1500.0,
) -> EloReport:
    """Run a leak-free chronological Elo evaluation over one competition/season.

    The database SHA-256 is captured before and after loading; any change raises
    RuntimeError (no writes to the production DB).
    """
    db = resolve_db_path(db_path)
    sha_before = _sha256(db)
    df = load_elo_matches(db, provider=provider,
                          provider_competition_id=str(provider_competition_id),
                          season_name=season_name)
    sha_after = _sha256(db)
    if sha_before is not None and sha_before != sha_after:
        raise RuntimeError(
            "Database SHA-256 changed during Elo evaluation -- aborting "
            "(writes to a read-only DB violate the leakage contract)."
        )
    names = _team_name_map(db)
    model = EloModel(K=K, home_advantage=home_advantage, initial_rating=initial_rating)
    model.fit(df)
    ratings = {names.get(tid, str(tid)): float(r) for tid, r in model.final_ratings().items()}
    teams = set(df["home_team_id"]).union(set(df["away_team_id"]))
    metrics = model.evaluate()
    return EloReport(
        dataset={
            "provider": provider,
            "provider_competition_id": str(provider_competition_id),
            "season_name": season_name,
            "n_matches": int(len(df)),
            "n_teams": int(len(teams)),
            "date_min": str(df["match_date"].min()),
            "date_max": str(df["match_date"].max()),
        },
        ratings=ratings,
        eval_metrics=metrics,
        config={"K": float(K), "home_advantage": float(home_advantage),
                "initial_rating": float(initial_rating),
                "home_win_threshold": 0.60, "away_win_threshold": 0.40},
        notes=[
            "Read-only evaluation: SELECT-only loaders; SHA-256 re-checked after "
            "load; any modification aborts.",
            "No cross-competition leakage: load-time provider_competition_id filter "
            "isolates a single competition.",
            "Online prediction: each match is scored from ratings learned from "
            "strictly-earlier results (chronological order, no shuffle, no future info).",
            "No model selection: K / home_advantage / initial_rating are fixed; "
            "prediction thresholds are fixed, not tuned on any held-out partition.",
            "Elo expected score is a continuous 0-1 score, not a calibrated "
            "3-class probability; metrics are descriptive, not ranked.",
        ],
        production_db_sha256=sha_after,
        production_db_unchanged=(True if sha_before is None else sha_before == sha_after),
    )



def save_elo_report(report: EloReport, output_dir: str,
                    name: str = "phase19_elo") -> dict:
    """Write JSON + readable TXT artifacts. NEVER writes to the production DB."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    base = out / f"{name}_{ts}"
    json_path = base.with_suffix(".json")
    txt_path = base.with_suffix(".txt")
    json_path.write_text(
        json.dumps(report.to_dict(), indent=2, default=_json_default),
        encoding="utf-8",
    )
    _write_elo_txt(report, txt_path)
    return {"json": str(json_path), "txt": str(txt_path), "dir": str(out)}


def _write_elo_txt(report: EloReport, path) -> None:
    L = []
    L.append("Phase-19 Elo Ratings Report")
    L.append("===========================")
    L.append(f"Produced: {report.produced_at}")
    d = report.dataset
    L.append(f"provider={d.get('provider')} competition_id={d.get('provider_competition_id')} season={d.get('season_name')}")
    L.append(f"n_matches={d.get('n_matches')} n_teams={d.get('n_teams')} date_min={d.get('date_min')} date_max={d.get('date_max')}")
    L.append(f"production DB sha256: {report.production_db_sha256}")
    L.append(f"production DB unchanged: {report.production_db_unchanged}")
    L.append("")
    L.append("Config")
    L.append("------")
    for k, v in report.config.items():
        L.append(f"  {k}: {v}")
    L.append("")
    L.append("Final ratings (top 15)")
    L.append("----------------------")
    ordered = sorted(report.ratings.items(), key=lambda kv: kv[1], reverse=True)
    for i, (team, rating) in enumerate(ordered[:15], 1):
        L.append(f"  {i:>2d}. {str(team):<24s} {rating:.1f}")
    if len(ordered) > 15:
        L.append(f"  ... and {len(ordered) - 15} more")
    L.append("")
    L.append("Online evaluation (each match vs. pre-match ratings)")
    L.append("----------------------------------------------------")
    m = report.eval_metrics
    L.append(f"  n={m.get('n')}")
    L.append(f"  accuracy: {m.get('accuracy'):.4f}")
    L.append(f"  macro_f1: {m.get('macro_f1'):.4f}")
    L.append(f"  brier_expected_score: {m.get('brier_expected_score'):.4f}")
    cm = m.get("confusion_matrix") or []
    if len(cm) == 3:
        L.append("  confusion (rows=true, cols=pred, home/draw/away):")
        for row in cm:
            L.append("    " + " ".join(f"{v:7.0f}" for v in row))
    cd = m.get("class_distribution") or {}
    if cd:
        L.append(f"  class distribution counts: {cd.get('counts')}")
    L.append("")
    L.append("Notes")
    L.append("-----")
    for note in report.notes:
        L.append(f"- {note}")
    Path(path).write_text("\n".join(L), encoding="utf-8")
