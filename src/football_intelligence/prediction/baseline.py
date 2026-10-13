"""Baseline models and evaluation utilities for match prediction.

Two model types are provided:

1. FrequencyBaseline -- a no-skill baseline that predicts class-prior
   probabilities from the training distribution.

2. LogisticRegressionClassifier -- a multinomial logistic-regression
   classifier trained via batch gradient descent using only numpy.

Outcome labels: home_goals > away_goals -> "home_win" (0), equal -> "draw" (1),
less -> "away_win" (2).  NULL goals cannot be labelled.

Design: no real DB modified, no external APIs, features use only historical
results (guaranteed by FeatureEngineer).
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

OUTCOME_LABELS: tuple[str, ...] = ("home_win", "draw", "away_win")
OUTCOME_INDEX: dict[str, int] = {"home_win": 0, "draw": 1, "away_win": 2}
N_OUTCOMES: int = 3


def compute_outcome_label(home_goals: float, away_goals: float) -> Optional[str]:
    """Determine the match outcome label from goal counts.

    Returns "home_win", "draw", "away_win", or None if either is NaN.
    """
    if pd.isna(home_goals) or pd.isna(away_goals):
        return None
    hg = float(home_goals)
    ag = float(away_goals)
    if hg > ag:
        return "home_win"
    if hg < ag:
        return "away_win"
    return "draw"


def chronological_split(
    df: pd.DataFrame,
    train_frac: float = 0.6,
    val_frac: float = 0.2,
    date_col: str = "match_date",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split df into train/val/test by chronological order. No shuffling."""
    if not 0 <= train_frac <= 1:
        raise ValueError(f"train_frac must be in [0, 1], got {train_frac}")
    if not 0 <= val_frac <= 1:
        raise ValueError(f"val_frac must be in [0, 1], got {val_frac}")
    if train_frac + val_frac >= 1.0:
        raise ValueError(
            f"train_frac ({train_frac}) + val_frac ({val_frac}) must be < 1"
        )
    if date_col not in df.columns:
        raise ValueError(f"Column '{date_col}' not found in DataFrame")

    sorted_df = df.sort_values(date_col).reset_index(drop=True)
    n = len(sorted_df)
    if n == 0:
        return (sorted_df.iloc[0:0].copy(),
                sorted_df.iloc[0:0].copy(),
                sorted_df.iloc[0:0].copy())

    n_train = int(np.floor(n * train_frac))
    n_val = int(np.floor(n * val_frac))
    n_train = max(n_train, 1)
    test_start = n_train + n_val

    train_df = sorted_df.iloc[:n_train].copy()
    val_df = sorted_df.iloc[n_train:test_start].copy()
    test_df = sorted_df.iloc[test_start:].copy()
    return train_df, val_df, test_df


# ---------------------------------------------------------------------------
# Preprocessing: impute + scale
# ---------------------------------------------------------------------------


def _impute_and_scale(
    X: np.ndarray,
    fill_value: float = 0.0,
) -> tuple[np.ndarray, dict]:
    """Replace NaN/inf with fill_value and standardize columns.

    Mean and std are computed from the input (training) data.  The scaler
    state is returned so validation/test data can use the same parameters.
    """
    if X.size == 0:
        return X, {"mean": np.array([]), "std": np.array([])}

    X_clean = np.where(np.isfinite(X), X, fill_value).astype(float)
    mean = X_clean.mean(axis=0)
    std = X_clean.std(axis=0)
    std_safe = np.where(std == 0, 1.0, std)
    X_scaled = (X_clean - mean) / std_safe
    return X_scaled, {"mean": mean, "std": std_safe}


def _apply_scaler(
    X: np.ndarray,
    scaler: dict,
    fill_value: float = 0.0,
) -> np.ndarray:
    """Apply a pre-fitted scaler to new data.

    NaN/inf values are replaced with fill_value before scaling.
    """
    if X.size == 0:
        return X
    X_clean = np.where(np.isfinite(X), X, fill_value).astype(float)
    mean = scaler["mean"]
    std = scaler["std"]
    if mean.size == 0:
        return X_clean
    n_cols = X_clean.shape[1]
    n_mean = mean.shape[0]
    if n_cols != n_mean:
        if n_cols > n_mean:
            pad = n_cols - n_mean
            mean = np.pad(mean, (0, pad))
            std = np.pad(std, (0, pad), constant_values=1.0)
        else:
            mean = mean[:n_cols]
            std = std[:n_cols]
    return (X_clean - mean) / std



# ---------------------------------------------------------------------------
# Frequency baseline model
# ---------------------------------------------------------------------------


class FrequencyBaseline:
    """A no-skill baseline that predicts class-prior probabilities.

    During fit, the frequency of each outcome class in the training
    labels is recorded.  During predict_proba, the same class distribution
    is returned for every input row, regardless of features.
    """

    def __init__(self) -> None:
        self._class_proba: np.ndarray = np.zeros(N_OUTCOMES)
        self._is_fitted: bool = False

    def fit(self, y: np.ndarray) -> "FrequencyBaseline":
        """Record the class distribution from training labels.

        Args:
            y: 1-D array of integer class labels (0, 1, or 2).
        """
        y = np.asarray(y, dtype=int)
        if len(y) == 0:
            self._class_proba = np.ones(N_OUTCOMES) / N_OUTCOMES
        else:
            counts = np.bincount(y, minlength=N_OUTCOMES)[:N_OUTCOMES]
            total = counts.sum()
            if total == 0:
                self._class_proba = np.ones(N_OUTCOMES) / N_OUTCOMES
            else:
                self._class_proba = counts / total
        self._is_fitted = True
        return self

    def predict_proba(self, X: Optional[np.ndarray] = None) -> np.ndarray:
        """Return class-prior probabilities for each row.

        Args:
            X: Ignored except for shape.  Only the number of rows matters.

        Returns:
            Array of shape (n_rows, 3) with identical rows.
        """
        if not self._is_fitted:
            raise RuntimeError(
                "Model must be fitted before calling predict_proba"
            )
        if X is not None and hasattr(X, "shape"):
            n = X.shape[0]
        else:
            n = 1
        return np.tile(self._class_proba, (n, 1))

    def predict(self, X: Optional[np.ndarray] = None) -> np.ndarray:
        """Return the most likely outcome class index for each row."""
        proba = self.predict_proba(X)
        return np.argmax(proba, axis=1)

    @property
    def class_distribution(self) -> dict[str, float]:
        """Return the class distribution as a {label: probability} dict."""
        return {
            OUTCOME_LABELS[i]: float(self._class_proba[i])
            for i in range(N_OUTCOMES)
        }



# ---------------------------------------------------------------------------
# Logistic regression (numpy, multinomial softmax)
# ---------------------------------------------------------------------------


class LogisticRegressionClassifier:
    """Multinomial logistic-regression classifier implemented with numpy.

    Uses softmax with L2 regularization, trained via batch gradient descent.
    """

    def __init__(
        self,
        learning_rate: float = 0.01,
        n_iterations: int = 1000,
        l2_lambda: float = 0.01,
        random_state: Optional[int] = 42,
    ) -> None:
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.l2_lambda = l2_lambda
        self.random_state = random_state
        self._weights: Optional[np.ndarray] = None
        self._bias: Optional[np.ndarray] = None
        self._scaler: dict = {"mean": np.array([]), "std": np.array([])}
        self._fill_value: float = 0.0
        self._is_fitted: bool = False

    @property
    def weights(self) -> Optional[np.ndarray]:
        """Weight matrix of shape (n_features, n_classes)."""
        return self._weights

    @property
    def bias(self) -> Optional[np.ndarray]:
        """Bias vector of shape (n_classes,)."""
        return self._bias

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LogisticRegressionClassifier":
        """Fit the model using batch gradient descent.

        Args:
            X: Feature matrix of shape (n_samples, n_features).
            y: Integer class labels with values in {0, 1, 2}.
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=int)
        n_samples, n_features = X.shape

        X_scaled, self._scaler = _impute_and_scale(
            X, fill_value=self._fill_value
        )

        rng = np.random.RandomState(self.random_state)
        if n_samples == 0:
            self._weights = np.zeros((max(n_features, 1), N_OUTCOMES))
            self._bias = np.zeros(N_OUTCOMES)
            self._is_fitted = True
            return self

        self._weights = rng.randn(n_features, N_OUTCOMES) * 0.01
        self._bias = np.zeros(N_OUTCOMES)

        y_onehot = np.zeros((n_samples, N_OUTCOMES))
        y_onehot[np.arange(n_samples), y] = 1.0

        for _ in range(self.n_iterations):
            linear = X_scaled @ self._weights + self._bias
            exp_vals = np.exp(linear - linear.max(axis=1, keepdims=True))
            probs = exp_vals / exp_vals.sum(axis=1, keepdims=True)
            grad_w = (X_scaled.T @ (probs - y_onehot)) / n_samples
            grad_w += (self.l2_lambda / n_samples) * self._weights
            grad_b = (probs - y_onehot).mean(axis=0)
            self._weights -= self.learning_rate * grad_w
            self._bias -= self.learning_rate * grad_b

        self._is_fitted = True
        return self


    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return predicted class probabilities of shape (n_samples, 3).

        Raises RuntimeError if the model has not been fitted.
        """
        if not self._is_fitted:
            raise RuntimeError(
                "Model must be fitted before calling predict_proba"
            )
        X = np.asarray(X, dtype=float)
        if X.size == 0:
            return np.zeros((0, N_OUTCOMES))
        X_scaled = _apply_scaler(
            X, self._scaler, fill_value=self._fill_value
        )
        linear = X_scaled @ self._weights + self._bias
        exp_vals = np.exp(linear - linear.max(axis=1, keepdims=True))
        probs = exp_vals / exp_vals.sum(axis=1, keepdims=True)
        return probs

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return predicted class indices."""
        return np.argmax(self.predict_proba(X), axis=1)

    def predict_label(self, X: np.ndarray) -> np.ndarray:
        """Return predicted class labels (strings)."""
        return np.array(
            [OUTCOME_LABELS[i] for i in self.predict(X)]
        )


# ---------------------------------------------------------------------------
# Feature / label preparation
# ---------------------------------------------------------------------------


def prepare_training_data(
    df: pd.DataFrame,
    engineer=None,
    feature_cols=None,
    label_col: str = "outcome_label",
    date_col: str = "target_match_date",
) -> tuple:
    """Prepare feature matrix X and label vector y from a match DataFrame.

    If engineer (a FeatureEngineer) is provided, features are computed via
    the leakage-safe pipeline.  Only rows with a valid outcome label are
    retained.  The result is sorted chronologically by date_col.

    Args:
        df: Match DataFrame with match_date, home_goals, away_goals, match_id.
        engineer: Optional FeatureEngineer to compute features.
        feature_cols: Column names to use as features (when no engineer).
        label_col: Name for the outcome label column.
        date_col: Column to sort by for chronological ordering.

    Returns:
        Tuple of (sorted_df, X, y) where sorted_df is sorted by date with a
        new outcome_label column; X is a 2-D float array; y is integer
        class indices.
    """
    from ..features.engineering import FeatureEngineer

    required = {"match_date", "home_goals", "away_goals", "match_id"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"DataFrame missing required columns: {missing}")

    if engineer is None:
        engineer = FeatureEngineer()

    features_df = engineer.compute_all_features(df)

    # Handle empty or column-less results (e.g. empty input DataFrame)
    if len(features_df) == 0 or "target_match_id" not in features_df.columns:
        return features_df, np.zeros((0, 0), dtype=float), np.zeros(0, dtype=int)

    label_lookup = {}
    for _, row in df.iterrows():
        label = compute_outcome_label(row["home_goals"], row["away_goals"])
        label_lookup[int(row["match_id"])] = label

    features_df["outcome_label"] = features_df["target_match_id"].map(label_lookup)
    valid = features_df[features_df["outcome_label"].notna()].copy()
    valid["_label_idx"] = valid["outcome_label"].map(OUTCOME_INDEX).astype(int)
    valid = valid.sort_values(date_col).reset_index(drop=True)

    if feature_cols is not None:
        cols = [c for c in feature_cols if c in valid.columns]
    else:
        exclude = {
            "target_match_id", "target_match_date",
            "home_team_id", "away_team_id", "features_valid",
            "outcome_label", "_label_idx",
        }
        cols = [c for c in valid.columns if c not in exclude]

    X = valid[cols].to_numpy(dtype=float)
    y = valid["_label_idx"].to_numpy(dtype=int)
    return valid, X, y



# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def compute_log_loss(
    y_true,
    y_prob: np.ndarray,
    labels=None,
    eps: float = 1e-15,
) -> float:
    """Compute multiclass log loss (cross-entropy)."""
    y_true = np.asarray(y_true)
    y_prob = np.clip(np.asarray(y_prob, dtype=float), eps, 1 - eps)

    if y_prob.ndim == 1:
        y_prob = y_prob.reshape(-1, 1)

    n_samples = y_prob.shape[0]

    if y_true.dtype == object or y_true.dtype.kind in ("U", "S"):
        if labels is None:
            raise ValueError("labels must be provided when y_true contains strings")
        label_to_idx = {lbl: i for i, lbl in enumerate(labels)}
        y_idx = np.array([label_to_idx[l] for l in y_true])
    else:
        y_idx = y_true.astype(int)

    n_classes = y_prob.shape[1]
    y_onehot = np.zeros((n_samples, n_classes))
    y_onehot[np.arange(n_samples), y_idx] = 1.0
    loss = -np.sum(y_onehot * np.log(y_prob)) / n_samples
    return float(loss)


def compute_brier_score(y_true, y_prob: np.ndarray, labels=None) -> float:
    """Compute the multiclass Brier score."""
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob, dtype=float)
    n_samples = y_prob.shape[0]

    if y_true.dtype == object or y_true.dtype.kind in ("U", "S"):
        if labels is None:
            raise ValueError("labels must be provided when y_true contains strings")
        label_to_idx = {lbl: i for i, lbl in enumerate(labels)}
        y_idx = np.array([label_to_idx[l] for l in y_true])
    else:
        y_idx = y_true.astype(int)

    n_classes = y_prob.shape[1]
    y_onehot = np.zeros((n_samples, n_classes))
    y_onehot[np.arange(n_samples), y_idx] = 1.0
    return float(np.sum((y_prob - y_onehot) ** 2) / n_samples)


def compute_accuracy(y_true, y_pred, labels=None) -> float:
    """Compute classification accuracy. Returns fraction in [0, 1]."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if (y_true.dtype == object or y_true.dtype.kind in ("U", "S")) and labels is not None:
        l2i = {lbl: i for i, lbl in enumerate(labels)}
        y_true = np.array([l2i[l] for l in y_true])
        y_pred = np.array([l2i[l] for l in y_pred])
    if len(y_true) == 0:
        return 0.0
    return float(np.mean(y_true == y_pred))



def build_confusion_matrix(y_true, y_pred, labels=None) -> np.ndarray:
    """Build a multiclass confusion matrix. Rows=actual, cols=predicted."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if (y_true.dtype == object or y_true.dtype.kind in ("U", "S")) and labels is not None:
        l2i = {lbl: i for i, lbl in enumerate(labels)}
        yt = np.array([l2i[l] for l in y_true])
        yp = np.array([l2i[l] for l in y_pred])
    else:
        yt = y_true.astype(int)
        yp = y_pred.astype(int)
        if labels is None:
            mv = 0
            if len(yt) > 0:
                mv = max(mv, int(yt.max()))
            if len(yp) > 0:
                mv = max(mv, int(yp.max()))
            labels = [str(i) for i in range(mv + 1)]

    n = len(labels) if labels else N_OUTCOMES
    matrix = np.zeros((n, n), dtype=int)
    for t, p in zip(yt, yp):
        if 0 <= t < n and 0 <= p < n:
            matrix[t, p] += 1
    return matrix


def class_distribution(y, labels=None) -> dict:
    """Compute the frequency distribution of outcome classes.

    Returns dict with 'counts', 'percentages', and 'total' keys.
    """
    y = np.asarray(y)
    if (y.dtype == object or y.dtype.kind in ("U", "S")) and labels is not None:
        idx = np.array([OUTCOME_INDEX.get(l, -1) for l in y])
    else:
        idx = y.astype(int)

    n = len(idx)
    counts = {}
    percentages = {}
    for lbl, i in OUTCOME_INDEX.items():
        c = int(np.sum(idx == i))
        counts[lbl] = c
        percentages[lbl] = (100.0 * c / n) if n > 0 else 0.0

    return {"counts": counts, "percentages": percentages, "total": n}


def evaluate_predictions(y_true, y_prob: np.ndarray, y_pred=None, labels=None) -> dict:
    """Compute a full evaluation report.

    Returns dict with log_loss, brier_score, accuracy, confusion_matrix,
    class_distribution, n_samples, labels.
    """
    if labels is None:
        labels = list(OUTCOME_LABELS)

    y_arr = np.asarray(y_true)
    if y_arr.dtype == object or y_arr.dtype.kind in ("U", "S"):
        l2i = {lbl: i for i, lbl in enumerate(labels)}
        y_idx = np.array([l2i[l] for l in y_arr])
    else:
        y_idx = y_arr.astype(int)

    if y_pred is None:
        y_pred_idx = np.argmax(y_prob, axis=1)
    else:
        yp = np.asarray(y_pred)
        if yp.dtype == object or yp.dtype.kind in ("U", "S"):
            y_pred_idx = np.array([l2i[l] for l in yp])
        else:
            y_pred_idx = yp.astype(int)

    return {
        "log_loss": compute_log_loss(y_idx, y_prob, labels),
        "brier_score": compute_brier_score(y_idx, y_prob, labels),
        "accuracy": compute_accuracy(y_idx, y_pred_idx, labels),
        "confusion_matrix": build_confusion_matrix(y_idx, y_pred_idx, labels),
        "class_distribution": class_distribution(y_idx, labels),
        "n_samples": int(len(y_idx)),
        "labels": labels,
    }
