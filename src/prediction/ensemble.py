"""Monotone regularized probability calibration."""
import numpy as np
from sklearn.linear_model import LogisticRegression

def fit_platt_calibrator(y, score, *, regularization_c: float = 1e6) -> LogisticRegression | None:
    """Fit a one-dimensional probability calibrator.

    With only a handful of positives, an effectively unregularized slope can
    create extreme probabilities while leaving AUC unchanged. New workflows
    can pass a smaller C explicitly; the legacy default is retained so the
    previously frozen baseline remains reproducible.
    """
    y = np.asarray(y, dtype=int)
    score = np.asarray(score, dtype=float)
    mask = np.isfinite(score)
    if len(np.unique(y[mask])) < 2:
        return None
    if regularization_c <= 0:
        raise ValueError("regularization_c must be positive")
    model = LogisticRegression(C=regularization_c, solver="lbfgs", max_iter=2000)
    model.fit(score[mask].reshape(-1, 1), y[mask])
    return model

def calibrate_probability(calibrator, score) -> np.ndarray:
    score = np.asarray(score, dtype=float)
    if calibrator is None:
        return np.clip(score, 0.0, 1.0)
    return calibrator.predict_proba(score.reshape(-1, 1))[:, 1]
