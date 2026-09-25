"""Transparent, bounded calibration hook for member A's future-risk labels."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .event_taxonomy import DIMENSION_LABELS
from .exposure_adjustment import monotone_percentile_penalty
from .scoring_rules import DEFAULT_WEIGHTS, DIMENSION_PENALTY_CAPS, validate_weights


def fit_penalty_model(rows: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    """Fit shrinkage and percentile references on training windows only."""
    fitted = rows.copy()
    model = {"dimensions": {}}
    for dimension in DIMENSION_LABELS:
        if dimension == "imu_evidence":
            rate = pd.Series(fitted[f"{dimension}_raw_rate"], dtype="float64")
            reference = sorted(float(value) for value in rate.dropna().clip(lower=0))
            fitted[f"{dimension}_rate"] = rate
            fitted[f"{dimension}_penalty"] = monotone_percentile_penalty(
                rate, available=rate.notna(), cap=DIMENSION_PENALTY_CAPS[dimension], reference_rate=reference
            ).to_numpy()
            model["dimensions"][dimension] = {
                "cap": DIMENSION_PENALTY_CAPS[dimension],
                "reference_rate": reference,
            }
            continue
        numerator = pd.to_numeric(fitted[f"{dimension}_decayed_severity_count"], errors="coerce").fillna(0.0)
        exposure = pd.to_numeric(fitted[f"{dimension}_exposure"], errors="coerce")
        valid = exposure.notna() & (exposure > 0)
        total_exposure = float(exposure[valid].sum())
        prior_mean = float(numerator[valid].sum() / total_exposure) if total_exposure > 0 else 0.0
        prior_exposure = float(exposure[valid].median()) if valid.any() else 1.0
        prior_exposure = max(prior_exposure, 1e-9)
        clean_exposure = exposure.fillna(0.0).clip(lower=0.0)
        rate = (numerator + prior_exposure * prior_mean) / (clean_exposure + prior_exposure)
        reference = sorted(float(value) for value in rate.dropna().clip(lower=0))
        fitted[f"{dimension}_rate"] = rate
        fitted[f"{dimension}_penalty"] = monotone_percentile_penalty(
            rate, cap=DIMENSION_PENALTY_CAPS[dimension], reference_rate=reference
        ).to_numpy()
        model["dimensions"][dimension] = {
            "prior_mean": prior_mean,
            "prior_exposure": prior_exposure,
            "cap": DIMENSION_PENALTY_CAPS[dimension],
            "reference_rate": reference,
        }
    return model, fitted


def apply_penalty_model(rows: pd.DataFrame, model: dict) -> pd.DataFrame:
    """Apply training-only shrinkage and percentile references to later data."""
    transformed = rows.copy()
    for dimension in DIMENSION_LABELS:
        parameters = model["dimensions"][dimension]
        if dimension == "imu_evidence":
            rate = pd.Series(transformed[f"{dimension}_raw_rate"], dtype="float64")
            transformed[f"{dimension}_rate"] = rate
            transformed[f"{dimension}_penalty"] = monotone_percentile_penalty(
                rate,
                available=rate.notna(),
                cap=float(parameters["cap"]),
                reference_rate=parameters["reference_rate"],
            ).to_numpy()
            continue
        numerator = pd.to_numeric(
            transformed[f"{dimension}_decayed_severity_count"], errors="coerce"
        ).fillna(0.0)
        exposure = pd.to_numeric(transformed[f"{dimension}_exposure"], errors="coerce")
        clean_exposure = exposure.fillna(0.0).clip(lower=0.0)
        prior_mean = float(parameters["prior_mean"])
        prior_exposure = float(parameters["prior_exposure"])
        rate = (numerator + prior_exposure * prior_mean) / (clean_exposure + prior_exposure)
        transformed[f"{dimension}_rate"] = rate
        transformed[f"{dimension}_penalty"] = monotone_percentile_penalty(
            rate,
            cap=float(parameters["cap"]),
            reference_rate=parameters["reference_rate"],
        ).to_numpy()
    return transformed


def binary_auc(y_true, score) -> float | None:
    """AUC from average ranks; return None when only one class exists."""
    frame = pd.DataFrame({"label": y_true, "score": score}).dropna()
    if frame.empty or frame["label"].nunique() < 2:
        return None
    labels = frame["label"].astype(int)
    if not labels.isin([0, 1]).all():
        raise ValueError("future labels must be 0/1")
    positives = int(labels.sum())
    negatives = len(labels) - positives
    ranks = frame["score"].rank(method="average")
    return float((ranks[labels == 1].sum() - positives * (positives + 1) / 2) / (positives * negatives))


def _signals_by_window(penalties, labels, window_id):
    dimensions = list(DEFAULT_WEIGHTS)
    windows = pd.Series("all", index=penalties.index) if window_id is None else pd.Series(window_id, index=penalties.index)
    aucs = {dimension: [] for dimension in dimensions}
    for _, indices in windows.groupby(windows).groups.items():
        for dimension in dimensions:
            value = binary_auc(labels.loc[indices], penalties.loc[indices, f"{dimension}_penalty"].fillna(0.0))
            if value is not None:
                aucs[dimension].append(value)
    return aucs


def _regularized_coefficients(penalties: pd.DataFrame, labels: pd.Series) -> dict[str, float]:
    try:
        from sklearn.linear_model import LogisticRegression
    except ImportError as exc:  # pragma: no cover - only reached for label calibration
        raise RuntimeError("权重校准需要scikit-learn，请安装 src/scoring/requirements.txt") from exc
    dimensions = list(DEFAULT_WEIGHTS)
    columns = [f"{dimension}_penalty" for dimension in dimensions]
    usable = labels.notna()
    x = penalties.loc[usable, columns].fillna(0.0).to_numpy(dtype=float) / 100.0
    y = labels.loc[usable].astype(int).to_numpy()
    model = LogisticRegression(
        penalty="l2",
        C=0.1,
        class_weight="balanced",
        solver="liblinear",
        random_state=20260923,
    )
    model.fit(x, y)
    return {dimension: float(coefficient) for dimension, coefficient in zip(dimensions, model.coef_[0])}


def calibrate_weights(
    penalties: pd.DataFrame | None,
    future_label: pd.Series | None,
    window_id: pd.Series | None = None,
    max_change: float = 0.05,
    blend: float = 0.25,
) -> tuple[dict[str, float], dict]:
    """Return constrained weights and an auditable calibration report."""
    initial = DEFAULT_WEIGHTS.copy()
    if future_label is None or penalties is None:
        return initial, {
            "status": "initial_weights_missing_future_labels",
            "message": "未发现A成员的可用未来标签；保留仓库初始权重。",
        }
    labels = pd.Series(future_label).reset_index(drop=True)
    penalty_frame = penalties.reset_index(drop=True)
    if labels.notna().sum() < 50 or labels.dropna().nunique() < 2:
        return initial, {
            "status": "initial_weights_insufficient_future_labels",
            "labeled_rows": int(labels.notna().sum()),
        }
    expected = {f"{dimension}_penalty" for dimension in initial}
    missing = expected - set(penalty_frame)
    if missing:
        raise ValueError(f"calibration penalties missing columns: {sorted(missing)}")
    windows = None if window_id is None else pd.Series(window_id).reset_index(drop=True)
    aucs = _signals_by_window(penalty_frame, labels, windows)
    coefficients = _regularized_coefficients(penalty_frame, labels)
    direction_stability = {
        dimension: {
            "direction": (
                "positive" if values and all(value >= 0.5 for value in values)
                else "negative" if values and all(value <= 0.5 for value in values)
                else "mixed"
            ),
            "stable": bool(values) and (
                all(value >= 0.5 for value in values) or all(value <= 0.5 for value in values)
            ),
        }
        for dimension, values in aucs.items()
    }
    signals = {
        dimension: max(0.0, coefficients[dimension])
        * max(0.0, float(np.median(values)) - 0.5)
        if values else 0.0
        for dimension, values in aucs.items()
    }
    signal_total = sum(signals.values())
    if signal_total <= 0:
        return initial, {
            "status": "initial_weights_no_stable_positive_signal",
            "window_auc": aucs,
            "logistic_coefficients": coefficients,
            "direction_stability": direction_stability,
        }
    target = {dimension: signals[dimension] / signal_total for dimension in initial}
    target["imu_evidence"] = min(target["imu_evidence"], 0.10)
    non_imu = [dimension for dimension in initial if dimension != "imu_evidence"]
    non_imu_total = sum(target[dimension] for dimension in non_imu)
    if non_imu_total > 0:
        scale = (1.0 - target["imu_evidence"]) / non_imu_total
        for dimension in non_imu:
            target[dimension] *= scale
    largest_delta = max(abs(target[dimension] - initial[dimension]) for dimension in initial)
    alpha = min(blend, max_change / largest_delta) if largest_delta > 0 else 0.0
    candidate = {
        dimension: initial[dimension] + alpha * (target[dimension] - initial[dimension])
        for dimension in initial
    }
    try:
        validate_weights(candidate)
    except ValueError:
        status = "initial_weights_constraint_fallback"
        candidate = initial
    else:
        status = "calibrated_with_future_labels"
    return candidate, {
        "status": status,
        "window_auc": aucs,
        "logistic_coefficients": coefficients,
        "direction_stability": direction_stability,
        "positive_calibration_dimensions": [
            dimension for dimension, signal in signals.items() if signal > 0
        ],
        "signals": signals,
        "max_change": max_change,
        "blend": blend,
        "applied_blend": alpha,
    }
