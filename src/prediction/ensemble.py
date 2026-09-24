"""Stable rank blending and out-of-fold probability calibration."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from .evaluate import aggregate_fold_metrics, binary_metrics, metric_sort_key
from .splitters import TimeFold


BLEND_WEIGHTS = (0.25, 0.50, 0.75)


def percentile_rank(values: np.ndarray) -> np.ndarray:
    return pd.Series(values).rank(method="average", pct=True).to_numpy(dtype=float)


def _fold_records(y, score, folds: list[TimeFold], model_name: str) -> list[dict]:
    records: list[dict] = []
    for fold in folds:
        mask = fold.valid_index
        record = binary_metrics(y[mask], score[mask])
        record.update(
            {
                "model_name": model_name,
                "fold": fold.name,
                "window_id": fold.valid_window,
                "valid_feature_end": fold.valid_feature_end,
                "valid_label_start": fold.valid_label_start,
            }
        )
        records.append(record)
    return records


def select_prediction_strategy(
    y,
    logistic_oof: np.ndarray,
    boosting_oof: np.ndarray,
    folds: list[TimeFold],
) -> dict:
    y = np.asarray(y, dtype=int)
    logistic_rank = np.full(len(y), np.nan)
    boosting_rank = np.full(len(y), np.nan)
    for fold in folds:
        mask = fold.valid_index
        logistic_rank[mask] = percentile_rank(logistic_oof[mask])
        boosting_rank[mask] = percentile_rank(boosting_oof[mask])

    candidates: list[dict] = []
    for name, score in (
        ("logistic", logistic_oof),
        ("lightgbm", boosting_oof),
    ):
        records = _fold_records(y, score, folds, name)
        candidates.append(
            {"strategy": name, "weight": None, "score": score, "records": records,
             "summary": aggregate_fold_metrics(records)}
        )
    best_single = max(candidates, key=lambda item: metric_sort_key(item["summary"]))

    eligible_blends: list[dict] = []
    if len(folds) >= 2:
        single_auc = np.asarray([r["roc_auc"] for r in best_single["records"]])
        for weight in BLEND_WEIGHTS:
            score = weight * boosting_rank + (1 - weight) * logistic_rank
            records = _fold_records(y, score, folds, f"rank_blend_{weight:.2f}")
            summary = aggregate_fold_metrics(records)
            blend_auc = np.asarray([r["roc_auc"] for r in records])
            no_worse_count = int(np.sum(blend_auc >= single_auc - 1e-6))
            if (
                no_worse_count >= 2
                and summary["min_roc_auc"] >= best_single["summary"]["min_roc_auc"] - 1e-6
            ):
                eligible_blends.append(
                    {
                        "strategy": "rank_blend",
                        "weight": float(weight),
                        "score": score,
                        "records": records,
                        "summary": summary,
                    }
                )
    if eligible_blends:
        return max(eligible_blends, key=lambda item: metric_sort_key(item["summary"]))
    return best_single


def fit_platt_calibrator(y, score) -> LogisticRegression | None:
    y = np.asarray(y, dtype=int)
    score = np.asarray(score, dtype=float)
    mask = np.isfinite(score)
    if len(np.unique(y[mask])) < 2:
        return None
    model = LogisticRegression(C=1e6, solver="lbfgs", max_iter=2000)
    model.fit(score[mask].reshape(-1, 1), y[mask])
    return model


def calibrate_probability(calibrator, score) -> np.ndarray:
    score = np.asarray(score, dtype=float)
    if calibrator is None:
        return np.clip(score, 0.0, 1.0)
    return calibrator.predict_proba(score.reshape(-1, 1))[:, 1]


def combine_final_predictions(
    strategy: dict,
    logistic_probability: np.ndarray,
    boosting_probability: np.ndarray,
) -> np.ndarray:
    if strategy["strategy"] == "logistic":
        return logistic_probability
    if strategy["strategy"] == "lightgbm":
        return boosting_probability
    weight = float(strategy["weight"])
    return (
        weight * percentile_rank(boosting_probability)
        + (1 - weight) * percentile_rank(logistic_probability)
    )
