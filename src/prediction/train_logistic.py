"""Regularised logistic-regression baseline."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from .evaluate import aggregate_fold_metrics, binary_metrics, metric_sort_key
from .preprocessing import build_preprocessor
from .splitters import TimeFold


DEFAULT_C_VALUES = (0.03, 0.1, 0.3, 1.0, 3.0)


def cross_validate_logistic(
    frame: pd.DataFrame,
    feature_columns: list[str],
    folds: list[TimeFold],
    *,
    c_values=DEFAULT_C_VALUES,
    random_state: int = 2026,
) -> dict:
    candidates: list[dict] = []
    X = frame[feature_columns]
    y = frame["label"].to_numpy()
    for c_value in c_values:
        oof = np.full(len(frame), np.nan, dtype=float)
        records: list[dict] = []
        for fold in folds:
            pipeline = Pipeline(
                [
                    (
                        "preprocess",
                        build_preprocessor(
                            frame.iloc[fold.train_index],
                            feature_columns,
                            scale_numeric=True,
                        ),
                    ),
                    (
                        "model",
                        LogisticRegression(
                            C=float(c_value),
                            class_weight="balanced",
                            max_iter=3000,
                            solver="liblinear",
                            random_state=random_state,
                        ),
                    ),
                ]
            )
            pipeline.fit(X.iloc[fold.train_index], y[fold.train_index])
            probability = pipeline.predict_proba(X.iloc[fold.valid_index])[:, 1]
            oof[fold.valid_index] = probability
            record = binary_metrics(y[fold.valid_index], probability)
            record.update(
                {
                    "model_name": "logistic",
                    "parameter": f"C={c_value}",
                    "fold": fold.name,
                    "window_id": fold.valid_window,
                    "valid_feature_end": fold.valid_feature_end,
                    "valid_label_start": fold.valid_label_start,
                    "train_positive": int(y[fold.train_index].sum()),
                    "train_negative": int(len(fold.train_index) - y[fold.train_index].sum()),
                }
            )
            records.append(record)
        summary = aggregate_fold_metrics(records)
        candidates.append(
            {
                "model_name": "logistic",
                "params": {"C": float(c_value)},
                "records": records,
                "summary": summary,
                "oof": oof,
            }
        )
    return max(candidates, key=lambda item: metric_sort_key(item["summary"]))


def fit_final_logistic(
    frame: pd.DataFrame,
    feature_columns: list[str],
    params: dict,
    *,
    random_state: int = 2026,
) -> Pipeline:
    pipeline = Pipeline(
        [
            (
                "preprocess",
                build_preprocessor(frame, feature_columns, scale_numeric=True),
            ),
            (
                "model",
                LogisticRegression(
                    C=float(params["C"]),
                    class_weight="balanced",
                    max_iter=3000,
                    solver="liblinear",
                    random_state=random_state,
                ),
            ),
        ]
    )
    pipeline.fit(frame[feature_columns], frame["label"])
    return pipeline


def logistic_importance(pipeline: Pipeline) -> pd.DataFrame:
    names = pipeline.named_steps["preprocess"].get_feature_names_out()
    coefficients = pipeline.named_steps["model"].coef_[0]
    result = pd.DataFrame(
        {
            "model": "logistic",
            "feature": names,
            "importance": np.abs(coefficients),
            "signed_effect": coefficients,
        }
    )
    return result.sort_values("importance", ascending=False).reset_index(drop=True)
