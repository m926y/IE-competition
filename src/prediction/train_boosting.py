"""Shallow LightGBM model selected by forward time validation."""

from __future__ import annotations

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, early_stopping
from sklearn.pipeline import Pipeline

from .evaluate import aggregate_fold_metrics, binary_metrics, metric_sort_key
from .preprocessing import build_preprocessor
from .splitters import TimeFold


DEFAULT_PARAMETER_GRID = (
    {
        "max_depth": 2,
        "num_leaves": 7,
        "learning_rate": 0.03,
        "min_child_samples": 40,
        "subsample": 0.8,
        "colsample_bytree": 0.7,
        "reg_alpha": 0.5,
        "reg_lambda": 5.0,
    },
    {
        "max_depth": 3,
        "num_leaves": 12,
        "learning_rate": 0.025,
        "min_child_samples": 30,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 1.0,
        "reg_lambda": 8.0,
    },
    {
        "max_depth": 4,
        "num_leaves": 20,
        "learning_rate": 0.02,
        "min_child_samples": 20,
        "subsample": 0.7,
        "colsample_bytree": 0.7,
        "reg_alpha": 2.0,
        "reg_lambda": 12.0,
    },
)


def _model(params: dict, random_state: int, n_estimators: int = 1200) -> LGBMClassifier:
    return LGBMClassifier(
        objective="binary",
        n_estimators=n_estimators,
        class_weight="balanced",
        random_state=random_state,
        n_jobs=-1,
        deterministic=True,
        force_col_wise=True,
        subsample_freq=1,
        verbosity=-1,
        **params,
    )


def cross_validate_boosting(
    frame: pd.DataFrame,
    feature_columns: list[str],
    folds: list[TimeFold],
    *,
    parameter_grid=DEFAULT_PARAMETER_GRID,
    random_state: int = 2026,
) -> dict:
    candidates: list[dict] = []
    X = frame[feature_columns]
    y = frame["label"].to_numpy()
    for params in parameter_grid:
        oof = np.full(len(frame), np.nan, dtype=float)
        records: list[dict] = []
        best_iterations: list[int] = []
        for fold in folds:
            preprocessor = build_preprocessor(
                frame.iloc[fold.train_index], feature_columns, scale_numeric=False
            )
            X_train = preprocessor.fit_transform(X.iloc[fold.train_index])
            X_valid = preprocessor.transform(X.iloc[fold.valid_index])
            model = _model(params, random_state)
            model.fit(
                X_train,
                y[fold.train_index],
                eval_X=X_valid,
                eval_y=y[fold.valid_index],
                eval_metric="auc",
                callbacks=[early_stopping(60, verbose=False)],
            )
            probability = model.predict_proba(X_valid)[:, 1]
            oof[fold.valid_index] = probability
            best_iterations.append(int(model.best_iteration_ or model.n_estimators))
            record = binary_metrics(y[fold.valid_index], probability)
            record.update(
                {
                    "model_name": "lightgbm",
                    "parameter": str(params),
                    "fold": fold.name,
                    "window_id": fold.valid_window,
                    "valid_feature_end": fold.valid_feature_end,
                    "valid_label_start": fold.valid_label_start,
                    "train_positive": int(y[fold.train_index].sum()),
                    "train_negative": int(len(fold.train_index) - y[fold.train_index].sum()),
                    "best_iteration": best_iterations[-1],
                }
            )
            records.append(record)
        summary = aggregate_fold_metrics(records)
        summary["median_best_iteration"] = float(np.median(best_iterations))
        candidates.append(
            {
                "model_name": "lightgbm",
                "params": dict(params),
                "records": records,
                "summary": summary,
                "oof": oof,
                "best_iterations": best_iterations,
            }
        )
    return max(candidates, key=lambda item: metric_sort_key(item["summary"]))


def fit_final_boosting(
    frame: pd.DataFrame,
    feature_columns: list[str],
    params: dict,
    *,
    n_estimators: int,
    random_state: int = 2026,
) -> Pipeline:
    pipeline = Pipeline(
        [
            (
                "preprocess",
                build_preprocessor(frame, feature_columns, scale_numeric=False),
            ),
            ("model", _model(params, random_state, n_estimators=n_estimators)),
        ]
    )
    pipeline.fit(frame[feature_columns], frame["label"])
    return pipeline


def boosting_importance(pipeline: Pipeline) -> pd.DataFrame:
    names = pipeline.named_steps["preprocess"].get_feature_names_out()
    importance = pipeline.named_steps["model"].feature_importances_
    result = pd.DataFrame(
        {
            "model": "lightgbm",
            "feature": names,
            "importance": importance,
            "signed_effect": np.nan,
        }
    )
    return result.sort_values("importance", ascending=False).reset_index(drop=True)
