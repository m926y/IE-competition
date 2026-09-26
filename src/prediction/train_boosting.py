"""Frozen shallow formal-target LightGBM with training-only preprocessing."""
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.pipeline import Pipeline
from .preprocessing import build_preprocessor

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
