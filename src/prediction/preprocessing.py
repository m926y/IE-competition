"""Leakage-safe preprocessing fitted independently inside each time fold."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


class QuantileClipper(BaseEstimator, TransformerMixin):
    """Clip numeric values using quantiles learned only from training rows."""

    def __init__(self, lower: float = 0.005, upper: float = 0.995):
        self.lower = lower
        self.upper = upper

    def fit(self, X, y=None):
        values = np.asarray(X, dtype=float)
        self.lower_bounds_ = np.nanquantile(values, self.lower, axis=0)
        self.upper_bounds_ = np.nanquantile(values, self.upper, axis=0)
        return self

    def transform(self, X):
        values = np.asarray(X, dtype=float)
        return np.clip(values, self.lower_bounds_, self.upper_bounds_)

    def get_feature_names_out(self, input_features=None):
        return np.asarray(input_features, dtype=object)


def infer_column_types(
    frame: pd.DataFrame, feature_columns: list[str]
) -> tuple[list[str], list[str]]:
    numeric = [
        column for column in feature_columns if pd.api.types.is_numeric_dtype(frame[column])
    ]
    categorical = [column for column in feature_columns if column not in numeric]
    return numeric, categorical


def build_preprocessor(
    frame: pd.DataFrame,
    feature_columns: list[str],
    *,
    scale_numeric: bool,
) -> ColumnTransformer:
    numeric, categorical = infer_column_types(frame, feature_columns)
    numeric_steps: list[tuple[str, object]] = [
        ("clip", QuantileClipper()),
        ("impute", SimpleImputer(strategy="median", add_indicator=True)),
    ]
    if scale_numeric:
        numeric_steps.append(("scale", StandardScaler()))
    transformers: list[tuple[str, object, list[str]]] = []
    if numeric:
        transformers.append(("numeric", Pipeline(numeric_steps), numeric))
    if categorical:
        categorical_pipeline = Pipeline(
            [
                ("impute", SimpleImputer(strategy="constant", fill_value="UNKNOWN")),
                (
                    "onehot",
                    OneHotEncoder(handle_unknown="ignore", min_frequency=2),
                ),
            ]
        )
        transformers.append(("categorical", categorical_pipeline, categorical))
    return ColumnTransformer(transformers, remainder="drop", verbose_feature_names_out=True)
