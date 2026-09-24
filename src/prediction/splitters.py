"""Forward-only validation splits for rolling vehicle windows."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class TimeFold:
    name: str
    train_index: np.ndarray
    valid_index: np.ndarray
    valid_window: str
    valid_feature_end: str
    valid_label_start: str


def make_time_folds(
    frame: pd.DataFrame,
    *,
    min_train_windows: int = 1,
    min_train_positives: int = 1,
    min_train_negatives: int = 1,
    min_valid_positives: int = 1,
    min_valid_negatives: int = 1,
) -> list[TimeFold]:
    """Create expanding-window folds without random event or row splitting.

    A training window must end its label period before the validation label
    period begins. Its feature cutoff must also be earlier than the validation
    cutoff. Folds lacking both classes are skipped because AUC is undefined.
    """
    summaries = (
        frame.groupby("window_id", as_index=False)
        .agg(
            feature_end=("feature_end", "max"),
            label_start=("label_start", "min"),
            label_end=("label_end", "max"),
        )
        .sort_values(["feature_end", "label_start", "window_id"])
        .reset_index(drop=True)
    )
    folds: list[TimeFold] = []
    for _, valid in summaries.iterrows():
        train_windows = summaries.loc[
            (summaries["feature_end"] < valid["feature_end"])
            & (summaries["label_end"] < valid["label_start"]),
            "window_id",
        ]
        if len(train_windows) < min_train_windows:
            continue
        train_mask = frame["window_id"].isin(train_windows)
        valid_mask = frame["window_id"].eq(valid["window_id"])
        y_train = frame.loc[train_mask, "label"]
        y_valid = frame.loc[valid_mask, "label"]
        train_positives = int(y_train.sum())
        train_negatives = int(len(y_train) - train_positives)
        positives = int(y_valid.sum())
        negatives = int(len(y_valid) - positives)
        if (
            train_positives < min_train_positives
            or train_negatives < min_train_negatives
            or positives < min_valid_positives
            or negatives < min_valid_negatives
        ):
            continue
        folds.append(
            TimeFold(
                name=f"fold_{len(folds) + 1}",
                train_index=np.flatnonzero(train_mask.to_numpy()),
                valid_index=np.flatnonzero(valid_mask.to_numpy()),
                valid_window=str(valid["window_id"]),
                valid_feature_end=pd.Timestamp(valid["feature_end"]).date().isoformat(),
                valid_label_start=pd.Timestamp(valid["label_start"]).date().isoformat(),
            )
        )
    if not folds:
        raise ValueError(
            "无法构造有效时间验证折：请确保至少两个按时间先后排列的窗口，"
            "且验证窗口同时包含正负样本"
        )
    return folds
