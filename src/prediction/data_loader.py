"""Input contracts for the accident-prediction pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


ID_COLUMN = "gpsno"
LABEL_COLUMN = "label"
WINDOW_COLUMN = "window_id"
DATE_COLUMNS = ("feature_start", "feature_end", "label_start", "label_end")
METADATA_COLUMNS = {ID_COLUMN, LABEL_COLUMN, WINDOW_COLUMN, *DATE_COLUMNS}
EXACT_LEAKAGE_COLUMNS = {
    "target",
    "future_target",
    "future_accident",
    "future_near_miss",
    "label_accident",
    "label_near_miss",
}


def read_table(path: str | Path) -> pd.DataFrame:
    """Read a supported tabular file without silently guessing formats."""
    table_path = Path(path)
    if not table_path.exists():
        raise FileNotFoundError(f"输入文件不存在：{table_path}")
    suffix = table_path.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(table_path)
    if suffix == ".csv":
        return pd.read_csv(table_path, dtype={ID_COLUMN: "string"})
    raise ValueError(f"不支持的文件格式：{table_path}；仅支持 Parquet 或 CSV")


def _normalise_id(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame[ID_COLUMN] = frame[ID_COLUMN].astype("string").str.strip()
    if frame[ID_COLUMN].isna().any() or frame[ID_COLUMN].eq("").any():
        raise ValueError("gpsno 存在空值")
    return frame


def load_training_data(path: str | Path) -> pd.DataFrame:
    """Load and validate A's rolling-window training features."""
    frame = read_table(path)
    required = {ID_COLUMN, LABEL_COLUMN, WINDOW_COLUMN, *DATE_COLUMNS}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"训练特征缺少字段：{missing}")
    frame = _normalise_id(frame)
    for column in DATE_COLUMNS:
        frame[column] = pd.to_datetime(frame[column], errors="coerce")
        if frame[column].isna().any():
            raise ValueError(f"{column} 含无法解析的日期")
    labels = pd.to_numeric(frame[LABEL_COLUMN], errors="coerce")
    if labels.isna().any() or not set(labels.unique()).issubset({0, 1}):
        raise ValueError("label 必须只包含 0 和 1")
    frame[LABEL_COLUMN] = labels.astype("int8")
    if frame.duplicated([ID_COLUMN, WINDOW_COLUMN]).any():
        raise ValueError("训练数据中 gpsno + window_id 不唯一")
    if (frame["feature_start"] > frame["feature_end"]).any():
        raise ValueError("存在 feature_start 晚于 feature_end 的窗口")
    if (frame["label_start"] > frame["label_end"]).any():
        raise ValueError("存在 label_start 晚于 label_end 的窗口")
    if (frame["feature_end"] >= frame["label_start"]).any():
        raise ValueError("特征期与标签期重叠，存在时间泄漏")
    return frame


def load_prediction_data(path: str | Path) -> pd.DataFrame:
    """Load and validate the 61-day vehicle-level prediction features."""
    frame = read_table(path)
    if ID_COLUMN not in frame.columns:
        raise ValueError("最终特征缺少 gpsno")
    frame = _normalise_id(frame)
    if frame[ID_COLUMN].duplicated().any():
        raise ValueError("最终特征中 gpsno 不唯一")
    return frame


def select_feature_columns(frame: pd.DataFrame) -> list[str]:
    """Return model features while rejecting explicit future-label fields."""
    lowered = {column: column.lower().strip() for column in frame.columns}
    leakage = [
        column
        for column, name in lowered.items()
        if name in EXACT_LEAKAGE_COLUMNS or name.startswith("future_")
    ]
    if leakage:
        raise ValueError(f"发现疑似未来信息字段：{sorted(leakage)}")
    features = [column for column in frame.columns if column not in METADATA_COLUMNS]
    if not features:
        raise ValueError("没有可用的模型特征")
    all_missing = [column for column in features if frame[column].isna().all()]
    if all_missing:
        raise ValueError(f"存在全空特征：{all_missing}")
    constant = [column for column in features if frame[column].nunique(dropna=False) <= 1]
    return [column for column in features if column not in constant]


def align_prediction_features(
    frame: pd.DataFrame, feature_columns: Iterable[str]
) -> pd.DataFrame:
    """Align final features to the exact training feature order."""
    result = frame.copy()
    columns = list(feature_columns)
    for column in columns:
        if column not in result.columns:
            result[column] = np.nan
    return result[[ID_COLUMN, *columns]]


def load_feature_dictionary(path: str | Path | None) -> pd.DataFrame | None:
    if path is None:
        return None
    dictionary_path = Path(path)
    if not dictionary_path.exists():
        return None
    frame = pd.read_csv(dictionary_path)
    required = {"feature_name", "feature_group"}
    if not required.issubset(frame.columns):
        raise ValueError("特征字典至少需要 feature_name 和 feature_group")
    return frame
