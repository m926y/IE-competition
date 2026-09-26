from __future__ import annotations

import argparse
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.prediction.data_loader import load_training_data, select_feature_columns
from src.prediction.splitters import make_time_folds
from src.prediction.validate_submission import validate_submission


def synthetic_data(vehicle_count: int = 60) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(2026)
    windows = [
        ("w1", "2026-06-01", "2026-06-07", "2026-06-08", "2026-06-14"),
        ("w2", "2026-06-15", "2026-06-21", "2026-06-22", "2026-06-28"),
        ("w3", "2026-06-29", "2026-07-05", "2026-07-06", "2026-07-12"),
        ("w4", "2026-07-13", "2026-07-19", "2026-07-20", "2026-07-26"),
    ]
    rows: list[dict] = []
    vehicle_effect = rng.normal(size=vehicle_count)
    for window_number, dates in enumerate(windows):
        window_id, feature_start, feature_end, label_start, label_end = dates
        risk = vehicle_effect + rng.normal(scale=0.6, size=vehicle_count) + 0.1 * window_number
        cutoff = float(np.median(risk))
        for index in range(vehicle_count):
            rows.append(
                {
                    "gpsno": f"V{index:04d}",
                    "window_id": window_id,
                    "feature_start": feature_start,
                    "feature_end": feature_end,
                    "label_start": label_start,
                    "label_end": label_end,
                    "label": int(risk[index] > cutoff),
                    "energy_type": "electric" if index % 3 == 0 else "diesel",
                    "monthly_mileage_km": 5000 + 900 * vehicle_effect[index],
                    "risk_event_rate_14d": max(0.0, risk[index] + 2.0),
                    "trajectory_speed_std": abs(rng.normal(8 + risk[index], 1.0)),
                    "imu_gyro_std": abs(rng.normal(2 + risk[index] * 0.2, 0.3)),
                }
            )
    train = pd.DataFrame(rows)
    full = (
        train.loc[train["window_id"].eq("w4")]
        .drop(columns=["window_id", "feature_start", "feature_end", "label_start", "label_end", "label"])
        .reset_index(drop=True)
    )
    return train, full


class PredictionPipelineTests(unittest.TestCase):
    def test_time_folds_are_forward_only(self):
        train, _ = synthetic_data(20)
        for column in ("feature_start", "feature_end", "label_start", "label_end"):
            train[column] = pd.to_datetime(train[column])
        folds = make_time_folds(train)
        self.assertEqual(len(folds), 3)
        for fold in folds:
            train_label_end = train.iloc[fold.train_index]["label_end"].max()
            valid_label_start = train.iloc[fold.valid_index]["label_start"].min()
            self.assertLess(train_label_end, valid_label_start)

    def test_loader_rejects_future_feature(self):
        train, _ = synthetic_data(20)
        train["future_accident"] = 0
        with self.assertRaisesRegex(ValueError, "未来信息"):
            select_feature_columns(train)


if __name__ == "__main__":
    unittest.main()
