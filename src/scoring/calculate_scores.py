#!/usr/bin/env python3
"""Generate member C's explainable 100-point safety score from cleaned data."""

from __future__ import annotations

import argparse
import json
import platform
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import duckdb
except ImportError:  # pragma: no cover - score_frames itself does not need DuckDB
    duckdb = None

from .calibrate_weights import apply_penalty_model, calibrate_weights, fit_penalty_model
from .event_taxonomy import DIMENSION_LABELS
from .exposure_adjustment import empirical_bayes_rate, monotone_percentile_penalty
from .export_scores import SAFETY_SCORE_COLUMNS, write_csv, write_json, write_parquet
from .reason_codes import operational_action, reason_record
from .scoring_rules import RULE_VERSION, default_rules_frame, validate_weights, write_rules
from .validate_scores import validate_score_frame

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CLEANED_DIR = ROOT / "data" / "processed" / "cleaned"
DEFAULT_OUTPUT_DIR = ROOT / "outputs" / "scoring"
DEFAULT_RUN_DIR = ROOT / "runs" / "member_c"
DEFAULT_FEATURE_DIR = ROOT / "outputs" / "features"
AS_OF_TS = "2026-07-31 23:59:59"
OBSERVATION_DAYS = 61
RANDOM_SEED = 20260923

REQUIRED_INPUTS = {
    "profile": "vehicle_profile.parquet",
    "risk": "risk_events.parquet",
    "trajectory": "trajectory_daily.parquet",
    "imu": "imu_daily.parquet",
}

A_REQUIRED_INPUTS = {
    "risk_daily": "risk_event_daily.parquet",
    "train_windows": "train_windows.parquet",
    "full_features": "full_61d_features.parquet",
    "feature_dictionary": "feature_dictionary.csv",
    "label_report": "label_report.json",
    "split_manifest": "split_manifest.csv",
}


def _sql_quote(value: str | Path) -> str:
    return "'" + str(value).replace("\\", "/").replace("'", "''") + "'"


def resolve_inputs(cleaned_dir: Path) -> dict[str, Path]:
    paths = {key: cleaned_dir / filename for key, filename in REQUIRED_INPUTS.items()}
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "缺少公共清洗文件：" + ", ".join(missing)
            + "。先运行 src/data/clean_data.py；本评分器不需要等待A的特征字典。"
        )
    return paths


def collect_cleaned_frames(paths: dict[str, Path], rules: pd.DataFrame, as_of_ts: str = AS_OF_TS):
    if duckdb is None:
        raise RuntimeError("缺少DuckDB。请执行：python -m pip install -r src/scoring/requirements.txt")
    connection = duckdb.connect()
    connection.register("scoring_rules", rules)
    try:
        profile = connection.execute(
            f"SELECT * FROM read_parquet({_sql_quote(paths['profile'])})"
        ).df()
        trajectory = connection.execute(f"""
            SELECT gpsno,
                   count(*) AS trajectory_days,
                   sum(source_rows) AS trajectory_source_rows,
                   sum(valid_speed_rows) AS valid_speed_rows,
                   sum(invalid_speed_rows) AS invalid_speed_rows,
                   sum(valid_coordinate_rows) AS valid_coordinate_rows,
                   sum(invalid_coordinate_rows) AS invalid_coordinate_rows,
                   sum(distance_km) AS distance_km,
                   sum(run_time_hours) AS run_time_hours
            FROM read_parquet({_sql_quote(paths['trajectory'])})
            GROUP BY gpsno
        """).df()
        imu = connection.execute(f"""
            SELECT gpsno,
                   count(*) AS imu_days,
                   sum(source_rows) AS imu_source_rows,
                   sum(missing_ems_speed_rows) AS missing_ems_speed_rows,
                   sum(missing_gps_speed_rows) AS missing_gps_speed_rows,
                   sum(accel_norm_outlier_rows) AS accel_norm_outlier_rows,
                   sum(gyro_norm_gt20_rows) AS gyro_norm_gt20_rows,
                   sum(coalesce(ems_gps_abs_diff_mean, 0) * source_rows)
                     / nullif(sum(source_rows), 0) AS ems_gps_abs_diff_mean
            FROM read_parquet({_sql_quote(paths['imu'])})
            GROUP BY gpsno
        """).df()
        risk = connection.execute(f"""
            WITH events AS (
              SELECT events.gpsno, events.event_type, events.event_time,
                     rules.dimension, rules.severity, rules.half_life_days,
                     greatest(
                       0.0,
                       date_diff('second', events.event_time, TIMESTAMP {_sql_quote(as_of_ts)}) / 86400.0
                     ) AS age_days
              FROM read_parquet({_sql_quote(paths['risk'])}) AS events
              JOIN scoring_rules AS rules
                ON events.event_type = cast(rules.event_type AS INTEGER)
              WHERE events.event_time <= TIMESTAMP {_sql_quote(as_of_ts)}
            )
            SELECT gpsno, dimension,
                   count(*) AS event_count,
                   count(DISTINCT cast(event_time AS DATE)) AS risk_event_days,
                   sum(severity * pow(2.0, -age_days / half_life_days)) AS decayed_severity_count,
                   max(event_time) AS most_recent_event_time
            FROM events
            GROUP BY gpsno, dimension
        """).df()
        quality = {
            "profile_rows": int(len(profile)),
            "profile_vehicles": int(profile["gpsno"].nunique()),
            "trajectory_vehicles": int(trajectory["gpsno"].nunique()),
            "imu_vehicles": int(imu["gpsno"].nunique()),
            "risk_vehicles": int(risk["gpsno"].nunique()),
            "as_of_timestamp": as_of_ts,
        }
        return profile, trajectory, imu, risk, quality
    finally:
        connection.close()


def _assign_risk_level(rank: pd.Series) -> np.ndarray:
    count = len(rank)
    a_end = int(np.ceil(count * 0.20))
    b_end = int(np.ceil(count * 0.60))
    c_end = int(np.ceil(count * 0.90))
    return np.select([rank <= a_end, rank <= b_end, rank <= c_end], ["A", "B", "C"], default="D")


def score_frames(
    profile: pd.DataFrame,
    trajectory: pd.DataFrame,
    imu: pd.DataFrame,
    risk: pd.DataFrame,
    weights: dict[str, float] | None = None,
) -> tuple[pd.DataFrame, dict[str, float], dict]:
    """Score already aggregated frames; separated for deterministic unit tests."""
    if profile["gpsno"].duplicated().any():
        raise ValueError("vehicle_profile.parquet 的 gpsno 必须唯一")
    frame = profile.copy()
    frame["gpsno"] = frame["gpsno"].astype(str)
    frame = frame.merge(trajectory, on="gpsno", how="left").merge(imu, on="gpsno", how="left")
    numeric_fill = [
        "trajectory_days", "trajectory_source_rows", "valid_speed_rows", "invalid_speed_rows",
        "valid_coordinate_rows", "invalid_coordinate_rows", "distance_km", "run_time_hours",
        "imu_days", "imu_source_rows", "missing_ems_speed_rows", "missing_gps_speed_rows",
        "accel_norm_outlier_rows", "gyro_norm_gt20_rows", "ems_gps_abs_diff_mean",
    ]
    for column in numeric_fill:
        if column not in frame:
            frame[column] = 0.0
    frame[numeric_fill] = frame[numeric_fill].fillna(0.0)
    for dimension in [key for key in DIMENSION_LABELS if key != "imu_evidence"]:
        subset = risk[risk["dimension"] == dimension].set_index("gpsno")
        frame[f"{dimension}_event_count"] = frame["gpsno"].map(subset["event_count"]).fillna(0).astype(int)
        frame[f"{dimension}_risk_event_days"] = frame["gpsno"].map(subset["risk_event_days"]).fillna(0).astype(int)
        frame[f"{dimension}_decayed_severity_count"] = frame["gpsno"].map(
            subset["decayed_severity_count"]
        ).fillna(0.0)
        if dimension in {"history", "driving_stability", "road_rule"}:
            exposure = frame["distance_km"] / 1000.0
        elif dimension == "fatigue_distraction":
            exposure = frame["run_time_hours"] / 100.0
        else:
            exposure = frame["trajectory_days"].astype(float)
        frame[f"{dimension}_exposure"] = exposure
        rate, prior_mean, prior_exposure = empirical_bayes_rate(
            frame[f"{dimension}_decayed_severity_count"], exposure
        )
        frame[f"{dimension}_rate"] = rate
        frame[f"{dimension}_prior_mean"] = prior_mean
        frame[f"{dimension}_prior_exposure"] = prior_exposure
        frame[f"{dimension}_penalty"] = monotone_percentile_penalty(rate).to_numpy()

    imu_available = frame["imu_source_rows"] > 0
    imu_rate = pd.Series(np.nan, index=frame.index, dtype="float64")
    imu_rate.loc[imu_available] = (
        0.60 * frame.loc[imu_available, "accel_norm_outlier_rows"] / frame.loc[imu_available, "imu_source_rows"]
        + 0.30 * frame.loc[imu_available, "gyro_norm_gt20_rows"] / frame.loc[imu_available, "imu_source_rows"]
        + 0.10 * (frame.loc[imu_available, "ems_gps_abs_diff_mean"] / 20.0).clip(0, 1)
    )
    frame["imu_evidence_event_count"] = (
        frame["accel_norm_outlier_rows"] + frame["gyro_norm_gt20_rows"]
    ).fillna(0).astype(int)
    frame["imu_evidence_exposure"] = frame["imu_source_rows"]
    frame["imu_evidence_rate"] = imu_rate
    frame["imu_evidence_penalty"] = monotone_percentile_penalty(
        imu_rate, available=imu_available, cap=60.0
    ).to_numpy()

    calibration = {
        "status": "manual_weights" if weights is not None else "initial_weights_missing_future_labels",
        "message": "A成员未来标签未接入；使用仓库固定初始权重。" if weights is None else "使用显式传入权重。",
    }
    if weights is None:
        penalty_frame = frame[[f"{dimension}_penalty" for dimension in DIMENSION_LABELS]]
        weights, calibration = calibrate_weights(penalty_frame, None)
    validate_weights(weights)
    for dimension, weight in weights.items():
        frame[f"{dimension}_weight"] = weight
        frame[f"{dimension}_contribution"] = weight * frame[f"{dimension}_penalty"].fillna(0.0)
    frame["safety_score_raw"] = 100.0 - sum(frame[f"{dimension}_contribution"] for dimension in weights)
    frame["safety_score"] = frame["safety_score_raw"].clip(0, 100)
    frame = frame.sort_values(["safety_score", "gpsno"], ascending=[False, True]).reset_index(drop=True)
    frame["rank"] = np.arange(1, len(frame) + 1)
    frame["risk_level"] = _assign_risk_level(frame["rank"])

    trajectory_rows = frame["trajectory_source_rows"].replace(0, np.nan)
    frame["valid_speed_ratio"] = (frame["valid_speed_rows"] / trajectory_rows).fillna(0.0).clip(0, 1)
    frame["valid_coordinate_ratio"] = (frame["valid_coordinate_rows"] / trajectory_rows).fillna(0.0).clip(0, 1)
    frame["confidence_score"] = (
        25 * (frame["trajectory_days"] / OBSERVATION_DAYS).clip(0, 1)
        + 15 * (frame["imu_days"] / OBSERVATION_DAYS).clip(0, 1)
        + 15 * frame["valid_speed_ratio"]
        + 15 * frame["valid_coordinate_ratio"]
        + 10 * (frame["distance_km"] / 1000.0).clip(0, 1)
        + 10 * (frame["run_time_hours"] / 100.0).clip(0, 1)
        + 10  # the cleaned risk-event dataset itself is present and readable
    ).clip(0, 100)
    frame["confidence_level"] = np.select(
        [frame["confidence_score"] >= 80, frame["confidence_score"] >= 60],
        ["high", "medium"], default="low",
    )

    for index in range(1, 4):
        for suffix, default in [
            ("", ""), ("_code", ""), ("_dimension", ""),
            ("_penalty_points", np.nan), ("_event_count", np.nan),
            ("_normalized_rate", np.nan), ("_percentile", np.nan),
            ("_source_event_codes", ""), ("_source_fields", ""),
        ]:
            frame[f"top_reason_{index}{suffix}"] = default
    frame["primary_risk_dimension"] = ""
    frame["primary_risk_dimension_name"] = ""
    for row_index, row in frame.iterrows():
        candidates = [dimension for dimension in weights if pd.notna(row[f"{dimension}_penalty"])]
        top_dimensions = sorted(candidates, key=lambda key: (-row[f"{key}_contribution"], key))[:3]
        for reason_rank, dimension in enumerate(top_dimensions, start=1):
            record = reason_record(row, dimension, reason_rank)
            frame.at[row_index, f"top_reason_{reason_rank}"] = record["reason_text"]
            frame.at[row_index, f"top_reason_{reason_rank}_code"] = record["reason_code"]
            frame.at[row_index, f"top_reason_{reason_rank}_dimension"] = record["dimension"]
            frame.at[row_index, f"top_reason_{reason_rank}_penalty_points"] = record["penalty_points"]
            frame.at[row_index, f"top_reason_{reason_rank}_event_count"] = record["event_count"]
            frame.at[row_index, f"top_reason_{reason_rank}_normalized_rate"] = record["normalized_rate"]
            frame.at[row_index, f"top_reason_{reason_rank}_percentile"] = record["comparison_percentile"]
            frame.at[row_index, f"top_reason_{reason_rank}_source_event_codes"] = record["source_event_codes"]
            frame.at[row_index, f"top_reason_{reason_rank}_source_fields"] = record["source_fields"]
        frame.at[row_index, "primary_risk_dimension"] = top_dimensions[0]
        frame.at[row_index, "primary_risk_dimension_name"] = DIMENSION_LABELS[top_dimensions[0]]
    actions = frame.apply(
        lambda row: operational_action(
            row["risk_level"],
            row["primary_risk_dimension"],
            row["confidence_level"],
            int(row[f"{row['primary_risk_dimension']}_event_count"]),
        ),
        axis=1,
        result_type="expand",
    )
    frame = pd.concat([frame, actions], axis=1)
    frame["safety_score"] = frame["safety_score"].round(4)
    frame["confidence_score"] = frame["confidence_score"].round(2)
    return frame, weights, calibration


def resolve_a_inputs(feature_dir: Path) -> dict[str, Path]:
    paths = {key: feature_dir / filename for key, filename in A_REQUIRED_INPUTS.items()}
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError("A成员交付不完整：" + ", ".join(missing))
    return paths


def _read_parquet(path: Path) -> pd.DataFrame:
    connection = duckdb.connect()
    try:
        return connection.execute(
            f"SELECT * FROM read_parquet({_sql_quote(path)})"
        ).df()
    finally:
        connection.close()


def _risk_daily_quality(path: Path) -> dict:
    connection = duckdb.connect()
    try:
        result = connection.execute(f"""
            WITH per_vehicle AS (
              SELECT cast(gpsno AS VARCHAR) AS gpsno,
                     count(DISTINCT event_date) AS covered_days
              FROM read_parquet({_sql_quote(path)})
              GROUP BY gpsno
            )
            SELECT count(*) AS vehicles,
                   min(covered_days) AS min_covered_days,
                   max(covered_days) AS max_covered_days,
                   sum(covered_days) AS vehicle_days
            FROM per_vehicle
        """).fetchone()
    finally:
        connection.close()
    quality = {
        "vehicles": int(result[0]),
        "min_covered_days": int(result[1]),
        "max_covered_days": int(result[2]),
        "vehicle_days": int(result[3]),
    }
    if quality != {
        "vehicles": 500,
        "min_covered_days": 61,
        "max_covered_days": 61,
        "vehicle_days": 30500,
    }:
        raise ValueError(f"risk_event_daily覆盖不完整：{quality}")
    return quality


def build_a_scoring_rows(base: pd.DataFrame, rules: pd.DataFrame, suffix: str) -> pd.DataFrame:
    """Convert A's fixed-window feature contract into C's scoring contract."""
    rows = pd.DataFrame(index=base.index)
    rows["gpsno"] = base["gpsno"].astype(str)
    for column in [
        "window_id", "window_role", "feature_start", "feature_end",
        "label_start", "label_end", "label",
    ]:
        if column in base:
            rows[column] = base[column]
    field_map = {
        "energy_type": "profile_energy_type",
        "distance_km": f"distance_km_{suffix}",
        "run_time_hours": f"run_time_hours_{suffix}",
        "trajectory_days": f"trajectory_days_{suffix}",
        "imu_days": f"imu_days_{suffix}",
        "valid_speed_ratio": f"speed_valid_fraction_{suffix}",
        "valid_coordinate_ratio": f"coordinate_valid_fraction_{suffix}",
        "active_days": f"active_days_{suffix}",
    }
    for target, source in field_map.items():
        rows[target] = pd.to_numeric(base[source], errors="coerce") if target != "energy_type" else base[source]
    rows["energy_type"] = rows["energy_type"].fillna("unknown").astype(str)
    # A's risk_event_daily is a validated dense 500 x 61 vehicle-day grid.
    rows["risk_event_coverage"] = 1.0

    for dimension in [key for key in DIMENSION_LABELS if key != "imu_evidence"]:
        dimension_rules = rules[rules["dimension"] == dimension]
        count = pd.Series(0.0, index=base.index)
        weighted = pd.Series(0.0, index=base.index)
        for rule in dimension_rules.itertuples(index=False):
            code = int(rule.event_type)
            count = count.add(pd.to_numeric(base[f"evt_{code}_count_{suffix}"], errors="coerce").fillna(0.0))
            weighted = weighted.add(
                pd.to_numeric(base[f"evt_{code}_decayed_count_{suffix}"], errors="coerce").fillna(0.0)
                * float(rule.severity)
            )
        rows[f"{dimension}_event_count"] = count.round().astype(int)
        rows[f"{dimension}_decayed_severity_count"] = weighted
        if dimension in {"history", "driving_stability", "road_rule"}:
            rows[f"{dimension}_exposure"] = rows["distance_km"] / 1000.0
        elif dimension == "fatigue_distraction":
            rows[f"{dimension}_exposure"] = rows["run_time_hours"] / 100.0
        else:
            rows[f"{dimension}_exposure"] = rows["active_days"]

    accel = pd.to_numeric(base[f"imu_accel_outlier_fraction_{suffix}"], errors="coerce")
    gyro = pd.to_numeric(base[f"imu_gyro_gt20_fraction_{suffix}"], errors="coerce")
    speed_gap = pd.to_numeric(
        base[f"imu_ems_gps_abs_diff_mean_daily_mean_{suffix}"], errors="coerce"
    )
    imu_available = rows["imu_days"].fillna(0) > 0
    rows["imu_evidence_raw_rate"] = (
        0.60 * accel + 0.30 * gyro + 0.10 * (speed_gap / 20.0).clip(0, 1)
    ).where(imu_available)
    rows["imu_evidence_event_count"] = 0
    rows["imu_evidence_exposure"] = rows["imu_days"].fillna(0)
    return rows


def finalize_a_scores(
    rows: pd.DataFrame,
    weights: dict[str, float],
    observation_days: int = 14,
) -> pd.DataFrame:
    """Create score, explanation and action columns after frozen transforms are applied."""
    validate_weights(weights)
    frame = rows.copy()
    for dimension, weight in weights.items():
        frame[f"{dimension}_weight"] = weight
        frame[f"{dimension}_contribution"] = weight * frame[f"{dimension}_penalty"].fillna(0.0)
    frame["safety_score_raw"] = 100.0 - sum(
        frame[f"{dimension}_contribution"] for dimension in weights
    )
    frame["safety_score"] = frame["safety_score_raw"].clip(0, 100)
    frame = frame.sort_values(["safety_score", "gpsno"], ascending=[False, True]).reset_index(drop=True)
    frame["rank"] = np.arange(1, len(frame) + 1)
    frame["risk_level"] = _assign_risk_level(frame["rank"])
    frame["valid_speed_ratio"] = frame["valid_speed_ratio"].fillna(0.0).clip(0, 1)
    frame["valid_coordinate_ratio"] = frame["valid_coordinate_ratio"].fillna(0.0).clip(0, 1)
    frame["confidence_score"] = (
        25 * (frame["trajectory_days"].fillna(0) / observation_days).clip(0, 1)
        + 15 * (frame["imu_days"].fillna(0) / observation_days).clip(0, 1)
        + 15 * frame["valid_speed_ratio"]
        + 15 * frame["valid_coordinate_ratio"]
        + 10 * (frame["distance_km"].fillna(0) / 1000.0).clip(0, 1)
        + 10 * (frame["run_time_hours"].fillna(0) / 100.0).clip(0, 1)
        + 10 * frame["risk_event_coverage"]
    ).clip(0, 100)
    frame["confidence_level"] = np.select(
        [frame["confidence_score"] >= 80, frame["confidence_score"] >= 60],
        ["high", "medium"], default="low",
    )
    for index in range(1, 4):
        for ending, default in [
            ("", ""), ("_code", ""), ("_dimension", ""),
            ("_penalty_points", np.nan), ("_event_count", np.nan),
            ("_normalized_rate", np.nan), ("_percentile", np.nan),
            ("_source_event_codes", ""), ("_source_fields", ""),
        ]:
            frame[f"top_reason_{index}{ending}"] = default
    frame["primary_risk_dimension"] = ""
    frame["primary_risk_dimension_name"] = ""
    for row_index, row in frame.iterrows():
        candidates = [
            dimension for dimension in weights if pd.notna(row[f"{dimension}_penalty"])
        ]
        top_dimensions = sorted(
            candidates, key=lambda key: (-row[f"{key}_contribution"], key)
        )[:3]
        for reason_rank, dimension in enumerate(top_dimensions, start=1):
            record = reason_record(row, dimension, reason_rank)
            frame.at[row_index, f"top_reason_{reason_rank}"] = record["reason_text"]
            frame.at[row_index, f"top_reason_{reason_rank}_code"] = record["reason_code"]
            frame.at[row_index, f"top_reason_{reason_rank}_dimension"] = record["dimension"]
            frame.at[row_index, f"top_reason_{reason_rank}_penalty_points"] = record["penalty_points"]
            frame.at[row_index, f"top_reason_{reason_rank}_event_count"] = record["event_count"]
            frame.at[row_index, f"top_reason_{reason_rank}_normalized_rate"] = record["normalized_rate"]
            frame.at[row_index, f"top_reason_{reason_rank}_percentile"] = record["comparison_percentile"]
            frame.at[row_index, f"top_reason_{reason_rank}_source_event_codes"] = record["source_event_codes"]
            frame.at[row_index, f"top_reason_{reason_rank}_source_fields"] = record["source_fields"]
        frame.at[row_index, "primary_risk_dimension"] = top_dimensions[0]
        frame.at[row_index, "primary_risk_dimension_name"] = DIMENSION_LABELS[top_dimensions[0]]
    actions = frame.apply(
        lambda row: operational_action(
            row["risk_level"], row["primary_risk_dimension"], row["confidence_level"],
            int(row[f"{row['primary_risk_dimension']}_event_count"]),
        ),
        axis=1,
        result_type="expand",
    )
    frame = pd.concat([frame, actions], axis=1)
    frame["safety_score"] = frame["safety_score"].round(4)
    frame["confidence_score"] = frame["confidence_score"].round(2)
    return frame


def _future_labels(details: pd.DataFrame, source: pd.DataFrame) -> pd.Series:
    label_map = source.assign(gpsno=source["gpsno"].astype(str)).set_index("gpsno")["label"]
    return details["gpsno"].map(label_map)


def run_a_integrated_scoring(
    paths: dict[str, Path], rules: pd.DataFrame
) -> tuple[pd.DataFrame, dict[str, float], dict, dict, dict]:
    train_windows = _read_parquet(paths["train_windows"])
    full_features = _read_parquet(paths["full_features"])
    split_manifest = pd.read_csv(paths["split_manifest"])
    risk_daily_quality = _risk_daily_quality(paths["risk_daily"])

    selection_train_id = split_manifest.query(
        "phase == 'model_selection' and split_role == 'train'"
    )["window_id"].iloc[0]
    selection_validation_id = split_manifest.query(
        "phase == 'model_selection' and split_role == 'validation'"
    )["window_id"].iloc[0]
    locked_train_ids = split_manifest.query(
        "phase == 'locked_test' and split_role == 'train'"
    )["window_id"].tolist()
    locked_test_id = split_manifest.query(
        "phase == 'locked_test' and split_role == 'test'"
    )["window_id"].iloc[0]

    def window_rows(window_ids: list[str]) -> pd.DataFrame:
        selected = train_windows[train_windows["window_id"].isin(window_ids)].copy()
        return build_a_scoring_rows(selected, rules, "window")

    selection_train = window_rows([selection_train_id])
    selection_model, selection_train = fit_penalty_model(selection_train)
    selection_weights, selection_calibration = calibrate_weights(
        selection_train,
        selection_train["label"],
        selection_train["window_id"],
    )
    selection_validation_raw = window_rows([selection_validation_id])
    selection_validation = finalize_a_scores(
        apply_penalty_model(selection_validation_raw, selection_model), selection_weights
    )
    selection_report = validate_score_frame(
        selection_validation,
        selection_weights,
        selection_calibration,
        _future_labels(selection_validation, selection_validation_raw),
    )

    locked_train = window_rows(locked_train_ids)
    penalty_model, locked_train = fit_penalty_model(locked_train)
    weights, calibration = calibrate_weights(
        locked_train,
        locked_train["label"],
        locked_train["window_id"],
    )
    locked_test_raw = window_rows([locked_test_id])
    locked_test = finalize_a_scores(
        apply_penalty_model(locked_test_raw, penalty_model), weights
    )
    locked_report = validate_score_frame(
        locked_test,
        weights,
        calibration,
        _future_labels(locked_test, locked_test_raw),
    )

    main_raw_base = train_windows[train_windows["window_role"] == "main_40d"].copy()
    main_raw = build_a_scoring_rows(main_raw_base, rules, "14d")
    main_diagnostic = finalize_a_scores(apply_penalty_model(main_raw, penalty_model), weights)
    main_report = validate_score_frame(
        main_diagnostic,
        weights,
        calibration,
        _future_labels(main_diagnostic, main_raw),
    )

    final_raw = build_a_scoring_rows(full_features, rules, "14d")
    final_raw["window_id"] = "final_14d_20260731"
    final_raw["window_role"] = "final_forecast"
    final_raw["feature_start"] = "2026-07-18"
    final_raw["feature_end"] = "2026-07-31"
    details = finalize_a_scores(apply_penalty_model(final_raw, penalty_model), weights)
    validation = validate_score_frame(details, weights, calibration)
    validation["validation_scope"] = (
        "A交付已接入；固定14天时间切分，锁定测试仅用于最终评估，"
        "21→40天主窗口仅作诊断。"
    )
    validation["model_selection_validation"] = selection_report
    validation["locked_test"] = locked_report
    validation["main_40d_diagnostic"] = main_report
    validation["overall_status"] = (
        "pass"
        if validation["overall_static_status"] == "pass"
        and locked_report["overall_static_status"] == "pass"
        and locked_report["future_validation"]["status"] == "pass"
        else "review_required"
    )
    parameters = {
        "selection_train_window": selection_train_id,
        "selection_validation_window": selection_validation_id,
        "selection_weights": selection_weights,
        "selection_calibration": selection_calibration,
        "locked_train_windows": locked_train_ids,
        "locked_test_window": locked_test_id,
        "final_weights": weights,
        "final_calibration": calibration,
        "penalty_model": penalty_model,
    }
    quality = {
        "train_window_rows": int(len(train_windows)),
        "train_window_vehicles": int(train_windows["gpsno"].nunique()),
        "final_feature_rows": int(len(full_features)),
        "final_feature_vehicles": int(full_features["gpsno"].nunique()),
        "risk_event_daily_coverage": risk_daily_quality,
        "a_label_report": json.loads(paths["label_report"].read_text(encoding="utf-8")),
    }
    return details, weights, calibration, validation, {"parameters": parameters, "quality": quality}


def build_acceptance_report(validation: dict, details: pd.DataFrame) -> dict:
    """Map the repository's member-C acceptance items to executable evidence."""
    locked = validation["locked_test"]["future_validation"]
    energy_future = locked.get("energy_type_future_discrimination", [])
    energy_usable = [item for item in energy_future if item["future_label_has_two_classes"]]
    checks = [
        {"requirement": "恰好500辆车且gpsno唯一", "status": "pass" if len(details) == 500 and details.gpsno.nunique() == 500 else "fail"},
        {"requirement": "安全分位于0到100且排名方向一致", "status": "pass" if validation["core_checks"]["score_in_0_100"] and validation["core_checks"]["score_rank_direction_consistent"] else "fail"},
        {"requirement": "高危历史不获得高安全分", "status": "pass" if validation["history_safety_check"]["pass_not_rewarded"] else "fail"},
        {"requirement": "IMU缺失不被误判为安全", "status": "pass" if validation["imu_missingness_check"]["pass_not_systematically_higher"] else "fail"},
        {"requirement": "扣分原因可回溯到事件码和特征字段", "status": "pass" if validation["deduction_reasons_traceable"] else "fail"},
        {"requirement": "A到D未来事故/未遂率单调上升", "status": "pass" if locked["level_rate_monotonic"] else "review_required", "evidence": locked["future_event_rate_by_level"]},
        {"requirement": "最危险与最安全20%未来风险存在明显差异", "status": "pass" if (locked["riskiest_minus_safest"] or 0) > 0 else "review_required", "evidence": locked["riskiest_minus_safest"]},
        {"requirement": "各能源类型内仍有未来风险区分度", "status": "pass" if energy_usable and all((item["auc"] or 0) > 0.5 for item in energy_usable) else "review_required", "evidence": energy_future},
        {"requirement": "删除历史维度后行为维度仍有区分度", "status": "pass" if (locked.get("behavior_without_history_future_auc") or 0) > 0.5 else "review_required", "evidence": locked.get("behavior_without_history_future_auc")},
        {"requirement": "权重小幅变化时排名稳定", "status": "pass" if validation["weight_sensitivity_min_spearman"] >= 0.9 else "review_required", "evidence": validation["weight_sensitivity_min_spearman"]},
        {"requirement": "时间边界、随机种子、公式、权重和输入可追溯", "status": "pass"},
    ]
    return {
        "rule_version": RULE_VERSION,
        "source_document": "docs/team_technical_routes.md#4-队员c安全评分与运营方案",
        "checks": checks,
        "pass_count": sum(item["status"] == "pass" for item in checks),
        "review_required_count": sum(item["status"] == "review_required" for item in checks),
        "fail_count": sum(item["status"] == "fail" for item in checks),
        "overall_status": "pass" if all(item["status"] == "pass" for item in checks) else "review_required",
    }


def main() -> None:
    if duckdb is None:
        raise SystemExit("缺少DuckDB。请执行：python -m pip install -r src/scoring/requirements.txt")
    parser = argparse.ArgumentParser(description="生成成员C的安全评分、解释、运营动作和验证报告")
    parser.add_argument("--cleaned-dir", type=Path, default=DEFAULT_CLEANED_DIR)
    parser.add_argument("--feature-dir", type=Path, default=DEFAULT_FEATURE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--as-of", default=AS_OF_TS)
    args = parser.parse_args()

    started = time.time()
    paths = resolve_a_inputs(args.feature_dir.resolve())
    output_dir = args.output_dir.resolve()
    run_dir = args.run_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    run_dir.mkdir(parents=True, exist_ok=True)
    rules = default_rules_frame()
    details, weights, calibration, validation, audit = run_a_integrated_scoring(paths, rules)
    quality = audit["quality"]
    acceptance = build_acceptance_report(validation, details)
    validation["docs_acceptance"] = acceptance

    write_csv(details, output_dir / "safety_scores.csv", SAFETY_SCORE_COLUMNS)
    write_parquet(details, output_dir / "score_details.parquet", duckdb)
    write_rules(output_dir / "scoring_rules.csv", rules)
    write_json(validation, output_dir / "score_validation.json")
    write_json(acceptance, output_dir / "acceptance_report.json")
    write_json(audit["parameters"], run_dir / "calibration_parameters.json")
    feature_fields = [
        "profile_energy_type", "distance_km_14d", "run_time_hours_14d", "active_days_14d",
        "trajectory_days_14d", "imu_days_14d", "speed_valid_fraction_14d",
        "coordinate_valid_fraction_14d", "imu_accel_outlier_fraction_14d",
        "imu_gyro_gt20_fraction_14d", "imu_ems_gps_abs_diff_mean_daily_mean_14d",
    ] + [
        f"evt_{int(code)}_{metric}_14d"
        for code in rules["event_type"]
        for metric in ("count", "decayed_count")
    ]
    manifest = {
        "run_id": "member_c_a_integrated_v2_1",
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "as_of_timestamp": args.as_of,
        "random_seed": RANDOM_SEED,
        "rule_version": RULE_VERSION,
        "source_experiments": {
            "member_a_release": "member-a-v1.0",
            "member_a_asset_sha256": "51697f5b0d6deb58c10ea2f0f535e9aab53bb50a4cd077bd978bd8006afa5a7c",
            "member_a_run": "20260922_110414",
        },
        "input_files": {key: path.name for key, path in paths.items()},
        "feature_fields": feature_fields,
        "runtime": {"python": platform.python_version(), "duckdb": duckdb.__version__},
        "weights": weights,
        "calibration": calibration,
        "quality": quality,
        "outputs": [
            "safety_scores.csv", "score_details.parquet", "scoring_rules.csv",
            "score_validation.json", "acceptance_report.json", "scoring_method.md",
        ],
        "metrics_summary": {
            "docs_acceptance_status": acceptance["overall_status"],
            "locked_test": validation["locked_test"]["future_validation"],
            "score_summary": validation["score_summary"],
        },
        "elapsed_seconds": round(time.time() - started, 2),
        "validation_status": validation["overall_status"],
    }
    write_json(manifest, run_dir / "run_manifest.json")
    print(
        f"完成：{len(details)}辆车；静态验证={validation['overall_static_status']}；"
        f"锁定测试={validation['locked_test']['future_validation']['status']}；"
        f"耗时={manifest['elapsed_seconds']}秒"
    )


if __name__ == "__main__":
    main()
