"""Train, validate and export the complete task-one prediction pipeline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .data_loader import (
    ID_COLUMN,
    align_prediction_features,
    load_feature_dictionary,
    load_prediction_data,
    load_training_data,
    read_table,
    select_feature_columns,
)
from .ensemble import (
    calibrate_probability,
    combine_final_predictions,
    fit_platt_calibrator,
    select_prediction_strategy,
)
from .evaluate import binary_metrics, choose_operating_threshold
from .splitters import make_time_folds
from .train_boosting import (
    boosting_importance,
    cross_validate_boosting,
    fit_final_boosting,
)
from .train_logistic import (
    cross_validate_logistic,
    fit_final_logistic,
    logistic_importance,
)
from .validate_submission import validate_submission


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TRAIN = ROOT / "outputs" / "features" / "train_windows.parquet"
DEFAULT_FULL = ROOT / "outputs" / "features" / "full_61d_features.parquet"
DEFAULT_DICTIONARY = ROOT / "outputs" / "features" / "feature_dictionary.csv"
DEFAULT_EXPECTED_IDS = ROOT / "data" / "processed" / "cleaned" / "vehicle_profile.csv"
DEFAULT_OUTPUT = ROOT / "outputs" / "prediction"
DEFAULT_SEED = 2026


def _json_default(value):
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"无法序列化：{type(value)!r}")


def _normalise_records(records: list[dict]) -> list[dict]:
    return [
        {
            key: (None if isinstance(value, float) and not np.isfinite(value) else value)
            for key, value in record.items()
        }
        for record in records
    ]


def _feature_group_map(
    feature_columns: list[str], dictionary: pd.DataFrame | None
) -> dict[str, str]:
    def normalise_group(value: str) -> str:
        lowered = str(value).strip().lower()
        aliases = {
            "profile": ("profile", "画像", "静态"),
            "risk": ("risk", "event", "风险", "事件"),
            "exposure": ("exposure", "暴露", "归一化"),
            "recency": ("recency", "trend", "近期", "趋势", "聚集"),
            "trajectory": ("trajectory", "轨迹"),
            "imu": ("imu", "惯性", "陀螺"),
            "other": ("other", "quality", "其他", "质量"),
        }
        return next(
            (group for group, tokens in aliases.items() if any(token in lowered for token in tokens)),
            "other",
        )

    result: dict[str, str] = {}
    if dictionary is not None:
        mapping = dictionary.set_index("feature_name")["feature_group"].astype(str).to_dict()
        result.update(
            {
                feature: normalise_group(mapping[feature])
                for feature in feature_columns
                if feature in mapping
            }
        )
    rules = (
        ("profile", ("energy", "monthly_", "highway_", "morning_", "dusk_", "night_")),
        ("imu", ("imu_", "gyro_", "accel_", "ems_", "gps_speed")),
        ("trajectory", ("trajectory_", "distance_", "mileage_", "run_time", "speed_")),
        ("recency", ("recent_", "recency", "trend", "days_since", "burst", "streak")),
        ("exposure", ("per_100", "per_1000", "exposure", "active_days", "coverage")),
        ("risk", ("risk_", "event_", "accident_", "near_miss", "fatigue", "distraction")),
    )
    for feature in feature_columns:
        if feature in result:
            continue
        lowered = feature.lower()
        result[feature] = next(
            (group for group, tokens in rules if any(token in lowered for token in tokens)),
            "other",
        )
    return result


def run_ablation(
    frame: pd.DataFrame,
    features: list[str],
    folds,
    dictionary: pd.DataFrame | None,
    boosting_params: dict,
    *,
    random_state: int,
) -> pd.DataFrame:
    group_map = _feature_group_map(features, dictionary)
    order = ["profile", "risk", "exposure", "recency", "trajectory", "imu", "other"]
    selected: list[str] = []
    rows: list[dict] = []
    for group in order:
        selected.extend([feature for feature in features if group_map[feature] == group])
        selected = list(dict.fromkeys(selected))
        if not selected:
            continue
        result = cross_validate_boosting(
            frame,
            selected,
            folds,
            parameter_grid=(boosting_params,),
            random_state=random_state,
        )
        rows.append(
            {"added_group": group, "feature_count": len(selected), **result["summary"]}
        )
    return pd.DataFrame(rows)


def _write_method(
    path: Path,
    *,
    feature_count: int,
    fold_count: int,
    logistic: dict,
    boosting: dict,
    strategy: dict,
    threshold: dict,
    random_state: int,
) -> None:
    blend_text = strategy["strategy"]
    if strategy["strategy"] == "rank_blend":
        blend_text += f"（LightGBM权重 {strategy['weight']:.2f}）"
    text = f"""# 事故预测方法说明

## 数据与时间边界

- 输入为队员A交付的滚动训练窗口和完整61天车辆特征；
- 样本单位为车辆与时间窗口；
- 训练窗口严格早于验证窗口，不随机拆分事件；
- 所有填补、截尾、标准化和独热编码只在训练折拟合；
- 模型未使用任何比赛外部数据，也未使用`gpsno`作为特征。

## 模型

- 有效特征数：{feature_count}
- 时间验证折数：{fold_count}
- 逻辑回归参数：`{logistic['params']}`
- 逻辑回归平均AUC：{logistic['summary']['mean_roc_auc']:.6f}
- LightGBM参数：`{boosting['params']}`
- LightGBM平均AUC：{boosting['summary']['mean_roc_auc']:.6f}
- 最终策略：{blend_text}
- 最终策略平均AUC：{strategy['summary']['mean_roc_auc']:.6f}

逻辑回归采用L2正则和类别权重；LightGBM限制树深、叶子数和最小叶节点样本量，并使用早停。最终策略由多个时间窗口的平均AUC与最差窗口AUC共同决定。排名融合只有在至少两个时间窗口不差于最佳单模型，且最差窗口不恶化时才启用。

## 概率与二分类

使用训练外预测拟合Platt校准器。连续概率写入`risk_prob`；二分类阈值为`{threshold['threshold']:.8f}`，对应验证召回率`{threshold['recall']:.6f}`和特异度`{threshold['specificity']:.6f}`。

## 复现

```powershell
python -m src.prediction.train_final
python -m src.prediction.validate_submission outputs/prediction/forecast_result.csv --expected-ids data/processed/cleaned/vehicle_profile.csv
```

随机种子固定为{random_state}。模型、预处理器、校准器、特征顺序和阈值均保存在`outputs/prediction/model/`。
"""
    path.write_text(text, encoding="utf-8")


def run(args: argparse.Namespace) -> dict:
    """Legacy synthetic-test harness; formal A inputs must use integrated.run."""
    train = load_training_data(args.train_features)
    if 'window_role' in train:
        raise ValueError('Use src.prediction.integrated for formal A feature contracts')
    full = load_prediction_data(args.full_features)
    features = select_feature_columns(train)
    aligned_full = align_prediction_features(full, features)
    folds = make_time_folds(train, min_train_windows=args.min_train_windows)
    dictionary = load_feature_dictionary(args.feature_dictionary)

    logistic = cross_validate_logistic(
        train, features, folds, random_state=args.random_state
    )
    boosting = cross_validate_boosting(
        train, features, folds, random_state=args.random_state
    )
    strategy = select_prediction_strategy(
        train["label"].to_numpy(), logistic["oof"], boosting["oof"], folds
    )
    calibrator = fit_platt_calibrator(train["label"].to_numpy(), strategy["score"])
    calibrated_oof = np.full(len(train), np.nan)
    valid_oof = np.isfinite(strategy["score"])
    calibrated_oof[valid_oof] = calibrate_probability(
        calibrator, strategy["score"][valid_oof]
    )
    threshold = choose_operating_threshold(
        train.loc[valid_oof, "label"], calibrated_oof[valid_oof]
    )

    logistic_model = fit_final_logistic(
        train, features, logistic["params"], random_state=args.random_state
    )
    final_iterations = max(20, int(round(boosting["summary"]["median_best_iteration"])))
    boosting_model = fit_final_boosting(
        train,
        features,
        boosting["params"],
        n_estimators=final_iterations,
        random_state=args.random_state,
    )
    logistic_probability = logistic_model.predict_proba(aligned_full[features])[:, 1]
    boosting_probability = boosting_model.predict_proba(aligned_full[features])[:, 1]
    raw_final_score = combine_final_predictions(
        strategy, logistic_probability, boosting_probability
    )
    final_probability = calibrate_probability(calibrator, raw_final_score)

    output = Path(args.output_dir)
    model_dir = output / "model"
    model_dir.mkdir(parents=True, exist_ok=True)
    submission = pd.DataFrame(
        {
            "gpsno": aligned_full[ID_COLUMN].astype(str),
            "accident": (final_probability >= threshold["threshold"]).astype("int8"),
            "risk_prob": final_probability,
        }
    )
    expected_ids = read_table(args.expected_ids)[ID_COLUMN].astype("string")
    validation = validate_submission(
        submission, expected_ids=expected_ids, expected_rows=len(expected_ids)
    )
    submission.to_csv(output / "forecast_result.csv", index=False, encoding="utf-8")

    oof = train[[ID_COLUMN, "window_id", "label", "feature_end", "label_start"]].copy()
    oof["logistic_probability"] = logistic["oof"]
    oof["lightgbm_probability"] = boosting["oof"]
    oof["selected_raw_score"] = strategy["score"]
    oof["calibrated_probability"] = calibrated_oof
    oof.loc[valid_oof].to_parquet(output / "oof_predictions.parquet", index=False)

    importance = pd.concat(
        [logistic_importance(logistic_model), boosting_importance(boosting_model)],
        ignore_index=True,
    )
    importance.to_csv(output / "feature_importance.csv", index=False, encoding="utf-8")
    if args.skip_ablation:
        ablation = pd.DataFrame(
            [{"added_group": "all", "feature_count": len(features), **boosting["summary"]}]
        )
    else:
        ablation = run_ablation(
            train,
            features,
            folds,
            dictionary,
            boosting["params"],
            random_state=args.random_state,
        )
    ablation.to_csv(output / "ablation_results.csv", index=False, encoding="utf-8")

    joblib.dump(logistic_model, model_dir / "logistic.joblib")
    joblib.dump(boosting_model, model_dir / "lightgbm.joblib")
    joblib.dump(calibrator, model_dir / "platt_calibrator.joblib")
    metadata = {
        "random_state": args.random_state,
        "feature_columns": features,
        "strategy": strategy["strategy"],
        "blend_weight": strategy["weight"],
        "threshold": threshold,
        "lightgbm_n_estimators": final_iterations,
        "logistic_params": logistic["params"],
        "lightgbm_params": boosting["params"],
    }
    (model_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, default=_json_default),
        encoding="utf-8",
    )

    metrics = {
        "data": {
            "training_rows": len(train),
            "vehicles": int(train[ID_COLUMN].nunique()),
            "feature_count": len(features),
            "fold_count": len(folds),
        },
        "logistic": {
            "params": logistic["params"],
            "summary": logistic["summary"],
            "folds": _normalise_records(logistic["records"]),
        },
        "lightgbm": {
            "params": boosting["params"],
            "summary": boosting["summary"],
            "folds": _normalise_records(boosting["records"]),
        },
        "selected_strategy": {
            "name": strategy["strategy"],
            "weight": strategy["weight"],
            "summary": strategy["summary"],
            "folds": _normalise_records(strategy["records"]),
        },
        "calibrated_oof": binary_metrics(
            train.loc[valid_oof, "label"], calibrated_oof[valid_oof]
        ),
        "threshold": threshold,
        "submission": validation,
    }
    (output / "validation_metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2, default=_json_default),
        encoding="utf-8",
    )
    _write_method(
        output / "prediction_method.md",
        feature_count=len(features),
        fold_count=len(folds),
        logistic=logistic,
        boosting=boosting,
        strategy=strategy,
        threshold=threshold,
        random_state=args.random_state,
    )
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="QA-compliant fixed-window prediction pipeline")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    from .integrated import run as run_integrated
    run_integrated(args.output_dir)


if __name__ == "__main__":
    main()
