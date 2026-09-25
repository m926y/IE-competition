"""Versioned, auditable event severities, exposure units, and weight constraints."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .event_taxonomy import DIMENSION_LABELS, EVENT_TAXONOMY, EXPECTED_EVENT_CODES

RULE_VERSION = "member_c_v2.1_docs_complete"
HALF_LIFE_DAYS = 14.0

DEFAULT_WEIGHTS = {
    "history": 0.35,
    "fatigue_distraction": 0.20,
    "driving_stability": 0.20,
    "road_rule": 0.10,
    "management": 0.10,
    "imu_evidence": 0.05,
}

EVENT_SEVERITY = {
    11803: 10.0, 11804: 7.0,
    41001: 5.0, 41029: 4.5, 41002: 3.5, 41003: 3.0,
    41004: 3.0, 41023: 3.5, 41009: 3.0, 41005: 2.0,
    30000: 5.0, 30005: 3.0, 30017: 2.5, 30002: 1.5,
    30003: 1.5, 60292: 2.0, 60294: 2.0,
    11401: 2.5, 11402: 3.0, 11403: 3.0, 11405: 3.0, 11406: 4.0,
    41006: 2.0, 41021: 1.5,
}

EXPOSURE_UNIT = {
    "history": "per_1000km",
    "fatigue_distraction": "per_100h",
    "driving_stability": "per_1000km",
    "road_rule": "per_1000km",
    "management": "per_active_day",
}

# Correlated alarms are aggregated inside one dimension and can never deduct
# more than this dimension-level cap, regardless of how many related event
# codes fire together. IMU remains a lower-weight corroborating signal.
DIMENSION_PENALTY_CAPS = {
    "history": 100.0,
    "fatigue_distraction": 100.0,
    "driving_stability": 100.0,
    "road_rule": 100.0,
    "management": 100.0,
    "imu_evidence": 60.0,
}


def default_rules_frame() -> pd.DataFrame:
    rows = []
    for event_type in sorted(EVENT_TAXONOMY):
        event_name, dimension = EVENT_TAXONOMY[event_type]
        rows.append({
            "event_type": event_type,
            "event_name": event_name,
            "dimension": dimension,
            "dimension_name": DIMENSION_LABELS[dimension],
            "severity": EVENT_SEVERITY[event_type],
            "exposure_unit": EXPOSURE_UNIT[dimension],
            "half_life_days": HALF_LIFE_DAYS,
            "dimension_penalty_cap": DIMENSION_PENALTY_CAPS[dimension],
            "rule_version": RULE_VERSION,
        })
    return pd.DataFrame(rows)


def validate_rules(rules: pd.DataFrame) -> None:
    required = {
        "event_type", "event_name", "dimension", "severity",
        "exposure_unit", "half_life_days", "dimension_penalty_cap", "rule_version",
    }
    missing = required - set(rules.columns)
    if missing:
        raise ValueError(f"scoring_rules.csv 缺少字段：{sorted(missing)}")
    codes = set(rules["event_type"].astype(int))
    if codes != EXPECTED_EVENT_CODES or rules["event_type"].duplicated().any():
        raise ValueError("scoring_rules.csv 必须且只能覆盖24个已知事件码")
    if (rules["severity"].astype(float) <= 0).any():
        raise ValueError("事件严重度必须大于0")
    if (rules["half_life_days"].astype(float) <= 0).any():
        raise ValueError("半衰期必须大于0")
    if not rules["dimension_penalty_cap"].astype(float).between(0, 100, inclusive="right").all():
        raise ValueError("维度扣分上限必须在(0, 100]内")
    for dimension, expected_cap in DIMENSION_PENALTY_CAPS.items():
        observed = rules.loc[rules["dimension"] == dimension, "dimension_penalty_cap"].unique()
        if dimension != "imu_evidence" and (len(observed) != 1 or observed[0] != expected_cap):
            raise ValueError(f"{dimension}维度扣分上限不一致")
    if set(rules["dimension"]) - set(DIMENSION_LABELS):
        raise ValueError("存在未知评分维度")


def load_rules(path: Path | None = None) -> pd.DataFrame:
    """Load an explicit rule table, or return the versioned repository defaults."""
    rules = default_rules_frame() if path is None else pd.read_csv(path, encoding="utf-8-sig")
    validate_rules(rules)
    return rules


def write_rules(path: Path, rules: pd.DataFrame | None = None) -> None:
    rules = default_rules_frame() if rules is None else rules
    validate_rules(rules)
    path.parent.mkdir(parents=True, exist_ok=True)
    rules.to_csv(path, index=False, encoding="utf-8-sig")


def validate_weights(weights: dict[str, float]) -> None:
    if set(weights) != set(DEFAULT_WEIGHTS):
        raise ValueError("权重必须覆盖六个固定维度")
    if any(value < 0 for value in weights.values()):
        raise ValueError("权重不得为负")
    if abs(sum(weights.values()) - 1.0) > 1e-9:
        raise ValueError("权重之和必须为1")
    if weights["history"] < max(value for key, value in weights.items() if key != "history"):
        raise ValueError("历史事故与未遂必须是最高权重")
    if weights["fatigue_distraction"] < weights["road_rule"]:
        raise ValueError("疲劳分心权重不得低于道路规则风险")
    if weights["driving_stability"] < weights["road_rule"]:
        raise ValueError("驾驶稳定性权重不得低于道路规则风险")
    if weights["imu_evidence"] > 0.10:
        raise ValueError("IMU权重不得超过10%")
