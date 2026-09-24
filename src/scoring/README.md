# 成员 C 安全评分运行说明

本目录实现任务二的可解释 100 分制安全评分。它直接读取公共清洗层，不依赖成员 A 的内部函数，也不会修改原始数据或清洗数据。

## 当前可运行范围

在 A 尚未交付时间标签和特征字典时，程序已经完成：24 类事件映射、严重度配置、14 天时间衰减、里程/时长/活跃日暴露归一化、经验贝叶斯收缩、六维惩罚、固定初始权重、排名、A/B/C/D 等级、可信度、前三扣分原因、运营动作和静态验证。

暂不能声称完成的内容只有依赖未来标签的三项：权重校准、风险等级对未来事故/未遂率的单调性、最危险与最安全 20% 的未来风险差。

## 安装与运行

从仓库根目录执行：

```powershell
python -m pip install -r src/scoring/requirements.txt
python -m src.scoring.calculate_scores
```

默认读取：

```text
data/processed/cleaned/
├─ vehicle_profile.parquet
├─ risk_events.parquet
├─ trajectory_daily.parquet
└─ imu_daily.parquet
```

如果清洗文件在其他只读目录：

```powershell
python -m src.scoring.calculate_scores `
  --cleaned-dir D:\path\to\cleaned `
  --output-dir outputs\scoring `
  --run-dir runs\member_c
```

运行测试：

```powershell
python -m unittest discover -s tests -v
```

## 固定输出

```text
outputs/scoring/
├─ safety_scores.csv
├─ score_details.parquet
├─ scoring_rules.csv
├─ score_validation.json
└─ scoring_method.md

runs/member_c/
└─ run_manifest.json
```

真实评分、Parquet 和运行记录受数据保密规则约束，不提交到公开 GitHub；方法文档、规则、源码、测试和输出契约会提交。

## A 交付后的接入点

`calibrate_weights.py` 已实现带时间窗口的单维度 AUC 检查、L2 正则逻辑回归方向估计和受约束有限校准。拿到 A 的 `risk_event_daily.parquet`、`feature_dictionary.csv` 和 `label_report.json` 后，先按字典核对事故/未遂字段并构造未来标签，再用同样的截止日期为每个窗口重算六维惩罚，最后把惩罚、`future_label` 和 `window_id` 传给 `calibrate_weights()`；不得用标签期数据计算事件率、总体均值、收缩参数或分位点。
