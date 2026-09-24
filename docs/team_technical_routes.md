# 答疑修订说明

以下原设计以当前整合实现为准：label=事故次数≥1或未遂次数≥2；21→40窗口仅诊断，正式采用A的固定14天开发/锁定测试；禁止将半年画像或标签辅助列放入早期回测；B/C只按开发集选参，最终测试仅报告；40天外推限制见docs/integration_review.md。

# 三人并行最优技术路线

本文规定三名成员统一采用的技术路线。三条工作流并行开发，通过稳定文件接口集成，避免重复工作、代码冲突和数据泄漏。

## 1. 总体架构

```text
公共清洗层 data/processed/cleaned/
          │
          ├───────────────┬────────────────┐
          │               │                │
          ▼               ▼                ▼
   A：标签与特征     B：预测框架      C：安全评分框架
          │               │                │
          └────特征接口───►│                │
          │               │                │
          └────特征字典────────────────────►│
                          │                │
                          ▼                ▼
                 forecast_result.csv  safety_scores.csv
                          │                │
                          └──────┬─────────┘
                                 ▼
                         outputs/final/
```

统一原则：

1. 原始数据和公共清洗层只读；
2. A、B、C只修改各自源码与输出目录；
3. 成员之间通过Parquet、CSV和JSON接口交付；
4. 所有特征、模型和评分都遵守时间边界；
5. 不使用任何比赛外部数据；
6. 最终结果必须覆盖500辆车，`gpsno`唯一。

---

# 2. 队员A：时间标签与特征工程

## 2.1 责任边界

队员A负责：

- 风险事件日表；
- 主验证和滚动时间标签；
- 训练窗口特征；
- 完整61天预测特征；
- 特征字典和质量报告。

队员A不负责：

- 最终预测模型选择；
- 最终评分权重；
- 修改公共清洗层。

独占目录：

```text
src/features/
tests/features/
outputs/features/
runs/member_a/
```

输入：

```text
data/processed/cleaned/
├─ vehicle_profile.parquet
├─ risk_events.parquet
├─ trajectory_daily.parquet
├─ imu_daily.parquet
└─ vehicle_day.parquet
```

## 2.2 代码结构

```text
src/features/
├─ event_groups.py              # 事件编码与风险维度映射
├─ build_risk_daily.py          # 风险事件日表
├─ build_labels.py              # 时间标签
├─ build_window_features.py     # 窗口特征
├─ build_full_features.py       # 完整61天特征
├─ feature_contract.py          # 字段契约
└─ validate_features.py         # 泄漏、缺失和唯一键检查
```

## 2.3 风险事件日表

从`risk_events.parquet`生成：

```text
outputs/features/risk_event_daily.parquet
```

主键：

```text
gpsno + event_date
```

事件按以下维度映射：

| 维度 | 事件示例 |
|---|---|
| 高危历史 | 事故11803、未遂事故11804 |
| 疲劳 | 闭眼、困倦、打哈欠 |
| 分心 | 低头、电话、手机、注意力分散、吸烟 |
| 稳定性 | 左右车道偏移、长时间压线、车距过近、前碰撞预警 |
| 道路风险 | 路口、弯道、危险路段、匝道、长下坡超速 |
| 管理配合 | 摄像头遮挡、摄像头角度异常 |
| 盲区风险 | 后盲区、右盲区报警 |

日表至少包含：

- 每个`event_type`日次数；
- 每个风险维度日次数；
- 事故和未遂事故日次数；
- 风险事件发生速度均值与最大值；
- 风险事件活跃小时数；
- 同日出现的风险种类数；
- 是否出现高危事件。

统计与标签必须使用`event_type`，不能依赖可能变化的`event_name`文本。

## 2.4 时间标签

### 主验证标签

```text
特征期：2026-06-01 至 2026-06-21
标签期：2026-06-22 至 2026-07-31
正类：标签期11803次数≥1或11804次数≥2
```

### 滚动训练标签

构造多个截止日，使用：

```text
过去14天特征 -> 未来14天标签
过去21天特征 -> 未来14天标签
过去30天特征 -> 未来14天标签
```

每条样本必须保留：

```text
gpsno
feature_start
feature_end
label_start
label_end
label
window_id
```

标签规则：

```text
label = 1，当标签期内count(11803)>=1 OR count(11804)>=2
label = 0，否则
```

防泄漏要求：

- 只允许使用`event_time <= feature_end 23:59:59`的数据；
- 标签期事故、未遂和其他风险事件均不得进入特征；
- 历史事故和未遂可以作为特征，但必须发生在特征截止日前；
- 任何全体均值、分位数和收缩参数都必须仅使用训练窗口估计；
- 不得随机拆分原始事件记录。

## 2.5 特征体系

### 2.5.1 静态画像

- 能源类型；
- 月均行驶里程；
- 月均行驶时长；
- 月平均停留次数；
- 高速里程占比；
- 早晨、黄昏、夜间驾驶占比。

### 2.5.2 暴露量

- 窗口总里程；
- 窗口总运行小时；
- 活跃驾驶天数；
- 轨迹覆盖天数；
- IMU覆盖天数；
- 移动轨迹点数；
- 平均每日里程和运行时长。

### 2.5.3 事件次数

对各事件类型和风险维度生成：

- 总次数；
- 有事件天数；
- 最大单日次数；
- 最近7、14、30天次数；
- 指数衰减次数。

指数衰减：

```text
weight(age_days) = 2 ^ (-age_days / 14)
decayed_count = Σ weight(age_days)
```

### 2.5.4 暴露归一化

```text
rate_per_1000km = event_count / (distance_km / 1000)
rate_per_100h = event_count / (run_time_hours / 100)
rate_per_active_day = event_count / active_days
```

分母为0时保留缺失，不转换为0。

低暴露车辆使用经验贝叶斯收缩：

```text
smoothed_rate = (n + τ × μ) / (exposure + τ)
```

其中：

- `n`：车辆事件数；
- `exposure`：千公里或百小时；
- `μ`：训练样本全体平均风险率；
- `τ`：先验暴露强度，只在训练窗口估计。

### 2.5.5 近期性和趋势

- 距最近一次事故的天数；
- 距最近一次未遂事故的天数；
- 距最近一次疲劳、分心、稳定性事件的天数；
- 最近窗口相对上一窗口的变化。

趋势：

```text
trend = log((recent_rate + ε) / (previous_rate + ε))
```

### 2.5.6 聚集性

- 最大单日风险事件数；
- 风险事件活跃天数占比；
- 最长连续风险天数；
- 同日出现三类及以上风险的天数；
- 事件小时集中度；
- 高危事件前后短时间内的预警数量。

### 2.5.7 轨迹特征

- 平均速度和最大速度；
- 速度标准差；
- 移动点比例；
- 每日里程波动；
- 每日运行时长波动；
- 无效坐标比例；
- 大距离跳变比例；
- 轨迹覆盖率。

### 2.5.8 IMU日级特征

- IMU覆盖率；
- EMS速度缺失率；
- GPS速度缺失率；
- EMS与GPS速度差均值；
- 各加速度轴标准差和绝对最大值；
- 各角速度轴标准差和绝对最大值；
- 加速度模长均值、标准差和最大值；
- 角速度模长均值、标准差和最大值；
- `gyro_norm > 20`候选比例；
- 加速度模长候选异常比例。

IMU缺失只作为覆盖率或可信度，不直接解释为低风险。

### 2.5.9 数据质量特征

- 轨迹有效日期比例；
- IMU有效日期比例；
- 速度有效比例；
- 坐标有效比例；
- 车辆是否有完整21天或61天数据。

## 2.6 特征控制

- 第一版正式模型特征控制在100个左右；
- 删除常量列；
- 删除完全重复列；
- 对相关系数过高且含义重复的特征只保留一个；
- 所有计数型特征同时保留原计数和`log1p`版本时，需要通过验证筛选，避免重复；
- 特征筛选只能使用训练时间窗口；
- 特征名称必须体现维度和窗口，例如`fatigue_rate_14d_per_100h`。

## 2.7 固定交付接口

```text
outputs/features/
├─ risk_event_daily.parquet
├─ train_windows.parquet
├─ full_61d_features.parquet
├─ feature_dictionary.csv
├─ label_report.json
└─ feature_quality.json
```

`feature_dictionary.csv`至少包含：

```text
feature_name
feature_group
description
unit
window_days
numerator
denominator
missing_policy
```

验收标准：

- `full_61d_features.parquet`恰好500行；
- `gpsno`唯一且与画像完全一致；
- 训练样本主键无重复；
- 标签期没有进入特征；
- 每个比例和风险率有明确分母；
- 没有全空列、常量列或无限值；
- 所有窗口边界均可追溯。

---

# 3. 队员B：事故预测

## 3.1 责任边界

队员B负责：

- 训练和验证框架；
- 逻辑回归基线；
- 浅层梯度提升树主模型；
- 特征族消融；
- 稳定模型融合；
- 最终提交文件。

队员B不修改A的特征定义，也不修改C的评分规则。

独占目录：

```text
src/prediction/
tests/prediction/
outputs/prediction/
runs/member_b/
```

## 3.2 代码结构

```text
src/prediction/
├─ data_loader.py
├─ splitters.py
├─ preprocessing.py
├─ train_logistic.py
├─ train_boosting.py
├─ ensemble.py
├─ evaluate.py
├─ train_final.py
└─ validate_submission.py
```

## 3.3 训练数据

正式输入：

```text
outputs/features/train_windows.parquet
outputs/features/full_61d_features.parquet
outputs/features/feature_dictionary.csv
```

训练样本单位为：

```text
一辆车 × 一个特征窗口 × 一个未来标签窗口
```

## 3.4 时间验证

验证遵循：

1. 主验证使用21天特征预测后续40天；
2. 辅助验证使用A提供的滚动时间窗口；
3. 训练窗口早于验证窗口；
4. 不随机拆分原始事件；
5. 预处理和特征选择只在训练部分拟合；
6. 每个窗口分别报告AUC和PR-AUC；
7. 模型选择同时考虑平均AUC和最差窗口AUC。

报告字段：

```text
model_name
window_id
train_positive
train_negative
valid_positive
valid_negative
roc_auc
pr_auc
log_loss
probability_min
probability_max
probability_mean
```

## 3.5 预处理

### 数值变量

- 缺失值使用训练集中的中位数；
- 逻辑回归前标准化；
- 树模型保留原始数值尺度；
- 极端值截尾阈值只在训练集估计；
- 数据质量和覆盖率缺失保留单独指示变量。

### 类别变量

- 能源类型独热编码；
- 未知类别使用统一`UNKNOWN`；
- 不把`gpsno`作为数值或类别特征。

### 禁止操作

- 不对车辆级样本直接做SMOTE；
- 不使用标签期计算的均值、分位数或编码；
- 不使用外部天气、地图、节假日或司机信息；
- 不用全体数据先筛选特征再验证。

## 3.6 逻辑回归基线

采用带正则化的逻辑回归：

- L2作为默认；
- 使用类别权重；
- 通过时间验证选择正则强度；
- 输出标准化后的系数；
- 检查重要特征方向是否符合常识。

逻辑回归作用：

- 建立稳定基线；
- 检查数据泄漏；
- 识别方向异常的特征；
- 为最终融合提供低方差模型。

## 3.7 浅层梯度提升树

主树模型固定使用LightGBM。

约束原则：

- 树深较小；
- 叶子数较少；
- 较大的最小叶节点样本数；
- 较低学习率；
- 早停；
- 类别权重；
- 行和列采样防止过拟合。

参数搜索范围：

```text
max_depth: 2～5
num_leaves: 4～24
learning_rate: 0.01～0.08
min_child_samples: 15～80
subsample: 0.6～1.0
colsample_bytree: 0.5～1.0
reg_alpha: 0～5
reg_lambda: 1～20
```

参数选择必须使用时间窗口结果，不能只看单一切分。

## 3.8 特征族消融

按以下顺序比较：

1. 画像；
2. 画像+风险事件；
3. 加入暴露归一化；
4. 加入近期性和趋势；
5. 加入轨迹；
6. 加入IMU；
7. 全部特征。

特征族保留条件：

- 多个时间窗口平均AUC提高；
- 最差窗口不明显恶化；
- 没有泄漏；
- 重要性方向合理；
- 增益能稳定复现。

## 3.9 模型融合

最终使用逻辑回归和梯度提升树的排名融合：

```text
rank_score = w × percentile_rank(tree_probability)
           + (1-w) × percentile_rank(logistic_probability)
```

融合权重只从以下候选中选择：

```text
w ∈ {0.25, 0.50, 0.75}
```

启用融合的必要条件：

- 至少两个时间窗口中不差于最佳单模型；
- 最差窗口优于或等于最佳单模型的最差窗口；
- 权重轻微变化不会导致大幅排名变化。

若不满足条件，最终使用时间验证更稳定的单模型。

## 3.10 概率和二分类阈值

- AUC模型选择使用连续概率或排序分；
- `risk_prob`必须保留连续信息；
- 概率校准使用训练外预测进行Platt scaling；
- 二分类`accident`阈值根据验证集Youden指数和业务召回率共同确定；
- 阈值选择与AUC模型选择分开；
- 检查是否出现大量0或1极端概率。

## 3.11 最终训练与提交

1. 使用选定特征、预处理器和参数；
2. 使用所有允许的历史训练窗口；
3. 对`full_61d_features.parquet`生成500辆车概率；
4. 生成二分类结果；
5. 与模板字段对齐；
6. 运行提交校验器。

提交格式：

```text
gpsno,accident,risk_prob
```

验收标准：

- 恰好500行；
- `gpsno`唯一且与画像完全一致；
- `accident`只包含0和1；
- `risk_prob`位于0～1；
- 无空值、无无限值；
- 模型、预处理器、特征顺序和参数已保存；
- 固定随机种子后结果可复现。

固定交付：

```text
outputs/prediction/
├─ validation_metrics.json
├─ oof_predictions.parquet
├─ feature_importance.csv
├─ ablation_results.csv
├─ model/
├─ forecast_result.csv
└─ prediction_method.md
```

---

# 4. 队员C：安全评分与运营方案

## 4.1 责任边界

队员C负责：

- 风险维度与事件映射；
- 100分制评分；
- 权重数据校准；
- 排名和风险等级；
- 评分可信度；
- 扣分原因；
- 运营动作；
- 评分区分度验证。

独占目录：

```text
src/scoring/
tests/scoring/
outputs/scoring/
runs/member_c/
```

## 4.2 代码结构

```text
src/scoring/
├─ event_taxonomy.py
├─ scoring_rules.py
├─ exposure_adjustment.py
├─ calibrate_weights.py
├─ calculate_scores.py
├─ reason_codes.py
├─ validate_scores.py
└─ export_scores.py
```

## 4.3 评分维度

| 维度 | 初始权重 | 数据来源 |
|---|---:|---|
| 历史事故与未遂 | 35% | 11803、11804 |
| 疲劳与分心 | 20% | 闭眼、困倦、哈欠、低头、电话、手机等 |
| 驾驶稳定性 | 20% | 车道偏移、压线、车距、碰撞预警 |
| 道路规则风险 | 10% | 路口、弯道、匝道、长下坡超速 |
| 管理配合度 | 10% | 摄像头遮挡、角度异常 |
| IMU风险佐证 | 5% | 模长波动、极端角速度、速度源差异 |

权重约束：

- 历史事故与未遂始终是最高权重；
- 疲劳分心与稳定性不得低于道路规则风险；
- IMU权重不超过10%；
- 权重总和为100%；
- 权重只能在约束范围内根据时间标签校准。

## 4.4 基础风险率

每个维度先计算暴露归一化风险率：

```text
rate_per_1000km = weighted_event_count / (distance_km / 1000)
rate_per_100h = weighted_event_count / (run_time_hours / 100)
rate_per_day = weighted_event_count / active_days
```

事件严重度由`scoring_rules.csv`配置，不能散落在多个脚本中。

初始事件严重度层级：

| 层级 | 示例 | 相对严重度 |
|---|---|---:|
| 极高 | 事故 | 10 |
| 很高 | 未遂事故 | 7 |
| 高 | 闭眼、困倦、前碰撞预警 | 4～5 |
| 中 | 打电话、看手机、车距过近、长时间压线 | 2～3 |
| 低 | 普通车道偏移、管理异常 | 1～2 |

相对严重度用于维度内部聚合，不直接等于最终扣分。

## 4.5 时间衰减

近期事件权重更高：

```text
weight(age_days) = 2 ^ (-age_days / half_life)
```

动态评分使用统一14天半衰期，月度报告可以额外展示30天统计，但最终主分保持同一公式，避免多个口径。

## 4.6 低暴露车辆收缩

```text
smoothed_rate = (n + τ × μ) / (exposure + τ)
```

- `n`为时间衰减后的严重度加权事件数；
- `exposure`为千公里、百小时或活跃日；
- `μ`为全体平均风险率；
- `τ`为先验暴露强度。

这样可以防止低里程车辆因为偶然零事件获得满分。

## 4.7 维度惩罚

对平滑风险率：

1. 使用`log1p`压缩极端值；
2. 按训练期分位数映射为0～100惩罚；
3. 检查惩罚与未来事故/未遂率是否单调；
4. 对同维度高相关事件设置扣分上限；
5. 分位点只能从训练期估计。

公式：

```text
dimension_penalty_k = monotonic_percentile(smoothed_rate_k)
safety_score = 100 - Σ(weight_k × dimension_penalty_k / 100)
```

最终分数限制在0～100。

## 4.8 权重数据校准

使用A的时间标签验证六个维度与未来风险的关系：

1. 检查单维度AUC和分箱坏样本率；
2. 使用带正则逻辑回归估计方向和相对强度；
3. 保持权重约束，只做有限调整；
4. 使用多个时间窗口检查权重方向；
5. 选择在各窗口下风险分层最稳定的权重；
6. 保留公式和权重变更记录。

权重校准不能把评分变成无法解释的黑箱模型。

## 4.9 IMU处理

- IMU缺失不扣风险分；
- IMU覆盖率单独进入评分可信度；
- IMU只作为风险佐证维度；
- 极端角速度不能直接解释为事故；
- IMU维度扣分设置较低上限；
- 无IMU车辆与有IMU车辆的非IMU五个维度计算完全一致。

## 4.10 评分可信度

可信度由以下信息计算：

- 轨迹覆盖天数；
- IMU覆盖天数；
- 有效速度比例；
- 有效坐标比例；
- 总里程和运行时长；
- 风险事件记录覆盖。

评分可信度输出：

```text
confidence_score: 0～100
confidence_level: high / medium / low
```

可信度只说明数据是否充分，不改变风险方向。

## 4.11 扣分原因

每辆车输出前三项原因：

```text
reason_code
reason_text
dimension
penalty_points
event_count
normalized_rate
comparison_percentile
```

原因示例：

- 近14天闭眼和困倦事件率位于全体前5%；
- 历史存在未遂事故且距今较近；
- 车距过近与前碰撞预警同时偏高；
- 摄像头遮挡持续多日；
- 评分可信度较低：IMU覆盖不足。

## 4.12 风险等级与运营动作

初始使用相对分位数：

| 等级 | 范围 | 运营动作 |
|---|---|---|
| A | 安全分前20% | 正向激励、常规观察 |
| B | 20%～60% | 月度提醒和复盘 |
| C | 60%～90% | 针对性培训、每周跟踪 |
| D | 后10% | 重点干预、事件复核和管理闭环 |

每个等级同时给出：

- 主要风险维度；
- 培训主题；
- 复核周期；
- 是否需要人工检查设备；
- 下周期改善目标。

## 4.13 评分验证

必须验证：

1. A到D未来事故/未遂率单调上升；
2. 最低20%与最高20%的未来风险差明显；
3. 各能源类型内仍有区分度；
4. 低IMU覆盖车辆不会系统性获得更高分；
5. 删除历史事故维度后，其余行为维度仍有区分能力；
6. 小幅改变权重时车辆排名不会大幅跳动；
7. 每个扣分原因可以回溯到原始事件或清洗统计。

固定交付：

```text
outputs/scoring/
├─ safety_scores.csv
├─ score_details.parquet
├─ scoring_rules.csv
├─ score_validation.json
└─ scoring_method.md
```

`safety_scores.csv`至少包含：

```text
gpsno
safety_score
rank
risk_level
confidence_score
confidence_level
top_reason_1
top_reason_2
top_reason_3
```

验收标准：

- 恰好500辆车；
- `gpsno`唯一；
- 安全分位于0～100；
- 分数和排名方向一致；
- 高危历史不会获得高安全分；
- IMU缺失不会被误判为安全；
- 风险等级与后续事故/未遂具有单调关系；
- 扣分原因可解释、可运营。

---

# 5. 三人接口与并行规则

## 5.1 A向B交付

```text
outputs/features/train_windows.parquet
outputs/features/full_61d_features.parquet
outputs/features/feature_dictionary.csv
outputs/features/label_report.json
```

## 5.2 A向C交付

```text
outputs/features/risk_event_daily.parquet
outputs/features/feature_dictionary.csv
outputs/features/label_report.json
```

## 5.3 B向最终集成交付

```text
outputs/prediction/forecast_result.csv
outputs/prediction/validation_metrics.json
outputs/prediction/prediction_method.md
```

## 5.4 C向最终集成交付

```text
outputs/scoring/safety_scores.csv
outputs/scoring/score_details.parquet
outputs/scoring/score_validation.json
outputs/scoring/scoring_method.md
```

## 5.5 并行规则

- A不修改`src/prediction/`和`src/scoring/`；
- B不修改`src/features/`和`src/scoring/`；
- C不修改`src/features/`和`src/prediction/`；
- 公共清洗层保持只读；
- 不直接调用其他成员未稳定的内部函数；
- 只通过固定输出文件交付；
- 接口字段发生变化时同步更新特征字典和README；
- 每次实验保存配置、随机种子、特征清单和指标；
- 正式产物记录来源实验编号。

# 6. 统一质量门槛

新特征、模型或评分规则进入正式结果前必须满足：

1. 无未来信息泄漏；
2. 多个时间窗口下方向一致；
3. 可以重复运行；
4. 车辆覆盖完整；
5. 缺失和异常处理明确；
6. 计算公式和字段单位可追溯；
7. 不使用外部数据；
8. 能在最终文档中解释。
