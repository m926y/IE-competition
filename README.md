# 2026 Agent 大赛：驾驶安全

本项目面向 Crazy Logi-AI 2.0「驾驶安全」赛题，使用 500 辆货车在 2026 年 6 月 1 日至 7 月 31 日共 61 天内的车辆画像、风险事件、轨迹和 IMU 数据，完成：

1. 任务一：预测每辆车未来 40 天发生事故或未遂事故的概率；
2. 任务二：建立可解释、可运营、支持动态更新的司机安全评分体系。

数据主键实际为车辆设备号 `gpsno`。原始文件没有独立司机 ID，因此本文中的“司机评分”技术上是车辆/设备关联的驾驶风险评分，最终报告需要明确这一限制。

## 1. 当前状态

已经完成：

- 30 个轨迹分片和 30 个 IMU 分片完整性检查；
- 约 13.26 亿条轨迹与 IMU 数据的流式清洗和日级聚合；
- 风险事件逐条清洗；
- 车辆画像标准化；
- 轨迹日表与 IMU 日表合并；
- 质量报告、清洗代码和团队协作结构；
- A部分时间标签、风险日表和多窗口特征流水线（`member-a-v1.0`）；
- 17个时间窗口、8,500行训练样本、500辆车最终特征和100维首版核心特征；
- A部分训练/验证/测试划分清单、15项自动化测试及独立产物验收报告。

尚未完成：

- 任务一预测模型；
- 任务二安全评分结果；
- 最终提交 CSV 和说明 PDF。

当前没有统一的 ZIP、7Z 或 RAR 项目压缩包。原始数据仍以 `.gz` 保存；清洗数据是独立的 Parquet、CSV 和 JSON 文件。Parquet 内部已经使用 ZSTD 压缩，不需要先解压才能读取。

## 2. 项目目录

```text
IE亮剑/
├─ README.md
├─ requirements.txt
├─ .gitignore
├─ .deps/                              # 项目本地依赖，不提交
│
├─ config/                             # 后续公共配置
│
├─ data/
│  ├─ raw/                             # 原始数据，只读
│  │  ├─ vehicle_profile.xlsx
│  │  ├─ risk_events.tsv.gz
│  │  ├─ trajectory/                   # 30个轨迹分片
│  │  └─ imu/                          # 30个IMU分片
│  ├─ templates/
│  │  └─ forecast_result_template.csv
│  └─ processed/
│     └─ cleaned/                      # 已完成的公共清洗层，只读
│        ├─ vehicle_profile.csv
│        ├─ vehicle_profile.parquet
│        ├─ risk_events.parquet
│        ├─ trajectory_daily.parquet
│        ├─ imu_daily.parquet
│        ├─ vehicle_day.parquet
│        └─ quality_report.json
│
├─ docs/
│  ├─ competition/
│  │  ├─ competition_rules.pdf
│  │  ├─ dataset_description.pdf
│  │  └─ imu_risk_reference.pdf
│  ├─ data_cleaning_and_modeling_route.md
│  └─ team_technical_routes.md          # 三人统一技术路线
│
├─ src/
│  ├─ data/
│  │  └─ clean_data.py                 # 清洗入口
│  ├─ common/                          # 稳定公共接口，修改需协调
│  ├─ features/                        # 队员A独占
│  ├─ prediction/                      # 队员B独占
│  └─ scoring/                         # 队员C独占
│
├─ tests/
│  ├─ features/
│  ├─ prediction/
│  └─ scoring/
│
├─ outputs/
│  ├─ features/                        # 队员A产物
│  ├─ prediction/                      # 队员B产物
│  ├─ scoring/                         # 队员C产物
│  └─ final/                           # 最终集成产物
│
└─ runs/
   ├─ member_a/                        # A的实验记录
   ├─ member_b/                        # B的实验记录
   └─ member_c/                        # C的实验记录
```

目录规则：

- `data/raw/`：原始比赛数据，任何成员不得编辑、重命名或覆盖；
- `data/processed/cleaned/`：公共清洗层，只能由清洗脚本生成，禁止手工改表；
- `src/features/`、`src/prediction/`、`src/scoring/`：分别由三名成员独占；
- `outputs/`：正式阶段产物；
- `runs/`：个人实验过程，只删除本人目录下的内容；
- `src/common/`、`config/`和README：共享区域，修改前需要在小组内说明。

## 3. 原始数据与清洗结果

| 数据 | 原始记录数 | 车辆数 | 清洗后形式 | 主要质量结果 |
|---|---:|---:|---|---|
| 车辆画像 | 500 | 500 | 每车一行 | 车辆号无重复、无缺失 |
| 风险事件 | 3,152,654 | 500 | 逐事件 Parquet | 5条速度异常、767条无效坐标 |
| 轨迹 | 584,310,388 | 500 | 车辆-日期聚合 | 115条速度异常、2,569条无效坐标 |
| IMU | 738,244,484 | 500 | 车辆-日期聚合 | EMS速度缺失约41.2%，GPS速度缺失约10.7% |
| 车辆日表 | 28,847 | 500 | 轨迹与IMU合并 | `gpsno + event_date` 无重复 |

所有明细数据实际覆盖左闭右开区间：

```text
[2026-06-01 00:00:00, 2026-08-01 00:00:00)
```

即完整61天。

## 4. 数据清洗采用的具体策略

清洗入口为`src/data/clean_data.py`。脚本使用DuckDB直接读取gzip，不先生成大型TSV明文文件。

### 4.1 总体策略

1. **流式读取压缩数据**
   - 直接读取`.gz`；
   - 避免产生数十到上百GB的临时明文；
   - 清洗产物使用ZSTD压缩Parquet。

2. **防御性类型转换**
   - 原始TSV先按字符串读取；
   - 使用安全转换将时间、整数和浮点数转成目标类型；
   - 无法转换的值变为缺失值或质量标记，避免整批任务失败。

3. **固定时间边界**
   - 只接受2026-06-01至2026-07-31的记录；
   - 使用左闭右开区间，避免截止日边界错误；
   - 日表日期统一从有效时间戳派生。

4. **标识符统一**
   - `gpsno`和`imei`按字符串处理；
   - 去除前后空格；
   - 不把设备号当作连续数值特征。

5. **异常值不静默删除**
   - 关键异常保留原始值；
   - 同时生成清洗值和异常标记；
   - 对轨迹和IMU聚合记录异常数量，供后续置信度评估。

6. **不在清洗阶段插值**
   - 不人为补造轨迹点、风险事件或IMU信号；
   - 缺失值、覆盖率和可用天数作为后续模型特征；
   - 填补只在模型管道内部完成，并且只能在训练折拟合。

7. **控制清洗粒度**
   - 风险事件保留逐条记录；
   - 轨迹和IMU默认生成车辆-日期聚合；
   - 不默认物化13亿多条逐点Parquet，防止磁盘不足。

### 4.2 车辆画像

输入：`data/raw/vehicle_profile.xlsx`

处理：

- 跳过表头后读取500辆车；
- `gpsno`转为字符串；
- 能源类型保留为分类文本；
- 里程、时长、停留次数和比例转为数值；
- 检查车辆号重复和空值；
- 同时输出CSV和Parquet，方便人工查看与建模读取。

### 4.3 风险事件

输入字段：

```text
gpsno, lat, lng, event_type, event_name, speed, start_time
```

处理规则：

- `event_type`转为整数并与24个已知事件编码核对；
- `event_name`仅用于展示，统计和分组以编码为准；
- 速度原值保存在`speed_raw`；
- 仅将`0 <= speed <= 160 km/h`写入`speed_kmh`；
- 超出范围的记录保留，但设置`speed_outlier=true`；
- 经纬度原值保存在`lat_raw/lng_raw`；
- 只将大致中国范围`纬度3～54、经度73～136`写入清洗坐标；
- 无效坐标保留记录，并设置`coordinate_valid=false`；
- 时间戳必须位于61天观察期；
- 保留事故`11803`和未遂事故`11804`，供后续标签构造。

风险事件仍保持一事件一行，便于按事件类型汇总、构造时间窗口、解释评分扣分和检查标签泄漏。

### 4.4 轨迹

输入字段：

```text
gpsno, distance, run_time, trigger_time, speed, course, lat, lng
```

处理规则：

- `distance`按厘米读取，日汇总除以100000转换为公里；
- 负距离不进入里程汇总，并计入`invalid_distance_rows`；
- 单点距离大于500000厘米，即5公里，标记为`large_distance_gap_rows`，暂不直接删除；
- `run_time`按秒读取，负值不进入运行时长汇总；
- 运行时长除以3600转换为小时；
- 速度只接受0～160km/h；
- 航向角只接受0～360度；
- 经纬度使用与风险事件相同的范围检查；
- 最终按`gpsno + event_date`聚合。

日级输出包括原始点数、速度统计、移动点数、坐标质量、航向异常数、总里程、运行小时和大距离跳变数。

大距离跳变可能来自长时间不上报，也可能是定位跳点，因此清洗阶段只标记，不武断删除。

### 4.5 IMU

输入字段：

```text
gpsno, imei, data_time, data_date,
ems_speed, gps_speed,
accel_x_raw, accel_y_raw, accel_z_raw,
gyro_x_raw, gyro_y_raw, gyro_z_raw
```

处理规则：

- 校验`data_time`日期与`data_date`是否一致；
- 分别统计EMS速度和GPS速度缺失；
- 速度统计只使用0～160km/h范围；
- 两类速度同时有效时计算绝对差；
- 不假定设备X/Y/Z轴天然对应车辆前向、横向和垂直方向；
- 保留各轴均值、标准差和绝对最大值；
- 计算加速度模长和角速度模长；
- `accel_norm`不在0.5～1.5范围时只记为候选异常；
- `gyro_norm > 20`只记为强转向或冲击候选，不能直接当事故标签；
- 最终按`gpsno + event_date`聚合。

两个重要原则：

1. IMU缺失不等于安全；
2. 极端IMU值不等于事故，需要结合速度、轨迹和风险事件验证。

### 4.6 车辆日表

`trajectory_daily.parquet`与`imu_daily.parquet`使用`gpsno + event_date`全外连接，得到`vehicle_day.parquet`。

该连接会：

- 保留只有轨迹、没有IMU的日期；
- 保留只有IMU、没有轨迹的日期；
- 明确区分0和缺失；
- 支持计算车辆61天内的轨迹覆盖率和IMU覆盖率。

### 4.7 质量门槛

清洗后至少检查：

- 画像是否恰好500辆车；
- 风险、轨迹和IMU的车辆覆盖；
- 日期是否覆盖6月1日至7月31日；
- `gpsno + event_date`是否重复；
- 风险事件编码是否都在已知清单；
- 速度、坐标、距离和IMU异常数量；
- 画像与清洗表之间是否有无法匹配的车辆号。

结果保存在`data/processed/cleaned/quality_report.json`。

## 5. 环境与运行

安装依赖：

```powershell
python -m pip install --target .deps -r requirements.txt
```

完整重跑：

```powershell
python src/data/clean_data.py --overwrite
```

按阶段运行：

```powershell
python src/data/clean_data.py --stages profile risk --overwrite
python src/data/clean_data.py --stages trajectory imu vehicle_day --overwrite
```

冒烟测试：

```powershell
python src/data/clean_data.py `
  --output-dir outputs/cleaned_smoke `
  --sample-files 1 `
  --limit 200000 `
  --overwrite
```

默认输出为`data/processed/cleaned/`。

如确需物化逐点轨迹和IMU：

```powershell
python src/data/clean_data.py --stages trajectory imu --row-level --overwrite
```

逐点物化会消耗大量磁盘，项目主线统一使用日级聚合数据；只有逐点复核任务才执行该命令。

## 6. 三人并行分工

各成员的统一技术路线、实现步骤和验收标准见：

```text
docs/team_technical_routes.md
```

项目主线固定为：A构建多窗口时间标签与增强特征；B使用正则逻辑回归和浅层梯度提升树完成事故预测，仅在时间验证证明稳定时进行排名融合；C使用数据校准且公式透明的安全评分。三名成员按照固定文件接口交付，内部实现互不依赖。

| 队员 | 主任务 | 独占源码目录 | 独占输出目录 |
|---|---|---|---|
| A | 时间标签和特征工程 | `src/features/` | `outputs/features/` |
| B | 事故预测和提交文件 | `src/prediction/` | `outputs/prediction/` |
| C | 安全评分和运营方案 | `src/scoring/` | `outputs/scoring/` |

三条路线可以同时启动：

- A基于公共清洗层生产正式特征；
- B先用现有画像、风险事件和车辆日表搭建训练框架；
- C直接用现有清洗层构建评分，不等待B的预测模型。

### 6.1 队员A：标签与特征

主要任务：

1. 生成`risk_event_daily.parquet`；
2. 构造21天特征预测后续40天的主验证样本；
3. 构造14/21/30天特征预测未来14天的辅助窗口；
4. 生成次数、风险率、覆盖率、趋势、聚集性和近期性特征；
5. 输出完整61天的最终预测特征。

固定交付：

```text
outputs/features/
├─ risk_event_daily.parquet
├─ train_windows.parquet
├─ full_61d_features.parquet
├─ feature_dictionary.csv
├─ label_report.json
└─ feature_quality.json
```

要求：

- `full_61d_features.parquet`恰好500行；
- `gpsno`唯一；
- 每个风险率都有明确分母；
- 标签期信息不得进入特征；
- 所有窗口保留起止日期。

#### 6.1.1 已实现的处理路径与代码位置

A部分已于2026年9月22日完成全量运行和验收。处理流程为：

```text
data/processed/cleaned/（只读）
    → 输入字段、主键、车辆集合与SHA256核对
    → 风险事件日表（24类事件、7个风险维度、完整车辆日历）
    → 主40天标签、滚动14天标签及额外验证窗口
    → 795个候选特征（事件、暴露量、近期性、趋势、聚集性、轨迹、IMU等）
    → 仅用最早训练窗口筛选100维核心特征
    → Parquet交付、特征字典、时间划分及质量报告
    → 独立回读验收与自动化测试
```

| 代码路径 | 职责 |
|---|---|
| [src/features/pipeline.py](src/features/pipeline.py) | 统一运行入口，串联计算、导出和校验 |
| [src/features/event_groups.py](src/features/event_groups.py) | 24类事件编码与7个风险维度映射 |
| [src/features/window_definitions.py](src/features/window_definitions.py) | 时间窗口及训练/验证/测试划分 |
| [src/features/build_risk_daily.py](src/features/build_risk_daily.py) | 风险日表、历史高危事件前后5分钟预警配对计数 |
| [src/features/build_labels.py](src/features/build_labels.py) | 事故11803或未遂11804的未来窗口标签 |
| [src/features/build_window_features.py](src/features/build_window_features.py) | 各特征窗口的统一计算逻辑，包含轨迹及IMU |
| [src/features/build_full_features.py](src/features/build_full_features.py) | 7月31日截止的完整61天及近期特征 |
| [src/features/feature_contract.py](src/features/feature_contract.py) | 字段类型、特征字典、版本和禁止入模字段 |
| [src/features/fold_transforms.py](src/features/fold_transforms.py) | 训练折内特征筛选及暴露量收缩的fit/transform接口 |
| [src/features/validate_features.py](src/features/validate_features.py) | 输入、输出、日期、标签、分母及字段一致性检查 |
| [src/features/audit_delivery.py](src/features/audit_delivery.py) | 回读已导出文件，独立核验并生成验收报告 |
| [src/features/runtime.py](src/features/runtime.py) | 本地依赖加载、Parquet/CSV/JSON读写及文件指纹 |
| [tests/features/test_features.py](tests/features/test_features.py) | 时间边界、防泄漏、缺失处理和统计公式测试 |

详细口径及B/C对接说明见 [src/features/README.md](src/features/README.md) 和交付副本 [outputs/features/IMPLEMENTATION.md](outputs/features/IMPLEMENTATION.md)。

#### 6.1.2 运行与复现

以下命令均在包含本README、`src/`和`requirements.txt`的项目根目录执行：

```powershell
# 首次生成；默认读取 data/processed/cleaned，写入 outputs/features
python -m src.features.pipeline

# 已有完整产物时，显式重跑
python -m src.features.pipeline --overwrite

# 单独运行测试
python -m unittest discover -s tests/features -v

# 回读交付数据、运行测试并更新 acceptance_report.json
python -m src.features.audit_delivery
```

本次运行使用Python 3.12、DuckDB 1.5.5。当前`.deps`包含Python 3.12的Windows二进制依赖，不应直接由系统其他版本的Python加载。本机可使用已验证的解释器运行：

```powershell
$pythonA = Join-Path $env:USERPROFILE '.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe'
& $pythonA -m src.features.pipeline --overwrite
& $pythonA -m src.features.audit_delivery
```

其他机器按`requirements.txt`配置匹配的Python环境即可；不要跨Python版本直接复制`.deps`。支持`--cleaned-dir`指定清洗目录，`--output-dir`限定在A独占的`outputs/features/`范围内。

本次正式流水线运行记录保存在 [runs/member_a/20260922_110414](runs/member_a/20260922_110414/)，其中`run_config.json`记录窗口、随机种子、依赖版本及输入/源码SHA256，`run_result.json`记录该次流水线完成时的结果和文件指纹。独立验收结果以随后生成的`acceptance_report.json`为准。每次重跑会创建新的运行记录目录。

#### 6.1.3 已交付数据及使用方式

GitHub数据交付：[member-a-v1.0 Release](https://github.com/m926y/IE-competition/releases/tag/member-a-v1.0)。下载附件 `member-a-features-v1.zip`，在项目根目录解压，恢复 `outputs/features/` 和本次 `runs/member_a/20260922_110414/`。下表中的本地数据链接在解压后有效；数据未提交到Git历史。仅使用已计算特征无需原始数据，重新运行流水线仍需另行准备公共清洗层。

| 产物（均位于outputs/features） | 本次结果 / 用途 |
|---|---|
| [risk_event_daily.parquet](outputs/features/risk_event_daily.parquet) | 30,500行，即500辆车×61天；主键`gpsno + event_date` |
| [train_windows.parquet](outputs/features/train_windows.parquet) | 8,500行，即17个窗口×500辆车；795个候选特征，另含元数据及标签 |
| [full_61d_features.parquet](outputs/features/full_61d_features.parquet) | 500行，包含完整61天和固定近期候选特征，无实际未来标签 |
| [feature_dictionary.csv](outputs/features/feature_dictionary.csv) | 特征来源、单位、窗口、分子、分母、缺失规则和`enabled_v1` |
| [label_report.json](outputs/features/label_report.json) | 各窗口起止日期、正负类数量、事故/未遂标签和预测期限说明 |
| [feature_quality.json](outputs/features/feature_quality.json) | 字段质量、输入检查、前后指纹一致性及自动验收结果 |
| [train_model_core.parquet](outputs/features/train_model_core.parquet) | 8,500行；100个核心特征，另含元数据和`label`，供B按划分清单取训练/验证/测试行 |
| [forecast_model_core.parquet](outputs/features/forecast_model_core.parquet) | 500行；与训练核心表含义一致的100个核心特征及元数据 |
| [core_model_features.json](outputs/features/core_model_features.json) | 有序入模列清单、筛选所用训练窗口及排除原因 |
| [split_manifest.csv](outputs/features/split_manifest.csv) | 按实验阶段指定训练、验证和测试窗口 |
| [acceptance_report.json](outputs/features/acceptance_report.json) | 最终独立验收结果、逐项测试日志及产物SHA256 |

候选表保留全部795个特征，供审计、特征族消融和C的评分聚合使用，其中允许出现稀有事件常量列或短窗口下不可计算的长周期列。**“无训练全空、常量、重复或高度相关列”的筛选约束适用于100维正式模型视图，不能理解为795列候选表均可直接入模。**

B应读取`core_model_features.json`的`columns`作为有序特征列表，按`split_manifest.csv`的`phase`、`split_role`和`window_id`筛选样本。`gpsno`、日期、窗口元数据、`label`及其他标签字段不得进入特征矩阵；不能把整个训练表的数值列直接作为模型输入。

核心特征只在6月14日截止的最早训练窗口上筛选：删除全空、常量和完全重复列，剔除绝对相关系数不小于0.995的冗余列，并按预设特征族轮转顺序最多保留100列。没有使用验证、测试或最终预测数据的分布选列。本次100列包括：暴露量3、数据质量6、轨迹5、事件维度28、事件类型24、聚集性7、IMU 13、近期次数7、趋势7。

#### 6.1.4 时间窗口和训练/验证/测试划分

标签定义：标签期内出现`event_type IN (11803,11804)`则为1，否则为0；日期范围按日展示，SQL使用左闭右开时间区间。

- 主窗口：6月1—21日特征，6月22日—7月31日40天标签；500辆车中正类128辆、负类372辆。
- 常规辅助窗口：过去14/21/30天预测后续14天，按7天步长滚动并增加7月17日终端截止窗口，共15个。
- 额外验证窗口：6月17—30日特征，7月1—14日标签；用于避免验证与最终本地测试的标签期重叠。
- 合计1个主窗口、15个辅助窗口、1个额外验证窗口，共17个、8,500行；窗口样本重复使用同一批500辆车，不能视作8,500辆独立车辆。

| 阶段 | 样本角色 | 特征期 | 标签期 | 行数 |
|---|---|---|---|---:|
| `model_selection` | 训练 | 6月1—14日 | 6月15—28日 | 500 |
| `model_selection` | 验证 | 6月17—30日 | 7月1—14日 | 500 |
| `locked_test` | 冻结参数后重训 | 截止6月14日、21日、28日、30日的4个14天窗口 | 各自后续14天，均在7月17日前已可知 | 2,000 |
| `locked_test` | 本地测试 | 7月4—17日 | 7月18—31日 | 500 |

这些是同一训练表在不同实验阶段的选取方式，不是可相加的新车辆集合。允许同一车辆跨时间出现在训练和测试中；禁止随机拆分窗口行。测试阶段只能在特征、超参数和阈值冻结后运行，不得根据最后测试窗口的结果反复调参。

主21天→40天窗口已按要求交付，但现有61天历史不足以在6月21日之前获得完整的14天特征＋14天标签训练样本，因此该主窗口标记为`main_40d`诊断样本，不纳入上述严格时间划分。最终比赛预测期为8月1日至9月9日40天，当前无真值；14天模型概率不能直接称为已校准的40天概率，期限迁移由B处理并记录。

#### 6.1.5 已执行的防泄漏与计算限制

1. 特征只读取对应`feature_start`至`feature_end`的数据，标签单独读取未来区间；历史事故/未遂可以成为特征，但未来事件不能进入特征。
2. `_window`表示该行完整特征期，最终表对应61天；`_14d`及近期7/14/30天字段严格按固定长度计算。历史长度不足时为NULL，不向窗口外取数。B不能把训练期21天计数替换成最终61天计数。
3. 事件率保留明确分母；分母为0或缺失时输出NULL。没有轨迹/IMU的日期不视为停车或安全日。风险日表中的0仅表示没有记录事件。
4. 近半年画像统计截止日未知，保留原值供审计，但全部画像快照字段暂不进入严格核心表。
5. 轨迹速度均值、标准差按有效点数和二阶矩合并。IMU日表缺少每个轴及速度差的有效点数，因此`daily_mean`表示非空日统计的等权平均，不声称是全窗口逐点均值或标准差。IMU阈值仅作为候选异常，不作为事故标签。
6. 暴露量收缩提供训练折内`fit/transform`接口，不用全量数据预先估计先验。填补、截尾、标准化和模型选择由B在每个训练折内拟合；交付数据保持缺失值，不预先填零。
7. 原始数据及公共清洗层保持只读；未使用比赛外部数据。A只交付标签、特征和验证接口，不代表B的预测模型或C的评分体系已完成。

#### 6.1.6 本次结果报告与验收

版本：`member-a-v1.0`；最终独立验收时间：2026-09-22 11:06:54；机器报告状态：`passed`。以下为本次生成结果，重跑后应以最新报告为准。

| 检查项 | 结果 |
|---|---|
| README要求的六项固定产物 | 已全部生成 |
| 最终车辆数及唯一键 | 500辆，与画像一致；车辆/窗口主键无重复 |
| 主窗口标签复核 | 128辆正类，与先前独立统计一致 |
| 风险日表计数守恒 | 与3,152,654条风险事件一致 |
| 窗口标签 | 独立SQL重建核对通过 |
| 防泄漏 | 日期边界、未来数据注入、真实数据截断后特征不变等检查通过 |
| 训练/最终特征一致性 | 最终核心列与独立最后14天窗口逐值一致 |
| 正式核心表 | 100列，包含IMU及趋势；训练集无全空、常量或高度冗余列 |
| 核心特征训练集最大绝对相关系数 | 0.993435，小于0.995阈值 |
| 风险率独立公式核验 | 1,674,000项通过；零/缺失分母保持NULL |
| 自动化测试 | 15项全部通过 |
| 公共清洗输入文件 | 运行前后SHA256一致 |
| Parquet回读 | 行数、核心字段及候选表与核心表取值一致性检查通过 |

这表示A的计算、数据接口及当前验收范围通过，不代表预测AUC、40天概率校准或评分效果已经验证。B/C后续仍需按各自职责完成模型和评分实验。

### 6.2 队员B：事故预测

实施步骤：

1. 建立数据加载和时间验证框架；
2. 使用画像、事件计数和车辆日表训练最简逻辑回归；
3. 实现AUC、PR-AUC、模型保存和提交校验；
4. 预留A的正式特征接口。
5. 接入`train_windows.parquet`；
6. 训练正则逻辑回归；
7. 训练一个浅层梯度提升树；
8. 做特征族消融和时间窗口稳定性分析；
9. 仅在多个时间切分中均稳定提升时进行排名融合；
10. 用完整61天特征生成最终预测。

固定交付：

```text
outputs/prediction/
├─ validation_metrics.json
├─ oof_predictions.parquet
├─ feature_importance.csv
├─ model/
├─ forecast_result.csv
└─ prediction_method.md
```

提交CSV要求：

```text
字段：gpsno,accident,risk_prob
行数：500
gpsno：唯一且与画像一致
accident：0或1
risk_prob：0到1之间，无空值
```

### 6.3 队员C：安全评分

固定初始权重：

| 维度 | 权重 |
|---|---:|
| 历史事故与未遂 | 35% |
| 疲劳和分心 | 20% |
| 驾驶稳定性 | 20% |
| 道路规则风险 | 10% |
| 管理配合度 | 10% |
| IMU风险佐证 | 5% |

基础公式：

```text
风险率 = 时间衰减后的严重度加权事件数 / 行驶暴露量
维度惩罚 = 风险率经过收缩、截尾和分位数映射后的0～100值
安全分 = 100 - Σ(维度权重 × 维度惩罚)
```

评分规则：

- 次数必须除以里程、时长或活跃天数；
- 近期事件权重高于较早事件；
- 低暴露车辆使用经验贝叶斯收缩；
- 高相关事件设置维度扣分上限；
- IMU缺失不扣安全分，只降低可信度；
- 每辆车输出前三项扣分原因。

固定交付：

```text
outputs/scoring/
├─ safety_scores.csv
├─ score_details.parquet
├─ scoring_rules.csv
├─ score_validation.json
└─ scoring_method.md
```

## 7. 成员间接口

成员之间通过文件交付，不直接依赖对方未稳定的内部函数：

```text
A -> B：train_windows.parquet、full_61d_features.parquet
A -> B（首版入模）：train_model_core.parquet、forecast_model_core.parquet、core_model_features.json、split_manifest.csv
A -> C：feature_dictionary.csv、可选聚合特征
B -> 集成：forecast_result.csv、validation_metrics.json
C -> 集成：safety_scores.csv、scoring_method.md
```

公共清洗层`data/processed/cleaned/`必须保持只读。若发现清洗问题，应修改`src/data/clean_data.py`并全量重跑，而不是手工编辑Parquet。

## 8. 协作规则

1. 每个人只修改自己的源码和输出目录；
2. `src/common/`、`config/`、README和依赖文件属于共享区域；
3. 实验写入`runs/member_a`、`runs/member_b`或`runs/member_c`；
4. 每次实验保存配置、随机种子、特征清单和指标；
5. 不使用`latest.csv`等不可追踪文件名；
6. 数据和模型文件不提交到公共Git仓库；
7. 不使用任何比赛外部数据。

分支约定：

```text
feature/member-a-features
feature/member-b-prediction
feature/member-c-scoring
```

## 9. 最终验收

A部分（2026-09-22已验收，详见6.1.6）：

- [x] 六项固定交付及补充核心模型表已生成；
- [x] 17个时间窗口、8,500行训练样本及500辆车最终特征齐全；
- [x] 主键、日期边界、标签、分母、防泄漏和字段对齐检查通过；
- [x] 100维首版特征仅在早期训练窗口筛选，包含IMU；
- [x] 15项测试和独立产物验收通过，清洗输入SHA256未改变；
- [x] 已记录候选表与入模表区别、画像时间来源和40天验证限制。

任务一：

- [ ] 500辆车预测齐全；
- [ ] 使用时间验证而非随机事件拆分；
- [ ] `risk_prob`位于0～1；
- [ ] 提交格式与模板一致；
- [ ] 模型和预处理可以复现。

任务二：

- [ ] 500辆车评分齐全；
- [ ] 分数位于0～100；
- [ ] 高危历史不会得到高安全分；
- [ ] IMU缺失不会被误判为安全；
- [ ] 扣分原因可以回溯到事件或统计字段；
- [ ] 风险等级与后续事故/未遂率具有区分度。

项目整体：

- [ ] 从原始数据可复现清洗、特征、训练、预测和评分；
- [ ] 时间边界和随机种子固定；
- [ ] 未使用外部数据；
- [ ] 未泄露比赛原始数据；
- [ ] 文档包含清洗策略、特征、模型、验证、评分和运营建议。
