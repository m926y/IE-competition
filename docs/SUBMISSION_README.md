# 最终提交版本安装与复现

版本2026-09-25。代码包仅供本次比赛授权使用，不附带比赛数据和数据访问凭据。最终交付为 `forecast_result.csv`、`safety_scores.csv`、`method_report.pdf`、`source_code.zip` 四个文件。

## 环境

实测Windows、Python 3.13、CPU，无需GPU。建议使用代码包内锁定依赖，其他平台需重新验收。

```powershell
python -m venv .venv
.venv/Scripts/python.exe -m pip install -r requirements-lock.txt
```

仓库可使用 `requirements.txt`；Linux/macOS将解释器改为 `.venv/bin/python`。不要跨Python版本复制 `.deps`。仅软件依赖允许下载，不引入外部训练数据。

## 数据与运行

将比赛授权原始文件放到：

```text
data/raw/vehicle_profile.xlsx
data/raw/risk_events.tsv.gz
data/raw/trajectory/*.gz  # 全部30个分片
data/raw/imu/*.gz         # 全部30个分片
```

从原始文件执行：

```powershell
.venv/Scripts/python.exe -m src.reproduce_submission --clean --rebuild-features --package
```

有清洗表和已校验特征时：

```powershell
.venv/Scripts/python.exe -m src.reproduce_submission --package
```

仅有清洗表时添加 `--rebuild-features`。全量原始清洗涉及约13.26亿条传感器记录，耗时取决于机器和磁盘。本轮复用已有清洗表，未再次扫描全部原始数据。DuckDB清洗内存设置4GB、特征层2GB，不等于总进程峰值；需要预留原始压缩文件、输出与临时空间。

配置必须完整包含 `configs/submission/baseline_selection.json`、`prediction_selection.json`、`scoring_frozen.json`。正式模型从冻结参数重训；辅助模型的7维收缩先验在每个训练期重估并保存在模型中。评分复用冻结权重与参考分布，不重新选型。已有目录可重复运行同一方案，无需旧版本目录或IMU五分钟风险表。

主要输出：

- `outputs/features/`：标签、日表、训练与最终特征、字典、独立验收。
- `outputs/prediction/`：模型、训练先验、500车预测、回顾结果和稳健性。
- `outputs/scoring/`：500车安全分、明细、8天动态评分、等级置信区间。
- `submission/final/`：仅四个正式文件；`outputs/delivery/acceptance.json`为交付验收。

源码包排除比赛明细和模型二进制，必须由获授权数据重建模型；不依赖本机其他项目目录。

## 动态评分与推理约束

```powershell
.venv/Scripts/python.exe -m src.scoring.enhanced --as-of 2026-07-31
```

每日、每周、每月运行均使用固定14天窗口、14天半衰期和训练参考分布。当前数据截止7月31日，超过数据覆盖日拒绝执行；先接入新清洗表再更新。评分冻结配置使用标签截至7月14日，不能在更早日期作为当时可用配置部署。

安全分按六维行为贡献扣分，IMU缺失不确定性另列为保守下界。贡献包含先验收缩，不全是直接观测违规。A/B/C/D为20%/40%/30%/10%相对容量分级，不能据等级判断车队整体改善。低置信度优先复核设备，等级不直接触发处罚。

预测入口 `src.prediction.optimize.predict_bundle` 要求完整500辆车、ID无重复、同一截止日；输入为同结构的固定14天核心及风险计数/暴露特征，不能逐车单独排名。最终风险来自14天校准与独立平稳Poisson40天外推，真实40天概率仍未验证。

## 验收与指标

```powershell
.venv/Scripts/python.exe -m unittest discover -s tests -v
.venv/Scripts/python.exe -m src.delivery.validate submission/final
```

历史回顾7月18日至31日：预测AUC 0.819991、AP 0.157706；评分AUC 0.655720。前冠军预测AUC 0.815602、AP 0.164210；新方案提高AUC但AP有所下降。晚期标签此前已被查看，不能称为新独立测试。等级风险率区间明显重叠，单调点估计不代表可靠等级差异。

CSV采用UTF-8无BOM与LF；两份文件各500个唯一gpsno。PDF合并任务一、二说明。ZIP使用源码白名单与SHA256清单。单文件按小于50MB保守检查，队名与平台字段以最终上传界面为准；本流程只生成本地文件，不代为上传。
