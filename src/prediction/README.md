# 队员B：事故预测模块

本目录实现任务一的完整训练、验证、融合、校准和提交生成流程，不负责构造队员A定义的标签与特征。

## 输入契约

默认读取：

```text
outputs/features/train_windows.parquet
outputs/features/full_61d_features.parquet
outputs/features/feature_dictionary.csv
data/processed/cleaned/vehicle_profile.csv
```

`train_windows.parquet`必须包含：

```text
gpsno, window_id, feature_start, feature_end,
label_start, label_end, label, <模型特征...>
```

`full_61d_features.parquet`必须每车一行，包含`gpsno`和与训练表一致的模型特征。缺失的正式特征列会以缺失值对齐，多出的列不会进入模型。

## 安装与运行

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe -m src.prediction.train_final
```

运行快速流程时可以暂时跳过特征族消融：

```powershell
.venv\Scripts\python.exe -m src.prediction.train_final --skip-ablation
```

正式提交必须执行完整流程，不添加`--skip-ablation`。

## 验证逻辑

- 只进行前向时间验证，不随机拆分事件或车辆窗口；
- 训练窗口的标签期必须早于验证标签期；
- 缺失填补、极端值截尾、标准化和独热编码仅在训练折拟合；
- 逻辑回归使用L2正则与类别权重；
- 主模型使用浅层LightGBM，并通过早停控制复杂度；
- 只有排名融合在多个时间折中稳定时才启用；
- 使用训练外预测完成Platt概率校准和二分类阈值选择；
- 不使用外部数据，也不把`gpsno`作为特征。

## 输出

```text
outputs/prediction/
├─ validation_metrics.json
├─ oof_predictions.parquet
├─ feature_importance.csv
├─ ablation_results.csv
├─ model/
│  ├─ logistic.joblib
│  ├─ lightgbm.joblib
│  ├─ platt_calibrator.joblib
│  └─ metadata.json
├─ forecast_result.csv
└─ prediction_method.md
```

提交文件可以单独复核：

```powershell
.venv\Scripts\python.exe -m src.prediction.validate_submission `
  outputs/prediction/forecast_result.csv `
  --expected-ids data/processed/cleaned/vehicle_profile.csv
```

## 测试

```powershell
.venv\Scripts\python.exe -m unittest discover -s tests -v
```

测试使用合成时间窗口，不读取或复制比赛原始数据。
