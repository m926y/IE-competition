# B 事故预测：本地整合版

运行：

```powershell
.venv/Scripts/python.exe -m src.prediction.train_final
```

正式入口为integrated.py。输入为A的train_model_core.parquet、forecast_model_core.parquet、core_model_features.json、split_manifest.csv和答疑修订后的label_report.json。拒绝旧标签与缺失核心字段。

只使用固定14天特征和14天标签；开发验证期选择模型、早停、校准与阈值，冻结后评估锁定测试，最终重训。仅一个独立开发窗口，禁用融合。分组消融只报告开发期影响，不据锁定测试调整模型。

输出位于outputs/prediction，包含预测CSV、训练外及锁定测试预测、模型、核心特征顺序、选择冻结记录、消融、指标和方法说明。probability_at_horizon使用明确的独立平稳Poisson假设，将14天概率转换为“40天事故≥1或未遂≥2”的概率；其实际40天校准效果未知。

旧train_final.run只保留给合成测试兼容并拒绝正式A表。CLI始终走正式整合入口，不支持旧的skip-ablation参数。

测试：python -m unittest discover -s tests -v。详见docs/integration_review.md。
