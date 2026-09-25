# 成员 C 安全评分运行说明

本目录实现任务二的可解释 100 分制安全评分。当前 v2.0 已正式接入 A 成员发布的时间窗口特征、未来事故/未遂标签和固定切分表，不再是 A 交付前的预校准版。

## 输入

默认从 `outputs/features/` 读取 A 的正式发布包：

- `risk_event_daily.parquet`
- `train_windows.parquet`
- `full_61d_features.parquet`
- `feature_dictionary.csv`
- `label_report.json`
- `split_manifest.csv`

## 时间切分

- 模型选择训练：6 月 1‑14 日特征，6 月 15‑28 日标签。
- 模型选择验证：6 月 17‑30 日特征，7 月 1‑14 日标签。
- 锁定测试：7 月 4‑17 日特征，7 月 18‑31 日标签。该窗口只做最后评估，不反向调权。
- 最终评分：使用 7 月 18‑31 日的最新 14 天行为，面向 8 月 1 日之后的风险管理。

## 运行

```powershell
python -m pip install -r src/scoring/requirements.txt
python -m src.scoring.calculate_scores
```

可用 `--feature-dir`、`--output-dir` 和 `--run-dir` 覆盖默认目录。

## 输出

`outputs/scoring/` 生成最终排名、明细、规则、验证报告和对照 docs 的 `acceptance_report.json`；`runs/member_c/` 生成运行清单和冻结的校准参数。由于仓库规则禁止直接提交数据派生文件，实际结果通过 GitHub Release 附件发布，源码、方法文档和规则表由 Git 跟踪。

## 当前验证结论

500 车静态检查全部通过。锁定测试中，最危险 20% 的未来事故/未遂率为 25%，最安全 20% 为 7%；但 B、C 档未呈严格单调，因此总状态如实标记为 `review_required`。
