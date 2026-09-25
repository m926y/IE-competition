# 评分交付目录

此目录是成员 C 的固定交付接口。仓库跟踪方法文档和规则表；以下数据派生产物由 `python -m src.scoring.calculate_scores` 生成，并放入 GitHub Release 附件：

- `safety_scores.csv`
- `score_details.parquet`
- `score_validation.json`
- `acceptance_report.json`

这样既让 Git 主分支保持精简，又能在 GitHub 上下载完整可复现结果。
