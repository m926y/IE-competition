# 评分交付目录

此目录是成员 C 的固定交付接口。仓库只跟踪不含比赛数据的方法文档和规则表；以下数据派生产物由 `python -m src.scoring.calculate_scores` 在本地生成，并被 `.gitignore` 排除：

- `safety_scores.csv`
- `score_details.parquet`
- `score_validation.json`

这样既保留完整复现路径，也遵守赛题“不得泄露数据”和仓库“不提交数据、模型及生成结果”的规则。
