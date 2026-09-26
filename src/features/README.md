# 特征与标签

入口：python -m src.features.pipeline --overwrite；独立验收：python -m src.features.audit_delivery。

从公共清洗表构造固定时间窗口、795维候选和100维训练期筛选核心列。正式正类为事故11803至少1条或未遂11804至少2条。核心不含ID、未来标签或截至7月底的画像快照。

shrunk_rates.py为辅助模型增加7维训练期暴露收缩风险率；只在训练期拟合先验，零或未知暴露保留缺失。最终解释、参数和复现步骤见docs/final_optimization.md与docs/SUBMISSION_README.md。
