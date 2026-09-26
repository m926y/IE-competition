# 驾驶安全比赛：最终方案

正式交付在 `submission/final/`，仅包含预测CSV、评分CSV、算法说明PDF、源码ZIP。当前统一版本为2026-09-25，安装与复现见 [提交说明](docs/SUBMISSION_README.md)，选择依据与改进见 [优化记录](docs/final_optimization.md)。

任务一采用正式/辅助目标LightGBM排名融合，新增7维训练期暴露收缩风险率。开发平均AUC从0.8620升至0.8682；历史回顾AUC从前冠军0.8156升至0.8200，AP由0.1642降至0.1577。比赛以AUC为主指标，不宣称其他指标全部提升。任务二保留较优固定参考评分，AUC 0.6557，补充不确定性下界、等级风险率区间和证据解释。

历史晚期已被查看，以上不是独立测试成绩；真实未来40天概率校准未知。仅使用比赛数据，结果只供比赛授权渠道。

```powershell
.venv/Scripts/python.exe -m src.reproduce_submission --package
.venv/Scripts/python.exe -m src.delivery.validate submission/final
```

目录：`data/raw`为原始数据，`data/processed/cleaned`为公共清洗表，`outputs/features`为已审计特征，`outputs/prediction`与`outputs/scoring`为唯一模型结果，`configs/submission`为冻结配置，`outputs/audit`保存简要选型和删除记录，`tests`为自动化检查。

特征截止7月31日，预测40天采用[2026-08-01,2026-09-10)。正式正类沿用项目答疑修订口径：事故11803≥1条或未遂11804≥2条。原题说明与IMU参考在 `docs/competition`，原始数据不随源码交付。
