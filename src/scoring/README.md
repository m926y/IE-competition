# C 安全评分：本地整合版

正式运行：python -m src.scoring.integrated。读取A修订标签与固定时间划分，以及公共清洗层。

历史评分从同一起止日读取事件、轨迹和IMU；训练期拟合经验先验和分位点，验证/测试使用固定参考参数。未来标签按gpsno匹配。权重只在开发阶段校准与选择，测试标签不参与。

输出outputs/scoring：500车safety_scores.csv、score_details.parquet、scoring_rules.csv、score_validation.json、scoring_method.md，以及temporal_validation.json、frozen_selection.json和参考参数。

当前历史锁定测试不满足全部等级单调性要求，报告保留review_required。静态检查通过不等于未来区分度验收通过。

单窗口诊断：python -m src.scoring.calculate_scores --start-date 2026-06-01 --as-of "2026-06-21 23:59:59"。该入口是初始权重诊断，不替代正式时间校准入口。

日级传感器输入仅支持日末截止；同长度动态评分需复用参考参数。完整61天最终评分重新拟合当期参考分布、保留开发期冻结权重，不与历史14天分数作直接纵向比较。

详见docs/integration_review.md。
