# 最终事故预测

唯一入口：python -m src.prediction.optimize。读取configs/submission冻结配置，输出outputs/prediction。正式目标浅层LightGBM与三种子辅助模型等权排名融合，辅助模型增加7维训练期收缩风险率。

模型包含预处理、校准、先验、阈值与完整车队ID；predict_bundle必须整批500车、同一截止日。40天概率为明确假设下的外推，未来真值未验证。历史AUC 0.819991、AP 0.157706；详见docs/final_optimization.md。
