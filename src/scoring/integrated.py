"""Train/reference-only scoring calibration and a locked historical test."""
from pathlib import Path
import json
import pandas as pd
import numpy as np
from .calculate_scores import collect_cleaned_frames, resolve_inputs, score_frames
from .scoring_rules import default_rules_frame, DEFAULT_WEIGHTS, write_rules
from .calibrate_weights import calibrate_weights, binary_auc
from .event_taxonomy import DIMENSION_LABELS
from .export_scores import SAFETY_SCORE_COLUMNS, write_csv, write_parquet, write_json
from .validate_scores import validate_score_frame
import duckdb
from src.common.provenance import record_run

ROOT=Path(__file__).resolve().parents[2]

def make_reference(details):
    result={}
    for dim in DIMENSION_LABELS:
        result[dim]={'rates':details[f'{dim}_rate'].dropna().tolist()}
        if dim!='imu_evidence':
            result[dim].update(mu=float(details[f'{dim}_prior_mean'].iloc[0]),
                               tau=float(details[f'{dim}_prior_exposure'].iloc[0]))
    return result

def aligned_labels(details, labels):
    if labels.gpsno.duplicated().any():
        raise ValueError('Nonunique vehicle label key')
    mapped=details.gpsno.map(labels.set_index('gpsno').label)
    if mapped.isna().any():
        raise ValueError('Missing vehicle labels')
    return mapped

def future_report(details, labels, weights, calibration):
    target=aligned_labels(details,labels)
    report=validate_score_frame(details,weights,calibration,target)
    report['future_risk_auc']=binary_auc(target,100-details.safety_score)
    report['energy_future_auc']={str(k):binary_auc(target.loc[g.index],100-g.safety_score)
                               for k,g in details.groupby('energy_type')}
    behavior=sum(details[f'{d}_contribution'] for d in weights if d!='history')
    report['behavior_without_history_auc']=binary_auc(target,behavior)
    return report

def run():
    out=ROOT/'outputs/scoring'; out.mkdir(parents=True,exist_ok=True)
    feature=ROOT/'outputs/features'
    label_report=json.loads((feature/'label_report.json').read_text(encoding='utf-8'))
    if 'count(11804)>=2' not in label_report['definition']:
        raise ValueError('Rebuild A features with the corrected QA label rule')
    windows=pd.read_parquet(feature/'train_model_core.parquet')
    plan=pd.read_csv(feature/'split_manifest.csv')
    paths=resolve_inputs(ROOT/'data/processed/cleaned'); rules=default_rules_frame()
    def get_window(row, weights=None, reference=None):
        begin=str(row.feature_start)[:10]; end=str(row.feature_end)[:10]
        frames=collect_cleaned_frames(paths,rules,end+' 23:59:59',begin)
        days=(pd.Timestamp(end)-pd.Timestamp(begin)).days+1
        return score_frames(*frames[:4],weights=weights,reference=reference,observation_days=days)
    development=plan[plan.phase.eq('model_selection')]
    training=development[development.split_role.eq('train')].iloc[0]
    validation=development[development.split_role.eq('validation')].iloc[0]
    test=plan[plan.split_role.eq('test')].iloc[0]
    fit_details,_,_=get_window(training)
    reference=make_reference(fit_details)
    target=aligned_labels(fit_details,windows[windows.window_id.eq(training.window_id)])
    candidate,calibration=calibrate_weights(fit_details,target)
    dev_labels=windows[windows.window_id.eq(validation.window_id)]
    base,_,_=get_window(validation,DEFAULT_WEIGHTS,reference)
    alternative,_,_=get_window(validation,candidate,reference)
    base_auc=binary_auc(aligned_labels(base,dev_labels),100-base.safety_score)
    candidate_auc=binary_auc(aligned_labels(alternative,dev_labels),100-alternative.safety_score)
    weights=candidate if candidate_auc>=base_auc else DEFAULT_WEIGHTS.copy()
    selection={'initial_auc':base_auc,'candidate_auc':candidate_auc,'weights':weights,
               'calibration_fit':calibration,'reference_window':training.window_id,
               'limitation':'Only one independent development window available; no multi-window stability claim'}
    write_json(selection,out/'frozen_selection.json')
    # Neither weights nor reference distribution is changed after viewing test.
    details,_,_=get_window(test,weights,reference)
    test_report=future_report(details,windows[windows.window_id.eq(test.window_id)],weights,selection)
    write_json(test_report,out/'temporal_validation.json')
    write_csv(details,out/'locked_test_scores.csv',SAFETY_SCORE_COLUMNS)
    # Final operational score uses all 61 days; fit a contemporaneous reference
    # for this different observation length, with weights frozen by development.
    frames=collect_cleaned_frames(paths,rules)
    final,_,_=score_frames(*frames[:4],weights=weights)
    report=validate_score_frame(final,weights,selection)
    report['historical_locked_test']=test_report
    report['future_validation']={'status':'future_40d_labels_unavailable',
                                'historical_evaluation':'temporal_validation.json'}
    write_csv(final,out/'safety_scores.csv',SAFETY_SCORE_COLUMNS)
    write_parquet(final,out/'score_details.parquet',duckdb)
    write_rules(out/'scoring_rules.csv',rules)
    write_json(report,out/'score_validation.json')
    write_json(reference,out/'historical_reference.json')
    write_json(make_reference(final),out/'final_reference.json')
    method='# 安全评分：答疑修订版\n\n'
    method+='gpsno按答疑视为司机唯一标识。事故≥1或未遂≥2用于未来正类；单次未遂仍计历史扣分。\n\n'
    method+='所有事件、轨迹、IMU均按同一起止日截断；管理维度分母为moving_rows>0的活跃天。'
    method+='IMU速度源差使用非空日均值的等权平均，不冒充逐点加权均值。\n\n'
    method+='历史验证：在6月1—14日评分数据拟合先验和分位点、使用6月15—28日标签校准权重；'
    method+='6月17—30日评分、7月1—14日标签仅作方案选择；冻结后在7月4—17日评分、7月18—31日标签测试。'
    method+='校准与评价按gpsno关联，禁止依赖行顺序。\n\n'
    method+='最终61天评分使用冻结权重及当前61天队列参考分布，与历史14天评分不直接作数值纵向比较。'
    method+='日/周/月更新需固定窗口和参考分布；当前同队列IMU缺失贡献为0并降低可信度。'
    method+='尚无8月真实标签，不宣称最终40天风险已验证。\n\n'
    method+='## 独立测试结果\n\n'+json.dumps({'auc':test_report['future_risk_auc'],
        'future_validation':test_report['future_validation'], 'selection':selection},ensure_ascii=False,indent=2)
    (out/'scoring_method.md').write_text(method,encoding='utf-8')
    record_run('member_c',list(paths.values())+[feature/'train_model_core.parquet',feature/'split_manifest.csv',feature/'label_report.json'],list(out.glob('*')))
    print(json.dumps({'rows':len(final),'static':report['overall_static_status'],
                     'test_auc':test_report['future_risk_auc'],
                     'test_future_validation':test_report['future_validation']},ensure_ascii=False,indent=2))
    return report

if __name__=='__main__':
    run()
