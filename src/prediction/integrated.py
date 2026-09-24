"""QA-compliant integration: frozen feature contract, selection then locked test."""
from pathlib import Path
import json
import hashlib
import joblib
import duckdb
import numpy as np
import pandas as pd
from scipy.optimize import brentq
from .data_loader import load_training_data, load_prediction_data, METADATA_COLUMNS
from .splitters import TimeFold
from .train_logistic import cross_validate_logistic, fit_final_logistic, logistic_importance
from .train_boosting import cross_validate_boosting, fit_final_boosting, boosting_importance
from .evaluate import binary_metrics, choose_operating_threshold, metric_sort_key
from .ensemble import fit_platt_calibrator, calibrate_probability
from .validate_submission import validate_submission
from src.common.provenance import record_run

ROOT = Path(__file__).resolve().parents[2]

def phase_fold(frame, manifest, phase):
    plan = manifest[manifest.phase.eq(phase)]
    train_ids = plan.loc[plan.split_role.eq('train'), 'window_id']
    evaluation = plan[~plan.split_role.eq('train')]
    if len(evaluation) != 1:
        raise ValueError('Exactly one evaluation window is required per phase')
    row = evaluation.iloc[0]
    tr = np.flatnonzero(frame.window_id.isin(train_ids))
    va = np.flatnonzero(frame.window_id.eq(row.window_id))
    if not len(tr) or not len(va):
        raise ValueError('Manifest references missing windows')
    if frame.iloc[tr].label_end.max() > pd.Timestamp(row.feature_end):
        raise ValueError('Training labels unavailable at evaluation forecast cutoff')
    if frame.iloc[tr].label.nunique()!=2 or frame.iloc[va].label.nunique()!=2:
        raise ValueError('Both classes are required in train/evaluation')
    return TimeFold(phase,tr,va,str(row.window_id),str(row.feature_end),str(row.label_start))

def probability_at_horizon(p14, accident_fraction, days=40):
    """Independent stationary Poisson counts; positive iff A>=1 or N>=2.

    Invert the 14-day positive probability for total intensity. The accident
    fraction is estimated only from recorded historical events. This is an
    explicit extrapolation assumption, not observed 40-day calibration.
    """
    a = float(np.clip(accident_fraction,0,1))
    rates = np.asarray([brentq(lambda x: 1-np.exp(-x)*(1+(1-a)*x)-p,
                              0,100) for p in np.clip(p14,1e-7,1-1e-7)])
    rates *= days/14
    return np.clip(1-np.exp(-rates)*(1+(1-a)*rates),0,1)

def accident_fraction(cutoff):
    path=ROOT/'data/processed/cleaned/risk_events.parquet'
    with duckdb.connect() as con:
        a,n=con.execute('''SELECT count_if(event_type=11803),count_if(event_type=11804)
          FROM read_parquet(?) WHERE event_time < CAST(? AS DATE)+INTERVAL 1 DAY''',
          [str(path),str(pd.Timestamp(cutoff).date())]).fetchone()
    return float(a/(a+n)) if a+n else 0.0

def dump(path, value):
    Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False,default=str),encoding='utf-8')

def run(output_dir=None):
    source=ROOT/'outputs/features'
    label_report=json.loads((source/'label_report.json').read_text(encoding='utf-8'))
    if 'count(11804)>=2' not in label_report['definition']:
        raise ValueError('A labels predate QA correction; rebuild features first')
    out=Path(output_dir or ROOT/'outputs/prediction'); out.mkdir(parents=True,exist_ok=True)
    contract=json.loads((source/'core_model_features.json').read_text(encoding='utf-8'))
    cols=contract['columns']
    if set(cols)&METADATA_COLUMNS:
        raise ValueError('Label or metadata present in core feature contract')
    train=load_training_data(source/'train_model_core.parquet')
    full=load_prediction_data(source/'forecast_model_core.parquet')
    if set(cols)-set(full) or set(cols)-set(train):
        raise ValueError('Core schema mismatch')
    plan=pd.read_csv(source/'split_manifest.csv')
    selection=phase_fold(train,plan,'model_selection')
    locked=phase_fold(train,plan,'locked_test')
    if train.iloc[selection.valid_index].label_end.max() >= train.iloc[locked.valid_index].label_start.min():
        raise ValueError('Selection and locked labels overlap')
    # Pass only training/validation rows to tuning code, not the locked test rows.
    ids=np.r_[selection.train_index,selection.valid_index]
    dev=train.iloc[ids].reset_index(drop=True)
    fold=phase_fold(dev,plan,'model_selection')
    lr=cross_validate_logistic(dev,cols,[fold])
    gb=cross_validate_boosting(dev,cols,[fold])
    selected=max([lr,gb],key=lambda x:metric_sort_key(x['summary']))
    name=selected['model_name']
    params=selected['params']
    iterations=max(1,int(gb['summary']['median_best_iteration']))
    def fit(data, model_name=name):
        if model_name=='logistic':
            return fit_final_logistic(data,cols,lr['params'])
        return fit_final_boosting(data,cols,gb['params'],n_estimators=iterations)
    score=selected['oof'][fold.valid_index]
    y=dev.iloc[fold.valid_index].label.to_numpy()
    calibrator=fit_platt_calibrator(y,score)
    if calibrator.coef_[0,0] <= 0:
        raise ValueError('Calibration would reverse risk ordering; inspect development data')
    p=calibrate_probability(calibrator,score)
    threshold=choose_operating_threshold(y,p)
    # Freeze parameters before accessing any locked-test model performance.
    frozen={'features':cols,'model':name,'params':params,'iterations':iterations,
            'threshold14':threshold,'blend':'disabled: only one independent selection window',
            'label':'accident_count>=1 OR near_miss_count>=2','seed':2026}
    dump(out/'frozen_selection.json',frozen)
    frozen_hash=hashlib.sha256((out/'frozen_selection.json').read_bytes()).hexdigest()
    test_model=fit(train.iloc[locked.train_index])
    test=train.iloc[locked.valid_index]
    test_raw=test_model.predict_proba(test[cols])[:,1]
    test_p=calibrate_probability(calibrator,test_raw)
    locked_metrics=binary_metrics(test.label,test_p)
    oof=dev.iloc[fold.valid_index][['gpsno','window_id','label']].copy()
    oof['raw_probability']=score; oof['phase']='model_selection'
    held=test[['gpsno','window_id','label']].copy()
    held['raw_probability']=test_raw; held['phase']='locked_test'
    pd.concat([oof,held]).to_parquet(out/'oof_predictions.parquet',index=False)
    # One fixed observation length and target horizon; never mix main_40d rows.
    final_train=train[(train.feature_window_days==14)&(train.label_horizon_days==14)]
    model=fit(final_train)
    p14=calibrate_probability(calibrator,model.predict_proba(full[cols])[:,1])
    fraction=accident_fraction('2026-07-31')
    p40=probability_at_horizon(p14,fraction)
    cut40=float(probability_at_horizon([threshold['threshold']],fraction)[0])
    submission=pd.DataFrame({'gpsno':full.gpsno,'accident':(p40>=cut40).astype(int),'risk_prob':p40})
    expected=pd.read_csv(ROOT/'data/processed/cleaned/vehicle_profile.csv',dtype={'gpsno':str}).gpsno
    valid=validate_submission(submission,expected_ids=expected,expected_rows=500)
    submission.to_csv(out/'forecast_result.csv',index=False,encoding='utf-8',lineterminator='\n')
    modeldir=out/'model'; modeldir.mkdir(exist_ok=True)
    joblib.dump({'model':model,'calibrator':calibrator,'columns':cols,'accident_fraction':fraction,
                 'threshold40':cut40},modeldir/'predictor.joblib')
    dump(modeldir/'metadata.json',frozen|{'accident_fraction':fraction,'threshold40':cut40})
    importance=logistic_importance(model) if name=='logistic' else boosting_importance(model)
    importance.to_csv(out/'feature_importance.csv',index=False)
    # Descriptive ablation on selection only; never use locked-test feedback.
    dictionary=pd.read_csv(source/'feature_dictionary.csv').set_index('feature_name')
    ablations=[]
    for group in ['profile','event_group','event_type','exposure','recent','trend','trajectory','imu','quality','clustering']:
        subset=[c for c in cols if dictionary.loc[c,'feature_group']!=group]
        if len(subset)==len(cols):
            ablations.append({'removed_group':group,'status':'not_in_core'}); continue
        result=cross_validate_logistic(dev,subset,[fold],c_values=[lr['params']['C']]) if name=='logistic' else cross_validate_boosting(dev,subset,[fold],parameter_grid=[gb['params']])
        ablations.append({'removed_group':group,'status':'development_only',**result['summary']})
    pd.DataFrame(ablations).to_csv(out/'ablation_results.csv',index=False)
    metrics={'label_definition':frozen['label'],'selection':{'logistic':lr['summary'],'lightgbm':gb['summary']},
             'selected_model':name,'locked_test':locked_metrics,'locked_test_window':locked.valid_window,
             'frozen_selection_sha256':frozen_hash,'submission':valid,
             'calibration':'fit on model-selection validation; assessed on locked test only',
             'horizon':'40d extrapolation via stationary independent Poisson accident/near-miss counts; not validated against 40d future truth',
             'accident_fraction':fraction,'threshold40':cut40}
    dump(out/'validation_metrics.json',metrics)
    (out/'prediction_method.md').write_text('# 事故预测方法（答疑修订版）\n\n'
        '正类：事故≥1或未遂≥2。核心列由A训练期筛选，排除ID、标签、窗口元数据和半年画像。'
        '只使用14天特征预测14天标签；读取split_manifest，开发集选择模型，冻结参数后评估7月18—31日。'
        '逻辑回归与LightGBM按开发集AUC选择；独立开发窗口不足两个，禁用融合。'
        '概率校准与阈值只使用开发验证集；锁定测试指标独立报告。\n\n'
        '40天外推假定事故、未遂为独立平稳Poisson计数：P=1-exp(-λ事故-λ未遂)(1+λ未遂)。'
        '按历史两类事件比例反推14天总强度，再按40/14放大。此假设未获真实未来40天标签验证，'
        '单调变换保留AUC排序；不宣称已完成40天概率校准。最终窗口为[2026-08-01,2026-09-10)。\n\n'
        +json.dumps(metrics,ensure_ascii=False,indent=2),encoding='utf-8')
    record_run('member_b',list(source.glob('*.json'))+[source/'train_model_core.parquet',source/'forecast_model_core.parquet',source/'split_manifest.csv'],list(out.glob('*'))+list(modeldir.glob('*')))
    print(json.dumps(metrics,ensure_ascii=False,indent=2))
    return metrics

if __name__=='__main__':
    run()
