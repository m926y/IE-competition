"""Frozen final predictor with training-only exposure shrinkage."""
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score, average_precision_score
from .data_loader import load_training_data, METADATA_COLUMNS
from .integrated import probability_at_horizon, accident_fraction, dump
from .ensemble import fit_platt_calibrator, calibrate_probability
from .evaluate import choose_operating_threshold
from .train_boosting import fit_final_boosting
from .validate_submission import validate_submission
from src.features.shrunk_rates import fit_rate_priors, transform_rates
from src.common.provenance import sha256

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / 'outputs/features'
OUT = ROOT / 'outputs/prediction'
CONFIG = ROOT / 'configs/submission'
SEEDS = [2026, 42, 137]


def load_inputs():
    report = json.loads((SOURCE/'label_report.json').read_text(encoding='utf-8'))
    if 'count(11804)>=2' not in report['definition']:
        raise ValueError('Rebuild labels using accident >= 1 OR near miss >= 2')
    frame = load_training_data(SOURCE/'train_windows.parquet')
    frame = frame[frame.feature_window_days.eq(14) & frame.label_horizon_days.eq(14)].reset_index(drop=True)
    full = pd.read_parquet(SOURCE/'full_61d_features.parquet')
    full.gpsno = full.gpsno.astype(str)
    frame['aux_any'] = frame[['accident_label', 'near_miss_label']].max(axis=1).astype(int)
    core = json.loads((SOURCE/'core_model_features.json').read_text())['columns']
    if set(core) & (METADATA_COLUMNS | {'aux_any', 'aux_count'}):
        raise ValueError('Target or metadata in features')
    return frame, full, {'core': core}


def rank(values):
    return pd.Series(values).rank(method='average', pct=True).to_numpy()


def baseline_fit(frame, core):
    frozen = json.loads((CONFIG/'baseline_selection.json').read_text())
    return fit_final_boosting(frame, core, frozen['params'], n_estimators=frozen['iterations'])


def fit_models(frame, columns, spec):
    y = frame[spec['target']].to_numpy()
    mass = 1/frame.groupby('gpsno').gpsno.transform('size').to_numpy()
    mass = mass/mass.mean() * np.where(y > 0, np.sqrt((y == 0).sum()/max(1, (y > 0).sum())), 1)
    models = []
    for seed in SEEDS:
        model = LGBMClassifier(n_estimators=spec['rounds'], max_depth=3, num_leaves=7,
            learning_rate=.03, min_child_samples=30, reg_lambda=10., reg_alpha=.5,
            colsample_bytree=.8, subsample=.8, subsample_freq=1, deterministic=True,
            force_col_wise=True, random_state=seed, n_jobs=4, verbosity=-1)
        model.fit(frame[columns], y, sample_weight=mass)
        models.append(model)
    return models


def predict(models, frame, columns, spec=None):
    return np.mean([m.predict_proba(frame[columns])[:, 1] for m in models], axis=0)


def fit_predict(frame, validation, choice):
    core = choice['feature_sets']['core']
    base = baseline_fit(frame, core)
    base_score = base.predict_proba(validation[core])[:, 1]
    priors = fit_rate_priors(frame)
    train, added = transform_rates(frame, priors)
    valid, _ = transform_rates(validation, priors)
    spec = choice['specs'][0]
    columns = core + added
    models = fit_models(train, columns, spec)
    score = .5*rank(base_score) + .5*rank(predict(models, valid, columns))
    return score, dict(baseline=base, models=models, columns=columns, spec=spec,
                       choice=choice, rate_priors=priors)


def validate_cohort(bundle, frame):
    ids = frame.gpsno.astype(str)
    if frame.gpsno.isna().any() or ids.duplicated().any() or set(ids) != set(bundle['cohort_ids']):
        raise ValueError('Rank blend requires the complete unique training vehicle cohort')
    if ('feature_end' not in frame or pd.to_datetime(frame.feature_end).isna().any()
            or pd.to_datetime(frame.feature_end).nunique() != 1):
        raise ValueError('A cohort must have exactly one forecast cutoff')


def predict_bundle(bundle, frame):
    validate_cohort(bundle, frame)
    base = bundle['baseline'].predict_proba(frame[bundle['choice']['feature_sets']['core']])[:, 1]
    transformed, _ = transform_rates(frame, bundle['rate_priors'])
    auxiliary = predict(bundle['models'], transformed, bundle['columns'])
    score = .5*rank(base) + .5*rank(auxiliary)
    return probability_at_horizon(calibrate_probability(bundle['calibrator'], score), bundle['accident_fraction'])


def paired_bootstrap(y, base, new):
    rng = np.random.default_rng(20260924)
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    differences = []
    for _ in range(1000):
        index = np.r_[rng.choice(pos, len(pos)), rng.choice(neg, len(neg))]
        differences.append(roc_auc_score(y[index], new[index])-roc_auc_score(y[index], base[index]))
    return {'delta_ci95': np.quantile(differences, [.025, .975]).tolist(),
            'bootstrap_fraction_improved': float(np.mean(np.array(differences) > 0)),
            'limitation': 'Conditional on retrospective window; excludes model-selection bias'}


def reproduce_frozen():
    OUT.mkdir(parents=True, exist_ok=True)
    choice = json.loads((CONFIG/'prediction_selection.json').read_text(encoding='utf-8'))
    if choice['seeds'] != SEEDS or choice['candidate'] != 'blend_aux_shrunk_rates':
        raise ValueError('Unexpected frozen model contract')
    if sha256(CONFIG/'baseline_selection.json') != choice['baseline_selection_sha256']:
        raise ValueError('Frozen baseline changed')
    frame, full, sets = load_inputs()
    if sets != choice['feature_sets']:
        raise ValueError('Core feature contract changed')
    records = []
    for window in choice['development_windows']:
        valid = frame[frame.window_id.eq(window)]
        train = frame[frame.label_end.le(valid.feature_end.min()) & frame.feature_end.lt(valid.feature_end.min())]
        score, _ = fit_predict(train, valid, choice)
        record = valid[['gpsno', 'window_id', 'label']].copy()
        record['score'] = score
        records.append(record)
    pd.concat(records).to_parquet(OUT/'development_predictions.parquet', index=False)
    dev = records[-1]
    calibrator = fit_platt_calibrator(dev.label, dev.score, regularization_c=1.0)
    if calibrator is None or calibrator.coef_[0, 0] <= 0:
        raise ValueError('Calibration must be monotone increasing')
    threshold = choose_operating_threshold(dev.label, calibrate_probability(calibrator, dev.score))
    late = frame[frame.window_id.eq('auxiliary_14d_20260717_h14')]
    train = frame[frame.label_end.le(late.feature_end.min()) & frame.feature_end.lt(late.feature_end.min())]
    score, _ = fit_predict(train, late, choice)
    base = baseline_fit(train, sets['core']).predict_proba(late[sets['core']])[:, 1]
    y = late.label.to_numpy()
    result = dict(candidate=choice['candidate'], evaluation_kind='previously viewed retrospective July18-31; NOT fresh holdout',
        baseline_auc=roc_auc_score(y, base), candidate_auc=roc_auc_score(y, score),
        baseline_ap=average_precision_score(y, base), candidate_ap=average_precision_score(y, score),
        auc_delta=roc_auc_score(y, score)-roc_auc_score(y, base), **paired_bootstrap(y, base, score))
    retrospective = late[['gpsno', 'label']].copy()
    retrospective['baseline'], retrospective['candidate'] = base, score
    retrospective.to_parquet(OUT/'retrospective_predictions.parquet', index=False)
    _, bundle = fit_predict(frame, full, choice)
    fraction = accident_fraction('2026-07-31')
    threshold40 = float(probability_at_horizon([threshold['threshold']], fraction)[0])
    bundle.update(calibrator=calibrator, calibration_regularization_c=1.0,
                  accident_fraction=fraction, threshold40=threshold40, cohort_ids=sorted(full.gpsno))
    p40 = predict_bundle(bundle, full)
    forecast = pd.DataFrame({'gpsno': full.gpsno, 'accident': (p40 >= threshold40).astype(int), 'risk_prob': p40})
    ids = pd.read_csv(ROOT/'data/processed/cleaned/vehicle_profile.csv', dtype={'gpsno': str}).gpsno
    result['submission'] = validate_submission(forecast, expected_ids=ids)
    forecast.to_csv(OUT/'forecast_result.csv', index=False, encoding='utf-8', lineterminator='\n')
    joblib.dump(bundle, OUT/'predictor.joblib')
    dump(OUT/'selection_frozen.json', choice)
    dump(OUT/'evaluation.json', result)
    dump(OUT/'rate_priors.json', bundle['rate_priors'])
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    reproduce_frozen()
