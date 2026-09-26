"""Post-selection diagnostics. Never use this audit to select model parameters."""
import json
import joblib
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from .optimize import ROOT, OUT, predict_bundle, predict
from .validate_submission import validate_submission
from src.features.shrunk_rates import transform_rates


def rank_change(base, altered, k=50):
    a = set(np.argsort(-base, kind='stable')[:k])
    b = set(np.argsort(-altered, kind='stable')[:k])
    return {'spearman': float(spearmanr(base, altered).statistic),
            'top50_overlap': len(a & b) / k,
            'mean_abs_probability_shift': float(np.abs(base - altered).mean())}


def run():
    bundle = joblib.load(OUT / 'predictor.joblib')
    frame = pd.read_parquet(ROOT / 'outputs/features/full_61d_features.parquet')
    dictionary = pd.read_csv(ROOT / 'outputs/features/feature_dictionary.csv').set_index('feature_name')
    cols = bundle['choice']['feature_sets']['core']
    # The 61-day container includes fixed 14d/7d columns; verify actual semantics.
    windows = dictionary.loc[cols, 'window_days'].astype(str)
    assert windows.isin(['14', '7']).all(), 'Variable-window feature in fixed-window model'
    base = predict_bundle(bundle, frame)
    csv = pd.read_csv(OUT / 'forecast_result.csv', dtype={'gpsno': str})
    assert frame.gpsno.astype(str).tolist() == csv.gpsno.tolist()
    assert np.allclose(csv.risk_prob, base, rtol=1e-12, atol=1e-14)
    assert np.array_equal(csv.accident, (base >= bundle['threshold40']).astype(int))
    validation = validate_submission(csv, expected_ids=frame.gpsno)
    rng = np.random.default_rng(20260924)
    stress = {}
    for fraction in [.2, 1.]:
        changed = frame.copy()
        rows = rng.choice(len(frame), int(len(frame) * fraction), replace=False)
        imu = [c for c in cols if dictionary.loc[c, 'feature_group'] == 'imu']
        changed.loc[changed.index[rows], imu] = np.nan
        for c in cols:
            if c.startswith('imu_coverage_'):
                changed.loc[changed.index[rows], c] = 0.
        stress[f'imu_dropout_{fraction:.0%}'] = rank_change(base, predict_bundle(bundle, changed))
    seed_stability = []
    if bundle.get('models'):
        transformed, _ = transform_rates(frame, bundle['rate_priors'])
        seed_predictions = [predict([model], transformed, bundle['columns'], bundle['spec']) for model in bundle['models']]
        for i in range(len(seed_predictions)):
            for j in range(i + 1, len(seed_predictions)):
                seed_stability.append({'seeds': [bundle['choice']['seeds'][i], bundle['choice']['seeds'][j]],
                                       **rank_change(seed_predictions[i], seed_predictions[j])})
    late = pd.read_parquet(OUT / 'retrospective_predictions.parquet')
    topk = {}
    for k in [25, 50, 100]:
        y = late.sort_values('candidate', ascending=False, kind='stable').label.iloc[:k]
        topk[str(k)] = {'positives': int(y.sum()), 'precision': float(y.mean()),
                       'recall': float(y.sum() / late.label.sum())}
    report = {'reproduction': validation, 'fixed_feature_windows': windows.value_counts().to_dict(),
              'stress': stress, 'auxiliary_seed_stability': seed_stability, 'retrospective_topk': topk,
              'limitations': ['Stress scenarios measure prediction sensitivity, not future AUC.',
                              'No adjustment to frozen models after this audit.',
                              '40-day probability calibration cannot be validated without future labels.']}
    (OUT / 'robustness.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    run()
