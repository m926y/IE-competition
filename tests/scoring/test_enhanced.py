import unittest
import numpy as np
import pandas as pd
from src.scoring.enhanced import (
    decayed_exposure,penalty_matrix,learn_weights,enhanced_score,DIMS,
    DEFAULT_LEVEL_POLICY,
    wilson_interval,
)
from src.scoring.scoring_rules import DEFAULT_WEIGHTS, validate_weights


class EnhancedTests(unittest.TestCase):
    def test_level_interval_exposes_small_sample_uncertainty(self):
        low, high = wilson_interval(6, 50)
        self.assertAlmostEqual(low, .0561760015)
        self.assertAlmostEqual(high, .2380481738)
        self.assertIsNone(wilson_interval(0, 0))

    def test_reason_metadata_and_text_agree_when_contributions_tie(self):
        from src.scoring.event_taxonomy import DIMENSION_LABELS
        row = {'gpsno': 'a', 'confidence_level': 'high'}
        for dim in DIMS:
            row[dim+'_penalty'] = 10.
            row[dim+'_contribution'] = 1.
            row[dim+'_event_count'] = 2
        # Simulate a stale upstream ordering that must be rebuilt together.
        row['top_reason_1_dimension'] = 'history'
        result, _ = enhanced_score(pd.DataFrame([row]), DEFAULT_WEIGHTS, [20, 40, 60])
        first = result.iloc[0]
        for k in [1, 2, 3]:
            dim = first[f'top_reason_{k}_dimension']
            self.assertTrue(first[f'top_reason_{k}'].startswith(DIMENSION_LABELS[dim]))
            self.assertEqual(first[f'top_reason_{k}_penalty_points'], first[dim+'_contribution'])

    def test_exposure_excludes_future_and_uses_same_decay(self):
        daily = pd.DataFrame({'gpsno': ['a'] * 3,
            'event_date': pd.to_datetime(['2026-06-01', '2026-06-14', '2026-07-01']),
            'distance_km': [100., 100., 99999.], 'run_time_hours': [1., 1., 99999.],
            'moving_rows': [1, 1, 1]})
        got = decayed_exposure(daily, '2026-06-01', '2026-06-14')
        expected = 2 ** (-13.5/14) + 2 ** (-.5/14)
        self.assertAlmostEqual(got.loc['a', 'distance_km'], 100 * expected)
        self.assertAlmostEqual(got.loc['a', 'active_days'], expected)

    def test_missing_imu_is_neutral_not_perfect(self):
        frame = pd.DataFrame({d + '_penalty': [10.] for d in DIMS})
        frame['imu_evidence_penalty'] = np.nan
        self.assertAlmostEqual(penalty_matrix(frame)[0, -1], .3)

    def test_pairwise_weights_obey_constraints(self):
        rng = np.random.default_rng(42)
        frame = pd.DataFrame(rng.uniform(0, 100, (40, 6)), columns=[d + '_penalty' for d in DIMS])
        y = (frame.history_penalty > 75).astype(int)
        weights, status = learn_weights(frame, y, y)
        validate_weights(weights)
        self.assertEqual(status['status'], 'fit')
        self.assertTrue(all(abs(weights[d] - DEFAULT_WEIGHTS[d]) <= .100001 for d in DIMS))
        self.assertEqual(status['evidence_status'],'low_positive_count')

    def test_missing_imu_changes_floor_not_behavior_score(self):
        row={'gpsno':'a','confidence_level':'low'}
        for d in DIMS:
            row[d+'_penalty']=10.; row[d+'_contribution']=DEFAULT_WEIGHTS[d]*10
            row[d+'_event_count']=0
        row['imu_evidence_penalty']=np.nan
        row['imu_evidence_contribution']=0.
        got,_=enhanced_score(pd.DataFrame([row]),DEFAULT_WEIGHTS,[20,40,60])
        self.assertAlmostEqual(got.safety_score.iloc[0],90.5)
        self.assertAlmostEqual(got.uncertainty_reserve.iloc[0],1.5)
        self.assertAlmostEqual(got.confidence_adjusted_floor.iloc[0],89.)

    def test_rank_level_policy_has_stable_capacity_and_preserves_scores(self):
        rows=[]
        for i in range(500):
            row={'gpsno':f'{i:03d}','confidence_level':'high'}
            for d in DIMS:
                row[d+'_penalty']=i/10
                row[d+'_contribution']=DEFAULT_WEIGHTS[d]*i/10
                row[d+'_event_count']=0
            rows.append(row)
        details=pd.DataFrame(rows)
        absolute,_=enhanced_score(details,DEFAULT_WEIGHTS,[10,20,30])
        ranked,_=enhanced_score(details,DEFAULT_WEIGHTS,[10,20,30],DEFAULT_LEVEL_POLICY)
        counts=ranked.risk_level.value_counts().to_dict()
        self.assertEqual(counts,{'B':200,'C':150,'A':100,'D':50})
        self.assertTrue(np.allclose(absolute.safety_score,ranked.safety_score))
        self.assertEqual(ranked.iloc[0].risk_rank_percentile,.002)
        self.assertEqual(ranked.iloc[-1].risk_rank_percentile,1.)

    def test_rank_level_policy_rejects_invalid_cutoffs(self):
        row={'gpsno':'a','confidence_level':'high'}
        for d in DIMS:
            row[d+'_penalty']=10.; row[d+'_contribution']=DEFAULT_WEIGHTS[d]*10
            row[d+'_event_count']=0
        with self.assertRaises(ValueError):
            enhanced_score(pd.DataFrame([row]),DEFAULT_WEIGHTS,
                           level_policy={'type':'rank_percentile','cutoffs':[.2,.2,.9]})


if __name__ == '__main__':
    unittest.main()
