import unittest
import numpy as np
import pandas as pd
from src.features.shrunk_rates import GROUPS, fit_rate_priors, transform_rates
from src.prediction.optimize import rank, validate_cohort
from src.prediction.ensemble import fit_platt_calibrator, calibrate_probability


class FinalPredictionTests(unittest.TestCase):
    def frame(self):
        frame = pd.DataFrame({'distance_km_14d': [10., 100., 0., np.nan],
                              'run_time_hours_14d': [1., 10., 0., np.nan]})
        for group in GROUPS:
            frame[group+'_count_14d'] = [1., 5., 10., 10.]
        return frame

    def test_shrinkage_uses_training_prior_and_preserves_missing(self):
        train = self.frame()
        prior = fit_rate_priors(train)
        test = train.copy()
        test.loc[0, 'severe_history_count_14d'] = 1000.
        got, columns = transform_rates(test, prior)
        self.assertEqual(len(columns), 7)
        self.assertAlmostEqual(prior['severe_history']['mu'], 6/110)
        expected = (1000+55*6/110)/(10+55)
        self.assertAlmostEqual(got.severe_history_shrunk_rate.iloc[0], expected)
        self.assertTrue(got[columns].iloc[2:].isna().all().all())
        # A changed validation cohort cannot refit priors or affect this row.
        single, _ = transform_rates(test.iloc[:1], prior)
        pd.testing.assert_series_equal(single.iloc[0], got.iloc[0])

    def test_rank_ties(self):
        np.testing.assert_allclose(rank([3, 1, 1]), [1, .5, .5])

    def test_cohort_rejects_partial_duplicate_and_mixed_dates(self):
        bundle = {'cohort_ids': ['a', 'b']}
        frame = pd.DataFrame({'gpsno': ['a', 'b'], 'feature_end': ['2026-07-31']*2})
        validate_cohort(bundle, frame.iloc[::-1])
        for altered in [frame.iloc[:1], pd.concat([frame, frame.iloc[:1]]),
                        frame.assign(feature_end=['2026-07-31', '2026-07-30']),
                        frame.assign(feature_end=['2026-07-31', None])]:
            with self.assertRaises(ValueError):
                validate_cohort(bundle, altered)

    def test_regularized_calibration_preserves_order(self):
        x = np.linspace(0, 1, 100)
        y = (x > .9).astype(int)
        model = fit_platt_calibrator(y, x, regularization_c=1.)
        self.assertGreater(model.coef_[0, 0], 0)
        self.assertTrue(np.all(np.diff(calibrate_probability(model, x)) > 0))
