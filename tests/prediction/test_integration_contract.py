import unittest
import numpy as np
import pandas as pd
from src.prediction.data_loader import select_feature_columns,align_prediction_features
from src.prediction.integrated import probability_at_horizon

class IntegrationContractTests(unittest.TestCase):
    def test_all_a_label_and_metadata_fields_excluded(self):
        frame=pd.DataFrame({'gpsno':['a','b'],'label':[0,1],'accident_label':[0,1],
            'near_miss_label':[1,1],'label_horizon_days':[14,40],
            'feature_window_days':[14,21],'window_role':['aux','main'],'signal':[1.,2.]})
        self.assertEqual(select_feature_columns(frame),['signal'])

    def test_missing_feature_fails(self):
        with self.assertRaises(ValueError):
            align_prediction_features(pd.DataFrame({'gpsno':['a']}),['signal'])

    def test_horizon_respects_two_near_miss_rule_and_order(self):
        p=np.array([.01,.1,.8])
        np.testing.assert_allclose(probability_at_horizon(p,0,14),p,atol=1e-8)
        x=probability_at_horizon(p,0,40)
        self.assertTrue((np.diff(x)>0).all())
        self.assertTrue((x>=p).all())
        np.testing.assert_allclose(probability_at_horizon(p,1,40),1-(1-p)**(40/14))
