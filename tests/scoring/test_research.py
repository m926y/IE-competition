import unittest
import numpy as np
import pandas as pd
from src.scoring.additive import basis, contributions, fit, DIMS
from src.scoring.scoring_rules import DEFAULT_WEIGHTS, validate_weights
from src.scoring.risk_evidence import difference, group_counts, hierarchical_intervals


class ResearchTests(unittest.TestCase):
    def test_optimizer_obeys_strict_business_constraints(self):
        rng=np.random.default_rng(42)
        details=pd.DataFrame({d+'_penalty':rng.uniform(0,60 if d=='imu_evidence' else 100,60) for d in DIMS})
        y=(details.history_penalty>80).astype(int)
        config=fit(details,y,y,'percentile',{},DEFAULT_WEIGHTS)
        validate_weights(config['weights'])
        self.assertTrue(np.asarray(config['coefficients']).min()>=-1e-10)

    def test_monotone_functions_and_nonnegative_deductions(self):
        config={'mapping':'percentile','coefficients':[[.01,.02,.03]]*6}
        details=pd.DataFrame({d+'_penalty':np.linspace(0,60 if d=='imu_evidence' else 100,101) for d in DIMS})
        values=contributions(details,config)
        self.assertTrue(np.all(values>=0))
        self.assertTrue(np.all(np.diff(values,axis=0)>=-1e-12))
        self.assertTrue(np.allclose(basis(np.array([0.,1.])),[[0,0,0],[1,1,1]]))

    def test_direct_difference_is_antisymmetric(self):
        groups=group_counts([0,1,0,0,1,1],['A','A','A','B','B','B'],['A','B'])
        forward=difference(groups['B'],groups['A'])
        backward=difference(groups['A'],groups['B'])
        self.assertAlmostEqual(forward['risk_difference'],-backward['risk_difference'])
        self.assertTrue(np.allclose(forward['difference_ci95'],-np.array(backward['difference_ci95'])[::-1]))

    def test_hierarchical_posterior_exposes_uncertainty(self):
        groups={'low':{'n':100,'positive':1},'high':{'n':50,'positive':6}}
        report=hierarchical_intervals(groups)
        for item in report['groups'].values():
            lo,hi=item['credible_interval95']
            self.assertTrue(0<lo<item['posterior_mean']<hi<1)
        self.assertLess(report['groups']['low']['posterior_mean'],report['groups']['high']['posterior_mean'])
