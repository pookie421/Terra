"""Optimization and provenance regressions independent of retained model outputs."""
import unittest
from types import SimpleNamespace
import numpy as np
from backend.ml.physics_informed import ContextModel,NAMES,MONOTONIC,matrix
from backend.physics.reference_context import columns

class FeatureFusion(unittest.TestCase):
    def test_extreme_physical_feature_does_not_lock_optimizer(self):
        groups=np.repeat(np.arange(30),4);y=np.tile([0,0,1,1],30)
        x=np.zeros((len(y),len(NAMES)));x[:,0]=y*100.;x[:,1]=y*200.
        x[:,NAMES.index('fosm_reference_score')]=1e-25
        x[:,NAMES.index('fosm_log10_p')]=-25.
        x[:,NAMES.index('gravity_release_admissibility')]=1.
        model=ContextModel().fit(x,y,groups)
        score=model.predict_learned_proba(x)
        self.assertGreater(float(score[y==1].mean()-score[y==0].mean()),.5)
        for item,(tree,_) in zip(model.tuning,model.folds):
            self.assertGreater(item['initial_hessian'],.2)
            self.assertFalse(set(item['fit_components'])&set(item['calibration_components']))
            self.assertEqual(tree.get_params()['base_score'],.5)
        self.assertEqual(len(NAMES),len(MONOTONIC))

    def test_newmark_requires_positive_pga_and_stable_block(self):
        cell=SimpleNamespace(cell_id='test',slope_deg=30.,twi=8.,profile_curvature=0.)
        window={'status':'AVAILABLE','rain_24h_mm':0.,'rain_prior_6d_mm':0.}
        base=columns([cell],window)[0]
        self.assertIsNone(base['newmark_displacement_cm'])
        shaken=columns([cell],{**window,'pga_reference_g':.5})[0]
        if shaken['fs_mid']>1:self.assertGreaterEqual(shaken['newmark_displacement_cm'],0.)
        self.assertEqual(base['fs_mid'],shaken['fs_mid'])
        self.assertEqual(matrix([0.,0.,.2,30.,8.,0.],shaken).shape,(len(NAMES),))

    def test_replay_name_whitelist(self):
        from backend.api.replay import load_replay
        with self.assertRaises(KeyError):load_replay('../../.env')

if __name__=='__main__':unittest.main()
