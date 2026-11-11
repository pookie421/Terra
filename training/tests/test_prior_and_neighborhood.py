import unittest
from types import SimpleNamespace
from dataclasses import replace
import numpy as np
from backend.ml.global_xgb import GlobalMonotonicXGB, prior_adjust_probability, mechanical_stability_gate
from backend.ml.build_real_labeled_dataset import populate_physics_features
from backend.physics.multi_causal_engine import MultiCausalEngine
from test_mechanisms import BASE

class PriorAndNeighborhood(unittest.TestCase):
    def test_prior_shift_identity_prevalence_and_order(self):
        p=np.array([0.,.01,.2,.676,.9,1.])
        out=prior_adjust_probability(p,.676,.15)
        self.assertAlmostEqual(out[3],.15)
        self.assertTrue(np.all(np.diff(out)>0))
        np.testing.assert_allclose(prior_adjust_probability(p,.3,.3),p)
        np.testing.assert_equal(out[[0,-1]],[0.,1.])
        for prior in (0.,1.,np.nan):
            with self.assertRaises(ValueError):prior_adjust_probability(p,prior,.15)

    def test_prior_adjust_each_fold_before_averaging_and_uncertainty(self):
        class Fold:
            def __init__(self,p):self.p=p
            def predict_proba(self,x):return np.tile([1-self.p,self.p],(len(x),1))
        model=GlobalMonotonicXGB(field_prevalence=.15);model.model=SimpleNamespace(calibrated_classifiers_=[Fold(.2),Fold(.8)])
        model.calibration_fold_prevalences_=[.2,.8]
        x=np.ones((2,14))
        np.testing.assert_allclose(model.predict_proba(x),.15)
        np.testing.assert_allclose(model.predict_calibration_population_proba(x),.5)
        np.testing.assert_allclose(model.ensemble_std(x),0.,atol=1e-15)

    def test_neighborhood_is_symmetric_excludes_distant_cliff(self):
        def cell(k,lat,slope,twi):return SimpleNamespace(cell_id=k,latitude=lat,longitude=7.,slope_deg=slope,twi=twi,profile_curvature=.001,elevation_m=100.)
        terrain=SimpleNamespace(cells={c.cell_id:c for c in [cell('local',46.,10.,4.),cell('near_scarp',46.004,35.,10.),cell('distant',46.03,70.,20.)]})
        results=[]
        for label in (0,1):
            row=dict(latitude=46.,longitude=7.,label=label,soil_moisture_72h_max_m3m3=.2,rain_effective_storm_mm=0.,snow_depth_72h_max_m=0.)
            populate_physics_features(row,terrain);results.append(row)
            self.assertEqual(row['representative_terrain_cell_id'],'local')
            self.assertEqual(row['slope_max_deg'],10.);self.assertEqual(row['twi'],4.)
        self.assertEqual(results[0]['physics_inputs_json'],results[1]['physics_inputs_json'])

    def test_dry_storage_gate_does_not_erase_explicit_groundwater(self):
        engine=MultiCausalEngine();inp=replace(BASE,soil_saturation_fraction=.99,rain_24h_mm=0.,snow_depth_m=0.)
        np.testing.assert_equal(engine.compute(inp).pore_pressure_head_m,0.)
        self.assertGreater(engine.compute(replace(inp,antecedent_pressure_head_m=.3)).pore_pressure_head_m[0],0.)
        self.assertGreater(engine.compute(replace(inp,rain_24h_mm=50.)).pore_pressure_head_m[0],0.)
        self.assertEqual(mechanical_stability_gate([20.],[1.39],[.1])[0],1.)
        self.assertGreater(mechanical_stability_gate([35.],[.9],[.99])[0],.99)
