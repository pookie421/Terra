import unittest
from dataclasses import replace
import numpy as np
from backend.physics.mechanisms import postfire,mamot_joint_strength
from backend.physics.multi_causal_engine import MultiCausalEngine,PhysicsInputs
from backend.physics.features import physics_feature_matrix
BASE=PhysicsInputs(.45,.2,.03,1.45,35.,8.,.3,0.,24.,0.,10.,8.,0.,0.,.5,0.,9999.,9999.)
class Mechanisms(unittest.TestCase):
 def test_mamot_published_endpoints_and_domain(self):
  self.assertAlmostEqual(float(mamot_joint_strength(-8,100)),851.3)
  self.assertAlmostEqual(float(mamot_joint_strength(-.5,100)),142.55)
  with self.assertRaises(ValueError):mamot_joint_strength(0,100)
 def test_postfire_requires_explicit_basin_inputs(self):
     with self.assertRaises(ValueError):
         postfire(.85,35.,.2,.45,65.,0.,2.)
     b=replace(BASE,burn_severity=.85,rain_15min_override_mm=10.,postfire_burned_steep_fraction=.5,postfire_mean_dnbr=500.,postfire_soil_erodibility_kf=.2,postfire_burned_area_km2=1.,postfire_basin_relief_m=100.)
     s=MultiCausalEngine().compute(b)
     self.assertGreater(s.p_failure_m1[0],.9)
     self.assertGreater(s.debris_volume_m3[0],0)
     self.assertFalse(MultiCausalEngine().compute(BASE).postfire_basin_inputs_available[0])
 def test_hydraulic_strength_changes_without_probability_union(self):
     e=MultiCausalEngine();dry=e.compute(BASE);wet=e.compute(replace(BASE,rain_24h_mm=300))
     self.assertGreater(wet.pore_pressure_head_m[0],dry.pore_pressure_head_m[0])
     self.assertLess(wet.fs_dynamic[0],dry.fs_dynamic[0])
     np.testing.assert_array_equal(wet.p_failure_physical,wet.p_failure_fosm)
     np.testing.assert_array_equal(wet.suction_cohesion_kpa,0.)
 def test_rain_and_melt_features_separate_and_cryo_bounded(self):
  inp=replace(BASE,elevation_m=2500,snow_depth_m=.4,warming_delta_c=8,freeze_thaw_cycles_7d=3)
  s=MultiCausalEngine().compute(inp);x=physics_feature_matrix(inp,s,np.array([0.]))
  self.assertEqual(x[0,7],0.)
  self.assertAlmostEqual(x[0,8],float(s.snowmelt_mm_day[0]),places=4)
  self.assertEqual(s.cryo_perched_head_m[0],0)
  self.assertTrue(np.all(s.pore_pressure_head_m<=s.regolith_depth_m*np.cos(np.radians(s.effective_slope_deg))**2+1e-12))
