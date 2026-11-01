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
 def test_fire_gates_and_rain_response(self):
  values=[postfire(np.array([b]),np.array([s]),.2,.45,r,0,2.) for b,s,r in [(0,35,300),(.85,2,300),(.85,35,0),(.85,35,65)]]
  self.assertEqual(values[0][0][0],0);self.assertEqual(values[1][0][0],0)
  self.assertGreater(values[3][0][0],values[2][0][0]);self.assertGreater(values[3][1][0],0)
 def test_union_preserves_mc_and_suction_decays(self):
  e=MultiCausalEngine();dry=e.compute(BASE);wet=e.compute(replace(BASE,rain_24h_mm=300))
  self.assertGreater(dry.suction_cohesion_kpa[0],wet.suction_cohesion_kpa[0])
  fire=e.compute(replace(BASE,burn_severity=.85,rain_24h_mm=65,pga_g=.4))
  np.testing.assert_allclose(fire.p_failure_physical,1-(1-fire.p_failure_fosm)*(1-fire.p_failure_seismic)*(1-fire.p_failure_m1))
  np.testing.assert_allclose(fire.fs_dynamic,np.minimum(fire.fs_mohr_coulomb,fire.fs_postfire_yield_index))
 def test_equivalent_flux_shared_and_cryo_bounds(self):
  inp=replace(BASE,elevation_m=2500,snow_depth_m=.4,warming_delta_c=8,freeze_thaw_cycles_7d=3)
  s=MultiCausalEngine().compute(inp);x=physics_feature_matrix(inp,s,np.array([0.]))
  self.assertAlmostEqual(x[0,7],float(s.equivalent_water_mm[0]),places=4)
  self.assertGreater(s.cryo_perched_head_m[0],0)
  self.assertTrue(np.all(s.pore_pressure_head_m<=s.regolith_depth_m*np.cos(np.radians(s.effective_slope_deg))**2+1e-12))
