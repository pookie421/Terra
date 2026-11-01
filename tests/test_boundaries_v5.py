import asyncio
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from backend.physics.multi_causal_engine import MultiCausalEngine
from backend.physics.alpine_weather import apply_alpine_weather,cryospheric_eligibility
from backend.physics.cascading import dimensionless_blockage_index,larsen_volume
from backend.ml.global_xgb import GlobalMonotonicXGB,TrainingDataError
from backend.ml.pinls import pinls_filter
from backend.realtime.global_ingestor import GlobalIngestor
from test_mechanisms import BASE

class PhysicalBoundaries(unittest.TestCase):
 def test_warm_hill_warming_does_not_create_cryo(self):
  b=replace(BASE,air_temperature_c=18,soil_temperature_c=16,snow_depth_m=0,latitude_deg=-33.7)
  a=apply_alpine_weather(b,[900,1000]);w=apply_alpine_weather(b,[900,1000],8)
  sa=MultiCausalEngine().compute(a);sw=MultiCausalEngine().compute(w)
  self.assertFalse(sw.cryospheric_eligible.any())
  np.testing.assert_array_equal(w.snow_depth_m,0)
  np.testing.assert_allclose(sa.fs_dynamic,sw.fs_dynamic)
  np.testing.assert_array_equal(sw.thaw_factor,1)
 def test_boundary_is_baseline_and_latitude_dependent(self):
  np.testing.assert_array_equal(cryospheric_eligibility(0,10,np.array([2500,2500]),np.array([0,50])),[False,True])
  self.assertTrue(cryospheric_eligibility(.006,18,1000,0))
 def test_barren_burn_has_no_physical_effect(self):
  b=replace(BASE,ndvi_norm=0,rain_24h_mm=100)
  e=MultiCausalEngine();a=e.compute(b);burn=e.compute(replace(b,burn_severity=1))
  np.testing.assert_allclose(a.fs_dynamic,burn.fs_dynamic)
  np.testing.assert_array_equal(burn.effective_burn_severity,0)
  np.testing.assert_array_equal(burn.p_failure_m1,0)
 def test_lateral_width_limit_and_hollow_factor(self):
  e=MultiCausalEngine();small=e.compute(BASE);wide=e.compute(replace(BASE,block_width_m=1e12));hollow=e.compute(replace(BASE,hollow_confinement_ratio=2))
  np.testing.assert_allclose(small.fs_dynamic,wide.fs_dynamic)
  np.testing.assert_allclose(hollow.lateral_resistance_kpa,small.lateral_resistance_kpa/2)
 def test_dbi_units_and_invalid_geometry(self):
  self.assertAlmostEqual(dimensionless_blockage_index(100e6,20,1e6),np.log10(2000))
  self.assertGreater(larsen_volume(100),0)
  with self.assertRaises(ValueError):dimensionless_blockage_index(1,0,1)
 def test_pinls_selection_does_not_rewrite_features(self):
  rows=[dict(event_id=str(i),label=y,fs_static_prior=fs,slope_max_deg=25) for i,(y,fs) in enumerate([(0,1.1),(0,1.21),(1,.8)])]
  selected,report=pinls_filter(rows)
  self.assertEqual([r['event_id'] for r in selected],['1','2']);self.assertEqual(rows[0]['fs_static_prior'],1.1)
 def test_conformal_rank_and_group_isolation(self):
  m=GlobalMonotonicXGB();p=np.arange(10)/10;y=np.zeros(10);groups=list(map(str,range(10)))
  result=m.calibrate_conformal(p,y,groups,['training'])
  self.assertEqual(result['q'],.9)
  intervals,width,_=m.conformal_intervals(np.array([.1,.9]));self.assertTrue(np.all((intervals>=0)&(intervals<=1)))
  with self.assertRaises(TrainingDataError):m.calibrate_conformal(p,y,groups,['1'])
  with self.assertRaises(ValueError):m.calibrate_conformal(p,y,groups,['training'],alpha=1)

class ProviderDeadline(unittest.IsolatedAsyncioTestCase):
 async def test_optional_providers_have_independent_two_second_deadlines(self):
  ingestor=GlobalIngestor(provider_timeout_s=25)
  async def slow():await asyncio.sleep(10)
  start=asyncio.get_running_loop().time()
  results=await asyncio.gather(*(ingestor._capture(k,k,slow()) for k in ['overpass','soilgrids']))
  self.assertLess(asyncio.get_running_loop().time()-start,2.5)
  self.assertTrue(all(p is None and not status.available for p,status in results))
