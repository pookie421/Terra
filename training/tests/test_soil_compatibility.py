"""Frozen provider-shaped payloads; no network or current-date dependencies."""
import unittest
from types import SimpleNamespace
from backend.realtime.global_ingestor import GlobalIngestor
from backend.realtime.soilgrids import estimate_pedotransfer, soilgrids_to_conventional
from backend.physics.multi_causal_engine import MultiCausalEngine, PhysicsInputs
from test_reference_equations import NetworkDisabled


class SoilCompatibility(NetworkDisabled):
    def setUp(self):
        super().setUp()
        self.ingestor = GlobalIngestor.__new__(GlobalIngestor)

    def test_raw_rest_and_wcs_units(self):
        raw = {'properties': {'layers': [
            {'name': name, 'depths': [{'values': {'mean': value}}]}
            for name,value in [('bdod',145),('soc',250),('sand',450),('clay',200)]]}}
        result = self.ingestor._soil_property_values(raw)
        self.assertEqual(result, {'bdod_kg_m3':1450., 'soc_g_kg':25., 'sand_g_kg':450., 'clay_g_kg':200.})
        self.assertEqual(soilgrids_to_conventional('soc',250),25.)
        hydro = self.ingestor._combine_hydrology(SimpleNamespace(soil_moisture_0_100cm_m3m3=.3),None,raw)
        self.assertAlmostEqual(hydro.soil_properties['organic_matter_fraction'],.0431)

    def test_static_fallback_and_consistent_partial_soil(self):
        self.assertEqual(estimate_pedotransfer(30,0,8,10),estimate_pedotransfer(30,0,8,90))
        raw = {'bdod_kg_m3':1000.,'sand_g_kg':500.,'clay_g_kg':200.}
        h = self.ingestor._combine_hydrology(SimpleNamespace(soil_moisture_0_100cm_m3m3=.3),None,raw,allow_estimated_soil=True)
        self.assertEqual(h.soil_properties['bulk_density_g_cm3'],1.)
        self.assertEqual(h.soil_properties['sand_fraction'],.5)
        inputs = PhysicsInputs(.5,.2,.03,1.,35.,8.,h.soil_saturation_pct/100,0.,24.,0.,10.,8.,0.,0.,.5,0.,9999.,9999.)
        state = MultiCausalEngine().compute(inputs)
        self.assertAlmostEqual(state.theta_s[0],h.soil_porosity_estimate)
        self.assertAlmostEqual(state.theta_s[0]*state.local_saturation[0],.3)

    def test_incompatible_water_is_not_clipped_or_replaced_by_index(self):
        raw = {'bdod_kg_m3':1473.595,'sand_g_kg':450.,'clay_g_kg':200.,'soc_g_kg':20.}
        power = {'properties':{'parameter':{'GWETROOT':{'20261104':.8}}}}
        h = self.ingestor._combine_hydrology(SimpleNamespace(soil_moisture_0_100cm_m3m3=.47384),power,raw)
        self.assertIsNone(h.soil_saturation_pct)
        self.assertEqual(h.blended_relative_wetness_pct,80.)
        self.assertEqual(h.moisture_compatibility,'incompatible_moisture_and_porosity')
        self.assertEqual(h.open_meteo_soil_moisture_m3m3,.47384)

    def test_static_prior_does_not_depend_on_other_viewport_cells(self):
        center=SimpleNamespace(cell_id='center',latitude=10.,longitude=20.,slope_deg=30.,profile_curvature=0.,twi=8.)
        distant=SimpleNamespace(cell_id='other',latitude=11.,longitude=21.,slope_deg=70.,profile_curvature=1.,twi=30.)
        terrain=SimpleNamespace(center_latitude=10.,center_longitude=20.,cells={'center':center})
        weather=SimpleNamespace(soil_moisture_0_100cm_m3m3=.3)
        first=self.ingestor._combine_hydrology(weather,None,None,terrain,allow_estimated_soil=True)
        terrain.cells['other']=distant
        second=self.ingestor._combine_hydrology(weather,None,None,terrain,allow_estimated_soil=True)
        self.assertEqual(first.soil_properties,second.soil_properties)
        self.assertEqual(first.soil_saturation_pct,second.soil_saturation_pct)
