"""Network-independent regressions for the stabilization changes."""
import asyncio
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import httpx
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from backend.physics.multi_causal_engine import PhysicsInputs, MultiCausalEngine
from backend.physics.features import physics_feature_matrix
from backend.ml.global_xgb import mechanical_stability_gate
from backend.ml.validation_suite import historical_hindcast_metrics
from backend.realtime.coolr import nearby_count, local_catalog_distances, haversine_km
from backend.realtime.cache import AsyncTTLCache
from backend.llm.qwen_agent import TerraShieldQwenAgent as Agent
from backend.api.reliability import reliability_report

BASE=PhysicsInputs(.45,.2,.03,1.45,35.,8.,.3,0.,24.,0.,10.,8.,0.,0.,.5,0.,9999.,9999.)

class LocalFeatures(unittest.TestCase):
    def test_neighbor_slope_does_not_overwrite_local_fs_or_slope(self):
        engine=MultiCausalEngine()
        inputs=replace(BASE,slope_deg=np.array([2.,40.]),profile_curvature=np.zeros(2))
        state=engine.compute(inputs)
        matrix=physics_feature_matrix(inputs,state,np.zeros(2))
        changed=replace(inputs,slope_deg=np.array([2.,70.]))
        after=physics_feature_matrix(changed,engine.compute(changed),np.zeros(2))
        np.testing.assert_allclose(matrix[0],after[0])
        np.testing.assert_allclose(matrix[:,4],[2.,40.])
        np.testing.assert_allclose(matrix[:,0],state.fs_dynamic)

    def test_legacy_gate_does_not_transform_calibrated_scores(self):
        np.testing.assert_array_equal(mechanical_stability_gate([0.,2.,8.5,35.],[5.,2.,1.16,.8]),1.)

    def test_recall_40_is_not_top_10_percent(self):
        y=np.array([1]*24+[0]*24);p=np.linspace(1,0,48)
        m=historical_hindcast_metrics(p,y,[str(i//2) for i in range(48)]).metrics
        self.assertEqual(m['recall_at_40_k'],40)
        self.assertEqual(m['recall_at_40'],1)
        self.assertEqual(m['recall_at_k_k'],5)
        self.assertAlmostEqual(m['recall_at_k'],5/24)

class CoolrTelemetry(unittest.IsolatedAsyncioTestCase):
    async def test_envelope_parameters_and_real_zero_count(self):
        requests=[]
        def handler(request):
            requests.append(request)
            return httpx.Response(200,json={'count':0})
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            result=await nearby_count(client,11.,76.,AsyncTTLCache())
        query=requests[0].url.params
        self.assertEqual(query['where'],'1=1')
        self.assertEqual(query['spatialRel'],'esriSpatialRelIntersects')
        self.assertEqual(query['geometry'],'75.55,10.55,76.45,11.45')
        self.assertEqual(query['returnCountOnly'],'true')
        self.assertIs(type(result['count']),int)
        self.assertEqual(result['count'],0)
        self.assertEqual(result['raw_count_response'],{'count':0})

    async def test_offline_fallback_discloses_coverage(self):
        local={'cached_count_50km':2,'nasa_coolr_regional_count_100km':4,'nearest_coolr_event_km':12.,'complete_global_catalog':False}
        def handler(request):raise httpx.ConnectError('offline',request=request)
        with patch('backend.realtime.coolr.local_catalog_distances',return_value=local):
            async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
                result=await nearby_count(client,11.,76.,AsyncTTLCache())
        self.assertEqual(result['count'],2)
        self.assertEqual(result['quality'],'partial_local_catalog_fallback_50km')
        self.assertFalse(result['complete_global_catalog'])

    def test_local_distance_deduplicates_sources_and_excludes_controls(self):
        row={'source_name':'NASA COOLR','event_record_type':'cited_manual_point_buffer_cell','source_event_id':'event-a','event_id':'cell-a','latitude':0.,'longitude':.1}
        rows=[row,{**row,'event_id':'cell-b','longitude':.2},{**row,'event_id':'control','source_event_id':'control','event_record_type':'temporal_absence_sample','longitude':0.}]
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'events.parquet';pq.write_table(pa.Table.from_pylist(rows),path)
            result=local_catalog_distances(0.,0.,path)
        self.assertEqual(result['nasa_coolr_regional_count_100km'],1)
        self.assertAlmostEqual(result['nearest_coolr_event_km'],haversine_km(0,0,0,.1))

class BriefingGuard(unittest.TestCase):
    def setUp(self):
        self.scan={'location':{'display_name':'Test town'},'summary':{'max_p_failure':.812345,'min_fs_dynamic':.934567,'critical_hotspots_count':2,'total_households_exposed':None,'kinematics_top_hotspot':{'h3_index':'8928308280fffff','slope_deg':30.,'peak_velocity_ms':12.34567,'failure_mode':'rapid','dominant_cause':'cryospheric'}}}
        self.canonical=Agent._briefing(self.scan,'LOCAL_EOC').strip()

    def test_rounded_template_is_grounded(self):
        self.assertTrue(Agent._numeric_parity_audit(self.canonical,self.scan))
        text,passed,status=Agent._guard_briefing(self.canonical,self.scan,'LOCAL_EOC')
        self.assertTrue(passed);self.assertEqual(status,'VERIFIED_LLM_PROSE')

    def test_numbers_names_and_percentages_are_intercepted(self):
        for addition in (' Evacuate 999 households.',' Failure chance 99.99%.'):
            text,passed,status=Agent._guard_briefing(self.canonical+addition,self.scan,'LOCAL_EOC')
            self.assertEqual(text,self.canonical);self.assertTrue(passed)
            self.assertEqual(status,'REPLACED_UNVERIFIED_PROSE')

    def test_invalid_template_fails_closed(self):
        with patch.object(Agent,'_briefing',return_value='Invented 999999.'):
            text,passed,status=Agent._guard_briefing(None,self.scan,'LOCAL_EOC')
        self.assertEqual(status,'WITHHELD_INVALID_TEMPLATE')
        self.assertFalse(passed);self.assertNotIn('999999',text)

    def test_percentage_and_lists_preserve_prose(self):
        candidate='1. Failure score 81.23%.\n2. Slope 30 degrees.'
        text,passed,status=Agent._guard_briefing(candidate,self.scan,'LOCAL_EOC')
        self.assertEqual(text,candidate);self.assertTrue(passed)
        self.assertEqual(status,'VERIFIED_LLM_PROSE')

    def test_no_numeric_half_unit_tolerance(self):
        self.assertFalse(Agent._numeric_parity_audit('Value 30.4',self.scan))

class Reliability(unittest.TestCase):
    def test_missing_and_available_feeds_are_distinct(self):
        source=SimpleNamespace(available=True)
        scan=SimpleNamespace(sources={'terrarium_dem':source,'open_meteo':source},hydrology=SimpleNamespace(soil_property_quality='estimated'),terrain=True,coolr_nearby={'count':0})
        report=reliability_report(scan)
        self.assertEqual(report['soil_source'],'PEDOTRANSFER_TOPOGRAPHIC_ESTIMATE')
        self.assertEqual(report['weather_source'],'LIVE_OR_CACHED_OPEN_METEO_BEST_MATCH')
        self.assertEqual(report['seismic_source'],'UNAVAILABLE')
        self.assertEqual(report['operational_confidence_tier'],'UNVALIDATED')

    def test_successful_http_does_not_hide_soil_fallback(self):
        scan=SimpleNamespace(sources={'soilgrids':SimpleNamespace(available=True)},hydrology=SimpleNamespace(soil_property_source='topographic_climatic_pedotransfer_estimate',soil_property_quality='estimated_not_measured'),terrain=True,coolr_nearby={'count':0})
        self.assertEqual(reliability_report(scan)['soil_source'],'PEDOTRANSFER_TOPOGRAPHIC_ESTIMATE')
        scan.hydrology.soil_property_source='hybrid_soilgrids_and_pedotransfer'
        scan.hydrology.soil_property_quality='mixed_measured_prediction_and_estimate'
        self.assertEqual(reliability_report(scan)['soil_source'],'MIXED_ISRIC_AND_PEDOTRANSFER')

if __name__=='__main__':unittest.main()
