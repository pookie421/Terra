"""Regression tests for time windows, feature parity, attribution and metrics."""
import unittest
from dataclasses import replace
from datetime import datetime,timedelta
from types import SimpleNamespace
import numpy as np
from backend.ml.archive_weather import summarize_archive,zero_crossings
from backend.ml.audit_dataset import mechanical_controls
from backend.ml.global_xgb import FEATURE_NAMES,spatial_group_splits,make_base_estimator
from backend.ml.build_real_labeled_dataset import populate_physics_features
from backend.ml.validation_suite import run_lhasa_benchmark_suite
from backend.physics.features import physics_feature_matrix
from backend.physics.causal_attribution import CAUSES,group_shap_by_cause,explain_causes
from backend.physics.multi_causal_engine import MultiCausalEngine,PhysicsInputs
from test_mechanisms import BASE

class UnifiedPipeline(unittest.TestCase):
    def test_storm_window_captures_cross_midnight_peak_without_future_data(self):
        start=datetime(2020,1,1)
        times=[(start+timedelta(hours=i)).isoformat(timespec='minutes') for i in range(216)]
        rain=[0.]*216
        rain[132:156]=[5.]*24  # Jan 6 noon through Jan 7 morning; event day Jan 8 is dry.
        rain[192:216]=[100.]*24  # Future rainfall must never enter the features.
        data={'hourly':{'time':times,'rain':rain,'snow_depth':[0.]*216,'soil_moisture_0_to_7cm':[.2]*216}}
        out=summarize_archive(data,'2020-01-08')
        self.assertEqual(out['rain_event_day_mm'],0.)
        self.assertEqual(out['rain_peak_24h_3d_mm'],120.)
        self.assertEqual(out['rain_7d_total_mm'],120.)
        self.assertEqual(out['rain_effective_storm_mm'],120.)
        data['hourly']['rain'][30]=10.
        self.assertEqual(summarize_archive(data,'2020-01-08')['rain_effective_storm_mm'],121.5)
        data['hourly']['rain'][133]=None
        self.assertIsNone(summarize_archive(data,'2020-01-08')['rain_peak_24h_3d_mm'])
        self.assertIsNone(summarize_archive(data,'2020-01-08')['rain_effective_storm_mm'])

    def test_storm_softening_preserves_gentle_and_normal_rain_strength(self):
        engine=MultiCausalEngine()
        for slope,rain in [(19.,300.),(30.,35.)]:
            state=engine.compute(replace(BASE,slope_deg=slope,rain_24h_mm=rain,pore_pressure_override_m=100.))
            np.testing.assert_array_equal(state.effective_soil_cohesion_kpa,state.soil_cohesion_kpa)
        storm=engine.compute(replace(BASE,slope_deg=35.,rain_24h_mm=180.,pore_pressure_override_m=100.))
        np.testing.assert_array_equal(storm.effective_soil_cohesion_kpa,storm.soil_cohesion_kpa)
        self.assertTrue(np.all(storm.effective_soil_cohesion_kpa >= .65*storm.soil_cohesion_kpa))

    def test_no_model_labeled_anchors_in_default_baseline(self):
        from backend.ml.global_xgb import training_sample_weights
        self.assertEqual(mechanical_controls(),[])
        np.testing.assert_allclose(training_sample_weights(['observed','synthetic:legacy','synthetic:regime:test']),[1.,.25,.60])

    def test_crossings_ignore_zero_plateaus_and_break_at_missing(self):
        self.assertEqual(zero_crossings([-1,0,0,1,0,-1]),2)
        self.assertEqual(zero_crossings([-1,None,1]),0)

    def test_archive_excludes_event_day_from_crossings_and_uses_liquid_rain(self):
        start=datetime(2020,1,1);times=[(start+timedelta(hours=i)).isoformat(timespec='minutes') for i in range(192)]
        air=[-1]*84+[1]*84+[10]*24
        data={'hourly':{'time':times,'temperature_2m':air,'rain':[0]*168+[1]*24,'precipitation':[0]*168+[5]*24,'snow_depth':[0]*192}}
        out=summarize_archive(data,'2020-01-08')
        self.assertEqual(out['freeze_thaw_cycles_7d'],1)
        self.assertEqual(out['rain_event_day_mm'],24)
        self.assertEqual(out['warming_delta_c'],10)
        data['hourly']['temperature_2m'][0]=None
        self.assertIsNone(summarize_archive(data,'2020-01-08')['freeze_thaw_cycles_7d'])

    def test_historical_feature_parity_and_label_blind_terrain(self):
        row=dict(latitude=46.,longitude=7.,sand_fraction=.4,clay_fraction=.2,organic_matter_fraction=.03,bulk_density_g_cm3=1.45,rain_event_day_mm=10.,snow_depth_event_day_max_m=.5,temperature_event_day_mean_c=4.,soil_temperature_event_day_mean_c=0.,soil_moisture_event_day_mean_m3m3=.3,freeze_thaw_cycles_7d=4,warming_delta_c=5.,terrain_elevation_m=3000.)
        near=SimpleNamespace(cell_id='a',latitude=46.,longitude=7.,slope_deg=10.,twi=8.,profile_curvature=0.)
        steep=SimpleNamespace(cell_id='b',latitude=46.001,longitude=7.,slope_deg=50.,twi=8.,profile_curvature=0.)
        terrain=SimpleNamespace(cells={'a':near,'b':steep})
        row.update(rain_effective_storm_mm=30.,soil_moisture_72h_max_m3m3=.35,snow_depth_72h_max_m=.6)
        a={**row,'label':1};b={**row,'label':0}
        populate_physics_features(a,terrain);populate_physics_features(b,terrain)
        np.testing.assert_equal([a[k] for k in FEATURE_NAMES],[b[k] for k in FEATURE_NAMES])
        self.assertEqual(a['slope_max_deg'],10.)
        import json
        inp=PhysicsInputs(**json.loads(a['physics_inputs_json']));state=MultiCausalEngine().compute(inp)
        np.testing.assert_allclose([a[k] for k in FEATURE_NAMES],physics_feature_matrix(inp,state,np.array([0.]))[0],rtol=1e-6)
        self.assertEqual(a['rain_24h_mm'],30.)
        self.assertGreater(a['snowmelt_mm_day'],0.)

    def test_composite_positive_shap_follows_physics_and_conserves_mass(self):
        shap=np.zeros((1,14));shap[0,0]=3;shap[0,1]=2;shap[0,7]=-4
        physical={c:np.array([100. if c=='seismic_tectonic' else 0.]) for c in CAUSES}
        out=group_shap_by_cause(shap,FEATURE_NAMES,physical)
        self.assertEqual(out['seismic_tectonic'][0],5)
        self.assertEqual(sum(v[0] for v in out.values()),5)
        state=MultiCausalEngine().compute(replace(BASE,air_temperature_c=12.,snow_depth_m=0.,freeze_thaw_cycles_7d=5.,warming_delta_c=0.))
        causes=explain_causes(replace(BASE,air_temperature_c=12.,snow_depth_m=0.,freeze_thaw_cycles_7d=5.,warming_delta_c=0.),state)
        self.assertEqual(causes.delta_tau_kpa['cryospheric'][0],0.)
        self.assertEqual(causes.delta_tau_kpa['post_wildfire_biotic'][0],0.)

    def test_transient_replacement_preserves_water_table_and_caps_total(self):
        inp=replace(BASE,rain_24h_mm=180.,soil_saturation_fraction=.8,transient_pressure_head_m=0.)
        engine=MultiCausalEngine();state=engine.compute(inp)
        self.assertEqual(state.pore_pressure_head_m[0],0.)
        override=engine.compute(replace(inp,pore_pressure_override_m=0.))
        self.assertEqual(override.pore_pressure_head_m[0],0.)
        extreme=engine.compute(replace(inp,transient_pressure_head_m=100.))
        np.testing.assert_allclose(extreme.pore_pressure_head_m,extreme.regolith_depth_m*np.cos(np.radians(extreme.effective_slope_deg))**2)

    def test_optional_synthetic_rows_remain_training_only(self):
        y=np.array([1,0,1,0,1,0,1]);g=np.array(['a','a','b','b','c','c','synthetic:regime:example'])
        for train,test in spatial_group_splits(y,g,3):
            self.assertIn(6,train)
            self.assertNotIn(6,test)

    def test_metrics_and_estimator_contract(self):
        out=run_lhasa_benchmark_suite([1,1,0,0],[.8,.05,.15,.01])
        self.assertEqual(out['tss'],0.)
        self.assertEqual(out['csi'],1/3)
        self.assertEqual(out['fnr'],.5)
        params=make_base_estimator().get_params()
        self.assertEqual(params['max_depth'],4);self.assertEqual(params['min_child_weight'],2.5)
        self.assertEqual(params['colsample_bynode'],.45)
        self.assertEqual(params['colsample_bytree'],.75)
        self.assertIsNotNone(params['interaction_constraints'])
        estimator=make_base_estimator(n_estimators=3);estimator.set_params(n_jobs=2)
        rng=np.random.default_rng(91);x=rng.normal(size=(40,14));y=np.array([0,1]*20)
        estimator.fit(x,y)
        from backend.ml.global_xgb import GlobalMonotonicXGB
        calibrated=GlobalMonotonicXGB(n_estimators=3).fit(x,y,np.array([str(i//10) for i in range(40)]),n_splits=2)
        self.assertEqual(calibrated.model.method,'sigmoid')
        self.assertTrue(np.isfinite(calibrated.predict_proba(x)).all())
        self.assertTrue(np.isfinite(estimator.predict_proba(x)).all())
        with self.assertRaises(ValueError):run_lhasa_benchmark_suite([0,0],[.1,.2])

    def test_quiescent_controls_require_complete_evidence(self):
        from backend.ml.pinls import quiescent_control
        row=dict(rain_effective_storm_mm=20.,pga_attenuation_proxy_g=0.,snow_depth_72h_max_m=0.,fs_dynamic=1.5,p_failure_physical=.1,usgs_source_status='live_no_matching_earthquake')
        self.assertTrue(quiescent_control(row))
        for key,value in [('rain_effective_storm_mm',35.01),('snow_depth_72h_max_m',None),('pga_attenuation_proxy_g',.01),('usgs_source_status','unavailable')]:
            self.assertFalse(quiescent_control({**row,key:value}))

class ArchiveRateLimits(unittest.IsolatedAsyncioTestCase):
    async def test_rate_limit_is_not_retried_as_a_different_weather_schema(self):
        import httpx,tempfile
        from pathlib import Path
        from unittest.mock import patch,AsyncMock
        from backend.ml.archive_weather import acquire_archive
        calls=[]
        def handler(request):
            calls.append(str(request.url));return httpx.Response(429,json={'reason':'rate limited'})
        event=SimpleNamespace(latitude=1.,longitude=1.,event_date='2020-01-08')
        with tempfile.TemporaryDirectory() as tmp, patch('backend.ml.archive_weather.CACHE',Path(tmp)), patch('backend.ml.archive_weather.asyncio.sleep',new_callable=AsyncMock):
            async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
                result=await acquire_archive(client,event)
        self.assertEqual(len(calls),3)
        self.assertEqual(len(set(calls)),1)
        self.assertEqual(result['weather_feature_version'],'hourly_v12_failed')
        self.assertIsNone(result['freeze_thaw_cycles_7d'])
