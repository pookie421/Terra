"""Network provenance and explicit scenario-initialization contracts."""
import asyncio
import unittest
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch
import httpx
import numpy as np
from backend.realtime.cache import AsyncTTLCache, PolicyCache
from backend.realtime.global_ingestor import GlobalIngestor, GeocodingError, SourceStatus
from backend.api.scan_service import GlobalScanService
from backend.physics.multi_causal_engine import MultiCausalEngine
from backend.physics.temporal_forecaster import forecast_72h, fosm_probability_sigma
from backend.ml.global_xgb import mechanical_stability_gate
from test_mechanisms import BASE
from test_phase4_forecast import weather


class LivePolicy(unittest.IsolatedAsyncioTestCase):
    async def test_fresh_requests_ignore_populated_memory(self):
        cache=AsyncTTLCache()
        await cache.get_or_create('x',100,AsyncMock(return_value='old'))
        factory=AsyncMock(side_effect=['fresh1','fresh2'])
        policy=PolicyCache(cache,use_cache=False,fallback_on_error=False)
        self.assertEqual(await policy.get_or_create('x',100,factory),'fresh1')
        self.assertEqual(await policy.get_or_create('x',100,factory),'fresh2')
        self.assertEqual(factory.await_count,2)

    async def test_mixed_hit_and_live_error_fallback_are_disclosed(self):
        cache=AsyncTTLCache()
        await cache.get_or_create('x',100,AsyncMock(return_value='cached'))
        ingestor=GlobalIngestor(cache=cache,use_cache=True)
        async def mixed():
            return await asyncio.gather(ingestor.cache.get_or_create('x',100,AsyncMock()),
                ingestor.cache.get_or_create('y',100,AsyncMock(return_value='fresh')))
        result,status=await ingestor._capture('test','test',mixed())
        self.assertEqual(result,['cached','fresh'])
        self.assertEqual(status.mode,'CACHED')
        self.assertIn('LIVE',status.cache_trace)
        ingestor=GlobalIngestor(cache=cache,use_cache=False)
        result,status=await ingestor._capture('test','test',
            ingestor.cache.get_or_create('x',100,AsyncMock(side_effect=httpx.ConnectError('offline'))))
        self.assertEqual(result,'cached');self.assertEqual(status.mode,'CACHED')
        self.assertIn('offline',status.error)

    async def test_live_only_never_falls_back(self):
        cache=AsyncTTLCache()
        await cache.get_or_create('x',100,AsyncMock(return_value='cached'))
        policy=PolicyCache(cache,use_cache=False,fallback_on_error=False)
        with self.assertRaises(httpx.ConnectError):
            await policy.get_or_create('x',100,AsyncMock(side_effect=httpx.ConnectError('offline')))

    async def test_named_geocoding_uses_network_even_with_cache_enabled(self):
        requests=[]
        def respond(request):
            requests.append(request)
            return httpx.Response(200,json={'results':[{'latitude':1.,'longitude':2.,'name':'From network'}]})
        ingestor=GlobalIngestor(use_cache=True)
        async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
            for _ in range(2):
                loc,status=await ingestor._resolve_location('Wayanad',client)
                self.assertEqual((loc.latitude,loc.longitude),(1.,2.))
                self.assertEqual(status.mode,'LIVE')
        self.assertEqual(len(requests),2)

    async def test_api_known_location_only_after_geocoder_failure(self):
        service=GlobalScanService.__new__(GlobalScanService)
        service.provider_timeout_s=15;service.model=None;service.rainfall_model=None
        raw=SimpleNamespace(terrain=None,sources={'geocoding':SourceStatus(True,'coordinates','now',mode='PROVIDED')})
        with patch('backend.api.scan_service.GlobalIngestor') as factory, \
             patch('backend.api.scan_service.evaluate_live_physics',return_value=None), \
             patch.object(service,'_empty_response',side_effect=lambda scan,agency:{}), \
             patch('backend.api.ground_assessment.attach_ground',side_effect=lambda response,*args,**kwargs:response), \
             patch('backend.api.source_assessment.attach',side_effect=lambda response,scan,*args:scan):
            ingestor=factory.return_value.__aenter__.return_value
            ingestor.scan=AsyncMock(return_value=raw)
            await service.scan(location_query='Wayanad',_reference_assumptions=True)
            ingestor.scan.assert_awaited_once_with('Wayanad',radius_km=3.5)
            ingestor.scan=AsyncMock(side_effect=[GeocodingError('no network result'),raw])
            result=await service.scan(location_query='Wayanad',_reference_assumptions=True)
            self.assertEqual(ingestor.scan.await_args_list[1].args,((11.548,76.134),))
            self.assertFalse(result.sources['geocoding'].available)
            self.assertEqual(result.sources['geocoding'].mode,'FALLBACK')
            ingestor.scan=AsyncMock(side_effect=RuntimeError('terrain failure'))
            with self.assertRaises(RuntimeError):await service.scan(location_query='Wayanad',_reference_assumptions=True)
            self.assertEqual(ingestor.scan.await_count,1)


class ScenarioContracts(unittest.TestCase):
    def test_injection_supplies_missing_liquid_rain_without_inventing_air(self):
        inputs=replace(BASE,snow_depth_m=0.)
        state=MultiCausalEngine().compute(inputs);rows=weather()
        for row in rows:row['liquid_rain_mm']=None
        forecast=forecast_72h(inputs,state,np.array([0.]),None,rows,injected_storm_mm=220)
        self.assertEqual(forecast['status'],'PROJECTED_SCENARIO')
        self.assertAlmostEqual(sum(r['liquid_rain_mm'] for r in forecast['paths']['p50']['hourly'][1:]),220.)
        rows[0]['temperature_2m']=None
        self.assertEqual(forecast_72h(inputs,state,np.array([0.]),None,rows,injected_storm_mm=220)['status'],'UNAVAILABLE')

    def test_injected_calm_reset_is_explicit_and_does_not_mutate_observation(self):
        inputs=replace(BASE,soil_saturation_fraction=.95,rain_24h_mm=180.,snow_depth_m=0.)
        state=MultiCausalEngine().compute(inputs);original=state.pore_pressure_head_m.copy()
        forecast=forecast_72h(inputs,state,np.array([0.]),None,weather(),injected_storm_mm=220)
        self.assertEqual(forecast['initial_condition_quality'],'SYNTHETIC_CALM_NOT_LIVE')
        self.assertTrue(forecast['injected_storm']['changes_current_state'])
        self.assertLess(forecast['current_risk_t0_pct'],forecast['observed_current_risk_t0_pct'])
        np.testing.assert_array_equal(state.pore_pressure_head_m,original)
        self.assertEqual(inputs.rain_24h_mm,180.)
        self.assertEqual(sum(r['liquid_rain_mm'] for r in forecast['paths']['p50']['hourly'][1:19]),0.)

    def test_missing_soil_temperature_uses_disclosed_air_fallback(self):
        inputs=replace(BASE,snow_depth_m=0.)
        state=MultiCausalEngine().compute(inputs);rows=weather()
        rows[3]['soil_temperature_0_to_7cm']=None
        forecast=forecast_72h(inputs,state,np.array([0.]),None,rows)
        self.assertEqual(forecast['status'],'PROJECTED_SCENARIO')
        self.assertEqual(forecast['soil_temperature_fallback_hours'],[4])
        rows[3]['temperature_2m']=None
        self.assertEqual(forecast_72h(inputs,state,np.array([0.]),None,rows)['status'],'UNAVAILABLE')

    def test_global_wetting_and_static_yield_follow_triggers(self):
        inputs=replace(BASE,soil_saturation_fraction=.66,rain_24h_mm=0.,twi=np.array([6.,18.]),slope_deg=35.)
        dry=MultiCausalEngine().compute(inputs)
        wet=MultiCausalEngine().compute(replace(inputs,rain_24h_mm=180.))
        np.testing.assert_array_equal(wet.local_saturation,dry.local_saturation)
        self.assertTrue(np.all(wet.pore_pressure_head_m>dry.pore_pressure_head_m))
        self.assertTrue(np.all(wet.critical_acceleration_g<=dry.critical_acceleration_g))
        self.assertEqual(mechanical_stability_gate([35],[1.6],[.99])[0],mechanical_stability_gate([35],[1.6],[.01])[0])

    def test_bounded_spread_retains_sigma_dependence(self):
        state=SimpleNamespace(beta_fosm=np.zeros(2),sigma_fs=np.array([.04,.12]),p_failure_seismic=np.zeros(2),p_failure_m1=np.zeros(2))
        sigma=fosm_probability_sigma(state)
        self.assertAlmostEqual(sigma[1]/sigma[0],3.)
