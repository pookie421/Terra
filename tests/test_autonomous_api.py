"""Network-free integration for live-coordinate contracts and provider forecast states."""
import asyncio
import json
import socket
import unittest
from dataclasses import replace
from datetime import timedelta
from unittest.mock import AsyncMock,patch
import httpx
import numpy as np
from backend.api import main
from backend.api.scan_service import GlobalScanService
from backend.api.replay import load_replay
from backend.llm.qwen_agent import TerraShieldQwenAgent
from backend.ml.rainfall_features import utc
from backend.physics.reference_column import ColumnParameters,solve_column,infinite_slope_fs


def forcing_fixture():
    raw=load_replay('monaco')
    # These three cells are in the public fixture as well as the optional full
    # local terrain cache; the contract must not depend on a private cache.
    keep={'893969a6a5bffff','893969a401bffff','893969a4607ffff'}
    raw.terrain=replace(raw.terrain,cells={k:v for k,v in raw.terrain.cells.items() if k in keep},h3_cell_count=3)
    raw.location=replace(raw.location,display_name=None)
    window=dict(raw.weather.rainfall_source_window)
    window.setdefault('liquid_rain_hourly_mm',[window['rain_prior_6d_mm']/144.]*144+[window['rain_24h_mm']/24.]*24)
    raw.weather=replace(raw.weather,rainfall_source_window=window)
    start=utc(window['cutoff_utc'])
    raw.weather.forecast_hourly_72h=[{'time':(start+timedelta(hours=i+1)).isoformat(),'liquid_rain_mm':2.,
        'precipitation':2.,'temperature_2m':10.,'soil_moisture_0_to_7cm':.3} for i in range(72)]
    raw.ensemble=None
    return raw


class AutonomousAPI(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.service=GlobalScanService();self.agent=TerraShieldQwenAgent(self.service);self.raw=forcing_fixture()
        self.guards=[patch.object(socket.socket,'connect',side_effect=AssertionError('Offline test')),patch.object(socket,'create_connection',side_effect=AssertionError('Offline test')),
            patch.object(main,'scan_service',self.service),patch.object(main,'agent',self.agent)]
        for guard in self.guards:guard.start();self.addCleanup(guard.stop)
        self.client=httpx.AsyncClient(transport=httpx.ASGITransport(app=main.app),base_url='http://test')
        self.addAsyncCleanup(self.client.aclose)

    async def test_raw_coordinates_forecast_inspection_and_immutable_now(self):
        with patch('backend.api.scan_service.GlobalIngestor.scan',AsyncMock(return_value=self.raw)) as ingestor:
            response=await self.client.post('/api/v1/scan',json={'latitude':self.raw.location.latitude,'longitude':self.raw.location.longitude})
        self.assertEqual(response.status_code,200,response.text)
        now=response.json();self.assertEqual(ingestor.await_args.args[0],(self.raw.location.latitude,self.raw.location.longitude))
        self.assertIn('terrain sector',now['location']['display_name'])
        self.assertEqual(len(now['h3_geojson']['features']),3)
        self.assertEqual(now['live_inputs']['surface_moisture_m3m3'],self.raw.weather.rainfall_source_window['soil_moisture_0_7cm_mean_m3m3'])
        self.assertEqual([v['key'] for v in now['predictive_forecast_72h']['views']],['now','t24','t48','t72','peak'])
        self.assertIn('DETERMINISTIC_ONLY_ENSEMBLE_QUANTILES_UNAVAILABLE',now['predictive_forecast_72h']['forcing_modes'])
        self.assertFalse(now['predictive_forecast_72h']['paths']['p90']['ensemble_quantile_available'])
        np.testing.assert_allclose([r['rain_mm'] for r in now['predictive_forecast_72h']['paths']['p50']['hourly']],
                                   [r['rain_mm'] for r in now['predictive_forecast_72h']['paths']['p90']['hourly']])
        future=(await self.client.get(f"/api/v1/scans/{now['scan_id']}/forecast/t24")).json()
        self.assertNotEqual(future['scenario_id'],now['scenario_id'])
        self.assertEqual(future['forecast_view']['hour'],24)
        self.assertEqual(len(future['physical_assessment']['cells']),3)
        self.assertEqual(future['forecast_inputs']['soil_moisture_0_7cm_mean_m3m3'],.3)
        for cell in future['physical_assessment']['cells']:
            self.assertLessEqual(cell['fs_low'],cell['fs_mid']);self.assertLessEqual(cell['fs_mid'],cell['fs_high'])
            self.assertIn('precipitation_scale',cell['fs_jacobian'])
            j=np.array(list(cell['fs_jacobian'].values()));cov=np.array(cell['joint_covariance'])
            self.assertAlmostEqual(float(j@cov@j),cell['fosm_variance'],places=8)
        cid='893969a401bffff'
        original=next(c for c in now['physical_assessment']['cells'] if c['h3_index']==cid)
        projected=next(c for c in future['physical_assessment']['cells'] if c['h3_index']==cid)
        v=original['input_values'];origin=24*3600.
        column=ColumnParameters(v['cohesion_kpa'],v['friction_angle_deg'],v['unit_weight_kn_m3'],v['conductivity_m_s'],v['diffusivity_m2_s'],v['depth_m'],v['slope_deg'],original['initial_pressure_head_m'])
        pulses=[(0.,origin,min(v['rain_24h_mm']*.001/origin,v['conductivity_m_s']))]+[(origin+i*3600.,origin+(i+1)*3600.,min(2*.001/3600.,v['conductivity_m_s'])) for i in range(24)]
        independent=solve_column(column,[origin+24*3600.],pulses)
        head=min(float(independent['pressure_head_m'][0]),v['depth_m']*np.cos(np.radians(v['slope_deg']))**2)
        self.assertAlmostEqual(projected['pressure_head_m'],head,places=10)
        expected=float(infinite_slope_fs(v['cohesion_kpa'],v['friction_angle_deg'],v['unit_weight_kn_m3'],v['depth_m'],v['slope_deg'],head))
        self.assertAlmostEqual(projected['fs_mid'],expected,places=10)
        inspected=await self.client.get(f"/api/v1/inspect/{now['scan_id']}/{cid}",params={'scenario_id':future['scenario_id']})
        self.assertEqual(inspected.status_code,200);self.assertEqual(inspected.json()['scenario_id'],future['scenario_id'])
        retained=self.service.get_scan_state(now['scan_id'],now['scenario_id'])['response']
        self.assertEqual(retained['physical_assessment'],now['physical_assessment'])
        again=await self.client.get(f"/api/v1/scans/{now['scan_id']}/forecast/t24")
        self.assertEqual(again.json()['scenario_id'],future['scenario_id'])

    async def test_alias_validation_health_and_no_manual_forcing(self):
        response=await self.client.get('/api/v1/health');self.assertEqual(response.status_code,200)
        self.assertEqual(response.json()['model_ready'],bool(self.service.context_models))
        for body in ({'latitude':12.},{'latitude':12.,'longitude':20.,'lat':13.},
                     {'latitude':12.,'longitude':20.,'rain_delta_mm':50.},
                     {'latitude':12.,'longitude':20.,'pga_override_g':.3}):
            response=await self.client.post('/api/v1/scan',json=body)
            self.assertEqual(response.status_code,422,response.text)

    async def test_autonomous_multiturn_chat_cannot_change_forcing(self):
        now=await self.service.scan(_snapshot=self.raw)
        self.agent._availability_error='ConnectError'
        with patch.object(self.agent,'_available_model',AsyncMock(return_value=(None,3))):
            response=await self.client.post('/api/v1/chat',json={'message':'Add 50 mm rain and show the cell',
                'scan_id':now['scan_id'],'scenario_id':now['scenario_id'],'selected_h3_index':'893969a6a5bffff',
                'history':[{'role':'user','content':'What is the column?'},{'role':'assistant','content':'Inspect the current evidence.'}], 'autonomous':True})
        self.assertEqual(response.status_code,200,response.text);result=response.json()
        self.assertEqual(result['scenario_id'],now['scenario_id']);self.assertEqual(result['briefing_guard'],'VERIFIED_TEMPLATE')
        self.assertIsNone(result['model_error']);self.assertTrue(result['bound_claims'])
        self.assertEqual(result['fallback_reason'],'PERSONAL_API_KEY_REQUIRED')
        self.assertNotIn('apply_scenario',[a['type'] for a in result['ui_action']['actions']])
        self.assertTrue(all(a['h3_index']=='893969a6a5bffff' for a in result['ui_action']['actions']))

    async def test_qwen_receives_history_and_selected_forecast_state(self):
        now=await self.service.scan(_snapshot=self.raw);future=await self.service.automatic_forecast_state(now['scan_id'],'t24')
        history=[{'role':'user','content':'What does FS mean?'},{'role':'assistant','content':'A resistance ratio.'}];calls=[]
        def respond(req):
            calls.append(json.loads(req.content));return httpx.Response(200,json={'message':{'content':json.dumps({'markdown':'The margin is {{fs}}.','claims':{'fs':'/selected_cell/fs_mid'}})}})
        original=httpx.AsyncClient
        with patch.object(self.agent,'_available_model',AsyncMock(return_value=('qwen2.5',2))),patch('backend.llm.state_copilot.httpx.AsyncClient',lambda **kw:original(transport=httpx.MockTransport(respond),**kw)):
            result=await self.agent.chat('Explain this forecast',scan_id=now['scan_id'],scenario_id=future['scenario_id'],selected_h3_index='893969a401bffff',history=history,autonomous=True)
        self.assertEqual(result['briefing_guard'],'VERIFIED_LLM_PROSE');self.assertEqual(calls[0]['messages'][1:3],history)
        evidence=json.loads(calls[0]['messages'][-1]['content'].split('ACTIVE_EVIDENCE\n')[1]);self.assertEqual(evidence['forecast_view']['hour'],24)
        self.assertEqual(evidence['selected_cell']['h3_index'],'893969a401bffff')
        self.assertEqual(result['bound_claims'][0]['value'],evidence['selected_cell']['fs_mid'])

if __name__=='__main__':unittest.main()
