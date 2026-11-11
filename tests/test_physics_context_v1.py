"""Independent analytical, provider-contract and state-bound fusion acceptance tests."""
import asyncio,json,math,socket,unittest
from datetime import datetime,timedelta,timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch,AsyncMock
import numpy as np
import httpx,h3
from backend.realtime.dem_processor import DEMProcessor,EARTH_CIRCUMFERENCE_M,TERRAIN_VERSION
from backend.physics.reference_context import joint_fosm,_solve,columns,CONTEXT_FEATURES
from backend.physics.kinematics import route_runout
from backend.api.scan_service import GlobalScanService
from backend.realtime.global_ingestor import GlobalScan,GlobalIngestor
from scripts.snapshot_replay import hydrate
ROOT=Path(__file__).resolve().parents[1]


class TerrainMath(unittest.TestCase):
    def test_planes_across_latitudes_and_azimuths(self):
        for latitude in (0,43.73718,65,-60):
            dx=EARTH_CIRCUMFERENCE_M*math.cos(math.radians(latitude))/(256*8192)
            yy,xx=np.mgrid[-10:11,-10:11]*dx
            for bearing in (0,45,90,135,180,225,270,315):
                a=math.radians(bearing);rise=math.tan(math.radians(32.))
                surface=-rise*math.sin(a)*xx+rise*math.cos(a)*yy
                slope,aspect,curv=DEMProcessor.derivatives(surface,dx,dx)
                np.testing.assert_allclose(slope,32.,atol=1e-10)
                np.testing.assert_allclose((aspect-bearing+180)%360-180,0,atol=1e-10)
                np.testing.assert_allclose(curv,0,atol=1e-10)
        slope,_,_=DEMProcessor.derivatives(np.ones((5,5)),10.,10.)
        np.testing.assert_array_equal(slope,0.)

    def test_specific_area_on_isolated_cell(self):
        cell=h3.latlng_to_cell(43.7,7.4,9);lat,lon=h3.cell_to_latlng(cell)
        c=SimpleNamespace(latitude=lat,longitude=lon,elevation_m=100.,slope_deg=10.,downslope_neighbors=[])
        DEMProcessor()._route_contributing_area({cell:c})
        self.assertAlmostEqual(c.twi,math.log(math.sqrt(h3.cell_area(cell,unit='m^2'))/math.tan(math.radians(10))))


class OfflineAsync(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.guards=[patch.object(socket.socket,'connect',side_effect=AssertionError('Network disabled')),patch.object(socket,'create_connection',side_effect=AssertionError('Network disabled'))]
        for g in self.guards:g.start();self.addCleanup(g.stop)


class TerrainSupport(OfflineAsync):
    async def test_viewport_invariance_on_continuous_analytic_tiles(self):
        dem=DEMProcessor()
        async def analytic_tile(client,x,y):
            # One continuous sloped surface across tile seams; independent of
            # local provider caches and suitable for a source-only checkout.
            yy,xx=np.mgrid[0:256,0:256]
            elevation=(1000.+.5*(x*256+xx)-.3*(y*256+yy)).astype(np.float32)
            return elevation,np.ones((256,256),dtype=bool)
        with patch.object(dem,'_fetch_tile',side_effect=analytic_tile):
            async with httpx.AsyncClient() as client:
                small=await dem.process(43.73718,7.42145,1.2,client)
                large=await dem.process(43.73718,7.42145,3.5,client)
        common=set(small.cells)&set(large.cells)
        self.assertGreater(len(common),30)
        for k in common:
            self.assertAlmostEqual(small.cells[k].twi,large.cells[k].twi,places=10)
            self.assertAlmostEqual(small.cells[k].slope_deg,large.cells[k].slope_deg,places=10)


class ReferenceMath(unittest.TestCase):
    def test_joint_variance_and_monte_carlo_small_noise(self):
        x=np.array([[5.,32.,19.,1.5,1e-5,1e-4,.2,30.,20.]])
        sd=np.array([[.02,.02,.01,.002,1e-8,1e-7,.001,.01,.02]])
        corr=np.eye(9);corr[0,1]=corr[1,0]=-.3
        var,j,cov=joint_fosm(x,sd,correlation=corr)
        np.testing.assert_allclose(var,np.einsum('ni,nij,nj->n',j,cov,j),rtol=1e-12)
        expected=1/(19*1.5*math.sin(math.pi/6)*math.cos(math.pi/6))
        self.assertAlmostEqual(j[0,0],expected,places=8)
        rng=np.random.default_rng(17);draws=rng.multivariate_normal(x[0],cov[0],size=20000)
        empirical=np.var(_solve(draws,86400.)[0])
        self.assertLess(abs(empirical/var[0]-1),.04)

    def test_reference_no_boreholes_and_positive_envelope(self):
        cell=SimpleNamespace(cell_id='column',slope_deg=3.7,twi=10.,profile_curvature=0.)
        w={'rain_24h_mm':6.9,'rain_prior_6d_mm':6.6,'soil_moisture_0_7cm_mean_m3m3':.336}
        c=columns([cell],w)[0]
        self.assertGreater(c['fs_mid'],5.)
        self.assertGreater(c['fosm_variance'],0.)
        self.assertGreater(c['fs_low'],0.)
        self.assertLess(c['p_failure_physical'],1e-4)
        self.assertEqual(set(c['fs_jacobian']),set(c['input_values']))
        self.assertEqual(c['provenance_label'],'pedotransfer_reference_column')

    def test_flat_slope_reliability_has_no_ratio_singularity(self):
        window={'rain_24h_mm':6.9,'rain_prior_6d_mm':6.6,'soil_moisture_0_7cm_mean_m3m3':.336}
        cells=[SimpleNamespace(cell_id=str(a),slope_deg=a,twi=10.,profile_curvature=0.) for a in (.01,.1,.5,1.,2.,3.7)]
        for c in columns(cells,window):
            self.assertLess(c['p_failure_physical'],1e-6)
            self.assertGreater(c['fs_low'],1.)
            self.assertLessEqual(c['fs_low'],c['fs_mid']);self.assertLessEqual(c['fs_mid'],c['fs_high'])
            self.assertGreater(c['fosm_variance'],0.)

    def test_voellmy_acceleration_and_exact_drag(self):
        a=SimpleNamespace(latitude=0.,longitude=0.,elevation_m=100.,downslope_neighbors=['b'])
        b=SimpleNamespace(latitude=0.,longitude=.001,elevation_m=0.,downslope_neighbors=[])
        route=route_runout('a',{'a':a,'b':b},0.,.2,500.,flow_depth_m=.5,released=True)
        self.assertEqual(route.path,['a','b']);self.assertGreater(route.peak_velocity_m_s,0.)
        dx=h3.great_circle_distance((0,0),(0,.001),unit='m');ds=math.hypot(dx,100.)
        decay=2*9.81/(500*.5);equilibrium=((100-.2*dx)/ds)/decay
        self.assertAlmostEqual(route.final_energy_head_m,equilibrium*(1-math.exp(-decay*ds)))
        static=route_runout('a',{'a':a,'b':b},0.,.2,500.)
        self.assertEqual(static.path,['a'])
        exposure={'a':{'roads':1,'roads_ids':['way/1']},'b':{'roads':1,'roads_ids':['way/1']}}
        routed=route_runout('a',{'a':a,'b':b},0.,.2,500.,flow_depth_m=.5,released=True,infrastructure_by_cell=exposure)
        self.assertEqual(routed.intersected_roads,1)

    @unittest.skipUnless((ROOT/'artifacts/physics_informed_models.pkl').exists() and
                         (ROOT/'data/physics_informed_training.json').exists(),
                         'Requires retained local learned artifacts and training frame')
    def test_fusion_marginal_monotonicity_and_shap_reconstruction(self):
        from backend.ml.physics_informed import NAMES,MONOTONIC
        from scipy.special import expit
        import joblib
        model=joblib.load(ROOT/'artifacts/physics_informed_models.pkl')['Physics-Informed ML']
        data=json.loads((ROOT/'data/physics_informed_training.json').read_text())['train']
        x=np.asarray([r['features'] for r in data],float)
        base=model.predict_proba(x)
        for i,direction in enumerate(MONOTONIC):
            if not direction:continue
            changed=x.copy();changed[:,i]+=np.maximum(abs(changed[:,i])*.01,1e-6)
            if NAMES[i] in ('fosm_reference_score','gravity_release_admissibility'):changed[:,i]=np.minimum(changed[:,i],1.)
            valid=np.isfinite(x[:,i])&(changed[:,i]>=x[:,i])
            difference=model.predict_proba(changed)-base
            self.assertTrue(np.all(direction*difference[valid]>=-2e-6),NAMES[i])
        explanation=model.explain(x[:3])
        for i,e in enumerate(explanation):
            probabilities=[expit(sum(f['terms'].values())+f['bias']+f['calibration_offset'])*e['mechanical_admissibility'] for f in e['folds']]
            self.assertAlmostEqual(float(np.mean(probabilities)),base[i],places=5)


class Providers(OfflineAsync):
    async def test_wcs_map_depth_projection_and_units(self):
        from backend.realtime.soilgrids import fetch_soilgrids_wcs,HOMOLOSINE_PROJ
        from rasterio.io import MemoryFile
        calls=[]
        def handle(req):
            calls.append(req);params=req.url.params;prop=params['map'].split('/')[-1].split('.')[0]
            v={'sand':450,'clay':200,'soc':250,'bdod':145}[prop]
            with MemoryFile() as m:
                with m.open(driver='GTiff',height=2,width=2,count=1,dtype='int16') as ds:ds.write(np.full((2,2),v,dtype=np.int16),1)
                body=m.read()
            return httpx.Response(200,content=body)
        async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
            result=await fetch_soilgrids_wcs(client,43.7,7.4)
        self.assertEqual(len(calls),12);self.assertEqual(result['bdod_kg_m3'],1450.);self.assertEqual(result['soc_g_kg'],25.)
        self.assertEqual({r.url.params['COVERAGEID'].split('_')[1] for r in calls},{'0-5cm','5-15cm','15-30cm'})
        self.assertIn('+proj=igh',HOMOLOSINE_PROJ)

    async def test_missing_rain_and_explicit_seismic_windows(self):
        self.assertIsNone(GlobalIngestor._sum_present([2.,None,3.]))
        self.assertEqual(GlobalIngestor._sum_present([0.,0.]),0.)
        now=datetime.now(timezone.utc)
        def event(name,age):return {'id':name,'properties':{'mag':5.,'time':(now-timedelta(seconds=age)).timestamp()*1000},'geometry':{'coordinates':[7.4,43.7,10.]}}
        ing=GlobalIngestor();requests=[]
        async def cached(name,params,*args):
            requests.append(params);return {'features':[event('recent',60),event('old',86400)]}
        ing._json_cached=cached
        result=await ing._get_seismic(43.7,7.4,None)
        self.assertEqual(result.event_count,1);self.assertEqual(len(result.background_events_30d),2)
        spans=sorted((datetime.fromisoformat(p['endtime'])-datetime.fromisoformat(p['starttime'])).total_seconds() for p in requests)
        self.assertEqual(spans,[300.,30*86400.])

    async def test_osm_geometry_crosses_cells_and_query_includes_exposure(self):
        from backend.realtime.osm_geometry import intersected_cells
        way={'type':'way','geometry':[{'lat':43.7,'lon':7.4},{'lat':43.7,'lon':7.42}],'tags':{'highway':'primary'}}
        self.assertGreater(len(intersected_cells(way)),3)
        requests=[]
        def respond(req):requests.append(req);return httpx.Response(200,json={'elements':[dict(way,id=1)]})
        async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
            result=await GlobalIngestor()._get_infrastructure(43.69,7.39,43.71,7.43,client)
        text=requests[0].content.decode();self.assertIn('building',text);self.assertIn('quarry',text)
        self.assertEqual(result.road_count,1);self.assertGreater(len(result.by_h3_cell),3)

    async def test_tls_failover_and_coastal_river_selection(self):
        from backend.realtime.ground_context import river_discharge
        requests=[];today=datetime.now(timezone.utc).date()
        def respond(req):
            requests.append(req)
            if 'overpass-api.de' in req.url.host:raise httpx.ConnectError('TLS certificate failure',request=req)
            if '/interpreter' in req.url.path:return httpx.Response(200,json={'elements':[]})
            self.assertEqual(req.url.params['cell_selection'],'land')
            return httpx.Response(200,json={'daily':{'time':[(today+timedelta(days=i)).isoformat() for i in range(-7,7)],'river_discharge':[10.]*14}})
        async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
            await GlobalIngestor()._get_infrastructure(43.5,7.1,43.6,7.2,client)
            await river_discharge(client,43.7,7.4)
        self.assertGreaterEqual(len(requests),3)


class LiveContract(OfflineAsync):
    @unittest.skipUnless((ROOT/'artifacts/physics_informed_models.pkl').exists(),
                         'Requires retained local physics-informed model artifact')
    async def test_retained_monaco_fusion_scores(self):
        raw=hydrate(GlobalScan,json.loads((ROOT/'tests/fixtures/monaco_physics_context.json').read_text())['snapshot'])
        service=GlobalScanService();before=await service.scan(_snapshot=raw)
        cid='893969a6a5bffff';cliff='893969a401bffff'
        flat=service.inspect_hexagon(cid,before['scan_id'],before['scenario_id'])
        self.assertLess(flat['physics_informed_assessment']['score'],.01)
        after=await service.apply_scenario(before['scan_id'],before['scenario_id'],{'rain_delta_mm':100.})
        a=service.inspect_hexagon(cliff,before['scan_id'],before['scenario_id'])
        b=service.inspect_hexagon(cliff,after['scan_id'],after['scenario_id'])
        self.assertGreaterEqual(b['physics_informed_assessment']['score'],a['physics_informed_assessment']['score'])

    async def test_monaco_state_and_copilot(self):
        raw=hydrate(GlobalScan,json.loads((ROOT/'tests/fixtures/monaco_physics_context.json').read_text())['snapshot'])
        service=GlobalScanService();before=await service.scan(_snapshot=raw)
        cid='893969a6a5bffff';cliff='893969a401bffff'
        flat=service.inspect_hexagon(cid,before['scan_id'],before['scenario_id'])
        self.assertGreater(flat['fs_mid'],5.)
        after=await service.apply_scenario(before['scan_id'],before['scenario_id'],{'rain_delta_mm':100.})
        self.assertNotEqual(before['scenario_id'],after['scenario_id'])
        unchanged=service.inspect_hexagon(cid,before['scan_id'],before['scenario_id'])
        self.assertEqual(flat,unchanged)
        a=service.inspect_hexagon(cliff,before['scan_id'],before['scenario_id']);b=service.inspect_hexagon(cliff,after['scan_id'],after['scenario_id'])
        self.assertGreater(a['fs_mid'],1.);self.assertLess(a['fs_mid'],2.)
        self.assertLessEqual(b['fs_mid'],a['fs_mid'])
        self.assertEqual(before['runout_assessment']['status'],'CONDITIONAL_RELEASE_SCENARIO')
        from backend.llm.qwen_agent import TerraShieldQwenAgent
        agent=TerraShieldQwenAgent(service)
        agent._availability_error='ConnectError'
        with patch.object(agent,'_available_model',AsyncMock(return_value=(None,3))):
            result=await agent.chat('Explain the mechanical margin',scan_id=before['scan_id'],scenario_id=before['scenario_id'])
        self.assertEqual(result['briefing_guard'],'VERIFIED_TEMPLATE');self.assertEqual(result['scenario_id'],before['scenario_id'])
        self.assertGreater(len(result['clean_markdown']),300)
        def respond(req):return httpx.Response(200,json={'message':{'content':json.dumps({'markdown':'The reference margin is {{fs}}; its variance is {{variance}}.','claims':{'fs':'/selected_cell/fs_mid','variance':'/selected_cell/fosm_variance'}})}})
        original=httpx.AsyncClient
        with patch.object(agent,'_available_model',AsyncMock(return_value=('qwen2.5',2))),patch('backend.llm.state_copilot.httpx.AsyncClient',lambda **kw:original(transport=httpx.MockTransport(respond),**kw)):
            result=await agent.chat('Explain the active cell',scan_id=before['scan_id'],scenario_id=before['scenario_id'])
        self.assertEqual(result['briefing_guard'],'VERIFIED_LLM_PROSE');self.assertTrue(result['numeric_audit_passed'])
        self.assertEqual(len(result['bound_claims']),2)
        # Returning to the original state is independent of later caller mutation.
        before['summary']['hotspot_fs_dynamic']=-999.
        self.assertNotEqual(service.get_scan_state(before['scan_id'],before['scenario_id'])['response']['summary']['hotspot_fs_dynamic'],-999.)

    async def test_claims_reject_unbound_and_unrelated_numbers(self):
        from backend.llm.state_copilot import verify_and_render
        with self.assertRaises(ValueError):verify_and_render({'markdown':'FS is 123','claims':{}},{})
        with self.assertRaises(KeyError):verify_and_render({'markdown':'FS {{fs}}','claims':{'fs':'/missing'}},{})

    async def test_chat_aliases_scenario_actions_and_runtime_errors(self):
        from backend.api.main import app
        paths={r.path for r in app.routes}
        self.assertTrue({'/api/chat','/api/v1/chat','/api/v1/agent/chat'}<=paths)
        raw=hydrate(GlobalScan,json.loads((ROOT/'tests/fixtures/monaco_physics_context.json').read_text())['snapshot'])
        service=GlobalScanService();scan=await service.scan(_snapshot=raw)
        from backend.llm.qwen_agent import TerraShieldQwenAgent
        agent=TerraShieldQwenAgent(service)
        with patch.object(agent,'_available_model',AsyncMock(return_value=(None,3))):
            result=await agent.chat('Add 50 mm rain and show the cell',scan_id=scan['scan_id'],scenario_id=scan['scenario_id'])
        self.assertEqual(result['scan_id'],scan['scan_id']);self.assertNotEqual(result['scenario_id'],scan['scenario_id'])
        self.assertTrue({'apply_scenario','fly_to_cell','inspect_cell'}<={a['type'] for a in result['ui_action']['actions']})
        agent._availability_error='ConnectError'
        with patch.object(agent,'_available_model',AsyncMock(return_value=(None,3))):
            result=await agent.chat('Explain',scan_id=scan['scan_id'],scenario_id=scan['scenario_id'])
        self.assertEqual(result['briefing_guard'],'VERIFIED_TEMPLATE')
        self.assertEqual(result['model_error'],'ConnectError')
        self.assertEqual(result['fallback_reason'],'LOCAL_MODEL_SERVICE_UNAVAILABLE')

    async def test_forecast_initial_state_matches_active_column(self):
        raw=hydrate(GlobalScan,json.loads((ROOT/'tests/fixtures/monaco_physics_context.json').read_text())['snapshot'])
        service=GlobalScanService();scan=await service.scan(_snapshot=raw,rain_delta_mm=30.,storm_duration_hr=6.,inject_forecast_storm_mm=100.)
        self.assertEqual(scan['predictive_forecast_72h']['status'],'PROJECTED_REFERENCE_SCENARIO')
        initial=scan['predictive_forecast_72h']['paths']['p50']['hourly'][0]['minimum_fs']
        self.assertAlmostEqual(initial,scan['summary']['min_fs_dynamic'],places=9)


if __name__=='__main__':unittest.main()
