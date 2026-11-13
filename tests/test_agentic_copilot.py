"""State binding, ordered controller plans and source-honest app SFT contracts."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch
import httpx

from backend.llm.app_actions import (ACTION_TYPES, MODELS, action_envelope, plan_actions,
                                    public_steps, validate_actions)
from backend.llm.qwen_agent import TerraShieldQwenAgent
from backend.ml.context_trajectories import training_plan
from backend.llm.state_copilot import context, verify_and_render

FLAT='893969a6a5bffff';CLIFF='893969a401bffff';RISK='893969a4607ffff'


def scan_fixture():
    cells=[{'h3_index':FLAT,'slope_deg':3.7,'score':.01,'fs_mid':5.,'fs_low':4.,'fs_high':6.,'fs_interval':[4.,5.,6.],'fosm_variance':.02},
           {'h3_index':CLIFF,'slope_deg':41.1,'score':.2,'fs_mid':1.38,'fs_low':1.1,'fs_high':1.6,'fs_interval':[1.1,1.38,1.6],'fosm_variance':.03},
           {'h3_index':RISK,'slope_deg':25.,'score':.9,'fs_mid':.9,'fs_low':.6,'fs_high':1.2,'fs_interval':[.6,.9,1.2],'fosm_variance':.04}]
    for c in cells:c['prediction_scores']={model:c['score'] for model in MODELS}
    return {'scan_id':'scan-a','scenario_id':'now-a','location':{'query':'fixture','latitude':1.,'longitude':2.},
            'scenario':{},'summary':{'hotspot_h3_index':RISK},'source_status':{},'selected_cell':cells[0],
            'cell_registry':{c['h3_index']:dict(c) for c in cells},
            'physics_informed_assessment':{'cells':cells,'hotspot':cells[-1]},
            'physical_assessment':{'cells':cells,'hotspot':cells[-1]}}


class OrderedPlans(unittest.TestCase):
    def test_large_grid_provider_and_routing_payloads_never_fill_the_llm_context(self):
        scan=scan_fixture()
        cell=scan['physics_informed_assessment']['hotspot']
        cell.update(nisar_coherence=0.,lhasa_probability=.1,joint_covariance=[[1.,0.],[0.,2.]])
        scan['nasa_lhasa']={'mode':'LIVE_NASA_IMAGESERVER','by_h3':{f'unselected-{i}':{'payload':'foreign-cell-marker'*100} for i in range(277)},'cells':[cell]*277}
        scan['nisar']={'mode':'METADATA_LIVE','granule_count':12,'by_h3':scan['nasa_lhasa']['by_h3'],'granules':[{'raw':'foreign-cell-marker'*100}]*12,'raster_assets':scan['nasa_lhasa']['by_h3']}
        scan['runout_assessment']={'status':'CONDITIONAL','path':['foreign-cell-marker']*200,'scenarios':[{'coordinates_lonlat':[[1.,2.]]*200,'peak_velocity_m_s':3.}]*3}
        evidence=context(scan)
        encoded=json.dumps(evidence)
        self.assertNotIn('foreign-cell-marker',encoded)
        self.assertNotIn('by_h3',encoded)
        self.assertLess(len(encoded),12000)
        self.assertEqual(evidence['selected_cell']['joint_covariance'],[[1.,0.],[0.,2.]])
        prose,claims=verify_and_render({'markdown':'Coherence is {{radar}}.','claims':{'radar':'/selected_cell/nisar_coherence'}},evidence)
        self.assertEqual(claims[0]['value'],0.)

    def test_coordinate_scan_retains_both_model_switches_and_defers_new_cell(self):
        actions=plan_actions('Take me to 11.548, 76.134, switch to NASA LHASA and then back to TERRA-SHIELD, and inspect the steepest cliff',scan_fixture(),FLAT)
        self.assertEqual([a['type'] for a in actions],['scan_coordinates','select_prediction_model','select_prediction_model','inspect_cell'])
        self.assertEqual([a['model'] for a in actions if 'model' in a],['nasa_lhasa_2','terra_physics_ml'])
        self.assertEqual((actions[0]['latitude'],actions[0]['longitude']),(11.548,76.134))
        self.assertEqual(actions[-1]['target'],'steepest');self.assertNotIn('h3_index',actions[-1])
        self.assertEqual([a['state_scope'] for a in actions],['bound','chain','chain','chain'])

    def test_unnamed_and_named_initial_scans_do_not_need_existing_state(self):
        actions=plan_actions('Take me to Kathmandu, Nepal, then switch to NASA LHASA and inspect the riskiest cell')
        self.assertEqual(actions[0]['type'],'search_location');self.assertEqual(actions[0]['query'],'Kathmandu, Nepal')
        self.assertIsNone(actions[0]['scan_id']);self.assertEqual(actions[-1]['target'],'riskiest')
        self.assertEqual(plan_actions('11.548, 76.134')[0]['type'],'scan_coordinates')
        precise=plan_actions('Scan 11.548123456789012, 76.134123456789012 then inspect the steepest cell',scan_fixture(),FLAT)
        self.assertEqual(precise[-1]['target'],'steepest')

    def test_semantic_cells_resolve_against_bound_evidence_without_selecting_a_label(self):
        scan=scan_fixture()
        self.assertEqual(plan_actions('Inspect the steepest cliff',scan,FLAT)[-1]['h3_index'],CLIFF)
        self.assertEqual(plan_actions('Compare our in-house Physics+ML model with NASA LHASA on the riskiest cell here',scan,FLAT)[-1]['h3_index'],RISK)
        actions=plan_actions('Compare the steepest cliff with the valley floor',scan,FLAT)
        self.assertEqual(actions[-1]['type'],'compare_cells')
        self.assertEqual((actions[-1]['primary_h3'],actions[-1]['secondary_h3']),(CLIFF,FLAT))

    def test_forecast_chain_defers_selectors_and_retains_display_order(self):
        actions=plan_actions('Show pore pressure layer, then set 72 hours and inspect the steepest cliff',scan_fixture(),FLAT)
        self.assertEqual([a['type'] for a in actions],['set_map_layer','set_time_horizon','inspect_cell'])
        self.assertEqual(actions[1]['horizon'],'t_72h');self.assertEqual(actions[-1]['target'],'steepest')
        self.assertEqual(plan_actions('Switch to NASA LHASA')[0]['model'],'nasa_lhasa_2')

    def test_riskiest_selector_uses_the_selected_prediction_model(self):
        scan=scan_fixture()
        scan['cell_registry'][CLIFF]['prediction_scores']['nasa_lhasa_2']=.99
        scan['active_prediction_model']='nasa_lhasa_2'
        self.assertEqual(plan_actions('Inspect the riskiest cell',scan,FLAT)[-1]['h3_index'],CLIFF)

    def test_nasa_ranking_keeps_official_and_reference_scores_separate(self):
        scan=scan_fixture();scan['active_prediction_model']='nasa_lhasa_2'
        scan['nasa_lhasa']={'cells':[{'h3_index':FLAT,'official_score':.01},{'h3_index':CLIFF,'official_score':.04},{'h3_index':RISK,'official_score':None}]}
        scan['cell_registry'][RISK]['prediction_scores']['nasa_lhasa_2']=.99
        self.assertEqual(plan_actions('Inspect the riskiest cell',scan,FLAT)[-1]['h3_index'],CLIFF)

    def test_validation_rejects_unknown_actions_nonfinite_coordinates_and_stale_cells(self):
        scan=scan_fixture()
        for action in ({'type':'open_url','url':'https://example.com'},
                       {'type':'scan_coordinates','latitude':float('nan'),'longitude':2.},
                       {'type':'scan_coordinates','latitude':True,'longitude':2.},
                       {'type':'select_prediction_model','model':'invented'},
                       {'type':'inspect_cell','h3_index':FLAT,'scan_id':'other'},
                       {'type':'fly_to_cell','h3_index':FLAT,'pitch':90.},
                       {'type':'inspect_cell','h3_index':'893969a460fffff'}):
            with self.subTest(action=action),self.assertRaises(ValueError):validate_actions([action],scan,'scan-a','now-a',FLAT)
        with self.assertRaisesRegex(ValueError,'selector'):
            validate_actions([{'type':'scan_coordinates','latitude':1.,'longitude':2.},{'type':'inspect_cell','h3_index':FLAT}],scan,'scan-a','now-a',FLAT)
        with self.assertRaisesRegex(ValueError,'distinct'):
            validate_actions([{'type':'compare_cells','primary_h3':FLAT,'secondary_h3':FLAT}],scan,'scan-a','now-a',FLAT)

    def test_training_examples_cover_exact_action_schema_without_new_source_rows(self):
        evidence=scan_fixture();seen=set()
        for index in range(8):
            prompt,envelope=training_plan(evidence,index,11.548,76.134)
            actions=validate_actions(envelope['actions'],evidence,'scan-a','now-a',FLAT)
            seen.update(a['type'] for a in actions)
            parsed=plan_actions(prompt,evidence,FLAT)
            self.assertEqual([a['type'] for a in parsed],[a['type'] for a in actions],prompt)
            self.assertEqual(envelope['schema_version'],'agentic_app_v2')
        self.assertEqual(seen,set(ACTION_TYPES))


class LocalEndpoint(unittest.TestCase):
    def test_running_app_discovers_only_safe_local_trained_server_and_keeps_overrides(self):
        with tempfile.TemporaryDirectory() as directory,patch.dict('os.environ',{'TERRA_QWEN_URL':''}):
            path=Path(directory)/'endpoint.json';agent=TerraShieldQwenAgent(scan_service=object());agent._endpoint_config_paths=[path]
            self.assertEqual(agent.ollama_url,'http://localhost:11434')
            path.write_text(json.dumps({'source':'local_trained_adapter_server','url':'http://127.0.0.1:8001'}))
            self.assertEqual(agent.ollama_url,'http://127.0.0.1:8001')
            for url in ('http://example.com:8001','http://127.0.0.1:8001/private','http://user:secret@localhost:8001','https://localhost:8001','http://localhost:8001?x=1'):
                path.write_text(json.dumps({'source':'local_trained_adapter_server','url':url}))
                self.assertEqual(agent.ollama_url,'http://localhost:11434')
            with patch.dict('os.environ',{'TERRA_QWEN_URL':'http://localhost:9876/'}):self.assertEqual(agent.ollama_url,'http://localhost:9876')
            explicit=TerraShieldQwenAgent(scan_service=object(),ollama_url='http://localhost:7654');explicit._endpoint_config_paths=[path]
            self.assertEqual(explicit.ollama_url,'http://localhost:7654')


class Service:
    def __init__(self):self.scan=scan_fixture()
    def get_scan_state(self,*args):return {'response':self.scan}
    def inspect_hexagon(self,cid,*args):return self.scan['cell_registry'][cid]
    def trace_runout(self,cid,*args):return {'h3_index':cid,'status':'CONDITIONAL'}


class ControllerReplies(unittest.IsolatedAsyncioTestCase):
    async def test_history_budget_keeps_recent_whole_turns_and_preserves_current_evidence(self):
        agent=TerraShieldQwenAgent(scan_service=Service());agent._available_model=AsyncMock(return_value=('qwen2.5',2));calls=[]
        original=httpx.AsyncClient
        def respond(request):
            calls.append(json.loads(request.content))
            return httpx.Response(200,json={'message':{'content':json.dumps({'markdown':'The margin is {{fs}}.','claims':{'fs':'/selected_cell/fs_mid'}})}})
        history=[{'role':'user','content':'OLD_CONTEXT_MARKER'+'x'*2600},{'role':'assistant','content':'The recent conversation concerns the selected cliff.'}]
        with patch('backend.llm.state_copilot.httpx.AsyncClient',lambda **kw:original(transport=httpx.MockTransport(respond),**kw)):
            reply=await agent.chat('Explain this cell',scan_id='scan-a',scenario_id='now-a',selected_h3_index=CLIFF,history=history,autonomous=True)
        self.assertNotIn('OLD_CONTEXT_MARKER',json.dumps(calls[0]['messages']))
        self.assertEqual(calls[0]['messages'][1],history[-1])
        self.assertEqual(reply['history_context']['omitted_oldest'],1)
        self.assertEqual(reply['bound_claims'][0]['value'],1.38)

    async def test_comparison_prose_is_bound_to_the_primary_cell_the_ui_will_select(self):
        agent=TerraShieldQwenAgent(scan_service=Service());agent._available_model=AsyncMock(return_value=('qwen2.5',2));calls=[]
        original=httpx.AsyncClient
        def respond(request):
            calls.append(json.loads(request.content))
            return httpx.Response(200,json={'message':{'content':json.dumps({'markdown':'The primary margin is {{fs}}.','claims':{'fs':'/selected_cell/fs_mid'}})}})
        with patch('backend.llm.state_copilot.httpx.AsyncClient',lambda **kw:original(transport=httpx.MockTransport(respond),**kw)):
            reply=await agent.chat('Compare the steepest cliff with the valley floor',scan_id='scan-a',scenario_id='now-a',selected_h3_index=RISK,autonomous=True)
        evidence=json.loads(calls[0]['messages'][-1]['content'].split('ACTIVE_EVIDENCE\n')[1])
        action=reply['ui_action']['actions'][-1]
        self.assertEqual(evidence['selected_cell']['h3_index'],action['primary_h3'])
        self.assertEqual(reply['bound_claims'][0]['value'],1.38)

    async def test_finetuned_full_action_envelope_is_accepted_but_server_owns_plan_id(self):
        agent=TerraShieldQwenAgent(scan_service=Service());agent._available_model=AsyncMock(return_value=('qwen2.5',2))
        actions=plan_actions('Inspect the selected cell',scan_fixture(),FLAT)
        envelope=action_envelope(actions,'scan-a','now-a');envelope['plan_id']='untrusted-model-id'
        payload={'markdown':'The margin is {{fs}}.','claims':{'fs':'/selected_cell/fs_mid'},'ui_action':envelope}
        original=httpx.AsyncClient
        def respond(_):return httpx.Response(200,json={'message':{'content':json.dumps(payload)}})
        with patch('backend.llm.state_copilot.httpx.AsyncClient',lambda **kw:original(transport=httpx.MockTransport(respond),**kw)):
            result=await agent.chat('Inspect the selected cell',scan_id='scan-a',scenario_id='now-a',selected_h3_index=FLAT,autonomous=True)
        self.assertEqual(result['briefing_guard'],'VERIFIED_LLM_PROSE');self.assertEqual(result['action_guard'],'VERIFIED_LLM_ACTIONS')
        self.assertNotEqual(result['ui_action']['plan_id'],'untrusted-model-id')

    async def test_initial_and_existing_location_chains_queue_one_ui_scan_without_old_prose(self):
        agent=TerraShieldQwenAgent(scan_service=Service());agent._available_model=AsyncMock(side_effect=AssertionError('No old-state prose before a new scan'))
        for ids in ({},{'scan_id':'scan-a','scenario_id':'now-a'}):
            result=await agent.chat('Take me to 11.548, 76.134, switch to NASA LHASA and then back to TERRA-SHIELD, and inspect the steepest cliff',autonomous=True,**ids)
            self.assertEqual(result['briefing_guard'],'VERIFIED_TEMPLATE')
            self.assertEqual([a['type'] for a in result['ui_action']['actions']].count('scan_coordinates'),1)
            self.assertEqual(len(result['thought_steps']),3);self.assertEqual(result['ui_action']['execution_status'],'PLANNED')
            self.assertNotIn('h3_index',result['ui_action']['actions'][-1])

    async def test_semantic_inspection_uses_the_target_cell_in_verified_evidence(self):
        agent=TerraShieldQwenAgent(scan_service=Service());agent._available_model=AsyncMock(return_value=('qwen2.5',2));calls=[]
        def respond(req):
            calls.append(json.loads(req.content));return httpx.Response(200,json={'message':{'content':json.dumps({'markdown':'The margin is {{fs}}.','claims':{'fs':'/selected_cell/fs_mid'},'thought_steps':[{'reasoning':'PRIVATE_PAYLOAD_MUST_NOT_BE_RENDERED'}]})}})
        original=httpx.AsyncClient
        with patch('backend.llm.state_copilot.httpx.AsyncClient',lambda **kw:original(transport=httpx.MockTransport(respond),**kw)):
            result=await agent.chat('Inspect the steepest cliff',scan_id='scan-a',scenario_id='now-a',selected_h3_index=FLAT,autonomous=True)
        self.assertEqual(result['briefing_guard'],'VERIFIED_LLM_PROSE')
        self.assertEqual(result['bound_claims'][0]['value'],1.38)
        self.assertNotIn('PRIVATE_PAYLOAD',json.dumps(result['thought_steps']))
        evidence=json.loads(calls[0]['messages'][-1]['content'].split('ACTIVE_EVIDENCE\n')[1])
        self.assertEqual(evidence['selected_cell']['h3_index'],CLIFF)
        self.assertEqual(calls[0]['options']['num_predict'],1536)

    async def test_model_cannot_smuggle_an_unrequested_scan_into_verified_numeric_prose(self):
        agent=TerraShieldQwenAgent(scan_service=Service());agent._available_model=AsyncMock(return_value=('qwen2.5',2))
        payload={'markdown':'The margin is {{fs}}.','claims':{'fs':'/selected_cell/fs_mid'},
                 'ui_action':{'actions':[{'type':'scan_coordinates','latitude':45.,'longitude':10.}]}}
        original=httpx.AsyncClient
        def respond(_):return httpx.Response(200,json={'message':{'content':json.dumps(payload)}})
        with patch('backend.llm.state_copilot.httpx.AsyncClient',lambda **kw:original(transport=httpx.MockTransport(respond),**kw)):
            result=await agent.chat('Explain the selected cell',scan_id='scan-a',scenario_id='now-a',selected_h3_index=FLAT,autonomous=True)
        self.assertEqual(result['briefing_guard'],'WITHHELD_UNVERIFIED_LLM_PROSE')
        self.assertNotIn('scan_coordinates',[a['type'] for a in result['ui_action']['actions']])
        self.assertEqual(len(result['thought_steps']),3)


if __name__=='__main__':unittest.main()
