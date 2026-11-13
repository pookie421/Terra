"""Hosted prose must obey the same evidence/action contract as local inference."""
import json
from pathlib import Path
import socket
import tempfile
import unittest
from unittest.mock import patch

import httpx

from backend.api import main
from backend.api.scan_service import GlobalScanService
from backend.llm.model_client import assistant_config, connection_summary, chat_json_content, AssistantConfigurationError
from backend.llm.qwen_agent import TerraShieldQwenAgent
from scripts.configure_assistant import save_settings
from test_autonomous_api import forcing_fixture

HOSTED={'TERRA_LLM_PROVIDER':'openai_compatible','TERRA_LLM_BASE_URL':'https://models.example/v1',
        'TERRA_LLM_MODEL':'example-json-model','TERRA_LLM_API_KEY':'test-key-never-public'}


class HostedConfiguration(unittest.TestCase):
    def test_only_explicit_operator_configuration_enables_hosted_requests(self):
        self.assertEqual(assistant_config({'OPENAI_API_KEY':'unrelated','EARTHDATA_TOKEN':'other'}).provider,'ollama')
        self.assertFalse(assistant_config({'TERRA_LLM_PROVIDER':'openai_compatible'}).configured)
        summary=connection_summary(HOSTED)
        self.assertTrue(summary['configured'])
        self.assertEqual(summary['availability'],'not_checked')
        self.assertNotIn(HOSTED['TERRA_LLM_API_KEY'],json.dumps(summary))
        self.assertNotIn(HOSTED['TERRA_LLM_BASE_URL'],json.dumps(summary))
        self.assertNotIn(HOSTED['TERRA_LLM_API_KEY'],repr(assistant_config(HOSTED)))

    def test_endpoint_validation_rejects_credentials_queries_and_nonlocal_plaintext(self):
        for url in ('http://public.example/v1','https://user:secret@models.example/v1',
                    'https://models.example/v1?key=secret','https://models.example/v1#secret',
                    'https://models.example:bad/v1'):
            with self.subTest(url=url),self.assertRaises(AssistantConfigurationError):
                assistant_config({**HOSTED,'TERRA_LLM_BASE_URL':url})
        self.assertTrue(assistant_config({**HOSTED,'TERRA_LLM_BASE_URL':'http://127.0.0.1:8001/v1','TERRA_LLM_API_KEY':''}).configured)

    def test_interactive_setup_preserves_other_provider_keys_and_validates_before_write(self):
        from dotenv import dotenv_values
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'.env'
            original='EARTHDATA_TOKEN=existing-earthdata-token\nTERRA_NISAR_MANIFEST=/local/radar.json\n'
            path.write_text(original)
            with self.assertRaises(AssistantConfigurationError):
                save_settings(path,{**HOSTED,'TERRA_LLM_BASE_URL':'http://external.example'})
            self.assertEqual(path.read_text(),original)
            save_settings(path,HOSTED)
            saved=dotenv_values(path,interpolate=False)
            self.assertEqual(saved['EARTHDATA_TOKEN'],'existing-earthdata-token')
            self.assertEqual(saved['TERRA_LLM_API_KEY'],HOSTED['TERRA_LLM_API_KEY'])


class HostedProse(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.service=GlobalScanService()
        self.agent=TerraShieldQwenAgent(self.service)
        self.scan=await self.service.scan(_snapshot=forcing_fixture())
        self.guard=patch.object(socket.socket,'connect',side_effect=AssertionError('No live provider requests'))
        self.guard.start();self.addCleanup(self.guard.stop)

    async def reply(self,responder):
        original=httpx.AsyncClient
        with patch.dict('os.environ',HOSTED,clear=True),patch('backend.llm.state_copilot.httpx.AsyncClient',
            lambda **kw:original(transport=httpx.MockTransport(responder),**kw)):
            return await self.agent.chat('Explain the current column',scan_id=self.scan['scan_id'],
                scenario_id=self.scan['scenario_id'],selected_h3_index='893969a401bffff',autonomous=True)

    async def test_hosted_response_is_bound_to_the_selected_state_and_secret_stays_in_header(self):
        calls=[]
        def respond(request):
            body=json.loads(request.content);calls.append(body)
            self.assertEqual(request.url.path,'/v1/chat/completions')
            self.assertEqual(request.headers['authorization'],'Bearer '+HOSTED['TERRA_LLM_API_KEY'])
            self.assertEqual(body['response_format'],{'type':'json_object'})
            self.assertEqual(body['max_tokens'],1536)
            self.assertNotIn(HOSTED['TERRA_LLM_API_KEY'],json.dumps(body))
            evidence=json.loads(body['messages'][-1]['content'].split('ACTIVE_EVIDENCE\n')[1])
            prose={'markdown':'The reference margin is {{margin}}.','claims':{'margin':'/selected_cell/fs_mid'},
                   'ui_action':{'actions':evidence['authorized_action_plan']}}
            return httpx.Response(200,json={'choices':[{'finish_reason':'stop','message':{
                'content':json.dumps(prose),'reasoning_content':'Private model text must not reach the interface'}}]})
        result=await self.reply(respond)
        self.assertEqual(len(calls),1)
        self.assertEqual(result['briefing_guard'],'VERIFIED_LLM_PROSE')
        self.assertEqual(result['scan_id'],self.scan['scan_id'])
        evidence=json.loads(calls[0]['messages'][-1]['content'].split('ACTIVE_EVIDENCE\n')[1])
        self.assertEqual(result['bound_claims'][0]['value'],evidence['selected_cell']['fs_mid'])
        self.assertNotIn(HOSTED['TERRA_LLM_API_KEY'],json.dumps(result))
        self.assertNotIn('Private model text',json.dumps(result))

    async def test_provider_failure_uses_verified_template_without_echoing_response_or_retrying(self):
        calls=[]
        def respond(request):
            calls.append(request)
            return httpx.Response(429,json={'error':{'message':'Echoed secret '+HOSTED['TERRA_LLM_API_KEY']}})
        result=await self.reply(respond)
        self.assertEqual(len(calls),1)
        self.assertEqual(result['briefing_guard'],'VERIFIED_TEMPLATE')
        self.assertEqual(result['model_error'],'ASSISTANT_PROVIDER_HTTP_429')
        self.assertNotIn(HOSTED['TERRA_LLM_API_KEY'],json.dumps(result))

    async def test_cloud_model_cannot_add_an_unrequested_map_action(self):
        def respond(request):
            prose={'markdown':'The margin is {{margin}}.','claims':{'margin':'/selected_cell/fs_mid'},
                   'ui_action':{'actions':[{'type':'scan_coordinates','latitude':0.,'longitude':0.,'label':'Unrequested'}]}}
            return httpx.Response(200,json={'choices':[{'message':{'content':json.dumps(prose)}}]})
        result=await self.reply(respond)
        self.assertEqual(result['briefing_guard'],'WITHHELD_UNVERIFIED_LLM_PROSE')
        self.assertEqual([action['type'] for action in result['ui_action']['actions']],['inspect_cell'])
        self.assertEqual(result['ui_action']['actions'][0]['h3_index'],'893969a401bffff')

    async def test_connections_endpoint_is_read_only_and_redacted(self):
        with patch.dict('os.environ',{**HOSTED,'EARTHDATA_TOKEN':'private-earthdata'},clear=True):
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=main.app),base_url='http://test') as client:
                response=await client.get('/api/v1/connections')
                self.assertEqual(response.status_code,200)
                result=response.json()
                self.assertEqual(result['scope'],'deployment')
                self.assertTrue(result['earthdata']['configured'])
                self.assertFalse(result['assistant']['requires_personal_key'])
                self.assertNotIn('private-earthdata',response.text)
                self.assertNotIn(HOSTED['TERRA_LLM_API_KEY'],response.text)
                self.assertEqual((await client.post('/api/v1/connections',json={'api_key':'untrusted'})).status_code,405)

    async def test_provider_specific_json_parameters_are_not_sent_to_other_endpoints(self):
        for base,model,thinking,router in (
            ('https://workspace.ap-southeast-1.maas.aliyuncs.com/compatible-mode/v1','qwen-plus',False,False),
            ('https://dashscope.aliyuncs.com/compatible-mode/v1','qwen2.5-7b-instruct',None,False),
            ('https://openrouter.ai/api/v1','qwen/example',None,True)):
            def respond(request):
                body=json.loads(request.content)
                self.assertEqual(body.get('enable_thinking'),thinking)
                self.assertEqual(body.get('provider',{}).get('require_parameters',False),router)
                return httpx.Response(200,json={'choices':[{'message':{'content':'{"markdown":"Bound evidence.","claims":{}}'}}]})
            with patch.dict('os.environ',{**HOSTED,'TERRA_LLM_BASE_URL':base,'TERRA_LLM_MODEL':model},clear=True):
                async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
                    await chat_json_content(client,model,[{'role':'user','content':'JSON only'}],lambda:(_ for _ in ()).throw(AssertionError('Hosted calls must not inspect local endpoint files')))


if __name__=='__main__':unittest.main()
