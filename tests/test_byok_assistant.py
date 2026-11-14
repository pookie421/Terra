"""Personal keys remain request scoped; every provider uses the same guards."""
import asyncio
import json
import socket
import unittest
from unittest.mock import AsyncMock, patch

import httpx

from backend.api import main
from backend.api.scan_service import GlobalScanService
from backend.llm.app_actions import plan_actions
from backend.llm.byok_client import (ENDPOINTS, MAX_RESPONSE_BYTES, request_assistant_config,
                                     request_chat_json_content)
from backend.llm.model_client import AssistantConfigurationError, AssistantServiceError
from backend.llm.qwen_agent import TerraShieldQwenAgent
from backend.llm.state_copilot import context, template
from test_agentic_copilot import scan_fixture, FLAT, CLIFF, RISK
from test_autonomous_api import forcing_fixture


class PersonalTransport(unittest.IsolatedAsyncioTestCase):
    async def test_all_named_providers_use_their_own_endpoint_and_secret_header(self):
        for provider, endpoint in ENDPOINTS.items():
            with self.subTest(provider=provider):
                config=request_assistant_config(provider,'test-model',api_key='secret-'+provider)
                def respond(request):
                    self.assertEqual(str(request.url),endpoint)
                    body=json.loads(request.content)
                    self.assertNotIn(config.api_key,json.dumps(body))
                    self.assertEqual(body['model'],'test-model')
                    content='{"markdown":"Bound evidence.","claims":{}}'
                    if provider=='anthropic':
                        self.assertEqual(request.headers['x-api-key'],config.api_key)
                        self.assertEqual(request.headers['anthropic-version'],'2023-06-01')
                        self.assertEqual(body['system'],'JSON only')
                        self.assertEqual([m['role'] for m in body['messages']],['user'])
                        return httpx.Response(200,json={'stop_reason':'end_turn','content':[
                            {'type':'thinking','thinking':'private'}, {'type':'text','text':content}]})
                    self.assertEqual(request.headers['authorization'],'Bearer '+config.api_key)
                    self.assertEqual(body['response_format'],{'type':'json_object'})
                    return httpx.Response(200,json={'choices':[{'message':{'content':content}}]})
                async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
                    value=await request_chat_json_content(client,config,[
                        {'role':'system','content':'JSON only'},{'role':'user','content':'Explain'}])
                self.assertNotIn('private',value)
                self.assertEqual(json.loads(value)['claims'],{})

    async def test_custom_requests_pin_public_dns_and_preserve_tls_identity(self):
        config=request_assistant_config('custom_openai','model','https://models.example/v1','custom-secret')
        addresses=[(socket.AF_INET,socket.SOCK_STREAM,6,'',('93.184.216.34',443))]
        def respond(request):
            self.assertEqual(request.url.host,'93.184.216.34')
            self.assertEqual(request.headers['host'],'models.example')
            self.assertEqual(request.extensions['sni_hostname'],'models.example')
            self.assertEqual(request.url.path,'/v1/chat/completions')
            return httpx.Response(200,json={'choices':[{'message':{'content':'{}'}}]})
        with patch.object(asyncio.get_running_loop(),'getaddrinfo',AsyncMock(return_value=addresses)):
            async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
                self.assertEqual(await request_chat_json_content(client,config,[]),'{}')

    async def test_private_dns_and_redirects_never_receive_a_forwarded_key(self):
        config=request_assistant_config('custom_openai','model','https://models.example/v1','secret')
        addresses=[(socket.AF_INET,socket.SOCK_STREAM,6,'',('169.254.169.254',443))]
        with patch.object(asyncio.get_running_loop(),'getaddrinfo',AsyncMock(return_value=addresses)):
            async with httpx.AsyncClient(transport=httpx.MockTransport(lambda _:self.fail('Private destination contacted'))) as client:
                with self.assertRaisesRegex(AssistantServiceError,'NOT_PUBLIC'):
                    await request_chat_json_content(client,config,[])
        calls=[]
        def redirect(request):
            calls.append(request)
            return httpx.Response(302,headers={'Location':'https://foreign.example/steal'})
        async with httpx.AsyncClient(transport=httpx.MockTransport(redirect),follow_redirects=True) as client:
            with self.assertRaisesRegex(AssistantServiceError,'HTTP_302'):
                await request_chat_json_content(client,request_assistant_config('openai','model',api_key='secret'),[])
        self.assertEqual(len(calls),1)

    async def test_dns_timeout_and_response_body_limits_are_enforced_before_parsing(self):
        config=request_assistant_config('custom_openai','model','https://models.example/v1','secret')
        async def expire(awaitable,timeout):
            self.assertEqual(timeout,5.)
            awaitable.close()
            raise asyncio.TimeoutError
        with patch.object(asyncio.get_running_loop(),'getaddrinfo',AsyncMock()),patch(
                'backend.llm.byok_client.asyncio.wait_for',side_effect=expire):
            async with httpx.AsyncClient(transport=httpx.MockTransport(lambda _:self.fail('DNS timeout contacted provider'))) as client:
                with self.assertRaisesRegex(AssistantServiceError,'DNS_FAILED'):
                    await request_chat_json_content(client,config,[])
        class OversizedStream(httpx.AsyncByteStream):
            sent=0
            async def __aiter__(self):
                for _ in range(100):
                    self.sent+=1
                    yield b'x'*16384
        stream=OversizedStream()
        def respond(request):
            self.assertEqual(request.headers['accept-encoding'],'identity')
            return httpx.Response(200,stream=stream)
        async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
            with self.assertRaisesRegex(AssistantServiceError,'RESPONSE_TOO_LARGE'):
                await request_chat_json_content(client,request_assistant_config('openai','model',api_key='secret'),[])
        self.assertLessEqual(stream.sent,MAX_RESPONSE_BYTES//16384+1)

    async def test_unrequested_compression_is_rejected_before_decompression(self):
        async with httpx.AsyncClient(transport=httpx.MockTransport(lambda _:httpx.Response(
                200,headers={'Content-Encoding':'gzip'},stream=httpx.ByteStream(b'unread')))) as client:
            with self.assertRaisesRegex(AssistantServiceError,'ENCODING_UNSUPPORTED'):
                await request_chat_json_content(client,request_assistant_config('openai','model',api_key='secret'),[])

    def test_invalid_custom_urls_and_credentials_have_value_free_errors(self):
        for url in ('http://models.example/v1','https://127.0.0.1/v1','https://[::1]/v1','https://224.0.0.1/v1',
                    'https://localhost/v1','https://user:private@models.example/v1',
                    'https://models.example/v1?key=private','https://models.example:8000/v1'):
            with self.subTest(url=url),self.assertRaises(AssistantConfigurationError) as error:
                request_assistant_config('custom_openai','model',url,'private')
            self.assertNotIn(url,str(error.exception))
        self.assertNotIn('secret',repr(request_assistant_config('openai','model',api_key='secret')))


class PersonalPublicChat(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.service=GlobalScanService()
        self.agent=TerraShieldQwenAgent(self.service)
        self.scan=await self.service.scan(_snapshot=forcing_fixture())
        self.old_agent=main.agent
        main.agent=self.agent
        self.addCleanup(setattr,main,'agent',self.old_agent)
        self.client=httpx.AsyncClient(transport=httpx.ASGITransport(app=main.app),base_url='http://test')
        self.addAsyncCleanup(self.client.aclose)
        self.guard=patch.object(socket.socket,'connect',side_effect=AssertionError('No live provider requests'))
        self.guard.start();self.addCleanup(self.guard.stop)

    def body(self,**extra):
        return {'message':'Explain the selected cell','scan_id':self.scan['scan_id'],
                'scenario_id':self.scan['scenario_id'],'selected_h3_index':CLIFF,
                'autonomous':True,**extra}

    async def test_absent_personal_key_ignores_every_server_key_and_local_service(self):
        server={'TERRA_LLM_PROVIDER':'openai_compatible','TERRA_LLM_BASE_URL':'https://provider.example/v1',
                'TERRA_LLM_MODEL':'paid-model','TERRA_LLM_API_KEY':'server-secret',
                'TERRA_DEFAULT_LLM_API_KEY':'default-server-secret'}
        with patch.dict('os.environ',server),patch.object(self.agent,'_available_model',AsyncMock(side_effect=AssertionError('Local discovery forbidden'))):
            response=await self.client.post('/api/v1/chat',json=self.body())
        self.assertEqual(response.status_code,200,response.text)
        reply=response.json()
        self.assertEqual(reply['briefing_guard'],'VERIFIED_TEMPLATE')
        self.assertTrue(reply['numeric_audit_passed'])
        self.assertEqual(reply['fallback_reason'],'PERSONAL_API_KEY_REQUIRED')
        self.assertEqual(reply['assistant_mode'],'deterministic')
        self.assertEqual(reply['ui_action']['actions'][-1]['type'],'inspect_cell')
        self.assertNotIn('server-secret',response.text)

    async def test_header_overrides_body_key_and_prose_is_numeric_verified(self):
        original=httpx.AsyncClient
        def respond(request):
            self.assertEqual(request.headers['authorization'],'Bearer header-secret')
            payload=json.loads(request.content)
            self.assertNotIn('body-secret',json.dumps(payload))
            evidence=json.loads(payload['messages'][-1]['content'].split('ACTIVE_EVIDENCE\n')[1])
            self.assertEqual(evidence['scan_id'],self.scan['scan_id'])
            prose={'markdown':'The reference margin is {{margin}}.',
                   'claims':{'margin':'/selected_cell/fs_mid'},
                   'ui_action':{'actions':evidence['authorized_action_plan']}}
            return httpx.Response(200,json={'choices':[{'message':{'content':json.dumps(prose)}}]})
        with patch('backend.llm.state_copilot.httpx.AsyncClient',lambda **kw:original(transport=httpx.MockTransport(respond),**kw)):
            response=await self.client.post('/api/v1/chat',headers={'X-LLM-API-Key':'header-secret'},
                json=self.body(provider='gemini',model='chosen-model',api_key='body-secret'))
        self.assertEqual(response.status_code,200,response.text)
        result=response.json()
        self.assertEqual(result['briefing_guard'],'VERIFIED_LLM_PROSE')
        self.assertEqual(result['assistant_provider'],'gemini')
        self.assertEqual(result['model_name'],'chosen-model')
        self.assertEqual(result['model_provenance']['availability'],'verified_response')
        self.assertEqual(result['bound_claims'][0]['path'],'/selected_cell/fs_mid')
        for secret in ('header-secret','body-secret'):
            self.assertNotIn(secret,response.text)

    async def test_concurrent_personal_requests_do_not_share_credentials(self):
        original=httpx.AsyncClient;seen={}
        async def respond(request):
            body=json.loads(request.content)
            await asyncio.sleep(0)
            seen[body['model']]=request.headers['authorization']
            return httpx.Response(200,json={'choices':[{'message':{'content':json.dumps({
                'markdown':'The margin is {{margin}}.','claims':{'margin':'/selected_cell/fs_mid'}})}}]})
        with patch('backend.llm.state_copilot.httpx.AsyncClient',lambda **kw:original(transport=httpx.MockTransport(respond),**kw)):
            responses=await asyncio.gather(*[self.client.post('/api/v1/chat',
                headers={'X-LLM-API-Key':'secret-'+name},json=self.body(provider='groq',model=name))
                for name in ('first','second')])
        self.assertEqual(seen,{'first':'Bearer secret-first','second':'Bearer secret-second'})
        self.assertTrue(all(r.json()['briefing_guard']=='VERIFIED_LLM_PROSE' for r in responses))

    async def test_provider_error_and_validation_never_echo_credentials(self):
        original=httpx.AsyncClient
        def respond(request):
            return httpx.Response(401,json={'error':'Reflected header-secret'})
        with patch('backend.llm.state_copilot.httpx.AsyncClient',lambda **kw:original(transport=httpx.MockTransport(respond),**kw)):
            response=await self.client.post('/api/v1/chat',headers={'X-LLM-API-Key':'header-secret'},
                json=self.body(provider='openai',model='chosen'))
        self.assertEqual(response.json()['briefing_guard'],'VERIFIED_TEMPLATE')
        self.assertEqual(response.json()['model_error'],'ASSISTANT_PROVIDER_HTTP_401')
        self.assertNotIn('header-secret',response.text)
        invalid=await self.client.post('/api/v1/chat',json=self.body(api_key=['malformed-secret']))
        self.assertEqual(invalid.status_code,422)
        self.assertNotIn('malformed-secret',invalid.text)

    async def test_unbound_numbers_and_reflected_keys_are_withheld(self):
        original=httpx.AsyncClient
        for markdown in ('The factor of safety is 0.1.','Reflected header-secret.'):
            calls=[]
            def respond(request):
                calls.append(request)
                return httpx.Response(200,json={'choices':[{'message':{'content':json.dumps({
                    'markdown':markdown,'claims':{}})}}]})
            with patch('backend.llm.state_copilot.httpx.AsyncClient',lambda **kw:original(transport=httpx.MockTransport(respond),**kw)):
                response=await self.client.post('/api/v1/chat',headers={'X-LLM-API-Key':'header-secret'},
                    json=self.body(provider='openai',model='chosen'))
            self.assertEqual(len(calls),2)
            self.assertEqual(response.json()['briefing_guard'],'WITHHELD_UNVERIFIED_LLM_PROSE')
            self.assertFalse(response.json()['numeric_audit_passed'])
            self.assertNotIn(markdown,response.text)
            self.assertNotIn('header-secret',response.text)


class DailyNasaEvidence(unittest.TestCase):
    def scan(self):
        scan=scan_fixture()
        scan['active_prediction_model']='nasa_lhasa_2'
        scan['nasa_lhasa']={'mode':'PARTIAL_LIVE','cells':[
            {'h3_index':FLAT,'official_score':.01,'tomorrow_probability':.8},
            {'h3_index':CLIFF,'official_score':.1,'tomorrow_probability':.02},
            {'h3_index':RISK,'official_score':None,'tomorrow_probability':None}]}
        return scan

    def test_product_changes_selected_evidence_and_critical_cell_using_official_only(self):
        scan=self.scan()
        self.assertEqual(plan_actions('Inspect the riskiest cell',scan,CLIFF)[-1]['h3_index'],CLIFF)
        scan['nasa_product']='tomorrow'
        self.assertEqual(plan_actions('Inspect the riskiest cell',scan,CLIFF)[-1]['h3_index'],FLAT)
        evidence=context(scan,'nasa_lhasa_2','tomorrow')
        self.assertEqual(evidence['nasa_product'],'tomorrow')
        self.assertIsNone(evidence['selected_cell']['lhasa_probability'])
        self.assertIn('official NASA Tomorrow score is unavailable',template(evidence))
        scan['physics_informed_assessment']['hotspot']=scan['cell_registry'][FLAT]
        evidence=context(scan,'nasa_lhasa_2','tomorrow')
        self.assertEqual(evidence['selected_cell']['lhasa_probability'],.8)
        self.assertEqual(evidence['selected_cell']['prediction_scores']['nasa_lhasa_2'],.8)
        self.assertIn("belong to Terra's separate conditional analysis, not NASA LHASA",template(evidence))

    def test_nasa_daily_actions_never_fabricate_hourly_forecasts_or_reference_ranking(self):
        scan=self.scan()
        actions=plan_actions('Show tomorrow and inspect the riskiest cell',scan,CLIFF)
        self.assertEqual(actions[0]['horizon'],'nasa_tomorrow')
        self.assertEqual(actions[-1]['target'],'riskiest')
        self.assertEqual(plan_actions('Show Today',scan,CLIFF)[0]['horizon'],'nasa_today')
        for horizon in ('24 hours','48 hours','72 hours','peak'):
            with self.subTest(horizon=horizon),self.assertRaisesRegex(ValueError,'relative Today and Tomorrow'):
                plan_actions('Set '+horizon,scan,CLIFF)
        scan['nasa_lhasa']={'cells':[]}
        with self.assertRaisesRegex(ValueError,'no official cell scores'):
            plan_actions('Inspect the riskiest cell',scan,CLIFF)


if __name__=='__main__':unittest.main()
