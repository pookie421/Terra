import test from 'node:test';
import assert from 'node:assert/strict';
import {ASSISTANT_STORAGE_KEY,DEFAULT_ASSISTANT,assistantReady,assistantRequest,changeAiProvider,readAssistantSettings,writeAssistantSettings} from '../.test-build/assistantSettings.js';
function storage(){const values=new Map();return {getItem:key=>values.get(key)??null,setItem:(key,value)=>values.set(key,value),removeItem:key=>values.delete(key)};}

test('personal keys use only supplied session storage and clear without retaining the key',()=>{
  const session=storage(),configured={...DEFAULT_ASSISTANT,apiKey:'personal-test-key'};
  assert.equal(writeAssistantSettings(session,configured),true);
  assert.deepEqual(readAssistantSettings(session),configured);
  writeAssistantSettings(session,{...configured,apiKey:''});
  assert.equal(readAssistantSettings(session).apiKey,'');
  assert.ok(!session.getItem(ASSISTANT_STORAGE_KEY).includes('personal-test-key'));
});
test('switching provider removes the previous provider credential and custom endpoint',()=>{
  const custom={provider:'custom_openai',model:'private-model',apiKey:'must-not-forward',baseUrl:'https://old.example/v1'};
  const next=changeAiProvider(custom,'gemini');
  assert.equal(next.apiKey,'');assert.equal(next.baseUrl,'');assert.equal(next.provider,'gemini');assert.ok(next.model.startsWith('gemini'));
});
test('invalid stored settings or unavailable storage safely preserve keyless mode',()=>{
  assert.equal(readAssistantSettings({getItem:()=>'{broken'}).apiKey,'');
  assert.equal(readAssistantSettings({getItem:()=>'{"provider":"invented","apiKey":"secret"}'}).apiKey,'');
  assert.equal(writeAssistantSettings({setItem:()=>{throw new Error('blocked');}},DEFAULT_ASSISTANT),false);
});
test('custom endpoint readiness requires a model and credential plus an uncredentialed HTTPS URL',()=>{
  const value={provider:'custom_openai',model:'model',apiKey:'key',baseUrl:'https://provider.example/v1'};
  assert.equal(assistantReady(value),true);
  for(const baseUrl of ['http://provider.example','https://user:password@provider.example','https://provider.example/?key=secret','not-a-url'])assert.equal(assistantReady({...value,baseUrl}),false);
  assert.equal(assistantReady({...value,apiKey:''}),false);
  assert.equal(assistantReady({...value,apiKey:'key\nother'}),false);
});
test('request credentials travel in the dedicated header and never in the JSON configuration',()=>{
  const value={provider:'custom_openai',model:' model ',apiKey:' secret-personal-key ',baseUrl:'https://provider.example/v1'};
  const request=assistantRequest(value);
  assert.equal(request.headers['X-LLM-API-Key'],'secret-personal-key');
  assert.equal(request.config.model,'model');assert.equal(request.config.base_url,value.baseUrl);
  assert.ok(!JSON.stringify(request.config).includes('secret-personal-key'));
  assert.deepEqual(assistantRequest({...value,apiKey:''}),{config:{},headers:{}});
});
