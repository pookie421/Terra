import test from 'node:test';
import assert from 'node:assert/strict';
import {ApiHttpError,requestJson,retryAfterDelayMs} from '../.test-build/apiRequest.js';

const busy=(code='ANALYSIS_CAPACITY_BUSY',retry='0.001')=>new Response(JSON.stringify({detail:'The analysis service is busy.',code}),{status:429,headers:{'Content-Type':'application/json','Retry-After':retry}});
const okay=()=>new Response(JSON.stringify({scan_id:'current-scan'}),{headers:{'Content-Type':'application/json'}});
async function mockedFetch(factory,run){
  const original=globalThis.fetch;globalThis.fetch=factory;
  try{return await run();}finally{globalThis.fetch=original;}
}

test('explicit pre-admission rejection retries the identical POST after waiting',async()=>{
  const calls=[],waiting=[];
  await mockedFetch(async(_url,init)=>{calls.push(init);return calls.length===1?busy():okay();},async()=>{
    const result=await requestJson('/api/v1/scan',{body:{latitude:1,longitude:2},headers:{'X-LLM-API-Key':'personal-secret'},admissionWaitMs:100,onAdmissionWait:value=>waiting.push(value)});
    assert.equal(result.scan_id,'current-scan');
  });
  assert.equal(calls.length,2);assert.equal(calls[0].method,'POST');assert.equal(calls[0].body,calls[1].body);
  assert.deepEqual(calls[0].headers,calls[1].headers);assert.deepEqual(waiting,[true,false]);
});

test('assistant admission rejection has the same bounded retry contract',async()=>{
  let calls=0;
  await mockedFetch(async()=>++calls===1?busy('ASSISTANT_CAPACITY_BUSY'):okay(),async()=>{
    assert.equal((await requestJson('/api/v1/chat',{body:{message:'Inspect this cell'},admissionWaitMs:100})).scan_id,'current-scan');
  });
  assert.equal(calls,2);
});

test('generic rate limits, missing Retry-After and upstream failures never replay a POST',async()=>{
  for(const response of [busy('PROVIDER_RATE_LIMIT'),new Response(JSON.stringify({detail:'Busy',code:'ANALYSIS_CAPACITY_BUSY'}),{status:429,headers:{'Content-Type':'application/json'}}),new Response(JSON.stringify({detail:'Provider unavailable'}),{status:502,headers:{'Content-Type':'application/json'}})]){
    let calls=0;
    await mockedFetch(async()=>{calls++;return response;},async()=>{
      await assert.rejects(requestJson('/api/v1/scan',{body:{latitude:1},admissionWaitMs:100}),error=>error instanceof ApiHttpError&&error.status===response.status&&error.backendResponded);
    });
    assert.equal(calls,1);
  }
});

test('retry requires explicit opt-in and preserves the server error detail',async()=>{
  let calls=0;
  await mockedFetch(async()=>{calls++;return busy();},async()=>{
    await assert.rejects(requestJson('/api/v1/scan',{body:{latitude:1}}),error=>error instanceof ApiHttpError&&error.message==='The analysis service is busy.');
  });
  assert.equal(calls,1);
});

test('network failure never retries or leaks an endpoint or credential',async()=>{
  let calls=0;
  await mockedFetch(async()=>{calls++;throw new TypeError('https://private.example/ personal-secret');},async()=>{
    await assert.rejects(requestJson('/api/v1/scan',{body:{api_key:'personal-secret'},admissionWaitMs:100}),error=>error instanceof TypeError&&!/private\.example|personal-secret/.test(error.message));
  });
  assert.equal(calls,1);
});

test('aborting admission wait cancels the timer and prevents another request',async()=>{
  const controller=new AbortController(),waiting=[];let calls=0;
  await mockedFetch(async()=>{calls++;return busy('ANALYSIS_CAPACITY_BUSY','10');},async()=>{
    await assert.rejects(requestJson('/api/v1/scan',{body:{latitude:1},signal:controller.signal,admissionWaitMs:1000,onAdmissionWait:value=>{waiting.push(value);if(value)controller.abort();}}),error=>error.name==='AbortError'&&/cancelled/.test(error.message));
  });
  assert.equal(calls,1);assert.deepEqual(waiting,[true,false]);
});

test('admission deadline expires without retrying before the declared delay',async()=>{
  let calls=0;const waiting=[];
  await mockedFetch(async()=>{calls++;return busy('ANALYSIS_CAPACITY_BUSY','10');},async()=>{
    await assert.rejects(requestJson('/api/v1/scan',{body:{latitude:1},admissionWaitMs:5,onAdmissionWait:value=>waiting.push(value)}),error=>error instanceof ApiHttpError&&error.status===429);
  });
  assert.equal(calls,1);assert.deepEqual(waiting,[true,false]);
});

test('one total timeout covers response parsing and admission waits',async()=>{
  const waiting=[];
  await mockedFetch(async()=>busy('ANALYSIS_CAPACITY_BUSY','10'),async()=>{
    await assert.rejects(requestJson('/api/v1/scan',{body:{latitude:1},timeoutMs:5,admissionWaitMs:1000,onAdmissionWait:value=>waiting.push(value)}),error=>error.name==='TimeoutError'&&/timed out/.test(error.message));
  });
  assert.deepEqual(waiting,[true,false]);
  await mockedFetch(async()=>({ok:true,status:200,headers:new Headers({'Content-Type':'application/json'}),json:()=>new Promise(()=>{})}),async()=>{
    await assert.rejects(requestJson('/api/v1/scan',{timeoutMs:5}),error=>error.name==='TimeoutError'&&/timed out/.test(error.message));
  });
});

test('an already aborted request makes no network call',async()=>{
  const controller=new AbortController();controller.abort();let calls=0;
  await mockedFetch(async()=>{calls++;return okay();},async()=>{
    await assert.rejects(requestJson('/api/v1/scan',{signal:controller.signal,admissionWaitMs:100}),error=>error.name==='AbortError');
  });
  assert.equal(calls,0);
});

test('Retry-After accepts positive delta/date values and safely bounds malformed or excessive delays',()=>{
  const now=Date.parse('2026-11-15T00:00:00Z');
  assert.equal(retryAfterDelayMs('3'),3000);assert.equal(retryAfterDelayMs('0.001'),1);
  assert.equal(retryAfterDelayMs('999'),10000);assert.equal(retryAfterDelayMs('Sun, 15 Nov 2026 00:00:05 GMT',now),5000);
  assert.equal(retryAfterDelayMs('Sun, 15 Nov 2026 00:01:00 GMT',now),10000);
  assert.equal(retryAfterDelayMs('Sun, 15 Nov 2026 00:00:00 GMT',now),1);
  for(const value of [null,'','garbage','-3','0','Infinity'])assert.equal(retryAfterDelayMs(value,now),3000);
});
