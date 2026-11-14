export class ApiHttpError extends Error {
  readonly status:number;
  readonly backendResponded:boolean;
  constructor(message:string,status:number,backendResponded:boolean){
    super(message);this.name='ApiHttpError';this.status=status;this.backendResponded=backendResponded;
  }
}

export interface RequestJsonOptions {
  body?:unknown;
  signal?:AbortSignal;
  timeoutMs?:number;
  headers?:Record<string,string>;
  admissionWaitMs?:number;
  onAdmissionWait?:(waiting:boolean)=>void;
}

const ADMISSION_CODES=new Set(['ANALYSIS_CAPACITY_BUSY','ASSISTANT_CAPACITY_BUSY']);
const MAX_ATTEMPTS=24;

/** Retry-After controls only requests the application explicitly rejected before admission. */
export function retryAfterDelayMs(value:string|null,now=Date.now()):number {
  const text=value?.trim()??'';
  if(/^\d+(?:\.\d+)?$/.test(text)){
    const seconds=Number(text);
    if(Number.isFinite(seconds)&&seconds>0)return Math.min(10000,Math.max(1,seconds*1000));
    return 3000;
  }
  const date=/^[A-Za-z]{3}, \d{2} [A-Za-z]{3} \d{4} \d{2}:\d{2}:\d{2} GMT$/.test(text)?Date.parse(text):NaN;
  return Number.isFinite(date)?Math.min(10000,Math.max(1,date-now)):3000;
}

function finitePositive(value:number|undefined,fallback:number){
  return typeof value==='number'&&Number.isFinite(value)&&value>0?value:fallback;
}

export async function requestJson<T>(url:string,options:RequestJsonOptions={}):Promise<T>{
  let timedOut=false;
  const controller=new AbortController();
  const cancelled=()=>new DOMException(timedOut?'The request timed out. Please try again.':'The request was cancelled or superseded.',timedOut?'TimeoutError':'AbortError');
  const check=()=>{if(controller.signal.aborted)throw cancelled();};
  if(options.signal?.aborted)throw cancelled();
  const timer=setTimeout(()=>{timedOut=true;controller.abort();},finitePositive(options.timeoutMs,180000));
  const abort=()=>controller.abort();options.signal?.addEventListener('abort',abort,{once:true});
  let waiting=false,firstRejection:number|undefined;
  const setWaiting=(next:boolean)=>{if(waiting!==next){waiting=next;options.onAdmissionWait?.(next);}};
  const waitLimit=Math.min(60000,finitePositive(options.admissionWaitMs,0));

  async function abortable<R>(operation:Promise<R>):Promise<R>{
    return new Promise<R>((resolve,reject)=>{
      const stop=()=>{cleanup();reject(cancelled());};
      const cleanup=()=>controller.signal.removeEventListener('abort',stop);
      controller.signal.addEventListener('abort',stop,{once:true});
      operation.then(value=>{cleanup();resolve(value);},error=>{cleanup();reject(error);});
      if(controller.signal.aborted)stop();
    });
  }
  async function sleep(ms:number){
    check();
    await new Promise<void>((resolve,reject)=>{
      const stop=()=>{clearTimeout(delay);cleanup();reject(cancelled());};
      const cleanup=()=>controller.signal.removeEventListener('abort',stop);
      const delay=setTimeout(()=>{cleanup();resolve();},ms);
      controller.signal.addEventListener('abort',stop,{once:true});
      if(controller.signal.aborted)stop();
    });
  }

  try{
    // Serialize once. A rejected attempt never changes a personal-key request or its plan.
    const serialized=options.body===undefined?undefined:JSON.stringify(options.body);
    const headers=options.body===undefined?options.headers:{'Content-Type':'application/json',...options.headers};
    for(let attempt=1;attempt<=MAX_ATTEMPTS;attempt++){
      check();setWaiting(false);check();
      let response:Response;
      try{
        response=await abortable(fetch(url,{signal:controller.signal,method:serialized===undefined?'GET':'POST',headers,body:serialized}));
      }catch(error){
        if(controller.signal.aborted||error instanceof DOMException&&error.name==='AbortError')throw cancelled();
        // Native/third-party transport errors can contain endpoint URLs. Never expose them.
        throw new TypeError('The service could not be reached. Check your connection and try again.');
      }
      check();
      const isJson=response.headers.get('content-type')?.toLowerCase().includes('application/json')??false;
      if(response.ok){
        if(!isJson)throw new ApiHttpError('API did not return JSON. Check the FastAPI service and Vite proxy.',response.status,false);
        try{return await abortable(response.json()) as T;}
        catch(error){if(controller.signal.aborted)throw cancelled();throw new ApiHttpError('The service returned an invalid JSON response.',response.status,true);}
      }
      let payload:Record<string,unknown>={};
      if(isJson){
        try{
          const parsed:unknown=await abortable(response.json());
          if(parsed&&typeof parsed==='object'&&!Array.isArray(parsed))payload=parsed as Record<string,unknown>;
        }catch(error){if(controller.signal.aborted)throw cancelled();}
      }
      const failure=new ApiHttpError(typeof payload.detail==='string'?payload.detail:`API request failed (${response.status})`,response.status,isJson);
      const retryHeader=response.headers.get('retry-after');
      const admission=response.status===429&&typeof payload.code==='string'&&ADMISSION_CODES.has(payload.code)&&retryHeader!==null;
      if(!admission||waitLimit<=0||attempt===MAX_ATTEMPTS)throw failure;
      firstRejection??=performance.now();
      const remaining=waitLimit-(performance.now()-firstRejection);
      if(remaining<=0)throw failure;
      const delay=retryAfterDelayMs(retryHeader),exhausts=delay>=remaining;
      setWaiting(true);
      await sleep(Math.min(delay,remaining));
      setWaiting(false);
      check();if(exhausts)throw failure;
    }
    throw new ApiHttpError('The service is still busy. Please try again shortly.',429,true);
  }finally{
    clearTimeout(timer);options.signal?.removeEventListener('abort',abort);setWaiting(false);
  }
}
