import type {Scan,ChatReply,Cell,Runout,Health,HorizonKey,PredictionModel} from './types';
const base=(import.meta.env.VITE_API_URL as string|undefined)?.replace(/\/$/,'')??'';
export class ApiHttpError extends Error {
  readonly status:number;
  readonly backendResponded:boolean;
  constructor(message:string,status:number,backendResponded:boolean){super(message);this.name='ApiHttpError';this.status=status;this.backendResponded=backendResponded;}
}
function conversationWindow(history:{role:'user'|'assistant';content:string}[]){
  const chosen:typeof history=[];let remaining=16000;
  for(let i=history.length-1;i>=0&&chosen.length<12;i--){
    const content=history[i].content.slice(0,4000);
    if(content.length>remaining)break;
    chosen.unshift({role:history[i].role,content});remaining-=content.length;
  }
  return chosen;
}
async function request<T>(path:string,body?:unknown,signal?:AbortSignal,timeoutMs=180000):Promise<T>{
  if(signal?.aborted)throw new DOMException('The request was superseded.','AbortError');
  const controller=new AbortController(),timer=setTimeout(()=>controller.abort(),timeoutMs);
  const abort=()=>controller.abort();signal?.addEventListener('abort',abort,{once:true});
  try{
    const response=await fetch(base+path,{signal:controller.signal,method:body===undefined?'GET':'POST',headers:body===undefined?undefined:{'Content-Type':'application/json'},body:body===undefined?undefined:JSON.stringify(body)});
    const isJson=response.headers.get('content-type')?.includes('application/json')??false;
    if(!response.ok){const error=isJson?await response.json().catch(()=>({})):{};throw new ApiHttpError(typeof error.detail==='string'?error.detail:`API request failed (${response.status})`,response.status,isJson);}
    if(!isJson)throw new ApiHttpError('API did not return JSON. Check the FastAPI service and Vite proxy.',response.status,false);
    return await response.json() as T;
  }finally{clearTimeout(timer);signal?.removeEventListener('abort',abort);}
}
export const api={
  health:()=>request<Health>('/api/v1/health',undefined,undefined,6000),
  scan:(location:string|{latitude:number;longitude:number},signal?:AbortSignal)=>request<Scan>('/api/v1/scan',typeof location==='string'?{query:location}:location,signal),
  forecast:(scanId:string,key:HorizonKey,signal?:AbortSignal)=>request<Scan>(`/api/v1/scans/${encodeURIComponent(scanId)}/forecast/${key}`,undefined,signal),
  inspect:(scan:Scan,id:string,signal?:AbortSignal)=>request<Cell&{physics_informed_assessment?:Cell}>(`/api/v1/inspect/${scan.scan_id}/${id}?scenario_id=${scan.scenario_id}`,undefined,signal,15000),
  runout:(scan:Scan,id:string,signal?:AbortSignal)=>request<Runout>(`/api/v1/scans/${scan.scan_id}/scenarios/${scan.scenario_id}/runout/${id}`,undefined,signal,15000),
  chat:(scan:Scan|null,id:string,message:string,history:{role:'user'|'assistant';content:string}[],signal?:AbortSignal,active_prediction_model:PredictionModel='terra_physics_ml')=>request<ChatReply>('/api/v1/chat',{message,scan_id:scan?.scan_id??null,scenario_id:scan?.scenario_id??null,selected_h3_index:id||null,history:conversationWindow(history),autonomous:true,active_prediction_model},signal,90000),
};
