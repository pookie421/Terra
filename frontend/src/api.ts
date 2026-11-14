import type {Scan,ChatReply,Cell,Runout,Health,HorizonKey,PredictionModel,Connections} from './types';
import {assistantRequest,type AssistantSettings} from './assistantSettings';
import type {NasaProduct} from './types';
import {requestJson} from './apiRequest';
export {ApiHttpError} from './apiRequest';
const base=(import.meta.env.VITE_API_URL as string|undefined)?.replace(/\/$/,'')??'';
function conversationWindow(history:{role:'user'|'assistant';content:string}[]){
  const chosen:typeof history=[];let remaining=16000;
  for(let i=history.length-1;i>=0&&chosen.length<12;i--){
    const content=history[i].content.slice(0,4000);
    if(content.length>remaining)break;
    chosen.unshift({role:history[i].role,content});remaining-=content.length;
  }
  return chosen;
}
async function request<T>(path:string,body?:unknown,signal?:AbortSignal,timeoutMs=180000,extraHeaders?:Record<string,string>,admissionWaitMs=0,onAdmissionWait?:(waiting:boolean)=>void):Promise<T>{
  return requestJson<T>(base+path,{body,signal,timeoutMs,headers:extraHeaders,admissionWaitMs,onAdmissionWait});
}
export const api={
  health:()=>request<Health>('/api/v1/health',undefined,undefined,6000),
  connections:(signal?:AbortSignal)=>request<Connections>('/api/v1/connections',undefined,signal,6000),
  scan:(location:string|{latitude:number;longitude:number},signal?:AbortSignal,onWait?:(waiting:boolean)=>void)=>request<Scan>('/api/v1/scan',typeof location==='string'?{query:location}:location,signal,180000,undefined,60000,onWait),
  forecast:(scanId:string,key:HorizonKey,signal?:AbortSignal,onWait?:(waiting:boolean)=>void)=>request<Scan>(`/api/v1/scans/${encodeURIComponent(scanId)}/forecast/${key}`,undefined,signal,180000,undefined,60000,onWait),
  refreshRadar:(scan:Scan,signal?:AbortSignal,onWait?:(waiting:boolean)=>void)=>request<Scan>(`/api/v1/scans/${encodeURIComponent(scan.scan_id)}/radar/refresh`,{scenario_id:scan.scenario_id},signal,180000,undefined,60000,onWait),
  inspect:(scan:Scan,id:string,signal?:AbortSignal)=>request<Cell&{physics_informed_assessment?:Cell}>(`/api/v1/inspect/${scan.scan_id}/${id}?scenario_id=${scan.scenario_id}`,undefined,signal,15000),
  runout:(scan:Scan,id:string,signal?:AbortSignal)=>request<Runout>(`/api/v1/scans/${scan.scan_id}/scenarios/${scan.scenario_id}/runout/${id}`,undefined,signal,15000),
  chat:(scan:Scan|null,id:string,message:string,history:{role:'user'|'assistant';content:string}[],signal?:AbortSignal,active_prediction_model:PredictionModel='terra_physics_ml',assistant?:AssistantSettings,nasa_product:NasaProduct='today',onWait?:(waiting:boolean)=>void)=>{const connection=assistantRequest(assistant);return request<ChatReply>('/api/v1/chat',{message,scan_id:scan?.scan_id??null,scenario_id:scan?.scenario_id??null,selected_h3_index:id||null,history:conversationWindow(history),autonomous:true,active_prediction_model,nasa_product,...connection.config},signal,90000,connection.headers,30000,onWait);},
};
