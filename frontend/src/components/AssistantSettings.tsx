import {KeyRound,X} from 'lucide-react';
import {AI_PROVIDERS,assistantReady,changeAiProvider,type AiProvider,type AssistantSettings as Settings} from '../assistantSettings';

interface Props {value:Settings;onChange:(value:Settings)=>void;onClose:()=>void;stored:boolean;connected:boolean}
export default function AssistantSettings({value,onChange,onClose,stored,connected}:Props){
  return <>
    <div className="panel-title"><h2 id="ai-settings-title">AI model & API key</h2><button onClick={onClose} aria-label="Close AI settings"><X size={18}/></button></div>
    <div className="ai-settings-content">
      <p className="support-note">Use your own provider account for generated answers. Map commands and verified telemetry work without a key.</p>
      <label htmlFor="ai-provider">Provider</label><select autoFocus id="ai-provider" value={value.provider} onChange={e=>onChange(changeAiProvider(value,e.target.value as AiProvider))}>{(Object.keys(AI_PROVIDERS) as AiProvider[]).map(id=><option key={id} value={id}>{AI_PROVIDERS[id].label}</option>)}</select>
      <label htmlFor="ai-model">Model ID</label><input id="ai-model" value={value.model} maxLength={200} onChange={e=>onChange({...value,model:e.target.value})} placeholder={value.provider==='custom_openai'?'Your endpoint’s model ID':AI_PROVIDERS[value.provider].model} autoComplete="off" spellCheck={false}/>
      {value.provider==='custom_openai'&&<><label htmlFor="ai-endpoint">OpenAI-compatible base URL</label><input id="ai-endpoint" type="url" value={value.baseUrl} onChange={e=>onChange({...value,baseUrl:e.target.value})} placeholder="https://your-provider.example/v1" autoComplete="off" spellCheck={false}/><p className="support-note">Use a public HTTPS endpoint. This website’s server cannot reach a model on your computer’s localhost.</p></>}
      <label htmlFor="ai-api-key">API key</label><div className="api-key-field"><input id="ai-api-key" type="password" value={value.apiKey} onChange={e=>onChange({...value,apiKey:e.target.value})} placeholder="Paste your personal API key" autoComplete="off" autoCapitalize="none" spellCheck={false}/><button disabled={!value.apiKey} onClick={()=>onChange({...value,apiKey:''})}>Clear key</button></div>
      <div className={`ai-status ${assistantReady(value)?'ready':''}`} role="status"><KeyRound size={14}/><span>{connected?`Connected · ${AI_PROVIDERS[value.provider].label} / ${value.model}`:assistantReady(value)?'Key ready · verified after a successful response':'No key · deterministic mode'}</span></div>
      <p className="support-note">Your key stays in this browser tab’s session storage and is sent with each assistant request to the selected provider through this website. It is never saved by the server. Closing the session or choosing Clear key removes it. Changing provider clears the previous key.</p>
      {!stored&&<p className="support-note amber">Session storage is unavailable. Settings remain in memory for this page only.</p>}
      <button className="hud-button ai-settings-done" onClick={onClose}>Done</button>
    </div>
  </>;
}
