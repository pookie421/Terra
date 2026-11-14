export const AI_PROVIDERS = {
  openai: {label:'OpenAI',model:'gpt-4o-mini'},
  gemini: {label:'Google Gemini',model:'gemini-2.5-flash'},
  anthropic: {label:'Anthropic Claude',model:'claude-sonnet-4-20250514'},
  openrouter: {label:'OpenRouter',model:'openrouter/free'},
  groq: {label:'Groq',model:'llama-3.3-70b-versatile'},
  custom_openai: {label:'Custom endpoint',model:''},
} as const;
export type AiProvider=keyof typeof AI_PROVIDERS;
export interface AssistantSettings {provider:AiProvider;model:string;apiKey:string;baseUrl:string}
export const ASSISTANT_STORAGE_KEY='terra.assistant.session.v1';
export const DEFAULT_ASSISTANT:AssistantSettings={provider:'openai',model:AI_PROVIDERS.openai.model,apiKey:'',baseUrl:''};
export function readAssistantSettings(storage:Pick<Storage,'getItem'>|undefined):AssistantSettings {
  try {
    const value=JSON.parse(storage?.getItem(ASSISTANT_STORAGE_KEY)??'null');
    if(!value||!Object.hasOwn(AI_PROVIDERS,value.provider))return {...DEFAULT_ASSISTANT};
    return {provider:value.provider,model:typeof value.model==='string'?value.model.slice(0,200):AI_PROVIDERS[value.provider as AiProvider].model,apiKey:typeof value.apiKey==='string'?value.apiKey:'',baseUrl:typeof value.baseUrl==='string'?value.baseUrl:''};
  } catch {return {...DEFAULT_ASSISTANT};}
}
export function writeAssistantSettings(storage:Pick<Storage,'setItem'|'removeItem'>|undefined,value:AssistantSettings):boolean {
  try {storage?.setItem(ASSISTANT_STORAGE_KEY,JSON.stringify(value));return !!storage;}catch{return false;}
}
export function changeAiProvider(value:AssistantSettings,provider:AiProvider):AssistantSettings {
  // Never forward one provider's credential to a newly selected provider.
  return value.provider===provider?value:{provider,model:AI_PROVIDERS[provider].model,apiKey:'',baseUrl:''};
}
export function assistantReady(value:AssistantSettings):boolean {
  if(!value.apiKey.trim()||/[\r\n]/.test(value.apiKey)||!value.model.trim())return false;
  if(value.provider!=='custom_openai')return true;
  try{const url=new URL(value.baseUrl);return url.protocol==='https:'&&!url.username&&!url.password&&!url.search&&!url.hash;}catch{return false;}
}
export function assistantLabel(value:AssistantSettings):string {
  return assistantReady(value)?`${AI_PROVIDERS[value.provider].label} · ${value.model.trim()}`:'No key · map commands available';
}
export function assistantRequest(value?:AssistantSettings):{config:{provider?:AiProvider;model?:string;base_url?:string};headers:Record<string,string>} {
  const key=value?.apiKey.trim();
  if(!value||!key)return {config:{},headers:{}};
  return {config:{provider:value.provider,model:value.model.trim(),...(value.provider==='custom_openai'?{base_url:value.baseUrl.trim()}: {})},headers:{'X-LLM-API-Key':key}};
}
