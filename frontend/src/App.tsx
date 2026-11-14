import {useEffect,useRef,useState} from 'react';
import {Search,ArrowLeft,ArrowUpRight,Crosshair,Download,MessageSquare,LoaderCircle,ChevronDown,Check,AlertCircle,Layers3,Navigation,MapPin,X,Clock3,ExternalLink,Settings2} from 'lucide-react';
import MapCanvas,{type MapHandle} from './components/MapCanvas';
import AssistantSettingsPanel from './components/AssistantSettings';
import RadarPanel from './components/RadarPanel';
import AssistantPanel,{type AssistantMessage as Message} from './components/AssistantPanel';
import {radarCoverage} from './radarPresentation';
import AnalysisDetails from './components/AnalysisDetails';
import {assistantLabel,assistantReady,DEFAULT_ASSISTANT,readAssistantSettings,writeAssistantSettings,type AssistantSettings} from './assistantSettings';
import {api,ApiHttpError} from './api';
import {executeActions} from './actionExecutor';
import {chatProductMatches,criticalCell,incompleteAnalysis,LAYER_LABELS,modelLabel,nasaDescription,nasaProductAvailable,nasaProductView,nasaResolution,predictionScore,MODEL_LABELS} from './modelPresentation';
import {cellIn,coord,fmt,HORIZONS,PRIMARY_PREDICTION_MODELS,DIAGNOSTIC_MODELS,SOURCES,SOURCE_LABELS,type Cell,type ForecastHour,type Health,type HorizonKey,type MapLayer,type NasaProduct,type PredictionModel,type Provider,type Runout,type Scan,type WorkspaceHorizon} from './types';
const time=(value?:string|null)=>{if(!value)return 'Awaiting source time';const d=new Date(value);return Number.isNaN(d.valueOf())?value:d.toLocaleString('en-GB',{timeZone:'UTC',day:'2-digit',month:'short',hour:'2-digit',minute:'2-digit'})+' UTC';};
const labels=LAYER_LABELS;
const providerLabel=(p?:Provider)=>({WAITING:'Not queried',LIVE:'Available',METADATA:'Granule metadata',REFERENCE:'Reference estimate',LOCAL:'Local subset',CACHED:'Cached',FALLBACK:'Fallback',UNAVAILABLE:'Unavailable',PROVIDED:'Provided'}[stateOf(p)]??stateOf(p).toLowerCase().replaceAll('_',' '));
function stateOf(p?:Provider){if(!p)return 'WAITING';const mode=(p.mode??'').toUpperCase();if(mode.includes('FALLBACK'))return 'FALLBACK';if(mode.includes('EMULATION')||mode.includes('REFERENCE'))return 'REFERENCE';if(mode.includes('CACHE'))return 'CACHED';if(mode.includes('METADATA'))return 'METADATA';if(!p.available)return 'UNAVAILABLE';if(mode.includes('LIVE'))return 'LIVE';if(mode.includes('LOCAL'))return 'LOCAL';return mode==='PROVIDED'?'PROVIDED':mode||'AVAILABLE';}
function tone(p?:Provider){const s=stateOf(p);return s==='LIVE'?'green':s==='CACHED'?'cyan':s==='FALLBACK'?'amber':s==='UNAVAILABLE'?'red':'muted';}
function errorText(e:unknown){return e instanceof Error?e.name==='AbortError'?'The request timed out or was superseded.':e.message:String(e);}
function primary(scan:Scan){return scan.physics_informed_assessment?.hotspot?.h3_index??scan.physical_assessment.hotspot?.h3_index??scan.physical_assessment.cells[0]?.h3_index??'';}
function numeric(object:Record<string,unknown>|undefined,name:string){const v=object?.[name];return typeof v==='number'&&Number.isFinite(v)?v:undefined;}

export default function App(){
  const [scan,setScan]=useState<Scan|null>(null),[baseline,setBaseline]=useState<Scan|null>(null),[selected,setSelected]=useState(''),[inspected,setInspected]=useState<Cell>(),[route,setRoute]=useState<Runout>();
  const [query,setQuery]=useState(''),[busy,setBusy]=useState(''),[error,setError]=useState(''),[engine,setEngine]=useState<'connecting'|'active'|'offline'>('connecting'),[health,setHealth]=useState<Health>();
  const [horizon,setHorizon]=useState<HorizonKey>('now'),[layer,setLayer]=useState<MapLayer>('probability'),[predictionModel,setPredictionModel]=useState<PredictionModel>('terra_physics_ml');
  const [provider,setProvider]=useState<string|null>(null),[sourcesOpen,setSourcesOpen]=useState(false),[overlay,setOverlay]=useState<'options'|'assistant'|null>(null),[chat,setChat]=useState<Message[]>([]),[draft,setDraft]=useState(''),[chatBusy,setChatBusy]=useState(false),[comparison,setComparison]=useState(''),[view,setView]=useState<'overview'|'inspect'>('overview'),[dockOpen,setDockOpen]=useState(false),[aiSettingsOpen,setAiSettingsOpen]=useState(false),[radarExpanded,setRadarExpanded]=useState(false);
  const [assistant,setAssistant]=useState<AssistantSettings>(()=>{try{return readAssistantSettings(window.sessionStorage);}catch{return {...DEFAULT_ASSISTANT};}}),[settingsStored,setSettingsStored]=useState(true),[nasaProduct,setNasaProduct]=useState<NasaProduct>('today');
  const [assistantVerified,setAssistantVerified]=useState(false);
  const optionsOpen=overlay==='options',copilotOpen=overlay==='assistant',dockVisible=dockOpen||overlay!==null;
  const map=useRef<MapHandle|null>(null),epoch=useRef(0),request=useRef<AbortController|null>(null),chatEnd=useRef<HTMLDivElement>(null),searchInput=useRef<HTMLInputElement>(null),latest=useRef<Scan|null>(scan),baselineRef=useRef<Scan|null>(baseline);
  const selection=useRef(selected),modelRef=useRef(predictionModel),chatRequest=useRef<AbortController|null>(null),chatSequence=useRef(0),chatInput=useRef<HTMLTextAreaElement|null>(null);
  const optionsPanel=useRef<HTMLElement|null>(null),optionsTrigger=useRef<HTMLButtonElement|null>(null),overlayReturn=useRef<HTMLElement|null>(null),optionsOrigin=useRef<{overlay:'assistant'|null;view:'overview'|'inspect';open:boolean}|null>(null),nasaProductRef=useRef(nasaProduct);
  latest.current=scan;selection.current=selected;baselineRef.current=baseline;modelRef.current=predictionModel;
  nasaProductRef.current=nasaProduct;
  const nasaActive=predictionModel==='nasa_lhasa_2';
  const displayScan=scan&&nasaActive?nasaProductView(scan,nasaProduct):scan;
  const incomplete=incompleteAnalysis(scan);
  const cell=nasaActive&&displayScan?cellIn(displayScan,selected):inspected?.h3_index===selected?inspected:scan?cellIn(scan,selected):undefined;
  const secondary=displayScan&&comparison?cellIn(displayScan,comparison):undefined;
  const comparisonNasa=baseline?nasaProductView(baseline,nasaProduct):null,comparisonNasaCell=comparisonNasa?cellIn(comparisonNasa,selected):undefined;
  const forecast=baseline?.predictive_forecast_72h,views=forecast?.views??[],currentView=views.find(v=>v.key===horizon);
  const minColumn=scan?.physical_assessment.cells.filter(c=>c.fs_mid!=null).reduce<Cell|undefined>((a,b)=>!a||b.fs_mid!<a.fs_mid!?b:a,undefined);
  const hotspot=displayScan?criticalCell(displayScan,predictionModel):undefined,peakScore=predictionScore(hotspot,predictionModel),activeRunout=route?.h3_index===selected?route:scan?.runout_assessment?.h3_index===selected?scan.runout_assessment:undefined;
  const inputs=baseline?.live_inputs,soil=inputs?.soil_properties,radarStatus=scan?radarCoverage(scan,selected):null;
  const locationTitle=scan?(/^-?\d+(?:\.\d+)?\s*,\s*-?\d/.test(scan.location.display_name)?'Selected area':scan.location.display_name||'Selected area'):'';
  const allocations=Object.entries(cell?.uncertainty_variance_allocation??{}).sort((a,b)=>Math.abs(b[1])-Math.abs(a[1]));
  const maxAllocation=Math.max(1e-12,...allocations.map(([,v])=>Math.abs(v))),feed=provider?scan?.source_status[provider]:undefined;
  useEffect(()=>{let cancelled=false;const check=()=>api.health().then(value=>{if(!cancelled){setHealth(value);setEngine('active');}}).catch(()=>{if(!cancelled)setEngine('offline');});void check();const timer=setInterval(check,30000);return()=>{cancelled=true;clearInterval(timer);request.current?.abort();chatSequence.current++;chatRequest.current?.abort();};},[]);
  useEffect(()=>{const key=(event:KeyboardEvent)=>{const target=event.target as HTMLElement;if(event.key==='/'&&!overlay&&!['INPUT','TEXTAREA','SELECT'].includes(target.tagName)){event.preventDefault();searchInput.current?.focus();}if(event.key==='Escape'){event.preventDefault();setProvider(null);setSourcesOpen(false);closeOverlay();}};window.addEventListener('keydown',key);return()=>window.removeEventListener('keydown',key);},[overlay]);
  useEffect(()=>{try{setSettingsStored(writeAssistantSettings(window.sessionStorage,assistant));}catch{setSettingsStored(false);}},[assistant]);
  useEffect(()=>{
    setInspected(undefined);setRoute(undefined);if(!scan||!selected)return;
    const controller=new AbortController();
    const active=()=>!controller.signal.aborted&&latest.current?.scan_id===scan.scan_id&&latest.current?.scenario_id===scan.scenario_id&&selection.current===selected;
    api.inspect(scan,selected,controller.signal).then(c=>{if(active()&&c.h3_index===selected)setInspected({...cellIn(scan,selected),...c,...c.physics_informed_assessment});}).catch(e=>{if(active())setError('Cell inspection: '+errorText(e));});
    api.runout(scan,selected,controller.signal).then(r=>{if(active()&&r.h3_index===selected)setRoute(r);}).catch(e=>{if(active())setError('Runout inspection: '+errorText(e));});
    return()=>controller.abort();
  },[scan?.scan_id,scan?.scenario_id,selected]);
  function install(next:Scan,selectCritical=false){
    const display=modelRef.current==='nasa_lhasa_2'?nasaProductView(next,nasaProductRef.current):next;
    const id=!selectCritical&&cellIn(next,selection.current)?selection.current:criticalCell(display,modelRef.current)?.h3_index??primary(next);
    const sameSelection=latest.current?.scan_id===next.scan_id&&latest.current?.scenario_id===next.scenario_id&&selection.current===id;
    latest.current=next;selection.current=id;setScan(next);setSelected(id);
    if(!sameSelection){setInspected(undefined);setRoute(undefined);}
    setComparison(previous=>!selectCritical&&cellIn(next,previous)?previous:'');
    if(selectCritical)document.querySelector('.cell-panel')?.scrollTo({top:0});
  }
  function cancelChat(reason='The action sequence stopped because the workspace changed.'){
    chatSequence.current++;chatRequest.current?.abort();chatRequest.current=null;setChatBusy(false);
    setChat(messages=>messages.map(message=>message.executing?{...message,executing:false,activeTask:undefined,cancelled:true,executionError:reason}:message));
  }
  async function acquire(location:string|{latitude:number;longitude:number},chain?:AbortController){
    if(chain?.signal.aborted)throw new DOMException('The action sequence was superseded.','AbortError');
    const id=++epoch.current;if(request.current!==chain)request.current?.abort();if(!chain)cancelChat();
    const controller=chain??new AbortController();request.current=controller;
    setBusy(typeof location==='string'?`Resolving ${location} and acquiring live inputs`:`Scanning ${coord(location.latitude,location.longitude)}`);setError('');setProvider(null);
    if(typeof location!=='string')map.current?.flyToCoordinates(location.latitude,location.longitude);
    try{
      const next=await api.scan(location,controller.signal);
      if(epoch.current!==id||controller.signal.aborted)throw new DOMException('The scan was superseded.','AbortError');
      nasaProductRef.current='today';setNasaProduct('today');install(next,true);baselineRef.current=next;setBaseline(next);setHorizon('now');if(!chain){setChat([]);showWorkspace('overview');}setEngine('active');return next;
    }finally{if(epoch.current===id)setBusy('');}
  }
  async function startScan(location:string|{latitude:number;longitude:number}){
    try{await acquire(location);}catch(e){if(e instanceof DOMException&&e.name==='AbortError')return;setError(errorText(e));if(e instanceof ApiHttpError)setEngine(e.backendResponded?'active':'offline');else if(e instanceof TypeError)setEngine('offline');}
  }
  function search(event:React.FormEvent){event.preventDefault();const value=query.trim();if(!value)return;
    const match=value.match(/^(-?\d+(?:\.\d+)?)\s*[,;]\s*(-?\d+(?:\.\d+)?)$/);
    if(match){const latitude=Number(match[1]),longitude=Number(match[2]);if(Math.abs(latitude)>90||Math.abs(longitude)>180){setError('Use latitude −90…90 and longitude −180…180.');return;}void startScan({latitude,longitude});}
    else void startScan(value);
  }
  async function selectHorizon(key:HorizonKey,chain?:AbortController){
    const original=baselineRef.current;if(!original)throw new Error('Scan a location before selecting a forecast.');
    if(modelRef.current==='nasa_lhasa_2'&&key!=='now')throw new Error('NASA LHASA uses Today and Tomorrow products, not Terra forecast hours.');
    if(!chain)cancelChat();const id=++epoch.current;if(request.current!==chain)request.current?.abort();
    if(key==='now'){install(original);setHorizon('now');setError('');setBusy('');return;}
    if(!original.predictive_forecast_72h?.views?.some(v=>v.key===key))throw new Error('This forecast horizon is unavailable for the current scan.');
    const controller=chain??new AbortController();request.current=controller;setBusy('Loading '+HORIZONS.find(h=>h.key===key)?.label+' forecast');setError('');
    try{const next=await api.forecast(original.scan_id,key,controller.signal);if(epoch.current!==id||controller.signal.aborted)throw new DOMException('The forecast was superseded.','AbortError');install(next);setHorizon(key);}
    finally{if(epoch.current===id)setBusy('');}
  }
  function selectCell(id:string,fly=false,cancelPending=true,camera?:{pitch?:number;bearing?:number;zoom?:number}){
    const current=latest.current,c=current?cellIn(current,id):undefined;if(!c)return;
    if(cancelPending&&selection.current!==id)cancelChat();selection.current=id;setSelected(id);if(fly)map.current?.flyTo(c,camera);
  }
  function selectModel(model:PredictionModel,agent=false,preserveCell=false){
    if(!agent)cancelChat();modelRef.current=model;setPredictionModel(model);setLayer('probability');
    const original=baselineRef.current;
    if(model==='nasa_lhasa_2'&&original){if(!agent){request.current?.abort();epoch.current++;setBusy('');}install(original);setHorizon('now');}
    const current=model==='nasa_lhasa_2'&&original?nasaProductView(original,nasaProductRef.current):latest.current;
    const critical=current?criticalCell(current,model):undefined;
    if(critical&&!preserveCell){selection.current=critical.h3_index;setSelected(critical.h3_index);}
  }
  function selectNasaProduct(product:NasaProduct,agent=false){
    if(modelRef.current!=='nasa_lhasa_2')throw new Error('Choose NASA LHASA before selecting a NASA daily product.');
    if(!agent)cancelChat();
    const original=baselineRef.current;if(!original)throw new Error('Scan an area before selecting a NASA product.');
    if(!nasaProductAvailable(original,product))throw new Error(`NASA ${product==='today'?'Today':'Tomorrow'} is unavailable for this area.`);
    nasaProductRef.current=product;setNasaProduct(product);install(original);setHorizon('now');
    const critical=criticalCell(nasaProductView(original,product),'nasa_lhasa_2');
    if(critical){selection.current=critical.h3_index;setSelected(critical.h3_index);}
  }
  async function selectWorkspaceHorizon(value:WorkspaceHorizon,controller:AbortController){
    if(value==='nasa_today'||value==='nasa_tomorrow'){selectNasaProduct(value==='nasa_today'?'today':'tomorrow',true);await renderAction();return;}
    await selectHorizon(value,controller);
  }
  function selectLayer(next:MapLayer,agent=false){if(!agent)cancelChat();setLayer(next);}
  function showWorkspace(next:'overview'|'inspect'|'assistant'){
    optionsOrigin.current=null;setDockOpen(true);setOverlay(next==='assistant'?'assistant':null);
    if(next!=='assistant')setView(next);
    else requestAnimationFrame(()=>chatInput.current?.focus({preventScroll:true}));
  }
  function openOptions(ai=false){
    if(!optionsOpen)optionsOrigin.current={overlay:copilotOpen?'assistant':null,view,open:dockOpen};
    overlayReturn.current=document.activeElement instanceof HTMLElement?document.activeElement:null;
    setDockOpen(true);setOverlay('options');if(ai)setAiSettingsOpen(true);
    if(ai)requestAnimationFrame(()=>{const field=document.getElementById('ai-provider');field?.scrollIntoView({block:'nearest'});field?.focus({preventScroll:true});});
  }
  function closeWorkspace(){
    setDockOpen(false);setOverlay(null);optionsOrigin.current=null;
    requestAnimationFrame(()=>optionsTrigger.current?.focus({preventScroll:true}));
  }
  function closeOverlay(){
    if(!optionsOpen){closeWorkspace();return;}
    const origin=optionsOrigin.current;optionsOrigin.current=null;
    setOverlay(origin?.overlay??null);setView(origin?.view??view);setDockOpen(origin?.open??false);
    requestAnimationFrame(()=>{if(overlayReturn.current?.isConnected)overlayReturn.current.focus({preventScroll:true});else optionsTrigger.current?.focus({preventScroll:true});});
  }
  function openCopilot(){if(copilotOpen)closeWorkspace();else showWorkspace('assistant');}
  function stopAssistant(){
    const waiting=chatBusy&&!chat.some(message=>message.executing);
    cancelChat('Stopped remaining actions. An already dispatched request may still finish.');
    if(waiting)setChat(messages=>[...messages,{role:'assistant',text:'Stopped waiting for the response. An already dispatched request may still finish.',cancelled:true,scanId:latest.current?.scan_id??null,state:latest.current?.scenario_id??null,cell:selection.current,model:modelRef.current}]);
    if(request.current?.signal.aborted){epoch.current++;setBusy('');}
  }
  function changeAssistant(value:AssistantSettings){cancelChat();setAssistantVerified(false);setAssistant(value);}
  async function refreshRadar(){
    const original=baselineRef.current;if(!original||busy)return;
    cancelChat();request.current?.abort();const controller=new AbortController(),id=++epoch.current;request.current=controller;
    setBusy('Checking NISAR measurements');setError('');
    try{const next=await api.refreshRadar(original,controller.signal);if(controller.signal.aborted||epoch.current!==id)return;baselineRef.current=next;setBaseline(next);install(next);setHorizon('now');}
    catch(e){if(!controller.signal.aborted)setError(errorText(e));}finally{if(epoch.current===id)setBusy('');}
  }
  function renderAction(){return new Promise<void>(resolve=>{let done=false;const finish=()=>{if(!done){done=true;clearTimeout(timer);resolve();}};const timer=setTimeout(finish,150);requestAnimationFrame(()=>requestAnimationFrame(finish));});}
  async function send(event?:React.FormEvent){event?.preventDefault();if(!draft.trim()||chatBusy||busy)return;
    if(assistant.apiKey.trim()&&!assistantReady(assistant)){setError('Choose an AI model and a valid custom endpoint before using this key.');openOptions(true);return;}
    const current=latest.current,text=draft.trim(),identity=current?.scenario_id??null,cellId=selection.current,scanId=current?.scan_id??null,model=modelRef.current,product=nasaProductRef.current;
    const history=chat.filter(m=>(m.scanId??null)===scanId&&(m.state??null)===identity&&m.cell===cellId&&m.model===model&&(model!=='nasa_lhasa_2'||m.nasaProduct===product)).map(m=>({role:m.role,content:m.text}));
    const token=++chatSequence.current,controller=new AbortController();chatRequest.current?.abort();chatRequest.current=controller;
    const alive=()=>!controller.signal.aborted&&chatSequence.current===token;
    const original=()=>alive()&&(latest.current?.scan_id??null)===scanId&&(latest.current?.scenario_id??null)===identity&&selection.current===cellId&&modelRef.current===model&&(model!=='nasa_lhasa_2'||nasaProductRef.current===product);
    const replyId='assistant-'+token;
    setDraft('');setChat(c=>[...c,{role:'user',text,scanId,state:identity,cell:cellId,model,nasaProduct:product}]);setChatBusy(true);
    try{
      const reply=await api.chat(current,cellId,text,history,controller.signal,model,assistant,product);
      if(!original()||reply.scan_id!==scanId||reply.scenario_id!==identity||(reply.active_prediction_model&&reply.active_prediction_model!==model)||!chatProductMatches(model,product,reply.nasa_product))return;
      // This reply is bound to the same immutable scan. Reinstalling it clears
      // inspection/runout without triggering their identity-bound reload.
      setAssistantVerified(reply.briefing_guard==='VERIFIED_LLM_PROSE');
      setChat(c=>[...c,{id:replyId,role:'assistant',text:reply.clean_markdown,guard:reply.briefing_guard,claims:reply.bound_claims,scanId,state:reply.scenario_id,cell:cellId,model,nasaProduct:product,assistantProvider:reply.assistant_provider,assistantModel:reply.model_name,error:reply.model_error,steps:reply.thought_steps,plannedActions:reply.ui_action?.actions??[],executing:!!reply.ui_action?.actions.length,outcomes:[]}]);
      const outcomes=await executeActions(reply,{
        scan:()=>latest.current&&modelRef.current==='nasa_lhasa_2'?nasaProductView(latest.current,nasaProductRef.current):latest.current,selected:()=>selection.current,model:()=>modelRef.current,isCurrent:alive,
        scanCoordinates:async(latitude,longitude,label)=>{if(label)setQuery(label);await acquire({latitude,longitude},controller);},
        searchLocation:async value=>{setQuery(value);await acquire(value,controller);},
        selectModel:async value=>{selectModel(value,true);await renderAction();},setHorizon:value=>selectWorkspaceHorizon(value,controller),setLayer:async value=>{selectLayer(value,true);await renderAction();},
        flyTo:(c,camera)=>selectCell(c.h3_index,true,false,camera),
        inspect:c=>{selectCell(c.h3_index,false,false);setView('inspect');},
        compare:(a,b)=>{selectCell(a.h3_index,false,false);setComparison(b.h3_index);setView('inspect');},
        start:step=>{if(alive())setChat(messages=>messages.map(m=>m.id===replyId?{...m,activeTask:step}:m));},
        outcome:outcome=>{if(alive())setChat(messages=>messages.map(m=>m.id===replyId?{...m,activeTask:undefined,outcomes:[...m.outcomes??[],outcome]}:m));},
      });
      if(alive())setChat(messages=>messages.map(m=>m.id===replyId?{...m,executing:false,activeTask:undefined,executionError:outcomes.find(o=>o.status==='failed')?.summary}:m));
    }catch(e){if(alive())setChat(c=>c.some(m=>m.id===replyId)?c.map(m=>m.id===replyId?{...m,executing:false,activeTask:undefined,executionError:errorText(e)}:m):[...c,{role:'assistant',text:'The assistant request could not complete: '+errorText(e),guard:'SERVICE_ERROR',scanId,state:identity,cell:cellId,model}]);}
    finally{if(chatRequest.current===controller){chatRequest.current=null;setChatBusy(false);}}
  }
  function exportScan(){if(!scan)return;const url=URL.createObjectURL(new Blob([JSON.stringify({...scan,display_context:{prediction_model:predictionModel,nasa_product:nasaActive?nasaProduct:null,map_layer:layer,selected_h3_index:selected,comparison_h3_index:comparison||null}},null,2)],{type:'application/json'}));const link=document.createElement('a');link.href=url;link.download=`terra-${scan.scan_id}-${scan.scenario_id}.json`;link.click();setTimeout(()=>URL.revokeObjectURL(url),1000);}
  return <div className={`command-center ${scan?'has-scan':''} ${dockVisible?'workspace-open':''} ${overlay?'overlay-'+overlay:''}`}>
    <MapCanvas scan={displayScan} selected={selected} compared={comparison} predictionModel={predictionModel} onSelect={id=>{selectCell(id);showWorkspace('inspect');}} onScanCoordinate={(latitude,longitude)=>void startScan({latitude,longitude})} handle={map} runout={route} layer={layer}/>
    <header className="command-bar glass" aria-label="Map workspace tools">
      <form className="global-search" onSubmit={search}>
        <Search size={18}/><input ref={searchInput} aria-label="Search any place or latitude, longitude" value={query} onChange={e=>setQuery(e.target.value)} placeholder="Search a place or coordinates"/>
        <kbd>/</kbd><button type="submit" title="Search and scan" aria-label="Search and scan"><ArrowUpRight size={18}/></button>
      </form>
      <button className="scan-center" title="Scan current map view" aria-label="Scan current map view" onClick={()=>map.current?.scanCenter()}><Crosshair size={16}/><span>Scan map view</span></button>
      <div className="toolbar-model">
        <label htmlFor="prediction-model">Prediction model</label>
        <select id="prediction-model" aria-label="Prediction model" value={PRIMARY_PREDICTION_MODELS.includes(predictionModel)?predictionModel:''} onChange={e=>selectModel(e.target.value as PredictionModel)}>
          {!PRIMARY_PREDICTION_MODELS.includes(predictionModel)&&<option value="" disabled>Diagnostic: {MODEL_LABELS[predictionModel]}</option>}
          {PRIMARY_PREDICTION_MODELS.map(model=><option key={model} value={model}>{MODEL_LABELS[model]}</option>)}
        </select>
      </div>
      <div className="toolbar-actions">
        <button className={`toolbar-button ${copilotOpen?'active':''}`} type="button" onClick={openCopilot} aria-label="Open assistant" title="Assistant" aria-expanded={copilotOpen} aria-controls="view-assistant"><MessageSquare size={16}/><span>Assistant</span></button>
        <button ref={optionsTrigger} className={`toolbar-button options-trigger ${optionsOpen?'active':''}`} type="button" onClick={()=>optionsOpen?closeOverlay():openOptions()} aria-label="Options" title="Options" aria-expanded={optionsOpen} aria-controls="options-panel"><Settings2 size={16}/><span>Options</span><ChevronDown size={13}/></button>
      </div>
      {engine==='offline'&&<span className="connection-error" role="status" aria-label="Service unavailable" title="Service unavailable"><AlertCircle size={14}/><span>Service unavailable</span></span>}
    </header>


    {!scan&&!dockVisible&&<section className="welcome-card glass"><h1>Explore an area</h1><p>Search a place or select any point on the map to scan live conditions and the automatic forecast.</p><button className="welcome-assistant" onClick={openCopilot}><MessageSquare size={14}/>Navigate with the assistant</button></section>}
    {scan&&<div className="sector-context"><h1 title={scan.location.display_name}>{locationTitle}</h1><p>{nasaActive?`NASA ${nasaProduct==='today'?'Today':'Tomorrow'} · ${scan.nasa_lhasa?.product_valid_time_utc?time(scan.nasa_lhasa.product_valid_time_utc):'Source valid time not declared'}`:time(currentView?.valid_time_utc??scan.generated_at_utc)}</p></div>}
    {dockVisible&&<aside className={`workspace-dock glass ${optionsOpen?'is-options':''}`} aria-label="Analysis workspace">
      {optionsOpen?<div className="workspace-options-title"><button onClick={closeOverlay} aria-label="Back from options"><ArrowLeft size={17}/></button><h2>Options</h2><button onClick={closeWorkspace} aria-label="Close workspace"><X size={18}/></button></div>:<div className="workspace-tabs" role="tablist" aria-label="Analysis views">{(['overview','inspect','assistant'] as const).map(tab=><button key={tab} role="tab" id={'tab-'+tab} aria-controls={'view-'+tab} aria-selected={copilotOpen?tab==='assistant':view===tab} disabled={tab!=='assistant'&&!scan} onClick={()=>showWorkspace(tab)}>{tab==='overview'?'Overview':tab==='inspect'?'Inspect':'Assistant'}{tab==='assistant'&&chatBusy&&<LoaderCircle size={12} className="spin"/>}</button>)}<button className="workspace-close" onClick={closeWorkspace} aria-label="Close workspace"><X size={17}/></button></div>}
      <div className="workspace-view">
        {busy&&!copilotOpen&&<div className="workspace-notice acquisition" role="status"><LoaderCircle size={17} className="spin"/><div><strong>{busy}</strong><span>Gathering data and preparing the forecast.</span></div></div>}
    {error&&<div className="workspace-notice error-banner" role="alert"><AlertCircle size={17}/><div><strong>Request could not complete</strong><span>{error}</span></div><button onClick={()=>setError('')} aria-label="Dismiss error"><X size={16}/></button></div>}
    {incomplete&&!busy&&!error&&!copilotOpen&&!optionsOpen&&<div className="workspace-notice analysis-notice" role="status"><AlertCircle size={16}/><div><strong>{incomplete.title}</strong><p>{incomplete.message}</p><button onClick={()=>{openOptions();setSourcesOpen(true);}}>View sources<ArrowUpRight size={13}/></button></div></div>}
        {optionsOpen?<section ref={optionsPanel} id="options-panel" className="options-popover workspace-pane" aria-label="Workspace options">
      <div className="options-section options-layer"><label htmlFor="map-layer">Map layer</label><select id="map-layer" disabled={!scan} value={layer} onChange={e=>selectLayer(e.target.value as MapLayer)}>{(Object.keys(labels) as MapLayer[]).map(key=><option key={key} value={key}>{labels[key]}</option>)}</select>{!scan&&<p className="support-note">Scan a location to inspect result layers.</p>}</div>
      <details className="options-section disclosure" open={sourcesOpen} onToggle={event=>setSourcesOpen(event.currentTarget.open)}><summary>Sources<ChevronDown size={15}/></summary>
        {scan?<><p className="support-note">Select a provider for current availability and provenance.</p>
          <div className="source-list">{SOURCES.map(name=>{const p=scan.source_status[name];return <button key={name} className={provider===name?'selected':''} onClick={()=>setProvider(v=>v===name?null:name)} aria-expanded={provider===name}><span>{SOURCE_LABELS[name]}</span><small className={tone(p)}>{providerLabel(p)}</small></button>;})}</div>
          {provider&&<div className="source-detail"><h3>{feed?.provider??SOURCE_LABELS[provider]}</h3><p>{feed?.error??'Source provenance refers to the current scan. Availability does not establish local measurement accuracy.'}</p><dl className="telemetry"><div><dt>Source mode</dt><dd>{stateOf(feed)}</dd></div><div><dt>Retrieved</dt><dd>{time(feed?.fetched_at_utc)}</dd></div><div><dt>Fetch time</dt><dd>{feed?.latency_ms==null?'Not recorded':fmt(feed.latency_ms,0)+' ms'}</dd></div><div><dt>Age</dt><dd>{feed?.age_seconds==null?'Not reported':fmt(feed.age_seconds,0)+' s'}</dd></div></dl></div>}
        </>:<p className="support-note">Provider results appear after a scan.</p>}
        <a className="text-link" href="https://github.com/pookie421/Terra/blob/main/docs/global_source_audit.md" target="_blank" rel="noreferrer">Source methods and limitations <ExternalLink size={13}/></a>
      </details>
      <details className="options-section disclosure options-ai-settings" open={aiSettingsOpen} onToggle={event=>setAiSettingsOpen(event.currentTarget.open)}><summary>AI model & API key<ChevronDown size={15}/></summary><AssistantSettingsPanel embedded value={assistant} stored={settingsStored} connected={assistantVerified} onChange={changeAssistant}/></details>
      <details className="options-section disclosure" open={DIAGNOSTIC_MODELS.includes(predictionModel)||undefined}><summary>Advanced diagnostics<ChevronDown size={15}/></summary>
        <p className="support-note">Inspect individual model components. Selecting a diagnostic updates the map and assistant context to that exact model.</p>
        <div className="diagnostic-models">{DIAGNOSTIC_MODELS.map(model=><button key={model} className={predictionModel===model?'active':''} aria-pressed={predictionModel===model} onClick={()=>selectModel(model)}>{MODEL_LABELS[model]}{predictionModel===model&&<Check size={13}/>}</button>)}</div>
        <p className="support-note">These are diagnostic components, not additional primary models. Choose a primary model from the toolbar to return.</p>
      </details>
      <div className="options-section"><button className="options-export" disabled={!scan} onClick={exportScan}><Download size={15}/>Export analysis<span>JSON</span></button></div>

        </section>:copilotOpen?<section id="view-assistant" role="tabpanel" aria-labelledby="tab-assistant" className="assistant-drawer"><AssistantPanel scan={displayScan} selectedCell={cell} predictionModel={predictionModel} nasaProduct={nasaProduct} messages={chat} draft={draft} busy={chatBusy} workspaceBusy={!!busy} connectionLabel={assistantLabel(assistant)} connected={assistantVerified} hasKey={assistantReady(assistant)} onDraftChange={setDraft} onSend={()=>void send()} onStop={stopAssistant} onOpenOptions={()=>openOptions(true)} inputRef={chatInput} endRef={chatEnd}/></section>:scan&&view==='overview'?<div id="view-overview" role="tabpanel" aria-labelledby="tab-overview" className="workspace-pane-slot">
      <section className="sector-panel workspace-pane" aria-label="Area overview">
        <div className="panel-title"><h2>{nasaActive?'NASA daily overview':'Area overview'}</h2></div>
        <section className="panel-section">
          <div className="sector-score"><span>{nasaActive?'Peak official NASA score':'Peak model score'}</span><strong className={peakScore!=null&&peakScore>=.65?'red':''}>{fmt(peakScore,2)}</strong></div>
          <p className="active-model-label">{DIAGNOSTIC_MODELS.includes(predictionModel)?'Diagnostic · ':''}{modelLabel(displayScan,predictionModel)}{nasaActive?` · ${nasaProduct==='today'?'Today':'Tomorrow'}`:''}</p>
          {!nasaActive&&<div className="summary-interval"><span>Lowest factor of safety</span><b>{fmt(minColumn?.fs_mid)}</b><small>Central estimate</small></div>}
          {nasaActive&&<p className="model-support">{nasaResolution(scan)}</p>}
          <button className="hud-button" disabled={!hotspot} onClick={()=>hotspot&&selectCell(hotspot.h3_index,true)}><Navigation size={15}/>{nasaActive?'View highest NASA pixel':'View critical area'}<ArrowUpRight size={15}/></button>
          <p className="support-note">{nasaActive?nasaDescription(displayScan):peakScore===null?'No supported score for this model and time.':'Experimental scores; physical columns use reference soil assumptions.'}</p>
        </section>
        {!nasaActive&&radarStatus&&<section className="panel-section radar-overview"><div className="section-title"><h3>NISAR radar</h3><span className={radarStatus.measuredCells?'green':'muted'}>{radarStatus.measuredCells?'Measured':radarStatus.status==='pending'?'Processing':radarStatus.status==='authentication'?'Access required':'Coverage'}</span></div><p className="support-note">{radarStatus.summary}</p><button className="hud-button" onClick={()=>{showWorkspace('inspect');setRadarExpanded(true);requestAnimationFrame(()=>document.querySelector('.radar-section')?.scrollIntoView({block:'start',behavior:'smooth'}));}}>View radar evidence<ArrowUpRight size={14}/></button></section>}
        {nasaActive?<section className="panel-section nasa-products">
          <div className="section-title"><h3>NASA daily product</h3></div>
          <div className="horizon-switch daily-products" role="group" aria-label="NASA daily product">{(['today','tomorrow'] as NasaProduct[]).map(product=><button key={product} className={nasaProduct===product?'active':''} aria-pressed={nasaProduct===product} disabled={!!busy||!nasaProductAvailable(baseline,product)} onClick={()=>selectNasaProduct(product)}>{product==='today'?'Today':'Tomorrow'}</button>)}</div>
          <p className="support-note">Daily source products. “Tomorrow” is not an exact 24-hour lead. Terra’s hourly forecast is a separate model.</p>
          <dl className="telemetry"><div><dt>Native support</dt><dd>{scan.nasa_lhasa?.native_support??nasaResolution(scan)}</dd></div><div><dt>Product version</dt><dd>{scan.nasa_lhasa?.product_version?.startsWith('NOT_')?'Not declared by service':scan.nasa_lhasa?.product_version??'Not declared by service'}</dd></div><div><dt>Retrieved</dt><dd>{time(scan.nasa_lhasa?.fetched_at_utc as string|undefined)}</dd></div></dl>
        </section>:<section className="panel-section forecast-section">
          <div className="section-title"><h3>{predictionModel==='rainfall_only'?'Terra physical forecast context':'72-hour forecast'}</h3></div>
          <div className="horizon-switch" role="group" aria-label="Forecast time">{HORIZONS.map(h=><button key={h.key} className={horizon===h.key?'active':''} aria-pressed={horizon===h.key} title={h.key==='peak'?'Lowest sector factor of safety in the forecast':h.sub} disabled={!!busy||(h.key!=='now'&&!views.some(v=>v.key===h.key))} onClick={()=>void selectHorizon(h.key).catch(e=>setError(errorText(e)))}>{h.label}</button>)}</div>
          <ForecastChart hourly={forecast?.paths?.p50?.hourly??[]}/>
          <p className="forecast-caption">{horizon==='peak'?'Lowest forecast stability':horizon==='now'?'Current conditions':'Forecast conditions'}{currentView?.hour!=null&&horizon!=='now'?` · +${currentView.hour}h`:''}</p>
          {forecast?.reason&&<p className="support-note amber">{forecast.reason}</p>}
        </section>}
        <details className="panel-section disclosure observed-section"><summary>{nasaActive?'Terra environmental context':'Current conditions'}<ChevronDown size={15}/></summary>
          <dl className="telemetry"><Metric label="Rain · last 24h" value={inputs?.rain_24h_mm} unit="mm" digits={1}/><Metric label="Rain · preceding 6 days" value={inputs?.rain_prior_6d_mm} unit="mm" digits={1}/><Metric label="Surface moisture · 24h mean" value={inputs?.surface_moisture_m3m3} unit="m³/m³" digits={3}/></dl>
          <details className="nested-details"><summary>Soil and other inputs</summary><dl className="telemetry"><Metric label="Clay · 0–30 cm" value={numeric(soil,'clay_g_kg')==null?undefined:numeric(soil,'clay_g_kg')!/10} unit="%"/><Metric label="Sand · 0–30 cm" value={numeric(soil,'sand_g_kg')==null?undefined:numeric(soil,'sand_g_kg')!/10} unit="%"/><Metric label="Bulk density" value={numeric(soil,'bdod_kg_m3')} unit="kg/m³"/><Metric label="Organic carbon" value={numeric(soil,'soc_g_kg')} unit="g/kg"/><Metric label="USGS PGA · catalog proxy" value={inputs?.pga_proxy_g} unit="g" digits={4}/><Metric label="River discharge · modeled" value={inputs?.river_discharge_m3_s} unit="m³/s"/></dl><p className="support-note">Soil source: {String(soil?.source??'Unavailable').replaceAll('_',' ')}<br/>Weather: {time(inputs?.weather_valid_time_utc)}</p></details>
        </details>
        <details className="panel-section disclosure model-section"><summary>Selected-cell comparison<ChevronDown size={15}/></summary>
          {PRIMARY_PREDICTION_MODELS.map(model=>{const score=predictionScore(model==='nasa_lhasa_2'?comparisonNasaCell:cell,model);return <div className={`model-row ${predictionModel===model?'primary':''}`} key={model}><div><button onClick={()=>selectModel(model,false,true)}>{modelLabel(model==='nasa_lhasa_2'?comparisonNasa:scan,model)}{model==='nasa_lhasa_2'?` · ${nasaProduct==='today'?'Today':'Tomorrow'}`:''}</button><b>{fmt(score,2)}</b></div><div className={`model-track ${score===null?'missing':''}`}><i style={{width:`${Math.max(0,Math.min(100,(score??0)*100))}%`}}/></div></div>;})}
          <details className="nested-details"><summary>Component diagnostics</summary>{DIAGNOSTIC_MODELS.map(model=>{const score=predictionScore(cell,model);return <div className={`model-row ${predictionModel===model?'primary':''}`} key={model}><div><button onClick={()=>selectModel(model)}>{modelLabel(scan,model)}</button><b>{fmt(score,2)}</b></div><div className={`model-track ${score===null?'missing':''}`}><i style={{width:`${Math.max(0,Math.min(100,(score??0)*100))}%`}}/></div></div>;})}</details>
          <p className="support-note">Same location; input support, time products and score meanings differ. NASA daily products remain separate from Terra’s forecast. {nasaDescription(comparisonNasa)}</p>
        </details>
        <AnalysisDetails scan={displayScan??scan} nasa={nasaActive} minimum={minColumn} health={health}/>
      </section>
        </div>:scan?<div id="view-inspect" role="tabpanel" aria-labelledby="tab-inspect" className="workspace-pane-slot">
      <section className="cell-panel workspace-pane" aria-label="Selected location">
        <div className="panel-title"><h2>Selected location</h2><button disabled={!cell} aria-label="Center selected location" title="Center selected location" onClick={()=>cell&&map.current?.flyTo(cell)}><Crosshair size={16}/></button></div>
        <section className="panel-section cell-identity"><p>{cell?coord(cell.latitude,cell.longitude):'Select an area on the map.'}</p><div className="cell-stats"><Datum name="Slope" value={cell?.slope_deg} unit="°"/><Datum name="Elevation" value={cell?.elevation_m} unit="m"/></div></section>
        {secondary&&<section className="panel-section pinned-comparison"><div className="section-title"><h3>Pinned cell comparison</h3><button onClick={()=>{cancelChat();setComparison('');}} aria-label="Remove pinned comparison"><X size={16}/></button></div><table><thead><tr><th>Metric</th><th>Selected</th><th>Pinned</th></tr></thead><tbody><tr><th>Slope</th><td>{fmt(cell?.slope_deg)}°</td><td>{fmt(secondary.slope_deg)}°</td></tr><tr><th>FS central</th><td>{fmt(cell?.fs_mid)}</td><td>{fmt(secondary.fs_mid)}</td></tr><tr><th>Active score</th><td>{fmt(predictionScore(cell,predictionModel))}</td><td>{fmt(predictionScore(secondary,predictionModel))}</td></tr></tbody></table><button className="hud-button" onClick={()=>{const previous=selected;selectCell(secondary.h3_index,true);setComparison(previous);}}>Inspect pinned cell<ArrowUpRight size={14}/></button><details className="nested-details"><summary>Exact cells</summary><p className="support-note">Selected: {selected}<br/>Pinned: {secondary.h3_index}<br/>{coord(secondary.latitude,secondary.longitude)}</p></details></section>}
        {nasaActive&&<section className="panel-section nasa-pixel">
          <div className="section-title"><h3>NASA {nasaProduct==='today'?'Today':'Tomorrow'} pixel</h3></div>
          <div className="nasa-pixel-score"><strong>{fmt(predictionScore(cell,'nasa_lhasa_2'),3)}</strong><span>{predictionScore(cell,'nasa_lhasa_2')===null?'No official value':'Official relative hazard score'}</span></div>
          <dl className="telemetry"><div><dt>Source pixel</dt><dd>{cell?.coarse_pixel_id??'Not reported'}</dd></div><Metric label="Susceptibility class" value={cell?.susceptibility_class??cell?.lhasa_susceptibility} digits={0}/><div><dt>Source spacing</dt><dd>{nasaResolution(scan)}</dd></div><div><dt>Valid time</dt><dd>{scan.nasa_lhasa?.product_valid_time_utc?time(scan.nasa_lhasa.product_valid_time_utc):'Not declared by service'}</dd></div></dl>
          <p className="support-note">NASA values apply to native raster pixels. The selected terrain coordinate provides location context within that pixel.</p>
          <button className="hud-button" onClick={()=>selectModel('terra_physics_ml',false,true)}>Inspect Terra physics at this cell<ArrowUpRight size={14}/></button>
          <p className="support-note">Factor of safety, radar context and potential runout come from Terra’s analysis, not NASA LHASA.</p>
        </section>}
        {!nasaActive&&<section className="panel-section physical-column">
          <div className="section-title"><h3>Factor of safety</h3></div>
          <div className="fs-envelope"><div><span>Low</span><strong>{fmt(cell?.fs_low)}</strong></div><div className="nominal"><span>Central</span><strong>{fmt(cell?.fs_mid)}</strong></div><div><span>High</span><strong>{fmt(cell?.fs_high)}</strong></div></div>
          <p className="fs-scale">A value below 1 indicates potential instability.</p>
          <p className="prior-label">{cell?.provenance_label?.includes('pedotransfer')?'Uses reference soil assumptions':cell?.provenance_label?.replaceAll('_',' ')??'Reference column'}</p>
          <dl className="telemetry"><Metric label="Pore-pressure head" value={cell?.pressure_head_m} unit="m"/><div><dt>Additional rain to instability</dt><dd>{cell?.rainfall_threshold_attainable_at_duration===false?'Not reached in this window':cell?.critical_rainfall_deficit_mm==null?'—':fmt(cell.critical_rainfall_deficit_mm,1)+' mm'}</dd></div></dl>
          <details className="nested-details technical-details"><summary>Physical details</summary><dl className="telemetry"><div><dt>H3 cell</dt><dd>{selected}</dd></div><Metric label="Specific TWI" value={cell?.twi}/><Metric label="Joint variance · JΣJᵀ" value={cell?.fosm_variance} digits={4}/><Metric label="Mechanical admissibility" value={cell?.mechanical_admissibility} digits={4}/><Metric label="Learned trigger score" value={cell?.learned_trigger_score} digits={4}/><div><dt>Newmark displacement</dt><dd title={cell?.newmark_status}>{cell?.newmark_displacement_cm==null?'Outside input/domain support':fmt(cell.newmark_displacement_cm,4)+' cm'}</dd></div><div><dt>Column provenance</dt><dd>{cell?.provenance_label??'Not reported'}</dd></div></dl>
            <details className="sensitivity-details"><summary>Variance sensitivities</summary><p className="support-note">Signed covariance allocations, not causal percentages.</p>{allocations.map(([key,value])=><div className="sensitivity" key={key}><span>{key.replaceAll('_',' ')}</span><div><i className={value<0?'negative':''} style={{width:`${Math.abs(value)/maxAllocation*100}%`}}/></div><b>{fmt(value,3)}</b></div>)}<details className="matrix-details"><summary>Jacobian and covariance</summary><pre>{JSON.stringify({J:cell?.fs_jacobian,Sigma:cell?.joint_covariance},null,2)}</pre></details></details>
          </details>
        </section>}
        {!nasaActive&&<RadarPanel expanded={radarExpanded} onExpandedChange={setRadarExpanded} scan={scan} cell={cell} busy={!!busy} onRefresh={()=>void refreshRadar()}/>}
        {!nasaActive&&<details className="panel-section disclosure runout-section"><summary>Potential runout<ChevronDown size={15}/></summary><div className="runout-numbers"><Datum name="Peak velocity" value={activeRunout?.peak_velocity_ms} unit="m/s"/><Datum name="Travel distance" value={activeRunout?.scenarios?.some(r=>r.travel_distance_m!=null)?Math.max(...activeRunout.scenarios.map(r=>r.travel_distance_m??0)):undefined} unit="m"/></div><p className="support-note">Conditional on release. Three rheology and depth assumptions; routes may stop at the scan boundary.</p><details className="nested-details"><summary>Routing assumptions</summary>{activeRunout?.scenarios?.map((r,index)=><div className="route-row" key={index}><i className={'route-'+index}/><span>μ {fmt(r.mu_b)} / ξ {fmt(r.xi_m_s2,0)} / depth {fmt(r.flow_depth_m)} m</span><b>{fmt(r.peak_velocity_m_s)} m/s</b><small>{r.stop_reason?.replaceAll('_',' ')}</small></div>)}<p className="support-note">{activeRunout?.corridor_h3_indices?.length??'—'} corridor cells</p></details></details>}
      </section>
        </div>:null}
      </div>
    </aside>}
    {scan&&<div className="mobile-panel-switch glass" role="group" aria-label="Workspace panel"><button className={!dockVisible?'active':''} aria-pressed={!dockVisible} onClick={closeWorkspace}><MapPin size={16}/>Map</button><button className={dockVisible&&!overlay&&view==='overview'?'active':''} aria-pressed={dockVisible&&!overlay&&view==='overview'} onClick={()=>showWorkspace('overview')}><Layers3 size={16}/>Overview</button><button className={dockVisible&&!overlay&&view==='inspect'?'active':''} aria-pressed={dockVisible&&!overlay&&view==='inspect'} onClick={()=>showWorkspace('inspect')}><Search size={16}/>Inspect</button><button className={copilotOpen?'active':''} aria-pressed={copilotOpen} onClick={()=>showWorkspace('assistant')}><MessageSquare size={16}/>Assistant</button></div>}
    {!dockVisible&&<div className="workspace-floating-notices">{busy&&!copilotOpen&&<div className="workspace-notice acquisition" role="status"><LoaderCircle size={17} className="spin"/><div><strong>{busy}</strong><span>Gathering data and preparing the forecast.</span></div></div>}
    {error&&<div className="workspace-notice error-banner" role="alert"><AlertCircle size={17}/><div><strong>Request could not complete</strong><span>{error}</span></div><button onClick={()=>setError('')} aria-label="Dismiss error"><X size={16}/></button></div>}
    {incomplete&&!busy&&!error&&!copilotOpen&&!optionsOpen&&<div className="workspace-notice analysis-notice" role="status"><AlertCircle size={16}/><div><strong>{incomplete.title}</strong><p>{incomplete.message}</p><button onClick={()=>{openOptions();setSourcesOpen(true);}}>View sources<ArrowUpRight size={13}/></button></div></div>}</div>}
  </div>;
}

function Metric({label,value,unit='',digits=2}:{label:string;value:unknown;unit?:string;digits?:number}){const display=typeof value==='number'&&value>0&&value<10**-digits?'<' + (10**-digits).toLocaleString('en-US',{maximumFractionDigits:digits}):fmt(value,digits);return <div><dt>{label}</dt><dd title={typeof value==='number'?String(value):undefined}>{display}{typeof value==='number'&&unit?' '+unit:''}</dd></div>;}
function Datum({name,value,unit=''}:{name:string;value:unknown;unit?:string}){return <div className="datum"><span>{name}</span><strong>{fmt(value)}{typeof value==='number'&&unit&&<small> {unit}</small>}</strong></div>;}
function ForecastChart({hourly}:{hourly:ForecastHour[]}){
  const values=hourly.map(h=>h.minimum_fs).filter((v):v is number=>typeof v==='number'&&Number.isFinite(v));
  if(!values.length)return <div className="forecast-empty"><Clock3 size={16}/><span>Forecast appears when weather data is available.</span></div>;
  const low=Math.min(1,...values)*.9,high=Math.max(1.1,...values)*1.05,span=Math.max(.1,high-low),y=(v:number)=>57-(v-low)/span*47;
  let path='',previous=false;hourly.forEach((h,i)=>{const v=h.minimum_fs;if(typeof v!=='number'||!Number.isFinite(v)){previous=false;return;}path+=`${previous?'L':'M'} ${((h.hour??i)/72*286+2).toFixed(2)} ${y(v).toFixed(2)} `;previous=true;});
  return <div className="forecast-chart"><span>Lowest forecast factor of safety</span><svg viewBox="0 0 290 72" role="img" aria-label="Automatic seventy-two hour minimum factor-of-safety trajectory"><path d={`M 2 ${y(1)} H 288`} stroke="#F59E0B" strokeOpacity=".45" strokeDasharray="3 4" fill="none"/><path d={path} fill="none" stroke="#90abc4" strokeWidth="2" vectorEffect="non-scaling-stroke"/></svg><div><span>Now</span><span>24h</span><span>48h</span><span>72h</span></div></div>;
}
