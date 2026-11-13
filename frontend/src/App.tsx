import {useEffect,useRef,useState} from 'react';
import {Search,ArrowUpRight,Crosshair,Download,MessageSquare,LoaderCircle,ChevronDown,Send,Check,AlertCircle,Layers3,Navigation,MapPin,X,Clock3,ExternalLink,Database} from 'lucide-react';
import MapCanvas,{type MapHandle} from './components/MapCanvas';
import {api,ApiHttpError} from './api';
import {executeActions} from './actionExecutor';
import {criticalCell,LAYER_LABELS,modelLabel,nasaDescription,predictionScore,MODEL_LABELS} from './modelPresentation';
import {cellIn,coord,fmt,HORIZONS,PREDICTION_MODELS,SOURCES,SOURCE_LABELS,type ActionOutcome,type BoundClaim,type Cell,type ForecastHour,type Health,type HorizonKey,type MapLayer,type PredictionModel,type Provider,type Runout,type Scan,type ThoughtStep} from './types';
interface Message{id?:string;role:'assistant'|'user';text:string;guard?:string;claims?:BoundClaim[];state?:string|null;cell?:string;model?:PredictionModel;error?:string;steps?:ThoughtStep[];outcomes?:ActionOutcome[];executing?:boolean;executionError?:string}
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
  const [provider,setProvider]=useState<string|null>(null),[sourcesOpen,setSourcesOpen]=useState(false),[chat,setChat]=useState<Message[]>([]),[draft,setDraft]=useState(''),[chatBusy,setChatBusy]=useState(false),[copilotOpen,setCopilotOpen]=useState(false),[comparison,setComparison]=useState(''),[mobilePanel,setMobilePanel]=useState<'sector'|'cell'|null>(null);
  const map=useRef<MapHandle|null>(null),epoch=useRef(0),request=useRef<AbortController|null>(null),chatEnd=useRef<HTMLDivElement>(null),searchInput=useRef<HTMLInputElement>(null),latest=useRef<Scan|null>(scan),baselineRef=useRef<Scan|null>(baseline);
  const selection=useRef(selected),modelRef=useRef(predictionModel),chatRequest=useRef<AbortController|null>(null),chatSequence=useRef(0),chatInput=useRef<HTMLTextAreaElement|null>(null);
  latest.current=scan;selection.current=selected;baselineRef.current=baseline;modelRef.current=predictionModel;
  const cell=inspected?.h3_index===selected?inspected:scan?cellIn(scan,selected):undefined;
  const secondary=scan&&comparison?cellIn(scan,comparison):undefined;
  const forecast=baseline?.predictive_forecast_72h,views=forecast?.views??[],currentView=views.find(v=>v.key===horizon);
  const minColumn=scan?.physical_assessment.cells.filter(c=>c.fs_mid!=null).reduce<Cell|undefined>((a,b)=>!a||b.fs_mid!<a.fs_mid!?b:a,undefined);
  const hotspot=scan?criticalCell(scan,predictionModel):undefined,peakScore=predictionScore(hotspot,predictionModel),activeRunout=route?.h3_index===selected?route:scan?.runout_assessment?.h3_index===selected?scan.runout_assessment:undefined;
  const inputs=baseline?.live_inputs,soil=inputs?.soil_properties;
  const locationTitle=scan?(/^-?\d+(?:\.\d+)?\s*,\s*-?\d/.test(scan.location.display_name)?'Selected area':scan.location.display_name||'Selected area'):'';
  const allocations=Object.entries(cell?.uncertainty_variance_allocation??{}).sort((a,b)=>Math.abs(b[1])-Math.abs(a[1]));
  const maxAllocation=Math.max(1e-12,...allocations.map(([,v])=>Math.abs(v))),feed=provider?scan?.source_status[provider]:undefined;
  useEffect(()=>{let cancelled=false;const check=()=>api.health().then(value=>{if(!cancelled){setHealth(value);setEngine('active');}}).catch(()=>{if(!cancelled)setEngine('offline');});void check();const timer=setInterval(check,30000);return()=>{cancelled=true;clearInterval(timer);request.current?.abort();chatSequence.current++;chatRequest.current?.abort();};},[]);
  useEffect(()=>{const key=(event:KeyboardEvent)=>{const target=event.target as HTMLElement;if(event.key==='/'&&!['INPUT','TEXTAREA'].includes(target.tagName)){event.preventDefault();searchInput.current?.focus();}if(event.key==='Escape'){setProvider(null);setSourcesOpen(false);setMobilePanel(null);setCopilotOpen(false);}};window.addEventListener('keydown',key);return()=>window.removeEventListener('keydown',key);},[]);
  useEffect(()=>{if(chat.length||chatBusy)chatEnd.current?.scrollIntoView({block:'nearest',behavior:'smooth'});},[chat,chatBusy]);
  useEffect(()=>{
    setInspected(undefined);setRoute(undefined);if(!scan||!selected)return;
    const controller=new AbortController();
    const active=()=>!controller.signal.aborted&&latest.current?.scan_id===scan.scan_id&&latest.current?.scenario_id===scan.scenario_id&&selection.current===selected;
    api.inspect(scan,selected,controller.signal).then(c=>{if(active()&&c.h3_index===selected)setInspected({...cellIn(scan,selected),...c,...c.physics_informed_assessment});}).catch(e=>{if(active())setError('Cell inspection: '+errorText(e));});
    api.runout(scan,selected,controller.signal).then(r=>{if(active()&&r.h3_index===selected)setRoute(r);}).catch(e=>{if(active())setError('Runout inspection: '+errorText(e));});
    return()=>controller.abort();
  },[scan?.scan_id,scan?.scenario_id,selected]);
  function install(next:Scan,selectCritical=false){
    const id=!selectCritical&&cellIn(next,selection.current)?selection.current:criticalCell(next,modelRef.current)?.h3_index??primary(next);
    latest.current=next;selection.current=id;setScan(next);setSelected(id);setInspected(undefined);setRoute(undefined);
    setComparison(previous=>!selectCritical&&cellIn(next,previous)?previous:'');
    if(selectCritical)document.querySelector('.cell-panel')?.scrollTo({top:0});
  }
  function cancelChat(){
    chatSequence.current++;chatRequest.current?.abort();chatRequest.current=null;setChatBusy(false);
    setChat(messages=>messages.map(message=>message.executing?{...message,executing:false,executionError:'The action sequence stopped because the workspace changed.'}:message));
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
      install(next,true);baselineRef.current=next;setBaseline(next);setHorizon('now');if(!chain)setChat([]);setEngine('active');return next;
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
  function selectModel(model:PredictionModel,agent=false){if(!agent)cancelChat();modelRef.current=model;setPredictionModel(model);setLayer('probability');}
  function selectLayer(next:MapLayer,agent=false){if(!agent)cancelChat();setLayer(next);}
  function openCopilot(){setCopilotOpen(true);requestAnimationFrame(()=>chatInput.current?.focus({preventScroll:true}));}
  function renderAction(){return new Promise<void>(resolve=>{let done=false;const finish=()=>{if(!done){done=true;clearTimeout(timer);resolve();}};const timer=setTimeout(finish,150);requestAnimationFrame(()=>requestAnimationFrame(finish));});}
  async function send(event:React.FormEvent){event.preventDefault();if(!draft.trim()||chatBusy||busy)return;
    const current=latest.current,text=draft.trim(),identity=current?.scenario_id??null,cellId=selection.current,scanId=current?.scan_id??null,model=modelRef.current;
    const history=chat.filter(m=>(m.state??null)===identity&&m.cell===cellId&&m.model===model).map(m=>({role:m.role,content:m.text}));
    const token=++chatSequence.current,controller=new AbortController();chatRequest.current?.abort();chatRequest.current=controller;
    const alive=()=>!controller.signal.aborted&&chatSequence.current===token;
    const original=()=>alive()&&(latest.current?.scan_id??null)===scanId&&(latest.current?.scenario_id??null)===identity&&selection.current===cellId&&modelRef.current===model;
    const replyId='assistant-'+token;
    setDraft('');setChat(c=>[...c,{role:'user',text,state:identity,cell:cellId,model}]);setChatBusy(true);
    try{
      const reply=await api.chat(current,cellId,text,history,controller.signal,model);
      if(!original()||reply.scan_id!==scanId||reply.scenario_id!==identity||(reply.active_prediction_model&&reply.active_prediction_model!==model))return;
      if(reply.updated_scan_data?.scan_id===scanId&&reply.updated_scan_data.scenario_id===identity)install(reply.updated_scan_data);
      setChat(c=>[...c,{id:replyId,role:'assistant',text:reply.clean_markdown,guard:reply.briefing_guard,claims:reply.bound_claims,state:reply.scenario_id,cell:cellId,model,error:reply.model_error,steps:reply.thought_steps,executing:!!reply.ui_action?.actions.length,outcomes:[]}]);
      const outcomes=await executeActions(reply,{
        scan:()=>latest.current,selected:()=>selection.current,model:()=>modelRef.current,isCurrent:alive,
        scanCoordinates:async(latitude,longitude,label)=>{if(label)setQuery(label);await acquire({latitude,longitude},controller);},
        searchLocation:async value=>{setQuery(value);await acquire(value,controller);},
        selectModel:async value=>{selectModel(value,true);await renderAction();},setHorizon:value=>selectHorizon(value,controller),setLayer:async value=>{selectLayer(value,true);await renderAction();},
        flyTo:(c,camera)=>selectCell(c.h3_index,true,false,camera),
        inspect:c=>{selectCell(c.h3_index,false,false);setMobilePanel('cell');},
        compare:(a,b)=>{selectCell(a.h3_index,false,false);setComparison(b.h3_index);setMobilePanel('cell');},
        outcome:outcome=>{if(alive())setChat(messages=>messages.map(m=>m.id===replyId?{...m,outcomes:[...m.outcomes??[],outcome]}:m));},
      });
      if(alive())setChat(messages=>messages.map(m=>m.id===replyId?{...m,executing:false,executionError:outcomes.find(o=>o.status==='failed')?.summary}:m));
    }catch(e){if(alive())setChat(c=>c.some(m=>m.id===replyId)?c.map(m=>m.id===replyId?{...m,executing:false,executionError:errorText(e)}:m):[...c,{role:'assistant',text:'The assistant request could not complete: '+errorText(e),guard:'SERVICE_ERROR',state:identity,cell:cellId,model}]);}
    finally{if(chatRequest.current===controller){chatRequest.current=null;setChatBusy(false);}}
  }
  function exportScan(){if(!scan)return;const url=URL.createObjectURL(new Blob([JSON.stringify({...scan,display_context:{prediction_model:predictionModel,map_layer:layer,selected_h3_index:selected,comparison_h3_index:comparison||null}},null,2)],{type:'application/json'}));const link=document.createElement('a');link.href=url;link.download=`terra-${scan.scan_id}-${scan.scenario_id}.json`;link.click();setTimeout(()=>URL.revokeObjectURL(url),1000);}
  return <div className={`command-center ${scan?'has-scan':''}`}>
    <MapCanvas scan={scan} selected={selected} compared={comparison} predictionModel={predictionModel} onSelect={id=>selectCell(id)} onScanCoordinate={(latitude,longitude)=>void startScan({latitude,longitude})} handle={map} runout={route} layer={layer}/>
    <header className="command-bar glass" aria-label="Map workspace tools">
      <form className="global-search" onSubmit={search}>
        <Search size={18}/><input ref={searchInput} aria-label="Search any place or latitude, longitude" value={query} onChange={e=>setQuery(e.target.value)} placeholder="Search a place or coordinates"/>
        <kbd>/</kbd><button type="submit" title="Search and scan" aria-label="Search and scan"><ArrowUpRight size={18}/></button>
      </form>
      <button className="scan-center" onClick={()=>map.current?.scanCenter()}><Crosshair size={16}/><span>Scan map view</span></button>
      <div className="toolbar-actions">
        {scan&&<button className={`toolbar-button ${sourcesOpen?'active':''}`} onClick={()=>setSourcesOpen(v=>!v)} aria-label="Data sources" title="Data sources" aria-expanded={sourcesOpen} aria-controls="source-panel"><Database size={16}/><span>Sources</span></button>}
        <button className={`toolbar-button ${copilotOpen?'active':''}`} type="button" onClick={openCopilot} aria-label="Open assistant" title="Assistant" aria-expanded={copilotOpen} aria-controls="assistant-drawer"><MessageSquare size={16}/><span>Assistant</span></button>
        {scan&&<button className="toolbar-button export-button" onClick={exportScan} title="Export analysis" aria-label="Export analysis"><Download size={17}/></button>}
      </div>
      {engine==='offline'&&<span className="connection-error" role="status"><AlertCircle size={14}/>Service unavailable</span>}
    </header>

    {sourcesOpen&&scan&&<section id="source-panel" className="source-popover glass" aria-label="Data sources">
      <div className="popover-title"><h2>Data sources</h2><button onClick={()=>setSourcesOpen(false)} aria-label="Close data sources"><X size={18}/></button></div>
      <p className="support-note">Availability and accuracy vary by source. Select a source for its provenance.</p>
      <div className="source-list">{SOURCES.map(name=>{const p=scan.source_status[name];return <button key={name} className={provider===name?'selected':''} onClick={()=>setProvider(v=>v===name?null:name)} aria-expanded={provider===name}><span>{SOURCE_LABELS[name]}</span><small className={tone(p)}>{providerLabel(p)}</small></button>;})}</div>
      {provider&&<div className="source-detail"><h3>{feed?.provider??SOURCE_LABELS[provider]}</h3><p>{feed?.error??'Source provenance refers to the current scan. Availability does not establish local measurement accuracy.'}</p><dl className="telemetry"><div><dt>Source mode</dt><dd>{stateOf(feed)}</dd></div><div><dt>Retrieved</dt><dd>{time(feed?.fetched_at_utc)}</dd></div><div><dt>Fetch time</dt><dd>{feed?.latency_ms==null?'Not recorded':fmt(feed.latency_ms,0)+' ms'}</dd></div><div><dt>Age</dt><dd>{feed?.age_seconds==null?'Not reported':fmt(feed.age_seconds,0)+' s'}</dd></div></dl></div>}
      <a className="text-link" href="https://github.com/pookie421/Terra/blob/main/docs/global_source_audit.md" target="_blank" rel="noreferrer">Source methods and limitations <ExternalLink size={13}/></a>
    </section>}

    {!scan&&<section className="welcome-card glass"><h1>Explore an area</h1><p>Search a location or click the map to view landslide estimates and the 72-hour forecast.</p><div className="engine-section"><label htmlFor="prediction-model-start">Prediction model</label><select id="prediction-model-start" value={predictionModel} onChange={e=>selectModel(e.target.value as PredictionModel)}>{PREDICTION_MODELS.map(model=><option key={model} value={model}>{MODEL_LABELS[model]}</option>)}</select></div><button className="welcome-assistant" onClick={openCopilot}><MessageSquare size={14}/>Ask the assistant to navigate</button></section>}
    {scan&&<>
      <div className="sector-context">
        <h1 title={scan.location.display_name}>{locationTitle}</h1>
        <p>{time(currentView?.valid_time_utc??scan.generated_at_utc)}</p>
        <div className="layer-switch glass" role="group" aria-label="Map layer">{(Object.keys(labels) as MapLayer[]).map(k=><button className={layer===k?'active':''} aria-pressed={layer===k} key={k} onClick={()=>selectLayer(k)}>{labels[k]}</button>)}</div>
        {scan.nisar&&<button className="radar-context glass" onClick={()=>{setSourcesOpen(true);setProvider('nisar');}}><Database size={12}/>NISAR L-SAR · {scan.nisar.mode==='METADATA_LIVE'?'metadata':cell?.nisar_coherence!=null?'measured subset':'source context'}</button>}
      </div>
      <aside className={`hud-panel sector-panel glass ${mobilePanel==='sector'?'mobile-open':''}`} aria-label="Area overview">
        <div className="panel-title"><h2>Area overview</h2></div>
        <section className="panel-section engine-section">
          <label htmlFor="prediction-model">Prediction model</label>
          <select id="prediction-model" value={predictionModel} onChange={e=>selectModel(e.target.value as PredictionModel)}>{PREDICTION_MODELS.map(model=><option key={model} value={model}>{model==='nasa_lhasa_2'&&scan.nasa_lhasa?.label?scan.nasa_lhasa.label:MODEL_LABELS[model]}</option>)}</select>
          <p className="support-note">{predictionModel==='nasa_lhasa_2'?nasaDescription(scan):'Terrain-informed estimates with inspectable physics and conditional runout.'}</p>
          <details className="nested-details"><summary>Resolution and scope</summary><p className="support-note">Terrain elevation sources are nominally 30 m where available. H3 resolution-9 cells average about 0.1 km² and span hundreds of metres. NASA LHASA rasters are about 1 km; sampling one pixel into several H3 cells preserves its original support.</p><p className="support-note">Finer terrain and radar context do not establish greater predictive skill. Compare shared cells, issue times and coverage.</p>{scan.nasa_lhasa&&<dl className="telemetry"><div><dt>NASA product</dt><dd>{scan.nasa_lhasa.product_version?.startsWith('NOT_')?'Served version unverified':scan.nasa_lhasa.product_version??'Served version unverified'}</dd></div><div><dt>NASA valid time</dt><dd>{scan.nasa_lhasa.valid_time_utc?time(scan.nasa_lhasa.valid_time_utc):'Not declared by service'}</dd></div><div><dt>Source mode</dt><dd>{scan.nasa_lhasa.provenance??scan.nasa_lhasa.mode??'Unavailable'}</dd></div></dl>}</details>
        </section>
        <section className="panel-section">
          <div className="sector-score"><span>Peak model score</span><strong className={peakScore!=null&&peakScore>=.65?'red':''}>{fmt(peakScore,2)}</strong></div>
          <p className="active-model-label">{modelLabel(scan,predictionModel)}</p>
          <div className="summary-interval"><span>Lowest factor of safety</span><b>{fmt(minColumn?.fs_mid)}</b><small>Central estimate</small></div>
          <button className="hud-button" disabled={!hotspot} onClick={()=>hotspot&&selectCell(hotspot.h3_index,true)}><Navigation size={15}/>View critical area<ArrowUpRight size={15}/></button>
          <p className="support-note">{peakScore===null?'No supported score for this model and time.':predictionModel==='nasa_lhasa_2'?nasaDescription(scan):'Experimental scores; physical columns use reference soil assumptions.'}</p>
        </section>
        <section className="panel-section forecast-section">
          <div className="section-title"><h3>72-hour forecast</h3></div>
          <div className="horizon-switch" role="group" aria-label="Forecast time">{HORIZONS.map(h=><button key={h.key} className={horizon===h.key?'active':''} aria-pressed={horizon===h.key} title={h.key==='peak'?'Lowest sector factor of safety in the forecast':h.sub} disabled={!!busy||(h.key!=='now'&&!views.some(v=>v.key===h.key))} onClick={()=>void selectHorizon(h.key).catch(e=>setError(errorText(e)))}>{h.label}</button>)}</div>
          <ForecastChart hourly={forecast?.paths?.p50?.hourly??[]}/>
          <p className="forecast-caption">{horizon==='peak'?'Lowest forecast stability':horizon==='now'?'Current conditions':'Forecast conditions'}{currentView?.hour!=null&&horizon!=='now'?` · +${currentView.hour}h`:''}</p>
          {forecast?.reason&&<p className="support-note amber">{forecast.reason}</p>}
          {predictionModel==='nasa_lhasa_2'&&horizon!=='now'&&<p className="support-note">NASA “Tomorrow” is a daily product, not an exact T+24h forecast. Unsupported later horizons remain unavailable.</p>}
        </section>
        <details className="panel-section disclosure observed-section" open><summary>Current conditions<ChevronDown size={15}/></summary>
          <dl className="telemetry"><Metric label="Rain · last 24h" value={inputs?.rain_24h_mm} unit="mm" digits={1}/><Metric label="Rain · preceding 6 days" value={inputs?.rain_prior_6d_mm} unit="mm" digits={1}/><Metric label="Surface moisture · 24h mean" value={inputs?.surface_moisture_m3m3} unit="m³/m³" digits={3}/></dl>
          <details className="nested-details"><summary>Soil and other inputs</summary><dl className="telemetry"><Metric label="Clay · 0–30 cm" value={numeric(soil,'clay_g_kg')==null?undefined:numeric(soil,'clay_g_kg')!/10} unit="%"/><Metric label="Sand · 0–30 cm" value={numeric(soil,'sand_g_kg')==null?undefined:numeric(soil,'sand_g_kg')!/10} unit="%"/><Metric label="Bulk density" value={numeric(soil,'bdod_kg_m3')} unit="kg/m³"/><Metric label="Organic carbon" value={numeric(soil,'soc_g_kg')} unit="g/kg"/><Metric label="USGS PGA · catalog proxy" value={inputs?.pga_proxy_g} unit="g" digits={4}/><Metric label="River discharge · modeled" value={inputs?.river_discharge_m3_s} unit="m³/s"/></dl><p className="support-note">Soil source: {String(soil?.source??'Unavailable').replaceAll('_',' ')}<br/>Weather: {time(inputs?.weather_valid_time_utc)}</p></details>
        </details>
        <details className="panel-section disclosure model-section" open><summary>Model comparison<ChevronDown size={15}/></summary>
          <div className="comparison-pair"><div><span>Terra · Physics + ML</span><strong>{fmt(predictionScore(cell,'terra_physics_ml'))}</strong><small>Conditional initiation score</small></div><div><span>{modelLabel(scan,'nasa_lhasa_2')}</span><strong>{fmt(predictionScore(cell,'nasa_lhasa_2'))}</strong><small>{cell?.official_score!=null||cell?.lhasa_probability!=null?'Official raster value':'Reference / unavailable'}</small></div></div>
          {PREDICTION_MODELS.map(model=>{const score=predictionScore(cell,model);return <div className={`model-row ${predictionModel===model?'primary':''}`} key={model}><div><button onClick={()=>selectModel(model)}>{modelLabel(scan,model)}</button><b>{fmt(score,2)}</b></div><div className={`model-track ${score===null?'missing':''}`}><i style={{width:`${Math.max(0,Math.min(100,(score??0)*100))}%`}}/></div></div>;})}
          <p className="support-note">Same selected H3 cell; native input support and score meanings differ. This display is not a matched predictive benchmark. {nasaDescription(scan)}</p>
        </details>
        <details className="panel-section disclosure analysis-details"><summary>Analysis details<ChevronDown size={15}/></summary><dl className="telemetry"><div><dt>Scan</dt><dd>{scan.scan_id}</dd></div><div><dt>State</dt><dd>{scan.scenario_id}</dd></div><div><dt>Minimum FS · low / central / high</dt><dd>{[minColumn?.fs_low,minColumn?.fs_mid,minColumn?.fs_high].map(v=>fmt(v)).join(' / ')}</dd></div><div><dt>Minimum FS cell</dt><dd>{minColumn?.h3_index}</dd></div><div><dt>Grid cells</dt><dd>{scan.physical_assessment.cells.length}</dd></div><div><dt>Elevation range</dt><dd>{range(scan.physical_assessment.cells.map(c=>c.elevation_m))} m</dd></div><div><dt>Slope range</dt><dd>{range(scan.physical_assessment.cells.map(c=>c.slope_deg))}°</dd></div><div><dt>Model</dt><dd>{health?.model_version??'Not reported'}</dd></div></dl><p className="support-note">Research estimates, not operational warnings.</p></details>
      </aside>

      <aside className={`hud-panel cell-panel glass ${mobilePanel==='cell'?'mobile-open':''}`} aria-label="Selected location">
        <div className="panel-title"><h2>Selected location</h2><button disabled={!cell} aria-label="Center selected location" title="Center selected location" onClick={()=>cell&&map.current?.flyTo(cell)}><Crosshair size={16}/></button></div>
        <section className="panel-section cell-identity"><p>{cell?coord(cell.latitude,cell.longitude):'Select an area on the map.'}</p><div className="cell-stats"><Datum name="Slope" value={cell?.slope_deg} unit="°"/><Datum name="Elevation" value={cell?.elevation_m} unit="m"/></div></section>
        {secondary&&<section className="panel-section pinned-comparison"><div className="section-title"><h3>Pinned cell comparison</h3><button onClick={()=>{cancelChat();setComparison('');}} aria-label="Remove pinned comparison"><X size={16}/></button></div><table><thead><tr><th>Metric</th><th>Selected</th><th>Pinned</th></tr></thead><tbody><tr><th>Slope</th><td>{fmt(cell?.slope_deg)}°</td><td>{fmt(secondary.slope_deg)}°</td></tr><tr><th>FS central</th><td>{fmt(cell?.fs_mid)}</td><td>{fmt(secondary.fs_mid)}</td></tr><tr><th>Active score</th><td>{fmt(predictionScore(cell,predictionModel))}</td><td>{fmt(predictionScore(secondary,predictionModel))}</td></tr></tbody></table><button className="hud-button" onClick={()=>{const previous=selected;selectCell(secondary.h3_index,true);setComparison(previous);}}>Inspect pinned cell<ArrowUpRight size={14}/></button><details className="nested-details"><summary>Exact cells</summary><p className="support-note">Selected: {selected}<br/>Pinned: {secondary.h3_index}<br/>{coord(secondary.latitude,secondary.longitude)}</p></details></section>}
        <section className="panel-section physical-column">
          <div className="section-title"><h3>Factor of safety</h3></div>
          <div className="fs-envelope"><div><span>Low</span><strong>{fmt(cell?.fs_low)}</strong></div><div className="nominal"><span>Central</span><strong>{fmt(cell?.fs_mid)}</strong></div><div><span>High</span><strong>{fmt(cell?.fs_high)}</strong></div></div>
          <p className="fs-scale">A value below 1 indicates potential instability.</p>
          <p className="prior-label">{cell?.provenance_label?.includes('pedotransfer')?'Uses reference soil assumptions':cell?.provenance_label?.replaceAll('_',' ')??'Reference column'}</p>
          <dl className="telemetry"><Metric label="Pore-pressure head" value={cell?.pressure_head_m} unit="m"/><div><dt>Additional rain to instability</dt><dd>{cell?.rainfall_threshold_attainable_at_duration===false?'Not reached in this window':cell?.critical_rainfall_deficit_mm==null?'—':fmt(cell.critical_rainfall_deficit_mm,1)+' mm'}</dd></div></dl>
          <details className="nested-details technical-details"><summary>Physical details</summary><dl className="telemetry"><div><dt>H3 cell</dt><dd>{selected}</dd></div><Metric label="Specific TWI" value={cell?.twi}/><Metric label="Joint variance · JΣJᵀ" value={cell?.fosm_variance} digits={4}/><Metric label="Mechanical admissibility" value={cell?.mechanical_admissibility} digits={4}/><Metric label="Learned trigger score" value={cell?.learned_trigger_score} digits={4}/><div><dt>Newmark displacement</dt><dd title={cell?.newmark_status}>{cell?.newmark_displacement_cm==null?'Outside input/domain support':fmt(cell.newmark_displacement_cm,4)+' cm'}</dd></div><div><dt>Column provenance</dt><dd>{cell?.provenance_label??'Not reported'}</dd></div></dl>
            <details className="sensitivity-details"><summary>Variance sensitivities</summary><p className="support-note">Signed covariance allocations, not causal percentages.</p>{allocations.map(([key,value])=><div className="sensitivity" key={key}><span>{key.replaceAll('_',' ')}</span><div><i className={value<0?'negative':''} style={{width:`${Math.abs(value)/maxAllocation*100}%`}}/></div><b>{fmt(value,3)}</b></div>)}<details className="matrix-details"><summary>Jacobian and covariance</summary><pre>{JSON.stringify({J:cell?.fs_jacobian,Sigma:cell?.joint_covariance},null,2)}</pre></details></details>
          </details>
        </section>
        <details className="panel-section disclosure radar-section"><summary>NISAR radar context<ChevronDown size={15}/></summary><p className="support-note">{scan.nisar?.mode==='METADATA_LIVE'?'Live acquisition metadata. No radar measurement is inferred from metadata.':scan.nisar?.scope??scan.nisar?.reason??'Radar observables require a compatible, georeferenced local or authenticated subset.'}</p><dl className="telemetry"><Metric label="L-band coherence" value={cell?.nisar_coherence} digits={3}/><Metric label="LOS velocity · measured series" value={cell?.nisar_los_velocity_mm_yr} unit="mm/yr"/><Metric label="Polarimetric index" value={cell?.nisar_polarimetric_moisture_index} digits={3}/><Metric label="Discovered granules" value={cell?cell.nisar_granule_count:scan.nisar?.granule_count} digits={0}/><div><dt>Latest discovered acquisition</dt><dd>{time(cell?cell.nisar_latest_acquisition_utc:scan.nisar?.latest_acquisition_utc)}</dd></div></dl><p className="support-note">Coherence can change with vegetation, geometry and processing. A single interferogram does not establish an annual deformation rate or soil strength.</p><details className="nested-details"><summary>Acquisition provenance</summary><p className="support-note">Discovery dates and measured-observation dates can differ. Measured sources retain their own acquisition intervals below.</p>{cell?.nisar_measurement_provenance&&<details className="nested-details"><summary>Measured radar sources</summary><pre>{JSON.stringify(cell.nisar_measurement_provenance,null,2)}</pre></details>}{scan.nisar?.granules?.map((granule,index)=><div className="granule-row" key={granule.granule_id??index}><span>{granule.granule_id??'Granule'}</span><small>{time(granule.time_start)} · {granule.orbit_direction??'Orbit not declared'}</small>{safeUrl(granule.browse_url)&&<a href={safeUrl(granule.browse_url)} target="_blank" rel="noreferrer">Browse acquisition<ExternalLink size={12}/></a>}</div>)}{!scan.nisar?.granules?.length&&<p className="support-note">No acquisition metadata is available in this scan.</p>}</details></details>
        <details className="panel-section disclosure runout-section"><summary>Potential runout<ChevronDown size={15}/></summary><div className="runout-numbers"><Datum name="Peak velocity" value={activeRunout?.peak_velocity_ms} unit="m/s"/><Datum name="Travel distance" value={activeRunout?.scenarios?.some(r=>r.travel_distance_m!=null)?Math.max(...activeRunout.scenarios.map(r=>r.travel_distance_m??0)):undefined} unit="m"/></div><p className="support-note">Conditional on release. Three rheology and depth assumptions; routes may stop at the scan boundary.</p><details className="nested-details"><summary>Routing assumptions</summary>{activeRunout?.scenarios?.map((r,index)=><div className="route-row" key={index}><i className={'route-'+index}/><span>μ {fmt(r.mu_b)} / ξ {fmt(r.xi_m_s2,0)} / depth {fmt(r.flow_depth_m)} m</span><b>{fmt(r.peak_velocity_m_s)} m/s</b><small>{r.stop_reason?.replaceAll('_',' ')}</small></div>)}<p className="support-note">{activeRunout?.corridor_h3_indices?.length??'—'} corridor cells</p></details></details>
      </aside>
      <div className="mobile-panel-switch glass" role="group" aria-label="Workspace panel"><button className={!mobilePanel?'active':''} aria-pressed={!mobilePanel} onClick={()=>setMobilePanel(null)}><MapPin size={16}/>Map</button><button className={mobilePanel==='sector'?'active':''} aria-pressed={mobilePanel==='sector'} onClick={()=>setMobilePanel('sector')}><Layers3 size={16}/>Overview</button><button className={mobilePanel==='cell'?'active':''} aria-pressed={mobilePanel==='cell'} onClick={()=>setMobilePanel('cell')}><Search size={16}/>Inspect</button></div>
    </>}
    {copilotOpen&&<aside id="assistant-drawer" className="assistant-drawer glass" aria-label="Assistant">
      <div className="panel-title"><h2>Assistant</h2><button onClick={()=>setCopilotOpen(false)} aria-label="Close assistant"><X size={18}/></button></div>
      <p className="assistant-context">{scan?modelLabel(scan,predictionModel)+' · '+locationTitle:'Search or scan a location from this conversation.'}</p>
      <div className="chat-log" aria-live="polite">
        {!chat.length&&<div className="chat-empty"><p>Ask about the current evidence or navigate the workspace in several steps.</p><button onClick={()=>setDraft(scan?'Compare Terra with NASA LHASA on the riskiest cell here.':'Take me to 11.548, 76.134 and inspect the steepest cell.')}><ArrowUpRight size={14}/>{scan?'Compare prediction models':'Navigate and inspect terrain'}</button></div>}
        {chat.map((m,index)=><article className={`chat-message ${m.role}`} key={m.id??index}>
          {m.guard&&<span title={m.guard} className={'verification '+(m.guard==='VERIFIED_LLM_PROSE'?'verified':m.guard==='VERIFIED_TEMPLATE'?'template':'failed')}>{m.guard==='VERIFIED_LLM_PROSE'||m.guard==='VERIFIED_TEMPLATE'?<Check size={12}/>:<AlertCircle size={12}/>} {m.guard==='VERIFIED_LLM_PROSE'?'Verified response':m.guard==='VERIFIED_TEMPLATE'?'Verified fallback':m.guard.toLowerCase().replaceAll('_',' ')}</span>}
          <p>{m.text}</p>
          {!!m.steps?.length&&<details className="agent-steps"><summary>Agent steps</summary><p className="support-note">Public action and evidence summaries.</p><ol>{m.steps.map(step=><li key={step.step}><strong>{step.phase}</strong><span>{step.reasoning}</span></li>)}</ol></details>}
          {!!m.outcomes?.length&&<details className="agent-outcomes" open><summary>Execution results</summary><ol>{m.outcomes.map(outcome=><li key={outcome.step} className={outcome.status==='failed'?'amber':''}>{outcome.status==='completed'?<Check size={13}/>:<AlertCircle size={13}/>}<span>{outcome.summary}</span></li>)}</ol></details>}
          {m.executing&&<p className="chat-thinking"><LoaderCircle size={13} className="spin"/>Executing the next workspace action…</p>}
          {m.executionError&&<p className="execution-error" role="status">{m.executionError}</p>}
          {(m.state||m.error||m.claims?.length||m.outcomes?.length)&&<details className="claim-citations"><summary>Response details</summary>{m.guard&&<p>{m.guard}</p>}{m.error&&<p className="amber">Local model service: {m.error}</p>}{m.state&&<p>Prose evidence state: {m.state}<br/>Cell: {m.cell}<br/>Model: {m.model}</p>}{m.claims?.map(c=><div key={c.name}><code>{c.path}</code><b>{fmt(c.value,5)}</b></div>)}{m.outcomes?.map(outcome=><p key={outcome.step}>Action {outcome.step}: {outcome.scan_id??'No scan'} / {outcome.scenario_id??'No forecast state'}</p>)}</details>}
        </article>)}
        {chatBusy&&!chat.some(m=>m.executing)&&<div className="chat-thinking"><LoaderCircle size={14} className="spin"/>Checking the request and current evidence…</div>}<div ref={chatEnd}/>
      </div>
      <form className="chat-form" onSubmit={send}><textarea ref={chatInput} rows={2} aria-label="Ask the assistant or control the workspace" value={draft} onChange={e=>setDraft(e.target.value)} onKeyDown={e=>{if(e.key==='Enter'&&!e.shiftKey){e.preventDefault();void send(e);}}} placeholder="Ask or describe the steps to perform…"/><button disabled={!draft.trim()||chatBusy||!!busy} aria-label="Send message"><Send size={16}/></button></form>
    </aside>}
    {busy&&<div className="acquisition glass" role="status"><LoaderCircle size={19} className="spin"/><div><strong>{busy}</strong><span>Gathering data and preparing the forecast.</span></div></div>}
    {error&&<div className="error-banner glass" role="alert"><AlertCircle size={18}/><div><strong>Unable to complete the request</strong><span>{error}</span></div><button onClick={()=>setError('')} aria-label="Dismiss error"><X size={17}/></button></div>}
  </div>;
}
function Metric({label,value,unit='',digits=2}:{label:string;value:unknown;unit?:string;digits?:number}){const display=typeof value==='number'&&value>0&&value<10**-digits?'<' + (10**-digits).toLocaleString('en-US',{maximumFractionDigits:digits}):fmt(value,digits);return <div><dt>{label}</dt><dd title={typeof value==='number'?String(value):undefined}>{display}{typeof value==='number'&&unit?' '+unit:''}</dd></div>;}
function Datum({name,value,unit=''}:{name:string;value:unknown;unit?:string}){return <div className="datum"><span>{name}</span><strong>{fmt(value)}{typeof value==='number'&&unit&&<small> {unit}</small>}</strong></div>;}
function range(values:number[]){const v=values.filter(Number.isFinite);return v.length?`${fmt(Math.min(...v),0)} — ${fmt(Math.max(...v),0)}`:'—';}
function ForecastChart({hourly}:{hourly:ForecastHour[]}){
  const values=hourly.map(h=>h.minimum_fs).filter((v):v is number=>typeof v==='number'&&Number.isFinite(v));
  if(!values.length)return <div className="forecast-empty"><Clock3 size={16}/><span>Forecast appears when weather data is available.</span></div>;
  const low=Math.min(1,...values)*.9,high=Math.max(1.1,...values)*1.05,span=Math.max(.1,high-low),y=(v:number)=>57-(v-low)/span*47;
  let path='',previous=false;hourly.forEach((h,i)=>{const v=h.minimum_fs;if(typeof v!=='number'||!Number.isFinite(v)){previous=false;return;}path+=`${previous?'L':'M'} ${((h.hour??i)/72*286+2).toFixed(2)} ${y(v).toFixed(2)} `;previous=true;});
  return <div className="forecast-chart"><span>Lowest forecast factor of safety</span><svg viewBox="0 0 290 72" role="img" aria-label="Automatic seventy-two hour minimum factor-of-safety trajectory"><path d={`M 2 ${y(1)} H 288`} stroke="#F59E0B" strokeOpacity=".45" strokeDasharray="3 4" fill="none"/><path d={path} fill="none" stroke="#90abc4" strokeWidth="2" vectorEffect="non-scaling-stroke"/></svg><div><span>Now</span><span>24h</span><span>48h</span><span>72h</span></div></div>;
}

function safeUrl(value?:string):string|undefined{if(!value)return;try{const url=new URL(value);return url.protocol==='https:'&&!url.username&&!url.password?url.href:undefined;}catch{return;}}
