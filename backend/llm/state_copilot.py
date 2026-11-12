"""Qwen co-pilot grounded in an immutable scan/scenario, with path-bound claims."""
import json,math,re
import httpx


def context(scan):
    p=scan.get('physics_informed_assessment',{});cell=p.get('hotspot') or scan.get('physical_assessment',{}).get('hotspot') or {}
    same=next((r for r in scan.get('physical_assessment',{}).get('cells',[]) if r['h3_index']==cell.get('h3_index')),{})
    location={k:scan['location'].get(k) for k in ('query','display_name','latitude','longitude','agency_mode')}
    summary={k:scan['summary'].get(k) for k in ('physics_informed_score','fs_interval','fosm_variance','hazard_summary','physics_status')}
    return {'scan_id':scan['scan_id'],'scenario_id':scan['scenario_id'],'location':location,
        'scenario':scan['scenario'],'forecast_view':scan.get('forecast_view'),
        'live_inputs':scan.get('live_inputs'),'forecast_inputs':scan.get('forecast_inputs'),'selected_cell':{**same,**cell},'summary':summary,
        'runout_corridor':scan.get('runout_assessment'),'source_provenance':scan['source_status'],
        'scope':'Conditional reference mechanics and experimental initiation score; no operational warning authority'}


def pointer(data,path):
    if not isinstance(path,str) or not path.startswith('/'):raise ValueError('Claim needs a JSON pointer')
    for key in path[1:].split('/'):
        key=key.replace('~1','/').replace('~0','~')
        data=data[int(key)] if isinstance(data,list) else data[key]
    return data


def verify_and_render(payload,evidence):
    """No floating numbers may originate in prose; all are bound to metric paths.

    Canonical field labels accompany substitutions, preventing an unrelated
    telemetry number from silently passing as a different metric.
    """
    if not isinstance(payload,dict) or not isinstance(payload.get('markdown'),str):raise ValueError('Structured prose required')
    prose=payload['markdown'];claims=payload.get('claims',{})
    if re.search(r'\b(?:certain failure|failure is certain|guaranteed safe|evacuate immediately)\b',prose,re.I):raise ValueError('Unsupported operational certainty')
    prose=re.sub(r'<think>.*?</think>','',prose,flags=re.S)
    if re.search(r'\d',re.sub(r'\{\{[a-z_]+\}\}','',prose)):raise ValueError('Numeric prose must use bound claims')
    used=set(re.findall(r'\{\{([a-z_]+)\}\}',prose))
    if used!=set(claims):raise ValueError('Every claim must be used exactly by a named binding')
    bound=[]
    for name,path in claims.items():
        value=pointer(evidence,path)
        if isinstance(value,bool) or not isinstance(value,(float,int)) or not math.isfinite(value):raise ValueError('Claim is not a finite numeric metric')
        label=path.rsplit('/',1)[-1]
        rendered=f'{label}: {value:.6g}'
        prose=prose.replace('{{'+name+'}}',rendered);bound.append({'name':name,'path':path,'value':value})
    return prose,bound


def template(e):
    c=e['selected_cell'];interval=c.get('fs_interval',[None,None,None])
    def f(v):return 'unavailable' if v is None else f'{v:.6g}'
    text=(f"The selected cell is {c.get('h3_index','unavailable')}. Its reference FS envelope is "
          f"[{', '.join(f(v) for v in interval)}], with FOSM variance {f(c.get('fosm_variance'))}. "
          f"The physics-informed initiation score is {f(c.get('score'))}. These are conditional model outputs.\n\n"
          "The reference column combines declared material/depth priors with antecedent water balance and rainfall pulses. "
          "Its feature attribution explains model evidence; it does not establish causal percentages or measured groundwater.\n\n"
          "The runout corridor shows conditional release trajectories across the exposed rheology scenarios. "
          "Inspect the source provenance and automatic provider-forced forecast horizons at the same cell before interpreting changes.")
    return text


async def chat_active(agent,message,scan_id,scenario_id,sliders,agency_mode,selected_h3_index=None,history=None,autonomous=False):
    from .qwen_agent import parse_scenario_request
    state=agent.scan_service.get_scan_state(scan_id,scenario_id);scan=state['response']
    if autonomous and sliders:raise ValueError('Autonomous chat cannot apply manual forcing')
    history=history or []
    if len(history)>12 or any(not isinstance(r,dict) or r.get('role') not in ('user','assistant') or not isinstance(r.get('content'),str) or len(r['content'])>4000 for r in history):
        raise ValueError('History must contain at most twelve bounded user/assistant messages')
    if sum(len(r['content']) for r in history)>16000:raise ValueError('Conversation history exceeds context budget')
    parsed={} if autonomous else parse_scenario_request(message,scan['location'].get('query',''),sliders)
    changes={k:parsed[k] for k in parsed.get('_explicit_fields',[]) if k in ('rain_delta_mm','storm_duration_hr','pga_override_g','snowmelt_temp_surge_c','burn_severity_override')}
    actions=[]
    if changes:
        scan=await agent.scan_service.apply_scenario(scan_id,scenario_id,changes)
        actions.append({'type':'apply_scenario','changes':changes,'scan_id':scan['scan_id'],'scenario_id':scan['scenario_id']})
    e=context(scan);selected=e['selected_cell'].get('h3_index')
    requested=re.search(r'\b[0-9a-f]{15}\b',message,re.I)
    selected=requested.group().lower() if requested else selected_h3_index or selected
    if selected:
        inspected=agent.scan_service.inspect_hexagon(selected,scan['scan_id'],scan['scenario_id'])
        e['selected_cell']={**inspected,**(inspected.get('physics_informed_assessment') or {})}
        e['runout_corridor']=agent.scan_service.trace_runout(selected,scan['scan_id'],scan['scenario_id'])
    if selected:
        kinds=['inspect_cell']
        if re.search(r'\b(?:fly|zoom|show|locate)\b',message,re.I):kinds.insert(0,'fly_to_cell')
        for kind in kinds:actions.append({'type':kind,'h3_index':selected,'scan_id':scan['scan_id'],'scenario_id':scan['scenario_id']})
    model,tier=await agent._available_model();claims=[]
    service_error=getattr(agent,'_availability_error',None)
    if model is None and not service_error:
        return {'scan_id':scan['scan_id'],'scenario_id':scan['scenario_id'],
            'clean_markdown':'The local service is responding but no compatible Qwen model is installed. Load the custom terra-shield-qwen model or Qwen2.5 in Ollama; the structured analysis remains available.',
            'numeric_audit_passed':False,'briefing_guard':'MODEL_UNAVAILABLE','bound_claims':[],
            'model_name':None,'model_tier':tier,'ui_action':{'actions':actions},'updated_scan_data':scan,'tool_calls':[]}
    if model:
        system=('Explain the active mathematical-physics/ML state and sources in useful prose. Return JSON only: '
            '{"markdown":"prose with {{fs}} numeric bindings","claims":{"fs":"/selected_cell/fs_mid"}}. '
            'All numbers must be bindings to finite numeric JSON pointers into the supplied evidence. Do not emit free numeric literals. '
            'Discuss uncertainty, mathematical margin, feature evidence and conditional runout. No claims of certain safety/failure or evacuation orders. '
            'Conversation history is context only, never a measurement source. The ACTIVE_EVIDENCE describes the selected displayed state. '
            'Do not invent or modify meteorological forcing; autonomous mode permits only inspection and camera actions.')
        messages=[{'role':'system','content':system},*history,{'role':'user','content':message+'\nACTIVE_EVIDENCE\n'+json.dumps(e,allow_nan=False)}]
        text=None;error=None
        async with httpx.AsyncClient(timeout=45.) as client:
            for attempt in range(2):
                try:
                    response=await client.post(agent.ollama_url+'/api/chat',json={'model':model,'messages':messages,'stream':False,'format':'json','options':{'temperature':.1,'num_ctx':8192}})
                    response.raise_for_status();candidate=response.json()['message']['content']
                    text,claims=verify_and_render(json.loads(candidate),e);break
                except httpx.TransportError as exc:
                    service_error=type(exc).__name__;break
                except (ValueError,KeyError,TypeError,httpx.HTTPError) as exc:
                    error=type(exc).__name__;messages.append({'role':'user','content':'Verification failed. Repair the JSON and use only numeric bindings to supplied paths.'})
        if text is None and service_error:
            text=template(e);guard='VERIFIED_TEMPLATE'
        elif text is None:
            return {'scan_id':scan['scan_id'],'scenario_id':scan['scenario_id'],'clean_markdown':'Local model prose failed evidence verification; structured scan remains available.',
                'numeric_audit_passed':False,'briefing_guard':'WITHHELD_UNVERIFIED_LLM_PROSE','verification_error':error,
                'model_name':model,'model_tier':tier,'ui_action':{'actions':actions},'updated_scan_data':scan,'tool_calls':[]}
        else:guard='VERIFIED_LLM_PROSE'
    else:text=template(e);guard='VERIFIED_TEMPLATE'
    if guard=='VERIFIED_TEMPLATE':
        claims=[{'name':k,'path':'/selected_cell/'+k,'value':v} for k,v in e['selected_cell'].items() if k in ('fs_low','fs_mid','fs_high','fosm_variance','score') and isinstance(v,(int,float)) and math.isfinite(v)]
    return {'scan_id':scan['scan_id'],'scenario_id':scan['scenario_id'],'clean_markdown':text,'numeric_audit_passed':True,
        'briefing_guard':guard,'bound_claims':claims,'model_error':service_error,
        'fallback_reason':'LOCAL_MODEL_SERVICE_UNAVAILABLE' if service_error else None,'model_name':model,'model_tier':tier,
        'ui_action':{'actions':actions},'updated_scan_data':scan,'tool_calls':[{'tool_name':'inspect_active_scan','arguments':{'scan_id':scan['scan_id'],'scenario_id':scan['scenario_id']}}]}
