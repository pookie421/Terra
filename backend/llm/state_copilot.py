"""Qwen co-pilot grounded in an immutable scan/scenario, with path-bound claims."""
import json,math,re
import httpx
from .model_client import chat_json_content, connection_summary, AssistantServiceError, AssistantConfigurationError
from .byok_client import request_chat_json_content
from .app_actions import (H3_TOKEN, MODELS, SCHEMA_VERSION, action_envelope, cell_registry,
                          plan_actions, public_steps, validate_actions)


def compact_cell(cell):
    result={k:v for k,v in cell.items() if k not in ('physics_informed_assessment','geometry','radar_feature_context','map_url','fs_dynamic','fs_distribution_quantiles_05_50_95','p_failure_linear_normal_diagnostic','fosm_log_variance')}
    if isinstance(result.get('model_explanation'),dict):
        result['model_explanation']={k:v for k,v in result['model_explanation'].items() if k!='folds'}
        terms=result['model_explanation'].get('feature_log_odds_terms')
        if isinstance(terms,dict):
            result['model_explanation']['feature_log_odds_terms']=dict(sorted(terms.items(),key=lambda item:abs(item[1]),reverse=True)[:6])
            result['model_explanation']['contribution_scope']='Largest absolute six terms; complete decomposition remains in the API'
    return result


def compact_runout(runout):
    if not runout:return runout
    keys=('status','h3_index','peak_velocity_ms','scope','stop_reason','peak_velocity_m_s','mu_b','xi_m_s2','flow_depth_m','total_distance_m','total_travel_time_s')
    result={k:runout[k] for k in keys if k in runout}
    result['scenarios']=[{k:row[k] for k in keys if k in row} for row in runout.get('scenarios',[])]
    result['corridor_cell_count']=len(runout.get('corridor_h3_indices',runout.get('path',[])))
    result['geometry_scope']='Complete paths and asset intersections remain in the state/runout API'
    return result


def provider_context(scan):
    nasa=scan.get('nasa_lhasa') or {};radar=scan.get('nisar') or {}
    common=('mode','provider','product_version','native_resolution_arcsec','native_resolution_m',
            'native_resolution_m_approx','valid_time_utc','scope','reason')
    return ({k:nasa[k] for k in common if k in nasa},
            {k:radar[k] for k in (*common,'granule_count','query_hit_count','latest_acquisition_utc','measurement_status') if k in radar})


def nasa_cell_context(scan, cell, product):
    """The NASA workspace binds only to its selected official daily product."""
    cell = dict(cell)
    source = next((row for row in (scan.get('nasa_lhasa') or {}).get('cells', [])
                   if row.get('h3_index') == cell.get('h3_index')), {})
    field = 'tomorrow_probability' if product == 'tomorrow' else 'official_score'
    value = source.get(field, source.get('probability') if product == 'today' else None)
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 1:
        value = None
    cell['lhasa_probability'] = value
    cell['nasa_product'] = product
    cell['lhasa_provenance'] = 'NASA_LHASA_LIVE_RASTER' if value is not None else 'NASA_PRODUCT_UNAVAILABLE'
    cell['prediction_scores'] = {**(cell.get('prediction_scores') or {}), 'nasa_lhasa_2': value}
    return cell


def context(scan, active_prediction_model='terra_physics_ml', nasa_product='today'):
    p=scan.get('physics_informed_assessment',{});cell=p.get('hotspot') or scan.get('physical_assessment',{}).get('hotspot') or {}
    same=next((r for r in scan.get('physical_assessment',{}).get('cells',[]) if r['h3_index']==cell.get('h3_index')),{})
    location={k:scan['location'].get(k) for k in ('query','display_name','latitude','longitude','agency_mode')}
    summary={k:scan['summary'].get(k) for k in ('physics_informed_score','fs_interval','fosm_variance','hazard_summary','physics_status')}
    nasa,radar=provider_context(scan)
    provenance={name:{k:v for k,v in status.items() if k in ('available','mode','error')}
                for name,status in scan['source_status'].items()}
    for status in provenance.values():
        if status.get('error'):status['error']=str(status['error'])[:140]
    forecast={k:v for k,v in (scan.get('forecast_inputs') or {}).items() if k in ('status','rain_24h_mm','rain_prior_6d_mm','soil_moisture_0_7cm_mean_m3m3','cutoff_utc','product','moisture_support')}
    nasa['selected_product']=nasa_product
    nasa['selected_product_time_note']='NASA daily service product; exact issue/valid time is undeclared unless source metadata provides it'
    selected_cell=compact_cell({**same,**cell})
    if active_prediction_model=='nasa_lhasa_2':selected_cell=nasa_cell_context(scan,selected_cell,nasa_product)
    return {'scan_id':scan['scan_id'],'scenario_id':scan['scenario_id'],'location':location,
        'scenario':scan['scenario'],'forecast_view':scan.get('forecast_view'),
        'live_inputs':scan.get('live_inputs'),'forecast_inputs':forecast or None,'selected_cell':selected_cell,'summary':summary,
        'runout_corridor':compact_runout(scan.get('runout_assessment')),'source_provenance':provenance,
        'active_prediction_model':active_prediction_model,'nasa_product':nasa_product,
        'prediction_models':[{k:v for k,v in row.items() if k in ('id','label','available','mode','product_version','peak_scope')}
            for row in (scan.get('prediction_models') or [])],
        'nasa_lhasa':nasa,'nisar':radar,
        'context_policy':'Selected physical column and source summaries; full grids, raw archives and routing geometry remain in the API',
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
    prose=re.sub(r'<think\b[^>]*>.*?</think\s*>','',prose,flags=re.S|re.I)
    if re.search(r'<\s*/?\s*think\b',prose,re.I):raise ValueError('Unclosed private reasoning markup')
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
    selected_model=e.get('active_prediction_model','terra_physics_ml')
    if selected_model=='nasa_lhasa_2':
        official=c.get('lhasa_probability')
        product=e.get('nasa_product','today').title()
        if isinstance(official,(int,float)) and not isinstance(official,bool):
            nasa_text=f"The selected official NASA {product} score is {f(official)}. Its native spatial support differs from the terrain mechanics grid; the service's exact issue and valid time are undeclared."
        else:
            nasa_text=f"The official NASA {product} score is unavailable for this cell. A Terra reference index is not substituted as NASA output."
        text=nasa_text+"\n\nThe following physical column, runout and automatic weather-forced forecast belong to Terra's separate conditional analysis, not NASA LHASA.\n\n"+text
    radar=[c.get(k) for k in ('nisar_coherence','nisar_los_velocity_mm_yr','nisar_polarimetric_moisture_index')]
    if not any(v is not None for v in radar):
        text+=' No measured radar field is included for this cell; discovery metadata does not establish displacement rate, moisture or strength.'
    else:
        text+=' Available radar fields retain acquisition, unit and quality context; their relationship to strength is not assumed calibrated.'
    return text


def planning_reply(message, actions, scan_id=None, scenario_id=None, scan=None):
    """Public plan only; new-location evidence is acquired by the UI executor."""
    return {'scan_id':scan_id,'scenario_id':scenario_id,'origin_scan_id':scan_id,'origin_scenario_id':scenario_id,
        'clean_markdown':('The requested app commands are ready. Location scans and forecast changes must finish before their returned evidence can be discussed.' if actions else
                          'Choose a map location or ask for a named-place or coordinate scan to begin the analysis.'),
        'numeric_audit_passed':True,'briefing_guard':'VERIFIED_TEMPLATE','bound_claims':[],
        'model_name':None,'model_tier':None,'model_error':None,
        'thought_steps':public_steps(actions,bool(scan)),
        'thought_steps_scope':'Public action and evidence summaries; not private model reasoning',
        'action_guard':'VALIDATED_USER_INTENT_PLAN','ui_action':action_envelope(actions,scan_id,scenario_id),
        'updated_scan_data':scan,'tool_calls':[]}


async def chat_active(agent,message,scan_id,scenario_id,sliders,agency_mode,selected_h3_index=None,history=None,autonomous=False,active_prediction_model='terra_physics_ml',request_assistant=None,nasa_product='today'):
    from .qwen_agent import parse_scenario_request
    state=agent.scan_service.get_scan_state(scan_id,scenario_id);scan=state['response']
    if active_prediction_model not in MODELS:raise ValueError('Unknown active prediction model')
    if nasa_product not in ('today','tomorrow'):raise ValueError('Unknown NASA daily product')
    if active_prediction_model=='nasa_lhasa_2' and ((scan.get('forecast_view') or {}).get('hour') or 0)>0:
        raise ValueError('Return to the original NOW scan before using a NASA daily product; Terra hourly forecasts are separate')
    if autonomous and sliders:raise ValueError('Autonomous chat cannot apply manual forcing')
    history=history or []
    if len(history)>12 or any(not isinstance(r,dict) or r.get('role') not in ('user','assistant') or not isinstance(r.get('content'),str) or len(r['content'])>4000 for r in history):
        raise ValueError('History must contain at most twelve bounded user/assistant messages')
    if sum(len(r['content']) for r in history)>16000:raise ValueError('Conversation history exceeds context budget')
    supplied_history_count=len(history)
    recent=[];history_bytes=0
    for row in reversed(history):
        length=len(row['content'].encode('utf-8'))
        if history_bytes+length>2400:break
        recent.append(row);history_bytes+=length
    history=list(reversed(recent))
    parsed={} if autonomous else parse_scenario_request(message,scan['location'].get('query',''),sliders)
    changes={k:parsed[k] for k in parsed.get('_explicit_fields',[]) if k in ('rain_delta_mm','storm_duration_hr','pga_override_g','snowmelt_temp_surge_c','burn_severity_override')}
    actions=[];origin_scan_id=scan_id;origin_scenario_id=scenario_id
    if changes:
        scan=await agent.scan_service.apply_scenario(scan_id,scenario_id,changes)
        actions.append({'type':'apply_scenario','changes':changes,'scan_id':scan['scan_id'],'scenario_id':scan['scenario_id']})
    e=context(scan,active_prediction_model,nasa_product);selected=e['selected_cell'].get('h3_index')
    requested=H3_TOKEN.search(message)
    selected=requested.group().lower() if requested else selected_h3_index or selected
    if selected:
        inspected=agent.scan_service.inspect_hexagon(selected,scan['scan_id'],scan['scenario_id'])
        e['selected_cell']=compact_cell({**inspected,**(inspected.get('physics_informed_assessment') or {})})
        if active_prediction_model=='nasa_lhasa_2':e['selected_cell']=nasa_cell_context(scan,e['selected_cell'],nasa_product)
        e['runout_corridor']=compact_runout(agent.scan_service.trace_runout(selected,scan['scan_id'],scan['scenario_id']))
    planner_scan={**scan,'active_prediction_model':active_prediction_model,'nasa_product':nasa_product}
    planned=plan_actions(message,planner_scan,selected)
    if any(a['type'] in ('scan_coordinates','search_location','set_time_horizon') for a in planned):
        return planning_reply(message,planned,scan['scan_id'],scan['scenario_id'],scan)
    for action in planned:
        if action['type']=='select_prediction_model':e['active_prediction_model']=action['model']
    if e['active_prediction_model']=='nasa_lhasa_2':e['selected_cell']=nasa_cell_context(scan,e['selected_cell'],nasa_product)
    planned_selection=next((a.get('primary_h3') if a['type']=='compare_cells' else a.get('h3_index')
        for a in reversed(planned) if a['type'] in ('inspect_cell','fly_to_cell','compare_cells')),None)
    if planned_selection and planned_selection!=selected:
        selected=planned_selection
        inspected=agent.scan_service.inspect_hexagon(selected,scan['scan_id'],scan['scenario_id'])
        e['selected_cell']=compact_cell({**inspected,**(inspected.get('physics_informed_assessment') or {})})
        if e['active_prediction_model']=='nasa_lhasa_2':e['selected_cell']=nasa_cell_context(scan,e['selected_cell'],nasa_product)
        e['runout_corridor']=compact_runout(agent.scan_service.trace_runout(selected,scan['scan_id'],scan['scenario_id']))
    actions.extend(planned)
    public=public_steps(actions,True)
    shared={'origin_scan_id':origin_scan_id,'origin_scenario_id':origin_scenario_id,
        'history_context':{'provided':supplied_history_count,'included':len(history),
            'omitted_oldest':supplied_history_count-len(history),'policy':'Most recent whole messages within a 2400-byte history budget; current evidence is preserved'},
        'thought_steps':public,'thought_steps_scope':'Public action and evidence summaries; not private model reasoning',
        'action_guard':'VALIDATED_USER_INTENT_PLAN','ui_action':action_envelope(actions,scan['scan_id'],scan['scenario_id'])}
    if request_assistant is None:
        # Retained programmatic/CLI compatibility; public API calls always supply
        # an explicit personal-key config, even in deterministic mode.
        model,tier=await agent._available_model()
        service_error=getattr(agent,'_availability_error',None)
    else:
        model=request_assistant.model if request_assistant.configured else None
        tier=0 if model else None
        service_error=None
    claims=[]
    if request_assistant is None and model is None and not service_error:
        return {'scan_id':scan['scan_id'],'scenario_id':scan['scenario_id'],
            'clean_markdown':'The local service is responding but no compatible Qwen model is installed. Load the custom terra-shield-qwen model or Qwen2.5 in Ollama; the structured analysis remains available.',
            'numeric_audit_passed':False,'briefing_guard':'MODEL_UNAVAILABLE','bound_claims':[],
            'model_name':None,'model_tier':tier,'updated_scan_data':scan,'tool_calls':[],**shared}
    if model:
        system=('Explain the active mathematical-physics/ML state and sources in useful prose. Return JSON only: '
            '{"markdown":"prose with {{fs}} numeric bindings","claims":{"fs":"/selected_cell/fs_mid"},"ui_action":{"actions":[]}}. '
            'All numbers in prose must be bindings to finite numeric JSON pointers into the supplied evidence. Do not emit free numeric literals in prose. '
            'Discuss uncertainty, mathematical margin, feature evidence and conditional runout. No claims of certain safety/failure or evacuation orders. '
            'Conversation history is context only, never a measurement source. The ACTIVE_EVIDENCE describes the selected displayed state. '
            'Do not invent or modify meteorological forcing. Optional ui_action.actions must reproduce authorized_action_plan exactly; '
            'do not add location or display commands that the user did not request. '
            'Radar metadata is not measured deformation; a coarse reference proxy is not an official NASA prediction. '
            'Finer terrain resolution does not establish superior predictive skill. Do not rank unvalidated proxy indices against official probabilities. '
            'Public action summaries are supplied by the server; do not reveal private reasoning or deliberation.')
        e['authorized_action_plan']=planned
        messages=[{'role':'system','content':system},*history,{'role':'user','content':message+'\nACTIVE_EVIDENCE\n'+json.dumps(e,allow_nan=False,separators=(',',':'))}]
        text=None;error=None
        async with httpx.AsyncClient(timeout=45.,trust_env=False) as client:
            for attempt in range(2):
                try:
                    candidate=(await request_chat_json_content(client,request_assistant,messages) if request_assistant is not None else
                               await chat_json_content(client,model,messages,lambda:agent.ollama_url))
                    if request_assistant is not None and request_assistant.api_key in candidate:
                        raise ValueError('Credential reflection is forbidden')
                    payload=json.loads(candidate);verified_text,verified_claims=verify_and_render(payload,e)
                    if 'ui_action' in payload:
                        candidate_actions=payload['ui_action']
                        if not isinstance(candidate_actions,dict) or 'actions' not in candidate_actions or set(candidate_actions)-{'actions','schema_version','execution_status','plan_id'}:raise ValueError('Invalid app plan envelope')
                        if candidate_actions.get('schema_version',SCHEMA_VERSION)!=SCHEMA_VERSION or candidate_actions.get('execution_status','PLANNED')!='PLANNED':raise ValueError('Invalid app plan version or execution status')
                        if 'plan_id' in candidate_actions and (not isinstance(candidate_actions['plan_id'],str) or len(candidate_actions['plan_id'])>128):raise ValueError('Invalid app plan identifier')
                        validated=validate_actions(candidate_actions['actions'],planner_scan,scan['scan_id'],scan['scenario_id'],selected)
                        if validated!=planned:raise ValueError('Generated actions differ from authorized user intent')
                        shared['action_guard']='VERIFIED_LLM_ACTIONS'
                    text,claims=verified_text,verified_claims
                    break
                except (AssistantServiceError,AssistantConfigurationError) as exc:
                    service_error=str(exc);break
                except httpx.TransportError as exc:
                    service_error=type(exc).__name__;break
                except (ValueError,KeyError,IndexError,TypeError,httpx.HTTPError) as exc:
                    error=type(exc).__name__;messages.append({'role':'user','content':'Verification failed. Repair the JSON and use only numeric bindings to supplied paths.'})
        if text is None and service_error:
            text=template(e);guard='VERIFIED_TEMPLATE'
        elif text is None:
            return {'scan_id':scan['scan_id'],'scenario_id':scan['scenario_id'],'clean_markdown':'Assistant prose failed evidence verification; structured scan remains available.',
                'numeric_audit_passed':False,'briefing_guard':'WITHHELD_UNVERIFIED_LLM_PROSE','verification_error':error,
                'model_name':model,'model_tier':tier,'updated_scan_data':scan,'tool_calls':[],**shared}
        else:guard='VERIFIED_LLM_PROSE'
    else:text=template(e);guard='VERIFIED_TEMPLATE'
    if guard=='VERIFIED_TEMPLATE':
        claims=[{'name':k,'path':'/selected_cell/'+k,'value':v} for k,v in e['selected_cell'].items() if k in ('fs_low','fs_mid','fs_high','fosm_variance','score') and isinstance(v,(int,float)) and math.isfinite(v)]
        claims.extend({'name':k,'path':'/selected_cell/prediction_scores/'+k,'value':v} for k,v in (e['selected_cell'].get('prediction_scores') or {}).items() if isinstance(v,(int,float)) and not isinstance(v,bool) and math.isfinite(v))
    return {'scan_id':scan['scan_id'],'scenario_id':scan['scenario_id'],'clean_markdown':text,'numeric_audit_passed':True,
        'briefing_guard':guard,'bound_claims':claims,'model_error':service_error,
        'fallback_reason':('ASSISTANT_SERVICE_UNAVAILABLE' if request_assistant is not None else
            'LOCAL_MODEL_SERVICE_UNAVAILABLE' if connection_summary()['provider']=='ollama' else 'ASSISTANT_SERVICE_UNAVAILABLE') if service_error else None,'model_name':model,'model_tier':tier,
        'updated_scan_data':scan,'tool_calls':[{'tool_name':'inspect_active_scan','arguments':{'scan_id':scan['scan_id'],'scenario_id':scan['scenario_id']}}],**shared}
