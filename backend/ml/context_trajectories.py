"""State-bound Qwen traces from retained sources; full-component holdout embargo."""
import hashlib,json
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import joblib
import pyarrow.parquet as pq
from .global_xgb import row_component_groups
from .physics_informed import matrix
from .rainfall_features import feature_vector
from backend.physics.reference_context import columns,VERSION
from backend.api.reference_assessment import reference_runout
from scripts.train_rainfall_baseline import extract
from backend.llm.app_actions import SCHEMA_VERSION, action_envelope, public_steps, validate_actions

SOURCE_NAMES=('geocoding','terrarium_dem','open_meteo','nasa_power','soilgrids','usgs','overpass',
              'eonet_wildfires','eonet_landslides','coolr','river','bedrock','ensemble','satellite','nisar','nasa_lhasa')


def training_plan(evidence, index, latitude, longitude):
    """Exercise the app contract on retained states without inventing new labels."""
    cid=evidence['selected_cell']['h3_index'];registry=evidence['cell_registry']
    prompts=[
        'Fly to the selected cell and inspect its mechanics and conditional runout.',
        'Compare the selected cell: switch to NASA LHASA, switch back to our in-house model and inspect this cell.',
        'Return to Now and inspect the selected cell. Keep forecast coverage explicit.',
        'Show the factor-of-safety layer and inspect the selected reference column.',
        'Compare the selected source cell with the flattest other cell using conditional mechanics, without assigning an invented stable label.',
        f'Scan {latitude}, {longitude}, switch to our in-house model, then inspect the steepest cell after the scan.',
        f'Search {latitude}, {longitude}, switch to NASA LHASA then inspect the riskiest returned cell after the scan.',
        'Show the NISAR radar layer and inspect this cell. Explain whether actual radar and NASA comparison measurements are present.'
    ]
    variant=index%len(prompts)
    plans=[
        [{'type':'fly_to_cell','h3_index':cid},{'type':'inspect_cell','h3_index':cid}],
        [{'type':'select_prediction_model','model':'nasa_lhasa_2'},{'type':'select_prediction_model','model':'terra_physics_ml'},{'type':'inspect_cell','h3_index':cid}],
        [{'type':'set_time_horizon','horizon':'now'},{'type':'inspect_cell','target':'selected'}],
        [{'type':'set_map_layer','layer':'fs_mid'},{'type':'inspect_cell','h3_index':cid}],
        [],
        [{'type':'scan_coordinates','latitude':latitude,'longitude':longitude,'label':f'{latitude:.6f}, {longitude:.6f}'},{'type':'select_prediction_model','model':'terra_physics_ml'},{'type':'inspect_cell','target':'steepest'}],
        [{'type':'search_location','query':f'{latitude}, {longitude}'},{'type':'select_prediction_model','model':'nasa_lhasa_2'},{'type':'inspect_cell','target':'riskiest'}],
        [{'type':'set_map_layer','layer':'nisar_radar'},{'type':'inspect_cell','h3_index':cid}]
    ]
    other=sorted((k for k in registry if k!=cid),key=lambda k:(registry[k]['slope_deg'],k))
    plans[4]=[{'type':'compare_cells','primary_h3':cid,'secondary_h3':other[0]}] if other else [{'type':'inspect_cell','h3_index':cid}]
    if other:prompts[4]=f'Compare {cid} with {other[0]} using conditional mechanics, without assigning an invented stable label.'
    else:prompts[4]='Inspect the selected source cell using conditional mechanics.'
    actions=validate_actions(plans[variant],evidence,evidence['scan_id'],evidence['scenario_id'],cid)
    prompt=prompts[variant]
    envelope=action_envelope(actions,evidence['scan_id'],evidence['scenario_id'])
    # The server computes its own idempotency hash; teaching the model to
    # invent or memorize opaque plan IDs adds no application capability.
    envelope.pop('plan_id')
    return prompt,envelope


def _preserve_old_instructions(data,art,content):
    target=data/'sft_context_trajectories.jsonl'
    if not target.exists():return None
    previous=target.read_bytes();digest=hashlib.sha256(previous).hexdigest()
    if previous==content:return None
    archive=data/'instruction_history'/digest;archive.mkdir(parents=True,exist_ok=True)
    archived=archive/target.name
    if archived.exists() and archived.read_bytes()!=previous:raise ValueError('Instruction history SHA collision or corrupted archive')
    archived.write_bytes(previous)
    report=art/'sft_trajectory_report.json'
    if report.exists():(archive/'sft_trajectory_report.json').write_bytes(report.read_bytes())
    return {'sha256':digest,'path':str(archived.relative_to(data.parent))}


def export(root,count=None):
    root=Path(root);data=root/'data';art=root/'artifacts'
    rows=pq.read_table(data/'real_global_landslide_observations.parquet').to_pylist()
    obs=[r for r in rows if r.get('label') in (0,1) and not r.get('is_synthetic')]
    components=row_component_groups(obs);manifest=json.loads((data/'event_holdout_manifest.json').read_text())
    held=set(manifest['heldout_groups']+manifest['legacy_reserved_groups'])
    blocked={g for r,g in zip(obs,components) if r['event_group_id'] in held}
    events={r['event_group_id'] for r,g in zip(obs,components) if g in blocked}
    regions={r['region_block'] for r,g in zip(obs,components) if g in blocked}
    chosen={}
    for r in rows:
        if r.get('is_synthetic') or r.get('region_block') in regions or r.get('event_group_id') in events:continue
        if not all(r.get(k) for k in ('event_id','event_date','h3_cell_id')):continue
        chosen[(r['h3_cell_id'],r['event_date'])]=r
    selected=sorted(chosen.values(),key=lambda r:r['event_id'])
    if count is not None:
        if len(selected)<count:raise ValueError(f'Only {len(selected)} distinct eligible cell/dates; no SFT padding permitted')
        selected=selected[:count]
    terrains={}
    for p in (data/'source_catalogs/terrain_metric_v1').glob('*.json'):
        d=json.loads(p.read_text());terrains[tuple(d['center'])]=(d,hashlib.sha256(p.read_bytes()).hexdigest())
    models=joblib.load(art/'physics_informed_models.pkl');model=models['Physics-Informed ML']
    output=[];scored=0;action_counts=Counter()
    for index,r in enumerate(selected):
        d,_=terrains[(r['latitude'],r['longitude'])]
        terrain={c['cell_id']:SimpleNamespace(**c) for c in d['cells']}
        cell=terrain.get(r['h3_cell_id']) or min(terrain.values(),key=lambda c:(c.latitude-r['latitude'])**2+(c.longitude-r['longitude'])**2)
        try:item=extract(r,data,terrains);window=item['window']
        except (ValueError,KeyError,OSError):window={'status':'FORCING_PRIOR','rain_24h_mm':0.,'rain_prior_6d_mm':0.}
        physical=columns([cell],window)[0];score=None;explanation=None;model_scores={}
        if window.get('status')=='AVAILABLE' and r.get('regime')=='monsoon_rain':
            x=matrix(feature_vector(window,cell.slope_deg,cell.twi,cell.profile_curvature),physical)[None,:]
            model_scores={name:float(fitted.predict_proba(x)[0]) for name,fitted in models.items()}
            score=model_scores['Physics-Informed ML'];explanation=model.explain(x)[0];scored+=1
        identity=hashlib.sha256((r['event_id']+r['event_date']).encode()).hexdigest()[:24]
        runout=reference_runout(SimpleNamespace(terrain=SimpleNamespace(cells=terrain),infrastructure=None),cell.cell_id)
        selected_cell={k:physical[k] for k in ('h3_index','fs_mid','fs_low','fs_high','fs_interval','fosm_variance','mechanistic_features','pressure_head_m','initial_pressure_head_m','provenance_label')}
        selected_cell.update(score=score,model_explanation=explanation)
        selected_cell['prediction_scores']={'terra_physics_ml':score,'nasa_lhasa_2':None,
            'physics_only':physical['p_failure_physical'],'weather_terrain_ml':model_scores.get('Weather+Terrain ML'),
            'rainfall_only':model_scores.get('Rainfall-Only')}
        selected_cell.update(nisar_coherence=None,nisar_los_velocity_mm_yr=None,nisar_polarimetric_moisture_index=None,
            nisar_granule_count=None,nisar_latest_acquisition_utc=None)
        # Geometry selectors are based on actual retained terrain. Additional
        # cells are display targets, never new negative labels or scored outcomes.
        options={cell.cell_id:cell}
        for option in (min(terrain.values(),key=lambda c:(c.slope_deg,c.cell_id)),max(terrain.values(),key=lambda c:(c.slope_deg,c.cell_id))):options[option.cell_id]=option
        others=[c for c in terrain.values() if c.cell_id!=cell.cell_id]
        if others:
            option=min(others,key=lambda c:(c.slope_deg,c.cell_id));options[option.cell_id]=option
        registry={cid:{'h3_index':cid,'slope_deg':c.slope_deg} for cid,c in options.items()}
        registry[cell.cell_id].update(fs_mid=physical['fs_mid'],score=score)
        evidence={'scan_id':'historical-'+identity,'scenario_id':'retained-event-day','selected_cell':selected_cell,
            'cell_registry':registry,'active_prediction_model':'terra_physics_ml',
            'nasa_lhasa':{'mode':'NOT_INCLUDED_HISTORICAL','official_score':None,'reference_score':None,
                'scope':'No as-issued NASA comparator payload is present in this retained trace; no NASA prediction is fabricated'},
            'nisar':{'mode':'NOT_INCLUDED_HISTORICAL','granules':[],'measurement_status':'NO_RETAINED_RADAR_MEASUREMENTS',
                'scope':'No retained SAR measurement; coherence, LOS velocity and polarimetric indices remain unavailable'},
            'runout_corridor':{k:runout[k] for k in ('status','path','peak_velocity_ms','corridor_h3_indices','scope')},
            'source_provenance':{k:{'mode':'RETAINED' if k in ('terrarium_dem','open_meteo','coolr') else 'NOT_INCLUDED',
                'scope':'Retained historical payload; no live availability claim'} for k in SOURCE_NAMES},
            'source_event_id':r['event_id'],'source_url':r.get('source_url'),'label_quality':r.get('label_quality'),
            'scope':'Historical event-day scenario, not advance prediction; no surveyed strength or operational warning'}
        claims={'low':'/selected_cell/fs_low','mid':'/selected_cell/fs_mid','high':'/selected_cell/fs_high','variance':'/selected_cell/fosm_variance'}
        prose=('The reference FS envelope extends from {{low}} to {{high}}, with nominal {{mid}} and joint uncertainty {{variance}}. '
               'These values use declared material/depth priors, antecedent water balance and rainfall pulses.\n\n')
        if score is not None:claims['score']='/selected_cell/score';prose+='The physics-informed initiation score is {{score}} on the sampled proxy population. '
        else:prose+='The rainfall ML model is outside this regime or lacks a complete forcing window. '
        prose+=('Feature attribution describes model evidence, not identified causes. The release corridor is conditional on the declared rheology; '
                'it does not assert that this slope has failed. NASA comparator outputs are unrecorded in this retained historical trace. '
                'Radar metadata or coherence alone cannot establish a displacement rate, moisture content or measured shear strength. '
                'App actions are requested plans; newly scanned or forecast states must return before their evidence is discussed.')
        # Flat zero-shear cells have undefined FS, so omit their unavailable claims.
        if physical['fs_mid'] is None:
            claims={};prose=('The column has no modeled gravity driving shear. Inspect the conditional release corridor and source provenance; the rainfall score does not establish an operational warning. '
                'NASA comparator outputs and radar measurements are unrecorded in this historical trace; no radar-derived deformation or strength is fabricated.')
        user_prompt,ui_action=training_plan(evidence,index,float(r['latitude']),float(r['longitude']))
        for action in ui_action['actions']:action_counts[action['type']]+=1
        answer={'markdown':prose,'claims':claims,'thought_steps':public_steps(ui_action['actions'],True),'ui_action':ui_action}
        from backend.llm.state_copilot import verify_and_render
        verify_and_render(answer,evidence)
        messages=[{'role':'system','content':'Explain active mathematical-physics/ML evidence. Return JSON with markdown, path-bound numeric claims, short public thought_steps action/evidence summaries and validated ui_action.actions. Never reveal private deliberation. Bind actions to the exact scan/state; resolve symbolic selectors after new scans. No free numeric literals in prose, fabricated NASA/radar observations, invented weather forcing or unsupported warnings.'},
            {'role':'user','content':user_prompt+' Explain the retained margin, uncertainty, model evidence and conditional runout separately from any new app actions.'},
            {'role':'assistant','tool_calls':[{'id':'active','type':'function','function':{'name':'inspect_active_scan','arguments':json.dumps({'scan_id':evidence['scan_id'],'scenario_id':evidence['scenario_id']})}}]},
            {'role':'tool','tool_call_id':'active','name':'inspect_active_scan','content':json.dumps(evidence,allow_nan=False)},
            {'role':'assistant','content':json.dumps(answer)}]
        output.append({'source_event_id':r['event_id'],'messages':messages})
    target=data/'sft_context_trajectories.jsonl';content=''.join(json.dumps(r,allow_nan=False)+'\n' for r in output).encode('utf-8')
    previous=_preserve_old_instructions(data,art,content)
    temporary=target.with_suffix('.jsonl.tmp');temporary.write_bytes(content);temporary.replace(target)
    (data/'sft_600_trajectories.jsonl').write_bytes(target.read_bytes())
    (data/'chatml_trajectories.jsonl').write_bytes(target.read_bytes())
    report={'rows':len(output),'distinct_cell_dates':len(output),'modeled_rows':len(output),'rainfall_ml_scored_rows':scored,
        'fusion_model_version':model.version,'fusion_model_sha256':hashlib.sha256((art/'physics_informed_models.pkl').read_bytes()).hexdigest(),
        'sha256':hashlib.sha256(target.read_bytes()).hexdigest(),'physics_model_version':VERSION,
        'excluded_xgb_holdout_groups':sorted(events),'excluded_region_blocks':sorted(regions),
        'holdout_component_overlap':0,'protocol':SCHEMA_VERSION,'action_counts':dict(action_counts),
        'source_channels':list(SOURCE_NAMES),'previous_instruction_corpus_preserved':previous,
        'radar_and_nasa_training_provenance':'Retained historical traces have no NASA as-issued or SAR measurements; missingness remains explicit',
        'scope':'Source-linked instruction corpus; calibration rows may appear. No independent LLM or CUDA validation.'}
    (art/'sft_trajectory_report.json').write_text(json.dumps(report,indent=2)+'\n');return report
