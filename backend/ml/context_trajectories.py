"""State-bound Qwen traces from retained sources; full-component holdout embargo."""
import hashlib,json
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

SOURCE_NAMES=('geocoding','terrarium_dem','open_meteo','nasa_power','soilgrids','usgs','overpass',
              'eonet_wildfires','eonet_landslides','coolr','river','bedrock','ensemble','satellite')


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
    model=joblib.load(art/'physics_informed_models.pkl')['Physics-Informed ML']
    output=[];scored=0
    for r in selected:
        d,_=terrains[(r['latitude'],r['longitude'])]
        terrain={c['cell_id']:SimpleNamespace(**c) for c in d['cells']}
        cell=terrain.get(r['h3_cell_id']) or min(terrain.values(),key=lambda c:(c.latitude-r['latitude'])**2+(c.longitude-r['longitude'])**2)
        try:item=extract(r,data,terrains);window=item['window']
        except (ValueError,KeyError,OSError):window={'status':'FORCING_PRIOR','rain_24h_mm':0.,'rain_prior_6d_mm':0.}
        physical=columns([cell],window)[0];score=None;explanation=None
        if window.get('status')=='AVAILABLE' and r.get('regime')=='monsoon_rain':
            x=matrix(feature_vector(window,cell.slope_deg,cell.twi,cell.profile_curvature),physical)[None,:]
            score=float(model.predict_proba(x)[0]);explanation=model.explain(x)[0];scored+=1
        identity=hashlib.sha256((r['event_id']+r['event_date']).encode()).hexdigest()[:24]
        runout=reference_runout(SimpleNamespace(terrain=SimpleNamespace(cells=terrain),infrastructure=None),cell.cell_id)
        selected_cell={k:physical[k] for k in ('h3_index','fs_mid','fs_low','fs_high','fs_interval','fosm_variance','mechanistic_features','pressure_head_m','initial_pressure_head_m','provenance_label')}
        selected_cell.update(score=score,model_explanation=explanation)
        evidence={'scan_id':'historical-'+identity,'scenario_id':'retained-event-day','selected_cell':selected_cell,
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
                'it does not assert that this slope has failed. Inspect the cell and apply a scenario using the active state identifiers.')
        # Flat zero-shear cells have undefined FS, so omit their unavailable claims.
        if physical['fs_mid'] is None:
            claims={};prose='The column has no modeled gravity driving shear. Inspect the conditional release corridor and source provenance; the rainfall score does not establish an operational warning.'
        answer={'markdown':prose,'claims':claims}
        from backend.llm.state_copilot import verify_and_render
        verify_and_render(answer,evidence)
        messages=[{'role':'system','content':'Explain active mathematical-physics/ML evidence. Return JSON with markdown and path-bound numeric claims; no free numeric literals or unsupported warnings.'},
            {'role':'user','content':'Inspect the active scan/scenario, explain its margin, uncertainty, feature evidence and conditional runout.'},
            {'role':'assistant','tool_calls':[{'id':'active','type':'function','function':{'name':'inspect_active_scan','arguments':json.dumps({'scan_id':evidence['scan_id'],'scenario_id':evidence['scenario_id']})}}]},
            {'role':'tool','tool_call_id':'active','name':'inspect_active_scan','content':json.dumps(evidence,allow_nan=False)},
            {'role':'assistant','content':json.dumps(answer)}]
        output.append({'source_event_id':r['event_id'],'messages':messages})
    target=data/'sft_context_trajectories.jsonl';target.write_text(''.join(json.dumps(r,allow_nan=False)+'\n' for r in output))
    (data/'sft_600_trajectories.jsonl').write_bytes(target.read_bytes())
    (data/'chatml_trajectories.jsonl').write_bytes(target.read_bytes())
    report={'rows':len(output),'distinct_cell_dates':len(output),'modeled_rows':len(output),'rainfall_ml_scored_rows':scored,
        'fusion_model_version':model.version,'fusion_model_sha256':hashlib.sha256((art/'physics_informed_models.pkl').read_bytes()).hexdigest(),
        'sha256':hashlib.sha256(target.read_bytes()).hexdigest(),'physics_model_version':VERSION,
        'excluded_xgb_holdout_groups':sorted(events),'excluded_region_blocks':sorted(regions),
        'holdout_component_overlap':0,'protocol':'active_scan_scenario_bound_claims_v1',
        'scope':'Source-linked instruction corpus; calibration rows may appear. No independent LLM or CUDA validation.'}
    (art/'sft_trajectory_report.json').write_text(json.dumps(report,indent=2)+'\n');return report
