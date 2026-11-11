"""Source-linked instruction traces; unavailable physics remains unavailable."""
from dataclasses import fields
import hashlib
import json
from pathlib import Path
import numpy as np
from backend.ml.global_xgb import FEATURE_NAMES, PHYSICS_MODEL_VERSION
from backend.physics.multi_causal_engine import MultiCausalEngine,PhysicsInputs
from backend.physics.features import physics_feature_matrix


def export_current_trajectories(rows,model,destination,count=600,excluded_groups=()):
    eligible={}
    for row in rows:
        if row.get('is_synthetic') or row.get('source_name') in (None,'user-provided project candidate list'):continue
        if not all(row.get(k) for k in ('event_id','event_date','h3_cell_id')):continue
        if row.get('event_group_id') in excluded_groups:continue
        key=(row.get('source_event_id') or row['event_id'],row['h3_cell_id'],row['event_date'])
        eligible[key]=row
    if len(eligible)<count:raise ValueError(f'Need {count} distinct source cell/dates; available {len(eligible)}; no padding')
    pools={}
    for key,row in sorted(eligible.items()):pools.setdefault(row.get('regime') or 'unknown',[]).append(row)
    chosen=[]
    while len(chosen)<count:
        for regime in sorted(pools):
            if pools[regime] and len(chosen)<count:chosen.append(pools[regime].pop(0))
    names={f.name for f in fields(PhysicsInputs)};engine=MultiCausalEngine()
    states={};matrices=[];indices=[]
    for i,row in enumerate(chosen):
        if not row.get('physics_inputs_json') or not all(row.get(k) is not None for k in FEATURE_NAMES):continue
        inp=PhysicsInputs(**{k:v for k,v in json.loads(row['physics_inputs_json']).items() if k in names})
        state=engine.compute(inp);matrix=physics_feature_matrix(inp,state,np.atleast_1d(inp.profile_curvature))
        np.testing.assert_allclose(matrix[0],[row[k] for k in FEATURE_NAMES],rtol=1e-5,atol=1e-6)
        states[i]=state;matrices.append(matrix[0]);indices.append(i)
    predictions=dict(zip(indices,model.predict_proba(np.asarray(matrices)))) if matrices else {}
    output=[]
    for i,row in enumerate(chosen):
        state=states.get(i)
        telemetry={'event_id':row['event_id'],'event_date':row['event_date'],'h3_cell_id':row['h3_cell_id'],
            'source_url':row.get('source_url'),'citation':row.get('citation'),'label_quality':row.get('label_quality'),
            'feature_provenance':row.get('physics_feature_provenance_json'),'physics_model_version':PHYSICS_MODEL_VERSION,
            'physics_status':'MODELED' if state is not None else 'UNAVAILABLE',
            'unavailable_reason':None if state is not None else row.get('physics_feature_error'),
            'features':{k:row.get(k) for k in FEATURE_NAMES},
            'p_xgb_calibrated':float(predictions[i]) if state is not None else None,
            'p_failure_physical':float(state.p_failure_physical[0]) if state is not None else None,
            'p_combined':None,'assumed_field_prevalence':None,
            'peak_velocity_ms':None,'motion_status':'UNAVAILABLE_NO_OBSERVED_RUNOUT_GEOMETRY',
            'forecast_status':'UNAVAILABLE_NO_EVENT_TIME_FORECAST_FORCING',
            'scope':'Source-linked calculation or explicit missing-data trace; not observed failure probability or operational warning'}
        if state is not None:
            answer=(f"Modeled FS is {state.fs_dynamic[0]:.4f}. The calibrated ML score is {100*predictions[i]:.2f}% for the sampled inventory/absence population; "
                    f"the separate FOSM diagnostic is {100*state.p_failure_physical[0]:.2f}%. Neither is independently validated as a global event probability. ")
        else:
            answer='Physics and ML scores are unavailable because the retained inputs are missing or incompatible. A relative wetness index cannot replace absolute saturation. '
        answer+='Runout geometry and event-time future forcing are unavailable, so speed and forecast failure time cannot be inferred.'
        messages=[{'role':'user','content':'Inspect this source-linked event cell with the tools. Distinguish observations, model calculations, and unavailable inputs.'},
            {'role':'assistant','tool_calls':[{'id':'telemetry','type':'function','function':{'name':'get_event_cell_telemetry','arguments':json.dumps({'event_id':row['event_id']})}}]},
            {'role':'tool','tool_call_id':'telemetry','name':'get_event_cell_telemetry','content':json.dumps(telemetry,sort_keys=True)},
            {'role':'assistant','content':answer}]
        output.append({'source_event_id':row['event_id'],'messages':messages})
    destination=Path(destination);destination.parent.mkdir(parents=True,exist_ok=True)
    destination.write_text(''.join(json.dumps(r,ensure_ascii=False,allow_nan=False)+'\n' for r in output))
    return {'rows':len(output),'distinct_cell_dates':len(chosen),'modeled_rows':len(states),'unavailable_rows':len(output)-len(states),
        'sha256':hashlib.sha256(destination.read_bytes()).hexdigest(),'physics_model_version':PHYSICS_MODEL_VERSION,
        'excluded_xgb_holdout_groups':sorted(excluded_groups),
        'scope':'Instruction corpus, not independent Qwen evaluation or 600 independent events. Missing physics is preserved; no invented forecast or motion.'}
