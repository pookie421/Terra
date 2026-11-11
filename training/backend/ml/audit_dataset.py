"""Deterministic feature refresh from retained source observations.

The old model-conditioned scenario-anchor and date-repair generators are retired.
Live catalog acquisition remains a separate operation in build_real_labeled_dataset.
"""
import hashlib
import json
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from .global_xgb import FEATURE_NAMES, PHYSICS_MODEL_VERSION


def feature_row(inputs, state):
    from backend.physics.features import physics_feature_matrix
    result=dict(zip(FEATURE_NAMES,map(float,physics_feature_matrix(inputs,state,np.atleast_1d(inputs.profile_curvature))[0])))
    result.update(fs_static_prior=float(state.fs_static[0]),p_failure_physical=float(state.p_failure_physical[0]))
    return result


def mechanical_controls():
    """No engine-labeled synthetic training anchors in the reference baseline."""
    return []


def refresh_frozen_dataset(path, source=None):
    import pyarrow as pa
    import pyarrow.parquet as pq
    from .build_real_labeled_dataset import populate_physics_features
    from .event_holdout import write_holdout
    path=Path(path)
    source=Path(source) if source else path.parent/'benchmark_history/v12/real_global_landslide_observations.parquet'
    if not source.exists():
        raise FileNotFoundError('Retained unscreened source required; refusing to reuse model-selected v13 controls: '+str(source))
    source_hash=hashlib.sha256(source.read_bytes()).hexdigest()
    rows=[r for r in pq.read_table(source).to_pylist() if not r.get('is_synthetic')]
    # Terrain is pinned to retained payloads. No fetch, fallback acquisition, or
    # highest-slope search can change this integration benchmark.
    terrains={}
    for file in sorted((path.parent/'source_catalogs').glob('terrain_v13_*.json')):
        payload=json.loads(file.read_text())
        if 'center' not in payload or 'cells' not in payload:continue
        terrains[tuple(payload['center'])]=SimpleNamespace(cells={c['cell_id']:SimpleNamespace(**c) for c in payload['cells']})
    failures=[]
    for row in rows:
        terrain=terrains.get((row['latitude'],row['longitude']))
        # Source record retains its original support if a neighborhood is absent.
        if terrain is None and row.get('terrain_slope_deg') is not None:
            cell=SimpleNamespace(cell_id=row['h3_cell_id'],latitude=row['latitude'],longitude=row['longitude'],
                slope_deg=row['terrain_slope_deg'],profile_curvature=row.get('terrain_profile_curvature') or 0.,
                twi=row.get('terrain_twi') or 6.,elevation_m=row.get('terrain_elevation_m') or 0.)
            terrain=SimpleNamespace(cells={cell.cell_id:cell})
        row['is_synthetic']=False
        row['physics_model_version']=PHYSICS_MODEL_VERSION
        row['quiescent_control_verified']=False  # inventory absence is never verified stability
        if row.get('label')==0:
            row['control_verification_scope']='Retained v12 dates; external-weather/date sampling, no FS/probability acceptance. Not global random backgrounds.'
        try:
            if terrain is None:raise ValueError('Retained terrain unavailable')
            populate_physics_features(row,terrain)
        except (ValueError,TypeError,KeyError) as exc:
            row.update({name:None for name in FEATURE_NAMES})
            row['physics_inputs_json']=None
            row['physics_feature_quality']='unavailable_reference_inputs'
            row['physics_feature_error']=str(exc)
        row['physics_feature_provenance_json']=json.dumps(row.get('physics_feature_provenance') or {},sort_keys=True)
        if not all(row.get(k) is not None for k in FEATURE_NAMES):
            failures.append({'event_id':row['event_id'],'label':row.get('label'),'reason':row.get('physics_feature_error')})
    keys=sorted(set().union(*(r.keys() for r in rows)))
    pq.write_table(pa.Table.from_pylist([{k:r.get(k) for k in keys} for r in rows]),path,compression='zstd')
    # Remove stale revised/model-filtered semantics while keeping compatible file names.
    split=write_holdout(path,rows)
    hold=pq.read_table(path.with_name('historical_hindcast_candidates.parquet'))
    pq.write_table(hold,path.with_name('historical_hindcast_revised_controls.parquet'),compression='zstd')
    pq.write_table(pa.table({'event_id':pa.array([],type=pa.string()),'label':pa.array([],type=pa.int8())}),path.with_name('mechanical_training_controls.parquet'))
    summary={'physics_model_version':PHYSICS_MODEL_VERSION,'row_count':len(rows),
        'source_path':str(source),'source_sha256':source_hash,'network_used':False,
        'labeled_positive_rows':sum(r.get('label')==1 for r in rows),'labeled_negative_rows':sum(r.get('label')==0 for r in rows),
        'synthetic_training_only_rows':0,'complete_14_feature_rows':len(rows)-len(failures),
        'unavailable_inputs':failures,'event_group_holdout':split,
        'label_policy':'Retained source proxies and externally selected temporal inventory absence; no model-conditioned screening',
        'evaluation_scope':'Existing development cohorts, not new independent global benchmarks. Excluding incompatible inputs introduces availability selection; coverage is reported.'}
    path.with_suffix('.metadata.json').write_text(json.dumps(summary,indent=2)+'\n')
    return summary


async def audit_refresh(path):
    return refresh_frozen_dataset(path)
