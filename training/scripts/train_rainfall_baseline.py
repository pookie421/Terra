#!/usr/bin/env python3
"""Reproducible rainfall baseline, using retained source observations."""
import hashlib,json,socket,sys,platform
from collections import Counter
from datetime import datetime,timedelta,timezone
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from backend.ml.rainfall_features import FEATURES,VERSION,source_window,feature_vector
from backend.ml.rainfall_baseline import RainfallBaseline,PRIMARY,CANDIDATES,grouped_folds
from backend.ml.global_xgb import row_component_groups
from backend.ml.validation_suite import ranking_metrics


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def load_terrains(data):
    terrains={}
    corrected=sorted((data/'source_catalogs/terrain_metric_v1').glob('*.json'))
    paths=corrected or sorted((data/'source_catalogs').glob('terrain_v13_*.json'))
    for path in paths:
        payload=json.loads(path.read_text())
        if 'center' in payload and 'cells' in payload:
            terrains[tuple(payload['center'])]=(payload,sha(path))
    return terrains


def extract(row,data,terrains):
    key=row.get('weather_archive_cache_key')
    if not key:raise ValueError('No retained hourly weather archive')
    path=data/'source_catalogs/historical_weather_v11'/f'{key}.json'
    payload=json.loads(path.read_text())
    # Retained archives end at 23:00, with preceding-hour rain sums. Never
    # fabricate the absent next-midnight sample or treat labels as onset times.
    cutoff=datetime.fromisoformat(row['event_date']).replace(tzinfo=timezone.utc)+timedelta(hours=23)
    window=source_window(payload['response'].get('hourly') or {},cutoff,product='archive')
    terrain,terrain_hash=terrains[(row['latitude'],row['longitude'])]
    cell=min(terrain['cells'],key=lambda c:(c['cell_id']!=row.get('h3_cell_id'),
             (c['latitude']-row['latitude'])**2+((c['longitude']-row['longitude'])*np.cos(np.radians(row['latitude'])))**2,c['cell_id']))
    vector=feature_vector(window,cell['slope_deg'],cell['twi'],cell['profile_curvature'])
    return {'event_id':row['event_id'],'event_group_id':row['event_group_id'],'region_block':row['region_block'],
        'event_date':row['event_date'],'label':row['label'],'label_quality':row.get('label_quality'),
        'source_url':row.get('source_url'),'h3_cell_id':cell['cell_id'],'regime':row['regime'],
        'features':[None if not np.isfinite(v) else float(v) for v in vector],
        'window':window,'weather_payload_sha256':sha(path),'terrain_payload_sha256':terrain_hash,
        'physics_available':row.get('fs_dynamic') is not None,
        'p_failure_physical':row.get('p_failure_physical')}


def prepare(data):
    import pyarrow.parquet as pq
    observations=pq.read_table(data/'real_global_landslide_observations.parquet').to_pylist()
    observed=[r for r in observations if r.get('label') in (0,1) and not r.get('is_synthetic')]
    components=row_component_groups(observed)  # full graph, before any feature/label-quality filtering
    manifest=json.loads((data/'event_holdout_manifest.json').read_text())
    held=set(manifest['heldout_groups']+manifest['legacy_reserved_groups'])
    embargo={g for r,g in zip(observed,components) if r['event_group_id'] in held}
    cal=pq.read_table(data/'conformal_calibration.parquet').to_pylist()
    cal_events={r['event_group_id'] for r in cal};cal_regions={r['region_block'] for r in cal}
    reserved={g for r,g in zip(observed,components) if r['event_group_id'] in cal_events or r['region_block'] in cal_regions}
    chosen=set(manifest['selected_ids']);terrains=load_terrains(data)
    train=[];hold=[];omissions=[];counts=Counter()
    for row,component in zip(observed,components):
        if row['regime']!='monsoon_rain':continue
        target=hold if row['event_id'] in chosen else train if component not in embargo|reserved else None
        if target is None:counts['embargoed_or_reserved']+=1;continue
        try:
            item=extract(row,data,terrains);item['component']=str(component);target.append(item)
        except (ValueError,KeyError,OSError,TypeError) as exc:
            omissions.append({'event_id':row['event_id'],'label':row['label'],'reason':str(exc)})
    water=Counter()
    for row in observations:
        theta=row.get('soil_moisture_72h_max_m3m3');n=row.get('soil_porosity_estimate')
        water['missing' if theta is None else 'incompatible_with_prior' if n is None or theta>n or theta<0 else 'compatible']+=1
    quality={'retained_rows':len(observations),'labeled_rows':len(observed),'physical_input_diagnostics':dict(water),
        'rainfall_labeled_rows':sum(r['regime']=='monsoon_rain' for r in observed),
        'embargoed_or_reserved_rainfall_rows':counts['embargoed_or_reserved'],
        'training_rows':len(train),'training_rows_with_physics':sum(r['physics_available'] for r in train),
        'recovered_training_rows_without_physics':sum(not r['physics_available'] for r in train),
        'omissions':omissions,'training_label_quality':dict(Counter(r['label_quality'] for r in train)),
        'scope':'Source grouping and missingness preserved; no model-selected controls, new labels, or density adjustment'}
    return train,hold,quality


def matrix(rows):return np.asarray([r['features'] for r in rows],float)


def evaluate_subset(rows,predictions):
    y=np.asarray([r['label'] for r in rows]);result={name:ranking_metrics(y,p) for name,p in predictions.items()}
    return result


def main_legacy_rainfall():
    import joblib
    import sklearn,xgboost
    from backend.release import invalidate_readiness
    def offline(*args,**kwargs):raise RuntimeError('No network in the retained rainfall benchmark')
    socket.socket.connect=offline
    data=ROOT/'data';art=ROOT/'artifacts'
    invalidate_readiness(art,'Rainfall statistical baseline and CLI need current verification')
    train,hold,coverage=prepare(data)
    x=matrix(train);y=np.array([r['label'] for r in train]);groups=np.array([r['component'] for r in train])
    folds=grouped_folds(y,groups)
    oof={name:np.full(len(y),np.nan) for name in ('prevalence_only',*CANDIDATES)};audits=[]
    for fold,(a,b) in enumerate(folds):
        for field in ('event_group_id','region_block'):
            assert not {train[i][field] for i in a}&{train[i][field] for i in b}
        oof['prevalence_only'][b]=float(y[a].mean())
        for kind in CANDIDATES:
            candidate=RainfallBaseline(kind).fit(x[a],y[a],groups[a])
            oof[kind][b]=candidate.predict_proba(x[b])
        audits.append({'fold':fold,'train_ids':[train[i]['event_id'] for i in a],
            'test_ids':[train[i]['event_id'] for i in b],'train_components':sorted(set(groups[a])),
            'test_components':sorted(set(groups[b])),'shared_events':0,'shared_regions':0})
        print(f'Finished outer fold {fold+1}/{len(folds)}',flush=True)
    primary=RainfallBaseline(PRIMARY).fit(x,y,groups)
    artifact=art/'rainfall_source_baseline.pkl';joblib.dump(primary,artifact)
    np.testing.assert_allclose(joblib.load(artifact).predict_proba(x),primary.predict_proba(x),rtol=0,atol=0)
    for field in ('event_group_id','region_block'):
        assert not {r[field] for r in train}&{r[field] for r in hold}
    common=np.array([r['physics_available'] for r in train])
    common_predictions={k:p[common] for k,p in oof.items()}
    common_predictions['fosm_diagnostic']=np.array([r['p_failure_physical'] for r in train if r['physics_available']])
    common_rows=[r for r in train if r['physics_available']]
    from backend.ml.uncertainty import cluster_intervals
    report={'feature_version':VERSION,'primary_model':PRIMARY,'model_selection':'Predeclared source logistic; no CV or holdout winner selection',
        'generated_at_utc':datetime.now(timezone.utc).isoformat(),
        'environment':{'python':platform.python_version(),'numpy':np.__version__,'sklearn':sklearn.__version__,'xgboost':xgboost.__version__},
        'network_used':False,'features':list(FEATURES),'coverage':coverage,
        'training_components':len(set(groups)),'positive_rows':int(np.count_nonzero(y==1)),
        'negative_rows':int(np.count_nonzero(y==0)),'cross_validation':evaluate_subset(train,oof),
        'fold_audit':audits,'primary_cluster_intervals':cluster_intervals(y,oof[PRIMARY],groups),
        'common_physics_subset':{'rows':len(common_rows),'metrics':evaluate_subset(common_rows,common_predictions)},
        'original_rainfall_development_holdout':{'rows':len(hold),
            'metrics':ranking_metrics([r['label'] for r in hold],primary.predict_proba(matrix(hold))) if hold else None},
        'oof_predictions':[{'event_id':r['event_id'],'label':int(label),'scores':{k:float(p[i]) for k,p in oof.items()}}
                           for i,(r,label) in enumerate(zip(train,y))],
        'model_sha256':sha(artifact),'observations_sha256':sha(data/'real_global_landslide_observations.parquet'),
        'limitations':['Retrospective features end at 23:00 UTC on the label date and can include weather after the actual event; not pre-event forecast validation',
            'Shared extraction windows do not remove reanalysis-to-live-model distribution shift',
            'Point/footprint labels and externally weather-selected temporal absences remain imperfect',
            'Raw moisture is a provider feature, not physical saturation or a groundwater estimate',
            'Small number of independent components; inspected holdout is development evidence',
            'No operational warning threshold, causal interpretation, or global probability calibration'],
        'independent_validation':False,'operational_warning_ready':False}
    (data/'rainfall_source_training.json').write_text(json.dumps({'version':VERSION,'train':train,'holdout':hold},indent=2,allow_nan=False)+'\n')
    report['dataset_sha256']=sha(data/'rainfall_source_training.json')
    (art/'rainfall_baseline_report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'coverage':coverage,'cross_validation':report['cross_validation']},indent=2))
    return 0
def main():
    from scripts.train_physics_informed import run
    run(ROOT)
    return 0

if __name__=='__main__':raise SystemExit(main())
