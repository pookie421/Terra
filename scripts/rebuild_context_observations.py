"""Recompute physical context for every retained observation without rewriting source labels."""
import hashlib,json,sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pyarrow.parquet as pq
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from backend.physics.reference_context import columns
from backend.ml.physics_informed import matrix
from backend.ml.rainfall_features import feature_vector
from scripts.train_rainfall_baseline import extract


def build(root=ROOT):
    root=Path(root);data=root/'data';terrains={}
    for p in sorted((data/'source_catalogs/terrain_metric_v1').glob('*.json')):
        d=json.loads(p.read_text());terrains[tuple(d['center'])]=(d,hashlib.sha256(p.read_bytes()).hexdigest())
    print('Loaded corrected terrain locations:',len(terrains),flush=True)
    rows=pq.read_table(data/'real_global_landslide_observations.parquet').to_pylist();out=[]
    for index,row in enumerate(rows):
        if index%100==0:print('Rebuilding reference context',index,'of',len(rows),flush=True)
        d,terrain_hash=terrains[(row['latitude'],row['longitude'])]
        cell=min(d['cells'],key=lambda c:(c['cell_id']!=row.get('h3_cell_id'),
            (c['latitude']-row['latitude'])**2+((c['longitude']-row['longitude'])*np.cos(np.radians(row['latitude'])))**2,c['cell_id']))
        c=SimpleNamespace(**cell)
        try:window=extract(row,data,terrains)['window']
        except (ValueError,KeyError,OSError,TypeError):window={'status':'FORCING_PRIOR','rain_24h_mm':0.,'rain_prior_6d_mm':0.,'reason':'Retained hourly forcing incomplete or absent'}
        pga=row.get('pga_attenuation_proxy_g')
        window['pga_reference_g']=float(pga) if pga is not None and pga>0 else None
        physical=columns([c],window)[0]
        try:features=matrix(feature_vector(window,c.slope_deg,c.twi,c.profile_curvature),physical)
        except ValueError:features=None
        footprint='footprint' in str(row.get('label_quality',''))
        out.append({'event_id':row['event_id'],'event_date':row['event_date'],'latitude':row['latitude'],'longitude':row['longitude'],
            'event_group_id':row.get('event_group_id'),'region_block':row.get('region_block'),'original_label':row.get('label'),
            'label_quality':row.get('label_quality'),'source_url':row.get('source_url'),'regime':row.get('regime'),
            'task':'failure_extent_unresolved_initiation' if footprint else 'initiation_proxy_or_temporal_absence' if row.get('label') in (0,1) else 'unlabeled_event_context',
            'window':window,'physical':physical,'terrain_payload_sha256':terrain_hash,
            'features':[float(v) if np.isfinite(v) else None for v in features] if features is not None else None})
    path=data/'reference_context_observations.jsonl';path.write_text(''.join(json.dumps(r,allow_nan=False)+'\n' for r in out))
    report={'observations':len(out),'reference_columns':len(out),'complete_weather_ml_contexts':sum(r['features'] is not None for r in out),
        'source_labels_unchanged':True,'source_sha256':hashlib.sha256((data/'real_global_landslide_observations.parquet').read_bytes()).hexdigest(),
        'dataset_sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
    (root/'artifacts/context_observation_coverage.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report),flush=True);return out

if __name__=='__main__':build()
