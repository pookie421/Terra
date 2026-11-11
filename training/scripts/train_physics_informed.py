#!/usr/bin/env python3
"""Offline identical-fold four-way ablation after terrain and target correction."""
import argparse,hashlib,json,socket,sys
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import joblib
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.train_rainfall_baseline import prepare
from backend.physics.reference_context import columns
from backend.ml.rainfall_features import feature_vector
from backend.ml.rainfall_baseline import grouped_folds
from backend.ml.physics_informed import ContextModel,KINDS,NAMES,VERSION,matrix
from backend.ml.validation_suite import ranking_metrics
from backend.ml.uncertainty import cluster_intervals
from backend.realtime.dem_processor import TERRAIN_VERSION


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def run(root=ROOT,device='cpu'):
    data=root/'data';art=root/'artifacts'
    from backend.release import invalidate_readiness
    invalidate_readiness(art,'Metric terrain, reference physics and physics-informed fusion require current verification')
    from scripts.rebuild_context_observations import build as build_contexts
    contexts={r['event_id']:r for r in build_contexts(root)}
    source_train,source_hold,coverage=prepare(data)
    exclusions=[]
    def build(rows):
        retained=[]
        for row in rows:
            if row['label_quality']=='mapped_failure_footprint_not_verified_initiation_crown':
                exclusions.append({'event_id':row['event_id'],'label':row['label'],'source_url':row['source_url'],
                    'h3_cell_id':row['h3_cell_id'],'task':'footprint_extent_unresolved_initiation','reason':'No independently mapped crown; retained in separate extent cohort, never relabeled stable'})
                continue
            context=contexts[row['event_id']]
            if context['features'] is None:raise ValueError('Declared training observation lacks complete weather context')
            retained.append({**row,'features':context['features'],'physical':context['physical'],'window':context['window'],
                'terrain_version':TERRAIN_VERSION,'terrain_payload_sha256':context['terrain_payload_sha256'],
                'target':'manual_point_within_300m_initiation_proxy' if row['label']==1 else 'matched_temporal_inventory_absence',
                'original_label_preserved':True})
        return retained
    train=build(source_train);hold=build(source_hold)
    positive_keys={(r['event_group_id'],r['physical']['h3_index']) for r in train if r['label']==1}
    negative_keys={(r['event_group_id'],r['physical']['h3_index']) for r in train if r['label']==0}
    assert positive_keys==negative_keys,'Initiation-proxy cohort must have matching temporal controls'
    x=np.asarray([r['features'] for r in train],float);y=np.array([r['label'] for r in train]);g=np.array([r['component'] for r in train])
    folds=grouped_folds(y,g);oof={k:np.full(len(y),np.nan) for k in KINDS};learned=np.full(len(y),np.nan);audit=[]
    for fold,(a,b) in enumerate(folds):
        for field in ('component','event_group_id','region_block'):
            assert not {train[i][field] for i in a}&{train[i][field] for i in b}
        for kind in KINDS:
            candidate=ContextModel(kind,device).fit(x[a],y[a],g[a])
            oof[kind][b]=candidate.predict_proba(x[b])
            if kind=='Physics-Informed ML':learned[b]=candidate.predict_learned_proba(x[b])
        audit.append({'fold':fold,'train_ids':[train[i]['event_id'] for i in a],'test_ids':[train[i]['event_id'] for i in b],
                      'train_components':sorted(set(g[a])),'test_components':sorted(set(g[b]))})
        print('Ablation fold',fold+1,'of',len(folds),flush=True)
    for field in ('component','event_group_id','region_block'):
        assert not {r[field] for r in train}&{r[field] for r in hold}
    models={k:ContextModel(k,device).fit(x,y,g) for k in KINDS}
    path=art/'physics_informed_models.pkl';temporary=path.with_suffix('.pkl.tmp')
    joblib.dump(models,temporary);temporary.replace(path)
    xx=np.asarray([r['features'] for r in hold],float)
    report={'version':VERSION,'terrain_version':TERRAIN_VERSION,'features':list(NAMES),'network_used':False,
        'primary_model':'Physics-Informed ML','selection':'Scientific thesis declared before evaluating corrected cohort; no holdout winner selection',
        'training_rows':len(train),'holdout_rows':len(hold),'components':len(set(g)),
        'training_labels':dict(Counter(map(int,y))),'excluded_footprint_rows':exclusions,
        'matched_positive_negative_cell_groups':len(positive_keys),
        'observations_sha256':sha(data/'real_global_landslide_observations.parquet'),
        'holdout_manifest_sha256':sha(data/'event_holdout_manifest.json'),
        'coverage_before_target_separation':coverage,'fold_audit':audit,
        'cross_validation':{k:ranking_metrics(y,p) for k,p in oof.items()},
        'uncoupled_feature_fusion_cv':ranking_metrics(y,learned),
        'uncoupled_feature_fusion_oof':list(map(float,learned)),
        'cluster_intervals':{k:cluster_intervals(y,p,g) for k,p in oof.items()},
        'development_holdout':{k:ranking_metrics([r['label'] for r in hold],m.predict_proba(xx)) for k,m in models.items()} if hold else {},
        'oof_predictions':[{'event_id':r['event_id'],'label':int(y[i]),'component':r['component'],'scores':{k:float(p[i]) for k,p in oof.items()}} for i,r in enumerate(train)],
        'model_sha256':sha(path),'limitations':['Manual points within 300 m are initiation proxies, not verified crowns',
            'Footprints without independently mapped crowns separated; no inferred crowns or synthetic stable labels',
            'Matched temporal absences are weather-selected and unverified; no representative background controls',
            'Retrospective event-day rain may follow failure; archive/live shift; inspected holdout remains development evidence',
            'Strength/depth/covariance assumptions are reference priors; output is an experimental sample score, not field-calibrated probability'],
        'independent_predictive_validation':False,'operational_warning_ready':False}
    import platform,sklearn,xgboost
    report['environment']={'python':platform.python_version(),'numpy':np.__version__,'sklearn':sklearn.__version__,'xgboost':xgboost.__version__,'device':device}
    report['fusion_definition']='Fold-prevalence initialization; monotonic source/mechanistic/interaction features, nested grouped regularization and disjoint offset calibration. Conditional score multiplies learned trigger score by exposed gravity-over-cohesion admissibility. Uncoupled metrics are reported separately; no per-row base_margin.'
    report['fusion_training_diagnostics']=[{'calibration_offset':offset,'split_features':tree.get_booster().get_score(importance_type='gain')}
        for tree,offset in models['Physics-Informed ML'].folds]
    report['nested_training_audit']=models['Physics-Informed ML'].tuning
    report['feature_units']={'effective_shear_stress_surge':'kPa; rainfall-induced frictional resistance reduction',
        'pressure_head_m':'m','critical_rain_deficit_mm':'mm; null denotes unattainable at the declared duration',
        'newmark_displacement_cm':'cm; missing outside regression/input domain',
        'gravity_release_admissibility':'Conditional necessary-condition score under declared lognormal moments; not a calibrated event probability'}
    dest=data/'physics_informed_training.json';dest.write_text(json.dumps({'version':VERSION,'train':train,'holdout':hold},indent=2,allow_nan=False)+'\n')
    report['dataset_sha256']=sha(dest)
    (art/'physics_informed_ablation_report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    (art/'scientific_validation_report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps(report['cross_validation'],indent=2));return report

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--device',choices=['cpu','cuda'],default='cpu');a=p.parse_args()
    socket.socket.connect=lambda *a,**k:(_ for _ in ()).throw(RuntimeError('Training is offline'))
    run(device=a.device)
