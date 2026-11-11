"""Global multi-regime proxy CV and unlabeled storm-grid consistency diagnostics.

Neighboring cells are NEVER fabricated stable labels. Inventory/extent proxy CV
is a different task from verified initiation, and all omissions remain visible.
"""
import json,socket,sys
from pathlib import Path
from collections import Counter
import numpy as np
import joblib
import pyarrow.parquet as pq
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from backend.ml.global_xgb import row_component_groups
from backend.ml.physics_informed import ContextModel,KINDS,VERSION
from backend.ml.rainfall_baseline import grouped_folds
from backend.ml.validation_suite import ranking_metrics


def run(root=ROOT,device='cpu'):
    data=root/'data';art=root/'artifacts'
    full=[json.loads(s) for s in (data/'reference_context_observations.jsonl').read_text().splitlines()]
    contexts={r['event_id']:r for r in full}
    source=[r for r in pq.read_table(data/'real_global_landslide_observations.parquet').to_pylist() if r.get('label') in (0,1) and not r.get('is_synthetic')]
    groups=row_component_groups(source);manifest=json.loads((data/'event_holdout_manifest.json').read_text())
    held=set(manifest['heldout_groups']+manifest['legacy_reserved_groups'])
    reserve=pq.read_table(data/'conformal_calibration.parquet').to_pylist();events={r['event_group_id'] for r in reserve};regions={r['region_block'] for r in reserve}
    blocked={g for r,g in zip(source,groups) if r['event_group_id'] in held|events or r['region_block'] in regions}
    rows=[];omitted=[]
    for r,g in zip(source,groups):
        c=contexts[r['event_id']]
        if g in blocked:omitted.append({'event_id':r['event_id'],'reason':'Full original holdout/legacy/calibration component embargo'});continue
        if c['features'] is None:omitted.append({'event_id':r['event_id'],'reason':'Incomplete source weather for learned models'});continue
        rows.append({**c,'label':r['label'],'component':str(g)})
    x=np.asarray([r['features'] for r in rows],float);y=np.asarray([r['label'] for r in rows]);g=np.asarray([r['component'] for r in rows])
    scores={k:np.full(len(y),np.nan) for k in KINDS};learned=np.full(len(y),np.nan);audits=[]
    for fold,(a,b) in enumerate(grouped_folds(y,g)):
        for field in ('component','region_block','event_group_id'):assert not {rows[i][field] for i in a}&{rows[i][field] for i in b}
        for kind in KINDS:
            model=ContextModel(kind,device).fit(x[a],y[a],g[a])
            scores[kind][b]=model.predict_proba(x[b])
            if kind=='Physics-Informed ML':learned[b]=model.predict_learned_proba(x[b])
        audits.append({'fold':fold,'train_ids':[rows[i]['event_id'] for i in a],'test_ids':[rows[i]['event_id'] for i in b]})
        print('Global proxy fold',fold+1,flush=True)
    report={'version':VERSION,'task':'Reported-failure/extent vs inventory absence; not verified initiation or prospective warning',
        'full_context_rows':len(full),'coverage_by_regime':dict(Counter(r['regime'] for r in full)),
        'fit_rows':len(rows),'components':len(set(g)),'omissions':omitted,'fold_audit':audits,
        'cross_validation':{k:ranking_metrics(y,p) for k,p in scores.items()},
        'uncoupled_feature_fusion_cv':ranking_metrics(y,learned),
        'by_regime':{regime:{'rows':sum(r['regime']==regime for r in rows),'metrics':{k:ranking_metrics(y[[r['regime']==regime for r in rows]],p[[r['regime']==regime for r in rows]]) for k,p in scores.items()} if len(set(y[[r['regime']==regime for r in rows]]))==2 else None} for regime in sorted(set(r['regime'] for r in rows))},
        'predictions':[{'event_id':r['event_id'],'label':r['label'],'regime':r['regime'],'scores':{k:float(v[i]) for k,v in scores.items()}} for i,r in enumerate(rows)],
        'limitations':['All 1611 physical columns retained, but incomplete weather is excluded from ML with explicit IDs',
            'Current column is a regolith reference; other mechanisms retain independent applicability requirements',
            'Rainfall-trained feature set includes missing Newmark flags, not invented shaking or burn measurements',
            'Extent/report labels are not crowns; absence controls are weather-selected; this is a secondary development task',
            'No neighborhood cell is assigned a non-failure label; spatial storm sweeps are unlabeled consistency diagnostics'],
        'holdout_membership_changed':False,'operational_warning_ready':False}
    (art/'global_context_benchmark.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');print(json.dumps({k:(v['roc_auc'],v['brier_score']) for k,v in report['cross_validation'].items()}));return report
if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--device',choices=('cpu','cuda'),default='cpu');a=p.parse_args()
    socket.socket.connect=lambda *a,**k:(_ for _ in ()).throw(RuntimeError('Offline benchmark'))
    run(device=a.device)
