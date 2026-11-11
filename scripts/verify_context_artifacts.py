"""Re-extract corrected features and verify the four-way model, labels and state traces."""
import hashlib,json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import joblib
from backend.ml.physics_informed import KINDS,VERSION,matrix
from backend.ml.rainfall_features import feature_vector
from backend.ml.validation_suite import ranking_metrics
from backend.physics.reference_context import columns


def verify(root):
    root=Path(root);art=root/'artifacts';data=root/'data'
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    report=json.loads((art/'physics_informed_ablation_report.json').read_text())
    saved=json.loads((data/'physics_informed_training.json').read_text())
    assert report['version']==saved['version']==VERSION
    assert sha(art/'physics_informed_models.pkl')==report['model_sha256']
    assert sha(data/'physics_informed_training.json')==report['dataset_sha256']
    assert sha(data/'real_global_landslide_observations.parquet')==report['observations_sha256']
    assert sha(data/'event_holdout_manifest.json')==report['holdout_manifest_sha256']
    models=joblib.load(art/'physics_informed_models.pkl');assert tuple(models)==KINDS
    terrains={}
    for p in (data/'source_catalogs/terrain_metric_v1').glob('*.json'):
        d=json.loads(p.read_text());terrains[sha(p)]=d
    for row in saved['train']+saved['holdout']:
        assert row['label_quality']!='mapped_failure_footprint_not_verified_initiation_crown'
        terrain=terrains[row['terrain_payload_sha256']]
        cell=next(c for c in terrain['cells'] if c['cell_id']==row['physical']['h3_index'])
        physical=columns([SimpleNamespace(**cell)],row['window'])[0]
        x=matrix(feature_vector(row['window'],cell['slope_deg'],cell['twi'],cell['profile_curvature']),physical)
        np.testing.assert_allclose(x,np.asarray(row['features'],float),rtol=1e-12,atol=1e-12,equal_nan=True)
        np.testing.assert_allclose(physical['fosm_variance'],row['physical']['fosm_variance'],rtol=1e-12)
    full_path=data/'reference_context_observations.jsonl'
    coverage=json.loads((art/'context_observation_coverage.json').read_text())
    assert sha(full_path)==coverage['dataset_sha256']
    full=[json.loads(line) for line in full_path.read_text().splitlines()]
    assert len(full)==coverage['observations']==coverage['reference_columns']
    assert all(r['physical']['provenance_label']=='pedotransfer_reference_column' for r in full)
    train=saved['train'];hold=saved['holdout'];byid={r['event_id']:r for r in train}
    assert len(train)==len(report['oof_predictions'])
    for field in ('component','event_group_id','region_block'):
        assert not {r[field] for r in train}&{r[field] for r in hold}
    scored=[]
    for f in report['fold_audit']:
        a=[byid[k] for k in f['train_ids']];b=[byid[k] for k in f['test_ids']];scored+=f['test_ids']
        for field in ('component','event_group_id','region_block'):
            assert not {r[field] for r in a}&{r[field] for r in b}
    assert sorted(scored)==sorted(byid)
    y=[r['label'] for r in train]
    xx=np.asarray([r['features'] for r in hold],float)
    for kind in KINDS:
        assert ranking_metrics(y,[r['scores'][kind] for r in report['oof_predictions']])==report['cross_validation'][kind]
        assert ranking_metrics([r['label'] for r in hold],models[kind].predict_proba(xx))==report['development_holdout'][kind]
    traces=data/'sft_context_trajectories.jsonl';sft=json.loads((art/'sft_trajectory_report.json').read_text())
    assert sft['fusion_model_sha256']==sha(art/'physics_informed_models.pkl')
    assert sft['fusion_model_version']==VERSION
    assert sha(traces)==sft['sha256']
    from backend.llm.state_copilot import verify_and_render
    from scripts.train_qwen25_agent import validate_chatml
    assert validate_chatml(traces,sft['rows'])==sft['distinct_cell_dates']
    for line in traces.read_text().splitlines():
        item=json.loads(line);evidence=json.loads(next(m['content'] for m in item['messages'] if m['role']=='tool'))
        verify_and_render(json.loads(item['messages'][-1]['content']),evidence)
    return {'passed':True,'training_rows':len(train),'holdout_rows':len(hold),'identical_four_way_folds':len(report['fold_audit']),
        'all_observation_reference_columns':coverage['reference_columns'],
        'sft_traces':sft['rows'],'checks':'Raw metric terrain + weather re-extraction, covariance/features, hashes, target separation, grouping, metrics, and bound-claim SFT parity'}
