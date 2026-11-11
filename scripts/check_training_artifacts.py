#!/usr/bin/env python3
"""Verify equations, frozen integration, and regenerated artifacts without network."""
import hashlib
import io
import json
import socket
import sys
import unittest
from datetime import datetime,timezone
from pathlib import Path
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rainfall_artifact_checks(data,art):
    import joblib
    import numpy as np
    from backend.ml.rainfall_features import FEATURES,VERSION
    from backend.ml.rainfall_baseline import PRIMARY,grouped_folds
    from backend.ml.validation_suite import ranking_metrics
    from scripts.train_rainfall_baseline import prepare,matrix,evaluate_subset
    report=json.loads((art/'rainfall_baseline_report.json').read_text())
    saved=json.loads((data/'rainfall_source_training.json').read_text())
    assert report['feature_version']==saved['version']==VERSION
    assert report['model_sha256']==sha(art/'rainfall_source_baseline.pkl')
    assert report['dataset_sha256']==sha(data/'rainfall_source_training.json')
    assert report['observations_sha256']==sha(data/'real_global_landslide_observations.parquet')
    # Re-extract from retained hourly weather and terrain, including full-graph
    # holdout/calibration embargo before feature availability filtering.
    train,hold,coverage=prepare(data)
    assert train==saved['train'] and hold==saved['holdout']
    assert coverage==report['coverage']
    model=joblib.load(art/'rainfall_source_baseline.pkl')
    assert model.feature_version==VERSION and model.kind==report['primary_model']==PRIMARY
    assert tuple(model.feature_names)==FEATURES
    y=np.array([r['label'] for r in train]);groups=np.array([r['component'] for r in train])
    predictions=report['oof_predictions']
    assert [(r['event_id'],r['label']) for r in train]==[(r['event_id'],r['label']) for r in predictions]
    scores={key:np.array([r['scores'][key] for r in predictions]) for key in report['cross_validation']}
    assert evaluate_subset(train,scores)==report['cross_validation']
    assert len(report['fold_audit'])==len(grouped_folds(y,groups))
    for (a,b),audit in zip(grouped_folds(y,groups),report['fold_audit']):
        assert audit['train_ids']==[train[i]['event_id'] for i in a]
        assert audit['test_ids']==[train[i]['event_id'] for i in b]
        assert audit['train_components']==sorted(set(groups[a]))
        assert audit['test_components']==sorted(set(groups[b]))
        np.testing.assert_allclose(scores['prevalence_only'][b],float(y[a].mean()),rtol=0,atol=0)
        for key in ('event_group_id','region_block','component'):
            assert not {train[i][key] for i in a}&{train[i][key] for i in b}
    for key in ('event_group_id','region_block','component'):
        assert not {r[key] for r in train}&{r[key] for r in hold}
    common=np.array([r['physics_available'] for r in train])
    subset=[r for r in train if r['physics_available']]
    common_scores={k:v[common] for k,v in scores.items()}
    common_scores['fosm_diagnostic']=np.array([r['p_failure_physical'] for r in subset])
    assert evaluate_subset(subset,common_scores)==report['common_physics_subset']['metrics']
    assert ranking_metrics([r['label'] for r in hold],model.predict_proba(matrix(hold)))==report['original_rainfall_development_holdout']['metrics']
    assert report['independent_validation'] is False and report['operational_warning_ready'] is False
    return {'training_rows':len(train),'components':len(set(groups)),
        'recovered_rows_without_physics':coverage['recovered_training_rows_without_physics'],
        'development_holdout_rows':len(hold),'feature_version':VERSION,
        'scope':'Raw source re-extraction, source/model hashes, full-graph embargo, fold membership, baseline and reported metric parity'}


def archived_v14_artifact_checks():
    import joblib
    import numpy as np
    import pyarrow.parquet as pq
    from backend.ml.global_xgb import FEATURE_NAMES,PHYSICS_MODEL_VERSION,row_component_groups,spatial_group_splits
    from backend.ml.validation_suite import ranking_metrics
    from backend.physics.multi_causal_engine import PhysicsInputs,MultiCausalEngine
    from backend.physics.features import physics_feature_matrix
    from scripts.train_qwen25_agent import validate_chatml
    data=ROOT/'data';art=ROOT/'artifacts'
    train=pq.read_table(data/'model_training.parquet').to_pylist()
    hold=pq.read_table(data/'historical_hindcast_candidates.parquet').to_pylist()
    calibration=pq.read_table(data/'conformal_calibration.parquet').to_pylist()
    observations=pq.read_table(data/'real_global_landslide_observations.parquet').to_pylist()
    assert train and hold and calibration
    assert not any(r.get('is_synthetic') for r in observations)
    source=pq.read_table(data/'benchmark_history/v12/real_global_landslide_observations.parquet').to_pylist()
    expected={r['event_id']:(r['event_date'],r.get('label')) for r in source if not r.get('is_synthetic')}
    assert expected=={r['event_id']:(r['event_date'],r.get('label')) for r in observations}
    manifest=json.loads((data/'event_holdout_manifest.json').read_text())
    assert [r['event_id'] for r in hold]==manifest['selected_ids']
    for left,right in ((train,hold),(train,calibration),(calibration,hold)):
        for key in ('event_group_id','region_block'):
            assert not {r[key] for r in left}&{r[key] for r in right},key
    model=joblib.load(art/'global_multi_causal_xgb.pkl')
    assert model.physics_model_version_==PHYSICS_MODEL_VERSION and model.field_prevalence is None
    report=json.loads((art/'reference_baseline_report.json').read_text())
    assert report['model_sha256']==sha(art/'global_multi_causal_xgb.pkl')
    assert report['training_sha256']==sha(data/'model_training.parquet')
    x=np.array([[r[k] for k in FEATURE_NAMES] for r in train],dtype=np.float32)
    labels=np.array([r['label'] for r in train]);groups=row_component_groups(train)
    engine=MultiCausalEngine()
    for row,expected_features in zip(train,x):
        assert row['physics_model_version']==PHYSICS_MODEL_VERSION
        inputs=PhysicsInputs(**json.loads(row['physics_inputs_json']))
        actual=physics_feature_matrix(inputs,engine.compute(inputs),np.atleast_1d(inputs.profile_curvature))[0]
        np.testing.assert_allclose(actual,expected_features,rtol=1e-5,atol=1e-6)
    folds=spatial_group_splits(labels,groups,len(report['cross_validation']['fold_audit']))
    for (a,b),audit in zip(folds,report['cross_validation']['fold_audit']):
        assert sorted(set(groups[a]))==audit['train_components']
        assert sorted(set(groups[b]))==audit['test_components']
        for key in ('event_group_id','region_block'):
            assert not {train[i][key] for i in a}&{train[i][key] for i in b}
    predictions=report['cross_validation']['predictions']
    assert [r['event_id'] for r in train]==[r['event_id'] for r in predictions]
    recomputed=ranking_metrics(labels,[r['p_xgb_calibrated'] for r in predictions])
    assert recomputed==report['cross_validation']['p_xgb_calibrated']
    sft=data/'sft_600_trajectories.jsonl'
    assert validate_chatml(sft)==600
    sft_report=json.loads((art/'sft_trajectory_report.json').read_text())
    assert sft_report['sha256']==sha(sft)
    assert sft_report['model_sha256']==sha(art/'global_multi_causal_xgb.pkl')
    assert sha(sft)==sha(data/'chatml_trajectories.jsonl')
    by_id={r['event_id']:r for r in observations}
    for line in sft.read_text().splitlines():
        record=json.loads(line);row=by_id[record['source_event_id']]
        assert row['event_group_id'] not in set(manifest['heldout_groups']) | set(manifest.get('legacy_reserved_groups',[]))
        telemetry=json.loads(next(m['content'] for m in record['messages'] if m['role']=='tool'))
        assert telemetry['p_combined'] is None
        if telemetry['physics_status']=='UNAVAILABLE':
            assert telemetry['p_xgb_calibrated'] is None and telemetry['p_failure_physical'] is None
        else:
            probability=model.predict_proba(np.array([[row[k] for k in FEATURE_NAMES]],np.float32))[0]
            np.testing.assert_allclose(telemetry['p_xgb_calibrated'],probability,rtol=1e-7,atol=1e-8)
    # Exercise the deployed score path as well as the physics-only fixture tests.
    import asyncio
    from backend.api.scan_service import GlobalScanService
    from backend.realtime.global_ingestor import GlobalScan
    from scripts.snapshot_replay import hydrate
    snapshot=hydrate(GlobalScan,json.loads((ROOT/'tests/fixtures/frozen_scan.json').read_text())['snapshot'])
    service=GlobalScanService();service.model=model
    response=asyncio.run(service.scan(_snapshot=snapshot,_reference_assumptions=True))
    assert response['assessment_mode']=='ARCHIVED_REFERENCE_ASSUMPTIONS_NOT_LIVE_ASSESSMENT'
    json.dumps(response,allow_nan=False)
    cells={f['properties']['h3_index']:f['properties'] for f in response['h3_geojson']['features']}
    assert cells
    for cell in cells.values():
        assert cell['p_failure']==cell['p_xgb_calibrated'] and cell['p_combined'] is None
    summary=response['summary'];hotspot=cells[summary['hotspot_h3_index']]
    assert summary['hotspot_probability']==hotspot['p_xgb_calibrated']
    assert summary['hotspot_fs_dynamic']==hotspot['fs_dynamic']
    # Exercise the current public path with retained real Monaco evidence.
    monaco=hydrate(GlobalScan,json.loads((ROOT/'tests/fixtures/monaco_provider_excerpt.json').read_text())['snapshot'])
    current=asyncio.run(service.scan(_snapshot=monaco))
    assert current['assessment_mode']=='SOURCE_EVIDENCE_AND_EXPLICIT_GROUND_INPUTS'
    assert current['physical_assessment']['status']=='UNAVAILABLE'
    assert current['statistical_assessment']['status']=='EXPERIMENTAL'
    from backend.ml.rainfall_features import FEATURES as RAIN_FEATURES
    for cell in current['statistical_assessment']['cells']:
        vector=np.array([[cell['features'][name] for name in RAIN_FEATURES]])
        np.testing.assert_allclose(service.rainfall_model.predict_proba(vector)[0],cell['score'],rtol=1e-12)
        assert cell['model_explanation']
    assert all(c['fs_dynamic'] is None and c['p_failure_physical'] is None for c in current['cell_comparisons'])
    assert current['summary']['critical_hotspots_count'] is None
    json.dumps(current,allow_nan=False)
    return {'passed':True,'training_rows':len(train),'original_holdout_rows':len(hold),
        'calibration_rows':len(calibration),'preserved_source_rows':len(observations),'sft_rows':600,
        'archived_reference_replay_with_fitted_model_cells':len(cells),
        'current_public_monaco_replay_cells':len(current['terrain_geojson']['features']),
        'rainfall_source_model':rainfall_artifact_checks(data,art),
        'scope':'Feature/model/report parity, original IDs/dates/labels, event/spatial isolation, SFT provenance and hashes'}


def artifact_checks():
    from scripts.verify_context_artifacts import verify
    return verify(ROOT)


def flatten(suite):
    for test in suite:
        if isinstance(test,unittest.TestSuite):yield from flatten(test)
        else:yield test


def run_tests(tests):
    stream=io.StringIO()
    result=unittest.TextTestRunner(stream=stream,verbosity=1).run(unittest.TestSuite(tests))
    if not result.wasSuccessful():print(stream.getvalue())
    return {'passed':result.wasSuccessful(),'tests_run':result.testsRun,'skipped':len(result.skipped),
            'failures':[{'test':test.id(),'traceback':trace} for test,trace in result.failures+result.errors]}


def main():
    from scripts.finalize_readiness import fingerprint
    suites={'equation_tests':[],'frozen_payload_replays':[],'other_deterministic_tests':[]}
    for test in flatten(unittest.defaultTestLoader.discover(str(ROOT/'tests'))):
        key='equation_tests' if test.id().startswith('test_reference_equations.') else (
            'frozen_payload_replays' if '.FrozenScan.' in test.id() else 'other_deterministic_tests')
        suites[key].append(test)
    report={'generated_at_utc':datetime.now(timezone.utc).isoformat(),'network_disabled':True}
    with patch.object(socket.socket,'connect',side_effect=AssertionError('No network in reference verification')),patch.object(socket,'create_connection',side_effect=AssertionError('No network in reference verification')):
        for name,suite in suites.items():report[name]=run_tests(suite)
        try:report['artifact_integrity']=artifact_checks()
        except Exception as exc:
            import traceback
            report['artifact_integrity']={'passed':False,'error':str(exc),'traceback':traceback.format_exc()}
    live_path=ROOT/'artifacts/live_source_provider_check.json'
    report['live_provider_availability']={'status':'recorded_separately' if live_path.exists() else 'not_run',
        'report':live_path.name if live_path.exists() else None,
        'scope':'Separate timestamped live check; never evidence of predictive skill or part of deterministic pass/fail'}
    report['external_solver_comparison']={'status':'not_run','scope':'Analytical equation checks are not a TRIGRS executable benchmark'}
    report['implementation_fingerprint']=fingerprint()
    report['passed']=all(report[k]['passed'] for k in (*suites,'artifact_integrity'))
    (ROOT/'artifacts/reference_verification.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))
    return 0 if report['passed'] else 1
if __name__=='__main__':raise SystemExit(main())
