#!/usr/bin/env python3
"""Record reference engineering verification separately from predictive validity."""
import hashlib
import json
from pathlib import Path
from datetime import datetime,timezone
ROOT=Path(__file__).resolve().parents[1]


def fingerprint():
    paths=[*ROOT.glob('backend/**/*.py'),*ROOT.glob('scripts/*.py'),*ROOT.glob('tests/**/*.py'),
           *ROOT.glob('tests/fixtures/*.json')]
    paths += [ROOT/p for p in ('requirements-training.txt','requirements-qwen.txt','scripts/run_training.bat',
        'data/model_training.parquet','data/conformal_calibration.parquet','data/historical_hindcast_candidates.parquet',
        'data/real_global_landslide_observations.parquet','data/event_holdout_manifest.json','data/sft_600_trajectories.jsonl',
        'artifacts/global_multi_causal_xgb.pkl','artifacts/global_multi_causal_xgb.metadata.json',
        'artifacts/reference_baseline_report.json','artifacts/sft_trajectory_report.json',
        'data/rainfall_source_training.json','data/replays/chooralmala_2024.json',
        'artifacts/rainfall_source_baseline.pkl','artifacts/rainfall_baseline_report.json')]
    paths += [ROOT/p for p in ('artifacts/physics_informed_models.pkl','artifacts/physics_informed_ablation_report.json',
        'artifacts/metric_terrain_rebuild.json','artifacts/context_observation_coverage.json',
        'data/physics_informed_training.json','data/reference_context_observations.jsonl','data/sft_context_trajectories.jsonl')]
    digest=hashlib.sha256()
    for path in sorted(set(paths)):
        digest.update(str(path.relative_to(ROOT)).encode())
        digest.update(path.read_bytes() if path.exists() else b'MISSING')
    return digest.hexdigest()


def main():
    current=fingerprint()
    path=ROOT/'artifacts/reference_verification.json'
    report=json.loads(path.read_text()) if path.exists() else {}
    fresh=report.get('implementation_fingerprint')==current
    checks={'fresh_verification':fresh,
            'equation_tests':bool(report.get('equation_tests',{}).get('passed')),
            'frozen_payload_replays':bool(report.get('frozen_payload_replays',{}).get('passed')),
            'other_deterministic_tests':bool(report.get('other_deterministic_tests',{}).get('passed')),
            'artifact_integrity':bool(report.get('artifact_integrity',{}).get('passed'))}
    ready=all(checks.values())
    payload={'generated_at_utc':datetime.now(timezone.utc).isoformat(),'implementation_fingerprint':current,
        'checks':checks,'engineering_ready':ready,'pretraining_ready':ready,
        'independent_predictive_validation':False,'operational_warning_ready':False,
        'windows_rtx5070ti_verified':False,'qlora_verified':False,
        'live_physics_requires_sourced_ground_inputs':False,'legacy_reference_physics_live_enabled':False,
        'reference_column_priors_live_enabled':True,'physics_informed_fusion_live_enabled':True,
        'conditional_runout_enabled':True,'copilot_state_bound':True,
        'live_numeric_placeholders_allowed':False,'cap_warning_export_enabled':False,
        'scope':'CPU engineering verification for corrected terrain, reference columns, four-way fusion ablation and state-bound co-pilot. No external predictive validation, CUDA certification, or operational warning approval.'}
    (ROOT/'artifacts/readiness.json').write_text(json.dumps(payload,indent=2)+'\n')
    validation_path=ROOT/'artifacts/scientific_validation_report.json'
    validation=json.loads(validation_path.read_text()) if validation_path.exists() else {}
    validation.update(engineering_ready=ready,pretraining_ready=ready,combined_audit=payload)
    validation_path.write_text(json.dumps(validation,indent=2)+'\n')
    print(json.dumps(payload,indent=2))
    return 0 if ready else 1
if __name__=='__main__':raise SystemExit(main())
