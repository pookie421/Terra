#!/usr/bin/env python3
"""Combine engineering audits and disclose the latest mandatory request gates."""
import hashlib
import json
from pathlib import Path
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]

def fingerprint():
    paths = sorted([*ROOT.glob('backend/**/*.py'), *ROOT.glob('scripts/*.py'),
                    *ROOT.glob('tests/*.py'), *ROOT.glob('windows_5070ti_trainer/*.py'),
                    ROOT/'data/model_training.parquet', ROOT/'data/historical_hindcast_candidates.parquet',
                    ROOT/'data/conformal_calibration.parquet', ROOT/'data/historical_hindcast_pinls.parquet',
                    ROOT/'data/event_holdout_manifest.json', ROOT/'IMPORT_ARTIFACTS_MAC.sh', ROOT/'LICENSE', ROOT/'pyproject.toml',
                    ROOT/'artifacts/global_multi_causal_xgb.pkl'])
    digest = hashlib.sha256()
    for path in paths:
        digest.update(str(path.relative_to(ROOT)).encode())
        digest.update(path.read_bytes() if path.exists() else b'MISSING')
    return digest.hexdigest()

def main():
    current = fingerprint()
    reports = {}
    for name in ('full_system_bug_audit', 'randomized_global_blind_audit', 'literature_boundary_audit'):
        path = ROOT/'artifacts'/f'{name}.json'
        reports[name] = json.loads(path.read_text()) if path.exists() else {}
    checks = {name: bool(report.get('engineering_ready') and
              report.get('implementation_fingerprint') == current)
              for name, report in reports.items()}
    engineering_ready = all(checks.values())
    request_passed = bool(reports['literature_boundary_audit'].get('all_requested_targets_passed'))
    ready = engineering_ready and request_passed
    payload = {'generated_at_utc': datetime.now(timezone.utc).isoformat(),
               'implementation_fingerprint': current, 'audits': checks,
               'engineering_ready': engineering_ready, 'pretraining_ready': ready,
               'all_requested_targets_passed': request_passed,
               'scope': 'Engineering invariants and latest requested pre-Phase-4 gates are reported separately. No operational certification.'}
    (ROOT/'artifacts/readiness.json').write_text(json.dumps(payload, indent=2)+'\n')
    path = ROOT/'artifacts/scientific_validation_report.json'
    validation = json.loads(path.read_text())
    validation.update(engineering_ready=engineering_ready, pretraining_ready=ready, combined_audit=payload)
    path.write_text(json.dumps(validation, indent=2)+'\n')
    for name, report in reports.items():
        if not report:
            continue
        report.update(pretraining_ready=ready, combined_engineering_ready=engineering_ready)
        (ROOT/'artifacts'/f'{name}.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(payload, indent=2))
    return 0 if ready else 1

if __name__ == '__main__':
    raise SystemExit(main())
