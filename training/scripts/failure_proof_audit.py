#!/usr/bin/env python3
"""Verify surgical regressions and summarize fresh scenario evidence without refitting targets."""
import json
from pathlib import Path
import subprocess
import sys
from datetime import datetime, timezone
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.finalize_readiness import fingerprint


def main():
    current=fingerprint()
    report={'generated_at_utc':datetime.now(timezone.utc).isoformat(),
            'implementation_fingerprint':current,'checks':{},'requested_targets':{},
            'scope':'Regression checks plus the existing fresh-network scenario audit; no additional acquisition or independent scientific validation.'}
    result=subprocess.run([sys.executable,'-m','unittest','discover','-s','tests','-p','test_surgical_physics.py','-v'],cwd=ROOT,capture_output=True,text=True)
    report['checks']['surgical_regressions']={'passed':result.returncode==0,'output':result.stdout+result.stderr}
    try:
        phase=json.loads((ROOT/'artifacts/phase4_forecast_audit.json').read_text())
        assert phase.get('implementation_fingerprint')==current,'Scenario evidence is stale'
        assert phase.get('engineering_ready'),'Scenario runtime checks failed'
        assert len(phase['rows'])==5 and len(phase['named_fresh_locations'])==4
        for row in phase['rows']:
            for label in ('baseline','perturbed','zero_trigger_control'):
                if label not in row:continue
                for status in row[label]['source_status'].values():
                    assert status['mode']!='CACHED' and 'CACHED' not in status.get('cache_trace',[])
            report['requested_targets'][row['name']]=row['targets']
            if 'zero_trigger_control' in row:
                report['requested_targets'][row['name']+' calm']=row['zero_trigger_control']['targets']
        for row in phase['named_fresh_locations']:
            assert row['no_cached_sources']
            report['requested_targets'][row['query']]=row['targets']
        report['requested_targets']['injection']=phase['forecast_targets']
        report['requested_targets']['categorical_scenarios']=phase.get('lhasa_scenario_benchmark',{}).get('scenario_targets',{'missing':False})
        report['checks']['fresh_scenario_evidence']={'passed':True,'generated_at_utc':phase['generated_at_utc']}
    except (AssertionError,KeyError,OSError,ValueError) as exc:
        report['checks']['fresh_scenario_evidence']={'passed':False,'error':str(exc)}
    report['engineering_ready']=all(c['passed'] for c in report['checks'].values())
    report['all_requested_targets_passed']=report['engineering_ready'] and all(all(v.values()) for v in report['requested_targets'].values())
    out=ROOT/'artifacts';out.mkdir(exist_ok=True)
    (out/'failure_proof_audit.json').write_text(json.dumps(report,indent=2)+'\n')
    lines=['# Surgical physics and scenario audit','',report['scope'],'','| Check | Result |','|---|---|']
    for name,value in report['checks'].items():lines.append('| '+name+' | '+('PASS' if value['passed'] else 'FAIL')+' |')
    lines+=['','| Scenario criterion | Result |','|---|---|']
    for name,targets in report['requested_targets'].items():
        for key,passed in targets.items():lines.append('| '+name+': '+key+' | '+('PASS' if passed else 'NOT MET')+' |')
    (out/'failure_proof_audit.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(report,indent=2))
    return 0 if report['engineering_ready'] else 1


if __name__=='__main__':raise SystemExit(main())
