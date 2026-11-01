#!/usr/bin/env python3
"""Boundary, split-isolation and cold-latency evidence for the v5 request.

Empirical misses and unresolved constitutive assumptions remain failed targets.
Cold timings use fresh in-memory caches and include acquisition plus API payload
construction/serialization, but exclude process startup and HTTP transport.
"""
import asyncio
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from backend.release import portable_payload
from backend.api.serialization import sanitize_json

def isolation():
    import pyarrow.parquet as pq
    train=pq.read_table(ROOT/'data/model_training.parquet').to_pylist()
    cal=pq.read_table(ROOT/'data/conformal_calibration.parquet').to_pylist()
    hold=pq.read_table(ROOT/'data/historical_hindcast_candidates.parquet').to_pylist()
    filtered=pq.read_table(ROOT/'data/historical_hindcast_pinls.parquet').to_pylist()
    groups=[{r['region_block'] for r in rows} for rows in (train,cal,hold)]
    assert all(not groups[i]&groups[j] for i in range(3) for j in range(i+1,3))
    assert not any(r.get('is_synthetic') for r in cal+hold)
    assert all(r['fs_static_prior']>1.20 for r in train+filtered if r['label']==0)
    ids=json.loads((ROOT/'data/event_holdout_manifest.json').read_text())['selected_ids']
    assert [r['event_id'] for r in hold]==ids
    meta=json.loads((ROOT/'data/real_global_landslide_observations.metadata.json').read_text())['event_group_holdout']
    return {'passed':True,'training_rows':len(train),'calibration_rows':len(cal),'holdout_rows':len(hold),
            'blocks_disjoint':True,'original_holdout_ids_unchanged':True,
            'negative_selection':{k:meta[k] for k in ('pinls_training','pinls_holdout')}}

def output_contract(scan):
    from scripts.randomized_global_blind_audit import geometry,cells
    geometry(scan)
    values=cells(scan)
    assert values,'No DEM cells in cold scan'
    for p in values.values():
        lower,upper=p['conformal_interval_90']
        assert 0<=lower<=upper<=1
        assert abs(p['uncertainty_width']-(upper-lower))<1e-12
        assert p['conformal_target']=='binary event label; not a confidence interval for true failure probability'
    top=scan['summary']['kinematics_top_hotspot']
    assert 'cascading_dbi' in top and 'impoundment_status' in top
    if top['dam_height_m'] is None:assert top['cascading_dbi'] is None
    return values

async def cold_and_blue():
    from backend.realtime.global_ingestor import GlobalIngestor
    from backend.realtime.cache import AsyncTTLCache
    from backend.api.scan_service import GlobalScanService
    service=GlobalScanService(provider_timeout_s=12)
    rows=[];blue={};snapshots=[]
    for name,lat,lon in [('Blue Mountains',-33.712,150.311),('Wayanad',11.548,76.134),('New Delhi',28.6139,77.2090)]:
        row={'name':name,'passed':False}
        started=time.perf_counter()
        try:
            async with GlobalIngestor(cache=AsyncTTLCache(),provider_timeout_s=12) as ingestor:
                raw=await ingestor.scan((lat,lon),radius_km=3.5)
            fetched=time.perf_counter()-started
            base=await service.scan(location_query=name,lat=lat,lon=lon,_snapshot=raw)
            base=sanitize_json(base)
            json.dumps(base,allow_nan=False)
            elapsed=time.perf_counter()-started
            values=output_contract(base)
            row.update(passed=True,provider_seconds=fetched,cold_scan_seconds=elapsed,
                       target_lt_2_5=elapsed<2.5,audit_target_lt_3_5=elapsed<3.5,
                       cell_count=len(values),source_status=base['source_status'])
            snapshots.append({'name':name,'snapshot':sanitize_json(raw.to_dict()),'scan':base})
            if name=='Blue Mountains':
                changed=sanitize_json(await service.scan(location_query=name,lat=lat,lon=lon,
                                                        snowmelt_temp_surge_c=8,_snapshot=raw))
                after=output_contract(changed)
                eligible=sum(p['cryospheric_eligible'] for p in after.values())
                delta=max(abs(after[h]['fs_dynamic']-p['fs_dynamic']) for h,p in values.items())
                blue={'passed':eligible==0 and delta<1e-10,
                      'eligible_cells':eligible,'max_abs_fs_change':delta,
                      'critical_fs':changed['summary']['min_fs_dynamic'],
                      'critical_score':changed['summary']['max_p_failure'],
                      'stable_fs_gt_1':changed['summary']['min_fs_dynamic']>1,
                      'same_provider_snapshot':True}
                snapshots[-1]['warming_scan']=changed
        except Exception as exc:
            row.update(error=type(exc).__name__+': '+str(exc),elapsed_seconds=time.perf_counter()-started)
        rows.append(row)
        print('COLD',json.dumps(portable_payload(row)),flush=True)
    (ROOT/'artifacts/literature_boundary_snapshots.json').write_text(json.dumps(portable_payload(snapshots),allow_nan=False))
    return rows,blue

def main():
    report={'generated_at_utc':datetime.now(timezone.utc).isoformat(),'engineering_ready':False,
            'timing_scope':'Fresh in-memory cache; public provider acquisition plus scan payload and JSON serialization; excludes process startup/HTTP transport'}
    try:report['isolation']=isolation()
    except Exception as exc:report['isolation']={'passed':False,'error':str(exc)}
    report['cold_scans'],report['blue_mountains']=asyncio.run(cold_and_blue())
    validation=json.loads((ROOT/'artifacts/scientific_validation_report.json').read_text())
    full=json.loads((ROOT/'artifacts/full_system_bug_audit.json').read_text())
    wayanad=next((r for r in full['scenarios'] if r['name']=='Wayanad'),{})
    report['conformal_holdout']=validation.get('conformal_holdout',{})
    report['pinls_holdout']=validation.get('stage2_pinls_filtered',{})
    report['requested_targets']={
        'blue_mountains_no_warming_created_cryo':report['blue_mountains'].get('passed',False),
        'blue_mountains_critical_fs_gt_1':report['blue_mountains'].get('stable_fs_gt_1',False),
        'stage2_original_auc_gte_0_82':validation['stage2']['metrics']['roc_auc']>=.82,
        'stage2_pinls_auc_gte_0_82':report['pinls_holdout'].get('metrics',{}).get('roc_auc',0)>=.82,
        'wayanad_score_gte_0_80':wayanad.get('p95_failure',0)>=.80,
        'all_cold_scans_lt_3_5_seconds':all(r.get('audit_target_lt_3_5',False) for r in report['cold_scans']),
        'all_cold_scans_lt_2_5_seconds':all(r.get('target_lt_2_5',False) for r in report['cold_scans']),
        'zero_unverified_constitutive_terms':False,
        'observed_geometry_for_live_dbi':False}
    report['unresolved_science']=[
        'Supplied Milledge DOI is Pudasaini 2012. Implemented lateral term is an adaptation of finite-block reasoning, not the full Milledge 2014 model.',
        'FS prior >1.20 is the requested negative selection rule; the cited Do 2026 primary full text was not verified.',
        'Live ice weakening and post-fire yield-index transformation remain explicitly marked scenario proxies, not Mamot or Staley constitutive equations.',
        'RGB vegetation proxy is not measured pre-fire NDVI. Snow floors, width 35 m and hollow factor are scenario assumptions.',
        'Conformal intervals predict binary labels under exchangeable blocks; they are not confidence intervals on true failure probability.',
        'Dam height, deposited dam volume and full upstream catchment are unavailable. DBI is null rather than fabricated.']
    report['engineering_ready']=bool(report['isolation']['passed'] and report['blue_mountains'].get('passed') and all(r['passed'] for r in report['cold_scans']))
    report['all_requested_targets_passed']=all(report['requested_targets'].values())
    from scripts.finalize_readiness import fingerprint
    report['implementation_fingerprint']=fingerprint()
    report=portable_payload(sanitize_json(report))
    (ROOT/'artifacts/literature_boundary_audit.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    lines=['# v5 boundary and literature audit','','| Requested target | Result |','|---|---|']
    lines += ['| {} | {} |'.format(k,'PASS' if v else 'NOT MET') for k,v in report['requested_targets'].items()]
    lines += ['','| Cold scan | Seconds |','|---|---:|']
    lines += ['| {} | {} |'.format(r['name'],r.get('cold_scan_seconds',r.get('error'))) for r in report['cold_scans']]
    lines += ['',*['- '+v for v in report['unresolved_science']]]
    (ROOT/'artifacts/literature_boundary_audit.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(report['requested_targets'],indent=2),flush=True)
    return 0 if report['all_requested_targets_passed'] else 1

if __name__=='__main__':raise SystemExit(main())
