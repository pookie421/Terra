#!/usr/bin/env python3
"""Record live and zero-trigger scenario targets without manipulating wetness."""
import asyncio
import json
import hashlib
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from backend.api.scan_service import GlobalScanService, known_location_coordinates
from backend.api.serialization import sanitize_json
from backend.realtime.global_ingestor import GlobalIngestor, GeocodingError, SourceStatus
from scripts.finalize_readiness import fingerprint
from backend.release import portable_payload

CASES=[('Wayanad',11.548,76.134,dict(rain_delta_mm=180)),
       ('Kaikoura',-42.260,173.620,dict(pga_override_g=.35)),
       ('Zermatt',46.020,7.749,dict(snowmelt_temp_surge_c=8)),
       ('Malibu',34.07022,-118.54453,dict(burn_severity_override=.85,rain_delta_mm=65)),
       ('New Delhi',28.6139,77.2090,dict(rain_delta_mm=300))]


def metrics(scan):
    s=scan['summary'];top=s.get('kinematics_top_hotspot') or {}
    return dict(fs=s.get('critical_fs'),score=s.get('critical_p_failure'),
        maximum_cell_score=max((f['properties']['p_failure'] for f in scan['h3_geojson']['features']),default=None),
        velocity=top.get('peak_velocity_ms'),newmark=top.get('newmark_displacement_cm'),
        m1=top.get('p_failure_m1'),cause=s.get('dominant_cause'),
        epistemic_interval_90=top.get('epistemic_interval_90'),
        source_status=scan['source_status'],forecast_status=scan.get('predictive_forecast_72h',{}).get('status'))


def no_cached_sources(scan):
    return all(s.get('mode')!='CACHED' and 'CACHED' not in s.get('cache_trace',[])
               for s in scan['source_status'].values())


def invariants(scan):
    json.dumps(sanitize_json(scan),allow_nan=False)
    if not scan['h3_geojson']['features']:
        assert scan['summary']['wmo_alert_level']=='UNAVAILABLE'
        assert not scan['runout_geojson']['features']
        return
    for f in scan['h3_geojson']['features']:
        p=f['properties'];assert p['shap_cause_pct'] is not None
        for key in ('epistemic_interval_90','conformal_binary_label_interval'):
            lo,hi=p[key];assert 0<=lo<=hi<=1
        assert abs(sum(p['causal_breakdown_pct'].values())-100)<1e-6
    forecast=scan['predictive_forecast_72h']
    if forecast['status']=='PROJECTED_SCENARIO':
        for path in forecast['paths'].values():
            assert [r['hour'] for r in path['hourly']]==list(range(73))
            assert all(0<=r['risk_pct']<=100 and np.isfinite(r['critical_fs']) for r in path['hourly'])
            expected=next((r['hour'] for r in path['hourly'] if r['critical_fs']<1 or r['risk_pct']>=65),None)
            assert expected==path['crossing_hour']


async def main():
    service=GlobalScanService();report={'generated_at_utc':datetime.now(timezone.utc).isoformat(),
        'rows':[],'checks':{},'engineering_ready':False,
        'acquisition_policy':'Fresh network only; TTL cache reads and cache-on-error disabled. Paired scenarios share each freshly fetched snapshot.',
        'acceptance_policy':'Wayanad calm FS>1.35 and score<10%; Zermatt calm FS>1.50 and score<15%; injected storm starts below15% and crosses T+19..30 with FS<.95. Kaikoura FS<.65 and score>=95%; Delhi<.01%; Grand Canyon<15%. Revised scenario criteria, not independent physical validation.',
        'scope':'Preselected research scenario projections; not blind scientific validation or forecast-skill validation'}
    snapshots=[];categorical=[]
    for name,lat,lon,changes in CASES:
        row={'name':name}
        try:
            async with GlobalIngestor(provider_timeout_s=15,use_cache=False,allow_cache_fallback=False) as ingestor:
                raw=await ingestor.scan((lat,lon))
            snapshots.append({'name':name,'snapshot':raw.to_dict()})
            baseline=await service.scan(location_query=name,_snapshot=raw)
            perturbed=await service.scan(location_query=name,_snapshot=raw,**changes)
            invariants(baseline);invariants(perturbed)
            assert no_cached_sources(baseline) and no_cached_sources(perturbed)
            row.update(baseline=metrics(baseline),perturbed=metrics(perturbed),runtime_passed=True)
            p=row['perturbed'];targets={}
            categorical.append({'name':name,'expected_active':name!='New Delhi','score':p['score']})
            if p['fs'] is None:targets={'terrain_available':False}
            elif name=='Wayanad':targets={'fs':p['fs']<.90,'score':p['score']>=.88,'velocity':p['velocity']>10,'cause':p['cause']=='hydro_meteorological'}
            elif name=='Kaikoura':targets={'fs':p['fs']<.65,'score':p['score']>=.95,'newmark':p['newmark']>=10,'velocity':p['velocity']>10,'cause':p['cause']=='seismic_tectonic'}
            elif name=='Zermatt':targets={'fs':p['fs']<.85,'score':p['score']>=.85,'cause':p['cause']=='cryospheric'}
            elif name=='Malibu':targets={'m1':p['m1']>=.90,'score':p['score']>=.90,'velocity':p['velocity']>=5,'cause':p['cause']=='post_wildfire_biotic','forecast':p['forecast_status']=='PROJECTED_SCENARIO'}
            else:targets={'score':p['score']<.0001,'maximum_cell_score_lt_5_percent':p['maximum_cell_score']<.05,'velocity':p['velocity']==0}
            row['targets']=targets
            if name in ('Wayanad','Zermatt') and raw.weather:
                # Explicit zero-trigger counterfactual; never alter moisture or terrain.
                calm=replace(raw,weather=replace(raw.weather,rain_24h_mm=0.,peak_rain_3h_mm=0.,snow_depth_m=0.,freeze_thaw_cycles_7d=0.),
                    seismic=replace(raw.seismic,max_pga_g=0.) if raw.seismic else None,
                    satellite_cells={k:{**v,'burn_severity_proxy':0.} for k,v in raw.satellite_cells.items()})
                scan=await service.scan(location_query=name,_snapshot=calm)
                invariants(scan);c=metrics(scan)
                categorical.append({'name':name+' zero triggers','expected_active':False,'score':c['score']})
                row['zero_trigger_control']={**c,'assumption':'Rain/snow/cycles/PGA/burn set to zero; actual hydrology, temperatures and terrain retained',
                    'targets':{'fs':c['fs'] is not None and c['fs']>(1.35 if name=='Wayanad' else 1.50),'score':c['score'] is not None and c['score']<(.10 if name=='Wayanad' else .15)}}
            if name=='Wayanad':
                toe_raw=replace(raw,river={**(raw.river or {}),'surge_ratio':5.})
                toe=await service.scan(location_query=name,_snapshot=toe_raw)
                report['toe_scenario']=metrics(toe)
                categorical.append({'name':'Wayanad river-surge scenario','expected_active':True,'score':metrics(toe)['score']})
                storm=await service.scan(location_query=name,_snapshot=raw,inject_forecast_storm_mm=220,storm_arrival_hour=18,storm_duration_hours=12)
                invariants(storm);f=storm['predictive_forecast_72h']
                report['injected_forecast']=f
                crossing=f.get('predicted_threshold_crossing_hour')
                risk_crossing=next((r['hour'] for r in f.get('paths',{}).get('p50',{}).get('hourly',[]) if r['risk_pct']>=65),None)
                report['injected_risk_65_crossing_hour']=risk_crossing
                joint_crossing=next((r['hour'] for r in f.get('paths',{}).get('p50',{}).get('hourly',[]) if r['risk_pct']>=65 and r['critical_fs']<.95),None)
                report['injected_joint_risk_fs_crossing_hour']=joint_crossing
                report['forecast_targets']={'available':f['status']=='PROJECTED_SCENARIO',
                    'calm_t0':f.get('current_risk_t0_pct',100)<15,
                    'risk_crossing_19_to_30':risk_crossing is not None and 19<=risk_crossing<=30,
                    'joint_risk_fs_crossing_19_to_30':joint_crossing is not None and 19<=joint_crossing<=30}
        except Exception as exc:
            row.update(runtime_passed=False,error=type(exc).__name__+': '+str(exc))
        report['rows'].append(row)
        print(name,json.dumps({k:v for k,v in row.items() if k in ('targets','runtime_passed','error')}),flush=True)
    report['named_fresh_locations']=[]
    for query,limit in [('Grand Canyon, Arizona',.15),('Ooty, Tamil Nadu',None),
                        ('Palakkad, Kerala',.05),('Chooralmala, Wayanad',None)]:
        row={'query':query}
        try:
            async with GlobalIngestor(provider_timeout_s=15,use_cache=False,allow_cache_fallback=False) as ingestor:
                try:
                    raw=await ingestor.scan(query)
                except GeocodingError as exc:
                    coordinates=known_location_coordinates(query)
                    if coordinates is None:raise
                    raw=await ingestor.scan(coordinates)
                    raw.sources['geocoding']=SourceStatus(available=False,provider='Project coordinate fallback',
                        fetched_at_utc=datetime.now(timezone.utc).isoformat(),mode='FALLBACK',error=str(exc),cache_policy='LIVE_ONLY')
            snapshots.append({'name':query,'snapshot':raw.to_dict()})
            scan=await service.scan(location_query=query,_snapshot=raw)
            invariants(scan)
            assert no_cached_sources(scan)
            location=dict(scan['location']);location['nasa_telemetry']=dict(location.get('nasa_telemetry',{}))
            events=location['nasa_telemetry'].get('nasa_eonet_active_events')
            if isinstance(events,list):
                location['nasa_telemetry']['nasa_eonet_active_events']={'count':len(events),'sha256':hashlib.sha256(json.dumps(events,sort_keys=True).encode()).hexdigest(),'records_location':'phase4_provider_snapshots.json'}
            row.update(metrics(scan),location=location,runtime_passed=True,no_cached_sources=True)
            if query in ('Grand Canyon, Arizona','Palakkad, Kerala'):
                categorical.append({'name':query,'expected_active':False,'score':row['score']})
            row['targets']={} if limit is None else {'score':row['score'] is not None and row['score']<limit}
            if query=='Chooralmala, Wayanad':
                row['targets']={'score_lt_25_percent':row['score'] is not None and row['score']<.25,'fs_gt_1_25':row['fs'] is not None and row['fs']>1.25}
        except Exception as exc:
            row.update(runtime_passed=False,error=type(exc).__name__+': '+str(exc))
        report['named_fresh_locations'].append(row)
        print(query,json.dumps({k:v for k,v in row.items() if k in ('score','fs','targets','runtime_passed','error')}),flush=True)
    report['engineering_ready']=all(r.get('runtime_passed',False) for r in report['rows']+report['named_fresh_locations'])
    report['checks']['fresh_named_locations_completed']=all(r.get('runtime_passed',False) for r in report['named_fresh_locations'])
    report['all_requested_targets_passed']=report['engineering_ready'] and all(all(r.get('targets',{}).values()) and all(r.get('zero_trigger_control',{}).get('targets',{}).values()) for r in report['rows']) and all(report.get('forecast_targets',{'missing':False}).values())
    report['all_requested_targets_passed'] &= report['checks']['fresh_named_locations_completed'] and all(all(r.get('targets',{}).values()) for r in report['named_fresh_locations'])
    from backend.ml.validation_suite import run_lhasa_benchmark_suite
    report['lhasa_scenario_benchmark']=run_lhasa_benchmark_suite(scenarios=categorical) if len(categorical)==10 else {'scenario_targets':{'complete':False},'available_rows':len(categorical)}
    report['all_requested_targets_passed'] &= all(report['lhasa_scenario_benchmark']['scenario_targets'].values())
    report['implementation_fingerprint']=fingerprint()
    out=ROOT/'artifacts';(out/'phase4_provider_snapshots.json').write_text(json.dumps(portable_payload(sanitize_json(snapshots))))
    (out/'phase4_forecast_audit.json').write_text(json.dumps(portable_payload(sanitize_json(report)),indent=2)+'\n')
    print('ENGINEERING',report['engineering_ready'],'REQUESTED_TARGETS',report['all_requested_targets_passed'])

if __name__=='__main__':asyncio.run(main())
