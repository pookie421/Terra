#!/usr/bin/env python3
"""Randomized scenario stress tests; not an independent labeled global benchmark.

Each pair uses one frozen live-provider snapshot so weather drift cannot masquerade
as a scenario response. Both seeds, selections, missing data and failures persist.
"""
from __future__ import annotations
import argparse,asyncio,hashlib,json,math,secrets,sys,time
from pathlib import Path
from datetime import datetime,timezone
from unittest.mock import patch,AsyncMock
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from backend.realtime.global_ingestor import GlobalIngestor
from backend.api.serialization import sanitize_json
from backend.release import portable_payload
POOL=[
 (1,'Wayanad',11.548,76.134),(1,'Munnar',10.088,77.059),(1,'Darjeeling',27.041,88.266),(1,'Mocoa',1.149,-76.646),(1,'Petropolis',-22.505,-43.178),(1,'Cameron Highlands',4.472,101.380),
 (2,'Kaikoura',-42.260,173.620),(2,'Langtang',28.214,85.505),(2,'Taroko',24.162,121.495),(2,'Atsuma',42.766,141.928),(2,'Banos',-1.396,-78.424),
 (3,'Zermatt',46.020,7.749),(3,'Bondo',46.295,9.602),(3,'Rohtang',32.371,77.246),(3,'Chamonix',45.923,6.869),(3,'Aoraki',-43.734,170.096),
 (4,'Malibu Canyon',34.050,-118.680),(4,'Montecito',34.448,-119.631),(4,'Glenwood Canyon',39.563,-107.224),(4,'Blue Mountains',-33.712,150.311),(4,'Penteli',38.055,23.868),
 (5,'Ramban',33.245,75.238),(5,'Chuquicamata',-22.290,-68.902),(5,'Geirangerfjord',62.101,7.205),(5,"Devil's Slide",37.577,-122.515),
 (6,'New Delhi',28.613,77.209),(6,'Kansas',38.500,-98.),(6,'Amsterdam',52.367,4.904),(6,'Riyadh',24.713,46.675),(6,'Manaus',-3.119,-60.021)]

def selection(seed):
 rng=np.random.default_rng(seed);selected=[]
 for a in range(1,7):
  candidates=[i for i,x in enumerate(POOL) if x[0]==a]
  selected.extend(int(i) for i in rng.choice(candidates,2,replace=False))
 return selected

def perturbation(rng, archetype, fuel_index):
 kinds=[0,1]
 if archetype==3:kinds.append(2)
 if archetype in (1,4) and fuel_index>=.20:kinds.append(3)
 kind=int(rng.choice(kinds))
 if kind==0:return dict(rain_delta_mm=float(rng.uniform(100,350)))
 if kind==1:return dict(pga_override_g=float(rng.uniform(.2,.6)))
 if kind==2:return dict(snowmelt_temp_surge_c=float(rng.uniform(4,12)))
 return dict(burn_severity_override=float(rng.uniform(.6,.95)),rain_delta_mm=float(rng.uniform(40,100)))

def finite(value):
 if isinstance(value,float):assert math.isfinite(value)
 elif isinstance(value,dict):
  for v in value.values():finite(v)
 elif isinstance(value,list):
  for v in value:finite(v)

def geometry(scan):
 from tests.geojson_helpers import check_geojson
 import h3
 check_geojson(scan);finite(scan)
 assert 'reliability_report' in scan['location']
 assert scan.get('serialization_quality',{}).get('nonfinite_replacement_count',0)==0
 for f in scan['h3_geojson']['features']:
  ring=[[lon,lat] for lat,lon in h3.cell_to_boundary(f['properties']['h3_index'])];ring.append(ring[0])
  np.testing.assert_allclose(f['geometry']['coordinates'][0],ring,atol=1e-9)

def cells(scan):return {f['properties']['h3_index']:f['properties'] for f in scan['h3_geojson']['features']}

def compare(base,changed):
 a,b=cells(base),cells(changed);assert set(a)==set(b),'Terrain support changed'
 deltas={h:b[h]['p_failure']-a[h]['p_failure'] for h in a}
 negative=[{'h3_index':h,'delta':d} for h,d in deltas.items() if d < -1e-6]
 contradictions=[]
 for p in [*a.values(),*b.values()]:
  stable=p['fs_dynamic']>1.25 and p['p_failure_m1']<.5
  if (stable and (p['peak_velocity_ms']>=.016/(365.25*86400) or p['cruden_varnes_class']!=1)) or (p['peak_velocity_ms']>.05 and p['fs_dynamic']>1.05 and p['p_failure_m1']<.5):contradictions.append(p['h3_index'])
 return {'monotonic_violations':len(negative),'min_delta':min(deltas.values()),'negative_examples':negative[:5],'kinematic_contradictions':len(contradictions)}

async def acquire(indices,timeout):
 sem=asyncio.Semaphore(3)
 async def one(i):
  async with sem:
   start=time.perf_counter()
   try:
    async with GlobalIngestor(provider_timeout_s=timeout,use_cache=False,allow_cache_fallback=False) as ingestor:
     scan=await ingestor.scan((POOL[i][2],POOL[i][3]),radius_km=3.5)
    return i,(scan,time.perf_counter()-start)
   except Exception as e:return i,(None,str(e))
 return dict(await asyncio.gather(*(one(i) for i in indices)))

def main():
 parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--seed',type=int,default=20261031);parser.add_argument('--second-seed',type=int,help='Repeat the recorded second selection while fetching fresh inputs');parser.add_argument('--provider-timeout',type=float,default=20);parser.add_argument('--replay-report',type=Path);parser.add_argument('--replay-snapshots',type=Path);args=parser.parse_args()
 if bool(args.replay_report)!=bool(args.replay_snapshots):parser.error('Both replay paths are required')
 previous=json.loads(args.replay_report.read_text()) if args.replay_report else None
 fresh=args.second_seed if args.second_seed is not None else secrets.randbits(32);seeds=previous['seeds'] if previous else [args.seed,fresh]
 selections=[selection(seed) for seed in seeds]
 report={'generated_at_utc':datetime.now(timezone.utc).isoformat(),'seeds':seeds,'selections':[[POOL[i][1] for i in s] for s in selections],
  'pool_size':len(POOL),'protocol':'Two stratified 12-location draws; all five named flat controls added each sweep. Paired perturbations share frozen live telemetry. Randomness does not imply independent observational validation.',
  'rows':[],'flat_controls':[],'causal_controls':[],'engineering_ready':False}
 out=ROOT/'artifacts/randomized_global_blind_audit.json';out.write_text(json.dumps(report,indent=2))
 indices=sorted(set(sum(selections,[])+[i for i,x in enumerate(POOL) if x[0]==6]))
 print('Frozen selections:',report['selections'],flush=True)
 if previous:
  from scripts.snapshot_replay import hydrate
  from backend.realtime.global_ingestor import GlobalScan
  raw=[json.loads(line) for line in args.replay_snapshots.read_text().splitlines()]
  latencies={r['name']:r['live_fetch_seconds'] for r in previous['rows']}
  snapshots={r['pool_index']:(hydrate(GlobalScan,r['snapshot']),latencies[POOL[r['pool_index']][1]]) for r in raw}
  assert set(indices)<=set(snapshots),'Incomplete frozen snapshot support'
  report['replay_of_generated_at_utc']=previous['generated_at_utc']
  report['acquisition_mode']='Replay of saved real-provider inputs; no fresh acquisition'
 else:
  snapshots=asyncio.run(acquire(indices,args.provider_timeout))
  report['acquisition_mode']='Live provider acquisition'
 # Retain source inputs locally for forensic replay; raw provider data is Git-ignored.
 with (ROOT/'artifacts/randomized_provider_snapshots.jsonl').open('w') as handle:
  for i,(snapshot,_) in snapshots.items():
   if snapshot is not None:handle.write(json.dumps(sanitize_json({'pool_index':i,'snapshot':snapshot.to_dict()}),allow_nan=False)+'\n')
 from fastapi.testclient import TestClient
 from backend.api.main import app,agent
 from scripts.full_system_bug_audit import feature_contract
 expected={1:('hydro_meteorological',{'rain_delta_mm':180}),2:('seismic_tectonic',{'pga_override_g':.4}),3:('cryospheric',{'snowmelt_temp_surge_c':8}),4:('post_wildfire_biotic',{'burn_severity_override':.85,'rain_delta_mm':65})}
 with TestClient(app) as client:
  for sweep,(seed,selected) in enumerate(zip(seeds,selections)):
   rng=np.random.default_rng(seed+1)
   for i in sorted(set(selected+[j for j,x in enumerate(POOL) if x[0]==6])):
    archetype,name,lat,lon=POOL[i];snapshot,latency=snapshots[i]
    row={'sweep':sweep,'name':name,'archetype':archetype,'random_draw':i in selected,'live_fetch_seconds':latency,'passed':False}
    try:
     assert snapshot is not None,'Live provider acquisition failed'
     row['source_status']={k:v.__dict__ for k,v in snapshot.sources.items()}
     row['snapshot_time_utc']=snapshot.generated_at_utc
     row['snapshot_sha256']=hashlib.sha256(json.dumps(sanitize_json(snapshot.to_dict()),sort_keys=True).encode()).hexdigest()
     with patch.object(GlobalIngestor,'scan',AsyncMock(return_value=snapshot)),patch.object(agent,'_available_model',AsyncMock(return_value=(None,3))):
      params={'query':name,'lat':lat,'lon':lon}
      def scan(scenario):
       response=client.get('/api/v1/scan',params={**params,**scenario});response.raise_for_status();payload=response.json();finite(payload)
       if payload['h3_geojson']['features']:geometry(payload)
       return payload
      start=time.perf_counter();base=scan({})
      if not cells(base):
       assert base['summary']['wmo_alert_level']=='UNAVAILABLE'
       assert base['summary']['max_p_failure'] is None
       assert not base['runout_geojson']['features']
       row.update(passed=True,status='FALLBACK_RUNTIME_VERIFIED_HAZARD_NOT_EVALUATED',numeric_parity=None,scenario=None)
       report['rows'].append(row)
       print(name,'explicit unavailable terrain response verified; hazard tests not evaluated',flush=True)
       continue
      scenario=perturbation(rng,archetype,float(np.median([p['prefire_fuel_index_proxy'] for p in cells(base).values()]))) if archetype!=6 else {'rain_delta_mm':300.};changed=scan(scenario)
      row.update(scenario=scenario,**compare(base,changed),baseline_score=base['summary']['max_p_failure'],perturbed_score=changed['summary']['max_p_failure'],critical_fs=changed['summary']['min_fs_dynamic'],max_velocity=max(p['peak_velocity_ms'] for p in cells(changed).values()),slope_p95=float(np.percentile([p['slope_deg'] for p in cells(base).values()],95)))
      chat=client.post('/api/v1/agent/chat',json={'message':'Scan '+name,'current_query':name,'sliders':scenario});chat.raise_for_status();reply=chat.json();finite(reply);geometry(reply['updated_scan_data'])
      np.testing.assert_allclose([p['p_failure'] for p in cells(changed).values()],[p['p_failure'] for p in cells(reply['updated_scan_data']).values()],atol=1e-12)
      row['numeric_parity']=reply['numeric_audit_passed'] and reply['briefing_guard']!='WITHHELD_INVALID_TEMPLATE'
      row['snapshot_api_compute_seconds']=time.perf_counter()-start
      if archetype==6:
       row['flat_immunity']=row['perturbed_score']<.01 and row['max_velocity']==0
       report['flat_controls'].append({'sweep':sweep,'name':name,'passed':row['flat_immunity'],'score':row['perturbed_score'],'slope_p95':row['slope_p95']})
      if archetype in expected:
       cause,trigger=expected[archetype];target=scan(trigger)
       candidate=[p for p in cells(target).values() if p['slope_deg']>=24]
       if archetype==3:candidate=[p for p in candidate if p['cryospheric_eligible'] and p['cell_snow_depth_m']>0]
       if archetype==4:candidate=[p for p in candidate if p['prefire_fuel_index_proxy']>=.20]
       base_cells=cells(base)
       active=[p for p in candidate if p['causal_attribution_valid'] and p['causal_breakdown_pct'][cause]>0 and any(c['causal_group']==cause for c in p['numbers_that_matter'])]
       increased=sum(p['causal_breakdown_pct'][cause]>base_cells[p['h3_index']]['causal_breakdown_pct'][cause]+1e-8 for p in active)
       met=target['summary']['meteorological_metrics'];base_met=base['summary']['meteorological_metrics']
       if archetype==1:assert met['rain_24h_mm_scenario']>=base_met['rain_24h_mm_scenario']+180-1e-8
       if archetype==2:assert met['pga_g_scenario']>=.4
       if archetype==3:assert met['air_temperature_c_scenario']>=base_met['air_temperature_c_scenario']+8-1e-8
       if archetype==4:assert met['burn_severity_override']==.85 and met['rain_24h_mm_scenario']>=base_met['rain_24h_mm_scenario']+65-1e-8
       aligned=bool(active) if candidate else None
       report['causal_controls'].append({'sweep':sweep,'name':name,'cause':cause,'eligible_cells':len(candidate),'activated_cells':len(active),'increased_share_cells':increased,'gate':'injected trigger present, attribution valid and positive, matching card; shares may plateau at constitutive bounds','passed':aligned,'status':'tested' if candidate else 'not_applicable_no_eligible_steep_cells'})
       row['causal_alignment']=aligned
      row['passed']=bool(row['monotonic_violations']==0 and row['kinematic_contradictions']==0 and row['numeric_parity'] and row.get('flat_immunity',True) and row.get('causal_alignment') is not False)
    except Exception as exc:row['error']=type(exc).__name__+': '+str(exc)
    report['rows'].append(row);out.write_text(json.dumps(portable_payload(sanitize_json(report)),indent=2,allow_nan=False))
    print(name,'sweep',sweep,'PASS' if row['passed'] else 'FAIL',row.get('perturbed_score'),row.get('error',''),flush=True)
 try:
  feature_contract()
  assert (ROOT/'LICENSE').read_bytes()==(ROOT/'training/LICENSE').read_bytes()
  report['bundle_sync_passed']=True
 except Exception as exc:report['bundle_sync_passed']=False;report['bundle_error']=str(exc)
 from scripts.finalize_readiness import fingerprint
 report['implementation_fingerprint']=fingerprint()
 report['engineering_ready']=all(r['passed'] for r in report['rows']) and report['bundle_sync_passed'] and len(report['rows'])==30
 report['hazard_evaluated_rows']=sum(not r.get('status','').startswith('FALLBACK') for r in report['rows']);report['fallback_only_rows']=len(report['rows'])-report['hazard_evaluated_rows']
 report['passed_rows']=sum(r['passed'] for r in report['rows']);report['total_rows']=len(report['rows'])
 out.write_text(json.dumps(portable_payload(sanitize_json(report)),indent=2,allow_nan=False))
 lines=['# Randomized global scenario audit','',report['protocol'],'',f"Seeds: {seeds}; engineering_ready = {report['engineering_ready']}",'','| Sweep | Location | Score | FS | Monotonic violations | Result |','|---|---|---:|---:|---:|---|']
 for r in report['rows']:lines.append('| {} | {} | {} | {} | {} | {} |'.format(r['sweep'],r['name'],round(r['perturbed_score'],5) if 'perturbed_score' in r else 'unavailable',round(r['critical_fs'],3) if 'critical_fs' in r else 'unavailable',r.get('monotonic_violations','unavailable'),'PASS' if r['passed'] else 'FAIL'))
 (ROOT/'artifacts/randomized_global_blind_audit.md').write_text('\n'.join(lines)+'\n')
 print('ENGINEERING_READY',report['engineering_ready'],flush=True)
 return 0 if report['engineering_ready'] else 1
if __name__=='__main__':raise SystemExit(main())
