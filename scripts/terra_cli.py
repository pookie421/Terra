#!/usr/bin/env python3
"""Interactive research dashboard. Provider failure never becomes fabricated terrain."""
import argparse,asyncio,json,sys,time
import httpx
import numpy as np
import h3
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from backend.api.scan_service import GlobalScanService,known_location_coordinates
from backend.realtime.global_ingestor import GlobalIngestor
from backend.llm.qwen_agent import TerraShieldQwenAgent

def show(scan,elapsed):
 loc=scan['location'];s=scan['summary'];features=scan['h3_geojson']['features'];top=s.get('kinematics_top_hotspot') or {}
 print('\nTERRA research scan —',loc.get('display_name') or loc.get('query'))
 print('Coordinates:',loc.get('latitude'),loc.get('longitude'),'Latency ms:',round(elapsed*1000),'H3 cells:',len(features))
 print('Elevation range:',loc.get('nasa_telemetry',{}).get('srtm_terrarium_elevation_range_m'))
 if features:print('P95 slope:',float(np.percentile([f['properties']['slope_deg'] for f in features],95)),'Max TWI:',max(f['properties'].get('twi',0) for f in features))
 geology=loc.get('bedrock_lithology') or {}
 print('Mapped geology:',geology.get('name'), '|',geology.get('lithology'), '| age:',geology.get('age'))
 print('\nFEEDS (successful responses may come from cache; fallback values are not measurements)')
 for name,status in scan['source_status'].items():print(f"{name:20} {status.get('mode','LIVE_OR_CACHED') if status['available'] else 'FALLBACK':16} {status.get('error') or status['provider']}")
 weather=s.get('meteorological_metrics',{});nasa=loc.get('nasa_telemetry',{});river=loc.get('river_discharge') or {}
 def sensor(label,value,unit,provider):
  status=scan['source_status'].get(provider,{})
  mode=('MISSING_VALUE' if value is None else status.get('mode','UNAVAILABLE')) if status.get('available') else 'FALLBACK'
  print(f"{label:30} {str(value):20} {unit:8} [{mode}]")
 for label,key,unit in [('Rain last 24 h','rain_24h_mm_observed','mm'),('Antecedent rain index 7 d','ari_7d_mm','mm'),('Rain forecast next 48 h','forecast_rain_48h_mm','mm'),('Snow depth','snow_depth_m','m'),('Air temperature (scenario)','air_temperature_c_scenario','C'),('Soil temperature observed','soil_temperature_c_observed','C')]:sensor(label,weather.get(key),unit,'open_meteo')
 print('Assimilated saturation:',weather.get('assimilated_soil_saturation_fraction'),'[DERIVED from available hydrology; see source statuses]')
 sensor('NASA POWER root wetness',nasa.get('nasa_power_gwetroot'),'fraction','nasa_power')
 sensor('NASA POWER daily rain',nasa.get('nasa_power_prectotcorr_mm'),'mm','nasa_power')
 sensor('NASA COOLR nearby count',nasa.get('nasa_coolr_nearby_count'),'events','coolr')
 sensor('Modeled river discharge',river.get('discharge_m3_s'),'m3/s','river')
 sensor('Modeled seven-day surge',river.get('surge_ratio'),'ratio','river')
 sensor('USGS-derived seismic PGA',weather.get('pga_g_observed'),'g','usgs')
 print('\nVERDICT:',s.get('wmo_alert_level'),'— research score, not an operational warning')
 if not features:
  print('Terrain unavailable: FS, hazard attribution, failure score, and runout were not evaluated.')
  return
 print('Critical FS static/dynamic:',s.get('critical_fs_static'),s.get('critical_fs'),'Hotspot FS:',top.get('fs_dynamic'))
 score=s.get('max_p_failure');bounds=top.get('conformal_interval_90')
 print('Critical failure score:',None if score is None else f'{100*score:.2f}%', 'Hotspot label interval (90% nominal):',None if bounds is None else [round(100*v,2) for v in bounds],'Class:',top.get('bivariate_class'))
 print('Interval targets binary labels; it is not a confidence interval for true failure probability.')
 print('Causes:',s.get('causal_breakdown_pct'));print('Mode:',top.get('failure_mode'),'Velocity m/s:',top.get('peak_velocity_ms'),'Class:',top.get('cruden_varnes_class'))
 for route in scan['runout_geojson']['features']:
  p=route['properties'];coords=route['geometry']['coordinates'];distance=sum(h3.great_circle_distance((a[1],a[0]),(b[1],b[0]),unit='m') for a,b in zip(coords,coords[1:]));print('Distance m:',round(distance));print('Runout:',p['source_cell'],'ETA min:',p['arrival_time_min'],'roads:',p['intersected_roads'],'households:',p['intersected_households'])
 for c in s.get('numbers_that_matter',[]):print(f"{c['metric']:30} {c['value']:.4g} {c['unit']} | {c['status']} | {c['threshold']}")

async def acquire(query):
 try:
  parts=query.split(',');coords=tuple(map(float,parts)) if len(parts)==2 else None
 except ValueError:coords=None
 async with GlobalIngestor() as ingestor:
  try:return await ingestor.scan(coords or query)
  except (ValueError,asyncio.TimeoutError,httpx.HTTPError):
   fallback=known_location_coordinates(query)
   if fallback is None:raise
   raw=await ingestor.scan(fallback)
   raw.sources['geocoding'].available=False;raw.sources['geocoding'].mode='FALLBACK';raw.sources['geocoding'].error='Known-coordinate fallback after geocoder failure'
   return raw

async def main(args):
 service=GlobalScanService();agent=TerraShieldQwenAgent(service)
 query=args.location
 while True:
  query=query or input("Enter any location or coordinate on Earth: ").strip()
  if query.lower() in ('q','quit'):return 0
  try:
   start=time.perf_counter();raw=await acquire(query)
   scan=await service.scan(location_query=query,_snapshot=raw);show(scan,time.perf_counter()-start)
   if args.non_interactive or args.location:
    return 0 if scan['h3_geojson']['features'] else 2
   while True:
    choice=input('\n[1] Another area [2] Forecast/stress [3] Qwen [4] Sanity suite [q] Quit: ').strip()
    if choice=='q':return 0
    if choice=='1':query=None;break
    if choice=='2':
     kind=input('forecast / rain / pga / warming / burn: ').strip().lower()
     keys={'rain':'rain_delta_mm','pga':'pga_override_g','warming':'snowmelt_temp_surge_c','burn':'burn_severity_override'}
     if kind=='forecast':
      value=scan['summary']['meteorological_metrics'].get('forecast_rain_48h_mm')
      if value is None:print('Forecast unavailable');continue
      from dataclasses import replace
      forecast_raw=replace(raw,weather=replace(raw.weather,rain_24h_mm=value))
      changes={'storm_duration_hr':48}
     elif kind in keys:
      value=float(input('Value (burn in %): '));changes={keys[kind]:value/100 if kind=='burn' else value}
     else:continue
     start=time.perf_counter();after=await service.scan(location_query=query,_snapshot=forecast_raw if kind=='forecast' else raw,**changes)
     if kind=='forecast':print('Scenario: forecast 48 h total replaces rain input; same antecedent conditions retained.')
     print('Before -> After score:',scan['summary']['max_p_failure'],'->',after['summary']['max_p_failure']);show(after,time.perf_counter()-start)
    elif choice=='3':
     result=await agent.chat(input('Question: '),current_query=query)
     print(result['tool_calls']);print(result['briefing_guard']);print(result['clean_markdown'])
    elif choice=='4':
     sanity_suite(service.model)
  except (ValueError,RuntimeError,asyncio.TimeoutError,httpx.HTTPError) as exc:
   print('Scan unavailable:',str(exc))
   if args.non_interactive or args.location:return 2
   query=None

def sanity_suite(model):
 from backend.ml.validation_suite import run_counterfactual_controls,run_causal_regime_controls,run_kinematic_branch_controls,monotonicity_sweep
 controls=run_counterfactual_controls();causes=run_causal_regime_controls();flow=run_kinematic_branch_controls()[0];mono=monotonicity_sweep(model)
 rows=[('Flat city, 350 mm rain',controls[0].passed,controls[0].assertion),('Bone-dry cliff',controls[1].passed,controls[1].assertion),('Active monsoon flow',flow.passed,flow.actual_mode),('Seismic fault shake',causes[0]['passed'],causes[0]['actual_dominant_cause']),('Alpine snowmelt',causes[1]['passed'],causes[1]['actual_dominant_cause']),('Monotonicity sweep',mono.status=='passed',str(mono.metrics))]
 print('Synthetic controls; these are not historical validation observations.')
 for name,passed,detail in rows:print(f"{'PASS' if passed else 'FAIL':4} | {name:25} | {detail}")
 return all(r[1] for r in rows)

if __name__=='__main__':
 parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('location',nargs='?');parser.add_argument('--non-interactive',action='store_true')
 try:raise SystemExit(asyncio.run(main(parser.parse_args())))
 except (EOFError,KeyboardInterrupt):print('\nGoodbye.')
