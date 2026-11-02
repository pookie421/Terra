#!/usr/bin/env python3
"""Interactive research dashboard. Provider failure never becomes fabricated terrain."""
import argparse,asyncio,json,sys,time
import httpx
import numpy as np
import h3
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from backend.api.scan_service import GlobalScanService,known_location_coordinates
from backend.realtime.global_ingestor import GlobalIngestor, GeocodingError
from backend.llm.qwen_agent import TerraShieldQwenAgent
from scripts.cli_location import coordinate, parse_location, prompt_location, location_query, describe_location
from scripts.cli_ui import banner, input_examples, show_summary, forecast_table, Busy, fmt, panel
from scripts.cli_scenarios import prompt_scenario

def show_details(scan,elapsed):
 loc=scan['location'];s=scan['summary'];features=scan['h3_geojson']['features'];top=s.get('kinematics_top_hotspot') or {}
 print('\nTERRA research scan —',loc.get('display_name') or loc.get('query'))
 print('Latitude:',loc.get('latitude'),'| Longitude:',loc.get('longitude'),'| Latency ms:',round(elapsed*1000),'| H3 cells:',len(features))
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
 print('Epistemic spread proxy (90% normal):',top.get('epistemic_interval_90_pct'),'%; coverage is unvalidated')
 print('Binary-label conformal interval is separate from this probability-spread proxy.')
 forecast=scan.get('predictive_forecast_72h',{})
 print('72-HOUR PROJECTION:',forecast.get('status'),forecast.get('reason',''))
 if forecast.get('status')=='PROJECTED_SCENARIO':
  print('Forcing:',forecast['forcing_modes'])
  print('Initial state:',forecast.get('initial_condition_quality'))
  print('Observed current score %:',forecast.get('observed_current_risk_t0_pct'))
  print('Soil temperature uses air for hours:',forecast.get('soil_temperature_fallback_hours',[]))
  print('P50/P90 peak score %:',forecast['peak_72h_risk_p50_pct'],forecast['peak_72h_risk_p90_pct'])
  print('Projected crossing hour:',forecast['predicted_threshold_crossing_hour'],'Lead hours:',forecast['advance_warning_lead_time_hours'])
  print('Hour | P50 FS | P50 risk % | P90 FS | P90 risk %')
  for row in forecast['hourly_horizon_table']:
   a,b=row['p50'],row['p90'];print(f"T+{row['hour']:02d} | {a['critical_fs']:.3f} | {a['risk_pct']:8.2f} | {b['critical_fs']:.3f} | {b['risk_pct']:8.2f}")
  print('These are hourly scenario crossings, not exact observed failure times.')
 print('Causes:',s.get('causal_breakdown_pct'));print('Mode:',top.get('failure_mode'),'Velocity m/s:',top.get('peak_velocity_ms'),'Class:',top.get('cruden_varnes_class'))
 for route in scan['runout_geojson']['features']:
  p=route['properties'];coords=route['geometry']['coordinates'];distance=sum(h3.great_circle_distance((a[1],a[0]),(b[1],b[0]),unit='m') for a,b in zip(coords,coords[1:]));print('Distance m:',round(distance));print('Runout:',p['source_cell'],'ETA min:',p['arrival_time_min'],'roads:',p['intersected_roads'],'households:',p['intersected_households'])
 for c in s.get('numbers_that_matter',[]):print(f"{c['metric']:30} {c['value']:.4g} {c['unit']} | {c['status']} | {c['threshold']}")

async def acquire(query, use_cache=False):
 location=parse_location(query)
 async with GlobalIngestor(use_cache=use_cache) as ingestor:
  try:return await ingestor.scan(location)
  except GeocodingError as exc:
   if isinstance(location,tuple):raise
   fallback=known_location_coordinates(query)
   if fallback is None:raise
   raw=await ingestor.scan(fallback)
   raw.sources['geocoding'].available=False;raw.sources['geocoding'].mode='FALLBACK';raw.sources['geocoding'].error='Known-coordinate fallback after live geocoder failure: '+str(exc)
   return raw

def show(scan, elapsed, details=False):
    show_summary(scan, elapsed)
    if details:
        show_details(scan, elapsed)


async def main(args):
    banner()
    if args.examples:
        input_examples()
        return 0
    print('Data policy:', 'cache reads enabled by --use-cache' if args.use_cache else 'fresh network first; memory cache only after a provider error')
    service = GlobalScanService()
    agent = TerraShieldQwenAgent(service)
    query = args.location
    recent = []
    one_shot = args.non_interactive or (args.location is not None and not args.interactive)
    while True:
        if query is None:
            selected = prompt_location(recent)
            if selected is None:
                return 0
            query = location_query(selected)
        try:
            print('\nScanning —', describe_location(parse_location(query)))
            start = time.perf_counter()
            async with Busy('Fetching terrain and environmental sources', args.plain):
                raw = await acquire(query, use_cache=args.use_cache)
                scan = await service.scan(location_query=query, _snapshot=raw,
                    inject_forecast_storm_mm=args.inject_forecast_storm_mm,
                    storm_arrival_hour=args.storm_arrival_hour,
                    storm_duration_hours=args.storm_duration_hours)
            elapsed = time.perf_counter() - start
            recent = [query] + [name for name in recent if name != query][:4]
            show(scan, elapsed, args.details)
            if one_shot:
                return 0 if scan['h3_geojson']['features'] else 2
        except (ValueError, RuntimeError, asyncio.TimeoutError, httpx.HTTPError) as exc:
            print('Scan unavailable:', str(exc))
            if one_shot:
                return 2
            query = None
            continue
        while True:
            panel('NEXT / Explore this scan')
            print('  [1] Another location     [2] What-if scenario')
            print('  [3] Forecast chart       [4] Full data and source details')
            print('  [5] Ask Qwen             [6] Synthetic sanity checks')
            print('  [r] Refresh live data    [?] Input examples    [q] Quit')
            choice = input('Next action: ').strip().lower()
            try:
                if choice in ('q', 'quit'):
                    return 0
                if choice == '1':
                    query = None
                    break
                if choice == 'r':
                    print('Refreshing this location; providers may serve cached responses.')
                    break
                if choice in ('?', 'help'):
                    input_examples()
                elif choice == '2':
                    changes = prompt_scenario()
                    if changes is None:
                        continue
                    start = time.perf_counter()
                    async with Busy('Computing scenario from the original snapshot', args.plain):
                        after = await service.scan(location_query=query, _snapshot=raw, **changes)
                    before_score = scan['summary'].get('max_p_failure')
                    after_score = after['summary'].get('max_p_failure')
                    print('Previous display -> new regional score:',
                          fmt(None if before_score is None else before_score * 100), '->',
                          fmt(None if after_score is None else after_score * 100), '%')
                    scan, elapsed = after, time.perf_counter() - start
                    show(scan, elapsed, args.details)
                elif choice == '3':
                    forecast_table(scan.get('predictive_forecast_72h', {}))
                elif choice == '4':
                    show_details(scan, elapsed)
                elif choice == '5':
                    question = input('Question (e.g. "What drives the risk in Wayanad?"; b = back): ').strip()
                    if question.lower() in ('b', 'back', ''):
                        continue
                    async with Busy('Asking Qwen', args.plain):
                        result = await agent.chat(question, current_query=query)
                    print(result['briefing_guard'])
                    print(result['clean_markdown'])
                elif choice == '6':
                    sanity_suite(service.model)
                else:
                    print('Choose a listed number, r, ?, or q.')
            except (ValueError, RuntimeError, asyncio.TimeoutError, httpx.HTTPError) as exc:
                print('Could not complete that action:', str(exc))
                print('Your last scan is still available. Choose another action.')


def sanity_suite(model):
 from backend.ml.validation_suite import run_counterfactual_controls,run_causal_regime_controls,run_kinematic_branch_controls,monotonicity_sweep
 controls=run_counterfactual_controls();causes=run_causal_regime_controls();flow=run_kinematic_branch_controls()[0];mono=monotonicity_sweep(model)
 rows=[('Flat city, 350 mm rain',controls[0].passed,controls[0].assertion),('Bone-dry cliff',controls[1].passed,controls[1].assertion),('Active monsoon flow',flow.passed,flow.actual_mode),('Seismic fault shake',causes[0]['passed'],causes[0]['actual_dominant_cause']),('Alpine snowmelt',causes[1]['passed'],causes[1]['actual_dominant_cause']),('Monotonicity sweep',mono.status=='passed',str(mono.metrics))]
 print('Synthetic controls; these are not historical validation observations.')
 for name,passed,detail in rows:print(f"{'PASS' if passed else 'FAIL':4} | {name:25} | {detail}")
 return all(r[1] for r in rows)

def parse_args(argv=None):
 parser=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter,
  epilog='Examples:\n  %(prog)s\n  %(prog)s --lat 11.548 --lon 76.134\n  %(prog)s "11.548, 76.134"\n  %(prog)s "longitude=76.134 latitude=11.548"\n\nUnlabeled coordinates always use LATITUDE, LONGITUDE.\nSouth and west use negative numbers. Labels can be supplied in either order.')
 parser.add_argument('location',nargs='?',help='Place name or quoted "latitude, longitude" in decimal degrees')
 parser.add_argument('--lat','--latitude',help='Latitude (-90 to 90); north positive, south negative')
 parser.add_argument('--lon','--longitude',help='Longitude (-180 to 180); east positive, west negative')
 mode=parser.add_mutually_exclusive_group()
 mode.add_argument('--non-interactive',action='store_true',help='Scan once; requires a location or both coordinate flags')
 mode.add_argument('--interactive',action='store_true',help='Keep the action menu open after scanning a supplied location')
 parser.add_argument('--use-cache',action='store_true',help='Allow memory-cache reads; default fetches fresh data, with disclosed error fallback')
 parser.add_argument('--examples',action='store_true',help='Show every input format without fetching data')
 parser.add_argument('--details',action='store_true',help='Also display full telemetry and provider errors')
 parser.add_argument('--plain',action='store_true',help='Disable animated progress; suitable for logs and screen readers')
 parser.add_argument('--inject-forecast-storm-mm',type=float)
 parser.add_argument('--storm-arrival-hour',type=int,default=18)
 parser.add_argument('--storm-duration-hours',type=int,default=12)
 args=parser.parse_args(argv)
 if (args.lat is None)!=(args.lon is None):parser.error('Provide both --lat and --lon.')
 if args.location is not None and args.lat is not None:parser.error('Use a place/coordinate pair OR --lat and --lon, not both.')
 try:
  if args.lat is not None:
   args.location=location_query((coordinate(args.lat,'latitude'),coordinate(args.lon,'longitude')))
  elif args.location is not None:
   args.location=location_query(parse_location(args.location))
 except ValueError as exc:parser.error(str(exc))
 if args.non_interactive and args.location is None and not args.examples:parser.error('--non-interactive needs a location or --lat and --lon.')
 if args.inject_forecast_storm_mm is not None:
  from backend.physics.temporal_forecaster import validate_injection
  try:validate_injection(args.inject_forecast_storm_mm,args.storm_arrival_hour,args.storm_duration_hours)
  except ValueError as exc:parser.error(str(exc))
 return args

if __name__=='__main__':
 try:raise SystemExit(asyncio.run(main(parse_args())))
 except (EOFError,KeyboardInterrupt):print('\nGoodbye.')
