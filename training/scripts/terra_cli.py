#!/usr/bin/env python3
"""Explore rainfall evidence, compare scenarios, and inspect source limitations."""
import argparse,asyncio,contextlib,json,math,socket,sys,time
from pathlib import Path
import httpx
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from backend.api.scan_service import GlobalScanService,known_location_coordinates
from backend.realtime.global_ingestor import GlobalIngestor,GlobalScan,GeocodingError
from backend.llm.qwen_agent import TerraShieldQwenAgent
from scripts.cli_location import coordinate,parse_location,prompt_location,location_query,describe_location
from scripts.cli_ui import banner,input_examples,show_summary,forecast_table,show_sources,show_comparison,Busy,fmt,panel,line,configure
from scripts.cli_scenarios import prompt_scenario


def show_details(scan,elapsed):
    show_sources(scan)
    panel('INPUTS AND LIMITATIONS')
    ev=scan.get('evidence',{});summary=scan.get('summary',{})
    line('Physical column water: '+fmt(ev.get('volumetric_moisture_m3m3'),3)+' m3/m3')
    line('Estimated pore volume: '+fmt(ev.get('estimated_porosity_fraction'),3))
    line('Absolute saturation: '+fmt(ev.get('absolute_saturation_fraction'),3))
    line('Compatibility: '+ev.get('moisture_compatibility','unavailable'))
    line(ev.get('relative_wetness_scope',''))
    assessment=scan.get('statistical_assessment',{})
    line(assessment.get('scope','Source-based rainfall assessment unavailable'))
    selected=assessment.get('hotspot') or {}
    if selected.get('map_url'):line('Selected cell map: '+selected['map_url'])
    for name,value in selected.get('features',{}).items():line(name+': '+fmt(value,3),4)
    physical=scan.get('physical_assessment',{})
    line('Optional site overrides: '+', '.join(physical.get('optional_site_overrides',physical.get('required_inputs',[]))))
    for row in physical.get('omitted_cells',[]):line(row['h3_index']+': '+row['reason'],4)
    line('Physical FS uses declared reference priors, antecedent water balance and optional site overrides; inspect the FS envelope and sensitivities.')
    line('Model evaluation: docs/source_model_status.md | historical examples are not independent forecast validation.')


async def acquire(query,use_cache=False):
    location=parse_location(query)
    async with GlobalIngestor(use_cache=use_cache) as ingestor:
        try:return await ingestor.scan(location)
        except GeocodingError as exc:
            if isinstance(location,tuple):raise
            fallback=known_location_coordinates(query)
            if fallback is None:raise
            raw=await ingestor.scan(fallback)
            raw.sources['geocoding'].available=False;raw.sources['geocoding'].mode='FALLBACK'
            raw.sources['geocoding'].error='Known-coordinate fallback after live Open-Meteo lookup failure: '+str(exc)
            return raw


def load_snapshot(path):
    from scripts.snapshot_replay import hydrate
    payload=json.loads(Path(path).read_text())
    if not isinstance(payload,dict) or not isinstance(payload.get('snapshot'),dict):
        raise ValueError('Replay JSON needs a snapshot object; --output exports a response, not a reusable provider snapshot')
    raw=hydrate(GlobalScan,payload['snapshot'])
    provenance=payload.get('provenance') or {}
    return raw,{'label':provenance.get('label','Retained provider snapshot'),
                'scope':provenance.get('scope','Offline provider replay; not live conditions'),
                'path':str(Path(path))}


def show(scan,elapsed,details=False):
    show_summary(scan,elapsed)
    if details:show_details(scan,elapsed)


def write_response(scan,path):
    target=Path(path).expanduser();target.parent.mkdir(parents=True,exist_ok=True)
    target.write_text(json.dumps(scan,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
    return target


def usable(scan):
    return bool(scan.get('h3_geojson',{}).get('features') or scan.get('statistical_assessment',{}).get('status')=='EXPERIMENTAL')


async def evaluate(service,query,raw,changes,replay,ground_columns=None):
    if replay:
        from unittest.mock import patch
        with patch.object(socket.socket,'connect',side_effect=RuntimeError('Network disabled during snapshot replay')), \
             patch.object(socket,'create_connection',side_effect=RuntimeError('Network disabled during snapshot replay')):
            result=await service.scan(location_query=query,_snapshot=raw,ground_columns=ground_columns,**changes)
        result['replay']=replay
    else:result=await service.scan(location_query=query,_snapshot=raw,ground_columns=ground_columns,**changes)
    return result


async def main(args):
    configure(args.plain)
    if not args.json:banner()
    if args.examples:input_examples();return 0
    ground_columns=None
    if args.ground_data:
        try:
            payload=json.loads(args.ground_data.read_text())
            if not isinstance(payload,dict) or not isinstance(payload.get('cells'),dict):
                raise ValueError('Ground-data JSON requires a cells object indexed by H3 IDs')
            ground_columns=payload['cells']
        except (OSError,ValueError) as exc:
            print('Could not load ground data: '+str(exc),file=sys.stderr);return 2
    service=GlobalScanService();agent=None
    query=args.location;recent=[]
    replay_path=args.snapshot
    one_shot=args.non_interactive or args.json or (args.location is not None and not args.interactive) or (bool(replay_path) and not sys.stdin.isatty() and not args.interactive)
    while True:
        raw=None;replay=None
        if replay_path:
            try:raw,replay=load_snapshot(replay_path)
            except (OSError,ValueError,TypeError,KeyError) as exc:
                print('Could not load replay: '+str(exc),file=sys.stderr);return 2
            query=raw.location.query
        elif query is None:
            selected=prompt_location(recent)
            if selected is None:return 0
            query=location_query(selected)
        try:
            start=time.perf_counter()
            if raw is None:
                if args.json:raw=await acquire(query,use_cache=args.use_cache)
                else:
                    async with Busy('Fetching terrain and environmental sources',args.plain):raw=await acquire(query,use_cache=args.use_cache)
            original=await evaluate(service,query,raw,{},replay,ground_columns)
            changes={}
            if args.rain_delta_mm:changes.update(rain_delta_mm=args.rain_delta_mm,storm_duration_hr=args.storm_duration_hr)
            if args.inject_forecast_storm_mm is not None:
                changes.update(inject_forecast_storm_mm=args.inject_forecast_storm_mm,storm_arrival_hour=args.storm_arrival_hour,storm_duration_hours=args.storm_duration_hours)
            scan=await evaluate(service,query,raw,changes,replay,ground_columns) if changes else original
            elapsed=time.perf_counter()-start
            recent=[query]+[name for name in recent if name!=query][:4]
            if args.json:print(json.dumps(scan,indent=2,allow_nan=False))
            else:
                if changes:show_comparison(original,scan)
                show(scan,elapsed,args.details)
            if args.output:
                target=write_response(scan,args.output)
                print('Saved response: '+str(target),file=sys.stderr if args.json else sys.stdout)
            if one_shot:return 0 if usable(scan) else 2
        except (ValueError,RuntimeError,asyncio.TimeoutError,httpx.HTTPError,OSError) as exc:
            print('Scan unavailable: '+str(exc),file=sys.stderr if args.json else sys.stdout)
            if one_shot:return 2
            query=None;replay_path=None;continue
        while True:
            panel('NEXT / Explore the evidence')
            line('[1] Another location       [2] What-if scenarios')
            line('[3] Forecast details       [4] Sources and limitations')
            line('[5] Optional assistant     [6] Reference equations')
            line('[s] Save response JSON     [0] Original snapshot')
            line('[r] '+('Replay original inputs' if replay else 'Refresh live data')+'    [?] Help    [q] Quit')
            choice=input('Next action: ').strip().lower()
            try:
                if choice in ('q','quit'):return 0
                if choice=='1':query=None;replay_path=None;break
                if choice=='r':break
                if choice=='0':scan=original;show(scan,elapsed,args.details)
                elif choice in ('?','help'):input_examples()
                elif choice=='2':
                    changes=prompt_scenario()
                    if changes is None:continue
                    start=time.perf_counter()
                    after=await evaluate(service,query,raw,changes,replay,ground_columns)
                    show_comparison(original,after)
                    scan,elapsed=after,time.perf_counter()-start
                    show(scan,elapsed,args.details)
                elif choice=='3':forecast_table(scan.get('predictive_forecast_72h',{}),scan.get('location',{}).get('forecast_weather_hourly_72h'))
                elif choice=='4':show_details(scan,elapsed)
                elif choice=='5':
                    if replay:
                        line('Snapshot replay keeps network connections disabled. Inspect the structured physical/ML evidence here; state-bound co-pilot is available in a connected session.')
                        continue
                    question=input('Ask about sources or assumptions [b = back]: ').strip()
                    if question.lower() in ('b','back',''):continue
                    if agent is None:agent=TerraShieldQwenAgent(service)
                    async with Busy('Asking the optional assistant',args.plain):result=await agent.chat(question,current_query=query,scan_id=scan['scan_id'],scenario_id=scan['scenario_id'])
                    line(result['briefing_guard']);print(result['clean_markdown'])
                elif choice=='6':sanity_suite(service.model)
                elif choice=='s':
                    path=input('Save response as [terra_scan.json; b = back]: ').strip() or 'terra_scan.json'
                    if path.lower()!='b':line('Saved response: '+str(write_response(scan,path)))
                elif choice not in ('0','1','r'):line('Choose a listed action, ? for help, or q to quit.')
            except (ValueError,RuntimeError,asyncio.TimeoutError,httpx.HTTPError,OSError) as exc:
                line('Action unavailable: '+str(exc));line('Your last scan is still available.')


def sanity_suite(model=None):
    import unittest
    from tests.test_reference_equations import ReferenceEquations
    panel('REFERENCE EQUATIONS / Network disabled')
    line('These checks verify equations and units, not observed forecasting skill.')
    suite=unittest.defaultTestLoader.loadTestsFromTestCase(ReferenceEquations)
    return unittest.TextTestRunner(stream=sys.stdout,verbosity=1).run(suite).wasSuccessful()


def parse_args(argv=None):
    parser=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='Examples:\n  %(prog)s\n  %(prog)s "Wayanad, India" --interactive\n  %(prog)s --lat 11.548 --lon 76.134 --rain-delta-mm 50\n  %(prog)s --snapshot data/replays/chooralmala_2024.json --json\n\nUnlabeled coordinates are LATITUDE, LONGITUDE. Snapshot replay makes no network requests.')
    parser.add_argument('location',nargs='?',help='Place name or quoted latitude, longitude')
    parser.add_argument('--lat','--latitude',help='Latitude (-90 to 90)')
    parser.add_argument('--lon','--longitude',help='Longitude (-180 to 180)')
    mode=parser.add_mutually_exclusive_group()
    mode.add_argument('--non-interactive',action='store_true',help='Scan once and exit')
    mode.add_argument('--interactive',action='store_true',help='Keep the exploration menu open')
    parser.add_argument('--snapshot',type=Path,help='Replay a provider snapshot JSON without network')
    parser.add_argument('--ground-data',type=Path,help='Optional sourced site overrides by H3 cell; missing parameters retain declared reference priors')
    parser.add_argument('--json',action='store_true',help='Emit only the API response JSON to stdout; scan once')
    parser.add_argument('--output',type=Path,help='Save the response JSON')
    parser.add_argument('--use-cache',action='store_true',help='Allow live ingestion memory-cache reads')
    parser.add_argument('--examples',action='store_true',help='Show input examples without fetching data')
    parser.add_argument('--details',action='store_true',help='Include full source details')
    parser.add_argument('--plain',action='store_true',help='Disable ANSI color and animated progress')
    parser.add_argument('--rain-delta-mm',type=float,default=0.,help='Add a rainfall sensitivity scenario, mm')
    parser.add_argument('--storm-duration-hr',type=float,default=24.,help='Physical pulse duration, hours; statistical feature remains a 24h total')
    parser.add_argument('--inject-forecast-storm-mm',type=float)
    parser.add_argument('--storm-arrival-hour',type=int,default=18)
    parser.add_argument('--storm-duration-hours',type=int,default=12)
    args=parser.parse_args(argv)
    if (args.lat is None)!=(args.lon is None):parser.error('Provide both --lat and --lon.')
    if args.location is not None and args.lat is not None:parser.error('Use a place/coordinate pair OR --lat and --lon, not both.')
    if args.snapshot and (args.location is not None or args.lat is not None):parser.error('Replay already specifies a location; omit location/coordinate flags.')
    if args.json and args.interactive:parser.error('--json is a one-shot output; omit --interactive.')
    if args.json and args.examples:parser.error('Use --examples without --json.')
    try:
        if args.lat is not None:args.location=location_query((coordinate(args.lat,'latitude'),coordinate(args.lon,'longitude')))
        elif args.location is not None:args.location=location_query(parse_location(args.location))
    except ValueError as exc:parser.error(str(exc))
    if (args.non_interactive or args.json) and args.location is None and not (args.examples or args.snapshot):
        parser.error('Supply a location, both coordinates, or --snapshot for one-shot output.')
    if not math.isfinite(args.rain_delta_mm) or args.rain_delta_mm<0:parser.error('Rain delta must be finite and nonnegative.')
    if not math.isfinite(args.storm_duration_hr) or args.storm_duration_hr<=0:parser.error('Storm duration must be finite and positive.')
    if args.inject_forecast_storm_mm is not None:
        from backend.physics.temporal_forecaster import validate_injection
        try:validate_injection(args.inject_forecast_storm_mm,args.storm_arrival_hour,args.storm_duration_hours)
        except ValueError as exc:parser.error(str(exc))
    return args

if __name__=='__main__':
    try:raise SystemExit(asyncio.run(main(parse_args())))
    except (EOFError,KeyboardInterrupt):print('\nGoodbye.',file=sys.stderr)
