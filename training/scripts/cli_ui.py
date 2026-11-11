"""Readable terminal views for source evidence and experimental scenarios."""
import asyncio
import contextlib
import math
import os
import shutil
import sys
import textwrap
import time

_PLAIN=False

def configure(plain=False):
    global _PLAIN
    _PLAIN=plain


def width():return max(32,min(88,shutil.get_terminal_size((88,24)).columns-2))


def paint(text,code='36'):
    return f'\033[{code}m{text}\033[0m' if sys.stdout.isatty() and not _PLAIN and not os.getenv('NO_COLOR') else text


def line(text='',indent=2):
    for part in str(text).splitlines() or ['']:
        for wrapped in textwrap.wrap(part,width=max(20,width()-indent),break_long_words=True) or ['']:
            print(' '*indent+wrapped)


def panel(title):
    print('\n'+paint('-'*width(),'2'))
    line(paint(title,'1;36'))
    print(paint('-'*width(),'2'))


def banner():
    panel('TERRA  /  Terrain and hazard analysis')
    line('Terrain. NASA data. Transparent scenarios.')
    line('Source evidence and model scenarios | scores are experimental')


def input_examples():
    panel('START EXPLORING')
    for title,example in [('Live place','python scripts/terra_cli.py "Wayanad, India" --interactive'),
        ('Coordinates','python scripts/terra_cli.py --lat 11.548 --lon 76.134'),
        ('Saved snapshot','python scripts/terra_cli.py --snapshot data/replays/chooralmala_2024.json'),
        ('Compare rainfall','python scripts/terra_cli.py "Wayanad, India" --rain-delta-mm 50'),
        ('Export response','python scripts/terra_cli.py "Wayanad, India" --json > scan.json')]:
        line(title+':');line(example,4)
    line('Location input also accepts longitude=76.134 latitude=11.548, or 42.26 S, 173.62 E.')
    line('Unlabeled pairs are LATITUDE, LONGITUDE. Use q to quit or b to go back.')


def fmt(value,digits=2):
    if value is None:return 'unavailable'
    try:
        n=float(value)
        if not math.isfinite(n):return 'unavailable'
        if n==0:return '0'
        if abs(n)<.5*10**(-digits):return f'{n:.2e}'
        return f'{n:.{digits}f}'
    except (ValueError,TypeError):return 'unavailable'


def bar(percent,width=18):
    if fmt(percent)=='unavailable':return '['+'?'*width+'] unavailable'
    filled=round(max(0.,min(100.,float(percent)))*width/100)
    return '['+'#'*filled+'.'*(width-filled)+f'] {float(percent):6.2f}%'


def score(value):
    if value is None:return 'unavailable'
    if value==0:return '0 (numerical model output; not evidence of safety)'
    if value==1:return '1 (numerical model output; not certainty)'
    if 0<1-value<.0005:return '1 − '+fmt(1-value,3)
    return fmt(value,3)


def show_cell(cell):
    line('Cell centre: '+fmt(cell.get('latitude'),6)+', '+fmt(cell.get('longitude'),6)+
         ' | '+fmt(cell.get('distance_from_query_m'),0)+' m from query')
    line('DEM slope: '+fmt(cell.get('slope_deg'),2)+' degrees | H3 grid ID: '+str(cell.get('h3_index','unavailable')))


def forecast_table(forecast,weather_hours=None):
    panel('FUTURE WEATHER SCENARIO')
    if forecast.get('status')=='PROJECTED_REFERENCE_SCENARIO':
        line(forecast['scope']);line('Hour       P50 minimum FS      P90-forcing minimum FS')
        for hour in (0,6,12,18,24,36,48,72):
            a=forecast['paths']['p50']['hourly'][hour];b=forecast['paths']['p90']['hourly'][hour]
            line(f"+{hour:02d}h       {fmt(a['minimum_fs'],3)}                 {fmt(b['minimum_fs'],3)}")
        return
    if forecast.get('status')!='PROJECTED_SCENARIO':
        line('Physical forecast unavailable: '+forecast.get('reason','Complete forecast inputs missing'))
        if weather_hours:
            line('Provider weather forecast only; no FS, failure probability or failure time is inferred.')
            line('UTC timestamp                Liquid rain (mm)       Air (C)')
            for row in weather_hours:
                line(str(row.get('time','unavailable'))+'   '+fmt(row.get('liquid_rain_mm'),2)+'   '+fmt(row.get('temperature_2m'),1))
        return
    line('Experimental physics path; separate from the rainfall source baseline.')
    line('Initial state: '+forecast.get('initial_condition_quality','modeled'))
    if forecast.get('initial_condition_quality')=='SYNTHETIC_CALM_NOT_LIVE':
        line('Synthetic forcing resets initial rain and head. It is not a historical reconstruction.')
    crossing=forecast.get('predicted_threshold_crossing_hour')
    line('First model-threshold crossing: '+('none within 72h' if crossing is None else f'T+{crossing}h'))
    line('Hour     P50 FS     P50 score     P90-forcing score')
    for row in forecast.get('hourly_horizon_table',[]):
        a,b=row['p50'],row['p90']
        line(f"+{row['hour']:02d}h     {fmt(a.get('critical_fs'),3):>6}        {fmt(a.get('risk_pct'),1):>5}%           {fmt(b.get('risk_pct'),1):>5}%")
    line('Threshold crossings are model behavior, not predicted landslide times. P90 describes forcing, not 90% failure certainty.')


def show_sources(scan):
    panel('SOURCES / Availability and provenance')
    for name,status in sorted(scan.get('source_status',{}).items(),key=lambda kv:(not ('nasa' in kv[0].lower() or kv[0] in ('coolr','eonet')),kv[0])):
        mode=status.get('mode','UNASSESSED') if status.get('available') else 'UNAVAILABLE'
        line(f"{name}  [{mode}]  {status.get('provider','')}")
        if status.get('error'):line(status['error'],4)
    ev=scan.get('evidence',{})
    line('NASA contributions:')
    for key,value in ev.get('nasa_contribution',{}).items():line(key+': '+value,4)
    line('Other providers: '+ev.get('other_providers','see provider records'))
    line('A responding provider does not establish measurement accuracy or independent validation.')


def show_summary(scan,elapsed):
    if scan.get('assessment_mode')=='PHYSICS_INFORMED_REFERENCE':return show_reference_summary(scan,elapsed)
    loc=scan.get('location',{});summary=scan.get('summary',{});ev=scan.get('evidence',{})
    assessment=scan.get('statistical_assessment',{});window=assessment.get('window',{})
    top=summary.get('kinematics_top_hotspot') or {}
    panel(str(loc.get('display_name') or loc.get('query') or 'Terrain scan'))
    replay=scan.get('replay',{})
    if replay:line(paint('OFFLINE REPLAY | '+replay.get('label','retained snapshot'),'1;33'))
    line(f"{fmt(loc.get('latitude'),5)}, {fmt(loc.get('longitude'),5)} | {ev.get('terrain_cells',len(scan.get('h3_geojson',{}).get('features',[])))} terrain cells | {elapsed:.1f}s")
    line('Snapshot: '+str(scan.get('generated_at_utc','unavailable')))
    if replay.get('scope'):line(replay['scope'])
    panel('SOURCE EVIDENCE')
    line('Rain, latest complete 24h: '+fmt(window.get('rain_24h_mm'),1)+' mm')
    line('Rain, preceding six days: '+fmt(window.get('rain_prior_6d_mm'),1)+' mm')
    line('Raw surface moisture (0-7 cm): '+fmt(window.get('soil_moisture_0_7cm_mean_m3m3'),3)+' m3/m3')
    line('Window cutoff: '+str(window.get('cutoff_utc','unavailable')))
    line('Soil: '+str(ev.get('soil_source','unavailable'))+' | '+str(ev.get('soil_quality','unavailable')))
    sources=scan.get('source_status',{});available=sum(bool(s.get('available')) for s in sources.values())
    line(f'{available}/{len(sources)} sources available; use [4] for NASA contributions and source details.')
    panel('INVENTORY CLASSIFIER / Unvalidated for local warning decisions')
    selected=assessment.get('hotspot') or {}
    if assessment.get('status')=='EXPERIMENTAL':
        show_cell(selected)
        line('Selection: highest inventory-classifier score in this scan.')
        line(paint('Classifier output: '+score(selected['score']),'1'))
        line('Cell score range: '+fmt(assessment['score_min'],3)+' to '+fmt(assessment['score_max'],3))
        line('Score reflects the sampled event/absence dataset; it is not your chance of experiencing a landslide.')
        if selected.get('missing_features'):line('Statistical missing-input handling: '+', '.join(selected['missing_features']))
        if selected.get('outside_training_range'):line('Outside fitted data ranges: '+', '.join(selected['outside_training_range']))
        leading=selected.get('model_explanation',{}).get('leading_upward_feature')
        if leading:line('Leading upward model influence: '+{'twi':'terrain wetness index (TWI)',
            'rain_24h_mm':'latest 24h rain','rain_prior_6d_mm':'preceding six days of rain',
            'soil_moisture_0_7cm_mean_m3m3':'raw surface moisture','slope_deg':'DEM slope',
            'profile_curvature':'terrain curvature'}.get(leading,leading)+'. This is model behavior, not a physical cause.')
        line('Training labels include report locations and failure footprints, not consistently verified initiation slopes.')
        line(assessment.get('scenario_note',''))
    else:line('Unavailable: '+assessment.get('reason','Source-based rainfall model or matching window unavailable'))
    panel('PHYSICAL COLUMN / Same cell as the classifier selection' if selected else 'PHYSICAL COLUMN / Sourced ground inputs')
    same=next((r for r in scan.get('cell_comparisons',[]) if r['h3_index']==selected.get('h3_index')),None)
    physical=scan.get('physical_assessment',{})
    if same and same.get('fs_dynamic') is not None:
        line('Conditional FS at '+same['h3_index']+': '+fmt(same['fs_dynamic'],3))
        line('Uses supplied, time-valid ground records and DEM slope. Their source citations are not independently verified.')
    else:
        line('FS unavailable: '+str(physical.get('reason') or 'No complete sourced ground column for this cell.'))
    if top and top.get('h3_index')!=selected.get('h3_index'):
        line('Lowest supported FS occurs at another cell with supplied ground data:')
        show_cell(top)
        matching=next((r for r in scan.get('cell_comparisons',[]) if r['h3_index']==top['h3_index']),{})
        line('At that same cell: classifier '+score(matching.get('statistical_score'))+' | FS '+fmt(top.get('fs_dynamic'),3))
    line('FOSM unavailable: no supported joint uncertainty model. No default zero is reported.')
    if ev.get('moisture_exceeds_porosity'):
        line('Input conflict: volumetric moisture '+fmt(ev.get('volumetric_moisture_m3m3'),3)+
             ' exceeds estimated porosity '+fmt(ev.get('estimated_porosity_fraction'),3)+'. Physical saturation was not clipped or invented.')
    line('Runout speed and failure time are not established. Missing physics does not mean safe terrain.')
    line('No local hazard verdict can be inferred from the inventory-classifier output.')
    line('Use [2] for a rainfall comparison, [3] for forecast details, or [4] for evidence.')


def show_reference_summary(scan,elapsed):
    loc=scan.get('location',{});summary=scan['summary'];p=scan.get('physics_informed_assessment',{})
    top=p.get('hotspot') or scan.get('physical_assessment',{}).get('hotspot') or {}
    physical=next((r for r in scan.get('physical_assessment',{}).get('cells',[]) if r['h3_index']==top.get('h3_index')),{})
    panel(str(loc.get('display_name') or loc.get('query') or 'Terrain scan'))
    if scan.get('replay'):line('OFFLINE REPLAY | '+scan['replay'].get('label','retained snapshot'))
    line('Scan '+scan.get('scan_id','')+' | scenario '+scan.get('scenario_id',''))
    line(str(len(scan.get('terrain_geojson',{}).get('features',[])))+' terrain cells | '+fmt(elapsed,1)+' s')
    panel('PHYSICS-INFORMED INITIATION')
    if top:show_cell(top)
    line('Physics-informed score: '+score(top.get('score')))
    line(p.get('scope') or p.get('reason','Model unavailable'))
    if top.get('ablation_scores'):
        for name,value in top['ablation_scores'].items():line(name+': '+score(value))
    panel('REFERENCE PHYSICAL COLUMN / Same selected cell')
    line('FS [low, mid, high]: '+', '.join(fmt(v,3) for v in physical.get('fs_interval',[None]*3)))
    line('FOSM variance: '+fmt(physical.get('fosm_variance'),6))
    line('Antecedent head: '+fmt(physical.get('initial_pressure_head_m'),4)+' m | current modeled head: '+fmt(physical.get('pressure_head_m'),4)+' m')
    line(physical.get('interval_scope','Physical column unavailable'))
    line('Provenance: '+str(physical.get('provenance_label','unavailable')))
    runout=scan.get('runout_assessment',{})
    line('Conditional release corridor: '+str(len(runout.get('corridor_h3_indices',[])))+' H3 cells')
    line('Conditional peak speed: '+fmt(runout.get('peak_velocity_ms'),3)+' m/s; release is a scenario, not asserted failure')
    ev=scan.get('evidence',{})
    if ev.get('moisture_exceeds_porosity'):line('Input conflict: surface moisture exceeds selected porosity; deeper storage uses antecedent water balance.')
    sources=scan.get('source_status',{})
    line(str(sum(bool(s.get('available')) for s in sources.values()))+'/'+str(len(sources))+' source responses available; inspect provenance for cache age and support.')


def show_comparison(before,after):
    panel('SCENARIO COMPARISON / Same retained snapshot')
    if before.get('assessment_mode')=='PHYSICS_INFORMED_REFERENCE':
        primary=before.get('physics_informed_assessment',{}).get('hotspot') or {}
        cell=primary.get('h3_index')
        right=next((r for r in after.get('physics_informed_assessment',{}).get('cells',[]) if r['h3_index']==cell),{})
        line('Physics-informed reference cell: '+str(cell or 'unavailable'))
        line('Fusion score: '+score(primary.get('score'))+' -> '+score(right.get('score')))
        line('FS envelope: '+str(primary.get('fs_interval'))+' -> '+str(right.get('fs_interval')))
    names={'rain_delta_mm':'Added rain (mm)','storm_duration_hr':'Pulse duration (hours)',
           'pga_override_g':'Shaking override (g)','snowmelt_temp_surge_c':'Warming (C)',
           'burn_severity_override':'Burn severity (0-1)'}
    for key,value in after.get('scenario',{}).items():
        if value:line(names.get(key,key)+': '+fmt(value,2))
    a=before.get('statistical_assessment',{});b=after.get('statistical_assessment',{})
    cell=(a.get('hotspot') or {}).get('h3_index')
    lhs=next((r for r in a.get('cells',[]) if r['h3_index']==cell),{})
    rhs=next((r for r in b.get('cells',[]) if r['h3_index']==cell),{})
    line('Reference cell: '+str(cell or 'unavailable'))
    line('                               Original       Scenario')
    for name,left,right in [('Latest 24h rain (mm)',a.get('window',{}).get('rain_24h_mm'),b.get('window',{}).get('rain_24h_mm')),
                            ('Statistical score (0-1)',lhs.get('score'),rhs.get('score'))]:
        line(f'{name:28} {fmt(left,3):>10}  {fmt(right,3):>12}')
    if b.get('status')!='EXPERIMENTAL':line(b.get('reason','Rainfall assessment unavailable'))
    physical_cell=before.get('summary',{}).get('hotspot_h3_index')
    if physical_cell:
        def properties(scan):
            return next((f['properties'] for f in scan.get('h3_geojson',{}).get('features',[])
                         if f['properties'].get('h3_index')==physical_cell),{})
        left,right=properties(before),properties(after)
        line('Physics reference cell: '+physical_cell)
        for label,key in [('Modeled FS','fs_dynamic'),('FOSM diagnostic','p_failure_physical')]:
            line(f'{label:28} {fmt(left.get(key),3):>10}  {fmt(right.get(key),3):>12}')
    line('Every scenario starts from the original inputs. This is a sensitivity comparison, not an observed outcome.')


class Busy:
    def __init__(self,label,plain=False):self.label,self.plain,self.task=label,plain,None
    async def __aenter__(self):
        self.started=time.monotonic();line(self.label+'...')
        if sys.stdout.isatty() and not self.plain:self.task=asyncio.create_task(self.animate())
        return self
    async def animate(self):
        i=0
        while True:
            print(f'\r  {"|/-"[i%3]} {self.label} | {time.monotonic()-self.started:.0f}s',end='',flush=True)
            i+=1;await asyncio.sleep(.15)
    async def __aexit__(self,*exc):
        if self.task:
            self.task.cancel()
            with contextlib.suppress(asyncio.CancelledError):await self.task
            print('\r'+' '*min(width(),80)+'\r',end='',flush=True)
