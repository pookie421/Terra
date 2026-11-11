"""Retained label-blind terrain and explicitly selected inventory-absence controls.

A radius-based representative is not a mapped initiation location or NASA's grid.
Control date repair creates a secondary benchmark; original dates stay frozen.
"""
import asyncio
import hashlib
import json
from dataclasses import asdict, replace
from datetime import date, timedelta
from pathlib import Path
from types import SimpleNamespace
import numpy as np


def terrain_object(payload):
    return SimpleNamespace(cells={c['cell_id']:SimpleNamespace(**c) for c in payload['cells']})


async def refresh_neighborhoods(rows, client, directory):
    from backend.realtime.dem_processor import DEMProcessor
    directory.mkdir(parents=True,exist_ok=True)
    dem=DEMProcessor(zoom=13);sem=asyncio.Semaphore(4)
    locations=sorted({(r['latitude'],r['longitude']) for r in rows})
    terrains={};failures=[]
    async def one(lat,lon):
        key=hashlib.sha256(json.dumps([lat,lon,1.2,13,9]).encode()).hexdigest()
        path=directory/('terrain_v13_'+key+'.json')
        try:
            if path.exists():payload=json.loads(path.read_text())
            else:
                async with sem:result=await dem.process(lat,lon,1.2,client)
                payload={'center':[lat,lon],'radius_km':1.2,'source':'AWS Terrarium tiles',
                         'tile_ids':result.tile_ids,'cells':[asdict(c) for c in result.cells.values()]}
                path.write_text(json.dumps(payload))
            terrains[(lat,lon)]=terrain_object(payload)
        except Exception as exc:failures.append({'lat':lat,'lon':lon,'error':str(exc)})
    await asyncio.gather(*(one(*loc) for loc in locations))
    print('NEIGHBORHOODS',len(terrains),'of',len(locations),'failures',len(failures),flush=True)
    # Never silently claim that a nearest-cell fallback is a measured 1km maximum.
    if failures:
        (directory/'terrain_v13_failures.json').write_text(json.dumps(failures))
        raise RuntimeError(f'{len(failures)} required terrain neighborhoods unavailable; retained successful acquisitions for retry')
    return terrains


def recompute(row,terrain):
    from backend.ml.build_real_labeled_dataset import populate_physics_features
    from backend.ml.global_xgb import PHYSICS_MODEL_VERSION
    populate_physics_features(row,terrain)
    row['physics_model_version']=PHYSICS_MODEL_VERSION
    row['physics_feature_provenance_json']=json.dumps(row['physics_feature_provenance'],sort_keys=True)
    return row


async def expand_controls(rows,terrains,client,data_dir,frozen_ids):
    from backend.ml.archive_weather import CACHE,summarize_archive,acquire_archive
    from backend.ml.build_real_labeled_dataset import EventCandidate,fetch_usgs
    from backend.ml.pinls import quiescent_control
    from backend.physics.multi_causal_engine import PhysicsInputs,MultiCausalEngine
    positives=[r for r in rows if r.get('label')==1]
    by_id={r['event_id']:r for r in rows};new=[];diagnostics=[];repaired={}
    sem=asyncio.Semaphore(5)
    # Weather histories already include many candidate days; use them before fetching.
    weather_by_location={}
    for r in rows:
        key=r.get('weather_archive_cache_key')
        if key:weather_by_location.setdefault((r['latitude'],r['longitude']),set()).add(key)
    async def search(positive):
        async with sem:
            location=(positive['latitude'],positive['longitude']);terrain=terrains[location]
            origin=date.fromisoformat(positive['event_date']);target=origin-timedelta(days=60)
            candidates={};old=by_id.get(positive['event_id']+':tminus60')
            # Persist successful searches and their selected dates across rebuilds.
            cache_key=hashlib.sha256((positive['event_id']+':external_weather_dates_v14').encode()).hexdigest()
            cache_dir=data_dir/'source_catalogs/control_search_v14';cache_dir.mkdir(parents=True,exist_ok=True)
            checkpoint=cache_dir/(cache_key+'.json')
            if checkpoint.exists():
                saved=json.loads(checkpoint.read_text());valid=[recompute(r,terrain) for r in saved['controls']]
                valid=[r for r in valid if quiescent_control(r)]
            else:valid=[]
            if old and quiescent_control(old) and old['event_date'] not in {r['event_date'] for r in valid}:valid.append(dict(old))
            def add_payload(payload,key):
                dates=sorted({t[:10] for t in payload['response'].get('hourly',{}).get('time',[])})
                for day in dates:
                    d=date.fromisoformat(day)
                    if not origin-timedelta(days=400)<=d<=origin-timedelta(days=30):continue
                    weather=summarize_archive(payload['response'],day)
                    if weather['rain_effective_storm_mm'] is None or weather['rain_effective_storm_mm']>35 or weather['snow_depth_72h_max_m']!=0:continue
                    weather.update(weather_source_url=payload['url'],weather_archive_cache_key=key,weather_acquisition='retained_archive_response',weather_feature_version='hourly_v12',weather_selected_date=day)
                    candidates[day]=weather
            for key in weather_by_location.get(location,[]):
                path=CACHE/(key+'.json')
                if path.exists():add_payload(json.loads(path.read_text()),key)
            attempted=set();attempt_count=0
            async def try_candidates():
                nonlocal attempt_count
                for day in sorted(candidates,key=lambda d:(abs((date.fromisoformat(d)-target).days),d)):
                    if len(valid)>=2:return
                    if day in attempted or day in {r['event_date'] for r in valid}:continue
                    attempted.add(day);weather=candidates[day]
                    row=dict(positive);row.update(weather)
                    row.update(event_id=positive['event_id']+':control:'+day,event_date=day,label=0,
                        event_record_type='temporal_absence_sample',label_quality='temporal_inventory_absence_pseudo_negative_unverified',
                        label_source='Inventory absence at same source location; external weather/seismic screen, not surveyed intact ground',
                        label_note='Selected from a declared pre-event date search; synthetic=False does not mean field-verified absence',
                        is_synthetic=False,sample_weight=1.,control_parent_event_id=positive['event_id'],
                        control_original_date=target.isoformat(),control_date_selection='nearest valid pre-event candidate to nominal tminus60; existing archive then fixed offsets',
                        pga_attenuation_proxy_g=0.,usgs_source_status='live_provisional_not_verified')
                    recompute(row,terrain)
                    if not quiescent_control(row):continue
                    fields={k:row[k] for k in EventCandidate.__dataclass_fields__ if k in row}
                    seismic=await fetch_usgs(client,EventCandidate(**fields));attempt_count+=1
                    row.update(seismic);recompute(row,terrain)
                    if quiescent_control(row):valid.append(row)
            await try_candidates()
            # Additional fixed seasonal offsets; no score-dependent or label-dependent features.
            for offset in (365,330,270,180,120,60,30):
                if len(valid)>=2:break
                fields={k:positive[k] for k in EventCandidate.__dataclass_fields__ if k in positive}
                event=EventCandidate(**fields);event=replace(event,event_date=(origin-timedelta(days=offset)).isoformat())
                weather=await acquire_archive(client,event,search_days=7)
                key=weather.get('weather_archive_cache_key');path=CACHE/(key+'.json') if key else None
                if path and path.exists():add_payload(json.loads(path.read_text()),key)
                await try_candidates()
            unique=[];seen=set()
            for row in valid:
                if row['event_date'] not in seen:unique.append(row);seen.add(row['event_date'])
            valid=unique[:2]
            checkpoint.write_text(json.dumps({'controls':valid,'offsets':[365,330,270,180,120,60,30]}))
            for row in valid:
                if row['event_id'] not in by_id:new.append(row)
            if old and old['event_id'] in frozen_ids:
                revised=dict(old)
                if not quiescent_control(old) and valid:
                    revised=dict(min(valid,key=lambda r:(abs((date.fromisoformat(r['event_date'])-date.fromisoformat(old['event_date'])).days),r['event_date'])))
                    revised.update(event_id=old['event_id'],original_holdout_control_date=old['event_date'],holdout_control_date_revised=True)
                repaired[old['event_id']]=revised
            diagnostics.append({'positive_id':positive['event_id'],'group':positive['event_group_id'],'valid_controls':len(valid),
                'dates':[r['event_date'] for r in valid],'candidate_days_checked':len(attempted),'seismic_requests':attempt_count,
                'scope':'Searched absence controls; no known stability observation'})
            if len(diagnostics)%25==0:print('CONTROL SEARCH',len(diagnostics),'/',len(positives),'new',len(new),flush=True)
    await asyncio.gather(*(search(p) for p in positives))
    (data_dir/'control_search_v13_report.json').write_text(json.dumps({'rows':diagnostics,'positive_rows':len(positives),
        'with_two_controls':sum(r['valid_controls']>=2 for r in diagnostics),'unfilled_positive_ids':[r['positive_id'] for r in diagnostics if r['valid_controls']<1],
        'scope':'Selected temporal absences, not independently observed stable ground'},indent=2))
    return new,repaired
