"""Time-matched reanalysis features; missing hours never count as freeze-thaw events."""
import hashlib
import json
import asyncio
from datetime import date, timedelta
from pathlib import Path
import numpy as np

ARCHIVE='https://archive-api.open-meteo.com/v1/archive'
CACHE=Path(__file__).resolve().parents[2]/'data/source_catalogs/historical_weather_v11'


async def archive_request(client,params):
    # Respect shared provider capacity across concurrent event enrichments.
    if not hasattr(client,'_terra_archive_lock'):
        client._terra_archive_lock=asyncio.Lock()
        client._terra_archive_next=0.
    for attempt in range(3):
        async with client._terra_archive_lock:
            loop=asyncio.get_running_loop()
            await asyncio.sleep(max(0.,client._terra_archive_next-loop.time()))
            client._terra_archive_next=loop.time()+.65
        response=await client.get(ARCHIVE,params=params,timeout=20.)
        if response.status_code!=429:return response
        await asyncio.sleep(20.*(attempt+1))
    return response


def zero_crossings(values):
    """Count sign changes through zero; missing samples break continuity."""
    count=0;last=None
    for value in values:
        if value is None or not np.isfinite(value):last=None;continue
        sign=np.sign(value)
        if sign==0:continue
        if last is not None and sign!=last:count+=1
        last=sign
    return count


def summarize_archive(data, event_day):
    hourly=data.get('hourly') or {};times=hourly.get('time') or []
    day=date.fromisoformat(event_day);start=(day-timedelta(days=7)).isoformat()
    event=[i for i,t in enumerate(times) if t[:10]==event_day]
    prior=[i for i,t in enumerate(times) if start<=t[:10]<event_day]
    window=[i for i,t in enumerate(times) if (day-timedelta(days=2)).isoformat()<=t[:10]<=event_day]
    week=[i for i,t in enumerate(times) if (day-timedelta(days=6)).isoformat()<=t[:10]<=event_day]
    def vals(key,indices):
        raw=hourly.get(key) or []
        return [float(raw[i]) if i<len(raw) and raw[i] is not None and np.isfinite(raw[i]) else None for i in indices]
    def present(key,indices):return [v for v in vals(key,indices) if v is not None]
    def mean(key,indices):
        v=present(key,indices);return float(np.mean(v)) if v else None
    def complete_sum(key,indices):
        v=present(key,indices);return float(sum(v)) if len(v)==len(indices)==24 else None
    rain=complete_sum('rain',event)
    def complete_values(key, indices, hours):
        values=present(key,indices)
        # Reject gaps, duplicates and out-of-order timestamps as well as nulls.
        from datetime import datetime
        stamps=[datetime.fromisoformat(times[i]) for i in indices]
        contiguous=len(stamps)==hours and all(b-a==timedelta(hours=1) for a,b in zip(stamps,stamps[1:]))
        return values if len(values)==hours and contiguous else None
    rain72=complete_values('rain',window,72)
    rainweek=complete_values('rain',week,168)
    peak=float(np.max(np.convolve(rain72,np.ones(24),mode='valid'))) if rain72 is not None else None
    total=float(sum(rainweek)) if rainweek is not None else None
    effective=peak+.15*max(0.,total-peak) if peak is not None and total is not None else None
    def peak72(key):
        values=complete_values(key,window,72)
        return max(values) if values is not None else None
    air=vals('temperature_2m',prior);soil=vals('soil_temperature_0_to_7cm',prior)
    soil_complete=len(soil)==168 and all(v is not None for v in soil)
    air_complete=len(air)==168 and all(v is not None for v in air)
    cycles=zero_crossings(soil if soil_complete else air) if soil_complete or air_complete else None
    tday=mean('temperature_2m',event);tprior=mean('temperature_2m',prior)
    snow=present('snow_depth',event)
    return {'weather_source_status':'open_meteo_archive_reanalysis',
        'weather_observation_start':times[0] if times else None,'weather_observation_end':times[-1] if times else None,
        'rain_event_day_mm':rain,'rain_event_day_quality':'liquid_rain_24_hour_sum' if rain is not None else 'missing_or_incomplete_liquid_rain',
        'rain_peak_24h_3d_mm':peak,'rain_7d_total_mm':total,'rain_effective_storm_mm':effective,
        'storm_window_quality':'complete_72h_peak_plus_15pct_7d_remainder' if effective is not None else 'unavailable_incomplete_storm_window',
        'snow_depth_72h_max_m':peak72('snow_depth'),
        'soil_moisture_72h_max_m3m3':peak72('soil_moisture_0_to_7cm'),
        'precipitation_event_day_mm':complete_sum('precipitation',event),
        'rain_antecedent_7d_mm':float(sum(present('rain',prior))) if len(present('rain',prior))==168 else None,
        'snowfall_event_day_mm':complete_sum('snowfall',event),
        'snow_depth_event_day_max_m':max(snow) if len(snow)==24 else None,
        'temperature_event_day_mean_c':tday,'temperature_7d_mean_c':tprior,
        'soil_temperature_event_day_mean_c':mean('soil_temperature_0_to_7cm',event),
        'soil_moisture_event_day_mean_m3m3':mean('soil_moisture_0_to_7cm',event),
        'soil_moisture_deep_event_day_mean_m3m3':mean('soil_moisture_28_to_100cm',event),
        'freeze_thaw_cycles_7d':cycles,
        'freeze_thaw_air_crossings_7d':zero_crossings(air) if air_complete else None,
        'freeze_thaw_soil_crossings_7d':zero_crossings(soil) if soil_complete else None,
        'freeze_thaw_quality':'soil_reanalysis_168_hours' if soil_complete else 'air_reanalysis_168_hours_proxy' if air_complete else 'unavailable_incomplete_7d',
        'warming_delta_c':max(0.,tday-tprior) if tday is not None and tprior is not None else None}


async def acquire_archive(client,event,search_days=0):
    if not event.event_date:return {'weather_source_status':'not_run_no_event_date'}
    day=date.fromisoformat(event.event_date)
    params={'latitude':event.latitude,'longitude':event.longitude,'start_date':(day-timedelta(days=7+search_days)).isoformat(),
        'end_date':day.isoformat(),'timezone':'UTC',
        'hourly':'precipitation,rain,snowfall,snow_depth,temperature_2m,soil_temperature_0_to_7cm,soil_moisture_0_to_7cm,soil_moisture_28_to_100cm'}
    key=hashlib.sha256(json.dumps(params,sort_keys=True).encode()).hexdigest();path=CACHE/(key+'.json')
    try:
        if path.exists():payload=json.loads(path.read_text());acquisition='retained_archive_response'
        else:
            try:
                response=await archive_request(client,params);response.raise_for_status()
            except Exception:
                if 'response' in locals() and response.status_code==429:raise
                params['hourly']='precipitation,rain,snowfall,snow_depth,temperature_2m'
                response=await archive_request(client,params);response.raise_for_status()
            payload={'request':params,'url':str(response.url),'response':response.json()}
            if 'hourly' not in payload['response']:raise ValueError('Archive response lacks hourly data')
            CACHE.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(payload));acquisition='fresh_archive_response'
        selected=day.isoformat()
        if search_days:
            candidates=[summarize_archive(payload['response'],(day-timedelta(days=i)).isoformat()) for i in range(search_days+1)]
            eligible=[(c,(day-timedelta(days=i)).isoformat()) for i,c in enumerate(candidates)
                if c['rain_effective_storm_mm'] is not None and c['rain_effective_storm_mm']<=35 and c['snow_depth_72h_max_m']==0]
            # Nearest eligible pre-offset day, not the driest day in the window.
            if eligible:selected=max(eligible,key=lambda x:x[1])[1]
        result=summarize_archive(payload['response'],selected)
        result.update(weather_source_url=payload['url'],weather_archive_cache_key=key,weather_acquisition=acquisition,weather_feature_version='hourly_v12',weather_selected_date=selected)
        return result
    except Exception as exc:
        result=summarize_archive({},event.event_date)
        result.update(weather_source_status='unavailable: '+type(exc).__name__+': '+str(exc),weather_feature_version='hourly_v12_failed')
        return result
