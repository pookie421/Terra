"""Shared, cutoff-bound source features; raw moisture is never physical saturation."""
from datetime import datetime,timedelta,timezone
import math
import numpy as np

VERSION='rainfall_source_windows_v2'
FEATURES=('rain_24h_mm','rain_prior_6d_mm','soil_moisture_0_7cm_mean_m3m3',
          'slope_deg','twi','profile_curvature')


def utc(value):
    stamp=datetime.fromisoformat(str(value).replace('Z','+00:00')) if not isinstance(value,datetime) else value
    return stamp.replace(tzinfo=timezone.utc) if stamp.tzinfo is None else stamp.astimezone(timezone.utc)


def finite(value):
    try:return value is not None and math.isfinite(float(value))
    except (ValueError,TypeError):return False


def source_window(hourly, cutoff, *, product):
    """168 complete preceding-hour sums ending at a UTC hour boundary.

    Archives use 23:00 of the labeled UTC day: retrospective event-day
    classification, not a prediction made before the landslide. Live inference
    uses the current hour boundary. The underlying reanalysis/forecast products
    still differ, even though definitions and extraction code are shared.
    """
    cutoff=utc(cutoff).replace(minute=0,second=0,microsecond=0)
    times=hourly.get('time') or []
    if product not in ('archive','forecast'):
        raise ValueError('Declare archive or forecast rain semantics')
    expected=[cutoff-timedelta(hours=i) for i in range(167,-1,-1)]
    by_time={};duplicates=set()
    for i,value in enumerate(times):
        try:stamp=utc(value)
        except (ValueError,TypeError):continue
        if stamp in by_time:duplicates.add(stamp)
        by_time[stamp]=i
    indices=[by_time.get(t) for t in expected]
    contiguous=all(i is not None for i in indices) and not duplicates.intersection(expected)
    def values(key,chosen,minimum=0.,maximum=None):
        raw=hourly.get(key) or []
        result=[]
        for i in chosen:
            v=raw[i] if i is not None and i<len(raw) else None
            if not finite(v) or float(v)<minimum or (maximum is not None and float(v)>maximum):return None
            result.append(float(v))
        return result
    rain=values('rain',indices) if contiguous else None
    # Archive rain includes convection; forecast rain and showers are separate.
    # Missing showers cannot silently mean zero or archival rain semantics.
    if product=='forecast':
        showers=values('showers',indices) if contiguous else None
        rain=[a+b for a,b in zip(rain,showers)] if rain is not None and showers is not None else None
    moisture=values('soil_moisture_0_to_7cm',indices[-24:],maximum=1.) if contiguous else None
    return {'version':VERSION,'cutoff_utc':cutoff.isoformat(),
        'status':'AVAILABLE' if rain is not None else 'UNAVAILABLE',
        'reason':None if rain is not None else '168 contiguous complete hours of nonnegative liquid rain are required through the cutoff (forecast requires rain and showers)',
        'product':product,'rain_definition':'archive rain' if product=='archive' else 'forecast rain + showers',
        'rain_24h_mm':sum(rain[-24:]) if rain is not None else None,
        'rain_prior_6d_mm':sum(rain[:-24]) if rain is not None else None,
        'liquid_rain_hourly_mm':rain,
        'soil_moisture_0_7cm_mean_m3m3':sum(moisture)/24 if moisture is not None else None,
        'moisture_quality':'raw_provider_volumetric_0_to_7cm_not_saturation' if moisture is not None else 'missing_optional_raw_moisture',
        'window':'Preceding-hour stamps in (cutoff-168h, cutoff]; latest 24h rain and nonoverlapping preceding 144h rain',
        'water_state_scope':'No division by estimated porosity; not groundwater or absolute saturation'}


def feature_vector(window, slope, twi, curvature):
    if window.get('version')!=VERSION or window.get('status')!='AVAILABLE':
        raise ValueError('Matching, complete source weather window required')
    values=[window.get(k) for k in FEATURES[:3]]+[slope,twi,curvature]
    for i in (0,1,3,4,5):
        if not finite(values[i]):raise ValueError('Missing required source feature: '+FEATURES[i])
    if not 0<=float(slope)<90 or float(twi)<=0 or min(float(values[0]),float(values[1]))<0:
        raise ValueError('Invalid source terrain or rain')
    water=values[2]
    if water is not None and (not finite(water) or not 0<=float(water)<=1):
        raise ValueError('Raw volumetric moisture must be in [0,1] or missing')
    return np.asarray([np.nan if v is None else float(v) for v in values],dtype=np.float64)
