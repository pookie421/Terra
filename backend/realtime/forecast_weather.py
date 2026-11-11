"""UTC-aligned forecast ingestion; missing observations never become dry hours."""
from datetime import datetime, timezone
import math
import numpy as np

SNOW_WATER_MM_PER_CM = 10.0 / 7.0
ENSEMBLE_URL = 'https://ensemble-api.open-meteo.com/v1/ensemble'


def finite(value):
    return value is not None and isinstance(value, (int, float)) and math.isfinite(value)


def liquid_rain(hourly, index):
    def value(key):
        items = hourly.get(key) or []
        return items[index] if index < len(items) else None
    rain, showers = value('rain'), value('showers')
    if finite(rain) and finite(showers):
        return max(0., rain + showers)
    precip, snow = value('precipitation'), value('snowfall')
    # Open-Meteo snowfall is cm; 7 cm snow equals 10 mm precipitation water equivalent.
    if finite(precip) and finite(snow):
        return max(0., precip - SNOW_WATER_MM_PER_CM * snow)
    return None


def future_hours(hourly, now):
    rows = []
    keys = ('precipitation','rain','showers','snowfall','snow_depth','temperature_2m',
            'soil_temperature_0_to_7cm','soil_moisture_0_to_7cm','soil_moisture_28_to_100cm')
    for i, text in enumerate(hourly.get('time') or []):
        stamp = datetime.fromisoformat(text).replace(tzinfo=timezone.utc)
        if stamp <= now:
            continue
        row = {'time': stamp.isoformat(), 'liquid_rain_mm': liquid_rain(hourly,i)}
        for key in keys:
            values = hourly.get(key) or []
            v = values[i] if i < len(values) else None
            row[key] = v if finite(v) else None
        rows.append(row)
        if len(rows) == 72:
            break
    return rows


async def ensemble_forecast(client, lat, lon):
    params = dict(latitude=lat,longitude=lon,models='icon_seamless,gfs_seamless',
                  hourly='precipitation,temperature_2m',forecast_days=5,timezone='UTC')
    response = await client.get(ENSEMBLE_URL, params=params)
    if response.status_code == 400:
        params['models'] = 'icon_seamless_eps,ncep_gefs_seamless'
        response = await client.get(ENSEMBLE_URL, params=params)
    response.raise_for_status()
    data = response.json()
    records = data if isinstance(data,list) else [data]
    by_time = {}
    for record in records:
        hourly = record.get('hourly') or {}
        for i, stamp in enumerate(hourly.get('time') or []):
            row = by_time.setdefault(stamp, {'precipitation':[], 'temperature_2m':[]})
            for key, values in hourly.items():
                for variable in row:
                    if key.startswith(variable) and i < len(values) and finite(values[i]):
                        row[variable].append(float(values[i]))
    result = {}
    for stamp, row in by_time.items():
        if len(row['precipitation']) >= 2:
            result[stamp] = {'precip_p50_mm':float(np.quantile(row['precipitation'],.5)),
                            'precip_p90_mm':float(np.quantile(row['precipitation'],.9)),
                            'member_count':len(row['precipitation']),
                            'temperature_p50_c':float(np.median(row['temperature_2m'])) if row['temperature_2m'] else None}
    if not result:
        raise ValueError('Ensemble members unavailable')
    return {'hours':result,'models':params['models'],'source_url':ENSEMBLE_URL,
            'quality':'Pooled hourly member quantiles; P90 is not a coherent member or a worst-case probability'}


def precipitation_paths(hours, ensemble):
    """Use total-precipitation quantiles times deterministic liquid fraction."""
    p50, p90, modes = [], [], []
    for row in hours:
        liquid = row['liquid_rain_mm']
        if not finite(liquid):
            raise ValueError('Missing hourly liquid rain; forecast cannot assume zero')
        key = row['time'][:16]
        spread = (ensemble or {}).get('hours',{}).get(key)
        total = row.get('precipitation')
        if spread and finite(total) and total > 0:
            fraction = min(1., liquid/total)
            p50.append(max(0.,spread['precip_p50_mm']*fraction))
            p90.append(max(0.,spread['precip_p90_mm']*fraction))
            modes.append('ENSEMBLE_WITH_DETERMINISTIC_LIQUID_FRACTION')
        elif spread and liquid == 0 and finite(row.get('temperature_2m')):
            # No deterministic precipitation: phase partition is an explicit temperature proxy.
            fraction = float(row['temperature_2m'] > 0)
            p50.append(max(0.,spread['precip_p50_mm']*fraction))
            p90.append(max(0.,spread['precip_p90_mm']*fraction))
            modes.append('ENSEMBLE_WITH_TEMPERATURE_PHASE_PROXY')
        else:
            p50.append(liquid); p90.append(liquid)
            modes.append('DETERMINISTIC_ONLY_ENSEMBLE_QUANTILES_UNAVAILABLE')
    return np.asarray(p50),np.asarray(p90),modes
