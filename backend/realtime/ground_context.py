"""Mapped geology and modeled river flow; neither is a local ground measurement."""
from datetime import datetime,timezone
import math

async def river_discharge(client,lat,lon):
    url='https://flood-api.open-meteo.com/v1/flood'
    response=await client.get(url,params=dict(latitude=lat,longitude=lon,daily='river_discharge,river_discharge_max',past_days=7,forecast_days=7,timezone='UTC',cell_selection='land'))
    response.raise_for_status();data=response.json();daily=data['daily']
    today=datetime.now(timezone.utc).date().isoformat();dates=daily['time'];i=dates.index(today)
    q=daily['river_discharge'][i];past=[v for v in daily['river_discharge'][max(0,i-7):i] if v is not None and math.isfinite(v)]
    if q is None or not math.isfinite(q) or q<0 or len(past)==0 or min(past)<0:raise ValueError('Missing discharge or available past-week history')
    forecast=[{'date':d,'discharge_m3_s':v,'surge_ratio':min(5.,max(1.,v/max(min(past),1.)))} for d,v in zip(dates[i:i+7],daily['river_discharge'][i:i+7]) if v is not None and math.isfinite(v) and v>=0]
    return dict(history_days_available=len(past),forecast_7d=forecast,discharge_m3_s=q,surge_ratio=min(5.,max(1.,q/max(min(past),1.))),date=today,source_url=url,quality='GloFAS modeled 5 km discharge; not gauged local flow')

async def bedrock(client,lat,lon):
    url='https://macrostrat.org/api/v2/geologic_units/map'
    response=await client.get(url,params=dict(lat=lat,lng=lon,response='long'));response.raise_for_status()
    records=response.json().get('success',{}).get('data',[])
    if not records:raise ValueError('No mapped geology returned')
    unit=records[0];lith=unit.get('lith') or unit.get('lithology') or ''
    return dict(name=unit.get('name'),lithology=lith,age=unit.get('age') or {'early':unit.get('t_age'),'late':unit.get('b_age')},source_url=url,quality='Descriptive mapped bedrock context only; no regolith strength adjustment',raw_unit=unit)
