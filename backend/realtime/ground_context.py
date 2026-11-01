"""Mapped geology and modeled river flow; neither is a local ground measurement."""
from datetime import datetime,timezone
import math

async def river_discharge(client,lat,lon):
    url='https://flood-api.open-meteo.com/v1/flood'
    response=await client.get(url,params=dict(latitude=lat,longitude=lon,daily='river_discharge,river_discharge_max',past_days=7,forecast_days=3,timezone='UTC'))
    response.raise_for_status();data=response.json();daily=data['daily']
    today=datetime.now(timezone.utc).date().isoformat();dates=daily['time'];i=dates.index(today)
    q=daily['river_discharge'][i];past=[v for v in daily['river_discharge'][max(0,i-7):i] if v is not None and math.isfinite(v)]
    if q is None or not math.isfinite(q) or q<0 or len(past)!=7 or min(past)<0:raise ValueError('Missing discharge or complete seven-day history')
    return dict(discharge_m3_s=q,surge_ratio=min(5.,max(1.,q/max(min(past),1.))),date=today,source_url=url,quality='GloFAS modeled 5 km discharge; not gauged local flow')

async def bedrock(client,lat,lon):
    url='https://macrostrat.org/api/v2/geologic_units/map'
    response=await client.get(url,params=dict(lat=lat,lng=lon,response='long'));response.raise_for_status()
    records=response.json().get('success',{}).get('data',[])
    if not records:raise ValueError('No mapped geology returned')
    unit=records[0];lith=unit.get('lith') or unit.get('lithology') or ''
    text=str(lith).lower();weak=any(k in text for k in ('shale','mudstone','phyllite','tuff'));strong=any(k in text for k in ('granite','basalt'))
    adjustment=-1.5 if weak and not strong else 1.5 if strong and not weak else 0.
    return dict(name=unit.get('name'),lithology=lith,age=unit.get('age') or {'early':unit.get('t_age'),'late':unit.get('b_age')},source_url=url,friction_adjustment_deg=adjustment,cohesion_adjustment_kpa=adjustment,quality='Mapped unit; requested strength adjustment is an uncalibrated scenario proxy',raw_unit=unit)
