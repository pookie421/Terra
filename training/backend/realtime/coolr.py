"""NASA regional envelope count plus independent cached-catalog geodesic telemetry."""
import math
import asyncio
from datetime import datetime, timezone
from pathlib import Path
URL='https://gis.earthdata.nasa.gov/portal/rest/services/Landslides/COOLR_Events_Points/FeatureServer/0/query'

def haversine_km(lat,lon,other_lat,other_lon):
    a,b=math.radians(lat),math.radians(other_lat)
    h=math.sin((b-a)/2)**2+math.cos(a)*math.cos(b)*math.sin(math.radians(other_lon-lon)/2)**2
    return 6371.0088*2*math.asin(math.sqrt(min(1.,max(0.,h))))

def local_catalog_distances(lat,lon,path=None):
    path=path or Path(__file__).resolve().parents[2]/'data/real_global_landslide_observations.parquet'
    distances={}
    quality='partial_local_catalog'
    if path.exists():
        try:
            import pyarrow.parquet as pq
            for r in pq.read_table(path).to_pylist():
                if 'COOLR' not in str(r.get('source_name')) or r.get('latitude') is None or r.get('event_record_type') not in ('cited_manual_point_buffer_cell','reported_landslide_point_proxy'):continue
                key=r.get('source_event_id') or r['event_id']
                distance=haversine_km(lat,lon,r['latitude'],r['longitude'])
                distances[key]=min(distance,distances.get(key,float('inf')))
        except (ImportError,OSError,ValueError):quality='local_catalog_unreadable'
    else:quality='local_catalog_absent'
    return {'nasa_coolr_regional_count_100km':sum(d<=100 for d in distances.values()),
            'nearest_coolr_event_km':min(distances.values()) if distances else None,
            'cached_count_50km':sum(d<=50 for d in distances.values()),
            'cached_unique_source_records':len(distances),'local_catalog_quality':quality,
            'local_distance_basis':'Haversine to cached point/cell coordinates; buffer proxies deduplicated by source ID',
            'complete_global_catalog':False}

async def nearby_count(client,lat,lon,cache):
    local=local_catalog_distances(lat,lon)
    params={'where':'1=1','geometry':f'{lon-.45},{lat-.45},{lon+.45},{lat+.45}',
            'geometryType':'esriGeometryEnvelope','inSR':4326,'spatialRel':'esriSpatialRelIntersects',
            'returnCountOnly':'true','f':'json'}
    async def fetch():
        response=await client.get(URL,params=params,timeout=10);response.raise_for_status();payload=response.json()
        count=payload.get('count')
        if type(count) is not int or count<0:raise ValueError('NASA returned no valid count')
        return {'count':count,'quality':'NASA_live_query_or_TTL_cache','source_url':str(response.url),
                'fetched_at_utc':datetime.now(timezone.utc).isoformat(),
                'geometry':'plus/minus 0.45 degree envelope; not an exact 50 km circle',
                'query':params,'raw_count_response':payload}
    try:result=await asyncio.wait_for(cache.get_or_create(('coolr-envelope-v3',round(lat,4),round(lon,4)),3600,fetch),timeout=2.0)
    except Exception as exc:
        result={'count':local['cached_count_50km'],'quality':'partial_local_catalog_fallback_50km',
                'live_error_type':type(exc).__name__,'source_url':URL,'query':params}
    return {**result,**local}
