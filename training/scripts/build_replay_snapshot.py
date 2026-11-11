"""Create a small, transparent historical replay from retained local sources."""
import json,hashlib,sys
from dataclasses import asdict
from datetime import datetime,timedelta,timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from unittest.mock import patch
import numpy as np
from backend.realtime.global_ingestor import GlobalIngestor,GlobalScan,LocationInfo,SourceStatus
from backend.realtime.dem_processor import TerrainScan,TerrainCell


def build_replay(root):
    import pyarrow.parquet as pq
    from scripts.train_rainfall_baseline import load_terrains
    root=Path(root);data=root/'data';rows=pq.read_table(data/'real_global_landslide_observations.parquet').to_pylist()
    row=next(r for r in rows if r['event_id']=='chooralmala_2024')
    terrain,_=load_terrains(data)[(row['latitude'],row['longitude'])]
    # Fixed nearest 25 cells; no selection by slope, model outcome or probability.
    cells=sorted(terrain['cells'],key=lambda c:((c['latitude']-row['latitude'])**2+
                 (c['longitude']-row['longitude'])**2,c['cell_id']))[:25]
    cells={c['cell_id']:TerrainCell(**c) for c in cells}
    elevations=[c.elevation_m for c in cells.values()]
    scan_terrain=TerrainScan(center_latitude=row['latitude'],center_longitude=row['longitude'],radius_km=1.2,
        resolution=9,zoom=13,pixel_spacing_x_m=0.,pixel_spacing_y_m=0.,tile_count=len(terrain.get('tile_ids',[])),
        tile_ids=terrain.get('tile_ids',[]),h3_cell_count=len(cells),min_elevation_m=min(elevations),max_elevation_m=max(elevations),
        slope_p95_deg=float(np.percentile([c.slope_deg for c in cells.values()],95)),max_twi=max(c.twi for c in cells.values()),
        interpolated_pixels=0,cells=cells,warnings=['Retained terrain sampled after the event; not a verified pre-event DEM'])
    path=data/'source_catalogs/historical_weather_v11'/f"{row['weather_archive_cache_key']}.json"
    payload=json.loads(path.read_text());cutoff=datetime.fromisoformat(row['event_date']).replace(tzinfo=timezone.utc)+timedelta(hours=23)
    class FrozenClock(datetime):
        @classmethod
        def now(cls,tz=None):return cutoff if tz else cutoff.replace(tzinfo=None)
    ingestor=GlobalIngestor.__new__(GlobalIngestor)
    with patch('backend.realtime.global_ingestor.datetime',FrozenClock):weather=ingestor._parse_weather(payload['response'],product='archive')
    hydrology=ingestor._combine_hydrology(weather,None,None,scan_terrain)
    sources={'open_meteo':SourceStatus(True,'Retained Open-Meteo event-day reanalysis',cutoff.isoformat(),mode='REPLAY'),
             'terrain':SourceStatus(True,'Retained Terrarium terrain neighborhood',cutoff.isoformat(),mode='REPLAY'),
             'soilgrids':SourceStatus(False,'Static terrain soil prior',cutoff.isoformat(),'No measured soil in this replay',mode='UNAVAILABLE')}
    for name,provider in [('nasa_power','NASA POWER'),('coolr','NASA COOLR'),('eonet','NASA EONET')]:
        sources[name]=SourceStatus(False,provider,cutoff.isoformat(),'No event-time payload included in this replay; source contribution is documented separately',mode='UNAVAILABLE')
    snapshot=GlobalScan(location=LocationInfo('Chooralmala historical replay',row['latitude'],row['longitude'],'Chooralmala / 30 July 2024','India'),
        radius_km=1.2,generated_at_utc=cutoff.isoformat(),terrain=scan_terrain,weather=weather,hydrology=hydrology,seismic=None,
        infrastructure=None,eonet_events=[],satellite_cells={},sources=sources,elapsed_seconds=0.)
    output={'provenance':{'label':'Chooralmala event-day example / 30 July 2024',
        'scope':'Retrospective event-day replay, not an untouched benchmark or an advance warning. Historical weather with retained later terrain; no invented NASA live feeds.',
        'weather_payload_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'source_event_id':row['event_id'],
        'selection':'Nearest 25 retained cells; selected before inspecting new model results'},'snapshot':asdict(snapshot)}
    target=data/'replays/chooralmala_2024.json';target.parent.mkdir(exist_ok=True)
    target.write_text(json.dumps(output,indent=2,allow_nan=False)+'\n')
    return target

if __name__=='__main__':print(build_replay(Path(__file__).resolve().parents[1]))
