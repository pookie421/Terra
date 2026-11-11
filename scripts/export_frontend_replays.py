"""Export curated, precomputed UI responses without environmental acquisition."""
import asyncio,copy,hashlib,json,socket,sys
from pathlib import Path
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from backend.api.replay import load_replay
from backend.api.scan_service import GlobalScanService
from backend.api.serialization import sanitize_json


def curate(response):
    r=copy.deepcopy(response)
    for key in ('predictive_forecast_72h','cell_comparisons'):r.pop(key,None)
    r['snapshot_note']='Curated retained response; forecasts and duplicate comparison rows omitted. Not current conditions.'
    for status in r['source_status'].values():
        status['original_mode']=status.get('mode');status['mode']='RETAINED' if status.get('available') else 'RETAINED_MISSING'
    for key in ('h3_geojson','terrain_geojson'):
        for f in r.get(key,{}).get('features',[]):
            f['properties']={k:v for k,v in f['properties'].items() if k in ('h3_index','latitude','longitude','slope_deg','elevation_m','fs_mid','fs_low','fs_high','score','physics_informed_score')}
    for c in r.get('physics_informed_assessment',{}).get('cells',[]):
        if c.get('model_explanation'):c['model_explanation'].pop('folds',None)
    return sanitize_json(r)


async def run():
    target=ROOT/'artifacts/frontend_replays';target.mkdir(parents=True,exist_ok=True);manifest={}
    for name in ('monaco','wayanad'):
        service=GlobalScanService();raw=load_replay(name)
        model_hash=hashlib.sha256((ROOT/'artifacts/physics_informed_models.pkl').read_bytes()).hexdigest()[:12]
        raw.scan_id='retained-'+name+'-'+model_hash
        baseline=await service.scan(_snapshot=raw)
        routes={c['h3_index']:service.trace_runout(c['h3_index'],baseline['scan_id'],baseline['scenario_id']) for c in baseline['physical_assessment']['cells']}
        (target/f'{name}-runouts.json').write_text(json.dumps(sanitize_json(routes),separators=(',',':'))+'\n')
        for rain in (0,1,10,50,100,300):
            scan=baseline if rain==0 else await service.apply_scenario(baseline['scan_id'],baseline['scenario_id'],{'rain_delta_mm':rain})
            payload=curate(scan);path=target/f'{name}-{rain}.json';path.write_text(json.dumps(payload,separators=(',',':'),allow_nan=False)+'\n')
            manifest[path.name]={'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'cells':len(payload['physical_assessment']['cells']),
                'model_version':payload.get('physics_informed_assessment',{}).get('version'),'rain_delta_mm':rain}
        print('Exported retained response scenarios:',name,flush=True)
    (target/'manifest.json').write_text(json.dumps({'scope':'Precomputed retained responses, no fresh provider claims','files':manifest},indent=2)+'\n')

if __name__=='__main__':
    with patch.object(socket.socket,'connect',side_effect=AssertionError('Replay export is offline')):asyncio.run(run())
