#!/usr/bin/env python3
"""Rebuild retained locations from decoded Terrarium tiles, with immutable raw tile cache.

Only --fetch permits network; no weather/event/provider scans are performed.
Original derived caches and labels are retained. Every result records its version.
"""
import argparse,asyncio,hashlib,json,sys
from dataclasses import asdict
from pathlib import Path
import httpx
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from backend.realtime.dem_processor import DEMProcessor,TERRAIN_VERSION


async def rebuild(root=ROOT,fetch=False):
    source=root/'data/source_catalogs';target=source/'terrain_metric_v1';target.mkdir(exist_ok=True)
    originals=[p for p in sorted(source.glob('terrain_v13_*.json')) if 'center' in json.loads(p.read_text())]
    # Include exact retained Monaco viewport, not a new location search.
    monaco=root/'artifacts/monaco_input_audit/provider_snapshot.json'
    jobs=[(p,json.loads(p.read_text())) for p in originals]
    if monaco.exists():
        s=json.loads(monaco.read_text())['snapshot'];jobs.append((monaco,dict(center=[s['location']['latitude'],s['location']['longitude']],radius_km=3.5)))
    sem=asyncio.Semaphore(3);dem=DEMProcessor(tile_concurrency=8);failures=[];completed=[]
    class OfflineTransport(httpx.AsyncBaseTransport):
        async def handle_async_request(self,request):raise RuntimeError('Raw tile absent; use --fetch for retained locations')
    async with httpx.AsyncClient(timeout=30.,limits=httpx.Limits(max_connections=16),transport=None if fetch else OfflineTransport(),follow_redirects=True) as client:
        async def one(path,payload):
            key=hashlib.sha256(json.dumps(payload['center']).encode()).hexdigest();out=target/(key+'.json')
            if out.exists():completed.append(str(out));return
            try:
                async with sem:scan=await dem.process(*payload['center'],payload.get('radius_km',1.2),client)
                d={'center':payload['center'],'radius_km':scan.radius_km,'terrain_version':TERRAIN_VERSION,
                    'original_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'source':'AWS Terrarium retained raw tiles; later terrain, not event-date survey',
                    'tile_ids':scan.tile_ids,'cells':[asdict(c) for c in scan.cells.values()],'scan':asdict(scan)}
                out.write_text(json.dumps(d,allow_nan=False));completed.append(str(out))
            except Exception as exc:failures.append({'source':str(path.relative_to(root)),'center':payload['center'],'error':str(exc)})
            if (len(completed)+len(failures))%20==0:print('Terrain',len(completed),'complete',len(failures),'failed of',len(jobs),flush=True)
        await asyncio.gather(*(one(p,d) for p,d in jobs))
    report={'version':TERRAIN_VERSION,'requested':len(jobs),'completed':len(completed),'failures':failures,'network_permitted':fetch}
    (root/'artifacts/metric_terrain_rebuild.json').write_text(json.dumps(report,indent=2)+'\n')
    return report

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--fetch',action='store_true');a=p.parse_args()
    r=asyncio.run(rebuild(fetch=a.fetch));print(json.dumps({k:v for k,v in r.items() if k!='failures'}));sys.exit(bool(r['failures']))
