"""Whitelisted retained inputs for stateful replay; never acquires providers."""
import hashlib,json
from dataclasses import replace
from pathlib import Path
from scripts.snapshot_replay import hydrate
from backend.realtime.global_ingestor import GlobalScan
from backend.realtime.dem_processor import TerrainScan,TerrainCell,TERRAIN_VERSION
ROOT=Path(__file__).resolve().parents[2]

def load_replay(name,root=ROOT):
    paths={'monaco':root/'tests/fixtures/monaco_physics_context.json',
           'wayanad':root/'data/replays/chooralmala_2024.json'}
    path=paths.get(name)
    if path is None or not path.exists():raise KeyError('Retained snapshot is not installed')
    raw=hydrate(GlobalScan,json.loads(path.read_text())['snapshot'])
    if not raw.location.display_name:
        raw.location=replace(raw.location,display_name='Monaco / coastal terrain' if name=='monaco' else 'Wayanad / monsoon terrain')
    key=hashlib.sha256(json.dumps([raw.location.latitude,raw.location.longitude]).encode()).hexdigest()
    terrain_path=root/'data/source_catalogs/terrain_metric_v1'/(key+'.json')
    if terrain_path.exists():
        d=json.loads(terrain_path.read_text())
        if d['terrain_version']!=TERRAIN_VERSION:raise ValueError('Replay terrain version mismatch')
        raw.terrain=TerrainScan(**{**d['scan'],'cells':{c['cell_id']:TerrainCell(**c) for c in d['cells']},'terrain_version':TERRAIN_VERSION})
    raw.scan_id=None
    return raw
