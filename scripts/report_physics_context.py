#!/usr/bin/env python3
"""Write a compact reproducible Phase 1/2 result from retained Monaco forcing."""
import asyncio,hashlib,json,socket,sys
from datetime import datetime,timezone
from pathlib import Path
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from backend.api.scan_service import GlobalScanService
from backend.realtime.global_ingestor import GlobalScan
from scripts.snapshot_replay import hydrate


async def report():
    raw=hydrate(GlobalScan,json.loads((ROOT/'tests/fixtures/monaco_physics_context.json').read_text())['snapshot'])
    service=GlobalScanService();scenarios=[]
    baseline=await service.scan(_snapshot=raw)
    for delta in (0.,1.,10.,50.,100.,300.):
        response=baseline if delta==0 else await service.apply_scenario(baseline['scan_id'],baseline['scenario_id'],{'rain_delta_mm':delta})
        cells=[]
        for k in ('893969a6a5bffff','893969a401bffff'):
            c=service.inspect_hexagon(k,response['scan_id'],response['scenario_id'])
            ml=c['physics_informed_assessment']
            cells.append({'h3_index':k,'corrected_slope_deg':c['slope_deg'],'fs_interval':c['fs_interval'],
                'fosm_variance':c['fosm_variance'],'physics_informed_score':ml['score'],
                'ablation_scores':ml['ablation_scores'],'pressure_head_m':c['pressure_head_m']})
        scenarios.append({'added_rain_mm':delta,'scan_id':response['scan_id'],'scenario_id':response['scenario_id'],'cells':cells})
    flat=[s['cells'][0]['physics_informed_score'] for s in scenarios];cliff=[s['cells'][1]['physics_informed_score'] for s in scenarios]
    assert flat[0]<.01
    assert all(b>=a-1e-6 for a,b in zip(cliff,cliff[1:]))
    ablation=json.loads((ROOT/'artifacts/physics_informed_ablation_report.json').read_text())
    result={'generated_at_utc':datetime.now(timezone.utc).isoformat(),
        'fusion_model_version':ablation['version'],
        'fusion_model_sha256':hashlib.sha256((ROOT/'artifacts/physics_informed_models.pkl').read_bytes()).hexdigest(),
        'terrain':json.loads((ROOT/'artifacts/metric_terrain_rebuild.json').read_text()),
        'physical_coverage':json.loads((ROOT/'artifacts/context_observation_coverage.json').read_text()),
        'monaco_retained_observed_rain_mm':6.9,'monaco_scenarios':scenarios,
        'cross_validation':ablation['cross_validation'],'development_holdout':ablation['development_holdout'],
        'training_rows':ablation['training_rows'],'holdout_rows':ablation['holdout_rows'],'components':ablation['components'],
        'footprint_rows_separated':len(ablation['excluded_footprint_rows']),
        'original_monaco_comparison':{'flat_slope_deg':3.698151741709028,'flat_inventory_score':.8215765363548586,
            'cliff_slope_deg':41.11468342372349,'cliff_reference_fs':1.3816947,
            'scope':'Archived values use different terrain math and priors; they are not tuning targets or measured truth'},
        'copilot_verification':'Mocked Ollama JSON protocol + bound claims; actual weights/CUDA not exercised',
        'provider_verification':'Mocked HTTP/geospatial contracts; only retained-location DEM tiles downloaded; no global weather/event sweep',
        'operational_warning_ready':False,'independent_predictive_validation':False}
    path=ROOT/'artifacts/phase1_phase2_report.json';path.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'monaco':scenarios[0],'report':str(path)},indent=2))

if __name__=='__main__':
    with patch.object(socket.socket,'connect',side_effect=AssertionError('Offline retained regression')):
        asyncio.run(report())
