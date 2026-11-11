"""Unlabeled spatial sensitivity diagnostics from published replay responses."""
import json,hashlib
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]

def run():
    records=[]
    for name in ('monaco','wayanad'):
        for rain in (0,1,10,50,100,300):
            path=ROOT/f'artifacts/frontend_replays/{name}-{rain}.json';scan=json.loads(path.read_text())
            cells=scan['physics_informed_assessment']['cells'];byid={c['h3_index']:c for c in scan['physical_assessment']['cells']}
            def band(mask):
                selected=[r for r in cells if mask(byid[r['h3_index']])]
                return {'cells':len(selected),'models':{k:{'mean':float(np.mean([r['ablation_scores'][k] for r in selected])),
                    'maximum':max(r['ablation_scores'][k] for r in selected)} for k in ('Rainfall-Only','Physics-Only','Weather+Terrain ML','Physics-Informed ML')} if selected else {}}
            records.append({'location':name,'added_rain_mm':rain,'response_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
                'all_cells':band(lambda c:True),'low_slope_reference':band(lambda c:c['slope_deg']<6 and c['fs_low'] is not None and c['fs_low']>3),
                'steep_reference':band(lambda c:c['slope_deg']>25),
                'monaco_targets':[{**r,'reference_fs':byid[r['h3_index']]['fs_interval']} for r in cells if r['h3_index'] in ('893969a6a5bffff','893969a401bffff')]})
    result={'scope':'Unlabeled same-storm spatial consistency, not observed non-failure classification; no AUC/FPR from assumed stable neighbors',
        'new_stable_labels_created':0,'model_targets_used_for_tuning':False,'scenarios':records}
    (ROOT/'artifacts/storm_grid_diagnostics.json').write_text(json.dumps(result,indent=2)+'\n')
    print('Recorded',len(records),'unlabeled grid scenarios')
if __name__=='__main__':run()
