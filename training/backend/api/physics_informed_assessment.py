"""The shared live/training mathematical-context feature contract."""
import hashlib,json
from dataclasses import replace
from pathlib import Path
import numpy as np
from backend.ml.physics_informed import ContextModel,VERSION,matrix
from backend.ml.rainfall_features import feature_vector
from backend.realtime.dem_processor import TERRAIN_VERSION
from .ground_assessment import located_cell
ROOT=Path(__file__).resolve().parents[2]


def load_models():
    import joblib
    path=ROOT/'artifacts/physics_informed_models.pkl'
    if not path.exists():return None
    try:models=joblib.load(path)
    except (OSError,ValueError,TypeError,AttributeError,ImportError,EOFError):return None
    if any(not isinstance(m,ContextModel) or m.version!=VERSION for m in models.values()):return None
    return models


def upgrade_retained_terrain(scan):
    """Replace retained derived values only when exact raw-tile recomputation exists."""
    if not scan.terrain or scan.terrain.terrain_version==TERRAIN_VERSION:return scan
    center=[scan.terrain.center_latitude,scan.terrain.center_longitude]
    key=hashlib.sha256(json.dumps(center).encode()).hexdigest()
    path=ROOT/'data/source_catalogs/terrain_metric_v1'/(key+'.json')
    if not path.exists():return scan
    from backend.realtime.dem_processor import TerrainCell
    payload=json.loads(path.read_text())
    corrected={c['cell_id']:TerrainCell(**c) for c in payload['cells'] if c['cell_id'] in scan.terrain.cells}
    if not corrected:return scan
    terrain=replace(scan.terrain,cells=corrected,h3_cell_count=len(corrected),terrain_version=TERRAIN_VERSION,
        pixel_spacing_y_m=scan.terrain.pixel_spacing_x_m,warnings=scan.terrain.warnings+['Recomputed from retained-location raw tiles: '+TERRAIN_VERSION])
    return replace(scan,terrain=terrain)


def attach(response,scan,models,rain_delta_mm=0.):
    window=dict(getattr(scan.weather,'rainfall_source_window',{}) or {})
    assessment={'status':'UNAVAILABLE','version':VERSION,'cells':[],'hotspot':None,
        'scope':'Physics-informed experimental initiation score on the declared sampled population; not annual probability',
        'reason':None,'operational_warning_ready':False}
    response['physics_informed_assessment']=assessment
    if not models:assessment['reason']='Corrected physics-informed model artifact unavailable';return response
    if not scan.terrain or scan.terrain.terrain_version!=TERRAIN_VERSION:
        assessment['reason']='Corrected metric terrain required; legacy features cannot enter this model';return response
    if window.get('status')!='AVAILABLE':assessment['reason']='Complete source weather window required for ML; physical reference retained';return response
    window['rain_24h_mm']+=rain_delta_mm
    by_id={r['h3_index']:r for r in response['physical_assessment']['cells']}
    cells=[c for c in sorted(scan.terrain.cells.values(),key=lambda c:c.cell_id) if c.cell_id in by_id]
    x=np.asarray([matrix(feature_vector(window,c.slope_deg,c.twi,c.profile_curvature),by_id[c.cell_id]) for c in cells])
    if not len(cells):assessment['reason']='No valid physical columns';return response
    scores={k:m.predict_proba(x) for k,m in models.items()}
    explanations=models['Physics-Informed ML'].explain(x)
    records=[]
    for i,c in enumerate(cells):
        physical=by_id[c.cell_id]
        records.append({**located_cell(c,scan.location),'score':float(scores['Physics-Informed ML'][i]),
            'learned_trigger_score':explanations[i]['learned_score'],
            'mechanical_admissibility':explanations[i]['mechanical_admissibility'],
            'ablation_scores':{k:float(v[i]) for k,v in scores.items()},'fs_interval':physical['fs_interval'],
            'fosm_variance':physical['fosm_variance'],'mechanistic_features':physical['mechanistic_features'],
            'model_explanation':explanations[i]})
    top=max(records,key=lambda r:(r['score'],r['h3_index']))
    assessment.update(status='EXPERIMENTAL_PHYSICS_INFORMED',cells=records,hotspot=top)
    # The statistical comparison uses the freshly retrained no-physics ablation,
    # never the old inventory model evaluated with changed feature definitions.
    from backend.ml.rainfall_features import FEATURES
    stat=[{**r,'score':r['ablation_scores']['Weather+Terrain ML'],
        'features':dict(zip(FEATURES,[float(v) if np.isfinite(v) else None for v in x[i,:6]]))} for i,r in enumerate(records)]
    response['statistical_assessment']={'status':'EXPERIMENTAL','score_name':'weather_terrain_initiation_proxy_score',
        'cells':stat,'hotspot':max(stat,key=lambda r:r['score']),'score_max':max(r['score'] for r in stat),
        'score_min':min(r['score'] for r in stat),'score_median':float(np.median(scores['Weather+Terrain ML'])),
        'scope':'Corrected weather+terrain comparator on matched initiation-proxy labels','reason':None,'window':window}
    summary=response['summary'];same=by_id[top['h3_index']]
    summary.update(physics_informed_score=top['score'],max_p_failure=float(max(scores['Physics-Informed ML'])),
        hotspot_h3_index=top['h3_index'],hotspot_probability=top['score'],hotspot_fs_dynamic=same['fs_mid'],
        fs_interval=same['fs_interval'],fosm_variance=same['fosm_variance'],kinematics_top_hotspot={**same,**top},
        critical_hotspots_count=sum(r['fs_low'] is not None and r['fs_low']<=1 for r in by_id.values()),
        hazard_summary={'status':'CONDITIONAL_REFERENCE_ASSESSMENT','initiation_score':top['score'],
            'mechanical_margin':'envelope_crosses_failure' if same['fs_low'] is not None and same['fs_low']<=1 else 'envelope_above_failure',
            'scope':assessment['scope']})
    if response.get('runout_assessment',{}).get('h3_index')!=top['h3_index']:
        from .reference_assessment import reference_runout
        response['runout_assessment']=reference_runout(scan,top['h3_index'])
    response['runout_geojson']['features']=[{'type':'Feature','geometry':{'type':'LineString','coordinates':r['coordinates_lonlat']},'properties':r}
        for r in response.get('runout_assessment',{}).get('scenarios',[]) if len(r['path'])>1]
    corridor=[]
    import h3
    for k in response.get('runout_assessment',{}).get('corridor_h3_indices',[]):
        ring=[[lon,lat] for lat,lon in h3.cell_to_boundary(k)];ring.append(ring[0])
        corridor.append({'type':'Feature','geometry':{'type':'Polygon','coordinates':[ring]},'properties':{'h3_index':k,'scope':'Conditional release corridor'}})
    response['runout_corridor_geojson']={'type':'FeatureCollection','features':corridor}
    pmap={r['h3_index']:r for r in records}
    for feature in response['h3_geojson']['features']:
        if feature['properties']['h3_index'] in pmap:feature['properties'].update(pmap[feature['properties']['h3_index']])
    for feature in response.get('terrain_geojson',{}).get('features',[]):
        record=pmap.get(feature['properties']['h3_index'])
        if record:feature['properties'].update(physics_informed_score=record['score'],
            statistical_rainfall_score=record['ablation_scores']['Weather+Terrain ML'])
    response['cell_comparisons']=[{**r,'fs_dynamic':by_id[r['h3_index']]['fs_mid'],
        'statistical_score':r['ablation_scores']['Weather+Terrain ML'],'physics_informed_score':r['score'],
        'p_failure_physical':by_id[r['h3_index']]['p_failure_physical'],
        'physics_status':by_id[r['h3_index']]['status']} for r in records]
    return response


def summary_metrics(response):
    summary=response['summary'];top=summary.get('kinematics_top_hotspot') or {}
    if not top:return
    values=[('dynamic_factor_of_safety',top.get('fs_mid'),'ratio'),
        ('physics_informed_initiation_score',top.get('score'),'sample score'),
        ('pore_pressure_head',top.get('pressure_head_m'),'m'),('slope_angle',top.get('slope_deg'),'degrees'),
        ('conditional_release_peak_velocity',response.get('runout_assessment',{}).get('peak_velocity_ms'),'m/s'),
        ('factor_of_safety_standard_deviation',top.get('sigma_fs'),'ratio')]
    summary['numbers_that_matter']=[{'metric':name,'value':value,'unit':unit,'available':value is not None,
        'status':'CONDITIONAL_REFERENCE' if value is not None else 'UNAVAILABLE','threshold':None,
        'causal_group':None,'weight_percent':None} for name,value,unit in values]
