"""Additive source-based rainfall assessment; never fabricates missing physics."""
from pathlib import Path
import numpy as np
from backend.ml.rainfall_features import FEATURES,VERSION,feature_vector
from backend.ml.rainfall_baseline import RainfallBaseline
from .ground_assessment import located_cell
ROOT=Path(__file__).resolve().parents[2]


def load_rainfall_model():
    path=ROOT/'artifacts/rainfall_source_baseline.pkl'
    if not path.exists():return None
    try:
        import joblib
        model=joblib.load(path)
        return model if isinstance(model,RainfallBaseline) and model.feature_version==VERSION else None
    except (ImportError,ValueError,TypeError,OSError,AttributeError):return None


def evidence(scan):
    h=scan.hydrology;w=scan.weather
    theta=h.open_meteo_soil_moisture_m3m3 if h else None
    n=h.soil_porosity_estimate if h else None
    compatibility=getattr(h,'moisture_compatibility','unavailable') if h else 'unavailable'
    return {'weather_time_utc':w.observed_at_utc if w else None,
        'terrain_cells':len(scan.terrain.cells) if scan.terrain else 0,
        'soil_source':h.soil_property_source if h else 'unavailable',
        'soil_quality':h.soil_property_quality if h else 'unavailable',
        'volumetric_moisture_m3m3':theta,'estimated_porosity_fraction':n,
        'absolute_saturation_fraction':h.soil_saturation_pct/100 if h and h.soil_saturation_pct is not None else None,
        'moisture_compatibility':compatibility,
        'moisture_exceeds_porosity':bool(theta is not None and n is not None and theta>n),
        'relative_wetness_scope':'NASA POWER relative wetness is an index; it does not replace physical saturation',
        'statistical_moisture_scope':'24h mean of provider 0-7cm volumetric moisture; separate from physical-column inputs',
        'nasa_contribution':{
            'COOLR':'Historical inventory labels and regional event context; reported locations can be imprecise',
            'POWER':'Relative wetness / daily precipitation context; not measured groundwater',
            'EONET':'Event context only; not local validation or a model-training label'},
        'other_providers':'Terrarium terrain, Open-Meteo weather, SoilGrids where available, USGS seismic context',
        'motion_status':'CONDITIONAL_RELEASE_SCENARIOS_WITH_DECLARED_RHEOLOGY',
        'scope':'Experimental rainfall exploration; no operational alert or evacuation decision'}


def assess(scan,model,rain_delta_mm=0.,nonrain_scenario=False):
    window=dict(getattr(scan.weather,'rainfall_source_window',{}) or {}) if scan.weather else {}
    result={'status':'UNAVAILABLE','feature_version':VERSION,'score_name':'experimental_rainfall_score',
        'scope':'Rainfall-related inventory/absence sample; not annual or personal failure probability; other mechanisms not assessed',
        'window':window,'cells':[],'hotspot':None,'score_min':None,'score_max':None,'score_median':None,
        'model_kind':getattr(model,'kind',None),'reason':None,
        'hazard_probability':None,'hazard_verdict':None,
        'validation_status':'NO_INDEPENDENT_LOCAL_OR_PROSPECTIVE_VALIDATION',
        'selection_reason':'Highest inventory-classifier score among complete source-feature cells; ties use H3 ID. This does not select the most mechanically unstable slope.',
        'scenario_note':'Additional rain changes only the latest 24h accumulation; raw moisture and previous six days remain fixed' if rain_delta_mm else 'Current retained/source window; not an advance forecast'}
    if nonrain_scenario:
        result.update(status='OUT_OF_SCOPE',reason='This statistical model supports rainfall-only comparisons; other mechanisms remain in the experimental physics path')
        return result
    if not scan.terrain or not scan.terrain.cells:
        result['reason']='Terrain unavailable';return result
    if not window or window.get('version')!=VERSION or window.get('status')!='AVAILABLE':
        result['reason']=window.get('reason') or 'Matching complete hourly source window unavailable; no legacy window substitution'
        return result
    if model is None:
        result['reason']='Source-based rainfall model unavailable; run scripts/train_rainfall_baseline.py'
        return result
    window['rain_24h_mm']+=rain_delta_mm
    cells=sorted(scan.terrain.cells.values(),key=lambda c:c.cell_id);vectors=[];selected=[];omitted=[]
    for cell in cells:
        try:
            vector=feature_vector(window,cell.slope_deg,cell.twi,cell.profile_curvature)
            if not np.isfinite(vector).all():raise ValueError('Raw surface moisture missing; live scoring does not substitute an imputed observation')
            located_cell(cell,scan.location)
            vectors.append(vector);selected.append(cell)
        except ValueError as exc:omitted.append({'h3_index':cell.cell_id,'reason':str(exc)})
    if not vectors:
        result.update(reason='No cells with valid source terrain',omitted_cells=omitted);return result
    x=np.asarray(vectors);p=model.predict_proba(x);outside=model.out_of_range(x)
    if len(p)!=len(x) or not np.isfinite(p).all() or np.any((p<0)|(p>1)):
        result['reason']='Model returned invalid scores; no numeric replacement';return result
    explanations=model.explain(x) if hasattr(model,'explain') else None
    records=[{**located_cell(c,scan.location),'score':float(score),
              'features':dict(zip(FEATURES,[None if not np.isfinite(v) else float(v) for v in vector])),
              'missing_features':[FEATURES[i] for i,v in enumerate(vector) if not np.isfinite(v)],
              'outside_training_range':out} for c,score,vector,out in zip(selected,p,x,outside)]
    if explanations:
        for record,explanation in zip(records,explanations):record['model_explanation']=explanation
    hotspot=records[int(np.argmax(p))]
    result.update(status='EXPERIMENTAL',cells=records,hotspot=hotspot,
        score_min=float(np.min(p)),score_max=float(np.max(p)),score_median=float(np.median(p)),
        omitted_cells=omitted,range_scope='Training min/max checks flag extrapolation; values inside these ranges are not evidence of validation')
    return result


def attach(response,scan,model,rain_delta_mm=0.,nonrain_scenario=False):
    response['evidence']=evidence(scan)
    response['statistical_assessment']=assess(scan,model,rain_delta_mm,nonrain_scenario)
    # Keep a terrain layer even when physics is unavailable. Legacy h3_geojson
    # remains unchanged, so existing consumers cannot mistake null FS for a value.
    import h3
    scores={c['h3_index']:c for c in response['statistical_assessment']['cells']}
    terrain=[]
    for cell in scan.terrain.cells.values() if scan.terrain else ():
        boundary=getattr(h3,'cell_to_boundary',None) or h3.h3_to_geo_boundary
        ring=[[float(lon),float(lat)] for lat,lon in boundary(cell.cell_id)]
        ring.append(ring[0])
        terrain.append({'type':'Feature','geometry':{'type':'Polygon','coordinates':[ring]},
            'properties':{'h3_index':cell.cell_id,'slope_deg':float(cell.slope_deg),'elevation_m':float(cell.elevation_m),
                          'statistical_rainfall_score':scores.get(cell.cell_id,{}).get('score')}})
    response['terrain_geojson']={'type':'FeatureCollection','features':terrain}
    # One joined record per cell, so the CLI never pairs scores from different
    # locations. Physical uncertainty and motion remain explicit nulls.
    physical={c['h3_index']:c for c in response.get('physical_assessment',{}).get('cells',[])}
    if not physical:
        physical={f['properties']['h3_index']:f['properties'] for f in response.get('h3_geojson',{}).get('features',[])}
    response['cell_comparisons']=[{**located_cell(cell,scan.location),
        'statistical_score':scores.get(cell.cell_id,{}).get('score'),
        'fs_dynamic':physical.get(cell.cell_id,{}).get('fs_dynamic'),
        'p_failure_physical':physical.get(cell.cell_id,{}).get('p_failure_physical'),
        'physics_status':physical.get(cell.cell_id,{}).get('status',response.get('physical_assessment',{}).get('reason','UNAVAILABLE'))}
        for cell in sorted(scan.terrain.cells.values(),key=lambda c:c.cell_id)] if scan.terrain else []
    return response
