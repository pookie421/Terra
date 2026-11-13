"""Presentation and immutable state construction for automatic forecast maps."""
import copy
import hashlib
from dataclasses import replace
import h3
from .reference_forecast import projected_columns,projected_window
from .reference_assessment import reference_runout
from .physics_informed_assessment import attach as attach_models,summary_metrics


def live_inputs(scan):
    window=getattr(scan.weather,'rainfall_source_window',{}) or {}
    soil=dict(getattr(scan.hydrology,'soil_properties',{}) or {})
    return {'rain_24h_mm':window.get('rain_24h_mm'),
        'rain_prior_6d_mm':window.get('rain_prior_6d_mm'),
        'surface_moisture_m3m3':window.get('soil_moisture_0_7cm_mean_m3m3'),
        'weather_valid_time_utc':window.get('cutoff_utc'),
        'soil_properties':soil,'soil_quality':getattr(scan.hydrology,'soil_property_quality',None),
        'pga_proxy_g':getattr(scan.seismic,'max_pga_g',None),
        'river_discharge_m3_s':(scan.river or {}).get('discharge_m3_s'),
        'source_notes':{'rain':'Liquid latest-24h and preceding nonoverlapping 144h, same source window as ML',
            'surface_moisture':'Latest-24h mean raw 0–7cm volumetric source feature, not saturation or groundwater head',
            'soil':'Predicted 0–30cm properties or explicit field-level priors, not borehole strength',
            'pga':'Catalog attenuation proxy; no accelerometer observation is implied',
            'river':'GloFAS modeled discharge, not a local stream gauge'}}


def scenario_id(origin,key,hour,valid_time):
    if key=='now':return origin
    return 'forecast-'+key+'-'+hashlib.sha256((origin+str(hour)+valid_time).encode()).hexdigest()[:16]


def sector_summary(response):
    cells=response['physical_assessment']['cells']
    finite=[c for c in cells if c['fs_mid'] is not None]
    minimum=min(finite,key=lambda c:(c['fs_mid'],c['h3_index'])) if finite else None
    summary=response['summary']
    summary.update(grid_cell_count=len(cells),
        minimum_fs_h3_index=minimum['h3_index'] if minimum else None,
        minimum_fs_interval=minimum['fs_interval'] if minimum else None,
        elevation_range_m=[min(c['elevation_m'] for c in cells),max(c['elevation_m'] for c in cells)] if cells else None,
        slope_range_deg=[min(c['slope_deg'] for c in cells),max(c['slope_deg'] for c in cells)] if cells else None)
    return response


def decorate_forecast(response):
    response=sector_summary(response)
    forecast=response['predictive_forecast_72h']
    for view in forecast.get('views',[]):
        view['scenario_id']=scenario_id(response['scenario_id'],view['key'],view['hour'],view['valid_time_utc'])
        view['summary']={'minimum_fs':view['minimum_fs'],'forcing_path':'p50'}
    response['forecast_view']={'key':'now','hour':0,'valid_time_utc':response.get('live_inputs',{}).get('weather_valid_time_utc') or response['generated_at_utc'],
        'kind':'CURRENT_PROVIDER_REFERENCE_COLUMN'}
    return response


def _polygon(cell):
    ring=[[lon,lat] for lat,lon in h3.cell_to_boundary(cell['h3_index'])];ring.append(ring[0])
    return {'type':'Feature','geometry':{'type':'Polygon','coordinates':[ring]},'properties':cell}


def project_response(state,models,key):
    base=state['response'];forecast=base['predictive_forecast_72h']
    if not forecast.get('automatic'):raise ValueError('Automatic provider forecast is unavailable for this state')
    view=next((r for r in forecast.get('views',[]) if r['key']==key),None)
    if view is None:raise ValueError(forecast.get('reason','Requested automatic horizon is unavailable'))
    if key=='now':return copy.deepcopy(base),copy.deepcopy(state['scan'])
    projection=forecast['projection_context'];hour=view['hour'];raw=copy.deepcopy(state['scan'])
    cells=projected_columns(raw,base['physical_assessment'],projection,hour)
    window=projected_window(raw,projection,hour)
    raw.weather=replace(raw.weather,rainfall_source_window=window)
    result=copy.deepcopy(base)
    physical={**result['physical_assessment'],'cells':cells,'forecast_hour':hour,
        'scope':'Automatic provider-forced shallow-regolith projection with retained material priors; no observed future failure claim'}
    finite=[c for c in cells if c['fs_mid'] is not None]
    physical['hotspot']=min(finite,key=lambda c:(c['fs_mid'],c['h3_index'])) if finite else cells[0] if cells else None
    result['physical_assessment']=physical
    result['h3_geojson']={'type':'FeatureCollection','features':[_polygon(c) for c in cells]}
    result['terrain_geojson']=copy.deepcopy(result['h3_geojson'])
    result['cell_comparisons']=[]
    result['statistical_assessment']={'status':'UNAVAILABLE','cells':[],'hotspot':None,'reason':'Projected source window has not been scored'}
    summary=result['summary']
    summary.update(physics_informed_score=None,max_p_failure=None,hotspot_probability=None,
        min_fs_dynamic=min((c['fs_mid'] for c in finite),default=None),
        critical_hotspots_count=sum(c['fs_low'] is not None and c['fs_low']<=1 for c in cells))
    top=physical['hotspot']
    if top:
        summary.update(hotspot_h3_index=top['h3_index'],hotspot_fs_dynamic=top['fs_mid'],fs_interval=top['fs_interval'],
            fosm_variance=top['fosm_variance'],kinematics_top_hotspot=top,p_failure_physical=top['p_failure_physical'])
        result['runout_assessment']=reference_runout(raw,top['h3_index'])
        result['runout_geojson']={'type':'FeatureCollection','features':[{'type':'Feature','geometry':{'type':'LineString','coordinates':r['coordinates_lonlat']},'properties':r}
            for r in result['runout_assessment']['scenarios'] if len(r['path'])>1]}
        result['runout_corridor_geojson']={'type':'FeatureCollection','features':[_polygon({'h3_index':k}) for k in result['runout_assessment']['corridor_h3_indices']]}
    result=attach_models(result,raw,models)
    summary_metrics(result)
    sector_summary(result)
    result.update(scenario_id=view['scenario_id'],forecast_view={**view,'kind':'AUTOMATIC_PROVIDER_FORECAST'},
        forecast_origin_scenario_id=base['scenario_id'],forecast_inputs=window,
        source_provenance_note='Source statuses describe acquisition at NOW; this state is a conditional forecast from those payloads')
    from .prediction_models import attach_prediction_models
    result=attach_prediction_models(result,raw)
    return result,raw
