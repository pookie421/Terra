"""Public conditional physical columns, site overrides and release corridors."""
import math
from collections.abc import Mapping
import h3
from .ground_assessment import located_cell,observation,REQUIRED,HYDRAULIC
from backend.ml.rainfall_features import utc
from backend.physics.reference_context import columns as reference_columns,VERSION


def source_window(scan):
    weather=scan.weather;window=dict(getattr(weather,'rainfall_source_window',{}) or {})
    if window.get('status')=='AVAILABLE':return window
    return {'status':'FORCING_PRIOR','rain_24h_mm':getattr(weather,'rain_24h_mm',None) or 0.,
        'rain_prior_6d_mm':0.,'soil_moisture_0_7cm_mean_m3m3':None,
        'reason':'Incomplete hourly forcing; nominal available rain/dry reference with enlarged rain uncertainty'}


def assess(scan,columns=None,rain_delta_mm=0.,storm_duration_hr=24.,unsupported_scenario=False,pga_override_g=0.):
    result={'status':'UNAVAILABLE','model_version':VERSION,'cells':[],'hotspot':None,'omitted_cells':[],
        'required_inputs':[],'optional_site_overrides':list(REQUIRED+HYDRAULIC),'reason':None,
        'scope':'Conditional shallow-regolith reference columns; site records override declared priors',
        'uncertainty_status':'JOINT_FOSM_REFERENCE_PRIORS','p_failure_physical':None,'peak_velocity_ms':None,'failure_time_utc':None}
    if not scan.terrain or not scan.terrain.cells:result['reason']='Terrain unavailable';return result
    if columns is not None and not isinstance(columns,Mapping):raise ValueError('Ground columns must be indexed by H3 cell')
    valid={};excluded=set();at=utc(scan.generated_at_utc)
    for cell_id,record in (columns or {}).items():
        try:
            if cell_id not in scan.terrain.cells:raise ValueError('Ground cell is outside this scan')
            if not isinstance(record,Mapping):raise ValueError('Ground column must contain sourced input objects')
            for key in record:
                if key not in REQUIRED+HYDRAULIC:raise ValueError('Unknown ground input: '+key)
                value=observation(record,key,at)
                if key in ('depth_m','unit_weight_kn_m3','conductivity_m_s','diffusivity_m2_s') and value<=0:raise ValueError('Positive '+key+' required')
                if key in ('cohesion_kpa','friction_angle_deg') and value<0:raise ValueError('Nonnegative '+key+' required')
                if key=='friction_angle_deg' and value>=90:raise ValueError('Friction angle must be below 90 degrees')
            valid[cell_id]=record
        except (ValueError,TypeError) as exc:
            excluded.add(cell_id);result['omitted_cells'].append({'h3_index':cell_id,'reason':str(exc)})
    cells=[c for k,c in sorted(scan.terrain.cells.items()) if k not in excluded]
    window=source_window(scan)
    pga=max(getattr(scan.seismic,'max_pga_g',0.) or 0.,pga_override_g)
    window['pga_reference_g']=pga if pga>0 else None
    rows=reference_columns(cells,window,getattr(scan.hydrology,'soil_properties',None),rain_delta_mm,storm_duration_hr,valid)
    for cell,row in zip(cells,rows):result['cells'].append({**located_cell(cell,scan.location),**row})
    if rows:
        finite=[c for c in result['cells'] if c['fs_mid'] is not None]
        result['hotspot']=min(finite,key=lambda c:(c['fs_mid'],c['h3_index'])) if finite else result['cells'][0]
        result.update(status='REFERENCE_COLUMN',selection_reason='Minimum uncapped median FS; material/water priors and site overrides exposed')
    else:result['reason']='No valid terrain columns after invalid supplied overrides were excluded'
    result['terrain_cells_without_ground_result']=len(scan.terrain.cells)-len(rows)
    if unsupported_scenario:result['additional_mechanism_status']='Requested non-rain mechanism requires its own model; reference baseline retained'
    return result


def reference_runout(scan,cell_id):
    from backend.physics.kinematics import route_runout
    routes=[];corridor=set()
    for mu,xi,depth in ((.1,1000.,1.),(.2,500.,.5),(.3,200.,.25)):
        route=route_runout(cell_id,scan.terrain.cells,0.,mu,xi,
            infrastructure_by_cell=scan.infrastructure.by_h3_cell if scan.infrastructure else {},flow_depth_m=depth,released=True)
        d=route.to_dict();corridor.update(route.path)
        if any(k not in scan.terrain.cells for k in h3.grid_disk(route.path[-1],1)):
            d['stop_reason']='terrain_domain_boundary; route may continue'
        d.update(mu_b=mu,xi_m_s2=xi,flow_depth_m=depth,
            coordinates_lonlat=[[scan.terrain.cells[k].longitude,scan.terrain.cells[k].latitude] for k in route.path])
        routes.append(d)
    return {'status':'CONDITIONAL_RELEASE_SCENARIO','h3_index':cell_id,'path':routes[1]['path'],
        'peak_velocity_ms':max(r['peak_velocity_m_s'] for r in routes),'scenarios':routes,
        'corridor_h3_indices':sorted(corridor),'scope':'Conditional release, not asserted failure; assumed rheology; H3 routing, no entrainment or mass conservation'}


def attach(response,scan,columns=None,**scenario):
    assessment=assess(scan,columns,**scenario)
    response.update(physical_assessment=assessment,assessment_mode='PHYSICS_INFORMED_REFERENCE',operational_warning_ready=False)
    summary=response['summary'];summary['physics_status']=assessment['reason'] or assessment['status']
    top=assessment['hotspot']
    if top:
        summary.update(hotspot_h3_index=top['h3_index'],hotspot_fs_dynamic=top['fs_mid'],
            fs_interval=top['fs_interval'],fosm_variance=top['fosm_variance'],p_failure_physical=top['p_failure_physical'],
            min_fs_dynamic=min((c['fs_mid'] for c in assessment['cells'] if c['fs_mid'] is not None),default=None),
            kinematics_top_hotspot=top,summary_statistics_quality='Uncapped reference column minima; not surveyed local stability')
        response['runout_assessment']=reference_runout(scan,top['h3_index'])
        for cell in assessment['cells']:
            ring=[[lon,lat] for lat,lon in h3.cell_to_boundary(cell['h3_index'])];ring.append(ring[0])
            response['h3_geojson']['features'].append({'type':'Feature','geometry':{'type':'Polygon','coordinates':[ring]},'properties':cell})
        response['runout_geojson']['features']=[{'type':'Feature','geometry':{'type':'LineString','coordinates':r['coordinates_lonlat']},'properties':r}
            for r in response['runout_assessment']['scenarios'] if len(r['path'])>1]
    return response
