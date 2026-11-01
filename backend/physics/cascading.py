"""DBI in consistent SI units; unresolved dam geometry never receives invented values."""
import math
import h3

def larsen_volume(area_m2):
    if not math.isfinite(area_m2) or area_m2<=0:raise ValueError('Positive mapped/scenario area required')
    # Larsen et al. 2010, Fig. 2a: log10(alpha)=-.836, gamma=1.332, all materials.
    return 10.**(-.836) * area_m2**1.332

def dimensionless_blockage_index(catchment_area_m2, dam_height_m, deposited_volume_m3):
    values=(catchment_area_m2,dam_height_m,deposited_volume_m3)
    if any(not math.isfinite(v) or v<=0 for v in values):raise ValueError('Positive finite SI geometry required')
    return math.log10(catchment_area_m2*dam_height_m/deposited_volume_m3)

def cascading_screen(source_cell, route_path, terrain_cells, dam_geometry=None):
    channel=next((h for h in route_path[1:] if terrain_cells[h].twi>=14.5),None)
    result={'cascading_dbi':None,'impoundment_status':'NO_MODELED_CHANNEL_INTERSECTION',
            'channel_h3_index':channel,'dam_height_m':None,'dam_volume_m3':None,'catchment_area_m2':None}
    if channel is None:return result
    area=h3.cell_area(source_cell,unit='m^2')
    result.update(impoundment_status='POTENTIAL_CHANNEL_INTERSECTION_GEOMETRY_UNRESOLVED',
                  source_area_scenario_m2=area,landslide_volume_scenario_m3=larsen_volume(area),
                  volume_quality='Full H3 source-cell failure scenario; not mapped scar or deposited dam volume',
                  catchment_area_in_view_m2=float(terrain_cells[channel].contributing_area_m2),
                  catchment_quality='Viewport DEM accumulation; full upstream catchment not established')
    if dam_geometry is None:return result
    a=dam_geometry['catchment_area_m2'];h=dam_geometry['dam_height_m'];v=dam_geometry['deposited_volume_m3']
    dbi=dimensionless_blockage_index(a,h,v)
    result.update(cascading_dbi=dbi,dam_height_m=h,dam_volume_m3=v,catchment_area_m2=a,
                  impoundment_status=('EMPIRICAL_STABLE_DAM_DOMAIN' if dbi<2.75 else
                                      'EMPIRICAL_UNSTABLE_DAM_DOMAIN' if dbi>3.08 else 'EMPIRICAL_TRANSITION_DOMAIN'),
                  geometry_source=dam_geometry.get('source','explicit scenario geometry'),
                  dbi_quality='Ermini-Casagli empirical domain; not evidence that a dam has formed')
    return result
