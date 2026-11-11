"""Reference columns with optional sourced ground overrides.

Missing boreholes select declared reference priors. Public assessments delegate
to the uncertainty-aware reference path; site records are optional overrides.
"""
import math
from collections.abc import Mapping
import h3
from backend.ml.rainfall_features import utc

REQUIRED=('cohesion_kpa','friction_angle_deg','unit_weight_kn_m3','depth_m','pressure_head_m')
HYDRAULIC=('conductivity_m_s','diffusivity_m2_s')
METHODS=('measurement','site_calibration','pedotransfer_reference_column','scenario')


def located_cell(cell,location):
    if not h3.is_valid_cell(cell.cell_id):raise ValueError('Invalid H3 cell identifier')
    lat,lon=h3.cell_to_latlng(cell.cell_id)
    return {'h3_index':cell.cell_id,'latitude':float(lat),'longitude':float(lon),
        'slope_deg':float(cell.slope_deg),'twi':float(cell.twi),'elevation_m':float(cell.elevation_m),
        'distance_from_query_m':float(h3.great_circle_distance((lat,lon),(location.latitude,location.longitude),unit='m')),
        'h3_resolution':h3.get_resolution(cell.cell_id),
        'map_url':f'https://www.openstreetmap.org/?mlat={lat:.7f}&mlon={lon:.7f}#map=16/{lat:.7f}/{lon:.7f}'}


def observation(record,name,at):
    entry=record.get(name)
    if not isinstance(entry,Mapping):raise ValueError('Missing sourced ground input: '+name)
    value=entry.get('value')
    if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value):
        raise ValueError('Invalid ground value: '+name)
    if not isinstance(entry.get('source'),str) or not entry['source'].strip():
        raise ValueError('Missing source citation: '+name)
    if entry.get('method') not in METHODS:
        raise ValueError('Unrecognized ground-input provenance: '+name)
    if entry['method'] in ('pedotransfer_reference_column','scenario'):return float(value)
    try:start,end=utc(entry['observed_at_utc']),utc(entry['valid_until_utc'])
    except (ValueError,TypeError,KeyError):raise ValueError('Missing or invalid validity dates: '+name)
    if not start<=at<=end:raise ValueError('Ground input expired or dated after the scan: '+name)
    return float(value)


def assess_ground(scan,columns=None,**scenario):
    from .reference_assessment import assess
    return assess(scan,columns,**scenario)


def attach_ground(response,scan,columns=None,**scenario):
    from .reference_assessment import attach
    return attach(response,scan,columns,**scenario)
