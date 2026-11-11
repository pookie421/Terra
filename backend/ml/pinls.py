"""Declared physical negative selection; features are never conditioned on labels."""
import hashlib
import math
from collections import Counter

def quiescent_control(row):
    """External-forcing diagnostic only; never consult modeled stability.

    This defines a calm-weather subset, not the default background population.
    """
    keys=('rain_effective_storm_mm','pga_attenuation_proxy_g','snow_depth_72h_max_m')
    if any(row.get(k) is None or not math.isfinite(float(row[k])) for k in keys):return False
    return (0.<=row[keys[0]]<=35. and 0.<=row[keys[1]]<.01 and row[keys[2]]==0.
            and str(row.get('usgs_source_status','')).startswith(('live_', 'retained_')))


def pinls_filter(rows, threshold=None):
    """Compatibility adapter: retain all labels; no model-screened negatives."""
    return list(rows), {'threshold':None,'removed_ids':[],
        'scope':'No FS/probability filtering; inventory absence is not verified stability',
        'temporal_rule':'Fixed-date or seeded external-data sampling only'}

def reserve_conformal_blocks(rows):
    from .global_xgb import row_component_groups
    eligible = [r for r in rows if r.get('label') in (0,1) and not r.get('is_synthetic')]
    if not eligible:
        return list(rows), [], []
    components = row_component_groups(eligible)
    ids = {r['event_id']:str(g) for r,g in zip(eligible,components)}
    groups=sorted(set(components),key=lambda g:hashlib.sha256(('terra-conformal-v14:'+g).encode()).hexdigest())
    reserved=set(groups[:max(1,len(groups)//5)])
    calibration=[r for r in rows if ids.get(r['event_id']) in reserved]
    proper=[r for r in rows if ids.get(r['event_id']) not in reserved]
    return proper,calibration,sorted(reserved)
