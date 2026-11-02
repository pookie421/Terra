"""Declared physical negative selection; features are never conditioned on labels."""
import hashlib
import math
from collections import Counter

def pinls_filter(rows, threshold=1.20):
    kept, rejected = [], []
    for row in rows:
        prior = row.get('fs_static_prior')
        negative_ok = prior is not None and math.isfinite(float(prior)) and float(prior) > threshold
        (kept if row.get('label') != 0 or negative_ok else rejected).append(row)
    strata = Counter('low_slope_lt15' if r.get('slope_max_deg',0)<15 else 'moderate_high_slope_ge15'
                     for r in kept if r.get('label')==0 and not r.get('is_synthetic'))
    return kept, {'threshold':threshold,'removed_ids':[r['event_id'] for r in rejected],
                  'retained_negative_slope_strata':dict(strata),
                  'scope':'Requested sampling rule; not verified intact ground. Filtering evaluation negatives changes the estimand.'}

def reserve_conformal_blocks(rows):
    groups=sorted({str(r['region_block']) for r in rows if r.get('label') in (0,1) and not r.get('is_synthetic')},
                  key=lambda g:hashlib.sha256(('terra-conformal-v5:'+g).encode()).hexdigest())
    reserved=set(groups[:max(10,len(groups)//5)])
    calibration=[r for r in rows if str(r.get('region_block')) in reserved and r.get('label') in (0,1)]
    proper=[r for r in rows if str(r.get('region_block')) not in reserved]
    return proper,calibration,sorted(reserved)
