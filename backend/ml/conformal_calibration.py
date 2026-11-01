"""Fit uncertainty on separately reserved, unfiltered spatial blocks."""
import numpy as np
from .global_xgb import FEATURE_NAMES, blend_failure_probability

def fit_reserved_conformal(model, data_directory, training_groups):
    import pyarrow.parquet as pq
    path=data_directory/'conformal_calibration.parquet'
    rows=pq.read_table(path).to_pylist()
    x=np.array([[r[k] for k in FEATURE_NAMES] for r in rows],dtype=np.float32)
    p=blend_failure_probability(model.predict_proba(x),np.array([r['p_failure_physical'] for r in rows]),x[:,4],x[:,0])
    return model.calibrate_conformal(p,[r['label'] for r in rows],[r['region_block'] for r in rows],training_groups)

def heldout_conformal_diagnostics(model, rows):
    """Descriptive coverage on untouched holdout labels, never used to choose q."""
    x=np.array([[r[k] for k in FEATURE_NAMES] for r in rows],dtype=np.float32)
    p=blend_failure_probability(model.predict_proba(x),np.array([r['p_failure_physical'] for r in rows]),x[:,4],x[:,0])
    bounds,width,_=model.conformal_intervals(p)
    y=np.array([r['label'] for r in rows]);g=np.array([r['region_block'] for r in rows])
    covered=(bounds[:,0]<=y)&(y<=bounds[:,1])
    return {'target':'binary event labels, not the latent probability',
            'rows':len(rows),'blocks':len(set(g)),
            'row_coverage':float(np.mean(covered)),
            'whole_block_coverage':float(np.mean([np.all(covered[g==v]) for v in set(g)])),
            'mean_interval_width':float(np.mean(width)),
            'q':model.conformal_['q'],
            'empty_binary_prediction_sets':int(np.sum((bounds[:,0]>0)&(bounds[:,1]<1))),
            'scope':'Descriptive on fixed source-location proxies; not a global coverage guarantee'}
