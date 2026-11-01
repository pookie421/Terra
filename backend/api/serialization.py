"""JSON boundary conversion with explicit non-finite replacement diagnostics."""
from dataclasses import asdict, is_dataclass
from pathlib import Path
import math
from collections.abc import Mapping
import numpy as np

def sanitize_json(value, report_replacements=True):
    replaced=[]
    def convert(item,path):
        if isinstance(item,np.ndarray):return convert(item.tolist(),path)
        if isinstance(item,np.generic):return convert(item.item(),path)
        if is_dataclass(item):return convert(asdict(item),path)
        if isinstance(item,Mapping):return {str(k):convert(v,path+'.'+str(k)) for k,v in item.items()}
        if isinstance(item,(list,tuple)):return [convert(v,path+'['+str(i)+']') for i,v in enumerate(item)]
        if isinstance(item,float) and not math.isfinite(item):
            replaced.append(path);return 0.0
        if isinstance(item,Path):return str(item)
        return item
    clean=convert(value,'$')
    if report_replacements and isinstance(clean,dict) and replaced:
        clean['serialization_quality']={'nonfinite_replacement_count':len(replaced),'replacement_value':0.0,'affected_paths':replaced}
    return clean
