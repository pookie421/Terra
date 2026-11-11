#!/usr/bin/env python3
"""Network-disabled refresh/refit of the retained reference baseline."""
import argparse
import json
import socket
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--refresh',action='store_true')
    parser.add_argument('--device',default='cpu',choices=['cpu','cuda'])
    args=parser.parse_args()
    def offline(*args,**kwargs):raise RuntimeError('Live network is disabled for this reproducible baseline')
    socket.socket.connect=offline
    if args.refresh:
        from backend.ml.audit_dataset import refresh_frozen_dataset
        result=refresh_frozen_dataset(ROOT/'data/real_global_landslide_observations.parquet')
        print(json.dumps({k:result[k] for k in ('row_count','complete_14_feature_rows','network_used')},indent=2),flush=True)
    from backend.ml.reference_training import fit_reference_baseline
    report=fit_reference_baseline(ROOT,device=args.device)
    print(json.dumps({k:report[k] for k in ('training_rows','positive_rows','negative_rows','connected_components')},indent=2))
    print('Report: artifacts/reference_baseline_report.json')
    return 0
if __name__=='__main__':raise SystemExit(main())
