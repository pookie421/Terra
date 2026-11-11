#!/usr/bin/env python3
"""Compatibility entry point for the offline, event/spatial-isolated baseline."""
import argparse
import asyncio
import json
import socket
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def _fit_real_model(path):
    from backend.ml.reference_training import fit_reference_baseline
    if Path(path).resolve() != (ROOT/'data/model_training.parquet').resolve():
        raise ValueError('Use the canonical embargoed model_training.parquet split')
    return fit_reference_baseline(ROOT,device='cpu')


async def run(args):
    from backend.release import invalidate_readiness
    invalidate_readiness(ROOT/'artifacts','Offline baseline rebuild requires fresh artifact verification')
    data_path=ROOT/args.dataset
    if not args.skip_dataset_build:
        from backend.ml.audit_dataset import refresh_frozen_dataset
        refresh_frozen_dataset(data_path)
    meta=data_path.with_suffix('.metadata.json')
    summary=json.loads(meta.read_text()) if meta.exists() else {}
    split=summary.get('event_group_holdout',{})
    if split.get('status') != 'ready':
        payload={'pretraining_ready':False,'engineering_ready':False,'training_model_fitted':False,
                 'audit_status':'blocked: frozen source refresh or holdout split incomplete','dataset_build':summary}
        (ROOT/'artifacts/scientific_validation_report.json').write_text(json.dumps(payload,indent=2)+'\n')
        print('Training stopped: complete the frozen source refresh and event/spatial split first.')
        return 2
    report=_fit_real_model(ROOT/str(split['training_rows']))
    print(json.dumps({k:report[k] for k in ('training_rows','positive_rows','negative_rows','connected_components')},indent=2))
    return 0


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset',default='data/real_global_landslide_observations.parquet')
    parser.add_argument('--skip-dataset-build',action='store_true')
    # Preserve accepted command-line options without coupling live acquisition to fitting.
    parser.add_argument('--provider-timeout',type=float,default=15.,help=argparse.SUPPRESS)
    parser.add_argument('--fetch-timeout',type=float,default=15.,help=argparse.SUPPRESS)
    parser.add_argument('--radius-km',type=float,default=1.,help=argparse.SUPPRESS)
    parser.add_argument('--max-events',type=int,default=450,help=argparse.SUPPRESS)
    args=parser.parse_args()
    def offline(*args,**kwargs):raise RuntimeError('Baseline fitting is network-disabled; acquire sources separately')
    socket.socket.connect=offline
    return asyncio.run(run(args))
if __name__=='__main__':raise SystemExit(main())
