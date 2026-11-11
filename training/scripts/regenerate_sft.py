#!/usr/bin/env python3
"""Regenerate source-grounded instruction traces with the current fitted model."""
import sys,json,shutil,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import joblib
import pyarrow.parquet as pq
from backend.ml.sft_trajectories import export_current_trajectories

def main():
    from backend.ml.context_trajectories import export
    report=export(ROOT)
    print(json.dumps(report,indent=2))
    return 0
if __name__=='__main__':main()
