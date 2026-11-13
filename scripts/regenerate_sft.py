#!/usr/bin/env python3
"""Regenerate state-bound app-controller traces without fetching new observations."""
import sys,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))

def main():
    from backend.ml.context_trajectories import export
    report=export(ROOT)
    print(json.dumps(report,indent=2))
    return 0
if __name__=='__main__':main()
