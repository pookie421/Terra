#!/usr/bin/env python3
"""Remove generated caches and workstation paths; never alter source observations."""
import shutil
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from backend.release import HOME_PREFIX, portable_text
TEXT_SUFFIXES={'.py','.md','.json','.jsonl','.txt','.csv','.bat','.sh','.yaml','.yml','.toml'}

def main():
    for path in list(ROOT.rglob('__pycache__')):
        shutil.rmtree(path)
    for path in ROOT.rglob('.DS_Store'):
        path.unlink()
    for path in ROOT.rglob('*'):
        if not path.is_file() or '.git' in path.parts:
            continue
        if path.suffix in TEXT_SUFFIXES:
            original=path.read_text(encoding='utf-8');clean=portable_text(original)
            if clean!=original:path.write_text(clean,encoding='utf-8')
    bad=[str(p.relative_to(ROOT)) for p in ROOT.rglob('*') if p.is_file() and '.git' not in p.parts and HOME_PREFIX.encode() in p.read_bytes()]
    if bad:raise RuntimeError('Workstation paths remain: '+str(bad))
    print('Portable workspace verified; generated caches removed.')

if __name__=='__main__':main()
