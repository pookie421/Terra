#!/usr/bin/env python3
"""Mirror shared source, local datasets and reports into the portable GPU bundle."""
import shutil
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from backend.release import HOME_PREFIX, portable_text
BUNDLE=ROOT/'windows_5070ti_trainer'

def main():
    for package in ('ml','physics','realtime'):
        source=ROOT/'backend'/package;target=BUNDLE/'backend'/package
        target.mkdir(parents=True,exist_ok=True)
        for path in target.glob('*.py'):
            if not (source/path.name).exists():path.unlink()
        for path in source.glob('*.py'):shutil.copy2(path,target/path.name)
    shutil.copy2(ROOT/'backend/release.py',BUNDLE/'backend/release.py')
    for path in (ROOT/'data').rglob('*'):
        if path.is_file():
            target=BUNDLE/path.relative_to(ROOT);target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,target)
    for name in ('global_multi_causal_xgb.pkl','global_multi_causal_xgb.metadata.json','scientific_validation_report.json','full_system_bug_audit.json','full_system_bug_audit.md','randomized_global_blind_audit.json','randomized_global_blind_audit.md','literature_boundary_audit.json','literature_boundary_audit.md','readiness.json'):
        path=ROOT/'artifacts'/name
        if path.exists():shutil.copy2(path,BUNDLE/'artifacts'/name)
    for name in ('LICENSE','THIRD_PARTY_NOTICES.md','PRETRAINING_AUDIT.md','pyproject.toml'):
        if (ROOT/name).exists():shutil.copy2(ROOT/name,BUNDLE/name)
    (BUNDLE/'docs').mkdir(exist_ok=True)
    shutil.copy2(ROOT/'docs/physics_sources.md',BUNDLE/'docs/physics_sources.md')
    shutil.copy2(ROOT/'scripts/check_training_artifacts.py',BUNDLE/'scripts/check_training_artifacts.py')
    for path in BUNDLE.rglob('*'):
        if path.is_file() and path.suffix in ('.py','.md','.json','.jsonl','.txt','.bat','.sh','.csv'):
            original=path.read_text(encoding='utf-8');clean=portable_text(original)
            if clean!=original:path.write_text(clean,encoding='utf-8')
    for path in list(BUNDLE.rglob('__pycache__')):shutil.rmtree(path)
    bad=[str(p.relative_to(BUNDLE)) for p in BUNDLE.rglob('*') if p.is_file() and HOME_PREFIX.encode() in p.read_bytes()]
    if bad:raise RuntimeError('Nonportable paths in '+str(bad))
    print('Windows bundle refreshed; shared source mirrored and portable.')

if __name__=='__main__':main()
