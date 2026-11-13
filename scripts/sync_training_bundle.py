#!/usr/bin/env python3
"""Build current portable training bundles from canonical source and local data.

Only obsolete top-level terrain_v13 derived caches are excluded. Their canonical
originals stay untouched; older bundle copies move to ignored archive_snapshots.
All observations, benchmark data, current metric terrain, raw tiles and retained
multi-causal source payloads remain in the current bundle inventory.
"""
import argparse
import hashlib
import json
import os
import shutil
import stat
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
DIRECTORIES=('backend','scripts','tests','docs','data')
ARTIFACTS=('global_multi_causal_xgb.pkl','global_multi_causal_xgb.metadata.json',
 'scientific_validation_report.json','reference_baseline_report.json','readiness.json',
 'sft_trajectory_report.json','lhasa_benchmark_suite.json','reference_verification.json','ACTIVE_REPORTS.md',
 'rainfall_source_baseline.pkl','rainfall_baseline_report.json','physics_informed_models.pkl',
 'physics_informed_ablation_report.json','metric_terrain_rebuild.json','phase1_phase2_report.json',
 'context_observation_coverage.json','global_context_benchmark.json','storm_grid_diagnostics.json',
 'MODEL_CARD.md','runtime_manifest.json','hardware_preflight.json','gpu_smoke_test.json','training_artifact_manifest.json')
ALIASES=(('scripts/train_gpu.py','train.py'),('scripts/train_qwen25_agent.py','train_qwen25_agent.py'),
 ('scripts/run_training.bat','run_training.bat'),
 ('scripts/run_copilot_server.bat','run_copilot_server.bat'),
 ('scripts/serve_qwen_copilot.py','serve_qwen_copilot.py'),
 ('scripts/export_qwen_ollama.py','export_qwen_ollama.py'),
 ('scripts/hardware_preflight.py','hardware_preflight.py'),
 ('constraints-runtime.txt','constraints-runtime.txt'),('requirements-training.txt','requirements.txt'),
 ('requirements-training.txt','requirements-training.txt'),('requirements-qwen.txt','requirements-qwen.txt'),
 ('requirements-geo.txt','requirements-geo.txt'),('docs/training_workflow.md','README.md'),
 ('LICENSE','LICENSE'),('THIRD_PARTY_NOTICES.md','THIRD_PARTY_NOTICES.md'),
 ('pyproject.toml','pyproject.toml'),('API_USAGE.md','API_USAGE.md'))
TEMP_SUFFIXES=('.sync-tmp','.restore-tmp','.recovery-tmp','.bounded-tmp')


def obsolete_terrain(relative):
    relative=Path(relative)
    return (relative.parent==Path('data/source_catalogs') and
            relative.name.startswith('terrain_v13_') and relative.suffix=='.json')


def source_file(path):
    return (path.is_file() and '__pycache__' not in path.parts and
            path.name!='.DS_Store' and path.suffix!='.pyc' and
            not path.name.endswith(TEMP_SUFFIXES))


def inventory(root=ROOT):
    root=Path(root);files={};excluded=[]
    for directory in DIRECTORIES:
        for path in sorted((root/directory).rglob('*')):
            if not source_file(path):continue
            relative=path.relative_to(root)
            if obsolete_terrain(relative):excluded.append(relative.as_posix());continue
            files[relative.as_posix()]=path
    for source,destination in ALIASES:files[destination]=root/source
    for name in ARTIFACTS:
        path=root/'artifacts'/name
        if path.exists():files['artifacts/'+name]=path
    return files,excluded


def read_bytes_bounded(source,timeout=30.,role='canonical source'):
    """A blocked cloud-backed read fails explicitly rather than hanging a sync.

    A child process can be terminated even if its OS file read stalls. No other
    copy, archive hash or stale destination substitutes for an unreadable source.
    """
    source=Path(source)
    cat=shutil.which('cat') if os.name!='nt' else None
    command=[cat,str(source)] if cat else [sys.executable,'-c',
        'import pathlib,sys;sys.stdout.buffer.write(pathlib.Path(sys.argv[1]).read_bytes())',str(source)]
    try:
        return subprocess.run(command,stdout=subprocess.PIPE,stderr=subprocess.PIPE,
                              timeout=timeout,check=True).stdout
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError('Timed out reading '+role+': '+source.name) from exc
    except subprocess.CalledProcessError as exc:
        raise RuntimeError('Failed reading '+role+': '+source.name) from exc


def copy_file(source,destination):
    """Atomic copy; verify destination bytes against the actual source read."""
    source,destination=Path(source),Path(destination)
    content=read_bytes_bounded(source)
    digest=hashlib.sha256(content).hexdigest()
    destination.parent.mkdir(parents=True,exist_ok=True)
    # Verify both actual files before avoiding an unnecessary cloud/disk write.
    # A destination read failure can be repaired from the verified source;
    # an unreadable source is never replaced by an older destination.
    locally_materialized=not (getattr(destination.stat(),'st_flags',0) &
        getattr(stat,'SF_DATALESS',0x40000000)) if destination.exists() else False
    if locally_materialized and destination.is_file() and not destination.is_symlink():
        try:
            current=read_bytes_bounded(destination,timeout=5.,role='existing destination')
        except RuntimeError:
            current=None
        if current is not None and hashlib.sha256(current).hexdigest()==digest:
            return digest
    temporary=destination.with_name(destination.name+'.sync-tmp')
    try:
        temporary.write_bytes(content)
        temporary.replace(destination)
        if hashlib.sha256(read_bytes_bounded(destination,role='copied destination')).hexdigest()!=digest:
            raise RuntimeError('Copied payload hash mismatch: '+destination.name)
    finally:
        if temporary.exists():temporary.unlink()
    return digest


def archive_obsolete_terrain(target):
    """Preserve previous derived files without reading or claiming verification."""
    target=Path(target);count=0
    for path in sorted((target/'data/source_catalogs').glob('terrain_v13_*.json')):
        archive=target/'archive_snapshots/legacy_terrain_v13'/path.name
        archive.parent.mkdir(parents=True,exist_ok=True)
        suffix=0
        while archive.exists():
            suffix+=1;archive=archive.with_name(path.name+'.previous-'+str(suffix))
        path.rename(archive);count+=1
    return count


def main(root=ROOT,plan=False):
    root=Path(root);files,excluded=inventory(root)
    if plan:
        print(json.dumps({'current_files':len(files),'excluded_obsolete_terrain_caches':len(excluded),
            'policy':'Canonical observations and current multi-causal inputs retained; obsolete derived terrain stays at canonical root and outside the current bundle manifest'},indent=2))
        return
    manifests={}
    for name in ('training','windows_5070ti_trainer'):
        target=root/name;target.mkdir(exist_ok=True)
        archived=archive_obsolete_terrain(target)
        if archived:print(name+': preserved '+str(archived)+' old terrain copies in archive_snapshots (not reread/verified)',flush=True)
        previous=target/'bundle_manifest.json'
        if previous.exists():
            archived_manifest=target/'archive_snapshots/bundle_manifest.previous.json'
            archived_manifest.parent.mkdir(parents=True,exist_ok=True)
            counter=0
            while archived_manifest.exists():
                counter+=1
                archived_manifest=archived_manifest.with_name('bundle_manifest.previous-'+str(counter)+'.json')
            previous.rename(archived_manifest)
        manifest={}
        def stage(item):
            relative,source=item
            return relative,copy_file(source,target/relative)
        with ThreadPoolExecutor(max_workers=16) as pool:
            futures=[pool.submit(stage,item) for item in files.items()]
            try:
                for completed,future in enumerate(as_completed(futures),1):
                    relative,digest=future.result();manifest[relative]=digest
                    if completed%500==0:print(name+': copied and hash-verified '+str(completed)+'/'+str(len(files)),flush=True)
            except BaseException:
                for future in futures:future.cancel()
                raise
        for cache in list(target.rglob('__pycache__')):shutil.rmtree(cache)
        expected_managed={relative for relative in files if relative.split('/')[0] in DIRECTORIES}
        actual_managed={path.relative_to(target).as_posix() for directory in DIRECTORIES
            for path in (target/directory).rglob('*') if source_file(path) and
            not obsolete_terrain(path.relative_to(target))}
        unexpected=actual_managed-expected_managed
        if unexpected:
            raise RuntimeError('Unexpected current bundle files require review; no fresh manifest written: '+str(sorted(unexpected)[:5]))
        # The current manifest covers exactly the selected canonical inventory.
        # Archive contents are preserved local history, not freshly verified data.
        manifest=dict(sorted(manifest.items()))
        temporary=target/'bundle_manifest.json.sync-tmp'
        temporary.write_text(json.dumps(manifest,indent=2)+'\n')
        temporary.replace(target/'bundle_manifest.json')
        manifests[name]=manifest
    if manifests['training']!=manifests['windows_5070ti_trainer']:
        raise RuntimeError('Bundle source/data mismatch; canonical sources may have changed during synchronization')
    print('Both current bundles copied from canonical bytes; matching verified manifest entries:',len(manifests['training']))
    print('Obsolete canonical terrain_v13 caches retained outside current bundles:',len(excluded))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan',action='store_true',help='Show inventory counts without reading payloads or changing either bundle')
    main(plan=parser.parse_args().plan)
