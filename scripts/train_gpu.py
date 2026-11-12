#!/usr/bin/env python3
"""Portable CUDA XGBoost training and explicit Windows BF16 QLoRA stages."""
import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
HERE = Path(__file__).resolve()
ROOT = HERE.parents[1] if HERE.parent.name == 'scripts' else HERE.parent
sys.path.insert(0, str(ROOT))


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def smoke_test(device='cuda', require_torch=False, require_bf16=False):
    started_at = datetime.now(timezone.utc).isoformat()
    import tempfile
    import numpy as np
    from xgboost import XGBClassifier
    rng = np.random.default_rng(42)
    x = rng.normal(size=(64, 4)).astype(np.float32)
    y = (x[:, 0] + x[:, 1] > 0).astype(int)
    model = XGBClassifier(n_estimators=4, max_depth=2, tree_method='hist', device=device,
                          n_jobs=2, random_state=42)
    model.fit(x, y)
    selected = json.loads(model.get_booster().save_config())['learner']['generic_param']['device']
    if device == 'cuda' and not selected.startswith('cuda'):
        raise RuntimeError('XGBoost silently fell back to CPU; CUDA training is not verified')
    expected = model.predict_proba(x)
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / 'smoke.ubj'
        model.save_model(path)
        digest = sha256(path)
        restored = XGBClassifier()
        restored.load_model(path)
        # Runtime device parameters are not the portable model-IO contract.
        restored.set_params(device=device)
        actual = restored.predict_proba(x)
        np.testing.assert_array_equal(actual, expected)
        cross_device = {'status': 'not_applicable_cpu_run'}
        if device == 'cuda':
            restored.set_params(device='cpu')
            cpu = restored.predict_proba(x)
            np.testing.assert_allclose(cpu, expected, rtol=1e-6, atol=1e-7)
            cross_device = {'status': 'passed', 'comparison': 'same saved model, CPU versus CUDA inference',
                            'rtol': 1e-6, 'atol': 1e-7,
                            'maximum_absolute_difference': float(np.max(np.abs(cpu - expected)))}
    result = {'status': 'passed', 'started_at_utc': started_at,
              'xgboost_device': selected, 'tree_method': 'hist', 'train_save_reload': 'passed',
              'same_device_prediction_parity': 'exact', 'serialized_model_sha256': digest,
              'prediction_sha256': hashlib.sha256(expected.tobytes()).hexdigest(),
              'cross_device_inference': cross_device, 'torch_kernel': 'not_requested',
              'scope': 'Exact same-device save/reload check; separately fitted CPU/GPU trees are not claimed identical'}
    if require_torch:
        from scripts.hardware_preflight import cuda_preflight
        result.update(torch_kernel='passed', torch_preflight=cuda_preflight(require_bf16=require_bf16))
    result['finished_at_utc'] = datetime.now(timezone.utc).isoformat()
    return result


def verify_training_artifacts(root, device, smoke):
    """Check the actual saved model/report/frame/holdout relationship, then hash it."""
    import joblib
    import numpy as np
    from backend.ml.physics_informed import KINDS, NAMES, VERSION
    from backend.ml.validation_suite import ranking_metrics
    root = Path(root)
    art = root / 'artifacts'
    model_path = art / 'physics_informed_models.pkl'
    report_path = art / 'physics_informed_ablation_report.json'
    frame_path = root / 'data/physics_informed_training.json'
    holdout_path = root / 'data/event_holdout_manifest.json'
    report = json.loads(report_path.read_text())
    frame = json.loads(frame_path.read_text())
    if report['version'] != VERSION or frame['version'] != VERSION or report['features'] != list(NAMES):
        raise RuntimeError('Saved model/report feature contract mismatch')
    if sha256(model_path) != report['model_sha256'] or sha256(frame_path) != report['dataset_sha256']:
        raise RuntimeError('Saved model or feature-frame SHA-256 differs from its ablation report')
    if sha256(holdout_path) != report['holdout_manifest_sha256']:
        raise RuntimeError('Holdout membership changed during training')
    if report.get('save_reload_prediction_parity', {}).get('status') != 'passed':
        raise RuntimeError('Ablation report lacks exact ensemble save/reload verification')
    if report['environment']['device'] != device:
        raise RuntimeError('Ablation report did not record the requested training device')
    models = joblib.load(model_path)
    if tuple(models) != KINDS or any(m.version != VERSION for m in models.values()):
        raise RuntimeError('Saved artifact lacks the declared four-model ablation')
    # Portable model IO need not retain runtime device parameters. Compare the
    # reloaded predictions on the same requested device as the recorded fit.
    for model in models.values():
        for tree, _ in getattr(model, 'folds', ()):
            tree.set_params(device=device)
    if frame['holdout']:
        x = np.asarray([row['features'] for row in frame['holdout']], float)
        y = [row['label'] for row in frame['holdout']]
        for kind, model in models.items():
            if ranking_metrics(y, model.predict_proba(x)) != report['development_holdout'][kind]:
                raise RuntimeError('Reloaded {} predictions disagree with development-holdout report'.format(kind))
    sft_path = root / 'data/sft_context_trajectories.jsonl'
    sft_report = json.loads((art / 'sft_trajectory_report.json').read_text())
    if sft_report['fusion_model_sha256'] != report['model_sha256'] or sft_report['sha256'] != sha256(sft_path):
        raise RuntimeError('Instruction corpus and fusion model provenance do not match')
    global_report = json.loads((art / 'global_context_benchmark.json').read_text())
    if global_report['version'] != VERSION or global_report['holdout_membership_changed']:
        raise RuntimeError('Global benchmark version or holdout membership mismatch')
    required = ('physics_informed_models.pkl', 'physics_informed_ablation_report.json',
                'scientific_validation_report.json', 'global_context_benchmark.json', 'sft_trajectory_report.json')
    files = {str((art / name).relative_to(root)): sha256(art / name) for name in required}
    files[str(frame_path.relative_to(root))] = sha256(frame_path)
    files[str(holdout_path.relative_to(root))] = sha256(holdout_path)
    files[str(sft_path.relative_to(root))] = sha256(sft_path)
    manifest = {'generated_at_utc': datetime.now(timezone.utc).isoformat(), 'status': 'verified',
                'requested_xgboost_device': device, 'tree_method': 'hist', 'model_version': VERSION,
                'sha256': files, 'hardware_smoke': smoke,
                'saved_model_development_metrics': 'exact',
                'ensemble_save_reload': report.get('save_reload_prediction_parity'),
                'scope': 'Artifact/data/report consistency and target smoke evidence; no independent predictive validation'}
    temporary = art / 'training_artifact_manifest.json.tmp'
    temporary.write_text(json.dumps(manifest, indent=2) + '\n')
    temporary.replace(art / 'training_artifact_manifest.json')
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--smoke-only', action='store_true', help='Only small XGBoost train/save/reload hardware check')
    parser.add_argument('--smoke-test', action='store_true', help='Only the longest-trajectory two-step QLoRA VRAM test; no full fitting')
    parser.add_argument('--device', default='cuda', choices=['cpu', 'cuda'])
    parser.add_argument('--train-llm', action='store_true')
    parser.add_argument('--llm-only', action='store_true')
    parser.add_argument('--check-torch', action='store_true')
    parser.add_argument('--skip-dataset-build', action='store_true', help='Compatibility flag; this trainer always uses frozen features')
    parser.add_argument('--optuna-trials', type=int, default=0)
    args = parser.parse_args()
    if args.optuna_trials:
        parser.error('The current four-way experiment uses fixed hyperparameters; legacy Optuna flags are not supported')
    qlora = args.train_llm or args.llm_only or args.smoke_test
    if qlora and args.device != 'cuda':
        parser.error('QLoRA requires native BF16 CUDA; CPU and FP16 fallbacks are disabled')
    if args.smoke_only and qlora:
        parser.error('--smoke-only is XGBoost-only; use --smoke-test for the QLoRA VRAM test')
    from scripts.hardware_preflight import write_check_report
    (ROOT / 'artifacts').mkdir(exist_ok=True)
    started_at = datetime.now(timezone.utc).isoformat()
    try:
        smoke = smoke_test(args.device, args.check_torch or qlora, require_bf16=qlora)
    except BaseException as exc:
        failure = {'status': 'failed', 'started_at_utc': started_at,
                   'finished_at_utc': datetime.now(timezone.utc).isoformat(),
                   'requested_device': args.device, 'error_type': type(exc).__name__, 'error': str(exc),
                   'scope': 'Current runtime smoke failed; no current GPU or training-readiness certification'}
        failure = write_check_report(ROOT / 'artifacts/gpu_smoke_test.json', failure)
        print(json.dumps(failure, indent=2), flush=True)
        raise
    smoke = write_check_report(ROOT / 'artifacts/gpu_smoke_test.json', smoke)
    print(json.dumps(smoke, indent=2), flush=True)
    if args.smoke_only:
        return 0
    if not args.llm_only and not args.smoke_test:
        from scripts.train_physics_informed import run
        run(ROOT, args.device)
        from scripts.benchmark_global_context import run as global_benchmark
        global_benchmark(ROOT, args.device)
        from scripts import regenerate_sft
        regenerate_sft.ROOT = ROOT
        regenerate_sft.main()
        manifest = verify_training_artifacts(ROOT, args.device, smoke)
        print('Verified {} saved artifacts/data inputs'.format(len(manifest['sha256'])), flush=True)
    if qlora:
        from scripts.train_qwen25_agent import train_qwen25
        train_qwen25(ROOT / 'data/sft_context_trajectories.jsonl', ROOT / 'artifacts/qwen2_5_7b_lora',
                     ROOT / 'artifacts/terra-shield-qwen-q4_k_m.gguf', smoke_test=args.smoke_test)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
