#!/usr/bin/env python3
"""Portable XGBoost trainer; QLoRA is an explicit separate stage."""
import argparse
import json
import sys
from pathlib import Path
HERE=Path(__file__).resolve()
ROOT=HERE.parents[1] if HERE.parent.name=='scripts' else HERE.parent
sys.path.insert(0,str(ROOT))


def smoke_test(device='cuda', require_torch=False):
    import tempfile
    import numpy as np
    from xgboost import XGBClassifier
    rng=np.random.default_rng(42);x=rng.normal(size=(64,4)).astype(np.float32);y=(x[:,0]+x[:,1]>0).astype(int)
    model=XGBClassifier(n_estimators=4,max_depth=2,tree_method='hist',device=device,random_state=42)
    model.fit(x,y)
    selected=json.loads(model.get_booster().save_config())['learner']['generic_param']['device']
    if device=='cuda' and not selected.startswith('cuda'):
        raise RuntimeError('XGBoost silently fell back to CPU; CUDA training is not verified')
    with tempfile.TemporaryDirectory() as directory:
        path=Path(directory)/'smoke.ubj';model.save_model(path)
        restored=XGBClassifier();restored.load_model(path)
        np.testing.assert_allclose(restored.predict_proba(x),model.predict_proba(x),rtol=1e-6,atol=1e-7)
    result={'xgboost_device':selected,'train_save_reload':'passed','torch_kernel':'not_requested'}
    if require_torch:
        import torch
        if not torch.cuda.is_available():raise RuntimeError('PyTorch cannot see CUDA')
        a=torch.ones((32,32),device='cuda');assert (a@a).sum().item()==32768.
        torch.cuda.synchronize()
        result.update(torch_kernel='passed',gpu=torch.cuda.get_device_name(0),torch_version=torch.__version__)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--smoke-only',action='store_true')
    parser.add_argument('--device',default='cuda',choices=['cpu','cuda'])
    parser.add_argument('--train-llm',action='store_true')
    parser.add_argument('--llm-only',action='store_true')
    parser.add_argument('--check-torch',action='store_true')
    parser.add_argument('--skip-dataset-build',action='store_true',help='Compatibility flag; this trainer always uses frozen features')
    parser.add_argument('--optuna-trials',type=int,default=0)
    args=parser.parse_args()
    if args.optuna_trials:parser.error('The current four-way experiment uses fixed hyperparameters; legacy Optuna flags are not supported')
    smoke=smoke_test(args.device,args.check_torch or args.train_llm or args.llm_only)
    print(json.dumps(smoke,indent=2),flush=True)
    (ROOT/'artifacts').mkdir(exist_ok=True)
    (ROOT/'artifacts/gpu_smoke_test.json').write_text(json.dumps(smoke,indent=2)+'\n')
    if args.smoke_only:return 0
    if not args.llm_only:
        from scripts.train_physics_informed import run
        run(ROOT,args.device)
        from scripts.benchmark_global_context import run as global_benchmark
        global_benchmark(ROOT,args.device)
        from scripts import regenerate_sft
        regenerate_sft.ROOT=ROOT
        regenerate_sft.main()
    if args.train_llm or args.llm_only:
        try:
            from train_qwen25_agent import train_qwen25
        except ImportError:
            from scripts.train_qwen25_agent import train_qwen25
        train_qwen25(ROOT/'data/sft_context_trajectories.jsonl',ROOT/'artifacts/qwen2_5_7b_lora',ROOT/'artifacts/terra-shield-qwen-q4_k_m.gguf')
    return 0
if __name__=='__main__':raise SystemExit(main())
