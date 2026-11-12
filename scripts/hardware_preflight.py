#!/usr/bin/env python3
"""Execute CUDA kernels before accepting a Windows/Blackwell training environment.

This is a target-machine check, not a claim that an unexecuted GPU path works.
Importing this module does not import torch or require a GPU.
"""
import argparse
import json
import platform
import re
from datetime import datetime, timezone
from pathlib import Path


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def write_check_report(path, value):
    """Publish current evidence atomically and retain the previous bytes locally.

    A failed new check must never leave an earlier successful report at the
    current path. No previous report is parsed or used as fallback evidence.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(value)
    previous = None
    if path.exists():
        history = path.parent / 'previous_checks'
        history.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')
        previous = history / '{}.{}{}'.format(path.stem, stamp, path.suffix)
        counter = 0
        while previous.exists():
            counter += 1
            previous = history / '{}.{}.{}{}'.format(path.stem, stamp, counter, path.suffix)
        payload['previous_report_preserved'] = str(previous.relative_to(path.parent))
    temporary = path.with_suffix(path.suffix + '.tmp')
    try:
        temporary.write_text(json.dumps(payload, indent=2, allow_nan=False) + '\n', encoding='utf-8')
        if previous is not None:
            path.rename(previous)
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return payload


def cuda_version(value):
    match = re.match(r'^(\d+)\.(\d+)', value or '')
    return tuple(map(int, match.groups())) if match else (0, 0)


def _probe_kernels(torch, index, bf16, require_bf16):
    import torch.nn.functional as functional
    device = 'cuda:{}'.format(index)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision('high')
    a = torch.ones((32, 32), device=device, requires_grad=True)
    value = (a @ a).sum()
    value.backward()
    torch.cuda.synchronize(index)
    if value.item() != 32768. or not torch.isfinite(a.grad).all().item():
        raise RuntimeError('CUDA float32 forward/backward kernel verification failed')
    result = {'float32_matmul_backward': 'passed', 'bf16_matmul_backward': 'not_supported',
              'fused_sdpa_backends': [], 'sdpa_probe_errors': {}}
    if bf16:
        b = torch.ones((32, 32), device=device, dtype=torch.bfloat16, requires_grad=True)
        v = (b @ b).float().sum()
        v.backward()
        torch.cuda.synchronize(index)
        if v.item() != 32768. or not torch.isfinite(b.grad).all().item():
            raise RuntimeError('CUDA BF16 forward/backward kernel verification failed')
        result['bf16_matmul_backward'] = 'passed'
        from torch.nn.attention import SDPBackend, sdpa_kernel
        # Never count the quadratic-memory MATH implementation as a fused kernel.
        for name in ('FLASH_ATTENTION', 'EFFICIENT_ATTENTION', 'CUDNN_ATTENTION'):
            backend = getattr(SDPBackend, name, None)
            if backend is None:
                continue
            try:
                q = torch.randn((1, 4, 16, 64), device=device, dtype=torch.bfloat16, requires_grad=True)
                with sdpa_kernel(backends=[backend]):
                    output = functional.scaled_dot_product_attention(q, q, q, is_causal=True)
                    output.float().square().mean().backward()
                torch.cuda.synchronize(index)
                if not torch.isfinite(q.grad).all().item():
                    raise RuntimeError('Non-finite SDPA gradient')
                result['fused_sdpa_backends'].append(name)
            except RuntimeError as exc:
                result['sdpa_probe_errors'][name] = str(exc)
        if require_bf16 and not result['fused_sdpa_backends']:
            raise RuntimeError('No native fused BF16 SDPA backend executed successfully; '
                               'refusing the high-memory MATH fallback for 8192-token QLoRA')
    return result


def _probe_bitsandbytes(torch, index):
    import bitsandbytes as bnb
    device = 'cuda:{}'.format(index)
    layer = bnb.nn.Linear4bit(64, 64, bias=False, compute_dtype=torch.bfloat16,
                             compress_statistics=True, quant_type='nf4').to(device)
    x = torch.randn((2, 64), device=device, dtype=torch.bfloat16, requires_grad=True)
    output = layer(x)
    output.float().square().mean().backward()
    torch.cuda.synchronize(index)
    if not torch.isfinite(output).all().item() or not torch.isfinite(x.grad).all().item():
        raise RuntimeError('bitsandbytes NF4 CUDA forward/backward produced non-finite values')
    # >100k elements exercises the paged-state allocation path, rather than
    # the small-parameter nonpaged exception in bitsandbytes optimizers.
    parameter = torch.nn.Parameter(torch.ones(131072, device=device, dtype=torch.bfloat16))
    optimizer = bnb.optim.PagedAdamW8bit([parameter], lr=1e-4)
    parameter.float().square().mean().backward()
    optimizer.step()
    torch.cuda.synchronize(index)
    states = optimizer.state[parameter]
    paged = bool(getattr(states.get('state1'), 'is_paged', False))
    if not paged:
        raise RuntimeError('PagedAdamW8bit did not allocate paged optimizer state')
    if not torch.isfinite(parameter).all().item():
        raise RuntimeError('PagedAdamW8bit produced non-finite parameters')
    return {'bitsandbytes_version': bnb.__version__, 'nf4_forward_backward': 'passed',
            'paged_adamw_8bit_step': 'passed', 'paged_state_allocation': paged}


def cuda_preflight(torch_module=None, require_bf16=False, require_blackwell=False,
                   check_bitsandbytes=False):
    """Return actual GPU/kernel evidence, or raise before model acquisition/training."""
    started_at = utc_now()
    if torch_module is None:
        import torch as torch_module
    torch = torch_module
    if not torch.cuda.is_available():
        raise RuntimeError('PyTorch cannot execute CUDA. Install a cu128 wheel and a compatible '
                           'NVIDIA driver; CPU PyTorch is not CUDA certification.')
    index = torch.cuda.current_device()
    capability = tuple(torch.cuda.get_device_capability(index))
    runtime = cuda_version(torch.version.cuda)
    architectures = list(torch.cuda.get_arch_list())
    if require_blackwell and capability < (12, 0):
        raise RuntimeError('This target check requires compute capability 12.0+; found {}'.format(capability))
    if capability >= (12, 0):
        if runtime < (12, 8):
            raise RuntimeError('Blackwell sm_120 requires PyTorch CUDA 12.8 or newer; found {}'.format(torch.version.cuda))
        native_target = 'sm_{}{}'.format(*capability)
        ptx_target = 'compute_{}{}'.format(*capability)
        # Same-major lower-minor SASS is compatible with a newer minor device.
        compatible_sass = any(re.fullmatch(r'sm_{}\d+'.format(capability[0]), arch)
                              and int(arch.split('_')[1][len(str(capability[0])):]) <= capability[1]
                              for arch in architectures)
        if native_target not in architectures and ptx_target not in architectures and not compatible_sass:
            raise RuntimeError('Torch does not advertise a compatible {} target: {}'.format(native_target, architectures))
    try:
        bf16 = bool(torch.cuda.is_bf16_supported(including_emulation=False))
    except TypeError:
        bf16 = capability >= (8, 0) and bool(torch.cuda.is_bf16_supported())
    if require_bf16 and not bf16:
        raise RuntimeError('BF16 CUDA support is required for QLoRA; FP16 fallback is disabled')
    properties = torch.cuda.get_device_properties(index)
    free_bytes, total_bytes = torch.cuda.mem_get_info(index)
    report = {'status': 'passed', 'started_at_utc': started_at,
              'platform': platform.system(), 'torch_version': torch.__version__,
              'cuda_runtime': torch.version.cuda, 'device_index': index,
              'gpu': torch.cuda.get_device_name(index), 'compute_capability': list(capability),
              'compiled_architectures': architectures, 'total_vram_bytes': int(properties.total_memory),
              'driver_visible_total_vram_bytes': int(total_bytes), 'free_vram_bytes': int(free_bytes),
              'total_vram_gib': round(properties.total_memory / 1024**3, 3), 'bf16_supported': bf16,
              'tf32_enabled': True,
              'scope': 'Actual small-kernel preflight; full longest-trajectory QLoRA VRAM smoke is still required'}
    report.update(_probe_kernels(torch, index, bf16, require_bf16))
    if check_bitsandbytes:
        if not bf16:
            raise RuntimeError('BF16 is required for the NF4/bitsandbytes preflight')
        report.update(_probe_bitsandbytes(torch, index))
    report['finished_at_utc'] = utc_now()
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--require-bf16', action='store_true')
    parser.add_argument('--require-blackwell', action='store_true')
    parser.add_argument('--check-bitsandbytes', action='store_true')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    started_at = utc_now()
    try:
        result = cuda_preflight(require_bf16=args.require_bf16,
                                require_blackwell=args.require_blackwell,
                                check_bitsandbytes=args.check_bitsandbytes)
    except BaseException as exc:
        result = {'status': 'failed', 'started_at_utc': started_at, 'finished_at_utc': utc_now(),
                  'error_type': type(exc).__name__, 'error': str(exc),
                  'scope': 'Current CUDA preflight failed; no current GPU or training-readiness certification'}
        if args.output:
            result = write_check_report(args.output, result)
        print(json.dumps(result, indent=2))
        if isinstance(exc, (KeyboardInterrupt, SystemExit)):
            raise
        return 1
    if args.output:
        result = write_check_report(args.output, result)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
