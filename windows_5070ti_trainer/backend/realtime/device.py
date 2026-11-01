"""Optional CUDA → Apple MPS → CPU selection for later tensor workloads."""

from __future__ import annotations

import shutil
import subprocess
from typing import Any, Optional


def detect_compute_device(torch_module: Optional[Any] = None) -> str:
    """Return the fastest available PyTorch device without requiring PyTorch for Phase 1."""
    if torch_module is None:
        try:
            import torch as torch_module  # type: ignore
        except ImportError:
            return "cpu"

    cuda = getattr(getattr(torch_module, "cuda", None), "is_available", None)
    if callable(cuda) and cuda():
        return "cuda"

    backends = getattr(torch_module, "backends", None)
    mps = getattr(backends, "mps", None)
    is_available = getattr(mps, "is_available", None)
    if callable(is_available) and is_available():
        return "mps"
    return "cpu"


def detect_xgboost_device() -> str:
    """XGBoost supports CUDA and CPU; MPS devices use its CPU tree method."""
    nvidia_smi = shutil.which("nvidia-smi")
    if nvidia_smi:
        try:
            result = subprocess.run(
                [nvidia_smi, "-L"], capture_output=True, text=True, timeout=1.0, check=False
            )
            if result.returncode == 0 and "GPU" in result.stdout:
                try:
                    import xgboost
                except ImportError:
                    # The standard Linux XGBoost wheel is CUDA-enabled; an
                    # explicitly CPU-only build will be rejected at fit time.
                    return "cuda"
                build_info = getattr(xgboost, "build_info", None)
                info = build_info() if callable(build_info) else {}
                if info.get("USE_CUDA", True):
                    return "cuda"
        except (OSError, subprocess.TimeoutExpired):
            pass
    return "cpu"
