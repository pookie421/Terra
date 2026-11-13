"""Start the single-process web runtime, including a root-owned volume bootstrap.

The image normally runs as UID 10001. Railway mounts persistent volumes as root;
RAILWAY_RUN_UID=0 allows this helper to prepare one cache directory before it
drops privileges and replaces itself with Uvicorn. No application runs as root.
"""
from __future__ import annotations

import os
from pathlib import Path
import sys
import tempfile
from typing import Mapping

APP_UID = 10001
APP_GID = 10001
VOLUME_ROOT = Path('/data')


def server_port(environ: Mapping[str, str]) -> int:
    value = environ.get('PORT', '8000')
    if not value.isascii() or not value.isdecimal() or not 1 <= int(value) <= 65535:
        raise ValueError('PORT must be an integer from 1 to 65535')
    return int(value)


def prepare_cache(environ: Mapping[str, str]) -> Path:
    cache = Path(environ.get('TERRA_TERRAIN_CACHE_DIR', '/data/terrarium'))
    if not cache.is_absolute():
        raise ValueError('TERRA_TERRAIN_CACHE_DIR must be an absolute path')
    # Restrict the privileged operation to a child of the intended data mount.
    # resolve() also refuses an existing symlink that escapes the mount.
    resolved = cache.resolve()
    if os.geteuid() == 0:
        root = VOLUME_ROOT.resolve()
        if resolved == root or root not in resolved.parents:
            raise ValueError('Root cache bootstrap is restricted to a child of /data')
        cache.mkdir(parents=True, exist_ok=True)
        os.chown(resolved, APP_UID, APP_GID)
        os.setgroups([])
        os.setgid(APP_GID)
        os.setuid(APP_UID)
    else:
        cache.mkdir(parents=True, exist_ok=True)
    # Fail at startup with a clear message, before a user's first terrain request.
    try:
        with tempfile.TemporaryFile(dir=cache):
            pass
    except OSError as exc:
        raise RuntimeError('Terrain cache is not writable; check the persistent volume and startup UID') from exc
    return cache


def main() -> None:
    port = server_port(os.environ)
    prepare_cache(os.environ)
    args = [sys.executable, '-m', 'uvicorn', 'backend.api.main:app',
            '--host', '0.0.0.0', '--port', str(port), '--workers', '1',
            '--no-server-header', '--timeout-graceful-shutdown', '60']
    os.execv(sys.executable, args)


if __name__ == '__main__':
    main()
