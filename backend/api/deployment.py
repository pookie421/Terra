"""Small single-process hosting helpers; these do not persist scientific state."""
from __future__ import annotations

from contextlib import contextmanager
import asyncio
import os
from pathlib import Path, PurePosixPath
from threading import Lock

from fastapi import HTTPException
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.responses import JSONResponse
from starlette.staticfiles import StaticFiles


def concurrency_limit(name: str, default: int) -> int:
    try:
        value = int(os.environ.get(name, str(default)))
    except ValueError as exc:
        raise ValueError(f'{name} must be an integer from 1 to 16') from exc
    if not 1 <= value <= 16:
        raise ValueError(f'{name} must be an integer from 1 to 16')
    return value


class AdmissionBudget:
    """Reject excess work immediately instead of retaining an unbounded queue."""

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.active = 0
        self._lock = Lock()

    def acquire(self) -> bool:
        with self._lock:
            if self.active >= self.capacity:
                return False
            self.active += 1
            return True

    def release(self) -> None:
        with self._lock:
            self.active -= 1

    @contextmanager
    def slot(self):
        if not self.acquire():
            raise HTTPException(429, 'Analysis service is busy. Retry in a few seconds.',
                                headers={'Retry-After': '3'})
        try:
            yield
        finally:
            self.release()


class AdmissionMiddleware:
    """Separate expensive analyses from chat so inference cannot occupy their slots."""

    def __init__(self, app, analyses: AdmissionBudget, chats: AdmissionBudget):
        self.app = app
        self.analyses = analyses
        self.chats = chats
        self._running = set()

    async def __call__(self, scope, receive, send):
        if scope['type'] != 'http':
            return await self.app(scope, receive, send)
        path = scope.get('path', '').rstrip('/')
        method = scope.get('method')
        budget = None
        if method == 'POST' and path in ('/api/v1/chat', '/api/v1/agent/chat', '/api/chat'):
            budget = self.chats
        elif ((path == '/api/v1/scan' and method in ('GET', 'POST'))
              or (path.startswith('/api/v1/replays/') and method == 'GET')
              or (path.startswith('/api/v1/scans/') and '/forecast/' in path and method == 'GET')
              or (path.startswith('/api/v1/scans/') and '/scenarios/' in path and method == 'POST')):
            budget = self.analyses
        if budget is None:
            return await self.app(scope, receive, send)
        if not budget.acquire():
            response = JSONResponse({'detail': 'Analysis service is busy. Retry in a few seconds.'},
                                    status_code=429, headers={'Retry-After': '3'})
            return await response(scope, receive, send)
        # A cancelled HTTP caller must not free capacity while a forecast's
        # to_thread work is still running. Keep only admitted tasks alive and
        # release their slot on actual completion, consuming orphaned errors.
        task = asyncio.create_task(self.app(scope, receive, send))
        self._running.add(task)

        def completed(finished):
            self._running.discard(finished)
            budget.release()
            if not finished.cancelled():
                finished.exception()

        task.add_done_callback(completed)
        return await asyncio.shield(task)


class FrontendFiles(StaticFiles):
    """Serve the built SPA without turning missing API/assets into HTML successes."""

    async def get_response(self, path, scope):
        # Starlette normalizes its lookup path with native OS separators.
        # Apply the same route policy on Windows while leaving lookup unchanged.
        policy_path = PurePosixPath(path.replace('\\', '/'))
        parts = policy_path.parts
        if (parts and parts[0] == 'api') or any(part.startswith('.') for part in parts if part != '.'):
            raise StarletteHTTPException(404)
        try:
            response = await super().get_response(path, scope)
        except StarletteHTTPException as exc:
            if (exc.status_code != 404 or scope['method'] not in ('GET', 'HEAD')
                    or policy_path.suffix or (parts and parts[0] == 'assets')):
                raise
            response = await super().get_response('index.html', scope)
        response.headers['X-Content-Type-Options'] = 'nosniff'
        response.headers['Cache-Control'] = ('public, max-age=31536000, immutable'
                                             if parts and parts[0] == 'assets' else 'no-cache')
        return response


def frontend_directory(root: Path) -> Path | None:
    enabled = os.environ.get('TERRA_SERVE_FRONTEND', '').strip().lower()
    if enabled not in ('', '0', 'false', 'no', '1', 'true', 'yes'):
        raise ValueError('TERRA_SERVE_FRONTEND must be true or false')
    if enabled not in ('1', 'true', 'yes'):
        return None
    return Path(os.environ.get('TERRA_FRONTEND_DIST', str(root / 'frontend' / 'dist'))).resolve()


def readiness(model_ready: bool, frontend: Path | None):
    frontend_ready = None if frontend is None else (frontend / 'index.html').is_file()
    ready = model_ready and frontend_ready is not False
    return JSONResponse({'status': 'ready' if ready else 'not_ready', 'model_ready': model_ready,
                         'frontend_required': frontend is not None, 'frontend_ready': frontend_ready,
                         'upstream_providers': 'not_checked', 'operational_warning_ready': False},
                        status_code=200 if ready else 503, headers={'Cache-Control': 'no-store'})
