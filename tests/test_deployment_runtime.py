"""Offline contracts for a single-process public website and bounded analysis work."""
import asyncio
import builtins
import importlib
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import httpx
from fastapi import FastAPI

from backend.api import main
from backend.api.deployment import AdmissionBudget, AdmissionMiddleware, FrontendFiles, concurrency_limit


class WebsiteServing(unittest.IsolatedAsyncioTestCase):
    async def test_absent_replay_returns_404_without_training_scripts(self):
        from backend.api import replay
        original_import = builtins.__import__

        def runtime_only_import(name, *args, **kwargs):
            if name == 'scripts.snapshot_replay':
                raise AssertionError('Production must not import absent replay tooling')
            return original_import(name, *args, **kwargs)

        with tempfile.TemporaryDirectory() as directory, patch('builtins.__import__', side_effect=runtime_only_import):
            importlib.reload(replay)
            original_load = replay.load_replay
            with patch.object(replay, 'load_replay', side_effect=lambda name: original_load(name, root=Path(directory))):
                async with httpx.AsyncClient(transport=httpx.ASGITransport(app=main.app), base_url='http://test') as client:
                    response = await client.get('/api/v1/replays/monaco')
                    self.assertEqual(response.status_code, 404)
                    self.assertIn('not installed', response.json()['detail'])

    async def test_spa_routes_assets_and_api_errors_remain_distinct(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dist = root / 'dist'
            (dist / 'assets').mkdir(parents=True)
            (dist / 'index.html').write_text('<html>Website entry</html>')
            (dist / 'assets' / 'app-123.js').write_text('console.log("website");')
            (root / 'private.txt').write_text('Never serve outside the build')
            (dist / '.env').write_text('Never serve hidden files')
            try:
                (dist / 'assets' / 'linked.txt').symlink_to(root / 'private.txt')
                symlink_available = True
            except (OSError, NotImplementedError):
                # Native Windows CI may not grant the symlink privilege.
                symlink_available = False
            app = FastAPI()

            @app.get('/api/v1/ping')
            async def ping():
                return {'alive': True}

            app.mount('/', FrontendFiles(directory=str(dist), html=True))
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url='http://test') as client:
                for path in ('/', '/analysis/shared-view'):
                    response = await client.get(path)
                    self.assertEqual(response.status_code, 200)
                    self.assertIn('Website entry', response.text)
                    self.assertEqual(response.headers['cache-control'], 'no-cache')
                script = await client.get('/assets/app-123.js')
                self.assertEqual(script.status_code, 200)
                self.assertIn('javascript', script.headers['content-type'])
                self.assertIn('immutable', script.headers['cache-control'])
                self.assertEqual(script.headers['x-content-type-options'], 'nosniff')
                self.assertEqual((await client.get('/api/v1/ping')).json(), {'alive': True})
                for path in ('/api/v1/missing', '/assets/missing.js', '/.env', '/%2e%2e/private.txt'):
                    with self.subTest(path=path):
                        response = await client.get(path)
                        self.assertEqual(response.status_code, 404)
                        self.assertNotIn('Website entry', response.text)
                        self.assertNotIn('Never serve', response.text)
                with self.subTest(path='/assets/linked.txt'):
                    if not symlink_available:
                        self.skipTest('This filesystem/account cannot create symlinks')
                    response = await client.get('/assets/linked.txt')
                    self.assertEqual(response.status_code, 404)
                    self.assertNotIn('Never serve', response.text)
                self.assertEqual((await client.post('/api/v1/missing')).status_code, 404)

    async def test_readiness_requires_model_and_enabled_frontend_without_calling_providers(self):
        with tempfile.TemporaryDirectory() as directory:
            dist = Path(directory)
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=main.app), base_url='http://test') as client:
                with patch.object(main.scan_service, 'context_models', None), patch.object(main, 'frontend_dist', None):
                    self.assertEqual((await client.get('/api/v1/health')).status_code, 200)
                    response = await client.get('/api/v1/ready')
                    self.assertEqual(response.status_code, 503)
                    self.assertFalse(response.json()['model_ready'])
                with patch.object(main.scan_service, 'context_models', object()), patch.object(main, 'frontend_dist', dist):
                    self.assertEqual((await client.get('/api/v1/ready')).status_code, 503)
                    (dist / 'index.html').write_text('<html>Ready</html>')
                    response = await client.get('/api/v1/ready')
                    self.assertEqual(response.status_code, 200)
                    self.assertTrue(response.json()['frontend_ready'])
                    self.assertEqual(response.json()['upstream_providers'], 'not_checked')
                    self.assertFalse(response.json()['operational_warning_ready'])
                    self.assertEqual(response.headers['cache-control'], 'no-store')


class AdmissionControl(unittest.IsolatedAsyncioTestCase):
    async def test_overload_rejects_without_queueing_while_health_and_chat_stay_available(self):
        analyses, chats = AdmissionBudget(1), AdmissionBudget(1)
        started, release = asyncio.Event(), asyncio.Event()
        app = FastAPI()
        app.add_middleware(AdmissionMiddleware, analyses=analyses, chats=chats)

        @app.post('/api/v1/scan')
        async def scan():
            started.set()
            await release.wait()
            return {'scan': 'finished'}

        @app.get('/api/v1/health')
        async def health():
            return {'alive': True}

        @app.post('/api/v1/chat')
        async def chat():
            return {'briefing_guard': 'VERIFIED_TEMPLATE'}

        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url='http://test') as client:
            first = asyncio.create_task(client.post('/api/v1/scan'))
            await asyncio.wait_for(started.wait(), timeout=2)
            try:
                excess = await client.post('/api/v1/scan')
                self.assertEqual(excess.status_code, 429)
                self.assertEqual(excess.headers['retry-after'], '3')
                self.assertEqual(analyses.active, 1)
                self.assertEqual((await client.get('/api/v1/health')).status_code, 200)
                self.assertEqual((await client.post('/api/v1/chat')).status_code, 200)
            finally:
                release.set()
                self.assertEqual((await first).status_code, 200)
            self.assertEqual(analyses.active, 0)
            self.assertEqual((await client.post('/api/v1/scan')).status_code, 200)

    async def test_caller_cancellation_retains_capacity_until_work_finishes_and_errors_release_it(self):
        budget = AdmissionBudget(1)
        started, finish, completed = asyncio.Event(), asyncio.Event(), asyncio.Event()
        app = FastAPI()
        app.add_middleware(AdmissionMiddleware, analyses=budget, chats=AdmissionBudget(1))

        @app.get('/api/v1/scan')
        async def blocked_scan():
            started.set()
            await finish.wait()
            completed.set()
            return {'scan': 'finished'}

        @app.post('/api/v1/scan')
        async def failed_scan():
            raise RuntimeError('Fixture failure')

        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app, raise_app_exceptions=False), base_url='http://test') as client:
            pending = asyncio.create_task(client.get('/api/v1/scan'))
            await asyncio.wait_for(started.wait(), timeout=2)
            pending.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await pending
            self.assertEqual(budget.active, 1)
            self.assertEqual((await client.post('/api/v1/scan')).status_code, 429)
            finish.set()
            await asyncio.wait_for(completed.wait(), timeout=2)
            await asyncio.sleep(0)  # Let the completed request's callback release its slot.
            self.assertEqual(budget.active, 0)
            self.assertEqual((await client.post('/api/v1/scan')).status_code, 500)
            self.assertEqual(budget.active, 0)

    async def test_forecast_replay_scenario_and_legacy_chat_cannot_bypass_analysis_limit(self):
        self.assertTrue(main.analysis_budget.acquire())
        remaining = main.analysis_budget.capacity - 1
        for _ in range(remaining):
            self.assertTrue(main.analysis_budget.acquire())
        try:
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=main.app), base_url='http://test') as client:
                for method, path, body in (
                    ('GET', '/api/v1/replays/fixture', None),
                    ('GET', '/api/v1/scans/fixture/forecast/24h', None),
                    ('POST', '/api/v1/scans/fixture/scenarios/fixture', {'changes': {}}),
                    ('POST', '/api/v1/chat', {'message': 'Legacy scan', 'autonomous': False}),
                ):
                    with self.subTest(path=path):
                        response = await client.request(method, path, json=body)
                        self.assertEqual(response.status_code, 429)
                        self.assertEqual(response.headers['retry-after'], '3')
        finally:
            for _ in range(remaining + 1):
                main.analysis_budget.release()

    def test_invalid_concurrency_settings_fail_explicitly(self):
        for value in ('0', '-1', '17', 'nope'):
            with self.subTest(value=value), patch.dict('os.environ', {'TERRA_MAX_CONCURRENT_ANALYSES': value}):
                with self.assertRaises(ValueError):
                    concurrency_limit('TERRA_MAX_CONCURRENT_ANALYSES', 2)


if __name__ == '__main__':
    unittest.main()
