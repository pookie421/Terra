"""Radar processing can finish without mutating a previously inspected scan."""
import copy
import threading
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import httpx

from backend.api.radar_refresh import refresh_radar_scan


class RadarRefreshTests(unittest.IsolatedAsyncioTestCase):
    def fixture(self, forecast_key='now'):
        raw = SimpleNamespace(
            scan_id='original-scan', location=SimpleNamespace(latitude=11.548, longitude=76.134),
            radius_km=3.5, terrain=SimpleNamespace(cells={'cell': SimpleNamespace()}),
            nisar={'mode': 'METADATA_ONLY'}, sources={}, generated_at_utc='2026-11-14T00:00:00Z')
        response = {'scan_id': 'original-scan', 'scenario_id': 'original-state',
                    'forecast_view': {'key': forecast_key}, 'scenario': {'rain_delta_mm': 0},
                    'physical_assessment': {'cells': [{'h3_index': 'cell', 'fs_mid': 1.4}]}}
        state = {'scan': raw, 'response': response, 'ground_columns': {'cell': {'source': 'site'}}}
        service = SimpleNamespace(provider_timeout_s=15, _context_lock=threading.RLock(), _scan_states={})
        service.get_scan_state = lambda *args: copy.deepcopy(state)

        async def rebuild(**kwargs):
            self.assertIsNone(kwargs['_snapshot'].scan_id)
            self.assertEqual(kwargs['ground_columns'], state['ground_columns'])
            self.assertEqual(kwargs['rain_delta_mm'], 0)
            self.assertEqual(kwargs['_snapshot'].generated_at_utc, raw.generated_at_utc)
            updated = {**copy.deepcopy(response), 'scan_id': 'refreshed-scan'}
            service._scan_states[('refreshed-scan', 'original-state')] = {'response': copy.deepcopy(updated)}
            return updated

        service.scan = AsyncMock(side_effect=rebuild)
        return service, state

    async def test_completed_measurements_rebuild_new_identity_and_keep_old_state_unchanged(self):
        service, original = self.fixture()
        before = copy.deepcopy(original['response'])
        radar = {'mode': 'MEASURED_NISAR_SUBSET', 'measurement_access': {'status': 'MEASURED_AVAILABLE'}}
        with patch('backend.api.radar_refresh.fetch_nisar', AsyncMock(return_value=radar)) as fetch:
            result = await refresh_radar_scan(service, 'original-scan', 'original-state')
        self.assertEqual(original['response'], before)
        self.assertEqual(original['scan'].nisar, {'mode': 'METADATA_ONLY'})
        snapshot = service.scan.call_args.kwargs['_snapshot']
        self.assertEqual(snapshot.nisar, radar)
        self.assertEqual(snapshot.sources['nisar'].mode, 'MEASURED_NISAR_SUBSET')
        self.assertTrue(result['radar_refresh']['reused_environmental_forcing'])
        self.assertEqual(result['scan_id'], 'refreshed-scan')
        saved = service._scan_states[('refreshed-scan', 'original-state')]['response']
        self.assertEqual(saved['radar_refresh'], result['radar_refresh'])
        self.assertIn('bbox', fetch.call_args.kwargs)

    async def test_pending_job_remains_pending_without_erasing_physics(self):
        service, _ = self.fixture()
        radar = {'mode': 'METADATA_ONLY', 'measurement_access': {'status': 'SUBSET_PENDING',
                 'jobs': [{'job_id': 'real-job', 'status': 'running'}]}}
        with patch('backend.api.radar_refresh.fetch_nisar', AsyncMock(return_value=radar)):
            result = await refresh_radar_scan(service, 'original-scan', 'original-state')
        self.assertEqual(service.scan.call_args.kwargs['_snapshot'].nisar, radar)
        self.assertEqual(result['physical_assessment']['cells'][0]['fs_mid'], 1.4)

    async def test_forecast_and_missing_terrain_are_rejected_before_network(self):
        for key, missing in [('t24', False), ('now', True)]:
            service, state = self.fixture(key)
            if missing:
                state['scan'].terrain.cells = {}
            with self.subTest(key=key, missing=missing), patch('backend.api.radar_refresh.fetch_nisar', AsyncMock()) as fetch:
                with self.assertRaises(ValueError):
                    await refresh_radar_scan(service, 'original-scan', 'original-state')
                fetch.assert_not_called()
                service.scan.assert_not_called()

    async def test_failed_refresh_retains_existing_analysis_and_redacts_error_details(self):
        service, state = self.fixture()
        with patch('backend.api.radar_refresh.fetch_nisar', AsyncMock(side_effect=RuntimeError('signed-url-secret'))):
            with self.assertRaises(ValueError) as error:
                await refresh_radar_scan(service, 'original-scan', 'original-state')
        self.assertNotIn('signed-url-secret', str(error.exception))
        service.scan.assert_not_called()
        self.assertEqual(state['scan'].scan_id, 'original-scan')

    async def test_route_requires_state_identity_and_returns_new_scan(self):
        from backend.api import main
        result = {'scan_id': 'new-scan', 'scenario_id': 'retained-forcing', 'nisar': {'mode': 'MEASURED_NISAR_SUBSET'}}
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=main.app), base_url='http://test') as client:
            with patch.object(main.scan_service, 'refresh_radar_state', AsyncMock(return_value=result)) as refresh:
                self.assertEqual((await client.post('/api/v1/scans/old/radar/refresh', json={})).status_code, 422)
                refresh.assert_not_called()
                response = await client.post('/api/v1/scans/old/radar/refresh', json={'scenario_id': 'old-state'})
                self.assertEqual(response.status_code, 200)
                self.assertEqual(response.json(), result)
                refresh.assert_awaited_once_with('old', 'old-state')
            with patch.object(main.scan_service, 'refresh_radar_state', AsyncMock(side_effect=KeyError('expired'))):
                self.assertEqual((await client.post('/api/v1/scans/old/radar/refresh', json={'scenario_id': 'old-state'})).status_code, 404)


if __name__ == '__main__':
    unittest.main()
