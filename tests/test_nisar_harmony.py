"""Real subset protocol, reusable pending jobs and credential/mask boundaries."""
import json
import importlib.util
import time
import unittest

import httpx
import numpy as np

from backend.realtime import nisar_harmony as harmony
from backend.realtime.nisar_processor import fetch_nisar


GUNW_PATH = '/science/LSAR/GUNW/grids/frequencyA/unwrappedInterferogram/HH/coherenceMagnitude'
HH_PATH = '/science/LSAR/GCOV/grids/frequencyA/HHHH'
HV_PATH = '/science/LSAR/GCOV/grids/frequencyA/HVHV'


def granule(kind='GUNW'):
    return {'granule_id': 'NISAR_FIXTURE_' + kind, 'cmr_concept_id': 'G123-ASF',
        'collection_concept_id': 'C123-ASF', 'collection_short_name': 'NISAR_L2_' + kind + '_V1',
        'time_start': '2026-10-01T00:00:00Z', 'time_end': '2026-10-13T00:00:00Z',
        'spatial_coverage': {'boxes': [[-1, -1, 3, 3]], 'polygons': []}}


def capability(kind='GUNW'):
    names = [GUNW_PATH] if kind == 'GUNW' else [HH_PATH, HV_PATH]
    return {'conceptId': 'C123-ASF', 'shortName': 'NISAR_L2_' + kind + '_V1',
        'summary': {'subsetting': {'bbox': True, 'variable': True}},
        'variables': [{'name': name, 'href': 'https://cmr.earthdata.nasa.gov/search/concepts/V{}-ASF'.format(i + 100)}
            for i, name in enumerate(names)]}


def output(name):
    return 'https://harmony-prod-public.s3.us-west-2.amazonaws.com/job/' + name.strip('/').replace('/', '_') + '_subsetted.tif?X-Amz-Signature=private-signature'


def raster_bytes(values):
    from rasterio.io import MemoryFile
    from rasterio.transform import from_origin
    values = np.asarray(values, dtype='float32')
    with MemoryFile() as memory:
        with memory.open(driver='GTiff', width=values.shape[1], height=values.shape[0], count=1,
                dtype='float32', crs='EPSG:4326', transform=from_origin(0, 1, 1, 1), nodata=float('nan')) as dataset:
            dataset.write(values, 1)
        return memory.read()


class HarmonyProtocol(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        for cache in (harmony._CAPABILITIES, harmony._JOBS, harmony._POLL):
            await cache.clear()

    async def test_key_absent_is_actionable_auth_state_without_provider_requests(self):
        async def handler(request):
            raise AssertionError('No authenticated request without a token')
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            assets, status = await harmony.measured_subsets(client, [granule()], [(0, 0, 1, 1)],
                token='', deadline=time.monotonic() + 1)
        self.assertEqual(assets, [])
        self.assertEqual(status['status'], 'AUTH_REQUIRED')
        self.assertFalse(status['authentication_configured'])

    async def test_pending_job_is_reused_then_ready_with_exact_variables_and_bbox(self):
        calls = []; ready = False
        async def handler(request):
            calls.append(request)
            if request.url.path == '/capabilities':
                self.assertNotIn('authorization', request.headers)
                return httpx.Response(200, json=capability())
            self.assertEqual(request.headers['authorization'], 'Bearer private-token')
            if request.url.path == '/jobs':
                self.assertTrue(request.url.params['label'].startswith('terra-nisar-'))
                return httpx.Response(200, json={'jobs': []})
            if request.url.path.endswith('/rangeset'):
                self.assertIn('/collections/V100-ASF/', request.url.path)
                self.assertEqual(request.url.params['granuleId'], 'G123-ASF')
                self.assertEqual(request.url.params.get_list('subset'), ['lon(0:1)', 'lat(0:1)'])
                self.assertEqual(request.url.params['maxResults'], '1')
                return httpx.Response(200, json={'jobID': 'job-1'})
            return httpx.Response(200, json={'jobID': 'job-1', 'status': 'successful' if ready else 'running',
                'links': [{'rel': 'data', 'type': 'image/tiff', 'href': output(GUNW_PATH)}] if ready else []})
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            assets, status = await harmony.measured_subsets(client, [granule()], [(0, 0, 1, 1)],
                token='private-token', deadline=time.monotonic() + 2)
            self.assertEqual(assets, [])
            self.assertEqual(status['status'], 'SUBSET_PENDING')
            self.assertEqual(status['jobs'][0]['job_id'], 'job-1')
            self.assertEqual(status['retry_after_seconds'], 10)
            self.assertNotIn('private-token', json.dumps(status))
            ready = True
            await harmony._POLL.clear()  # Simulate the next permitted poll.
            assets, status = await harmony.measured_subsets(client, [granule()], [(0, 0, 1, 1)],
                token='private-token', deadline=time.monotonic() + 2)
        self.assertEqual(len(assets), 1)
        self.assertEqual(assets[0]['observable'], 'coherence')
        self.assertEqual(assets[0]['source_dataset_paths'], [GUNW_PATH])
        self.assertEqual(status['status'], 'RASTER_READY')
        self.assertEqual(sum(r.url.path.endswith('/rangeset') for r in calls), 1)
        self.assertNotIn('private-signature', json.dumps(status))

    async def test_restart_recovers_labeled_job_without_submitting_another(self):
        async def handler(request):
            if request.url.path == '/capabilities':
                return httpx.Response(200, json=capability())
            if request.url.path == '/jobs':
                return httpx.Response(200, json={'jobs': [{'jobID': 'existing-1', 'status': 'running'}]})
            self.assertEqual(request.url.path, '/jobs/existing-1')
            return httpx.Response(200, json={'status': 'running'})
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            _, status = await harmony.measured_subsets(client, [granule()], [(0, 0, 1, 1)],
                token='token', deadline=time.monotonic() + 1)
        self.assertEqual(status['jobs'][0]['job_id'], 'existing-1')
        self.assertEqual(status['status'], 'SUBSET_PENDING')

    async def test_rejected_authorization_and_untrusted_results_do_not_become_measurements(self):
        async def handler(request):
            if request.url.path == '/capabilities':
                return httpx.Response(200, json=capability())
            return httpx.Response(403, json={'message': 'not authorized'})
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            assets, status = await harmony.measured_subsets(client, [granule()], [(0, 0, 1, 1)],
                token='token', deadline=time.monotonic() + 1)
        self.assertEqual(assets, [])
        self.assertEqual(status['status'], 'AUTH_REQUIRED')
        self.assertFalse(harmony.allowed_result_url('https://127.0.0.1/private.tif'))
        self.assertFalse(harmony.allowed_result_url('https://harmony.earthdata.nasa.gov@attacker.example/file.tif'))
        self.assertFalse(harmony.allowed_result_url('http://harmony.earthdata.nasa.gov/file.tif'))


@unittest.skipUnless(importlib.util.find_spec('rasterio'), 'Optional rasterio needed for raster pixel contracts')
class AutomaticRaster(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        await HarmonyProtocol.asyncSetUp(self)

    async def run_scan(self, kind, rasters, redirects=False):
        requests = []
        async def handler(request):
            requests.append(request)
            if request.url.host == 'cmr.earthdata.nasa.gov':
                self.assertNotIn('authorization', request.headers)
                if request.url.path.endswith('.umm_json'):
                    return httpx.Response(200, json={'items': [{'meta': {'concept-id': 'G123-ASF'},
                        'umm': {'CollectionReference': {'ShortName': 'NISAR_L2_' + kind + '_V1'}}}]})
                item = granule(kind)
                item.update(id=item['cmr_concept_id'], producer_granule_id=item['granule_id'], boxes=['-1 -1 3 3'])
                return httpx.Response(200, json={'feed': {'entry': [item]}})
            if request.url.host.endswith(('amazonaws.com', 'cloudfront.net')):
                self.assertNotIn('authorization', request.headers)
                name = HV_PATH if 'HVHV' in request.url.path else (HH_PATH if 'HHHH' in request.url.path else GUNW_PATH)
                return httpx.Response(200, content=rasters[name])
            if request.url.path == '/capabilities':
                return httpx.Response(200, json=capability(kind))
            self.assertEqual(request.headers['authorization'], 'Bearer secret-token')
            if request.url.path.startswith('/service-results/'):
                name = HV_PATH if 'HVHV' in request.url.path else (HH_PATH if 'HHHH' in request.url.path else GUNW_PATH)
                return httpx.Response(307, headers={'location': output(name).replace(
                    'harmony-prod-public.s3.us-west-2.amazonaws.com', 'example.cloudfront.net')})
            if request.url.path == '/jobs':
                return httpx.Response(200, json={'jobs': []})
            if request.url.path.endswith('/rangeset'):
                if redirects:
                    return httpx.Response(303, headers={'location': '/jobs/real-subset-job'})
                return httpx.Response(200, json={'jobID': 'real-subset-job'})
            return httpx.Response(200, json={'jobID': 'real-subset-job', 'status': 'successful', 'links': [
                {'rel': 'data', 'type': 'image/tiff', 'href': (harmony.HARMONY + '/service-results/' +
                    name.strip('/').replace('/', '_') + '.tif') if redirects else output(name)} for name in rasters]})
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            result = await fetch_nisar(client, .5, .5, cells=[
                {'h3_index': 'a', 'latitude': .5, 'longitude': .5},
                {'h3_index': 'b', 'latitude': .5, 'longitude': 1.5}], bbox=(0, 0, 2, 1),
                local_manifest={'assets': []}, earthdata_token='secret-token', timeout_s=20)
        self.assertNotIn('secret-token', json.dumps(result))
        self.assertNotIn('private-signature', json.dumps(result))
        self.assertNotIn('X-Amz-Signature', json.dumps(result))
        return result, requests

    async def test_scan_automatically_samples_actual_coherence_bytes_and_preserves_nodata(self):
        result, requests = await self.run_scan('GUNW', {GUNW_PATH: raster_bytes([[.75, float('nan')]])})
        self.assertEqual(result['mode'], 'AUTHENTICATED_RASTER_MEASURED')
        self.assertEqual(result['measurement_access']['status'], 'MEASURED_AVAILABLE')
        self.assertEqual(result['by_h3']['a']['nisar_coherence'], .75)
        self.assertIsNone(result['by_h3']['b']['nisar_coherence'])
        self.assertIsNone(result['by_h3']['a']['nisar_los_velocity_mm_yr'])
        provenance = result['by_h3']['a']['measurement_provenance']['coherence']
        self.assertEqual(provenance['harmony_job_id'], 'real-subset-job')
        self.assertEqual(provenance['source_dataset_paths'], [GUNW_PATH])
        self.assertEqual(provenance['hash_verification'], 'ACTUAL_DOWNLOADED_SUBSET_BYTES')

    async def test_gcov_ratio_uses_matching_channels_and_rejects_zero_denominator(self):
        result, _ = await self.run_scan('GCOV', {HH_PATH: raster_bytes([[4., 0.]]), HV_PATH: raster_bytes([[2., 2.]])})
        self.assertEqual(result['by_h3']['a']['nisar_polarimetric_moisture_index'], .5)
        self.assertIsNone(result['by_h3']['b']['nisar_polarimetric_moisture_index'])
        self.assertIsNone(result['by_h3']['a']['nisar_coherence'])
        provenance = result['by_h3']['a']['measurement_provenance']['polarimetric_ratio']
        self.assertEqual(provenance['interpretation'], 'UNCALIBRATED_HV_HH_RADAR_RATIO_NOT_SOIL_WATER')
        self.assertIn('denominator_sha256', provenance)

    async def test_official_metadata_and_raster_redirects_keep_token_off_cdn(self):
        result, requests = await self.run_scan('GUNW', {GUNW_PATH: raster_bytes([[.5, float('nan')]])}, redirects=True)
        self.assertEqual(result['by_h3']['a']['nisar_coherence'], .5)
        self.assertEqual(result['measurement_access']['status'], 'MEASURED_AVAILABLE')
        self.assertTrue(any(r.url.host == 'example.cloudfront.net' for r in requests))
        self.assertTrue(all('authorization' not in r.headers for r in requests if r.url.host == 'example.cloudfront.net'))
