"""NASA raster/metadata contracts; provider traffic uses deterministic fixtures."""
import asyncio
import importlib.util
import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

import httpx
import numpy as np

from backend.realtime.nasa_lhasa import fetch_lhasa
from backend.realtime.nisar_processor import (CMR, CMR_UMM, COLLECTIONS, _sample_hdf5, _covers,
    _validate_asset, fetch_nisar, parse_granule)


def cell(identity='a', lat=.001, lon=.001, slope=30.):
    return {'h3_index': identity, 'latitude': lat, 'longitude': lon, 'slope_deg': slope}


def granule():
    return {'id': 'G123-ASF', 'producer_granule_id': 'NISAR_FIXTURE_GUNW',
        'collection_concept_id': 'C123-ASF', 'time_start': '2026-10-01T00:00:00Z',
        'time_end': '2026-10-01T00:01:00Z', 'boxes': ['-1 -1 1 1'],
        'links': [{'rel': 'http://esipfed.org/ns/fedsearch/1.1/browse#',
            'href': 's3://bucket/browse.png'}, {'rel': 'http://esipfed.org/ns/fedsearch/1.1/browse#',
            'href': 'https://nisar.asf.earthdatacloud.nasa.gov/browse.png?private=signed'},
            {'rel': 'http://esipfed.org/ns/fedsearch/1.1/data#',
             'href': 'https://nisar.asf.earthdatacloud.nasa.gov/a.h5'}]}


class LhasaNativeSampling(unittest.IsolatedAsyncioTestCase):
    async def test_native_batch_zero_nodata_and_categorical_susceptibility(self):
        calls = []
        async def handler(request):
            calls.append(request)
            if not request.url.path.endswith('getSamples'):
                return httpx.Response(200, json={'extent': {'xmin': -180., 'ymax': 90.,
                    'spatialReference': {'wkid': 4326}}, 'pixelSizeX': 1/120, 'pixelSizeY': 1/120})
            self.assertEqual(request.url.params['interpolation'], 'RSP_NearestNeighbor')
            points = json.loads(request.url.params['geometry'])['points']
            self.assertEqual(len(points), 2)  # Three H3 cells, two native pixels.
            samples = []
            for i, (x, y) in enumerate(points):
                value = '4' if 'Susceptibility' in request.url.path else ('0' if i == 0 else 'NoData')
                samples.append({'location': {'x': x, 'y': y}, 'locationId': i, 'value': value})
            return httpx.Response(200, json={'samples': list(reversed(samples))})
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            result = await fetch_lhasa(client, [cell('a'), cell('b', lon=.002), cell('c', lon=.02)],
                                       weather={'rain_24h_mm': 20, 'ari_7d_mm': 60})
        self.assertEqual(len(calls), 6)
        self.assertEqual(result['by_h3']['a']['probability'], 0.)
        self.assertEqual(result['by_h3']['b']['probability'], 0.)
        self.assertIsNone(result['by_h3']['c']['probability'])
        self.assertIsNotNone(result['by_h3']['c']['reference_proxy_score'])
        self.assertEqual(result['by_h3']['a']['susceptibility_class'], 4)
        self.assertEqual(result['mode'], 'MIXED_LIVE_AND_REFERENCE')
        self.assertEqual(result['product_version'], 'NOT_DECLARED_BY_SERVICE')
        self.assertIsNone(result['product_valid_time_utc'])
        self.assertFalse(result['capabilities']['supports_72h_forecast'])
        self.assertFalse(result['capabilities']['supports_100m_predictions'])
        self.assertEqual(result['capabilities']['forecast_horizons_hours'], [])
        pixels = result['native_pixels_geojson_by_product']['today']['features']
        self.assertEqual(len(pixels), 1)  # Shared official pixel, not two H3 predictions.
        self.assertEqual(pixels[0]['properties']['value'], 0.)
        self.assertEqual(pixels[0]['properties']['source_resolution_arcsec'], [30., 30.])
        self.assertEqual(result['by_h3']['a']['coarse_pixel_id_by_product']['today'],
                         result['by_h3']['a']['coarse_pixel_id'])

    async def test_each_product_keeps_its_own_native_grid_geometry(self):
        async def handler(request):
            if not request.url.path.endswith('getSamples'):
                size = 1/60 if 'Tomorrow' in request.url.path else 1/120
                return httpx.Response(200, json={'extent': {'xmin': -180., 'ymax': 90.,
                    'spatialReference': {'wkid': 4326}}, 'pixelSizeX': size, 'pixelSizeY': size})
            x, y = json.loads(request.url.params['geometry'])['points'][0]
            return httpx.Response(200, json={'samples': [{'location': {'x': x, 'y': y},
                'value': '3' if 'Susceptibility' in request.url.path else '.25'}]})
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            result = await fetch_lhasa(client, [cell()])
        row = result['cells'][0]
        self.assertNotEqual(row['coarse_pixel_id_by_product']['today'], row['coarse_pixel_id_by_product']['tomorrow'])
        pixels = result['native_pixels_geojson_by_product']
        self.assertEqual(pixels['today']['features'][0]['properties']['source_resolution_arcsec'], [30., 30.])
        self.assertEqual(pixels['tomorrow']['features'][0]['properties']['source_resolution_arcsec'], [60., 60.])
        ring = pixels['tomorrow']['features'][0]['geometry']['coordinates'][0]
        self.assertAlmostEqual(ring[1][0] - ring[0][0], 1/60)

    async def test_out_of_domain_keeps_explicit_reference_and_no_nasa_probability(self):
        async def handler(request):
            raise AssertionError('All products are outside support at80N')
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            result = await fetch_lhasa(client, [cell(lat=80)], weather={'rain_24h_mm': 30, 'ari_7d_mm': 60})
        row = result['cells'][0]
        self.assertFalse(row['in_hazard_domain'])
        self.assertIsNone(row['probability'])
        self.assertEqual(row['provenance'], 'LHASA_REFERENCE_EMULATION')
        self.assertTrue(0 < row['reference_proxy_score'] < 1)
        self.assertFalse(result['reference_proxy']['is_probability'])
        self.assertFalse(result['exact_lhasa_2_0_reproduction'])

    async def test_one_aggregate_deadline_and_missing_rain_is_not_dry(self):
        async def handler(request):
            await asyncio.sleep(.3)
            return httpx.Response(200, json={'extent': {'xmin': -180, 'ymax': 90}, 'pixelSizeX': 1/120, 'pixelSizeY': 1/120})
        started = time.monotonic()
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            result = await fetch_lhasa(client, [cell()], timeout_s=.025)
        self.assertLess(time.monotonic() - started, .2)
        self.assertIsNone(result['cells'][0]['reference_proxy_score'])
        self.assertTrue(all(v == 'AGGREGATE_TIMEOUT' for v in result['errors'].values()))

    async def test_invalid_scaled_probability_is_rejected_not_rescaled(self):
        async def handler(request):
            if request.url.path.endswith('getSamples'):
                point = json.loads(request.url.params['geometry'])['points'][0]
                return httpx.Response(200, json={'samples': [{'location': {'x': point[0], 'y': point[1]}, 'value': '255'}]})
            return httpx.Response(200, json={'extent': {'xmin': -180, 'ymax': 90}, 'pixelSizeX': 1/120, 'pixelSizeY': 1/120})
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            result = await fetch_lhasa(client, [cell()])
        self.assertIsNone(result['cells'][0]['probability'])
        self.assertIsNone(result['cells'][0]['susceptibility_class'])


class NisarMetadata(unittest.IsolatedAsyncioTestCase):
    async def test_real_metadata_identity_orbit_enrichment_not_radar_measurement(self):
        calls = []
        async def handler(request):
            calls.append(request)
            self.assertNotIn('authorization', request.headers)
            if request.url.host == 'harmony.earthdata.nasa.gov':
                return httpx.Response(200, json={'conceptId': 'C123-ASF', 'summary': {}, 'variables': []})
            if request.url.path.endswith('.umm_json'):
                return httpx.Response(200, json={'items': [{'meta': {'concept-id': 'G123-ASF'},
                    'umm': {'CollectionReference': {'ShortName': 'NISAR_L2_GUNW_PROVISIONAL_V1'},
                        'AdditionalAttributes': [{'Name': 'ASCENDING_DESCENDING', 'Values': ['ASCENDING']},
                        {'Name': 'PRODUCT_VERSION', 'Values': ['1.0.20']}]}}]})
            self.assertEqual(set(request.url.params.get_list('short_name[]')), set(COLLECTIONS))
            return httpx.Response(200, json={'feed': {'entry': [granule()]}}, headers={'CMR-Hits': '42'})
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            result = await fetch_nisar(client, 0, 0, cells=[cell()], earthdata_token='must-not-be-exposed')
        self.assertEqual(result['mode'], 'METADATA_LIVE')
        self.assertEqual(result['granules'][0]['orbit_direction'], 'ASCENDING')
        self.assertEqual(result['granules'][0]['product_version'], '1.0.20')
        self.assertEqual(result['granules'][0]['collection_concept_id'], 'C123-ASF')
        self.assertEqual(result['granules'][0]['browse_url'], 'https://nisar.asf.earthdatacloud.nasa.gov/browse.png')
        self.assertEqual(result['by_h3']['a']['nisar_granule_count'], 1)
        self.assertIsNone(result['by_h3']['a']['nisar_coherence'])
        self.assertIsNone(result['by_h3']['a']['nisar_los_velocity_mm_yr'])
        self.assertNotIn('must-not-be-exposed', json.dumps(result))
        self.assertNotIn('private=', json.dumps(result))
        self.assertEqual(len(calls), 3)  # Public CMR/UMM plus collection capability check.
        self.assertEqual(result['measurement_access']['status'], 'SUBSET_UNAVAILABLE')

    async def test_dateline_search_deduplicates_granules_and_preserves_total_hits(self):
        async def handler(request):
            if request.url.path.endswith('.umm_json'):
                return httpx.Response(200, json={'items': []})
            return httpx.Response(200, json={'feed': {'entry': [granule()]}}, headers={'CMR-Hits': '12'})
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            result = await fetch_nisar(client, 0, 179.99, earthdata_token='')
        self.assertEqual(len(result['bbox_queries']), 2)
        self.assertEqual(result['granule_count'], 1)
        self.assertEqual(result['query_hits_by_bbox'], [12, 12])

    async def test_no_cmr_response_is_not_verified_no_coverage(self):
        async def handler(request):
            raise httpx.ConnectError('unreachable')
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            result = await fetch_nisar(client, 0, 0, earthdata_token='')
        self.assertEqual(result['mode'], 'UNAVAILABLE')
        self.assertIsNone(result['cells'][0]['nisar_granule_count'])
        self.assertIsNone(result['cells'][0]['nisar_coherence'])

    def test_pair_phase_is_never_annualized_into_velocity(self):
        asset = {'observable': 'los_velocity', 'radar_band': 'L', 'units': 'mm/yr',
                 'time_start': '2026-01-01T00:00:00Z', 'time_end': '2026-01-13T00:00:00Z',
                 'observation_count': 2, 'derivation': 'pair_phase'}
        with self.assertRaisesRegex(ValueError, 'multi-acquisition timeseries'):
            _validate_asset(asset)
        asset.update(observation_count=5, derivation='documented_timeseries',
                     reference='https://example.org/site-calibration', sign_convention='toward satellite positive')
        self.assertEqual(_validate_asset(asset), ('los_velocity', 1.))

    def test_unknown_footprint_is_unknown_per_cell_coverage(self):
        item = granule(); item.pop('boxes')
        parsed = parse_granule(item)
        self.assertIsNone(parsed['orbit_direction'])
        self.assertEqual(parsed['spatial_coverage'], {'boxes': [], 'polygons': []})
        dateline = {'spatial_coverage': {'polygons': [[[179, -1], [-179, -1], [-179, 1], [179, 1], [179, -1]]]}}
        self.assertTrue(_covers(dateline, 0, -179.5))
        self.assertFalse(_covers(dateline, 0, 0))
        triangle = {'spatial_coverage': {'polygons': [[[0, 0], [1, 0], [0, 1], [0, 0]]], 'boxes': [[0, 0, 1, 1]]}}
        self.assertFalse(_covers(triangle, .9, .9))


@unittest.skipUnless(importlib.util.find_spec('rasterio'), 'Optional rasterio needed for local subset tests')
class NisarRaster(unittest.IsolatedAsyncioTestCase):
    async def test_authenticated_subset_reads_real_pixels_without_exposing_token(self):
        import rasterio
        from rasterio.io import MemoryFile
        from rasterio.transform import from_origin
        with MemoryFile() as memory:
            with memory.open(driver='GTiff', width=1, height=1, count=1, dtype='float32',
                crs='EPSG:4326', transform=from_origin(0, 1, 1, 1)) as dataset:
                dataset.write(np.array([[.75]], dtype='float32'), 1)
            content = memory.read()
        observed = []
        async def handler(request):
            observed.append(request)
            if request.url.host == 'cmr.earthdata.nasa.gov':
                self.assertNotIn('authorization', request.headers)
                return httpx.Response(200, json={'feed': {'entry': []}})
            self.assertEqual(request.headers['authorization'], 'Bearer private-token')
            return httpx.Response(200, content=content)
        manifest = {'assets': [{'url': 'https://nisar.asf.earthdatacloud.nasa.gov/coherence.tif',
            'observable': 'coherence', 'radar_band': 'L', 'units': '1'}]}
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            result = await fetch_nisar(client, .5, .5, local_manifest=manifest, earthdata_token='private-token')
        self.assertEqual(result['mode'], 'AUTHENTICATED_RASTER_MEASURED')
        self.assertEqual(result['cells'][0]['nisar_coherence'], .75)
        self.assertEqual(result['measured_assets'][0]['hash_verification'], 'ACTUAL_DOWNLOADED_SUBSET_BYTES')
        self.assertNotIn('private-token', json.dumps(result))
        self.assertEqual(len(observed), 2)

    async def test_oversize_remote_container_is_refused_without_inventing_values(self):
        async def handler(request):
            if request.url.host == 'cmr.earthdata.nasa.gov':
                return httpx.Response(200, json={'feed': {'entry': []}})
            return httpx.Response(200, content=b'not a subset', headers={'content-length': str(20*1024*1024)})
        manifest = {'assets': [{'url': 'https://nisar.asf.earthdatacloud.nasa.gov/large.tif',
            'observable': 'coherence', 'radar_band': 'L', 'units': '1'}]}
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            result = await fetch_nisar(client, 0, 0, local_manifest=manifest, earthdata_token='private-token')
        self.assertIsNone(result['cells'][0]['nisar_coherence'])
        self.assertEqual(result['mode'], 'METADATA_LIVE')
        self.assertTrue(any('exceeds 16 MiB' in error for error in result['errors']))

    async def test_local_raster_zero_mask_scale_and_calibration_passthrough(self):
        import rasterio
        from rasterio.transform import from_origin
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'coherence.tif'
            with rasterio.open(path, 'w', driver='GTiff', width=2, height=2, count=1,
                dtype='int16', crs='EPSG:4326', transform=from_origin(0, 2, 1, 1), nodata=-9999) as dataset:
                dataset.write(np.array([[0, 800], [-9999, 400]], dtype='int16'), 1)
                dataset.scales = (.001,)
            velocity = Path(directory)/'documented_velocity.tif'
            with rasterio.open(velocity, 'w', driver='GTiff', width=2, height=2, count=1,
                dtype='float64', crs='EPSG:4326', transform=from_origin(0, 2, 1, 1)) as dataset:
                dataset.write(np.full((2, 2), .002, dtype='float64'), 1)
            manifest = {'version': 'nisar_subset_manifest_v1', 'assets': [
                {'path': str(path), 'observable': 'coherence', 'radar_band': 'L', 'units': '1',
                 'granule_id': 'NISAR_LOCAL_FIXTURE', 'acquisition_utc': '2026-10-02T00:00:00Z'},
                {'path': str(velocity), 'observable': 'los_velocity', 'radar_band': 'L', 'units': 'm/yr',
                 'derivation': 'documented_timeseries', 'observation_count': 8,
                 'time_start': '2026-01-01T00:00:00Z', 'time_end': '2026-10-02T00:00:00Z',
                 'reference': 'https://example.org/fixture-timeseries-method', 'sign_convention': 'toward satellite positive'}],
                'strength_calibration_by_h3': {'a': {'source': 'documented calibration'}, 'outside': {'source': 'not requested'}}}
            async def handler(request):
                raise httpx.ConnectError('CMR temporarily unavailable')
            async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
                result = await fetch_nisar(client, 1, 1, cells=[cell('a', 1.5, .5), cell('b', 1.5, 1.5),
                    cell('c', .5, .5), cell('outside-raster', 20, 20)], local_manifest=manifest, earthdata_token='')
        self.assertEqual(result['mode'], 'LOCAL_RASTER_MEASURED')
        self.assertEqual(result['by_h3']['a']['nisar_coherence'], 0.)
        self.assertAlmostEqual(result['by_h3']['b']['nisar_coherence'], .8)
        self.assertIsNone(result['by_h3']['c']['nisar_coherence'])
        self.assertIsNone(result['by_h3']['outside-raster']['nisar_coherence'])
        self.assertAlmostEqual(result['by_h3']['b']['nisar_los_velocity_mm_yr'], 2.)
        self.assertEqual(set(result['strength_calibration_by_h3']), {'a'})
        self.assertEqual(result['by_h3']['a']['strength_relation_status'], 'NO_CALIBRATED_STRENGTH_RELATION')
        self.assertEqual(result['latest_acquisition_utc'], '2026-10-02T00:00:00Z')

    async def test_token_never_sent_to_untrusted_asset_host(self):
        requests = []
        async def handler(request):
            requests.append(request)
            return httpx.Response(200, json={'feed': {'entry': []}})
        manifest = {'assets': [{'url': 'https://example.org/data.tif', 'observable': 'coherence',
                               'radar_band': 'L', 'units': '1'}]}
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            result = await fetch_nisar(client, 0, 0, local_manifest=manifest, earthdata_token='secret')
        self.assertTrue(all(request.url.host == 'cmr.earthdata.nasa.gov' for request in requests))
        self.assertTrue(all('authorization' not in request.headers for request in requests))
        self.assertIsNone(result['observables']['nisar_coherence'])
        self.assertNotIn('secret', json.dumps(result))


@unittest.skipUnless(importlib.util.find_spec('h5py'), 'Optional h5py needed for native HDF5 tests')
class NisarHDF5(unittest.TestCase):
    def test_native_hdf5_discovery_retains_real_product_time(self):
        import h5py
        from backend.realtime.nisar_processor import _manifest
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'NISAR_NATIVE_FIXTURE.h5'
            grid = '/science/LSAR/GUNW/grids/frequencyA'
            with h5py.File(path, 'w') as handle:
                handle[grid+'/unwrappedInterferogram/HH/coherenceMagnitude'] = np.ones((2, 2))
                handle[grid+'/xCoordinates'] = [499995., 500005.]
                handle[grid+'/yCoordinates'] = [5., -5.]
                handle[grid+'/projection'] = 32631
                handle['/science/LSAR/identification/referenceZeroDopplerStartTime'] = b'2026-10-02T00:00:00Z'
                handle['/science/LSAR/identification/secondaryZeroDopplerStartTime'] = b'2026-09-20T00:00:00Z'
            with patch.dict('os.environ', {'TERRA_NISAR_DIR': directory, 'TERRA_NISAR_MANIFEST': ''}):
                manifest, root, digest = _manifest(None)
            self.assertEqual(len(manifest['assets']), 1)
            self.assertEqual(manifest['assets'][0]['acquisition_utc'], '2026-10-02T00:00:00Z')
            self.assertEqual(manifest['assets'][0]['x_dataset'], grid+'/xCoordinates')
            self.assertEqual(manifest['assets'][0]['observable'], 'coherence')

    def test_projected_coordinates_native_units_and_hv_hh_ratio(self):
        import h5py
        from pyproj import Transformer
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'sample.h5'
            with h5py.File(path, 'w') as handle:
                handle['x'] = [499995., 500005.]; handle['y'] = [5., -5.]
                handle['projection'] = 32631
                handle['coh'] = np.array([[0., .8], [float('nan'), .5]])
                handle['coh'].attrs['units'] = '1'
                handle['hv'] = np.array([[2., 8.], [float('nan'), 5.]])
                handle['hh'] = np.array([[4., 10.], [0., 10.]])
            inverse = Transformer.from_crs(32631, 4326, always_xy=True)
            lon, lat = inverse.transform(499995., 5.)
            asset = {'observable': 'coherence', 'radar_band': 'L', 'units': '1', 'dataset': 'coh',
                     'x_dataset': 'x', 'y_dataset': 'y', 'projection_dataset': 'projection'}
            observable, values, support = _sample_hdf5(path, asset, [cell(lat=lat, lon=lon)])
            self.assertEqual(values['a'], 0.)
            self.assertEqual(support['crs'], 'EPSG:32631')
            asset.update(observable='polarimetric_ratio', dataset='hv', denominator_dataset='hh')
            self.assertEqual(_sample_hdf5(path, asset, [cell(lat=lat, lon=lon)])[1]['a'], .5)
