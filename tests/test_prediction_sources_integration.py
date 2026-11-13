"""Provider/multimodel integration on explicit fixtures, no network or new labels."""
import copy
from datetime import timedelta
import socket
import unittest
from unittest.mock import patch
from backend.api.scan_service import GlobalScanService
from backend.ml.rainfall_features import utc
from test_autonomous_api import forcing_fixture


class PredictionIntegration(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.service = GlobalScanService()
        self.raw = forcing_fixture()
        self.guard = patch.object(socket.socket, 'connect', side_effect=AssertionError('Offline provider integration'))
        self.guard.start(); self.addCleanup(self.guard.stop)

    async def test_nasa_zero_is_preserved_and_daily_product_does_not_become_72h_forecast(self):
        ids = list(self.raw.terrain.cells)
        self.raw.nasa_lhasa = {'mode': 'LIVE_NASA_IMAGESERVER', 'product_version': 'NOT_DECLARED_BY_SERVICE',
            'cells': [{'h3_index': key, 'probability': 0., 'tomorrow_probability': .25,
                'reference_proxy_score': None, 'susceptibility_class': 4,
                'provenance': 'LIVE_NASA_IMAGESERVER'} for key in ids]}
        self.raw.nisar = {'mode': 'METADATA_LIVE', 'granule_count': 1, 'granules': [{'granule_id': 'fixture'}], 'cells': []}
        before = copy.deepcopy(self.raw.nasa_lhasa)
        now = await self.service.scan(_snapshot=self.raw)
        self.assertEqual(len(now['prediction_models']), 5)
        for feature in now['h3_geojson']['features']:
            row = feature['properties']
            self.assertEqual(row['prediction_scores']['nasa_lhasa_2'], 0.)
            self.assertIsNone(row['nisar_coherence'])
            self.assertFalse(row['radar_feature_context']['trained_radar_weights_available'])
            self.assertGreater(now['spatial_support']['h3_mean_cell_area_m2'], 100000)
        tomorrow = await self.service.automatic_forecast_state(now['scan_id'], 't24')
        self.assertEqual(tomorrow['nasa_lhasa']['cells'][0]['official_score'], .25)
        future = await self.service.automatic_forecast_state(now['scan_id'], 't72')
        for feature in future['h3_geojson']['features']:
            self.assertIsNone(feature['properties']['prediction_scores']['nasa_lhasa_2'])
        self.assertTrue(all(r['probability'] is None and r['reference_proxy_score'] is None for r in future['nasa_lhasa']['cells']))
        self.assertEqual(self.raw.nasa_lhasa, before)
        inspected = self.service.inspect_hexagon(ids[0], now['scan_id'], now['scenario_id'])
        self.assertEqual(inspected['prediction_scores']['nasa_lhasa_2'], 0.)

    async def test_metadata_alone_leaves_full_physics_and_scores_identical(self):
        baseline = await self.service.scan(_snapshot=copy.deepcopy(self.raw))
        self.raw.nisar = {'mode': 'METADATA_LIVE', 'granule_count': 3, 'cells': [],
                          'latest_acquisition_utc': self.raw.generated_at_utc}
        updated = await self.service.scan(_snapshot=self.raw)
        old = {r['h3_index']: r for r in baseline['physical_assessment']['cells']}
        for row in updated['physical_assessment']['cells']:
            self.assertEqual(row['fs_interval'], old[row['h3_index']]['fs_interval'])
            self.assertEqual(row['prediction_scores']['terra_physics_ml'], old[row['h3_index']]['prediction_scores']['terra_physics_ml'])
            self.assertFalse(row['radar_constraint']['applied'])
            self.assertIsNotNone(row['fosm_variance'])

    async def test_only_anchored_site_calibration_changes_strength_and_recomputes_ml(self):
        key = '893969a401bffff'
        baseline = await self.service.scan(_snapshot=copy.deepcopy(self.raw))
        original = next(r for r in baseline['physical_assessment']['cells'] if r['h3_index'] == key)
        at = utc(self.raw.generated_at_utc)
        self.raw.nisar = {'mode': 'LOCAL_RASTER', 'granule_count': 1,
            'granules': [{'granule_id': 'fixture-granule'}],
            'cells': [{'h3_index': key, 'nisar_coherence': .3}],
            'strength_calibration_by_h3': {key: {'granule_id': 'fixture-granule', 'relation_version': 'deterministic_test_fixture_v1',
                'cohesion_kpa': {'value': .1, 'source': 'Explicit synthetic calibration fixture, not a scientific coefficient',
                    'method': 'site_calibration', 'observed_at_utc': (at-timedelta(days=1)).isoformat(),
                    'valid_until_utc': (at+timedelta(days=1)).isoformat()}}}}
        changed = await self.service.scan(_snapshot=copy.deepcopy(self.raw))
        row = next(r for r in changed['physical_assessment']['cells'] if r['h3_index'] == key)
        self.assertEqual(row['input_values']['cohesion_kpa'], .1)
        self.assertTrue(row['radar_constraint']['applied'])
        self.assertLess(row['fs_mid'], original['fs_mid'])
        self.assertNotEqual(row['prediction_scores']['terra_physics_ml'], original['prediction_scores']['terra_physics_ml'])
        self.raw.nisar['cells'] = []
        rejected = await self.service.scan(_snapshot=self.raw)
        unchanged = next(r for r in rejected['physical_assessment']['cells'] if r['h3_index'] == key)
        self.assertEqual(unchanged['fs_interval'], original['fs_interval'])
        self.assertEqual(unchanged['radar_constraint']['status'], 'RADAR_CALIBRATION_REJECTED_REFERENCE_RETAINED')


if __name__ == '__main__':unittest.main()
