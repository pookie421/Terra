import json
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from backend.ml.global_xgb import connected_component_groups, spatial_group_splits, GlobalMonotonicXGB
from backend.ml.pinls import pinls_filter, quiescent_control
from backend.ml.validation_suite import ranking_metrics
from backend.api.scan_service import GlobalScanService
from backend.realtime.global_ingestor import GlobalScan
from scripts.snapshot_replay import hydrate


class EventIsolation(unittest.TestCase):
    def test_transitive_event_region_components_and_training_only_synthetics(self):
        regions=['a','b','b','c','d','e','f','synthetic:regime:x']
        events=['event1','event1','event2','event2','event3','event4','event5',None]
        groups=connected_component_groups(regions,events)
        self.assertEqual(len(set(groups[:4])),1)
        labels=np.ones(len(groups),int)
        for train,test in spatial_group_splits(labels,regions,3,event_groups=events):
            self.assertFalse(set(np.array(regions)[train]) & set(np.array(regions)[test]))
            self.assertFalse(set(np.array(events,dtype=object)[train]) & set(np.array(events,dtype=object)[test]))
            self.assertIn(7,train)
            self.assertNotIn(7,test)

    def test_physics_cannot_select_background_rows(self):
        rows=[dict(event_id=str(i),label=0,fs_static_prior=fs,fs_dynamic=fs,p_failure_physical=p,
            rain_effective_storm_mm=20,pga_attenuation_proxy_g=0,snow_depth_72h_max_m=0,usgs_source_status='live_no_matching_earthquake')
              for i,(fs,p) in enumerate(((.2,.99),(5.,0.)))]
        self.assertEqual(pinls_filter(rows)[0],rows)
        self.assertTrue(all(quiescent_control(r) for r in rows))
        self.assertIsNone(GlobalMonotonicXGB().field_prevalence)

    def test_attainable_fpr_points_and_ap_are_explicit(self):
        y=np.array([1,0,1,0]);p=np.array([.9,.8,.7,.6])
        result=ranking_metrics(y,p)
        self.assertEqual(result['empirical_fpr_step'],.5)
        self.assertEqual(result['operating_points']['0.01']['achieved_fpr'],0.)
        self.assertEqual(result['operating_points']['0.01']['tpr'],.5)
        self.assertAlmostEqual(result['average_precision'],5/6)

    def test_large_int8_label_counts_and_mixed_component_folds(self):
        y=np.array([1]*155+[0]*103,dtype=np.int8)
        self.assertEqual(ranking_metrics(y,np.full(len(y),.5))['positive_rows'],155)
        groups=np.repeat(np.array(['mixed_a','positive_a','mixed_b','negative_a','mixed_c']),[4,8,4,7,4])
        labels=np.array([0,0,1,1]+[1]*8+[0,0,1,1]+[0]*7+[0,0,1,1])
        for train,test in spatial_group_splits(labels,groups,3):
            self.assertEqual(set(labels[test]),{0,1})
            self.assertEqual(set(labels[train]),{0,1})


class FrozenScan(unittest.IsolatedAsyncioTestCase):
    async def test_response_keys_and_cell_coherent_hotspot_offline(self):
        directory=Path(__file__).parent/'fixtures'
        snapshot=hydrate(GlobalScan,json.loads((directory/'frozen_scan.json').read_text())['snapshot'])
        contract=json.loads((directory/'scan_response_keys_v13.json').read_text())
        with patch('socket.socket.connect',side_effect=AssertionError('No live network in frozen replay')), patch('backend.api.scan_service._load_model',return_value=None):
            service=GlobalScanService()
            result=await service.scan(lat=snapshot.location.latitude,lon=snapshot.location.longitude,_snapshot=snapshot,_reference_assumptions=True)
        self.assertEqual(result['assessment_mode'],'ARCHIVED_REFERENCE_ASSUMPTIONS_NOT_LIVE_ASSESSMENT')
        json.dumps(result,allow_nan=False)
        self.assertTrue(set(contract['top']) <= set(result))
        self.assertTrue(set(contract['summary']) <= set(result['summary']))
        cells={f['properties']['h3_index']:f['properties'] for f in result['h3_geojson']['features']}
        summary=result['summary'];hotspot=cells[summary['hotspot_h3_index']]
        self.assertTrue(set(contract['cell']) <= set(hotspot))
        self.assertEqual(summary['hotspot_fs_dynamic'],hotspot['fs_dynamic'])
        self.assertEqual(summary['hotspot_probability'],hotspot['p_failure'])
        self.assertEqual(summary['hotspot_newmark_displacement_cm'],hotspot['newmark_displacement_reference_cm'])
        self.assertIsNone(hotspot['p_combined'])

    async def test_incompatible_hydrology_is_unavailable(self):
        directory=Path(__file__).parent/'fixtures'
        snapshot=hydrate(GlobalScan,json.loads((directory/'frozen_scan.json').read_text())['snapshot'])
        snapshot.hydrology.soil_saturation_pct=None
        snapshot.hydrology.moisture_compatibility='incompatible_moisture_and_porosity'
        with patch('backend.api.scan_service._load_model',return_value=None):
            result=await GlobalScanService().scan(_snapshot=snapshot,_reference_assumptions=True)
        self.assertIsNone(result['summary']['max_p_failure'])
        self.assertEqual(result['summary']['wmo_alert_level'],'UNAVAILABLE')
        self.assertIn('incompatible',result['predictive_forecast_72h']['reason'])
