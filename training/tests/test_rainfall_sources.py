"""Source timing, independent availability, calibration isolation and CLI contracts."""
import copy
import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout, redirect_stderr
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import AsyncMock, patch
import numpy as np

from backend.ml.rainfall_features import source_window, feature_vector, VERSION
from backend.ml.rainfall_baseline import RainfallBaseline, grouped_folds
from backend.api.source_assessment import assess
from backend.api.scan_service import GlobalScanService
from backend.realtime.global_ingestor import GlobalScan, GlobalIngestor
from scripts.snapshot_replay import hydrate
from scripts import terra_cli, cli_ui

ROOT=Path(__file__).resolve().parents[1]
CUTOFF=datetime(2024,7,30,23,tzinfo=timezone.utc)


def hourly():
    # Include an extra old and future sample to catch off-by-one selection.
    times=[CUTOFF+timedelta(hours=i) for i in range(-168,2)]
    return {'time':[t.isoformat() for t in times],
            'rain':[999.]+[1.]*144+[2.]*24+[999.],
            'showers':[100.]+[.5]*168+[100.],
            'soil_moisture_0_to_7cm':[.6]*170}


def raw_scan():
    payload=json.loads((ROOT/'tests/fixtures/frozen_scan.json').read_text())
    raw=hydrate(GlobalScan,payload['snapshot'])
    raw.weather.rainfall_source_window=source_window(hourly(),CUTOFF,product='archive')
    raw.hydrology=replace(raw.hydrology,soil_saturation_pct=None,
        open_meteo_soil_moisture_m3m3=.6,soil_porosity_estimate=.4)
    return raw


class Scorer:
    kind='test_scorer'
    def predict_proba(self,x):return np.full(len(x),.42)
    def out_of_range(self,x):return [[] for _ in x]


class SourceWindows(unittest.TestCase):
    def test_hour_endpoints_and_nonoverlapping_totals(self):
        result=source_window(hourly(),CUTOFF+timedelta(minutes=32),product='archive')
        self.assertEqual(result['cutoff_utc'],CUTOFF.isoformat())
        self.assertEqual(result['rain_24h_mm'],48.)
        self.assertEqual(result['rain_prior_6d_mm'],144.)
        self.assertAlmostEqual(result['soil_moisture_0_7cm_mean_m3m3'],.6)

    def test_live_showers_match_archive_liquid_rain(self):
        live=hourly();archive=copy.deepcopy(live)
        archive['rain']=[a+b for a,b in zip(live['rain'],live['showers'])]
        del archive['showers']
        a=source_window(live,CUTOFF,product='forecast')
        b=source_window(archive,CUTOFF,product='archive')
        self.assertEqual(a['rain_24h_mm'],60.)
        for key in ('rain_24h_mm','rain_prior_6d_mm','soil_moisture_0_7cm_mean_m3m3'):
            self.assertEqual(a[key],b[key])

    def test_live_missing_showers_cannot_become_archive_rain(self):
        data=hourly();del data['showers']
        self.assertEqual(source_window(data,CUTOFF,product='forecast')['status'],'UNAVAILABLE')
        self.assertEqual(source_window(data,CUTOFF,product='archive')['status'],'AVAILABLE')

    def test_gaps_duplicates_and_missing_rain_are_unavailable(self):
        for mode in ('gap','duplicate','null','negative'):
            with self.subTest(mode=mode):
                data=hourly()
                if mode=='gap':data['time'][15]=(CUTOFF+timedelta(hours=5)).isoformat()
                elif mode=='duplicate':data['time'][15]=data['time'][16]
                else:data['rain'][15]=None if mode=='null' else -1
                self.assertEqual(source_window(data,CUTOFF,product='archive')['status'],'UNAVAILABLE')

    def test_unknown_product_is_rejected(self):
        with self.assertRaises(ValueError):source_window(hourly(),CUTOFF,product='unknown')

    def test_missing_soil_keeps_rain_available_without_inventing_water(self):
        data=hourly();data['soil_moisture_0_to_7cm'][160]=None
        window=source_window(data,CUTOFF,product='archive')
        self.assertEqual(window['status'],'AVAILABLE')
        vector=feature_vector(window,30.,10.,-.1)
        self.assertTrue(np.isnan(vector[2]))
        for bad in (None,float('nan')):
            with self.assertRaises(ValueError):feature_vector(window,bad,10.,-.1)

    def test_raw_water_above_soil_prior_remains_a_provider_feature(self):
        vector=feature_vector(source_window(hourly(),CUTOFF,product='archive'),30.,10.,-.1)
        self.assertAlmostEqual(vector[2],.6)  # No porosity division or saturation cap.

    def test_live_parser_uses_identical_feature_extractor(self):
        class Clock(datetime):
            @classmethod
            def now(cls,tz=None):return CUTOFF
        data=hourly()
        with patch('backend.realtime.global_ingestor.datetime',Clock):
            weather=GlobalIngestor.__new__(GlobalIngestor)._parse_weather({'hourly':data})
        self.assertEqual(weather.rainfall_source_window,source_window(data,CUTOFF,product='forecast'))


class CalibrationIsolation(unittest.TestCase):
    def test_imputation_is_fitted_only_inside_each_calibration_training_fold(self):
        groups=np.repeat(np.arange(12),2);y=np.tile([0,1],12)
        x=np.column_stack([np.arange(24),np.arange(24)*2,np.arange(24)/30,
                           np.full(24,30.),np.full(24,10.),np.zeros(24)])
        x[::3,2]=np.nan
        model=RainfallBaseline().fit(x,y,groups)
        for (train,test),fitted in zip(grouped_folds(y,groups,3),model.model.calibrated_classifiers_):
            self.assertFalse(set(groups[train])&set(groups[test]))
            self.assertAlmostEqual(fitted.estimator.named_steps['missing'].statistics_[2],np.nanmedian(x[train,2]))
        probe=x[:1].copy();probe[0,2]=np.nan
        self.assertTrue(np.isfinite(model.predict_proba(probe)).all())
        probe[0,3]=np.nan
        with self.assertRaises(ValueError):model.predict_proba(probe)

    def test_all_missing_optional_moisture_keeps_feature_shape(self):
        groups=np.repeat(np.arange(6),2);y=np.tile([0,1],6)
        x=np.tile([10.,20.,np.nan,30.,10.,0.],(12,1));x[:,0]=np.arange(12)
        model=RainfallBaseline().fit(x,y,groups)
        self.assertTrue(np.isfinite(model.predict_proba(x)).all())


class IndependentAssessment(unittest.IsolatedAsyncioTestCase):
    async def test_api_retains_terrain_and_statistical_score_when_physics_is_missing(self):
        raw=raw_scan();service=GlobalScanService();service.rainfall_model=Scorer()
        response=await service.scan(_snapshot=raw)
        self.assertTrue(response['h3_geojson']['features'])
        self.assertEqual(len(response['terrain_geojson']['features']),len(raw.terrain.cells))
        self.assertEqual(response['physical_assessment']['status'],'REFERENCE_COLUMN')
        self.assertIsNone(response['evidence']['absolute_saturation_fraction'])
        self.assertTrue(response['evidence']['moisture_exceeds_porosity'])
        json.dumps(response,allow_nan=False)

    async def test_scenarios_do_not_mutate_source_or_accumulate(self):
        raw=raw_scan();original=copy.deepcopy(raw.weather.rainfall_source_window)
        first=assess(raw,Scorer(),50.)
        second=assess(raw,Scorer(),20.)
        self.assertEqual(first['window']['rain_24h_mm'],98.)
        self.assertEqual(second['window']['rain_24h_mm'],68.)
        self.assertEqual(raw.weather.rainfall_source_window,original)
        self.assertEqual(first['window']['soil_moisture_0_7cm_mean_m3m3'],original['soil_moisture_0_7cm_mean_m3m3'])

    async def test_other_mechanisms_do_not_receive_rainfall_scores(self):
        result=assess(raw_scan(),Scorer(),nonrain_scenario=True)
        self.assertEqual(result['status'],'OUT_OF_SCOPE');self.assertFalse(result['cells'])

    async def test_old_snapshot_never_substitutes_legacy_peak_weather_features(self):
        raw=raw_scan();raw.weather.rainfall_source_window={}
        self.assertEqual(assess(raw,Scorer())['status'],'UNAVAILABLE')
        raw.weather.rainfall_source_window={'status':'AVAILABLE','version':'old_version'}
        self.assertEqual(assess(raw,Scorer())['status'],'UNAVAILABLE')


class CliContracts(unittest.IsolatedAsyncioTestCase):
    async def test_default_interactive_start_acquires_live_location(self):
        raw=raw_scan();service=GlobalScanService();service.rainfall_model=Scorer()
        with patch.object(terra_cli,'GlobalScanService',return_value=service), \
             patch.object(terra_cli,'acquire',new_callable=AsyncMock,return_value=raw) as acquire, \
             patch.object(terra_cli,'prompt_location',return_value=(11.548,76.134)), \
             patch('builtins.input',return_value='q'),redirect_stdout(io.StringIO()):
            self.assertEqual(await terra_cli.main(terra_cli.parse_args(['--plain'])),0)
            acquire.assert_awaited_once()

    async def test_json_stdout_and_export_are_the_same_complete_response(self):
        raw=raw_scan();service=GlobalScanService();service.rainfall_model=Scorer()
        with tempfile.TemporaryDirectory() as tmp:
            output=Path(tmp)/'scan.json';stdout=io.StringIO()
            with patch.object(terra_cli,'GlobalScanService',return_value=service), \
                 patch.object(terra_cli,'acquire',new_callable=AsyncMock,return_value=raw), \
                 redirect_stdout(stdout),redirect_stderr(io.StringIO()):
                code=await terra_cli.main(terra_cli.parse_args(['Wayanad','--json','--output',str(output)]))
            self.assertEqual(code,0)
            self.assertEqual(json.loads(stdout.getvalue()),json.loads(output.read_text()))
            self.assertNotIn('replay',json.loads(stdout.getvalue()))

    async def test_explicit_replay_never_fetches_live_data(self):
        from dataclasses import asdict
        service=GlobalScanService();service.rainfall_model=Scorer()
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'snapshot.json';path.write_text(json.dumps({'snapshot':asdict(raw_scan())}))
            stdout=io.StringIO()
            with patch.object(terra_cli,'GlobalScanService',return_value=service), \
                 patch.object(terra_cli,'acquire',new_callable=AsyncMock) as acquire,redirect_stdout(stdout):
                code=await terra_cli.main(terra_cli.parse_args(['--snapshot',str(path),'--json']))
            self.assertEqual(code,0);acquire.assert_not_awaited()
            self.assertIn('replay',json.loads(stdout.getvalue()))

    async def test_cli_rejects_ambiguous_or_nonfinite_scenarios(self):
        for argv in (['--json'],['Wayanad','--rain-delta-mm','nan'],['Wayanad','--rain-delta-mm','-1'],
                     ['--snapshot','x','Wayanad'],['Wayanad','--interactive','--json']):
            with self.subTest(argv=argv),redirect_stderr(io.StringIO()),self.assertRaises(SystemExit):
                terra_cli.parse_args(argv)

    async def test_comparison_uses_original_cell_even_when_hotspot_changes(self):
        before={'statistical_assessment':{'hotspot':{'h3_index':'a'},'cells':[{'h3_index':'a','score':.2}]}}
        after={'statistical_assessment':{'hotspot':{'h3_index':'b'},'cells':[{'h3_index':'a','score':.4},{'h3_index':'b','score':.9}]}}
        stdout=io.StringIO()
        with redirect_stdout(stdout):cli_ui.show_comparison(before,after)
        self.assertIn('0.200',stdout.getvalue());self.assertIn('0.400',stdout.getvalue())
        self.assertNotIn('0.900',stdout.getvalue())

    async def test_plain_summary_never_displays_unavailable_motion_as_zero(self):
        service=GlobalScanService();service.rainfall_model=Scorer()
        response=await service.scan(_snapshot=raw_scan());stdout=io.StringIO()
        with redirect_stdout(stdout):cli_ui.show_summary(response,0.)
        self.assertNotIn('\x1b',stdout.getvalue())
        self.assertNotIn('0.00 m/s',stdout.getvalue())
        self.assertIn('Input conflict:',stdout.getvalue())


if __name__=='__main__':unittest.main()
