"""Missing evidence must never become a measured zero or a safe verdict."""
import copy
import io
import json
import math
import unittest
from contextlib import redirect_stdout
from datetime import timedelta
from pathlib import Path
from unittest.mock import patch
from types import SimpleNamespace
import h3
import numpy as np
from backend.api.ground_assessment import assess_ground,REQUIRED
from backend.api.scan_service import GlobalScanService
from backend.api.serialization import sanitize_json
from backend.api.source_assessment import assess
from backend.ml.rainfall_features import utc
from backend.ml.rainfall_baseline import RainfallBaseline
from backend.physics.reference_column import infinite_slope_fs
from backend.realtime.global_ingestor import GlobalIngestor,GlobalScan
from scripts.snapshot_replay import hydrate
from scripts.cli_ui import fmt,score,show_summary
from test_rainfall_sources import raw_scan,Scorer


def ground_columns(raw,values=None):
    # Deliberate test parameters, never a source fallback or shipped field data.
    values=values or dict(cohesion_kpa=8.,friction_angle_deg=32.,unit_weight_kn_m3=19.,depth_m=2.,pressure_head_m=.4)
    at=utc(raw.generated_at_utc)
    return {next(iter(raw.terrain.cells)):{key:{'value':value,'source':'Synthetic unit-test input',
        'method':'measurement','observed_at_utc':(at-timedelta(hours=1)).isoformat(),
        'valid_until_utc':(at+timedelta(hours=1)).isoformat()} for key,value in values.items()}}


class GroundEvidence(unittest.TestCase):
    def test_default_hydrology_labels_static_prior(self):
        weather=SimpleNamespace(soil_moisture_0_100cm_m3m3=.3)
        h=GlobalIngestor.__new__(GlobalIngestor)._combine_hydrology(weather,None,None)
        self.assertGreater(h.soil_porosity_estimate,0.)
        self.assertIn('estimated',h.soil_property_quality)
        self.assertEqual(h.open_meteo_soil_moisture_m3m3,.3)

    def test_partial_soil_keeps_raw_field_without_filling_others(self):
        h=GlobalIngestor.__new__(GlobalIngestor)._combine_hydrology(None,None,{'bdod_kg_m3':1300.})
        self.assertEqual(h.soil_properties['bdod_kg_m3'],1300.)
        self.assertGreater(h.soil_properties['clay_g_kg'],0.)
        self.assertEqual(h.soil_properties['property_sources']['bdod_kg_m3'],'ISRIC SoilGrids')
        self.assertIn('prior',h.soil_properties['property_sources']['clay_g_kg'])

    def test_complete_ground_column_matches_reference_equation_without_fs_cap(self):
        raw=raw_scan();columns=ground_columns(raw,dict(cohesion_kpa=100.,friction_angle_deg=35.,
            unit_weight_kn_m3=18.,depth_m=1.,pressure_head_m=0.))
        result=assess_ground(raw,columns);cell=next(c for c in result['cells'] if c['h3_index'] in columns);v=cell['input_values']
        expected=float(infinite_slope_fs(v['cohesion_kpa'],v['friction_angle_deg'],v['unit_weight_kn_m3'],
            v['depth_m'],cell['slope_deg'],v['initial_pressure_head_m']))
        self.assertAlmostEqual(cell['fs_dynamic'],expected);self.assertGreater(expected,5.)
        self.assertGreater(cell['fosm_variance'],0.);self.assertEqual(v['initial_pressure_head_m'],0.)

    def test_each_required_input_is_required_including_explicit_initial_head(self):
        raw=raw_scan()
        for missing in REQUIRED:
            with self.subTest(missing=missing):
                columns=ground_columns(raw);del next(iter(columns.values()))[missing]
                result=assess_ground(raw,columns)
                self.assertEqual(result['status'],'REFERENCE_COLUMN');self.assertTrue(result['cells'])
                cell=next(c for c in result['cells'] if c['h3_index'] in columns)
                self.assertNotIn(missing,cell['ground_provenance'])

    def test_invalid_provenance_or_time_never_falls_back(self):
        raw=raw_scan()
        for change in ({'source':''},{'method':'assumed'},{'value':float('nan')},{'value':True},
                       {'observed_at_utc':'2999-01-01'},{'valid_until_utc':'1900-01-01'}):
            with self.subTest(change=change):
                columns=ground_columns(raw);next(iter(columns.values()))['depth_m'].update(change)
                result=assess_ground(raw,columns)
                self.assertNotIn(next(iter(columns)),[c['h3_index'] for c in result['cells']])
                self.assertEqual(result['omitted_cells'][0]['h3_index'],next(iter(columns)))

    def test_ground_data_does_not_apply_to_another_cell(self):
        raw=raw_scan();columns=ground_columns(raw)
        result=assess_ground(raw,columns)
        self.assertEqual(len(result['cells']),len(raw.terrain.cells))
        self.assertEqual(sum(bool(c['ground_provenance']) for c in result['cells']),1)
        cell=next(c for c in result['cells'] if c['h3_index'] in columns)
        self.assertEqual((cell['latitude'],cell['longitude']),h3.cell_to_latlng(cell['h3_index']))

    def test_zero_strength_can_produce_real_zero_fs(self):
        raw=raw_scan();columns=ground_columns(raw,dict(cohesion_kpa=0.,friction_angle_deg=0.,
            unit_weight_kn_m3=19.,depth_m=2.,pressure_head_m=0.))
        cell=next(c for c in assess_ground(raw,columns)['cells'] if c['h3_index'] in columns)
        self.assertEqual(cell['fs_dynamic'],0.)

    def test_rain_scenario_requires_hydraulic_inputs_and_keeps_initial_head(self):
        raw=raw_scan();columns=ground_columns(raw)
        self.assertEqual(assess_ground(raw,columns,rain_delta_mm=50.)['status'],'REFERENCE_COLUMN')
        record=next(iter(columns.values()))
        for name,value in [('conductivity_m_s',1.e-5),('diffusivity_m2_s',1.e-4)]:
            record[name]={**record['depth_m'],'value':value}
        before=next(c for c in assess_ground(raw,columns)['cells'] if c['h3_index'] in columns)
        after=next(c for c in assess_ground(raw,columns,rain_delta_mm=50.)['cells'] if c['h3_index'] in columns)
        self.assertGreater(after['pressure_head_m'],before['pressure_head_m'])
        self.assertLess(after['fs_dynamic'],before['fs_dynamic'])
        self.assertEqual(record['pressure_head_m']['value'],.4)

    def test_unsupported_scenario_is_not_silently_ignored(self):
        raw=raw_scan();result=assess_ground(raw,ground_columns(raw),unsupported_scenario=True)
        self.assertEqual(result['status'],'REFERENCE_COLUMN');self.assertTrue(result['cells'])
        self.assertIn('additional_mechanism_status',result)

    def test_nonfinite_json_is_null_and_real_zero_remains_zero(self):
        result=sanitize_json({'value':float('nan'),'nested':[float('inf'),0.]})
        self.assertIsNone(result['value']);self.assertIsNone(result['nested'][0]);self.assertEqual(result['nested'][1],0.)
        self.assertIsNone(result['serialization_quality']['replacement_value'])
        json.dumps(result,allow_nan=False)

    def test_small_nonzero_values_do_not_display_as_zero(self):
        self.assertEqual(fmt(8.975590673867619e-14,3),'8.98e-14')
        self.assertNotIn('0.000',score(8.975590673867619e-14))
        self.assertIn('not evidence of safety',score(0.))

    def test_missing_source_moisture_abstains_without_median_substitution(self):
        raw=raw_scan();raw.weather.rainfall_source_window['soil_moisture_0_7cm_mean_m3m3']=None
        with patch.object(Scorer,'predict_proba',side_effect=AssertionError('Do not score missing source inputs')):
            result=assess(raw,Scorer())
        self.assertEqual(result['status'],'UNAVAILABLE')
        self.assertIn('imputed',result['omitted_cells'][0]['reason'])

    def test_logistic_explanation_reconstructs_served_probabilities(self):
        x=np.tile([10.,20.,.4,30.,10.,.001],(24,1));x[:,0]=np.arange(24);x[:,4]=np.arange(24)+1.
        y=np.tile([0,1],12);groups=np.repeat(np.arange(12),2)
        model=RainfallBaseline().fit(x,y,groups);expected=model.predict_proba(x)
        explanation=model.explain(x)
        for probability,record in zip(expected,explanation):
            reconstructed=[]
            for fold in record['folds']:
                logit=fold['intercept']+sum(fold['terms'].values())
                p=1./(1.+math.exp(-logit));self.assertAlmostEqual(p,fold['probability'],places=12)
                reconstructed.append(p)
            self.assertAlmostEqual(float(np.mean(reconstructed)),probability,places=12)


class StrictLiveInterface(unittest.IsolatedAsyncioTestCase):
    async def test_live_scan_never_calls_archived_assumption_builder(self):
        raw=raw_scan();service=GlobalScanService();service.rainfall_model=Scorer()
        with patch('backend.api.scan_service.evaluate_live_physics',side_effect=AssertionError('No guessed strength, head, or motion')):
            result=await service.scan(_snapshot=raw)
        self.assertEqual(result['physical_assessment']['status'],'REFERENCE_COLUMN')
        self.assertEqual(len(result['physical_assessment']['cells']),len(raw.terrain.cells))
        self.assertIsNone(result['summary']['causal_breakdown_pct'])
        self.assertIn('fs_interval',result['summary'])
        self.assertFalse(result['operational_warning_ready'])
        cell=next(iter(raw.terrain.cells));inspection=service.inspect_hexagon(cell)
        self.assertIsNotNone(inspection['fs_dynamic'])
        self.assertTrue(service.trace_runout(cell)['path'])

    async def test_cell_comparison_joins_same_cell_with_sourced_ground_inputs(self):
        raw=raw_scan();service=GlobalScanService();service.rainfall_model=Scorer()
        columns=ground_columns(raw);result=await service.scan(_snapshot=raw,ground_columns=columns)
        physical=result['physical_assessment']['cells'][0]
        joined=next(c for c in result['cell_comparisons'] if c['h3_index']==physical['h3_index'])
        self.assertEqual(joined['fs_dynamic'],physical['fs_dynamic'])
        self.assertIsNone(result['summary']['max_p_failure']);self.assertEqual(result['summary']['wmo_alert_level'],'UNAVAILABLE')

    async def test_monaco_ids_are_real_cells_and_cli_locates_them(self):
        payload=json.loads((Path(__file__).parent/'fixtures/monaco_provider_excerpt.json').read_text())
        raw=hydrate(GlobalScan,payload['snapshot']);service=GlobalScanService();service.rainfall_model=Scorer()
        result=await service.scan(_snapshot=raw);out=io.StringIO()
        with redirect_stdout(out):show_summary(result,0.)
        for cell in result['cell_comparisons']:
            self.assertTrue(h3.is_valid_cell(cell['h3_index']))
            self.assertEqual((cell['latitude'],cell['longitude']),h3.cell_to_latlng(cell['h3_index']))
            self.assertIsNotNone(cell['fs_dynamic'])
        self.assertIn('Cell centre:',out.getvalue());self.assertIn('FS',out.getvalue())
        self.assertNotIn('Modeled FS: 3.781',out.getvalue())

    async def test_cap_export_cannot_promote_client_score_to_warning(self):
        from backend.api.main import export_cap_xml,CapXmlRequest
        from fastapi import HTTPException
        with self.assertRaises(HTTPException) as caught:
            await export_cap_xml(CapXmlRequest(scan={'operational_warning_ready':True,'summary':{'wmo_alert_level':'RED','max_p_failure':1.}}))
        self.assertEqual(caught.exception.status_code,409)

    async def test_assistant_cannot_promote_a_classifier_score_to_certain_failure(self):
        from backend.llm.qwen_agent import TerraShieldQwenAgent
        service=GlobalScanService();service.rainfall_model=Scorer()
        response=await service.scan(_snapshot=raw_scan())
        text,valid,status=TerraShieldQwenAgent._guard_briefing('Failure is certain. Evacuate immediately.',response,'LOCAL_EOC')
        self.assertTrue(valid);self.assertEqual(status,'VERIFIED_TEMPLATE')
        self.assertNotIn('Evacuate immediately',text);self.assertIn('conditional',text.lower())


if __name__=='__main__':unittest.main()
