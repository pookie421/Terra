"""Compact measured-radar evidence must distinguish a masked cell from the scan."""
import json
import unittest
from types import SimpleNamespace

from backend.llm.state_copilot import chat_active, context, provider_context, template, verify_and_render

CLIFF='893969a401bffff';VALLEY='893969a6a5bffff';HOTSPOT='893969a4607ffff'


def scan_fixture():
    cells=[{'h3_index':CLIFF,'slope_deg':41.,'score':.2,'fs_mid':1.4,'fs_interval':[1.1,1.4,1.7],'fosm_variance':.03},
           {'h3_index':VALLEY,'slope_deg':4.,'score':.01,'fs_mid':5.,'fs_interval':[4.,5.,6.],'fosm_variance':.02},
           {'h3_index':HOTSPOT,'slope_deg':25.,'score':.7,'fs_mid':.9,'fs_interval':[.6,.9,1.2],'fosm_variance':.04,'nisar_coherence':.9}]
    radar=[{'h3_index':CLIFF,'nisar_coherence':0.},
           {'h3_index':VALLEY,'nisar_coherence':.323,'nisar_polarimetric_moisture_index':0.},
           {'h3_index':HOTSPOT,'nisar_coherence':None}]
    return {'scan_id':'scan-a','scenario_id':'now-a','scenario':{},'summary':{},'source_status':{},
            'location':{'query':'fixture','latitude':11.,'longitude':76.},
            'physical_assessment':{'cells':cells,'hotspot':cells[-1]},
            'physics_informed_assessment':{'cells':cells,'hotspot':cells[-1]},
            'nisar':{'mode':'AUTHENTICATED_RASTER_MEASURED','measurement_access':{'status':'MEASURED_AVAILABLE'},
                     'cells':radar,'by_h3':{row['h3_index']:row for row in radar},
                     'granules':[{'raw_archive':'DO_NOT_PUT_FULL_RASTER_CATALOG_IN_CONTEXT'*1000}]}}


class RadarEvidence(unittest.TestCase):
    def test_scan_coverage_is_compact_selected_scoped_and_accepts_zero(self):
        evidence=context(scan_fixture());radar=evidence['nisar'];coverage=radar['sample_coverage']
        self.assertEqual((coverage['total_cells'],coverage['measured_cells']),(3,2))
        self.assertEqual(coverage['by_observable']['coherence'],2)
        self.assertEqual(coverage['by_observable']['polarimetric_ratio'],1)
        self.assertFalse(radar['selected_cell']['measured'])
        self.assertTrue(radar['model_integration']['measured_context_available'])
        self.assertFalse(radar['model_integration']['sar_trained_weights_available'])
        self.assertNotIn('DO_NOT_PUT_FULL_RASTER_CATALOG_IN_CONTEXT',json.dumps(evidence))
        self.assertNotIn('by_h3',json.dumps(evidence))
        self.assertLess(len(json.dumps(evidence)),6000)
        self.assertIn('other cells in this scan do have measured radar observations',template(evidence))
        self.assertNotIn('No measured radar field is included',template(evidence))

    def test_coverage_counts_are_available_to_the_existing_numeric_parity_checker(self):
        evidence=context(scan_fixture())
        prose,claims=verify_and_render({'markdown':'Measured cell count: {{count}}.',
            'claims':{'count':'/nisar/sample_coverage/measured_cells'}},evidence)
        self.assertEqual(claims,[{'name':'count','path':'/nisar/sample_coverage/measured_cells','value':2}])
        self.assertIn('measured_cells: 2',prose)

    def test_invalid_values_and_outside_points_do_not_become_coverage(self):
        scan=scan_fixture();scan['nisar']['by_h3']={}
        scan['nisar']['cells']=[{'h3_index':CLIFF,'nisar_coherence':1.1},
            {'h3_index':VALLEY,'nisar_coherence':False,'nisar_polarimetric_moisture_index':-.1},
            {'h3_index':HOTSPOT,'nisar_coherence':float('nan')},
            {'h3_index':'outside','nisar_coherence':.5}]
        radar=provider_context(scan,{'h3_index':CLIFF})[1]
        self.assertEqual(radar['sample_coverage']['measured_cells'],0)
        self.assertEqual(radar['sample_coverage']['total_cells'],3)
        self.assertFalse(radar['selected_cell']['measured'])

    def test_pending_and_unknown_coverage_do_not_claim_measured_data(self):
        scan=scan_fixture();scan['nisar']={'measurement_access':{'status':'SUBSET_PENDING'}}
        for row in scan['physical_assessment']['cells']:row.pop('nisar_coherence',None)
        self.assertIn('processing is pending',template(context(scan)))
        radar=provider_context({'nisar':{'measurement_access':{'status':'MEASURED_AVAILABLE'}}})[1]
        self.assertIsNone(radar['sample_coverage']['total_cells'])
        self.assertIsNone(radar['sample_coverage']['measured_cells'])


class RadarSelection(unittest.IsolatedAsyncioTestCase):
    async def test_post_plan_selected_cell_is_rebound_in_template_without_network(self):
        scan=scan_fixture()
        class Service:
            def get_scan_state(self,*_):return {'response':scan}
            def inspect_hexagon(self,cell_id,*_):return next(row for row in scan['physical_assessment']['cells'] if row['h3_index']==cell_id)
            def trace_runout(self,cell_id,*_):return {'h3_index':cell_id,'status':'CONDITIONAL'}
        agent=SimpleNamespace(scan_service=Service())
        reply=await chat_active(agent,'Inspect the steepest cliff','scan-a','now-a',{},'default',
            selected_h3_index=HOTSPOT,autonomous=True,request_assistant=SimpleNamespace(configured=False,model=None))
        self.assertEqual(reply['briefing_guard'],'VERIFIED_TEMPLATE')
        self.assertIn('measured radar observables for this selected cell',reply['clean_markdown'])
        self.assertNotIn('other cells in this scan do have',reply['clean_markdown'])
        self.assertEqual(reply['ui_action']['actions'][-1]['h3_index'],CLIFF)


if __name__=='__main__':unittest.main()
