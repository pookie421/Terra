import test from 'node:test';
import assert from 'node:assert/strict';
import {radarCoverage} from '../.test-build/radarPresentation.js';

function fixture(){
  const cells=['a','b','c'].map(h3_index=>({h3_index,latitude:11,longitude:76,slope_deg:30,elevation_m:100}));
  return {generated_at_utc:'2026-11-14T00:00:00Z',h3_geojson:{features:cells.map(properties=>({properties}))},physical_assessment:{cells},nisar:{measurement_access:{status:'MEASURED_AVAILABLE'},latest_acquisition_utc:'2026-11-13T00:00:00Z',cells:[
    {h3_index:'a',nisar_coherence:0,measurement_provenance:{coherence:{time_start:'2026-10-08T00:00:00Z',time_end:'2026-10-20T00:00:00Z',support:{crs:'EPSG:32643',pixel_size_native_units:[80,-80]}}}},
    {h3_index:'b',nisar_coherence:.323,nisar_polarimetric_moisture_index:0,measurement_provenance:{coherence:{time_end:'2026-10-20T00:00:00Z'},polarimetric_ratio:{acquisition_utc:'2026-11-10T00:00:00Z'}}},
    {h3_index:'c',nisar_coherence:null},
  ]}};
}
test('coverage preserves zero samples, deduplicates cells, and distinguishes a selected mask from sector measurements',()=>{
  const scan=fixture();scan.terrain_geojson=scan.h3_geojson;
  scan.nisar.by_h3=Object.fromEntries(scan.nisar.cells.map(row=>[row.h3_index,row]));
  const view=radarCoverage(scan,'c');
  assert.equal(view.measuredCells,2);assert.equal(view.totalCells,3);
  assert.equal(view.byObservable.coherence,2);assert.equal(view.byObservable.polarimetricRatio,1);
  assert.equal(view.selectedMeasured,false);assert.match(view.selectedMessage,/other cells have measurements/);
  assert.equal(view.summary,'2 / 3 cells with valid samples');
});
test('masked raw radar overrides stale model copies and does not include points outside the scanned terrain',()=>{
  const scan=fixture();scan.physical_assessment.cells[2].nisar_coherence=.9;
  scan.nisar.cells.push({h3_index:'outside',nisar_coherence:.6});
  const view=radarCoverage(scan,'c');assert.equal(view.measuredCells,2);assert.equal(view.totalCells,3);assert.equal(view.selectedMeasured,false);
});
test('measurement date and posting come from sampled provenance, never latest discovery',()=>{
  const view=radarCoverage(fixture(),'a');
  assert.equal(view.oldestMeasurementAgeDays,25);assert.deepEqual(view.nativePostings,['80 × 80 m']);
  assert.equal(view.measuredAcquisitionDates.includes('2026-11-13T00:00:00.000Z'),false);
  assert.match(view.selectedMessage,/measured coherence/);
});
test('invalid numeric fields cannot inflate measured coverage and unknown age stays unknown',()=>{
  const scan=fixture();scan.nisar.cells=[{h3_index:'a',nisar_coherence:1.01,nisar_los_velocity_mm_yr:NaN,nisar_polarimetric_moisture_index:-.1},{h3_index:'b',nisar_coherence:.2},{h3_index:'c',nisar_coherence:false}];
  const view=radarCoverage(scan);assert.equal(view.measuredCells,1);assert.equal(view.oldestMeasurementAgeDays,null);
});
test('authoritative empty raw provenance cannot borrow an older model-copy acquisition date',()=>{
  const scan=fixture();scan.physical_assessment.cells[0].nisar_measurement_provenance={coherence:{time_end:'2026-11-01T00:00:00Z'}};
  scan.nisar.cells[0].measurement_provenance={};
  const view=radarCoverage(scan);assert.equal(view.oldestMeasurementAgeDays,null);
  assert.equal(view.measuredAcquisitionDates.includes('2026-11-01T00:00:00.000Z'),false);
});
test('metadata-only, authentication, pending and unreported coverage remain distinct',()=>{
  const scan=fixture();scan.nisar.cells=[];
  scan.nisar.measurement_access={status:'SUBSET_PENDING',jobs:[{status:'running'},{status:'successful'}]};
  assert.equal(radarCoverage(scan).status,'pending');assert.equal(radarCoverage(scan).pendingJobs,1);
  scan.nisar.measurement_access.status='AUTH_REQUIRED';assert.equal(radarCoverage(scan).canRefresh,false);
  scan.nisar={mode:'METADATA_LIVE'};assert.equal(radarCoverage(scan).status,'metadata');
  const unknown=radarCoverage({nisar:{measurement_access:{status:'MEASURED_AVAILABLE'}}});
  assert.equal(unknown.totalCells,null);assert.equal(unknown.measuredCells,null);assert.match(unknown.summary,/not reported/);
  assert.equal(radarCoverage(null).canRefresh,false);
});
test('future or incomplete measurement timestamps never become fabricated acquisition ages',()=>{
  const scan=fixture();scan.nisar.cells[1].measurement_provenance.coherence.time_end='2026-11-15T00:00:00Z';
  assert.equal(radarCoverage(scan).oldestMeasurementAgeDays,null);
});
