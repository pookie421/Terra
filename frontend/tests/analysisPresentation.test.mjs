import test from 'node:test';
import assert from 'node:assert/strict';
import {chatProductMatches,incompleteAnalysis} from '../.test-build/modelPresentation.js';

test('a completed empty-grid scan is visibly incomplete while its retained response remains intact',()=>{
  const response={scan_id:'retained',scenario_id:'now',h3_geojson:{type:'FeatureCollection',features:[]},source_status:{terrarium_dem:{available:false,error:'AGGREGATE_TIMEOUT'}},live_inputs:{rain_24h_mm:6}};
  const before=structuredClone(response),notice=incompleteAnalysis(response);
  assert.equal(notice.title,'Analysis incomplete');assert.match(notice.message,/No terrain result grid/);assert.match(notice.message,/Sources/);
  assert.deepEqual(response,before,'the warning must not discard retained inputs or source errors');
  assert.equal(incompleteAnalysis(null),null,'initial world view is not a failed analysis');
  assert.equal(incompleteAnalysis({...response,h3_geojson:{type:'FeatureCollection',features:[{}]}}),null,'a returned grid does not trigger an empty-grid warning');
});
test('NASA chat replies must echo the exact daily product before prose or actions are accepted',()=>{
  assert.equal(chatProductMatches('nasa_lhasa_2','tomorrow','today'),false);
  assert.equal(chatProductMatches('nasa_lhasa_2','tomorrow',undefined),false);
  assert.equal(chatProductMatches('nasa_lhasa_2','tomorrow','tomorrow'),true);
  assert.equal(chatProductMatches('terra_physics_ml','today',undefined),true);
});
