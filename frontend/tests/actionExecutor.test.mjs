import test from 'node:test';
import assert from 'node:assert/strict';
import {executeActions,resolveCell} from '../.test-build/actionExecutor.js';
import {mapValue,predictionScore,modelLabel} from '../.test-build/modelPresentation.js';

function scan(id='origin',state=id,cells=[cell('flat',3,.2),cell('steep',41,.8)]){
  return {scan_id:id,scenario_id:state,location:{latitude:0,longitude:0},physical_assessment:{cells},h3_geojson:{type:'FeatureCollection',features:cells.map(properties=>({type:'Feature',geometry:{type:'Point',coordinates:[0,0]},properties}))}};
}
function cell(id,slope,score){return {h3_index:id,latitude:0,longitude:0,slope_deg:slope,elevation_m:100,fs_mid:2-score,prediction_scores:{terra_physics_ml:score,nasa_lhasa_2:score/2}};}
function workspace(initial=scan()){
  const current={scan:initial,selected:initial?.physical_assessment.cells[0].h3_index??'',model:'terra_physics_ml',alive:true,calls:[]};
  const handlers={
    scan:()=>current.scan,selected:()=>current.selected,model:()=>current.model,isCurrent:()=>current.alive,
    scanCoordinates:async(lat,lon)=>{await Promise.resolve();current.calls.push(['scan',lat,lon]);current.scan=scan('new');current.selected='flat';},
    searchLocation:async(query)=>{await Promise.resolve();current.calls.push(['search',query]);current.scan=scan('named');current.selected='flat';},
    selectModel:model=>{current.calls.push(['model',model]);current.model=model;},
    setHorizon:async(horizon)=>{await Promise.resolve();current.calls.push(['horizon',horizon]);current.scan=scan('new','future',[cell('future-flat',2,.3),cell('future-steep',48,.9)]);},
    setLayer:layer=>current.calls.push(['layer',layer]),
    flyTo:(target,camera)=>current.calls.push(['fly',target.h3_index,camera.pitch,camera.zoom]),
    inspect:target=>{current.calls.push(['inspect',target.h3_index]);current.selected=target.h3_index;},
    compare:(a,b)=>current.calls.push(['compare',a.h3_index,b.h3_index]),
  };
  return {current,handlers};
}
function reply(initial,actions){return {scan_id:initial?.scan_id??null,scenario_id:initial?.scenario_id??null,ui_action:{actions:actions.map((action,index)=>({scan_id:initial?.scan_id??null,scenario_id:initial?.scenario_id??null,state_scope:index===0?'bound':'chain',...action}))}};}

test('an ordered chain awaits scans and forecasts, retains both model switches, and resolves new-grid targets',async()=>{
  const {current,handlers}=workspace(null);
  const result=await executeActions(reply(null,[
    {type:'scan_coordinates',latitude:11.548,longitude:76.134},
    {type:'select_prediction_model',model:'nasa_lhasa_2'},
    {type:'select_prediction_model',model:'terra_physics_ml'},
    {type:'set_time_horizon',horizon:'t_24h'},
    {type:'inspect_cell',target:'steepest'},
    {type:'fly_to_cell',target:'selected',pitch:60,zoom:15},
    {type:'compare_cells',primary_target:'steepest',secondary_target:'flattest'},
    {type:'set_map_layer',layer:'nisar_radar'},
  ]),handlers);
  assert.deepEqual(current.calls,[['scan',11.548,76.134],['model','nasa_lhasa_2'],['model','terra_physics_ml'],['horizon','t24'],['inspect','future-steep'],['fly','future-steep',60,15],['compare','future-steep','future-flat'],['layer','nisar_radar']]);
  assert.equal(result.length,8);assert.ok(result.every(v=>v.status==='completed'));
  assert.match(result[1].summary,/NASA LHASA/);assert.match(result[2].summary,/Terra/);
  assert.equal(result.at(-1).scenario_id,'future');
});

test('a named location chain works before the first scan',async()=>{
  const {current,handlers}=workspace(null);
  await executeActions(reply(null,[{type:'search_location',query:'Unnamed valley'},{type:'inspect_cell',target:'riskiest'}]),handlers);
  assert.deepEqual(current.calls,[['search','Unnamed valley'],['inspect','steep']]);
});

test('stale response identity is rejected before any application mutation',async()=>{
  const {current,handlers}=workspace(scan('current'));
  await assert.rejects(executeActions(reply(scan('old'),[{type:'set_map_layer',layer:'slope'}]),handlers),/earlier scan/);
  assert.deepEqual(current.calls,[]);
});

test('all static arguments are validated before starting a partial chain',async()=>{
  const {current,handlers}=workspace(null);
  await assert.rejects(executeActions(reply(null,[{type:'scan_coordinates',latitude:12,longitude:76},{type:'select_prediction_model',model:'invented_engine'}]),handlers),/Unknown prediction model/);
  assert.deepEqual(current.calls,[]);
});

test('manual workspace changes cancel remaining chained actions after an awaited request',async()=>{
  const {current,handlers}=workspace(null);let release;
  handlers.scanCoordinates=async()=>{current.calls.push(['scan']);await new Promise(resolve=>{release=resolve;});};
  const pending=executeActions(reply(null,[{type:'scan_coordinates',latitude:1,longitude:2},{type:'select_prediction_model',model:'nasa_lhasa_2'}]),handlers);
  await Promise.resolve();current.alive=false;release();
  await assert.rejects(pending,error=>error.name==='AbortError');
  assert.deepEqual(current.calls,[['scan']]);
});

test('fixed cells from an old grid fail explicitly and stop subsequent actions',async()=>{
  const original=scan('old','old',[cell('old-only',42,.9)]),{current,handlers}=workspace(original);
  const outcomes=await executeActions(reply(original,[{type:'scan_coordinates',latitude:1,longitude:2},{type:'inspect_cell',h3_index:'old-only'},{type:'set_map_layer',layer:'slope'}]),handlers);
  assert.equal(outcomes.at(-1).status,'failed');
  assert.match(outcomes.at(-1).summary,/does not belong/);
  assert.deepEqual(current.calls,[['scan',1,2]]);
});

test('post-location actions must explicitly use the chain state scope',async()=>{
  const {handlers}=workspace(null);
  const outcomes=await executeActions(reply(null,[{type:'search_location',query:'Valley'},{type:'inspect_cell',target:'steepest',state_scope:'bound'}]),handlers);
  assert.equal(outcomes.at(-1).status,'failed');assert.match(outcomes.at(-1).summary,/new action-chain state/);
});

test('unavailable forecast stops the chain and preserves the error',async()=>{
  const original=scan(),{current,handlers}=workspace(original);
  handlers.setHorizon=async()=>{throw new Error('Forecast unavailable');};
  const outcomes=await executeActions(reply(original,[{type:'set_time_horizon',horizon:'t_72h'},{type:'inspect_cell',target:'steepest'}]),handlers);
  assert.equal(outcomes.length,1);assert.match(outcomes[0].summary,/Forecast unavailable/);assert.deepEqual(current.calls,[]);
});

test('out-of-range coordinates and camera values cannot reach handlers',async()=>{
  const original=scan(),{current,handlers}=workspace(original);
  await assert.rejects(executeActions(reply(original,[{type:'scan_coordinates',latitude:91,longitude:2}]),handlers),/latitude/);
  await assert.rejects(executeActions(reply(original,[{type:'fly_to_cell',target:'selected',zoom:100}]),handlers),/camera zoom/);
  assert.deepEqual(current.calls,[]);
});

test('explicit missing scores mask stale NASA values and genuine zeros remain valid',()=>{
  const value={...cell('a',3,0),lhasa_probability:.9,prediction_scores:{terra_physics_ml:0,nasa_lhasa_2:null}};
  assert.equal(predictionScore(value,'terra_physics_ml'),0);
  assert.equal(predictionScore(value,'nasa_lhasa_2'),null);
  assert.equal(mapValue({...value,nisar_coherence:0},'nisar_radar','terra_physics_ml'),0);
  assert.equal(mapValue(value,'nisar_radar','terra_physics_ml'),null);
});

test('missing radar and NASA metadata never acquire invented measurements or a known product version',()=>{
  const value={...cell('a',3,.2),prediction_scores:{terra_physics_ml:.2}};
  assert.equal(predictionScore(value,'nasa_lhasa_2'),null);
  assert.equal(mapValue(value,'nisar_radar','terra_physics_ml'),null);
  assert.equal(modelLabel({...scan(),nasa_lhasa:{mode:'LHASA_REFERENCE_EMULATION'}},'nasa_lhasa_2'),'Coarse reference estimate');
});

test('risk target follows the selected model and refuses missing scores',()=>{
  const current=scan('s','s',[{...cell('first',4,.9),prediction_scores:{terra_physics_ml:.9,nasa_lhasa_2:.1}},{...cell('second',40,.4),prediction_scores:{terra_physics_ml:.4,nasa_lhasa_2:.8}}]);
  assert.equal(resolveCell(current,undefined,'riskiest','first','terra_physics_ml').h3_index,'first');
  assert.equal(resolveCell(current,undefined,'riskiest','first','nasa_lhasa_2').h3_index,'second');
  assert.throws(()=>resolveCell(scan('s','s',[{...cell('a',4,.2),prediction_scores:{nasa_lhasa_2:null}}]),undefined,'riskiest','a','nasa_lhasa_2'),/no supported values/);
});

test('mixed NASA/reference coverage does not rank reference indices against official probabilities',()=>{
  const official={...cell('official',10,.4),official_score:0,lhasa_probability:0,prediction_scores:{nasa_lhasa_2:0}};
  const proxy={...cell('proxy',42,.8),reference_score:.99,prediction_scores:{nasa_lhasa_2:.99}};
  const current=scan('s','s',[official,proxy]);current.nasa_lhasa={mode:'MIXED_LIVE_AND_REFERENCE'};
  assert.equal(resolveCell(current,undefined,'riskiest','proxy','nasa_lhasa_2').h3_index,'official');
  assert.equal(modelLabel(current,'nasa_lhasa_2'),'NASA / coarse reference');
});
