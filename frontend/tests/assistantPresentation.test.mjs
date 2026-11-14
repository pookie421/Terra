import test from 'node:test';
import assert from 'node:assert/strict';
import {actionTaskLabel,assistantTasks,taskResultSummary,assistantSuggestions,proseBlocks,inlineParts,safeAssistantLink} from '../.test-build/assistantPresentation.js';

const plan=[{type:'scan_coordinates',latitude:11.548,longitude:76.134},{type:'select_prediction_model',model:'nasa_lhasa_2'},{type:'inspect_cell',target:'steepest'}];
const completed={step:1,type:'scan_coordinates',status:'completed',summary:'Scanned coordinates',scan_id:'new',scenario_id:'now'};
function scan(forecast=true,hasCells=true){
  const cells=hasCells?[{h3_index:'example',score:.4,prediction_scores:{terra_physics_ml:.4,nasa_lhasa_2:.2}}]:[];
  return {physical_assessment:{cells},h3_geojson:{type:'FeatureCollection',features:cells.map(properties=>({type:'Feature',properties,geometry:{type:'Point',coordinates:[0,0]}}))},nasa_lhasa:{cells:cells.map(cell=>({...cell,official_score:.2,tomorrow_probability:.3}))},predictive_forecast_72h:{views:forecast?[{key:'now'},{key:'t72'}]:[{key:'now'}]}};
}

test('a returned plan stays planned until the executor reports actual dispatch',()=>{
  assert.deepEqual(assistantTasks({plannedActions:plan,executing:true}).map(task=>task.status),['planned','planned','planned']);
  assert.deepEqual(assistantTasks({plannedActions:plan,outcomes:[completed],executing:true,activeTask:2}).map(task=>task.status),['completed','running','planned']);
  assert.equal(assistantTasks({plannedActions:plan,activeTask:2})[1].status,'planned');
});

test('cancelling a chain preserves confirmed results and never invents later completion',()=>{
  const evidence={plannedActions:plan,outcomes:[completed],executing:false,activeTask:2,cancelled:true};
  assert.deepEqual(assistantTasks(evidence).map(task=>task.status),['completed','cancelled','cancelled']);
  assert.equal(taskResultSummary(evidence),'Stopped · 1 of 3 completed');
  assert.equal(assistantTasks(evidence)[1].detail,'Not completed');
});

test('a failed action remains failed and prevents later tasks being presented as running',()=>{
  const evidence={plannedActions:plan,outcomes:[completed,{step:2,type:'select_prediction_model',status:'failed',summary:'Official product unavailable'}],executing:true,activeTask:3};
  const tasks=assistantTasks(evidence);
  assert.deepEqual(tasks.map(task=>task.status),['completed','failed','cancelled']);
  assert.equal(tasks[1].detail,'Official product unavailable');
  assert.equal(taskResultSummary(evidence),'Stopped after a failed task · 1 of 3 completed');
});

test('an interrupted dispatch without an outcome is never called successful',()=>{
  const evidence={plannedActions:plan,outcomes:[completed],executionError:'The active workspace changed.'};
  assert.deepEqual(assistantTasks(evidence).map(task=>task.status),['completed','cancelled','cancelled']);
  assert.equal(taskResultSummary(evidence),'Stopped · 1 of 3 completed');
});

test('old outcome-only messages retain only known actions without inventing a plan',()=>{
  const tasks=assistantTasks({outcomes:[{step:3,type:'inspect_cell',status:'failed',summary:'Cell unavailable'},completed,{step:0,type:'search_location',status:'completed',summary:'Invalid step'}],executing:true,activeTask:2});
  assert.deepEqual(tasks.map(task=>[task.step,task.status]),[[1,'completed'],[3,'failed']]);
  assert.deepEqual(assistantTasks({executing:true}),[]);
  assert.equal(taskResultSummary({executing:true}),null);
});

test('completion summaries require every planned outcome to be confirmed',()=>{
  assert.equal(taskResultSummary({plannedActions:[plan[0]],outcomes:[completed]}),'Task completed');
  assert.equal(taskResultSummary({plannedActions:plan,outcomes:[completed]}),'Plan ready · 1 of 3 completed');
  assert.equal(taskResultSummary({plannedActions:plan,outcomes:plan.map((action,index)=>({step:index+1,type:action.type,status:'completed',summary:'Done'}))}),'3 tasks completed');
});

test('task labels explain intent without exposing cell identifiers in workspace chrome',()=>{
  assert.equal(actionTaskLabel({type:'inspect_cell',h3_index:'893969a6a5bffff'}),'Inspect the requested cell');
  assert.equal(actionTaskLabel({type:'fly_to_cell',target:'steepest'}),'Focus the steepest cell');
  assert.equal(actionTaskLabel({type:'set_time_horizon',horizon:'nasa_tomorrow'}),'Show NASA Tomorrow');
});

test('suggestions respect NASA daily products and available Terra forecast states',()=>{
  const nasa=assistantSuggestions(scan(),'nasa_lhasa_2','tomorrow');
  assert.ok(!nasa.some(item=>item.id==='forecast'));
  assert.match(nasa.find(item=>item.id==='critical').description,/Tomorrow/);
  assert.ok(assistantSuggestions(scan(),'terra_physics_ml').some(item=>item.id==='forecast'));
  assert.ok(!assistantSuggestions(scan(false),'terra_physics_ml').some(item=>item.id==='forecast'));
  assert.ok(assistantSuggestions(scan(),'terra_physics_ml').some(item=>item.id==='radar'));
});

test('empty-workspace suggestions prepare editable drafts and missing grids do not suggest a critical cell',()=>{
  assert.deepEqual(assistantSuggestions(null,'terra_physics_ml').map(item=>item.prompt),['Scan at ','Find ']);
  const suggestions=assistantSuggestions(scan(false,false),'terra_physics_ml');
  assert.ok(suggestions.some(item=>item.id==='sources'));
  assert.ok(!suggestions.some(item=>item.id==='critical'||item.id==='compare'));
});

test('missing official NASA pixels cannot produce critical-pixel or model-comparison suggestions',()=>{
  const current=scan();current.nasa_lhasa={cells:[{h3_index:'example',official_score:null,tomorrow_probability:null}]};
  const nasa=assistantSuggestions(current,'nasa_lhasa_2');
  assert.ok(!nasa.some(item=>item.id==='critical'||item.id==='compare'));
  assert.ok(nasa.some(item=>item.id==='sources'));
  assert.ok(!assistantSuggestions(current,'terra_physics_ml').some(item=>item.id==='compare'));
});

test('readable blocks preserve exact numeric wording and explicit ordered-list numbers',()=>{
  const blocks=proseBlocks('## Current evidence\n\nFS is **1.3817** at 41.1147°.\nRain is 6.9 mm.\n\n3. Inspect 3.6982°.\n7. Compare 0.822 with 0.000021.\n\n- Coverage is missing.');
  assert.deepEqual(blocks,[{kind:'heading',text:'Current evidence'},{kind:'paragraph',text:'FS is **1.3817** at 41.1147°. Rain is 6.9 mm.'},{kind:'list',ordered:true,items:['Inspect 3.6982°.','Compare 0.822 with 0.000021.'],numbers:[3,7]},{kind:'list',ordered:false,items:['Coverage is missing.']}]);
});

test('tables and code preserve source values rather than interpreting executable markup',()=>{
  assert.deepEqual(proseBlocks('| State | FS |\n| --- | ---: |\n| Now | 1.3817 |\n\n```json\n{"value":0.000021}\n```'),[
    {kind:'table',headers:['State','FS'],rows:[['Now','1.3817']]},
    {kind:'code',text:'{"value":0.000021}'},
  ]);
  assert.deepEqual(inlineParts('<img src=x onerror=alert(1)>'),[{kind:'text',text:'<img src=x onerror=alert(1)>'}]);
});

test('inline formatting stays limited and hostile links remain inert text',()=>{
  const parts=inlineParts('**FS** `1.3817` *conditional* [NASA](https://example.nasa.gov/data) [unsafe](javascript:alert)');
  assert.deepEqual(parts.filter(part=>part.kind!=='text').map(part=>part.kind),['strong','code','emphasis','link']);
  assert.equal(parts.find(part=>part.kind==='link').href,'https://example.nasa.gov/data');
  assert.ok(parts.some(part=>part.kind==='text'&&part.text==='[unsafe](javascript:alert)'));
  for(const url of ['javascript:alert(1)','data:text/html,hello','http://example.com','https://user:secret@example.com','https://example.com/\nunsafe','/api/v1/scan'])assert.equal(safeAssistantLink(url),null);
});
