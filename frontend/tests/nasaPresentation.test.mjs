import test from 'node:test';
import assert from 'node:assert/strict';
import {cellIn} from '../.test-build/types.js';
import {criticalCell,nasaNativeFeatures,nasaProductAvailable,nasaProductView,nasaResolution,predictionScore} from '../.test-build/modelPresentation.js';

function fixture(){
  const cells=['a','b','c'].map((h3_index,index)=>({h3_index,latitude:10,longitude:76+index/1000,slope_deg:20,elevation_m:50,fs_mid:1.5,prediction_scores:{terra_physics_ml:.6,nasa_lhasa_2:index===2?.99:0}}));
  return {scan_id:'scan',scenario_id:'now',physical_assessment:{cells},h3_geojson:{type:'FeatureCollection',features:cells.map(properties=>({type:'Feature',geometry:{type:'Point',coordinates:[properties.longitude,10]},properties}))},predictive_forecast_72h:{status:'available',views:[{key:'t72',hour:72,scenario_id:'future'}]},nasa_lhasa:{native_resolution_arcsec:30,native_resolution_m:927,mode:'MIXED_LIVE_AND_REFERENCE',cells:[
    {h3_index:'a',official_score:0,tomorrow_probability:.7,coarse_pixel_id:'today:1',coarse_pixel_id_by_product:{today:'1:1',tomorrow:'2:2'}},
    {h3_index:'b',official_score:.2,tomorrow_probability:0,coarse_pixel_id_by_product:{today:'1:1',tomorrow:'2:2'}},
    {h3_index:'c',official_score:null,reference_score:.99,active_score:.99,tomorrow_probability:null},
  ]}};
}
test('NASA Today uses official values, retaining zero and masking Terra reference indices',()=>{
  const base=fixture(),view=nasaProductView(base,'today');
  assert.equal(predictionScore(cellIn(view,'a'),'nasa_lhasa_2'),0);
  assert.equal(predictionScore(cellIn(view,'c'),'nasa_lhasa_2'),null);
  assert.equal(criticalCell(view,'nasa_lhasa_2').h3_index,'b');
  assert.equal(base.nasa_lhasa.cells[2].reference_score,.99,'source evidence is preserved');
});
test('NASA Tomorrow switches daily scores without manufacturing a Terra forecast or new issue time',()=>{
  const base=fixture(),view=nasaProductView(base,'tomorrow');
  assert.equal(predictionScore(cellIn(view,'a'),'nasa_lhasa_2'),.7);
  assert.equal(predictionScore(cellIn(view,'b'),'nasa_lhasa_2'),0);
  assert.equal(criticalCell(view,'nasa_lhasa_2').h3_index,'a');
  assert.equal(cellIn(view,'a').coarse_pixel_id,'2:2');
  assert.equal(view.scenario_id,'now');assert.equal(view.predictive_forecast_72h,base.predictive_forecast_72h);
  assert.equal(view.nasa_lhasa.valid_time_utc,undefined);
});
test('NASA native geometry is reused once per source pixel and retains selected terrain membership',()=>{
  const base=fixture(),geometry={type:'Polygon',coordinates:[[[76,10],[76.01,10],[76.01,10.01],[76,10.01],[76,10]]]};
  base.nasa_lhasa.native_pixels_geojson_by_product={today:{type:'FeatureCollection',features:[{type:'Feature',geometry,properties:{native_pixel_id:'1:1',product:'today',value:0}}]}};
  const native=nasaNativeFeatures(nasaProductView(base,'today'),'b');
  assert.equal(native.features.length,1);assert.equal(native.features[0].geometry,geometry);
  assert.equal(native.features[0].properties.h3_index,'b');assert.equal(native.features[0].properties.map_value,0);
  assert.equal(nasaNativeFeatures(nasaProductView(fixture(),'today'),'a'),undefined,'missing grid metadata never creates fake native pixels');
});
test('daily coverage and resolution come from source support rather than Terra cell spacing',()=>{
  const base=fixture();assert.equal(nasaProductAvailable(base,'tomorrow'),true);
  base.nasa_lhasa.cells.forEach(row=>{row.tomorrow_probability=null;});
  assert.equal(nasaProductAvailable(base,'tomorrow'),false);
  assert.match(nasaResolution(base),/30 arc-second.*0.93 km/);
  assert.equal(nasaResolution({...base,nasa_lhasa:{}}),'Native pixel spacing not reported');
});
