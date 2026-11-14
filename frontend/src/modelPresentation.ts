import {MODELS,cellIn,type Cell,type MapLayer,type NasaProduct,type PredictionModel,type PredictionModelInfo,type Scan} from './types.js';
import type {FeatureCollection,Geometry} from 'geojson';

export const MODEL_LABELS:Record<PredictionModel,string>={terra_physics_ml:'Terra In House Model',nasa_lhasa_2:'NASA LHASA',physics_only:'Physics only',weather_terrain_ml:'Weather + terrain ML',rainfall_only:'Rainfall baseline'};
const MODEL_ABLATION:Partial<Record<PredictionModel,typeof MODELS[number]>>={terra_physics_ml:'Physics-Informed ML',physics_only:'Physics-Only',weather_terrain_ml:'Weather+Terrain ML',rainfall_only:'Rainfall-Only'};
export const LAYER_LABELS:Record<MapLayer,string>={probability:'Model score',fs_mid:'Factor of safety',lhasa_2:'NASA / reference',nisar_radar:'Radar coherence',slope:'Slope',pore_pressure:'Pressure head'};
export const finite=(value:unknown):number|null=>typeof value==='number'&&Number.isFinite(value)?value:null;

export function incompleteAnalysis(scan:Scan|null):{title:string;message:string}|null {
  if(!scan||scan.h3_geojson.features.length)return null;
  return {title:'Analysis incomplete',message:'No terrain result grid was returned. View Sources for provider status and retained inputs, or try scanning again.'};
}
export function chatProductMatches(model:PredictionModel,requested:NasaProduct,returned:NasaProduct|undefined):boolean {
  return model!=='nasa_lhasa_2'||requested===returned;
}

export function modelInfo(scan:Scan|null,model:PredictionModel):PredictionModelInfo|undefined {
  const catalog=scan?.prediction_models;
  return Array.isArray(catalog)?catalog.find(v=>v.id===model):catalog?.[model];
}
export function modelLabel(scan:Scan|null,model:PredictionModel):string {
  if(model==='nasa_lhasa_2'){
    const nasa=scan?.nasa_lhasa;
    if(nasa?.mode?.includes('EMULATION')||nasa?.provenance?.includes('EMULATION'))return 'Coarse reference estimate';
    if(nasa?.mode?.includes('REFERENCE'))return 'NASA / coarse reference';
    return nasa?.label??modelInfo(scan,model)?.label??MODEL_LABELS[model];
  }
  return MODEL_LABELS[model];
}
export function predictionScore(cell:Cell|undefined,model:PredictionModel):number|null {
  if(!cell)return null;
  // Explicit nulls are coverage masks, including NASA horizons without a product.
  if(cell.prediction_scores&&Object.hasOwn(cell.prediction_scores,model))return finite(cell.prediction_scores[model]);
  if(model==='nasa_lhasa_2')return finite(cell.active_score??cell.lhasa_probability??cell.lhasa_reference_score);
  const ablation=MODEL_ABLATION[model];
  if(ablation&&cell.ablation_scores&&Object.hasOwn(cell.ablation_scores,ablation))return finite(cell.ablation_scores[ablation]);
  return model==='terra_physics_ml'?finite(cell.score??cell.physics_informed_score):null;
}
export function mapValue(cell:Cell|undefined,layer:MapLayer,model:PredictionModel):number|null {
  if(!cell)return null;
  switch(layer){
    case 'probability':return predictionScore(cell,model);
    case 'lhasa_2':return predictionScore(cell,'nasa_lhasa_2');
    case 'nisar_radar':return finite(cell.nisar_coherence);
    case 'fs_mid':return finite(cell.fs_mid);
    case 'slope':return finite(cell.slope_deg);
    case 'pore_pressure':return finite(cell.pressure_head_m);
  }
}
export function scanCells(scan:Scan):Cell[]{
  return scan.h3_geojson.features.map(f=>cellIn(scan,f.properties.h3_index)).filter((c):c is Cell=>!!c);
}
export function criticalCell(scan:Scan,model:PredictionModel):Cell|undefined {
  let cells=scanCells(scan).filter(c=>predictionScore(c,model)!==null);
  if(model==='nasa_lhasa_2'){
    const official=cells.filter(c=>finite(c.official_score??c.lhasa_probability)!==null);
    if(official.length)cells=official;
  }
  return cells.sort((a,b)=>(predictionScore(b,model)!-predictionScore(a,model)!)||a.h3_index.localeCompare(b.h3_index))[0];
}
export function nasaDescription(scan:Scan|null):string {
  const nasa=scan?.nasa_lhasa;
  if(!nasa)return 'NASA raster coverage is queried with a scan.';
  if(nasa.mode?.includes('EMULATION')||nasa.provenance?.includes('EMULATION'))return 'Unvalidated coarse reference estimate. This is a local proxy, not a NASA-produced probability.';
  if(nasa.mode?.includes('REFERENCE'))return 'Mixed coverage: NASA raster values and separate unvalidated reference indices. The critical-cell view uses NASA pixels where available; unlike score meanings are not ranked together.';
  return nasa.scope??nasa.reason??'NASA-produced coarse raster. Repeating its pixel value across H3 cells does not create finer resolution.';
}

export function nasaResolution(scan:Scan|null):string {
  const source=scan?.nasa_lhasa;
  const arcsec=finite(source?.native_resolution_arcsec);
  const metres=finite(source?.native_resolution_m??source?.native_resolution_m_approx);
  if(arcsec)return `${arcsec} arc-second source pixels${metres?` · approximately ${(metres/1000).toLocaleString('en-US',{maximumFractionDigits:2})} km`:''}`;
  if(metres)return `Approximately ${metres>=1000?`${(metres/1000).toLocaleString('en-US',{maximumFractionDigits:2})} km`:`${metres} m`} source pixels`;
  return 'Native pixel spacing not reported';
}
export function nasaProductAvailable(scan:Scan|null,product:NasaProduct):boolean {
  return !!scan?.nasa_lhasa?.cells?.some(row=>finite(product==='today'?row.official_score??row.probability:row.tomorrow_probability)!==null);
}
/** Display NASA's two relative daily products without borrowing Terra forecast states.
 * Source pixels stay at their native support. No reference formula is relabelled NASA.
 */
export function nasaProductView(scan:Scan,product:NasaProduct):Scan {
  const rows=(scan.nasa_lhasa?.cells??[]).map(row=>{
    const official=finite(product==='today'?row.official_score??row.probability:row.tomorrow_probability);
    return {...row,official_score:official,probability:official,lhasa_probability:official,active_score:official,coarse_pixel_id:row.coarse_pixel_id_by_product?.[product]??row.coarse_pixel_id,
      lhasa_reference_score:null,reference_score:null,
      prediction_scores:{...row.prediction_scores,nasa_lhasa_2:official},
      provenance:official===null?'NASA_PRODUCT_UNAVAILABLE':'LIVE_NASA_IMAGESERVER'};
  });
  const lookup=new Map(rows.map(row=>[row.h3_index,row]));
  const scores=Object.fromEntries(scan.h3_geojson.features.map(feature=>{
    const id=feature.properties.h3_index;
    return [id,{...scan.prediction_scores?.[id]??feature.properties.prediction_scores,nasa_lhasa_2:lookup.get(id)?.official_score??null}];
  }));
  const officialCount=rows.filter(row=>row.official_score!==null).length;
  return {...scan,prediction_scores:scores,nasa_lhasa:{...scan.nasa_lhasa,cells:rows,active_product:product,
    label:'NASA LHASA',mode:officialCount?'LIVE_NASA_IMAGESERVER':'UNAVAILABLE',
    scope:officialCount?'Official NASA daily raster. Displayed H3 cells reuse the native source pixels.':'No official NASA value is available for this area and daily product.',
  }};
}
export function nasaNativeFeatures(scan:Scan,selected:string):FeatureCollection<Geometry,Cell&{map_value:number|null;map_reference:boolean}>|undefined {
  const product=scan.nasa_lhasa?.active_product??'today';
  const native=scan.nasa_lhasa?.native_pixels_geojson_by_product?.[product];
  if(!native)return;
  return {type:'FeatureCollection',features:native.features.flatMap(feature=>{
    const pixel=feature.properties.native_pixel_id;
    const members=scan.nasa_lhasa?.cells?.filter(row=>(row.coarse_pixel_id_by_product?.[product]??row.coarse_pixel_id)===pixel)??[];
    const target=members.find(row=>row.h3_index===selected)??members[0];
    const cell=target?cellIn(scan,target.h3_index):undefined;
    return cell?[{...feature,properties:{...cell,map_value:finite(feature.properties.value),map_reference:false}}]:[];
  })};
}
