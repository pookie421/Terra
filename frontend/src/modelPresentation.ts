import {MODELS,cellIn,type Cell,type MapLayer,type PredictionModel,type PredictionModelInfo,type Scan} from './types.js';

export const MODEL_LABELS:Record<PredictionModel,string>={terra_physics_ml:'Terra · Physics + ML',nasa_lhasa_2:'NASA LHASA · ~1 km',physics_only:'Physics only · FS / FOSM',weather_terrain_ml:'Weather + terrain ML',rainfall_only:'Rainfall only'};
const MODEL_ABLATION:Partial<Record<PredictionModel,typeof MODELS[number]>>={terra_physics_ml:'Physics-Informed ML',physics_only:'Physics-Only',weather_terrain_ml:'Weather+Terrain ML',rainfall_only:'Rainfall-Only'};
export const LAYER_LABELS:Record<MapLayer,string>={probability:'Model score',fs_mid:'Factor of safety',lhasa_2:'NASA / reference',nisar_radar:'Radar coherence',slope:'Slope',pore_pressure:'Pressure head'};
export const finite=(value:unknown):number|null=>typeof value==='number'&&Number.isFinite(value)?value:null;

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
  return modelInfo(scan,model)?.label??MODEL_LABELS[model];
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
