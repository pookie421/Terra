import type {FeatureCollection,Geometry} from 'geojson';
export type ModelName='Physics-Informed ML'|'Physics-Only'|'Weather+Terrain ML'|'Rainfall-Only';
export const MODELS:ModelName[]=['Physics-Informed ML','Physics-Only','Weather+Terrain ML','Rainfall-Only'];
export type PredictionModel='terra_physics_ml'|'nasa_lhasa_2'|'physics_only'|'weather_terrain_ml'|'rainfall_only';
export const PREDICTION_MODELS:PredictionModel[]=['terra_physics_ml','nasa_lhasa_2','physics_only','weather_terrain_ml','rainfall_only'];
// Keep every accepted action ID above; only these three are primary model choices.
export const PRIMARY_PREDICTION_MODELS:PredictionModel[]=['terra_physics_ml','nasa_lhasa_2','rainfall_only'];
export const DIAGNOSTIC_MODELS:PredictionModel[]=['physics_only','weather_terrain_ml'];
export type MapLayer='probability'|'fs_mid'|'lhasa_2'|'nisar_radar'|'slope'|'pore_pressure';
export const MAP_LAYERS:MapLayer[]=['probability','fs_mid','lhasa_2','nisar_radar','slope','pore_pressure'];
export interface Cell {
  h3_index:string;latitude:number;longitude:number;slope_deg:number;elevation_m:number;twi?:number;
  fs_mid?:number|null;fs_low?:number|null;fs_high?:number|null;fs_interval?:Array<number|null>;
  fosm_variance?:number|null;pressure_head_m?:number|null;critical_rainfall_deficit_mm?:number|null;
  rainfall_threshold_attainable_at_duration?:boolean;score?:number|null;physics_informed_score?:number|null;
  learned_trigger_score?:number;mechanical_admissibility?:number;newmark_displacement_cm?:number|null;newmark_status?:string;
  provenance_label?:string;ablation_scores?:Partial<Record<ModelName,number>>;
  uncertainty_variance_allocation?:Record<string,number>;fs_jacobian?:Record<string,number>;
  input_values?:Record<string,number>;joint_covariance?:number[][];
  prediction_scores?:Partial<Record<PredictionModel,number|null>>;
  lhasa_probability?:number|null;lhasa_reference_score?:number|null;lhasa_susceptibility?:number|null;
  official_score?:number|null;reference_score?:number|null;active_score?:number|null;provenance?:string;
  nisar_coherence?:number|null;nisar_los_velocity_mm_yr?:number|null;nisar_polarimetric_moisture_index?:number|null;
  nisar_granule_count?:number|null;nisar_latest_acquisition_utc?:string|null;nisar_mode?:string;
  nisar_measurement_provenance?:Record<string,unknown>|null;
}
export interface Provider {available?:boolean;mode?:string;provider?:string;fetched_at_utc?:string;error?:string|null;age_seconds?:number;latency_ms?:number|null;[key:string]:unknown}
export interface Route {path:string[];peak_velocity_m_s?:number;travel_time_s?:number;travel_distance_m?:number;stop_reason?:string;mu_b?:number;xi_m_s2?:number;flow_depth_m?:number;coordinates_lonlat?:number[][]}
export interface Runout {status:string;h3_index?:string;scope?:string;peak_velocity_ms?:number;scenarios?:Route[];corridor_h3_indices?:string[]}
export type HorizonKey='now'|'t24'|'t48'|'t72'|'peak';
export interface ForecastView {key:HorizonKey;hour:number;valid_time_utc?:string;scenario_id:string;summary?:Record<string,unknown>}
export interface ForecastHour {hour?:number;minimum_fs?:number|null;[key:string]:unknown}
export interface Forecast {status:string;reason?:string;views?:ForecastView[];peak_selection?:string;paths?:Record<string,{hourly?:ForecastHour[];[key:string]:unknown}>;[key:string]:unknown}
export interface LiveInputs {rain_24h_mm?:number|null;rain_prior_6d_mm?:number|null;surface_moisture_m3m3?:number|null;weather_valid_time_utc?:string;soil_properties?:Record<string,unknown>;pga_proxy_g?:number|null;river_discharge_m3_s?:number|null;source_notes?:unknown}
export interface Scan {
  scan_id:string;scenario_id:string;schema_version:string;generated_at_utc:string;
  location:{display_name:string;query:string;latitude:number;longitude:number};
  source_status:Record<string,Provider>;source_count?:number;
  h3_geojson:FeatureCollection<Geometry,Cell>;terrain_geojson?:FeatureCollection<Geometry,Cell>;
  runout_geojson:FeatureCollection;runout_corridor_geojson?:FeatureCollection;runout_assessment?:Runout;
  physical_assessment:{status:string;cells:Cell[];hotspot?:Cell;additional_mechanism_status?:string};
  physics_informed_assessment?:{status:string;cells:Cell[];hotspot?:Cell;reason?:string;version?:string};
  live_inputs?:LiveInputs;predictive_forecast_72h?:Forecast;forecast_view?:ForecastView;
  evidence?:Record<string,unknown>;summary:Record<string,unknown>;
  nasa_lhasa?:NasaLhasa;nisar?:NisarContext;
  prediction_models?:Partial<Record<PredictionModel,PredictionModelInfo>>|PredictionModelInfo[];
  prediction_scores?:Record<string,Partial<Record<PredictionModel,number|null>>>;
}
export interface PredictionModelInfo {id?:PredictionModel;label?:string;status?:string;scope?:string;native_resolution_m?:number;product_version?:string;[key:string]:unknown}
export interface NasaLhasa {status?:string;mode?:string;provenance?:string;label?:string;product_version?:string;native_resolution_m?:number;valid_time_utc?:string;scope?:string;reason?:string;cells?:Array<Partial<Cell>&{h3_index:string}>;[key:string]:unknown}
export interface NisarGranule {granule_id?:string;collection_concept_id?:string;time_start?:string;time_end?:string;orbit_direction?:string;browse_url?:string;opendap_url?:string;[key:string]:unknown}
export interface NisarContext {status?:string;mode?:string;scope?:string;reason?:string;granule_count?:number;latest_acquisition_utc?:string;granules?:NisarGranule[];cells?:Array<Partial<Cell>&{h3_index:string}>;[key:string]:unknown}
export type CellTarget='steepest'|'riskiest'|'selected'|'lowest_fs'|'flattest';
export type AgentHorizon='now'|'t_24h'|'t_48h'|'t_72h'|'peak_72h';
interface ActionBase {scan_id?:string|null;scenario_id?:string|null;state_scope?:'bound'|'chain'}
export type Action=ActionBase&(
  {type:'scan_coordinates';latitude:number;longitude:number;label?:string}|
  {type:'search_location';query:string}|
  {type:'select_prediction_model';model:PredictionModel}|
  {type:'set_time_horizon';horizon:AgentHorizon}|
  {type:'set_map_layer';layer:MapLayer}|
  {type:'fly_to_cell';h3_index?:string;target?:CellTarget;pitch?:number;bearing?:number;zoom?:number}|
  {type:'inspect_cell';h3_index?:string;target?:CellTarget}|
  {type:'compare_cells';primary_h3?:string;secondary_h3?:string;primary_target?:CellTarget;secondary_target?:CellTarget}
);
export interface ThoughtStep {step:number;phase:string;reasoning:string}
export interface ActionOutcome {step:number;type:string;status:'completed'|'failed';summary:string;scan_id?:string;scenario_id?:string}
export interface BoundClaim {name:string;path:string;value:number}
export interface ChatReply {scan_id:string|null;scenario_id:string|null;active_prediction_model?:PredictionModel;clean_markdown:string;briefing_guard:string;bound_claims?:BoundClaim[];model_error?:string;ui_action?:{actions:Action[]};updated_scan_data?:Scan;thought_steps?:ThoughtStep[]}
export interface Health {status?:string;engine?:string;model_ready?:boolean;model_version?:string;[key:string]:unknown}
export interface Connections {
  scope:'deployment';
  assistant:{provider:string;model:string|null;configured:boolean;mode:string;requires_personal_key:false};
  earthdata:{configured:boolean;requires_personal_key:false};
}
export const SOURCES=['geocoding','terrarium_dem','open_meteo','nasa_power','soilgrids','usgs','overpass','eonet_wildfires','eonet_landslides','coolr','river','bedrock','ensemble','satellite','nasa_lhasa','nisar'];
export const SOURCE_LABELS:Record<string,string>={geocoding:'Location',terrarium_dem:'Terrain',open_meteo:'Weather',nasa_power:'NASA POWER',soilgrids:'SoilGrids',usgs:'USGS seismic',overpass:'OpenStreetMap',eonet_wildfires:'EONET fires',eonet_landslides:'EONET landslides',coolr:'NASA COOLR',river:'GloFAS rivers',bedrock:'Bedrock geology',ensemble:'Weather ensemble',satellite:'Satellite imagery',nasa_lhasa:'NASA LHASA',nisar:'NISAR L-SAR'};
export function cellIn(scan:Scan,id:string):Cell|undefined {
  const physical=scan.physical_assessment.cells.find(c=>c.h3_index===id);
  const learned=scan.physics_informed_assessment?.cells.find(c=>c.h3_index===id);
  const feature=scan.h3_geojson.features.find(f=>f.properties.h3_index===id)?.properties;
  const nasa=scan.nasa_lhasa?.cells?.find(c=>c.h3_index===id),radar=scan.nisar?.cells?.find(c=>c.h3_index===id);
  return physical||feature?{...feature,...physical,...learned,...nasa,...radar,prediction_scores:scan.prediction_scores?.[id]??learned?.prediction_scores??physical?.prediction_scores??feature?.prediction_scores} as Cell:undefined;
}
export const fmt=(v:unknown,digits=2):string=>typeof v==='number'&&Number.isFinite(v)?(v!==0&&Math.abs(v)<.001?v.toExponential(2):v.toLocaleString('en-US',{maximumFractionDigits:digits})):'—';
export function coord(latitude:number,longitude:number){return `${Math.abs(latitude).toFixed(4)}° ${latitude<0?'S':'N'}, ${Math.abs(longitude).toFixed(4)}° ${longitude<0?'W':'E'}`;}
export const HORIZONS:{key:HorizonKey;label:string;sub:string}[]=[{key:'now',label:'Now',sub:'Observed'},{key:'t24',label:'+24h',sub:'Forecast'},{key:'t48',label:'+48h',sub:'Forecast'},{key:'t72',label:'+72h',sub:'Forecast'},{key:'peak',label:'Peak',sub:'72h mechanical'}];
