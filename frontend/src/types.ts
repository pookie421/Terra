import type {FeatureCollection,Geometry} from 'geojson';
export type ModelName='Physics-Informed ML'|'Physics-Only'|'Weather+Terrain ML'|'Rainfall-Only';
export const MODELS:ModelName[]=['Physics-Informed ML','Physics-Only','Weather+Terrain ML','Rainfall-Only'];
export interface Cell {
  h3_index:string;latitude:number;longitude:number;slope_deg:number;elevation_m:number;twi?:number;
  fs_mid?:number|null;fs_low?:number|null;fs_high?:number|null;fs_interval?:Array<number|null>;
  fosm_variance?:number|null;pressure_head_m?:number|null;critical_rainfall_deficit_mm?:number|null;
  rainfall_threshold_attainable_at_duration?:boolean;score?:number|null;physics_informed_score?:number|null;
  learned_trigger_score?:number;mechanical_admissibility?:number;newmark_displacement_cm?:number|null;newmark_status?:string;
  provenance_label?:string;ablation_scores?:Partial<Record<ModelName,number>>;
  uncertainty_variance_allocation?:Record<string,number>;fs_jacobian?:Record<string,number>;
  input_values?:Record<string,number>;joint_covariance?:number[][];
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
}
export interface Action {type:'fly_to_cell'|'inspect_cell';h3_index?:string;scan_id:string;scenario_id:string}
export interface BoundClaim {name:string;path:string;value:number}
export interface ChatReply {scan_id:string;scenario_id:string;clean_markdown:string;briefing_guard:string;bound_claims?:BoundClaim[];model_error?:string;ui_action?:{actions:Action[]};updated_scan_data?:Scan}
export interface Health {status?:string;engine?:string;model_ready?:boolean;model_version?:string;[key:string]:unknown}
export const SOURCES=['geocoding','terrarium_dem','open_meteo','nasa_power','soilgrids','usgs','overpass','eonet_wildfires','eonet_landslides','coolr','river','bedrock','ensemble','satellite'];
export const SOURCE_LABELS:Record<string,string>={geocoding:'Location',terrarium_dem:'Terrain',open_meteo:'Weather',nasa_power:'NASA POWER',soilgrids:'SoilGrids',usgs:'USGS seismic',overpass:'OpenStreetMap',eonet_wildfires:'EONET fires',eonet_landslides:'EONET landslides',coolr:'NASA COOLR',river:'GloFAS rivers',bedrock:'Bedrock geology',ensemble:'Weather ensemble',satellite:'Satellite imagery'};
export function cellIn(scan:Scan,id:string):Cell|undefined {
  const physical=scan.physical_assessment.cells.find(c=>c.h3_index===id);
  const learned=scan.physics_informed_assessment?.cells.find(c=>c.h3_index===id);
  return physical?{...physical,...learned}:undefined;
}
export const fmt=(v:unknown,digits=2):string=>typeof v==='number'&&Number.isFinite(v)?(v!==0&&Math.abs(v)<.001?v.toExponential(2):v.toLocaleString('en-US',{maximumFractionDigits:digits})):'—';
export function coord(latitude:number,longitude:number){return `${Math.abs(latitude).toFixed(4)}° ${latitude<0?'S':'N'}, ${Math.abs(longitude).toFixed(4)}° ${longitude<0?'W':'E'}`;}
export const HORIZONS:{key:HorizonKey;label:string;sub:string}[]=[{key:'now',label:'Now',sub:'Observed'},{key:'t24',label:'+24h',sub:'Forecast'},{key:'t48',label:'+48h',sub:'Forecast'},{key:'t72',label:'+72h',sub:'Forecast'},{key:'peak',label:'Peak',sub:'72h mechanical'}];
