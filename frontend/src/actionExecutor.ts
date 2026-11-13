import {cellIn,MAP_LAYERS,PREDICTION_MODELS,type Action,type ActionOutcome,type Cell,type CellTarget,type ChatReply,type HorizonKey,type MapLayer,type PredictionModel,type Scan} from './types.js';
import {scanCells,criticalCell,MODEL_LABELS,LAYER_LABELS} from './modelPresentation.js';

const TARGETS:CellTarget[]=['steepest','riskiest','selected','lowest_fs','flattest'];
const HORIZON_KEYS={now:'now',t_24h:'t24',t_48h:'t48',t_72h:'t72',peak_72h:'peak'} as const;
export interface ActionHandlers {
  scan:()=>Scan|null;selected:()=>string;model:()=>PredictionModel;isCurrent:()=>boolean;
  scanCoordinates:(latitude:number,longitude:number,label?:string)=>Promise<void>;
  searchLocation:(query:string)=>Promise<void>;
  selectModel:(model:PredictionModel)=>void|Promise<void>;
  setHorizon:(horizon:HorizonKey)=>Promise<void>;
  setLayer:(layer:MapLayer)=>void|Promise<void>;
  flyTo:(cell:Cell,camera:{pitch?:number;bearing?:number;zoom?:number})=>void;
  inspect:(cell:Cell)=>void;
  compare:(primary:Cell,secondary:Cell)=>void;
  outcome?:(outcome:ActionOutcome)=>void;
}
function fail(message:string):never{throw new Error(message);}
function bounded(v:unknown,min:number,max:number,name:string){if(typeof v!=='number'||!Number.isFinite(v)||v<min||v>max)fail('Invalid '+name+'.');}
function validateTarget(id:string|undefined,target:CellTarget|undefined){
  if(id&&target)fail('A cell action cannot mix a fixed cell and a dynamic target.');
  if(!id&&!target)fail('A cell action needs a cell or target.');
  if(id&&(typeof id!=='string'||id.length>32))fail('Invalid H3 cell identifier.');
  if(target&&!TARGETS.includes(target))fail('Unknown cell target.');
}
function validate(action:Action){
  switch(action.type){
    case 'scan_coordinates':bounded(action.latitude,-90,90,'latitude');bounded(action.longitude,-180,180,'longitude');break;
    case 'search_location':if(typeof action.query!=='string'||!action.query.trim()||action.query.length>500)fail('Invalid location search.');break;
    case 'select_prediction_model':if(!PREDICTION_MODELS.includes(action.model))fail('Unknown prediction model.');break;
    case 'set_time_horizon':if(!Object.hasOwn(HORIZON_KEYS,action.horizon))fail('Unknown forecast horizon.');break;
    case 'set_map_layer':if(!MAP_LAYERS.includes(action.layer))fail('Unknown map layer.');break;
    case 'fly_to_cell':
      if(action.pitch!==undefined)bounded(action.pitch,0,80,'camera pitch');
      if(action.bearing!==undefined)bounded(action.bearing,-360,360,'camera bearing');
      if(action.zoom!==undefined)bounded(action.zoom,0,22,'camera zoom');
      validateTarget(action.h3_index,action.target);break;
    case 'inspect_cell':validateTarget(action.h3_index,action.target);break;
    case 'compare_cells':validateTarget(action.primary_h3,action.primary_target);validateTarget(action.secondary_h3,action.secondary_target);break;
    default:fail('Unsupported application action.');
  }
}
function completed(action:Action):string {
  switch(action.type){
    case 'scan_coordinates':return `Scanned ${action.latitude}, ${action.longitude}.`;
    case 'search_location':return `Searched and scanned ${action.query}.`;
    case 'select_prediction_model':return 'Prediction model: '+MODEL_LABELS[action.model]+'.';
    case 'set_map_layer':return 'Map layer: '+LAYER_LABELS[action.layer]+'.';
    case 'set_time_horizon':return 'Forecast view: '+({now:'Now',t_24h:'+24 hours',t_48h:'+48 hours',t_72h:'+72 hours',peak_72h:'72-hour peak'}[action.horizon])+'.';
    case 'fly_to_cell':return 'Focused the map on the requested cell.';
    case 'inspect_cell':return 'Opened the requested physical column and runout.';
    case 'compare_cells':return 'Pinned the requested cells for comparison.';
  }
}
export function resolveCell(scan:Scan|null,id:string|undefined,target:CellTarget|undefined,selected:string,model:PredictionModel):Cell {
  if(!scan)fail('Scan a location before selecting a terrain cell.');
  validateTarget(id,target);
  if(id){const cell=cellIn(scan,id);return cell??fail('The requested cell does not belong to the current scan.');}
  if(target==='selected')return cellIn(scan,selected)??fail('The selected cell is no longer available.');
  if(target==='riskiest')return criticalCell(scan,model)??fail('This target has no supported values in the current scan.');
  const metric=(cell:Cell)=>target==='lowest_fs'?cell.fs_mid:cell.slope_deg;
  const cells=scanCells(scan).filter(c=>typeof metric(c)==='number'&&Number.isFinite(metric(c)));
  const ascending=target==='flattest'||target==='lowest_fs';
  cells.sort((a,b)=>(ascending?1:-1)*(metric(a)!-metric(b)!)||a.h3_index.localeCompare(b.h3_index));
  return cells[0]??fail('This target has no supported values in the current scan.');
}
export async function executeActions(reply:Pick<ChatReply,'scan_id'|'scenario_id'|'ui_action'>,handlers:ActionHandlers):Promise<ActionOutcome[]> {
  const origin=handlers.scan(),actions=reply.ui_action?.actions??[],outcomes:ActionOutcome[]=[];
  if((origin?.scan_id??null)!==reply.scan_id||(origin?.scenario_id??null)!==reply.scenario_id)fail('The action plan belongs to an earlier scan or forecast state.');
  if(actions.length>24)fail('Too many application actions.');
  for(const action of actions){
    if((action.scan_id??null)!==reply.scan_id||(action.scenario_id??null)!==reply.scenario_id)fail('An action has a different originating state.');
    validate(action);
  }
  let locationChanged=false;
  const current=()=>{if(!handlers.isCurrent())throw new DOMException('The action chain was superseded.','AbortError');};
  for(const [index,action] of actions.entries()){
    current();
    try{
      if(locationChanged&&action.state_scope!=='chain')fail('A post-scan action must be bound to the new action-chain state.');
      switch(action.type){
        case 'scan_coordinates':await handlers.scanCoordinates(action.latitude,action.longitude,action.label);locationChanged=true;break;
        case 'search_location':await handlers.searchLocation(action.query);locationChanged=true;break;
        case 'select_prediction_model':await handlers.selectModel(action.model);break;
        case 'set_time_horizon':await handlers.setHorizon(HORIZON_KEYS[action.horizon]);break;
        case 'set_map_layer':await handlers.setLayer(action.layer);break;
        case 'fly_to_cell':handlers.flyTo(resolveCell(handlers.scan(),action.h3_index,action.target,handlers.selected(),handlers.model()),action);break;
        case 'inspect_cell':handlers.inspect(resolveCell(handlers.scan(),action.h3_index,action.target,handlers.selected(),handlers.model()));break;
        case 'compare_cells':handlers.compare(resolveCell(handlers.scan(),action.primary_h3,action.primary_target,handlers.selected(),handlers.model()),resolveCell(handlers.scan(),action.secondary_h3,action.secondary_target,handlers.selected(),handlers.model()));break;
      }
      current();
      const state=handlers.scan();
      const outcome:ActionOutcome={step:index+1,type:action.type,status:'completed',summary:completed(action),scan_id:state?.scan_id,scenario_id:state?.scenario_id};
      outcomes.push(outcome);handlers.outcome?.(outcome);
    }catch(error){
      if(error instanceof DOMException&&error.name==='AbortError')throw error;
      const state=handlers.scan(),outcome:ActionOutcome={step:index+1,type:action.type,status:'failed',summary:error instanceof Error?error.message:String(error),scan_id:state?.scan_id,scenario_id:state?.scenario_id};
      outcomes.push(outcome);handlers.outcome?.(outcome);break;
    }
  }
  return outcomes;
}
