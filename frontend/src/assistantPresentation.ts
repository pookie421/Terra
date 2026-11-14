import type {Action,ActionOutcome,CellTarget,NasaProduct,PredictionModel,Scan} from './types.js';
import {MODEL_LABELS,LAYER_LABELS,nasaProductAvailable,predictionScore,scanCells} from './modelPresentation.js';

export type TaskStatus='planned'|'running'|'completed'|'failed'|'cancelled';
export interface AssistantTask {
  step:number;label:string;status:TaskStatus;detail?:string;
}
export interface TaskEvidence {
  plannedActions?:Action[];outcomes?:ActionOutcome[];executing?:boolean;
  activeTask?:number;cancelled?:boolean;executionError?:string;
}
export interface AssistantSuggestion {id:string;label:string;description:string;prompt:string}
export type ProseBlock=
  {kind:'paragraph'|'heading';text:string}|
  {kind:'list';ordered:boolean;items:string[];numbers?:number[]}|
  {kind:'code';text:string}|
  {kind:'table';headers:string[];rows:string[][]};
export type InlinePart={kind:'text'|'strong'|'emphasis'|'code';text:string}|{kind:'link';text:string;href:string};

const targets:Record<CellTarget,string>={steepest:'the steepest cell',riskiest:'the critical cell',selected:'the selected cell',lowest_fs:'the lowest-stability cell',flattest:'the flattest cell'};
const horizons={now:'current conditions',t_24h:'the 24-hour forecast',t_48h:'the 48-hour forecast',t_72h:'the 72-hour forecast',peak_72h:'the peak forecast state',nasa_today:'NASA Today',nasa_tomorrow:'NASA Tomorrow'};
export function actionTaskLabel(action:Action):string {
  switch(action.type){
    case 'scan_coordinates':return 'Scan the requested coordinates';
    case 'search_location':return 'Find '+action.query;
    case 'select_prediction_model':return 'Use '+MODEL_LABELS[action.model];
    case 'set_time_horizon':return 'Show '+horizons[action.horizon];
    case 'set_map_layer':return 'Show '+LAYER_LABELS[action.layer].toLowerCase();
    case 'fly_to_cell':return 'Focus '+(action.target?targets[action.target]:'the requested cell');
    case 'inspect_cell':return 'Inspect '+(action.target?targets[action.target]:'the requested cell');
    case 'compare_cells':return 'Compare the requested cells';
  }
}

function outcomeLabel(type:string):string {
  return ({scan_coordinates:'Scan location',search_location:'Find location',select_prediction_model:'Select model',set_time_horizon:'Select time product',set_map_layer:'Select map layer',fly_to_cell:'Focus cell',inspect_cell:'Inspect cell',compare_cells:'Compare cells'} as Record<string,string>)[type]??'Workspace task';
}

export function assistantTasks(evidence:TaskEvidence):AssistantTask[] {
  const outcomes=new Map((evidence.outcomes??[]).filter(row=>Number.isInteger(row.step)&&row.step>0).map(row=>[row.step,row]));
  const plan=evidence.plannedActions??[];
  if(!plan.length)return [...outcomes.values()].sort((a,b)=>a.step-b.step).map(outcome=>({step:outcome.step,label:outcomeLabel(outcome.type),status:outcome.status,detail:outcome.summary}));
  const failedStep=Math.min(...[...outcomes.values()].filter(row=>row.status==='failed').map(row=>row.step));
  return plan.map((action,index)=>{
    const step=index+1,outcome=outcomes.get(step);
    if(outcome)return {step,label:actionTaskLabel(action),status:outcome.status,detail:outcome.summary};
    const cancelled=!!evidence.cancelled||!!evidence.executionError||step>failedStep;
    const status:TaskStatus=cancelled?'cancelled':evidence.executing&&evidence.activeTask===step?'running':'planned';
    return {step,label:actionTaskLabel(action),status,detail:status==='cancelled'?'Not completed':undefined};
  });
}

export function taskResultSummary(evidence:TaskEvidence):string|null {
  const tasks=assistantTasks(evidence);
  if(!tasks.length)return null;
  const completed=tasks.filter(task=>task.status==='completed').length;
  if(completed===tasks.length)return tasks.length===1?'Task completed':`${tasks.length} tasks completed`;
  const progress=`${completed} of ${tasks.length} completed`;
  if(tasks.some(task=>task.status==='failed'))return `Stopped after a failed task · ${progress}`;
  if(evidence.cancelled||evidence.executionError)return `Stopped · ${progress}`;
  return evidence.executing?progress:`Plan ready · ${progress}`;
}

export function assistantSuggestions(scan:Scan|null,model:PredictionModel,nasaProduct:NasaProduct='today'):AssistantSuggestion[] {
  if(!scan)return [
    {id:'coordinates',label:'Scan coordinates',description:'Explore any latitude and longitude',prompt:'Scan at '},
    {id:'place',label:'Find a place',description:'Search a town, mountain or region',prompt:'Find '},
  ];
  const suggestions:AssistantSuggestion[]=[];
  const cells=scanCells(scan),terraAvailable=cells.some(cell=>predictionScore(cell,'terra_physics_ml')!==null);
  const nasaAvailable=nasaProductAvailable(scan,nasaProduct);
  const activeAvailable=model==='nasa_lhasa_2'?nasaAvailable:cells.some(cell=>predictionScore(cell,model)!==null);
  if(activeAvailable){
    suggestions.push({id:'critical',label:model==='nasa_lhasa_2'?'Inspect highest NASA pixel':'Inspect critical cell',description:model==='nasa_lhasa_2'?`Use the official ${nasaProduct==='today'?'Today':'Tomorrow'} product`:'Review the active model’s hotspot',prompt:'Inspect the riskiest cell in the current model.'});
    if(terraAvailable&&nasaAvailable)suggestions.push({id:'compare',label:'Compare the models',description:'Keep the same area and source context',prompt:'Compare Terra In House Model with NASA LHASA for the selected cell.'});
  }else suggestions.push({id:'sources',label:'Review source coverage',description:'See which inputs are missing',prompt:'Explain which sources are available for this scan and what is missing.'});
  if(model!=='nasa_lhasa_2'&&terraAvailable&&scan.predictive_forecast_72h?.views?.some(view=>view.key==='t72'))suggestions.push({id:'forecast',label:'Explore the 72-hour forecast',description:'Inspect the forecast’s critical cell',prompt:'Switch to Terra In House Model, show the 72-hour forecast, and inspect the riskiest cell.'});
  suggestions.push({id:'radar',label:'Review NISAR evidence',description:'Check measured pixels and acquisition dates',prompt:'What measured NISAR evidence is available for this cell, and what is missing?'});
  return suggestions.slice(0,4);
}

/** Parse display structure only. Numeric claims and their wording stay intact. */
export function proseBlocks(markdown:string):ProseBlock[] {
  const lines=markdown.replace(/\r\n?/g,'\n').split('\n'),blocks:ProseBlock[]=[];
  const tableRow=(line:string)=>line.trim().replace(/^\|/,'').replace(/\|$/,'').split('|').map(cell=>cell.trim());
  let index=0;
  while(index<lines.length){
    const line=lines[index];
    if(!line.trim()){index++;continue;}
    if(/^\s*```/.test(line)){
      const content:string[]=[];index++;
      while(index<lines.length&&!/^\s*```/.test(lines[index]))content.push(lines[index++]);
      if(index<lines.length)index++;
      blocks.push({kind:'code',text:content.join('\n')});continue;
    }
    if(line.includes('|')&&index+1<lines.length&&/^\s*\|?\s*:?-{3,}:?\s*(?:\|\s*:?-{3,}:?\s*)+\|?\s*$/.test(lines[index+1])){
      const headers=tableRow(line),rows:string[][]=[];index+=2;
      while(index<lines.length&&lines[index].trim()&&lines[index].includes('|'))rows.push(tableRow(lines[index++]));
      blocks.push({kind:'table',headers,rows});continue;
    }
    const heading=line.match(/^\s{0,3}#{1,6}\s+(.+?)\s*#*$/);
    if(heading){blocks.push({kind:'heading',text:heading[1]});index++;continue;}
    const bullet=line.match(/^\s*(?:([-+*])|\d+[.)])\s+(.+)$/);
    if(bullet){
      const ordered=!bullet[1],items:string[]=[],numbers:number[]=[];
      while(index<lines.length){
        const item=lines[index].match(/^\s*(?:([-+*])|\d+[.)])\s+(.+)$/);
        if(!item||(!item[1])!==ordered)break;
        items.push(item[2]);
        if(ordered)numbers.push(Number(lines[index].match(/^\s*(\d+)[.)]\s+/)?.[1]));
        index++;
      }
      blocks.push({kind:'list',ordered,items,...(ordered?{numbers}:{})});continue;
    }
    const paragraph=[line.trim()];index++;
    while(index<lines.length&&lines[index].trim()&&!/^\s*(?:```|#{1,6}\s|[-+*]\s|\d+[.)]\s)/.test(lines[index])&&!(lines[index].includes('|')&&index+1<lines.length&&/^\s*\|?\s*:?-{3,}/.test(lines[index+1])))paragraph.push(lines[index++].trim());
    blocks.push({kind:'paragraph',text:paragraph.join(' ')});
  }
  return blocks;
}

export function safeAssistantLink(value:string):string|null {
  try{
    const url=new URL(value);
    return url.protocol==='https:'&&!url.username&&!url.password&&!/[\s\u0000-\u001f]/.test(value)?url.href:null;
  }catch{return null;}
}

export function inlineParts(text:string):InlinePart[] {
  const pattern=/(\*\*([^*\n]+)\*\*|`([^`\n]+)`|\*([^*\n]+)\*|\[([^\]\n]+)\]\(([^)\s]+)\))/g;
  const parts:InlinePart[]=[];let end=0;
  for(const match of text.matchAll(pattern)){
    if((match.index??0)>end)parts.push({kind:'text',text:text.slice(end,match.index)});
    if(match[2])parts.push({kind:'strong',text:match[2]});
    else if(match[3])parts.push({kind:'code',text:match[3]});
    else if(match[4])parts.push({kind:'emphasis',text:match[4]});
    else {const href=safeAssistantLink(match[6]);parts.push(href?{kind:'link',text:match[5],href}:{kind:'text',text:match[0]});}
    end=match.index!+match[0].length;
  }
  if(end<text.length)parts.push({kind:'text',text:text.slice(end)});
  return parts;
}
