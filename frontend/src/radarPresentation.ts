import type {Scan} from './types.js';

type Row=Record<string,unknown>;
export type RadarCoverageStatus='measured'|'pending'|'authentication'|'empty'|'metadata';
export interface RadarCoverage {
  status:RadarCoverageStatus;
  summary:string;
  totalCells:number|null;
  measuredCells:number|null;
  byObservable:{coherence:number;losVelocity:number;polarimetricRatio:number;losDisplacement:number};
  selectedMeasured:boolean|null;
  selectedMessage:string;
  measuredAcquisitionDates:string[];
  measurementIntervals:Array<{start:string|null;end:string|null}>;
  oldestMeasurementAgeDays:number|null;
  nativePostings:string[];
  pendingJobs:number;
  canRefresh:boolean;
}

const OBSERVABLES=[
  {key:'coherence',field:'nisar_coherence',label:'coherence'},
  {key:'losVelocity',field:'nisar_los_velocity_mm_yr',label:'LOS velocity'},
  {key:'polarimetricRatio',field:'nisar_polarimetric_moisture_index',label:'polarimetric ratio'},
  {key:'losDisplacement',field:'nisar_los_displacement_mm',label:'LOS displacement'},
] as const;
const PROVENANCE_KEYS={coherence:'coherence',losVelocity:'los_velocity',polarimetricRatio:'polarimetric_ratio',losDisplacement:'los_displacement'};
const record=(value:unknown):Row=>value&&typeof value==='object'&&!Array.isArray(value)?value as Row:{};
const rows=(value:unknown):Row[]=>Array.isArray(value)?value.map(record):[];
const identity=(row:Row)=>typeof row.h3_index==='string'&&row.h3_index?row.h3_index:null;
function validValue(value:unknown,key:typeof OBSERVABLES[number]['key']):value is number{
  return typeof value==='number'&&Number.isFinite(value)&&(key!=='coherence'||value>=0&&value<=1)&&(key!=='polarimetricRatio'||value>=0);
}
function date(value:unknown):string|null{
  if(typeof value!=='string'||!value.trim())return null;
  const parsed=Date.parse(value);
  return Number.isFinite(parsed)?new Date(parsed).toISOString():null;
}
function posting(source:Row):string|null{
  const support=record(source.support),pixels=support.pixel_size_native_units;
  if(!Array.isArray(pixels)||pixels.length!==2||!pixels.every(value=>typeof value==='number'&&Number.isFinite(value)&&value!==0))return null;
  const crs=typeof support.crs==='string'?support.crs:'';
  const units=/EPSG:32[67]\d{2}(?:\D|$)/i.test(crs)?'m':crs?`native units (${crs})`:'native units';
  return `${pixels.map(value=>Number(Math.abs(value).toFixed(3))).join(' × ')} ${units}`;
}

/** Valid H3 samples across this scan, not radar footprint area or native-pixel counts. */
export function radarCoverage(scan:Scan|null,selectedCellId?:string):RadarCoverage{
  const radar=scan?.nisar,access=radar?.measurement_access,byId=new Map<string,Row>();
  const merge=(row:Row)=>{const id=identity(row);if(id)byId.set(id,{...byId.get(id),...row});};
  for(const feature of scan?.h3_geojson?.features??[])merge(record(feature.properties));
  for(const feature of scan?.terrain_geojson?.features??[])merge(record(feature.properties));
  for(const row of scan?.physical_assessment?.cells??[])merge(record(row));
  for(const row of scan?.physics_informed_assessment?.cells??[])merge(record(row));
  const terrainIds=new Set(byId.keys());
  const rawRows=[...rows(radar?.cells),...Object.values(record(radar?.by_h3)).map(record)];
  // A returned masked radar sample is authoritative, including its null values.
  for(const row of rawRows){const id=identity(row);if(id&&(!terrainIds.size||terrainIds.has(id)))merge(row);}
  const knownSamples=byId.size>0,totalCells=knownSamples?byId.size:null;
  const byObservable={coherence:0,losVelocity:0,polarimetricRatio:0,losDisplacement:0};
  const measuredIds=new Set<string>(),selectedLabels:string[]=[],acquisitions=new Set<string>(),postings=new Set<string>();
  const intervals=new Map<string,{start:string|null;end:string|null}>(),ages:number[]=[];
  const generated=date(scan?.generated_at_utc);let unknownAge=false;
  for(const [id,row] of byId){
    const provenance=Object.prototype.hasOwnProperty.call(row,'measurement_provenance')?
      record(row.measurement_provenance):record(row.nisar_measurement_provenance);
    for(const observable of OBSERVABLES){
      if(!validValue(row[observable.field],observable.key))continue;
      byObservable[observable.key]++;measuredIds.add(id);
      if(id===selectedCellId)selectedLabels.push(observable.label);
      const source=record(provenance[PROVENANCE_KEYS[observable.key]]);
      const start=date(source.time_start),end=date(source.time_end??source.acquisition_utc);
      if(start)acquisitions.add(start);if(end)acquisitions.add(end);
      if(start||end)intervals.set(`${start??''}/${end??''}`,{start,end});
      if(end&&generated&&Date.parse(end)<=Date.parse(generated))ages.push((Date.parse(generated)-Date.parse(end))/86400000);
      else unknownAge=true;
      const support=posting(source);if(support)postings.add(support);
    }
  }
  const measuredCells=knownSamples?measuredIds.size:null;
  const jobs=access?.jobs??[];
  const pendingJobs=jobs.filter(job=>/pending|running|accepted|queued|started|processing/i.test(job.status??job.provider_status??'')).length;
  const pending=access?.status==='SUBSET_PENDING'||pendingJobs>0;
  const status:RadarCoverageStatus=measuredIds.size?'measured':access?.status==='AUTH_REQUIRED'?'authentication':pending?'pending':
    access?.status==='NO_VALID_PIXELS'||access?.status==='NO_PRODUCT_COVERAGE'||access?.status==='MEASURED_AVAILABLE'&&knownSamples?'empty':'metadata';
  const summary=status==='measured'?`${measuredCells} / ${totalCells} cells with valid samples`:
    status==='authentication'?'Earthdata access required':status==='pending'?'NASA subset processing':
    status==='empty'?'No valid samples in this scan':access?.status==='MEASURED_AVAILABLE'?'Measured raster · cell coverage not reported':'No measured samples returned';
  const selectedKnown=!!selectedCellId&&byId.has(selectedCellId);
  const selectedMeasured=selectedKnown?measuredIds.has(selectedCellId!):null;
  const selectedMessage=selectedMeasured?`Selected cell: measured ${selectedLabels.join(', ')}`:
    selectedMeasured===false&&measuredIds.size?'Selected cell has no valid radar pixel; other cells have measurements.':
    selectedMeasured===false?'Selected cell has no valid radar sample.':'Select a terrain cell to inspect radar measurements.';
  return {status,summary,totalCells,measuredCells,byObservable,selectedMeasured,selectedMessage,
    measuredAcquisitionDates:[...acquisitions].sort(),measurementIntervals:[...intervals.values()],
    oldestMeasurementAgeDays:ages.length&&!unknownAge?Math.max(...ages):null,nativePostings:[...postings].sort(),
    pendingJobs,canRefresh:!!scan&&access?.status!=='AUTH_REQUIRED'};
}
