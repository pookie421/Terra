import {ChevronDown,ExternalLink,LoaderCircle,RefreshCw} from 'lucide-react';
import {fmt,type Cell,type Scan} from '../types';
import {radarCoverage} from '../radarPresentation';
interface Props {scan:Scan;cell?:Cell;busy:boolean;onRefresh:()=>void;expanded?:boolean;onExpandedChange?:(expanded:boolean)=>void}
const dates=(value?:string|null)=>value&&!Number.isNaN(Date.parse(value))?new Date(value).toLocaleString('en-GB',{timeZone:'UTC'})+' UTC':'Not reported';
function link(value?:string){try{const url=new URL(value??'');return url.protocol==='https:'&&!url.username&&!url.password?url.href:undefined;}catch{return;}}
const STATUS_LABELS:Record<string,string>={AUTH_REQUIRED:'Earthdata access required',SUBSET_PENDING:'Preparing measured subset',MEASURED_AVAILABLE:'Measured raster available',NO_PRODUCT_COVERAGE:'No compatible product covers this area',NO_VALID_PIXELS:'No valid measured pixels returned',SUBSET_UNAVAILABLE:'Measurement access unavailable'};
function dateOnly(value:unknown){if(typeof value!=='string')return 'Unknown date';const date=new Date(value);return Number.isNaN(date.valueOf())?'Unknown date':date.toLocaleDateString('en-GB',{timeZone:'UTC',day:'2-digit',month:'short',year:'numeric'});}
function record(value:unknown):Record<string,unknown>{return value&&typeof value==='object'&&!Array.isArray(value)?value as Record<string,unknown>:{};}
function interval(start:unknown,end:unknown){return start?`${dateOnly(start)} — ${dateOnly(end)}`:dateOnly(end);}
export default function RadarPanel({scan,cell,busy,onRefresh,expanded,onExpandedChange}:Props){
  const radar=scan.nisar,access=radar?.measurement_access,coverage=radarCoverage(scan,cell?.h3_index);
  const raw=cell?record(record(radar?.by_h3)[cell.h3_index]??radar?.cells?.find(row=>row.h3_index===cell.h3_index)):{};
  const selected:Record<string,unknown>={...cell,...raw};
  const measurements=[
    {observable:'coherence',label:'L-band coherence',value:selected.nisar_coherence,unit:''},
    {observable:'los_velocity',label:'LOS velocity · measured series',value:selected.nisar_los_velocity_mm_yr,unit:'mm/yr'},
    {observable:'polarimetric_ratio',label:'Polarimetric power ratio',value:selected.nisar_polarimetric_moisture_index,unit:''},
    {observable:'los_displacement',label:'LOS displacement',value:selected.nisar_los_displacement_mm,unit:'mm'},
  ];
  const measured=measurements.filter(({observable,value})=>typeof value==='number'&&Number.isFinite(value)&&
    (observable!=='coherence'||value>=0&&value<=1)&&(observable!=='polarimetric_ratio'||value>=0));
  const provenance=Object.prototype.hasOwnProperty.call(selected,'measurement_provenance')?record(selected.measurement_provenance):record(selected.nisar_measurement_provenance);
  const pending=access?.status==='SUBSET_PENDING'||coverage.pendingJobs>0;
  const summaryLabel=coverage.status==='measured'?'Measured':coverage.status==='pending'?'Processing':coverage.status==='authentication'?'Access needed':coverage.measuredCells===null&&access?.status==='MEASURED_AVAILABLE'?'Raster available':'No samples';
  return <details className="panel-section disclosure radar-section" open={expanded} onToggle={event=>{if(event.target===event.currentTarget)onExpandedChange?.(event.currentTarget.open);}}><summary>NISAR radar<span className="radar-coverage-summary">{summaryLabel}</span><ChevronDown size={15}/></summary>
    <div className={`radar-access ${coverage.status==='measured'?'ready':''}`} role="status">{coverage.status==='pending'&&<LoaderCircle size={14} className="spin"/>}<span>{coverage.summary}</span></div>
    <p className="support-note">Coverage counts terrain cells with valid radar samples.</p>
    <p className="support-note radar-selected-status">{coverage.selectedMessage}</p>
    {!!measured.length&&<dl className="telemetry">{measured.map(({label,value,unit})=><div key={label}><dt>{label}</dt><dd>{fmt(value,3)}{unit?' '+unit:''}</dd></div>)}</dl>}
    {measured.map(({observable})=>{const source=record(provenance[observable]);if(!Object.keys(source).length)return null;const support=record(source.support),pixels=Array.isArray(support.pixel_size_native_units)?support.pixel_size_native_units.filter((v):v is number=>typeof v==='number'&&Number.isFinite(v)&&v!==0):[],metres=/EPSG:32[67]\d{2}/.test(String(support.crs));return <div className="measured-source-brief" key={observable}><strong>{observable==='coherence'?'Coherence acquisition pair':observable.replaceAll('_',' ')+' acquisition'}</strong><span>{interval(source.time_start,source.time_end??source.acquisition_utc)}</span>{pixels.length===2&&<small>{pixels.map(v=>fmt(Math.abs(v),1)).join(' × ')} {metres?'m':'native CRS units'} source posting</small>}</div>;})}
    {!measured.length&&coverage.status==='measured'&&<div className="measured-source-brief"><strong>Measurements elsewhere in this scan</strong>{coverage.measurementIntervals.map((source,index)=><span key={index}>{interval(source.start,source.end)}</span>)}{coverage.nativePostings.length>0&&<small>{coverage.nativePostings.join('; ')} source posting</small>}</div>}
    {pending&&<p className="support-note">NASA is processing {coverage.status==='measured'?'additional':'bounded'} subsets. Check again to add returned measurements to a new analysis state.</p>}
    {coverage.canRefresh&&<button className="hud-button" disabled={busy} onClick={onRefresh}>{busy?<LoaderCircle size={14} className="spin"/>:<RefreshCw size={14}/>} {busy?'Checking NASA subsets…':pending?'Check for measurements':'Refresh radar measurements'}</button>}
    {access?.status==='AUTH_REQUIRED'&&<p className="support-note">The deployment owner must configure Earthdata access before this server can retrieve protected NISAR measurements. Your AI key does not grant Earthdata access.</p>}
    <details className="nested-details"><summary>How radar contributes</summary><p className="support-note">Valid radar samples enter the source and assistant context. The current classifier has not been refitted with SAR features. Only a separately documented site calibration can constrain physical inputs; no generic radar-to-strength conversion is applied.</p><p className="support-note">Coherence is not a deformation rate. A single interferogram does not establish annual motion or soil strength, and a polarimetric power ratio is not calibrated soil moisture.</p></details>
    <details className="nested-details"><summary>Coverage and source details</summary><dl className="telemetry radar-coverage-counts"><div><dt>Cells with coherence</dt><dd>{coverage.byObservable.coherence}</dd></div><div><dt>Cells with LOS velocity</dt><dd>{coverage.byObservable.losVelocity}</dd></div><div><dt>Cells with polarimetric ratio</dt><dd>{coverage.byObservable.polarimetricRatio}</dd></div><div><dt>Cells with LOS displacement</dt><dd>{coverage.byObservable.losDisplacement}</dd></div>{coverage.oldestMeasurementAgeDays!==null&&<div><dt>Oldest measured acquisition</dt><dd>{fmt(coverage.oldestMeasurementAgeDays,1)} days before scan</dd></div>}<div><dt>Discovered granules</dt><dd>{cell?.nisar_granule_count??radar?.granule_count??'Not reported'}</dd></div><div><dt>Latest discovered acquisition</dt><dd>{dates(cell?.nisar_latest_acquisition_utc??radar?.latest_acquisition_utc)}</dd></div></dl>
      <p className="support-note">{access?.message??radar?.reason??STATUS_LABELS[access?.status??'']??'Acquisition metadata alone does not provide a radar measurement.'}</p>
      {access?.jobs?.map(job=><div className="granule-row" key={job.job_id??job.granule_id}><span>{job.product_type??'NISAR'} · {(job.status??'Pending').toLowerCase()}</span><small>{job.granule_id??job.collection_short_name}<br/>{dates(job.acquisition_utc)}</small></div>)}
      {!!Object.keys(provenance).length&&<details className="nested-details"><summary>Selected-cell raster provenance</summary><pre>{JSON.stringify(provenance,null,2)}</pre></details>}
      {radar?.granules?.map((granule,index)=><div className="granule-row" key={granule.granule_id??index}><span>{granule.granule_id??'Granule'}</span><small>{dates(granule.time_start)} · {granule.orbit_direction??'Orbit not declared'}</small>{link(granule.browse_url)&&<a href={link(granule.browse_url)} target="_blank" rel="noreferrer">Browse acquisition<ExternalLink size={12}/></a>}</div>)}
      {!radar?.granules?.length&&<p className="support-note">No acquisition metadata was returned.</p>}
    </details>
  </details>;
}
