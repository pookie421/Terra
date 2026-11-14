import {ChevronDown,ExternalLink,LoaderCircle,RefreshCw} from 'lucide-react';
import {fmt,type Cell,type Scan} from '../types';
interface Props {scan:Scan;cell?:Cell;busy:boolean;onRefresh:()=>void}
const dates=(value?:string|null)=>value?new Date(value).toLocaleString('en-GB',{timeZone:'UTC'})+' UTC':'Not reported';
function link(value?:string){try{const url=new URL(value??'');return url.protocol==='https:'&&!url.username&&!url.password?url.href:undefined;}catch{return;}}
const STATUS_LABELS:Record<string,string>={AUTH_REQUIRED:'Earthdata access required',SUBSET_PENDING:'Preparing measured subset',MEASURED_AVAILABLE:'Measured raster available',NO_PRODUCT_COVERAGE:'No compatible product covers this area',NO_VALID_PIXELS:'No valid measured pixels returned',SUBSET_UNAVAILABLE:'Measurement access unavailable'};
function dateOnly(value:unknown){if(typeof value!=='string')return 'Unknown date';const date=new Date(value);return Number.isNaN(date.valueOf())?'Unknown date':date.toLocaleDateString('en-GB',{timeZone:'UTC',day:'2-digit',month:'short',year:'numeric'});}
function record(value:unknown):Record<string,unknown>{return value&&typeof value==='object'&&!Array.isArray(value)?value as Record<string,unknown>:{};}
export default function RadarPanel({scan,cell,busy,onRefresh}:Props){
  const radar=scan.nisar,access=radar?.measurement_access;
  const measurements=[['L-band coherence',cell?.nisar_coherence,''],['LOS velocity · measured series',cell?.nisar_los_velocity_mm_yr,'mm/yr'],['Polarimetric power ratio',cell?.nisar_polarimetric_moisture_index,'']] as const;
  const measured=measurements.filter(([,value])=>typeof value==='number'&&Number.isFinite(value));
  return <details className="panel-section disclosure radar-section"><summary>NISAR measurements<ChevronDown size={15}/></summary>
    <div className={`radar-access ${access?.status==='MEASURED_AVAILABLE'?'ready':''}`} role="status">{access?.status==='SUBSET_PENDING'&&<LoaderCircle size={14} className="spin"/>}<span>{STATUS_LABELS[access?.status??'']??(measured.length?'Measured raster available':'No measured raster returned')}</span></div>
    <p className="support-note">{access?.message??radar?.reason??'Acquisition metadata alone does not provide a radar measurement.'}</p>
    {!!measured.length&&<dl className="telemetry">{measured.map(([label,value,unit])=><div key={label}><dt>{label}</dt><dd>{fmt(value,3)}{unit?' '+unit:''}</dd></div>)}</dl>}
    {Object.entries(cell?.nisar_measurement_provenance??{}).map(([observable,value])=>{const source=record(value),support=record(source.support),pixels=Array.isArray(support.pixel_size_native_units)?support.pixel_size_native_units.filter((v):v is number=>typeof v==='number'&&Number.isFinite(v)):[],metres=/EPSG:32[67]\d{2}/.test(String(support.crs));return <div className="measured-source-brief" key={observable}><strong>{observable==='coherence'?'Coherence acquisition pair':observable.replaceAll('_',' ')+' acquisition'}</strong><span>{dateOnly(source.time_start)} — {dateOnly(source.time_end??source.acquisition_utc)}</span>{pixels.length===2&&<small>{pixels.map(v=>fmt(Math.abs(v),1)).join(' × ')} {metres?'m':'native CRS units'} source posting</small>}</div>;})}
    {!measured.length&&access?.status==='MEASURED_AVAILABLE'&&<p className="support-note">This sector has measured data; no valid value was returned at the selected cell.</p>}
    {access?.status==='SUBSET_PENDING'&&<p className="support-note">NASA is processing a bounded area. Check again to incorporate returned measurements into a new analysis state.</p>}
    {access?.status!=='AUTH_REQUIRED'&&<button className="hud-button" disabled={busy} onClick={onRefresh}><RefreshCw size={14}/>{access?.status==='SUBSET_PENDING'?'Check for measurements':'Refresh radar measurements'}</button>}
    {access?.status==='AUTH_REQUIRED'&&<p className="support-note">The deployment owner must configure Earthdata access before this server can retrieve protected NISAR measurements. Your AI key does not grant Earthdata access.</p>}
    <p className="support-note">Coherence is not a deformation rate. A single interferogram does not establish annual motion or soil strength.</p>
    <details className="nested-details"><summary>Acquisition and measurement sources</summary><dl className="telemetry"><div><dt>Discovered granules</dt><dd>{cell?.nisar_granule_count??radar?.granule_count??'Not reported'}</dd></div><div><dt>Latest discovered acquisition</dt><dd>{dates(cell?.nisar_latest_acquisition_utc??radar?.latest_acquisition_utc)}</dd></div></dl>
      {access?.jobs?.map(job=><div className="granule-row" key={job.job_id??job.granule_id}><span>{job.product_type??'NISAR'} · {(job.status??'Pending').toLowerCase()}</span><small>{job.granule_id??job.collection_short_name}<br/>{dates(job.acquisition_utc)}</small></div>)}
      {cell?.nisar_measurement_provenance&&<details className="nested-details"><summary>Measured raster provenance</summary><pre>{JSON.stringify(cell.nisar_measurement_provenance,null,2)}</pre></details>}
      {radar?.granules?.map((granule,index)=><div className="granule-row" key={granule.granule_id??index}><span>{granule.granule_id??'Granule'}</span><small>{dates(granule.time_start)} · {granule.orbit_direction??'Orbit not declared'}</small>{link(granule.browse_url)&&<a href={link(granule.browse_url)} target="_blank" rel="noreferrer">Browse acquisition<ExternalLink size={12}/></a>}</div>)}
      {!radar?.granules?.length&&<p className="support-note">No acquisition metadata was returned.</p>}
    </details>
  </details>;
}
