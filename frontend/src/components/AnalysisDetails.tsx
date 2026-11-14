import {ChevronDown} from 'lucide-react';
import {fmt,type Cell,type Health,type Scan} from '../types';
import {nasaResolution} from '../modelPresentation';
interface Props {scan:Scan;nasa:boolean;minimum?:Cell;health?:Health}
const range=(values:number[])=>{const valid=values.filter(Number.isFinite);return valid.length?`${fmt(Math.min(...valid),0)} — ${fmt(Math.max(...valid),0)}`:'Not reported';};
export default function AnalysisDetails({scan,nasa,minimum,health}:Props){
  const support=scan.spatial_support,source=scan.nasa_lhasa,product=source?.active_product??'today';
  const area=support?.h3_mean_cell_area_m2;
  return <details className="panel-section disclosure analysis-details"><summary>Analysis details<ChevronDown size={15}/></summary>
    <dl className="telemetry"><div><dt>Scan</dt><dd>{scan.scan_id}</dd></div><div><dt>State</dt><dd>{scan.scenario_id}</dd></div>
      {nasa?<><div><dt>Daily product</dt><dd>{product==='today'?'Today':'Tomorrow'}</dd></div><div><dt>Official sampled pixels</dt><dd>{source?.native_pixels_geojson_by_product?.[product]?.features.length??'Not reported'}</dd></div></>:<><div><dt>Minimum FS · low / central / high</dt><dd>{[minimum?.fs_low,minimum?.fs_mid,minimum?.fs_high].map(value=>fmt(value)).join(' / ')}</dd></div><div><dt>Minimum FS cell</dt><dd>{minimum?.h3_index}</dd></div><div><dt>Elevation range</dt><dd>{range(scan.physical_assessment.cells.map(cell=>cell.elevation_m))} m</dd></div><div><dt>Slope range</dt><dd>{range(scan.physical_assessment.cells.map(cell=>cell.slope_deg))}°</dd></div><div><dt>Terra model</dt><dd>{health?.model_version??'Not reported'}</dd></div></>}
      <div><dt>{nasa?'Terra context cells':'H3 display cells'}</dt><dd>{scan.physical_assessment.cells.length}</dd></div>
    </dl>
    <details className="nested-details"><summary>Resolution and scope</summary>
      {nasa?<><p className="support-note">{nasaResolution(scan)}. East–west pixel width varies with latitude. Source pixel footprints are displayed when service grid metadata is available; H3 terrain context does not increase NASA prediction resolution.</p><dl className="telemetry"><div><dt>NASA product version</dt><dd>{source?.product_version?.startsWith('NOT_')?'Not declared by service':source?.product_version??'Not declared by service'}</dd></div><div><dt>NASA valid time</dt><dd>{source?.product_valid_time_utc??'Not declared by service'}</dd></div></dl></>:<><p className="support-note">{support?.dem_nominal_pixel_size_m?`Nominal terrain tile spacing: ${support.dem_nominal_pixel_size_m} m; native elevation support varies. `:'Native elevation support varies. '}{area?`H3 cells average ${(area/1e6).toLocaleString('en-US',{maximumFractionDigits:3})} km². `:'H3 resolution-9 cells span hundreds of metres. '}Display cells, terrain pixels and coarser environmental inputs have different support.</p><p className="support-note">Reference columns and conditional routing remain available without boreholes. Finer display detail does not establish greater predictive skill.</p></>}
    </details>
    <p className="support-note">Research estimates, not operational warnings.</p>
  </details>;
}
