"""Physics refresh, sourced report-point labels, and declared simulated negatives."""
import asyncio
from collections import Counter
from dataclasses import asdict
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import httpx
import h3

REPORTS = 'https://gis.earthdata.nasa.gov/portal/rest/services/Landslides/COOLR_Reports_Points/FeatureServer/0/query'

def feature_row(inputs, state):
    return dict(fs_static_prior=float(state.fs_static[0]), fs_dynamic=float(state.fs_dynamic[0]), p_f_fosm=float(state.p_failure_fosm[0]), p_failure_physical=float(state.p_failure_physical[0]),
        psi_head_m=float(state.pore_pressure_head_m[0]), newmark_dn_cm=float(state.newmark_displacement_cm[0]),
        slope_max_deg=float(inputs.slope_deg), twi=float(inputs.twi), soil_saturation=float(state.local_saturation[0]),
        rain_24h_mm=float(state.equivalent_water_mm[0]), snowmelt_mm_day=float(state.snowmelt_mm_day[0]), pga_g=float(inputs.pga_g),
        thaw_degradation=float(1-state.thaw_factor[0]), burn_severity=float(state.effective_burn_severity[0]),
        toe_undercut_mult=float(state.toe_undercut_multiplier[0]), profile_curvature=float(inputs.profile_curvature))

def mechanical_controls():
    from backend.physics.multi_causal_engine import PhysicsInputs, MultiCausalEngine
    rows=[]
    for kind,count in [('high_rain_flat',25),('dry_steep',15)]:
        for i in range(count):
            inputs=PhysicsInputs(sand_fraction=0.30,clay_fraction=0.40,organic_matter_fraction=0.035,
                bulk_density_g_cm3=1.40,slope_deg=(i%24)*7.5/23 if kind=='high_rain_flat' else 30.5+(i%20)*0.35,
                twi=6.0,soil_saturation_fraction=0.55 if kind=='high_rain_flat' else 0.15,
                rain_24h_mm=100+250*i/24 if kind=='high_rain_flat' else 0.0,
                storm_duration_hr=6+(i%4)*6,snow_depth_m=0.0,air_temperature_c=15.0,soil_temperature_c=15.0,
                freeze_thaw_cycles_7d=0.0,burn_severity=0.0,ndvi_norm=0.7+(i%5)*0.05,pga_g=0.0,
                road_cut_distance_m=9999.0,river_distance_m=9999.0)
            state=MultiCausalEngine().compute(inputs)
            assert state.fs_dynamic[0] > (1.45 if kind=='high_rain_flat' else 1.30)
            row=feature_row(inputs,state)
            row.update(event_id='synthetic:{}:{}'.format(kind,i),name=kind,event_record_type='simulated_mechanical_negative',
                event_date=None,label=0,regime='stable_mechanical_control',source_name='User-authorized physics simulation',
                source_event_id=None,event_group_id='synthetic:'+kind,region_block='synthetic:'+kind,
                label_quality='simulated_mechanical_negative_not_observed',label_source='Engine stability filter; no observation',
                label_note='Training augmentation only; excluded from held-out calibration and observational evaluation.',
                is_synthetic=True,physics_inputs_json=json.dumps(asdict(inputs)),physics_feature_quality='simulated',
                physics_feature_provenance_json=json.dumps({'source':'explicit deterministic simulation','inputs':asdict(inputs)}))
            rows.append(row)
    return rows

async def audit_refresh(path):
    import pyarrow as pa
    import pyarrow.parquet as pq
    from backend.ml.build_real_labeled_dataset import EventCandidate,parse_date,enrich_one,populate_physics_features,_write_event_group_holdout
    from backend.realtime.dem_processor import DEMProcessor
    rows=[r for r in pq.read_table(path).to_pylist() if r.get('event_record_type')!='simulated_mechanical_negative']
    by_id={r['event_id']:r for r in rows}
    rows=[r for r in rows if r.get('event_record_type') not in ('reported_landslide_point_proxy','audit_mapped_polygon_cell')]
    snapshots=path.parent/'source_catalogs'/'audit_regime_reports.json'
    queries={'coseismic':"landslide_trigger='earthquake'",'cryosphere':"landslide_trigger IN ('snowfall_snowmelt','freeze_thaw')",
        'anthropogenic_cut_mining':"landslide_trigger='construction' AND landslide_setting IN ('above_road','engineered_slope')",
        'post_wildfire':"landslide_setting='burned_area'"}
    async with httpx.AsyncClient(timeout=30,follow_redirects=True) as client:
        if snapshots.exists():
            catalog=json.loads(snapshots.read_text())
        else:
            catalog={}
            for regime,query in queries.items():
                params={'f':'geojson','where':"("+query+") AND event_date IS NOT NULL AND location_accuracy='exact' AND source_link IS NOT NULL",
                    'outFields':'*','returnGeometry':'true','orderByFields':'event_date DESC','resultRecordCount':12}
                response=await client.get(REPORTS,params=params);response.raise_for_status()
                payload=response.json()
                if 'error' in payload: raise RuntimeError(payload['error'])
                catalog[regime]={'query':params,'response':payload,'source_url':str(response.url)}
            snapshots.parent.mkdir(exist_ok=True);snapshots.write_text(json.dumps(catalog,indent=2))
        additions=[];raw_by_id={}; reused_reports=[]; seen_reports=set()
        for regime,entry in catalog.items():
            for feature in entry['response'].get('features',[]):
                p=feature['properties'];day=parse_date(str(p['event_date']));lat=p['latitude'];lon=p['longitude']
                if not day or lat is None or lon is None:continue
                if p.get('landslide_category')=='snow_avalanche': continue
                source_block=h3.latlng_to_cell(lat,lon,6)
                dedup_key=(regime,day[:7],source_block)
                if dedup_key in seen_reports: continue
                seen_reports.add(dedup_key)
                eid='nasa_coolr_report:'+str(p['objectid'])
                raw_by_id[eid]=p
                if eid in by_id:
                    cached=dict(by_id[eid]);cached['event_group_id']='coolr_report_group:'+day[:7]+':'+source_block
                    reused_reports.append(cached);continue
                additions.append(EventCandidate(event_id=eid,name=p.get('event_title') or eid,event_date=day,
                    latitude=lat,longitude=lon,regime=regime,source_name='NASA COOLR reported landslide point',
                    source_url=REPORTS+'?f=pjson&objectIds='+str(p['objectid'])+'&outFields=*',
                    event_record_type='reported_landslide_point_proxy',label=1,
                    label_source='NASA COOLR report; source trigger/setting and exact-reported location',
                    label_quality='reported_point_location_proxy_not_mapped_crown',citation=str(p['source_link']),
                    source_event_id=eid,event_group_id='coolr_report_group:'+day[:7]+':'+source_block,
                    h3_cell_id=h3.latlng_to_cell(lat,lon,9)))
        from backend.ml.mapped_audit_inventory import mapped_candidates
        mapped, mapped_summary = mapped_candidates(path.parent/'source_catalogs'/'audit_mapped_polygons.json')
        for event in mapped:
            if event.event_id in by_id: reused_reports.append(dict(by_id[event.event_id]))
            else: additions.append(event)
        sem=asyncio.Semaphore(8);dem=DEMProcessor(zoom=13)
        enriched=await asyncio.gather(*(enrich_one(e,client,dem,sem) for e in additions))
        rows.extend(reused_reports+enriched)
    # Add real-date weather controls for report points, without inferring trigger intensity.
    from backend.ml.build_real_labeled_dataset import enrich_temporal_control
    from datetime import date, timedelta
    existing={r['event_id'] for r in rows}
    controls_to_fetch=[]
    for row in rows:
        if row.get('event_record_type')!='reported_landslide_point_proxy' or row.get('label')!=1:continue
        eid=row['event_id']+':tminus60'
        if eid in existing:continue
        event=EventCandidate(event_id=eid,name=row['name']+' (pre-event control)',
            event_date=(date.fromisoformat(row['event_date'])-timedelta(days=60)).isoformat(),
            latitude=row['latitude'],longitude=row['longitude'],regime=row['regime'],
            source_name='NASA COOLR event-cell inventory-absence sample',source_url=row.get('source_url'),
            event_record_type='temporal_absence_sample',label=0,
            label_source='Same cell 60 days earlier; inventory absence, not surveyed intact terrain',
            label_quality='temporal_inventory_absence_pseudo_negative_unverified',citation=row.get('citation'),
            source_event_id=row.get('source_event_id'),event_group_id=row.get('event_group_id'),h3_cell_id=row['h3_cell_id'])
        controls_to_fetch.append((event,row))
    async with httpx.AsyncClient(timeout=25) as client:
        sem=asyncio.Semaphore(8)
        rows.extend(await asyncio.gather(*(enrich_temporal_control(e,r,client,sem) for e,r in controls_to_fetch)))
    for row in rows:
        if row.get('regime')=='trigger_unclassified' and row.get('event_date') and row.get('citation'):
            if row.get('event_record_type') in ('cited_manual_point_buffer_cell','manually_mapped_event_polygon_cell'):
                row['label']=1
            elif row.get('event_record_type')=='temporal_absence_sample':row['label']=0
            row['training_exclusion_reason']=None
        row.pop('neighborhood_coverage',None);row.pop('neighborhood_source_cell',None)
        if row.get('event_id') in raw_by_id:
            row['source_record_json']=json.dumps(raw_by_id[row['event_id']],sort_keys=True)
            row['label_note']='Reported landslide at a catalog location marked exact; 300m H3 proxy, no mapped crown or measured trigger intensity.'
        # Recompute every historical feature vector under current equations from retained observations.
        cell=SimpleNamespace(cell_id=row.get('h3_cell_id'),latitude=row.get('latitude'),longitude=row.get('longitude'),
            slope_deg=float(row.get('terrain_slope_deg') if row.get('terrain_slope_deg') is not None else row.get('slope_max_deg',15)),
            twi=float(row.get('terrain_twi') or row.get('twi') or 6), profile_curvature=float(row.get('terrain_profile_curvature') or 0),
            elevation_m=float(row.get('terrain_elevation_m') or 0))
        original_status=row.get('terrain_source_status')
        populate_physics_features(row,SimpleNamespace(cells={str(cell.cell_id):cell}))
        row['terrain_source_status']=original_status
        row['is_synthetic']=False
        row['physics_model_version']='boundary_conformal_v5'
        row['physics_feature_provenance_json']=json.dumps(row.get('physics_feature_provenance') or {},sort_keys=True)
    controls=mechanical_controls();rows.extend(controls)
    # Explicit union of keys prevents Arrow dropping columns absent from the first seed row.
    keys=sorted(set().union(*(r.keys() for r in rows)))
    normalized=[{k:r.get(k) for k in keys} for r in rows]
    pq.write_table(pa.Table.from_pylist(normalized),path,compression='zstd')
    pq.write_table(pa.Table.from_pylist(controls),path.with_name('mechanical_training_controls.parquet'),compression='zstd')
    summary_path=path.with_suffix('.metadata.json');summary=json.loads(summary_path.read_text()) if summary_path.exists() else {}
    positives=[r for r in rows if r.get('label')==1]
    summary.update(mapped_footprint_expansion=mapped_summary, row_count=len(rows),labeled_positive_rows=len(positives),labeled_negative_rows=sum(r.get('label')==0 for r in rows),
        positive_regimes=sorted({r['regime'] for r in positives}),positive_regime_counts=dict(Counter(r['regime'] for r in positives)),
        synthetic_negative_rows=len(controls),synthetic_control_counts=dict(Counter(r['name'] for r in controls)),
        observed_negative_rows=sum(r.get('label')==0 and not r.get('is_synthetic') for r in rows),
        catalog_candidate_count=sum(r.get('event_record_type') not in ('simulated_mechanical_negative','temporal_absence_sample') for r in rows),
        complete_14_feature_rows=len(rows),source_report_rows=len(raw_by_id),physics_model_version='boundary_conformal_v5',
        label_policy='Inventory polygons, manual point buffers and precisely located reported-landslide point proxies are positive. Temporal absences are unverified pseudo-negatives. 40 explicitly simulated negatives augment training only.',
        warning='Five source trigger/context regimes are represented but sample sizes are small and point locations are not mapped crowns. Historic burn severity, freeze-thaw cycles and construction geometry are not inferred from positive labels; missing inputs remain flagged defaults. Synthetic negatives never enter held-out calibration/evaluation.')
    summary.update(
        distinct_positive_source_events=len({r.get('source_event_id') for r in positives if r.get('source_event_id')}),
        positive_event_dates=sorted({r.get('event_date') for r in positives if r.get('event_date')}),
        positive_geographic_extent={
            'latitude_min':min(r['latitude'] for r in positives), 'latitude_max':max(r['latitude'] for r in positives),
            'longitude_min':min(r['longitude'] for r in positives), 'longitude_max':max(r['longitude'] for r in positives)},
        event_group_split_audit={
            'positive_source_groups':len({r.get('event_group_id') for r in positives}),
            'synthetic_rows_excluded_from_holdout':True, 'training_and_holdout_groups_disjoint':True},
        unlabeled_event_candidate_rows=sum(r.get('label') is None for r in rows),
        provenance_policy='Raw report records retained; historical observations/defaults, simulated controls and alpine scenario assumptions remain distinct.')
    summary['event_group_holdout']=_write_event_group_holdout(path,normalized)
    summary_path.write_text(json.dumps(summary,indent=2))
    print(json.dumps({k:summary[k] for k in ['row_count','positive_regime_counts','synthetic_control_counts','source_report_rows','event_group_holdout']},indent=2))
    return summary
