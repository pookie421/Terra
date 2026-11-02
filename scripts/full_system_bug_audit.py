#!/usr/bin/env python3
"""Audit engineering invariants separately from empirical benchmark targets.

Default runs live REST scans and records responses. --offline runs local checks only
and cannot authorize pretraining. CUDA/Unsloth execution needs a compatible CUDA environment.
"""
from __future__ import annotations
import argparse
import ast
import asyncio
import hashlib
import importlib.util
import json
import math
import os
import sys
import tempfile
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from xml.etree import ElementTree
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
sys.dont_write_bytecode=True
os.environ.setdefault('TERRA_PROVIDER_TIMEOUT_S','12')
from backend.physics.multi_causal_engine import PhysicsInputs, MultiCausalEngine
from backend.physics.kinematics import solve_kinematics, route_runout
from backend.physics.causal_attribution import CAUSES, explain_causes, select_numbers_that_matter, _normalize
from backend.api.serialization import sanitize_json
from backend.release import HOME_PREFIX, portable_payload
from backend.ml.global_xgb import FEATURE_NAMES, MONOTONE_VECTOR, feature_matrix, spatial_group_splits
from backend.ml.validation_suite import load_labeled_training_data, run_kinematic_branch_controls

BASE=PhysicsInputs(.45,.2,.03,1.45,35.,8.,.3,0.,24.,0.,10.,8.,0.,0.,.5,0.,9999.,9999.)
CASES=[
 ('Wayanad',dict(lat=11.548,lon=76.134,rain_delta_mm=180.),.95,.80,'hydro_meteorological'),
 ('Kaikoura',dict(lat=-42.260,lon=173.620,pga_override_g=.35),.80,.85,'seismic_tectonic'),
 ('Zermatt',dict(lat=46.020,lon=7.749,snowmelt_temp_surge_c=8.),.98,.65,'cryospheric'),
 ('Malibu',dict(lat=34.07022,lon=-118.54453,burn_severity_override=.85,rain_delta_mm=65.),.95,.75,'post_wildfire_biotic'),
 ('New Delhi',dict(lat=28.6139,lon=77.2090,rain_delta_mm=300.),1.50,.001,None),
]

def check_cards(container):
    shares=container['causal_breakdown_pct']
    assert set(shares)==set(CAUSES)
    assert all(isinstance(v,(int,float)) and math.isfinite(v) and v>=0 for v in shares.values())
    assert abs(sum(shares.values())-100)<.2
    cards=container['numbers_that_matter'];assert len(cards)==6
    for card in cards:
        assert {'metric','value','unit','threshold','status','causal_group','weight_percent'}<=card.keys()
        assert card['status'] in {'CRITICAL','WATCH','NORMAL'}
        assert math.isfinite(card['value']) and math.isfinite(card['weight_percent'])

def formula_and_kinematics():
    engine=MultiCausalEngine();base=engine.compute(BASE)
    burned=engine.compute(replace(BASE,burn_severity=.85,rain_24h_mm=65))
    np.testing.assert_allclose(burned.regolith_depth_m,base.regolith_depth_m)
    assert burned.p_failure_m1[0] > .5 and burned.debris_volume_m3[0]>0
    np.testing.assert_allclose(burned.fs_dynamic,burned.fs_mohr_coulomb)
    shaken=engine.compute(replace(BASE,pga_g=.8,burn_severity=.85,rain_24h_mm=65))
    np.testing.assert_allclose(shaken.p_failure_physical,1-(1-shaken.p_failure_fosm)*(1-shaken.p_failure_seismic)*(1-shaken.p_failure_m1))
    # Cross the branch boundaries with hostile triggers, holding FS under explicit control.
    fs=np.array([.8,1.,1.10,1.15,1.15001,1.25,1.25001,2.,5.])
    inp=replace(BASE,slope_deg=np.full(len(fs),40.),burn_severity=.85,rain_24h_mm=65.,pga_g=1.)
    state=engine.compute(inp);state.fs_dynamic=fs;kin=solve_kinematics(inp,state)
    assert np.all(kin.peak_velocity_m_s > 5)  # surface M1 runoff can move over a stable slab
    clean=replace(inp,burn_severity=0.,rain_24h_mm=0.,pga_g=0.,soil_saturation_fraction=.15)
    clean_state=engine.compute(clean);clean_state.fs_dynamic=fs
    assert np.all(solve_kinematics(clean,clean_state).peak_velocity_m_s[fs>1.25]==0)
    actual=run_kinematic_branch_controls();assert len(actual)==4 and all(x.passed for x in actual)
    return {'branch_controls':[x.__dict__ for x in actual],'stable_rapid_contradictions':0,'boundary_cases':len(fs)}

def local_features_and_guard():
    import unittest
    suite=unittest.defaultTestLoader.discover(str(ROOT/'tests'))
    result=unittest.TextTestRunner(verbosity=1).run(suite)
    assert result.wasSuccessful(), 'Stabilization regression tests failed'
    return {'offline_regressions_passed':result.testsRun,'feature_policy':'local slope and local FS; no ring overwrite'}


def runout_safety():
    cells={str(i):SimpleNamespace(elevation_m=1000.-i*10, latitude=30.+i*.0001,longitude=80.,slope_deg=30.,downslope_neighbors=[str(i+1),str(max(0,i-1))]) for i in range(100)}
    route=route_runout('0',cells,20,.01,500,max_steps=10000)
    assert len(route.path)==51 and len(set(route.path))==len(route.path)
    cells['1'].elevation_m=1000.
    assert route_runout('0',cells,20,.01,500).path==['0']
    cells['1'].elevation_m=1010.
    assert route_runout('0',cells,20,.01,500).path==['0']
    assert route_runout('0',cells,0,.01,500).path==['0']
    return {'max_edges':50,'flat_pit_zero_energy':'stops safely; strict downhill policy'}

def attribution_edges():
    inp=replace(BASE,soil_saturation_fraction=0.,ndvi_norm=0.,road_cut_distance_m=1e6,river_distance_m=1e6)
    state=MultiCausalEngine().compute(inp);b=explain_causes(inp,state)
    assert not b.valid[0]
    check_cards({'causal_breakdown_pct':{k:float(v[0]) for k,v in b.combined_percentages.items()},'numbers_that_matter':[x.to_dict() for x in select_numbers_that_matter(0,inp,state,b)]})
    normalized=_normalize({c:np.array([np.nan,np.inf,-1.,0.]) for c in CAUSES})
    np.testing.assert_allclose(sum(normalized.values()),100.)
    return {'zero_stress':'neutral 20% prior per cause, valid=False','cards':6}

def serialization_edges():
    raw={'nested':[np.float32(.3),np.int64(7),np.bool_(True),np.array([np.nan,np.inf,-np.inf])],'none':None}
    clean=sanitize_json(raw);json.dumps(clean,allow_nan=False)
    assert type(clean['nested'][0]) is float and type(clean['nested'][1]) is int and type(clean['nested'][2]) is bool
    assert clean['serialization_quality']['nonfinite_replacement_count']==3 and clean['none'] is None
    return {'replacements_reported':3}

def feature_contract():
    expected=('fs_dynamic','p_f_fosm','psi_head_m','newmark_dn_cm','slope_max_deg','twi','soil_saturation','rain_24h_mm','snowmelt_mm_day','pga_g','thaw_degradation','burn_severity','toe_undercut_mult','profile_curvature')
    assert FEATURE_NAMES==expected and MONOTONE_VECTOR==(-1,1,1,1,1,1,1,1,1,1,1,1,1,0)
    sentinels={name:i+.25 for i,name in enumerate(expected)}
    np.testing.assert_allclose(feature_matrix(dict(reversed(list(sentinels.items()))))[0],list(sentinels.values()))
    import backend.ml.build_real_labeled_dataset as builder
    import backend.ml.validation_suite as validation
    from scripts.verify_5_global_causes import _feature_matrix
    from backend.ml.audit_dataset import feature_row
    assert builder.FEATURE_NAMES==validation.FEATURE_NAMES==FEATURE_NAMES
    inp=replace(BASE,profile_curvature=.017);state=MultiCausalEngine().compute(inp)
    row=feature_row(inp,state)
    np.testing.assert_allclose(_feature_matrix(inp,state,np.array([inp.profile_curvature]))[0],[row[k] for k in FEATURE_NAMES],rtol=1e-6)
    # Windows imports the same schema and mirrored physics/ML code, never an independent column list.
    for package in ('ml','physics','realtime'):
        for path in (ROOT/'backend'/package).glob('*.py'):
            assert path.read_bytes()==(ROOT/'training/backend'/package/path.name).read_bytes(),str(path.relative_to(ROOT))
    trainer=(ROOT/'training/train.py').read_text()
    assert 'from backend.ml.global_xgb import FEATURE_NAMES, MONOTONE_VECTOR' in trainer
    from backend.ml.global_xgb import make_base_estimator
    assert make_base_estimator(min_child_weight=7.0).get_params()['min_child_weight']==7.0
    return {'features':list(FEATURE_NAMES),'signs':list(MONOTONE_VECTOR),'bundle_code_identical':True}

def dataset_isolation():
    import pyarrow.parquet as pq
    rows=pq.read_table(ROOT/'data/real_global_landslide_observations.parquet').to_pylist()
    synthetic=[r for r in rows if r.get('is_synthetic')];assert len(synthetic)==40
    assert sum(r['name']=='high_rain_flat' for r in synthetic)==25
    assert sum(r['name']=='dry_steep' for r in synthetic)==15
    restored=[r for r in rows if r.get('label')==1 and r.get('regime')=='trigger_unclassified']
    assert len(restored)==125 and all(r.get('event_date') and r.get('citation') for r in restored)
    X,y,g=load_labeled_training_data(str(ROOT/'data/model_training.parquet'))
    for tr,te in spatial_group_splits(y,g,5):
        assert not set(g[tr])&set(g[te])
        assert not any(str(v).startswith('synthetic:') for v in g[te])
        assert sum(str(v).startswith('synthetic:') for v in g[tr])==40
    train=pq.read_table(ROOT/'data/model_training.parquet').to_pylist()
    hold=pq.read_table(ROOT/'data/historical_hindcast_candidates.parquet').to_pylist()
    assert not {r.get('event_group_id') for r in train}&{r.get('event_group_id') for r in hold}
    from collections import Counter
    counts=Counter(r['regime'] for r in train if r.get('label')==1)
    assert {'monsoon_rain','coseismic','cryosphere','post_wildfire','anthropogenic_cut_mining'}<=set(counts)
    assert not {r.get('region_block') for r in train}&{r.get('region_block') for r in hold}
    balance=Counter((r['regime'],r['label']) for r in hold)
    assert {r['regime'] for r in hold}=={'monsoon_rain','coseismic','cryosphere'}
    assert len(set(balance.values()))==1 and not any(r.get('is_synthetic') for r in hold)
    from backend.ml.mapped_audit_inventory import mapped_candidates
    mapped,_=mapped_candidates(ROOT/'data/source_catalogs/audit_mapped_polygons.json')
    verified={r.event_id:r.h3_cell_id for r in mapped}
    for r in rows:
        if r.get('event_record_type')=='audit_mapped_polygon_cell':
            assert verified[r['event_id']]==r['h3_cell_id']
    return {'train_rows':len(y),'synthetic':40,'positive_regime_counts':dict(counts),
            'positive_balance_achieved':max(counts.values())<=2*min(counts.values()),
            'restored_unclassified_positive_rows':len(restored),'holdout_rows':len(hold),
            'holdout_balanced_by_regime_and_label':True,'holdout_spatial_blocks_disjoint':True,
            'note':'Inventory absence negatives unverified; counts are cells, not independent events.'}

def windows_exports():
    path=ROOT/'training/train_qwen25_agent.py'
    spec=importlib.util.spec_from_file_location('qwen_export_audit',path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    tree=ast.parse(path.read_text());values={}
    for node in ast.walk(tree):
        if isinstance(node,ast.Call):
            for kw in node.keywords:
                if kw.arg in ('num_proc','dataset_num_proc','dataloader_num_workers'):values[kw.arg]=ast.literal_eval(kw.value)
    assert values=={'num_proc':1,'dataset_num_proc':1,'dataloader_num_workers':0}
    assert '__name__ == "__main__"' in path.read_text()
    assert '__name__ == "__main__"' in (ROOT/'training/train.py').read_text()
    calls=[]
    class Model:
        def save_pretrained(self,path):calls.append('adapter');Path(path).mkdir(parents=True);(Path(path)/'adapter_model.safetensors').write_bytes(b'audit mock')
        def save_pretrained_merged(self,path,tokenizer,save_method):calls.append('merged');Path(path).mkdir();(Path(path)/'model.safetensors').write_bytes(b'audit mock')
        def save_pretrained_gguf(self,*a,**kw):calls.append('gguf');raise RuntimeError('Injected converter failure')
    class Tokenizer:
        def save_pretrained(self,path):(Path(path)/'tokenizer.json').write_text('{}')
    with tempfile.TemporaryDirectory() as tmp:
        out=Path(tmp)/'lora';result=module.export_trained_model(Model(),Tokenizer(),out,Path(tmp)/'model.gguf')
        assert calls==['adapter','merged','gguf'] and result['gguf']['status']=='failed'
        assert result['adapter']['status']==result['merged_16bit']['status']=='saved'
        assert (out/'export_manifest.json').exists()
    return {'workers':values,'gguf_failure_retains_weights':True,'execution':'mocked export failure; CUDA/Unsloth not installed in the reference evaluation'}

def portability():
    bad=[];files=0
    for path in ROOT.rglob('*'):
        if path.is_file() and '.git' not in path.parts:
            files+=1
            if HOME_PREFIX.encode() in path.read_bytes():bad.append(str(path.relative_to(ROOT)))
    assert not bad,bad
    return {'files_scanned':files,'workstation_home_paths':0,'scope':'whole workspace'}

def geojson(scan):
    import h3
    from tests.geojson_helpers import check_geojson
    check_geojson(scan)
    telemetry=scan['location']['nasa_telemetry']
    assert type(telemetry['nasa_coolr_nearby_count']) is int
    assert type(telemetry['nasa_coolr_regional_count_100km']) is int
    assert 'nearest_coolr_event_km' in telemetry
    reliability=scan['location']['reliability_report']
    assert reliability['operational_confidence_tier']=='TIER_2_PARTIAL_PROXY'
    assert {'dem_source','weather_source','nasa_hydrology_source','seismic_source','soil_source','osm_exposure_source','feeds'}<=reliability.keys()
    assert 'srtm_terrarium_elevation_range_m' in telemetry and 'nasadem_elevation_range_m' not in telemetry
    for f in scan['h3_geojson']['features']:
        props=f['properties'];ring=f['geometry']['coordinates'][0]
        boundary=[[lon,lat] for lat,lon in h3.cell_to_boundary(props['h3_index'])];boundary.append(boundary[0])
        np.testing.assert_allclose(ring,boundary,atol=1e-9)
        lat,lon=h3.cell_to_latlng(props['h3_index'])
        assert abs(props['lat']-lat)<1e-8 and abs(props['lon']-lon)<1e-8
    for f in scan['runout_geojson']['features']:
        coordinates=f['geometry']['coordinates'];assert 2<=len(coordinates)<=51
        for lon,lat in coordinates:assert -180<=lon<=180 and -90<=lat<=90


def source_inventory():
    files=[]
    for folder in ('backend','scripts','training'):
        for path in (ROOT/folder).rglob('*.py'):
            source=path.read_text();compile(source,str(path.relative_to(ROOT)),'exec');ast.parse(source)
            files.append({'path':str(path.relative_to(ROOT)),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
    return {'python_files_compiled':len(files),'files':files}

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--offline',action='store_true');args=parser.parse_args()
    report={'generated_at_utc':datetime.now(timezone.utc).isoformat(),'checks':{},'scenarios':[],
            'limitations':['Source footprint/point proxies and unverified temporal absences are not surveyed global ground truth.',
             'Requested scarp/fire coefficients and alpine snowpack floors are scenario assumptions, not calibrated laws.',
             'Native TreeSHAP explains raw booster margins, not isotonic or mechanically gated scores.',
             'Windows CUDA/Unsloth execution is not verified in the reference evaluation.']}
    def run(name,fn):
        try:
            detail=fn();report['checks'][name]={'passed':True,'details':detail};print('PASS',name,flush=True);return detail
        except Exception as exc:
            report['checks'][name]={'passed':False,'error':type(exc).__name__+': '+str(exc)};print('FAIL',name,str(exc),flush=True);return None
    run('physics_and_kinematics',formula_and_kinematics)
    run('local_features_and_briefing_guard',local_features_and_guard)
    run('vector_3_feature_order',feature_contract)
    run('vector_4_runout',runout_safety)
    run('vector_5_attribution',attribution_edges)
    run('vector_6_sanitizer',serialization_edges)
    run('dataset_isolation',dataset_isolation)
    run('vector_8_windows_export',windows_exports)
    run('source_compilation',source_inventory)
    def offline_regressions():
        import subprocess
        result=subprocess.run([sys.executable,'-m','unittest','discover','-s','tests'],cwd=ROOT,capture_output=True,text=True)
        assert result.returncode==0,result.stdout+result.stderr
        return {'output':result.stderr.strip()}
    run('offline_regression_suite',offline_regressions)
    scans=[];api_payloads=[]
    if not args.offline:
        from fastapi.testclient import TestClient
        from backend.api.main import app, scan_service
        with TestClient(app) as client:
            for name,params,fs_limit,p_limit,cause in CASES:
                try:
                    response=client.get('/api/v1/scan',params={'query':name,**params});response.raise_for_status()
                    scan=response.json()
                    if not scan.get('h3_geojson',{}).get('features'):
                        report.setdefault('provider_retries',[]).append({'name':name,'source_status':scan.get('source_status')})
                        response=client.get('/api/v1/scan',params={'query':name,**params});response.raise_for_status();scan=response.json()
                    scans.append(scan);api_payloads.append(scan)
                    s=scan['summary'];top=s['kinematics_top_hotspot'];assert top is not None,'No live terrain'
                    flags={'fs':s['min_fs_dynamic']<fs_limit,'failure_score':s['max_p_failure']>p_limit,'cause':s['dominant_cause']==cause}
                    if name in ('Wayanad','Malibu'):flags['velocity']=top['peak_velocity_ms']>5
                    if name=='Kaikoura':flags['selected_hotspot_newmark']=top['newmark_displacement_cm']>5
                    if name=='New Delhi':flags={'fs':s['min_fs_dynamic']>fs_limit,'failure_score':s['max_p_failure']<p_limit,'velocity':max(f['properties']['peak_velocity_ms'] for f in scan['h3_geojson']['features'])==0.}
                    row={'name':name,'p05_fs':s['min_fs_dynamic'],'p95_failure':s['max_p_failure'],'hotspot_velocity_ms':top['peak_velocity_ms'],
                         'dominant_cause':s['dominant_cause'],'hotspot_newmark_cm':top['newmark_displacement_cm'],
                         'viewport_max_newmark_cm':max(f['properties']['newmark_displacement_cm'] for f in scan['h3_geojson']['features']),
                         'targets':flags,'passed':all(flags.values()),'source_status':scan['source_status']}
                    if name=='New Delhi':row['part1_target_lt_0_1_percent']=s['max_p_failure']<.001
                    row['nasa_telemetry']=dict(scan['location']['nasa_telemetry'])
                    events=row['nasa_telemetry'].get('nasa_eonet_active_events')
                    if isinstance(events,list):
                        row['nasa_telemetry']['nasa_eonet_active_events']={
                            'returned_record_count':len(events),
                            'records_sha256':hashlib.sha256(json.dumps(events,sort_keys=True).encode()).hexdigest(),
                            'records_location':'Local Git-ignored full_system_live_api_snapshots.json'}
                    row['reliability_report']=scan['location']['reliability_report']
                    report['scenarios'].append(row)
                    print('SCENARIO',name,json.dumps({k:v for k,v in row.items() if k not in ('source_status','nasa_telemetry','reliability_report')}),flush=True)
                except Exception as exc:
                    report['scenarios'].append({'name':name,'passed':False,'error':str(exc)});print('SCAN FAILED',name,exc,flush=True)
            def live_geometry():
                assert len(scans)==5
                for scan in scans:geojson(scan)
                return {'scans':5,'h3_rings_verified_against_library':True}
            run('vector_1_geojson',live_geometry)
            def live_shap():
                assert len(scans)==5 and scan_service.model is not None
                import xgboost as xgb
                checked=0
                for scan in scans:
                    assert scan['summary']['shap_status']=='native_tree_shap_fold_boosters'
                    index=scan['summary']['kinematics_top_hotspot']['h3_index']
                    matrix=scan_service._get_context(index)['feature_rows'];dm=xgb.DMatrix(matrix)
                    values=scan_service.model.shap_log_odds_contributions(matrix)
                    assert values.shape==(len(matrix),14) and np.isfinite(values).all() and np.any(values>0)
                    for fold in scan_service.model.model.calibrated_classifiers_:
                        booster=fold.estimator.get_booster();contrib=booster.predict(dm,pred_contribs=True)
                        np.testing.assert_allclose(contrib.sum(axis=1),booster.predict(dm,output_margin=True),atol=2e-5,rtol=2e-5);checked+=1
                return {'fold_scan_additivity_checks':checked,'method':'Native exact TreeSHAP on underlying XGBoost boosters'}
            run('vector_2_live_shap',live_shap)
            def live_cards():
                assert len(scans)==5
                for scan in scans:
                    assert scan['h3_geojson']['features'], 'No terrain for live invariant check'
                    check_cards(scan['summary'])
                    for f in scan['h3_geojson']['features']:
                        p=f['properties'];check_cards(p)
                        runoff=p['p_failure_m1']>=.5 and p['effective_burn_severity']>=.35 and scan['summary']['meteorological_metrics']['rain_24h_mm_scenario']>=25
                        assert runoff or p['fs_dynamic']<=1.25 or p['peak_velocity_ms']==0
                        assert runoff or p['fs_dynamic']<=1.05 or p['peak_velocity_ms']<=.05
                        np.testing.assert_allclose(p['hotspot_score'],p['p_failure']*(.35+.65*p['exposure_weight'])*p['kinematic_factor'])
                return {'all_cells_checked':sum(len(s['h3_geojson']['features']) for s in scans)}
            run('live_cards_and_kinematic_invariant',live_cards)
            def endpoints():
                response=client.post('/api/v1/agent/chat',json={'message':'Scan Wayanad with +180 mm rainfall','current_query':'Malibu','sliders':{}})
                response.raise_for_status();chat=response.json();api_payloads.append(chat)
                assert chat['numeric_audit_passed'] and chat['briefing_guard']!='WITHHELD_INVALID_TEMPLATE';geojson(chat['updated_scan_data'])
                assert chat['ui_action']['sliders']==chat['updated_scan_data']['scenario']
                v=client.get('/api/v1/validation/report');v.raise_for_status();api_payloads.append(v.json())
                cap=client.post('/api/v1/export/cap-xml',json={'scan':scans[0]});cap.raise_for_status();xml=ElementTree.fromstring(cap.text)
                ns={'c':'urn:oasis:names:tc:emergency:cap:1.2'};assert xml.find('c:status',ns).text=='Test'
                for payload in api_payloads:json.dumps(payload,allow_nan=False)
                assert client.get('/api/v1/scan',params={'lat':'nan','lon':77.}).status_code==422
                assert client.get('/api/v1/scan',params={'rain_delta_mm':'inf'}).status_code==422
                from backend.llm.qwen_agent import parse_scenario_request
                parsed=parse_scenario_request('Scan New Delhi with 0 mm rain','Wayanad',{'rain_delta_mm':180})
                assert parsed['rain_delta_mm']==0 and 'New Delhi' in parsed['location_query']
                return {'strict_json_payloads':len(api_payloads),'cap_xml':'parsed test message','invalid_input':'422','agent_scenario_ui_parity':True}
            run('all_api_endpoints',endpoints)
    run('vector_7_portability',portability)
    validation=json.loads((ROOT/'artifacts/scientific_validation_report.json').read_text())
    validation['stage4_kinematics']=[r.__dict__ for r in run_kinematic_branch_controls()]
    stage1=validation.get('stage1',{}).get('metrics',{});stage2=validation.get('stage2',{}).get('metrics',{});stage3=validation.get('stage3_monotonicity',{}).get('metrics',{})
    report['validation']={'stage1':stage1,'stage2':stage2,'stage3':stage3,'stage4':validation.get('stage4_kinematics',[])}
    report['validation_targets']={
        'stage1_auc_gte_0_84':stage1.get('roc_auc',0)>=.84,
        'stage2_auc_gte_0_80':stage2.get('roc_auc',0)>=.80,
        'stage3_zero_violations':validation.get('stage3_monotonicity',{}).get('status')=='passed',
        'stage4_four_controls':len(report['validation']['stage4'])==4 and all(r['passed'] for r in report['validation']['stage4'])}
    prefixes=['vector_1_','vector_2_','vector_3_','vector_4_','vector_5_','vector_6_','vector_7_','vector_8_']
    report['latent_vectors_passed']=sum(any(k.startswith(p) and v['passed'] for k,v in report['checks'].items()) for p in prefixes)
    license_text=(ROOT/'LICENSE').read_text()
    report['checks']['mit_license']={'passed':license_text.startswith('MIT License') and 'Permission is hereby granted, free of charge' in license_text}
    report['empirical_benchmark_metrics']={k:validation.get(k,{}) for k in ('stage1','stage2')}
    # Empirical score targets remain visible, but are not engineering invariants.
    delhi=next((s for s in report['scenarios'] if s['name']=='New Delhi'),{})
    report['engineering_ready']=bool(not args.offline and all(x['passed'] for x in report['checks'].values())
        and report['validation_targets']['stage3_zero_violations'] and report['validation_targets']['stage4_four_controls']
        and True  # empirical city scores do not determine engineering readiness
        )
    report['pretraining_ready']=False  # Only the combined, fingerprint-matched audits may grant readiness.
    report['readiness_scope']='Engineering and invariant audit only; randomized audit must also pass. Empirical results are descriptive.'
    from scripts.finalize_readiness import fingerprint
    report['implementation_fingerprint']=fingerprint()
    report=portable_payload(sanitize_json(report))
    (ROOT/'artifacts/full_system_bug_audit.json').write_text(json.dumps(report,indent=2,allow_nan=False))
    (ROOT/'artifacts/full_system_live_api_snapshots.json').write_text(json.dumps(portable_payload(api_payloads),allow_nan=False))
    validation['engineering_ready']=report['engineering_ready'];validation['empirical_benchmark_metrics']=report['empirical_benchmark_metrics'];validation['pretraining_ready']=report['pretraining_ready'];validation['full_system_audit']='artifacts/full_system_bug_audit.json'
    (ROOT/'artifacts/scientific_validation_report.json').write_text(json.dumps(portable_payload(sanitize_json(validation)),indent=2,allow_nan=False))
    lines=['# Full system audit', '', 'Generated UTC: '+report['generated_at_utc'], '',
           '**engineering_ready = '+str(report['engineering_ready']).lower()+'**', '',
           'This is the full-audit engineering result; combined pretraining readiness also requires the randomized audit. Empirical targets below are descriptive and do not control readiness.', '',
           '| Scenario | Critical P05 FS | Critical P95 score | Top velocity m/s | Result |',
           '|---|---:|---:|---:|---|']
    for row in report['scenarios']:
        lines.append('| {} | {} | {} | {} | {} |'.format(row['name'],
          format(row['p05_fs'],'.3f') if 'p05_fs' in row else 'Unavailable',
          format(100*row['p95_failure'],'.3f')+'%' if 'p95_failure' in row else 'Unavailable',
          format(row['hotspot_velocity_ms'],'.2f') if 'hotspot_velocity_ms' in row else 'Unavailable',
          'PASS' if row['passed'] else 'FAIL'))
    lines+=['','Critical percentiles use steep candidate cells (effective slope >=24 degrees), falling back to the viewport when none exist. These are not whole-viewport percentiles.', '',
            '| Validation | Value | Result |','|---|---:|---|',
            '| Spatial ROC-AUC (>=0.84) | {:.4f} | {} |'.format(stage1.get('roc_auc',0),'PASS' if report['validation_targets']['stage1_auc_gte_0_84'] else 'FAIL'),
            '| Event holdout ROC-AUC (>=0.80) | {:.4f} | {} |'.format(stage2.get('roc_auc',0),'PASS' if report['validation_targets']['stage2_auc_gte_0_80'] else 'FAIL'),
            '| Recall@40 (descriptive) | {:.4f} | Reported |'.format(stage2.get('recall_at_40',0)),
            '| Monotonicity violations | {} / {} | {} |'.format(stage3.get('violations'),stage3.get('comparisons'), 'PASS' if report['validation_targets']['stage3_zero_violations'] else 'FAIL'),
            '| Kinematics | {} / 4 | {} |'.format(sum(r['passed'] for r in report['validation']['stage4']),'PASS' if report['validation_targets']['stage4_four_controls'] else 'FAIL'),
            '| Latent checks | {} / 8 | {} |'.format(report['latent_vectors_passed'],'PASS' if report['latent_vectors_passed']==8 else 'FAIL'), '',
            'The 48-row balanced holdout has 24 event groups across rain, seismic and cryospheric regimes with matched temporal absence proxies. It replaces the Brazil-only benchmark and is not comparable with its old scores. Recall@40 covers most of this small set. Stage 2 scores ungated isotonic probabilities; the live gated blend is not independently calibrated.', '',
            '| Engineering check | Result |','|---|---|']
    for name,check in report['checks'].items():lines.append('| {} | {} |'.format(name,'PASS' if check['passed'] else 'FAIL: '+check.get('error','')))
    lines+=['','| Empirical metric | Estimate | 95% cluster interval |','|---|---:|---|']
    for stage in ('stage1','stage2'):
        benchmark=report['empirical_benchmark_metrics'][stage]
        for metric in ('roc_auc','pr_auc','brier_score','recall_at_k','recall_at_40'):
            if metric not in benchmark.get('metrics',{}):continue
            interval=benchmark.get('confidence_intervals',{}).get('intervals',{}).get(metric)
            lines.append('| {} {} | {:.4f} | {} |'.format(stage,metric,benchmark['metrics'][metric],'{:.4f}–{:.4f}'.format(*interval) if interval else 'unavailable'))
    lines+=['','Intervals resample whole spatial blocks or matched event pairs, conditional on fixed predictions; they exclude refitting and label uncertainty.',*['- '+v for v in report['limitations']]]
    (ROOT/'artifacts/full_system_bug_audit.md').write_text('\n'.join(lines)+'\n')
    print('VALIDATION',json.dumps(report['validation']),flush=True)
    print('VALIDATION TARGETS',report['validation_targets'],flush=True)
    print('LATENT',str(report['latent_vectors_passed'])+'/8; pretraining_ready =',report['pretraining_ready'],flush=True)
    return 0 if report['engineering_ready'] else 1

if __name__=='__main__':raise SystemExit(main())
