#!/usr/bin/env python3
"""Independent formula, source-label and evaluation-isolation regression checks."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from dataclasses import replace
import numpy as np
import pyarrow.parquet as pq
from backend.ml.audit_dataset import mechanical_controls
from backend.ml.global_xgb import mechanical_stability_gate,blend_failure_probability,spatial_group_splits
from backend.ml.validation_suite import load_labeled_training_data
from backend.physics.multi_causal_engine import PhysicsInputs,MultiCausalEngine
from backend.physics.alpine_weather import apply_alpine_weather

base=PhysicsInputs(.45,.2,.03,1.45,40.,8.,.3,0.,24.,0.,10.,8.,0.,0.,.5,0.,9999.,9999.)
engine=MultiCausalEngine()
bare=engine.compute(base);burned=engine.compute(replace(base,burn_severity=.85,rain_24h_mm=65.))
np.testing.assert_allclose(burned.regolith_depth_m,bare.regolith_depth_m)
assert np.all(burned.pore_pressure_head_m<=burned.regolith_depth_m*np.cos(np.deg2rad(burned.effective_slope_deg))**2+1e-12)
dry=engine.compute(replace(base,pga_g=.35))
wet=engine.compute(replace(base,pga_g=.35,rain_24h_mm=350.,soil_saturation_fraction=.99))
assert np.all(wet.critical_acceleration_g <= dry.critical_acceleration_g)
np.testing.assert_allclose(wet.critical_acceleration_g,np.clip((wet.fs_static-1)*np.sin(np.deg2rad(wet.effective_slope_deg))*.75,.05,.6))
assert np.all((dry.critical_acceleration_g>=.05)&(dry.critical_acceleration_g<=.6))
extreme=engine.compute(replace(base,pga_g=1e6))
assert extreme.newmark_displacement_cm[0]==120.
np.testing.assert_allclose(extreme.p_failure_seismic,.335*(1-np.exp(-.048*extreme.newmark_displacement_cm**1.565)))
assert np.all(extreme.p_failure_physical>=extreme.p_failure_seismic)
slope=np.array([0.,8.,40.]);fs=np.array([5.,1.35,.8])
expected=1/(1+np.exp(-(slope-8.5)/.65))/(1+np.exp((fs-1.35)/.12))
np.testing.assert_allclose(mechanical_stability_gate(slope,fs),expected)
yield_blend=1-(.35+.50/(1+np.exp((fs-1.05)/.06)))
np.testing.assert_allclose(blend_failure_probability(np.ones(3),np.zeros(3),slope,fs),np.where(fs<=1.05,np.maximum(.65,yield_blend),.65)*mechanical_stability_gate(slope,fs,np.zeros(3)))
alpine=apply_alpine_weather(base,[1600.,2500.,3310.],8.,latitude_deg=46.02)
np.testing.assert_allclose(alpine.air_temperature_c,[23.85,18.,12.735])
np.testing.assert_allclose(alpine.snow_depth_m,[0.,.28,.6]);assert np.all(alpine.freeze_thaw_cycles_7d[1:]>=3)
controls=mechanical_controls()
high=[r for r in controls if r['name']=='high_rain_flat'];steep=[r for r in controls if r['name']=='dry_steep']
assert len(high)==25 and len(steep)==15
assert all(100<=r['rain_24h_mm']<=350 and r['slope_max_deg']<8 and r['fs_dynamic']>1.45 for r in high)
assert all(r['rain_24h_mm']==r['pga_g']==0 and r['slope_max_deg']>30 and r['fs_dynamic']>1.3 for r in steep)
X,y,g=load_labeled_training_data(str(ROOT/'data/model_training.parquet'))
for train,test in spatial_group_splits(y,g,5):
 assert not any(str(v).startswith('synthetic:') for v in g[test])
 assert sum(str(v).startswith('synthetic:') for v in g[train])==40
train=pq.read_table(ROOT/'data/model_training.parquet').to_pylist();hold=pq.read_table(ROOT/'data/historical_hindcast_candidates.parquet').to_pylist()
assert not ({r.get('event_group_id') for r in train}&{r.get('event_group_id') for r in hold})
assert not any(r.get('is_synthetic') for r in hold)
regimes={r['regime'] for r in train if r.get('label')==1}
assert {'monsoon_rain','coseismic','cryosphere','post_wildfire','anthropogenic_cut_mining'}<=regimes
print('PASS: gate, effective depth, pre-seismic static yield acceleration, Newmark cap/probability, alpine assumptions, 40 simulated controls, five source regimes and calibration/holdout isolation.')
