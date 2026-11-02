"""Independent unit/causality checks for Phase 4; no city-score fitting."""
import unittest
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
import numpy as np
from test_mechanisms import BASE
from backend.physics.multi_causal_engine import MultiCausalEngine
from backend.physics.alpine_weather import apply_alpine_weather
from backend.physics.kinematics import solve_kinematics
from backend.physics.causal_attribution import group_shap_by_cause, explain_causes
from backend.physics.temporal_forecaster import response_function,superposed_head,forecast_creep_failure_hour,forecast_72h
from backend.realtime.forecast_weather import liquid_rain,precipitation_paths,future_hours
from backend.ml.global_xgb import FEATURE_NAMES,GlobalMonotonicXGB


def weather(rain=0.):
    now=datetime(2026,1,1,tzinfo=timezone.utc)
    return [dict(time=(now+timedelta(hours=i)).isoformat(),liquid_rain_mm=rain,precipitation=rain,
                 temperature_2m=15.,soil_temperature_0_to_7cm=14.,snowfall=0.) for i in range(1,73)]


class Phase4Physics(unittest.TestCase):
    def test_field_capacity_has_no_positive_head(self):
        b=replace(BASE,soil_saturation_fraction=.3,rain_24h_mm=0,snow_depth_m=0,air_temperature_c=15,soil_temperature_c=14)
        s=MultiCausalEngine().compute(b)
        self.assertLess(s.local_saturation[0],s.theta_33[0]/s.theta_s[0])
        np.testing.assert_array_equal(s.pore_pressure_head_m,0)
        self.assertGreater(s.suction_cohesion_kpa[0],0)
        wet=MultiCausalEngine().compute(replace(b,soil_saturation_fraction=.95))
        self.assertGreater(wet.pore_pressure_head_m[0],0)

    def test_small_vg_n_remains_finite(self):
        with np.errstate(over='raise',invalid='raise'):
            s=MultiCausalEngine().compute(replace(BASE,vg_n=1.0001,soil_saturation_fraction=.1))
        self.assertTrue(np.isfinite(s.fs_dynamic).all())

    def test_no_baseline_snow_creation(self):
        b=replace(BASE,snow_depth_m=0.,freeze_thaw_cycles_7d=0.,air_temperature_c=8.,latitude_deg=46.)
        a=apply_alpine_weather(b,[2500.,3000.])
        np.testing.assert_array_equal(a.snow_depth_m,0)
        np.testing.assert_array_equal(a.freeze_thaw_cycles_7d,0)
        dry=MultiCausalEngine().compute(a)
        np.testing.assert_array_equal(dry.thaw_factor,1.)
        scenario=apply_alpine_weather(b,[2500.,3000.],8.)
        self.assertTrue(np.all(scenario.snow_depth_m>0))

    def test_shap_mass_conserved_and_used(self):
        shap=np.arange(14,dtype=float)[None,:]
        grouped=group_shap_by_cause(shap,FEATURE_NAMES)
        self.assertAlmostEqual(sum(v[0] for v in grouped.values()),shap.sum())
        state=MultiCausalEngine().compute(BASE)
        b=explain_causes(BASE,state,shap_by_cause=grouped)
        self.assertIsNotNone(b.shap_percentages)
        self.assertAlmostEqual(sum(v[0] for v in b.combined_percentages.values()),100.)

    def test_surface_flow_can_move_above_stable_slab(self):
        b=replace(BASE,burn_severity=.85,rain_24h_mm=65.,slope_deg=35.,ndvi_norm=.8)
        s=MultiCausalEngine().compute(b);s.fs_dynamic[:]=2.
        self.assertGreater(s.p_failure_m1[0],.5)
        self.assertGreater(solve_kinematics(b,s).peak_velocity_m_s[0],5.)
        b=replace(b,burn_severity=0.);s=MultiCausalEngine().compute(b);s.fs_dynamic[:]=2.
        self.assertEqual(solve_kinematics(b,s).peak_velocity_m_s[0],0.)

    def test_geology_and_river_inputs_affect_physics(self):
        b=replace(BASE,twi=18.,river_distance_m=0.)
        e=MultiCausalEngine();s=e.compute(b)
        rock=e.compute(replace(b,geology_friction_adjustment_deg=1.5,geology_cohesion_adjustment_kpa=1.5))
        self.assertGreater(rock.fs_dynamic[0],s.fs_dynamic[0])
        flood=e.compute(replace(b,river_surge_ratio=4.))
        self.assertGreater(flood.toe_undercut_multiplier[0],s.toe_undercut_multiplier[0])


class ForecastMath(unittest.TestCase):
    def test_constant_pulse_telescopes_to_single_storm(self):
        k,d,z=1e-5,2e-5,1.5
        flux=np.full((12,1),k/2)
        expected=z*.5*response_function(np.array([d*12*3600/z**2]))
        np.testing.assert_allclose(superposed_head(flux,k,d,z),expected,rtol=1e-12)
        np.testing.assert_array_equal(response_function(np.array([0.])),0)
        np.testing.assert_array_equal(superposed_head(flux*0,k,d,z),0)

    def test_incoming_storm_does_not_change_present(self):
        b=replace(BASE,soil_saturation_fraction=.3,snow_depth_m=0.,air_temperature_c=15.,soil_temperature_c=14.,rain_24h_mm=0.,slope_deg=35.)
        state=MultiCausalEngine().compute(b)
        normal=forecast_72h(b,state,np.array([0.]),None,weather())
        storm=forecast_72h(b,state,np.array([0.]),None,weather(),injected_storm_mm=220,storm_arrival_hour=18,storm_duration_hours=12)
        self.assertEqual(normal['current_risk_t0_pct'],storm['current_risk_t0_pct'])
        for i in range(19):self.assertEqual(normal['paths']['p50']['hourly'][i],storm['paths']['p50']['hourly'][i])
        self.assertAlmostEqual(sum(r['liquid_rain_mm'] for r in storm['paths']['p50']['hourly'][1:]),220.)
        self.assertEqual(len(storm['paths']['p50']['hourly']),73)
        self.assertLess(min(r['critical_fs'] for r in storm['paths']['p50']['hourly']),normal['paths']['p50']['hourly'][0]['critical_fs'])

    def test_missing_hour_does_not_become_dry_forecast(self):
        b=replace(BASE,air_temperature_c=15,soil_temperature_c=14)
        state=MultiCausalEngine().compute(b);rows=weather();rows[5]['liquid_rain_mm']=None
        self.assertEqual(forecast_72h(b,state,np.array([0.]),None,rows)['status'],'UNAVAILABLE')
        self.assertEqual(forecast_72h(b,state,np.array([0.]),None,rows[:-1])['status'],'UNAVAILABLE')

    def test_inverse_velocity_root_and_rejection(self):
        t=np.arange(5.)
        self.assertAlmostEqual(forecast_creep_failure_hour(1/(10-t),t),10.)
        self.assertIsNone(forecast_creep_failure_hour(np.ones(5),t))
        with self.assertRaises(ValueError):forecast_creep_failure_hour([1,0,2],[0,1,2])

    def test_showers_and_explicit_ensemble_fallback(self):
        self.assertEqual(liquid_rain({'rain':[2.],'showers':[8.]},0),10.)
        self.assertAlmostEqual(liquid_rain({'precipitation':[10.],'snowfall':[7.]},0),0.)
        self.assertIsNone(liquid_rain({'rain':[2.]},0))
        a,b,modes=precipitation_paths(weather(2.),None)
        np.testing.assert_allclose(a,2);np.testing.assert_allclose(b,3)
        self.assertIn('FALLBACK',modes[0])

    def test_epistemic_formula_not_binary_label_interval(self):
        class Fold:
            def __init__(self,p):self.p=p
            def predict_proba(self,x):return np.tile([1-self.p,self.p],(len(x),1))
        model=GlobalMonotonicXGB();model.model=SimpleNamespace(calibrated_classifiers_=[Fold(p) for p in [.4,.45,.5,.55,.6]])
        x=np.zeros((1,14));s=.03;p=np.array([.5])
        actual=model.epistemic_probability_intervals(x,p,np.array([s]))
        half=1.645*np.sqrt(np.var([.4,.45,.5,.55,.6])+s*s)
        np.testing.assert_allclose(actual,[[.5-half,.5+half]])


class ProviderUnits(unittest.TestCase):
    def test_nasa_fraction_converted_only_at_percent_consumers(self):
        from unittest.mock import patch
        from backend.realtime.global_ingestor import GlobalIngestor
        from backend.realtime.soilgrids import estimate_pedotransfer
        power={'properties':{'parameter':{'GWETROOT':{'20260101':.8}}}}
        with patch('backend.realtime.global_ingestor.estimate_pedotransfer',wraps=estimate_pedotransfer) as estimate:
            h=GlobalIngestor.__new__(GlobalIngestor)._combine_hydrology(None,power,None)
        self.assertEqual(estimate.call_args.args[3],80.)
        self.assertEqual(h.blended_relative_wetness_pct,80.)
        self.assertEqual(h.nasa_gwetroot_pct,.8)

    def test_invalid_injection_rejected_even_without_weather(self):
        with self.assertRaises(ValueError):
            forecast_72h(None,None,None,None,[],injected_storm_mm=-1)
        with self.assertRaises(ValueError):
            superposed_head(np.ones((1,1)),np.nan,1.,1.)


class PartialRiverHistory(unittest.IsolatedAsyncioTestCase):
    async def test_five_valid_history_days_and_seven_future_days(self):
        from unittest.mock import AsyncMock
        from backend.realtime.ground_context import river_discharge
        today=datetime.now(timezone.utc).date()
        days=[(today+timedelta(days=i)).isoformat() for i in range(-5,7)]
        response=SimpleNamespace(raise_for_status=lambda:None,json=lambda:{'daily':{'time':days,'river_discharge':[2.]*5+[8.]*7}})
        result=await river_discharge(SimpleNamespace(get=AsyncMock(return_value=response)),0,0)
        self.assertEqual(result['history_days_available'],5)
        self.assertEqual(len(result['forecast_7d']),7)
        self.assertEqual(result['surge_ratio'],4.)
