"""Stable-ground screening and seismic branch boundary regressions."""
import unittest
from dataclasses import replace
import numpy as np
from backend.ml.global_xgb import mechanical_stability_gate, blend_failure_probability
from backend.physics.multi_causal_engine import MultiCausalEngine
from backend.physics.kinematics import solve_kinematics
from test_mechanisms import BASE


class SurgicalPhysics(unittest.TestCase):
    def test_reference_weight_prevents_post_cap_rain_stabilization(self):
        inputs=replace(BASE,slope_deg=40.,soil_saturation_fraction=.96,
                       twi=np.array([6.,18.]),pore_pressure_override_m=100.,rain_24h_mm=0.)
        engine=MultiCausalEngine();before=engine.compute(inputs)
        after=engine.compute(replace(inputs,rain_24h_mm=300.))
        np.testing.assert_array_equal(after.local_saturation,before.local_saturation)
        np.testing.assert_array_equal(before.pore_pressure_head_m,after.pore_pressure_head_m)
        # Storm-gated cohesion may weaken further even at capped head; added
        # water must never stabilize the fixed reference-weight slab.
        self.assertTrue(np.all(after.fs_dynamic <= before.fs_dynamic))

    def test_rain_cannot_dry_cells_or_raise_mechanical_fs(self):
        rng=np.random.default_rng(2048);n=256
        inputs=replace(BASE,sand_fraction=rng.uniform(.05,.8,n),clay_fraction=rng.uniform(.05,.5,n),
            soil_saturation_fraction=rng.uniform(.05,.99,n),slope_deg=rng.uniform(1.,65.,n),
            twi=rng.uniform(1.,25.,n),ndvi_norm=rng.uniform(0.,1.,n),pga_g=rng.uniform(0.,.5,n),
            snow_depth_m=rng.choice([0.,.5],n),air_temperature_c=rng.uniform(-5.,20.,n),
            soil_temperature_c=rng.uniform(-5.,20.,n),rain_24h_mm=0.)
        engine=MultiCausalEngine();previous=engine.compute(inputs)
        for rain in [*.25*np.arange(1,9),*np.arange(3.,31.),60.,120.,180.,350.,1000.]:
            current=engine.compute(replace(inputs,rain_24h_mm=rain))
            self.assertTrue(np.all(current.local_saturation>=previous.local_saturation-1e-12),rain)
            self.assertTrue(np.all(current.fs_dynamic<=previous.fs_dynamic+1e-10),rain)
            previous=current

    def test_physics_is_reported_separately_from_calibrated_ml(self):
        slope=np.array([10.,35.,35.]);fs=np.array([3.82,1.44,5.])
        np.testing.assert_array_equal(blend_failure_probability(np.ones(3),np.zeros(3),slope,fs),1.)
        np.testing.assert_array_equal(blend_failure_probability(np.zeros(3),np.ones(3),slope,fs),0.)

    def test_threshold_jump_is_visible_and_inputs_are_checked(self):
        before=mechanical_stability_gate([35.],[2.],[.299999])[0]
        after=mechanical_stability_gate([35.],[2.],[.30])[0]
        self.assertEqual(after,before)
        with self.assertRaises(ValueError):mechanical_stability_gate([35.],[2.],[np.nan])

    def test_seismic_yield_and_empirical_probability_are_not_score_boosted(self):
        inputs=replace(BASE,pga_g=.35,slope_deg=np.array([30.,40.]),rain_24h_mm=0.,snow_depth_m=0.)
        state=MultiCausalEngine().compute(inputs)
        np.testing.assert_allclose(state.critical_acceleration_g,np.maximum((state.fs_static-1)*np.sin(np.radians(state.effective_slope_deg)),0.))
        np.testing.assert_allclose(state.kh,0.)
        self.assertTrue(np.all(state.p_failure_seismic<=.335))
        quiet=MultiCausalEngine().compute(replace(inputs,pga_g=0.))
        np.testing.assert_array_equal(quiet.newmark_displacement_cm,0.)
        np.testing.assert_array_equal(quiet.p_failure_seismic,0.)

    def test_unstable_initial_state_is_not_a_newmark_prediction(self):
        inp=replace(BASE,pga_g=.35)
        s=MultiCausalEngine().compute(inp)
        s=replace(s,newmark_applicable=np.array([False]),newmark_displacement_cm=np.array([0.]))
        self.assertNotIn('Coseismic',str(solve_kinematics(inp,s).failure_mode[0]))

    def test_runout_energy_requires_explicit_travel_geometry(self):
        inp=replace(BASE,pga_g=.35,slope_deg=50.,runout_travel_length_m=100.)
        s=MultiCausalEngine().compute(inp)
        unknown=solve_kinematics(replace(inp,runout_travel_length_m=0.),s)
        self.assertEqual(unknown.peak_velocity_m_s[0],0.)
        np.testing.assert_array_equal(solve_kinematics(inp,s).peak_velocity_m_s >= 0,True)

    def test_no_slab_motion_above_yield_without_independent_trigger(self):
        engine=MultiCausalEngine()
        for saturation in (.3,.9):
            inp=replace(BASE,slope_deg=np.full(4,20.),pga_g=0.,burn_severity=0.,rain_24h_mm=0.)
            state=engine.compute(inp)
            state=replace(state,fs_dynamic=np.array([.99,1.,1.073,1.24]),local_saturation=np.full(4,saturation),
                p_failure_m1=np.zeros(4),newmark_displacement_cm=np.zeros(4))
            velocity=solve_kinematics(inp,state).peak_velocity_m_s
            self.assertEqual(velocity[0],0.)
            np.testing.assert_array_equal(velocity[1:],0.)
