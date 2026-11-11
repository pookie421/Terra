"""Independent equation verification, with network access explicitly disabled.

This verifies algebra and units, not observed landslide prediction skill.
"""
import math
import socket
import unittest
from dataclasses import replace
from unittest.mock import patch
import numpy as np
from scipy.integrate import quad
from backend.physics.reference_column import (ColumnParameters, solve_column,
    response_function, infinite_slope_fs, critical_acceleration_g,
    jibson_displacement_cm, jibson_failure_probability)
from backend.physics.temporal_forecaster import superposed_head
from backend.physics.mechanisms import staley_m1_probability, gartner_volume_m3
from backend.physics.multi_causal_engine import MultiCausalEngine, PhysicsInputs
from backend.ml.global_xgb import blend_failure_probability


def independent_response(t):
    # Integrate dR/dt instead of calling or duplicating the production erfc form.
    return quad(lambda u: math.exp(-1/u)/(2*math.sqrt(math.pi*u)), 0, t,
                epsabs=1e-13, epsrel=1e-12)[0] if t > 0 else 0.


class NetworkDisabled(unittest.TestCase):
    def setUp(self):
        for target in ('socket.create_connection', 'socket.socket.connect'):
            guard = patch(target, side_effect=AssertionError('Network is forbidden in equation verification'))
            guard.start()
            self.addCleanup(guard.stop)


class ReferenceEquations(NetworkDisabled):
    def test_iverson_response_against_independent_quadrature(self):
        times = np.array([0, .01, .1, 1., 10., 100.])
        np.testing.assert_allclose(response_function(times), [independent_response(t) for t in times], rtol=1e-10, atol=1e-13)

    def test_explicit_column_pulse_recession_and_fs(self):
        column = ColumnParameters(5., 35., 18., 1e-5, 1e-4, 1., 30., .1)
        times = np.array([0., 3600., 21600., 43200., 86400., 172800.])
        head = np.array([.1+.5*(independent_response(.0003*t)-independent_response(.0003*max(t-21600, 0))) for t in times])
        # Independent conventional infinite-slope expression, vertical Z convention.
        theta, phi = math.radians(30), math.radians(35)
        fs = math.tan(phi)/math.tan(theta)+(5.-9.81*head*math.tan(phi))/(18.*math.sin(theta)*math.cos(theta))
        actual = solve_column(column, times, [(0., 21600., 5e-6)])
        np.testing.assert_allclose(actual['pressure_head_m'], head, rtol=1e-10)
        np.testing.assert_allclose(actual['factor_of_safety'], fs, rtol=1e-10)
        self.assertLess(head[-1], head[-2])
        self.assertEqual(actual['pressure_head_m'][0], .1)
        # A parameter set excluded by the former wrapper clips remains expressible.
        shallow = replace(column, depth_m=.4, cohesion_kpa=.5)
        self.assertTrue(np.isfinite(solve_column(shallow, [600.], [(0., 600., 1e-5)])['factor_of_safety']).all())

    def test_temporal_pulses_match_independent_reference(self):
        flux = np.array([[5e-6], [1e-5], [0.], [0.]])
        expected = sum(q/1e-5*(independent_response(.0003*(4-i)*3600)-independent_response(.0003*(3-i)*3600)) for i, q in enumerate(flux[:, 0]))
        np.testing.assert_allclose(superposed_head(flux, 1e-5, 1e-4, 1., slope_deg=30), [expected], rtol=1e-10)
        np.testing.assert_array_equal(superposed_head(flux*0, 1e-5, 1e-4, 1., slope_deg=30), [0.])

    def test_staley_and_gartner_design_storm_units(self):
        intensity = np.array([16., 24., 40.])
        # Explicit basin fixture: T=.5, mean dNBR=500, KF=.2.
        expected = np.array([1/(1+math.exp(-x)) for x in [-.91, .45, 3.17]])
        np.testing.assert_allclose(staley_m1_probability(intensity/4, .5, 500., .2), expected, rtol=1e-13)
        volume = [math.exp(4.22+.39*math.sqrt(i)+.36*math.log(1.)+.13*math.sqrt(100.)) for i in intensity]
        np.testing.assert_allclose(gartner_volume_m3(intensity, 1., 100.), volume, rtol=1e-13)
        with self.assertRaises(ValueError):
            gartner_volume_m3(16, 0, 100)

    def test_newmark_domain_regression_and_northridge_probability(self):
        np.testing.assert_allclose(critical_acceleration_g([.9, 1., 1.5, 2.], 30.), [0., 0., .25, .5], atol=1e-15)
        ac, pga = np.array([.1, .25, .5]), np.array([.4, .5, .4])
        expected = [10**(.215+2.341*math.log10(1-a/p)-1.438*math.log10(a/p)) if p>a else 0 for a,p in zip(ac,pga)]
        actual = jibson_displacement_cm(ac, pga)
        np.testing.assert_allclose(actual, expected, rtol=1e-13)
        np.testing.assert_allclose(jibson_failure_probability(actual), [.335*(1-math.exp(-.048*d**1.565)) for d in expected], atol=1e-15)
        self.assertTrue(np.isnan(jibson_displacement_cm(0., .5)))
        self.assertGreater(float(jibson_displacement_cm(.001, .5)), 120.)

    def test_wrapper_units_batch_independence_and_no_score_cliffs(self):
        base = PhysicsInputs(.45,.2,.03,1.45,35.,8.,.3,100.,24.,0.,10.,8.,0.,0.,.5,0.,9999.,9999.)
        engine = MultiCausalEngine()
        alone = engine.compute(base)
        batch = engine.compute(replace(base, twi=[8., 16., 24.]))
        for key in ('pore_pressure_head_m','fs_dynamic','local_saturation'):
            self.assertEqual(getattr(alone,key)[0],getattr(batch,key)[0])
        np.testing.assert_allclose(alone.hydraulic_diffusivity_m2_s,
            alone.hydraulic_conductivity_m_s/alone.specific_moisture_capacity_m_inv)
        np.testing.assert_allclose(alone.dimensionless_time,
            4*alone.hydraulic_diffusivity_m2_s*math.cos(math.radians(35))**2*86400/alone.regolith_depth_m**2)
        for fs in (1.049999, 1.05, 1.050001, 2.):
            np.testing.assert_array_equal(blend_failure_probability([.5,.5],[.299999,.3],35.,fs),[.5,.5])
