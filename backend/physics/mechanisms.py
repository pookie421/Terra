"""Published equation forms with explicitly identified scenario input proxies.

References and domain restrictions: docs/physics_sources.md. The ice factors and
post-fire yield index are project scenario parameterizations, not paper equations.
"""
import numpy as np

def sigmoid(x):
    return 1.0/(1.0+np.exp(-np.clip(x,-700,700)))


def staley_m1_probability(rain_15min_mm, burned_steep_fraction, mean_dnbr, soil_erodibility_kf):
    """Staley 2017 M1: R15 in mm, T in [0,1], dNBR scaled by 1000.

    T is the basin fraction BOTH moderately/highly burned AND slope >=23°.
    This empirical basin model is conditional on a recently burned watershed.
    """
    rain, terrain, dnbr, kf = np.broadcast_arrays(*[np.asarray(v, float) for v in
        (rain_15min_mm, burned_steep_fraction, mean_dnbr, soil_erodibility_kf)])
    if not all(np.isfinite(v).all() for v in (rain, terrain, dnbr, kf)) or np.any(rain < 0) or np.any((terrain < 0) | (terrain > 1)) or np.any(dnbr < 0) or np.any(kf < 0):
        raise ValueError('Invalid Staley basin parameters')
    return sigmoid(-3.63 + rain*(.41*terrain + .67*dnbr/1000. + .70*kf))


def gartner_volume_m3(intensity_15min_mm_hr, burned_area_km2, basin_relief_m):
    """Gartner 2014 emergency volume; actual burned basin area in km²."""
    intensity, area, relief = np.broadcast_arrays(*[np.asarray(v, float) for v in
        (intensity_15min_mm_hr, burned_area_km2, basin_relief_m)])
    if not all(np.isfinite(v).all() for v in (intensity, area, relief)) or np.any(intensity < 0) or np.any(area <= 0) or np.any(relief < 0):
        raise ValueError('Invalid Gartner basin parameters')
    return np.exp(4.22+.39*np.sqrt(intensity)+.36*np.log(area)+.13*np.sqrt(relief))

def postfire(*args, **kwargs):
    """Retired cell-proxy API: basin parameters cannot be inferred from burn pixels."""
    raise ValueError('Use staley_m1_probability and gartner_volume_m3 with explicit basin inputs')

def mamot_joint_strength(temperature_c,normal_stress_kpa):
    """Mamot 2018 Eqs.4–6, for verified ice-filled joints in the tested domain."""
    t,stress=np.broadcast_arrays(np.asarray(temperature_c,float),np.asarray(normal_stress_kpa,float))
    if np.any(~np.isfinite(t)) or np.any(~np.isfinite(stress)) or np.any((t < -8)|(t > -.5)|(stress <100)|(stress>400)):
        raise ValueError('Mamot calibration requires -8..-0.5 C and 100..400 kPa, with known ice-filled joints')
    cohesion=53.3-73.5*t
    friction=.42-.21*t
    return cohesion+stress*friction
