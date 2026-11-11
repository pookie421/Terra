"""Network-free semi-infinite Iverson column and infinite-slope mechanics.

Vertical depth Z, pressure head psi (m), effective cohesion c (kPa), unit
weight gamma (kN/m³), conductivity Ks (m/s), hydraulic diffusivity D0 (m²/s).
Iverson's response argument is t*=4*D0*cos(slope)**2*t/Z**2. This module
does not implement TRIGRS' finite-depth or unsaturated boundary conditions.
Sources: Iverson (2000), doi:10.1029/2000WR900090; USGS OFR 2008-1159.
"""
from dataclasses import dataclass
import math
import numpy as np


def response_function(t_star):
    t = np.asarray(t_star, dtype=float)
    if np.any(~np.isfinite(t)) or np.any(t < 0):
        raise ValueError('Dimensionless time must be finite and nonnegative')
    safe = np.maximum(t, np.finfo(float).tiny)
    erfc = np.frompyfunc(math.erfc, 1, 1)
    result = np.sqrt(safe / math.pi) * np.exp(-1 / safe) - erfc(1 / np.sqrt(safe)).astype(float)
    return np.where(t > 0, result, 0.)


def infinite_slope_fs(cohesion_kpa, friction_deg, unit_weight_kn_m3,
                      depth_m, slope_deg, pressure_head_m, kh=0.):
    c, phi, gamma, z, angle, psi, seismic = np.broadcast_arrays(
        *[np.asarray(v, float) for v in (cohesion_kpa, friction_deg,
          unit_weight_kn_m3, depth_m, slope_deg, pressure_head_m, kh)])
    if not all(np.isfinite(v).all() for v in (c, phi, gamma, z, angle, psi, seismic)):
        raise ValueError('Column parameters must be finite')
    if np.any(c < 0) or np.any(gamma <= 0) or np.any(z <= 0) or np.any((phi < 0) | (phi >= 90)) or np.any((angle < 0) | (angle >= 90)) or np.any(seismic < 0):
        raise ValueError('Invalid column material or slope')
    theta = np.radians(angle)
    sin, cos = np.sin(theta), np.cos(theta)
    normal = gamma*z*(cos*cos-seismic*sin*cos)-9.81*psi
    # Soil cannot transmit tensile friction. This is an explicit contact boundary.
    resistance = c + np.maximum(normal, 0.) * np.tan(np.radians(phi))
    driving = gamma*z*(sin*cos+seismic*cos*cos)
    return np.divide(resistance, driving, out=np.full_like(resistance, np.inf), where=driving > 0)


@dataclass(frozen=True)
class ColumnParameters:
    cohesion_kpa: float
    friction_angle_deg: float
    unit_weight_kn_m3: float
    conductivity_m_s: float
    diffusivity_m2_s: float
    depth_m: float
    slope_deg: float
    initial_pressure_head_m: float = 0.


def solve_column(column: ColumnParameters, times_s, pulses):
    """Evaluate (start_s, end_s, infiltration_m_s) pulses at requested times.

    Initial head is the specified steady profile at Z. Flux is infiltration,
    not gross rainfall; 0 <= flux <= Ks is the saturated reference domain.
    No provider calls, pedotransfer, strength adjustment, or head/FS clipping.
    """
    values = np.asarray(list(column.__dict__.values()), float)
    times = np.asarray(times_s, float)
    if not np.isfinite(values).all() or not np.isfinite(times).all() or np.any(times < 0):
        raise ValueError('Finite column parameters and nonnegative times required')
    if column.conductivity_m_s <= 0 or column.diffusivity_m2_s <= 0 or column.depth_m <= 0:
        raise ValueError('Ks, D0, and depth must be positive')
    head = np.full_like(times, column.initial_pressure_head_m)
    scale = 4*column.diffusivity_m2_s*np.cos(np.radians(column.slope_deg))**2/column.depth_m**2
    previous_end = 0.
    for start, end, flux in pulses:
        if not all(math.isfinite(v) for v in (start, end, flux)) or start < previous_end or end <= start or not 0 <= flux <= column.conductivity_m_s:
            raise ValueError('Pulses must be ordered, nonoverlapping, and within infiltration capacity')
        previous_end = end
        head += column.depth_m*flux/column.conductivity_m_s*(
            response_function(scale*np.maximum(times-start, 0.))-
            response_function(scale*np.maximum(times-end, 0.)))
    fs = infinite_slope_fs(column.cohesion_kpa, column.friction_angle_deg,
                          column.unit_weight_kn_m3, column.depth_m,
                          column.slope_deg, head)
    return {'time_s': times, 'pressure_head_m': head, 'factor_of_safety': fs}


def critical_acceleration_g(fs_static, slope_deg):
    fs, slope = np.broadcast_arrays(np.asarray(fs_static, float), np.asarray(slope_deg, float))
    if not np.isfinite(fs).all() or not np.isfinite(slope).all() or np.any((slope < 0) | (slope >= 90)):
        raise ValueError('Invalid static FS or slope')
    return np.maximum((fs-1.)*np.sin(np.radians(slope)), 0.)


def jibson_displacement_cm(critical_g, pga_g):
    """Jibson 2007 PGA-ratio median regression; log10 residual sigma=0.510.

    NaN explicitly denotes ac=0 (no stable sliding-block initial state).
    PGA <= ac gives zero displacement. No empirical floor or displacement cap.
    """
    ac, pga = np.broadcast_arrays(np.asarray(critical_g, float), np.asarray(pga_g, float))
    if not np.isfinite(ac).all() or not np.isfinite(pga).all() or np.any(ac < 0) or np.any(pga < 0):
        raise ValueError('Accelerations must be finite and nonnegative')
    result = np.zeros_like(ac)
    result[ac <= 0] = np.nan
    active = (ac > 0) & (pga > ac)
    ratio = ac[active]/pga[active]
    result[active] = 10**(.215+2.341*np.log10(1-ratio)-1.438*np.log10(ratio))
    return result


def jibson_failure_probability(displacement_cm):
    """Northridge inventory relation (Jibson et al. 2000), not global calibration."""
    d = np.asarray(displacement_cm, float)
    if np.any(d < 0) or np.any(np.isinf(d)):
        raise ValueError('Displacement must be nonnegative or explicitly unavailable')
    return .335*(-np.expm1(-.048*d**1.565))
