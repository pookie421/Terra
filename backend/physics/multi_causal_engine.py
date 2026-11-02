"""Vectorized five-regime slope-stability equations for H3 cells."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Union

import numpy as np
from .mechanisms import postfire, suction_cohesion, ice_factors

ArrayInput = Union[float, int, Sequence[float], np.ndarray]
GAMMA_W_KN_M3 = 9.81
PARTICLE_DENSITY_G_CM3 = 2.65


@dataclass
class PhysicsInputs:
    """Measured or explicitly scenario-defined inputs; no silent imputation is applied."""

    sand_fraction: ArrayInput
    clay_fraction: ArrayInput
    organic_matter_fraction: ArrayInput
    bulk_density_g_cm3: ArrayInput
    slope_deg: ArrayInput
    twi: ArrayInput
    soil_saturation_fraction: ArrayInput
    rain_24h_mm: ArrayInput
    storm_duration_hr: ArrayInput
    snow_depth_m: ArrayInput
    air_temperature_c: ArrayInput
    soil_temperature_c: ArrayInput
    freeze_thaw_cycles_7d: ArrayInput
    burn_severity: ArrayInput
    ndvi_norm: ArrayInput
    pga_g: ArrayInput
    road_cut_distance_m: ArrayInput
    river_distance_m: ArrayInput
    profile_curvature: ArrayInput = 0.0
    elevation_m: ArrayInput = 0.0
    warming_delta_c: ArrayInput = 0.0
    peak_3h_intensity_mmhr: ArrayInput = 0.0
    basal_conductivity_ratio: ArrayInput = 0.01
    vg_alpha_kpa_inv: ArrayInput = 0.10
    vg_n: ArrayInput = 2.0
    latitude_deg: ArrayInput = 0.0
    baseline_air_temperature_c: ArrayInput = -999.0
    observed_snow_depth_m: ArrayInput = -1.0
    river_surge_ratio: ArrayInput = 1.0
    geology_friction_adjustment_deg: ArrayInput = 0.0
    geology_cohesion_adjustment_kpa: ArrayInput = 0.0
    rain_15min_override_mm: ArrayInput = -1.0
    snowmelt_override_mm_day: ArrayInput = -1.0
    pore_pressure_override_m: ArrayInput = -1.0  # temporal solver: -1 uses the static storm calculation
    block_width_m: ArrayInput = 35.0
    hollow_confinement_ratio: ArrayInput = 1.0


@dataclass
class PhysicsState:
    theta_s: np.ndarray
    theta_1500: np.ndarray
    theta_33: np.ndarray
    hydraulic_conductivity_m_s: np.ndarray
    friction_angle_deg: np.ndarray
    soil_cohesion_kpa: np.ndarray
    regolith_depth_m: np.ndarray
    snowmelt_mm_day: np.ndarray
    thaw_factor: np.ndarray
    root_cohesion_kpa: np.ndarray
    fire_runoff_multiplier: np.ndarray
    effective_conductivity_m_s: np.ndarray
    effective_infiltration_m_s: np.ndarray
    local_saturation: np.ndarray
    hydraulic_diffusivity_m2_s: np.ndarray
    dimensionless_time: np.ndarray
    iverson_response: np.ndarray
    pore_pressure_head_m: np.ndarray
    toe_undercut_multiplier: np.ndarray
    wet_unit_weight_kn_m3: np.ndarray
    effective_cohesion_kpa: np.ndarray
    kh: np.ndarray
    fs_static: np.ndarray
    fs_dynamic: np.ndarray
    sigma_fs: np.ndarray
    beta_fosm: np.ndarray
    p_failure_physical: np.ndarray
    critical_acceleration_g: np.ndarray
    newmark_displacement_cm: np.ndarray
    unburned_regolith_depth_m: np.ndarray
    fs_static_dry: np.ndarray
    p_failure_seismic: np.ndarray
    p_failure_fosm: np.ndarray
    effective_slope_deg: np.ndarray
    effective_soil_cohesion_kpa: np.ndarray
    fire_driving_multiplier: np.ndarray
    fs_mohr_coulomb: np.ndarray
    fs_postfire_yield_index: np.ndarray
    p_failure_m1: np.ndarray
    debris_volume_m3: np.ndarray
    rain_15min_proxy_mm: np.ndarray
    suction_cohesion_kpa: np.ndarray
    cryo_perched_head_m: np.ndarray
    ice_friction_factor: np.ndarray
    equivalent_water_mm: np.ndarray
    effective_burn_severity: np.ndarray
    cryospheric_eligible: np.ndarray
    lateral_resistance_kpa: np.ndarray
    warnings: List[str] = field(default_factory=list)

    @property
    def cell_count(self) -> int:
        return int(self.fs_dynamic.size)

    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        for key, value in result.items():
            if isinstance(value, np.ndarray):
                result[key] = value.tolist()
        return result


class MultiCausalEngine:
    """Evaluate soil hydraulics, five causal mechanisms, FS, FOSM, and Newmark."""

    def compute(self, inputs: PhysicsInputs) -> PhysicsState:
        values = self._broadcast_inputs(inputs)
        sand = np.clip(values["sand_fraction"], 0.05, 0.85)
        clay = np.clip(values["clay_fraction"], 0.05, 0.65)
        organic = 100.0 * np.clip(values["organic_matter_fraction"], 0.005, 0.06)  # coefficients use percent OM
        bulk_density = np.clip(values["bulk_density_g_cm3"], 1.15, 1.70)
        slope_deg = np.clip(values["slope_deg"], 0.0, 89.9)
        curvature = np.abs(values["profile_curvature"])
        amplified = slope_deg * (1.0 + 0.20 * np.clip(curvature / 0.015, 0.25, 1.0))
        # Never flatten a DEM slope already steeper than the 48 degree scenario cap.
        effective_slope = np.where(slope_deg >= 15.0, np.maximum(slope_deg, np.minimum(amplified, 48.0)), slope_deg)
        slope = np.radians(effective_slope)
        sin_slope, cos_slope = np.sin(slope), np.cos(slope)
        cos2 = cos_slope**2
        twi = values["twi"]
        initial_saturation = np.clip(values["soil_saturation_fraction"], 0.0, 1.0)
        rain = np.maximum(values["rain_24h_mm"], 0.0)
        storm_hr = values["storm_duration_hr"]
        snow_depth = np.maximum(values["snow_depth_m"], 0.0)
        air_temp = values["air_temperature_c"]
        soil_temp = values["soil_temperature_c"]
        freeze_cycles = np.maximum(values["freeze_thaw_cycles_7d"], 0.0)
        ndvi = np.clip(values["ndvi_norm"], 0.0, 1.0)
        burn = np.clip(values["burn_severity"], 0.0, 1.0) * np.clip(ndvi / .22, 0.0, 1.0)
        pga = np.maximum(values["pga_g"], 0.0)
        road_distance = np.maximum(values["road_cut_distance_m"], 0.0)
        river_distance = np.maximum(values["river_distance_m"], 0.0)
        if np.any(storm_hr <= 0):
            raise ValueError("storm_duration_hr must be positive for every cell")
        if np.any(twi <= 0):
            raise ValueError("TWI must be positive for every cell")

        # Saxton & Rawls 2006 equations, retaining the specified clip bounds.
        theta_s = np.clip(1.0 - bulk_density / PARTICLE_DENSITY_G_CM3, 0.34, 0.58)
        theta_1500_t = (
            -0.024 * sand
            + 0.487 * clay
            + 0.006 * organic
            + 0.005 * sand * organic
            - 0.013 * clay * organic
            + 0.068 * sand * clay
            + 0.031
        )
        theta_1500 = np.clip(theta_1500_t + 0.14 * theta_1500_t - 0.02, 0.05, 0.38)
        theta_33_t = (
            -0.251 * sand
            + 0.195 * clay
            + 0.011 * organic
            + 0.006 * sand * organic
            - 0.027 * clay * organic
            + 0.452 * sand * clay
            + 0.299
        )
        theta_33_raw = theta_33_t + 1.283 * theta_33_t**2 - 0.374 * theta_33_t - 0.015
        theta_33 = np.clip(theta_33_raw, theta_1500 + 0.05, theta_s - 0.04)
        retention_gap = np.log(theta_33) - np.log(theta_1500)
        b_parameter = np.divide(
            math.log(1500.0) - math.log(33.0),
            retention_gap,
            out=np.full_like(retention_gap, np.inf),
            where=np.abs(retention_gap) > 1e-12,
        )
        lambda_parameter = np.divide(
            1.0,
            b_parameter,
            out=np.zeros_like(b_parameter),
            where=np.isfinite(b_parameter) & (np.abs(b_parameter) > 1e-12),
        )
        ksat_raw = (
            1930.0
            * np.power(np.maximum(theta_s - theta_33, 1e-12), 3.0 - lambda_parameter)
            * (1e-3 / 3600.0)
        )
        ksat = np.clip(ksat_raw, 1e-7, 1e-3)
        friction_deg = np.clip(24.0 + 14.0 * sand - 8.0 * clay, 18.0, 39.0)
        friction_deg += values["geology_friction_adjustment_deg"]
        tan_phi = np.tan(np.radians(friction_deg))
        cohesion_soil = np.clip(4.0 + 16.0 * clay - 3.5 * sand, 2.0, 20.0)
        cohesion_soil = np.maximum(0.,cohesion_soil + values["geology_cohesion_adjustment_kpa"])
        base_depth = np.clip(2.8 * np.exp(-0.85 * np.tan(slope)), 0.60, 3.00)
        # Keep the slab thickness independent of the separate post-fire runoff branch.
        regolith_depth = base_depth.copy()  # slab depth is independent of the runoff branch

        snowmelt = np.where(
            (snow_depth > 0.01) & (air_temp > 0.0),
            np.maximum(0.0, 4.5 * air_temp + 0.0126 * rain * air_temp),
            0.0,
        )
        snowmelt = np.where(values["snowmelt_override_mm_day"] >= 0, values["snowmelt_override_mm_day"], snowmelt)
        from .alpine_weather import cryospheric_eligibility
        baseline_air = np.where(values['baseline_air_temperature_c'] > -998,
                                values['baseline_air_temperature_c'], air_temp-values['warming_delta_c'])
        observed_snow = np.where(values['observed_snow_depth_m'] >= 0,
                                 values['observed_snow_depth_m'], snow_depth)
        cryo_active = cryospheric_eligibility(observed_snow, baseline_air, values['elevation_m'], values['latitude_deg'])
        # Elevation is an eligibility screen, not evidence that an ice-filled joint exists.
        cryo_active &= (observed_snow > .005) | (freeze_cycles > 0) | (values['warming_delta_c'] > 0)
        _, thaw, ice_friction = ice_factors(values['elevation_m'],soil_temp,snow_depth,freeze_cycles,values['warming_delta_c'],eligible=cryo_active)
        tan_phi_base = tan_phi.copy()
        tan_phi = tan_phi * ice_friction
        friction_deg = np.degrees(np.arctan(tan_phi))
        root_cohesion_baseline = 12.0 * np.power(ndvi, 1.35)
        root_cohesion = root_cohesion_baseline * (1.0 - 0.90 * burn)
        fire_multiplier = np.ones_like(burn)
        keff = ksat.copy()

        median_twi = float(np.median(twi))
        # Project storm/convergence proxies, not a solved TRIGRS water balance.
        active_water_mm = rain + 1.35 * snowmelt
        wetness_convergence = 1.0 + (np.power(twi / median_twi, 0.28) - 1.0) * np.clip(active_water_mm / 35.0, 0.08, 1.0)
        storm_wetting = (1.0 - initial_saturation) * np.clip(active_water_mm / 210.0, 0.0, 0.85)
        local_saturation = np.clip((initial_saturation + storm_wetting) * wetness_convergence, 0.10, 0.99)

        fire_driving = np.ones_like(burn)
        soil_cohesion_effective = cohesion_soil.copy()
        infiltration = (
            rain * 1e-3 / (storm_hr * 3600.0) + snowmelt * 1e-3 / 86400.0
        ) * fire_multiplier
        hydraulic_diffusivity = 4.0 * keff * cos2 / np.maximum(theta_s * (1.0 - local_saturation), 0.015)
        dimensionless_time = storm_hr * 3600.0 * hydraulic_diffusivity / (regolith_depth**2)
        safe_time = np.maximum(dimensionless_time, 1e-12)
        iverson_response = (
            np.sqrt(safe_time / math.pi) * np.exp(-1.0 / safe_time)
            - self._erfc(1.0 / np.sqrt(safe_time))
        )
        s_fc = np.clip(theta_33 / np.maximum(theta_s, 0.30), 0.48, 0.76)
        drainage_gate = np.clip(active_water_mm / 15.0, 0.04, 1.0)
        mobile_water_fraction = drainage_gate * np.clip((local_saturation-s_fc) / np.maximum(1.0-s_fc,0.15),0.0,1.0)
        pressure_head = np.clip(
            regolith_depth * mobile_water_fraction * cos2
            + regolith_depth * np.minimum(infiltration / keff, 4.0) * iverson_response,
            0.0,
            regolith_depth * cos2,
        )

        # Bounded active-layer water storage above an assumed low-permeability base.
        # This mass-balance approximation is not the TRIGRS PDE solution.
        basal_ratio=np.clip(values['basal_conductivity_ratio'],0,1)
        perched=np.where(cryo_active & (snowmelt>0),
            np.minimum(regolith_depth*cos2, np.maximum(0.,infiltration-ksat*basal_ratio)*storm_hr*3600/np.maximum(theta_s,.01)),0.)
        pressure_head=np.minimum(regolith_depth*cos2,pressure_head+perched)
        override=values["pore_pressure_override_m"]
        pressure_head=np.where(override>=0,np.clip(override,0,regolith_depth*cos2),pressure_head)
        se=np.clip((local_saturation-.08)/.92,.02,.999)
        alpha=values['vg_alpha_kpa_inv'];n=values['vg_n']
        if np.any(alpha<=0) or np.any(n<=1):raise ValueError('Invalid van Genuchten parameters')
        suction_stress=-se/alpha*np.expm1(np.clip(np.log(se)*n/(1-n),0.,700.))**(1/n)
        taper=np.clip((.85-se)/.10,0,1);taper=taper*taper*(3-2*taper)
        storm_activation=np.clip((rain+snowmelt)/25.,0,1)
        suction=np.clip(-suction_stress*tan_phi,0,8)*taper*(1-storm_activation)
        suction=np.where(perched>0,0.,suction)
        p_m1,volume,fs_fire,r15=postfire(burn,effective_slope,clay,sand,rain,values['peak_3h_intensity_mmhr'],base_depth,values['rain_15min_override_mm'])

        toe_multiplier = (
            1.0
            + 0.35 * np.exp(-road_distance / 60.0) * (slope_deg > 16.0)
            + 0.15 * np.exp(-river_distance / 80.0) * (twi / 15.0) * np.where((twi>=14.5)&(values["river_surge_ratio"]>1.3),np.sqrt(np.clip(values["river_surge_ratio"],1,5)),1.)
        )
        wet_unit_weight = (bulk_density + local_saturation * theta_s) * GAMMA_W_KN_M3
        effective_cohesion = (soil_cohesion_effective + root_cohesion) * thaw + suction
        kh = 0.50 * pga

        lateral = np.zeros_like(regolith_depth)  # compatibility output; no lateral force
        normal_static = wet_unit_weight * regolith_depth * cos2 - pressure_head * GAMMA_W_KN_M3
        driving_static = wet_unit_weight * regolith_depth * sin_slope * cos_slope * toe_multiplier * fire_driving
        fs_static_raw = (effective_cohesion + normal_static * tan_phi) / np.maximum(driving_static, 1.0)
        fs_static = np.clip(fs_static_raw, 0.10, 5.00)

        normal_dynamic = (
            wet_unit_weight * regolith_depth * (cos2 - kh * sin_slope * cos_slope)
            - pressure_head * GAMMA_W_KN_M3
        )
        driving_dynamic = (
            wet_unit_weight
            * regolith_depth
            * (sin_slope * cos_slope + kh * cos2)
            * toe_multiplier * fire_driving
        )
        denominator_dynamic = np.maximum(driving_dynamic, 1.0)
        fs_dynamic_raw = (effective_cohesion + normal_dynamic * tan_phi) / denominator_dynamic
        fs_mohr = np.clip(fs_dynamic_raw, 0.10, 5.00)
        fs_dynamic = fs_mohr.copy()

        # Delta-method FOSM, including the derivative of the specified FS clip.
        sigma_cohesion = 0.18 * effective_cohesion
        lateral_coefficient = 0.0
        sigma_tan_phi = 0.12 * tan_phi
        sigma_pressure = 0.15 * pressure_head
        clip_active = (fs_dynamic_raw > 0.10) & (fs_dynamic_raw < 5.00)
        d_fs_dc = np.where(clip_active, 1.0 / denominator_dynamic, 0.0)
        d_fs_dtan = np.where(clip_active, normal_dynamic / denominator_dynamic, 0.0)
        d_fs_dpressure = np.where(
            clip_active, -GAMMA_W_KN_M3 * tan_phi / denominator_dynamic, 0.0
        )
        sigma_fs = np.sqrt(
            (d_fs_dc * sigma_cohesion) ** 2
            + (d_fs_dtan * sigma_tan_phi) ** 2
            + (d_fs_dpressure * sigma_pressure) ** 2
        )
        beta = (fs_mohr - 1.0) / np.maximum(sigma_fs, 0.05)
        p_failure = 0.5 * self._erfc(beta / math.sqrt(2.0))

        # Pre-storm, unsaturated, unburned baseline; excludes seismic load and thaw.
        dry_gamma = bulk_density * GAMMA_W_KN_M3
        dry_normal = dry_gamma * base_depth * cos2
        dry_driving = dry_gamma * base_depth * sin_slope * cos_slope * toe_multiplier
        fs_static_dry = np.clip((cohesion_soil + root_cohesion_baseline + dry_normal * tan_phi_base)
                               / np.maximum(dry_driving, 1.0), 0.10, 5.0)
        # Project static-state yield with a heuristic 0.75 factor; geology is still included.
        critical_acceleration = np.clip((fs_static - 1.0) * sin_slope * 0.75, 0.05, 0.60)
        pga_ratio = np.clip(
            critical_acceleration / np.maximum(pga, 1e-12), 1e-12, 1.0 - 1e-12
        )
        newmark = np.where(
            pga > critical_acceleration,
            np.power(10.0, 0.215 + 2.341 * np.log10(1.0 - pga_ratio) - 1.438 * np.log10(pga_ratio)),
            0.0,
        )

        # Project-defined detachment cap; retain a separate regional empirical component.
        newmark = np.minimum(newmark, 120.0)
        p_seismic = 0.335 * (-np.expm1(-0.048 * np.power(newmark, 1.565)))
        # Requested probabilistic union; this treats the three components as independent.
        p_fosm = p_failure.copy()
        p_failure = 1.0-(1.0-p_fosm)*(1.0-p_seismic)*(1.0-p_m1)
        warnings = ["M1/Gartner inputs are cell proxies; mechanical FS is separate from surface runoff initiation. Ice weakening, basal drainage and VG parameters are scenario assumptions. Probability union assumes independent components."]
        reversed_bounds = np.where(theta_33 <= theta_1500)[0]
        if reversed_bounds.size:
            warnings.append(
                "Saxton-Rawls bounds produced theta_33 <= theta_1500 for {} cell(s); configured clipping was retained".format(
                    int(reversed_bounds.size)
                )
            )
        return PhysicsState(
            theta_s=theta_s,
            theta_1500=theta_1500,
            theta_33=theta_33,
            hydraulic_conductivity_m_s=ksat,
            friction_angle_deg=friction_deg,
            soil_cohesion_kpa=cohesion_soil,
            regolith_depth_m=regolith_depth,
            snowmelt_mm_day=snowmelt,
            thaw_factor=thaw,
            root_cohesion_kpa=root_cohesion,
            fire_runoff_multiplier=fire_multiplier,
            effective_conductivity_m_s=keff,
            effective_infiltration_m_s=infiltration,
            local_saturation=local_saturation,
            hydraulic_diffusivity_m2_s=hydraulic_diffusivity,
            dimensionless_time=dimensionless_time,
            iverson_response=iverson_response,
            pore_pressure_head_m=pressure_head,
            toe_undercut_multiplier=toe_multiplier,
            wet_unit_weight_kn_m3=wet_unit_weight,
            effective_cohesion_kpa=effective_cohesion,
            kh=kh,
            fs_static=fs_static,
            fs_dynamic=fs_dynamic,
            sigma_fs=sigma_fs,
            beta_fosm=beta,
            p_failure_physical=p_failure,
            critical_acceleration_g=critical_acceleration,
            newmark_displacement_cm=newmark,
            effective_slope_deg=effective_slope, effective_soil_cohesion_kpa=soil_cohesion_effective,
            fire_driving_multiplier=fire_driving,
            unburned_regolith_depth_m=base_depth, fs_static_dry=fs_static_dry,
            p_failure_seismic=p_seismic, p_failure_fosm=p_fosm, warnings=warnings,
            fs_mohr_coulomb=fs_mohr, fs_postfire_yield_index=fs_fire, p_failure_m1=p_m1,
            debris_volume_m3=volume, rain_15min_proxy_mm=r15, suction_cohesion_kpa=suction,
            cryo_perched_head_m=perched, ice_friction_factor=ice_friction,
            equivalent_water_mm=rain+1.35*snowmelt,
            effective_burn_severity=burn, cryospheric_eligible=cryo_active, lateral_resistance_kpa=lateral,
        )

    @staticmethod
    def _broadcast_inputs(inputs: PhysicsInputs) -> Dict[str, np.ndarray]:
        arrays = {}
        for name in PhysicsInputs.__dataclass_fields__:
            value = np.asarray(getattr(inputs, name), dtype=np.float64)
            if value.ndim > 1:
                raise ValueError("{} must be a scalar or one-dimensional per-cell array".format(name))
            arrays[name] = np.atleast_1d(value)
        broadcasted = np.broadcast_arrays(*arrays.values())
        result = {name: array.copy() for name, array in zip(arrays, broadcasted)}
        for name, array in result.items():
            if not np.isfinite(array).all():
                raise ValueError("{} contains missing or non-finite values; provide explicit observations/scenarios".format(name))
        return result

    @staticmethod
    def _erfc(values: np.ndarray) -> np.ndarray:
        # math.erfc is correctly rounded by the platform libm; vectorize without SciPy dependency.
        function = np.frompyfunc(math.erfc, 1, 1)
        return function(values).astype(np.float64)
