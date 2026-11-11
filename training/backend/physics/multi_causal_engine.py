"""Vectorized five-regime slope-stability equations for H3 cells."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Union

import numpy as np
from .mechanisms import staley_m1_probability, gartner_volume_m3
from .reference_column import response_function, critical_acceleration_g, jibson_displacement_cm, jibson_failure_probability

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
    # Deprecated compatibility inputs: bedrock context never modifies regolith strength.
    geology_friction_adjustment_deg: ArrayInput = 0.0
    geology_cohesion_adjustment_kpa: ArrayInput = 0.0
    rain_15min_override_mm: ArrayInput = -1.0
    snowmelt_override_mm_day: ArrayInput = -1.0
    pore_pressure_override_m: ArrayInput = -1.0  # temporal solver: -1 uses the static storm calculation
    block_width_m: ArrayInput = 35.0
    hollow_confinement_ratio: ArrayInput = 1.0
    transient_pressure_head_m: ArrayInput = -1.0  # replace transient response only
    antecedent_pressure_head_m: ArrayInput = 0.0  # specified initial head, not inferred from wetness
    specific_moisture_capacity_m_inv: ArrayInput = -1.0
    horizontal_seismic_coefficient: ArrayInput = 0.0
    postfire_burned_steep_fraction: ArrayInput = -1.0
    postfire_mean_dnbr: ArrayInput = -1.0  # dNBR in the scaled 0..1000 convention
    postfire_soil_erodibility_kf: ArrayInput = -1.0
    postfire_burned_area_km2: ArrayInput = -1.0
    postfire_basin_relief_m: ArrayInput = -1.0
    weather_reference_elevation_m: ArrayInput = -999.0
    runout_travel_length_m: ArrayInput = 0.0  # unavailable geometry => no predicted speed


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
    newmark_applicable: Optional[np.ndarray] = None
    postfire_basin_inputs_available: Optional[np.ndarray] = None
    specific_moisture_capacity_m_inv: Optional[np.ndarray] = None

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
        organic = 100.0 * values["organic_matter_fraction"]  # coefficients use percent OM
        bulk_density = values["bulk_density_g_cm3"]
        if np.any((bulk_density <= 0) | (bulk_density >= PARTICLE_DENSITY_G_CM3)):
            raise ValueError('Bulk density must define a positive pore volume')
        slope_deg = np.clip(values["slope_deg"], 0.0, 89.9)
        effective_slope = slope_deg.copy()
        slope = np.radians(effective_slope)
        sin_slope, cos_slope = np.sin(slope), np.cos(slope)
        cos2 = cos_slope**2
        twi = values["twi"]
        initial_saturation = values["soil_saturation_fraction"]
        if np.any((initial_saturation < 0) | (initial_saturation > 1)):
            raise ValueError('Absolute saturation must be compatible with the selected soil porosity')
        rain = np.maximum(values["rain_24h_mm"], 0.0)
        storm_hr = values["storm_duration_hr"]
        snow_depth = np.maximum(values["snow_depth_m"], 0.0)
        air_temp = values["air_temperature_c"]
        soil_temp = values["soil_temperature_c"]
        freeze_cycles = np.maximum(values["freeze_thaw_cycles_7d"], 0.0)
        ndvi = np.clip(values["ndvi_norm"], 0.0, 1.0)
        burn = np.clip(values["burn_severity"], 0.0, 1.0)
        pga = np.maximum(values["pga_g"], 0.0)
        road_distance = np.maximum(values["road_cut_distance_m"], 0.0)
        river_distance = np.maximum(values["river_distance_m"], 0.0)
        if np.any(storm_hr <= 0):
            raise ValueError("storm_duration_hr must be positive for every cell")
        if np.any(twi <= 0):
            raise ValueError("TWI must be positive for every cell")

        # Saxton & Rawls 2006 equations, retaining the specified clip bounds.
        theta_s = 1.0 - bulk_density / PARTICLE_DENSITY_G_CM3
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
        tan_phi = np.tan(np.radians(friction_deg))
        cohesion_soil = np.clip(4.0 + 16.0 * clay - 3.5 * sand, 2.0, 20.0)
        base_depth = np.clip(2.8 * np.exp(-0.85 * np.tan(slope)), 0.60, 3.00)
        # Keep the slab thickness independent of the separate post-fire runoff branch.
        regolith_depth = base_depth.copy()  # slab depth is independent of the runoff branch

        snowmelt = np.where(
            (snow_depth > 0.01) & (air_temp > 0.0),
            np.maximum(0.0, 4.5 * air_temp + 0.0126 * rain * air_temp),
            0.0,
        )
        snowmelt = np.where(values["snowmelt_override_mm_day"] >= 0, values["snowmelt_override_mm_day"], snowmelt)
        # No unsurveyed ice-joint, root, toe-erosion, or wet-strength modifiers.
        # Their compatibility fields remain present; independent observations are
        # required before introducing these mechanisms into reference mechanics.
        cryo_active = np.zeros_like(slope_deg, dtype=bool)
        thaw = np.ones_like(slope_deg)
        ice_friction = np.ones_like(slope_deg)
        tan_phi_base = tan_phi.copy()
        root_cohesion_baseline = np.zeros_like(slope_deg)
        root_cohesion = root_cohesion_baseline.copy()
        fire_multiplier = np.ones_like(slope_deg)
        keff = ksat.copy()
        local_saturation = initial_saturation.copy()
        active_water_mm = rain + snowmelt
        fire_driving = np.ones_like(slope_deg)
        soil_cohesion_effective = cohesion_soil.copy()
        infiltration = np.minimum(rain*1e-3/(storm_hr*3600.) + snowmelt*1e-3/86400., keff)
        alpha = values['vg_alpha_kpa_inv']
        if np.any(alpha <= 0):
            raise ValueError('Characteristic retention alpha must be positive')
        # Linearized capacity across a characteristic suction head hc=1/(alpha*gamma_w).
        # This is a disclosed pedotransfer approximation, not measured diffusivity.
        # Explicit capacity overrides the approximation. C [1/m], Ks/C [m²/s].
        capillary_head_m = 1. / (alpha * GAMMA_W_KN_M3)
        supplied_capacity = values['specific_moisture_capacity_m_inv']
        if np.any((supplied_capacity != -1.) & (supplied_capacity <= 0.)):
            raise ValueError('Specific moisture capacity must be positive, or -1 for the declared approximation')
        capacity = np.where(values['specific_moisture_capacity_m_inv'] > 0,
                            values['specific_moisture_capacity_m_inv'], theta_s/capillary_head_m)
        hydraulic_diffusivity = keff / capacity
        dimensionless_time = 4.*hydraulic_diffusivity*cos2*storm_hr*3600./regolith_depth**2
        iverson_response = response_function(dimensionless_time)
        transient_head = np.where(values['transient_pressure_head_m'] >= 0,
                                   values['transient_pressure_head_m'],
                                   regolith_depth*infiltration/keff*iverson_response)
        pressure_head = np.minimum(regolith_depth*cos2,
                                   values['antecedent_pressure_head_m'] + transient_head)
        override = values['pore_pressure_override_m']
        pressure_head = np.where(override >= 0, np.minimum(override, regolith_depth*cos2), pressure_head)
        perched = np.zeros_like(pressure_head)
        suction = np.zeros_like(pressure_head)
        s_fc = theta_33 / theta_s
        toe_multiplier = np.ones_like(slope_deg)
        r15 = np.where(values['rain_15min_override_mm'] >= 0,
                        values['rain_15min_override_mm'], .25*values['peak_3h_intensity_mmhr'])
        fire_available = ((values['postfire_burned_steep_fraction'] >= 0) &
                          (values['postfire_mean_dnbr'] >= 0) &
                          (values['postfire_soil_erodibility_kf'] >= 0) &
                          (values['rain_15min_override_mm'] >= 0))
        p_m1 = np.zeros_like(slope_deg)
        if fire_available.any():
            p_m1[fire_available] = staley_m1_probability(r15[fire_available],
                values['postfire_burned_steep_fraction'][fire_available],
                values['postfire_mean_dnbr'][fire_available], values['postfire_soil_erodibility_kf'][fire_available])
        volume = np.zeros_like(slope_deg)
        volume_available = fire_available & (values['postfire_burned_area_km2'] > 0) & (values['postfire_basin_relief_m'] >= 0)
        if volume_available.any():
            volume[volume_available] = gartner_volume_m3(4*r15[volume_available],
                values['postfire_burned_area_km2'][volume_available], values['postfire_basin_relief_m'][volume_available])
        fs_fire = np.full_like(slope_deg, 5.)  # deprecated compatibility diagnostic
        # Reference moist slab weight is fixed during a rain perturbation. This
        # decouples the approximate pressure response from storm-driven mass gain.
        wet_unit_weight = (bulk_density + initial_saturation * theta_s) * GAMMA_W_KN_M3
        effective_cohesion = (soil_cohesion_effective + root_cohesion) * thaw + suction
        kh = values["horizontal_seismic_coefficient"]  # explicit load only; PGA is used in Newmark

        lateral = np.zeros_like(regolith_depth)  # compatibility output; no lateral force
        normal_static = wet_unit_weight * regolith_depth * cos2 - pressure_head * GAMMA_W_KN_M3
        driving_static = wet_unit_weight * regolith_depth * sin_slope * cos_slope * toe_multiplier * fire_driving
        fs_static_raw = (effective_cohesion + np.maximum(normal_static, 0.) * tan_phi) / np.maximum(driving_static, 1e-12)
        fs_static = fs_static_raw.copy()

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
        denominator_dynamic = np.maximum(driving_dynamic, 1e-12)
        fs_dynamic_raw = (effective_cohesion + np.maximum(normal_dynamic, 0.) * tan_phi) / denominator_dynamic
        fs_mohr = fs_dynamic_raw.copy()
        fs_dynamic = fs_mohr.copy()

        # Joint delta method, including geometry, hydraulics, water and forcing.
        from .reference_context import joint_fosm,angular_reliability
        from scipy.special import ndtr
        supplied_head=(values['pore_pressure_override_m']>=0)|(values['transient_pressure_head_m']>=0)
        reference_rain=rain+snowmelt*storm_hr/24.
        reference_initial=values['antecedent_pressure_head_m']
        reference_rain=np.where(supplied_head,0.,reference_rain)
        reference_initial=np.where(supplied_head,pressure_head,reference_initial)
        parameters=np.column_stack([effective_cohesion,friction_deg,wet_unit_weight,regolith_depth,
            keff,hydraulic_diffusivity,reference_initial,np.maximum(effective_slope,1e-6),reference_rain,kh])
        spreads=np.column_stack([.4*effective_cohesion+.5,np.full_like(slope_deg,4.),.08*wet_unit_weight,
            .35*regolith_depth,.5*keff,.6*hydraulic_diffusivity,.15*regolith_depth*cos2+.02,
            np.full_like(slope_deg,1.),.2*reference_rain+.5,np.zeros_like(kh)])
        correlation=np.eye(10);correlation[0,1]=correlation[1,0]=-.25;correlation[4,5]=correlation[5,4]=.6
        # Durations may differ by cell in the vectorized scenario interface.
        variance=np.zeros_like(slope_deg)
        for duration in np.unique(storm_hr):
            mask=storm_hr==duration
            variance[mask]=joint_fosm(parameters[mask],spreads[mask],duration*3600.,correlation)[0]
        sigma_fs=np.sqrt(variance)
        sigma_log=sigma_fs/np.maximum(fs_mohr,1e-12)
        beta=np.log(np.maximum(fs_mohr,1e-12))/np.maximum(sigma_log,1e-12)
        p_failure=ndtr(-beta)
        for duration in np.unique(storm_hr):
            mask=(storm_hr==duration)&(fs_mohr>1e-9)
            if mask.any():p_failure[mask]=angular_reliability(parameters[mask],spreads[mask],duration*3600.,correlation)[0]

        # Pre-storm, unsaturated, unburned baseline; excludes seismic load and thaw.
        dry_gamma = bulk_density * GAMMA_W_KN_M3
        dry_normal = dry_gamma * base_depth * cos2
        dry_driving = dry_gamma * base_depth * sin_slope * cos_slope * toe_multiplier
        fs_static_dry = (cohesion_soil + root_cohesion_baseline + dry_normal * tan_phi_base) / np.maximum(dry_driving, 1e-12)
        critical_acceleration = critical_acceleration_g(fs_static_raw, effective_slope)
        newmark_raw = jibson_displacement_cm(critical_acceleration, pga)
        newmark_applicable = (fs_static_raw > 1.) & (critical_acceleration > 0)
        # Numeric placeholder only for the legacy 14-feature interface. Public
        # cell output additionally carries applicability and null displacement.
        newmark = np.where(newmark_applicable, newmark_raw, 0.)
        p_seismic = np.where(newmark_applicable, jibson_failure_probability(newmark), 0.)
        p_fosm = p_failure.copy()
        # Components have different calibration populations and are dependent.
        # Keep them separate; the legacy physical score denotes FOSM only.
        p_failure = p_fosm.copy()
        warnings = ["Reference-column approximation; live material properties and capacity are pedotransfer estimates. "
                    "Initial pressure head defaults to an explicit zero-head scenario, not a wetness conversion. "
                    "FOSM spread is assumed, not calibrated event probability. M1 requires basin inputs. "
                    "Unavailable Newmark uses a zero feature placeholder and an explicit applicability flag. "
                    "No validated ice, root, erosion, or runout geometry is inferred from imagery or terrain."]
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
            equivalent_water_mm=rain+snowmelt,
            effective_burn_severity=burn, cryospheric_eligible=cryo_active, lateral_resistance_kpa=lateral,
            newmark_applicable=newmark_applicable, postfire_basin_inputs_available=fire_available,
            specific_moisture_capacity_m_inv=capacity,
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
