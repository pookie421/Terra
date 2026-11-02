"""Regime-level stress decomposition and location-specific threshold cards."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, replace
from typing import Dict, List, Mapping, Optional

import numpy as np

from .multi_causal_engine import GAMMA_W_KN_M3, MultiCausalEngine, PhysicsInputs, PhysicsState

CAUSES = (
    "hydro_meteorological",
    "seismic_tectonic",
    "cryospheric",
    "post_wildfire_biotic",
    "anthropogenic_fluvial",
)


@dataclass
class CausalBreakdown:
    delta_tau_kpa: Dict[str, np.ndarray]
    physical_percentages: Dict[str, np.ndarray]
    shap_percentages: Optional[Dict[str, np.ndarray]]
    combined_percentages: Dict[str, np.ndarray]
    absolute_shap_log_odds: Optional[Dict[str, np.ndarray]]
    valid: np.ndarray

    def to_dict(self) -> Dict[str, object]:
        return {
            "delta_tau_kpa": {key: value.tolist() for key, value in self.delta_tau_kpa.items()},
            "physical_percentages": {key: value.tolist() for key, value in self.physical_percentages.items()},
            "shap_percentages": None
            if self.shap_percentages is None
            else {key: value.tolist() for key, value in self.shap_percentages.items()},
            "combined_percentages": {key: value.tolist() for key, value in self.combined_percentages.items()},
            "absolute_shap_log_odds": None
            if self.absolute_shap_log_odds is None
            else {key: value.tolist() for key, value in self.absolute_shap_log_odds.items()},
            "valid": self.valid.tolist(),
        }


@dataclass
class NumbersThatMatterItem:
    metric: str
    value: float
    unit: str
    threshold: str
    status: str
    causal_group: str
    weight_percent: float

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


def group_shap_by_cause(shap_values, feature_names):
    """Partition absolute raw-margin SHAP; shared stability features split equally.

    This is an explicit display allocation, not identification of causal effects.
    Every column is accounted for exactly once across the five groups.
    """
    shap = np.abs(np.asarray(shap_values, dtype=float))
    if shap.ndim != 2 or shap.shape[1] != len(feature_names) or not np.isfinite(shap).all():
        raise ValueError('SHAP values must match finite feature columns')
    groups = {
        'hydro_meteorological': ('psi_head_m','soil_saturation','rain_24h_mm','twi'),
        'seismic_tectonic': ('pga_g','newmark_dn_cm'),
        'cryospheric': ('snowmelt_mm_day','thaw_degradation'),
        'post_wildfire_biotic': ('burn_severity',),
        'anthropogenic_fluvial': ('toe_undercut_mult','profile_curvature'),
    }
    result = {cause: np.zeros(shap.shape[0]) for cause in CAUSES}
    for column, name in enumerate(feature_names):
        owners = [cause for cause, names in groups.items() if name in names] or list(CAUSES)
        for cause in owners:
            result[cause] += shap[:, column] / len(owners)
    return result


def explain_causes(
    inputs: PhysicsInputs,
    state: PhysicsState,
    shap_values: Optional[np.ndarray] = None,
    feature_names: Optional[List[str]] = None,
    shap_by_cause: Optional[Mapping[str, np.ndarray]] = None,
) -> CausalBreakdown:
    """Return five physical shear reductions and optional absolute TreeSHAP shares.

    Physical contributions remain in kPa. They are normalized separately from
    dimensionless SHAP log-odds values; the combined percentage is the mean of
    the two normalized distributions when both are available.
    """
    slope = np.radians(np.clip(state.effective_slope_deg, 0.0, 89.9))
    ndvi = np.clip(np.asarray(inputs.ndvi_norm, dtype=np.float64), 0.0, 1.0)
    burn = state.effective_burn_severity
    tan_phi = np.tan(np.radians(state.friction_angle_deg))
    root_cohesion_unburned = 12.0 * ndvi**1.35
    slope_cos2 = np.cos(slope) ** 2
    slope_deg = np.asarray(inputs.slope_deg, dtype=np.float64)
    twi = np.maximum(np.asarray(inputs.twi, dtype=np.float64), 0.01)
    road_distance = np.broadcast_to(np.asarray(inputs.road_cut_distance_m, dtype=np.float64), (state.cell_count,))
    river_distance = np.broadcast_to(np.asarray(inputs.river_distance_m, dtype=np.float64), (state.cell_count,))
    field_capacity_saturation = np.clip(state.theta_33 / np.maximum(state.theta_s, 1e-6), 0.0, 0.95)
    field_capacity_head = np.zeros(state.cell_count)  # positive head already excludes field-capacity storage
    excess_pore_pressure_head = np.maximum(0.0, state.pore_pressure_head_m - field_capacity_head)
    snowmelt_activation = np.clip(state.snowmelt_mm_day / 60.0, 0.0, 0.75)
    cryospheric_degradation = np.maximum(1.0 - state.thaw_factor, snowmelt_activation)
    if np.any(burn > 0.0):
        unburned_state = MultiCausalEngine().compute(replace(inputs, burn_severity=np.zeros_like(burn)))
        unburned_excess_head = np.maximum(
            0.0, unburned_state.pore_pressure_head_m * state.regolith_depth_m / unburned_state.regolith_depth_m - field_capacity_head
        )
        fire_amplified_excess_head = np.maximum(
            0.0, excess_pore_pressure_head - unburned_excess_head
        )
        fire_runoff_stress = fire_amplified_excess_head * GAMMA_W_KN_M3 * tan_phi
    else:
        fire_runoff_stress = np.zeros(state.cell_count, dtype=np.float64)
    road_undercut = 0.35 * np.exp(-road_distance / 60.0) * (slope_deg > 16.0)
    # River distance is estimated from high-TWI DEM cells, not a mapped stream;
    # discount that proxy contribution while retaining measured road-cut effects.
    fluvial_proxy_undercut = 0.15 * np.exp(-river_distance / 80.0) * (twi / 15.0)
    fire_drag_stress = state.wet_unit_weight_kn_m3 * state.regolith_depth_m * (
        np.sin(slope)*np.cos(slope) + state.kh*slope_cos2
    ) * state.toe_undercut_multiplier * (state.fire_driving_multiplier - 1.0)
    fire_cohesion_loss = (state.soil_cohesion_kpa-state.effective_soil_cohesion_kpa + root_cohesion_unburned-state.root_cohesion_kpa) * state.thaw_factor
    stress = {
        "hydro_meteorological": np.maximum(
            0.0, excess_pore_pressure_head * GAMMA_W_KN_M3 * tan_phi - fire_runoff_stress
        ),
        "seismic_tectonic": np.maximum(
            0.0,
            state.wet_unit_weight_kn_m3
            * state.regolith_depth_m
            * state.kh
            * np.cos(slope) ** 2,
        ),
        "cryospheric": np.maximum(
            0.0,
            (state.effective_soil_cohesion_kpa + state.root_cohesion_kpa)
            * cryospheric_degradation,
        ),
        "post_wildfire_biotic": np.maximum(
            0.0, fire_cohesion_loss + fire_runoff_stress + fire_drag_stress + state.p_failure_m1 * state.wet_unit_weight_kn_m3 * state.regolith_depth_m * np.sin(slope)
        ),
        "anthropogenic_fluvial": np.maximum(
            0.0,
            state.wet_unit_weight_kn_m3
            * state.regolith_depth_m
            * np.sin(slope)
            * np.cos(slope)
            * (road_undercut + 0.5 * fluvial_proxy_undercut),
        ),
    }
    physical = _normalize(stress)
    if shap_values is not None:
        if shap_by_cause is not None or feature_names is None:
            raise ValueError('Provide either grouped SHAP or SHAP columns and names')
        shap_by_cause = group_shap_by_cause(shap_values, feature_names)
    shap_pct = None
    if shap_by_cause is not None:
        if set(shap_by_cause) != set(CAUSES):
            raise ValueError('SHAP groups must contain all five causes')
        shap_by_cause = {k: np.asarray(v, dtype=float) for k,v in shap_by_cause.items()}
        if any(v.shape != (state.cell_count,) or not np.isfinite(v).all() or np.any(v < 0) for v in shap_by_cause.values()):
            raise ValueError('Grouped SHAP must be nonnegative per-cell arrays')
        shap_pct = _normalize(shap_by_cause)

    if shap_pct is None:
        combined = physical
    else:
        physical_total = np.sum(np.stack(list(stress.values()), axis=0), axis=0)
        shap_total = np.sum(np.stack(list(shap_by_cause.values()), axis=0), axis=0)
        combined = {}
        for cause in CAUSES:
            combined[cause] = np.where(
                (physical_total > 0) & (shap_total > 0),
                0.5 * (physical[cause] + shap_pct[cause]),
                np.where(shap_total > 0, shap_pct[cause], physical[cause]),
            )
    valid = np.sum(np.stack(list(stress.values()), axis=0), axis=0) > 1e-8
    if shap_by_cause is not None:
        valid = valid | (np.sum(np.stack(list(shap_by_cause.values()), axis=0), axis=0) > 0)
    return CausalBreakdown(stress, physical, shap_pct, combined, shap_by_cause, valid)


def select_numbers_that_matter(
    index: int,
    inputs: PhysicsInputs,
    state: PhysicsState,
    breakdown: CausalBreakdown,
    probability_failure: Optional[np.ndarray] = None,
) -> List[NumbersThatMatterItem]:
    """Return exactly six complete threshold metrics weighted by active cause share."""
    if index < 0 or index >= state.cell_count:
        raise IndexError("cell index outside PhysicsState")
    slope = float(_value(inputs.slope_deg, index))
    pga = float(_value(inputs.pga_g, index))
    saturation = float(state.local_saturation[index])
    rain = float(_value(inputs.rain_24h_mm, index))
    snow = float(state.snowmelt_mm_day[index])
    burn = float(state.effective_burn_severity[index])
    freeze = float(_value(inputs.freeze_thaw_cycles_7d, index))
    prob = float(probability_failure[index]) if probability_failure is not None else float(state.p_failure_physical[index])
    cause_weights = {cause: float(breakdown.combined_percentages[cause][index]) for cause in CAUSES}
    candidates = [
        _item("dynamic_factor_of_safety", state.fs_dynamic[index], "ratio", "failure threshold 1.00; rapid-deformation watch 1.25", "critical" if state.fs_dynamic[index] <= 1.0 else ("watch" if state.fs_dynamic[index] <= 1.25 else "below threshold"), "hydro_meteorological", cause_weights["hydro_meteorological"]),
        _item(
            "blended_failure_probability" if probability_failure is not None else "physical_failure_probability",
            prob * 100.0,
            "%",
            "hotspot threshold 65%",
            "critical" if prob >= 0.65 else ("watch" if prob >= 0.20 else "below threshold"),
            "hydro_meteorological",
            cause_weights["hydro_meteorological"],
        ),
        _item("pore_pressure_head", state.pore_pressure_head_m[index], "m", "full saturation head {:.3f} m".format(state.regolith_depth_m[index] * math.cos(math.radians(slope)) ** 2), "critical" if state.pore_pressure_head_m[index] >= 0.8 * state.regolith_depth_m[index] * math.cos(math.radians(slope)) ** 2 else "below threshold", "hydro_meteorological", cause_weights["hydro_meteorological"]),
        _item("24_hour_rainfall", rain, "mm", "25 mm when burn severity ≥0.35 activates rapid-flow branch", "critical" if burn >= 0.35 and rain >= 25.0 else "below threshold", "hydro_meteorological", cause_weights["hydro_meteorological"]),
        _item("local_soil_saturation", saturation * 100.0, "%", "68% rapid-flow branch threshold", "critical" if saturation >= 0.68 else "below threshold", "hydro_meteorological", cause_weights["hydro_meteorological"]),
        _item("pga", pga, "g", "critical acceleration {:.4f} g".format(state.critical_acceleration_g[index]), "critical" if pga > state.critical_acceleration_g[index] else "below threshold", "seismic_tectonic", cause_weights["seismic_tectonic"]),
        _item("newmark_displacement", state.newmark_displacement_cm[index], "cm", "2 cm coseismic velocity branch threshold", "critical" if state.newmark_displacement_cm[index] >= 2.0 else "below threshold", "seismic_tectonic", cause_weights["seismic_tectonic"]),
        _item("snowmelt_flux", snow, "mm/day", "positive melt requires snow depth >0.01 m and air temperature >0°C", "active" if snow > 0 else "inactive", "cryospheric", cause_weights["cryospheric"]),
        _item("freeze_thaw_cycles_7d", freeze, "cycles", "thaw degradation increases at ≥1 cycle", "active" if freeze >= 1 else "inactive", "cryospheric", cause_weights["cryospheric"]),
        _item("burn_severity", burn, "0–1", "0.35 for fire-rain rapid-flow branch", "critical" if burn >= 0.35 else "below threshold", "post_wildfire_biotic", cause_weights["post_wildfire_biotic"]),
        _item("root_cohesion_after_fire", state.root_cohesion_kpa[index], "kPa", "computed from NDVI and burn severity", "derived", "post_wildfire_biotic", cause_weights["post_wildfire_biotic"]),
        _item("toe_undercut_multiplier", state.toe_undercut_multiplier[index], "ratio", "1.0 baseline; >1 adds undercut demand", "active" if state.toe_undercut_multiplier[index] > 1.0 else "inactive", "anthropogenic_fluvial", cause_weights["anthropogenic_fluvial"]),
        _item("slope_angle", slope, "degrees", "16° road-cut; 28° coseismic branch thresholds", "critical" if slope >= 28 else ("watch" if slope > 16 else "below threshold"), "anthropogenic_fluvial", cause_weights["anthropogenic_fluvial"]),
    ]
    core=[replace(item,causal_group='core_stability') for item in candidates[:2]]
    ranked=sorted(candidates[2:],key=lambda item:(item.weight_percent,item.status in ('CRITICAL','WATCH')),reverse=True)
    chosen=[];seen=set()
    for item in ranked:
        if item.causal_group not in seen:
            chosen.append(item);seen.add(item.causal_group)
        if len(chosen)==4:break
    return core+chosen


def _normalize(values: Mapping[str, np.ndarray]) -> Dict[str, np.ndarray]:
    clean = {key: np.maximum(0.0, np.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0)) for key,value in values.items()}
    total = np.sum(np.stack(list(clean.values()), axis=0), axis=0)
    # A neutral prior preserves the UI contract when no stress is active; valid=False signals no attribution.
    return {key: np.divide(value*100.0,total,out=np.full_like(value,100.0/len(clean),dtype=float),where=total>1e-8) for key,value in clean.items()}


def _value(value: object, index: int) -> float:
    array = np.atleast_1d(np.asarray(value, dtype=np.float64))
    return float(array[0] if array.size == 1 else array[index])


def _item(metric: str, value: object, unit: str, threshold: str, status: str, cause: str, weight: float) -> NumbersThatMatterItem:
    status = {"critical":"CRITICAL", "watch":"WATCH", "active":"WATCH"}.get(status, "NORMAL")
    numeric = float(value)
    return NumbersThatMatterItem(metric, numeric if math.isfinite(numeric) else 0.0, unit, threshold, status, cause, float(weight))
