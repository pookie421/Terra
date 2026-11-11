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


def group_shap_by_cause(shap_values, feature_names, physical=None):
    """Allocate positive raw-margin evidence, not causal effects or probability SHAP.

    Composite columns follow per-cell physical stress shares. Zero-stress cells
    use an explicit equal allocation; all positive column mass is conserved.
    """
    shap = np.maximum(np.asarray(shap_values, dtype=float), 0.)
    if shap.ndim != 2 or shap.shape[1] != len(feature_names) or not np.isfinite(shap).all():
        raise ValueError('SHAP values must match finite feature columns')
    groups = {
        'hydro_meteorological': ('psi_head_m','soil_saturation','rain_24h_mm'),
        'seismic_tectonic': ('pga_g','newmark_dn_cm'),
        'cryospheric': ('snowmelt_mm_day','thaw_degradation'),
        'post_wildfire_biotic': ('burn_severity',),
        'anthropogenic_fluvial': ('toe_undercut_mult','twi'),
    }
    composite = {'fs_dynamic','p_f_fosm','slope_max_deg','profile_curvature'}
    weights = np.ones((len(CAUSES),len(shap))) if physical is None else np.stack([np.broadcast_to(physical[c],(len(shap),)) for c in CAUSES])
    if not np.isfinite(weights).all() or np.any(weights<0):raise ValueError('Invalid physical shares')
    total=weights.sum(axis=0)
    weights=np.divide(weights,total,out=np.full_like(weights,1/len(CAUSES),dtype=float),where=total>0)
    result = {cause: np.zeros(shap.shape[0]) for cause in CAUSES}
    for column, name in enumerate(feature_names):
        owners = [cause for cause, names in groups.items() if name in names]
        if owners:
            result[owners[0]] += shap[:, column]
        elif name in composite:
            for i,cause in enumerate(CAUSES):result[cause]+=shap[:,column]*weights[i]
        else:raise ValueError('Unmapped SHAP feature: '+name)
    return result


def explain_causes(
    inputs: PhysicsInputs,
    state: PhysicsState,
    shap_values: Optional[np.ndarray] = None,
    feature_names: Optional[List[str]] = None,
    shap_by_cause: Optional[Mapping[str, np.ndarray]] = None,
) -> CausalBreakdown:
    """Return five physical shear reductions and optional positive TreeSHAP shares.

    Physical contributions remain in kPa. They are normalized separately from
    dimensionless SHAP log-odds values; the combined percentage is the mean of
    the two normalized distributions when both are available.
    """
    slope = np.radians(np.clip(state.effective_slope_deg, 0.0, 89.9))
    ndvi = np.clip(np.asarray(inputs.ndvi_norm, dtype=np.float64), 0.0, 1.0)
    burn = state.effective_burn_severity
    tan_phi = np.tan(np.radians(state.friction_angle_deg))
    rain=np.broadcast_to(np.maximum(np.asarray(inputs.rain_24h_mm,float),0.),(state.cell_count,))
    melt=np.maximum(state.snowmelt_mm_day,0.)
    rain_share=np.divide(rain,rain+melt,out=np.ones_like(rain),where=(rain+melt)>0)
    head_stress=np.maximum(state.pore_pressure_head_m,0.)*GAMMA_W_KN_M3*tan_phi
    zeros=np.zeros(state.cell_count)
    # Only modeled terms receive mechanical attribution. M1 basin likelihood and
    # unobserved root/ice/erosion effects are not manufactured shear stresses.
    stress = {
        'hydro_meteorological':head_stress*rain_share,
        'cryospheric':head_stress*(1-rain_share),
        'seismic_tectonic':state.wet_unit_weight_kn_m3*state.regolith_depth_m*np.maximum(
            np.asarray(inputs.pga_g)-state.critical_acceleration_g,0.)*np.cos(slope)**2,
        'post_wildfire_biotic':zeros.copy(),
        'anthropogenic_fluvial':zeros.copy(),
    }
    physical = _normalize(stress)
    if shap_values is not None:
        if shap_by_cause is not None or feature_names is None:
            raise ValueError('Provide either grouped SHAP or SHAP columns and names')
        shap_by_cause = group_shap_by_cause(shap_values, feature_names, physical)
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
