"""Local-cell feature schema; no neighboring cell overwrites."""
import numpy as np
from typing import Any
from backend.physics.multi_causal_engine import PhysicsInputs
from backend.ml.global_xgb import feature_matrix

def physics_feature_matrix(inputs: PhysicsInputs, state: Any, curvature: np.ndarray) -> np.ndarray:
    count = state.cell_count
    rain = np.broadcast_to(np.asarray(inputs.rain_24h_mm, dtype=np.float64), (count,))
    pga = np.broadcast_to(np.asarray(inputs.pga_g, dtype=np.float64), (count,))
    burn = np.broadcast_to(np.asarray(inputs.burn_severity, dtype=np.float64), (count,))
    feature_map = {
        "fs_dynamic": state.fs_dynamic, "p_f_fosm": state.p_failure_fosm,
        "psi_head_m": state.pore_pressure_head_m, "newmark_dn_cm": state.newmark_displacement_cm,
        "slope_max_deg": inputs.slope_deg, "twi": inputs.twi,
        "soil_saturation": state.local_saturation, "rain_24h_mm": state.equivalent_water_mm,
        "snowmelt_mm_day": state.snowmelt_mm_day, "pga_g": pga,
        "thaw_degradation": 1.0 - state.thaw_factor, "burn_severity": state.effective_burn_severity,
        "toe_undercut_mult": state.toe_undercut_multiplier,
        "profile_curvature": curvature,
    }
    return feature_matrix(feature_map)

