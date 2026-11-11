"""Archived v14 assumption-based input builder for reference regression only.

Live CLI/API assessments use api.ground_assessment and do not execute this
wrapper's estimated material, initial-head or compatibility defaults.
"""
from typing import Optional
import numpy as np
from backend.physics.alpine_weather import apply_alpine_weather, ALPINE_ASSUMPTION
from backend.ml.global_xgb import FEATURE_NAMES, GlobalMonotonicXGB, blend_failure_probability
from backend.realtime.global_ingestor import GlobalScan
from backend.realtime.soilgrids import estimate_pedotransfer
from backend.physics.multi_causal_engine import MultiCausalEngine, PhysicsInputs
from backend.physics.kinematics import solve_kinematics
from backend.physics.causal_attribution import explain_causes, select_numbers_that_matter
from backend.physics.features import physics_feature_matrix as _feature_matrix


def evaluate_live_physics(scan: GlobalScan, model: Optional[GlobalMonotonicXGB] = None) -> Optional[dict]:
    """Compute a complete, provenance-marked physics state for every sampled land H3 cell."""
    if not scan.terrain or not scan.terrain.cells:
        return None
    cells = list(scan.terrain.cells.values())
    cell_ids = [cell.cell_id for cell in cells]
    count = len(cells)
    hydrology = scan.hydrology
    properties = hydrology.soil_properties if hydrology else {}
    if hydrology is None or hydrology.soil_saturation_pct is None:
        return None
    # Ingestion derives every field from one finalized raw soil payload. Do not
    # derive a second viewport-dependent soil prior after computing saturation.
    required=('sand_fraction','clay_fraction','organic_matter_fraction','bulk_density_g_cm3')
    if any(properties.get(key) is None for key in required):
        return None

    def soil_array(key: str, fallback_key: str) -> np.ndarray:
        value = properties.get(key)
        return np.full(count, float(value), dtype=np.float64)

    sand = soil_array("sand_fraction", "sand_fraction")
    clay = soil_array("clay_fraction", "clay_fraction")
    organic = soil_array("organic_matter_fraction", "organic_matter_fraction")
    bulk_density = soil_array("bulk_density_g_cm3", "bulk_density_g_cm3")

    weather = scan.weather
    weather_fallbacks = []
    rain = weather.rain_24h_mm if weather and weather.rain_24h_mm is not None else 0.0
    if weather is None or weather.rain_24h_mm is None:
        weather_fallbacks.append("rain_24h_mm=0.0 (unavailable observation)")
    air = weather.air_temperature_c if weather and weather.air_temperature_c is not None else 10.0
    if weather is None or weather.air_temperature_c is None:
        weather_fallbacks.append("air_temperature_c=10.0 (neutral fallback)")
    soil_temp = weather.soil_temperature_0_7cm_c if weather and weather.soil_temperature_0_7cm_c is not None else air
    if weather is None or weather.soil_temperature_0_7cm_c is None:
        weather_fallbacks.append("soil_temperature_c=air temperature / 10.0 fallback")
    snow = weather.snow_depth_m if weather and weather.snow_depth_m is not None else 0.0
    if weather is None or weather.snow_depth_m is None:
        weather_fallbacks.append("snow_depth_m=0.0 (unavailable observation)")
    freeze_cycles = weather.freeze_thaw_cycles_7d if weather and weather.freeze_thaw_cycles_7d is not None else 0
    if weather is None or weather.freeze_thaw_cycles_7d is None:
        weather_fallbacks.append("freeze_thaw_cycles_7d=0 (unavailable observation)")

    if hydrology is None or hydrology.soil_saturation_pct is None:
        # Preserve unavailable state. A relative index or incompatible theta/n
        # cannot supply an absolute hydraulic initial condition.
        return None
    saturation = float(hydrology.soil_saturation_pct / 100.0)
    saturation_source = "Compatible modeled volumetric moisture / selected soil porosity; not observed groundwater"

    satellite = scan.satellite_cells
    vegetation = np.asarray([
        float(satellite.get(cell_id, {}).get("vegetation_index_norm_proxy", 0.50)) for cell_id in cell_ids
    ], dtype=np.float64)
    burn = np.asarray([
        float(satellite.get(cell_id, {}).get("burn_severity_proxy", 0.0)) for cell_id in cell_ids
    ], dtype=np.float64)
    vegetation_source = "Esri visible-band VDVI proxy" if satellite else "0.50 neutral fallback; imagery unavailable"
    burn_sources = sorted({
        str(satellite.get(cell_id, {}).get("burn_proxy_quality", "no_imagery_default_zero")) for cell_id in cell_ids
    })

    high_twi_cells = [cell for cell in cells if cell.twi >= 14.5]
    river_distance = np.full(count, 9999.0, dtype=np.float64)
    if high_twi_cells:
        earth_radius_m = 6371008.8
        high_lat = np.radians([cell.latitude for cell in high_twi_cells])
        high_lon = np.radians([cell.longitude for cell in high_twi_cells])
        for index, cell in enumerate(cells):
            lat = np.radians(cell.latitude)
            lon = np.radians(cell.longitude)
            dlat = high_lat - lat
            dlon = high_lon - lon
            a = np.sin(dlat / 2) ** 2 + np.cos(lat) * np.cos(high_lat) * np.sin(dlon / 2) ** 2
            river_distance[index] = earth_radius_m * 2.0 * np.arctan2(np.sqrt(a), np.sqrt(np.maximum(0.0, 1.0 - a))).min()
    else:
        weather_fallbacks.append("d_river_m=9999.0; no H3 cell has TWI >= 14.5")

    infra_cells = scan.infrastructure.by_h3_cell if scan.infrastructure else {}
    road_distance = np.asarray([
        0.0 if infra_cells.get(cell_id, {}).get("road_cuts", 0) > 0 else 9999.0 for cell_id in cell_ids
    ], dtype=np.float64)
    pga = scan.seismic.max_pga_g if scan.seismic and scan.seismic.max_pga_g is not None else 0.0
    if scan.seismic is None or scan.seismic.max_pga_g is None:
        weather_fallbacks.append("pga_g=0.0; seismic source unavailable")

    inputs = PhysicsInputs(
        sand_fraction=sand, clay_fraction=clay, organic_matter_fraction=organic, bulk_density_g_cm3=bulk_density,
        slope_deg=np.asarray([cell.slope_deg for cell in cells]),
        twi=np.maximum(np.asarray([cell.twi for cell in cells]), 0.01),
        soil_saturation_fraction=saturation, rain_24h_mm=float(rain), storm_duration_hr=24.0,
        snow_depth_m=float(snow), air_temperature_c=float(air), soil_temperature_c=float(soil_temp),
        freeze_thaw_cycles_7d=float(freeze_cycles), burn_severity=burn, ndvi_norm=vegetation,
        profile_curvature=np.asarray([cell.profile_curvature for cell in cells]),
        peak_3h_intensity_mmhr=float(scan.weather.peak_rain_3h_mm or 0)/3 if scan.weather else 0.,
        river_surge_ratio=(scan.river or {}).get("surge_ratio",1.),
        latitude_deg=scan.location.latitude, pga_g=float(pga), road_cut_distance_m=road_distance, river_distance_m=river_distance,
    )
    station_inputs = inputs
    inputs = apply_alpine_weather(inputs, [cell.elevation_m for cell in cells])
    state = MultiCausalEngine().compute(inputs)
    kinematics = solve_kinematics(inputs, state)
    matrix = _feature_matrix(inputs, state, np.asarray([cell.profile_curvature for cell in cells]))
    p_xgb = model.predict_proba(matrix) if model is not None else None
    probability = blend_failure_probability(p_xgb, state.p_failure_physical, matrix[:, 4], state.fs_dynamic) if p_xgb is not None else state.p_failure_physical
    shap_values = None
    if model is not None:
        try:
            shap_values = model.shap_log_odds_contributions(matrix, absolute=False)
        except Exception:
            shap_values = None
    breakdown = explain_causes(inputs, state, shap_values=shap_values, feature_names=list(FEATURE_NAMES) if shap_values is not None else None)
    infrastructure = scan.infrastructure.by_h3_cell if scan.infrastructure else {}
    households = np.asarray([
        infrastructure.get(cell_id, {}).get("households") for cell_id in cell_ids
    ], dtype=object)
    buildings = np.asarray([
        float(infrastructure.get(cell_id, {}).get("buildings", 0)) for cell_id in cell_ids
    ], dtype=np.float64)
    exposure = np.asarray([
        max(float(households[index]), 1.0) if households[index] is not None else max(buildings[index], 1.0)
        for index in range(count)
    ], dtype=np.float64)
    hotspot_scores = np.asarray(probability, dtype=np.float64) * (0.35 + 0.65 * exposure)
    top_hotspot_index = int(np.argmax(hotspot_scores))
    steep_mask = np.asarray(inputs.slope_deg, dtype=np.float64) >= 15.0
    steep_fs = state.fs_dynamic[steep_mask] if np.any(steep_mask) else state.fs_dynamic
    min_fs_dynamic = float(np.percentile(steep_fs, 5))
    max_p_failure = float(np.percentile(probability, 95))
    top_cause_percentages = {
        cause: float(values[top_hotspot_index])
        for cause, values in breakdown.combined_percentages.items()
    }
    dominant = max(top_cause_percentages, key=top_cause_percentages.get)
    numbers = select_numbers_that_matter(
        top_hotspot_index, inputs, state, breakdown, probability_failure=probability
    )
    provenance = {
        "soil": "{} ({})".format(hydrology.soil_property_source, hydrology.soil_property_quality),
        "soil_saturation": saturation_source,
        "vegetation": vegetation_source,
        "burn": burn_sources,
        "d_road_cut_m": "OSM mapped cutting tag at H3 centroid; otherwise 9999 m default",
        "d_river_m": "H3-center distance to nearest DEM cell with TWI >= 14.5; 9999 m if absent",
        "weather_and_other_defaults": weather_fallbacks,
        "probability": "Calibrated ML; physical components reported separately; no combined probability" if p_xgb is not None else "FOSM physical probability; no calibrated model artifact",
    }
    return {
        "cell_count": count,
        "min_fs_dynamic": min_fs_dynamic,
        "max_p_failure": max_p_failure,
        "fs_dynamic_median": float(np.median(state.fs_dynamic)),
        "p_failure_median": float(np.median(probability)),
        "p_physical_median": float(np.median(state.p_failure_physical)),
        "p_xgb_median": float(np.median(p_xgb)) if p_xgb is not None else None,
        "velocity_median_m_s": float(np.median(kinematics.peak_velocity_m_s)),
        "velocity_class_representative": str(kinematics.velocity_class_name[top_hotspot_index]),
        "dominant_cause": dominant,
        "cause_percentages": top_cause_percentages,
        "numbers_that_matter": [item.to_dict() for item in numbers],
        "representative_cell_id": cell_ids[top_hotspot_index],
        "kinematics_top_hotspot": {
            "h3_index": cell_ids[top_hotspot_index],
            "latitude": float(cells[top_hotspot_index].latitude),
            "longitude": float(cells[top_hotspot_index].longitude),
            "slope_deg": float(inputs.slope_deg[top_hotspot_index]),
            "fs_dynamic": float(state.fs_dynamic[top_hotspot_index]),
            "p_failure": float(probability[top_hotspot_index]),
            "p_failure_physical": float(state.p_failure_physical[top_hotspot_index]),
            "p_xgb_calibrated": float(p_xgb[top_hotspot_index]) if p_xgb is not None else None,
            "p_combined": None,
            "newmark_applicable": bool(state.newmark_applicable[top_hotspot_index]),
            "newmark_displacement_cm": float(state.newmark_displacement_cm[top_hotspot_index]),
            "exposure_weight": float(exposure[top_hotspot_index]),
            "hotspot_score": float(hotspot_scores[top_hotspot_index]),
            "peak_velocity_ms": float(kinematics.peak_velocity_m_s[top_hotspot_index]),
            "cruden_varnes_class": int(kinematics.velocity_class[top_hotspot_index]),
            "cruden_varnes_label": str(kinematics.velocity_class_name[top_hotspot_index]),
            "failure_mode": str(kinematics.failure_mode[top_hotspot_index]),
        },
        "provenance": provenance,
        "feature_rows": matrix,
        "inputs": inputs,
        "station_inputs": station_inputs,
        "alpine_weather_assumption": ALPINE_ASSUMPTION,
        "profile_curvature": np.asarray([cell.profile_curvature for cell in cells]),
        "state": state,
        "kinematics": kinematics,
        "breakdown": breakdown,
        "cell_ids": cell_ids,
        "cells": cells,
        "exposure_by_cell": exposure,
        "households_by_cell": households,
    }
