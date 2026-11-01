#!/usr/bin/env python3
"""Live five-environment physics readiness and validation report."""

from __future__ import annotations

import argparse
import asyncio
import sys
from dataclasses import replace
from statistics import median
from pathlib import Path
from typing import Any, List, Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.physics.alpine_weather import apply_alpine_weather, ALPINE_ASSUMPTION
from backend.ml.global_xgb import FEATURE_NAMES, GlobalMonotonicXGB, blend_failure_probability, feature_matrix
from backend.ml.validation_suite import (
    build_validation_report,
    load_labeled_training_csv,
    run_kinematic_branch_controls,
)
from backend.realtime.global_ingestor import GlobalIngestor, GlobalScan
from backend.realtime.soilgrids import estimate_pedotransfer
from backend.physics.multi_causal_engine import MultiCausalEngine, PhysicsInputs
from backend.physics.kinematics import solve_kinematics
from backend.physics.causal_attribution import CAUSES, explain_causes, select_numbers_that_matter

ENVIRONMENTS = [
    ("Wayanad Monsoon", 11.548, 76.134, "hydro-meteorological"),
    ("Seaward Kaikoura Range", -42.260, 173.620, "seismic / tectonic"),
    ("Zermatt Snowmelt", 46.020, 7.749, "cryospheric"),
    # CAL FIRE's published Palisades incident point, in the Malibu/Palisades fire area.
    ("Malibu Post-Fire Burn Scar", 34.07022, -118.54453, "post-wildfire / biotic"),
    ("New Delhi Flat Plain", 28.6139, 77.2090, "negative-control context"),
]


def fmt(value: Any, digits: int = 2, suffix: str = "") -> str:
    if value is None:
        return "unavailable"
    try:
        return "{:.{digits}f}{suffix}".format(float(value), digits=digits, suffix=suffix)
    except (TypeError, ValueError):
        return str(value)


def missing_physics_inputs(scan: GlobalScan) -> List[str]:
    missing = []
    soil = scan.hydrology.soil_properties if scan.hydrology else {}
    for key, label in (
        ("sand_g_kg", "SoilGrids sand fraction"),
        ("clay_g_kg", "SoilGrids clay fraction"),
        ("bdod_kg_m3", "SoilGrids bulk density"),
        ("soc_g_kg", "SoilGrids organic carbon for OM estimate"),
    ):
        if soil.get(key) is None:
            missing.append(label)
    if not scan.hydrology or scan.hydrology.soil_saturation_pct is None:
        missing.append("soil saturation/porosity estimate")
    if not scan.weather or scan.weather.rain_24h_mm is None:
        missing.append("24-hour rainfall")
    if not scan.weather or scan.weather.air_temperature_c is None or scan.weather.soil_temperature_0_7cm_c is None:
        missing.append("air and surface-soil temperatures")
    if not scan.terrain:
        missing.append("H3 terrain cells and TWI")
    if not scan.satellite_cells:
        missing.extend(("RGB vegetation-index proxy", "burn-severity proxy"))
    elif not any(
        cell.get("burn_proxy_quality") in ("estimated_eonet_rgb_proxy", "simulation_override")
        for cell in scan.satellite_cells.values()
    ):
        missing.append("EONET-linked burn evidence or simulation override")
    # OSM feature counts are available, but nearest road-cut and river distances
    # are not part of the current geometry contract.
    missing.extend(("road-cut distance", "river distance"))
    return missing


def evaluate_live_physics(scan: GlobalScan, model: Optional[GlobalMonotonicXGB] = None) -> Optional[dict]:
    """Compute a complete, provenance-marked physics state for every sampled land H3 cell."""
    if not scan.terrain or not scan.terrain.cells:
        return None
    cells = list(scan.terrain.cells.values())
    cell_ids = [cell.cell_id for cell in cells]
    count = len(cells)
    hydrology = scan.hydrology
    properties = hydrology.soil_properties if hydrology else {}
    soil_fallback = estimate_pedotransfer(
        float(np.mean([cell.slope_deg for cell in cells])),
        float(np.mean([cell.profile_curvature for cell in cells])),
        float(np.mean([cell.twi for cell in cells])),
        hydrology.blended_relative_wetness_pct if hydrology else None,
    )

    def soil_array(key: str, fallback_key: str) -> np.ndarray:
        value = properties.get(key)
        if value is None:
            value = soil_fallback[fallback_key]
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

    if hydrology and hydrology.soil_saturation_pct is not None:
        saturation = float(np.clip(hydrology.soil_saturation_pct / 100.0, 0.0, 1.0))
        saturation_source = "Open-Meteo moisture / bulk-density porosity" if "porosity" in " ".join(hydrology.source_mix) else "hydrology soil saturation"
    elif hydrology and hydrology.blended_relative_wetness_pct is not None:
        saturation = float(np.clip(hydrology.blended_relative_wetness_pct / 100.0, 0.0, 1.0))
        saturation_source = "relative wetness proxy, used as saturation fallback"
    else:
        saturation = 0.50
        saturation_source = "0.50 neutral fallback; no wetness observation"

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
            shap_values = model.shap_log_odds_contributions(matrix)
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
        "soil": "{} ({})".format(hydrology.soil_property_source, hydrology.soil_property_quality) if hydrology else soil_fallback["source"],
        "soil_saturation": saturation_source,
        "vegetation": vegetation_source,
        "burn": burn_sources,
        "d_road_cut_m": "OSM mapped cutting tag at H3 centroid; otherwise 9999 m default",
        "d_river_m": "H3-center distance to nearest DEM cell with TWI >= 14.5; 9999 m if absent",
        "weather_and_other_defaults": weather_fallbacks,
        "probability": "mechanically gated XGBoost/physical blend; source inventory and reported-point proxy labels" if p_xgb is not None else "FOSM physical probability; no calibrated model artifact",
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


from backend.physics.features import physics_feature_matrix as _feature_matrix


async def scan_environments(timeout_s: float, radius_km: float) -> List[GlobalScan]:
    scans = []
    async with GlobalIngestor(provider_timeout_s=timeout_s) as ingestor:
        for _, latitude, longitude, _ in ENVIRONMENTS:
            scans.append(await ingestor.scan((latitude, longitude), radius_km=radius_km))
    return scans


def print_environment_table(scans: List[GlobalScan], model: Optional[GlobalMonotonicXGB] = None) -> List[dict]:
    print("\nLIVE ENVIRONMENT / PHYSICS READINESS")
    print("Environment | H3 | Elevation m | slope P95 | max TWI | rain24 / ARI7d mm | PGA g | soil source | sat cells | med veg proxy | max burn proxy | physics")
    print("---|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---")
    physics_rows = []
    physics_internal = {}
    for (name, _, _, regime), scan in zip(ENVIRONMENTS, scans):
        terrain = scan.terrain
        weather = scan.weather
        seismic = scan.seismic
        if terrain:
            elev = "{}–{}".format(fmt(terrain.min_elevation_m), fmt(terrain.max_elevation_m))
            h3_count = str(terrain.h3_cell_count)
            slope = fmt(terrain.slope_p95_deg, suffix="°")
            twi = fmt(terrain.max_twi)
        else:
            elev, h3_count, slope, twi = "unavailable", "unavailable", "unavailable", "unavailable"
        rain = fmt(weather.rain_24h_mm) if weather else "unavailable"
        ari = fmt(weather.ari_7d_mm) if weather else "unavailable"
        pga = fmt(seismic.max_pga_g, digits=6) if seismic else "unavailable"
        veg = [float(item["vegetation_index_norm_proxy"]) for item in scan.satellite_cells.values()]
        burn = [float(item["burn_severity_proxy"]) for item in scan.satellite_cells.values()]
        median_veg = fmt(median(veg), digits=3) if veg else "unavailable"
        max_burn = fmt(max(burn), digits=3) if burn else "unavailable"
        missing = missing_physics_inputs(scan)
        physics = evaluate_live_physics(scan, model)
        physics_status = "computed for {} land cells".format(physics["cell_count"]) if physics else "not computed (no DEM land cells)"
        if physics:
            physics_internal[name] = physics
        print(
            "{} — {} | {} | {} | {} | {} | {} / {} | {} | {} | {} | {} | {} | {}".format(
                name, regime, h3_count, elev, slope, twi, rain, ari, pga,
                scan.hydrology.soil_property_source if scan.hydrology else "unavailable",
                len(scan.satellite_cells), median_veg, max_burn, physics_status
            )
        )
        if name.startswith("New Delhi") and terrain and terrain.slope_p95_deg >= 2.0:
            print("  note: this live viewport's slope P95 is above the <2° flat-control threshold; the synthetic flat control is reported separately below")
        print(
            "  source status: DEM={}, weather={}, NASA={}, soil={}, USGS={}, OSM={}, EONET fire/slide={}/{}, Esri={}".format(
                *["live" if scan.sources[source].available else "unavailable" for source in
                  ("terrarium_dem", "open_meteo", "nasa_power", "soilgrids", "usgs", "overpass")],
                "live" if scan.sources["eonet_wildfires"].available else "unavailable",
                "live" if scan.sources["eonet_landslides"].available else "unavailable",
                "live" if scan.sources["satellite"].available else "unavailable",
            )
        )
        if scan.hydrology:
            props = scan.hydrology.soil_properties
            print("  soil: sand={:.3f}, clay={:.3f}, OM={:.3f}, bulk density={:.3f} g/cm³ [{}]".format(
                float(props["sand_fraction"]), float(props["clay_fraction"]),
                float(props["organic_matter_fraction"]), float(props["bulk_density_g_cm3"]),
                scan.hydrology.soil_property_quality,
            ))
        if scan.satellite_cells and not any(
            item.get("burn_proxy_quality") in ("estimated_eonet_rgb_proxy", "simulation_override")
            for item in scan.satellite_cells.values()
        ):
            print("  burn proxy: no EONET wildfire geometry was available; zero signals do not confirm a burn-free surface")
        if missing:
            print("  source gaps handled by marked defaults/proxies: {}".format(", ".join(missing)))
        if scan.weather:
            print(
                "  rain peak3={} mm, forecast +24h={} mm/+24–48h={} mm, snow={} m, FT7d={}".format(
                    fmt(scan.weather.peak_rain_3h_mm),
                    fmt(scan.weather.forecast_rain_next_24h_mm),
                    fmt(scan.weather.forecast_rain_24_48h_mm),
                    fmt(scan.weather.snow_depth_m, digits=3),
                    scan.weather.freeze_thaw_cycles_7d,
                )
            )
        if scan.seismic:
            print("  earthquakes within 250km: {}".format(scan.seismic.event_count))
        if scan.infrastructure:
            print("  OSM road cuts: {}".format(scan.infrastructure.road_cut_count))
        else:
            print("  OSM road cuts: unavailable")
        if physics:
            physics_rows.append({
                "name": name,
                "cell_count": physics["cell_count"],
                "min_fs_dynamic": physics["min_fs_dynamic"],
                "max_p_failure": physics["max_p_failure"],
                "fs_dynamic_median": physics["fs_dynamic_median"],
                "p_failure_median": physics["p_failure_median"],
                "p_physical_median": physics["p_physical_median"],
                "p_xgb_median": physics["p_xgb_median"],
                "velocity_median_m_s": physics["velocity_median_m_s"],
                "velocity_class_representative": physics["velocity_class_representative"],
                "dominant_cause": physics["dominant_cause"],
                "cause_percentages": physics["cause_percentages"],
                "numbers_that_matter": physics["numbers_that_matter"],
                "kinematics_top_hotspot": physics["kinematics_top_hotspot"],
                "representative_cell_id": physics["representative_cell_id"],
                "provenance": physics["provenance"],
            })

    print("\nLIVE TOP HOTSPOTS BY REGION (FS is P05 among slopes ≥15°; P is P95; XGBoost blend remains exploratory)")
    print("Region | Land H3 | min_fs_dynamic | max_p_failure | peak_velocity_ms | Dominant cause | Cause share")
    print("---|---:|---:|---:|---:|---|---")
    for row in physics_rows:
        shares = ", ".join("{}={:.1f}%".format(cause, share) for cause, share in row["cause_percentages"].items())
        top = row["kinematics_top_hotspot"]
        print("{} | {} | {:.3f} | {:.4f} | {:.6g} | {} | {}".format(
            row["name"], row["cell_count"], row["min_fs_dynamic"], row["max_p_failure"],
            top["peak_velocity_ms"], row["dominant_cause"], shares,
        ))
        important = ", ".join("{}={:.3g} {}".format(item["metric"], item["value"], item["unit"]) for item in row["numbers_that_matter"])
        print("  numbers_that_matter: {}".format(important or "no active threshold metrics"))
        print("  top hotspot: H3={} | score={:.5g} | FS={:.3f} | P={:.4f} | class={} | mode={}".format(
            top["h3_index"], top["hotspot_score"], top["fs_dynamic"], top["p_failure"],
            top["cruden_varnes_label"], top["failure_mode"],
        ))
        print("  provenance: soil={}, saturation={}, vegetation={}, burn={}, roads={}, rivers={}".format(
            row["provenance"]["soil"], row["provenance"]["soil_saturation"], row["provenance"]["vegetation"],
            ",".join(row["provenance"]["burn"]), row["provenance"]["d_road_cut_m"], row["provenance"]["d_river_m"],
        ))

    delhi = physics_internal.get("New Delhi Flat Plain")
    if delhi:
        flat_inputs = replace(delhi["inputs"], slope_deg=np.zeros(delhi["cell_count"], dtype=np.float64))
        flat_state = MultiCausalEngine().compute(flat_inputs)
        flat_velocity = solve_kinematics(flat_inputs, flat_state).peak_velocity_m_s
        flat_control = {
            "name": "New Delhi live-input flat-plain counterfactual",
            "slope_deg": 0.0,
            "fs_dynamic_median": float(np.median(flat_state.fs_dynamic)),
            "p_failure_physical_median": float(np.median(flat_state.p_failure_physical)),
            "velocity_median_m_s": float(np.median(flat_velocity)),
            "passed": bool(np.median(flat_state.fs_dynamic) > 3.0 and np.median(flat_state.p_failure_physical) < 0.02),
            "assumption": "New Delhi live soil, weather, saturation, and other inputs held fixed; slope set to zero for a flat-ground counterfactual.",
        }
        for row in physics_rows:
            if row["name"] == "New Delhi Flat Plain":
                row["flat_plain_control"] = flat_control
        print("\nNEW DELHI FLAT-PLAIN CONTROL (live soil/weather held fixed; slope set to 0°)")
        print("FS_dynamic median={:.3f} | P_phys median={:.6f} | velocity median={:.6g} m/s | {}".format(
            flat_control["fs_dynamic_median"], flat_control["p_failure_physical_median"],
            flat_control["velocity_median_m_s"], "PASS" if flat_control["passed"] else "FAIL",
        ))

    scenario_changes = [
        ("Wayanad Monsoon", "+180 mm/24 h rainfall", {"rain_24h_mm": 180.0}),
        ("Seaward Kaikoura Range", "+0.35 g PGA", {"pga_g": 0.35}),
        ("Zermatt Snowmelt", "+5 °C air temperature with 0.5 m snow cover", {"air_temperature_c": 5.0, "snow_depth_m": 0.5}),
        ("Malibu Post-Fire Burn Scar", "burn severity 0.50 and at least 25 mm/24 h rain", {"burn_severity": 0.50, "rain_24h_mm": 25.0}),
    ]
    print("\nACTIVE WHAT-IF SCENARIOS (assumptions shown; hotspot percentiles and top-cell kinematics)")
    print("Region | Scenario | min_fs_dynamic | max_p_failure | peak_velocity_ms | Dominant cause | numbers_that_matter")
    print("---|---|---:|---:|---:|---|---")
    for name, title, changes in scenario_changes:
        baseline = physics_internal.get(name)
        if not baseline:
            continue
        inputs = baseline["inputs"]
        replacement = dict(changes)
        if "rain_24h_mm" in replacement:
            replacement["rain_24h_mm"] = max(float(inputs.rain_24h_mm), replacement["rain_24h_mm"])
        if "pga_g" in replacement:
            replacement["pga_g"] = max(float(inputs.pga_g), replacement["pga_g"])
        if "air_temperature_c" in replacement:
            replacement["air_temperature_c"] = np.maximum(inputs.air_temperature_c, replacement["air_temperature_c"])
        if "snow_depth_m" in replacement:
            replacement["snow_depth_m"] = np.maximum(inputs.snow_depth_m, replacement["snow_depth_m"])
        if "burn_severity" in replacement:
            replacement["burn_severity"] = np.full_like(np.asarray(inputs.burn_severity, dtype=np.float64), replacement["burn_severity"])
        scenario_inputs = replace(inputs, **replacement)
        state = MultiCausalEngine().compute(scenario_inputs)
        speeds = solve_kinematics(scenario_inputs, state).peak_velocity_m_s
        matrix = _feature_matrix(scenario_inputs, state, baseline["profile_curvature"])
        p_xgb = model.predict_proba(matrix) if model is not None else None
        probability = blend_failure_probability(p_xgb, state.p_failure_physical, matrix[:, 4], state.fs_dynamic) if p_xgb is not None else state.p_failure_physical
        hotspot_score = probability * (0.35 + 0.65 * baseline["exposure_by_cell"])
        top_index = int(np.argmax(hotspot_score))
        steep = np.asarray(scenario_inputs.slope_deg, dtype=np.float64) >= 15.0
        fs_values = state.fs_dynamic[steep] if np.any(steep) else state.fs_dynamic
        max_probability = float(np.percentile(probability, 95))
        min_fs = float(np.percentile(fs_values, 5))
        causal = explain_causes(scenario_inputs, state)
        cause = max(CAUSES, key=lambda key: float(causal.combined_percentages[key][top_index]))
        numbers = select_numbers_that_matter(top_index, scenario_inputs, state, causal, probability_failure=probability)
        numbers_text = ", ".join("{}={:.3g} {}".format(item.metric, item.value, item.unit) for item in numbers[:3])
        print("{} | {} | {:.3f} | {:.4f} | {:.6g} | {} | {}".format(
            name, title, min_fs, max_probability, float(speeds[top_index]), cause, numbers_text,
        ))
    return physics_rows


def print_validation(training_csv: str, hindcast_csv: str) -> int:
    model: Optional[GlobalMonotonicXGB] = None
    try:
        X, y, groups = load_labeled_training_csv(training_csv)
        model = GlobalMonotonicXGB().fit(X, y, groups)
        print("\nFitted monotonic model using {} labeled rows across {} spatial blocks.".format(len(y), len(set(groups))))
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        print("\nXGBoost model: NOT FIT — {}".format(exc))

    report = build_validation_report(training_csv, hindcast_csv, model)
    print("\nSCIENTIFIC VALIDATION")
    print("Stage | Status | Metrics | Details")
    print("---|---|---|---")
    for stage in (report.stage1, report.stage2, report.stage3_monotonicity):
        metrics = ", ".join(
            "{}={}".format(name, fmt(value, digits=4)) for name, value in stage.metrics.items()
        ) or "N/A"
        details = "; ".join(stage.details) or "—"
        print("{} | {} | {} | {}".format(stage.name, stage.status.upper(), metrics, details))

    print("\nStage 3 counterfactual controls (FOSM physical probability; no trained ML blend):")
    for control in report.stage3_controls:
        print(
            "{} | {} | FS_static={} | FS_dynamic={} | P_phys={} | velocity={} m/s (Class {}) | mode={} | expected {}".format(
                control.name,
                "PASS" if control.passed else "FAIL",
                fmt(control.fs_static, digits=3),
                fmt(control.fs_dynamic, digits=3),
                fmt(control.p_failure_physical, digits=5),
                fmt(control.peak_velocity_m_s, digits=6),
                control.velocity_class,
                control.failure_mode,
                control.assertion,
            )
        )
        shares = ", ".join("{}={:.1f}%".format(cause, share) for cause, share in control.cause_percentages.items())
        metrics = ", ".join(item["metric"] for item in control.numbers_that_matter)
        print("  cause shares: {}".format(shares))
        print("  numbers that matter: {}".format(metrics))

    print("\nKinematics branch equation checks:")
    branch_checks = run_kinematic_branch_controls()
    for check in branch_checks:
        print(
            "{} | {} | {} | v={} m/s | Class {}".format(
                check.name,
                "PASS" if check.passed else "FAIL",
                check.actual_mode,
                fmt(check.peak_velocity_m_s, digits=6),
                check.velocity_class,
            )
        )
    validation_failed = (
        any(not item.passed for item in report.stage3_controls)
        or any(not item.passed for item in branch_checks)
        or report.stage1.status == "failed"
        or report.stage3_monotonicity.status == "failed"
    )
    return 1 if validation_failed else 0


async def main_async(args: argparse.Namespace) -> int:
    scans = await scan_environments(args.provider_timeout, args.radius_km)
    print_environment_table(scans)
    return print_validation(args.training_csv, args.hindcast_csv)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider-timeout", type=float, default=1.25)
    parser.add_argument("--radius-km", type=float, default=3.5)
    parser.add_argument("--training-csv", default="data/landslide_training.csv")
    parser.add_argument("--hindcast-csv", default="data/historical_hindcast.csv")
    return asyncio.run(main_async(parser.parse_args()))


if __name__ == "__main__":
    sys.exit(main())
