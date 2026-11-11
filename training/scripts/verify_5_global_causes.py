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


from backend.api.live_input_builder import evaluate_live_physics


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
