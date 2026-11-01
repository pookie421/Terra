"""Live location scans, cell physics, hotspot summaries, and GeoJSON output."""

from __future__ import annotations

import json
import math
import os
import threading
import unicodedata
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import h3
import numpy as np

from backend.api.reliability import reliability_report
from backend.physics.alpine_weather import apply_alpine_weather, ALPINE_ASSUMPTION
from backend.ml.global_xgb import GlobalMonotonicXGB, blend_failure_probability
from backend.physics.causal_attribution import CAUSES, explain_causes, select_numbers_that_matter
from backend.physics.kinematics import route_runout, solve_kinematics
from backend.physics.multi_causal_engine import MultiCausalEngine, PhysicsInputs
from backend.realtime.global_ingestor import GlobalIngestor, GlobalScan
from scripts.verify_5_global_causes import _feature_matrix, evaluate_live_physics

ROOT = Path(__file__).resolve().parents[2]
RUNOUT_MU_B = 0.10
RUNOUT_XI = 500.0
RUNOUT_PARAMETER_QUALITY = "un_calibrated_display_default"
ALERT_THRESHOLDS = {"warning": 0.15, "watch": 0.35, "alert": 0.65}

# These coordinates were supplied as project scan locations, not inferred labels.
KNOWN_LOCATION_COORDINATES = {
    "chooralmala": (11.548, 76.134),
    "wayanad": (11.548, 76.134),
    "pettimudi": (10.174, 77.012),
    "kaikoura": (-42.260, 173.620),
    "zermatt": (46.020, 7.749),
    "malibu": (34.07022, -118.54453),
    "new delhi": (28.6139, 77.2090),
}


def known_location_coordinates(query: str) -> Optional[Tuple[float, float]]:
    normalized = unicodedata.normalize("NFKD", str(query)).encode("ascii", "ignore").decode("ascii").lower()
    for name, coordinates in KNOWN_LOCATION_COORDINATES.items():
        if name in normalized:
            return coordinates
    return None


def _load_model() -> Optional[GlobalMonotonicXGB]:
    path = ROOT / "artifacts" / "global_multi_causal_xgb.pkl"
    if not path.exists():
        return None
    try:
        import joblib
        model = joblib.load(path)
        return model if isinstance(model, GlobalMonotonicXGB) else None
    except Exception:
        return None


def _h3_boundary_lonlat(cell_id: str) -> List[List[float]]:
    boundary_fn = getattr(h3, "cell_to_boundary", None) or getattr(h3, "h3_to_geo_boundary")
    ring = [[float(lon), float(lat)] for lat, lon in boundary_fn(cell_id)]
    if ring and ring[0] != ring[-1]:
        ring.append(ring[0])
    return ring


def _finite(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def _field_capacity_stress_mask(inputs: PhysicsInputs, state: Any) -> np.ndarray:
    slope = np.radians(np.clip(np.asarray(inputs.slope_deg, dtype=np.float64), 0.0, 89.9))
    field_capacity_saturation = np.clip(state.theta_33 / np.maximum(state.theta_s, 1e-6), 0.0, 0.95)
    field_capacity_head = state.regolith_depth_m * field_capacity_saturation * np.cos(slope) ** 2
    return state.pore_pressure_head_m > field_capacity_head


class GlobalScanService:
    """The shared application layer used by REST routes and agent tools."""

    def __init__(self, provider_timeout_s: Optional[float] = None) -> None:
        self.provider_timeout_s = provider_timeout_s or float(os.getenv("TERRA_PROVIDER_TIMEOUT_S", "8"))
        self.model = _load_model()
        self.engine = MultiCausalEngine()
        self._cell_contexts: Dict[str, Dict[str, Any]] = {}
        self._context_order: List[str] = []
        self._context_lock = threading.RLock()

    async def scan(
        self,
        location_query: str = "Chooralmala, Wayanad",
        lat: Optional[float] = None,
        lon: Optional[float] = None,
        rain_delta_mm: float = 0.0,
        storm_duration_hr: float = 24.0,
        pga_override_g: float = 0.0,
        snowmelt_temp_surge_c: float = 0.0,
        burn_severity_override: float = 0.0,
        agency_mode: str = "LOCAL_EOC",
        _snapshot=None,
    ) -> Dict[str, Any]:
        if (lat is None) != (lon is None):
            raise ValueError("lat and lon must be provided together")
        if storm_duration_hr <= 0:
            raise ValueError("storm_duration_hr must be positive")
        numeric = [rain_delta_mm, storm_duration_hr, pga_override_g, snowmelt_temp_surge_c, burn_severity_override]
        numeric.extend(value for value in (lat, lon) if value is not None)
        if not all(math.isfinite(float(value)) for value in numeric):
            raise ValueError("coordinates and scenario inputs must be finite")
        if rain_delta_mm < 0 or pga_override_g < 0 or snowmelt_temp_surge_c < 0:
            raise ValueError("scenario deltas and overrides must be nonnegative")
        if not 0.0 <= burn_severity_override <= 1.0:
            raise ValueError("burn_severity_override must be between 0 and 1")
        used_named_fallback = False
        if lat is not None and lon is not None:
            location: Union[str, Tuple[float, float]] = (lat, lon)
        else:
            known_coordinates = known_location_coordinates(location_query)
            if known_coordinates is not None:
                location = known_coordinates
                used_named_fallback = True
            else:
                location = location_query

        if _snapshot is None:
            async with GlobalIngestor(provider_timeout_s=self.provider_timeout_s) as ingestor:
                live_scan = await ingestor.scan(location, radius_km=3.5)
        else:
            live_scan = _snapshot
        base = evaluate_live_physics(live_scan, self.model)
        if base is None:
            return self._empty_response(live_scan, agency_mode)

        base_inputs: PhysicsInputs = base["station_inputs"]
        count = int(base["cell_count"])
        rain = max(0.0, _finite(base_inputs.rain_24h_mm)) + rain_delta_mm
        pga = max(_finite(base_inputs.pga_g), pga_override_g)
        air = _finite(base_inputs.air_temperature_c)
        soil_temp = _finite(base_inputs.soil_temperature_c)
        burn = np.maximum(
            np.asarray(base_inputs.burn_severity, dtype=np.float64), burn_severity_override
        )
        inputs = replace(
            base_inputs,
            rain_24h_mm=rain,
            storm_duration_hr=storm_duration_hr,
            pga_g=pga,
            air_temperature_c=air,
            soil_temperature_c=soil_temp,
            burn_severity=burn,
        )
        inputs = apply_alpine_weather(inputs, [cell.elevation_m for cell in base["cells"]], snowmelt_temp_surge_c)
        state = self.engine.compute(inputs)
        kinematics = solve_kinematics(inputs, state)
        feature_rows = _feature_matrix(inputs, state, base["profile_curvature"])
        p_xgb = self.model.predict_proba(feature_rows) if self.model is not None else None
        probability = blend_failure_probability(p_xgb, state.p_failure_physical, inputs.slope_deg, state.fs_dynamic) if p_xgb is not None else state.p_failure_physical
        shap_values = self.model.shap_log_odds_contributions(feature_rows) if self.model is not None else None
        breakdown = explain_causes(inputs, state)

        cell_ids = list(base["cell_ids"])
        cells = base["cells"]
        infra_by_cell = live_scan.infrastructure.by_h3_cell if live_scan.infrastructure else {}
        households: List[Optional[int]] = []
        exposure = np.ones(count, dtype=np.float64)
        exposure_basis: List[str] = []
        for index, cell_id in enumerate(cell_ids):
            infra = infra_by_cell.get(cell_id, {})
            household_count = infra.get("households")
            building_count = max(0, int(infra.get("buildings", 0) or 0))
            households.append(int(household_count) if household_count is not None else None)
            if household_count is not None:
                exposure[index] = max(1.0, float(household_count))
                exposure_basis.append("mapped_household_count")
            elif building_count > 0:
                exposure[index] = float(building_count)
                exposure_basis.append("OSM_building_count_proxy")
            else:
                exposure[index] = 1.0
                exposure_basis.append("unit_weight_exposure_fallback")

        kinematic_factor = 1.0 + .75*np.clip(state.newmark_displacement_cm/10.,0.,1.5) + .50*np.clip(kinematics.peak_velocity_m_s/12.,0.,1.)
        hotspot_score = probability * (0.35 + 0.65 * exposure) * kinematic_factor
        top_index = int(np.argmax(hotspot_score))
        viewport_fs_p05 = float(np.percentile(state.fs_dynamic,5))
        viewport_probability_p95 = float(np.percentile(probability,95))
        steep = state.effective_slope_deg >= 24.0
        steep_fs = state.fs_dynamic[steep] if np.any(steep) else state.fs_dynamic
        min_fs_dynamic = float(np.percentile(steep_fs, 5))
        max_p_failure = float(np.percentile(probability[steep] if np.any(steep) else probability, 95))
        causal_top = {cause: float(values[top_index]) for cause, values in breakdown.combined_percentages.items()}
        dominant_cause = max(causal_top, key=causal_top.get) if breakdown.valid[top_index] else "no_active_trigger"
        numbers_top = [
            item.to_dict()
            for item in select_numbers_that_matter(
                top_index, inputs, state, breakdown, probability_failure=probability
            )
        ]

        if self.model is not None:
            intervals, widths, bivariate = self.model.conformal_intervals(probability)
        else:
            intervals=np.tile([0.,1.],(count,1)); widths=np.ones(count)
            bivariate=np.where(probability>=.5,'HIGH_SUSCEPTIBILITY_HIGH_UNCERTAINTY','LOW_SUSCEPTIBILITY_HIGH_UNCERTAINTY')
        geo_features = []
        serialized_cell_contexts: Dict[str, Dict[str, Any]] = {}
        for index, cell_id in enumerate(cell_ids):
            cell = cells[index]
            cell_causes = {cause: float(values[index]) for cause, values in breakdown.combined_percentages.items()}
            cell_numbers = [
                item.to_dict()
                for item in select_numbers_that_matter(index, inputs, state, breakdown, probability_failure=probability)
            ]
            cell_dominant = max(cell_causes, key=cell_causes.get) if breakdown.valid[index] else "no_active_trigger"
            cell_households = households[index]
            properties = {
                "h3_index": cell_id,
                "lat": float(cell.latitude),
                "lon": float(cell.longitude),
                "elevation_m": float(cell.elevation_m),
                "slope_deg": float(np.asarray(inputs.slope_deg)[index]),
                "twi": float(np.asarray(inputs.twi)[index]),
                "fs_dynamic": float(state.fs_dynamic[index]),
                "effective_scarp_slope_deg": float(state.effective_slope_deg[index]),
                "causal_attribution_valid": bool(breakdown.valid[index]),
                "effective_mobilization_depth_m": float(state.regolith_depth_m[index]),
                "fs_static_dry": float(state.fs_static_dry[index]),
                "p_failure_seismic": float(state.p_failure_seismic[index]),
                "p_failure_m1": float(state.p_failure_m1[index]),
                "effective_burn_severity": float(state.effective_burn_severity[index]),
                "prefire_fuel_index_proxy": float(np.asarray(inputs.ndvi_norm)[index]),
                "cryospheric_eligible": bool(state.cryospheric_eligible[index]),
                "lateral_resistance_kpa": float(state.lateral_resistance_kpa[index]),
                "conformal_interval_90": intervals[index].tolist(),
                "uncertainty_width": float(widths[index]),
                "bivariate_class": str(bivariate[index]),
                "conformal_target": "binary event label; not a confidence interval for true failure probability",
                "uncertainty_status": "calibrated_under_exchangeable_blocks_assumption" if self.model is not None and getattr(self.model,'conformal_',None) else "uncalibrated_vacuous_interval",
                "fs_mohr_coulomb": float(state.fs_mohr_coulomb[index]),
                "fs_postfire_yield_index": float(state.fs_postfire_yield_index[index]),
                "debris_volume_proxy_m3": float(state.debris_volume_m3[index]),
                "rain_15min_proxy_mm": float(state.rain_15min_proxy_mm[index]),
                "suction_cohesion_kpa": float(state.suction_cohesion_kpa[index]),
                "cryo_perched_head_m": float(state.cryo_perched_head_m[index]),
                "equivalent_water_mm": float(state.equivalent_water_mm[index]),
                "mechanism_quality": "M1/Gartner cell-input proxies; postfire FS is a yield index; assumed ice joints and basal drainage",
                "newmark_displacement_cm": float(state.newmark_displacement_cm[index]),
                "cell_snow_depth_m": float(np.asarray(inputs.snow_depth_m)[index]),
                "cell_air_temperature_c": float(np.asarray(inputs.air_temperature_c)[index]),
                "alpine_weather_quality": ALPINE_ASSUMPTION,
                "critical_acceleration_g": float(state.critical_acceleration_g[index]),
                "p_failure": float(probability[index]),
                "hotspot_score": float(hotspot_score[index]),
                "kinematic_factor": float(kinematic_factor[index]),
                "extrusion_height_m": float(probability[index] * 100.0),
                "cruden_varnes_class": int(kinematics.velocity_class[index]),
                "cruden_varnes_label": str(kinematics.velocity_class_name[index]),
                "peak_velocity_ms": float(kinematics.peak_velocity_m_s[index]),
                "failure_mode": str(kinematics.failure_mode[index]),
                "dominant_cause": cell_dominant,
                "road_name": None,
                "households": cell_households,
                "exposure_weight": float(exposure[index]),
                "exposure_basis": exposure_basis[index],
                "causal_breakdown_pct": cell_causes,
                "numbers_that_matter": cell_numbers,
            }
            geo_features.append({"type": "Feature", "geometry": {"type": "Polygon", "coordinates": [_h3_boundary_lonlat(cell_id)]}, "properties": properties})
            serialized_cell_contexts[cell_id] = properties

        top_hotspot = {
            **serialized_cell_contexts[cell_ids[top_index]],
            "causal_breakdown_pct": causal_top,
            "numbers_that_matter": numbers_top,
        }
        routes = self._runout_features(live_scan, cell_ids, state, kinematics, probability, hotspot_score)
        top_route=next((r['properties']['path'] for r in routes if r['properties']['source_cell']==cell_ids[top_index]),[cell_ids[top_index]])
        from backend.physics.cascading import cascading_screen
        top_hotspot.update(cascading_screen(cell_ids[top_index],top_route,live_scan.terrain.cells))
        critical_mask = (state.fs_dynamic <= 1.05) | (probability >= 0.65)
        household_values = [value for value in households if value is not None]
        weather = live_scan.weather
        summary = {
            "wmo_alert_level": self._alert_level(max_p_failure, rain),
            "max_p_failure": max_p_failure,
            "critical_fs": min_fs_dynamic, "critical_p_failure": max_p_failure,
            "summary_population": "effective_slope_ge_24_deg" if np.any(steep) else "all_cells",
            "summary_population_count": int(np.count_nonzero(steep)) if np.any(steep) else count,
            "viewport_fs_p05": viewport_fs_p05, "viewport_probability_p95": viewport_probability_p95,
            "min_fs_dynamic": min_fs_dynamic,
            "dominant_cause": dominant_cause,
            "causal_attribution_valid": bool(breakdown.valid[top_index]),
            "causal_attribution_quality": "stress_budget" if breakdown.valid[top_index] else "neutral_prior_no_active_stress",
            "critical_hotspots_count": int(np.count_nonzero(critical_mask)),
            "total_households_exposed": int(sum(household_values)) if household_values else None,
            "household_exposure_quality": "OSM household counts" if household_values else "household counts unavailable; building-count/unit-weight proxy used for hotspot ranking",
            "causal_breakdown_pct": causal_top,
            "kinematics_top_hotspot": top_hotspot,
            "numbers_that_matter": numbers_top,
            "meteorological_metrics": {
                "rain_24h_mm_observed": weather.rain_24h_mm if weather else None,
                "rain_24h_mm_scenario": rain,
                "storm_duration_hr": storm_duration_hr,
                "peak_rain_3h_mm": weather.peak_rain_3h_mm if weather else None,
                "ari_7d_mm": weather.ari_7d_mm if weather else None,
                "snow_depth_m": weather.snow_depth_m if weather else None,
                "air_temperature_c_scenario": air + snowmelt_temp_surge_c,
                "cell_air_temperature_range_c": [float(np.min(inputs.air_temperature_c)), float(np.max(inputs.air_temperature_c))],
                "cell_snow_depth_range_m": [float(np.min(inputs.snow_depth_m)), float(np.max(inputs.snow_depth_m))],
                "alpine_weather_assumption": ALPINE_ASSUMPTION,
                "pga_g_observed": live_scan.seismic.max_pga_g if live_scan.seismic else None,
                "pga_g_scenario": pga,
                "burn_severity_override": burn_severity_override,
            },
            "shap_status": "native_tree_shap_fold_boosters" if shap_values is not None else "model_unavailable",
            "shap_mean_absolute_log_odds": None if shap_values is None else dict(zip(self.model.feature_names_in_, np.mean(shap_values,axis=0).tolist())),
            "probability_source": "mechanical gate × regime-weighted isotonic XGBoost/physical union; gated score not separately calibrated" if p_xgb is not None else "physical FOSM/seismic/M1 union; no trained model artifact loaded",
            "runout_parameter_quality": RUNOUT_PARAMETER_QUALITY,
        }
        response = {
            "location": {
                "query": location_query if used_named_fallback or (location_query and location_query != "Chooralmala, Wayanad" and (lat is None or lon is None)) else live_scan.location.query,
                "display_name": location_query if used_named_fallback else live_scan.location.display_name,
                "latitude": live_scan.location.latitude,
                "longitude": live_scan.location.longitude,
                "agency_mode": agency_mode,
                "nasa_telemetry": self._nasa_telemetry(live_scan),
                "reliability_report": reliability_report(live_scan),
            },
            "summary": summary,
            "h3_geojson": {"type": "FeatureCollection", "features": geo_features},
            "runout_geojson": {"type": "FeatureCollection", "features": routes},
            "source_status": {key: value.__dict__ for key, value in live_scan.sources.items()},
            "generated_at_utc": live_scan.generated_at_utc,
            "elapsed_seconds": live_scan.elapsed_seconds,
            "scenario": {
                "rain_delta_mm": rain_delta_mm,
                "storm_duration_hr": storm_duration_hr,
                "pga_override_g": pga_override_g,
                "snowmelt_temp_surge_c": snowmelt_temp_surge_c,
                "burn_severity_override": burn_severity_override,
            },
        }
        self._remember_contexts(cell_ids, {
            "scan": live_scan,
            "inputs": inputs,
            "state": state,
            "kinematics": kinematics,
            "probability": probability,
            "breakdown": breakdown,
            "hotspot_score": hotspot_score,
            "exposure": exposure,
            "top_index": top_index,
            "feature_rows": feature_rows,
        })
        return response

    def inspect_hexagon(self, h3_index: str) -> Dict[str, Any]:
        context = self._get_context(h3_index)
        scan: GlobalScan = context["scan"]
        cell = scan.terrain.cells[h3_index]
        properties = context["state"]
        index = context["cell_ids"].index(h3_index) if "cell_ids" in context else list(scan.terrain.cells).index(h3_index)
        state = context["state"]
        breakdown = context["breakdown"]
        return {
            "h3_index": h3_index,
            "latitude": cell.latitude,
            "longitude": cell.longitude,
            "elevation_m": cell.elevation_m,
            "slope_deg": cell.slope_deg,
            "twi": cell.twi,
            "fs_dynamic": float(state.fs_dynamic[index]),
            "p_f_fosm": float(state.p_failure_fosm[index]),
            "p_failure": float(context["probability"][index]),
            "pore_pressure_head_m": float(state.pore_pressure_head_m[index]),
            "newmark_displacement_cm": float(state.newmark_displacement_cm[index]),
            "peak_velocity_ms": float(context["kinematics"].peak_velocity_m_s[index]),
            "cruden_varnes_label": str(context["kinematics"].velocity_class_name[index]),
            "failure_mode": str(context["kinematics"].failure_mode[index]),
            "causal_breakdown_pct": {cause: float(values[index]) for cause, values in breakdown.combined_percentages.items()},
            "hotspot_score": float(context["hotspot_score"][index]),
            "baseline_hydrostatic_credited": bool(_field_capacity_stress_mask(context["inputs"], state)[index]),
            "physics_feature_provenance": self._cell_provenance(scan, h3_index),
        }

    def trace_runout(self, h3_index: str) -> Dict[str, Any]:
        context = self._get_context(h3_index)
        scan: GlobalScan = context["scan"]
        state = context["state"]
        kinematics = context["kinematics"]
        index = context["cell_ids"].index(h3_index)
        infra = scan.infrastructure.by_h3_cell if scan.infrastructure else {}
        route = route_runout(
            h3_index,
            scan.terrain.cells,
            float(kinematics.peak_velocity_m_s[index]),
            RUNOUT_MU_B,
            RUNOUT_XI,
            infrastructure_by_cell=infra,
            max_steps=50,
        )
        return {
            **route.to_dict(),
            "coordinates_lonlat": [
                [float(scan.terrain.cells[cell_id].longitude), float(scan.terrain.cells[cell_id].latitude)]
                for cell_id in route.path
            ],
            "kinematics": {
                "peak_velocity_ms": float(kinematics.peak_velocity_m_s[index]),
                "cruden_varnes_label": str(kinematics.velocity_class_name[index]),
                "failure_mode": str(kinematics.failure_mode[index]),
                "fs_dynamic": float(state.fs_dynamic[index]),
                "p_failure": float(context["probability"][index]),
            },
            "evacuation_planning_note": "Planning aid only; verify route conditions, occupancy, and agency guidance before operational use.",
            "route_parameters": {"mu_b": RUNOUT_MU_B, "xi": RUNOUT_XI, "quality": RUNOUT_PARAMETER_QUALITY},
        }

    def _runout_features(
        self,
        scan: GlobalScan,
        cell_ids: Sequence[str],
        state: Any,
        kinematics: Any,
        probability: np.ndarray,
        hotspot_score: np.ndarray,
    ) -> List[Dict[str, Any]]:
        critical = np.flatnonzero((state.fs_dynamic <= 1.05) | (probability >= 0.65))
        ordered = sorted(critical.tolist(), key=lambda index: float(hotspot_score[index]), reverse=True)[:5]
        infra = scan.infrastructure.by_h3_cell if scan.infrastructure else {}
        result = []
        for index in ordered:
            cell_id = cell_ids[index]
            route = route_runout(
                cell_id, scan.terrain.cells,
                float(kinematics.peak_velocity_m_s[index]), RUNOUT_MU_B, RUNOUT_XI,
                infrastructure_by_cell=infra, max_steps=50,
            )
            if len(route.path) < 2:
                continue
            coordinates = [
                [float(scan.terrain.cells[item].longitude), float(scan.terrain.cells[item].latitude)]
                for item in route.path
            ]
            result.append({
                "type": "Feature",
                "geometry": {"type": "LineString", "coordinates": coordinates},
                "properties": {
                    **route.to_dict(),
                    "peak_velocity_ms": float(kinematics.peak_velocity_m_s[index]),
                    "p_failure": float(probability[index]),
                    "hotspot_score": float(hotspot_score[index]),
                    "route_parameters": {"mu_b": RUNOUT_MU_B, "xi": RUNOUT_XI, "quality": RUNOUT_PARAMETER_QUALITY},
                },
            })
        return result

    @staticmethod
    def _alert_level(max_probability: float, rain_24h_mm: float) -> str:
        if max_probability >= ALERT_THRESHOLDS["alert"]:
            return "RED"
        if max_probability >= ALERT_THRESHOLDS["watch"]:
            return "ORANGE"
        if max_probability >= ALERT_THRESHOLDS["warning"]:
            return "YELLOW"
        return "GREEN"

    @staticmethod
    def _cell_provenance(scan: GlobalScan, cell_id: str) -> Dict[str, Any]:
        sat = scan.satellite_cells.get(cell_id, {})
        hydro = scan.hydrology
        soil = hydro.soil_property_quality if hydro else "fallback"
        return {
            "soil": soil,
            "vegetation": sat.get("vegetation_proxy_quality", "Esri visible-band VDVI or neutral fallback"),
            "burn": sat.get("burn_proxy_quality", "not assessed; default zero"),
            "road_distance": "OSM cut tag when mapped; otherwise 9999 m",
            "river_distance": "DEM H3 TWI >=14.5 proxy",
        }

    def _remember_contexts(self, cell_ids: Sequence[str], context: Dict[str, Any]) -> None:
        context["cell_ids"] = list(cell_ids)
        with self._context_lock:
            for cell_id in cell_ids:
                self._cell_contexts[cell_id] = context
                if cell_id in self._context_order:
                    self._context_order.remove(cell_id)
                self._context_order.append(cell_id)
            while len(self._context_order) > 12000:
                old = self._context_order.pop(0)
                self._cell_contexts.pop(old, None)

    def _get_context(self, cell_id: str) -> Dict[str, Any]:
        try:
            is_valid = h3.is_valid_cell(cell_id) if hasattr(h3, "is_valid_cell") else h3.h3_is_valid(cell_id)
        except Exception:
            is_valid = False
        if not is_valid:
            raise KeyError("invalid H3 cell index")
        with self._context_lock:
            context = self._cell_contexts.get(cell_id)
        if context is None:
            raise KeyError("H3 cell has no active scan context; run /api/v1/scan first")
        return context

    @staticmethod
    def _nasa_telemetry(scan):
        h=scan.hydrology
        return {
            "nasa_power_gwetroot": h.nasa_gwetroot_pct if h and h.nasa_gwetroot_pct is not None else None,
            "nasa_power_prectotcorr_mm": h.nasa_precipitation_mm_day if h else None,
            "nasa_eonet_active_events": scan.eonet_events,
            "nasa_coolr_regional_count_100km": scan.coolr_nearby.get("nasa_coolr_regional_count_100km",0),
            "nearest_coolr_event_km": scan.coolr_nearby.get("nearest_coolr_event_km"),
            "nasa_coolr_nearby_count": scan.coolr_nearby["count"], "coolr_count_provenance": scan.coolr_nearby,
            "srtm_terrarium_elevation_range_m": [scan.terrain.min_elevation_m, scan.terrain.max_elevation_m] if scan.terrain else None,
            "quality": {"power": "latest available daily values; not instantaneous",
                        "eonet": "returned event records with their dates; not a local hazard count",
                        "coolr": scan.coolr_nearby["quality"],
                        "nasadem": "not fetched; terrain uses AWS Terrarium, not verified NASADEM"},
            "source_status": {k:v.__dict__ for k,v in scan.sources.items() if k.startswith('nasa_') or k.startswith('eonet_')},
        }

    def _empty_response(self, scan: GlobalScan, agency_mode: str) -> Dict[str, Any]:
        return {
            "location": {"query": scan.location.query, "display_name": scan.location.display_name, "latitude": scan.location.latitude, "longitude": scan.location.longitude, "agency_mode": agency_mode, "nasa_telemetry": self._nasa_telemetry(scan), "reliability_report": reliability_report(scan)},
            "summary": {
                "wmo_alert_level": "UNAVAILABLE", "max_p_failure": None, "min_fs_dynamic": None,
                "dominant_cause": None, "critical_hotspots_count": 0, "total_households_exposed": None,
                "causal_breakdown_pct": {cause: 20.0 for cause in CAUSES}, "kinematics_top_hotspot": None,
                "causal_attribution_valid": False, "causal_attribution_quality": "neutral_prior_no_terrain",
                "numbers_that_matter": [{"metric": key, "value": 0.0, "unit": unit, "threshold": "unavailable", "status": "NORMAL", "causal_group": "unavailable", "weight_percent": 0.0, "available": False}
                    for key,unit in [("dynamic_factor_of_safety","ratio"),("blended_failure_probability","%"),("peak_velocity","m/s"),("pore_pressure_head","m"),("slope_angle","degrees"),("newmark_displacement","cm")]],
                "meteorological_metrics": {}, "physics_status": "no sampled land cells",
            },
            "h3_geojson": {"type": "FeatureCollection", "features": []},
            "runout_geojson": {"type": "FeatureCollection", "features": []},
            "source_status": {key: value.__dict__ for key, value in scan.sources.items()},
            "generated_at_utc": scan.generated_at_utc,
            "elapsed_seconds": scan.elapsed_seconds,
        }
