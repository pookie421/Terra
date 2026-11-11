"""Baseline-gated cryospheric scenarios; warming alone cannot create snow."""
from dataclasses import replace
import numpy as np

ALPINE_ASSUMPTION = "Temperature lapse is applied only with explicit source elevation. Warming never creates snow or freeze-thaw observations; unverified joint ice does not weaken soil."

def cryospheric_eligibility(snow_depth_m, baseline_temperature_c, elevation_m, latitude_deg):
    line = np.maximum(1800.0, 4600.0 - 55.0 * np.abs(latitude_deg))
    return (np.asarray(snow_depth_m) > .005) | (np.asarray(baseline_temperature_c) <= 2.5) | (np.asarray(elevation_m) >= line)

def apply_alpine_weather(inputs, elevations_m, snowmelt_temp_surge_c=0.0, latitude_deg=None):
    elevation = np.asarray(elevations_m, dtype=float)
    latitude = inputs.latitude_deg if latitude_deg is None else latitude_deg
    reference = np.asarray(inputs.weather_reference_elevation_m)
    lapse = np.where(reference > -998, .0065*(elevation-reference), 0.)
    baseline = np.asarray(inputs.air_temperature_c) - lapse
    eligible = cryospheric_eligibility(inputs.snow_depth_m, baseline, elevation, latitude)
    snow = np.broadcast_to(np.asarray(inputs.snow_depth_m), elevation.shape).copy()
    cycles = np.broadcast_to(np.asarray(inputs.freeze_thaw_cycles_7d), elevation.shape).copy()
    return replace(inputs, air_temperature_c=baseline + snowmelt_temp_surge_c,
                   soil_temperature_c=np.asarray(inputs.soil_temperature_c)-lapse+.5*snowmelt_temp_surge_c,
                   snow_depth_m=snow, freeze_thaw_cycles_7d=cycles, elevation_m=elevation,
                   latitude_deg=latitude, warming_delta_c=snowmelt_temp_surge_c,
                   baseline_air_temperature_c=baseline, observed_snow_depth_m=inputs.snow_depth_m)
