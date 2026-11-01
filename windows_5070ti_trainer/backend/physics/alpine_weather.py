"""Baseline-gated cryospheric scenarios; warming alone cannot create snow."""
from dataclasses import replace
import numpy as np

ALPINE_ASSUMPTION = "Lapse-rate and latitude/elevation screen are scenario proxies, not observed permafrost. Snow/cycle floors apply only inside the baseline cryospheric screen."

def cryospheric_eligibility(snow_depth_m, baseline_temperature_c, elevation_m, latitude_deg):
    line = np.maximum(1800.0, 4600.0 - 55.0 * np.abs(latitude_deg))
    return (np.asarray(snow_depth_m) > .005) | (np.asarray(baseline_temperature_c) <= 2.5) | (np.asarray(elevation_m) >= line)

def apply_alpine_weather(inputs, elevations_m, snowmelt_temp_surge_c=0.0, latitude_deg=None):
    elevation = np.asarray(elevations_m, dtype=float)
    latitude = inputs.latitude_deg if latitude_deg is None else latitude_deg
    lapse = .0065 * np.maximum(0.0, elevation - elevation.min())
    baseline = np.asarray(inputs.air_temperature_c) - lapse
    eligible = cryospheric_eligibility(inputs.snow_depth_m, baseline, elevation, latitude)
    snow_floor = .40 * np.clip((elevation - 1800.0) / 1000.0, .2, 1.5)
    snow = np.where(eligible, np.maximum(inputs.snow_depth_m, snow_floor), inputs.snow_depth_m)
    cycles = np.where(eligible, np.maximum(inputs.freeze_thaw_cycles_7d, 3.), inputs.freeze_thaw_cycles_7d)
    return replace(inputs, air_temperature_c=baseline + snowmelt_temp_surge_c,
                   soil_temperature_c=np.asarray(inputs.soil_temperature_c)-lapse+.5*snowmelt_temp_surge_c,
                   snow_depth_m=snow, freeze_thaw_cycles_7d=cycles, elevation_m=elevation,
                   latitude_deg=latitude, warming_delta_c=snowmelt_temp_surge_c,
                   baseline_air_temperature_c=baseline, observed_snow_depth_m=inputs.snow_depth_m)
