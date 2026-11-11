#!/usr/bin/env python3
"""Run the requested three-location live ingestion proof."""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path
from typing import Any, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.realtime.global_ingestor import GlobalIngestor

LOCATIONS = [
    ("Chooralmala, Wayanad", 11.548, 76.134),
    ("Zermatt, Switzerland", 46.020, 7.749),
    ("Kaikoura, New Zealand", -42.400, 173.680),
]
REQUIRED_LIVE_SOURCES = ("terrarium_dem", "open_meteo", "nasa_power", "usgs", "overpass")


def fmt(value: Any, digits: int = 2, suffix: str = "") -> str:
    if value is None:
        return "unavailable"
    try:
        return "{:.{digits}f}{suffix}".format(float(value), digits=digits, suffix=suffix)
    except (TypeError, ValueError):
        return str(value)


def source_label(scan: Any, name: str) -> str:
    status = scan.sources[name]
    return "live" if status.available else "UNAVAILABLE ({})".format(status.error)


async def run(provider_timeout: float, strict: bool, radius_km: float) -> int:
    failures = []
    async with GlobalIngestor(provider_timeout_s=provider_timeout) as ingestor:
        for place, latitude, longitude in LOCATIONS:
            scan = await ingestor.scan((latitude, longitude), radius_km=radius_km)
            terrain = scan.terrain
            weather = scan.weather
            hydrology = scan.hydrology
            seismic = scan.seismic
            osm = scan.infrastructure
            print("\n{} ({:.3f}, {:.3f}) — scan {:.2f}s".format(place, latitude, longitude, scan.elapsed_seconds))
            print("  H3 res-9 cells: {} [{}]".format(terrain.h3_cell_count if terrain else "unavailable", source_label(scan, "terrarium_dem")))
            print(
                "  Elevation: {} to {} m; slope P95: {}; max TWI: {}".format(
                    fmt(terrain.min_elevation_m) if terrain else "unavailable",
                    fmt(terrain.max_elevation_m) if terrain else "unavailable",
                    fmt(terrain.slope_p95_deg, suffix="°") if terrain else "unavailable",
                    fmt(terrain.max_twi) if terrain else "unavailable",
                )
            )
            print(
                "  Rain: 24h {}; peak 3h {}; forecast +24h {}; +24–48h {}; ARI7d {} mm [{}]".format(
                    fmt(weather.rain_24h_mm, suffix=" mm") if weather else "unavailable",
                    fmt(weather.peak_rain_3h_mm, suffix=" mm") if weather else "unavailable",
                    fmt(weather.forecast_rain_next_24h_mm, suffix=" mm") if weather else "unavailable",
                    fmt(weather.forecast_rain_24_48h_mm, suffix=" mm") if weather else "unavailable",
                    fmt(weather.ari_7d_mm) if weather else "unavailable",
                    source_label(scan, "open_meteo"),
                )
            )
            saturation = hydrology.soil_saturation_pct if hydrology else None
            wetness = hydrology.blended_relative_wetness_pct if hydrology else None
            print(
                "  Soil: volumetric moisture {}; saturation {}; NASA/Open-Meteo blended wetness {}% [{}]".format(
                    fmt(weather.soil_moisture_0_100cm_m3m3, digits=3, suffix=" m³/m³") if weather else "unavailable",
                    fmt(saturation, suffix="%") if saturation is not None else "unavailable (porosity unavailable)",
                    fmt(wetness) if wetness is not None else "unavailable",
                    "NASA POWER: {}; SoilGrids: {}".format(
                        source_label(scan, "nasa_power"), source_label(scan, "soilgrids")
                    ),
                )
            )
            print(
                "  Snow depth: {}; freeze-thaw cycles (7d): {}".format(
                    fmt(weather.snow_depth_m, digits=3, suffix=" m") if weather else "unavailable",
                    weather.freeze_thaw_cycles_7d if weather else "unavailable",
                )
            )
            print(
                "  Seismic: {} events; max PGA {}; road cuts: {} [{}]".format(
                    seismic.event_count if seismic else "unavailable",
                    fmt(seismic.max_pga_g, digits=6, suffix=" g") if seismic else "unavailable",
                    osm.road_cut_count if osm else "unavailable",
                    "USGS: {}; Overpass: {}".format(source_label(scan, "usgs"), source_label(scan, "overpass")),
                )
            )
            if terrain and terrain.interpolated_pixels:
                print("  Terrain interpolation: {} missing pixels filled from observed neighbors".format(terrain.interpolated_pixels))
            for source in REQUIRED_LIVE_SOURCES:
                if not scan.sources[source].available:
                    failures.append("{}: {}".format(place, scan.sources[source].error))
    if strict and failures:
        print("\nLIVE PROOF FAILED: required live sources were unavailable:")
        for failure in failures:
            print("  - {}".format(failure))
        return 1
    if failures:
        print("\nPartial live scan: some required providers were unavailable; see source status above.")
    else:
        print("\nLIVE PROOF PASSED for all required sources at all three locations.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider-timeout", type=float, default=12.0, help="per-provider deadline in seconds")
    parser.add_argument("--radius-km", type=float, default=3.5)
    parser.add_argument("--allow-partial", action="store_true", help="return success while printing unavailable sources")
    args = parser.parse_args()
    return asyncio.run(run(args.provider_timeout, not args.allow_partial, args.radius_km))


if __name__ == "__main__":
    sys.exit(main())
