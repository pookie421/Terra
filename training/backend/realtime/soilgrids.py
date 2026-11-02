"""SoilGrids WCS retrieval and auditable topographic-climatic fallback."""

from __future__ import annotations

import asyncio
import io
import math
from typing import Any, Dict, Mapping, Optional, Tuple

import httpx
import numpy as np

SOILGRIDS_WCS = "https://maps.isric.org/mapserv"
PROPERTY_MAPS = {"bdod": "/map/bdod.map", "clay": "/map/clay.map", "sand": "/map/sand.map", "soc": "/map/soc.map"}
CRS_152160 = "http://www.opengis.net/def/crs/EPSG/0/152160"


def estimate_pedotransfer(
    slope_deg: float,
    profile_curvature: float,
    twi: float,
    moisture_regime_pct: Optional[float] = None,
) -> Dict[str, Any]:
    """Return plausible coupled fractions with an explicit estimated provenance."""
    slope = float(np.clip(slope_deg, 0.0, 80.0)) / 45.0
    curvature = float(np.clip(profile_curvature, -0.08, 0.08))
    wetness = float(np.clip((twi - 4.0) / 12.0, 0.0, 1.0))
    if moisture_regime_pct is not None and math.isfinite(float(moisture_regime_pct)):
        wetness = 0.5 * wetness + 0.5 * float(np.clip(moisture_regime_pct / 100.0, 0.0, 1.0))
    concavity = float(np.clip(-curvature / 0.08, 0.0, 1.0))
    sand = float(np.clip(0.40 + 0.28 * slope - 0.10 * wetness - 0.04 * concavity, 0.10, 0.82))
    clay = float(np.clip(0.24 - 0.12 * slope + 0.17 * wetness + 0.10 * concavity, 0.06, 0.58))
    if sand + clay > 0.92:
        clay = 0.92 - sand
    silt = float(max(0.03, 1.0 - sand - clay))
    organic = float(np.clip(0.012 + 0.038 * wetness + 0.012 * concavity - 0.007 * slope, 0.005, 0.06))
    bulk_density = float(np.clip(1.58 + 0.12 * slope - 0.30 * organic - 0.16 * wetness, 1.10, 1.85))
    return {
        "sand_fraction": sand,
        "clay_fraction": clay,
        "silt_fraction": silt,
        "organic_matter_fraction": organic,
        "bulk_density_g_cm3": bulk_density,
        "sand_g_kg": sand * 1000.0,
        "clay_g_kg": clay * 1000.0,
        "soc_g_kg": organic * 1000.0 / 1.724,
        "bdod_kg_m3": bulk_density * 1000.0,
        "source": "topographic_climatic_pedotransfer_estimate",
        "quality": "estimated_not_measured",
    }


async def fetch_soilgrids_wcs(
    client: httpx.AsyncClient,
    latitude: float,
    longitude: float,
    timeout_s: float = 2.5,
) -> Dict[str, float]:
    """Fetch small WCS subsets when optional pyproj/rasterio decoding is installed."""
    try:
        from pyproj import Transformer
        import rasterio
    except ImportError as exc:
        raise RuntimeError("SoilGrids WCS decoding requires optional pyproj and rasterio") from exc
    transform = Transformer.from_crs("EPSG:4326", "EPSG:152160", always_xy=True)
    x, y = transform.transform(longitude, latitude)
    half_width = 25000.0
    bounds = (x - half_width, x + half_width, y - half_width, y + half_width)
    properties = ("sand", "clay", "bdod", "soc")

    async def fetch_one(prop: str) -> Tuple[str, float]:
        params = [
            ("SERVICE", "WCS"), ("VERSION", "2.0.1"), ("REQUEST", "GetCoverage"),
            ("COVERAGEID", "{}_0-5cm_Q0.5".format(prop)), ("FORMAT", "GEOTIFF_INT16"),
            ("SUBSET", "X({:.3f},{:.3f})".format(bounds[0], bounds[1])),
            ("SUBSET", "Y({:.3f},{:.3f})".format(bounds[2], bounds[3])),
            ("SUBSETTINGCRS", CRS_152160), ("OUTPUTCRS", CRS_152160),
        ]
        response = await client.get(SOILGRIDS_WCS, params=params, timeout=timeout_s)
        response.raise_for_status()
        with rasterio.MemoryFile(response.content) as memfile:
            with memfile.open() as dataset:
                data = dataset.read(1, masked=True).compressed()
        valid = data[np.isfinite(data) & (data >= 0)]
        if not len(valid):
            raise ValueError("WCS returned no valid {} coverage pixels".format(prop))
        value = float(np.mean(valid))
        if prop == "bdod":
            value *= 10.0  # centigrams/cm³ to kg/m³
        return prop, value

    results = await asyncio.gather(*(fetch_one(prop) for prop in properties))
    values = dict(results)
    return {"sand_g_kg": values["sand"], "clay_g_kg": values["clay"], "bdod_kg_m3": values["bdod"], "soc_g_kg": values["soc"], "source": "ISRIC SoilGrids WCS", "quality": "soilgrids_prediction"}
