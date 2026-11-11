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
HOMOLOSINE_PROJ = "+proj=igh +lat_0=0 +lon_0=0 +datum=WGS84 +units=m +no_defs"


def estimate_pedotransfer(
    slope_deg: float,
    profile_curvature: float,
    twi: float,
    moisture_regime_pct: Optional[float] = None,
) -> Dict[str, Any]:
    """Static terrain prior, not a measured soil or a weather-dependent estimate.

    ``moisture_regime_pct`` is retained for call compatibility and ignored.
    """
    slope = float(np.clip(slope_deg, 0.0, 80.0)) / 45.0
    curvature = float(np.clip(profile_curvature, -0.08, 0.08))
    wetness = float(np.clip((twi - 4.0) / 12.0, 0.0, 1.0))
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
        "source": "static_topographic_soil_prior",
        "quality": "estimated_not_measured",
    }


def soilgrids_to_conventional(name: str, value: float) -> float:
    """Decode ISRIC mapped units once: bdod cg/cm³, SOC dg/kg."""
    return float(value) * {"bdod": 10.0, "soc": 0.1}.get(name, 1.0)


def derive_soil_properties(raw: Mapping[str, Any]) -> Dict[str, Any]:
    """Derive every physical property from the same completed raw record."""
    result = dict(raw)
    sand, clay = float(raw['sand_g_kg']) / 1000., float(raw['clay_g_kg']) / 1000.
    density = float(raw['bdod_kg_m3']) / 1000.
    organic = float(raw['soc_g_kg']) * 1.724 / 1000.
    if not all(math.isfinite(v) for v in (sand, clay, density, organic)):
        raise ValueError('Nonfinite soil properties')
    if min(sand, clay, organic) < 0 or sand + clay > 1 or organic > 1 or not 0 < density < 2.65:
        raise ValueError('Incompatible soil composition or bulk density')
    result.update(sand_fraction=sand, clay_fraction=clay, silt_fraction=1.-sand-clay,
                  organic_matter_fraction=organic, bulk_density_g_cm3=density,
                  porosity_fraction=1.-density/2.65)
    return result


async def fetch_soilgrids_wcs(
    client: httpx.AsyncClient,
    latitude: float,
    longitude: float,
    timeout_s: float = 2.5,
) -> Dict[str, Any]:
    """Fetch small WCS subsets when optional pyproj/rasterio decoding is installed."""
    try:
        from pyproj import Transformer
        import rasterio
    except ImportError as exc:
        raise RuntimeError("SoilGrids WCS decoding requires optional pyproj and rasterio") from exc
    transform = Transformer.from_crs("EPSG:4326", HOMOLOSINE_PROJ, always_xy=True)
    x, y = transform.transform(longitude, latitude)
    half_width = 1500.0
    bounds = (x - half_width, x + half_width, y - half_width, y + half_width)
    properties = ("sand", "clay", "bdod", "soc")

    semaphore = asyncio.Semaphore(4)
    async def fetch_one(prop: str, depth: str) -> Tuple[str, str, float]:
        params = [
            ("map", PROPERTY_MAPS[prop]),
            ("SERVICE", "WCS"), ("VERSION", "2.0.1"), ("REQUEST", "GetCoverage"),
            ("COVERAGEID", "{}_{}_mean".format(prop,depth)), ("FORMAT", "GEOTIFF_INT16"),
            ("SUBSET", "X({:.3f},{:.3f})".format(bounds[0], bounds[1])),
            ("SUBSET", "Y({:.3f},{:.3f})".format(bounds[2], bounds[3])),
            ("SUBSETTINGCRS", CRS_152160), ("OUTPUTCRS", CRS_152160),
        ]
        async with semaphore:
            response = await client.get(SOILGRIDS_WCS, params=params, timeout=timeout_s)
        response.raise_for_status()
        with rasterio.MemoryFile(response.content) as memfile:
            with memfile.open() as dataset:
                data = dataset.read(1, masked=True).compressed()
        valid = data[np.isfinite(data) & (data >= 0)]
        if not len(valid):
            raise ValueError("WCS returned no valid {} coverage pixels".format(prop))
        value = float(np.mean(valid))
        return prop, depth, soilgrids_to_conventional(prop, value)

    depths = {"0-5cm":5., "5-15cm":10., "15-30cm":15.}
    fetched = await asyncio.gather(*(fetch_one(prop,depth) for prop in properties for depth in depths),return_exceptions=True)
    results=[r for r in fetched if not isinstance(r,BaseException)]
    values = {prop:sum(v*depths[d] for p,d,v in results if p==prop)/30. if sum(p==prop for p,d,v in results)==3 else None for prop in properties}
    if all(v is None for v in values.values()):raise ValueError('No complete 0-30 cm SoilGrids property returned')
    return {"sand_g_kg": values["sand"], "clay_g_kg": values["clay"], "bdod_kg_m3": values["bdod"], "soc_g_kg": values["soc"], "source": "ISRIC SoilGrids WCS", "quality": "soilgrids_prediction" if all(v is not None for v in values.values()) else 'partial_soilgrids_prediction', "support":"3 km patch; thickness-weighted 0-30 cm mean predictions",
        'property_errors':[str(r) for r in fetched if isinstance(r,BaseException)]}
