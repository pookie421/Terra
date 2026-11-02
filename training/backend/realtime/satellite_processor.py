"""Esri World Imagery RGB proxies, explicitly not multispectral NDVI products."""

from __future__ import annotations

import asyncio
import io
import math
from dataclasses import asdict, dataclass
from typing import Any, Dict, Mapping, Optional, Tuple

import httpx
import numpy as np
from PIL import Image

from .cache import AsyncTTLCache, DEFAULT_CACHE
from .natural_events import NaturalEvent, iter_lon_lat, point_in_event_perimeter

ESRI_WORLD_IMAGERY = "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"


@dataclass
class SatelliteCell:
    cell_id: str
    red: int
    green: int
    blue: int
    vdvi: float
    vegetation_index_norm_proxy: float
    bare_char_spectral_proxy: float
    wildfire_distance_km: Optional[float]
    burn_severity_proxy: float
    burn_override_applied: bool
    imagery_provider: str = "Esri World Imagery RGB z13"
    index_note: str = "Visible-band VDVI transform; not true NIR NDVI"
    burn_proxy_quality: str = "not_assessed_no_eonet_fire_evidence"
    burn_proxy_note: str = "Zero means no EONET-linked visible-band signal was available, not verified absence of burn."


class SatelliteProcessor:
    def __init__(self, zoom: int = 13, cache: Optional[AsyncTTLCache] = None) -> None:
        self.zoom = zoom
        self.cache = cache or DEFAULT_CACHE
        self._semaphore = asyncio.Semaphore(16)

    async def process_cells(
        self,
        cell_coordinates: Mapping[str, Tuple[float, float]],
        events: Mapping[str, NaturalEvent],
        client: httpx.AsyncClient,
        burn_override: Optional[Mapping[str, float]] = None,
    ) -> Dict[str, SatelliteCell]:
        tile_keys = {self.tile_xy(lat, lon) for lat, lon in cell_coordinates.values()}
        fetched = await asyncio.gather(*(self._get_tile(client, x, y) for x, y in tile_keys), return_exceptions=True)
        tiles = {key: value for key, value in zip(tile_keys, fetched) if not isinstance(value, BaseException)}
        overrides = burn_override or {}
        fire_events = [event for event in events.values() if "wildfire" in event.category.lower()]
        fire_points = [point for event in fire_events for point in iter_lon_lat(event.coordinates)]
        fire_latitudes = np.radians(np.asarray([point[1] for point in fire_points], dtype=np.float64))
        fire_longitudes = np.radians(np.asarray([point[0] for point in fire_points], dtype=np.float64))
        output: Dict[str, SatelliteCell] = {}
        for cell_id, (lat, lon) in cell_coordinates.items():
            x, y = self.tile_xy(lat, lon)
            if (x, y) not in tiles:
                continue
            rgb = self._pixel(tiles[(x, y)], lat, lon)
            if rgb is None:
                continue
            r, g, b = rgb
            vf = float((2 * g - r - b) / (2 * g + r + b + 1.0))
            vegetation = float(np.clip((vf + 0.10) / 0.45, 0.02, 0.98))
            brightness = max(r + g + b, 1)
            red_ratio = r / brightness
            # A visible-band bare/char indicator only; RGB cannot distinguish ash from soil reliably.
            bare_char = float(np.clip((red_ratio - 0.34) / 0.18, 0.0, 1.0)) if vegetation < 0.18 else 0.0
            inside_perimeter = any(
                point_in_event_perimeter(lat, lon, event)
                for event in fire_events if event.geometry_type.lower() == "polygon"
            )
            distance = self._nearest_distance_km(lat, lon, fire_latitudes, fire_longitudes)
            local_fire_evidence = inside_perimeter or (distance is not None and distance <= 25.0)
            proximity = 1.0 if inside_perimeter else (
                math.exp(-distance / 2.0) if distance is not None and distance <= 25.0 else 0.0
            )
            inferred_burn = float(np.clip(proximity * bare_char, 0.0, 1.0))
            override = cell_id in overrides
            burn = float(np.clip(overrides[cell_id], 0.0, 1.0)) if override else inferred_burn
            quality = "simulation_override" if override else (
                "estimated_eonet_rgb_proxy" if local_fire_evidence else "not_assessed_no_nearby_eonet_fire_evidence"
            )
            note = "User simulation override applied." if override else (
                "Visible-band bare/char proxy combined with EONET wildfire proximity."
                if local_fire_evidence else
                "No EONET wildfire geometry within 25 km; zero is no signal, not verified absence of burn."
            )
            output[cell_id] = SatelliteCell(
                cell_id, r, g, b, vf, vegetation, bare_char, distance, burn, override,
                burn_proxy_quality=quality, burn_proxy_note=note,
            )
        return output

    @staticmethod
    def _nearest_distance_km(
        latitude: float,
        longitude: float,
        event_latitudes_rad: np.ndarray,
        event_longitudes_rad: np.ndarray,
    ) -> Optional[float]:
        if not event_latitudes_rad.size:
            return None
        lat1 = math.radians(latitude)
        lon1 = math.radians(longitude)
        delta_lat = event_latitudes_rad - lat1
        delta_lon = event_longitudes_rad - lon1
        haversine = (
            np.sin(delta_lat / 2.0) ** 2
            + math.cos(lat1) * np.cos(event_latitudes_rad) * np.sin(delta_lon / 2.0) ** 2
        )
        nearest = float(np.min(haversine))
        return 6371.0088 * 2.0 * math.atan2(math.sqrt(nearest), math.sqrt(max(0.0, 1.0 - nearest)))

    async def _get_tile(self, client: httpx.AsyncClient, x: int, y: int) -> np.ndarray:
        key = ("esri-world-imagery", self.zoom, x, y)

        async def load() -> np.ndarray:
            async with self._semaphore:
                response = await client.get(ESRI_WORLD_IMAGERY.format(z=self.zoom, x=x, y=y))
            response.raise_for_status()
            with Image.open(io.BytesIO(response.content)) as image:
                rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
            if rgb.shape[:2] != (256, 256):
                raise ValueError("unexpected Esri tile dimensions {}".format(rgb.shape[:2]))
            return rgb

        return await self.cache.get_or_create(key, 7 * 24 * 3600, load)

    def tile_xy(self, latitude: float, longitude: float) -> Tuple[int, int]:
        lat = min(max(latitude, -85.05112878), 85.05112878)
        n = 2**self.zoom
        x_float = (longitude + 180.0) / 360.0 * n
        lat_rad = math.radians(lat)
        y_float = (1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n
        return int(x_float) % n, min(max(int(y_float), 0), n - 1)

    def _pixel(self, tile: np.ndarray, latitude: float, longitude: float) -> Optional[Tuple[int, int, int]]:
        n = 2**self.zoom
        lat = min(max(latitude, -85.05112878), 85.05112878)
        x_float = (longitude + 180.0) / 360.0 * n
        lat_rad = math.radians(lat)
        y_float = (1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n
        px = min(max(int((x_float % 1.0) * 256), 0), 255)
        py = min(max(int((y_float % 1.0) * 256), 0), 255)
        values = tile[py, px]
        return int(values[0]), int(values[1]), int(values[2])
