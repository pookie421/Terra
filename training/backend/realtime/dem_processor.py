"""On-demand Terrarium DEM mosaics and H3 resolution-9 terrain metrics."""

from __future__ import annotations

import asyncio
import io
import math
from pathlib import Path
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

import h3
import httpx
import numpy as np
from PIL import Image

from .cache import AsyncTTLCache, DEFAULT_CACHE

EARTH_CIRCUMFERENCE_M = 40075016.686
TERRARIUM_URL = "https://s3.amazonaws.com/elevation-tiles-prod/terrarium/{z}/{x}/{y}.png"
TERRAIN_VERSION = "metric_specific_area_bounded_1000m_v1"
CATCHMENT_LENGTH_M = 1000.0


class TerrainDataError(RuntimeError):
    """Raised when real terrain observations are not sufficient to derive a surface."""


@dataclass
class TerrainCell:
    cell_id: str
    latitude: float
    longitude: float
    elevation_m: float
    slope_deg: float
    aspect_deg: float
    profile_curvature: float
    contributing_area_m2: float = 0.0
    twi: float = 2.0
    downslope_neighbors: List[str] = field(default_factory=list)


@dataclass
class TerrainScan:
    center_latitude: float
    center_longitude: float
    radius_km: float
    resolution: int
    zoom: int
    pixel_spacing_x_m: float
    pixel_spacing_y_m: float
    tile_count: int
    tile_ids: List[str]
    h3_cell_count: int
    min_elevation_m: float
    max_elevation_m: float
    slope_p95_deg: float
    max_twi: float
    interpolated_pixels: int
    cells: Dict[str, TerrainCell]
    warnings: List[str] = field(default_factory=list)
    terrain_version: str = 'legacy_unverified'

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class DEMProcessor:
    """Fetch and analyze Terrarium tiles for a global coordinate window."""

    def __init__(
        self,
        zoom: int = 13,
        resolution: int = 9,
        cache: Optional[AsyncTTLCache] = None,
        tile_concurrency: int = 16,
        max_fill_iterations: int = 8,
    ) -> None:
        self.zoom = zoom
        self.resolution = resolution
        self.cache = cache or DEFAULT_CACHE
        self._tile_concurrency=max(1,tile_concurrency)
        self._tile_semaphore=None
        self.max_fill_iterations = max_fill_iterations

    async def process(
        self, latitude: float, longitude: float, radius_km: float, client: httpx.AsyncClient
    ) -> TerrainScan:
        self._validate_coordinate(latitude, longitude)
        if radius_km <= 0 or radius_km > 100:
            raise ValueError("radius_km must be in (0, 100]")
        if abs(latitude) > 85.05112878:
            raise TerrainDataError(
                "Terrarium XYZ tiles do not cover latitudes beyond 85.0511 degrees; polar terrain is unavailable"
            )

        output_bounds = self._bbox(latitude, longitude, radius_km)
        output_cells = self._cover_bbox(*output_bounds)
        # Every output cell gets the same bounded upstream support regardless of
        # viewport size. Extra 600 m covers overlap cells, samples and derivatives.
        south, west, north, east = self._bbox(latitude, longitude, radius_km + CATCHMENT_LENGTH_M/1000. + .6)
        cells = self._cover_bbox(south, west, north, east)
        tile_xs, min_y, max_y = self._tile_ranges(south, west, north, east)
        tile_coords = [(x, y) for y in range(min_y, max_y + 1) for x in tile_xs]
        tile_results = await asyncio.gather(
            *(self._fetch_tile(client, x, y) for x, y in tile_coords), return_exceptions=True
        )

        mosaic = np.full((len(range(min_y, max_y + 1)) * 256, len(tile_xs) * 256), np.nan, dtype=np.float32)
        warnings: List[str] = []
        if latitude - radius_km / 111.32 < -85.05112878 or latitude + radius_km / 111.32 > 85.05112878:
            warnings.append("Viewport extends beyond the supported Web Mercator latitude coverage")
        tile_ids: List[str] = []
        successful = 0
        for (tile_x, tile_y), result in zip(tile_coords, tile_results):
            tile_ids.append("{}/{}/{}".format(self.zoom, tile_x, tile_y))
            if isinstance(result, BaseException):
                warnings.append("Terrarium tile {}/{} unavailable: {}".format(tile_x, tile_y, result))
                continue
            tile, tile_valid = result
            x_position = tile_xs.index(tile_x)
            row0 = (tile_y - min_y) * 256
            col0 = x_position * 256
            values = mosaic[row0 : row0 + 256, col0 : col0 + 256]
            values[tile_valid] = tile[tile_valid]
            successful += 1

        if successful == 0:
            raise TerrainDataError("No Terrarium elevation tiles were available")
        mosaic, interpolated_pixels = self._interpolate_small_voids(mosaic)
        if not np.isfinite(mosaic).all():
            missing = int(np.size(mosaic) - np.isfinite(mosaic).sum())
            raise TerrainDataError(
                "Terrain mosaic still has {} unsupported pixels after bounded neighbor interpolation".format(missing)
            )

        delta_x = EARTH_CIRCUMFERENCE_M * math.cos(math.radians(latitude)) / (256.0 * (2**self.zoom))
        delta_y = delta_x
        slope_deg, aspect_deg, curvature = self.derivatives(mosaic, delta_x, delta_y)

        cell_map = self._sample_cells(
            cells, latitude, longitude, tile_xs, min_y, mosaic, slope_deg, aspect_deg, curvature
        )
        sampled_count = len(cell_map)
        cell_map = {cell_id: cell for cell_id, cell in cell_map.items() if cell.elevation_m > 0.5}
        removed_ocean = sampled_count - len(cell_map)
        if removed_ocean:
            warnings.append("Elevation <=0.5 m mask is provisional; below-sea-level land needs a land mask")
        if not cell_map:
            raise TerrainDataError("No sampled land cells")
        self._route_contributing_area(cell_map)
        cell_map = {key: value for key, value in cell_map.items() if key in output_cells}
        values = list(cell_map.values())
        if not values:
            raise TerrainDataError("The requested bounding box did not produce usable H3 terrain cells")
        warnings.append(TERRAIN_VERSION + "; contributing paths limited to 1000 m; contour width=sqrt(H3 area)")

        return TerrainScan(
            center_latitude=latitude, center_longitude=longitude, radius_km=radius_km,
            resolution=self.resolution, zoom=self.zoom, pixel_spacing_x_m=delta_x,
            pixel_spacing_y_m=delta_y, tile_count=successful, tile_ids=tile_ids,
            h3_cell_count=len(cell_map), min_elevation_m=float(min(c.elevation_m for c in values)),
            max_elevation_m=float(max(c.elevation_m for c in values)),
            slope_p95_deg=float(np.percentile([c.slope_deg for c in values],95)),
            max_twi=float(max(c.twi for c in values)), interpolated_pixels=interpolated_pixels,
            cells=cell_map, warnings=warnings,terrain_version=TERRAIN_VERSION)

    @staticmethod
    def derivatives(mosaic, delta_x, delta_y):
        """Metric derivatives; raster rows increase southwards, bearings clockwise from north."""
        q, p = np.gradient(np.asarray(mosaic, float), delta_y, delta_x)
        dp_dy, r = np.gradient(p, delta_y, delta_x)
        t, dq_dx = np.gradient(q, delta_y, delta_x)
        s = (dp_dy + dq_dx) * 0.5

        gradient_squared = p * p + q * q
        slope_rad = np.arctan(np.sqrt(gradient_squared))
        slope_deg = np.degrees(slope_rad)
        aspect_deg = np.mod(np.degrees(np.arctan2(-p, q)), 360.0)
        curvature = -(
            p * p * r + 2.0 * p * q * s + q * q * t
        ) / ((gradient_squared + 1e-6) * np.power(1.0 + gradient_squared, 1.5))
        curvature = np.clip(curvature, -0.08, 0.08)

        return slope_deg, aspect_deg, curvature

    async def _fetch_tile(self, client: httpx.AsyncClient, tile_x: int, tile_y: int) -> Tuple[np.ndarray, np.ndarray]:
        if self._tile_semaphore is None:self._tile_semaphore=asyncio.Semaphore(self._tile_concurrency)
        key = ("terrarium", self.zoom, tile_x, tile_y)

        async def load() -> Tuple[np.ndarray, np.ndarray]:
            url = TERRARIUM_URL.format(z=self.zoom, x=tile_x, y=tile_y)
            path = Path(__file__).resolve().parents[2]/'data/source_catalogs/terrarium_tiles'/str(self.zoom)/str(tile_x)/(str(tile_y)+'.png')
            if path.exists():
                content = path.read_bytes()
            else:
                async with self._tile_semaphore:
                    response = await client.get(url)
                response.raise_for_status()
                content = response.content
                # Decode before persisting, so provider error bodies never poison the cache.
                with Image.open(io.BytesIO(content)) as checked:
                    checked.verify()
                path.parent.mkdir(parents=True,exist_ok=True)
                path.write_bytes(content)
            with Image.open(io.BytesIO(content)) as image:
                rgba = np.asarray(image.convert("RGBA"), dtype=np.uint8)
            if rgba.shape[:2] != (256, 256):
                raise TerrainDataError("Unexpected Terrarium tile size {}".format(rgba.shape[:2]))
            rgb = rgba[:, :, :3].astype(np.float32)
            elevation = 256.0 * rgb[:, :, 0] + rgb[:, :, 1] + rgb[:, :, 2] / 256.0 - 32768.0
            valid = (rgba[:, :, 3] > 0) & np.isfinite(elevation) & (elevation >= -12000.0) & (elevation <= 10000.0)
            elevation[~valid] = np.nan
            return elevation, valid

        return await self.cache.get_or_create(key, 7 * 24 * 3600, load)

    @staticmethod
    def _validate_coordinate(latitude: float, longitude: float) -> None:
        if not (-90.0 <= latitude <= 90.0 and -180.0 <= longitude <= 180.0):
            raise ValueError("latitude/longitude are outside valid global bounds")

    @staticmethod
    def _bbox(latitude: float, longitude: float, radius_km: float) -> Tuple[float, float, float, float]:
        lat_delta = radius_km / 111.32
        cos_lat = max(abs(math.cos(math.radians(latitude))), 0.01)
        lon_delta = min(180.0, radius_km / (111.32 * cos_lat))
        return (
            max(-85.05112878, latitude - lat_delta),
            longitude - lon_delta,
            min(85.05112878, latitude + lat_delta),
            longitude + lon_delta,
        )

    def _cover_bbox(self, south: float, west: float, north: float, east: float) -> Set[str]:
        polygon_boxes: List[Tuple[float, float, float, float]]
        if west < -180.0:
            polygon_boxes = [(south, west + 360.0, north, 180.0), (south, -180.0, north, east)]
        elif east > 180.0:
            polygon_boxes = [(south, west, north, 180.0), (south, -180.0, north, east - 360.0)]
        else:
            polygon_boxes = [(south, west, north, east)]

        cover: Set[str] = set()
        for s, w, n, e in polygon_boxes:
            points = [(s, w), (s, e), (n, e), (n, w), (s, w)]
            shape = None
            if hasattr(h3, "LatLngPoly"):
                shape = h3.LatLngPoly(points)
            generated: Iterable[str] = []
            try:
                if shape is not None and hasattr(h3, "h3shape_to_cells_experimental"):
                    generated = h3.h3shape_to_cells_experimental(shape, self.resolution, contain="overlap")
                elif shape is not None and hasattr(h3, "polygon_to_cells_experimental"):
                    generated = h3.polygon_to_cells_experimental(shape, self.resolution, contain="overlap")
                elif hasattr(h3, "geo_to_cells"):
                    generated = h3.geo_to_cells(
                        {"type": "Polygon", "coordinates": [[(w, s), (e, s), (e, n), (w, n), (w, s)]]},
                        self.resolution,
                    )
                elif shape is not None and hasattr(h3, "polygon_to_cells"):
                    generated = h3.polygon_to_cells(shape, self.resolution)
                elif hasattr(h3, "polyfill"):
                    generated = h3.polyfill(
                        {"type": "Polygon", "coordinates": [[(w, s), (e, s), (e, n), (w, n), (w, s)]]},
                        self.resolution,
                        geo_json_conformant=True,
                    )
            except (TypeError, ValueError, AttributeError):
                # Older h3-py lacks overlap containment; center-fill plus one ring
                # includes boundary-crossing cells while remaining globally valid.
                generated = []
            generated = list(generated)
            cover.update(generated)
            if generated:
                continue
            center_filler = getattr(h3, "geo_to_cells", None)
            try:
                if center_filler:
                    centers = list(center_filler(
                        {"type": "Polygon", "coordinates": [[(w, s), (e, s), (e, n), (w, n), (w, s)]]},
                        self.resolution,
                    ))
                elif shape is not None and hasattr(h3, "polygon_to_cells"):
                    centers = list(h3.polygon_to_cells(shape, self.resolution))
                elif hasattr(h3, "polyfill"):
                    centers = list(h3.polyfill(
                        {"type": "Polygon", "coordinates": [[(w, s), (e, s), (e, n), (w, n), (w, s)]]},
                        self.resolution,
                        geo_json_conformant=True,
                    ))
                else:
                    raise TerrainDataError("Installed h3-py version has no supported polygon fill API")
            except (TypeError, ValueError) as exc:
                raise TerrainDataError("Unable to cover bounding box with H3 cells: {}".format(exc))
            cover.update(centers)
            disk = getattr(h3, "grid_disk", getattr(h3, "k_ring", None))
            if disk:
                for cell in centers:
                    cover.update(disk(cell, 1))

        if not cover:
            raise TerrainDataError("H3 returned an empty cover for the requested bounding box")
        return cover

    def _tile_ranges(
        self, south: float, west: float, north: float, east: float
    ) -> Tuple[List[int], int, int]:
        n = 2**self.zoom

        def tile_x(lon: float) -> int:
            return int(math.floor((lon + 180.0) / 360.0 * n)) % n

        def tile_y(lat: float) -> int:
            lat = min(85.05112878, max(-85.05112878, lat))
            lat_rad = math.radians(lat)
            y = (1.0 - math.asinh(math.tan(lat_rad)) / math.pi) * 0.5 * n
            return max(0, min(n - 1, int(math.floor(y))))

        x_start, x_end = tile_x(west), tile_x(east)
        if east - west >= 360.0:
            x_values = list(range(n))
        else:
            x_values = []
            x = x_start
            while True:
                x_values.append(x)
                if x == x_end or len(x_values) > n:
                    break
                x = (x + 1) % n
        y_north, y_south = tile_y(north), tile_y(south)
        return x_values, min(y_north, y_south), max(y_north, y_south)

    def _interpolate_small_voids(self, surface: np.ndarray) -> Tuple[np.ndarray, int]:
        missing = ~np.isfinite(surface)
        initial_count = int(missing.sum())
        if initial_count == 0:
            return surface, 0
        height, width = surface.shape
        for _ in range(self.max_fill_iterations):
            missing = ~np.isfinite(surface)
            if not missing.any():
                break
            sums = np.zeros_like(surface, dtype=np.float32)
            weights = np.zeros_like(surface, dtype=np.float32)
            for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)):
                source_rows = slice(max(0, -dy), min(height, height - dy))
                source_cols = slice(max(0, -dx), min(width, width - dx))
                target_rows = slice(max(0, dy), min(height, height + dy))
                target_cols = slice(max(0, dx), min(width, width + dx))
                source = surface[source_rows, source_cols]
                valid = np.isfinite(source)
                weight = 0.70710678 if dx and dy else 1.0
                target_sum = sums[target_rows, target_cols]
                target_weight = weights[target_rows, target_cols]
                target_sum[valid] += source[valid] * weight
                target_weight[valid] += weight
            can_fill = missing & (weights > 0)
            surface[can_fill] = sums[can_fill] / weights[can_fill]
        remaining = int((~np.isfinite(surface)).sum())
        if remaining:
            return surface, initial_count - remaining
        return surface, initial_count

    def _sample_cells(
        self,
        cells: Set[str],
        center_lat: float,
        center_lon: float,
        tile_xs: Sequence[int],
        min_tile_y: int,
        elevation: np.ndarray,
        slope: np.ndarray,
        aspect: np.ndarray,
        curvature: np.ndarray,
    ) -> Dict[str, TerrainCell]:
        n = 2**self.zoom
        cell_ids = list(cells)
        center_points = [self._cell_to_latlng(cell) for cell in cell_ids]
        latitudes: List[float] = []
        longitudes: List[float] = []
        owner: List[int] = []
        # Center plus an interior ring samples the hex instead of one noisy pixel.
        bearings = [0.0] + [math.radians(angle) for angle in range(0, 360, 60)]
        radii_m = [0.0] + [90.0] * 6
        for idx, (lat, lon) in enumerate(center_points):
            latitudes.append(lat)
            longitudes.append(lon)
            owner.append(idx)
            for radius, bearing in zip(radii_m[1:], bearings[1:]):
                latitudes.append(lat + math.degrees(radius * math.cos(bearing) / 6371008.8))
                cos_lat = max(0.01, abs(math.cos(math.radians(lat))))
                longitudes.append(lon + math.degrees(radius * math.sin(bearing) / (6371008.8 * cos_lat)))
                owner.append(idx)

        lat_arr = np.asarray(latitudes)
        lon_arr = np.asarray(longitudes)
        x_float = ((lon_arr + 180.0) / 360.0) * n
        lat_clamped = np.radians(np.clip(lat_arr, -85.05112878, 85.05112878))
        y_float = (1.0 - np.arcsinh(np.tan(lat_clamped)) / math.pi) * 0.5 * n
        tile_x = np.floor(x_float).astype(np.int64) % n
        tile_y = np.floor(y_float).astype(np.int64)
        pixel_x_local = np.floor((x_float - np.floor(x_float)) * 256).astype(np.int64)
        pixel_y_local = np.floor((y_float - np.floor(y_float)) * 256).astype(np.int64)
        x_lookup = {value: idx for idx, value in enumerate(tile_xs)}
        mosaic_cols = np.full(len(tile_x), -1, dtype=np.int64)
        for idx, value in enumerate(tile_x):
            if int(value) in x_lookup:
                mosaic_cols[idx] = x_lookup[int(value)] * 256 + pixel_x_local[idx]
        mosaic_rows = (tile_y - min_tile_y) * 256 + pixel_y_local
        in_bounds = (
            (mosaic_cols >= 0)
            & (mosaic_rows >= 0)
            & (mosaic_rows < elevation.shape[0])
            & (mosaic_cols < elevation.shape[1])
            & (lat_arr >= -85.05112878)
            & (lat_arr <= 85.05112878)
        )
        sample_values: Dict[str, np.ndarray] = {}
        for name, raster in (("elevation", elevation), ("slope", slope), ("aspect", aspect), ("curvature", curvature)):
            samples = np.full(len(owner), np.nan, dtype=np.float64)
            valid_indexes = np.where(in_bounds)[0]
            samples[valid_indexes] = raster[mosaic_rows[valid_indexes], mosaic_cols[valid_indexes]]
            sample_values[name] = samples

        owners = np.asarray(owner, dtype=np.int64)
        result: Dict[str, TerrainCell] = {}
        for idx, cell_id in enumerate(cell_ids):
            indexes = np.where(owners == idx)[0]
            elev = sample_values["elevation"][indexes]
            valid = np.isfinite(elev)
            if not valid.any():
                continue
            slope_values = sample_values["slope"][indexes][valid]
            aspect_values = sample_values["aspect"][indexes][valid]
            curvature_values = sample_values["curvature"][indexes][valid]
            aspect_rad = np.radians(aspect_values)
            mean_aspect = math.degrees(math.atan2(float(np.sin(aspect_rad).mean()), float(np.cos(aspect_rad).mean()))) % 360.0
            cell_lat, cell_lon = center_points[idx]
            result[cell_id] = TerrainCell(
                cell_id=cell_id,
                latitude=float(cell_lat),
                longitude=float(cell_lon),
                elevation_m=float(np.mean(elev[valid])),
                slope_deg=float(np.mean(slope_values)),
                aspect_deg=mean_aspect,
                profile_curvature=float(np.mean(curvature_values)),
            )
        return result

    def _route_contributing_area(self, cells: Dict[str, TerrainCell]) -> None:
        if not cells:
            return
        areas = {cell_id: self._cell_area_m2(cell_id) for cell_id in cells}
        neighbors: Dict[str, List[Tuple[str, float]]] = {}
        disk = getattr(h3, "grid_disk", getattr(h3, "k_ring", None))
        for cell_id, cell in cells.items():
            lower: List[Tuple[str, float]] = []
            candidates = disk(cell_id, 1) if disk else []
            for candidate in candidates:
                if candidate == cell_id or candidate not in cells:
                    continue
                neighbor = cells[candidate]
                drop = cell.elevation_m - neighbor.elevation_m
                if drop <= 0:
                    continue
                distance = max(self._distance_m(cell, neighbor), 1.0)
                tangent = drop / distance
                lower.append((candidate, tangent**4))
            neighbors[cell_id] = lower
            cell.downslope_neighbors = [neighbor for neighbor, _ in lower]

        accumulated = dict(areas)
        # Transport each origin's area only along paths within a fixed physical
        # length. The process() halo contains every possible contributing origin.
        for origin in cells:
            pending = [(origin, 0., areas[origin])]
            while pending:
                current, travelled, mass = pending.pop()
                lower = neighbors[current]
                total_weight = sum(weight for _,weight in lower)
                for target, weight in lower:
                    distance = travelled + self._distance_m(cells[current],cells[target])
                    if distance > CATCHMENT_LENGTH_M:
                        continue
                    contribution = mass*weight/total_weight
                    accumulated[target] += contribution
                    pending.append((target,distance,contribution))

        for cell_id, cell in cells.items():
            area = accumulated[cell_id]
            cell.contributing_area_m2 = float(area)
            tangent_slope = max(math.tan(math.radians(cell.slope_deg)), 0.005)
            contour_width = math.sqrt(areas[cell_id])
            cell.twi = float(math.log(max(area/contour_width, 1e-12) / tangent_slope))

    @staticmethod
    def _latlng_to_cell(latitude: float, longitude: float, resolution: int) -> str:
        fn = getattr(h3, "latlng_to_cell", getattr(h3, "geo_to_h3", None))
        if fn is None:
            raise TerrainDataError("Installed h3-py version has no coordinate-to-cell API")
        return str(fn(latitude, longitude, resolution))

    def _cell_to_latlng(self, cell_id: str) -> Tuple[float, float]:
        fn = getattr(h3, "cell_to_latlng", getattr(h3, "h3_to_geo", None))
        if fn is None:
            raise TerrainDataError("Installed h3-py version has no cell-to-coordinate API")
        point = fn(cell_id)
        return float(point[0]), float(point[1])

    def _cell_area_m2(self, cell_id: str) -> float:
        fn = getattr(h3, "cell_area", None)
        if fn:
            try:
                return float(fn(cell_id, unit="m^2"))
            except TypeError:
                return float(fn(cell_id, unit="m2"))
        area_fn = getattr(h3, "hex_area", None)
        if area_fn:
            return float(area_fn(self.resolution, unit="m^2"))
        raise TerrainDataError("Installed h3-py version has no cell-area API")

    @staticmethod
    def _distance_m(a: TerrainCell, b: TerrainCell) -> float:
        fn = getattr(h3, "great_circle_distance", None)
        if fn:
            try:
                return float(fn((a.latitude, a.longitude), (b.latitude, b.longitude), unit="m"))
            except TypeError:
                return float(fn((a.latitude, a.longitude), (b.latitude, b.longitude), unit="m"))
        lat1, lat2 = math.radians(a.latitude), math.radians(b.latitude)
        dlat = lat2 - lat1
        dlon = math.radians(b.longitude - a.longitude)
        h = math.sin(dlat / 2.0) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2.0) ** 2
        return 2.0 * 6371008.8 * math.asin(math.sqrt(h))
