# Exact core source for external audit

Line-number prefixes are added; text after each prefix is verbatim source. Inclusion is not a claim that every equation is literature-verified. See docs/physics_sources.md for proxy and domain disclosures.

## backend/realtime/dem_processor.py

SHA-256: `0567d6120f146a6b0c465315b19b451139bcbe32ec7ae09be105a64d659a112c`

```python
0001 """On-demand Terrarium DEM mosaics and H3 resolution-9 terrain metrics."""
0002
0003 from __future__ import annotations
0004
0005 import asyncio
0006 import io
0007 import math
0008 from dataclasses import asdict, dataclass, field
0009 from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple
0010
0011 import h3
0012 import httpx
0013 import numpy as np
0014 from PIL import Image
0015
0016 from .cache import AsyncTTLCache, DEFAULT_CACHE
0017
0018 EARTH_CIRCUMFERENCE_M = 40075016.686
0019 TERRARIUM_URL = "https://s3.amazonaws.com/elevation-tiles-prod/terrarium/{z}/{x}/{y}.png"
0020
0021
0022 class TerrainDataError(RuntimeError):
0023     """Raised when real terrain observations are not sufficient to derive a surface."""
0024
0025
0026 @dataclass
0027 class TerrainCell:
0028     cell_id: str
0029     latitude: float
0030     longitude: float
0031     elevation_m: float
0032     slope_deg: float
0033     aspect_deg: float
0034     profile_curvature: float
0035     contributing_area_m2: float = 0.0
0036     twi: float = 2.0
0037     downslope_neighbors: List[str] = field(default_factory=list)
0038
0039
0040 @dataclass
0041 class TerrainScan:
0042     center_latitude: float
0043     center_longitude: float
0044     radius_km: float
0045     resolution: int
0046     zoom: int
0047     pixel_spacing_x_m: float
0048     pixel_spacing_y_m: float
0049     tile_count: int
0050     tile_ids: List[str]
0051     h3_cell_count: int
0052     min_elevation_m: float
0053     max_elevation_m: float
0054     slope_p95_deg: float
0055     max_twi: float
0056     interpolated_pixels: int
0057     cells: Dict[str, TerrainCell]
0058     warnings: List[str] = field(default_factory=list)
0059
0060     def to_dict(self) -> Dict[str, Any]:
0061         return asdict(self)
0062
0063
0064 class DEMProcessor:
0065     """Fetch and analyze Terrarium tiles for a global coordinate window."""
0066
0067     def __init__(
0068         self,
0069         zoom: int = 13,
0070         resolution: int = 9,
0071         cache: Optional[AsyncTTLCache] = None,
0072         tile_concurrency: int = 16,
0073         max_fill_iterations: int = 8,
0074     ) -> None:
0075         self.zoom = zoom
0076         self.resolution = resolution
0077         self.cache = cache or DEFAULT_CACHE
0078         self._tile_semaphore = asyncio.Semaphore(max(1, tile_concurrency))
0079         self.max_fill_iterations = max_fill_iterations
0080
0081     async def process(
0082         self, latitude: float, longitude: float, radius_km: float, client: httpx.AsyncClient
0083     ) -> TerrainScan:
0084         self._validate_coordinate(latitude, longitude)
0085         if radius_km <= 0 or radius_km > 100:
0086             raise ValueError("radius_km must be in (0, 100]")
0087         if abs(latitude) > 85.05112878:
0088             raise TerrainDataError(
0089                 "Terrarium XYZ tiles do not cover latitudes beyond 85.0511 degrees; polar terrain is unavailable"
0090             )
0091
0092         south, west, north, east = self._bbox(latitude, longitude, radius_km)
0093         cells = self._cover_bbox(south, west, north, east)
0094         tile_xs, min_y, max_y = self._tile_ranges(south, west, north, east)
0095         tile_coords = [(x, y) for y in range(min_y, max_y + 1) for x in tile_xs]
0096         tile_results = await asyncio.gather(
0097             *(self._fetch_tile(client, x, y) for x, y in tile_coords), return_exceptions=True
0098         )
0099
0100         mosaic = np.full((len(range(min_y, max_y + 1)) * 256, len(tile_xs) * 256), np.nan, dtype=np.float32)
0101         warnings: List[str] = []
0102         if latitude - radius_km / 111.32 < -85.05112878 or latitude + radius_km / 111.32 > 85.05112878:
0103             warnings.append("Viewport extends beyond the supported Web Mercator latitude coverage")
0104         tile_ids: List[str] = []
0105         successful = 0
0106         for (tile_x, tile_y), result in zip(tile_coords, tile_results):
0107             tile_ids.append("{}/{}/{}".format(self.zoom, tile_x, tile_y))
0108             if isinstance(result, BaseException):
0109                 warnings.append("Terrarium tile {}/{} unavailable: {}".format(tile_x, tile_y, result))
0110                 continue
0111             tile, tile_valid = result
0112             x_position = tile_xs.index(tile_x)
0113             row0 = (tile_y - min_y) * 256
0114             col0 = x_position * 256
0115             values = mosaic[row0 : row0 + 256, col0 : col0 + 256]
0116             values[tile_valid] = tile[tile_valid]
0117             successful += 1
0118
0119         if successful == 0:
0120             raise TerrainDataError("No Terrarium elevation tiles were available")
0121         mosaic, interpolated_pixels = self._interpolate_small_voids(mosaic)
0122         if not np.isfinite(mosaic).all():
0123             missing = int(np.size(mosaic) - np.isfinite(mosaic).sum())
0124             raise TerrainDataError(
0125                 "Terrain mosaic still has {} unsupported pixels after bounded neighbor interpolation".format(missing)
0126             )
0127
0128         delta_x = EARTH_CIRCUMFERENCE_M * math.cos(math.radians(latitude)) / (256.0 * (2**self.zoom))
0129         delta_y = EARTH_CIRCUMFERENCE_M / (256.0 * (2**self.zoom))
0130         q, p = np.gradient(mosaic, delta_y, delta_x)
0131         dp_dy, r = np.gradient(p, delta_y, delta_x)
0132         t, dq_dx = np.gradient(q, delta_y, delta_x)
0133         s = (dp_dy + dq_dx) * 0.5
0134
0135         gradient_squared = p * p + q * q
0136         slope_rad = np.clip(np.arctan(np.sqrt(gradient_squared)), 0.0175, 1.4835)
0137         slope_deg = np.degrees(slope_rad)
0138         aspect_deg = np.mod(270.0 - np.degrees(np.arctan2(q, p)), 360.0)
0139         curvature = -(
0140             p * p * r + 2.0 * p * q * s + q * q * t
0141         ) / ((gradient_squared + 1e-6) * np.power(1.0 + gradient_squared, 1.5))
0142         curvature = np.clip(curvature, -0.08, 0.08)
0143
0144         cell_map = self._sample_cells(
0145             cells, latitude, longitude, tile_xs, min_y, mosaic, slope_deg, aspect_deg, curvature
0146         )
0147         sampled_count = len(cell_map)
0148         cell_map = {
0149             cell_id: cell for cell_id, cell in cell_map.items()
0150             if cell.elevation_m > 0.5
0151         }
0152         removed_ocean = sampled_count - len(cell_map)
0153         if removed_ocean:
0154             warnings.append(
0155                 "Filtered {} H3 cells with mean elevation <= 0.5 m as ocean/bathymetry".format(removed_ocean)
0156             )
0157         if not cell_map:
0158             raise TerrainDataError("The requested bounding box contains no sampled land H3 cells above 0.5 m")
0159         self._route_contributing_area(cell_map)
0160         values = list(cell_map.values())
0161         if not values:
0162             raise TerrainDataError("The requested bounding box did not produce usable H3 terrain cells")
0163
0164         terrain = TerrainScan(
0165             center_latitude=latitude,
0166             center_longitude=longitude,
0167             radius_km=radius_km,
0168             resolution=self.resolution,
0169             zoom=self.zoom,
0170             pixel_spacing_x_m=delta_x,
0171             pixel_spacing_y_m=delta_y,
0172             tile_count=successful,
0173             tile_ids=tile_ids,
0174             h3_cell_count=len(cell_map),
0175             min_elevation_m=float(min(cell.elevation_m for cell in values)),
0176             max_elevation_m=float(max(cell.elevation_m for cell in values)),
0177             slope_p95_deg=float(np.percentile([cell.slope_deg for cell in values], 95)),
0178             max_twi=float(max(cell.twi for cell in values)),
0179             interpolated_pixels=interpolated_pixels,
0180             cells=cell_map,
0181             warnings=warnings,
0182         )
0183         if len(cell_map) < len(cells):
0184             warnings.append("{} H3 cover cells had no raster samples inside fetched tiles".format(len(cells) - len(cell_map)))
0185         return terrain
0186
0187     async def _fetch_tile(self, client: httpx.AsyncClient, tile_x: int, tile_y: int) -> Tuple[np.ndarray, np.ndarray]:
0188         key = ("terrarium", self.zoom, tile_x, tile_y)
0189
0190         async def load() -> Tuple[np.ndarray, np.ndarray]:
0191             url = TERRARIUM_URL.format(z=self.zoom, x=tile_x, y=tile_y)
0192             async with self._tile_semaphore:
0193                 response = await client.get(url)
0194             response.raise_for_status()
0195             with Image.open(io.BytesIO(response.content)) as image:
0196                 rgba = np.asarray(image.convert("RGBA"), dtype=np.uint8)
0197             if rgba.shape[:2] != (256, 256):
0198                 raise TerrainDataError("Unexpected Terrarium tile size {}".format(rgba.shape[:2]))
0199             rgb = rgba[:, :, :3].astype(np.float32)
0200             elevation = 256.0 * rgb[:, :, 0] + rgb[:, :, 1] + rgb[:, :, 2] / 256.0 - 32768.0
0201             valid = (rgba[:, :, 3] > 0) & np.isfinite(elevation) & (elevation >= -12000.0) & (elevation <= 10000.0)
0202             elevation[~valid] = np.nan
0203             return elevation, valid
0204
0205         return await self.cache.get_or_create(key, 7 * 24 * 3600, load)
0206
0207     @staticmethod
0208     def _validate_coordinate(latitude: float, longitude: float) -> None:
0209         if not (-90.0 <= latitude <= 90.0 and -180.0 <= longitude <= 180.0):
0210             raise ValueError("latitude/longitude are outside valid global bounds")
0211
0212     @staticmethod
0213     def _bbox(latitude: float, longitude: float, radius_km: float) -> Tuple[float, float, float, float]:
0214         lat_delta = radius_km / 111.32
0215         cos_lat = max(abs(math.cos(math.radians(latitude))), 0.01)
0216         lon_delta = min(180.0, radius_km / (111.32 * cos_lat))
0217         return (
0218             max(-85.05112878, latitude - lat_delta),
0219             longitude - lon_delta,
0220             min(85.05112878, latitude + lat_delta),
0221             longitude + lon_delta,
0222         )
0223
0224     def _cover_bbox(self, south: float, west: float, north: float, east: float) -> Set[str]:
0225         polygon_boxes: List[Tuple[float, float, float, float]]
0226         if west < -180.0:
0227             polygon_boxes = [(south, west + 360.0, north, 180.0), (south, -180.0, north, east)]
0228         elif east > 180.0:
0229             polygon_boxes = [(south, west, north, 180.0), (south, -180.0, north, east - 360.0)]
0230         else:
0231             polygon_boxes = [(south, west, north, east)]
0232
0233         cover: Set[str] = set()
0234         for s, w, n, e in polygon_boxes:
0235             points = [(s, w), (s, e), (n, e), (n, w), (s, w)]
0236             shape = None
0237             if hasattr(h3, "LatLngPoly"):
0238                 shape = h3.LatLngPoly(points)
0239             generated: Iterable[str] = []
0240             try:
0241                 if shape is not None and hasattr(h3, "h3shape_to_cells_experimental"):
0242                     generated = h3.h3shape_to_cells_experimental(shape, self.resolution, contain="overlap")
0243                 elif shape is not None and hasattr(h3, "polygon_to_cells_experimental"):
0244                     generated = h3.polygon_to_cells_experimental(shape, self.resolution, contain="overlap")
0245                 elif hasattr(h3, "geo_to_cells"):
0246                     generated = h3.geo_to_cells(
0247                         {"type": "Polygon", "coordinates": [[(w, s), (e, s), (e, n), (w, n), (w, s)]]},
0248                         self.resolution,
0249                     )
0250                 elif shape is not None and hasattr(h3, "polygon_to_cells"):
0251                     generated = h3.polygon_to_cells(shape, self.resolution)
0252                 elif hasattr(h3, "polyfill"):
0253                     generated = h3.polyfill(
0254                         {"type": "Polygon", "coordinates": [[(w, s), (e, s), (e, n), (w, n), (w, s)]]},
0255                         self.resolution,
0256                         geo_json_conformant=True,
0257                     )
0258             except (TypeError, ValueError, AttributeError):
0259                 # Older h3-py lacks overlap containment; center-fill plus one ring
0260                 # includes boundary-crossing cells while remaining globally valid.
0261                 generated = []
0262             generated = list(generated)
0263             cover.update(generated)
0264             if generated:
0265                 continue
0266             center_filler = getattr(h3, "geo_to_cells", None)
0267             try:
0268                 if center_filler:
0269                     centers = list(center_filler(
0270                         {"type": "Polygon", "coordinates": [[(w, s), (e, s), (e, n), (w, n), (w, s)]]},
0271                         self.resolution,
0272                     ))
0273                 elif shape is not None and hasattr(h3, "polygon_to_cells"):
0274                     centers = list(h3.polygon_to_cells(shape, self.resolution))
0275                 elif hasattr(h3, "polyfill"):
0276                     centers = list(h3.polyfill(
0277                         {"type": "Polygon", "coordinates": [[(w, s), (e, s), (e, n), (w, n), (w, s)]]},
0278                         self.resolution,
0279                         geo_json_conformant=True,
0280                     ))
0281                 else:
0282                     raise TerrainDataError("Installed h3-py version has no supported polygon fill API")
0283             except (TypeError, ValueError) as exc:
0284                 raise TerrainDataError("Unable to cover bounding box with H3 cells: {}".format(exc))
0285             cover.update(centers)
0286             disk = getattr(h3, "grid_disk", getattr(h3, "k_ring", None))
0287             if disk:
0288                 for cell in centers:
0289                     cover.update(disk(cell, 1))
0290
0291         if not cover:
0292             raise TerrainDataError("H3 returned an empty cover for the requested bounding box")
0293         return cover
0294
0295     def _tile_ranges(
0296         self, south: float, west: float, north: float, east: float
0297     ) -> Tuple[List[int], int, int]:
0298         n = 2**self.zoom
0299
0300         def tile_x(lon: float) -> int:
0301             return int(math.floor((lon + 180.0) / 360.0 * n)) % n
0302
0303         def tile_y(lat: float) -> int:
0304             lat = min(85.05112878, max(-85.05112878, lat))
0305             lat_rad = math.radians(lat)
0306             y = (1.0 - math.asinh(math.tan(lat_rad)) / math.pi) * 0.5 * n
0307             return max(0, min(n - 1, int(math.floor(y))))
0308
0309         x_start, x_end = tile_x(west), tile_x(east)
0310         if east - west >= 360.0:
0311             x_values = list(range(n))
0312         else:
0313             x_values = []
0314             x = x_start
0315             while True:
0316                 x_values.append(x)
0317                 if x == x_end or len(x_values) > n:
0318                     break
0319                 x = (x + 1) % n
0320         y_north, y_south = tile_y(north), tile_y(south)
0321         return x_values, min(y_north, y_south), max(y_north, y_south)
0322
0323     def _interpolate_small_voids(self, surface: np.ndarray) -> Tuple[np.ndarray, int]:
0324         missing = ~np.isfinite(surface)
0325         initial_count = int(missing.sum())
0326         if initial_count == 0:
0327             return surface, 0
0328         height, width = surface.shape
0329         for _ in range(self.max_fill_iterations):
0330             missing = ~np.isfinite(surface)
0331             if not missing.any():
0332                 break
0333             sums = np.zeros_like(surface, dtype=np.float32)
0334             weights = np.zeros_like(surface, dtype=np.float32)
0335             for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)):
0336                 source_rows = slice(max(0, -dy), min(height, height - dy))
0337                 source_cols = slice(max(0, -dx), min(width, width - dx))
0338                 target_rows = slice(max(0, dy), min(height, height + dy))
0339                 target_cols = slice(max(0, dx), min(width, width + dx))
0340                 source = surface[source_rows, source_cols]
0341                 valid = np.isfinite(source)
0342                 weight = 0.70710678 if dx and dy else 1.0
0343                 target_sum = sums[target_rows, target_cols]
0344                 target_weight = weights[target_rows, target_cols]
0345                 target_sum[valid] += source[valid] * weight
0346                 target_weight[valid] += weight
0347             can_fill = missing & (weights > 0)
0348             surface[can_fill] = sums[can_fill] / weights[can_fill]
0349         remaining = int((~np.isfinite(surface)).sum())
0350         if remaining:
0351             return surface, initial_count - remaining
0352         return surface, initial_count
0353
0354     def _sample_cells(
0355         self,
0356         cells: Set[str],
0357         center_lat: float,
0358         center_lon: float,
0359         tile_xs: Sequence[int],
0360         min_tile_y: int,
0361         elevation: np.ndarray,
0362         slope: np.ndarray,
0363         aspect: np.ndarray,
0364         curvature: np.ndarray,
0365     ) -> Dict[str, TerrainCell]:
0366         n = 2**self.zoom
0367         cell_ids = list(cells)
0368         center_points = [self._cell_to_latlng(cell) for cell in cell_ids]
0369         latitudes: List[float] = []
0370         longitudes: List[float] = []
0371         owner: List[int] = []
0372         # Center plus an interior ring samples the hex instead of one noisy pixel.
0373         bearings = [0.0] + [math.radians(angle) for angle in range(0, 360, 60)]
0374         radii_m = [0.0] + [90.0] * 6
0375         for idx, (lat, lon) in enumerate(center_points):
0376             latitudes.append(lat)
0377             longitudes.append(lon)
0378             owner.append(idx)
0379             for radius, bearing in zip(radii_m[1:], bearings[1:]):
0380                 latitudes.append(lat + math.degrees(radius * math.cos(bearing) / 6371008.8))
0381                 cos_lat = max(0.01, abs(math.cos(math.radians(lat))))
0382                 longitudes.append(lon + math.degrees(radius * math.sin(bearing) / (6371008.8 * cos_lat)))
0383                 owner.append(idx)
0384
0385         lat_arr = np.asarray(latitudes)
0386         lon_arr = np.asarray(longitudes)
0387         x_float = ((lon_arr + 180.0) / 360.0) * n
0388         lat_clamped = np.radians(np.clip(lat_arr, -85.05112878, 85.05112878))
0389         y_float = (1.0 - np.arcsinh(np.tan(lat_clamped)) / math.pi) * 0.5 * n
0390         tile_x = np.floor(x_float).astype(np.int64) % n
0391         tile_y = np.floor(y_float).astype(np.int64)
0392         pixel_x_local = np.floor((x_float - np.floor(x_float)) * 256).astype(np.int64)
0393         pixel_y_local = np.floor((y_float - np.floor(y_float)) * 256).astype(np.int64)
0394         x_lookup = {value: idx for idx, value in enumerate(tile_xs)}
0395         mosaic_cols = np.full(len(tile_x), -1, dtype=np.int64)
0396         for idx, value in enumerate(tile_x):
0397             if int(value) in x_lookup:
0398                 mosaic_cols[idx] = x_lookup[int(value)] * 256 + pixel_x_local[idx]
0399         mosaic_rows = (tile_y - min_tile_y) * 256 + pixel_y_local
0400         in_bounds = (
0401             (mosaic_cols >= 0)
0402             & (mosaic_rows >= 0)
0403             & (mosaic_rows < elevation.shape[0])
0404             & (mosaic_cols < elevation.shape[1])
0405             & (lat_arr >= -85.05112878)
0406             & (lat_arr <= 85.05112878)
0407         )
0408         sample_values: Dict[str, np.ndarray] = {}
0409         for name, raster in (("elevation", elevation), ("slope", slope), ("aspect", aspect), ("curvature", curvature)):
0410             samples = np.full(len(owner), np.nan, dtype=np.float64)
0411             valid_indexes = np.where(in_bounds)[0]
0412             samples[valid_indexes] = raster[mosaic_rows[valid_indexes], mosaic_cols[valid_indexes]]
0413             sample_values[name] = samples
0414
0415         owners = np.asarray(owner, dtype=np.int64)
0416         result: Dict[str, TerrainCell] = {}
0417         for idx, cell_id in enumerate(cell_ids):
0418             indexes = np.where(owners == idx)[0]
0419             elev = sample_values["elevation"][indexes]
0420             valid = np.isfinite(elev)
0421             if not valid.any():
0422                 continue
0423             slope_values = sample_values["slope"][indexes][valid]
0424             aspect_values = sample_values["aspect"][indexes][valid]
0425             curvature_values = sample_values["curvature"][indexes][valid]
0426             aspect_rad = np.radians(aspect_values)
0427             mean_aspect = math.degrees(math.atan2(float(np.sin(aspect_rad).mean()), float(np.cos(aspect_rad).mean()))) % 360.0
0428             cell_lat, cell_lon = center_points[idx]
0429             result[cell_id] = TerrainCell(
0430                 cell_id=cell_id,
0431                 latitude=float(cell_lat),
0432                 longitude=float(cell_lon),
0433                 elevation_m=float(np.mean(elev[valid])),
0434                 slope_deg=float(np.mean(slope_values)),
0435                 aspect_deg=mean_aspect,
0436                 profile_curvature=float(np.mean(curvature_values)),
0437             )
0438         return result
0439
0440     def _route_contributing_area(self, cells: Dict[str, TerrainCell]) -> None:
0441         if not cells:
0442             return
0443         areas = {cell_id: self._cell_area_m2(cell_id) for cell_id in cells}
0444         accumulated = dict(areas)
0445         neighbors: Dict[str, List[Tuple[str, float]]] = {}
0446         disk = getattr(h3, "grid_disk", getattr(h3, "k_ring", None))
0447         for cell_id, cell in cells.items():
0448             lower: List[Tuple[str, float]] = []
0449             candidates = disk(cell_id, 1) if disk else []
0450             for candidate in candidates:
0451                 if candidate == cell_id or candidate not in cells:
0452                     continue
0453                 neighbor = cells[candidate]
0454                 drop = cell.elevation_m - neighbor.elevation_m
0455                 if drop <= 0:
0456                     continue
0457                 distance = max(self._distance_m(cell, neighbor), 1.0)
0458                 tangent = drop / distance
0459                 lower.append((candidate, tangent**4))
0460             neighbors[cell_id] = lower
0461             cell.downslope_neighbors = [neighbor for neighbor, _ in lower]
0462
0463         for cell_id in sorted(cells, key=lambda item: cells[item].elevation_m, reverse=True):
0464             lower = neighbors[cell_id]
0465             total_weight = sum(weight for _, weight in lower)
0466             if total_weight <= 0:
0467                 continue
0468             for target, weight in lower:
0469                 accumulated[target] += accumulated[cell_id] * weight / total_weight
0470
0471         for cell_id, cell in cells.items():
0472             area = accumulated[cell_id]
0473             cell.contributing_area_m2 = float(area)
0474             tangent_slope = max(math.tan(math.radians(cell.slope_deg)), 0.005)
0475             cell.twi = float(np.clip(math.log(max(area, 1.0) / tangent_slope), 2.0, 22.0))
0476
0477     @staticmethod
0478     def _latlng_to_cell(latitude: float, longitude: float, resolution: int) -> str:
0479         fn = getattr(h3, "latlng_to_cell", getattr(h3, "geo_to_h3", None))
0480         if fn is None:
0481             raise TerrainDataError("Installed h3-py version has no coordinate-to-cell API")
0482         return str(fn(latitude, longitude, resolution))
0483
0484     def _cell_to_latlng(self, cell_id: str) -> Tuple[float, float]:
0485         fn = getattr(h3, "cell_to_latlng", getattr(h3, "h3_to_geo", None))
0486         if fn is None:
0487             raise TerrainDataError("Installed h3-py version has no cell-to-coordinate API")
0488         point = fn(cell_id)
0489         return float(point[0]), float(point[1])
0490
0491     def _cell_area_m2(self, cell_id: str) -> float:
0492         fn = getattr(h3, "cell_area", None)
0493         if fn:
0494             try:
0495                 return float(fn(cell_id, unit="m^2"))
0496             except TypeError:
0497                 return float(fn(cell_id, unit="m2"))
0498         area_fn = getattr(h3, "hex_area", None)
0499         if area_fn:
0500             return float(area_fn(self.resolution, unit="m^2"))
0501         raise TerrainDataError("Installed h3-py version has no cell-area API")
0502
0503     @staticmethod
0504     def _distance_m(a: TerrainCell, b: TerrainCell) -> float:
0505         fn = getattr(h3, "great_circle_distance", None)
0506         if fn:
0507             try:
0508                 return float(fn((a.latitude, a.longitude), (b.latitude, b.longitude), unit="m"))
0509             except TypeError:
0510                 return float(fn((a.latitude, a.longitude), (b.latitude, b.longitude), unit="m"))
0511         lat1, lat2 = math.radians(a.latitude), math.radians(b.latitude)
0512         dlat = lat2 - lat1
0513         dlon = math.radians(b.longitude - a.longitude)
0514         h = math.sin(dlat / 2.0) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2.0) ** 2
0515         return 2.0 * 6371008.8 * math.asin(math.sqrt(h))
```

## backend/realtime/global_ingestor.py

SHA-256: `bab41de39681bd014d29cca4725d77eb517d41e7318b3b90e5ef8458650a0b96`

```python
0001 """Parallel live-data ingestion for a global TERRA-SHIELD scan."""
0002
0003 from __future__ import annotations
0004
0005 import asyncio
0006 import math
0007 import time
0008 from dataclasses import asdict, dataclass, field
0009 from datetime import date, datetime, timedelta, timezone
0010 from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union
0011
0012 import httpx
0013 import numpy as np
0014
0015 from .cache import AsyncTTLCache, DEFAULT_CACHE
0016 from .dem_processor import DEMProcessor, TerrainScan
0017 from .natural_events import NaturalEvent, fetch_eonet_events
0018 from .satellite_processor import SatelliteCell, SatelliteProcessor
0019 from .soilgrids import estimate_pedotransfer, fetch_soilgrids_wcs
0020
0021 OPEN_METEO_FORECAST = "https://api.open-meteo.com/v1/forecast"
0022 OPEN_METEO_GEOCODING = "https://geocoding-api.open-meteo.com/v1/search"
0023 NASA_POWER = "https://power.larc.nasa.gov/api/temporal/daily/point"
0024 USGS_EVENTS = "https://earthquake.usgs.gov/fdsnws/event/1/query"
0025 OVERPASS = "https://overpass-api.de/api/interpreter"
0026 OVERPASS_FALLBACKS = (
0027     OVERPASS,
0028     "https://overpass.private.coffee/api/interpreter",
0029     "https://overpass.kumi.systems/api/interpreter",
0030     "https://lz4.overpass-api.de/api/interpreter",
0031 )
0032 SOILGRIDS = "https://rest.isric.org/soilgrids/v2.0/properties/query"
0033
0034 Location = Union[str, Tuple[float, float]]
0035
0036
0037 @dataclass
0038 class SourceStatus:
0039     available: bool
0040     provider: str
0041     fetched_at_utc: str
0042     error: Optional[str] = None
0043     mode: str = "UNASSESSED"
0044
0045
0046 @dataclass
0047 class LocationInfo:
0048     query: str
0049     latitude: float
0050     longitude: float
0051     display_name: Optional[str] = None
0052     country: Optional[str] = None
0053
0054
0055 @dataclass
0056 class WeatherSnapshot:
0057     observed_at_utc: Optional[str]
0058     rain_24h_mm: Optional[float]
0059     peak_rain_3h_mm: Optional[float]
0060     forecast_rain_next_24h_mm: Optional[float]
0061     forecast_rain_24_48h_mm: Optional[float]
0062     ari_7d_mm: Optional[float]
0063     ari_days_available: int
0064     soil_moisture_0_100cm_m3m3: Optional[float]
0065     soil_moisture_layers_m3m3: Dict[str, Optional[float]]
0066     snow_depth_m: Optional[float]
0067     air_temperature_c: Optional[float]
0068     soil_temperature_0_7cm_c: Optional[float]
0069     soil_temperature_28_100cm_c: Optional[float]
0070     freeze_thaw_cycles_7d: Optional[int]
0071
0072
0073 @dataclass
0074 class HydrologySnapshot:
0075     observed_date: Optional[str]
0076     nasa_gwetroot_pct: Optional[float]
0077     nasa_gwetprof_pct: Optional[float]
0078     nasa_precipitation_mm_day: Optional[float]
0079     open_meteo_soil_moisture_m3m3: Optional[float]
0080     soil_porosity_estimate: Optional[float]
0081     soil_saturation_pct: Optional[float]
0082     blended_relative_wetness_pct: Optional[float]
0083     source_mix: List[str]
0084     soil_properties: Dict[str, Any]
0085     soil_property_source: str = "unknown"
0086     soil_property_quality: str = "unknown"
0087
0088
0089 @dataclass
0090 class EarthquakeEvent:
0091     event_id: str
0092     magnitude: float
0093     magnitude_type: Optional[str]
0094     latitude: float
0095     longitude: float
0096     depth_km: float
0097     distance_km: float
0098     pga_g: float
0099     time_utc: Optional[str]
0100     place: Optional[str]
0101
0102
0103 @dataclass
0104 class SeismicSnapshot:
0105     event_count: int
0106     max_pga_g: Optional[float]
0107     events: List[EarthquakeEvent]
0108
0109
0110 @dataclass
0111 class InfrastructureSummary:
0112     total_features: int
0113     road_count: int
0114     road_cut_count: int
0115     waterway_count: int
0116     quarry_count: int
0117     building_count: int
0118     by_h3_cell: Dict[str, Dict[str, int]]
0119     query_bbox: Tuple[float, float, float, float]
0120
0121
0122 @dataclass
0123 class GlobalScan:
0124     location: LocationInfo
0125     radius_km: float
0126     generated_at_utc: str
0127     terrain: Optional[TerrainScan]
0128     weather: Optional[WeatherSnapshot]
0129     hydrology: Optional[HydrologySnapshot]
0130     seismic: Optional[SeismicSnapshot]
0131     infrastructure: Optional[InfrastructureSummary]
0132     eonet_events: List[Dict[str, Any]]
0133     satellite_cells: Dict[str, Dict[str, Any]]
0134     sources: Dict[str, SourceStatus]
0135     elapsed_seconds: float
0136     river: Optional[Dict[str, Any]] = None
0137     bedrock: Optional[Dict[str, Any]] = None
0138     coolr_nearby: Dict[str, Any] = field(default_factory=lambda: {"count":0,"quality":"not_queried"})
0139
0140     def to_dict(self) -> Dict[str, Any]:
0141         return asdict(self)
0142
0143
0144 class GlobalIngestor:
0145     """Query public global datasets concurrently, retaining source-level failures."""
0146
0147     def __init__(
0148         self,
0149         cache: Optional[AsyncTTLCache] = None,
0150         dem_processor: Optional[DEMProcessor] = None,
0151         provider_timeout_s: float = 1.25,
0152         user_agent: str = "TERRA-SHIELD/0.1 (research prototype; contact: terra-shield)",
0153     ) -> None:
0154         self.cache = cache or DEFAULT_CACHE
0155         self.dem_processor = dem_processor or DEMProcessor(cache=self.cache)
0156         self.satellite_processor = SatelliteProcessor(cache=self.cache)
0157         self.provider_timeout_s = max(0.5, provider_timeout_s)
0158         self.user_agent = user_agent
0159         self._client: Optional[httpx.AsyncClient] = None
0160
0161     async def __aenter__(self) -> "GlobalIngestor":
0162         self._client = httpx.AsyncClient(
0163             timeout=httpx.Timeout(self.provider_timeout_s),
0164             limits=httpx.Limits(max_connections=32, max_keepalive_connections=16),
0165             headers={"User-Agent": self.user_agent, "Accept": "application/json,image/png,*/*"},
0166         )
0167         return self
0168
0169     async def __aexit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
0170         if self._client is not None:
0171             await self._client.aclose()
0172             self._client = None
0173
0174     async def scan(
0175         self,
0176         location: Location,
0177         radius_km: float = 3.5,
0178         burn_override: Optional[Mapping[str, float]] = None,
0179     ) -> GlobalScan:
0180         started = time.perf_counter()
0181         if radius_km <= 0 or radius_km > 100:
0182             raise ValueError("radius_km must be in (0, 100]")
0183         if self._client is None:
0184             async with httpx.AsyncClient(
0185                 timeout=httpx.Timeout(self.provider_timeout_s),
0186                 limits=httpx.Limits(max_connections=32, max_keepalive_connections=16),
0187                 headers={"User-Agent": self.user_agent, "Accept": "application/json,image/png,*/*"},
0188             ) as client:
0189                 return await self._scan_with_client(location, radius_km, client, started, burn_override)
0190         return await self._scan_with_client(location, radius_km, self._client, started, burn_override)
0191
0192     async def _scan_with_client(
0193         self, location: Location, radius_km: float, client: httpx.AsyncClient, started: float,
0194         burn_override: Optional[Mapping[str, float]] = None,
0195     ) -> GlobalScan:
0196         location_info, geocode_status = await self._resolve_location(location, client)
0197         latitude, longitude = location_info.latitude, location_info.longitude
0198         south, west, north, east = self._bbox(latitude, longitude, radius_km)
0199
0200         from .ground_context import river_discharge, bedrock
0201         from .coolr import nearby_count
0202         source_names = ["terrarium_dem", "open_meteo", "nasa_power", "soilgrids", "usgs", "overpass", "eonet_wildfires", "eonet_landslides", "coolr", "river", "bedrock"]
0203         results = await asyncio.gather(
0204             self._capture("terrarium_dem", "AWS Terrarium", self.dem_processor.process(latitude, longitude, radius_km, client)),
0205             self._capture("open_meteo", "Open-Meteo", self._get_weather(latitude, longitude, client)),
0206             self._capture("nasa_power", "NASA POWER", self._get_power(latitude, longitude, client)),
0207             self._capture("soilgrids", "ISRIC SoilGrids", self._get_soilgrids(latitude, longitude, client)),
0208             self._capture("usgs", "USGS Earthquake Catalog", self._get_seismic(latitude, longitude, client)),
0209             self._capture("overpass", "OpenStreetMap Overpass", self._get_infrastructure(south, west, north, east, client)),
0210             self._capture("eonet_wildfires", "NASA EONET wildfires", fetch_eonet_events(client, "wildfires", max(10.0, self.provider_timeout_s), self.cache)),
0211             self._capture("eonet_landslides", "NASA EONET landslides", fetch_eonet_events(client, "landslides", max(6.0, self.provider_timeout_s), self.cache)),
0212             self._capture("coolr", "NASA COOLR", nearby_count(client,latitude,longitude,self.cache)),
0213             self._capture("river", "Open-Meteo GloFAS", river_discharge(client,latitude,longitude)),
0214             self._capture("bedrock", "Macrostrat", bedrock(client,latitude,longitude)),
0215             return_exceptions=False,
0216         )
0217         payloads: Dict[str, Any] = {}
0218         statuses: Dict[str, SourceStatus] = {"geocoding": geocode_status}
0219         for name, (payload, status) in zip(source_names, results):
0220             payloads[name] = payload
0221             statuses[name] = status
0222
0223         event_values = []
0224         for source in ("eonet_wildfires", "eonet_landslides"):
0225             event_values.extend(payloads.get(source) or [])
0226         event_map: Dict[str, NaturalEvent] = {}
0227         for event in event_values:
0228             event_map["{}:{}".format(event.event_id, event.date_utc)] = event
0229         satellite_cells: Dict[str, SatelliteCell] = {}
0230         if payloads.get("terrarium_dem"):
0231             satellite_cells, statuses['satellite'] = await self._capture(
0232                 'satellite', 'Esri World Imagery RGB z13', self.satellite_processor.process_cells(
0233                     {cell_id: (cell.latitude, cell.longitude) for cell_id, cell in payloads["terrarium_dem"].cells.items()},
0234                     event_map,
0235                     client,
0236                     burn_override=burn_override,
0237                 ))
0238             if not satellite_cells:
0239                 satellite_cells = {}
0240                 statuses['satellite'].available = False
0241                 statuses['satellite'].mode = 'FALLBACK'
0242                 statuses['satellite'].error = statuses['satellite'].error or 'No imagery cells were decoded'
0243         else:
0244             statuses["satellite"] = SourceStatus(False, "Esri World Imagery RGB z13", datetime.now(timezone.utc).isoformat(), "Terrain cells unavailable", "FALLBACK")
0245
0246         weather = payloads["open_meteo"]
0247         power = payloads["nasa_power"]
0248         soilgrids = payloads["soilgrids"]
0249         terrain = payloads["terrarium_dem"]
0250         hydrology = self._combine_hydrology(weather, power, soilgrids, terrain)
0251         coolr_count = payloads['coolr']
0252         if coolr_count is None:
0253             from .coolr import local_catalog_distances, URL
0254             local=local_catalog_distances(latitude,longitude)
0255             coolr_count={**local,'count':local['cached_count_50km'],'quality':'partial_local_catalog_fallback_50km','source_url':URL}
0256         if not coolr_count['quality'].startswith('NASA'):
0257             statuses['coolr'].available=False
0258             statuses['coolr'].mode='FALLBACK'
0259             statuses['coolr'].error=coolr_count.get('live_error_type','provider_deadline')
0260         return GlobalScan(
0261             location=location_info,
0262             radius_km=radius_km,
0263             generated_at_utc=datetime.now(timezone.utc).isoformat(),
0264             terrain=terrain,
0265             weather=weather,
0266             hydrology=hydrology,
0267             seismic=payloads["usgs"],
0268             infrastructure=payloads["overpass"],
0269             coolr_nearby=coolr_count, river=payloads["river"], bedrock=payloads["bedrock"],
0270             eonet_events=[event.to_dict() for event in event_values],
0271             satellite_cells={key: asdict(value) for key, value in satellite_cells.items()},
0272             sources=statuses,
0273             elapsed_seconds=time.perf_counter() - started,
0274         )
0275
0276     async def _capture(self, name: str, provider: str, awaitable: Any) -> Tuple[Any, SourceStatus]:
0277         from .cache import CACHE_TRACE
0278         trace=[];token=CACHE_TRACE.set(trace)
0279         fetched_at = datetime.now(timezone.utc).isoformat()
0280         try:
0281             deadline = 2.2
0282             payload = await asyncio.wait_for(awaitable, timeout=deadline)
0283             return payload, SourceStatus(True, provider, fetched_at, mode="CACHED" if trace and all(x=="CACHED" for x in trace) else "LIVE")
0284         except Exception as exc:
0285             return None, SourceStatus(False, provider, fetched_at, "{}: {}".format(type(exc).__name__, exc), "FALLBACK")
0286         finally:
0287             CACHE_TRACE.reset(token)
0288
0289     async def _resolve_location(self, location: Location, client: httpx.AsyncClient) -> Tuple[LocationInfo, SourceStatus]:
0290         fetched_at = datetime.now(timezone.utc).isoformat()
0291         if isinstance(location, str):
0292             query = location.strip()
0293             if not query:
0294                 raise ValueError("place name cannot be empty")
0295             key = ("geocode", query.casefold())
0296
0297             async def load() -> Dict[str, Any]:
0298                 response = await client.get(
0299                     OPEN_METEO_GEOCODING,
0300                     params={"name": query, "count": 1, "language": "en", "format": "json"},
0301                     timeout=self.provider_timeout_s,
0302                 )
0303                 response.raise_for_status()
0304                 return response.json()
0305
0306             data, status = await self._capture('geocoding', 'Open-Meteo Geocoding',
0307                 self.cache.get_or_create(key, 30 * 24 * 3600, load))
0308             if data is None:
0309                 raise ValueError('Geocoding unavailable: '+str(status.error))
0310             results = data.get("results") or []
0311             if not results:
0312                 raise ValueError("Open-Meteo geocoding found no result for {!r}".format(query))
0313             result = results[0]
0314             lat, lon = float(result["latitude"]), float(result["longitude"])
0315             self._validate_coordinate(lat, lon)
0316             info = LocationInfo(
0317                 query=query,
0318                 latitude=lat,
0319                 longitude=lon,
0320                 display_name=result.get("name"),
0321                 country=result.get("country"),
0322             )
0323             return info, status
0324         if isinstance(location, (tuple, list)) and len(location) == 2:
0325             lat, lon = float(location[0]), float(location[1])
0326             self._validate_coordinate(lat, lon)
0327             return LocationInfo("{:.6f}, {:.6f}".format(lat, lon), lat, lon), SourceStatus(
0328                 True, "Provided coordinates", fetched_at, mode="PROVIDED"
0329             )
0330         raise TypeError("location must be a place-name string or (latitude, longitude) pair")
0331
0332     async def _get_weather(self, latitude: float, longitude: float, client: httpx.AsyncClient) -> WeatherSnapshot:
0333         params = {
0334             "latitude": latitude,
0335             "longitude": longitude,
0336             "timezone": "UTC",
0337             "past_days": 7,
0338             "forecast_days": 3,
0339             "hourly": ",".join(
0340                 [
0341                     "precipitation",
0342                     "rain",
0343                     "snowfall",
0344                     "snow_depth",
0345                     "temperature_2m",
0346                     "soil_temperature_0_to_7cm",
0347                     "soil_temperature_28_to_100cm",
0348                     "soil_moisture_0_to_7cm",
0349                     "soil_moisture_7_to_28cm",
0350                     "soil_moisture_28_to_100cm",
0351                 ]
0352             ),
0353             "daily": "precipitation_sum",
0354         }
0355         data = await self._json_cached("open_meteo", params, 600, client, OPEN_METEO_FORECAST)
0356         return self._parse_weather(data)
0357
0358     def _parse_weather(self, data: Mapping[str, Any]) -> WeatherSnapshot:
0359         hourly = data.get("hourly") or {}
0360         times = list(hourly.get("time") or [])
0361         now = datetime.now(timezone.utc)
0362         datetimes = [self._parse_provider_time(value) for value in times]
0363         precipitation = self._float_values(hourly.get("precipitation"), len(times))
0364         rain = self._float_values(hourly.get("rain"), len(times))
0365         current = [idx for idx, dt in enumerate(datetimes) if dt is not None and dt <= now]
0366         recent_24 = [idx for idx in current if datetimes[idx] >= now - timedelta(hours=24)]
0367         rain_24 = self._sum_present(rain[idx] for idx in recent_24)
0368         peak3 = None
0369         windows: List[float] = []
0370         for start in range(max(0, len(recent_24) - 2)):
0371             indexes = recent_24[start : start + 3]
0372             if len(indexes) != 3 or any(rain[idx] is None for idx in indexes):
0373                 continue
0374             if any(datetimes[b] - datetimes[a] != timedelta(hours=1) for a, b in zip(indexes, indexes[1:])):
0375                 continue
0376             windows.append(sum(float(rain[idx]) for idx in indexes))
0377         if windows:
0378             peak3 = max(windows)
0379
0380         future = [idx for idx, dt in enumerate(datetimes) if dt is not None and dt > now]
0381         next24 = [idx for idx in future if datetimes[idx] <= now + timedelta(hours=24)]
0382         next48 = [idx for idx in future if now + timedelta(hours=24) < datetimes[idx] <= now + timedelta(hours=48)]
0383         forecast24 = self._sum_present(precipitation[idx] for idx in next24)
0384         forecast48 = self._sum_present(precipitation[idx] for idx in next48)
0385
0386         daily_precip: Dict[date, float] = {}
0387         for observed_time, amount in zip(datetimes, precipitation):
0388             if observed_time is None or amount is None or observed_time > now or observed_time < now - timedelta(days=7):
0389                 continue
0390             daily_precip[observed_time.date()] = daily_precip.get(observed_time.date(), 0.0) + amount
0391         ari_values = [daily_precip[day] for day in sorted(daily_precip, reverse=True)[:7]]
0392         ari = sum(amount * (0.85**index) for index, amount in enumerate(ari_values)) if ari_values else None
0393
0394         layers: Dict[str, Optional[float]] = {}
0395         for key in ("soil_moisture_0_to_7cm", "soil_moisture_7_to_28cm", "soil_moisture_28_to_100cm"):
0396             values = self._float_values(hourly.get(key), len(times))
0397             latest = next((values[idx] for idx in reversed(current) if values[idx] is not None), None)
0398             layers[key] = latest
0399         layer_weights = {
0400             "soil_moisture_0_to_7cm": 7.0,
0401             "soil_moisture_7_to_28cm": 21.0,
0402             "soil_moisture_28_to_100cm": 72.0,
0403         }
0404         moist_present = [(layers[key], weight) for key, weight in layer_weights.items() if layers[key] is not None]
0405         soil_moisture = (
0406             sum(float(value) * weight for value, weight in moist_present) / sum(weight for _, weight in moist_present)
0407             if moist_present
0408             else None
0409         )
0410
0411         def latest_hourly(key: str) -> Optional[float]:
0412             values = self._float_values(hourly.get(key), len(times))
0413             return next((values[idx] for idx in reversed(current) if values[idx] is not None), None)
0414
0415         air_temp = latest_hourly("temperature_2m")
0416         soil_temp_surface = latest_hourly("soil_temperature_0_to_7cm")
0417         soil_temp_deep = latest_hourly("soil_temperature_28_to_100cm")
0418         snow_depth = latest_hourly("snow_depth")
0419
0420         temp_values = self._float_values(hourly.get("soil_temperature_0_to_7cm"), len(times))
0421         if not any(value is not None for value in temp_values):
0422             temp_values = self._float_values(hourly.get("temperature_2m"), len(times))
0423         last_week = [idx for idx in current if datetimes[idx] >= now - timedelta(days=7)]
0424         cycles = 0
0425         prior_temp: Optional[float] = None
0426         for idx in last_week:
0427             value = temp_values[idx]
0428             if value is None:
0429                 continue
0430             if prior_temp is not None and ((prior_temp < 0 <= value) or (prior_temp > 0 >= value)):
0431                 cycles += 1
0432             prior_temp = value
0433
0434         observed_idx = current[-1] if current else None
0435         observed_at = datetimes[observed_idx].isoformat() if observed_idx is not None else None
0436         return WeatherSnapshot(
0437             observed_at_utc=observed_at,
0438             rain_24h_mm=rain_24,
0439             peak_rain_3h_mm=peak3,
0440             forecast_rain_next_24h_mm=forecast24,
0441             forecast_rain_24_48h_mm=forecast48,
0442             ari_7d_mm=ari,
0443             ari_days_available=len(ari_values),
0444             soil_moisture_0_100cm_m3m3=soil_moisture,
0445             soil_moisture_layers_m3m3=layers,
0446             snow_depth_m=snow_depth,
0447             air_temperature_c=air_temp,
0448             soil_temperature_0_7cm_c=soil_temp_surface,
0449             soil_temperature_28_100cm_c=soil_temp_deep,
0450             freeze_thaw_cycles_7d=cycles if last_week else None,
0451         )
0452
0453     async def _get_power(self, latitude: float, longitude: float, client: httpx.AsyncClient) -> Dict[str, Any]:
0454         today = datetime.now(timezone.utc).date()
0455         params = {
0456             "parameters": "PRECTOTCORR,GWETROOT,GWETPROF",
0457             "community": "RE",
0458             "longitude": round(longitude, 4),
0459             "latitude": round(latitude, 4),
0460             "start": (today - timedelta(days=10)).strftime("%Y%m%d"),
0461             "end": today.strftime("%Y%m%d"),
0462             "format": "JSON",
0463         }
0464         return await self._json_cached("nasa_power", params, 24 * 3600, client, NASA_POWER)
0465
0466     async def _get_soilgrids(self, latitude: float, longitude: float, client: httpx.AsyncClient) -> Dict[str, Any]:
0467         params = [
0468             ("lon", round(longitude, 5)),
0469             ("lat", round(latitude, 5)),
0470             ("property", "bdod"),
0471             ("property", "clay"),
0472             ("property", "sand"),
0473             ("property", "soc"),
0474             ("depth", "0-5cm"),
0475             ("depth", "5-15cm"),
0476             ("depth", "15-30cm"),
0477             ("value", "mean"),
0478         ]
0479         key_params = tuple(params)
0480
0481         async def load() -> Dict[str, Any]:
0482             try:
0483                 response = await client.get(SOILGRIDS, params=params, timeout=min(1.0, self.provider_timeout_s))
0484                 response.raise_for_status()
0485                 return response.json()
0486             except Exception as rest_error:
0487                 try:
0488                     wcs = await fetch_soilgrids_wcs(client, latitude, longitude, timeout_s=min(2.0, self.provider_timeout_s))
0489                     return {**wcs, "properties": wcs}
0490                 except Exception as wcs_error:
0491                     raise RuntimeError("SoilGrids REST failed ({}); WCS failed ({})".format(rest_error, wcs_error))
0492
0493         return await self.cache.get_or_create(("soilgrids", key_params), 30 * 24 * 3600, load)
0494
0495     async def _get_seismic(self, latitude: float, longitude: float, client: httpx.AsyncClient) -> SeismicSnapshot:
0496         params = {
0497             "format": "geojson",
0498             "latitude": latitude,
0499             "longitude": longitude,
0500             "maxradiuskm": 250,
0501             "minmagnitude": 2.5,
0502             "orderby": "time",
0503         }
0504         data = await self._json_cached("usgs", params, 30, client, USGS_EVENTS)
0505         events: List[EarthquakeEvent] = []
0506         for feature in data.get("features") or []:
0507             props = feature.get("properties") or {}
0508             geometry = feature.get("geometry") or {}
0509             coords = geometry.get("coordinates") or []
0510             if len(coords) < 3 or props.get("mag") is None:
0511                 continue
0512             lon, lat, depth_km = float(coords[0]), float(coords[1]), float(coords[2])
0513             epicentral = self._haversine_km(latitude, longitude, lat, lon)
0514             hypocentral = math.sqrt(epicentral * epicentral + depth_km * depth_km)
0515             magnitude = float(props["mag"])
0516             distance_for_log = max(hypocentral, 1.0)
0517             pga = 10.0 ** (
0518                 -1.02 + 0.249 * magnitude - math.log10(distance_for_log) - 0.00255 * distance_for_log
0519             )
0520             event_id = str(feature.get("id", ""))
0521             event_time = props.get("time")
0522             event_time_iso = datetime.fromtimestamp(event_time / 1000.0, timezone.utc).isoformat() if event_time else None
0523             events.append(
0524                 EarthquakeEvent(
0525                     event_id=event_id,
0526                     magnitude=magnitude,
0527                     magnitude_type=props.get("magType"),
0528                     latitude=lat,
0529                     longitude=lon,
0530                     depth_km=depth_km,
0531                     distance_km=hypocentral,
0532                     pga_g=pga,
0533                     time_utc=event_time_iso,
0534                     place=props.get("place"),
0535                 )
0536             )
0537         return SeismicSnapshot(
0538             event_count=len(events),
0539             max_pga_g=max((event.pga_g for event in events), default=0.0),
0540             events=events,
0541         )
0542
0543     async def _get_infrastructure(
0544         self,
0545         south: float,
0546         west: float,
0547         north: float,
0548         east: float,
0549         client: httpx.AsyncClient,
0550     ) -> InfrastructureSummary:
0551         if west < -180.0:
0552             boxes = [(south, west + 360.0, north, 180.0), (south, -180.0, north, east)]
0553         elif east > 180.0:
0554             boxes = [(south, west, north, 180.0), (south, -180.0, north, east - 360.0)]
0555         else:
0556             boxes = [(south, west, north, east)]
0557
0558         async def query_box(box: Tuple[float, float, float, float]) -> Dict[str, Any]:
0559             bbox = "{:.5f},{:.5f},{:.5f},{:.5f}".format(*box)
0560             query = """
0561 [timeout:8][out:json];
0562 (
0563   way[\"highway\"~\"^(trunk|primary|secondary|tertiary)$\"]({bbox});
0564   way[\"waterway\"=\"river\"]({bbox});
0565 );
0566 out geom;
0567 """.format(bbox=bbox)
0568             key = ("overpass",) + tuple(round(value, 3) for value in box)
0569
0570             async def load() -> Dict[str, Any]:
0571                 last_error: Optional[Exception] = None
0572                 endpoint_timeout = min(self.provider_timeout_s, 2.0)
0573                 for endpoint in OVERPASS_FALLBACKS:
0574                     try:
0575                         response = await client.post(
0576                             endpoint,
0577                             data={"data": query},
0578                             headers={"Content-Type": "application/x-www-form-urlencoded"},
0579                             timeout=endpoint_timeout,
0580                         )
0581                         response.raise_for_status()
0582                         return response.json()
0583                     except (httpx.ConnectError, httpx.TimeoutException) as exc:
0584                         last_error = exc
0585                     except httpx.HTTPStatusError as exc:
0586                         if exc.response.status_code == 429 or exc.response.status_code >= 500:
0587                             last_error = exc
0588                         else:
0589                             raise
0590                 if last_error is not None:
0591                     raise last_error
0592                 raise RuntimeError("No Overpass endpoints were configured")
0593
0594             return await self.cache.get_or_create(key, 6 * 3600, load)
0595
0596         responses = await asyncio.gather(*(query_box(box) for box in boxes))
0597         unique_elements: Dict[Tuple[str, int], Dict[str, Any]] = {}
0598         for payload in responses:
0599             for element in payload.get("elements") or []:
0600                 unique_elements[(str(element.get("type", "")), int(element.get("id", -1)))] = element
0601         data = {"elements": list(unique_elements.values())}
0602         by_cell: Dict[str, Dict[str, int]] = {}
0603         totals = {"roads": 0, "road_cuts": 0, "waterways": 0, "quarries": 0, "buildings": 0}
0604         for element in data.get("elements") or []:
0605             tags = element.get("tags") or {}
0606             center = element.get("center") or {}
0607             lat = element.get("lat", center.get("lat"))
0608             lon = element.get("lon", center.get("lon"))
0609             if (lat is None or lon is None) and element.get("geometry"):
0610                 coordinates = element.get("geometry") or []
0611                 if coordinates:
0612                     lat = sum(float(point["lat"]) for point in coordinates) / len(coordinates)
0613                     lon = sum(float(point["lon"]) for point in coordinates) / len(coordinates)
0614             if lat is None or lon is None:
0615                 continue
0616             cell_id = self._coordinate_to_h3(float(lat), float(lon))
0617             cell_counts = by_cell.setdefault(
0618                 cell_id, {"roads": 0, "road_cuts": 0, "waterways": 0, "quarries": 0, "buildings": 0}
0619             )
0620             categories = []
0621             if tags.get("highway"):
0622                 categories.append("roads")
0623             if tags.get("cutting", "no") not in ("no", "false", "0", ""):
0624                 categories.append("road_cuts")
0625             if tags.get("waterway"):
0626                 categories.append("waterways")
0627             if tags.get("landuse") == "quarry" or tags.get("quarry"):
0628                 categories.append("quarries")
0629             if tags.get("building"):
0630                 categories.append("buildings")
0631             for category in set(categories):
0632                 totals[category] += 1
0633                 cell_counts[category] += 1
0634         return InfrastructureSummary(
0635             total_features=len(data.get("elements") or []),
0636             road_count=totals["roads"],
0637             road_cut_count=totals["road_cuts"],
0638             waterway_count=totals["waterways"],
0639             quarry_count=totals["quarries"],
0640             building_count=totals["buildings"],
0641             by_h3_cell=by_cell,
0642             query_bbox=(south, west, north, east),
0643         )
0644
0645     async def _json_cached(
0646         self,
0647         provider: str,
0648         params: Mapping[str, Any],
0649         ttl_seconds: int,
0650         client: httpx.AsyncClient,
0651         url: str,
0652     ) -> Dict[str, Any]:
0653         key_params = tuple(sorted((key, str(value)) for key, value in params.items()))
0654         key = (provider, key_params)
0655
0656         async def load() -> Dict[str, Any]:
0657             response = await client.get(url, params=params, timeout=self.provider_timeout_s)
0658             response.raise_for_status()
0659             return response.json()
0660
0661         return await self.cache.get_or_create(key, ttl_seconds, load)
0662
0663     def _combine_hydrology(
0664         self,
0665         weather: Optional[WeatherSnapshot],
0666         power: Optional[Mapping[str, Any]],
0667         soilgrids: Optional[Mapping[str, Any]],
0668         terrain: Optional[TerrainScan] = None,
0669     ) -> Optional[HydrologySnapshot]:
0670         if weather is None and power is None and soilgrids is None and terrain is None:
0671             return None
0672         params = ((power or {}).get("properties") or {}).get("parameter") or {}
0673         gwetroot, root_date = self._latest_parameter(params.get("GWETROOT") or {})
0674         gwetprof, _ = self._latest_parameter(params.get("GWETPROF") or {})
0675         precip, _ = self._latest_parameter(params.get("PRECTOTCORR") or {})
0676         moisture = weather.soil_moisture_0_100cm_m3m3 if weather else None
0677         props = self._soil_property_values(soilgrids)
0678         if any(props.get(key) is None for key in ("bdod_kg_m3", "clay_g_kg", "sand_g_kg", "soc_g_kg")):
0679             measured_keys = [key for key in ("bdod_kg_m3", "clay_g_kg", "sand_g_kg", "soc_g_kg") if props.get(key) is not None]
0680             if terrain and terrain.cells:
0681                 cells = list(terrain.cells.values())
0682                 slope = float(np.mean([cell.slope_deg for cell in cells]))
0683                 curvature = float(np.mean([cell.profile_curvature for cell in cells]))
0684                 twi = float(np.mean([cell.twi for cell in cells]))
0685             else:
0686                 slope, curvature, twi = 15.0, 0.0, 6.0
0687             moisture_regime = gwetroot if gwetroot is not None else (moisture * 100.0 if moisture is not None else None)
0688             estimate = estimate_pedotransfer(slope, curvature, twi, moisture_regime)
0689             for key in ("bdod_kg_m3", "clay_g_kg", "sand_g_kg", "soc_g_kg"):
0690                 if props.get(key) is None:
0691                     props[key] = estimate[key]
0692             for key in ("sand_fraction", "clay_fraction", "silt_fraction", "organic_matter_fraction", "bulk_density_g_cm3"):
0693                 props.setdefault(key, estimate[key])
0694             props["property_sources"] = {
0695                 key: ("ISRIC SoilGrids" if key in measured_keys else "topographic_climatic_pedotransfer_estimate")
0696                 for key in ("bdod_kg_m3", "clay_g_kg", "sand_g_kg", "soc_g_kg")
0697             }
0698             props["source"] = "hybrid_soilgrids_and_pedotransfer" if measured_keys else "topographic_climatic_pedotransfer_estimate"
0699             props["quality"] = "mixed_measured_prediction_and_estimate" if measured_keys else "estimated_not_measured"
0700         else:
0701             props["sand_fraction"] = float(props["sand_g_kg"]) / 1000.0
0702             props["clay_fraction"] = float(props["clay_g_kg"]) / 1000.0
0703             props["silt_fraction"] = max(0.0, 1.0 - props["sand_fraction"] - props["clay_fraction"])
0704             props["organic_matter_fraction"] = float(props["soc_g_kg"]) * 1.724 / 1000.0
0705             props["bulk_density_g_cm3"] = float(props["bdod_kg_m3"]) / 1000.0
0706             props["source"] = (soilgrids or {}).get("source", "ISRIC SoilGrids")
0707             props["quality"] = (soilgrids or {}).get("quality", "soilgrids_prediction")
0708         bulk_density = props.get("bdod_kg_m3")
0709         porosity = 1.0 - bulk_density / 2650.0 if bulk_density is not None else None
0710         saturation = None
0711         if moisture is not None and porosity is not None and porosity > 0:
0712             saturation = float(np.clip(moisture / porosity * 100.0, 0.0, 100.0))
0713         sources: List[str] = []
0714         wetness_values: List[float] = []
0715         if saturation is not None:
0716             wetness_values.append(saturation)
0717             sources.append("Open-Meteo volumetric moisture / SoilGrids bulk-density porosity")
0718         elif moisture is not None:
0719             # This is volumetric moisture expressed as percent, not true pore saturation.
0720             wetness_values.append(float(np.clip(moisture * 100.0, 0.0, 100.0)))
0721             sources.append("Open-Meteo volumetric moisture (relative proxy; porosity unavailable)")
0722         if gwetroot is not None:
0723             wetness_values.append(float(np.clip(gwetroot, 0.0, 100.0)))
0724             sources.append("NASA POWER GWETROOT")
0725         blended = sum(wetness_values) / len(wetness_values) if wetness_values else None
0726         return HydrologySnapshot(
0727             observed_date=root_date,
0728             nasa_gwetroot_pct=gwetroot,
0729             nasa_gwetprof_pct=gwetprof,
0730             nasa_precipitation_mm_day=precip,
0731             open_meteo_soil_moisture_m3m3=moisture,
0732             soil_porosity_estimate=porosity,
0733             soil_saturation_pct=saturation,
0734             blended_relative_wetness_pct=blended,
0735             source_mix=sources,
0736             soil_properties=props,
0737             soil_property_source=str(props.get("source", "unknown")),
0738             soil_property_quality=str(props.get("quality", "unknown")),
0739         )
0740
0741     @staticmethod
0742     def _soil_property_values(data: Optional[Mapping[str, Any]]) -> Dict[str, Optional[float]]:
0743         values: Dict[str, Optional[float]] = {
0744             "bdod_kg_m3": None,
0745             "clay_g_kg": None,
0746             "sand_g_kg": None,
0747             "soc_g_kg": None,
0748         }
0749         if not data:
0750             return values
0751         if all(data.get(key) is not None for key in values):
0752             return {key: float(data[key]) for key in values}
0753         if all(data.get(key) is not None for key in values):
0754             return {key: float(data[key]) for key in values}
0755         raw_properties = data.get("properties") or []
0756         if isinstance(raw_properties, Mapping):
0757             raw_properties = raw_properties.get("layers", raw_properties)
0758         if isinstance(raw_properties, Mapping):
0759             properties = []
0760             for property_name, property_value in raw_properties.items():
0761                 if isinstance(property_value, Mapping):
0762                     normalized = dict(property_value)
0763                     normalized.setdefault("name", property_name)
0764                     properties.append(normalized)
0765                 elif isinstance(property_value, list):
0766                     properties.extend(
0767                         dict(layer, name=layer.get("name", property_name))
0768                         for layer in property_value
0769                         if isinstance(layer, Mapping)
0770                     )
0771         else:
0772             properties = [prop for prop in raw_properties if isinstance(prop, Mapping)]
0773         for prop in properties:
0774             name = prop.get("name")
0775             key = {
0776                 "bdod": "bdod_kg_m3",
0777                 "clay": "clay_g_kg",
0778                 "sand": "sand_g_kg",
0779                 "soc": "soc_g_kg",
0780             }.get(name)
0781             if not key:
0782                 continue
0783             depth_values = []
0784             raw_depths = prop.get("depths") or []
0785             if isinstance(raw_depths, Mapping):
0786                 depths = []
0787                 for depth_label, depth_value in raw_depths.items():
0788                     if isinstance(depth_value, Mapping):
0789                         normalized_depth = dict(depth_value)
0790                         normalized_depth.setdefault("label", depth_label)
0791                         depths.append(normalized_depth)
0792             else:
0793                 depths = [depth for depth in raw_depths if isinstance(depth, Mapping)]
0794             for depth in depths:
0795                 mean = ((depth.get("values") or {}).get("mean"))
0796                 if mean is not None and float(mean) >= 0:
0797                     thickness = float(str(depth.get("range", {}).get("bottom_depth", "0")).replace("cm", ""))
0798                     top = float(str(depth.get("range", {}).get("top_depth", "0")).replace("cm", ""))
0799                     weight = max(1.0, thickness - top)
0800                     depth_values.append((float(mean), weight))
0801             if depth_values:
0802                 mean_value = sum(value * weight for value, weight in depth_values) / sum(weight for _, weight in depth_values)
0803                 # SoilGrids bdod uses cg/cm3; 1 cg/cm3 equals 10 kg/m3.
0804                 values[key] = mean_value * 10.0 if key == "bdod_kg_m3" else mean_value
0805         return values
0806
0807     @staticmethod
0808     def _latest_parameter(parameter: Mapping[str, Any]) -> Tuple[Optional[float], Optional[str]]:
0809         candidates = []
0810         for day, value in parameter.items():
0811             try:
0812                 number = float(value)
0813             except (TypeError, ValueError):
0814                 continue
0815             if math.isfinite(number) and number > -900:
0816                 candidates.append((str(day), number))
0817         if not candidates:
0818             return None, None
0819         day, value = max(candidates, key=lambda item: item[0])
0820         return value, day
0821
0822     @staticmethod
0823     def _float_values(values: Any, expected_length: int) -> List[Optional[float]]:
0824         result: List[Optional[float]] = []
0825         values = list(values or [])
0826         for raw in values[:expected_length]:
0827             try:
0828                 value = float(raw)
0829                 result.append(value if math.isfinite(value) else None)
0830             except (TypeError, ValueError):
0831                 result.append(None)
0832         if len(result) < expected_length:
0833             result.extend([None] * (expected_length - len(result)))
0834         return result
0835
0836     @staticmethod
0837     def _sum_present(values: Sequence[Optional[float]]) -> Optional[float]:
0838         usable = [float(value) for value in values if value is not None]
0839         return sum(usable) if usable else None
0840
0841     @staticmethod
0842     def _parse_provider_time(value: Any) -> Optional[datetime]:
0843         if not value:
0844             return None
0845         try:
0846             parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
0847             if parsed.tzinfo is None:
0848                 parsed = parsed.replace(tzinfo=timezone.utc)
0849             return parsed.astimezone(timezone.utc)
0850         except ValueError:
0851             return None
0852
0853     @staticmethod
0854     def _bbox(latitude: float, longitude: float, radius_km: float) -> Tuple[float, float, float, float]:
0855         lat_delta = radius_km / 111.32
0856         lon_delta = min(180.0, radius_km / (111.32 * max(abs(math.cos(math.radians(latitude))), 0.01)))
0857         return (
0858             max(-85.05112878, latitude - lat_delta),
0859             longitude - lon_delta,
0860             min(85.05112878, latitude + lat_delta),
0861             longitude + lon_delta,
0862         )
0863
0864     @staticmethod
0865     def _validate_coordinate(latitude: float, longitude: float) -> None:
0866         if not (-90.0 <= latitude <= 90.0 and -180.0 <= longitude <= 180.0):
0867             raise ValueError("latitude/longitude are outside valid global bounds")
0868
0869     @staticmethod
0870     def _coordinate_to_h3(latitude: float, longitude: float) -> str:
0871         fn = getattr(__import__("h3"), "latlng_to_cell", getattr(__import__("h3"), "geo_to_h3", None))
0872         if fn is None:
0873             raise RuntimeError("Installed h3-py version has no coordinate-to-cell API")
0874         return str(fn(latitude, longitude, 9))
0875
0876     @staticmethod
0877     def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
0878         earth_radius_km = 6371.0088
0879         phi1, phi2 = math.radians(lat1), math.radians(lat2)
0880         dphi, dlambda = phi2 - phi1, math.radians(lon2 - lon1)
0881         hav = math.sin(dphi / 2.0) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2.0) ** 2
0882         return 2.0 * earth_radius_km * math.asin(math.sqrt(hav))
```

## backend/realtime/ground_context.py

SHA-256: `86b2851ab3a803dc3b466d9655b7fb316acc5830043922a027d6197940c3e2ee`

```python
0001 """Mapped geology and modeled river flow; neither is a local ground measurement."""
0002 from datetime import datetime,timezone
0003 import math
0004
0005 async def river_discharge(client,lat,lon):
0006     url='https://flood-api.open-meteo.com/v1/flood'
0007     response=await client.get(url,params=dict(latitude=lat,longitude=lon,daily='river_discharge,river_discharge_max',past_days=7,forecast_days=3,timezone='UTC'))
0008     response.raise_for_status();data=response.json();daily=data['daily']
0009     today=datetime.now(timezone.utc).date().isoformat();dates=daily['time'];i=dates.index(today)
0010     q=daily['river_discharge'][i];past=[v for v in daily['river_discharge'][max(0,i-7):i] if v is not None and math.isfinite(v)]
0011     if q is None or not math.isfinite(q) or q<0 or len(past)!=7 or min(past)<0:raise ValueError('Missing discharge or complete seven-day history')
0012     return dict(discharge_m3_s=q,surge_ratio=min(5.,max(1.,q/max(min(past),1.))),date=today,source_url=url,quality='GloFAS modeled 5 km discharge; not gauged local flow')
0013
0014 async def bedrock(client,lat,lon):
0015     url='https://macrostrat.org/api/v2/geologic_units/map'
0016     response=await client.get(url,params=dict(lat=lat,lng=lon,response='long'));response.raise_for_status()
0017     records=response.json().get('success',{}).get('data',[])
0018     if not records:raise ValueError('No mapped geology returned')
0019     unit=records[0];lith=unit.get('lith') or unit.get('lithology') or ''
0020     text=str(lith).lower();weak=any(k in text for k in ('shale','mudstone','phyllite','tuff'));strong=any(k in text for k in ('granite','basalt'))
0021     adjustment=-1.5 if weak and not strong else 1.5 if strong and not weak else 0.
0022     return dict(name=unit.get('name'),lithology=lith,age=unit.get('age') or {'early':unit.get('t_age'),'late':unit.get('b_age')},source_url=url,friction_adjustment_deg=adjustment,cohesion_adjustment_kpa=adjustment,quality='Mapped unit; requested strength adjustment is an uncalibrated scenario proxy',raw_unit=unit)
```

## backend/api/scan_service.py

SHA-256: `11839d7d1eed87145ff9770c6509fbaac09b6f6d3a1593f3fbfb3b8edac579af`

```python
0001 """Live location scans, cell physics, hotspot summaries, and GeoJSON output."""
0002
0003 from __future__ import annotations
0004
0005 import json
0006 import math
0007 import os
0008 import threading
0009 import unicodedata
0010 from dataclasses import replace
0011 from pathlib import Path
0012 from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union
0013
0014 import h3
0015 import numpy as np
0016
0017 from backend.api.reliability import reliability_report
0018 from backend.physics.alpine_weather import apply_alpine_weather, ALPINE_ASSUMPTION
0019 from backend.ml.global_xgb import GlobalMonotonicXGB, blend_failure_probability
0020 from backend.physics.causal_attribution import CAUSES, explain_causes, select_numbers_that_matter
0021 from backend.physics.kinematics import route_runout, solve_kinematics
0022 from backend.physics.multi_causal_engine import MultiCausalEngine, PhysicsInputs
0023 from backend.realtime.global_ingestor import GlobalIngestor, GlobalScan
0024 from scripts.verify_5_global_causes import _feature_matrix, evaluate_live_physics
0025
0026 ROOT = Path(__file__).resolve().parents[2]
0027 RUNOUT_MU_B = 0.10
0028 RUNOUT_XI = 500.0
0029 RUNOUT_PARAMETER_QUALITY = "un_calibrated_display_default"
0030 ALERT_THRESHOLDS = {"warning": 0.15, "watch": 0.35, "alert": 0.65}
0031
0032 # These coordinates were supplied as project scan locations, not inferred labels.
0033 KNOWN_LOCATION_COORDINATES = {
0034     "chooralmala": (11.548, 76.134),
0035     "wayanad": (11.548, 76.134),
0036     "pettimudi": (10.174, 77.012),
0037     "kaikoura": (-42.260, 173.620),
0038     "zermatt": (46.020, 7.749),
0039     "malibu": (34.07022, -118.54453),
0040     "new delhi": (28.6139, 77.2090),
0041 }
0042
0043
0044 def known_location_coordinates(query: str) -> Optional[Tuple[float, float]]:
0045     normalized = unicodedata.normalize("NFKD", str(query)).encode("ascii", "ignore").decode("ascii").lower()
0046     for name, coordinates in KNOWN_LOCATION_COORDINATES.items():
0047         if name in normalized:
0048             return coordinates
0049     return None
0050
0051
0052 def _load_model() -> Optional[GlobalMonotonicXGB]:
0053     path = ROOT / "artifacts" / "global_multi_causal_xgb.pkl"
0054     if not path.exists():
0055         return None
0056     try:
0057         import joblib
0058         model = joblib.load(path)
0059         return model if isinstance(model, GlobalMonotonicXGB) else None
0060     except Exception:
0061         return None
0062
0063
0064 def _h3_boundary_lonlat(cell_id: str) -> List[List[float]]:
0065     boundary_fn = getattr(h3, "cell_to_boundary", None) or getattr(h3, "h3_to_geo_boundary")
0066     ring = [[float(lon), float(lat)] for lat, lon in boundary_fn(cell_id)]
0067     if ring and ring[0] != ring[-1]:
0068         ring.append(ring[0])
0069     return ring
0070
0071
0072 def _finite(value: Any, default: float = 0.0) -> float:
0073     try:
0074         result = float(value)
0075     except (TypeError, ValueError):
0076         return default
0077     return result if math.isfinite(result) else default
0078
0079
0080 def _field_capacity_stress_mask(inputs: PhysicsInputs, state: Any) -> np.ndarray:
0081     slope = np.radians(np.clip(np.asarray(inputs.slope_deg, dtype=np.float64), 0.0, 89.9))
0082     field_capacity_saturation = np.clip(state.theta_33 / np.maximum(state.theta_s, 1e-6), 0.0, 0.95)
0083     field_capacity_head = state.regolith_depth_m * field_capacity_saturation * np.cos(slope) ** 2
0084     return state.pore_pressure_head_m > field_capacity_head
0085
0086
0087 class GlobalScanService:
0088     """The shared application layer used by REST routes and agent tools."""
0089
0090     def __init__(self, provider_timeout_s: Optional[float] = None) -> None:
0091         self.provider_timeout_s = provider_timeout_s or float(os.getenv("TERRA_PROVIDER_TIMEOUT_S", "8"))
0092         self.model = _load_model()
0093         self.engine = MultiCausalEngine()
0094         self._cell_contexts: Dict[str, Dict[str, Any]] = {}
0095         self._context_order: List[str] = []
0096         self._context_lock = threading.RLock()
0097
0098     async def scan(
0099         self,
0100         location_query: str = "Chooralmala, Wayanad",
0101         lat: Optional[float] = None,
0102         lon: Optional[float] = None,
0103         rain_delta_mm: float = 0.0,
0104         storm_duration_hr: float = 24.0,
0105         pga_override_g: float = 0.0,
0106         snowmelt_temp_surge_c: float = 0.0,
0107         burn_severity_override: float = 0.0,
0108         agency_mode: str = "LOCAL_EOC",
0109         _snapshot=None,
0110     ) -> Dict[str, Any]:
0111         if (lat is None) != (lon is None):
0112             raise ValueError("lat and lon must be provided together")
0113         if storm_duration_hr <= 0:
0114             raise ValueError("storm_duration_hr must be positive")
0115         numeric = [rain_delta_mm, storm_duration_hr, pga_override_g, snowmelt_temp_surge_c, burn_severity_override]
0116         numeric.extend(value for value in (lat, lon) if value is not None)
0117         if not all(math.isfinite(float(value)) for value in numeric):
0118             raise ValueError("coordinates and scenario inputs must be finite")
0119         if rain_delta_mm < 0 or pga_override_g < 0 or snowmelt_temp_surge_c < 0:
0120             raise ValueError("scenario deltas and overrides must be nonnegative")
0121         if not 0.0 <= burn_severity_override <= 1.0:
0122             raise ValueError("burn_severity_override must be between 0 and 1")
0123         used_named_fallback = False
0124         if lat is not None and lon is not None:
0125             location: Union[str, Tuple[float, float]] = (lat, lon)
0126         else:
0127             known_coordinates = known_location_coordinates(location_query)
0128             if known_coordinates is not None:
0129                 location = known_coordinates
0130                 used_named_fallback = True
0131             else:
0132                 location = location_query
0133
0134         if _snapshot is None:
0135             async with GlobalIngestor(provider_timeout_s=self.provider_timeout_s) as ingestor:
0136                 live_scan = await ingestor.scan(location, radius_km=3.5)
0137         else:
0138             live_scan = _snapshot
0139         base = evaluate_live_physics(live_scan, self.model)
0140         if base is None:
0141             return self._empty_response(live_scan, agency_mode)
0142
0143         base_inputs: PhysicsInputs = base["station_inputs"]
0144         count = int(base["cell_count"])
0145         rain = max(0.0, _finite(base_inputs.rain_24h_mm)) + rain_delta_mm
0146         pga = max(_finite(base_inputs.pga_g), pga_override_g)
0147         air = _finite(base_inputs.air_temperature_c)
0148         soil_temp = _finite(base_inputs.soil_temperature_c)
0149         burn = np.maximum(
0150             np.asarray(base_inputs.burn_severity, dtype=np.float64), burn_severity_override
0151         )
0152         inputs = replace(
0153             base_inputs,
0154             rain_24h_mm=rain,
0155             storm_duration_hr=storm_duration_hr,
0156             pga_g=pga,
0157             air_temperature_c=air,
0158             soil_temperature_c=soil_temp,
0159             burn_severity=burn,
0160         )
0161         inputs = apply_alpine_weather(inputs, [cell.elevation_m for cell in base["cells"]], snowmelt_temp_surge_c)
0162         state = self.engine.compute(inputs)
0163         kinematics = solve_kinematics(inputs, state)
0164         feature_rows = _feature_matrix(inputs, state, base["profile_curvature"])
0165         p_xgb = self.model.predict_proba(feature_rows) if self.model is not None else None
0166         probability = blend_failure_probability(p_xgb, state.p_failure_physical, inputs.slope_deg, state.fs_dynamic) if p_xgb is not None else state.p_failure_physical
0167         shap_values = self.model.shap_log_odds_contributions(feature_rows) if self.model is not None else None
0168         breakdown = explain_causes(inputs, state)
0169
0170         cell_ids = list(base["cell_ids"])
0171         cells = base["cells"]
0172         infra_by_cell = live_scan.infrastructure.by_h3_cell if live_scan.infrastructure else {}
0173         households: List[Optional[int]] = []
0174         exposure = np.ones(count, dtype=np.float64)
0175         exposure_basis: List[str] = []
0176         for index, cell_id in enumerate(cell_ids):
0177             infra = infra_by_cell.get(cell_id, {})
0178             household_count = infra.get("households")
0179             building_count = max(0, int(infra.get("buildings", 0) or 0))
0180             households.append(int(household_count) if household_count is not None else None)
0181             if household_count is not None:
0182                 exposure[index] = max(1.0, float(household_count))
0183                 exposure_basis.append("mapped_household_count")
0184             elif building_count > 0:
0185                 exposure[index] = float(building_count)
0186                 exposure_basis.append("OSM_building_count_proxy")
0187             else:
0188                 exposure[index] = 1.0
0189                 exposure_basis.append("unit_weight_exposure_fallback")
0190
0191         kinematic_factor = 1.0 + .75*np.clip(state.newmark_displacement_cm/10.,0.,1.5) + .50*np.clip(kinematics.peak_velocity_m_s/12.,0.,1.)
0192         hotspot_score = probability * (0.35 + 0.65 * exposure) * kinematic_factor
0193         top_index = int(np.argmax(hotspot_score))
0194         viewport_fs_p05 = float(np.percentile(state.fs_dynamic,5))
0195         viewport_probability_p95 = float(np.percentile(probability,95))
0196         steep = state.effective_slope_deg >= 24.0
0197         steep_fs = state.fs_dynamic[steep] if np.any(steep) else state.fs_dynamic
0198         min_fs_dynamic = float(np.percentile(steep_fs, 5))
0199         max_p_failure = float(np.percentile(probability[steep] if np.any(steep) else probability, 95))
0200         causal_top = {cause: float(values[top_index]) for cause, values in breakdown.combined_percentages.items()}
0201         dominant_cause = max(causal_top, key=causal_top.get) if breakdown.valid[top_index] else "no_active_trigger"
0202         numbers_top = [
0203             item.to_dict()
0204             for item in select_numbers_that_matter(
0205                 top_index, inputs, state, breakdown, probability_failure=probability
0206             )
0207         ]
0208
0209         if self.model is not None:
0210             intervals, widths, bivariate = self.model.conformal_intervals(probability,feature_rows)
0211         else:
0212             intervals=np.tile([0.,1.],(count,1)); widths=np.ones(count)
0213             bivariate=np.where(probability>=.5,'HIGH_SUSCEPTIBILITY_HIGH_UNCERTAINTY','LOW_SUSCEPTIBILITY_HIGH_UNCERTAINTY')
0214         geo_features = []
0215         serialized_cell_contexts: Dict[str, Dict[str, Any]] = {}
0216         for index, cell_id in enumerate(cell_ids):
0217             cell = cells[index]
0218             cell_causes = {cause: float(values[index]) for cause, values in breakdown.combined_percentages.items()}
0219             cell_numbers = [
0220                 item.to_dict()
0221                 for item in select_numbers_that_matter(index, inputs, state, breakdown, probability_failure=probability)
0222             ]
0223             cell_dominant = max(cell_causes, key=cell_causes.get) if breakdown.valid[index] else "no_active_trigger"
0224             cell_households = households[index]
0225             properties = {
0226                 "h3_index": cell_id,
0227                 "lat": float(cell.latitude),
0228                 "lon": float(cell.longitude),
0229                 "elevation_m": float(cell.elevation_m),
0230                 "slope_deg": float(np.asarray(inputs.slope_deg)[index]),
0231                 "twi": float(np.asarray(inputs.twi)[index]),
0232                 "fs_dynamic": float(state.fs_dynamic[index]),
0233                 "effective_scarp_slope_deg": float(state.effective_slope_deg[index]),
0234                 "causal_attribution_valid": bool(breakdown.valid[index]),
0235                 "effective_mobilization_depth_m": float(state.regolith_depth_m[index]),
0236                 "fs_static_dry": float(state.fs_static_dry[index]),
0237                 "p_failure_seismic": float(state.p_failure_seismic[index]),
0238                 "p_failure_m1": float(state.p_failure_m1[index]),
0239                 "effective_burn_severity": float(state.effective_burn_severity[index]),
0240                 "prefire_fuel_index_proxy": float(np.asarray(inputs.ndvi_norm)[index]),
0241                 "cryospheric_eligible": bool(state.cryospheric_eligible[index]),
0242                 "lateral_resistance_kpa": float(state.lateral_resistance_kpa[index]),
0243                 "conformal_interval_90": intervals[index].tolist(),
0244                 "uncertainty_width": float(widths[index]),
0245                 "bivariate_class": str(bivariate[index]),
0246                 "conformal_target": "binary event label; not a confidence interval for true failure probability",
0247                 "uncertainty_status": "ensemble_scaled_cell_residual_interval_no_spatial_coverage_guarantee" if self.model is not None and getattr(self.model,'conformal_',None) else "uncalibrated_vacuous_interval",
0248                 "fs_mohr_coulomb": float(state.fs_mohr_coulomb[index]),
0249                 "fs_postfire_yield_index": float(state.fs_postfire_yield_index[index]),
0250                 "debris_volume_proxy_m3": float(state.debris_volume_m3[index]),
0251                 "rain_15min_proxy_mm": float(state.rain_15min_proxy_mm[index]),
0252                 "suction_cohesion_kpa": float(state.suction_cohesion_kpa[index]),
0253                 "cryo_perched_head_m": float(state.cryo_perched_head_m[index]),
0254                 "equivalent_water_mm": float(state.equivalent_water_mm[index]),
0255                 "mechanism_quality": "M1/Gartner cell-input proxies; postfire FS is a yield index; assumed ice joints and basal drainage",
0256                 "newmark_displacement_cm": float(state.newmark_displacement_cm[index]),
0257                 "cell_snow_depth_m": float(np.asarray(inputs.snow_depth_m)[index]),
0258                 "cell_air_temperature_c": float(np.asarray(inputs.air_temperature_c)[index]),
0259                 "alpine_weather_quality": ALPINE_ASSUMPTION,
0260                 "critical_acceleration_g": float(state.critical_acceleration_g[index]),
0261                 "p_failure": float(probability[index]),
0262                 "hotspot_score": float(hotspot_score[index]),
0263                 "kinematic_factor": float(kinematic_factor[index]),
0264                 "extrusion_height_m": float(probability[index] * 100.0),
0265                 "cruden_varnes_class": int(kinematics.velocity_class[index]),
0266                 "cruden_varnes_label": str(kinematics.velocity_class_name[index]),
0267                 "peak_velocity_ms": float(kinematics.peak_velocity_m_s[index]),
0268                 "failure_mode": str(kinematics.failure_mode[index]),
0269                 "dominant_cause": cell_dominant,
0270                 "road_name": None,
0271                 "households": cell_households,
0272                 "exposure_weight": float(exposure[index]),
0273                 "exposure_basis": exposure_basis[index],
0274                 "causal_breakdown_pct": cell_causes,
0275                 "numbers_that_matter": cell_numbers,
0276             }
0277             geo_features.append({"type": "Feature", "geometry": {"type": "Polygon", "coordinates": [_h3_boundary_lonlat(cell_id)]}, "properties": properties})
0278             serialized_cell_contexts[cell_id] = properties
0279
0280         top_hotspot = {
0281             **serialized_cell_contexts[cell_ids[top_index]],
0282             "causal_breakdown_pct": causal_top,
0283             "numbers_that_matter": numbers_top,
0284         }
0285         routes = self._runout_features(live_scan, cell_ids, state, kinematics, probability, hotspot_score)
0286         top_route=next((r['properties']['path'] for r in routes if r['properties']['source_cell']==cell_ids[top_index]),[cell_ids[top_index]])
0287         from backend.physics.cascading import cascading_screen
0288         top_hotspot.update(cascading_screen(cell_ids[top_index],top_route,live_scan.terrain.cells))
0289         critical_mask = (state.fs_dynamic <= 1.05) | (probability >= 0.65)
0290         household_values = [value for value in households if value is not None]
0291         weather = live_scan.weather
0292         summary = {
0293             "wmo_alert_level": self._alert_level(max_p_failure, rain),
0294             "max_p_failure": max_p_failure,
0295             "critical_fs_static": float(np.percentile(state.fs_static[steep] if np.any(steep) else state.fs_static,5)),
0296             "critical_fs": min_fs_dynamic, "critical_p_failure": max_p_failure,
0297             "summary_population": "effective_slope_ge_24_deg" if np.any(steep) else "all_cells",
0298             "summary_population_count": int(np.count_nonzero(steep)) if np.any(steep) else count,
0299             "viewport_fs_p05": viewport_fs_p05, "viewport_probability_p95": viewport_probability_p95,
0300             "min_fs_dynamic": min_fs_dynamic,
0301             "dominant_cause": dominant_cause,
0302             "causal_attribution_valid": bool(breakdown.valid[top_index]),
0303             "causal_attribution_quality": "stress_budget" if breakdown.valid[top_index] else "neutral_prior_no_active_stress",
0304             "critical_hotspots_count": int(np.count_nonzero(critical_mask)),
0305             "total_households_exposed": int(sum(household_values)) if household_values else None,
0306             "household_exposure_quality": "OSM household counts" if household_values else "household counts unavailable; building-count/unit-weight proxy used for hotspot ranking",
0307             "causal_breakdown_pct": causal_top,
0308             "kinematics_top_hotspot": top_hotspot,
0309             "numbers_that_matter": numbers_top,
0310             "meteorological_metrics": {
0311                 "forecast_rain_48h_mm": (weather.forecast_rain_next_24h_mm + weather.forecast_rain_24_48h_mm) if weather and weather.forecast_rain_next_24h_mm is not None and weather.forecast_rain_24_48h_mm is not None else None,
0312                 "rain_24h_mm_observed": weather.rain_24h_mm if weather else None,
0313                 "soil_temperature_c_observed": weather.soil_temperature_0_7cm_c if weather else None,
0314                 "assimilated_soil_saturation_fraction": float(np.mean(inputs.soil_saturation_fraction)),
0315                 "rain_24h_mm_scenario": rain,
0316                 "storm_duration_hr": storm_duration_hr,
0317                 "peak_rain_3h_mm": weather.peak_rain_3h_mm if weather else None,
0318                 "ari_7d_mm": weather.ari_7d_mm if weather else None,
0319                 "snow_depth_m": weather.snow_depth_m if weather else None,
0320                 "air_temperature_c_scenario": air + snowmelt_temp_surge_c,
0321                 "cell_air_temperature_range_c": [float(np.min(inputs.air_temperature_c)), float(np.max(inputs.air_temperature_c))],
0322                 "cell_snow_depth_range_m": [float(np.min(inputs.snow_depth_m)), float(np.max(inputs.snow_depth_m))],
0323                 "alpine_weather_assumption": ALPINE_ASSUMPTION,
0324                 "pga_g_observed": live_scan.seismic.max_pga_g if live_scan.seismic else None,
0325                 "pga_g_scenario": pga,
0326                 "burn_severity_override": burn_severity_override,
0327             },
0328             "shap_status": "native_tree_shap_fold_boosters" if shap_values is not None else "model_unavailable",
0329             "shap_mean_absolute_log_odds": None if shap_values is None else dict(zip(self.model.feature_names_in_, np.mean(shap_values,axis=0).tolist())),
0330             "probability_source": "mechanical gate × regime-weighted isotonic XGBoost/physical union; gated score not separately calibrated" if p_xgb is not None else "physical FOSM/seismic/M1 union; no trained model artifact loaded",
0331             "runout_parameter_quality": RUNOUT_PARAMETER_QUALITY,
0332         }
0333         response = {
0334             "location": {
0335                 "query": location_query if used_named_fallback or (location_query and location_query != "Chooralmala, Wayanad" and (lat is None or lon is None)) else live_scan.location.query,
0336                 "display_name": location_query if used_named_fallback else live_scan.location.display_name,
0337                 "latitude": live_scan.location.latitude,
0338                 "longitude": live_scan.location.longitude,
0339                 "agency_mode": agency_mode,
0340                 "bedrock_lithology": live_scan.bedrock, "river_discharge": live_scan.river,
0341                 "nasa_telemetry": self._nasa_telemetry(live_scan),
0342                 "reliability_report": reliability_report(live_scan),
0343             },
0344             "summary": summary,
0345             "h3_geojson": {"type": "FeatureCollection", "features": geo_features},
0346             "runout_geojson": {"type": "FeatureCollection", "features": routes},
0347             "source_status": {key: value.__dict__ for key, value in live_scan.sources.items()},
0348             "generated_at_utc": live_scan.generated_at_utc,
0349             "elapsed_seconds": live_scan.elapsed_seconds,
0350             "scenario": {
0351                 "rain_delta_mm": rain_delta_mm,
0352                 "storm_duration_hr": storm_duration_hr,
0353                 "pga_override_g": pga_override_g,
0354                 "snowmelt_temp_surge_c": snowmelt_temp_surge_c,
0355                 "burn_severity_override": burn_severity_override,
0356             },
0357         }
0358         self._remember_contexts(cell_ids, {
0359             "scan": live_scan,
0360             "inputs": inputs,
0361             "state": state,
0362             "kinematics": kinematics,
0363             "probability": probability,
0364             "breakdown": breakdown,
0365             "hotspot_score": hotspot_score,
0366             "exposure": exposure,
0367             "top_index": top_index,
0368             "feature_rows": feature_rows,
0369         })
0370         return response
0371
0372     def inspect_hexagon(self, h3_index: str) -> Dict[str, Any]:
0373         context = self._get_context(h3_index)
0374         scan: GlobalScan = context["scan"]
0375         cell = scan.terrain.cells[h3_index]
0376         properties = context["state"]
0377         index = context["cell_ids"].index(h3_index) if "cell_ids" in context else list(scan.terrain.cells).index(h3_index)
0378         state = context["state"]
0379         breakdown = context["breakdown"]
0380         return {
0381             "h3_index": h3_index,
0382             "latitude": cell.latitude,
0383             "longitude": cell.longitude,
0384             "elevation_m": cell.elevation_m,
0385             "slope_deg": cell.slope_deg,
0386             "twi": cell.twi,
0387             "fs_dynamic": float(state.fs_dynamic[index]),
0388             "p_f_fosm": float(state.p_failure_fosm[index]),
0389             "p_failure": float(context["probability"][index]),
0390             "pore_pressure_head_m": float(state.pore_pressure_head_m[index]),
0391             "newmark_displacement_cm": float(state.newmark_displacement_cm[index]),
0392             "peak_velocity_ms": float(context["kinematics"].peak_velocity_m_s[index]),
0393             "cruden_varnes_label": str(context["kinematics"].velocity_class_name[index]),
0394             "failure_mode": str(context["kinematics"].failure_mode[index]),
0395             "causal_breakdown_pct": {cause: float(values[index]) for cause, values in breakdown.combined_percentages.items()},
0396             "hotspot_score": float(context["hotspot_score"][index]),
0397             "baseline_hydrostatic_credited": bool(_field_capacity_stress_mask(context["inputs"], state)[index]),
0398             "physics_feature_provenance": self._cell_provenance(scan, h3_index),
0399         }
0400
0401     def trace_runout(self, h3_index: str) -> Dict[str, Any]:
0402         context = self._get_context(h3_index)
0403         scan: GlobalScan = context["scan"]
0404         state = context["state"]
0405         kinematics = context["kinematics"]
0406         index = context["cell_ids"].index(h3_index)
0407         infra = scan.infrastructure.by_h3_cell if scan.infrastructure else {}
0408         route = route_runout(
0409             h3_index,
0410             scan.terrain.cells,
0411             float(kinematics.peak_velocity_m_s[index]),
0412             RUNOUT_MU_B,
0413             RUNOUT_XI,
0414             infrastructure_by_cell=infra,
0415             max_steps=50,
0416         )
0417         return {
0418             **route.to_dict(),
0419             "coordinates_lonlat": [
0420                 [float(scan.terrain.cells[cell_id].longitude), float(scan.terrain.cells[cell_id].latitude)]
0421                 for cell_id in route.path
0422             ],
0423             "kinematics": {
0424                 "peak_velocity_ms": float(kinematics.peak_velocity_m_s[index]),
0425                 "cruden_varnes_label": str(kinematics.velocity_class_name[index]),
0426                 "failure_mode": str(kinematics.failure_mode[index]),
0427                 "fs_dynamic": float(state.fs_dynamic[index]),
0428                 "p_failure": float(context["probability"][index]),
0429             },
0430             "evacuation_planning_note": "Planning aid only; verify route conditions, occupancy, and agency guidance before operational use.",
0431             "route_parameters": {"mu_b": RUNOUT_MU_B, "xi": RUNOUT_XI, "quality": RUNOUT_PARAMETER_QUALITY},
0432         }
0433
0434     def _runout_features(
0435         self,
0436         scan: GlobalScan,
0437         cell_ids: Sequence[str],
0438         state: Any,
0439         kinematics: Any,
0440         probability: np.ndarray,
0441         hotspot_score: np.ndarray,
0442     ) -> List[Dict[str, Any]]:
0443         critical = np.flatnonzero((state.fs_dynamic <= 1.05) | (probability >= 0.65))
0444         ordered = sorted(critical.tolist(), key=lambda index: float(hotspot_score[index]), reverse=True)[:5]
0445         infra = scan.infrastructure.by_h3_cell if scan.infrastructure else {}
0446         result = []
0447         for index in ordered:
0448             cell_id = cell_ids[index]
0449             route = route_runout(
0450                 cell_id, scan.terrain.cells,
0451                 float(kinematics.peak_velocity_m_s[index]), RUNOUT_MU_B, RUNOUT_XI,
0452                 infrastructure_by_cell=infra, max_steps=50,
0453                 flow_depth_m=float(state.regolith_depth_m[index]*np.clip(state.pore_pressure_head_m[index]/max(state.regolith_depth_m[index],1e-9),.25,1.)),
0454             )
0455             if len(route.path) < 2:
0456                 continue
0457             coordinates = [
0458                 [float(scan.terrain.cells[item].longitude), float(scan.terrain.cells[item].latitude)]
0459                 for item in route.path
0460             ]
0461             result.append({
0462                 "type": "Feature",
0463                 "geometry": {"type": "LineString", "coordinates": coordinates},
0464                 "properties": {
0465                     **route.to_dict(),
0466                     "peak_velocity_ms": float(kinematics.peak_velocity_m_s[index]),
0467                     "p_failure": float(probability[index]),
0468                     "hotspot_score": float(hotspot_score[index]),
0469                     "route_parameters": {"mu_b": RUNOUT_MU_B, "xi": RUNOUT_XI, "quality": RUNOUT_PARAMETER_QUALITY},
0470                 },
0471             })
0472         return result
0473
0474     @staticmethod
0475     def _alert_level(max_probability: float, rain_24h_mm: float) -> str:
0476         if max_probability >= ALERT_THRESHOLDS["alert"]:
0477             return "RED"
0478         if max_probability >= ALERT_THRESHOLDS["watch"]:
0479             return "ORANGE"
0480         if max_probability >= ALERT_THRESHOLDS["warning"]:
0481             return "YELLOW"
0482         return "GREEN"
0483
0484     @staticmethod
0485     def _cell_provenance(scan: GlobalScan, cell_id: str) -> Dict[str, Any]:
0486         sat = scan.satellite_cells.get(cell_id, {})
0487         hydro = scan.hydrology
0488         soil = hydro.soil_property_quality if hydro else "fallback"
0489         return {
0490             "soil": soil,
0491             "vegetation": sat.get("vegetation_proxy_quality", "Esri visible-band VDVI or neutral fallback"),
0492             "burn": sat.get("burn_proxy_quality", "not assessed; default zero"),
0493             "road_distance": "OSM cut tag when mapped; otherwise 9999 m",
0494             "river_distance": "DEM H3 TWI >=14.5 proxy",
0495         }
0496
0497     def _remember_contexts(self, cell_ids: Sequence[str], context: Dict[str, Any]) -> None:
0498         context["cell_ids"] = list(cell_ids)
0499         with self._context_lock:
0500             for cell_id in cell_ids:
0501                 self._cell_contexts[cell_id] = context
0502                 if cell_id in self._context_order:
0503                     self._context_order.remove(cell_id)
0504                 self._context_order.append(cell_id)
0505             while len(self._context_order) > 12000:
0506                 old = self._context_order.pop(0)
0507                 self._cell_contexts.pop(old, None)
0508
0509     def _get_context(self, cell_id: str) -> Dict[str, Any]:
0510         try:
0511             is_valid = h3.is_valid_cell(cell_id) if hasattr(h3, "is_valid_cell") else h3.h3_is_valid(cell_id)
0512         except Exception:
0513             is_valid = False
0514         if not is_valid:
0515             raise KeyError("invalid H3 cell index")
0516         with self._context_lock:
0517             context = self._cell_contexts.get(cell_id)
0518         if context is None:
0519             raise KeyError("H3 cell has no active scan context; run /api/v1/scan first")
0520         return context
0521
0522     @staticmethod
0523     def _nasa_telemetry(scan):
0524         h=scan.hydrology
0525         return {
0526             "nasa_power_gwetroot": h.nasa_gwetroot_pct if h and h.nasa_gwetroot_pct is not None else None,
0527             "nasa_power_prectotcorr_mm": h.nasa_precipitation_mm_day if h else None,
0528             "nasa_eonet_active_events": scan.eonet_events,
0529             "nasa_coolr_regional_count_100km": scan.coolr_nearby.get("nasa_coolr_regional_count_100km",0),
0530             "nearest_coolr_event_km": scan.coolr_nearby.get("nearest_coolr_event_km"),
0531             "nasa_coolr_nearby_count": scan.coolr_nearby["count"], "coolr_count_provenance": scan.coolr_nearby,
0532             "srtm_terrarium_elevation_range_m": [scan.terrain.min_elevation_m, scan.terrain.max_elevation_m] if scan.terrain else None,
0533             "quality": {"power": "latest available daily values; not instantaneous",
0534                         "eonet": "returned event records with their dates; not a local hazard count",
0535                         "coolr": scan.coolr_nearby["quality"],
0536                         "nasadem": "not fetched; terrain uses AWS Terrarium, not verified NASADEM"},
0537             "source_status": {k:v.__dict__ for k,v in scan.sources.items() if k.startswith('nasa_') or k.startswith('eonet_')},
0538         }
0539
0540     def _empty_response(self, scan: GlobalScan, agency_mode: str) -> Dict[str, Any]:
0541         weather=scan.weather
0542         return {
0543             "location": {"query": scan.location.query, "display_name": scan.location.display_name, "latitude": scan.location.latitude, "longitude": scan.location.longitude, "agency_mode": agency_mode, "bedrock_lithology": scan.bedrock, "river_discharge": scan.river, "nasa_telemetry": self._nasa_telemetry(scan), "reliability_report": reliability_report(scan)},
0544             "summary": {
0545                 "wmo_alert_level": "UNAVAILABLE", "max_p_failure": None, "min_fs_dynamic": None,
0546                 "dominant_cause": None, "critical_hotspots_count": 0, "total_households_exposed": None,
0547                 "causal_breakdown_pct": {cause: 20.0 for cause in CAUSES}, "kinematics_top_hotspot": None,
0548                 "causal_attribution_valid": False, "causal_attribution_quality": "neutral_prior_no_terrain",
0549                 "numbers_that_matter": [{"metric": key, "value": 0.0, "unit": unit, "threshold": "unavailable", "status": "NORMAL", "causal_group": "unavailable", "weight_percent": 0.0, "available": False}
0550                     for key,unit in [("dynamic_factor_of_safety","ratio"),("blended_failure_probability","%"),("peak_velocity","m/s"),("pore_pressure_head","m"),("slope_angle","degrees"),("newmark_displacement","cm")]],
0551                 "meteorological_metrics": {
0552                     "forecast_rain_48h_mm": (weather.forecast_rain_next_24h_mm + weather.forecast_rain_24_48h_mm) if weather and weather.forecast_rain_next_24h_mm is not None and weather.forecast_rain_24_48h_mm is not None else None,
0553                     "rain_24h_mm_observed": weather.rain_24h_mm if weather else None,
0554                     "ari_7d_mm": weather.ari_7d_mm if weather else None,
0555                     "snow_depth_m": weather.snow_depth_m if weather else None,
0556                     "air_temperature_c_scenario": weather.air_temperature_c if weather else None,
0557                     "soil_temperature_c_observed": weather.soil_temperature_0_7cm_c if weather else None,
0558                     "pga_g_observed": scan.seismic.max_pga_g if scan.seismic else None,
0559                 }, "physics_status": "no sampled land cells",
0560             },
0561             "h3_geojson": {"type": "FeatureCollection", "features": []},
0562             "runout_geojson": {"type": "FeatureCollection", "features": []},
0563             "source_status": {key: value.__dict__ for key, value in scan.sources.items()},
0564             "generated_at_utc": scan.generated_at_utc,
0565             "elapsed_seconds": scan.elapsed_seconds,
0566         }
```

## windows_5070ti_trainer/train_all_on_5070ti.py

SHA-256: `c0830345e856a90cadf5050fa70f5d401825d8e955b035001618fdf400f8d1c6`

```python
0001 """Portable CUDA XGBoost and guarded QLoRA training entry point."""
0002
0003 from __future__ import annotations
0004
0005 import argparse
0006 import hashlib
0007 import json
0008 import shutil
0009 import subprocess
0010 import sys
0011 from datetime import datetime, timezone
0012 from pathlib import Path
0013 from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple
0014
0015 import numpy as np
0016
0017 ROOT = Path(__file__).resolve().parent
0018 sys.path.insert(0, str(ROOT))
0019
0020 from backend.ml.global_xgb import FEATURE_NAMES, MONOTONE_VECTOR, GlobalMonotonicXGB, make_base_estimator, spatial_group_splits
0021 from backend.release import portable_payload
0022 from backend.ml.validation_suite import (
0023     build_validation_report,
0024     load_labeled_training_data,
0025     run_kinematic_branch_controls,
0026 )
0027
0028
0029 def _load_rows(path: Path) -> List[Dict[str, Any]]:
0030     import pyarrow.parquet as pq
0031     return pq.read_table(path).to_pylist()
0032
0033
0034 def _write_block_csv_if_needed(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
0035     """No synthetic records are emitted; used only for the existing CSV hindcast contract."""
0036     del path, rows
0037
0038
0039 def tune_cuda_xgb(X: np.ndarray, y: np.ndarray, groups: np.ndarray, trials: int = 20) -> Dict[str, Any]:
0040     import optuna
0041     from sklearn.metrics import roc_auc_score
0042     from xgboost import XGBClassifier
0043
0044     positive_blocks = len(np.unique(groups[y == 1]))
0045     n_splits = min(5, positive_blocks)
0046     if n_splits < 3:
0047         raise ValueError("at least three positive-bearing spatial blocks are required for grouped tuning")
0048     folds = spatial_group_splits(y, groups, n_splits)
0049     for fold, (train_idx, valid_idx) in enumerate(folds, start=1):
0050         if len(np.unique(y[train_idx])) < 2 or len(np.unique(y[valid_idx])) < 2:
0051             raise ValueError("spatial fold {} lacks both labeled classes; tuning stopped".format(fold))
0052
0053     def objective(trial: Any) -> float:
0054         depth = trial.suggest_int("max_depth", 3, 9)
0055         n_estimators = trial.suggest_int("n_estimators", 200, 900, step=100)
0056         learning_rate = trial.suggest_float("learning_rate", 0.015, 0.12, log=True)
0057         min_child_weight = trial.suggest_float("min_child_weight", 1.0, 12.0)
0058         scores = []
0059         for train_idx, valid_idx in folds:
0060             model = XGBClassifier(
0061                 n_estimators=n_estimators, max_depth=depth, learning_rate=learning_rate,
0062                 min_child_weight=min_child_weight, objective="binary:logistic", eval_metric="logloss",
0063                 tree_method="hist", device="cuda", monotone_constraints=MONOTONE_VECTOR,
0064                 n_jobs=-1, verbosity=0,
0065             )
0066             model.fit(X[train_idx], y[train_idx], sample_weight=np.asarray([0.25 if str(g).startswith("synthetic:") else 1.0 for g in groups[train_idx]]))
0067             scores.append(float(roc_auc_score(y[valid_idx], model.predict_proba(X[valid_idx])[:, 1])))
0068         return float(np.mean(scores))
0069
0070     study = optuna.create_study(direction="maximize", study_name="terra_shield_spatial_cuda_xgb")
0071     study.optimize(objective, n_trials=trials, show_progress_bar=False)
0072     return {"best_value_spatial_auc": study.best_value, "best_params": study.best_params, "trials": trials, "spatial_folds": n_splits}
0073
0074
0075 def verify_rows(rows: Sequence[Mapping[str, Any]]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
0076     training_path = ROOT / "data" / "model_training.parquet"
0077     if not training_path.exists():
0078         raise FileNotFoundError("Required embargoed model_training.parquet is missing; rebuild the frozen split")
0079     metadata=json.loads((ROOT/'data/real_global_landslide_observations.metadata.json').read_text())
0080     if metadata.get('event_group_holdout',{}).get('status')!='ready':
0081         raise ValueError("Source enrichment or frozen holdout split incomplete; refusing full-dataset training")
0082     X, y, groups = load_labeled_training_data(str(training_path))
0083     labeled = [row for row in rows if row.get("label") in (0, 1) and all(row.get(name) is not None for name in FEATURE_NAMES)]
0084     if len(np.unique(y)) != 2:
0085         raise ValueError("training requires both source-backed inventory positives and temporal inventory-absence controls")
0086     if any(row.get("label") == 0 and not str(row.get("label_quality", "")).startswith(("temporal_inventory_absence_pseudo_negative", "simulated_mechanical_negative")) for row in labeled):
0087         raise ValueError("negative labels must carry the explicit temporal inventory-absence quality marker")
0088     positive_blocks = len(np.unique(groups[y == 1]))
0089     if len(np.unique(groups)) < 5 or positive_blocks < 3:
0090         raise ValueError("training requires at least five coarse blocks and three positive-bearing blocks")
0091     if len(y) < 450:
0092         raise ValueError("training requires at least 450 complete labeled rows; found {}".format(len(y)))
0093     if any(row.get("label_quality", "").startswith("user_supplied") for row in rows if row.get("label") in (0, 1)):
0094         raise ValueError("user-supplied candidate points are not accepted as labeled ground truth")
0095     return X, y, groups
0096
0097
0098 def export_chatml_from_observations(rows: Sequence[Mapping[str, Any]], destination: Path) -> int:
0099     """Convert one real event observation per record; never clone rows to reach 600."""
0100     candidates = [
0101         row for row in rows
0102         if row.get("event_candidate") == 1
0103         and row.get("source_name") not in (None, "user-provided project candidate list")
0104         and row.get("event_id") and row.get("event_date") and row.get("h3_cell_id")
0105     ]
0106     unique = {}
0107     for row in candidates:
0108         trajectory_id = "{}|{}|{}".format(row.get("source_event_id") or row["event_id"], row["h3_cell_id"], row["event_date"])
0109         unique[trajectory_id] = row
0110     if len(unique) < 600:
0111         raise ValueError("600 distinct source-linked cell/date trajectories are required for QLoRA; found {} (no duplicates will be generated)".format(len(unique)))
0112     pools = {}
0113     for row in unique.values():
0114         pools.setdefault(str(row.get("regime") or "unknown"), []).append(row)
0115     selected = []
0116     while len(selected) < 600:
0117         for key in sorted(pools):
0118             if pools[key] and len(selected) < 600:
0119                 selected.append(pools[key].pop(0))
0120     destination.parent.mkdir(parents=True, exist_ok=True)
0121     with destination.open("w", encoding="utf-8") as handle:
0122         for row in selected:
0123             obs = {
0124                 key: row.get(key) for key in (
0125                     "event_id", "name", "event_date", "latitude", "longitude", "regime", "source_name",
0126                     "weather_source_status", "rain_event_day_mm", "rain_antecedent_7d_mm", "snowfall_event_day_mm",
0127                     "temperature_event_day_mean_c", "soil_moisture_event_day_mean_m3m3", "terrain_source_status",
0128                     "terrain_slope_deg", "terrain_twi", "terrain_elevation_m", "usgs_source_status",
0129                     "usgs_max_magnitude", "pga_attenuation_proxy_g", "label", "label_quality",
0130                     *FEATURE_NAMES, "physics_feature_provenance", "physics_feature_quality", "source_event_id", "h3_cell_id", "source_url", "citation",
0131                 )
0132             }
0133             messages = [
0134                 {"role": "user", "content": "Summarize the sourced observations for this landslide event candidate. Distinguish measurements, estimates, and missing evidence; do not infer a cell failure probability."},
0135                 {"role": "assistant", "content": "I will inspect the event-linked measurements and their provenance."},
0136                 {"role": "assistant", "tool_calls": [{"id": "obs1", "type": "function", "function": {"name": "get_event_observations", "arguments": json.dumps({"event_id": row["event_id"]})}}]},
0137                 {"role": "tool", "tool_call_id": "obs1", "name": "get_event_observations", "content": json.dumps(obs, sort_keys=True)},
0138                 {"role": "assistant", "content": "These are event-linked cell/date observations. Use the label only with its stated quality: polygon positives are inventory observations, point/report labels are location proxies, and 60-day pre-event negatives are inventory-absence assumptions. Synthetic mechanical controls are excluded from these source trajectories. Feature provenance distinguishes observations, estimates, and explicit defaults; no calibrated probability is supplied here."},
0139             ]
0140             handle.write(json.dumps({"messages": messages, "source_event_id": row["event_id"]}, ensure_ascii=False) + "\n")
0141     return 600
0142
0143
0144 def finetune_qwen(dataset_path: Path, output_dir: Path, gguf_path: Path) -> Dict[str, Any]:
0145     from train_qwen25_agent import train_qwen25
0146     return train_qwen25(dataset_path, output_dir, gguf_path)
0147
0148
0149 def write_hindcast_scores(candidates_path: Path, model: GlobalMonotonicXGB, destination: Path) -> Dict[str, Any]:
0150     import csv
0151     import pyarrow.parquet as pq
0152     from backend.ml.global_xgb import blend_failure_probability
0153
0154     rows = [row for row in pq.read_table(candidates_path).to_pylist()
0155             if row.get("label") in (0, 1) and all(row.get(name) is not None for name in FEATURE_NAMES)]
0156     if not rows or {row["label"] for row in rows} != {0, 1}:
0157         raise ValueError("held-out inventory must contain complete features for both source labels")
0158     X = np.asarray([[float(row[name]) for name in FEATURE_NAMES] for row in rows], dtype=np.float32)
0159     probability = model.predict_proba(X)  # Ungated event-buffer evaluation only.
0160     columns = ["case_id", "p_failure", "observed_failure", "event_date", "source_event_id", "event_group_id", "source_url", "label_quality", "split", "probability_source"]
0161     destination.parent.mkdir(parents=True, exist_ok=True)
0162     with destination.open("w", newline="", encoding="utf-8") as handle:
0163         writer = csv.DictWriter(handle, fieldnames=columns)
0164         writer.writeheader()
0165         for row, score in zip(rows, probability):
0166             writer.writerow({
0167                 "case_id": row.get("event_group_id"), "p_failure": float(score),
0168                 "observed_failure": int(row["label"]), "event_date": row.get("event_date"),
0169                 "source_event_id": row.get("source_event_id"), "event_group_id": row.get("event_group_id"),
0170                 "source_url": row.get("source_url"), "label_quality": row.get("label_quality"),
0171                 "split": "inventory_group_holdout_excluded_from_training",
0172                 "probability_source": "isotonic XGBoost event-buffer score; live scores remain gated",
0173             })
0174     return {"status": "written", "path": str(destination.relative_to(ROOT)), "rows": len(rows), "positive_rows": int(sum(row["label"] == 1 for row in rows)), "negative_rows": int(sum(row["label"] == 0 for row in rows))}
0175
0176
0177 def main() -> int:
0178     parser = argparse.ArgumentParser(description=__doc__)
0179     parser.add_argument("--skip-dataset-build", action="store_true")
0180     parser.add_argument("--rebuild-dataset", action="store_true", help="fetch a fresh source catalog even when the bundled Parquet already exists")
0181     parser.add_argument("--train-llm", action="store_true")
0182     parser.add_argument("--optuna-trials", type=int, default=20)
0183     parser.add_argument("--max-events", type=int, default=450)
0184     args = parser.parse_args()
0185     data_path = ROOT / "data" / "real_global_landslide_observations.parquet"
0186     artifacts = ROOT / "artifacts"
0187     artifacts.mkdir(parents=True, exist_ok=True)
0188     from backend.release import invalidate_readiness
0189     invalidate_readiness(artifacts, "GPU training started; copied audit results no longer grant readiness")
0190     report: Dict[str, Any] = {"generated_at_utc": datetime.now(timezone.utc).isoformat(), "device": "cuda", "engineering_ready": False, "pretraining_ready": False, "audit_status": "GPU retraining requires a fresh full-system audit", "label_policy": "COOLR polygon/manual/report-point proxies, temporal absence pseudo-negatives and explicit simulated training-only mechanical negatives."}
0191     if not args.skip_dataset_build and (args.rebuild_dataset or not data_path.exists()):
0192         result = subprocess.run([sys.executable, str(ROOT / "backend" / "ml" / "build_real_labeled_dataset.py"), "--output", str(data_path), "--max-events", str(args.max_events)], cwd=str(ROOT), check=False)
0193         report["dataset_build_exit_code"] = result.returncode
0194     else:
0195         report["dataset_build"] = "using bundled existing observation Parquet"
0196     try:
0197         import torch
0198         if not torch.cuda.is_available():
0199             raise RuntimeError("PyTorch CUDA is not available")
0200         if not data_path.exists():
0201             raise FileNotFoundError("dataset not found: {}".format(data_path))
0202         rows = _load_rows(data_path)
0203         X, y, groups = verify_rows(rows)
0204         training_path = ROOT / "data" / "model_training.parquet"
0205         if not training_path.exists():
0206             training_path = data_path
0207         report["labeled_rows"] = len(y)
0208         report["positive_rows"] = int((y == 1).sum())
0209         report["negative_rows"] = int((y == 0).sum())
0210         positive_rows = [row for row in _load_rows(training_path) if row.get("label") == 1]
0211         report["positive_source_events"] = len({row.get("source_event_id") for row in positive_rows if row.get("source_event_id")})
0212         report["positive_event_dates"] = sorted({row.get("event_date") for row in positive_rows if row.get("event_date")})
0213         report["positive_regimes"] = sorted({row.get("regime") for row in positive_rows if row.get("regime")})
0214         report["model_scope_warning"] = "Inventory polygon and point-buffer proxies differ in spatial precision; temporal-absence controls are not surveyed intact slopes; validation does not establish global or five-regime generalization."
0215         report["spatial_blocks"] = int(len(np.unique(groups)))
0216         report["cuda_device"] = torch.cuda.get_device_name(0)
0217         tuning = tune_cuda_xgb(X, y, groups, args.optuna_trials)
0218         report["optuna"] = tuning
0219         model = GlobalMonotonicXGB(
0220             n_estimators=int(tuning["best_params"]["n_estimators"]),
0221             max_depth=int(tuning["best_params"]["max_depth"]),
0222             learning_rate=float(tuning["best_params"]["learning_rate"]),
0223             min_child_weight=float(tuning["best_params"]["min_child_weight"]),
0224         )
0225         model.device = "cuda"
0226         model.fit(X, y, groups, n_splits=int(tuning["spatial_folds"]))
0227         from backend.ml.conformal_calibration import fit_reserved_conformal
0228         report['conformal_calibration']=fit_reserved_conformal(model,ROOT/'data',groups)
0229         import joblib
0230         joblib.dump(model, artifacts / "global_multi_causal_xgb.pkl")
0231         (artifacts / "global_multi_causal_xgb.metadata.json").write_text(json.dumps({
0232             "training_rows":int(len(y)), "positive_rows":int((y==1).sum()), "negative_rows":int((y==0).sum()),
0233             "synthetic_training_only_rows":int(sum(str(g).startswith("synthetic:") for g in groups)),
0234             "physics_model_version":"mechanical_fs_v6", "conformal":model.conformal_, "feature_names":list(FEATURE_NAMES),
0235             "calibration":"isotonic on source/absence holdouts; simulated controls enter fitting only",
0236             "probability_policy":"mechanically gated blend; deployed score not separately calibrated",
0237         }, indent=2), encoding="utf-8")
0238         report["model"] = {"status": "trained", "calibration": "spatial GroupKFold isotonic", "feature_names": list(FEATURE_NAMES), "monotonic_constraints": list(MONOTONE_VECTOR)}
0239         try:
0240             shap_values = model.shap_log_odds_contributions(X)
0241             report["tree_shap_mean_absolute_log_odds"] = {
0242                 name: float(value) for name, value in zip(FEATURE_NAMES, np.mean(shap_values, axis=0))
0243             }
0244         except Exception as exc:
0245             report["tree_shap_status"] = "unavailable: {}: {}".format(type(exc).__name__, exc)
0246         hindcast_path = ROOT / "data" / "historical_hindcast.csv"
0247         metadata_path = data_path.with_suffix(".metadata.json")
0248         metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else {}
0249         holdout = metadata.get("event_group_holdout", {})
0250         if holdout.get("status") == "ready":
0251             candidate_path = ROOT / "data" / "historical_hindcast_candidates.parquet"
0252             if candidate_path.exists():
0253                 try:
0254                     report["historical_hindcast_export"] = write_hindcast_scores(candidate_path, model, hindcast_path)
0255                 except Exception as exc:
0256                     report["historical_hindcast_export"] = {"status": "failed", "error": "{}: {}".format(type(exc).__name__, exc)}
0257         validation = build_validation_report(str(training_path), str(hindcast_path), model)
0258         validation.stage1.details.append("Training labels include 60-day inventory-absence pseudo-negatives; scores are not verified non-failure metrics.")
0259         validation.stage1.details.append("Spatial CV and the event-group holdout have different scopes; point-buffer labels are approximate source-location proxies.")
0260         report["validation"] = validation.to_dict()
0261         from backend.ml.conformal_calibration import heldout_conformal_diagnostics
0262         report['conformal_calibration']=model.conformal_
0263         report['conformal_holdout']=heldout_conformal_diagnostics(model,_load_rows(ROOT/'data/historical_hindcast_candidates.parquet'))
0264         filtered_csv=ROOT/'data/historical_hindcast_pinls.csv'
0265         write_hindcast_scores(ROOT/'data/historical_hindcast_pinls.parquet',model,filtered_csv)
0266         from backend.ml.validation_suite import load_hindcast_csv,historical_hindcast_metrics
0267         hp,hy,hc=load_hindcast_csv(str(filtered_csv))
0268         report['stage2_pinls_filtered']=historical_hindcast_metrics(hp,hy,hc).to_dict()
0269         report['stage2_pinls_filtered']['scope']='Secondary stability-filtered cohort; original holdout retained separately'
0270         report["stage4_kinematics"] = [item.__dict__ for item in run_kinematic_branch_controls()]
0271         report["model_status"] = "trained"
0272     except Exception as exc:
0273         report["model_status"] = "not_run"
0274         report["model_error"] = "{}: {}".format(type(exc).__name__, exc)
0275         report["validation"] = build_validation_report(str(data_path), str(ROOT / "data" / "historical_hindcast.csv"), None).to_dict()
0276         report["stage4_kinematics"] = [item.__dict__ for item in run_kinematic_branch_controls()]
0277         report["stage3_controls_status"] = "These physics controls are synthetic checks, not observed scores."
0278         report_path = artifacts / "scientific_validation_report.json"
0279         report_path.write_text(json.dumps(portable_payload(report), indent=2, allow_nan=False), encoding="utf-8")
0280         print("Training stopped safely: {}".format(report["model_error"]))
0281         print("Report: {}".format(report_path))
0282         return 2
0283
0284     if args.train_llm:
0285         try:
0286             chatml = ROOT / "data" / "chatml_trajectories.jsonl"
0287             if chatml.exists():
0288                 from train_qwen25_agent import validate_chatml
0289                 count = validate_chatml(chatml)
0290             else:
0291                 count = export_chatml_from_observations(rows, chatml)
0292             report["chatml_trajectory_count"] = count
0293             report["qwen_exports"] = finetune_qwen(chatml, artifacts / "qwen2_5_7b_lora", artifacts / "terra-shield-qwen-q4_k_m.gguf")
0294             report["qwen_status"] = "trained; see qwen_exports for conversion status"
0295         except Exception as exc:
0296             report["qwen_status"] = "not_run: {}: {}".format(type(exc).__name__, exc)
0297             print("QLoRA not run: {}".format(report["qwen_status"]))
0298     report_path = artifacts / "scientific_validation_report.json"
0299     report_path.write_text(json.dumps(portable_payload(report), indent=2, allow_nan=False), encoding="utf-8")
0300     print("Training report written to {}".format(report_path))
0301     return 0
0302
0303
0304 if __name__ == "__main__":
0305     raise SystemExit(main())
```

## windows_5070ti_trainer/train_qwen25_agent.py

SHA-256: `7c63b66894235c122e669a97dab0bfa60127ee741d04573f719acc803a4164d4`

```python
0001 """QLoRA fine-tuning and GGUF export for the RTX 5070 Ti trainer."""
0002
0003 from __future__ import annotations
0004
0005 import argparse
0006 import json
0007 import shutil
0008 from pathlib import Path
0009 from typing import Any, Dict, List, Mapping
0010
0011 ROOT = Path(__file__).resolve().parent
0012
0013
0014 def validate_chatml(path: Path, expected_rows: int = 600) -> int:
0015     if not path.exists():
0016         raise FileNotFoundError("source-linked ChatML trajectory file is missing: {}".format(path))
0017     seen = set()
0018     count = 0
0019     with path.open("r", encoding="utf-8") as handle:
0020         for line_number, line in enumerate(handle, start=1):
0021             if not line.strip():
0022                 continue
0023             row = json.loads(line)
0024             source_id = str(row.get("source_event_id") or "")
0025             messages = row.get("messages")
0026             if not source_id or not isinstance(messages, list) or not messages:
0027                 raise ValueError("ChatML row {} lacks a source ID or messages".format(line_number))
0028             if source_id in seen:
0029                 raise ValueError("duplicate source-linked ChatML trajectory: {}".format(source_id))
0030             if not any(message.get("role") == "tool" for message in messages):
0031                 raise ValueError("ChatML row {} does not contain a source observation tool result".format(line_number))
0032             seen.add(source_id)
0033             count += 1
0034     if count != expected_rows:
0035         raise ValueError("expected {} distinct trajectories, found {}".format(expected_rows, count))
0036     return count
0037
0038
0039 def export_trained_model(model: Any, tokenizer: Any, output_dir: Path, gguf_path: Path) -> Dict[str, Any]:
0040     """Preserve trained weights before optional conversions; record actual outputs."""
0041     output_dir.mkdir(parents=True, exist_ok=True)
0042     adapter = output_dir / "adapter"
0043     model.save_pretrained(str(adapter))
0044     tokenizer.save_pretrained(str(adapter))
0045     manifest = {"adapter": {"status": "saved", "path": "adapter"}}
0046     for kind in ("merged_16bit", "gguf"):
0047         try:
0048             if kind == "merged_16bit":
0049                 model.save_pretrained_merged(str(output_dir / kind), tokenizer, save_method="merged_16bit")
0050                 manifest[kind] = {"status": "saved", "path": kind}
0051             else:
0052                 # A fresh staging directory prevents stale GGUF files masquerading as a new export.
0053                 import tempfile
0054                 gguf_path.parent.mkdir(parents=True, exist_ok=True)
0055                 with tempfile.TemporaryDirectory(prefix="gguf_export_", dir=str(output_dir)) as temporary:
0056                     model.save_pretrained_gguf(temporary, tokenizer, quantization_method="q4_k_m")
0057                     exports = [p for p in Path(temporary).glob("*.gguf") if "q4_k_m" in p.name.lower()]
0058                     if not exports:
0059                         raise RuntimeError("No Q4_K_M GGUF produced")
0060                     shutil.move(str(exports[0]), str(gguf_path))
0061                 (gguf_path.parent / "Modelfile").write_text("FROM ./{}\nPARAMETER temperature 0.2\n".format(gguf_path.name), encoding="utf-8")
0062                 manifest[kind] = {"status": "saved", "path": gguf_path.name}
0063         except Exception as exc:
0064             manifest[kind] = {"status": "failed", "error_type": type(exc).__name__, "error": str(exc)}
0065             print("{} export failed; trained adapter retained: {}".format(kind, exc))
0066         (output_dir / "export_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
0067     return manifest
0068
0069
0070 def train_qwen25(dataset_path: Path, output_dir: Path, gguf_path: Path) -> Dict[str, Any]:
0071     import torch
0072
0073     if not torch.cuda.is_available():
0074         raise RuntimeError("Qwen2.5 QLoRA training requires a visible CUDA GPU; use the Windows RTX trainer.")
0075     validate_chatml(dataset_path)
0076
0077     from datasets import load_dataset
0078     from trl import SFTConfig, SFTTrainer
0079     from unsloth import FastLanguageModel
0080
0081     dataset = load_dataset("json", data_files={"train": str(dataset_path)}, split="train")
0082     model, tokenizer = FastLanguageModel.from_pretrained(
0083         model_name="unsloth/Qwen2.5-7B-Instruct",
0084         max_seq_length=2048,
0085         load_in_4bit=True,
0086     )
0087     model = FastLanguageModel.get_peft_model(
0088         model,
0089         r=16,
0090         lora_alpha=16,
0091         lora_dropout=0.0,
0092         target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
0093         use_gradient_checkpointing="unsloth",
0094         random_state=3407,
0095     )
0096
0097     def render(example: Mapping[str, Any]) -> Dict[str, str]:
0098         return {"text": tokenizer.apply_chat_template(example["messages"], tokenize=False, add_generation_prompt=False)}
0099
0100     dataset = dataset.map(render, num_proc=1)
0101     output_dir.mkdir(parents=True, exist_ok=True)
0102     config = SFTConfig(
0103         output_dir=str(output_dir),
0104         per_device_train_batch_size=1,
0105         gradient_accumulation_steps=8,
0106         warmup_steps=10,
0107         num_train_epochs=2,
0108         learning_rate=2e-4,
0109         logging_steps=10,
0110         optim="adamw_8bit",
0111         weight_decay=0.01,
0112         lr_scheduler_type="linear",
0113         seed=3407,
0114         bf16=bool(torch.cuda.is_bf16_supported()),
0115         fp16=not bool(torch.cuda.is_bf16_supported()),
0116         max_length=2048,
0117         dataset_text_field="text",
0118         dataset_num_proc=1,
0119         dataloader_num_workers=0,
0120         packing=False,
0121         report_to="none",
0122     )
0123     trainer = SFTTrainer(model=model, processing_class=tokenizer, train_dataset=dataset, args=config)
0124     trainer.train()
0125
0126     return export_trained_model(model, tokenizer, output_dir, gguf_path)
0127
0128
0129 def main() -> int:
0130     parser = argparse.ArgumentParser(description=__doc__)
0131     parser.add_argument("--dataset", default=str(ROOT / "data" / "chatml_trajectories.jsonl"))
0132     parser.add_argument("--output-dir", default=str(ROOT / "artifacts" / "qwen2_5_7b_lora"))
0133     parser.add_argument("--gguf", default=str(ROOT / "artifacts" / "terra-shield-qwen-q4_k_m.gguf"))
0134     args = parser.parse_args()
0135     train_qwen25(Path(args.dataset), Path(args.output_dir), Path(args.gguf))
0136     return 0
0137
0138
0139 if __name__ == "__main__":
0140     raise SystemExit(main())
0141
```

## backend/physics/multi_causal_engine.py

SHA-256: `31bd29796bc4008b99fb0db7edf146a55490f2a228dd82866745cb9f1e0f9773`

```python
0001 """Vectorized five-regime slope-stability equations for H3 cells."""
0002
0003 from __future__ import annotations
0004
0005 import math
0006 from dataclasses import asdict, dataclass, field
0007 from typing import Any, Dict, List, Mapping, Optional, Sequence, Union
0008
0009 import numpy as np
0010 from .mechanisms import postfire, suction_cohesion, ice_factors
0011
0012 ArrayInput = Union[float, int, Sequence[float], np.ndarray]
0013 GAMMA_W_KN_M3 = 9.81
0014 PARTICLE_DENSITY_G_CM3 = 2.65
0015
0016
0017 @dataclass
0018 class PhysicsInputs:
0019     """Measured or explicitly scenario-defined inputs; no silent imputation is applied."""
0020
0021     sand_fraction: ArrayInput
0022     clay_fraction: ArrayInput
0023     organic_matter_fraction: ArrayInput
0024     bulk_density_g_cm3: ArrayInput
0025     slope_deg: ArrayInput
0026     twi: ArrayInput
0027     soil_saturation_fraction: ArrayInput
0028     rain_24h_mm: ArrayInput
0029     storm_duration_hr: ArrayInput
0030     snow_depth_m: ArrayInput
0031     air_temperature_c: ArrayInput
0032     soil_temperature_c: ArrayInput
0033     freeze_thaw_cycles_7d: ArrayInput
0034     burn_severity: ArrayInput
0035     ndvi_norm: ArrayInput
0036     pga_g: ArrayInput
0037     road_cut_distance_m: ArrayInput
0038     river_distance_m: ArrayInput
0039     profile_curvature: ArrayInput = 0.0
0040     elevation_m: ArrayInput = 0.0
0041     warming_delta_c: ArrayInput = 0.0
0042     peak_3h_intensity_mmhr: ArrayInput = 0.0
0043     basal_conductivity_ratio: ArrayInput = 0.01
0044     vg_alpha_kpa_inv: ArrayInput = 0.10
0045     vg_n: ArrayInput = 2.0
0046     latitude_deg: ArrayInput = 0.0
0047     baseline_air_temperature_c: ArrayInput = -999.0
0048     observed_snow_depth_m: ArrayInput = -1.0
0049     river_surge_ratio: ArrayInput = 1.0
0050     geology_friction_adjustment_deg: ArrayInput = 0.0
0051     geology_cohesion_adjustment_kpa: ArrayInput = 0.0
0052     block_width_m: ArrayInput = 35.0
0053     hollow_confinement_ratio: ArrayInput = 1.0
0054
0055
0056 @dataclass
0057 class PhysicsState:
0058     theta_s: np.ndarray
0059     theta_1500: np.ndarray
0060     theta_33: np.ndarray
0061     hydraulic_conductivity_m_s: np.ndarray
0062     friction_angle_deg: np.ndarray
0063     soil_cohesion_kpa: np.ndarray
0064     regolith_depth_m: np.ndarray
0065     snowmelt_mm_day: np.ndarray
0066     thaw_factor: np.ndarray
0067     root_cohesion_kpa: np.ndarray
0068     fire_runoff_multiplier: np.ndarray
0069     effective_conductivity_m_s: np.ndarray
0070     effective_infiltration_m_s: np.ndarray
0071     local_saturation: np.ndarray
0072     hydraulic_diffusivity_m2_s: np.ndarray
0073     dimensionless_time: np.ndarray
0074     iverson_response: np.ndarray
0075     pore_pressure_head_m: np.ndarray
0076     toe_undercut_multiplier: np.ndarray
0077     wet_unit_weight_kn_m3: np.ndarray
0078     effective_cohesion_kpa: np.ndarray
0079     kh: np.ndarray
0080     fs_static: np.ndarray
0081     fs_dynamic: np.ndarray
0082     sigma_fs: np.ndarray
0083     beta_fosm: np.ndarray
0084     p_failure_physical: np.ndarray
0085     critical_acceleration_g: np.ndarray
0086     newmark_displacement_cm: np.ndarray
0087     unburned_regolith_depth_m: np.ndarray
0088     fs_static_dry: np.ndarray
0089     p_failure_seismic: np.ndarray
0090     p_failure_fosm: np.ndarray
0091     effective_slope_deg: np.ndarray
0092     effective_soil_cohesion_kpa: np.ndarray
0093     fire_driving_multiplier: np.ndarray
0094     fs_mohr_coulomb: np.ndarray
0095     fs_postfire_yield_index: np.ndarray
0096     p_failure_m1: np.ndarray
0097     debris_volume_m3: np.ndarray
0098     rain_15min_proxy_mm: np.ndarray
0099     suction_cohesion_kpa: np.ndarray
0100     cryo_perched_head_m: np.ndarray
0101     ice_friction_factor: np.ndarray
0102     equivalent_water_mm: np.ndarray
0103     effective_burn_severity: np.ndarray
0104     cryospheric_eligible: np.ndarray
0105     lateral_resistance_kpa: np.ndarray
0106     warnings: List[str] = field(default_factory=list)
0107
0108     @property
0109     def cell_count(self) -> int:
0110         return int(self.fs_dynamic.size)
0111
0112     def to_dict(self) -> Dict[str, Any]:
0113         result = asdict(self)
0114         for key, value in result.items():
0115             if isinstance(value, np.ndarray):
0116                 result[key] = value.tolist()
0117         return result
0118
0119
0120 class MultiCausalEngine:
0121     """Evaluate soil hydraulics, five causal mechanisms, FS, FOSM, and Newmark."""
0122
0123     def compute(self, inputs: PhysicsInputs) -> PhysicsState:
0124         values = self._broadcast_inputs(inputs)
0125         sand = np.clip(values["sand_fraction"], 0.05, 0.85)
0126         clay = np.clip(values["clay_fraction"], 0.05, 0.65)
0127         organic = 100.0 * np.clip(values["organic_matter_fraction"], 0.005, 0.06)  # coefficients use percent OM
0128         bulk_density = np.clip(values["bulk_density_g_cm3"], 1.15, 1.70)
0129         slope_deg = np.clip(values["slope_deg"], 0.0, 89.9)
0130         curvature = np.abs(values["profile_curvature"])
0131         amplified = slope_deg * (1.0 + 0.20 * np.clip(curvature / 0.015, 0.25, 1.0))
0132         # Never flatten a DEM slope already steeper than the 48 degree scenario cap.
0133         effective_slope = np.where(slope_deg >= 15.0, np.maximum(slope_deg, np.minimum(amplified, 48.0)), slope_deg)
0134         slope = np.radians(effective_slope)
0135         sin_slope, cos_slope = np.sin(slope), np.cos(slope)
0136         cos2 = cos_slope**2
0137         twi = values["twi"]
0138         initial_saturation = np.clip(values["soil_saturation_fraction"], 0.0, 1.0)
0139         rain = np.maximum(values["rain_24h_mm"], 0.0)
0140         storm_hr = values["storm_duration_hr"]
0141         snow_depth = np.maximum(values["snow_depth_m"], 0.0)
0142         air_temp = values["air_temperature_c"]
0143         soil_temp = values["soil_temperature_c"]
0144         freeze_cycles = np.maximum(values["freeze_thaw_cycles_7d"], 0.0)
0145         ndvi = np.clip(values["ndvi_norm"], 0.0, 1.0)
0146         burn = np.clip(values["burn_severity"], 0.0, 1.0) * np.clip(ndvi / .22, 0.0, 1.0)
0147         pga = np.maximum(values["pga_g"], 0.0)
0148         road_distance = np.maximum(values["road_cut_distance_m"], 0.0)
0149         river_distance = np.maximum(values["river_distance_m"], 0.0)
0150         if np.any(storm_hr <= 0):
0151             raise ValueError("storm_duration_hr must be positive for every cell")
0152         if np.any(twi <= 0):
0153             raise ValueError("TWI must be positive for every cell")
0154
0155         # Saxton & Rawls 2006 equations, retaining the specified clip bounds.
0156         theta_s = np.clip(1.0 - bulk_density / PARTICLE_DENSITY_G_CM3, 0.34, 0.58)
0157         theta_1500_t = (
0158             -0.024 * sand
0159             + 0.487 * clay
0160             + 0.006 * organic
0161             + 0.005 * sand * organic
0162             - 0.013 * clay * organic
0163             + 0.068 * sand * clay
0164             + 0.031
0165         )
0166         theta_1500 = np.clip(theta_1500_t + 0.14 * theta_1500_t - 0.02, 0.05, 0.38)
0167         theta_33_t = (
0168             -0.251 * sand
0169             + 0.195 * clay
0170             + 0.011 * organic
0171             + 0.006 * sand * organic
0172             - 0.027 * clay * organic
0173             + 0.452 * sand * clay
0174             + 0.299
0175         )
0176         theta_33_raw = theta_33_t + 1.283 * theta_33_t**2 - 0.374 * theta_33_t - 0.015
0177         theta_33 = np.clip(theta_33_raw, theta_1500 + 0.05, theta_s - 0.04)
0178         retention_gap = np.log(theta_33) - np.log(theta_1500)
0179         b_parameter = np.divide(
0180             math.log(1500.0) - math.log(33.0),
0181             retention_gap,
0182             out=np.full_like(retention_gap, np.inf),
0183             where=np.abs(retention_gap) > 1e-12,
0184         )
0185         lambda_parameter = np.divide(
0186             1.0,
0187             b_parameter,
0188             out=np.zeros_like(b_parameter),
0189             where=np.isfinite(b_parameter) & (np.abs(b_parameter) > 1e-12),
0190         )
0191         ksat_raw = (
0192             1930.0
0193             * np.power(np.maximum(theta_s - theta_33, 1e-12), 3.0 - lambda_parameter)
0194             * (1e-3 / 3600.0)
0195         )
0196         ksat = np.clip(ksat_raw, 1e-7, 1e-3)
0197         friction_deg = np.clip(24.0 + 14.0 * sand - 8.0 * clay, 18.0, 39.0)
0198         friction_deg += values["geology_friction_adjustment_deg"]
0199         tan_phi = np.tan(np.radians(friction_deg))
0200         cohesion_soil = np.clip(4.0 + 16.0 * clay - 3.5 * sand, 2.0, 20.0)
0201         cohesion_soil = np.maximum(0.,cohesion_soil + values["geology_cohesion_adjustment_kpa"])
0202         base_depth = np.clip(2.8 * np.exp(-0.85 * np.tan(slope)), 0.60, 3.00)
0203         # Keep the slab thickness independent of the separate post-fire runoff branch.
0204         regolith_depth = base_depth.copy()  # slab depth is independent of the runoff branch
0205
0206         snowmelt = np.where(
0207             (snow_depth > 0.01) & (air_temp > 0.0),
0208             np.maximum(0.0, 4.5 * air_temp + 0.0126 * rain * air_temp),
0209             0.0,
0210         )
0211         from .alpine_weather import cryospheric_eligibility
0212         baseline_air = np.where(values['baseline_air_temperature_c'] > -998,
0213                                 values['baseline_air_temperature_c'], air_temp-values['warming_delta_c'])
0214         observed_snow = np.where(values['observed_snow_depth_m'] >= 0,
0215                                  values['observed_snow_depth_m'], snow_depth)
0216         cryo_active = cryospheric_eligibility(observed_snow, baseline_air, values['elevation_m'], values['latitude_deg'])
0217         _, thaw, ice_friction = ice_factors(values['elevation_m'],soil_temp,snow_depth,freeze_cycles,values['warming_delta_c'],eligible=cryo_active)
0218         tan_phi_base = tan_phi.copy()
0219         tan_phi = tan_phi * ice_friction
0220         friction_deg = np.degrees(np.arctan(tan_phi))
0221         root_cohesion_baseline = 12.0 * np.power(ndvi, 1.35)
0222         root_cohesion = root_cohesion_baseline * (1.0 - 0.90 * burn)
0223         fire_multiplier = np.ones_like(burn)
0224         keff = ksat.copy()
0225
0226         median_twi = float(np.median(twi))
0227         local_saturation = np.clip(initial_saturation * np.power(twi / median_twi, 0.28), 0.10, 0.99)
0228
0229         fire_driving = np.ones_like(burn)
0230         soil_cohesion_effective = cohesion_soil.copy()
0231         infiltration = (
0232             rain * 1e-3 / (storm_hr * 3600.0) + snowmelt * 1e-3 / 86400.0
0233         ) * fire_multiplier
0234         hydraulic_diffusivity = 4.0 * keff * cos2 / np.maximum(theta_s * (1.0 - local_saturation), 0.015)
0235         dimensionless_time = storm_hr * 3600.0 * hydraulic_diffusivity / (regolith_depth**2)
0236         safe_time = np.maximum(dimensionless_time, 1e-12)
0237         iverson_response = (
0238             np.sqrt(safe_time / math.pi) * np.exp(-1.0 / safe_time)
0239             - self._erfc(1.0 / np.sqrt(safe_time))
0240         )
0241         pressure_head = np.clip(
0242             regolith_depth * local_saturation * cos2
0243             + regolith_depth * np.minimum(infiltration / keff, 4.0) * iverson_response,
0244             0.0,
0245             regolith_depth * cos2,
0246         )
0247
0248         # Bounded active-layer water storage above an assumed low-permeability base.
0249         # This mass-balance approximation is not the TRIGRS PDE solution.
0250         basal_ratio=np.clip(values['basal_conductivity_ratio'],0,1)
0251         perched=np.where(cryo_active & (snowmelt>0),
0252             np.minimum(regolith_depth*cos2, np.maximum(0.,infiltration-ksat*basal_ratio)*storm_hr*3600/np.maximum(theta_s,.01)),0.)
0253         pressure_head=np.minimum(regolith_depth*cos2,pressure_head+perched)
0254         se=np.clip((local_saturation-.08)/.92,.02,.999)
0255         alpha=values['vg_alpha_kpa_inv'];n=values['vg_n']
0256         if np.any(alpha<=0) or np.any(n<=1):raise ValueError('Invalid van Genuchten parameters')
0257         suction_stress=-se/alpha*np.expm1(np.log(se)*n/(1-n))**(1/n)
0258         taper=np.clip((.85-se)/.10,0,1);taper=taper*taper*(3-2*taper)
0259         storm_activation=np.clip((rain+snowmelt)/25.,0,1)
0260         suction=np.clip(-suction_stress*tan_phi,0,8)*taper*(1-storm_activation)
0261         suction=np.where(perched>0,0.,suction)
0262         p_m1,volume,fs_fire,r15=postfire(burn,effective_slope,clay,sand,rain,values['peak_3h_intensity_mmhr'],base_depth)
0263
0264         toe_multiplier = (
0265             1.0
0266             + 0.35 * np.exp(-road_distance / 60.0) * (slope_deg > 16.0)
0267             + 0.15 * np.exp(-river_distance / 80.0) * (twi / 15.0) * np.where((twi>=14.5)&(values["river_surge_ratio"]>1.3),np.sqrt(np.clip(values["river_surge_ratio"],1,5)),1.)
0268         )
0269         wet_unit_weight = (bulk_density + local_saturation * theta_s) * GAMMA_W_KN_M3
0270         effective_cohesion = (soil_cohesion_effective + root_cohesion) * thaw + suction
0271         kh = 0.50 * pga
0272
0273         lateral = np.zeros_like(regolith_depth)  # compatibility output; no lateral force
0274         normal_static = wet_unit_weight * regolith_depth * cos2 - pressure_head * GAMMA_W_KN_M3
0275         driving_static = wet_unit_weight * regolith_depth * sin_slope * cos_slope * toe_multiplier * fire_driving
0276         fs_static_raw = (effective_cohesion + normal_static * tan_phi) / np.maximum(driving_static, 1.0)
0277         fs_static = np.clip(fs_static_raw, 0.10, 5.00)
0278
0279         normal_dynamic = (
0280             wet_unit_weight * regolith_depth * (cos2 - kh * sin_slope * cos_slope)
0281             - pressure_head * GAMMA_W_KN_M3
0282         )
0283         driving_dynamic = (
0284             wet_unit_weight
0285             * regolith_depth
0286             * (sin_slope * cos_slope + kh * cos2)
0287             * toe_multiplier * fire_driving
0288         )
0289         denominator_dynamic = np.maximum(driving_dynamic, 1.0)
0290         fs_dynamic_raw = (effective_cohesion + normal_dynamic * tan_phi) / denominator_dynamic
0291         fs_mohr = np.clip(fs_dynamic_raw, 0.10, 5.00)
0292         fs_dynamic = fs_mohr.copy()
0293
0294         # Delta-method FOSM, including the derivative of the specified FS clip.
0295         sigma_cohesion = 0.18 * effective_cohesion
0296         lateral_coefficient = 0.0
0297         sigma_tan_phi = 0.12 * tan_phi
0298         sigma_pressure = 0.15 * pressure_head
0299         clip_active = (fs_dynamic_raw > 0.10) & (fs_dynamic_raw < 5.00)
0300         d_fs_dc = np.where(clip_active, 1.0 / denominator_dynamic, 0.0)
0301         d_fs_dtan = np.where(clip_active, normal_dynamic / denominator_dynamic, 0.0)
0302         d_fs_dpressure = np.where(
0303             clip_active, -GAMMA_W_KN_M3 * tan_phi / denominator_dynamic, 0.0
0304         )
0305         sigma_fs = np.sqrt(
0306             (d_fs_dc * sigma_cohesion) ** 2
0307             + (d_fs_dtan * sigma_tan_phi) ** 2
0308             + (d_fs_dpressure * sigma_pressure) ** 2
0309         )
0310         beta = (fs_mohr - 1.0) / np.maximum(sigma_fs, 0.05)
0311         p_failure = 0.5 * self._erfc(beta / math.sqrt(2.0))
0312
0313         # Pre-storm, unsaturated, unburned baseline; excludes seismic load and thaw.
0314         dry_gamma = bulk_density * GAMMA_W_KN_M3
0315         dry_normal = dry_gamma * base_depth * cos2
0316         dry_driving = dry_gamma * base_depth * sin_slope * cos_slope * toe_multiplier
0317         fs_static_dry = np.clip((cohesion_soil + root_cohesion_baseline + dry_normal * tan_phi_base)
0318                                / np.maximum(dry_driving, 1.0), 0.10, 5.0)
0319         critical_acceleration = np.clip((fs_static_dry - 1.0) * sin_slope, 0.05, 0.60)
0320         pga_ratio = np.clip(
0321             critical_acceleration / np.maximum(pga, 1e-12), 1e-12, 1.0 - 1e-12
0322         )
0323         newmark = np.where(
0324             pga > critical_acceleration,
0325             np.power(10.0, 0.215 + 2.341 * np.log10(1.0 - pga_ratio) - 1.438 * np.log10(pga_ratio)),
0326             0.0,
0327         )
0328
0329         # User-specified detachment cap; retain a separate regional empirical component.
0330         newmark = np.minimum(newmark, 120.0)
0331         p_seismic = 0.335 * (-np.expm1(-0.048 * np.power(newmark, 1.565)))
0332         # Requested probabilistic union; this treats the three components as independent.
0333         p_fosm = p_failure.copy()
0334         p_failure = 1.0-(1.0-p_fosm)*(1.0-p_seismic)*(1.0-p_m1)
0335         warnings = ["M1/Gartner inputs are cell proxies; postfire FS is a yield index. Ice weakening, basal drainage and VG parameters are scenario assumptions. Probability union assumes independent components."]
0336         reversed_bounds = np.where(theta_33 <= theta_1500)[0]
0337         if reversed_bounds.size:
0338             warnings.append(
0339                 "Saxton-Rawls bounds produced theta_33 <= theta_1500 for {} cell(s); exact requested clipping was retained".format(
0340                     int(reversed_bounds.size)
0341                 )
0342             )
0343         return PhysicsState(
0344             theta_s=theta_s,
0345             theta_1500=theta_1500,
0346             theta_33=theta_33,
0347             hydraulic_conductivity_m_s=ksat,
0348             friction_angle_deg=friction_deg,
0349             soil_cohesion_kpa=cohesion_soil,
0350             regolith_depth_m=regolith_depth,
0351             snowmelt_mm_day=snowmelt,
0352             thaw_factor=thaw,
0353             root_cohesion_kpa=root_cohesion,
0354             fire_runoff_multiplier=fire_multiplier,
0355             effective_conductivity_m_s=keff,
0356             effective_infiltration_m_s=infiltration,
0357             local_saturation=local_saturation,
0358             hydraulic_diffusivity_m2_s=hydraulic_diffusivity,
0359             dimensionless_time=dimensionless_time,
0360             iverson_response=iverson_response,
0361             pore_pressure_head_m=pressure_head,
0362             toe_undercut_multiplier=toe_multiplier,
0363             wet_unit_weight_kn_m3=wet_unit_weight,
0364             effective_cohesion_kpa=effective_cohesion,
0365             kh=kh,
0366             fs_static=fs_static,
0367             fs_dynamic=fs_dynamic,
0368             sigma_fs=sigma_fs,
0369             beta_fosm=beta,
0370             p_failure_physical=p_failure,
0371             critical_acceleration_g=critical_acceleration,
0372             newmark_displacement_cm=newmark,
0373             effective_slope_deg=effective_slope, effective_soil_cohesion_kpa=soil_cohesion_effective,
0374             fire_driving_multiplier=fire_driving,
0375             unburned_regolith_depth_m=base_depth, fs_static_dry=fs_static_dry,
0376             p_failure_seismic=p_seismic, p_failure_fosm=p_fosm, warnings=warnings,
0377             fs_mohr_coulomb=fs_mohr, fs_postfire_yield_index=fs_fire, p_failure_m1=p_m1,
0378             debris_volume_m3=volume, rain_15min_proxy_mm=r15, suction_cohesion_kpa=suction,
0379             cryo_perched_head_m=perched, ice_friction_factor=ice_friction,
0380             equivalent_water_mm=rain+1.35*snowmelt,
0381             effective_burn_severity=burn, cryospheric_eligible=cryo_active, lateral_resistance_kpa=lateral,
0382         )
0383
0384     @staticmethod
0385     def _broadcast_inputs(inputs: PhysicsInputs) -> Dict[str, np.ndarray]:
0386         arrays = {}
0387         for name in PhysicsInputs.__dataclass_fields__:
0388             value = np.asarray(getattr(inputs, name), dtype=np.float64)
0389             if value.ndim > 1:
0390                 raise ValueError("{} must be a scalar or one-dimensional per-cell array".format(name))
0391             arrays[name] = np.atleast_1d(value)
0392         broadcasted = np.broadcast_arrays(*arrays.values())
0393         result = {name: array.copy() for name, array in zip(arrays, broadcasted)}
0394         for name, array in result.items():
0395             if not np.isfinite(array).all():
0396                 raise ValueError("{} contains missing or non-finite values; provide explicit observations/scenarios".format(name))
0397         return result
0398
0399     @staticmethod
0400     def _erfc(values: np.ndarray) -> np.ndarray:
0401         # math.erfc is correctly rounded by the platform libm; vectorize without SciPy dependency.
0402         function = np.frompyfunc(math.erfc, 1, 1)
0403         return function(values).astype(np.float64)
```

## backend/physics/mechanisms.py

SHA-256: `a9cecdca021d92182a5ca9f307c7438eb9d148aff1cf532bea50ca481537c0a3`

```python
0001 """Published equation forms with explicitly identified scenario input proxies.
0002
0003 References and domain restrictions: docs/physics_sources.md. The ice factors and
0004 post-fire yield index are requested scenario parameterizations, not paper equations.
0005 """
0006 import numpy as np
0007
0008 def sigmoid(x):
0009     return 1.0/(1.0+np.exp(-np.clip(x,-700,700)))
0010
0011 def postfire(burn,slope_deg,clay,sand,rain,peak_mmhr,depth):
0012     r15=np.maximum(.25*np.maximum(peak_mmhr,0),.12*np.maximum(rain,0))
0013     prop=burn*sigmoid((slope_deg-21.)/2.5)
0014     dnbr=.85*burn
0015     kf=np.clip(.18+.35*(1-clay)-.15*sand,.15,.45)
0016     link=-3.63+(.41*prop+.67*dnbr+.70*kf)*r15
0017     active=(burn>.05)&(slope_deg>=12.)
0018     probability=np.where(active,sigmoid(link),0.)
0019     # Gartner emergency model coefficients; area and relief here are cell proxies.
0020     area=np.maximum(.105*burn,.01)
0021     relief=np.maximum(depth*50*np.sin(np.radians(slope_deg)),10.)
0022     ln_volume=4.22+.39*np.sqrt(4*r15)+.36*np.log(area)+.13*np.sqrt(relief)
0023     volume=np.where(active,np.exp(np.clip(ln_volume,-700,700)),0.)
0024     yield_index=np.full_like(probability,5.)  # deprecated diagnostic, never mechanical FS
0025     return probability,volume,yield_index,r15
0026
0027 def suction_cohesion(saturation,tan_phi,active_head,alpha_kpa_inv=.1,n=2.):
0028     if alpha_kpa_inv<=0 or n<=1:raise ValueError('van Genuchten alpha >0 and n >1 required')
0029     se=np.clip((saturation-.08)/.92,.02,.999)
0030     stress=-se/alpha_kpa_inv*np.expm1(np.log(se)*n/(1-n))**(1/n)
0031     # Smooth switch reaches zero at Se=.85, avoiding a discontinuous cutoff.
0032     taper=np.clip((.85-se)/.10,0,1);taper=taper*taper*(3-2*taper)
0033     return np.where(active_head,0.,np.clip(-stress*tan_phi,0,8)*taper)
0034
0035 def ice_factors(elevation,soil_temp,snow,cycles,warming,eligible=None):
0036     if eligible is None:
0037         from .alpine_weather import cryospheric_eligibility
0038         eligible=cryospheric_eligibility(snow,soil_temp,elevation,0.)
0039     # Requested joint-temperature proxy, NOT an observed joint temperature.
0040     joint=np.clip(np.asarray(soil_temp)-1.5-.30*np.asarray(cycles)+.50*np.asarray(warming),-8.,-.5)
0041     strength=mamot_joint_strength(joint,200.)
0042     friction_raw=.42-.21*joint
0043     cohesion_raw=strength-200.*friction_raw
0044     cohesion=np.where(eligible,np.clip(cohesion_raw/(53.3-73.5*(-8.)),.14,1.),1.)
0045     friction=np.where(eligible,np.clip(friction_raw/(.42-.21*(-8.)),.40,1.),1.)
0046     return eligible,cohesion,friction
0047
0048 def mamot_joint_strength(temperature_c,normal_stress_kpa):
0049     """Mamot 2018 Eqs.4–6, for verified ice-filled joints in the tested domain."""
0050     t,stress=np.broadcast_arrays(np.asarray(temperature_c,float),np.asarray(normal_stress_kpa,float))
0051     if np.any(~np.isfinite(t)) or np.any(~np.isfinite(stress)) or np.any((t < -8)|(t > -.5)|(stress <100)|(stress>400)):
0052         raise ValueError('Mamot calibration requires -8..-0.5 C and 100..400 kPa, with known ice-filled joints')
0053     cohesion=53.3-73.5*t
0054     friction=.42-.21*t
0055     return cohesion+stress*friction
```

## backend/physics/alpine_weather.py

SHA-256: `9064cefdda6d8ba57b741d2b4274bedafff2dc2bb0f041d22ac6f03c47d04092`

```python
0001 """Baseline-gated cryospheric scenarios; warming alone cannot create snow."""
0002 from dataclasses import replace
0003 import numpy as np
0004
0005 ALPINE_ASSUMPTION = "Lapse-rate and latitude/elevation screen are scenario proxies, not observed permafrost. Snow/cycle floors apply only inside the baseline cryospheric screen."
0006
0007 def cryospheric_eligibility(snow_depth_m, baseline_temperature_c, elevation_m, latitude_deg):
0008     line = np.maximum(1800.0, 4600.0 - 55.0 * np.abs(latitude_deg))
0009     return (np.asarray(snow_depth_m) > .005) | (np.asarray(baseline_temperature_c) <= 2.5) | (np.asarray(elevation_m) >= line)
0010
0011 def apply_alpine_weather(inputs, elevations_m, snowmelt_temp_surge_c=0.0, latitude_deg=None):
0012     elevation = np.asarray(elevations_m, dtype=float)
0013     latitude = inputs.latitude_deg if latitude_deg is None else latitude_deg
0014     lapse = .0065 * (elevation - float(np.median(elevation)))
0015     baseline = np.asarray(inputs.air_temperature_c) - lapse
0016     eligible = cryospheric_eligibility(inputs.snow_depth_m, baseline, elevation, latitude)
0017     snow_floor = .40 * np.clip((elevation - 1800.0) / 1000.0, .2, 1.5)
0018     snow = np.where(eligible, np.maximum(inputs.snow_depth_m, snow_floor), inputs.snow_depth_m)
0019     cycles = np.where(eligible, np.maximum(inputs.freeze_thaw_cycles_7d, 3.), inputs.freeze_thaw_cycles_7d)
0020     return replace(inputs, air_temperature_c=baseline + snowmelt_temp_surge_c,
0021                    soil_temperature_c=np.asarray(inputs.soil_temperature_c)-lapse+.5*snowmelt_temp_surge_c,
0022                    snow_depth_m=snow, freeze_thaw_cycles_7d=cycles, elevation_m=elevation,
0023                    latitude_deg=latitude, warming_delta_c=snowmelt_temp_surge_c,
0024                    baseline_air_temperature_c=baseline, observed_snow_depth_m=inputs.snow_depth_m)
```

## backend/physics/kinematics.py

SHA-256: `741203f6775a82a439a9c22fdb668ac490c3ae6abcb5dddbbcae2167a519aae2`

```python
0001 """Four-branch landslide velocity solver and H3 Voellmy-style routing."""
0002
0003 from __future__ import annotations
0004
0005 import math
0006 from dataclasses import asdict, dataclass, field
0007 from typing import Dict, List, Mapping, Optional, Sequence, Tuple
0008
0009 import h3
0010 import numpy as np
0011
0012 from .multi_causal_engine import PhysicsInputs, PhysicsState
0013
0014 GRAVITY_M_S2 = 9.81
0015
0016
0017 @dataclass
0018 class KinematicsState:
0019     peak_velocity_m_s: np.ndarray
0020     velocity_class: np.ndarray
0021     velocity_class_name: np.ndarray
0022     failure_mode: np.ndarray
0023
0024     def to_dict(self) -> Dict[str, object]:
0025         return {
0026             "peak_velocity_m_s": self.peak_velocity_m_s.tolist(),
0027             "velocity_class": self.velocity_class.tolist(),
0028             "velocity_class_name": self.velocity_class_name.tolist(),
0029             "failure_mode": self.failure_mode.tolist(),
0030         }
0031
0032
0033 @dataclass
0034 class RunoutRoute:
0035     source_cell: str
0036     path: List[str]
0037     arrival_time_min: float
0038     final_energy_head_m: float
0039     intersected_roads: int
0040     intersected_buildings: int
0041     intersected_households: Optional[int]
0042     stopped_at: str
0043
0044     def to_dict(self) -> Dict[str, object]:
0045         return asdict(self)
0046
0047
0048 def solve_kinematics(inputs: PhysicsInputs, state: PhysicsState) -> KinematicsState:
0049     """Apply the ordered rapid-flow, coseismic, slump, and creep branches."""
0050     slope = np.radians(np.clip(state.effective_slope_deg, 0.0, 89.9))
0051     burn = state.effective_burn_severity
0052     rain = np.maximum(np.asarray(inputs.rain_24h_mm, dtype=np.float64), 0.0)
0053     snow_depth = np.maximum(np.asarray(inputs.snow_depth_m, dtype=np.float64), 0.0)
0054     pga = np.maximum(np.asarray(inputs.pga_g, dtype=np.float64), 0.0)
0055     tan_phi = np.tan(np.radians(state.friction_angle_deg))
0056
0057     rapid = (state.fs_dynamic <= 1.0) & (((state.fs_dynamic <= 1.0) & (state.local_saturation >= 0.68)) | ((burn >= 0.35) & (rain >= 25.0)))
0058     coseismic = (
0059         (~rapid) & (state.fs_dynamic <= 1.0)
0060         & (pga > state.critical_acceleration_g)
0061         & (state.newmark_displacement_cm >= 2.0)
0062         & (np.asarray(inputs.slope_deg) >= 28.0)
0063     )
0064     slump = (~rapid) & (~coseismic) & (state.fs_dynamic <= 1.25)
0065     stable = (~rapid) & (~coseismic) & (~slump)
0066
0067     flow_depth = state.regolith_depth_m * np.clip(
0068         state.pore_pressure_head_m / (state.regolith_depth_m * np.cos(slope) ** 2), 0.25, 1.0
0069     )
0070     rapid_speed = np.sqrt(
0071         np.maximum(
0072             0.0,
0073             (300.0 + 150.0 * burn)
0074             * flow_depth
0075             * np.maximum(0.002, np.sin(slope) - 0.18 * np.cos(slope)),
0076         )
0077     )
0078     coseismic_speed = np.sqrt(
0079         2.0
0080         * GRAVITY_M_S2
0081         * 120.0
0082         * np.maximum(
0083             0.005,
0084             np.sin(slope)
0085             + state.kh * np.cos(slope)
0086             - 0.65 * tan_phi * np.cos(slope),
0087         )
0088     )
0089     slump_speed = (
0090         0.08
0091         * np.sin(slope)
0092         * np.power(np.maximum(0.0001, (1.28 / state.fs_dynamic) - 1.0), 2.2)
0093     )
0094     creep_speed = np.zeros_like(state.fs_dynamic)  # stable: no resolved mass movement
0095     speed = np.select([rapid, coseismic, slump, stable], [rapid_speed, coseismic_speed, slump_speed, creep_speed])
0096     velocity_class = np.select(
0097         [
0098             speed >= 5.0,
0099             speed >= 5e-2,
0100             speed >= 5e-4,
0101             speed >= 5e-6,
0102             speed >= 5e-8,
0103             speed >= 5e-10,
0104         ],
0105         [7, 6, 5, 4, 3, 2],
0106         default=1,
0107     ).astype(np.int8)
0108     class_names = np.asarray(
0109         [
0110             "Stable",
0111             "Extremely Slow",
0112             "Very Slow",
0113             "Slow",
0114             "Moderate",
0115             "Rapid",
0116             "Very Rapid",
0117             "Extremely Rapid",
0118         ],
0119         dtype=object,
0120     )[velocity_class]
0121     modes = np.select(
0122         [rapid, coseismic, slump, stable],
0123         ["Rapid fluidized debris flow / avalanche", "Coseismic rock avalanche / block slide", "Cohesive slump / thaw slump / toe-cut wedge", "Stable / sub-perception creep"],
0124         default="Unclassified",
0125     )
0126     return KinematicsState(speed, velocity_class, class_names, modes)
0127
0128
0129 def select_runout_hotspots(state: PhysicsState, probability: Optional[np.ndarray] = None) -> np.ndarray:
0130     """Indices meeting the requested FS or probability hotspot thresholds."""
0131     selected = state.fs_dynamic <= 1.05
0132     if probability is not None:
0133         probability = np.asarray(probability, dtype=np.float64)
0134         if probability.shape != state.fs_dynamic.shape:
0135             raise ValueError("probability must have the same shape as PhysicsState")
0136         selected = selected | (probability >= 0.65)
0137     return np.where(selected)[0]
0138
0139
0140 def route_runout(
0141     source_cell: str,
0142     terrain_cells: Mapping[str, object],
0143     peak_velocity_m_s: float,
0144     mu_b: float,
0145     xi: float,
0146     infrastructure_by_cell: Optional[Mapping[str, Mapping[str, int]]] = None,
0147     delta_s_by_edge: Optional[Mapping[Tuple[str, str], float]] = None,
0148     max_steps: int = 50,
0149     flow_depth_m: float = 0.5,
0150 ) -> RunoutRoute:
0151     """Route positive energy head over lower H3 neighbors.
0152
0153     For each edge, Δs is supplied in ``delta_s_by_edge`` when available; otherwise
0154     it is the current cell's dimensionless sin(slope) factor. The user-specified
0155     Voellmy head change is applied exactly as ΔE = Δz - μb Δx - (v²/(g ξ)) Δs.
0156     Friction and ξ are required inputs because the prompt gives no calibrated values.
0157     """
0158     if source_cell not in terrain_cells:
0159         raise KeyError("source cell is absent from the terrain graph")
0160     if not math.isfinite(mu_b) or not math.isfinite(xi) or mu_b < 0 or xi <= 0 or max_steps < 1:
0161         raise ValueError("mu_b must be nonnegative, xi positive, and max_steps at least 1")
0162     if peak_velocity_m_s < 0 or not math.isfinite(peak_velocity_m_s):
0163         raise ValueError("peak velocity must be finite and nonnegative")
0164
0165     if not math.isfinite(flow_depth_m) or flow_depth_m <= 0:
0166         raise ValueError("Flow depth must be positive and finite")
0167     h_f = max(float(flow_depth_m), 0.5)
0168     energy_head = peak_velocity_m_s**2 / (2.0 * GRAVITY_M_S2)
0169     path = [source_cell]
0170     elapsed_seconds = 0.0
0171     current = source_cell
0172     visited = {source_cell}
0173     infra = infrastructure_by_cell or {}
0174     edge_s = delta_s_by_edge or {}
0175
0176     for _ in range(min(int(max_steps), 50)):
0177         if energy_head <= 0.0:
0178             break
0179         cell = terrain_cells[current]
0180         neighbors = list(getattr(cell, "downslope_neighbors", []) or [])
0181         candidates = []
0182         current_elevation = float(getattr(cell, "elevation_m"))
0183         current_speed = math.sqrt(max(0.0, 2.0 * GRAVITY_M_S2 * energy_head))
0184         for neighbor_id in neighbors:
0185             if neighbor_id in visited or neighbor_id not in terrain_cells:
0186                 continue
0187             neighbor = terrain_cells[neighbor_id]
0188             dz = current_elevation - float(getattr(neighbor, "elevation_m"))
0189             if dz <= 0.0:
0190                 continue
0191             dx = _distance_m(cell, neighbor)
0192             delta_s = math.hypot(dx, dz)
0193             delta_energy = dz - mu_b * dx - (current_speed**2 / (GRAVITY_M_S2 * xi * h_f)) * delta_s
0194             next_head = energy_head + delta_energy
0195             if next_head > 0.0:
0196                 candidates.append((next_head, neighbor_id, dx))
0197         if not candidates:
0198             break
0199         next_head, next_cell, dx = max(candidates, key=lambda candidate: candidate[0])
0200         next_speed = math.sqrt(2.0 * GRAVITY_M_S2 * next_head)
0201         mean_speed = max(1e-4, (current_speed + next_speed) / 2.0)
0202         elapsed_seconds += dx / mean_speed
0203         energy_head = next_head
0204         current = next_cell
0205         visited.add(current)
0206         path.append(current)
0207
0208     road_hits = sum(int(infra.get(cell_id, {}).get("roads", 0)) for cell_id in path)
0209     building_hits = sum(int(infra.get(cell_id, {}).get("buildings", 0)) for cell_id in path)
0210     household_counts = [infra.get(cell_id, {}).get("households") for cell_id in path]
0211     household_hits = (
0212         sum(int(value) for value in household_counts)
0213         if all(value is not None for value in household_counts)
0214         else None
0215     )
0216     return RunoutRoute(
0217         source_cell=source_cell,
0218         path=path,
0219         arrival_time_min=elapsed_seconds / 60.0,
0220         final_energy_head_m=energy_head,
0221         intersected_roads=road_hits,
0222         intersected_buildings=building_hits,
0223         intersected_households=household_hits,
0224         stopped_at=path[-1],
0225     )
0226
0227
0228 def route_hotspots(
0229     terrain_cells: Mapping[str, object],
0230     cell_ids: Sequence[str],
0231     state: PhysicsState,
0232     kinematics: KinematicsState,
0233     mu_b: float,
0234     xi: float,
0235     probability: Optional[np.ndarray] = None,
0236     infrastructure_by_cell: Optional[Mapping[str, Mapping[str, int]]] = None,
0237     max_routes: int = 10,
0238 ) -> List[RunoutRoute]:
0239     """Route up to ``max_routes`` cells meeting FS≤1.05 or failure probability≥0.65."""
0240     if len(cell_ids) != state.cell_count or kinematics.peak_velocity_m_s.size != state.cell_count:
0241         raise ValueError("cell_ids, PhysicsState, and KinematicsState must have matching cell counts")
0242     if max_routes < 1:
0243         return []
0244     hotspots = select_runout_hotspots(state, probability)
0245     if probability is not None:
0246         p = np.asarray(probability, dtype=np.float64)
0247         hotspots = sorted(hotspots, key=lambda idx: (p[idx], -state.fs_dynamic[idx]), reverse=True)
0248     else:
0249         hotspots = sorted(hotspots, key=lambda idx: (state.p_failure_physical[idx], -state.fs_dynamic[idx]), reverse=True)
0250     routes = []
0251     for idx in hotspots[:max_routes]:
0252         cell_id = cell_ids[int(idx)]
0253         if cell_id not in terrain_cells:
0254             continue
0255         routes.append(
0256             route_runout(
0257                 cell_id,
0258                 terrain_cells,
0259                 float(kinematics.peak_velocity_m_s[idx]),
0260                 mu_b,
0261                 xi,
0262                 infrastructure_by_cell=infrastructure_by_cell,
0263             )
0264         )
0265     return routes
0266
0267
0268 def _distance_m(a: object, b: object) -> float:
0269     a_lat, a_lon = float(getattr(a, "latitude")), float(getattr(a, "longitude"))
0270     b_lat, b_lon = float(getattr(b, "latitude")), float(getattr(b, "longitude"))
0271     try:
0272         return float(h3.great_circle_distance((a_lat, a_lon), (b_lat, b_lon), unit="m"))
0273     except TypeError:
0274         lat1, lat2 = math.radians(a_lat), math.radians(b_lat)
0275         dlat, dlon = lat2 - lat1, math.radians(b_lon - a_lon)
0276         hav = math.sin(dlat / 2.0) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2.0) ** 2
0277         return 2.0 * 6371008.8 * math.asin(math.sqrt(hav))
```

## backend/physics/causal_attribution.py

SHA-256: `0971d9e7dc11d71aaaa61023d7663faa57a56e0863f948d3c46d47cf3a3f3f5c`

```python
0001 """Regime-level stress decomposition and location-specific threshold cards."""
0002
0003 from __future__ import annotations
0004
0005 import math
0006 from dataclasses import asdict, dataclass, replace
0007 from typing import Dict, List, Mapping, Optional
0008
0009 import numpy as np
0010
0011 from .multi_causal_engine import GAMMA_W_KN_M3, MultiCausalEngine, PhysicsInputs, PhysicsState
0012
0013 CAUSES = (
0014     "hydro_meteorological",
0015     "seismic_tectonic",
0016     "cryospheric",
0017     "post_wildfire_biotic",
0018     "anthropogenic_fluvial",
0019 )
0020
0021
0022 @dataclass
0023 class CausalBreakdown:
0024     delta_tau_kpa: Dict[str, np.ndarray]
0025     physical_percentages: Dict[str, np.ndarray]
0026     shap_percentages: Optional[Dict[str, np.ndarray]]
0027     combined_percentages: Dict[str, np.ndarray]
0028     absolute_shap_log_odds: Optional[Dict[str, np.ndarray]]
0029     valid: np.ndarray
0030
0031     def to_dict(self) -> Dict[str, object]:
0032         return {
0033             "delta_tau_kpa": {key: value.tolist() for key, value in self.delta_tau_kpa.items()},
0034             "physical_percentages": {key: value.tolist() for key, value in self.physical_percentages.items()},
0035             "shap_percentages": None
0036             if self.shap_percentages is None
0037             else {key: value.tolist() for key, value in self.shap_percentages.items()},
0038             "combined_percentages": {key: value.tolist() for key, value in self.combined_percentages.items()},
0039             "absolute_shap_log_odds": None
0040             if self.absolute_shap_log_odds is None
0041             else {key: value.tolist() for key, value in self.absolute_shap_log_odds.items()},
0042             "valid": self.valid.tolist(),
0043         }
0044
0045
0046 @dataclass
0047 class NumbersThatMatterItem:
0048     metric: str
0049     value: float
0050     unit: str
0051     threshold: str
0052     status: str
0053     causal_group: str
0054     weight_percent: float
0055
0056     def to_dict(self) -> Dict[str, object]:
0057         return asdict(self)
0058
0059
0060 def explain_causes(
0061     inputs: PhysicsInputs,
0062     state: PhysicsState,
0063     shap_values: Optional[np.ndarray] = None,
0064     feature_names: Optional[List[str]] = None,
0065 ) -> CausalBreakdown:
0066     """Return five physical shear reductions and optional absolute TreeSHAP shares.
0067
0068     Physical contributions remain in kPa. They are normalized separately from
0069     dimensionless SHAP log-odds values; the combined percentage is the mean of
0070     the two normalized distributions when both are available.
0071     """
0072     slope = np.radians(np.clip(state.effective_slope_deg, 0.0, 89.9))
0073     ndvi = np.clip(np.asarray(inputs.ndvi_norm, dtype=np.float64), 0.0, 1.0)
0074     burn = state.effective_burn_severity
0075     tan_phi = np.tan(np.radians(state.friction_angle_deg))
0076     root_cohesion_unburned = 12.0 * ndvi**1.35
0077     slope_cos2 = np.cos(slope) ** 2
0078     slope_deg = np.asarray(inputs.slope_deg, dtype=np.float64)
0079     twi = np.maximum(np.asarray(inputs.twi, dtype=np.float64), 0.01)
0080     road_distance = np.broadcast_to(np.asarray(inputs.road_cut_distance_m, dtype=np.float64), (state.cell_count,))
0081     river_distance = np.broadcast_to(np.asarray(inputs.river_distance_m, dtype=np.float64), (state.cell_count,))
0082     field_capacity_saturation = np.clip(state.theta_33 / np.maximum(state.theta_s, 1e-6), 0.0, 0.95)
0083     field_capacity_head = state.regolith_depth_m * field_capacity_saturation * slope_cos2
0084     excess_pore_pressure_head = np.maximum(0.0, state.pore_pressure_head_m - field_capacity_head)
0085     snowmelt_activation = np.clip(state.snowmelt_mm_day / 60.0, 0.0, 0.75)
0086     cryospheric_degradation = np.maximum(1.0 - state.thaw_factor, snowmelt_activation)
0087     if np.any(burn > 0.0):
0088         unburned_state = MultiCausalEngine().compute(replace(inputs, burn_severity=np.zeros_like(burn)))
0089         unburned_excess_head = np.maximum(
0090             0.0, unburned_state.pore_pressure_head_m * state.regolith_depth_m / unburned_state.regolith_depth_m - field_capacity_head
0091         )
0092         fire_amplified_excess_head = np.maximum(
0093             0.0, excess_pore_pressure_head - unburned_excess_head
0094         )
0095         fire_runoff_stress = fire_amplified_excess_head * GAMMA_W_KN_M3 * tan_phi
0096     else:
0097         fire_runoff_stress = np.zeros(state.cell_count, dtype=np.float64)
0098     road_undercut = 0.35 * np.exp(-road_distance / 60.0) * (slope_deg > 16.0)
0099     # River distance is estimated from high-TWI DEM cells, not a mapped stream;
0100     # discount that proxy contribution while retaining measured road-cut effects.
0101     fluvial_proxy_undercut = 0.15 * np.exp(-river_distance / 80.0) * (twi / 15.0)
0102     fire_drag_stress = state.wet_unit_weight_kn_m3 * state.regolith_depth_m * (
0103         np.sin(slope)*np.cos(slope) + state.kh*slope_cos2
0104     ) * state.toe_undercut_multiplier * (state.fire_driving_multiplier - 1.0)
0105     fire_cohesion_loss = (state.soil_cohesion_kpa-state.effective_soil_cohesion_kpa + root_cohesion_unburned-state.root_cohesion_kpa) * state.thaw_factor
0106     stress = {
0107         "hydro_meteorological": np.maximum(
0108             0.0, excess_pore_pressure_head * GAMMA_W_KN_M3 * tan_phi - fire_runoff_stress
0109         ),
0110         "seismic_tectonic": np.maximum(
0111             0.0,
0112             state.wet_unit_weight_kn_m3
0113             * state.regolith_depth_m
0114             * state.kh
0115             * np.cos(slope) ** 2,
0116         ),
0117         "cryospheric": np.maximum(
0118             0.0,
0119             (state.effective_soil_cohesion_kpa + state.root_cohesion_kpa)
0120             * cryospheric_degradation,
0121         ),
0122         "post_wildfire_biotic": np.maximum(
0123             0.0, fire_cohesion_loss + fire_runoff_stress + fire_drag_stress + state.p_failure_m1 * state.wet_unit_weight_kn_m3 * state.regolith_depth_m * np.sin(slope)
0124         ),
0125         "anthropogenic_fluvial": np.maximum(
0126             0.0,
0127             state.wet_unit_weight_kn_m3
0128             * state.regolith_depth_m
0129             * np.sin(slope)
0130             * np.cos(slope)
0131             * (road_undercut + 0.5 * fluvial_proxy_undercut),
0132         ),
0133     }
0134     physical = _normalize(stress)
0135     shap_by_cause = None
0136     shap_pct = None
0137     if shap_values is not None:
0138         shap = np.asarray(shap_values, dtype=np.float64)
0139         if shap.ndim != 2 or shap.shape[0] != state.cell_count:
0140             raise ValueError("shap_values must have shape (cell_count, feature_count)")
0141         if feature_names is None or len(feature_names) != shap.shape[1]:
0142             raise ValueError("feature_names matching shap_values columns are required")
0143         index = {name: idx for idx, name in enumerate(feature_names)}
0144         feature_groups = {
0145             "hydro_meteorological": ("psi_head_m", "soil_saturation", "rain_24h_mm"),
0146             "seismic_tectonic": ("pga_g", "newmark_dn_cm"),
0147             "cryospheric": ("snowmelt_mm_day", "thaw_degradation"),
0148             "post_wildfire_biotic": ("burn_severity",),
0149             "anthropogenic_fluvial": ("toe_undercut_mult",),
0150         }
0151         shap_by_cause = {}
0152         for cause, names in feature_groups.items():
0153             columns = [index[name] for name in names if name in index]
0154             shap_by_cause[cause] = (
0155                 np.abs(shap[:, columns]).sum(axis=1)
0156                 if columns
0157                 else np.zeros(state.cell_count, dtype=np.float64)
0158             )
0159         shap_pct = _normalize(shap_by_cause)
0160
0161     if shap_pct is None:
0162         combined = physical
0163     else:
0164         physical_total = np.sum(np.stack(list(stress.values()), axis=0), axis=0)
0165         shap_total = np.sum(np.stack(list(shap_by_cause.values()), axis=0), axis=0)
0166         combined = {}
0167         for cause in CAUSES:
0168             combined[cause] = np.where(
0169                 (physical_total > 0) & (shap_total > 0),
0170                 0.5 * (physical[cause] + shap_pct[cause]),
0171                 np.where(shap_total > 0, shap_pct[cause], physical[cause]),
0172             )
0173     valid = np.sum(np.stack(list(stress.values()), axis=0), axis=0) > 1e-8
0174     if shap_by_cause is not None:
0175         valid = valid | (np.sum(np.stack(list(shap_by_cause.values()), axis=0), axis=0) > 0)
0176     return CausalBreakdown(stress, physical, shap_pct, combined, shap_by_cause, valid)
0177
0178
0179 def select_numbers_that_matter(
0180     index: int,
0181     inputs: PhysicsInputs,
0182     state: PhysicsState,
0183     breakdown: CausalBreakdown,
0184     probability_failure: Optional[np.ndarray] = None,
0185 ) -> List[NumbersThatMatterItem]:
0186     """Return exactly six complete threshold metrics weighted by active cause share."""
0187     if index < 0 or index >= state.cell_count:
0188         raise IndexError("cell index outside PhysicsState")
0189     slope = float(_value(inputs.slope_deg, index))
0190     pga = float(_value(inputs.pga_g, index))
0191     saturation = float(state.local_saturation[index])
0192     rain = float(_value(inputs.rain_24h_mm, index))
0193     snow = float(state.snowmelt_mm_day[index])
0194     burn = float(state.effective_burn_severity[index])
0195     freeze = float(_value(inputs.freeze_thaw_cycles_7d, index))
0196     prob = float(probability_failure[index]) if probability_failure is not None else float(state.p_failure_physical[index])
0197     cause_weights = {cause: float(breakdown.combined_percentages[cause][index]) for cause in CAUSES}
0198     candidates = [
0199         _item("dynamic_factor_of_safety", state.fs_dynamic[index], "ratio", "failure threshold 1.00; rapid-deformation watch 1.25", "critical" if state.fs_dynamic[index] <= 1.0 else ("watch" if state.fs_dynamic[index] <= 1.25 else "below threshold"), "hydro_meteorological", cause_weights["hydro_meteorological"]),
0200         _item(
0201             "blended_failure_probability" if probability_failure is not None else "physical_failure_probability",
0202             prob * 100.0,
0203             "%",
0204             "hotspot threshold 65%",
0205             "critical" if prob >= 0.65 else ("watch" if prob >= 0.20 else "below threshold"),
0206             "hydro_meteorological",
0207             cause_weights["hydro_meteorological"],
0208         ),
0209         _item("pore_pressure_head", state.pore_pressure_head_m[index], "m", "full saturation head {:.3f} m".format(state.regolith_depth_m[index] * math.cos(math.radians(slope)) ** 2), "critical" if state.pore_pressure_head_m[index] >= 0.8 * state.regolith_depth_m[index] * math.cos(math.radians(slope)) ** 2 else "below threshold", "hydro_meteorological", cause_weights["hydro_meteorological"]),
0210         _item("24_hour_rainfall", rain, "mm", "25 mm when burn severity ≥0.35 activates rapid-flow branch", "critical" if burn >= 0.35 and rain >= 25.0 else "below threshold", "hydro_meteorological", cause_weights["hydro_meteorological"]),
0211         _item("local_soil_saturation", saturation * 100.0, "%", "68% rapid-flow branch threshold", "critical" if saturation >= 0.68 else "below threshold", "hydro_meteorological", cause_weights["hydro_meteorological"]),
0212         _item("pga", pga, "g", "critical acceleration {:.4f} g".format(state.critical_acceleration_g[index]), "critical" if pga > state.critical_acceleration_g[index] else "below threshold", "seismic_tectonic", cause_weights["seismic_tectonic"]),
0213         _item("newmark_displacement", state.newmark_displacement_cm[index], "cm", "2 cm coseismic velocity branch threshold", "critical" if state.newmark_displacement_cm[index] >= 2.0 else "below threshold", "seismic_tectonic", cause_weights["seismic_tectonic"]),
0214         _item("snowmelt_flux", snow, "mm/day", "positive melt requires snow depth >0.01 m and air temperature >0°C", "active" if snow > 0 else "inactive", "cryospheric", cause_weights["cryospheric"]),
0215         _item("freeze_thaw_cycles_7d", freeze, "cycles", "thaw degradation increases at ≥1 cycle", "active" if freeze >= 1 else "inactive", "cryospheric", cause_weights["cryospheric"]),
0216         _item("burn_severity", burn, "0–1", "0.35 for fire-rain rapid-flow branch", "critical" if burn >= 0.35 else "below threshold", "post_wildfire_biotic", cause_weights["post_wildfire_biotic"]),
0217         _item("root_cohesion_after_fire", state.root_cohesion_kpa[index], "kPa", "computed from NDVI and burn severity", "derived", "post_wildfire_biotic", cause_weights["post_wildfire_biotic"]),
0218         _item("toe_undercut_multiplier", state.toe_undercut_multiplier[index], "ratio", "1.0 baseline; >1 adds undercut demand", "active" if state.toe_undercut_multiplier[index] > 1.0 else "inactive", "anthropogenic_fluvial", cause_weights["anthropogenic_fluvial"]),
0219         _item("slope_angle", slope, "degrees", "16° road-cut; 28° coseismic branch thresholds", "critical" if slope >= 28 else ("watch" if slope > 16 else "below threshold"), "anthropogenic_fluvial", cause_weights["anthropogenic_fluvial"]),
0220     ]
0221     core=[replace(item,causal_group='core_stability') for item in candidates[:2]]
0222     ranked=sorted(candidates[2:],key=lambda item:(item.weight_percent,item.status in ('CRITICAL','WATCH')),reverse=True)
0223     chosen=[];seen=set()
0224     for item in ranked:
0225         if item.causal_group not in seen:
0226             chosen.append(item);seen.add(item.causal_group)
0227         if len(chosen)==4:break
0228     return core+chosen
0229
0230
0231 def _normalize(values: Mapping[str, np.ndarray]) -> Dict[str, np.ndarray]:
0232     clean = {key: np.maximum(0.0, np.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0)) for key,value in values.items()}
0233     total = np.sum(np.stack(list(clean.values()), axis=0), axis=0)
0234     # A neutral prior preserves the UI contract when no stress is active; valid=False signals no attribution.
0235     return {key: np.divide(value*100.0,total,out=np.full_like(value,100.0/len(clean),dtype=float),where=total>1e-8) for key,value in clean.items()}
0236
0237
0238 def _value(value: object, index: int) -> float:
0239     array = np.atleast_1d(np.asarray(value, dtype=np.float64))
0240     return float(array[0] if array.size == 1 else array[index])
0241
0242
0243 def _item(metric: str, value: object, unit: str, threshold: str, status: str, cause: str, weight: float) -> NumbersThatMatterItem:
0244     status = {"critical":"CRITICAL", "watch":"WATCH", "active":"WATCH"}.get(status, "NORMAL")
0245     numeric = float(value)
0246     return NumbersThatMatterItem(metric, numeric if math.isfinite(numeric) else 0.0, unit, threshold, status, cause, float(weight))
```

## backend/ml/global_xgb.py

SHA-256: `b3b381ccffa917d2023b17afca50d348f64df449c782313989741d1089c28add`

```python
0001 """Monotonic XGBoost with spatially grouped isotonic calibration."""
0002
0003 from __future__ import annotations
0004
0005 from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple
0006
0007 import numpy as np
0008
0009 from backend.realtime.device import detect_xgboost_device
0010
0011 FEATURE_CONSTRAINTS: Tuple[Tuple[str, int], ...] = (
0012     ("fs_dynamic", -1),
0013     ("p_f_fosm", 1),
0014     ("psi_head_m", 1),
0015     ("newmark_dn_cm", 1),
0016     ("slope_max_deg", 1),
0017     ("twi", 1),
0018     ("soil_saturation", 1),
0019     ("rain_24h_mm", 1),
0020     ("snowmelt_mm_day", 1),
0021     ("pga_g", 1),
0022     ("thaw_degradation", 1),
0023     ("burn_severity", 1),
0024     ("toe_undercut_mult", 1),
0025     ("profile_curvature", 0),
0026 )
0027 FEATURE_NAMES = tuple(name for name, _ in FEATURE_CONSTRAINTS)
0028 MONOTONE_VECTOR = tuple(direction for _, direction in FEATURE_CONSTRAINTS)
0029
0030
0031 class TrainingDataError(ValueError):
0032     """Raised when labeled spatial training data is insufficient or malformed."""
0033
0034
0035 def spatial_group_splits(y: Sequence[int], groups: Sequence[Any], n_splits: int) -> List[Tuple[np.ndarray, np.ndarray]]:
0036     """Build deterministic grouped folds with a positive-bearing block in each holdout."""
0037     labels = np.asarray(y, dtype=np.int8)
0038     group_values = np.asarray(groups)
0039     simulated = np.asarray([str(group).startswith("synthetic:") for group in group_values])
0040     unique = np.unique(group_values[~simulated])
0041     positive_groups = [group for group in unique if np.any((group_values == group) & (labels == 1))]
0042     if n_splits < 2 or len(positive_groups) < n_splits:
0043         raise TrainingDataError("{} positive-bearing spatial groups cannot support {} holdout folds".format(len(positive_groups), n_splits))
0044     assignments: Dict[Any, int] = {}
0045     for index, group in enumerate(positive_groups):
0046         key = group.item() if hasattr(group, "item") else group
0047         assignments[key] = index % n_splits
0048     negative_only = [group for group in unique if group not in positive_groups]
0049     for index, group in enumerate(negative_only):
0050         key = group.item() if hasattr(group, "item") else group
0051         assignments[key] = index % n_splits
0052     splits = []
0053     for fold in range(n_splits):
0054         held_out = [group for group in unique if assignments[group.item() if hasattr(group, "item") else group] == fold]
0055         test = np.flatnonzero(np.isin(group_values, held_out))
0056         train = np.flatnonzero(~np.isin(group_values, held_out))
0057         splits.append((train, test))
0058     return splits
0059
0060
0061 class GlobalMonotonicXGB:
0062     """XGB binary classifier; construction and training require optional ML packages."""
0063
0064     def __init__(self, n_estimators: int = 500, max_depth: int = 6, learning_rate: float = 0.04, min_child_weight: float = 1.0) -> None:
0065         self.n_estimators = n_estimators
0066         self.max_depth = max_depth
0067         self.learning_rate = learning_rate
0068         self.min_child_weight = min_child_weight
0069         self.device = detect_xgboost_device()
0070         self.model: Optional[Any] = None
0071         self.feature_names_in_ = FEATURE_NAMES
0072         self.conformal_ = None
0073
0074     def fit(self, X: np.ndarray, y: np.ndarray, spatial_groups: Sequence[Any], n_splits: int = 5) -> "GlobalMonotonicXGB":
0075         matrix, labels = _validate_xy(X, y)
0076         groups = np.asarray(spatial_groups)
0077         if groups.ndim != 1 or groups.size != labels.size:
0078             raise TrainingDataError("spatial_groups must provide one coarse-region ID per row")
0079         unique_groups = np.unique(groups)
0080         if unique_groups.size < n_splits:
0081             raise TrainingDataError(
0082                 "{} spatial blocks are available; at least {} are required for grouped isotonic calibration".format(
0083                     unique_groups.size, n_splits
0084                 )
0085             )
0086         try:
0087             from sklearn.calibration import CalibratedClassifierCV
0088         except ImportError as exc:
0089             raise RuntimeError("Install requirements-ml.txt to train the CUDA/CPU XGBoost model") from exc
0090
0091         splits = spatial_group_splits(labels, groups, n_splits)
0092         for train_idx, test_idx in splits:
0093             if np.unique(labels[train_idx]).size < 2 or np.unique(labels[test_idx]).size < 2:
0094                 raise TrainingDataError(
0095                     "each spatial calibration fold must contain both event and non-event labels; revise the blocks or data"
0096                 )
0097         estimator = make_base_estimator(
0098             n_estimators=self.n_estimators,
0099             max_depth=self.max_depth,
0100             learning_rate=self.learning_rate,
0101             min_child_weight=getattr(self, "min_child_weight", 1.0),
0102             device=self.device,
0103         )
0104         self.model = CalibratedClassifierCV(estimator=estimator, method="isotonic", cv=splits, ensemble=True)
0105         weights = np.asarray([0.25 if str(g).startswith("synthetic:") else 1.0 for g in groups])
0106         self.model.fit(matrix, labels, sample_weight=weights)
0107         return self
0108
0109     def predict_proba(self, X: np.ndarray) -> np.ndarray:
0110         model = self._require_fitted()
0111         matrix = _validate_X(X)
0112         return np.asarray(model.predict_proba(matrix)[:, 1], dtype=np.float64)
0113
0114     def calibrate_conformal(self, probabilities, labels, groups, training_groups, alpha=.10, scales=None):
0115         """Ensemble-scaled cell residuals; spatial dependence prevents a coverage guarantee."""
0116         if not np.isfinite(alpha) or not 0 < alpha < 1:
0117             raise ValueError('Conformal alpha must be between zero and one')
0118         p=np.asarray(probabilities,float); y=np.asarray(labels,float); g=np.asarray(groups).astype(str)
0119         if p.ndim!=1 or p.shape!=y.shape or p.shape!=g.shape or not len(p):
0120             raise ValueError('Conformal vectors must be nonempty and matching')
0121         if not np.isfinite(p).all() or np.any((p<0)|(p>1)) or not np.isin(y,[0,1]).all():
0122             raise ValueError('Conformal probabilities and binary labels are invalid')
0123         if set(g)&set(map(str,training_groups)):
0124             raise TrainingDataError('Conformal blocks overlap training/isotonic folds')
0125         scale=np.ones_like(p) if scales is None else np.maximum(np.asarray(scales,float),1e-6)
0126         if scale.shape!=p.shape or not np.isfinite(scale).all():raise ValueError('Invalid ensemble scales')
0127         scores=np.abs(y-p)/scale
0128         rank=int(np.ceil((len(scores)+1)*(1-alpha)))
0129         q=float(np.sort(scores)[rank-1]) if rank<=len(scores) else 1.0
0130         self.conformal_={'q':q,'alpha':alpha,'blocks':len(np.unique(g)),'rows':len(y),
0131                          'target':'binary_event_label_not_true_probability', 'scale':'ensemble_std' if scales is not None else 'unit',
0132                          'assumption':'exchangeable calibration/test cells; spatial dependence and global shift invalidate nominal coverage guarantees'}
0133         return self.conformal_
0134
0135     def ensemble_std(self, features):
0136         x=_validate_X(features)
0137         predictions=[fold.predict_proba(x)[:,1] for fold in self._require_fitted().calibrated_classifiers_]
0138         return np.std(predictions,axis=0)
0139
0140     def conformal_intervals(self, probabilities, features=None):
0141         p=np.asarray(probabilities,float)
0142         if not np.isfinite(p).all() or np.any((p<0)|(p>1)):
0143             raise ValueError('Invalid conformal prediction score')
0144         calibration=getattr(self,'conformal_',None)
0145         q=calibration['q'] if calibration else 1.0
0146         if calibration and calibration.get('scale')=='ensemble_std' and features is not None:
0147             q=q*np.maximum(self.ensemble_std(features),1e-6)
0148         lower=np.maximum(0.,p-q); upper=np.minimum(1.,p+q)
0149         width=upper-lower
0150         classes=np.char.add(np.where(p>=.5,'HIGH_SUSCEPTIBILITY_','LOW_SUSCEPTIBILITY_'),
0151                             np.where(width>=.25,'HIGH_UNCERTAINTY','LOW_UNCERTAINTY'))
0152         return np.column_stack((lower,upper)),width,classes
0153
0154     def shap_log_odds_contributions(self, X: np.ndarray, absolute: bool = True) -> np.ndarray:
0155         """Average absolute TreeSHAP contributions from calibrated fold models.
0156
0157         XGBoost's ``pred_contribs`` are exact TreeSHAP values in raw-margin
0158         (log-odds) space. The bias column is intentionally omitted.
0159         """
0160         model = self._require_fitted()
0161         matrix = _validate_X(X)
0162         try:
0163             import xgboost as xgb
0164         except ImportError as exc:
0165             raise RuntimeError("Install requirements-ml.txt for TreeSHAP extraction") from exc
0166         contributions = []
0167         for calibrated in getattr(model, "calibrated_classifiers_", []):
0168             estimator = getattr(calibrated, "estimator", None)
0169             if estimator is None:
0170                 estimator = getattr(calibrated, "base_estimator", None)
0171             if estimator is None or not hasattr(estimator, "get_booster"):
0172                 continue
0173             # Training uses NumPy matrices without column metadata; retain that same
0174             # positional feature contract for booster contribution predictions.
0175             dmatrix = xgb.DMatrix(matrix)
0176             raw_contrib = estimator.get_booster().predict(dmatrix, pred_contribs=True)
0177             contributions.append(np.asarray(raw_contrib[:, :-1], dtype=np.float64))
0178         if not contributions:
0179             raise RuntimeError("calibrated model does not expose XGBoost fold estimators for TreeSHAP")
0180         return np.mean(np.stack([np.abs(value) if absolute else value for value in contributions], axis=0), axis=0)
0181
0182     def feature_importance(self) -> Dict[str, float]:
0183         model = self._require_fitted()
0184         importances = []
0185         for calibrated in getattr(model, "calibrated_classifiers_", []):
0186             estimator = getattr(calibrated, "estimator", None)
0187             if estimator is not None and hasattr(estimator, "feature_importances_"):
0188                 importances.append(np.asarray(estimator.feature_importances_, dtype=np.float64))
0189         if not importances:
0190             return {name: 0.0 for name in FEATURE_NAMES}
0191         mean = np.mean(np.stack(importances, axis=0), axis=0)
0192         return {name: float(value) for name, value in zip(FEATURE_NAMES, mean)}
0193
0194     def _require_fitted(self) -> Any:
0195         if self.model is None:
0196             raise RuntimeError("GlobalMonotonicXGB has no fitted model; labeled training data is required")
0197         return self.model
0198
0199
0200 def mechanical_stability_gate(slope_deg: np.ndarray, fs_dynamic: np.ndarray) -> np.ndarray:
0201     """User-specified smooth mechanical screen; not a calibrated probability."""
0202     slope, fs = np.broadcast_arrays(np.asarray(slope_deg, dtype=float), np.asarray(fs_dynamic, dtype=float))
0203     if not np.isfinite(slope).all() or not np.isfinite(fs).all():
0204         raise ValueError("slope and factor of safety must be finite")
0205     return (1.0 / (1.0 + np.exp(np.clip(-(slope - 8.5) / 0.65, -700, 700)))) * (1.0 / (1.0 + np.exp(np.clip((fs - 2.15) / 0.18, -700, 700))))
0206
0207
0208 def blend_failure_probability(p_xgb: np.ndarray, p_physical: np.ndarray,
0209                               slope_deg: np.ndarray, fs_dynamic: np.ndarray) -> np.ndarray:
0210     ml, physical, slope, fs = np.broadcast_arrays(
0211         np.asarray(p_xgb, dtype=float), np.asarray(p_physical, dtype=float),
0212         np.asarray(slope_deg, dtype=float), np.asarray(fs_dynamic, dtype=float))
0213     if not np.isfinite(ml).all() or not np.isfinite(physical).all():
0214         raise ValueError("probabilities must be finite")
0215     base_blend = .65*ml + .35*physical
0216     yield_weight = .35 + .50/(1.+np.exp(np.clip((fs-1.05)/.06,-700,700)))
0217     yield_blend = (1.-yield_weight)*ml + yield_weight*physical
0218     combined = np.where(fs<=1.05,np.maximum(base_blend,yield_blend),base_blend)
0219     return mechanical_stability_gate(slope,fs)*np.clip(combined,0.,1.)
0220
0221
0222 def make_base_estimator(
0223     n_estimators: int = 500,
0224     max_depth: int = 6,
0225     learning_rate: float = 0.04,
0226     device: Optional[str] = None,
0227     min_child_weight: float = 1.0,
0228 ) -> Any:
0229     """Construct the constrained XGB estimator, preferring CUDA when available."""
0230     try:
0231         from xgboost import XGBClassifier
0232     except ImportError as exc:
0233         raise RuntimeError("Install requirements-ml.txt for XGBoost") from exc
0234     selected_device = device or detect_xgboost_device()
0235     return XGBClassifier(
0236         n_estimators=n_estimators,
0237         max_depth=max_depth,
0238         learning_rate=learning_rate,
0239         min_child_weight=min_child_weight,
0240         objective="binary:logistic",
0241         eval_metric="logloss",
0242         tree_method="hist",
0243         device=selected_device,
0244         monotone_constraints=MONOTONE_VECTOR,
0245         n_jobs=-1,
0246     )
0247
0248
0249 def feature_matrix(features: Mapping[str, Any]) -> np.ndarray:
0250     missing = [name for name in FEATURE_NAMES if name not in features]
0251     if missing:
0252         raise ValueError("missing model features: {}".format(", ".join(missing)))
0253     arrays = [np.atleast_1d(np.asarray(features[name], dtype=np.float64)) for name in FEATURE_NAMES]
0254     matrix = np.column_stack(np.broadcast_arrays(*arrays))
0255     return _validate_X(matrix)
0256
0257
0258 def _validate_xy(X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
0259     matrix = _validate_X(X)
0260     labels = np.asarray(y, dtype=np.int8)
0261     if labels.ndim != 1 or labels.size != matrix.shape[0]:
0262         raise TrainingDataError("labels must be a one-dimensional vector matching X rows")
0263     if not np.isin(labels, (0, 1)).all() or np.unique(labels).size != 2:
0264         raise TrainingDataError("binary training labels must contain both 0 (no failure) and 1 (failure)")
0265     return matrix, labels
0266
0267
0268 def _validate_X(X: np.ndarray) -> np.ndarray:
0269     matrix = np.asarray(X, dtype=np.float32)
0270     if matrix.ndim != 2 or matrix.shape[1] != len(FEATURE_NAMES):
0271         raise ValueError("X must have shape (n_cells, 14) in FEATURE_NAMES order")
0272     if not np.isfinite(matrix).all():
0273         raise ValueError("X contains missing or non-finite values; imputation must be explicit and trained")
0274     return matrix
```

## backend/ml/conformal_calibration.py

SHA-256: `eb7de4457965b4e7c5ad4b2300378368f27d1a9323e015665eed3ae148b58eb4`

```python
0001 """Fit uncertainty on separately reserved, unfiltered spatial blocks."""
0002 import numpy as np
0003 from .global_xgb import FEATURE_NAMES, blend_failure_probability
0004
0005 def fit_reserved_conformal(model, data_directory, training_groups):
0006     import pyarrow.parquet as pq
0007     path=data_directory/'conformal_calibration.parquet'
0008     rows=pq.read_table(path).to_pylist()
0009     x=np.array([[r[k] for k in FEATURE_NAMES] for r in rows],dtype=np.float32)
0010     p=blend_failure_probability(model.predict_proba(x),np.array([r['p_failure_physical'] for r in rows]),x[:,4],x[:,0])
0011     return model.calibrate_conformal(p,[r['label'] for r in rows],[r['region_block'] for r in rows],training_groups,scales=model.ensemble_std(x))
0012
0013 def heldout_conformal_diagnostics(model, rows):
0014     """Descriptive coverage on untouched holdout labels, never used to choose q."""
0015     x=np.array([[r[k] for k in FEATURE_NAMES] for r in rows],dtype=np.float32)
0016     p=blend_failure_probability(model.predict_proba(x),np.array([r['p_failure_physical'] for r in rows]),x[:,4],x[:,0])
0017     bounds,width,_=model.conformal_intervals(p,x)
0018     y=np.array([r['label'] for r in rows]);g=np.array([r['region_block'] for r in rows])
0019     covered=(bounds[:,0]<=y)&(y<=bounds[:,1])
0020     return {'target':'binary event labels, not the latent probability',
0021             'rows':len(rows),'blocks':len(set(g)),
0022             'row_coverage':float(np.mean(covered)),
0023             'whole_block_coverage':float(np.mean([np.all(covered[g==v]) for v in set(g)])),
0024             'mean_interval_width':float(np.mean(width)),
0025             'q':model.conformal_['q'],
0026             'empty_binary_prediction_sets':int(np.sum((bounds[:,0]>0)&(bounds[:,1]<1))),
0027             'scope':'Descriptive on fixed source-location proxies; not a global coverage guarantee'}
```

## backend/physics/cascading.py

SHA-256: `c4614ae442243200f17927ce59c31d7a0942f7e2ef058c14755ae368479c0786`

```python
0001 """DBI in consistent SI units; unresolved dam geometry never receives invented values."""
0002 import math
0003 import h3
0004
0005 def larsen_volume(area_m2):
0006     if not math.isfinite(area_m2) or area_m2<=0:raise ValueError('Positive mapped/scenario area required')
0007     # Larsen et al. 2010, Fig. 2a: log10(alpha)=-.836, gamma=1.332, all materials.
0008     return 10.**(-.836) * area_m2**1.332
0009
0010 def dimensionless_blockage_index(catchment_area_m2, dam_height_m, deposited_volume_m3):
0011     values=(catchment_area_m2,dam_height_m,deposited_volume_m3)
0012     if any(not math.isfinite(v) or v<=0 for v in values):raise ValueError('Positive finite SI geometry required')
0013     return math.log10(catchment_area_m2*dam_height_m/deposited_volume_m3)
0014
0015 def cascading_screen(source_cell, route_path, terrain_cells, dam_geometry=None):
0016     channel=next((h for h in route_path[1:] if terrain_cells[h].twi>=14.5),None)
0017     result={'cascading_dbi':None,'impoundment_status':'NO_MODELED_CHANNEL_INTERSECTION',
0018             'channel_h3_index':channel,'dam_height_m':None,'dam_volume_m3':None,'catchment_area_m2':None}
0019     if channel is None:return result
0020     area=h3.cell_area(source_cell,unit='m^2')
0021     result.update(impoundment_status='POTENTIAL_CHANNEL_INTERSECTION_GEOMETRY_UNRESOLVED',
0022                   source_area_scenario_m2=area,landslide_volume_scenario_m3=larsen_volume(area),
0023                   volume_quality='Full H3 source-cell failure scenario; not mapped scar or deposited dam volume',
0024                   catchment_area_in_view_m2=float(terrain_cells[channel].contributing_area_m2),
0025                   catchment_quality='Viewport DEM accumulation; full upstream catchment not established')
0026     if dam_geometry is None:return result
0027     a=dam_geometry['catchment_area_m2'];h=dam_geometry['dam_height_m'];v=dam_geometry['deposited_volume_m3']
0028     dbi=dimensionless_blockage_index(a,h,v)
0029     result.update(cascading_dbi=dbi,dam_height_m=h,dam_volume_m3=v,catchment_area_m2=a,
0030                   impoundment_status=('EMPIRICAL_STABLE_DAM_DOMAIN' if dbi<2.75 else
0031                                       'EMPIRICAL_UNSTABLE_DAM_DOMAIN' if dbi>3.08 else 'EMPIRICAL_TRANSITION_DOMAIN'),
0032                   geometry_source=dam_geometry.get('source','explicit scenario geometry'),
0033                   dbi_quality='Ermini-Casagli empirical domain; not evidence that a dam has formed')
0034     return result
```

## backend/llm/qwen_agent.py

SHA-256: `46547f28447c81beaa9495766cc6aef902a7d027d5eb68f888333bd23a80144b`

```python
0001 """Three-tier Ollama tool agent with a deterministic live-scan fallback."""
0002
0003 from __future__ import annotations
0004
0005 import json
0006 import math
0007 import re
0008 import time
0009 from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple
0010
0011 import httpx
0012
0013 from backend.api.scan_service import GlobalScanService
0014
0015
0016 TOOL_SCHEMAS: List[Dict[str, Any]] = [
0017     {
0018         "type": "function",
0019         "function": {
0020             "name": "scan_global_location",
0021             "description": "Run a live global landslide scan with the requested scenario perturbations.",
0022             "parameters": {
0023                 "type": "object",
0024                 "properties": {
0025                     "location_query": {"type": "string"},
0026                     "rain_delta_mm": {"type": "number", "default": 0.0},
0027                     "storm_duration_hr": {"type": "number", "default": 24.0},
0028                     "pga_override_g": {"type": "number", "default": 0.0},
0029                     "snowmelt_temp_surge_c": {"type": "number", "default": 0.0},
0030                     "burn_severity_override": {"type": "number", "default": 0.0},
0031                 },
0032                 "required": ["location_query"],
0033             },
0034         },
0035     },
0036     {
0037         "type": "function",
0038         "function": {
0039             "name": "inspect_hexagon_causal_physics",
0040             "description": "Inspect the active scan's physics and causal breakdown for one H3 cell.",
0041             "parameters": {"type": "object", "properties": {"h3_index": {"type": "string"}}, "required": ["h3_index"]},
0042         },
0043     },
0044     {
0045         "type": "function",
0046         "function": {
0047             "name": "trace_runout_and_evacuation",
0048             "description": "Trace a provisional Voellmy runout path and list mapped exposure intersections.",
0049             "parameters": {"type": "object", "properties": {"h3_index": {"type": "string"}}, "required": ["h3_index"]},
0050         },
0051     },
0052 ]
0053
0054 _NUMBER = r"[-+]?\d+(?:\.\d+)?"
0055 _NUMERIC_TOKEN = re.compile(r"(?<![\w])" + _NUMBER + r"(?:[eE][-+]?\d+)?")
0056 _XML_TAG = re.compile(r"<\s*/?\s*(?:ui_action|tool_call)\b[^>]*>", re.IGNORECASE)
0057 _ANY_TAG = re.compile(r"<\s*/?[A-Za-z][^>]*>")
0058
0059
0060 def parse_scenario_request(
0061     message: str, current_query: str = "", sliders: Optional[Mapping[str, Any]] = None
0062 ) -> Dict[str, Any]:
0063     """Extract only explicit perturbations; Mw is never converted into PGA."""
0064     text = str(message or "")
0065     lowered = text.lower()
0066     values: Dict[str, Any] = {
0067         "rain_delta_mm": 0.0,
0068         "storm_duration_hr": 24.0,
0069         "pga_override_g": 0.0,
0070         "snowmelt_temp_surge_c": 0.0,
0071         "burn_severity_override": 0.0,
0072     }
0073     explicit = set()
0074     rain_matches = re.findall("(" + _NUMBER + r")\s*mm\b", lowered)
0075     if rain_matches:
0076         values["rain_delta_mm"] = max(0.0, float(rain_matches[-1]))
0077         explicit.add("rain_delta_mm")
0078     pga_matches = re.findall("(" + _NUMBER + r")\s*g\b", lowered)
0079     if pga_matches:
0080         values["pga_override_g"] = max(0.0, float(pga_matches[-1]))
0081         explicit.add("pga_override_g")
0082     temp_matches = re.findall("(" + _NUMBER + r")\s*(?:°\s*c|degrees?\s*celsius|celsius)\b", lowered)
0083     if temp_matches:
0084         values["snowmelt_temp_surge_c"] = max(0.0, float(temp_matches[-1]))
0085         explicit.add("snowmelt_temp_surge_c")
0086     burn_matches = re.findall("(" + _NUMBER + r")\s*%", lowered)
0087     if burn_matches:
0088         values["burn_severity_override"] = min(1.0, max(0.0, float(burn_matches[-1]) / 100.0))
0089         explicit.add("burn_severity_override")
0090     duration_matches = re.findall("(" + _NUMBER + r")\s*(?:h|hr|hrs|hours?)\b", lowered)
0091     if duration_matches:
0092         values["storm_duration_hr"] = max(0.01, float(duration_matches[-1]))
0093         explicit.add("storm_duration_hr")
0094
0095     for key in values:
0096         if sliders and key in sliders:
0097             try:
0098                 slider_value = float(sliders[key])
0099                 if not math.isfinite(slider_value):
0100                     raise ValueError("slider must be finite")
0101                 if key == "burn_severity_override":
0102                     slider_value = min(1.0, max(0.0, slider_value))
0103                 elif key == "storm_duration_hr":
0104                     slider_value = max(0.01, slider_value)
0105                 else:
0106                     slider_value = max(0.0, slider_value)
0107                 if not math.isfinite(slider_value):
0108                     raise ValueError("slider must be finite")
0109                 if key not in explicit:
0110                     values[key] = slider_value
0111                     explicit.add(key)
0112             except (TypeError, ValueError):
0113                 continue
0114
0115     query = str(current_query or "").strip()
0116     from backend.api.scan_service import known_location_coordinates
0117     candidate = _location_from_message(text)
0118     if candidate and (not query or known_location_coordinates(candidate) is not None or re.search(r"\bscan\b",text,re.I)):
0119         query = candidate
0120     values["location_query"] = query or "Chooralmala, Wayanad"
0121     values["magnitude_only"] = bool(re.search(r"\bM\s*w\s*" + _NUMBER, text, re.IGNORECASE)) and not pga_matches
0122     values["_explicit_fields"] = sorted(explicit)
0123     return values
0124
0125
0126 def _location_from_message(message: str) -> str:
0127     text = message.strip()
0128     text = re.sub(r"\bM\s*w\s*" + _NUMBER, " ", text, flags=re.IGNORECASE)
0129     text = re.sub(_NUMBER + r"\s*(?:mm\b|g\b|°\s*c\b|degrees?\s*celsius\b|celsius\b|%|M\s*w\s*\d+(?:\.\d+)?)", " ", text, flags=re.IGNORECASE)
0130     text = re.sub(r"\b(?:scan|check|assess|estimate|show|tell\s+me|landslide|risk|hazard|with|under|after|for|at|in|around|near|the|and|burn|scar|rain|rainfall|earthquake|snowmelt|warming|temperature|please)\b", " ", text, flags=re.IGNORECASE)
0131     text = re.sub(r"[+,:;()\[\]{}]", " ", text)
0132     text = re.sub(r"\s+", " ", text).strip(" .-/")
0133     return text
0134
0135
0136 class TerraShieldQwenAgent:
0137     """Select a local Qwen model when available and ground every briefing in tools."""
0138
0139     def __init__(self, scan_service: Optional[GlobalScanService] = None, ollama_url: Optional[str] = None) -> None:
0140         self.scan_service = scan_service or GlobalScanService()
0141         self.ollama_url = (ollama_url or "http://localhost:11434").rstrip("/")
0142
0143     async def scan_global_location(
0144         self,
0145         location_query: str,
0146         rain_delta_mm: float = 0.0,
0147         storm_duration_hr: float = 24.0,
0148         pga_override_g: float = 0.0,
0149         snowmelt_temp_surge_c: float = 0.0,
0150         burn_severity_override: float = 0.0,
0151     ) -> Dict[str, Any]:
0152         return await self.scan_service.scan(
0153             location_query=location_query,
0154             rain_delta_mm=rain_delta_mm,
0155             storm_duration_hr=storm_duration_hr,
0156             pga_override_g=pga_override_g,
0157             snowmelt_temp_surge_c=snowmelt_temp_surge_c,
0158             burn_severity_override=burn_severity_override,
0159         )
0160
0161     async def _run_scan(self, arguments: Mapping[str, Any], agency_mode: str) -> Dict[str, Any]:
0162         return await self.scan_service.scan(**dict(arguments), agency_mode=agency_mode)
0163
0164     def inspect_hexagon_causal_physics(self, h3_index: str) -> Dict[str, Any]:
0165         return self.scan_service.inspect_hexagon(h3_index)
0166
0167     def trace_runout_and_evacuation(self, h3_index: str) -> Dict[str, Any]:
0168         return self.scan_service.trace_runout(h3_index)
0169
0170     async def chat(
0171         self,
0172         message: str,
0173         current_query: str = "",
0174         agency_mode: str = "LOCAL_EOC",
0175         sliders: Optional[Mapping[str, Any]] = None,
0176     ) -> Dict[str, Any]:
0177         parsed = parse_scenario_request(message, current_query, sliders)
0178         model_name, tier = await self._available_model()
0179         tool_calls: List[Dict[str, Any]] = []
0180         scan_data: Optional[Dict[str, Any]] = None
0181         candidate = None
0182         if model_name:
0183             try:
0184                 scan_data, candidate = await self._ollama_tool_roundtrip(
0185                     model_name, message, parsed, agency_mode, tool_calls
0186                 )
0187             except Exception:
0188                 # If Ollama disconnects mid-conversation, continue with live deterministic tools.
0189                 model_name, tier = None, 3
0190         if scan_data is None:
0191             args = self._scan_arguments(parsed)
0192             started = time.perf_counter()
0193             scan_data = await self._run_scan(args, agency_mode)
0194             tool_calls.append({
0195                 "tool_name": "scan_global_location",
0196                 "arguments": args,
0197                 "latency_ms": round((time.perf_counter() - started) * 1000.0, 2),
0198             })
0199
0200         clean_markdown, numeric_audit_passed, briefing_guard = self._guard_briefing(candidate, scan_data, agency_mode, bool(parsed.get("magnitude_only")))
0201         location = scan_data.get("location") or {}
0202         ui_action = {
0203             "fly_to": [location.get("longitude"), location.get("latitude")],
0204             "zoom": 13.4,
0205             "sliders": {key: parsed[key] for key in (
0206                 "rain_delta_mm", "storm_duration_hr", "pga_override_g",
0207                 "snowmelt_temp_surge_c", "burn_severity_override",
0208             )},
0209             "agency_mode": agency_mode,
0210         }
0211         return {
0212             "clean_markdown": clean_markdown,
0213             "tool_calls": tool_calls,
0214             "ui_action": ui_action,
0215             "updated_scan_data": scan_data,
0216             "numeric_audit_passed": bool(numeric_audit_passed),
0217             "briefing_guard": briefing_guard,
0218             "model_tier": tier,
0219             "model_name": model_name,
0220         }
0221
0222     async def _available_model(self) -> Tuple[Optional[str], int]:
0223         try:
0224             async with httpx.AsyncClient(timeout=1.25) as client:
0225                 response = await client.get(self.ollama_url + "/api/tags")
0226                 response.raise_for_status()
0227                 names = [str(item.get("name", "")) for item in response.json().get("models", [])]
0228             custom = next((name for name in names if name.split(":", 1)[0].lower() == "terra-shield-qwen"), None)
0229             if custom:
0230                 return custom, 1
0231             qwen = next((name for name in names if "qwen2.5" in name.lower()), None)
0232             if qwen:
0233                 return qwen, 2
0234         except Exception:
0235             pass
0236         return None, 3
0237
0238     async def _ollama_tool_roundtrip(
0239         self,
0240         model_name: str,
0241         message: str,
0242         parsed: Mapping[str, Any],
0243         agency_mode: str,
0244         tool_calls: List[Dict[str, Any]],
0245     ) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
0246         messages: List[Dict[str, Any]] = [
0247             {"role": "system", "content": "Use the supplied live tools for all measurements. Do not invent coordinates, probabilities, causes, or runout. Call scan_global_location for the requested location and perturbations. Then call inspect/trace only for an H3 index returned by the scan. Write a grounded final answer using only returned telemetry."},
0248             {"role": "user", "content": message or "Scan the current location."},
0249         ]
0250         scan_data: Optional[Dict[str, Any]] = None
0251         async with httpx.AsyncClient(timeout=45.0) as client:
0252             for _ in range(4):
0253                 response = await client.post(self.ollama_url + "/api/chat", json={
0254                     "model": model_name, "messages": messages, "tools": TOOL_SCHEMAS, "stream": False,
0255                 })
0256                 response.raise_for_status()
0257                 assistant_message = response.json().get("message") or {}
0258                 messages.append(assistant_message)
0259                 calls = assistant_message.get("tool_calls") or []
0260                 if not calls:
0261                     break
0262                 for call in calls:
0263                     function = call.get("function") or {}
0264                     name = str(function.get("name") or "")
0265                     arguments = function.get("arguments") or {}
0266                     if isinstance(arguments, str):
0267                         try:
0268                             arguments = json.loads(arguments)
0269                         except ValueError:
0270                             arguments = {}
0271                     arguments = dict(arguments) if isinstance(arguments, Mapping) else {}
0272                     if name == "scan_global_location":
0273                         # All scenario values come from explicit user text/sliders or deterministic defaults.
0274                         # A language model cannot silently add a perturbation that the UI does not show.
0275                         arguments = self._scan_arguments(parsed)
0276                     elif name in ("inspect_hexagon_causal_physics", "trace_runout_and_evacuation") and scan_data is None:
0277                         arguments_scan = self._scan_arguments(parsed)
0278                         started = time.perf_counter()
0279                         scan_data = await self._run_scan(arguments_scan, agency_mode)
0280                         tool_calls.append({"tool_name": "scan_global_location", "arguments": arguments_scan, "latency_ms": round((time.perf_counter() - started) * 1000.0, 2)})
0281                     started = time.perf_counter()
0282                     try:
0283                         result = await self._dispatch(name, arguments, agency_mode)
0284                         if name == "scan_global_location":
0285                             scan_data = result
0286                         tool_result = result
0287                     except Exception as exc:
0288                         tool_result = {"error": "{}: {}".format(type(exc).__name__, exc)}
0289                     tool_calls.append({"tool_name": name, "arguments": arguments, "latency_ms": round((time.perf_counter() - started) * 1000.0, 2)})
0290                     messages.append({"role": "tool", "name": name, "content": json.dumps(tool_result, ensure_ascii=False, default=str, allow_nan=False)})
0291         return scan_data, assistant_message.get("content", "")
0292
0293     async def _dispatch(self, name: str, arguments: Mapping[str, Any], agency_mode: str) -> Dict[str, Any]:
0294         if name == "scan_global_location":
0295             return await self._run_scan(self._clean_scan_arguments(arguments), agency_mode)
0296         if name == "inspect_hexagon_causal_physics":
0297             return self.inspect_hexagon_causal_physics(str(arguments.get("h3_index", "")))
0298         if name == "trace_runout_and_evacuation":
0299             return self.trace_runout_and_evacuation(str(arguments.get("h3_index", "")))
0300         raise ValueError("unknown tool: {}".format(name))
0301
0302     @staticmethod
0303     def _clean_scan_arguments(arguments: Mapping[str, Any]) -> Dict[str, Any]:
0304         allowed = {
0305             "location_query", "rain_delta_mm", "storm_duration_hr", "pga_override_g",
0306             "snowmelt_temp_surge_c", "burn_severity_override",
0307         }
0308         return {key: value for key, value in arguments.items() if key in allowed}
0309
0310     @staticmethod
0311     def _scan_arguments(parsed: Mapping[str, Any]) -> Dict[str, Any]:
0312         return {
0313             "location_query": parsed.get("location_query") or "Chooralmala, Wayanad",
0314             "rain_delta_mm": float(parsed.get("rain_delta_mm", 0.0)),
0315             "storm_duration_hr": float(parsed.get("storm_duration_hr", 24.0)),
0316             "pga_override_g": float(parsed.get("pga_override_g", 0.0)),
0317             "snowmelt_temp_surge_c": float(parsed.get("snowmelt_temp_surge_c", 0.0)),
0318             "burn_severity_override": float(parsed.get("burn_severity_override", 0.0)),
0319         }
0320
0321     @staticmethod
0322     def _briefing(scan: Mapping[str, Any], agency_mode: str, magnitude_only: bool = False) -> str:
0323         location = scan.get("location") or {}
0324         summary = scan.get("summary") or {}
0325         metrics = summary.get("meteorological_metrics") or {}
0326         hotspot = summary.get("kinematics_top_hotspot") or {}
0327         name = location.get("display_name") or location.get("query") or "Requested location"
0328         lines = ["## TERRA-SHIELD scan: {}".format(name), "", "- **Alert level:** {}".format(summary.get("wmo_alert_level", "UNAVAILABLE"))]
0329         if agency_mode.upper() == "WMO_MET":
0330             lines.extend([
0331                 "- **Daily rainfall scenario:** {} mm; storm duration: {} h".format(
0332                     TerraShieldQwenAgent._fmt(metrics.get("rain_24h_mm_scenario")),
0333                     TerraShieldQwenAgent._fmt(metrics.get("storm_duration_hr")),
0334                 ),
0335                 "- **Scenario PGA:** {} g; air temperature: {} °C; burn severity: {}".format(
0336                     TerraShieldQwenAgent._fmt(metrics.get("pga_g_scenario")),
0337                     TerraShieldQwenAgent._fmt(metrics.get("air_temperature_c_scenario")),
0338                     TerraShieldQwenAgent._fmt(metrics.get("burn_severity_override")),
0339                 ),
0340                 "- **P95 failure probability:** {}; **P05 dynamic factor of safety:** {}".format(
0341                     TerraShieldQwenAgent._fmt(summary.get("max_p_failure")),
0342                     TerraShieldQwenAgent._fmt(summary.get("min_fs_dynamic")),
0343                 ),
0344                 "- **Critical hotspots:** {}".format(summary.get("critical_hotspots_count", 0)),
0345             ])
0346         else:
0347             lines.extend([
0348                 "- **P95 failure probability:** {}; **P05 dynamic factor of safety:** {}".format(
0349                     TerraShieldQwenAgent._fmt(summary.get("max_p_failure")),
0350                     TerraShieldQwenAgent._fmt(summary.get("min_fs_dynamic")),
0351                 ),
0352                 "- **Top exposed hotspot:** H3 `{}`; slope {}°; peak velocity {} m/s; {}.".format(
0353                     hotspot.get("h3_index", "unavailable"),
0354                     TerraShieldQwenAgent._fmt(hotspot.get("slope_deg")),
0355                     TerraShieldQwenAgent._fmt(hotspot.get("peak_velocity_ms")),
0356                     hotspot.get("failure_mode", "mode unavailable"),
0357                 ),
0358                 "- **Hotspot cause:** {}".format(hotspot.get("dominant_cause", summary.get("dominant_cause", "unavailable"))),
0359                 "- **Critical hotspots:** {}; mapped households exposed: {}".format(
0360                     summary.get("critical_hotspots_count", 0),
0361                     summary.get("total_households_exposed") if summary.get("total_households_exposed") is not None else "unavailable",
0362                 ),
0363             ])
0364         cause = hotspot.get("causal_breakdown_pct") or summary.get("causal_breakdown_pct") or {}
0365         if cause:
0366             top_causes = sorted(cause.items(), key=lambda item: float(item[1]), reverse=True)[:3]
0367             lines.append("- **Causal breakdown:** {}".format(", ".join("{} {}%".format(key, TerraShieldQwenAgent._fmt(value, 1)) for key, value in top_causes)))
0368         numbers = hotspot.get("numbers_that_matter") or summary.get("numbers_that_matter") or []
0369         if numbers:
0370             lines.append("- **Numbers that matter:** {}".format("; ".join(
0371                 "{} {} {}".format(item.get("metric"), TerraShieldQwenAgent._fmt(item.get("value")), item.get("unit", ""))
0372                 for item in numbers[:3]
0373             )))
0374         lines.extend([
0375             "",
0376             "Probability is an exploratory model/physics blend when a model is available; source gaps and fallback inputs are listed in the scan payload.",
0377             "Runout routes use display defaults that are not calibrated for operational evacuation decisions.",
0378         ])
0379         if magnitude_only:
0380             lines.append("Mw was noted, but no site PGA was estimated because magnitude alone does not specify shaking at the location.")
0381         return "\n".join(lines)
0382
0383     @staticmethod
0384     def _fmt(value: Any, digits: int = 3) -> str:
0385         if value is None:
0386             return "unavailable"
0387         try:
0388             return format(float(value), ".{}f".format(digits))
0389         except (TypeError, ValueError):
0390             return str(value)
0391
0392     @staticmethod
0393     def _guard_briefing(candidate, telemetry, agency_mode, magnitude_only=False):
0394         cleaned=re.sub(r'<(?:ui_action|tool_call)\b[^>]*>.*?</(?:ui_action|tool_call)>','',candidate or '',flags=re.S|re.I).strip()
0395         if cleaned and TerraShieldQwenAgent._numeric_parity_audit(cleaned,telemetry):
0396             return cleaned,True,'VERIFIED_LLM_PROSE'
0397         canonical=TerraShieldQwenAgent._briefing(telemetry,agency_mode,magnitude_only)
0398         canonical=_ANY_TAG.sub('',_XML_TAG.sub('',canonical)).strip()
0399         if not TerraShieldQwenAgent._numeric_parity_audit(canonical,telemetry):
0400             return 'Verified numeric briefing unavailable. Inspect structured telemetry.',False,'WITHHELD_INVALID_TEMPLATE'
0401         return canonical,True,'VERIFIED_TEMPLATE' if not cleaned else 'REPLACED_UNVERIFIED_PROSE'
0402
0403     @staticmethod
0404     def _numeric_parity_audit(markdown: str, telemetry: Mapping[str, Any]) -> bool:
0405         available=[];strings=[]
0406         def collect(value):
0407             if isinstance(value,bool) or value is None:return
0408             if isinstance(value,(int,float)) and math.isfinite(float(value)):
0409                 available.append(float(value))
0410                 if 0<=value<=1:available.append(float(value)*100.)
0411             elif isinstance(value,str) and value:strings.append(value)
0412             elif isinstance(value,Mapping):
0413                 for child in value.values():collect(child)
0414             elif isinstance(value,(list,tuple)):
0415                 for child in value:collect(child)
0416         collect(telemetry)
0417         # Verbatim identifiers/location strings are not numeric measurements.
0418         for value in sorted(strings,key=len,reverse=True):
0419             if any(c.isdigit() for c in value):markdown=markdown.replace(value,'SOURCE_IDENTIFIER')
0420         markdown=re.sub(r'(?m)^\s*\d+[\.\)]\s+','',markdown)
0421         markdown=re.sub(r'\[T\+\d+m\]|Class\s+[1-7]\b|\b(?:24h|48h|7d|30m)\b|\b90%','',markdown)
0422         for token in _NUMERIC_TOKEN.findall(markdown):
0423             precision=len(token.partition('.')[2]) if '.' in token else 0
0424             if not any(abs(float(token)-value)<=max(10**(-precision),.05) for value in available):return False
0425         return True
```
