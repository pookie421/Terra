"""Parallel live-data ingestion for a global TERRA-SHIELD scan."""

from __future__ import annotations

import asyncio
import math
import time
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timedelta, timezone
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import httpx
import numpy as np

from .cache import AsyncTTLCache, DEFAULT_CACHE
from .dem_processor import DEMProcessor, TerrainScan
from .natural_events import NaturalEvent, fetch_eonet_events
from .satellite_processor import SatelliteCell, SatelliteProcessor
from .soilgrids import estimate_pedotransfer, fetch_soilgrids_wcs

OPEN_METEO_FORECAST = "https://api.open-meteo.com/v1/forecast"
OPEN_METEO_GEOCODING = "https://geocoding-api.open-meteo.com/v1/search"
NASA_POWER = "https://power.larc.nasa.gov/api/temporal/daily/point"
USGS_EVENTS = "https://earthquake.usgs.gov/fdsnws/event/1/query"
OVERPASS = "https://overpass-api.de/api/interpreter"
OVERPASS_FALLBACKS = (
    OVERPASS,
    "https://overpass.private.coffee/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
    "https://lz4.overpass-api.de/api/interpreter",
)
SOILGRIDS = "https://rest.isric.org/soilgrids/v2.0/properties/query"

Location = Union[str, Tuple[float, float]]


@dataclass
class SourceStatus:
    available: bool
    provider: str
    fetched_at_utc: str
    error: Optional[str] = None


@dataclass
class LocationInfo:
    query: str
    latitude: float
    longitude: float
    display_name: Optional[str] = None
    country: Optional[str] = None


@dataclass
class WeatherSnapshot:
    observed_at_utc: Optional[str]
    rain_24h_mm: Optional[float]
    peak_rain_3h_mm: Optional[float]
    forecast_rain_next_24h_mm: Optional[float]
    forecast_rain_24_48h_mm: Optional[float]
    ari_7d_mm: Optional[float]
    ari_days_available: int
    soil_moisture_0_100cm_m3m3: Optional[float]
    soil_moisture_layers_m3m3: Dict[str, Optional[float]]
    snow_depth_m: Optional[float]
    air_temperature_c: Optional[float]
    soil_temperature_0_7cm_c: Optional[float]
    soil_temperature_28_100cm_c: Optional[float]
    freeze_thaw_cycles_7d: Optional[int]


@dataclass
class HydrologySnapshot:
    observed_date: Optional[str]
    nasa_gwetroot_pct: Optional[float]
    nasa_gwetprof_pct: Optional[float]
    nasa_precipitation_mm_day: Optional[float]
    open_meteo_soil_moisture_m3m3: Optional[float]
    soil_porosity_estimate: Optional[float]
    soil_saturation_pct: Optional[float]
    blended_relative_wetness_pct: Optional[float]
    source_mix: List[str]
    soil_properties: Dict[str, Any]
    soil_property_source: str = "unknown"
    soil_property_quality: str = "unknown"


@dataclass
class EarthquakeEvent:
    event_id: str
    magnitude: float
    magnitude_type: Optional[str]
    latitude: float
    longitude: float
    depth_km: float
    distance_km: float
    pga_g: float
    time_utc: Optional[str]
    place: Optional[str]


@dataclass
class SeismicSnapshot:
    event_count: int
    max_pga_g: Optional[float]
    events: List[EarthquakeEvent]


@dataclass
class InfrastructureSummary:
    total_features: int
    road_count: int
    road_cut_count: int
    waterway_count: int
    quarry_count: int
    building_count: int
    by_h3_cell: Dict[str, Dict[str, int]]
    query_bbox: Tuple[float, float, float, float]


@dataclass
class GlobalScan:
    location: LocationInfo
    radius_km: float
    generated_at_utc: str
    terrain: Optional[TerrainScan]
    weather: Optional[WeatherSnapshot]
    hydrology: Optional[HydrologySnapshot]
    seismic: Optional[SeismicSnapshot]
    infrastructure: Optional[InfrastructureSummary]
    eonet_events: List[Dict[str, Any]]
    satellite_cells: Dict[str, Dict[str, Any]]
    sources: Dict[str, SourceStatus]
    elapsed_seconds: float
    coolr_nearby: Dict[str, Any] = field(default_factory=lambda: {"count":0,"quality":"not_queried"})

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class GlobalIngestor:
    """Query public global datasets concurrently, retaining source-level failures."""

    def __init__(
        self,
        cache: Optional[AsyncTTLCache] = None,
        dem_processor: Optional[DEMProcessor] = None,
        provider_timeout_s: float = 1.25,
        user_agent: str = "TERRA-SHIELD/0.1 (research prototype; contact: terra-shield)",
    ) -> None:
        self.cache = cache or DEFAULT_CACHE
        self.dem_processor = dem_processor or DEMProcessor(cache=self.cache)
        self.satellite_processor = SatelliteProcessor(cache=self.cache)
        self.provider_timeout_s = max(0.5, provider_timeout_s)
        self.user_agent = user_agent
        self._client: Optional[httpx.AsyncClient] = None

    async def __aenter__(self) -> "GlobalIngestor":
        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(self.provider_timeout_s),
            limits=httpx.Limits(max_connections=32, max_keepalive_connections=16),
            headers={"User-Agent": self.user_agent, "Accept": "application/json,image/png,*/*"},
        )
        return self

    async def __aexit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def scan(
        self,
        location: Location,
        radius_km: float = 3.5,
        burn_override: Optional[Mapping[str, float]] = None,
    ) -> GlobalScan:
        started = time.perf_counter()
        if radius_km <= 0 or radius_km > 100:
            raise ValueError("radius_km must be in (0, 100]")
        if self._client is None:
            async with httpx.AsyncClient(
                timeout=httpx.Timeout(self.provider_timeout_s),
                limits=httpx.Limits(max_connections=32, max_keepalive_connections=16),
                headers={"User-Agent": self.user_agent, "Accept": "application/json,image/png,*/*"},
            ) as client:
                return await self._scan_with_client(location, radius_km, client, started, burn_override)
        return await self._scan_with_client(location, radius_km, self._client, started, burn_override)

    async def _scan_with_client(
        self, location: Location, radius_km: float, client: httpx.AsyncClient, started: float,
        burn_override: Optional[Mapping[str, float]] = None,
    ) -> GlobalScan:
        location_info, geocode_status = await self._resolve_location(location, client)
        latitude, longitude = location_info.latitude, location_info.longitude
        south, west, north, east = self._bbox(latitude, longitude, radius_km)

        from .coolr import nearby_count
        source_names = ["terrarium_dem", "open_meteo", "nasa_power", "soilgrids", "usgs", "overpass", "eonet_wildfires", "eonet_landslides", "coolr"]
        results = await asyncio.gather(
            self._capture("terrarium_dem", "AWS Terrarium", self.dem_processor.process(latitude, longitude, radius_km, client)),
            self._capture("open_meteo", "Open-Meteo", self._get_weather(latitude, longitude, client)),
            self._capture("nasa_power", "NASA POWER", self._get_power(latitude, longitude, client)),
            self._capture("soilgrids", "ISRIC SoilGrids", self._get_soilgrids(latitude, longitude, client)),
            self._capture("usgs", "USGS Earthquake Catalog", self._get_seismic(latitude, longitude, client)),
            self._capture("overpass", "OpenStreetMap Overpass", self._get_infrastructure(south, west, north, east, client)),
            self._capture("eonet_wildfires", "NASA EONET wildfires", fetch_eonet_events(client, "wildfires", max(10.0, self.provider_timeout_s), self.cache)),
            self._capture("eonet_landslides", "NASA EONET landslides", fetch_eonet_events(client, "landslides", max(6.0, self.provider_timeout_s), self.cache)),
            self._capture("coolr", "NASA COOLR", nearby_count(client,latitude,longitude,self.cache)),
            return_exceptions=False,
        )
        payloads: Dict[str, Any] = {}
        statuses: Dict[str, SourceStatus] = {"geocoding": geocode_status}
        for name, (payload, status) in zip(source_names, results):
            payloads[name] = payload
            statuses[name] = status

        event_values = []
        for source in ("eonet_wildfires", "eonet_landslides"):
            event_values.extend(payloads.get(source) or [])
        event_map: Dict[str, NaturalEvent] = {}
        for event in event_values:
            event_map["{}:{}".format(event.event_id, event.date_utc)] = event
        satellite_cells: Dict[str, SatelliteCell] = {}
        if payloads.get("terrarium_dem"):
            try:
                satellite_cells = await self.satellite_processor.process_cells(
                    {cell_id: (cell.latitude, cell.longitude) for cell_id, cell in payloads["terrarium_dem"].cells.items()},
                    event_map,
                    client,
                    burn_override=burn_override,
                )
                statuses["satellite"] = SourceStatus(bool(satellite_cells), "Esri World Imagery RGB z13", datetime.now(timezone.utc).isoformat(), None if satellite_cells else "No imagery cells were decoded")
            except Exception as exc:
                statuses["satellite"] = SourceStatus(False, "Esri World Imagery RGB z13", datetime.now(timezone.utc).isoformat(), "{}: {}".format(type(exc).__name__, exc))
        else:
            statuses["satellite"] = SourceStatus(False, "Esri World Imagery RGB z13", datetime.now(timezone.utc).isoformat(), "Terrain cells or recent EONET events unavailable")

        weather = payloads["open_meteo"]
        power = payloads["nasa_power"]
        soilgrids = payloads["soilgrids"]
        terrain = payloads["terrarium_dem"]
        hydrology = self._combine_hydrology(weather, power, soilgrids, terrain)
        coolr_count = payloads['coolr']
        if coolr_count is None:
            from .coolr import local_catalog_distances, URL
            local=local_catalog_distances(latitude,longitude)
            coolr_count={**local,'count':local['cached_count_50km'],'quality':'partial_local_catalog_fallback_50km','source_url':URL}
        if not coolr_count['quality'].startswith('NASA'):
            statuses['coolr'].available=False
            statuses['coolr'].error=coolr_count.get('live_error_type','provider_deadline')
        return GlobalScan(
            location=location_info,
            radius_km=radius_km,
            generated_at_utc=datetime.now(timezone.utc).isoformat(),
            terrain=terrain,
            weather=weather,
            hydrology=hydrology,
            seismic=payloads["usgs"],
            infrastructure=payloads["overpass"],
            coolr_nearby=coolr_count,
            eonet_events=[event.to_dict() for event in event_values],
            satellite_cells={key: asdict(value) for key, value in satellite_cells.items()},
            sources=statuses,
            elapsed_seconds=time.perf_counter() - started,
        )

    async def _capture(self, name: str, provider: str, awaitable: Any) -> Tuple[Any, SourceStatus]:
        fetched_at = datetime.now(timezone.utc).isoformat()
        try:
            deadline = 2.0 if name in ('soilgrids','overpass') else self.provider_timeout_s
            payload = await asyncio.wait_for(awaitable, timeout=deadline)
            return payload, SourceStatus(True, provider, fetched_at)
        except Exception as exc:
            return None, SourceStatus(False, provider, fetched_at, "{}: {}".format(type(exc).__name__, exc))

    async def _resolve_location(self, location: Location, client: httpx.AsyncClient) -> Tuple[LocationInfo, SourceStatus]:
        fetched_at = datetime.now(timezone.utc).isoformat()
        if isinstance(location, str):
            query = location.strip()
            if not query:
                raise ValueError("place name cannot be empty")
            key = ("geocode", query.casefold())

            async def load() -> Dict[str, Any]:
                response = await client.get(
                    OPEN_METEO_GEOCODING,
                    params={"name": query, "count": 1, "language": "en", "format": "json"},
                    timeout=self.provider_timeout_s,
                )
                response.raise_for_status()
                return response.json()

            data = await self.cache.get_or_create(key, 30 * 24 * 3600, load)
            results = data.get("results") or []
            if not results:
                raise ValueError("Open-Meteo geocoding found no result for {!r}".format(query))
            result = results[0]
            lat, lon = float(result["latitude"]), float(result["longitude"])
            self._validate_coordinate(lat, lon)
            info = LocationInfo(
                query=query,
                latitude=lat,
                longitude=lon,
                display_name=result.get("name"),
                country=result.get("country"),
            )
            return info, SourceStatus(True, "Open-Meteo Geocoding", fetched_at)
        if isinstance(location, (tuple, list)) and len(location) == 2:
            lat, lon = float(location[0]), float(location[1])
            self._validate_coordinate(lat, lon)
            return LocationInfo("{:.6f}, {:.6f}".format(lat, lon), lat, lon), SourceStatus(
                True, "Provided coordinates", fetched_at
            )
        raise TypeError("location must be a place-name string or (latitude, longitude) pair")

    async def _get_weather(self, latitude: float, longitude: float, client: httpx.AsyncClient) -> WeatherSnapshot:
        params = {
            "latitude": latitude,
            "longitude": longitude,
            "timezone": "UTC",
            "past_days": 7,
            "forecast_days": 3,
            "hourly": ",".join(
                [
                    "precipitation",
                    "rain",
                    "snowfall",
                    "snow_depth",
                    "temperature_2m",
                    "soil_temperature_0_to_7cm",
                    "soil_temperature_28_to_100cm",
                    "soil_moisture_0_to_7cm",
                    "soil_moisture_7_to_28cm",
                    "soil_moisture_28_to_100cm",
                ]
            ),
            "daily": "precipitation_sum",
        }
        data = await self._json_cached("open_meteo", params, 600, client, OPEN_METEO_FORECAST)
        return self._parse_weather(data)

    def _parse_weather(self, data: Mapping[str, Any]) -> WeatherSnapshot:
        hourly = data.get("hourly") or {}
        times = list(hourly.get("time") or [])
        now = datetime.now(timezone.utc)
        datetimes = [self._parse_provider_time(value) for value in times]
        precipitation = self._float_values(hourly.get("precipitation"), len(times))
        rain = self._float_values(hourly.get("rain"), len(times))
        current = [idx for idx, dt in enumerate(datetimes) if dt is not None and dt <= now]
        recent_24 = [idx for idx in current if datetimes[idx] >= now - timedelta(hours=24)]
        rain_24 = self._sum_present(rain[idx] for idx in recent_24)
        peak3 = None
        windows: List[float] = []
        for start in range(max(0, len(recent_24) - 2)):
            indexes = recent_24[start : start + 3]
            if len(indexes) != 3 or any(rain[idx] is None for idx in indexes):
                continue
            if any(datetimes[b] - datetimes[a] != timedelta(hours=1) for a, b in zip(indexes, indexes[1:])):
                continue
            windows.append(sum(float(rain[idx]) for idx in indexes))
        if windows:
            peak3 = max(windows)

        future = [idx for idx, dt in enumerate(datetimes) if dt is not None and dt > now]
        next24 = [idx for idx in future if datetimes[idx] <= now + timedelta(hours=24)]
        next48 = [idx for idx in future if now + timedelta(hours=24) < datetimes[idx] <= now + timedelta(hours=48)]
        forecast24 = self._sum_present(precipitation[idx] for idx in next24)
        forecast48 = self._sum_present(precipitation[idx] for idx in next48)

        daily_precip: Dict[date, float] = {}
        for observed_time, amount in zip(datetimes, precipitation):
            if observed_time is None or amount is None or observed_time > now or observed_time < now - timedelta(days=7):
                continue
            daily_precip[observed_time.date()] = daily_precip.get(observed_time.date(), 0.0) + amount
        ari_values = [daily_precip[day] for day in sorted(daily_precip, reverse=True)[:7]]
        ari = sum(amount * (0.85**index) for index, amount in enumerate(ari_values)) if ari_values else None

        layers: Dict[str, Optional[float]] = {}
        for key in ("soil_moisture_0_to_7cm", "soil_moisture_7_to_28cm", "soil_moisture_28_to_100cm"):
            values = self._float_values(hourly.get(key), len(times))
            latest = next((values[idx] for idx in reversed(current) if values[idx] is not None), None)
            layers[key] = latest
        layer_weights = {
            "soil_moisture_0_to_7cm": 7.0,
            "soil_moisture_7_to_28cm": 21.0,
            "soil_moisture_28_to_100cm": 72.0,
        }
        moist_present = [(layers[key], weight) for key, weight in layer_weights.items() if layers[key] is not None]
        soil_moisture = (
            sum(float(value) * weight for value, weight in moist_present) / sum(weight for _, weight in moist_present)
            if moist_present
            else None
        )

        def latest_hourly(key: str) -> Optional[float]:
            values = self._float_values(hourly.get(key), len(times))
            return next((values[idx] for idx in reversed(current) if values[idx] is not None), None)

        air_temp = latest_hourly("temperature_2m")
        soil_temp_surface = latest_hourly("soil_temperature_0_to_7cm")
        soil_temp_deep = latest_hourly("soil_temperature_28_to_100cm")
        snow_depth = latest_hourly("snow_depth")

        temp_values = self._float_values(hourly.get("soil_temperature_0_to_7cm"), len(times))
        if not any(value is not None for value in temp_values):
            temp_values = self._float_values(hourly.get("temperature_2m"), len(times))
        last_week = [idx for idx in current if datetimes[idx] >= now - timedelta(days=7)]
        cycles = 0
        prior_temp: Optional[float] = None
        for idx in last_week:
            value = temp_values[idx]
            if value is None:
                continue
            if prior_temp is not None and ((prior_temp < 0 <= value) or (prior_temp > 0 >= value)):
                cycles += 1
            prior_temp = value

        observed_idx = current[-1] if current else None
        observed_at = datetimes[observed_idx].isoformat() if observed_idx is not None else None
        return WeatherSnapshot(
            observed_at_utc=observed_at,
            rain_24h_mm=rain_24,
            peak_rain_3h_mm=peak3,
            forecast_rain_next_24h_mm=forecast24,
            forecast_rain_24_48h_mm=forecast48,
            ari_7d_mm=ari,
            ari_days_available=len(ari_values),
            soil_moisture_0_100cm_m3m3=soil_moisture,
            soil_moisture_layers_m3m3=layers,
            snow_depth_m=snow_depth,
            air_temperature_c=air_temp,
            soil_temperature_0_7cm_c=soil_temp_surface,
            soil_temperature_28_100cm_c=soil_temp_deep,
            freeze_thaw_cycles_7d=cycles if last_week else None,
        )

    async def _get_power(self, latitude: float, longitude: float, client: httpx.AsyncClient) -> Dict[str, Any]:
        today = datetime.now(timezone.utc).date()
        params = {
            "parameters": "PRECTOTCORR,GWETROOT,GWETPROF",
            "community": "RE",
            "longitude": round(longitude, 4),
            "latitude": round(latitude, 4),
            "start": (today - timedelta(days=10)).strftime("%Y%m%d"),
            "end": today.strftime("%Y%m%d"),
            "format": "JSON",
        }
        return await self._json_cached("nasa_power", params, 24 * 3600, client, NASA_POWER)

    async def _get_soilgrids(self, latitude: float, longitude: float, client: httpx.AsyncClient) -> Dict[str, Any]:
        params = [
            ("lon", round(longitude, 5)),
            ("lat", round(latitude, 5)),
            ("property", "bdod"),
            ("property", "clay"),
            ("property", "sand"),
            ("property", "soc"),
            ("depth", "0-5cm"),
            ("depth", "5-15cm"),
            ("depth", "15-30cm"),
            ("value", "mean"),
        ]
        key_params = tuple(params)

        async def load() -> Dict[str, Any]:
            try:
                response = await client.get(SOILGRIDS, params=params, timeout=min(1.0, self.provider_timeout_s))
                response.raise_for_status()
                return response.json()
            except Exception as rest_error:
                try:
                    wcs = await fetch_soilgrids_wcs(client, latitude, longitude, timeout_s=min(2.0, self.provider_timeout_s))
                    return {**wcs, "properties": wcs}
                except Exception as wcs_error:
                    raise RuntimeError("SoilGrids REST failed ({}); WCS failed ({})".format(rest_error, wcs_error))

        return await self.cache.get_or_create(("soilgrids", key_params), 30 * 24 * 3600, load)

    async def _get_seismic(self, latitude: float, longitude: float, client: httpx.AsyncClient) -> SeismicSnapshot:
        params = {
            "format": "geojson",
            "latitude": latitude,
            "longitude": longitude,
            "maxradiuskm": 250,
            "minmagnitude": 2.5,
            "orderby": "time",
        }
        data = await self._json_cached("usgs", params, 30, client, USGS_EVENTS)
        events: List[EarthquakeEvent] = []
        for feature in data.get("features") or []:
            props = feature.get("properties") or {}
            geometry = feature.get("geometry") or {}
            coords = geometry.get("coordinates") or []
            if len(coords) < 3 or props.get("mag") is None:
                continue
            lon, lat, depth_km = float(coords[0]), float(coords[1]), float(coords[2])
            epicentral = self._haversine_km(latitude, longitude, lat, lon)
            hypocentral = math.sqrt(epicentral * epicentral + depth_km * depth_km)
            magnitude = float(props["mag"])
            distance_for_log = max(hypocentral, 1.0)
            pga = 10.0 ** (
                -1.02 + 0.249 * magnitude - math.log10(distance_for_log) - 0.00255 * distance_for_log
            )
            event_id = str(feature.get("id", ""))
            event_time = props.get("time")
            event_time_iso = datetime.fromtimestamp(event_time / 1000.0, timezone.utc).isoformat() if event_time else None
            events.append(
                EarthquakeEvent(
                    event_id=event_id,
                    magnitude=magnitude,
                    magnitude_type=props.get("magType"),
                    latitude=lat,
                    longitude=lon,
                    depth_km=depth_km,
                    distance_km=hypocentral,
                    pga_g=pga,
                    time_utc=event_time_iso,
                    place=props.get("place"),
                )
            )
        return SeismicSnapshot(
            event_count=len(events),
            max_pga_g=max((event.pga_g for event in events), default=0.0),
            events=events,
        )

    async def _get_infrastructure(
        self,
        south: float,
        west: float,
        north: float,
        east: float,
        client: httpx.AsyncClient,
    ) -> InfrastructureSummary:
        if west < -180.0:
            boxes = [(south, west + 360.0, north, 180.0), (south, -180.0, north, east)]
        elif east > 180.0:
            boxes = [(south, west, north, 180.0), (south, -180.0, north, east - 360.0)]
        else:
            boxes = [(south, west, north, east)]

        async def query_box(box: Tuple[float, float, float, float]) -> Dict[str, Any]:
            bbox = "{:.5f},{:.5f},{:.5f},{:.5f}".format(*box)
            query = """
[timeout:8][out:json];
(
  way[\"highway\"~\"^(trunk|primary|secondary|tertiary)$\"]({bbox});
  way[\"waterway\"=\"river\"]({bbox});
);
out geom;
""".format(bbox=bbox)
            key = ("overpass",) + tuple(round(value, 3) for value in box)

            async def load() -> Dict[str, Any]:
                last_error: Optional[Exception] = None
                endpoint_timeout = min(self.provider_timeout_s, 2.0)
                for endpoint in OVERPASS_FALLBACKS:
                    try:
                        response = await client.post(
                            endpoint,
                            data={"data": query},
                            headers={"Content-Type": "application/x-www-form-urlencoded"},
                            timeout=endpoint_timeout,
                        )
                        response.raise_for_status()
                        return response.json()
                    except (httpx.ConnectError, httpx.TimeoutException) as exc:
                        last_error = exc
                    except httpx.HTTPStatusError as exc:
                        if exc.response.status_code == 429 or exc.response.status_code >= 500:
                            last_error = exc
                        else:
                            raise
                if last_error is not None:
                    raise last_error
                raise RuntimeError("No Overpass endpoints were configured")

            return await self.cache.get_or_create(key, 6 * 3600, load)

        responses = await asyncio.gather(*(query_box(box) for box in boxes))
        unique_elements: Dict[Tuple[str, int], Dict[str, Any]] = {}
        for payload in responses:
            for element in payload.get("elements") or []:
                unique_elements[(str(element.get("type", "")), int(element.get("id", -1)))] = element
        data = {"elements": list(unique_elements.values())}
        by_cell: Dict[str, Dict[str, int]] = {}
        totals = {"roads": 0, "road_cuts": 0, "waterways": 0, "quarries": 0, "buildings": 0}
        for element in data.get("elements") or []:
            tags = element.get("tags") or {}
            center = element.get("center") or {}
            lat = element.get("lat", center.get("lat"))
            lon = element.get("lon", center.get("lon"))
            if (lat is None or lon is None) and element.get("geometry"):
                coordinates = element.get("geometry") or []
                if coordinates:
                    lat = sum(float(point["lat"]) for point in coordinates) / len(coordinates)
                    lon = sum(float(point["lon"]) for point in coordinates) / len(coordinates)
            if lat is None or lon is None:
                continue
            cell_id = self._coordinate_to_h3(float(lat), float(lon))
            cell_counts = by_cell.setdefault(
                cell_id, {"roads": 0, "road_cuts": 0, "waterways": 0, "quarries": 0, "buildings": 0}
            )
            categories = []
            if tags.get("highway"):
                categories.append("roads")
            if tags.get("cutting", "no") not in ("no", "false", "0", ""):
                categories.append("road_cuts")
            if tags.get("waterway"):
                categories.append("waterways")
            if tags.get("landuse") == "quarry" or tags.get("quarry"):
                categories.append("quarries")
            if tags.get("building"):
                categories.append("buildings")
            for category in set(categories):
                totals[category] += 1
                cell_counts[category] += 1
        return InfrastructureSummary(
            total_features=len(data.get("elements") or []),
            road_count=totals["roads"],
            road_cut_count=totals["road_cuts"],
            waterway_count=totals["waterways"],
            quarry_count=totals["quarries"],
            building_count=totals["buildings"],
            by_h3_cell=by_cell,
            query_bbox=(south, west, north, east),
        )

    async def _json_cached(
        self,
        provider: str,
        params: Mapping[str, Any],
        ttl_seconds: int,
        client: httpx.AsyncClient,
        url: str,
    ) -> Dict[str, Any]:
        key_params = tuple(sorted((key, str(value)) for key, value in params.items()))
        key = (provider, key_params)

        async def load() -> Dict[str, Any]:
            response = await client.get(url, params=params, timeout=self.provider_timeout_s)
            response.raise_for_status()
            return response.json()

        return await self.cache.get_or_create(key, ttl_seconds, load)

    def _combine_hydrology(
        self,
        weather: Optional[WeatherSnapshot],
        power: Optional[Mapping[str, Any]],
        soilgrids: Optional[Mapping[str, Any]],
        terrain: Optional[TerrainScan] = None,
    ) -> Optional[HydrologySnapshot]:
        if weather is None and power is None and soilgrids is None and terrain is None:
            return None
        params = ((power or {}).get("properties") or {}).get("parameter") or {}
        gwetroot, root_date = self._latest_parameter(params.get("GWETROOT") or {})
        gwetprof, _ = self._latest_parameter(params.get("GWETPROF") or {})
        precip, _ = self._latest_parameter(params.get("PRECTOTCORR") or {})
        moisture = weather.soil_moisture_0_100cm_m3m3 if weather else None
        props = self._soil_property_values(soilgrids)
        if any(props.get(key) is None for key in ("bdod_kg_m3", "clay_g_kg", "sand_g_kg", "soc_g_kg")):
            measured_keys = [key for key in ("bdod_kg_m3", "clay_g_kg", "sand_g_kg", "soc_g_kg") if props.get(key) is not None]
            if terrain and terrain.cells:
                cells = list(terrain.cells.values())
                slope = float(np.mean([cell.slope_deg for cell in cells]))
                curvature = float(np.mean([cell.profile_curvature for cell in cells]))
                twi = float(np.mean([cell.twi for cell in cells]))
            else:
                slope, curvature, twi = 15.0, 0.0, 6.0
            moisture_regime = gwetroot if gwetroot is not None else (moisture * 100.0 if moisture is not None else None)
            estimate = estimate_pedotransfer(slope, curvature, twi, moisture_regime)
            for key in ("bdod_kg_m3", "clay_g_kg", "sand_g_kg", "soc_g_kg"):
                if props.get(key) is None:
                    props[key] = estimate[key]
            for key in ("sand_fraction", "clay_fraction", "silt_fraction", "organic_matter_fraction", "bulk_density_g_cm3"):
                props.setdefault(key, estimate[key])
            props["property_sources"] = {
                key: ("ISRIC SoilGrids" if key in measured_keys else "topographic_climatic_pedotransfer_estimate")
                for key in ("bdod_kg_m3", "clay_g_kg", "sand_g_kg", "soc_g_kg")
            }
            props["source"] = "hybrid_soilgrids_and_pedotransfer" if measured_keys else "topographic_climatic_pedotransfer_estimate"
            props["quality"] = "mixed_measured_prediction_and_estimate" if measured_keys else "estimated_not_measured"
        else:
            props["sand_fraction"] = float(props["sand_g_kg"]) / 1000.0
            props["clay_fraction"] = float(props["clay_g_kg"]) / 1000.0
            props["silt_fraction"] = max(0.0, 1.0 - props["sand_fraction"] - props["clay_fraction"])
            props["organic_matter_fraction"] = float(props["soc_g_kg"]) * 1.724 / 1000.0
            props["bulk_density_g_cm3"] = float(props["bdod_kg_m3"]) / 1000.0
            props["source"] = (soilgrids or {}).get("source", "ISRIC SoilGrids")
            props["quality"] = (soilgrids or {}).get("quality", "soilgrids_prediction")
        bulk_density = props.get("bdod_kg_m3")
        porosity = 1.0 - bulk_density / 2650.0 if bulk_density is not None else None
        saturation = None
        if moisture is not None and porosity is not None and porosity > 0:
            saturation = float(np.clip(moisture / porosity * 100.0, 0.0, 100.0))
        sources: List[str] = []
        wetness_values: List[float] = []
        if saturation is not None:
            wetness_values.append(saturation)
            sources.append("Open-Meteo volumetric moisture / SoilGrids bulk-density porosity")
        elif moisture is not None:
            # This is volumetric moisture expressed as percent, not true pore saturation.
            wetness_values.append(float(np.clip(moisture * 100.0, 0.0, 100.0)))
            sources.append("Open-Meteo volumetric moisture (relative proxy; porosity unavailable)")
        if gwetroot is not None:
            wetness_values.append(float(np.clip(gwetroot, 0.0, 100.0)))
            sources.append("NASA POWER GWETROOT")
        blended = sum(wetness_values) / len(wetness_values) if wetness_values else None
        return HydrologySnapshot(
            observed_date=root_date,
            nasa_gwetroot_pct=gwetroot,
            nasa_gwetprof_pct=gwetprof,
            nasa_precipitation_mm_day=precip,
            open_meteo_soil_moisture_m3m3=moisture,
            soil_porosity_estimate=porosity,
            soil_saturation_pct=saturation,
            blended_relative_wetness_pct=blended,
            source_mix=sources,
            soil_properties=props,
            soil_property_source=str(props.get("source", "unknown")),
            soil_property_quality=str(props.get("quality", "unknown")),
        )

    @staticmethod
    def _soil_property_values(data: Optional[Mapping[str, Any]]) -> Dict[str, Optional[float]]:
        values: Dict[str, Optional[float]] = {
            "bdod_kg_m3": None,
            "clay_g_kg": None,
            "sand_g_kg": None,
            "soc_g_kg": None,
        }
        if not data:
            return values
        if all(data.get(key) is not None for key in values):
            return {key: float(data[key]) for key in values}
        if all(data.get(key) is not None for key in values):
            return {key: float(data[key]) for key in values}
        raw_properties = data.get("properties") or []
        if isinstance(raw_properties, Mapping):
            raw_properties = raw_properties.get("layers", raw_properties)
        if isinstance(raw_properties, Mapping):
            properties = []
            for property_name, property_value in raw_properties.items():
                if isinstance(property_value, Mapping):
                    normalized = dict(property_value)
                    normalized.setdefault("name", property_name)
                    properties.append(normalized)
                elif isinstance(property_value, list):
                    properties.extend(
                        dict(layer, name=layer.get("name", property_name))
                        for layer in property_value
                        if isinstance(layer, Mapping)
                    )
        else:
            properties = [prop for prop in raw_properties if isinstance(prop, Mapping)]
        for prop in properties:
            name = prop.get("name")
            key = {
                "bdod": "bdod_kg_m3",
                "clay": "clay_g_kg",
                "sand": "sand_g_kg",
                "soc": "soc_g_kg",
            }.get(name)
            if not key:
                continue
            depth_values = []
            raw_depths = prop.get("depths") or []
            if isinstance(raw_depths, Mapping):
                depths = []
                for depth_label, depth_value in raw_depths.items():
                    if isinstance(depth_value, Mapping):
                        normalized_depth = dict(depth_value)
                        normalized_depth.setdefault("label", depth_label)
                        depths.append(normalized_depth)
            else:
                depths = [depth for depth in raw_depths if isinstance(depth, Mapping)]
            for depth in depths:
                mean = ((depth.get("values") or {}).get("mean"))
                if mean is not None and float(mean) >= 0:
                    thickness = float(str(depth.get("range", {}).get("bottom_depth", "0")).replace("cm", ""))
                    top = float(str(depth.get("range", {}).get("top_depth", "0")).replace("cm", ""))
                    weight = max(1.0, thickness - top)
                    depth_values.append((float(mean), weight))
            if depth_values:
                mean_value = sum(value * weight for value, weight in depth_values) / sum(weight for _, weight in depth_values)
                # SoilGrids bdod uses cg/cm3; 1 cg/cm3 equals 10 kg/m3.
                values[key] = mean_value * 10.0 if key == "bdod_kg_m3" else mean_value
        return values

    @staticmethod
    def _latest_parameter(parameter: Mapping[str, Any]) -> Tuple[Optional[float], Optional[str]]:
        candidates = []
        for day, value in parameter.items():
            try:
                number = float(value)
            except (TypeError, ValueError):
                continue
            if math.isfinite(number) and number > -900:
                candidates.append((str(day), number))
        if not candidates:
            return None, None
        day, value = max(candidates, key=lambda item: item[0])
        return value, day

    @staticmethod
    def _float_values(values: Any, expected_length: int) -> List[Optional[float]]:
        result: List[Optional[float]] = []
        values = list(values or [])
        for raw in values[:expected_length]:
            try:
                value = float(raw)
                result.append(value if math.isfinite(value) else None)
            except (TypeError, ValueError):
                result.append(None)
        if len(result) < expected_length:
            result.extend([None] * (expected_length - len(result)))
        return result

    @staticmethod
    def _sum_present(values: Sequence[Optional[float]]) -> Optional[float]:
        usable = [float(value) for value in values if value is not None]
        return sum(usable) if usable else None

    @staticmethod
    def _parse_provider_time(value: Any) -> Optional[datetime]:
        if not value:
            return None
        try:
            parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=timezone.utc)
            return parsed.astimezone(timezone.utc)
        except ValueError:
            return None

    @staticmethod
    def _bbox(latitude: float, longitude: float, radius_km: float) -> Tuple[float, float, float, float]:
        lat_delta = radius_km / 111.32
        lon_delta = min(180.0, radius_km / (111.32 * max(abs(math.cos(math.radians(latitude))), 0.01)))
        return (
            max(-85.05112878, latitude - lat_delta),
            longitude - lon_delta,
            min(85.05112878, latitude + lat_delta),
            longitude + lon_delta,
        )

    @staticmethod
    def _validate_coordinate(latitude: float, longitude: float) -> None:
        if not (-90.0 <= latitude <= 90.0 and -180.0 <= longitude <= 180.0):
            raise ValueError("latitude/longitude are outside valid global bounds")

    @staticmethod
    def _coordinate_to_h3(latitude: float, longitude: float) -> str:
        fn = getattr(__import__("h3"), "latlng_to_cell", getattr(__import__("h3"), "geo_to_h3", None))
        if fn is None:
            raise RuntimeError("Installed h3-py version has no coordinate-to-cell API")
        return str(fn(latitude, longitude, 9))

    @staticmethod
    def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        earth_radius_km = 6371.0088
        phi1, phi2 = math.radians(lat1), math.radians(lat2)
        dphi, dlambda = phi2 - phi1, math.radians(lon2 - lon1)
        hav = math.sin(dphi / 2.0) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2.0) ** 2
        return 2.0 * earth_radius_km * math.asin(math.sqrt(hav))
