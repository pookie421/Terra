"""NASA EONET event retrieval and geometry-aware distance helpers."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import httpx

from .cache import AsyncTTLCache

EONET_EVENTS = "https://eonet.gsfc.nasa.gov/api/v3/events"


@dataclass
class NaturalEvent:
    event_id: str
    title: str
    category: str
    date_utc: Optional[str]
    geometry_type: str
    coordinates: Any
    source_url: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class NaturalEventsSnapshot:
    events: List[NaturalEvent] = field(default_factory=list)
    wildfire_count: int = 0
    landslide_count: int = 0
    nearest_fire_distance_km: Optional[float] = None
    eonet_url: str = EONET_EVENTS


async def fetch_eonet_events(
    client: httpx.AsyncClient,
    category: str,
    timeout_s: float = 2.5,
    cache: Optional[AsyncTTLCache] = None,
) -> List[NaturalEvent]:
    """Fetch recent/all-status EONET events for one category without credentials."""
    async def load() -> List[NaturalEvent]:
        response = await client.get(
            EONET_EVENTS,
            params={"category": category, "status": "all", "days": 365},
            timeout=timeout_s,
        )
        response.raise_for_status()
        payload = response.json()
        events: List[NaturalEvent] = []
        for item in payload.get("events") or []:
            categories = item.get("categories") or []
            category_name = str((categories[0] or {}).get("id", category)) if categories else category
            geometries = item.get("geometry") or []
            if not geometries:
                geometries = [{"date": None, "type": "Unknown", "coordinates": None}]
            for geometry in geometries:
                sources = item.get("sources") or []
                events.append(
                    NaturalEvent(
                        event_id=str(item.get("id", "")),
                        title=str(item.get("title", "")),
                        category=category_name,
                        date_utc=geometry.get("date"),
                        geometry_type=str(geometry.get("type", "Unknown")),
                        coordinates=geometry.get("coordinates"),
                        source_url=(sources[0] or {}).get("url") if sources else None,
                    )
                )
        return events

    if cache is None:
        return await load()
    return await cache.get_or_create(("nasa-eonet", category, 365), 15 * 60, load)


def nearest_event_distance_km(latitude: float, longitude: float, events: Iterable[NaturalEvent], category: str) -> Optional[float]:
    distances: List[float] = []
    for event in events:
        if category.lower() not in event.category.lower():
            continue
        for lon, lat in iter_lon_lat(event.coordinates):
            distances.append(haversine_km(latitude, longitude, lat, lon))
    return min(distances) if distances else None


def point_in_event_perimeter(latitude: float, longitude: float, event: NaturalEvent) -> bool:
    """Return true only for polygon geometries; point events have no perimeter."""
    if event.geometry_type.lower() != "polygon":
        return False
    polygons = event.coordinates or []
    if not polygons:
        return False
    outer_ring = polygons[0]
    return _point_in_ring(longitude, latitude, outer_ring)


def _point_in_ring(x: float, y: float, ring: Sequence[Sequence[float]]) -> bool:
    inside = False
    if len(ring) < 3:
        return False
    previous = ring[-1]
    for current in ring:
        x1, y1 = float(previous[0]), float(previous[1])
        x2, y2 = float(current[0]), float(current[1])
        if (y1 > y) != (y2 > y):
            crossing_x = (x2 - x1) * (y - y1) / ((y2 - y1) or 1e-15) + x1
            if x < crossing_x:
                inside = not inside
        previous = current
    return inside


def iter_lon_lat(coordinates: Any) -> Iterable[Tuple[float, float]]:
    """Yield lon/lat vertices from EONET Point, Polygon, MultiPolygon or LineString."""
    if not isinstance(coordinates, (list, tuple)):
        return
    if len(coordinates) >= 2 and all(isinstance(value, (int, float)) for value in coordinates[:2]):
        yield float(coordinates[0]), float(coordinates[1])
        return
    for child in coordinates:
        yield from iter_lon_lat(child)


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    radius = 6371.0088
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = p2 - p1
    dl = math.radians(lon2 - lon1)
    value = math.sin(dp / 2.0) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2.0) ** 2
    return radius * 2.0 * math.atan2(math.sqrt(value), math.sqrt(max(0.0, 1.0 - value)))
