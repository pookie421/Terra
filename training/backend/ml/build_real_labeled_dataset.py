"""Build event-time feature rows with source-labeled positives and marked absence samples.

NASA COOLR manually mapped polygons support inventory-positive cells. A matched
pre-event row is an inventory-absence pseudo-negative, not a surveyed intact slope.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import hashlib
import io
import json
import math
import sys
from dataclasses import asdict, dataclass, replace
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import httpx
import h3
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from backend.ml.global_xgb import FEATURE_NAMES
from backend.physics.multi_causal_engine import MultiCausalEngine, PhysicsInputs
from backend.realtime.dem_processor import DEMProcessor
from backend.realtime.global_ingestor import SOILGRIDS, GlobalIngestor
from backend.realtime.natural_events import NaturalEvent, fetch_eonet_events
from backend.realtime.soilgrids import estimate_pedotransfer, fetch_soilgrids_wcs, derive_soil_properties

NASA_GLC_CSV = "https://data.nasa.gov/docs/legacy/Global_Landslide_Catalog_Export/Global_Landslide_Catalog_Export_rows.csv"
NASA_COOLR_EVENTS = "https://gis.earthdata.nasa.gov/portal/rest/services/Landslides/COOLR_Events_Points/FeatureServer/0/query"
NASA_COOLR_POLYGONS = "https://gis.earthdata.nasa.gov/portal/rest/services/Landslides/COOLR_Events_Polygons/FeatureServer/0/query"
OPEN_METEO_ARCHIVE = "https://archive-api.open-meteo.com/v1/archive"
USGS_FDSN = "https://earthquake.usgs.gov/fdsnws/event/1/query"
OVERPASS_ENDPOINTS = (
    "https://overpass-api.de/api/interpreter", "https://overpass.private.coffee/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter", "https://lz4.overpass-api.de/api/interpreter",
)


@dataclass
class EventCandidate:
    event_id: str
    name: str
    event_date: Optional[str]
    latitude: float
    longitude: float
    regime: str
    source_name: str
    source_url: Optional[str]
    event_record_type: str
    label: Optional[int] = None
    label_source: Optional[str] = None
    label_quality: str = "event_candidate_point_not_cell_ground_truth"
    citation: Optional[str] = None
    failure_geometry: Any = None
    h3_cell_id: Optional[str] = None
    source_event_id: Optional[str] = None
    event_group_id: Optional[str] = None
    mapping_method: Optional[str] = None


# Coordinates are retained as user-provided event candidates and are cross-checked
# against public event catalogs below. They are not treated as independently verified.
SEED_EVENTS = [
    ("chooralmala_2024", "Chooralmala-Mundakkai Wayanad", "2024-07-30", 11.548, 76.134, "monsoon_rain"),
    ("kavalappara_2019", "Kavalappara", "2019-08-08", 11.412, 76.241, "monsoon_rain"),
    ("pettimudi_2020", "Pettimudi Munnar", "2020-08-06", 10.174, 77.012, "monsoon_rain"),
    ("idukki_2018", "Idukki Cheruthoni", "2018-08-16", 9.849, 76.972, "monsoon_rain"),
    ("mocoa_2017", "Mocoa Colombia", "2017-04-01", 1.149, -76.646, "monsoon_rain"),
    ("atami_2021", "Atami Japan", "2021-07-03", 35.103, 139.071, "monsoon_rain"),
    ("petropolis_2022", "Petropolis Brazil", "2022-02-15", -22.505, -43.178, "monsoon_rain"),
    ("kaikoura_2016", "Kaikoura Hapuku NZ", "2016-11-13", -42.316, 173.734, "coseismic"),
    ("langtang_2015", "Langtang Nepal", "2015-04-25", 28.214, 85.505, "coseismic"),
    ("atsuma_2018", "Hokkaido Atsuma Japan", "2018-09-05", 42.766, 141.928, "coseismic"),
    ("wenchuan_2008", "Wenchuan China", "2008-05-12", 31.021, 103.367, "coseismic"),
    ("bondo_2017", "Piz Cengalo Bondo Alps", "2017-08-23", 46.295, 9.602, "cryosphere"),
    ("chamoli_2021", "Chamoli Uttarakhand", "2021-02-07", 30.378, 79.731, "cryosphere"),
    ("zermatt_2024", "Zermatt Mattertal", "2024-06-21", 46.098, 7.781, "cryosphere"),
    ("brienz_2023", "Brienz-Brinzauls Switzerland", "2023-06-15", 46.668, 9.596, "cryosphere"),
    ("montecito_2018", "Montecito California", "2018-01-09", 34.448, -119.631, "post_wildfire"),
    ("malibu_2024", "Malibu Topanga", "2024-02-05", 34.082, -118.595, "post_wildfire"),
    ("glenwood_2021", "Glenwood Canyon Colorado", "2021-07-29", 39.563, -107.224, "post_wildfire"),
    ("ramban_2024", "Ramban NH-44 Jammu", "2024-04-29", 33.245, 75.238, "anthropogenic_cut_mining"),
    ("bingham_2020", "Bingham-Andes Escarpment", "2020-06-15", -22.290, -68.902, "anthropogenic_cut_mining"),
]


def _first(row: Mapping[str, Any], *keys: str) -> Optional[str]:
    lowered = {str(key).strip().lower(): value for key, value in row.items()}
    for key in keys:
        value = lowered.get(key.lower())
        if value not in (None, ""):
            return str(value).strip()
    return None


def parse_date(value: Optional[str]) -> Optional[str]:
    if not value:
        return None
    text = value.strip()
    if text.isdigit() and len(text) >= 12:
        try:
            return datetime.fromtimestamp(int(text) / 1000.0, tz=timezone.utc).date().isoformat()
        except (ValueError, OSError, OverflowError):
            return None
    for fmt in ("%Y-%m-%d", "%m/%d/%Y", "%Y/%m/%d", "%Y-%m-%d %H:%M:%S"):
        try:
            return datetime.strptime(text[:19], fmt).date().isoformat()
        except ValueError:
            continue
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00")).date().isoformat()
    except ValueError:
        return None


def _regime(trigger: Optional[str]) -> str:
    value = (trigger or "").lower()
    if any(token in value for token in ("earthquake", "seismic", "co-seismic", "coseismic")):
        return "coseismic"
    if any(token in value for token in ("snow", "freeze", "thaw", "glacier", "ice")):
        return "cryosphere"
    if any(token in value for token in ("fire", "burn", "wildfire")):
        return "post_wildfire"
    if any(token in value for token in ("human", "construction", "mining", "road", "cut")):
        return "anthropogenic_cut_mining"
    if any(token in value for token in ("rain", "monsoon", "precipitation", "storm")):
        return "monsoon_rain"
    return "trigger_unclassified"


async def fetch_glc(client: httpx.AsyncClient, limit: int) -> List[EventCandidate]:
    try:
        response = await client.get(NASA_GLC_CSV, timeout=20.0)
        response.raise_for_status()
        records: List[EventCandidate] = []
        seen_events = set()
        reader = csv.DictReader(io.StringIO(response.text))
        for row in reader:
            lat_raw = _first(row, "latitude", "lat")
            lon_raw = _first(row, "longitude", "lon", "lng")
            event_day = parse_date(_first(row, "event_date", "date", "event date"))
            if not lat_raw or not lon_raw or not event_day:
                continue
            try:
                lat, lon = float(lat_raw), float(lon_raw)
            except ValueError:
                continue
            if not (-90 <= lat <= 90 and -180 <= lon <= 180):
                continue
            external_id = _first(row, "event_id", "id") or "glc-{}".format(len(records) + 1)
            title = _first(row, "location_name", "location", "event_name") or "NASA GLC event {}".format(external_id)
            trigger = _first(row, "landslide_trigger", "trigger", "landslide trigger")
            records.append(EventCandidate(
                event_id="nasa_glc:{}".format(external_id), name=title, event_date=event_day,
                latitude=lat, longitude=lon, regime=_regime(trigger), source_name="NASA Global Landslide Catalog",
                source_url=NASA_GLC_CSV, event_record_type="reported_event_point",
                label=None, label_source=None, label_quality="NASA_reported_point_no_failure_polygon",
            ))
            if len(records) >= limit:
                break
        return records
    except Exception as exc:
        # Do not print redirect URLs; the NASA catalog endpoint may redirect through
        # a short-lived signed object-storage URL.
        print("NASA GLC catalog unavailable: {}".format(type(exc).__name__))
        return []


async def fetch_nasa_coolr_events(client: httpx.AsyncClient, limit: int) -> List[EventCandidate]:
    """Query cited COOLR points across pages and sample inventory groups fairly.

    A cited manual point is used as a 300 m initiation-zone proxy. The point
    represents an approximate source location; it is not a mapped crown polygon.
    """
    base_params = {
        "where": "event_date IS NOT NULL AND latitude IS NOT NULL AND longitude IS NOT NULL AND citation IS NOT NULL",
        "outFields": "objectid,event_id,event_date,event_title,location_description,landslide_trigger,landslide_category,citation,source_link,source_name,method,latitude,longitude",
        "returnGeometry": "true", "outSR": "4326", "resultRecordCount": "2000",
        "orderByFields": "event_date DESC", "f": "geojson",
    }
    try:
        grouped: Dict[str, List[EventCandidate]] = {}
        seen_events = set()
        # The first page can contain thousands of points from one inventory.
        # Paginate enough of the public layer to sample multiple dated inventories.
        for offset in range(0, 24000, 2000):
            response = await client.get(
                NASA_COOLR_EVENTS,
                params={**base_params, "resultOffset": str(offset)},
                timeout=20.0,
            )
            response.raise_for_status()
            features = response.json().get("features") or []
            for feature in features:
                props = feature.get("properties") or {}
                lat, lon = props.get("latitude"), props.get("longitude")
                coordinates = (feature.get("geometry") or {}).get("coordinates") or []
                if (lat is None or lon is None) and len(coordinates) >= 2:
                    lon, lat = coordinates[:2]
                if lat is None or lon is None:
                    continue
                try:
                    lat, lon = float(lat), float(lon)
                except (ValueError, TypeError):
                    continue
                day = parse_date(str(props.get("event_date")) if props.get("event_date") is not None else None)
                citation = str(props.get("citation") or "").strip()
                if day is None or not citation:
                    continue
                external_id = props.get("event_id") or props.get("objectid")
                if external_id is None or str(external_id) in seen_events:
                    continue
                seen_events.add(str(external_id))
                title = props.get("event_title") or props.get("location_description") or "NASA COOLR landslide event"
                method = str(props.get("method") or "").strip()
                group_seed = "|".join((str(title).strip(), day, citation))
                group_id = "nasa_coolr_inventory:{}".format(hashlib.sha1(group_seed.encode("utf-8")).hexdigest()[:12])
                cell_id = str((getattr(h3, "latlng_to_cell", None) or getattr(h3, "geo_to_h3"))(lat, lon, 9))
                cell_lat, cell_lon = (getattr(h3, "cell_to_latlng", None) or getattr(h3, "h3_to_geo"))(cell_id)
                within_point_buffer = haversine_km(lat, lon, float(cell_lat), float(cell_lon)) * 1000.0 <= 300.0
                cited_manual_point = method.lower() == "manual" and within_point_buffer
                records = grouped.setdefault(group_id, [])
                records.append(EventCandidate(
                    event_id="nasa_coolr:{}".format(external_id), name=str(title), event_date=day,
                    latitude=lat, longitude=lon, regime=_regime(str(props.get("landslide_trigger") or "")),
                    source_name="NASA COOLR", source_url=props.get("source_link") or NASA_COOLR_EVENTS,
                    event_record_type="cited_manual_point_buffer_cell" if cited_manual_point else "nasa_coolr_event_point",
                    label=1 if cited_manual_point else None,
                    label_source=("NASA COOLR manual point within 300 m; source citation: {}".format(citation) if cited_manual_point else None),
                    label_quality=(
                        "nasa_coolr_manual_point_h3_center_within_300m_initiation_proxy; no crown polygon"
                        if cited_manual_point else "NASA_event_point_with_catalog_citation_no_cell_polygon"
                    ),
                    citation=citation, h3_cell_id=cell_id, source_event_id=str(external_id),
                    event_group_id=group_id, mapping_method=method or None,
                ))
            if len(features) < 2000:
                break

        # Round-robin groups so one large event inventory cannot consume the
        # entire candidate budget or the positive sample.
        group_ids = sorted(
            grouped,
            key=lambda key: max((row.event_date or "" for row in grouped[key]), default=""),
            reverse=True,
        )
        records: List[EventCandidate] = []
        positions = {key: 0 for key in group_ids}
        while len(records) < limit:
            added = False
            for group_id in group_ids:
                index = positions[group_id]
                if index < len(grouped[group_id]):
                    records.append(grouped[group_id][index])
                    positions[group_id] += 1
                    added = True
                    if len(records) >= limit:
                        break
            if not added:
                break
        return records
    except Exception as exc:
        print("NASA COOLR events unavailable: {}".format(type(exc).__name__))
        return []


def _point_inside_polygon(lon: float, lat: float, rings: Sequence[Sequence[Sequence[float]]]) -> bool:
    def inside_ring(x: float, y: float, ring: Sequence[Sequence[float]]) -> bool:
        result = False
        if len(ring) < 3:
            return False
        previous = ring[-1]
        for current in ring:
            x1, y1 = float(previous[0]), float(previous[1])
            x2, y2 = float(current[0]), float(current[1])
            if (y1 > y) != (y2 > y):
                crossing = (x2 - x1) * (y - y1) / ((y2 - y1) or 1e-15) + x1
                if x < crossing:
                    result = not result
            previous = current
        return result
    return bool(rings) and inside_ring(lon, lat, rings[0]) and not any(inside_ring(lon, lat, ring) for ring in rings[1:])


def _interior_h3_cell(geometry: Mapping[str, Any], resolution: int = 9) -> Optional[Tuple[str, float, float]]:
    kind = str(geometry.get("type", ""))
    coords = geometry.get("coordinates") or []
    if kind == "Polygon":
        polygons = [coords]
    elif kind == "MultiPolygon":
        polygons = coords
    else:
        return None
    cell_from_latlng = getattr(h3, "latlng_to_cell", None) or getattr(h3, "geo_to_h3")
    cell_to_latlng = getattr(h3, "cell_to_latlng", None) or getattr(h3, "h3_to_geo")
    for rings in polygons:
        if not rings or not rings[0]:
            continue
        exterior = rings[0]
        xs = [float(point[0]) for point in exterior]
        ys = [float(point[1]) for point in exterior]
        candidates = [(sum(xs) / len(xs), sum(ys) / len(ys)), ((min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2)]
        # For concave polygons, sample a bounded grid until an interior cell center is found.
        for yi in range(1, 12):
            for xi in range(1, 12):
                candidates.append((min(xs) + (max(xs) - min(xs)) * xi / 12.0, min(ys) + (max(ys) - min(ys)) * yi / 12.0))
        for lon, lat in candidates:
            if not _point_inside_polygon(lon, lat, rings):
                continue
            try:
                cell = str(cell_from_latlng(lat, lon, resolution))
                center_lat, center_lon = map(float, cell_to_latlng(cell))
            except Exception:
                continue
            if _point_inside_polygon(center_lon, center_lat, rings):
                return cell, center_lat, center_lon
    return None


def _mapped_polygon_h3_cells(geometry: Mapping[str, Any], resolution: int = 9) -> List[Tuple[str, float, float]]:
    """Return H3 cell centers inside a mapped polygon, retaining every inventory cell."""
    kind = str(geometry.get("type", ""))
    coordinates = geometry.get("coordinates") or []
    polygons = [coordinates] if kind == "Polygon" else coordinates if kind == "MultiPolygon" else []
    cell_to_latlng = getattr(h3, "cell_to_latlng", None) or getattr(h3, "h3_to_geo")
    found: Dict[str, Tuple[str, float, float]] = {}
    for rings in polygons:
        if not rings or not rings[0]:
            continue
        try:
            if hasattr(h3, "LatLngPoly") and hasattr(h3, "polygon_to_cells"):
                outer = [(float(point[1]), float(point[0])) for point in rings[0]]
                holes = [[(float(point[1]), float(point[0])) for point in ring] for ring in rings[1:]]
                shape = h3.LatLngPoly(outer, *holes)
                candidates = h3.polygon_to_cells(shape, resolution)
            else:
                candidates = []
        except Exception:
            candidates = []
        for raw_cell in candidates:
            cell_id = str(raw_cell)
            try:
                lat, lon = map(float, cell_to_latlng(cell_id))
            except Exception:
                continue
            if _point_inside_polygon(lon, lat, rings):
                found[cell_id] = (cell_id, lat, lon)
    if not found:
        interior = _interior_h3_cell(geometry, resolution)
        if interior is not None:
            found[interior[0]] = interior
    return [found[key] for key in sorted(found)]


async def fetch_nasa_coolr_polygons(client: httpx.AsyncClient, limit: int) -> List[EventCandidate]:
    """Create positive inventory cells only from cited, manually mapped polygons."""
    params = {
        "where": "event_date IS NOT NULL AND citation IS NOT NULL",
        "outFields": "objectid,event_id,event_date,event_title,location_description,landslide_trigger,landslide_category,citation,source_link,source_name,method,latitude,longitude",
        "returnGeometry": "true", "outSR": "4326", "resultRecordCount": "2000",
        "resultOffset": "0", "orderByFields": "event_date DESC", "f": "geojson",
    }
    try:
        response = await client.get(NASA_COOLR_POLYGONS, params=params, timeout=15.0)
        response.raise_for_status()
        features = response.json().get("features") or []
        records: List[EventCandidate] = []
        seen_cells = set()
        for feature in features:
            props = feature.get("properties") or {}
            if str(props.get("method", "")).strip().lower() != "manual":
                continue
            geometry = feature.get("geometry") or {}
            mapped_cells = _mapped_polygon_h3_cells(geometry)
            if not mapped_cells:
                continue
            day = parse_date(str(props.get("event_date")) if props.get("event_date") is not None else None)
            if day is None:
                continue
            external_id = props.get("event_id") or props.get("objectid") or len(records)
            title = props.get("event_title") or props.get("location_description") or "NASA COOLR mapped landslide inventory"
            citation = str(props.get("citation") or "")
            group_seed = "|".join((str(title).strip(), day, citation))
            group_id = "nasa_coolr_inventory:{}".format(hashlib.sha1(group_seed.encode("utf-8")).hexdigest()[:12])
            for cell_id, lat, lon in mapped_cells:
                row_key = (str(external_id), cell_id)
                if row_key in seen_cells:
                    continue
                seen_cells.add(row_key)
                records.append(EventCandidate(
                    event_id="nasa_coolr_polygon:{}:{}".format(external_id, cell_id), name=str(title), event_date=day,
                    latitude=lat, longitude=lon, regime=_regime(str(props.get("landslide_trigger") or "")),
                    source_name="NASA COOLR manually mapped polygon", source_url=props.get("source_link") or NASA_COOLR_POLYGONS,
                    event_record_type="manually_mapped_event_polygon_cell", label=1,
                    label_source="NASA COOLR manual event polygon: {}".format(citation),
                    label_quality="polygon_inventory_positive_h3_center_inside; not initiation_crown",
                    citation=citation, failure_geometry=geometry, h3_cell_id=cell_id,
                    source_event_id=str(external_id),
                    event_group_id=group_id, mapping_method=str(props.get("method") or "Manual"),
                ))
                if len(records) >= limit:
                    break
            if len(records) >= limit:
                break
        return records
    except Exception as exc:
        print("NASA COOLR polygons unavailable: {}".format(type(exc).__name__))
        return []


async def fetch_osm_candidates(client: httpx.AsyncClient, seeds: Sequence[EventCandidate]) -> Tuple[List[EventCandidate], Dict[str, Any]]:
    """Query dated-event neighborhoods for mapped natural=landslide features."""
    semaphore = asyncio.Semaphore(3)

    async def one(seed: EventCandidate) -> Tuple[List[EventCandidate], bool, Optional[str]]:
        query = """[out:json][timeout:10];(node[natural=landslide](around:1500,{lat},{lon});way[natural=landslide](around:1500,{lat},{lon});relation[natural=landslide](around:1500,{lat},{lon}););out center tags;""".format(lat=seed.latitude, lon=seed.longitude)
        last_error = None
        async with semaphore:
            for endpoint in OVERPASS_ENDPOINTS:
                try:
                    response = await client.post(endpoint, data={"data": query}, timeout=8.0)
                    response.raise_for_status()
                    payload = response.json()
                    results = []
                    for element in payload.get("elements") or []:
                        center = element.get("center") or {}
                        lat, lon = element.get("lat", center.get("lat")), element.get("lon", center.get("lon"))
                        if lat is None or lon is None:
                            continue
                        tags = element.get("tags") or {}
                        ident = "osm:{}:{}".format(element.get("type", "feature"), element.get("id", ""))
                        results.append(EventCandidate(
                            event_id=ident, name=str(tags.get("name") or "OSM natural=landslide feature"),
                            event_date=None, latitude=float(lat), longitude=float(lon), regime="trigger_unclassified",
                            source_name="OpenStreetMap Overpass", source_url="https://www.openstreetmap.org/{}s/{}".format(element.get("type", "way"), element.get("id", "")),
                            event_record_type="mapped_landslide_feature_without_event_date", label=None,
                        label_source=None, label_quality="mapped_feature_not_dated_failure_cell",
                        source_event_id=ident,
                        ))
                    return results, True, None
                except Exception as exc:
                    last_error = "{}: {}".format(type(exc).__name__, exc)
                    continue
        return [], False, last_error or "all Overpass endpoints failed"

    result_groups = await asyncio.gather(*(one(seed) for seed in seeds))
    unique: Dict[str, EventCandidate] = {}
    successful_queries = sum(1 for _, succeeded, _ in result_groups if succeeded)
    errors = [error for _, succeeded, error in result_groups if not succeeded and error]
    for group, _, _ in result_groups:
        for item in group:
            unique[item.event_id] = item
    return list(unique.values()), {
        "available": successful_queries > 0,
        "successful_queries": successful_queries,
        "failed_queries": len(result_groups) - successful_queries,
        "feature_count": len(unique),
        "error": errors[0] if errors and not successful_queries else None,
    }


def eonet_candidates(events: Iterable[NaturalEvent]) -> List[EventCandidate]:
    result = []
    for event in events:
        coords = event.coordinates
        if event.geometry_type.lower() != "point" or not isinstance(coords, (list, tuple)) or len(coords) < 2:
            continue
        lon, lat = float(coords[0]), float(coords[1])
        try:
            event_day = parse_date(event.date_utc)
        except Exception:
            event_day = None
        category = "post_wildfire" if "wildfire" in event.category.lower() else "trigger_unclassified"
        result.append(EventCandidate(
            event_id="eonet:{}:{}".format(event.event_id, event.date_utc or ""), name=event.title,
            event_date=event_day, latitude=lat, longitude=lon, regime=category,
            source_name="NASA EONET", source_url=event.source_url or "https://eonet.gsfc.nasa.gov/docs/v3",
            event_record_type="eonet_event_point", label=None, label_source=None,
            label_quality="event_candidate_point_not_cell_ground_truth",
            source_event_id=str(event.event_id),
        ))
    return result


def event_candidates(glc: Sequence[EventCandidate], coolr: Sequence[EventCandidate], polygons: Sequence[EventCandidate], live: Sequence[EventCandidate], osm: Sequence[EventCandidate], target: int) -> List[EventCandidate]:
    items = [EventCandidate(
        event_id=ident, name=name, event_date=day, latitude=lat, longitude=lon, regime=regime,
        source_name="user-provided project candidate list", source_url=None,
        event_record_type="user_supplied_candidate_unverified", label=None, label_source=None,
        label_quality="user_supplied_candidate_not_independently_verified",
        source_event_id=ident,
    ) for ident, name, day, lat, lon, regime in SEED_EVENTS]
    seen = set()
    result: List[EventCandidate] = []
    pools = [items, list(polygons), list(coolr), list(live), list(osm), list(glc)]
    quotas = (target, target, target, target, target)
    # Keep the five causal-regime seed candidates, then reserve slots for cited
    # mapped polygons, cited event points, EONET points, and OSM features.
    for item in items[:target]:
        if item.event_id in seen:
            continue
        seen.add(item.event_id)
        result.append(item)
    remaining = max(0, target - len(result))
    for pool_index, pool in enumerate(pools[1:]):
        if remaining <= 0:
            break
        quota = min(remaining, quotas[pool_index])
        added = 0
        for item in pool:
            if item.event_id in seen:
                continue
            seen.add(item.event_id)
            result.append(item)
            added += 1
            if added >= quota or len(result) >= target:
                break
        remaining = target - len(result)
    for pool in pools[1:]:
        for item in pool:
            if len(result) >= target:
                break
            if item.event_id not in seen:
                seen.add(item.event_id)
                result.append(item)
    return result


async def fetch_archive(client: httpx.AsyncClient, event: EventCandidate) -> Dict[str, Any]:
    from backend.ml.archive_weather import acquire_archive
    return await acquire_archive(client,event)


async def fetch_usgs(client: httpx.AsyncClient, event: EventCandidate) -> Dict[str, Any]:
    if not event.event_date:
        return {"usgs_source_status": "not_run_no_event_date", "usgs_event_count": None}
    try:
        day = date.fromisoformat(event.event_date)
        params = {
            "format": "geojson", "starttime": (day - timedelta(days=1)).isoformat(),
            "endtime": (day + timedelta(days=1)).isoformat(), "latitude": event.latitude,
            "longitude": event.longitude, "maxradiuskm": 250, "minmagnitude": 2.5,
            "orderby": "magnitude",
        }
        import hashlib
        cache=Path(__file__).resolve().parents[2]/'data/source_catalogs/seismic_archive'
        key=hashlib.sha256(json.dumps(params,sort_keys=True).encode()).hexdigest()
        retained=cache/(key+'.json')
        if retained.exists():payload=json.loads(retained.read_text())['response']
        else:
            if not hasattr(client,'_terra_seismic_lock'):
                client._terra_seismic_lock=asyncio.Lock();client._terra_seismic_next=0.
            async with client._terra_seismic_lock:
                loop=asyncio.get_running_loop()
                await asyncio.sleep(max(0.,client._terra_seismic_next-loop.time()))
                client._terra_seismic_next=loop.time()+.75
            response = await client.get(USGS_FDSN, params=params, timeout=12.0)
            response.raise_for_status();payload=response.json()
            cache.mkdir(parents=True,exist_ok=True)
            retained.write_text(json.dumps({'request':params,'url':str(response.url),'response':payload}))
        features = payload.get("features") or []
        if not features:
            return {"usgs_source_status": "live_no_matching_earthquake", "usgs_event_count": 0, "usgs_max_magnitude": None, "pga_attenuation_proxy_g": 0.0}
        ranked = []
        for feature in features:
            props = feature.get("properties") or {}
            xyz = ((feature.get("geometry") or {}).get("coordinates") or [])
            if len(xyz) < 3 or props.get("mag") is None:
                continue
            lon, lat, depth = map(float, xyz[:3])
            distance = haversine_km(event.latitude, event.longitude, lat, lon)
            hyp = math.sqrt(distance * distance + depth * depth)
            magnitude = float(props["mag"])
            pga = 10 ** (-1.02 + 0.249 * magnitude - math.log10(max(hyp, 1.0)) - 0.00255 * max(hyp, 1.0))
            ranked.append((pga, magnitude, feature.get("id"), distance))
        ranked.sort(reverse=True)
        if not ranked:
            return {"usgs_source_status": "live_no_parseable_earthquake", "usgs_event_count": len(features)}
        pga, magnitude, event_id, distance = ranked[0]
        return {
            "usgs_source_status": "live_usgs_event_with_attenuation_estimate",
            "usgs_event_count": len(features), "usgs_event_id": event_id,
            "usgs_max_magnitude": magnitude, "usgs_distance_km": distance,
            "pga_attenuation_proxy_g": pga,
            "pga_quality": "attenuation_model_estimate_not_station_observation",
        }
    except Exception as exc:
        return {"usgs_source_status": "unavailable: {}: {}".format(type(exc).__name__, exc), "usgs_event_count": None}


async def fetch_soil_properties(client: httpx.AsyncClient, event: EventCandidate) -> Dict[str, Any]:
    params = [
        ("lon", round(event.longitude, 5)), ("lat", round(event.latitude, 5)),
        ("property", "bdod"), ("property", "clay"), ("property", "sand"), ("property", "soc"),
        ("depth", "0-5cm"), ("depth", "5-15cm"), ("depth", "15-30cm"), ("value", "mean"),
    ]
    errors = []
    try:
        response = await client.get(SOILGRIDS, params=params, timeout=2.5)
        response.raise_for_status()
        raw = response.json()
        values = GlobalIngestor._soil_property_values(raw)
        if all(values.get(key) is not None for key in ("bdod_kg_m3", "clay_g_kg", "sand_g_kg", "soc_g_kg")):
            return _soil_values_to_fractions(values, "ISRIC SoilGrids REST", "soilgrids_prediction")
        errors.append("SoilGrids REST had incomplete properties")
    except Exception as exc:
        errors.append("REST {}: {}".format(type(exc).__name__, exc))
    try:
        raw_wcs = await fetch_soilgrids_wcs(client, event.latitude, event.longitude, timeout_s=2.5)
        return _soil_values_to_fractions(raw_wcs, "ISRIC SoilGrids WCS", "soilgrids_prediction")
    except Exception as exc:
        errors.append("WCS {}: {}".format(type(exc).__name__, exc))
    return {"soilgrids_source_status": "unavailable; " + "; ".join(errors)}


def _soil_values_to_fractions(values: Mapping[str, Any], source: str, quality: str) -> Dict[str, Any]:
    derived = derive_soil_properties(values)
    sand, clay, organic = (derived[k] for k in ('sand_fraction','clay_fraction','organic_matter_fraction'))
    return {
        "soilgrids_source_status": source,
        "soil_property_source": source,
        "soil_property_quality": quality,
        "sand_fraction": sand, "clay_fraction": clay,
        "silt_fraction": max(0.0, 1.0 - sand - clay),
        "organic_matter_fraction": organic,
        "bulk_density_g_cm3": float(values["bdod_kg_m3"]) / 1000.0,
        "soil_sand_g_kg": float(values["sand_g_kg"]),
        "soil_clay_g_kg": float(values["clay_g_kg"]),
        "soil_soc_g_kg": float(values["soc_g_kg"]),
        "soil_bulk_density_kg_m3": float(values["bdod_kg_m3"]),
    }


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    radius = 6371.0088
    a1, a2 = math.radians(lat1), math.radians(lat2)
    da, db = math.radians(lat2 - lat1), math.radians(lon2 - lon1)
    h = math.sin(da / 2) ** 2 + math.cos(a1) * math.cos(a2) * math.sin(db / 2) ** 2
    return radius * 2.0 * math.atan2(math.sqrt(h), math.sqrt(max(0.0, 1.0 - h)))


def populate_physics_features(base: Dict[str, Any], terrain_result: Any) -> None:
    """Fill the 14 model columns using observed values plus explicit fallbacks."""
    cells = list(terrain_result.cells.values()) if terrain_result is not None and not isinstance(terrain_result, BaseException) and terrain_result.cells else []
    if cells:
        nearby = [c for c in cells if haversine_km(base["latitude"],base["longitude"],c.latitude,c.longitude)<=1.0]
        nearest = min(cells, key=lambda c:(c.cell_id != base.get("h3_cell_id"), haversine_km(base["latitude"],base["longitude"],c.latitude,c.longitude), str(c.cell_id)))
        base["report_neighborhood_slope_p90_context_only"] = float(np.percentile([c.slope_deg for c in nearby],90)) if nearby else None
        base["terrain_selection_policy"] = "labeled_h3_cell_or_nearest_source_coordinate; no slope-based selection"
        base["representative_terrain_cell_id"] = str(nearest.cell_id)
        base["representative_elevation_m"] = float(getattr(nearest,'elevation_m',base.get('terrain_elevation_m') or 0.))
        base["representative_terrain_candidate_count"] = len(nearby or cells)
        slope = float(nearest.slope_deg)
        curvature = float(nearest.profile_curvature)
        twi = max(0.01, float(nearest.twi))
        base["terrain_source_status"] = "live_aws_terrarium_dem"
    else:
        slope, curvature, twi = 15.0, 0.0, 6.0
        base.setdefault("terrain_source_status", "unavailable; explicit terrain feature fallback used")

    soil = estimate_pedotransfer(slope, curvature, twi)
    fallback = ('estimate' in str(base.get('soil_property_source', '')) or
                'prior' in str(base.get('soil_property_source', '')) or
                any(base.get(key) is None for key in ('sand_fraction','clay_fraction','organic_matter_fraction','bulk_density_g_cm3')))
    if fallback:
        for key in ('sand_fraction','clay_fraction','organic_matter_fraction','bulk_density_g_cm3'):
            base[key] = soil[key]
        base['soil_property_source'] = soil['source']
        base['soil_property_quality'] = 'static_prior_not_measured'
    moisture = base.get('soil_moisture_72h_max_m3m3', base.get('soil_moisture_event_day_mean_m3m3'))
    porosity = 1.0 - float(base['bulk_density_g_cm3']) / 2.65
    base['soil_porosity_estimate'] = porosity
    if moisture is None or not math.isfinite(float(moisture)) or not 0 <= float(moisture) <= porosity:
        base.update({name:None for name in FEATURE_NAMES})
        base['p_failure_physical'] = None
        base['physics_inputs_json'] = None
        base['physics_feature_quality'] = 'unavailable_missing_or_incompatible_moisture_porosity'
        base['physics_feature_error'] = 'Cannot infer absolute saturation from missing or incompatible volumetric moisture'
        return
    saturation = float(moisture) / porosity
    rain = float(base.get('rain_effective_storm_mm', base.get('rain_event_day_mm')) or 0.)
    snow_depth = float(base.get('snow_depth_72h_max_m', base.get('snow_depth_event_day_max_m')) or 0.)
    air_temp = float(base['temperature_event_day_mean_c'] if base.get('temperature_event_day_mean_c') is not None else 10.)
    soil_temp = float(base['soil_temperature_event_day_mean_c'] if base.get('soil_temperature_event_day_mean_c') is not None else air_temp)
    pga = float(base.get('pga_attenuation_proxy_g') or 0.)
    high_twi = [cell for cell in cells if float(cell.twi) >= 14.5]
    river_distance = min((haversine_km(nearest.latitude, nearest.longitude, cell.latitude, cell.longitude) * 1000.0 for cell in high_twi), default=9999.0) if cells else 9999.0
    burn = float(base.get("historical_burn_severity") or 0.0)
    vegetation = float(base["historical_ndvi"] if base.get("historical_ndvi") is not None else 0.50)
    cycles = float(base.get("freeze_thaw_cycles_7d") or 0.)
    warming = float(base.get("warming_delta_c") or 0.)
    road = float(base["historical_road_cut_distance_m"] if base.get("historical_road_cut_distance_m") is not None else 9999.)
    inputs = PhysicsInputs(
        sand_fraction=float(base["sand_fraction"]), clay_fraction=float(base["clay_fraction"]),
        organic_matter_fraction=float(base["organic_matter_fraction"]), bulk_density_g_cm3=float(base["bulk_density_g_cm3"]),
        slope_deg=slope, twi=twi, soil_saturation_fraction=saturation, rain_24h_mm=rain,
        storm_duration_hr=24.0, snow_depth_m=snow_depth, air_temperature_c=air_temp,
        soil_temperature_c=soil_temp, freeze_thaw_cycles_7d=cycles, burn_severity=burn,
        warming_delta_c=warming, baseline_air_temperature_c=air_temp-warming, observed_snow_depth_m=snow_depth,
        ndvi_norm=vegetation, pga_g=pga, road_cut_distance_m=road, river_distance_m=river_distance, profile_curvature=curvature, elevation_m=float(base.get("representative_elevation_m") or base.get("terrain_elevation_m") or 0), latitude_deg=float(base.get("latitude") or 0),
    )
    state = MultiCausalEngine().compute(inputs)
    from backend.physics.features import physics_feature_matrix
    row_features = dict(zip(FEATURE_NAMES, map(float,physics_feature_matrix(inputs,state,np.array([curvature]))[0])))
    row_features.update(fs_static_prior=float(state.fs_static[0]),p_failure_physical=float(state.p_failure_physical[0]))
    base.update(row_features)
    base["physics_inputs_json"] = json.dumps(asdict(inputs), sort_keys=True)
    base["physics_feature_provenance"] = {
        "soil": base.get("soil_property_source", soil["source"]),
        "saturation": "Open-Meteo 72h peak moisture / estimated porosity" if moisture is not None else "0.50 fallback; peak soil moisture unavailable",
        "storm_forcing": "72h peak rolling 24h liquid rain plus 15% of remaining inclusive 7-day rain; same rule for all labels; not an observed instantaneous 24h depth",
        "storm_window_quality": base.get("storm_window_quality", "legacy_event_day_fallback"),
        "historical_vegetation_and_burn": "time-matched fields when supplied; otherwise vegetation=0.50 and burn=0.0 explicit missing-data defaults",
        "road_cut_distance_m": "time-matched distance when supplied; otherwise 9999.0 missing-geometry default",
        "river_distance_m": "distance to nearest event-window DEM H3 cell with TWI >= 14.5" if high_twi else "9999.0; no high-TWI DEM cell in sample",
        "weather": base.get("weather_source_status", "unavailable"),
        "freeze_thaw": base.get("freeze_thaw_quality", "unavailable; zero fallback"),
        "warming": "event-day minus preceding 7-day mean, applied label-blind; missing values use zero",
        "terrain": "Labeled H3 cell or nearest source coordinate, never selected by slope; point/footprint location uncertainty retained"
    }
    base["physics_feature_error"] = None
    base["physics_feature_quality"] = "complete_with_explicit_defaults" if not cells or moisture is None else "complete_with_proxy_inputs"


async def enrich_one(event: EventCandidate, client: httpx.AsyncClient, dem: DEMProcessor, semaphore: asyncio.Semaphore) -> Dict[str, Any]:
    base = asdict(event)
    base.pop("failure_geometry", None)
    base.update({name: None for name in FEATURE_NAMES})
    cell_fn = getattr(h3, "latlng_to_cell", None) or getattr(h3, "geo_to_h3")
    base.update({
        "event_candidate": 1, "label": event.label,
        "h3_cell_id": event.h3_cell_id or str((getattr(h3, "latlng_to_cell", None) or getattr(h3, "geo_to_h3"))(event.latitude, event.longitude, 9)),
        "region_block": str(cell_fn(event.latitude, event.longitude, 4)),
        "label_note": (
            "H3 cell center lies inside cited manually mapped NASA COOLR event polygon."
            if event.label == 1 else event.label_quality if event.label == 0 else
            "Event presence is not a cell-level failure observation; label remains null."
        ),
    })
    async with semaphore:
        weather_task = fetch_archive(client, event)
        usgs_task = fetch_usgs(client, event)
        soil_task = fetch_soil_properties(client, event)
        terrain_task = dem.process(event.latitude, event.longitude, 1.2, client)
        weather, seismic, soil, terrain_result = await asyncio.gather(weather_task, usgs_task, soil_task, terrain_task, return_exceptions=True)
    if not isinstance(weather, BaseException):
        base.update(weather)
    else:
        base["weather_source_status"] = "unavailable: {}".format(weather)
    if not isinstance(seismic, BaseException):
        base.update(seismic)
    else:
        base["usgs_source_status"] = "unavailable: {}".format(seismic)
    if not isinstance(soil, BaseException):
        base.update(soil)
    else:
        base["soilgrids_source_status"] = "unavailable: {}".format(soil)
    if not isinstance(terrain_result, BaseException) and terrain_result.cells:
        cells = list(terrain_result.cells.values())
        nearest = min(cells, key=lambda cell: haversine_km(event.latitude, event.longitude, cell.latitude, cell.longitude))
        base.update({
            "terrain_source_status": "live_aws_terrarium_dem",
            "terrain_cell_id": nearest.cell_id,
            "terrain_slope_deg": nearest.slope_deg,
            "terrain_profile_curvature": nearest.profile_curvature,
            "terrain_twi": nearest.twi,
            "terrain_elevation_m": nearest.elevation_m,
            "terrain_pixel_spacing_x_m": terrain_result.pixel_spacing_x_m,
            "terrain_pixel_spacing_y_m": terrain_result.pixel_spacing_y_m,
        })
        if base.get("sand_fraction") is None:
            soil_moisture = base.get("soil_moisture_event_day_mean_m3m3")
            moisture_pct = float(soil_moisture) * 100.0 if soil_moisture is not None else None
            estimate = estimate_pedotransfer(nearest.slope_deg, nearest.profile_curvature, nearest.twi, moisture_pct)
            base.update({
                "soil_property_source": "topographic_climatic_pedotransfer_estimate",
                "soil_property_quality": "estimated_not_measured",
                "sand_fraction": estimate["sand_fraction"], "clay_fraction": estimate["clay_fraction"],
                "silt_fraction": estimate["silt_fraction"], "organic_matter_fraction": estimate["organic_matter_fraction"],
                "bulk_density_g_cm3": estimate["bulk_density_g_cm3"],
                "soil_sand_g_kg": estimate["sand_g_kg"], "soil_clay_g_kg": estimate["clay_g_kg"],
                "soil_soc_g_kg": estimate["soc_g_kg"], "soil_bulk_density_kg_m3": estimate["bdod_kg_m3"],
            })
    else:
        base["terrain_source_status"] = "unavailable: {}".format(terrain_result)
        if base.get("sand_fraction") is None:
            estimate = estimate_pedotransfer(15.0, 0.0, 6.0)
            base.update({
                "soil_property_source": "topographic_climatic_pedotransfer_estimate",
                "soil_property_quality": "estimated_not_measured_missing_terrain",
                "sand_fraction": estimate["sand_fraction"], "clay_fraction": estimate["clay_fraction"],
                "silt_fraction": estimate["silt_fraction"], "organic_matter_fraction": estimate["organic_matter_fraction"],
                "bulk_density_g_cm3": estimate["bulk_density_g_cm3"],
                "soil_sand_g_kg": estimate["sand_g_kg"], "soil_clay_g_kg": estimate["clay_g_kg"],
                "soil_soc_g_kg": estimate["soc_g_kg"], "soil_bulk_density_kg_m3": estimate["bdod_kg_m3"],
            })
    populate_physics_features(base, terrain_result)
    return base


async def enrich_temporal_control(
    event: EventCandidate,
    original_row: Mapping[str, Any],
    client: httpx.AsyncClient,
    semaphore: asyncio.Semaphore,
    allow_date_search: bool = True,
) -> Dict[str, Any]:
    """Reuse static terrain/soil provenance and fetch only date-specific control data."""
    base = dict(original_row)
    base.update(asdict(event))
    base.pop("failure_geometry", None)
    base["h3_cell_id"] = original_row.get("h3_cell_id") or event.h3_cell_id or str(
        (getattr(h3, "latlng_to_cell", None) or getattr(h3, "geo_to_h3"))(event.latitude, event.longitude, 9)
    )
    base["event_candidate"] = 1
    base["label"] = 0
    base["label_note"] = event.label_quality
    base["label_source"] = event.label_source
    base["label_quality"] = event.label_quality
    async with semaphore:
        from backend.ml.archive_weather import acquire_archive
        from dataclasses import replace
        weather = await acquire_archive(client,event,search_days=30 if allow_date_search else 0)
        selected=weather.get("weather_selected_date",event.event_date)
        event=replace(event,event_date=selected)
        seismic = ({k:v for k,v in original_row.items() if k.startswith(('usgs_', 'pga_', 'earthquake_'))}
                   if selected==original_row.get('event_date') and str(original_row.get('usgs_source_status','')).startswith('live_')
                   else await fetch_usgs(client,event))
    base["control_original_date"] = base.get("control_original_date") or base["event_date"]
    base["event_date"] = event.event_date
    base["control_date_selection"] = "nearest complete effective storm <=35 mm day in 30-day pre-offset search" if allow_date_search else "frozen holdout date; no reselection"
    base.update(weather)
    base.update(seismic)
    try:
        from types import SimpleNamespace
        terrain_cell = SimpleNamespace(
            cell_id=base["h3_cell_id"], latitude=event.latitude, longitude=event.longitude,
            elevation_m=float(base.get("terrain_elevation_m") or 0.0),
            slope_deg=float(base["terrain_slope_deg"] if base.get("terrain_slope_deg") is not None else 15.0),
            profile_curvature=float(base.get("terrain_profile_curvature") or 0.0),
            twi=float(base.get("terrain_twi") or 6.0),
        )
        populate_physics_features(base, SimpleNamespace(cells={terrain_cell.cell_id: terrain_cell}))
    except Exception as exc:
        base["physics_feature_error"] = "{}: {}".format(type(exc).__name__, exc)
    from backend.ml.pinls import quiescent_control
    base["quiescent_control_verified"] = quiescent_control(base)
    base["control_verification_scope"] = "Reanalysis and modeled stability only; not surveyed intact ground"
    return base


def _write_event_group_holdout(output: Path, rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """Build a fixed, balanced multi-event holdout with spatial embargo."""
    from backend.ml.event_holdout import write_holdout
    return write_holdout(output, rows)


async def build_dataset(output: Path, max_events: int, timeout_s: float) -> Dict[str, Any]:
    headers = {"User-Agent": "TERRA-SHIELD/0.2 (open-data research prototype)", "Accept": "application/json,text/csv,*/*"}
    timeout = httpx.Timeout(timeout_s)
    async with httpx.AsyncClient(timeout=timeout, headers=headers, limits=httpx.Limits(max_connections=24, max_keepalive_connections=12), follow_redirects=True) as client:
        coolr_task = fetch_nasa_coolr_events(client, max_events)
        polygons_task = fetch_nasa_coolr_polygons(client, max_events)
        live_tasks = await asyncio.gather(
            fetch_eonet_events(client, "landslides", min(timeout_s, 8.0)),
            fetch_eonet_events(client, "wildfires", min(timeout_s, 8.0)),
            return_exceptions=True,
        )
        live_events: List[EventCandidate] = []
        for result in live_tasks:
            if not isinstance(result, BaseException):
                live_events.extend(eonet_candidates(result))
        coolr, polygons = await asyncio.gather(coolr_task, polygons_task)
        # The old one-time NASA GLC CSV is only a fallback; the current COOLR
        # service contains GLC records plus cited inventories and polygons.
        glc = [] if coolr or polygons else await fetch_glc(client, max_events)
        osm, osm_status = await fetch_osm_candidates(client, [EventCandidate(
            event_id=item[0], name=item[1], event_date=item[2], latitude=item[3], longitude=item[4], regime=item[5],
            source_name="project candidate", source_url=None, event_record_type="seed", label=None,
        ) for item in (SEED_EVENTS[0], SEED_EVENTS[4], SEED_EVENTS[7], SEED_EVENTS[12], SEED_EVENTS[15], SEED_EVENTS[18])])
        catalog = event_candidates(glc, coolr, polygons, live_events, osm, max_events)
        temporal_controls = []
        control_source_names = {"NASA COOLR", "NASA GLC", "NASA EONET", "NASA COOLR manually mapped polygon"}
        for positive in catalog:
            if not positive.event_date or positive.source_name not in control_source_names:
                continue
            control_day = date.fromisoformat(positive.event_date) - timedelta(days=60)
            temporal_controls.append(replace(
                positive,
                event_id="{}:tminus60".format(positive.event_id),
                name="{} (60-day pre-event sample)".format(positive.name),
                event_date=control_day.isoformat(),
                source_name="NASA COOLR event-cell inventory-absence sample",
                event_record_type="temporal_absence_sample",
                label=0,
                label_source="Same source-event cell 60 days before reported date; inventory absence assumed, not field-verified",
                label_quality="temporal_inventory_absence_pseudo_negative_unverified",
                failure_geometry=None,
            ))
        catalog.extend(temporal_controls)
        dem = DEMProcessor(zoom=13)
        semaphore = asyncio.Semaphore(4)
        existing_by_event: Dict[str, Dict[str, Any]] = {}
        if output.exists():
            try:
                import pyarrow.parquet as pq
                existing_by_event = {str(row.get("event_id")): row for row in pq.read_table(output).to_pylist() if row.get("event_id")}
            except Exception:
                existing_by_event = {}
        catalog_by_id = {event.event_id: event for event in catalog}

        async def build_one(event: EventCandidate) -> Dict[str, Any]:
            existing = existing_by_event.get(event.event_id)
            if existing is not None:
                reused = dict(existing)
                reused.update({
                    "event_record_type": event.event_record_type,
                    "label": event.label,
                    "label_source": event.label_source,
                    "label_quality": event.label_quality,
                    "label_note": (
                        "H3 cell center lies inside cited manually mapped NASA COOLR event polygon."
                        if event.event_record_type == "manually_mapped_event_polygon_cell" else
                        "H3 cell center falls within 300 m of a cited NASA COOLR manual point; point is an initiation-zone proxy."
                        if event.label == 1 and event.event_record_type == "cited_manual_point_buffer_cell" else
                        event.label_quality if event.label in (0, 1) else
                        "Event presence is not a cell-level failure observation; label remains null."
                    ),
                    "event_group_id": event.event_group_id,
                    "mapping_method": event.mapping_method,
                })
                if not reused.get("h3_cell_id"):
                    reused["h3_cell_id"] = event.h3_cell_id or str(
                        (getattr(h3, "latlng_to_cell", None) or getattr(h3, "geo_to_h3"))(event.latitude, event.longitude, 9)
                    )
                return reused
            if event.event_record_type == "temporal_absence_sample":
                parent_id = event.event_id.removesuffix(":tminus60")
                original = existing_by_event.get(parent_id)
                if original is not None:
                    return await enrich_temporal_control(event, original, client, semaphore)
            return await enrich_one(event, client, dem, semaphore)

        rows = await asyncio.gather(*(build_one(event) for event in catalog))

    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError("Parquet output requires pyarrow; install requirements-ml.txt") from exc
    table = pa.Table.from_pylist(rows)
    group_array = pa.array([row.get("event_group_id") for row in rows], type=pa.string())
    if "event_group_id" in table.column_names:
        table = table.set_column(table.schema.get_field_index("event_group_id"), "event_group_id", group_array)
    else:
        table = table.append_column("event_group_id", group_array)
    metadata = dict(table.schema.metadata or {})
    metadata[b"label_policy"] = b"y=1: H3 centers inside cited manually mapped NASA COOLR polygons or within a 300 m buffer of a cited manual COOLR point (point is an approximate initiation-zone proxy, not a crown polygon). y=0: matched event-cell 60-day pre-event inventory-absence pseudo-negatives, not surveyed or verified intact terrain. Other event-only candidates remain unlabeled."
    metadata[b"dataset_scope"] = b"Event and matched temporal-control observations; current imagery is not used as historical imagery."
    metadata[b"catalog_sources"] = b"NASA COOLR cited event points and manually mapped polygons, NASA EONET, OpenStreetMap Overpass, NASA GLC fallback, and user-provided candidate seeds; source columns distinguish them."
    table = table.replace_schema_metadata(metadata)
    pq.write_table(table, output, compression="zstd")
    labels = [row.get("label") for row in rows]
    summary = {
        "output": str(output), "row_count": len(rows), "catalog_candidate_count": len(catalog),
        "nasa_glc_candidate_count": len(glc), "eonet_candidate_count": len(live_events), "osm_candidate_count": len(osm),
        "nasa_coolr_candidate_count": len(coolr),
        "nasa_coolr_manual_polygon_cell_count": len(polygons),
        "osm_source_status": osm_status,
        "eonet_source_status": {
            category: {
                "available": not isinstance(result, BaseException),
                "geometry_count": len(result) if not isinstance(result, BaseException) else 0,
                "error": type(result).__name__ if isinstance(result, BaseException) else None,
            }
            for category, result in zip(("landslides", "wildfires"), live_tasks)
        },
        "labeled_positive_rows": sum(value == 1 for value in labels),
        "labeled_negative_rows": sum(value == 0 for value in labels),
        "distinct_positive_source_events": len({row.get("source_event_id") for row in rows if row.get("label") == 1 and row.get("source_event_id")}),
        "positive_event_dates": sorted({row.get("event_date") for row in rows if row.get("label") == 1 and row.get("event_date")}),
        "positive_regimes": sorted({row.get("regime") for row in rows if row.get("label") == 1 and row.get("regime")}),
        "unlabeled_event_candidate_rows": sum(value is None for value in labels),
        "distinct_dated_external_source_events": len({
            item.source_event_id or item.event_id for item in catalog
            if item.event_date and item.source_name != "user-provided project candidate list"
        }),
        "distinct_dated_nasa_coolr_events": len({
            item.source_event_id or item.event_id for item in catalog
            if item.event_date and item.source_name in ("NASA COOLR", "NASA COOLR manually mapped polygon")
        }),
        "regimes": sorted({item.regime for item in catalog}),
        "complete_14_feature_rows": sum(all(row.get(name) is not None and math.isfinite(float(row[name])) for name in FEATURE_NAMES) for row in rows),
        "temporal_absence_controls": len(temporal_controls),
        "label_policy": "Cited manual NASA COOLR polygon H3 centers and H3 centers within 300 m of cited manual COOLR points are positive inventory proxies. Matched 60-day pre-event samples are inventory-absence pseudo-negatives, not surveyed or verified intact terrain. Other event points remain unlabeled.",
        "provenance_policy": "Rows retain field-level physics-feature provenance; temporal controls are explicitly marked pseudo-negative and unverified.",
    }
    # Freeze/evaluate the multi-regime split only AFTER report enrichment. The base
    # catalog alone may not contain the frozen seismic/cryosphere source records.
    metadata_path = output.with_suffix(".metadata.json")
    previous = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
    summary["event_group_holdout"] = {
        **previous.get("event_group_holdout", {}), "status": "pending_audit_refresh",
    }
    from backend.release import portable_payload
    metadata_path.write_text(json.dumps(portable_payload(summary), indent=2), encoding="utf-8")
    from backend.ml.audit_dataset import audit_refresh
    return await audit_refresh(output)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="data/real_global_landslide_observations.parquet")
    parser.add_argument("--audit-refresh", action="store_true", help="refresh stored physics, add sourced regime reports and simulated stability controls")
    parser.add_argument("--max-events", type=int, default=450)
    parser.add_argument("--timeout", type=float, default=15.0)
    args = parser.parse_args()
    if args.max_events < 1:
        parser.error("--max-events must be at least 1")
    if args.audit_refresh:
        from backend.ml.audit_dataset import audit_refresh
        summary = asyncio.run(audit_refresh(Path(args.output)))
        return 0
    summary = asyncio.run(build_dataset((ROOT / args.output).resolve() if not Path(args.output).is_absolute() else Path(args.output), args.max_events, args.timeout))
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
