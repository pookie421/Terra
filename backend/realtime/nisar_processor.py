"""NISAR discovery and optional, explicitly described L-SAR raster subsets.

CMR discovery is public. Product data need Earthdata credentials or local files.
GUNW coherence is dimensionless; a repeat-pair phase is not an annual velocity.
GCOV HV/HH power ratio is an uncalibrated radar index, not soil water content.
Source: https://nisar-docs.asf.alaska.edu/earthaccess/ and /products-overview/.

TERRA_NISAR_MANIFEST (or local_manifest=path/dict) describes optional subsets:
  {"version":"nisar_subset_manifest_v1", "assets":[
    {"path":"coherence.tif", "observable":"coherence", "radar_band":"L",
     "units":"1", "granule_id":"...", "acquisition_utc":"...", "band":1},
    {"path":"subset.h5", "observable":"coherence", "radar_band":"L",
     "dataset":"/science/LSAR/GUNW/grids/frequencyA/unwrappedInterferogram/HH/coherenceMagnitude",
     "x_dataset":".../xCoordinates", "y_dataset":".../yCoordinates",
     "projection_dataset":".../projection", "units":"1", ...}
  ], "strength_calibration_by_h3":{...}}
HDF5 readers use the actual coordinate vectors/projection, never array indices
as latitude/longitude. LOS velocity requires a documented multi-acquisition
timeseries, units, dates and reference; no velocity is inferred from GUNW phase.
Local NISAR HDF5 products in TERRA_NISAR_DIR (default data/nisar) are recognized
for native coherence and HV/HH only. GeoTIFF semantics need a manifest.
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
from statistics import median
import time
from urllib.parse import urljoin, urlsplit, urlunsplit

from .nasa_lhasa import _cells, _finite
from .natural_events import _point_in_ring
from .nisar_harmony import allowed_result_url, measured_subsets

CMR = 'https://cmr.earthdata.nasa.gov/search/granules.json'
CMR_UMM = 'https://cmr.earthdata.nasa.gov/search/granules.umm_json'
COLLECTIONS = ('NISAR_L2_GUNW_V1', 'NISAR_L2_GCOV_V1',
    'NISAR_L2_GUNW_PROVISIONAL_V1', 'NISAR_L2_GCOV_PROVISIONAL_V1',
    'NISAR_L2_GUNW_BETA_V1', 'NISAR_L2_GCOV_BETA_V1')
MAX_GRANULES = 12
MAX_REMOTE_BYTES = 16 * 1024 * 1024
FIELDS = {'coherence': 'nisar_coherence', 'los_velocity': 'nisar_los_velocity_mm_yr',
          'polarimetric_ratio': 'nisar_polarimetric_moisture_index',
          'los_displacement': 'nisar_los_displacement_mm'}


def _date(value):
    try:
        result = datetime.fromisoformat(str(value).replace('Z', '+00:00'))
        return result if result.tzinfo else None
    except (ValueError, TypeError):
        return None


def _latest(values):
    dates = [_date(value) for value in values]
    dates = [value for value in dates if value is not None]
    return max(dates).astimezone(timezone.utc).isoformat().replace('+00:00', 'Z') if dates else None


def _url(value):
    try:
        parts = urlsplit(str(value))
        return urlunsplit((parts.scheme, parts.netloc.rsplit('@', 1)[-1], parts.path, '', '')) if parts.scheme == 'https' else None
    except ValueError:
        return None


def _allowed_data_host(value):
    parts = urlsplit(value)
    host = (parts.hostname or '').lower()
    return (parts.scheme == 'https' and not parts.username and not parts.password and
            (host.endswith('.nasa.gov') or host.endswith('.asf.alaska.edu')))


def _bbox(latitude, longitude, bbox):
    if bbox is None:
        south, north = max(-90., latitude - .05), min(90., latitude + .05)
        west, east = longitude - .05, longitude + .05
    elif isinstance(bbox, dict):
        west, south, east, north = [float(bbox[k]) for k in ('west', 'south', 'east', 'north')]
    else:
        west, south, east, north = map(float, bbox)
    if not all(math.isfinite(v) for v in (west, south, east, north)) or not -90 <= south <= north <= 90:
        raise ValueError('Invalid NISAR bounding box')
    if east - west >= 360:
        return [(-180., south, 180., north)]
    west, east = (west + 180) % 360 - 180, (east + 180) % 360 - 180
    return [(west, south, east, north)] if west <= east else [(west, south, 180., north), (-180., south, east, north)]


def _coverage(entry):
    boxes, polygons = [], []
    for box in entry.get('boxes') or []:
        try:
            south, west, north, east = map(float, box.split())
            boxes.append([west, south, east, north])
        except (ValueError, AttributeError):
            continue
    for group in entry.get('polygons') or []:
        for ring in group:
            try:
                values = list(map(float, ring.split()))
                polygons.append([[values[i + 1], values[i]] for i in range(0, len(values) - 1, 2)])
            except (ValueError, AttributeError):
                continue
    return {'boxes': boxes, 'polygons': polygons}


def _covers(granule, lat, lon):
    geometry = granule.get('spatial_coverage') or {}
    for ring in geometry.get('polygons') or []:
        if not ring:
            continue
        # Unwrap consecutive vertices, rather than independently about the
        # query point (which would make a dateline swath appear almost global).
        unwrapped = [ring[0]]
        for x, y in ring[1:]:
            previous = unwrapped[-1][0]
            unwrapped.append([previous + ((x - previous + 180) % 360 - 180), y])
        query_lon = unwrapped[0][0] + ((lon - unwrapped[0][0] + 180) % 360 - 180)
        if _point_in_ring(query_lon, lat, unwrapped):
            return True
    if geometry.get('polygons'):
        return False  # A bounding rectangle must not override an exact swath.
    for west, south, east, north in geometry.get('boxes') or []:
        if south <= lat <= north and (west <= lon <= east if west <= east else lon >= west or lon <= east):
            return True
    return False if geometry.get('boxes') or geometry.get('polygons') else None


def parse_granule(entry):
    """Preserve real identifiers/time/URLs; absent direction stays unknown."""
    links = [link for link in entry.get('links') or [] if not link.get('inherited')]
    browse = next((_url(x.get('href')) for x in links if 'browse' in str(x.get('rel', '')).lower() and _url(x.get('href'))), None)
    opendap = next((_url(x.get('href')) for x in links if 'opendap' in (str(x.get('title', '')) + str(x.get('href', ''))).lower() and _url(x.get('href'))), None)
    data_links = [_url(x.get('href')) for x in links if 'data#' in str(x.get('rel', ''))]
    direction = entry.get('orbit_direction')
    for orbit in entry.get('orbit_calculated_spatial_domains') or []:
        direction = direction or orbit.get('orbit_direction')
    return {'granule_id': entry.get('producer_granule_id') or entry.get('title') or entry.get('id'),
        'cmr_concept_id': entry.get('id'), 'collection_concept_id': entry.get('collection_concept_id'),
        'dataset_id': entry.get('dataset_id'), 'time_start': entry.get('time_start'),
        'time_end': entry.get('time_end'), 'updated': entry.get('updated'), 'orbit_direction': direction,
        'collection_short_name': None, 'product_version': None, 'frequency_subbands': [], 'polarizations': {},
        'browse_url': browse, 'opendap_url': opendap, 'data_urls': [u for u in data_links if u],
        'spatial_coverage': _coverage(entry), 'measurement_status': 'METADATA_ONLY'}


def _manifest(local_manifest):
    path = local_manifest or os.environ.get('TERRA_NISAR_MANIFEST')
    root = Path(os.environ.get('TERRA_NISAR_DIR', str(Path(__file__).resolve().parents[2] / 'data/nisar')))
    if isinstance(path, dict):
        return path, root, None
    path = Path(path) if path else root / 'manifest.json'
    if path.exists():
        if path.stat().st_size > 1024 * 1024:
            raise ValueError('NISAR manifest exceeds 1 MiB')
        content = path.read_bytes()
        payload = json.loads(content)
        if payload.get('version') != 'nisar_subset_manifest_v1':
            raise ValueError('Unsupported NISAR subset manifest version')
        return payload, path.parent, hashlib.sha256(content).hexdigest()
    # Native HDF5 semantic paths are self-describing; arbitrary GeoTIFFs are not.
    assets = []
    for candidate in sorted(root.glob('*.h5'))[:2] if root.exists() else []:
        try:
            import h5py
            with h5py.File(candidate, 'r') as handle:
                names = []
                handle.visititems(lambda name, obj: names.append(name) if isinstance(obj, h5py.Dataset) else None)
                dates = {}
                for name in names:
                    if '/identification/' in name and name.rsplit('/', 1)[-1] in (
                            'zeroDopplerStartTime', 'zeroDopplerEndTime',
                            'referenceZeroDopplerStartTime', 'secondaryZeroDopplerStartTime'):
                        value = handle[name][()]
                        dates[name.rsplit('/', 1)[-1]] = value.decode() if isinstance(value, bytes) else str(value)
                acquisition = dates.get('zeroDopplerEndTime') or dates.get('referenceZeroDopplerStartTime')
                for name in names:
                    if not name.startswith('science/LSAR/') or '/grids/' not in name:
                        continue
                    if name.endswith('/coherenceMagnitude'):
                        grid = name[:name.index('/grids/') + len('/grids/')] + name.split('/grids/')[1].split('/')[0]
                        assets.append({'path': str(candidate.resolve()), 'observable': 'coherence', 'radar_band': 'L',
                            'dataset': '/' + name, 'x_dataset': '/' + grid + '/xCoordinates',
                            'y_dataset': '/' + grid + '/yCoordinates', 'projection_dataset': '/' + grid + '/projection',
                            'units': '1', 'granule_id': candidate.stem, 'acquisition_utc': acquisition,
                            'time_start': dates.get('secondaryZeroDopplerStartTime') or dates.get('zeroDopplerStartTime'),
                            'time_end': acquisition, 'metadata_origin': 'SELF_DESCRIBING_NISAR_HDF5'})
                    elif name.endswith('/HVHV') and name[:-4] + 'HHHH' in handle:
                        grid = name.rsplit('/', 1)[0]
                        assets.append({'path': str(candidate.resolve()), 'observable': 'polarimetric_ratio', 'radar_band': 'L',
                            'dataset': '/' + name, 'denominator_dataset': '/' + name[:-4] + 'HHHH',
                            'x_dataset': '/' + grid + '/xCoordinates', 'y_dataset': '/' + grid + '/yCoordinates',
                            'projection_dataset': '/' + grid + '/projection', 'units': '1', 'granule_id': candidate.stem,
                            'acquisition_utc': acquisition, 'time_start': dates.get('zeroDopplerStartTime'),
                            'time_end': acquisition, 'metadata_origin': 'SELF_DESCRIBING_NISAR_HDF5'})
        except ImportError:
            raise ValueError('Install optional h5py to read local NISAR HDF5 subsets')
    return {'assets': assets}, root, None


def _validate_asset(asset):
    observable = asset.get('observable')
    if observable not in FIELDS or asset.get('radar_band') != 'L':
        raise ValueError('Only explicitly identified L-SAR observables are accepted')
    unit = str(asset.get('units', '')).lower().replace(' ', '')
    if observable in ('coherence', 'polarimetric_ratio') and unit not in ('1', 'unitless', 'dimensionless'):
        raise ValueError('Radar index/coherence units must be dimensionless')
    if observable == 'los_velocity':
        start, end = _date(asset.get('time_start')), _date(asset.get('time_end'))
        if (asset.get('derivation') != 'documented_timeseries' or int(asset.get('observation_count', 0)) < 3 or
            start is None or end is None or end <= start or not asset.get('reference') or not asset.get('sign_convention')):
            raise ValueError('LOS annual velocity requires a documented multi-acquisition timeseries and reference')
        if unit not in ('mm/yr', 'mm/year', 'm/yr', 'm/year'):
            raise ValueError('Unsupported LOS velocity units')
    if observable == 'los_displacement' and (unit not in ('m', 'mm') or not asset.get('reference') or
            not _date(asset.get('time_start')) or not _date(asset.get('time_end'))):
        raise ValueError('Displacement requires explicit pair dates, units and reference; phase is not accepted as velocity')
    return observable, 1000. if unit in ('m', 'm/yr', 'm/year') else 1.


def _valid_value(value, asset, scale, missing=None):
    value = _finite(value)
    if value is None or missing is not None and value == missing:
        return None
    value *= scale
    if asset['observable'] == 'coherence' and not 0 <= value <= 1:
        return None
    if asset['observable'] == 'polarimetric_ratio' and value < 0:
        return None
    return value


def _sample_raster(source, asset, points):
    import rasterio
    from rasterio.warp import transform
    observable, scale = _validate_asset(asset)
    values, support = {}, {}
    with rasterio.open(source) as dataset:
        if dataset.crs is None:
            raise ValueError('NISAR subset lacks a coordinate reference system')
        band = int(asset.get('band', 1))
        if not 1 <= band <= dataset.count:
            raise ValueError('NISAR band index outside raster')
        xs, ys = transform('EPSG:4326', dataset.crs, [p['longitude'] for p in points], [p['latitude'] for p in points])
        coordinates = list(zip(xs, ys))
        for point, xy, sample in zip(points, coordinates, dataset.sample(coordinates, indexes=band, masked=True)):
            if not (dataset.bounds.left <= xy[0] < dataset.bounds.right and dataset.bounds.bottom < xy[1] <= dataset.bounds.top):
                continue
            raw = sample[0]
            if getattr(raw, 'mask', False):
                continue
            raw = float(raw) * dataset.scales[band - 1] + dataset.offsets[band - 1]
            value = _valid_value(raw, asset, scale)
            if value is not None:
                values[point['h3_index']] = value
        support = {'crs': str(dataset.crs), 'pixel_size_native_units': list(dataset.res),
                   'interpolation': 'nearest source pixel', 'raster_width': dataset.width, 'raster_height': dataset.height,
                   'transform_native': list(dataset.transform), 'bounds_native': list(dataset.bounds)}
    return observable, values, support


def _sample_hdf5(path, asset, points):
    import h5py
    import numpy as np
    from pyproj import Transformer
    observable, scale = _validate_asset(asset)
    with h5py.File(path, 'r') as handle:
        data = handle[asset['dataset']]
        x = np.asarray(handle[asset['x_dataset']][:], dtype=float)
        y = np.asarray(handle[asset['y_dataset']][:], dtype=float)
        if len(x) < 2 or len(y) < 2 or data.shape != (len(y), len(x)):
            raise ValueError('NISAR HDF5 geolocation grid shape mismatch')
        if not (np.all(np.diff(x) > 0) or np.all(np.diff(x) < 0)) or not (np.all(np.diff(y) > 0) or np.all(np.diff(y) < 0)):
            raise ValueError('NISAR HDF5 coordinates must be monotonic')
        epsg = asset.get('epsg')
        if epsg is None:
            projection = handle[asset['projection_dataset']]
            epsg = projection.attrs.get('epsg_code') or int(projection[()])
        transformer = Transformer.from_crs(4326, int(epsg), always_xy=True)
        dx, dy = abs(x[1] - x[0]), abs(y[1] - y[0])
        xmin, xmax = float(x.min()) - dx/2, float(x.max()) + dx/2
        ymin, ymax = float(y.min()) - dy/2, float(y.max()) + dy/2
        missing = _finite(data.attrs.get('_FillValue', data.attrs.get('missing_value')))
        native_units = data.attrs.get('units')
        if isinstance(native_units, bytes):
            native_units = native_units.decode()
        if observable in ('coherence', 'polarimetric_ratio') and native_units is not None and str(native_units).lower() not in ('1', 'unitless', 'dimensionless'):
            raise ValueError('HDF5 native units disagree with dimensionless radar observable')
        denominator = handle[asset['denominator_dataset']] if asset.get('denominator_dataset') else None
        values = {}
        for point in points:
            px, py = transformer.transform(point['longitude'], point['latitude'])
            if not xmin <= px <= xmax or not ymin <= py <= ymax:
                continue
            col, row = int(np.abs(x - px).argmin()), int(np.abs(y - py).argmin())
            raw = _finite(data[row, col])
            if raw is not None and missing is not None and raw == missing:
                raw = None
            if raw is not None:
                raw = raw * float(data.attrs.get('scale_factor', 1.)) + float(data.attrs.get('add_offset', 0.))
            if denominator is not None:
                den = _finite(denominator[row, col])
                if den is not None:
                    den = den * float(denominator.attrs.get('scale_factor', 1.)) + float(denominator.attrs.get('add_offset', 0.))
                raw = raw / den if raw is not None and den is not None and den > 0 else None
            value = _valid_value(raw, asset, scale)
            if value is not None:
                values[point['h3_index']] = value
        return observable, values, {'crs': 'EPSG:{}'.format(epsg), 'pixel_size_native_units': [dx, dy],
            'interpolation': 'nearest source pixel', 'dataset': asset['dataset'], 'container': 'HDF5'}


async def _read_asset(client, asset, root, points, token, timeout_s):
    _validate_asset(asset)
    url = asset.get('url')
    if url:
        harmony = asset.get('metadata_origin') == 'NASA_HARMONY_OFFICIAL_CMR_VARIABLE_SUBSET'
        if not token or not (_allowed_data_host(url) or harmony and allowed_result_url(url)):
            raise ValueError('Authenticated raster reading requires token and an allowlisted NASA/ASF HTTPS host')
        async def download(target, limit):
            for _ in range(4):
                if not (_allowed_data_host(target) or harmony and allowed_result_url(target)):
                    raise ValueError('Untrusted NISAR raster result host')
                if urlsplit(target).hostname == 'urs.earthdata.nasa.gov':
                    raise PermissionError('Earthdata authorization is required for the raster')
                content = bytearray()
                # Harmony's service-results endpoint can redirect to a signed
                # CloudFront/S3 URL. Re-evaluate every host and drop the token.
                headers = {'Authorization': 'Bearer ' + token} if _allowed_data_host(target) else {}
                async with client.stream('GET', target, headers=headers, timeout=timeout_s, follow_redirects=False) as response:
                    if response.status_code in (301, 302, 303, 307, 308):
                        location = response.headers.get('location')
                        if not location:
                            raise ValueError('Raster redirect has no destination')
                        target = urljoin(target, location)
                        continue
                    response.raise_for_status()
                    if 'text/html' in response.headers.get('content-type', '').lower():
                        raise PermissionError('Authentication HTML is not a NISAR raster')
                    size = response.headers.get('content-length')
                    if size and int(size) > limit:
                        raise ValueError('Remote subset exceeds 16 MiB; provide a local windowed subset')
                    async for part in response.aiter_bytes():
                        content.extend(part)
                        if len(content) > limit:
                            raise ValueError('Remote subset exceeds 16 MiB; provide a local windowed subset')
                return bytes(content)
            raise ValueError('Too many raster redirects')
        content = await download(url, MAX_REMOTE_BYTES)
        denominator_url = asset.get('denominator_url')
        denominator = await download(denominator_url, MAX_REMOTE_BYTES - len(content)) if denominator_url else None
        from rasterio.io import MemoryFile
        def sample():
            with MemoryFile(content) as memory:
                result = _sample_raster(memory.name, asset, points)
            if denominator is None:
                return result
            with MemoryFile(denominator) as memory:
                denominator_result = _sample_raster(memory.name, asset, points)
            for key in ('crs', 'transform_native', 'raster_width', 'raster_height'):
                if result[2][key] != denominator_result[2][key]:
                    raise ValueError('GCOV power channels do not share the same native pixel grid')
            ratios = {identity: value / denominator_result[1][identity] for identity, value in result[1].items()
                if identity in denominator_result[1] and denominator_result[1][identity] > 0}
            return result[0], ratios, dict(result[2], derivation='HVHV / HHHH from the same granule and native pixel; native mask union')
        result = await asyncio.to_thread(sample)
        digest, verification = hashlib.sha256(content).hexdigest(), 'ACTUAL_DOWNLOADED_SUBSET_BYTES'
        source = _url(url)
    else:
        path = Path(asset.get('path', ''))
        path = path if path.is_absolute() else root / path
        if not path.is_file():
            raise ValueError('Local NISAR subset does not exist')
        function = _sample_hdf5 if asset.get('dataset') else _sample_raster
        result = await asyncio.to_thread(function, path, asset, points)
        if path.stat().st_size <= MAX_REMOTE_BYTES:
            digest = await asyncio.to_thread(lambda: hashlib.sha256(path.read_bytes()).hexdigest())
            verification = 'ACTUAL_LOCAL_SUBSET_BYTES'
        else:
            digest, verification = None, 'LARGE_CONTAINER_NOT_HASHED_SAMPLE_VALUES_HASHED_SEPARATELY'
        source = path.name
    if digest is not None and asset.get('sha256') and digest != asset['sha256']:
        raise ValueError('NISAR subset checksum mismatch')
    observable, values, support = result
    provenance = {'granule_id': asset.get('granule_id'), 'source': source, 'observable': observable,
        'access_mode': 'AUTHENTICATED_RASTER_MEASURED' if url else 'LOCAL_RASTER_MEASURED',
        'metadata_origin': asset.get('metadata_origin', 'OPERATOR_SUBSET_MANIFEST'),
        'radar_band': 'L', 'units': 'mm/yr' if observable == 'los_velocity' else ('mm' if observable == 'los_displacement' else '1'),
        'acquisition_utc': asset.get('acquisition_utc'), 'time_start': asset.get('time_start'), 'time_end': asset.get('time_end'),
        'reference': asset.get('reference'), 'sign_convention': asset.get('sign_convention'),
        'sha256': digest, 'hash_verification': verification,
        'sample_values_sha256': hashlib.sha256(json.dumps(values, sort_keys=True, allow_nan=False).encode()).hexdigest(),
        'support': support, 'interpretation': 'UNCALIBRATED_HV_HH_RADAR_RATIO_NOT_SOIL_WATER' if observable == 'polarimetric_ratio' else 'RETRIEVED_RADAR_OBSERVABLE'}
    for key in ('collection_concept_id', 'collection_short_name', 'granule_concept_id', 'product_maturity',
            'product_version', 'variable_concept_ids', 'source_dataset_paths', 'harmony_job_id', 'quality_scope'):
        if asset.get(key) is not None:
            provenance[key] = asset[key]
    if url and asset.get('denominator_url'):
        provenance['denominator_source'] = _url(asset['denominator_url'])
        provenance['denominator_sha256'] = hashlib.sha256(denominator).hexdigest()
        provenance['derivation'] = 'HVHV / HHHH linear power; same granule and native pixel; HH>0 and HV>=0'
    return observable, values, provenance


async def fetch_nisar(client, latitude, longitude, *, cells=None, bbox=None, timeout_s=3.5,
                      cache=None, local_manifest=None, earthdata_token=None):
    """Discover latest intersecting NISAR products; optional bounded subset reads."""
    latitude, longitude = float(latitude), float(longitude)
    if not math.isfinite(latitude) or not math.isfinite(longitude) or not -90 <= latitude <= 90 or not -180 <= longitude <= 180:
        raise ValueError('Invalid NISAR scan coordinate')
    if not 0 < timeout_s <= 30:
        raise ValueError('Invalid NISAR timeout budget')
    points = _cells(cells, latitude, longitude)
    rectangles = _bbox(latitude, longitude, bbox)
    started, deadline = datetime.now(timezone.utc).isoformat(), time.monotonic() + timeout_s
    token = earthdata_token if earthdata_token is not None else os.environ.get('EARTHDATA_TOKEN')
    local_root = Path(os.environ.get('TERRA_NISAR_DIR', str(Path(__file__).resolve().parents[2] / 'data/nisar')))
    has_local = bool(local_manifest or os.environ.get('TERRA_NISAR_MANIFEST') or local_root.exists())
    metadata_deadline = time.monotonic() + timeout_s * (.6 if has_local or token else 1.)
    manifest_task = asyncio.create_task(asyncio.to_thread(_manifest, local_manifest))
    errors, granules, hits = [], [], []
    metadata_live = False

    async def query(rectangle, collections=COLLECTIONS, page_size=MAX_GRANULES):
        params = [('short_name[]', name) for name in collections]
        params += [('bounding_box', ','.join(map(str, rectangle))), ('page_size', str(page_size)), ('sort_key[]', '-start_date')]
        response = await client.get(CMR, params=params, timeout=timeout_s)
        response.raise_for_status()
        payload = response.json()
        if payload.get('errors'):
            raise ValueError('CMR rejected metadata query')
        if not isinstance(payload.get('feed'), dict) or not isinstance(payload['feed'].get('entry', []), list):
            raise ValueError('CMR metadata schema is missing a valid feed')
        return payload.get('feed', {}).get('entry') or [], response.headers.get('CMR-Hits')

    try:
        async def discover():
            return await asyncio.gather(*(query(rectangle) for rectangle in rectangles))
        discovered = await asyncio.wait_for(discover(), timeout=max(.001, metadata_deadline - time.monotonic()))
        metadata_live = True
        unique = {}
        for entries, count in discovered:
            hits.append(int(count) if count and str(count).isdigit() else None)
            for entry in entries:
                item = parse_granule(entry)
                if not item['cmr_concept_id'] or not item['granule_id']:
                    errors.append('MALFORMED_CMR_GRANULE_IDENTITY')
                    continue
                unique[item['cmr_concept_id'] or item['granule_id']] = item
        granules = sorted(unique.values(), key=lambda g: g.get('time_start') or '', reverse=True)[:MAX_GRANULES]
        # GCOV has more recent acquisitions than repeat-pair GUNW. A combined
        # newest-12 search can otherwise hide every available interferogram.
        # Reserve a bounded GUNW discovery query when the first page is GCOV-only.
        if token and granules and not any('GUNW' in str(g['granule_id']) for g in granules) and metadata_deadline - time.monotonic() > .1:
            try:
                extra = await asyncio.wait_for(asyncio.gather(*(query(rectangle,
                    tuple(name for name in COLLECTIONS if 'GUNW' in name), MAX_GRANULES // 2)
                    for rectangle in rectangles)), timeout=max(.001, metadata_deadline - time.monotonic()))
                for entries, _ in extra:
                    for entry in entries:
                        item = parse_granule(entry)
                        if item['cmr_concept_id'] and item['granule_id']:
                            unique[item['cmr_concept_id']] = item
                ordered = sorted(unique.values(), key=lambda g: g.get('time_start') or '', reverse=True)
                gunw = [g for g in ordered if 'GUNW' in str(g['granule_id'])][:MAX_GRANULES // 2]
                others = [g for g in ordered if 'GUNW' not in str(g['granule_id'])][:MAX_GRANULES - len(gunw)]
                granules = sorted(gunw + others, key=lambda g: g.get('time_start') or '', reverse=True)
            except Exception:
                errors.append('GUNW_DISCOVERY_UNAVAILABLE')
        # Feed JSON often omits orbital direction. UMM supplies it when present.
        if granules and metadata_deadline - time.monotonic() > .1 and any(not g['orbit_direction'] for g in granules):
            params = [('concept_id[]', g['cmr_concept_id']) for g in granules if g['cmr_concept_id']]
            params.append(('page_size', str(MAX_GRANULES)))
            try:
                response = await asyncio.wait_for(client.get(CMR_UMM, params=params,
                    timeout=max(.001, metadata_deadline - time.monotonic())), timeout=max(.001, metadata_deadline - time.monotonic()))
                response.raise_for_status()
                byid = {g['cmr_concept_id']: g for g in granules}
                for item in response.json().get('items') or []:
                    target = byid.get((item.get('meta') or {}).get('concept-id'))
                    if target is None:
                        continue
                    umm = item.get('umm') or {}
                    target['collection_short_name'] = (umm.get('CollectionReference') or {}).get('ShortName')
                    orbital = ((umm.get('SpatialExtent') or {}).get('OrbitParameters') or {})
                    target['orbit_direction'] = orbital.get('OrbitDirection')
                    for attribute in umm.get('AdditionalAttributes') or []:
                        name = str(attribute.get('Name', '')).upper()
                        values = attribute.get('Values') or []
                        if name.replace('_', '') in ('ORBITDIRECTION', 'ASCENDINGDESCENDING'):
                            target['orbit_direction'] = next(iter(attribute.get('Values') or []), None)
                        elif name == 'PRODUCT_VERSION':
                            target['product_version'] = next(iter(values), None)
                        elif name == 'FREQUENCIES':
                            target['frequency_subbands'] = values
                        elif name in ('FREQUENCY_A_POLARIZATION', 'FREQUENCY_B_POLARIZATION'):
                            target['polarizations'][name.split('_')[1]] = values
            except Exception:
                errors.append('ORBIT_METADATA_ENRICHMENT_UNAVAILABLE')
    except Exception as exc:
        errors.append('CMR_METADATA_' + type(exc).__name__.upper())

    manifest, root, manifest_sha = {}, Path('.'), None
    try:
        manifest, root, manifest_sha = await asyncio.wait_for(manifest_task, timeout=max(.001, deadline - time.monotonic()))
    except Exception as exc:
        errors.append('LOCAL_MANIFEST_' + type(exc).__name__.upper() + ': ' + str(exc) if isinstance(exc, ValueError) else 'LOCAL_MANIFEST_UNAVAILABLE')
    assets = list(manifest.get('assets') or [])[:8]
    # CMR coherence COG sidecars are identifiable; full multi-GB HDF5 is not
    # downloaded during a point scan. A size limit is enforced before reading.
    if token and not assets:
        for granule in granules:
            for url in granule['data_urls']:
                if 'coherence' in url.lower() and url.lower().endswith(('.tif', '.tiff')) and _allowed_data_host(url):
                    assets.append({'url': url, 'observable': 'coherence', 'radar_band': 'L', 'units': '1',
                        'granule_id': granule['granule_id'], 'acquisition_utc': granule['time_end']})
                    break
            if assets:
                break
    by_h3 = {}
    for point in points:
        covering = [g for g in granules if _covers(g, point['latitude'], point['longitude']) is True]
        known_coverage = metadata_live and all(_covers(g, point['latitude'], point['longitude']) is not None for g in granules)
        by_h3[point['h3_index']] = {'h3_index': point['h3_index'], **{field: None for field in FIELDS.values()},
            'nisar_granule_count': len(covering) if known_coverage else None,
            'nisar_latest_acquisition_utc': _latest(g['time_end'] or g['time_start'] for g in covering),
            'measurement_provenance': {}, 'radar_constraint_status': 'NO_MEASURED_RADAR_CONSTRAINT',
            'strength_relation_status': 'NO_CALIBRATED_STRENGTH_RELATION'}
    measured_assets = []
    async def sample_assets(descriptors):
        for asset in descriptors:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                errors.append('RASTER_READ_AGGREGATE_TIMEOUT')
                break
            try:
                observable, values, provenance = await asyncio.wait_for(
                    _read_asset(client, asset, root, points, token, remaining), timeout=remaining)
                provenance['granule_identity_in_cmr_results'] = any(
                    g['granule_id'] == provenance.get('granule_id') for g in granules)
                if values:
                    measured_assets.append(provenance)
                for identity, value in values.items():
                    by_h3[identity][FIELDS[observable]] = value
                    by_h3[identity]['measurement_provenance'][observable] = provenance
                    by_h3[identity]['radar_constraint_status'] = 'MEASURED_RADAR_CONTEXT'
                    by_h3[identity]['nisar_latest_acquisition_utc'] = _latest([
                        by_h3[identity]['nisar_latest_acquisition_utc'], provenance.get('acquisition_utc'), provenance.get('time_end')])
            except Exception as exc:
                errors.append('RASTER_' + type(exc).__name__.upper() + (': ' + str(exc) if isinstance(exc, ValueError) else ''))
    await sample_assets(assets)
    missing_observables = [observable for observable in ('coherence', 'polarimetric_ratio')
        if not any(row[FIELDS[observable]] is not None for row in by_h3.values())]
    automatic_assets, measurement_access = [], {'method': 'LOCAL_OR_AUTHENTICATED_RASTER_SUBSET',
        'authentication_configured': bool(token), 'status': 'SUBSET_UNAVAILABLE' if token else 'AUTH_REQUIRED',
        'retry_after_seconds': None, 'jobs': [], 'message': 'The deployment needs Earthdata access to retrieve measured NISAR subsets.'
            if not token else 'Radar acquisition did not complete within this request budget.'}
    if missing_observables and deadline - time.monotonic() > .02:
        eligible_granules = [g for g in granules if any(
            _covers(g, p['latitude'], p['longitude']) is not False for p in points)]
        try:
            automatic_assets, measurement_access = await asyncio.wait_for(measured_subsets(
                client, eligible_granules, rectangles, token=token, deadline=max(time.monotonic(), deadline - .05),
                observables=missing_observables), timeout=max(.001, deadline - time.monotonic()))
            if not metadata_live and measurement_access['status'] == 'NO_PRODUCT_COVERAGE':
                measurement_access.update(status='SUBSET_UNAVAILABLE',
                    message='The product catalogue was unavailable; absent coverage has not been established.')
            raster_errors_before = len(errors)
            await sample_assets(automatic_assets)
            if len(errors) > raster_errors_before:
                measurement_access['sample_error_codes'] = errors[raster_errors_before:]
        except Exception as exc:
            measurement_access.update(status='SUBSET_UNAVAILABLE',
                message='The bounded raster request did not finish; no radar values were inferred.')
            errors.append('HARMONY_' + type(exc).__name__.upper())
    if measured_assets:
        measurement_access.update(status='MEASURED_AVAILABLE',
            message='Valid NISAR pixels were sampled. Acquisition age, native masks and observable provenance remain explicit.')
    elif automatic_assets:
        if measurement_access.get('sample_error_codes'):
            measurement_access.update(status='SUBSET_UNAVAILABLE',
                message='NASA prepared the subsets, but raster access or decoding failed. No radar values were inferred.')
        else:
            measurement_access.update(status='NO_VALID_PIXELS',
                message='NASA returned rasters, but this scan has no valid sampled pixels. Native missing values were retained.')
    observed = {}
    for observable, field in FIELDS.items():
        values = sorted(row[field] for row in by_h3.values() if row[field] is not None)
        observed[field] = median(values) if values else None
    measured_modes = {asset['access_mode'] for asset in measured_assets}
    mode = (next(iter(measured_modes)) if len(measured_modes) == 1 else 'MIXED_RASTER_MEASURED') if measured_modes else (
        'METADATA_LIVE' if metadata_live else 'UNAVAILABLE')
    latest = _latest([g['time_end'] or g['time_start'] for g in granules] +
                     [a.get('acquisition_utc') or a.get('time_end') for a in measured_assets])
    age = (datetime.now(timezone.utc) - _date(latest)).total_seconds() / 86400 if latest else None
    return {'provider': 'NASA-ISRO NISAR / ASF DAAC / NASA CMR', 'mode': mode,
        'fetched_at_utc': started, 'source_url': CMR, 'queried_collections': list(COLLECTIONS),
        'granules': granules, 'granule_count': len(granules), 'query_hits_by_bbox': hits,
        'granule_count_scope': 'Latest returned intersecting metadata records, capped at12; not all mission acquisitions',
        'latest_acquisition_utc': latest, 'latest_acquisition_age_days': age,
        'bbox_queries': rectangles, 'metadata_live': metadata_live,
        'cells': list(by_h3.values()), 'by_h3': by_h3, 'observables': observed,
        'observable_aggregation': 'Median valid sampled H3 value; see per-cell measurements for source support',
        'measured_assets': measured_assets, 'manifest_sha256': manifest_sha,
        'measurement_access': measurement_access,
        'strength_calibration_by_h3': {key: value for key, value in (manifest.get('strength_calibration_by_h3') or {}).items() if key in by_h3},
        'radar_band': 'L', 'wavelength_m_approx': .24, 's_band_status': 'NOT_ASSUMED_FROM_L_SAR_METADATA',
        'deformation_prior': {'mean_strain': 0., 'status': 'NEUTRAL_REFERENCE_UNCONSTRAINED_WHEN_UNMEASURED',
            'uncertainty': 'Existing broad material/structural reference uncertainty retained; no invented radar precision'},
        'errors': errors, 'limitations': ['Metadata footprints/acquisition times are not measured deformation.',
            'Coherence loss can reflect vegetation, geometry, noise or temporal decorrelation, not just landslide motion.',
            'HV/HH radar ratio is not a calibrated soil-moisture retrieval; pair phase is not an annual displacement rate.',
            'No generic shear-strength change is inferred from radar; any site-calibrated relation requires separate provenance.',
            'Beta/provisional/validated maturity is preserved in collection identity; finer grid spacing does not prove accuracy.']}
