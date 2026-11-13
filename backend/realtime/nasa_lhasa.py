"""NASA ImageServer sampling at native support, with a separately named proxy.

LHASA 2 is a trained XGBoost system, not an ARI/road/slope algebraic formula.
An outage never turns a project heuristic into an official NASA probability.
References: https://github.com/nasa/LHASA and doi:10.3389/feart.2021.640043.
"""
from __future__ import annotations

import asyncio
from collections import defaultdict
from datetime import datetime, timezone
import json
import math
import re
from typing import Mapping

BASE = "https://gis.earthdata.nasa.gov/portal/rest/services/Landslides/"
SERVICES = {"today": BASE + "LHASA_Hazard_Today/ImageServer",
            "tomorrow": BASE + "LHASA_Hazard_Tomorrow/ImageServer",
            "susceptibility": BASE + "Global_Landslide_Susceptibility/ImageServer"}
ARC_DEGREES = 1 / 120
MAX_SAMPLES = 1000


def _get(obj, name, default=None):
    return obj.get(name, default) if isinstance(obj, Mapping) else getattr(obj, name, default)


def _finite(value):
    try:
        result = float(value)
        return result if math.isfinite(result) else None
    except (ValueError, TypeError, OverflowError):
        return None


def _cells(cells, latitude=None, longitude=None):
    items = cells.values() if isinstance(cells, Mapping) else cells or []
    result = []
    for index, cell in enumerate(items):
        lat, lon = _finite(_get(cell, 'latitude')), _finite(_get(cell, 'longitude'))
        identity = _get(cell, 'cell_id') or _get(cell, 'h3_index') or str(index)
        if lat is None or lon is None or not -90 <= lat <= 90 or not -180 <= lon <= 180:
            raise ValueError('Invalid cell coordinates')
        result.append({'h3_index': str(identity), 'latitude': lat,
                       'longitude': ((lon + 180) % 360) - 180,
                       'slope_deg': _finite(_get(cell, 'slope_deg'))})
    if not result and latitude is not None and longitude is not None:
        return _cells([{'h3_index': 'scan_center', 'latitude': latitude, 'longitude': longitude}])
    return result


def _pixel(lat, lon, metadata=None):
    metadata = metadata or {}
    extent = metadata.get('extent') or {}
    dx = abs(_finite(metadata.get('pixelSizeX')) or ARC_DEGREES)
    dy = abs(_finite(metadata.get('pixelSizeY')) or ARC_DEGREES)
    west = _finite(extent.get('xmin'))
    north = _finite(extent.get('ymax'))
    west = -180. if west is None else west
    north = 90. if north is None else north
    col, row = math.floor((lon - west) / dx), math.floor((north - lat) / dy)
    return '{}:{}'.format(col, row), west + (col + .5) * dx, north - (row + .5) * dy


def _proxy(rows, weather):
    """Unvalidated index; project scales are exposed and never called LHASA2."""
    rain = _finite(_get(weather, 'rain_24h_mm'))
    antecedent = _finite(_get(weather, 'ari_7d_mm'))
    slopes = [r['slope_deg'] for r in rows if r['slope_deg'] is not None and 0 <= r['slope_deg'] <= 90]
    if rain is None or rain < 0 or antecedent is None or antecedent < 0 or not slopes:
        return None
    # Mean slope over scan-covered native bins, not outcome-selected steep cells.
    slope = sum(slopes) / len(slopes)
    return math.sin(math.radians(slope)) * ((1 - math.exp(-rain / 100.)) +
                                           (1 - math.exp(-antecedent / 100.))) / 2


async def fetch_lhasa(client, cells, *, latitude=None, longitude=None, weather=None,
                      timeout_s=3.5, cache=None):
    """Sample three products under one wall-clock budget, not per-H3 requests.

    ImageServer metadata and nearest-neighbor multipoint getSamples preserve
    the source grid, categorical susceptibility and zero/NoData distinction.
    ``probability`` is official service output only. ``reference_proxy_score``
    remains a disclosed project index when that official output is absent.
    """
    if not 0 < timeout_s <= 30:
        raise ValueError('Invalid LHASA timeout budget')
    rows = _cells(cells, latitude, longitude)
    errors, metadata, sampled = {}, {}, {}
    started = datetime.now(timezone.utc).isoformat()

    async def product(name):
        url = SERVICES[name]
        eligible = [r for r in rows if (-60 <= r['latitude'] <= (72 if name == 'susceptibility' else 60))]
        if not eligible:
            errors[name] = 'OUTSIDE_PRODUCT_LATITUDE_DOMAIN'
            return
        try:
            async def load_metadata():
                response = await client.get(url, params={'f': 'json'}, timeout=timeout_s)
                response.raise_for_status()
                payload = response.json()
                if payload.get('error'):
                    raise ValueError('ImageServer metadata error {}'.format(payload['error'].get('code', 'unknown')))
                if not isinstance(payload.get('extent'), dict) or not _finite(payload.get('pixelSizeX')) or not _finite(payload.get('pixelSizeY')):
                    raise ValueError('ImageServer native grid metadata is missing')
                sr = (payload.get('extent') or {}).get('spatialReference') or {}
                if sr.get('latestWkid', sr.get('wkid', 4326)) != 4326:
                    raise ValueError('Unsupported source CRS; cannot assume degree pixels')
                return payload
            meta = await cache.get_or_create(('lhasa-metadata', name), 3600, load_metadata) if cache else await load_metadata()
            metadata[name] = {k: meta.get(k) for k in ('name', 'description', 'serviceDescription',
                'pixelSizeX', 'pixelSizeY', 'extent', 'timeInfo', 'copyrightText')}
            groups = {}
            for cell in eligible:
                key, x, y = _pixel(cell['latitude'], cell['longitude'], meta)
                groups[key] = (x, y)
            if len(groups) > MAX_SAMPLES:
                raise ValueError('Scan exceeds the native sample limit; request a smaller sector')
            points = list(groups.values())
            response = await client.get(url + '/getSamples', params={
                'f': 'json', 'geometryType': 'esriGeometryMultipoint',
                'geometry': json.dumps({'points': points, 'spatialReference': {'wkid': 4326}}),
                'returnFirstValueOnly': 'true', 'interpolation': 'RSP_NearestNeighbor', 'outFields': '*'}, timeout=timeout_s)
            response.raise_for_status()
            payload = response.json()
            if payload.get('error'):
                raise ValueError('ImageServer samples error {}'.format(payload['error'].get('code', 'unknown')))
            samples = {}
            for item in payload.get('samples') or []:
                location = item.get('location') or {}
                x, y = _finite(location.get('x')), _finite(location.get('y'))
                if x is None or y is None:
                    continue  # Never assume partial responses retain input order.
                sr = location.get('spatialReference') or {}
                if sr.get('latestWkid', sr.get('wkid', 4326)) != 4326:
                    continue
                key = _pixel(y, x, meta)[0]
                value = _finite(item.get('value'))
                if value is None or value == -9999:
                    continue
                if name == 'susceptibility':
                    valid = value.is_integer() and 0 <= value <= 5
                    value = int(value)
                else:
                    valid = 0 <= value <= 1
                if valid and key in groups:
                    samples[key] = {'value': value, 'raster_id': item.get('rasterId'),
                                    'source_resolution': item.get('resolution'), 'attributes': item.get('attributes') or {}}
            sampled[name] = samples
            if not samples:
                errors[name] = 'NO_VALID_PIXELS_RETURNED'
        except asyncio.CancelledError:
            errors[name] = 'AGGREGATE_TIMEOUT'
            raise
        except Exception as exc:
            # No request URL/token echo from network exception text.
            errors[name] = '{}: {}'.format(type(exc).__name__, str(exc) if isinstance(exc, ValueError) else 'provider request failed')

    tasks = [asyncio.create_task(product(name)) for name in SERVICES]
    done, pending = await asyncio.wait(tasks, timeout=timeout_s)
    for task in pending:
        task.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)
    coarse = defaultdict(list)
    for row in rows:
        coarse[_pixel(row['latitude'], row['longitude'])[0]].append(row)
    output = []
    for row in rows:
        result = {k: v for k, v in row.items() if k != 'slope_deg'}
        keys = {name: _pixel(row['latitude'], row['longitude'], metadata.get(name))[0] for name in SERVICES}
        source = {name: (sampled.get(name) or {}).get(keys[name]) for name in SERVICES}
        probability = source['today']['value'] if source['today'] else None
        tomorrow = source['tomorrow']['value'] if source['tomorrow'] else None
        proxy = _proxy(coarse[_pixel(row['latitude'], row['longitude'])[0]], weather) if probability is None else None
        result.update(probability=probability, tomorrow_probability=tomorrow,
            susceptibility_class=source['susceptibility']['value'] if source['susceptibility'] else None,
            coarse_pixel_id=keys['today'], reference_proxy_score=proxy,
            provenance='LIVE_NASA_IMAGESERVER' if probability is not None else 'LHASA_REFERENCE_EMULATION',
            score_kind='NASA_SERVICE_RELATIVE_PROBABILITY' if probability is not None else 'UNVALIDATED_REFERENCE_INDEX',
            in_hazard_domain=-60 <= row['latitude'] <= 60, source_samples=source)
        output.append(result)
    descriptions = ' '.join(str(v.get('description') or '') + ' ' + str(v.get('serviceDescription') or '') for v in metadata.values())
    version = re.search(r'LHASA\s*(?:version\s*)?(2\.[0-9]+(?:\.[0-9]+)?)', descriptions, re.I)
    available = {name: bool(sampled.get(name)) for name in SERVICES}
    live = sum(r['probability'] is not None for r in output)
    return {'provider': 'NASA GSFC / Earthdata GIS', 'mode': 'LIVE_NASA_IMAGESERVER' if live == len(output) and live else
            ('MIXED_LIVE_AND_REFERENCE' if live else 'LHASA_REFERENCE_EMULATION'),
        'product_version': version.group(1) if version else 'NOT_DECLARED_BY_SERVICE',
        'requested_comparator_version': '2.0', 'exact_lhasa_2_0_reproduction': False,
        'native_resolution_arcsec': 30, 'native_resolution_m_approx': 1000,
        'native_resolution_arcsec_by_product': {name: [abs(meta['pixelSizeX']) * 3600, abs(meta['pixelSizeY']) * 3600]
                                                for name, meta in metadata.items()},
        'native_support': '30-arcsecond geographic pixels; east-west metric width varies with latitude',
        'sampling': 'nearest-neighbor source-pixel-center multipoint samples, reused across H3 cells',
        'fetched_at_utc': started, 'product_valid_time_utc': None,
        'valid_time_note': 'Today/tomorrow service names are relative products; acquisition/issue time is unknown unless source metadata declares it',
        'cells': output, 'by_h3': {r['h3_index']: r for r in output}, 'availability': available,
        'source_urls': SERVICES.copy(), 'metadata': metadata, 'errors': errors,
        'coverage': {'cells': len(output), 'live_today_cells': live, 'reference_cells': len(output) - live},
        'reference_proxy': {'name': 'LHASA-2.0 Open Reference Emulation — unvalidated Terra proxy',
            'version': 'terra_coarse_reference_index_v1', 'is_nasa_model': False, 'is_probability': False,
            'formula': 'sin(mean_scan_covered_bin_slope) * ((1-exp(-R24/100mm))+(1-exp(-ARI7/100mm)))/2',
            'rainfall_scale_mm': 100, 'coefficient_status': 'PROJECT_REFERENCE_SCALES_NOT_FITTED_OR_VALIDATED',
            'missing_lhasa_inputs': ['IMERG native forcing', 'SMAP L4 wetness/snow', 'published trained model weights', 'fault/lithology predictors'],
            'soil_moisture_used': False, 'scope': 'Scan-covered arithmetic mean terrain in nominal30arcsecond bins and supplied weather; no road/fault inputs are invented'},
        'limitations': ['Official2.0 skill is not reproduced by a reference formula; current NASA services may use a newer model.',
            'NASA likelihood, FOSM and Terra experimental initiation scores have different support/calibration.',
            'No NASA+48h/+72h forecast is fabricated from today/tomorrow products.']}
