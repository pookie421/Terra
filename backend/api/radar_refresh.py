"""Refresh NISAR acquisition without repeating the other provider requests."""
import asyncio
import copy
from datetime import datetime, timezone

import httpx

from backend.realtime.cache import DEFAULT_CACHE, PolicyCache
from backend.realtime.global_ingestor import GlobalIngestor, SourceStatus
from backend.realtime.nisar_processor import fetch_nisar


async def refresh_radar_scan(service, scan_id, scenario_id):
    state = service.get_scan_state(scan_id, scenario_id)
    response = state['response']
    if response.get('forecast_view', {}).get('key', 'now') != 'now':
        raise ValueError('Refresh radar from the current-condition analysis, then select a forecast.')
    raw = copy.deepcopy(state['scan'])
    cells = getattr(getattr(raw, 'terrain', None), 'cells', None)
    if not cells:
        raise ValueError('Terrain cells are required to sample a measured radar product.')
    south, west, north, east = GlobalIngestor._bbox(
        raw.location.latitude, raw.location.longitude, raw.radius_km)
    timeout = min(15.0, max(3.5, service.provider_timeout_s))
    checked_at = datetime.now(timezone.utc).isoformat()
    try:
        async with httpx.AsyncClient(timeout=timeout, follow_redirects=False) as client:
            radar = await asyncio.wait_for(fetch_nisar(
                client, raw.location.latitude, raw.location.longitude,
                cells=cells, bbox=(west, south, east, north), timeout_s=timeout,
                cache=PolicyCache(DEFAULT_CACHE, False, True)), timeout=timeout + 1)
    except Exception as exc:
        # Do not return HTTP request details, signed URLs or authentication data.
        raise ValueError('Radar refresh could not complete; the existing analysis is retained ('
                         + type(exc).__name__ + ').') from None
    if not isinstance(radar, dict):
        raise ValueError('Radar refresh returned no source record; the existing analysis is retained.')
    raw.nisar = radar
    mode = radar.get('mode', 'UNAVAILABLE')
    errors = radar.get('errors') or []
    if isinstance(errors, dict):
        errors = list(errors.values())
    raw.sources['nisar'] = SourceStatus(
        available=mode not in ('UNAVAILABLE', 'METADATA_UNAVAILABLE'),
        provider='NASA-ISRO NISAR / ASF DAAC', fetched_at_utc=checked_at,
        error='; '.join(str(error) for error in errors) or None, mode=mode,
        cache_policy='LIVE_FIRST_CACHE_ON_ERROR')
    # The old snapshot remains immutable. Recompute against identical retained
    # forcing with a new scan identity, including any valid sourced calibration.
    raw.scan_id = None
    updated = await service.scan(_snapshot=raw, ground_columns=state['ground_columns'],
                                 **response.get('scenario', {}))
    metadata = {'source_scan_id': scan_id, 'source_scenario_id': scenario_id,
                'checked_at_utc': checked_at, 'reused_environmental_forcing': True,
                'scope': 'Only NISAR was reacquired; weather, terrain and other inputs retain their original times.'}
    updated['radar_refresh'] = metadata
    with service._context_lock:
        saved = service._scan_states.get((updated['scan_id'], updated['scenario_id']))
        if saved is not None:
            saved['response']['radar_refresh'] = copy.deepcopy(metadata)
    return updated
