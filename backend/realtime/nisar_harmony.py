"""Bounded, reusable NASA Harmony jobs for actual NISAR raster subsets.

Uses collection-advertised CMR variable identifiers, one real granule and the
scan's spatial subset. Processing can outlive a scan: a pending job is retained
and polled on refresh, rather than downloading a multi-GB HDF5 or resubmitting.
Only the returned COG pixels become measurements. No phase/strength inference.
API: https://harmony.earthdata.nasa.gov/docs
NISAR: https://nisar-docs.asf.alaska.edu/harmony/
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import re
import time
from urllib.parse import unquote, urljoin, urlsplit

from .cache import AsyncTTLCache

HARMONY = 'https://harmony.earthdata.nasa.gov'
_CAPABILITIES = AsyncTTLCache(24)
_JOBS = AsyncTTLCache(128)
_POLL = AsyncTTLCache(128)
_IDENTITY = re.compile(r'^[A-Z][0-9]+-[A-Za-z0-9_-]+$')
_JOB_ID = re.compile(r'^[A-Za-z0-9_-]{1,80}$')


def allowed_result_url(value):
    """Only HTTPS NASA/ASF, S3 or CloudFront results from NASA's jobs API.

    AWS signed query strings are used internally and never returned as public
    provenance. Earthdata authorization is never forwarded to AWS.
    """
    try:
        p = urlsplit(str(value)); host = (p.hostname or '').lower()
        return (p.scheme == 'https' and not p.username and not p.password and
            (p.port in (None, 443)) and (host.endswith('.nasa.gov') or
            host.endswith('.asf.alaska.edu') or host.endswith('.cloudfront.net') or
            bool(re.fullmatch(r'(?:[a-z0-9.-]+\.)?s3(?:[.-][a-z0-9-]+)?\.amazonaws\.com', host))))
    except (ValueError, TypeError):
        return False


def _kind(granule):
    name = str(granule.get('collection_short_name') or '') + ' ' + str(granule.get('granule_id') or '')
    return 'GUNW' if 'GUNW' in name else ('GCOV' if 'GCOV' in name else None)


def _maturity(granule):
    name = str(granule.get('collection_short_name') or '')
    return 'BETA' if 'BETA' in name else ('PROVISIONAL' if 'PROVISIONAL' in name else 'VALIDATED_OR_UNDECLARED')


async def _json(client, url, *, token=None, params=None, deadline):
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise TimeoutError('NISAR subset request budget exhausted')
    headers = {'Authorization': 'Bearer ' + token} if token else {}
    response = await asyncio.wait_for(client.get(url, params=params, headers=headers,
        timeout=remaining, follow_redirects=False), timeout=remaining)
    for _ in range(3):
        if response.status_code not in (301, 302, 303, 307, 308):
            break
        destination = urljoin(str(response.request.url), response.headers.get('location') or '')
        target = urlsplit(destination)
        if target.hostname == 'urs.earthdata.nasa.gov':
            raise PermissionError('Earthdata authorization or application approval is required')
        if target.scheme != 'https' or target.hostname != 'harmony.earthdata.nasa.gov' or target.username or target.password or target.port not in (None, 443):
            raise ValueError('Harmony returned an unexpected metadata redirect')
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError('NISAR subset request budget exhausted')
        response = await asyncio.wait_for(client.get(destination, headers=headers,
            timeout=remaining, follow_redirects=False), timeout=remaining)
    if response.status_code in (401, 403):
        raise PermissionError('Earthdata authorization was rejected')
    response.raise_for_status()
    if len(response.content) > 2 * 1024 * 1024:
        raise ValueError('Harmony metadata exceeds the bounded response size')
    payload = response.json()
    if not isinstance(payload, dict):
        raise ValueError('Harmony returned invalid metadata')
    return payload


def _variables(capabilities, kind):
    variables = []
    for item in capabilities.get('variables') or []:
        if not isinstance(item, dict):
            continue
        name = str(item.get('name') or '')
        identity = urlsplit(str(item.get('href') or '')).path.rsplit('/', 1)[-1]
        if not _IDENTITY.fullmatch(identity) or not identity.startswith('V'):
            continue
        if kind == 'GUNW' and re.search(r'/GUNW/grids/frequencyA/unwrappedInterferogram/(HH|VV)/coherenceMagnitude$', name):
            variables.append({'id': identity, 'name': name, 'role': 'coherence'})
        elif kind == 'GCOV' and name in ('/science/LSAR/GCOV/grids/frequencyA/HHHH',
                                        '/science/LSAR/GCOV/grids/frequencyA/HVHV'):
            variables.append({'id': identity, 'name': name,
                'role': 'denominator' if name.endswith('/HHHH') else 'numerator'})
    if kind == 'GCOV' and {v['role'] for v in variables} != {'numerator', 'denominator'}:
        return []
    return sorted(variables, key=lambda x: x['name'])[:2]


def _output_links(payload):
    result = []
    for link in payload.get('links') or []:
        href = link.get('href') if isinstance(link, dict) else None
        if not href or not allowed_result_url(href):
            continue
        p = urlsplit(href)
        if str(link.get('rel') or '').lower() in ('data', 'output') or (
                str(link.get('type') or '').lower().startswith('image/tiff')):
            if p.path.lower().endswith(('.tif', '.tiff')):
                result.append(href)
    return result


def _assets(payload, variables, granule, job_id):
    outputs = _output_links(payload)
    matches = {}
    for variable in variables:
        # ASF's extractor preserves the full HDF5 dataset path in the COG
        # basename. Do not assign arbitrary output ordering to HH/HV semantics.
        marker = variable['name'].strip('/').replace('/', '_')
        candidates = [url for url in outputs if marker in unquote(urlsplit(url).path)]
        if len(candidates) == 1:
            matches[variable['role'] + (variable['name'] if variable['role'] == 'coherence' else '')] = (variable, candidates[0])
    common = {'radar_band': 'L', 'units': '1', 'granule_id': granule.get('granule_id'),
        'granule_concept_id': granule.get('cmr_concept_id'),
        'collection_concept_id': granule.get('collection_concept_id'),
        'collection_short_name': granule.get('collection_short_name'),
        'product_maturity': _maturity(granule), 'product_version': granule.get('product_version'),
        'acquisition_utc': granule.get('time_end') or granule.get('time_start'),
        'time_start': granule.get('time_start'), 'time_end': granule.get('time_end'),
        'metadata_origin': 'NASA_HARMONY_OFFICIAL_CMR_VARIABLE_SUBSET', 'harmony_job_id': job_id,
        'quality_scope': 'Native nodata, finite/range checks; no deformation, moisture or strength calibration'}
    if _kind(granule) == 'GUNW':
        return [dict(common, observable='coherence', url=url, source_dataset_paths=[variable['name']],
            variable_concept_ids=[variable['id']]) for variable, url in matches.values()]
    if 'numerator' in matches and 'denominator' in matches:
        num, den = matches['numerator'], matches['denominator']
        return [dict(common, observable='polarimetric_ratio', url=num[1], denominator_url=den[1],
            source_dataset_paths=[num[0]['name'], den[0]['name']],
            variable_concept_ids=[num[0]['id'], den[0]['id']])]
    return []


async def measured_subsets(client, granules, rectangles, *, token, deadline,
                            observables=('coherence', 'polarimetric_ratio')):
    """Return ready raster descriptors and a credential-free access summary.

    A maximum of four product jobs is considered per scan (two products and
    two dateline rectangles). The most recent
    product is selected from discovery, with maturity breaking equal-time ties.
    Pending work is not a measurement. Job labels also recover work after a
    process restart, avoiding repeated submissions for the same exact request.
    """
    summary = {'method': 'NASA_HARMONY_SPATIAL_VARIABLE_SUBSET',
        'authentication_configured': bool(token), 'status': 'AUTH_REQUIRED' if not token else 'NO_PRODUCT_COVERAGE',
        'retry_after_seconds': None, 'jobs': [],
        'message': 'The deployment needs Earthdata access to retrieve measured NISAR subsets.' if not token else
                   'No supported NISAR product was discovered for this scan.'}
    if not token:
        return [], summary
    assets = []
    kinds = [kind for kind, obs in (('GUNW', 'coherence'), ('GCOV', 'polarimetric_ratio')) if obs in observables]
    for kind, rectangle in ((kind, rectangle) for kind in kinds for rectangle in rectangles):
        candidates = [g for g in granules if _kind(g) == kind and _IDENTITY.fullmatch(str(g.get('collection_concept_id') or ''))
            and _IDENTITY.fullmatch(str(g.get('cmr_concept_id') or ''))]
        if not candidates:
            continue
        granule = max(candidates, key=lambda g: (g.get('time_end') or g.get('time_start') or '',
            {'VALIDATED_OR_UNDECLARED': 2, 'PROVISIONAL': 1, 'BETA': 0}[_maturity(g)]))
        job = {'product_type': kind, 'status': 'UNAVAILABLE', 'provider_status': None, 'job_id': None,
            'collection_short_name': granule.get('collection_short_name'), 'granule_id': granule.get('granule_id'),
            'acquisition_utc': granule.get('time_end') or granule.get('time_start')}
        summary['jobs'].append(job)
        try:
            collection = granule['collection_concept_id']
            async def capabilities():
                return await _json(client, HARMONY + '/capabilities',
                    params={'collectionId': collection, 'version': '3'}, deadline=deadline)
            capability = await _CAPABILITIES.get_or_create(collection, 86400, capabilities)
            if capability.get('conceptId') != collection:
                raise ValueError('Harmony capabilities do not match the discovered collection')
            if capability.get('shortName'):
                granule['collection_short_name'] = capability['shortName']
                job['collection_short_name'] = capability['shortName']
            subset = (capability.get('summary') or {}).get('subsetting') or {}
            if not subset.get('bbox') or not subset.get('variable'):
                raise ValueError('Collection does not advertise variable and spatial subsetting')
            variables = _variables(capability, kind)
            if not variables:
                raise ValueError('Requested radar variables are not advertised by this collection')
            # A dateline sector has two small non-wrapping rectangles. Use one
            # job per product/rectangle; never turn it into a near-global bbox.
            if rectangle[0] >= rectangle[2] or rectangle[2] - rectangle[0] > 2 or rectangle[3] - rectangle[1] > 2:
                raise ValueError('Automatic NISAR subset is limited to a two-degree scan window')
            west, south, east, north = rectangle
            identity = json.dumps([granule['cmr_concept_id'], [v['id'] for v in variables], rectangle], separators=(',', ':'))
            key = hashlib.sha256((hashlib.sha256(token.encode()).hexdigest() + identity).encode()).hexdigest()
            label = 'terra-nisar-' + key[:32]
            async def submit():
                existing = await _json(client, HARMONY + '/jobs', token=token,
                    params={'label': label, 'limit': '2'}, deadline=deadline)
                jobs = [j for j in existing.get('jobs') or [] if _JOB_ID.fullmatch(str(j.get('jobID') or ''))
                    and str(j.get('status', '')).lower() not in ('failed', 'canceled', 'cancelled', 'error')]
                if jobs:
                    return {'jobID': jobs[0]['jobID']}
                path = '/{}/ogc-api-coverages/1.0.0/collections/{}/coverage/rangeset'.format(
                    collection, ','.join(v['id'] for v in variables))
                params = [('granuleId', granule['cmr_concept_id']), ('subset', 'lon({}:{})'.format(west, east)),
                    ('subset', 'lat({}:{})'.format(south, north)), ('format', 'image/tiff'),
                    ('forceAsync', 'true'), ('skipPreview', 'true'), ('maxResults', '1'),
                    ('ignoreErrors', 'false'), ('label', label)]
                result = await _json(client, HARMONY + path, token=token, params=params, deadline=deadline)
                if not _JOB_ID.fullmatch(str(result.get('jobID') or '')):
                    raise ValueError('Harmony did not return a usable job identifier')
                return {'jobID': result['jobID']}
            submitted = await _JOBS.get_or_create(key, 86400, submit)
            job['job_id'] = submitted['jobID']
            async def poll():
                return await _json(client, HARMONY + '/jobs/' + submitted['jobID'], token=token, deadline=deadline)
            payload = await _POLL.get_or_create((key, submitted['jobID']), 5, poll)
            provider_status = str(payload.get('status') or '').lower()
            job['provider_status'] = provider_status
            if provider_status in ('successful', 'success', 'complete', 'completed'):
                ready = _assets(payload, variables, granule, submitted['jobID'])
                job['status'] = 'SUCCESSFUL' if ready else 'UNAVAILABLE'
                assets.extend(ready)
            elif provider_status in ('failed', 'error', 'canceled', 'cancelled'):
                job['status'] = 'FAILED'
            elif provider_status in ('accepted', 'running', 'not_started', 'queued', 'paused',
                    'previewing', 'starting', 'processing'):
                job['status'] = 'PENDING'
            else:
                job['status'] = 'UNAVAILABLE'
        except PermissionError:
            job['status'] = 'AUTH_REQUIRED'
        except Exception as exc:
            job['status'] = 'PENDING' if job['job_id'] and 'TIMEOUT' in type(exc).__name__.upper() else 'UNAVAILABLE'
            # Provider exception strings may contain signed URLs or headers.
            job['error_code'] = type(exc).__name__.upper()
            if getattr(exc, 'response', None) is not None:
                job['http_status'] = exc.response.status_code
                job['request_phase'] = ('CAPABILITIES' if exc.request.url.path == '/capabilities' else
                    ('JOB_LOOKUP' if exc.request.url.path == '/jobs' else
                     ('SUBMISSION' if exc.request.url.path.endswith('/rangeset') else 'JOB_POLL')))
    if any(j['status'] == 'AUTH_REQUIRED' for j in summary['jobs']):
        summary.update(status='AUTH_REQUIRED', message='Earthdata rejected access; check the deployment credential and NISAR application authorization.')
    elif any(j['status'] == 'PENDING' for j in summary['jobs']):
        summary.update(status='SUBSET_PENDING', retry_after_seconds=10,
            message='NASA is preparing the requested raster subset. Refresh radar evidence to reuse and check this job.')
    elif assets:
        summary.update(status='RASTER_READY', message='NASA returned raster subsets; only valid sampled pixels will be reported as measurements.')
    elif summary['jobs']:
        summary.update(status='SUBSET_UNAVAILABLE', message='The discovered products did not yield supported raster subsets. No radar values were inferred.')
    return assets, summary
