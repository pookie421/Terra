#!/usr/bin/env python3
"""Check a deployed website and its bundled engine without acquiring providers."""
from __future__ import annotations

import argparse
import json
import re
import time
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen


def check(base: str) -> dict:
    def request(path: str, body=None):
        data = None if body is None else json.dumps(body).encode()
        req = Request(base + path, data=data, headers={'Content-Type': 'application/json'} if data else {})
        try:
            with urlopen(req, timeout=10) as response:
                return response.status, response.headers, response.read()
        except HTTPError as response:
            return response.code, response.headers, response.read()

    status, headers, raw = request('/api/v1/ready')
    assert status == 200, f'Readiness returned HTTP {status}'
    ready = json.loads(raw)
    assert ready['model_ready'] and ready['frontend_required'] and ready['frontend_ready'], ready
    assert ready['upstream_providers'] == 'not_checked', ready
    status, headers, raw = request('/')
    assert status == 200 and 'text/html' in headers.get('Content-Type', ''), 'Website entry is unavailable'
    html = raw.decode()
    assets = re.findall(r'(?:src|href)="(/assets/[^"?#]+\.(?:js|css))"', html)
    assert any(path.endswith('.js') for path in assets), 'Built JavaScript entry is missing'
    for path in dict.fromkeys(assets):
        status, headers, raw = request(path)
        kind = headers.get('Content-Type', '')
        assert status == 200 and ('javascript' in kind or 'text/css' in kind), f'Invalid build asset: {path}'
        assert 'immutable' in headers.get('Cache-Control', ''), f'Build asset is not cacheable: {path}'
    for path in ('/api/v1/missing-deployment-check', '/assets/missing-deployment-check.js', '/.env'):
        status, _, _ = request(path)
        assert status == 404, f'Missing/private path became a website success: {path}'
    status, headers, _ = request('/api/v1/scan', {'latitude': 91, 'longitude': 0})
    assert status == 422 and 'application/json' in headers.get('Content-Type', ''), 'Scan API validation is unavailable'
    return {'passed': True, 'website': base, 'model_ready': True, 'static_assets_checked': len(set(assets)),
            'provider_requests': 0, 'scope': 'Website routing, loaded engine and input validation; no live scan or forecast-skill claim'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('url', help='Website origin, for example https://example.up.railway.app')
    parser.add_argument('--wait-seconds', type=int, default=0, help='Retry startup checks for this many seconds')
    args = parser.parse_args()
    parsed = urlsplit(args.url)
    if (parsed.scheme not in ('http', 'https') or not parsed.hostname or parsed.username or parsed.password
            or parsed.query or parsed.fragment or parsed.path not in ('', '/')):
        parser.error('Pass an HTTP(S) origin without credentials, a path or query parameters')
    if not 0 <= args.wait_seconds <= 300:
        parser.error('--wait-seconds must be between 0 and 300')
    base = args.url.rstrip('/')
    deadline = time.monotonic() + args.wait_seconds
    while True:
        try:
            result = check(base)
            break
        except (AssertionError, URLError, ConnectionError, TimeoutError, ValueError, KeyError):
            if time.monotonic() >= deadline:
                raise
            time.sleep(1)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
