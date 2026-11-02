"""Small terminal graphics using only the standard library."""
import asyncio
import contextlib
import math
import shutil
import sys
import time


def panel(title):
    width = min(76, max(36, shutil.get_terminal_size((76, 24)).columns - 2))
    print('\n' + '+' + '-' * (width - 2) + '+')
    print('| ' + title[:width - 4].ljust(width - 4) + ' |')
    print('+' + '-' * (width - 2) + '+')


def banner():
    panel('T E R R A  /  Terrain research explorer')
    print('       /\\       /\\')
    print('  ____/  \\_____/  \\____   Explore. Compare. Inspect.')
    print('  Research scenarios; not operational warnings.')
    print('  [?] Examples and help    [q] Quit    Ctrl+C to exit')


def input_examples():
    panel('LOCATION INPUT / Copy any example')
    for title, example in (
        ('Place name', 'Zermatt, Switzerland'),
        ('Latitude, longitude', '11.548, 76.134'),
        ('Space-separated pair', '11.548 76.134'),
        ('Labels in either order', 'longitude=76.134 latitude=11.548'),
        ('Short labels', 'lat: 11.548, lon: 76.134'),
        ('Signed south latitude', '-42.26, 173.62'),
        ('Signed west longitude', '34.07, -118.54'),
        ('Hemisphere letters', '42.26 S, 173.62 E'),
        ('Degree symbols', '42.26° S, 173.62° E'),
    ):
        print(f'  {title}\n    {example}')
    print('\n  Guided entry: choose 2, then latitude 11.548, longitude 76.134.')
    print('  Decimal degrees only. Unlabeled pairs: LATITUDE first.')
    print('             North (+latitude)')
    print('                     |')
    print('  West (-longitude) -+- East (+longitude)')
    print('                     |')
    print('             South (-latitude)')
    print('\n  From your shell (not inside the location prompt):')
    print('    python scripts/terra_cli.py --lat 11.548 --lon 76.134')
    print('    python scripts/terra_cli.py "42.26 S, 173.62 E"')
    print('    python scripts/terra_cli.py --lat 34.07 --lon -118.54')
    print('    python scripts/terra_cli.py --lat 11.548 --lon 76.134 --interactive')
    print('  Add --non-interactive for one scan; --details for full telemetry.')


def fmt(value, digits=2):
    if value is None:
        return 'unavailable'
    try:
        number = float(value)
        return f'{number:.{digits}f}' if math.isfinite(number) else 'unavailable'
    except (ValueError, TypeError):
        return 'unavailable'


def bar(percent, width=18):
    if fmt(percent) == 'unavailable':
        return '[' + '?' * width + '] unavailable'
    filled = round(max(0., min(100., float(percent))) * width / 100)
    return '[' + '#' * filled + '.' * (width - filled) + f'] {float(percent):6.2f}%'


def forecast_table(forecast):
    panel('72-HOUR SCENARIO / P50 and P90 forcing paths')
    if forecast.get('status') != 'PROJECTED_SCENARIO':
        print('  Unavailable:', forecast.get('reason', 'Complete forecast inputs missing'))
        return
    print('  Initial state:', forecast.get('initial_condition_quality', 'modeled'))
    if forecast.get('initial_condition_quality') == 'SYNTHETIC_CALM_NOT_LIVE':
        print('  Injected scenario resets initial rain and water head to zero.')
        print('  Live-input regional score remains:', fmt(forecast.get('observed_current_risk_t0_pct')), '%')
    if forecast.get('soil_temperature_fallback_hours'):
        print('  Soil-temperature fallback to air:', len(forecast['soil_temperature_fallback_hours']), 'hours')
    crossing = forecast.get('predicted_threshold_crossing_hour')
    when = 'No crossing in this horizon' if crossing is None else 'Already critical at T+0' if crossing == 0 else f'T+{crossing} h'
    print('  First regional threshold crossing:', when)
    print('  P50 / P90 peak score:', fmt(forecast.get('peak_72h_risk_p50_pct')),
          '/', fmt(forecast.get('peak_72h_risk_p90_pct')), '%')
    print('  Forcing:', ', '.join(forecast.get('forcing_modes', [])))
    print('\n  Hour   P50 FS   P50 score                 P90 score')
    for row in forecast.get('hourly_horizon_table', []):
        a, b = row['p50'], row['p90']
        print(f"  +{row['hour']:02d}h   {fmt(a.get('critical_fs'), 3):>6}   {bar(a.get('risk_pct'), 12)}   {fmt(b.get('risk_pct')):>6}%")
    print('  Hourly projections, not observed failure times.')
    print('  P90 forcing is not a 90% failure probability or a worst-case bound.')


def show_summary(scan, elapsed):
    loc, summary = scan['location'], scan['summary']
    top = summary.get('kinematics_top_hotspot') or {}
    panel('SCAN / ' + str(loc.get('display_name') or loc.get('query') or 'Location'))
    print(f"  Latitude {fmt(loc.get('latitude'), 6)} | Longitude {fmt(loc.get('longitude'), 6)}")
    print(f"  {len(scan['h3_geojson']['features'])} terrain cells | {elapsed:.1f} seconds")
    sources = scan.get('source_status', {})
    missing = [name for name, status in sources.items() if not status.get('available')]
    print(f'  Sources: {len(sources) - len(missing)}/{len(sources)} available (may include cache)')
    if missing:
        print('  Missing / fallback:', ', '.join(missing))
    if not scan['h3_geojson']['features']:
        print('  Terrain unavailable. Hazard and runout were not evaluated.')
        return
    score = summary.get('critical_p_failure', summary.get('max_p_failure'))
    print('\n  Regional research score:', bar(None if score is None else score * 100))
    print(f"  Critical FS: {fmt(summary.get('critical_fs'), 3)} | Alert label: {summary.get('wmo_alert_level', 'UNAVAILABLE')}")
    print(f"  Selected hotspot: {fmt(top.get('peak_velocity_ms'))} m/s | {top.get('failure_mode', 'unavailable')}")
    print('  FS < 1 denotes modeled mechanical instability.')
    print('  Regional percentiles and the selected hotspot are different statistics.')
    print('\n  Model / physics attribution (heuristic shares):')
    for cause, share in sorted(summary.get('causal_breakdown_pct', {}).items(), key=lambda item: item[1], reverse=True):
        print(f"  {cause.replace('_', ' '):25} {bar(share, 12)}")
    bounds = top.get('epistemic_interval_90_pct')
    print('\n  Hotspot probability spread:', f'{fmt(bounds[0])}–{fmt(bounds[1])}%' if bounds else 'unavailable')
    print('  Spread coverage is unvalidated; binary-label intervals are separate.')
    forecast_table(scan.get('predictive_forecast_72h', {}))


class Busy:
    """Elapsed-time spinner on a TTY; plain output when redirected."""
    def __init__(self, label, plain=False):
        self.label, self.plain, self.task = label, plain, None

    async def __aenter__(self):
        self.started = time.monotonic()
        print(f'  {self.label}... Ctrl+C to cancel.')
        if sys.stdout.isatty() and not self.plain:
            self.task = asyncio.create_task(self.animate())
        return self

    async def animate(self):
        frames = '|/-\\'
        i = 0
        while True:
            print(f'\r  {frames[i % len(frames)]} Working... {time.monotonic() - self.started:.0f}s', end='', flush=True)
            i += 1
            await asyncio.sleep(.15)

    async def __aexit__(self, *exc):
        if self.task:
            self.task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self.task
            print('\r' + ' ' * 32 + '\r', end='', flush=True)
