"""Explicit latitude/longitude input for the interactive terminal."""
import math
import re

_NUMBER = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)"
_VALUE = rf"{_NUMBER}\s*°?\s*[NSEW]?"
_LABELED = re.compile(rf"\b(latitude|longitude|lat|lon|lng)\s*[:=]?\s*({_VALUE})", re.I)
_PAIR = re.compile(rf"^\s*({_VALUE})(?:\s*,\s*|\s+)({_VALUE})\s*$", re.I)


def coordinate(value, axis):
    """Parse decimal degrees, optionally with a hemisphere; never swap axes."""
    match = re.fullmatch(rf"\s*({_NUMBER})\s*°?\s*([NSEW]?)\s*", str(value), re.I)
    limit = 90 if axis == 'latitude' else 180
    if not match:
        raise ValueError(f'{axis.capitalize()} must be decimal degrees (for example 11.548 or 11.548 N).')
    number = float(match[1])
    hemisphere = match[2].upper()
    if hemisphere:
        allowed = 'NS' if axis == 'latitude' else 'EW'
        if hemisphere not in allowed:
            raise ValueError(f'{axis.capitalize()} uses {allowed[0]}/{allowed[1]}, not {hemisphere}.')
        if match[1].startswith(('-', '+')):
            raise ValueError('Use either a signed number or a hemisphere, for example -42.26 or 42.26 S.')
        number *= -1 if hemisphere in 'SW' else 1
    if not math.isfinite(number) or not -limit <= number <= limit:
        raise ValueError(f'{axis.capitalize()} must be between -{limit} and {limit} degrees. Coordinates were not swapped.')
    return number


def parse_location(text):
    """Return a place name or (latitude, longitude) tuple.

    Unlabeled pairs are always latitude first. Labels allow either order.
    """
    text = text.strip()
    if not text:
        raise ValueError('Enter a place name or coordinates.')
    labeled = list(_LABELED.finditer(text))
    if labeled:
        rest = _LABELED.sub('', text)
        if rest.strip(' ,;'):
            raise ValueError('Use latitude=11.548 longitude=76.134, with decimal degrees.')
        values = {}
        for match in labeled:
            axis = 'latitude' if match[1].lower() in ('lat', 'latitude') else 'longitude'
            if axis in values:
                raise ValueError(f'{axis.capitalize()} was supplied twice. Enter one latitude and one longitude.')
            values[axis] = coordinate(match[2], axis)
        if len(values) != 2:
            raise ValueError('Both latitude and longitude are required.')
        return values['latitude'], values['longitude']
    pair = _PAIR.fullmatch(text)
    if pair:
        return coordinate(pair[1], 'latitude'), coordinate(pair[2], 'longitude')
    # Malformed numeric input must not silently become a geocoding request.
    if (re.match(r'^[+\-\d.]', text) or
            re.match(r'^(?:lat(?:itude)?|lon(?:gitude)?|lng)\s*[:=]', text, re.I) or
            re.match(r'^(?:nan|[+-]?inf(?:inity)?)\b', text, re.I)):
        raise ValueError('Use latitude, longitude in decimal degrees: 11.548, 76.134. For longitude first, label both values.')
    return text


def location_query(location):
    return f'{location[0]:.8f}, {location[1]:.8f}' if isinstance(location, tuple) else location


def describe_location(location):
    if isinstance(location, tuple):
        return f'Latitude: {location[0]:.6f}° | Longitude: {location[1]:.6f}°'
    return f'Place: {location}'


def prompt_location(recent=None):
    from scripts.cli_ui import input_examples, panel
    recent = recent or []
    panel('LOCATION / Where would you like to explore?')
    print('\nChoose a location:')
    print('  [1] Place name   [2] Enter latitude and longitude separately   [q] Quit')
    print('  [3] Example locations   [r] Recent locations   [?] All input examples')
    print('  Place example: Zermatt, Switzerland')
    print('  Or paste: 11.548, 76.134  (LATITUDE first, LONGITUDE second)')
    print('  Labels work in either order: longitude=76.134 latitude=11.548')
    print('  Southern hemisphere example: 42.26 S, 173.62 E')
    if recent:
        print('  Enter to repeat:', recent[0])
    while True:
        text = input('Location or choice: ').strip()
        if not text and recent:
            return parse_location(recent[0])
        if text.lower() in ('?', 'help', 'examples'):
            input_examples()
            continue
        if text.lower() in ('3', 'r', 'recent'):
            choices = ['Chooralmala, Wayanad', 'Zermatt, Switzerland', 'Kaikoura', 'Malibu', 'New Delhi'] if text == '3' else recent
            if not choices:
                print('  No recent scans yet. Enter a location to begin.')
                continue
            for i, name in enumerate(choices, 1):
                print(f'  [{i}] {name}')
            choice = input('Choose a number (b to go back, q to quit): ').strip().lower()
            if choice in ('q', 'quit'):
                return None
            if choice.isdigit() and 1 <= int(choice) <= len(choices):
                return parse_location(choices[int(choice) - 1])
            if choice != 'b':
                print('Choose one of the listed numbers.')
            continue
        if text.lower() in ('q', 'quit'):
            return None
        if text == '1':
            text = input('Place name (e.g. Zermatt, Switzerland; q to quit): ').strip()
            if text.lower() in ('q', 'quit'):
                return None
        if text.lower() in ('2', 'c', 'coords', 'coordinates'):
            values = []
            for axis, hint in [('latitude', '-90 to 90; north +, south −; e.g. 11.548'),
                               ('longitude', '-180 to 180; east +, west −; e.g. 76.134')]:
                while True:
                    value = input(f'{axis.capitalize()} ({hint}; q to quit): ').strip()
                    if value.lower() in ('q', 'quit'):
                        return None
                    try:
                        values.append(coordinate(value, axis))
                        break
                    except ValueError as exc:
                        print(f'Invalid coordinate: {exc}')
            return tuple(values)
        try:
            return parse_location(text)
        except ValueError as exc:
            print(f'Invalid location: {exc}')
