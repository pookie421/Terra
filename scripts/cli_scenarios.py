"""Guided scenario controls with units, defaults and retryable validation."""
import math
from scripts.cli_ui import panel


class BackToMenu(Exception):
    pass


def ask_number(label, default, minimum=0., maximum=None, integer=False):
    while True:
        text = input(f'{label} [Enter = {default}; b = back]: ').strip()
        if text.lower() in ('b', 'back', 'q', 'quit'):
            raise BackToMenu()
        try:
            value = float(text) if text else float(default)
            if not math.isfinite(value) or value < minimum or (maximum is not None and value > maximum):
                raise ValueError()
            if integer and not value.is_integer():
                raise ValueError()
            return int(value) if integer else value
        except ValueError:
            bounds = f'{minimum} to {maximum}' if maximum is not None else f'{minimum} or greater'
            print(f'  Enter a finite {"whole " if integer else ""}number, {bounds}.')


def prompt_scenario():
    panel('WHAT IF? / Compare with the original scan')
    print('  [1] Extra rainfall       e.g. +180 mm over 24 hours')
    print('  [2] Earthquake shaking   e.g. 0.35 g PGA (not magnitude)')
    print('  [3] Warming             e.g. +8 degrees C')
    print('  [4] Burn + rain         e.g. 85% burn and +65 mm')
    print('  [5] Incoming storm      e.g. 220 mm arriving in 18 h, lasting 12 h')
    print('  [0] Original conditions and provider forecast   [b] Back')
    print('  Each scenario starts from the original snapshot; changes do not accumulate.')
    while True:
        choice = input('Scenario: ').strip().lower()
        try:
            if choice in ('b', 'back', 'q', 'quit'):
                return None
            if choice == '0':
                return {}
            if choice == '1':
                return {'rain_delta_mm': ask_number('Additional rainfall, mm (example: 180)', 180),
                        'storm_duration_hr': ask_number('Storm duration, hours (example: 24)', 24, minimum=.01)}
            if choice == '2':
                return {'pga_override_g': ask_number('PGA, g (example: 0.35)', .35)}
            if choice == '3':
                return {'snowmelt_temp_surge_c': ask_number('Temperature increase, degrees C (example: 8)', 8)}
            if choice == '4':
                return {'burn_severity_override': ask_number('Burn severity, percent (example: 85)', 85, maximum=100) / 100,
                        'rain_delta_mm': ask_number('Additional rainfall, mm (example: 65)', 65)}
            if choice == '5':
                amount = ask_number('Future storm total, mm (example: 220)', 220)
                arrival = ask_number('Arrival, hours from now (example: 18)', 18, maximum=71, integer=True)
                duration = ask_number('Duration, hours (example: 12)', min(12, 72-arrival), minimum=1, maximum=72-arrival, integer=True)
                print('  Synthetic calm-start experiment: initial rain and water head reset to zero; live summary remains separate.')
                return {'inject_forecast_storm_mm': amount, 'storm_arrival_hour': arrival, 'storm_duration_hours': duration}
            print('  Choose 0–5 or b to go back.')
        except BackToMenu:
            return None
