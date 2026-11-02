"""Scenario units and navigation must preserve the intended inputs/state."""
import contextlib
import io
import unittest
from unittest.mock import ANY, AsyncMock, patch
from scripts.cli_scenarios import prompt_scenario
from scripts.terra_cli import main, parse_args


class ScenarioInputs(unittest.TestCase):
    def prompt(self, answers):
        with patch('builtins.input', side_effect=answers), contextlib.redirect_stdout(io.StringIO()):
            return prompt_scenario()

    def test_burn_percentage_and_rain_units(self):
        self.assertEqual(self.prompt(['4', '150', '85', '65']),
                         {'burn_severity_override': .85, 'rain_delta_mm': 65.})

    def test_incoming_storm_cannot_extend_past_horizon(self):
        self.assertEqual(self.prompt(['5', '220', '71', '12', '1']),
                         {'inject_forecast_storm_mm': 220., 'storm_arrival_hour': 71, 'storm_duration_hours': 1})

    def test_defaults_and_back(self):
        self.assertEqual(self.prompt(['1', '', '0', '']),
                         {'rain_delta_mm': 180., 'storm_duration_hr': 24.})
        self.assertIsNone(self.prompt(['5', '220', 'b']))
        self.assertEqual(self.prompt(['0']), {})


class MenuState(unittest.IsolatedAsyncioTestCase):
    async def test_details_follow_latest_scenario_and_original_snapshot_is_reused(self):
        initial = {'summary': {'max_p_failure': .2}, 'h3_geojson': {'features': [{}]}}
        changed = {'summary': {'max_p_failure': .7}, 'h3_geojson': {'features': [{}]}}
        snapshot = object()
        with patch('scripts.terra_cli.GlobalScanService') as factory, \
             patch('scripts.terra_cli.TerraShieldQwenAgent'), \
             patch('scripts.terra_cli.acquire', AsyncMock(return_value=snapshot)) as acquire, \
             patch('scripts.terra_cli.show'), \
             patch('scripts.terra_cli.show_details') as details, \
             patch('scripts.terra_cli.prompt_scenario', return_value={'rain_delta_mm': 180.}), \
             patch('builtins.input', side_effect=['2', '4', 'q']), \
             contextlib.redirect_stdout(io.StringIO()):
            service = factory.return_value
            service.scan = AsyncMock(side_effect=[initial, changed])
            result = await main(parse_args(['--lat', '11.548', '--lon', '76.134', '--interactive', '--plain']))
            self.assertEqual(result, 0)
            self.assertEqual(acquire.await_count, 1)
            self.assertIs(service.scan.await_args_list[1].kwargs['_snapshot'], snapshot)
            self.assertEqual(service.scan.await_args_list[1].kwargs['rain_delta_mm'], 180.)
            details.assert_called_once_with(changed, ANY)
