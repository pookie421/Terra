"""Coordinate-order regressions: input must reach the intended location."""
import contextlib
import io
import unittest
from unittest.mock import AsyncMock, patch

from scripts.cli_location import parse_location, prompt_location
from scripts.terra_cli import acquire, parse_args


class CoordinateInput(unittest.TestCase):
    def test_decimal_and_hemisphere_pairs(self):
        for text, expected in [
            ('11.548, 76.134', (11.548, 76.134)),
            ('11.548 76.134', (11.548, 76.134)),
            ('42.26° S, 173.62° E', (-42.26, 173.62)),
            ('-42.26, 173.62', (-42.26, 173.62)),
            ('0, 0', (0., 0.)),
        ]:
            with self.subTest(text=text):
                self.assertEqual(parse_location(text), expected)

    def test_labeled_longitude_first(self):
        for text in ('longitude=76.134 latitude=11.548', 'lon: 76.134, lat: 11.548',
                     'latitude: 11.548; longitude: 76.134'):
            with self.subTest(text=text):
                self.assertEqual(parse_location(text), (11.548, 76.134))

    def test_invalid_coordinates_do_not_become_place_names(self):
        for text in ('91, 20', '20, 181', '173.62, -42.26', 'nan, 20', '11.548',
                     'lat=11', 'lat=11 lat=12 lon=76', '11, bad',
                     '11 E, 76 N', '-11 S, 76 E', '11 32 52 N, 76 8 2 E', ''):
            with self.subTest(text=text), self.assertRaises(ValueError):
                parse_location(text)

    def test_places_with_commas_remain_places(self):
        self.assertEqual(parse_location(' Zermatt, Switzerland '), 'Zermatt, Switzerland')

    def test_guided_entry_retries_only_invalid_field(self):
        with patch('builtins.input', side_effect=['2', '100', '11.548', 'bad', '76.134']), contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(prompt_location(), (11.548, 76.134))

    def test_quit_from_guided_coordinates(self):
        with patch('builtins.input', side_effect=['2', '11.548', 'q']), contextlib.redirect_stdout(io.StringIO()):
            self.assertIsNone(prompt_location())

    def test_flags_allow_explicit_longitude_first(self):
        args = parse_args(['--lon', '76.134', '--lat', '11.548', '--non-interactive'])
        self.assertEqual(parse_location(args.location), (11.548, 76.134))
        for flags in (['--lat', '11'], ['--non-interactive'],
                      ['Delhi', '--lat', '11', '--lon', '76'], ['--lat', 'nan', '--lon', '76']):
            with self.subTest(flags=flags), contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as exc:
                parse_args(flags)
            self.assertEqual(exc.exception.code, 2)


class AcquisitionInput(unittest.IsolatedAsyncioTestCase):
    async def test_coordinates_reach_ingestor_in_lat_lon_order(self):
        with patch('scripts.terra_cli.GlobalIngestor') as factory:
            ingestor = factory.return_value.__aenter__.return_value
            ingestor.scan = AsyncMock(return_value='snapshot')
            self.assertEqual(await acquire('longitude=76.134 latitude=11.548'), 'snapshot')
            ingestor.scan.assert_awaited_once_with((11.548, 76.134))

    async def test_invalid_coordinates_never_start_network_ingestion(self):
        with patch('scripts.terra_cli.GlobalIngestor') as factory:
            with self.assertRaises(ValueError):
                await acquire('190, 20')
            factory.assert_not_called()
