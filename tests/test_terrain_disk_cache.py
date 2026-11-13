"""Offline contracts for deployment cache paths, bounds and complete tile writes."""

import io
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import httpx
import numpy as np
from PIL import Image

from backend.realtime.cache import AsyncTTLCache
from backend.realtime.dem_processor import DEMProcessor, TerrainDataError, _TerrainDiskCache, _terrain_disk_cache


def png(size=(256, 256), green=100):
    output = io.BytesIO()
    Image.new('RGB', size, (128, green, 0)).save(output, format='PNG')
    return output.getvalue()


class TerrainDiskCacheTests(unittest.TestCase):
    def test_configuration_preserves_default_and_shares_configured_cache(self):
        with patch.dict(os.environ, {}, clear=True):
            cache = _terrain_disk_cache()
            expected = Path(__file__).resolve().parents[1] / 'data/source_catalogs/terrarium_tiles'
            self.assertEqual(cache.root, expected.resolve())
            self.assertEqual(cache.max_bytes, 1024 * 1024 * 1024)
        with tempfile.TemporaryDirectory() as folder:
            with patch.dict(os.environ, {'TERRA_TERRAIN_CACHE_DIR': folder, 'TERRA_TERRAIN_CACHE_MAX_MB': '2'}):
                cache = _terrain_disk_cache()
                self.assertEqual(cache.root, Path(folder).resolve())
                self.assertEqual(cache.max_bytes, 2 * 1024 * 1024)
                self.assertIs(_terrain_disk_cache(), cache)
            for invalid in ('0', '-1', '1.5', 'unlimited', ''):
                with self.subTest(invalid=invalid), patch.dict(os.environ, {'TERRA_TERRAIN_CACHE_MAX_MB': invalid}):
                    with self.assertRaisesRegex(ValueError, 'positive integer'):
                        _terrain_disk_cache()

    def test_eviction_is_bounded_and_preserves_unrelated_files_and_symlinks(self):
        tiles = [png(green=value) for value in (70, 80, 90)]
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder) / 'terrain'
            root.mkdir()
            sentinel = root / 'keep.txt'
            sentinel.write_text('unrelated')
            target = Path(folder) / 'outside.png'
            target.write_bytes(b'outside')
            linked = root / '3/0/7.png'
            linked.parent.mkdir(parents=True)
            try:
                linked.symlink_to(target)
            except OSError:
                linked = None  # Windows may require privileges for symlinks.
            budget = 2 * max(map(len, tiles))
            cache = _TerrainDiskCache(root, budget)
            for tile_y, content in enumerate(tiles):
                cache.write(3, 0, tile_y, content)
            self.assertFalse((root / '3/0/0.png').exists())
            self.assertEqual(cache.read(3, 0, 1), tiles[1])
            self.assertEqual(cache.read(3, 0, 2), tiles[2])
            self.assertLessEqual(sum(path.stat().st_size for path in root.glob('*/*/*.png') if not path.is_symlink()), budget)
            self.assertEqual(sentinel.read_text(), 'unrelated')
            self.assertEqual(target.read_bytes(), b'outside')
            if linked is not None:
                self.assertTrue(linked.is_symlink())

    def test_failed_atomic_replacement_preserves_previous_complete_file(self):
        before, after = png(green=40), png(green=60)
        with tempfile.TemporaryDirectory() as folder:
            cache = _TerrainDiskCache(Path(folder), 1024 * 1024)
            cache.write(3, 0, 0, before)
            with patch('backend.realtime.dem_processor.os.replace', side_effect=OSError('simulated full volume')):
                with self.assertRaises(OSError):
                    cache.write(3, 0, 0, after)
            self.assertEqual(cache.read(3, 0, 0), before)
            self.assertEqual(list(Path(folder).rglob('*.tmp')), [])

    def test_existing_cache_is_brought_under_budget_on_first_use(self):
        content = png()
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            directory = root / '3/0'
            directory.mkdir(parents=True)
            for tile_y in range(3):
                path = directory / (str(tile_y) + '.png')
                path.write_bytes(content)
                os.utime(path, (tile_y + 1, tile_y + 1))
            cache = _TerrainDiskCache(root, 2 * len(content))
            self.assertEqual(cache.read(3, 0, 2), content)
            self.assertFalse((directory / '0.png').exists())
            self.assertEqual(len(list(directory.glob('*.png'))), 2)

    def test_concurrent_writers_share_the_budget_and_leave_complete_pngs(self):
        tiles = [png(green=value) for value in range(40, 48)]
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            budget = 2 * max(map(len, tiles))
            cache = _TerrainDiskCache(root, budget)
            with ThreadPoolExecutor(max_workers=4) as executor:
                futures = [executor.submit(cache.write, 3, 0, tile_y, content) for tile_y, content in enumerate(tiles)]
                for future in futures:
                    future.result()
            retained = list(root.glob('*/*/*.png'))
            self.assertEqual(len(retained), 2)
            self.assertLessEqual(sum(path.stat().st_size for path in retained), budget)
            for path in retained:
                with Image.open(path) as image:
                    image.verify()
            self.assertEqual(list(root.rglob('*.tmp')), [])


class TerrainDiskFetchTests(unittest.IsolatedAsyncioTestCase):
    async def test_configured_volume_is_reused_without_a_second_provider_request(self):
        requests = []
        def respond(request):
            requests.append(str(request.url))
            return httpx.Response(200, content=png())
        with tempfile.TemporaryDirectory() as folder, patch.dict(os.environ, {'TERRA_TERRAIN_CACHE_DIR': folder}):
            async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
                first = await DEMProcessor(cache=AsyncTTLCache())._fetch_tile(client, 2, 3)
                second = await DEMProcessor(cache=AsyncTTLCache())._fetch_tile(client, 2, 3)
            self.assertEqual(len(requests), 1)
            self.assertTrue((Path(folder) / '13/2/3.png').is_file())
            np.testing.assert_array_equal(first[0], second[0])
            self.assertTrue(first[1].all())
            self.assertEqual(float(first[0][0, 0]), 100.0)

    async def test_invalid_dimensions_are_never_persisted(self):
        with tempfile.TemporaryDirectory() as folder, patch.dict(os.environ, {'TERRA_TERRAIN_CACHE_DIR': folder}):
            async with httpx.AsyncClient(transport=httpx.MockTransport(lambda request: httpx.Response(200, content=png((16, 16))))) as client:
                with self.assertRaisesRegex(TerrainDataError, 'Unexpected Terrarium tile size'):
                    await DEMProcessor(cache=AsyncTTLCache())._fetch_tile(client, 2, 3)
            self.assertEqual(list(Path(folder).rglob('*.png')), [])

    async def test_unwritable_volume_keeps_valid_provider_terrain_available(self):
        with tempfile.TemporaryDirectory() as folder, patch.dict(os.environ, {'TERRA_TERRAIN_CACHE_DIR': folder}):
            processor = DEMProcessor(cache=AsyncTTLCache())
            with patch.object(processor._disk_cache, 'write', side_effect=PermissionError('read only')):
                async with httpx.AsyncClient(transport=httpx.MockTransport(lambda request: httpx.Response(200, content=png()))) as client:
                    with self.assertLogs('backend.realtime.dem_processor', level='WARNING'):
                        values, valid = await processor._fetch_tile(client, 2, 3)
            self.assertTrue(valid.all())
            self.assertEqual(float(values[0, 0]), 100.0)
            self.assertFalse((Path(folder) / '13/2/3.png').exists())


if __name__ == '__main__':
    unittest.main()
