"""Offline checks for the production launcher's volume and process boundaries."""
import importlib.util
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('terra_container_start', ROOT / 'deployment/start.py')
start = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(start)


class ContainerStart(unittest.TestCase):
    def test_port_is_validated_before_launch(self):
        self.assertEqual(start.server_port({}), 8000)
        self.assertEqual(start.server_port({'PORT': '34567'}), 34567)
        for value in ('', '0', '65536', '-1', '1.2', '8000; echo unsafe', '８０００'):
            with self.subTest(value=value), self.assertRaises(ValueError):
                start.server_port({'PORT': value})

    def test_non_root_prepares_a_writable_cache(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(start.os, 'geteuid', return_value=10001, create=True):
            path = Path(tmp) / 'cache'
            self.assertEqual(start.prepare_cache({'TERRA_TERRAIN_CACHE_DIR': str(path)}), path)
            self.assertTrue(path.is_dir())
            self.assertEqual(list(path.iterdir()), [])

    @unittest.skipUnless(os.name == 'posix', 'Container privilege changes are POSIX-only')
    def test_root_bootstrap_changes_only_cache_then_drops_privileges(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            cache = root / 'terrarium'
            with patch.object(start, 'VOLUME_ROOT', root), \
                    patch.object(start.os, 'geteuid', return_value=0), \
                    patch.object(start.os, 'chown') as chown, \
                    patch.object(start.os, 'setgroups') as groups, \
                    patch.object(start.os, 'setgid') as gid, \
                    patch.object(start.os, 'setuid') as uid:
                self.assertEqual(start.prepare_cache({'TERRA_TERRAIN_CACHE_DIR': str(cache)}), cache)
                chown.assert_called_once_with(cache, 10001, 10001)
                groups.assert_called_once_with([])
                gid.assert_called_once_with(10001)
                uid.assert_called_once_with(10001)

    @unittest.skipUnless(os.name == 'posix', 'Container privilege changes are POSIX-only')
    def test_root_bootstrap_refuses_other_directories_and_escaping_symlinks(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            mount = root / 'data'
            mount.mkdir()
            other = root / 'other'
            other.mkdir()
            link = mount / 'link'
            link.symlink_to(other, target_is_directory=True)
            with patch.object(start, 'VOLUME_ROOT', mount), \
                    patch.object(start.os, 'geteuid', return_value=0), \
                    patch.object(start.os, 'chown') as chown:
                for path in (mount, other, link, Path('relative')):
                    with self.subTest(path=path), self.assertRaises(ValueError):
                        start.prepare_cache({'TERRA_TERRAIN_CACHE_DIR': str(path)})
                chown.assert_not_called()

    def test_launch_replaces_helper_with_one_worker_without_a_shell(self):
        with patch.dict(start.os.environ, {'PORT': '9123', 'WEB_CONCURRENCY': '12'}, clear=True), \
                patch.object(start, 'prepare_cache') as cache, \
                patch.object(start.os, 'execv') as execv:
            start.main()
            cache.assert_called_once()
            args = execv.call_args.args[1]
            self.assertEqual(args[:4], [start.sys.executable, '-m', 'uvicorn', 'backend.api.main:app'])
            self.assertEqual(args[args.index('--port') + 1], '9123')
            self.assertEqual(args[args.index('--workers') + 1], '1')


if __name__ == '__main__':
    unittest.main()
