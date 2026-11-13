"""Startup checks tolerate transient resets without hiding a persistent failure."""
import contextlib
import io
import json
import unittest
from unittest.mock import patch

from scripts import check_web_deployment as deployment_check


class StartupRetry(unittest.TestCase):
    def test_connection_reset_is_retried_then_ready_result_is_reported(self):
        result = {'passed': True, 'model_ready': True, 'provider_requests': 0}
        output = io.StringIO()
        with patch('sys.argv', ['check_web_deployment.py', 'http://localhost:8000', '--wait-seconds', '5']), \
                patch.object(deployment_check, 'check', side_effect=[ConnectionResetError('Starting'), result]) as check, \
                patch.object(deployment_check.time, 'monotonic', side_effect=[10., 10.1]), \
                patch.object(deployment_check.time, 'sleep') as sleep, \
                contextlib.redirect_stdout(output):
            deployment_check.main()
        self.assertEqual(check.call_args_list, [unittest.mock.call('http://localhost:8000')] * 2)
        sleep.assert_called_once_with(1)
        self.assertEqual(json.loads(output.getvalue()), result)

    def test_persistent_connection_reset_fails_at_the_deadline(self):
        output = io.StringIO()
        with patch('sys.argv', ['check_web_deployment.py', 'http://localhost:8000', '--wait-seconds', '2']), \
                patch.object(deployment_check, 'check', side_effect=ConnectionResetError('Still unavailable')) as check, \
                patch.object(deployment_check.time, 'monotonic', side_effect=[10., 10.5, 12.]), \
                patch.object(deployment_check.time, 'sleep') as sleep, \
                contextlib.redirect_stdout(output):
            with self.assertRaisesRegex(ConnectionResetError, 'Still unavailable'):
                deployment_check.main()
        self.assertEqual(check.call_count, 2)
        sleep.assert_called_once_with(1)
        self.assertEqual(output.getvalue(), '')


if __name__ == '__main__':
    unittest.main()
