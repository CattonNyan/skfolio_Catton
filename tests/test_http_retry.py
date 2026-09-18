import io
import unittest
import urllib.error
from unittest.mock import MagicMock, patch

from scripts.http_retry_helper import (
    execute_request_with_retry,
    fetch_json_with_retry,
    with_retry,
)


class HttpRetryTests(unittest.TestCase):
    @patch("urllib.request.urlopen")
    def test_execute_request_success_first_try(self, mock_urlopen):
        mock_resp = MagicMock()
        mock_resp.read.return_value = b'{"status": "ok"}'
        mock_resp.__enter__.return_value = mock_resp
        mock_urlopen.return_value = mock_resp

        data = execute_request_with_retry("https://api.example.com", max_retries=2, base_delay=0.01)
        self.assertEqual(data, b'{"status": "ok"}')
        self.assertEqual(mock_urlopen.call_count, 1)

    @patch("time.sleep", return_value=None)
    @patch("urllib.request.urlopen")
    def test_retry_on_429_then_succeed(self, mock_urlopen, mock_sleep):
        err_429 = urllib.error.HTTPError(
            url="https://api.example.com",
            code=429,
            msg="Too Many Requests",
            hdrs={"Retry-After": "0.1"},
            fp=io.BytesIO(b"Rate limit exceeded"),
        )
        mock_success = MagicMock()
        mock_success.read.return_value = b'{"result": "success"}'
        mock_success.__enter__.return_value = mock_success

        mock_urlopen.side_effect = [err_429, mock_success]

        data = fetch_json_with_retry("https://api.example.com", max_retries=2, base_delay=0.01)
        self.assertEqual(data, {"result": "success"})
        self.assertEqual(mock_urlopen.call_count, 2)
        mock_sleep.assert_called_once()

    @patch("urllib.request.urlopen")
    def test_non_retryable_404_error_raises_immediately(self, mock_urlopen):
        err_404 = urllib.error.HTTPError(
            url="https://api.example.com/notfound",
            code=404,
            msg="Not Found",
            hdrs={},
            fp=io.BytesIO(b"Not Found"),
        )
        mock_urlopen.side_effect = err_404

        with self.assertRaises(urllib.error.HTTPError) as ctx:
            execute_request_with_retry("https://api.example.com/notfound", max_retries=3, base_delay=0.01)
        self.assertEqual(ctx.exception.code, 404)
        self.assertEqual(mock_urlopen.call_count, 1)

    @patch("time.sleep", return_value=None)
    @patch("urllib.request.urlopen")
    def test_max_retries_exceeded(self, mock_urlopen, mock_sleep):
        err_503 = urllib.error.HTTPError(
            url="https://api.example.com",
            code=503,
            msg="Service Unavailable",
            hdrs={},
            fp=io.BytesIO(b"Service Unavailable"),
        )
        mock_urlopen.side_effect = err_503

        with self.assertRaises(urllib.error.HTTPError):
            execute_request_with_retry("https://api.example.com", max_retries=2, base_delay=0.01)
        self.assertEqual(mock_urlopen.call_count, 3)

    def test_invalid_parameters(self):
        with self.assertRaises(ValueError):
            execute_request_with_retry("https://api.example.com", max_retries=-1)
        with self.assertRaises(ValueError):
            execute_request_with_retry("https://api.example.com", base_delay=-0.5)
        with self.assertRaises(ValueError):
            execute_request_with_retry("https://api.example.com", timeout=0.0)


if __name__ == "__main__":
    unittest.main()
