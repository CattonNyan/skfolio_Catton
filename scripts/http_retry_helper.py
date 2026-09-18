"""Resilient HTTP Request & Exponential Backoff Retry Utility for Crypto Fetchers.

Features:
- Handles transient network timeouts and socket errors
- Retries on HTTP 429 (Rate Limit / Too Many Requests) and 5xx server errors
- Parses and respects server-sent 'Retry-After' headers
- Applies exponential backoff with random jitter to prevent thundering herd problems
"""

from __future__ import annotations

import functools
import json
import random
import time
import urllib.error
import urllib.request
from typing import Any, Callable

DEFAULT_RETRY_STATUSES = (429, 500, 502, 503, 504)


def execute_request_with_retry(
    req_or_url: str | urllib.request.Request,
    max_retries: int = 3,
    base_delay: float = 0.5,
    backoff_factor: float = 2.0,
    timeout: float = 5.0,
    retry_statuses: tuple[int, ...] = DEFAULT_RETRY_STATUSES,
) -> bytes:
    """Execute urllib HTTP request with exponential backoff and jitter on retryable errors."""
    if max_retries < 0:
        raise ValueError("max_retries must be non-negative.")
    if base_delay < 0:
        raise ValueError("base_delay must be non-negative.")
    if timeout <= 0:
        raise ValueError("timeout must be strictly positive.")

    if isinstance(req_or_url, str):
        req = urllib.request.Request(
            req_or_url,
            headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) skfolio-catton/1.6.0"},
        )
    else:
        req = req_or_url

    attempt = 0
    last_error: Exception | None = None

    while attempt <= max_retries:
        try:
            with urllib.request.urlopen(req, timeout=timeout) as response:
                return response.read()
        except urllib.error.HTTPError as error:
            last_error = error
            if error.code not in retry_statuses or attempt == max_retries:
                raise

            # Check for Retry-After header
            retry_after = error.headers.get("Retry-After") if error.headers else None
            delay = 0.0
            if retry_after:
                try:
                    delay = float(retry_after)
                except (ValueError, TypeError):
                    delay = 0.0

            if delay <= 0:
                jitter = 0.8 + 0.4 * random.random()
                delay = base_delay * (backoff_factor ** attempt) * jitter

            time.sleep(delay)
            attempt += 1

        except (urllib.error.URLError, TimeoutError, ConnectionResetError, OSError) as error:
            last_error = error
            if attempt == max_retries:
                raise

            jitter = 0.8 + 0.4 * random.random()
            delay = base_delay * (backoff_factor ** attempt) * jitter
            time.sleep(delay)
            attempt += 1

    if last_error is not None:
        raise last_error
    raise RuntimeError("Request execution failed unexpectedly without raising.")


def fetch_json_with_retry(
    req_or_url: str | urllib.request.Request,
    max_retries: int = 3,
    base_delay: float = 0.5,
    timeout: float = 5.0,
) -> Any:
    """Fetch URL and parse response body as JSON with automatic retry."""
    raw_bytes = execute_request_with_retry(
        req_or_url=req_or_url,
        max_retries=max_retries,
        base_delay=base_delay,
        timeout=timeout,
    )
    return json.loads(raw_bytes.decode("utf-8"))


def with_retry(
    max_retries: int = 3,
    base_delay: float = 0.5,
) -> Callable:
    """Decorator to apply retry logic to a network-fetching function."""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            attempt = 0
            while attempt <= max_retries:
                try:
                    return func(*args, **kwargs)
                except urllib.error.HTTPError as err:
                    if err.code not in DEFAULT_RETRY_STATUSES or attempt == max_retries:
                        raise
                    time.sleep(base_delay * (2 ** attempt))
                    attempt += 1
                except (urllib.error.URLError, TimeoutError, OSError):
                    if attempt == max_retries:
                        raise
                    time.sleep(base_delay * (2 ** attempt))
                    attempt += 1
        return wrapper
    return decorator
