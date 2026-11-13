"""Shared throttled HTTP transport for TradingView scanner requests.

All modules that talk to the scanner hosts (Screener, Markets, batch
collector) route their POST requests through this module so that
per-host rate limiting is applied consistently, no matter which
entry point issued the request.
"""

import threading
import time
from urllib.parse import urlparse

import requests


class RateLimiter:
    """Simple per-host rate limiter.

    Guarantees a minimum interval between consecutive requests to the
    same host. A rate of zero (or less) disables throttling.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._interval = 0.0
        self._next_allowed = {}

    def configure(self, requests_per_second: float) -> None:
        """Set the global per-host request rate. <= 0 disables throttling."""
        with self._lock:
            if requests_per_second and requests_per_second > 0:
                self._interval = 1.0 / float(requests_per_second)
            else:
                self._interval = 0.0
            self._next_allowed = {}

    def wait(self, url: str) -> None:
        """Block until a request to the URL's host is allowed."""
        host = urlparse(url).netloc
        with self._lock:
            if self._interval <= 0:
                return
            now = time.monotonic()
            slot = self._next_allowed.get(host, now)
            delay = slot - now
            self._next_allowed[host] = max(now, slot) + self._interval
        if delay > 0:
            time.sleep(delay)


limiter = RateLimiter()


def configure_rate_limit(requests_per_second: float) -> None:
    """Configure the shared rate limiter used by all scanner requests."""
    limiter.configure(requests_per_second)


def post(url: str, **kwargs) -> requests.Response:
    """Throttled wrapper around requests.post."""
    limiter.wait(url)
    return requests.post(url, **kwargs)
