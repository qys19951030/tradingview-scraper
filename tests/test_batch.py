import json
import os
import sys

import pytest

# Add the current working directory to the system path
path = str(os.getcwd())
if path not in sys.path:
    sys.path.append(path)

from tradingview_scraper.batch import (
    BatchCollector,
    BatchSpec,
    SpecError,
    load_spec,
    main,
)
from tradingview_scraper.symbols import http_transport


def make_spec_dict(**overrides):
    spec = {
        "markets": ["america", "uk"],
        "columns": ["name", "close", "change", "volume", "market_cap_basic"],
        "filters": [],
        "page_size": 2,
        "max_rows": 10,
        "requests_per_second": 1000,
    }
    spec.update(overrides)
    return spec


def write_spec(tmp_path, **overrides):
    spec_path = tmp_path / "spec.json"
    spec_path.write_text(json.dumps(make_spec_dict(**overrides)), encoding="utf-8")
    return str(spec_path)


class TestSpecParsing:
    def test_load_valid_spec(self, tmp_path):
        spec = load_spec(write_spec(tmp_path))
        assert spec.markets == ["america", "uk"]
        assert spec.columns == ["name", "close", "change", "volume", "market_cap_basic"]
        assert spec.page_size == 2
        assert spec.max_rows == 10
        assert spec.requests_per_second == 1000.0

    def test_defaults_applied(self, tmp_path):
        spec_path = tmp_path / "spec.json"
        spec_path.write_text(json.dumps({
            "markets": ["america"],
            "columns": ["name", "close"],
        }), encoding="utf-8")
        spec = load_spec(str(spec_path))
        assert spec.page_size == 100
        assert spec.max_rows == 1000
        assert spec.requests_per_second == 1.0
        assert spec.filters is None

    def test_missing_markets_rejected(self, tmp_path):
        with pytest.raises(SpecError):
            load_spec(write_spec(tmp_path, markets=[]))

    def test_unknown_market_rejected(self, tmp_path):
        with pytest.raises(SpecError, match="Unsupported markets"):
            load_spec(write_spec(tmp_path, markets=["atlantis"]))

    def test_missing_columns_rejected(self, tmp_path):
        with pytest.raises(SpecError):
            load_spec(write_spec(tmp_path, columns=[]))

    def test_invalid_page_size_rejected(self, tmp_path):
        with pytest.raises(SpecError):
            load_spec(write_spec(tmp_path, page_size=0))

    def test_invalid_max_rows_rejected(self, tmp_path):
        with pytest.raises(SpecError):
            load_spec(write_spec(tmp_path, max_rows=-5))

    def test_invalid_rate_rejected(self, tmp_path):
        with pytest.raises(SpecError):
            load_spec(write_spec(tmp_path, requests_per_second=0))

    def test_missing_file_rejected(self, tmp_path):
        with pytest.raises(SpecError):
            load_spec(str(tmp_path / "nope.json"))

    def test_invalid_json_rejected(self, tmp_path):
        spec_path = tmp_path / "spec.json"
        spec_path.write_text("{not json", encoding="utf-8")
        with pytest.raises(SpecError):
            load_spec(str(spec_path))

    def test_cli_bad_spec_returns_1(self, tmp_path, capsys):
        spec_path = tmp_path / "spec.json"
        spec_path.write_text("{}", encoding="utf-8")
        rc = main(["--spec", str(spec_path), "--out", str(tmp_path / "out")])
        assert rc == 1


class TestRateLimiter:
    def test_throttles_per_host(self, monkeypatch):
        sleeps = []
        monkeypatch.setattr(http_transport.time, "sleep", lambda s: sleeps.append(s))
        limiter = http_transport.RateLimiter()
        limiter.configure(2)  # one request every 0.5s per host
        limiter.wait("https://scanner.tradingview.com/america/scan")
        limiter.wait("https://scanner.tradingview.com/america/scan")
        limiter.wait("https://scanner.tradingview.com/america/scan")
        assert len(sleeps) == 2
        assert sleeps[0] == pytest.approx(0.5, abs=1e-3)
        assert sleeps[1] == pytest.approx(1.0, abs=1e-3)

    def test_hosts_throttled_independently(self, monkeypatch):
        sleeps = []
        monkeypatch.setattr(http_transport.time, "sleep", lambda s: sleeps.append(s))
        limiter = http_transport.RateLimiter()
        limiter.configure(2)
        limiter.wait("https://scanner.tradingview.com/america/scan")
        limiter.wait("https://example.com/other")
        assert sleeps == []

    def test_zero_rate_disables_throttling(self, monkeypatch):
        sleeps = []
        monkeypatch.setattr(http_transport.time, "sleep", lambda s: sleeps.append(s))
        limiter = http_transport.RateLimiter()
        limiter.configure(0)
        limiter.wait("https://scanner.tradingview.com/america/scan")
        limiter.wait("https://scanner.tradingview.com/america/scan")
        assert sleeps == []
