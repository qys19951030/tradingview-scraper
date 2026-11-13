import json
import os
import sys

import pytest
import requests

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
from tradingview_scraper.symbols.screener import Screener


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


class FakeResponse:
    def __init__(self, status_code=200, payload=None, headers=None, text=""):
        self.status_code = status_code
        self._payload = payload
        self.headers = headers or {}
        self.text = text

    def json(self):
        if self._payload is None:
            raise ValueError("no json")
        return self._payload


def make_rows(symbols):
    return {
        "data": [
            {"s": symbol, "d": [symbol, 100.0, 1.5, 5000, 1e9]}
            for symbol in symbols
        ],
        "totalCount": len(symbols),
    }


class FakeTransport:
    """Scripted scanner transport keyed by (market, range_start)."""

    def __init__(self):
        self.pages = {}
        self.calls = []

    def add_page(self, market, start, response):
        self.pages.setdefault((market, start), []).append(response)

    def __call__(self, url, json=None, headers=None, timeout=None):
        market = next(
            name for name, u in Screener.SUPPORTED_MARKETS.items() if u == url
        )
        start = json["range"][0]
        self.calls.append((market, start))
        queue = self.pages[(market, start)]
        if len(queue) > 1:
            return queue.pop(0)
        return queue[0]


def run_collector(tmp_path, transport, **spec_overrides):
    spec = BatchSpec.from_dict(make_spec_dict(**spec_overrides))
    out_dir = str(tmp_path / "out")
    collector = BatchCollector(
        spec=spec,
        out_dir=out_dir,
        transport_fn=transport,
        sleep_fn=lambda _: None,
    )
    rc = collector.run()
    return rc, out_dir


def read_jsonl(path):
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


class TestCollection:
    def test_pagination_and_duplicate_skip(self, tmp_path):
        transport = FakeTransport()
        transport.add_page("america", 0, FakeResponse(200, {
            "data": [
                {"s": "NYSE:A", "d": ["A", 1.0, 0.1, 100, 1e6]},
                {"s": "NYSE:B", "d": ["B", 2.0, 0.2, 200, 2e6]},
            ],
            "totalCount": 5,
        }))
        transport.add_page("america", 2, FakeResponse(200, {
            "data": [
                {"s": "NYSE:B", "d": ["B", 2.0, 0.2, 200, 2e6]},
                {"s": "NYSE:C", "d": ["C", 3.0, 0.3, 300, 3e6]},
            ],
            "totalCount": 5,
        }))
        transport.add_page("america", 4, FakeResponse(200, {
            "data": [
                {"s": "NYSE:D", "d": ["D", 4.0, 0.4, 400, 4e6]},
            ],
            "totalCount": 5,
        }))
        transport.add_page("uk", 0, FakeResponse(200, {
            "data": [{"s": "LSE:U", "d": ["U", 9.0, 0.9, 900, 9e6]}],
            "totalCount": 1,
        }))

        rc, out_dir = run_collector(tmp_path, transport)
        assert rc == 0

        dataset = read_jsonl(os.path.join(out_dir, "dataset.jsonl"))
        assert [r["symbol"] for r in dataset] == [
            "NYSE:A", "NYSE:B", "NYSE:C", "NYSE:D", "LSE:U",
        ]
        assert dataset[0]["market"] == "america"
        assert dataset[0]["name"] == "A"
        assert dataset[0]["close"] == 1.0

        manifest = json.loads(
            open(os.path.join(out_dir, "manifest.json"), encoding="utf-8").read()
        )
        america = manifest["markets"]["america"]
        assert america["status"] == "success"
        assert america["pages_fetched"] == 3
        assert america["written"] == 4
        assert america["skipped"] == 1
        assert america["received"] == 5
        assert america["last_error"] is None
        assert manifest["markets"]["uk"]["written"] == 1

    def test_stops_on_empty_page(self, tmp_path):
        transport = FakeTransport()
        transport.add_page("america", 0, FakeResponse(200, make_rows(["S1", "S2"])))
        transport.add_page("america", 2, FakeResponse(200, {"data": [], "totalCount": 100}))
        transport.add_page("uk", 0, FakeResponse(200, make_rows(["U1"])))

        rc, out_dir = run_collector(tmp_path, transport, max_rows=10)
        assert rc == 0
        assert (("america", 4) not in transport.calls)
        manifest = json.loads(
            open(os.path.join(out_dir, "manifest.json"), encoding="utf-8").read()
        )
        assert manifest["markets"]["america"]["status"] == "success"

    def test_max_rows_caps_collection(self, tmp_path):
        transport = FakeTransport()
        transport.add_page("america", 0, FakeResponse(200, make_rows(["S1", "S2"])))
        transport.add_page("uk", 0, FakeResponse(200, make_rows(["U1"])))

        rc, out_dir = run_collector(tmp_path, transport, max_rows=2)
        assert rc == 0
        dataset = read_jsonl(os.path.join(out_dir, "dataset.jsonl"))
        assert [r["symbol"] for r in dataset] == ["S1", "S2", "U1"]

    def test_429_retry_after_then_success(self, tmp_path):
        sleeps = []
        transport = FakeTransport()
        transport.add_page("america", 0, FakeResponse(429, headers={"Retry-After": "7"}))
        transport.add_page("america", 0, FakeResponse(200, make_rows(["S1"])))
        transport.add_page("uk", 0, FakeResponse(200, make_rows(["U1"])))

        spec = BatchSpec.from_dict(make_spec_dict(max_rows=5))
        collector = BatchCollector(
            spec=spec,
            out_dir=str(tmp_path / "out"),
            transport_fn=transport,
            sleep_fn=sleeps.append,
        )
        rc = collector.run()
        assert rc == 0
        assert sleeps == [7.0]
        dataset = read_jsonl(os.path.join(str(tmp_path / "out"), "dataset.jsonl"))
        assert [r["symbol"] for r in dataset] == ["S1", "U1"]

    def test_retry_exhaustion_fails_single_market(self, tmp_path):
        transport = FakeTransport()
        for _ in range(5):
            transport.add_page("america", 0, FakeResponse(500, text="boom"))
        transport.add_page("uk", 0, FakeResponse(200, make_rows(["U1"])))

        rc, out_dir = run_collector(tmp_path, transport)
        assert rc == 2

        manifest = json.loads(
            open(os.path.join(out_dir, "manifest.json"), encoding="utf-8").read()
        )
        assert manifest["markets"]["america"]["status"] == "failed"
        assert "500" in manifest["markets"]["america"]["last_error"]
        assert manifest["markets"]["uk"]["status"] == "success"

        dataset = read_jsonl(os.path.join(out_dir, "dataset.jsonl"))
        assert [r["symbol"] for r in dataset] == ["U1"]

    def test_all_markets_failed_returns_1(self, tmp_path):
        transport = FakeTransport()
        for _ in range(5):
            transport.add_page("america", 0, FakeResponse(500))
            transport.add_page("uk", 0, FakeResponse(500))

        rc, out_dir = run_collector(tmp_path, transport)
        assert rc == 1

    def test_connection_error_retried(self, tmp_path):
        calls = []

        def flaky_transport(url, json=None, headers=None, timeout=None):
            calls.append(url)
            if len(calls) == 1:
                raise requests.ConnectionError("reset by peer")
            if "uk" in url:
                return FakeResponse(200, make_rows(["U1"]))
            return FakeResponse(200, make_rows(["S1"]))

        rc, out_dir = run_collector(tmp_path, flaky_transport, max_rows=5)
        assert rc == 0
        assert len(calls) == 3

    def test_read_timeout_retried(self, tmp_path):
        calls = []

        def flaky_transport(url, json=None, headers=None, timeout=None):
            calls.append(url)
            if len(calls) == 1:
                raise requests.ReadTimeout("timed out")
            if "uk" in url:
                return FakeResponse(200, make_rows(["U1"]))
            return FakeResponse(200, make_rows(["S1"]))

        rc, out_dir = run_collector(tmp_path, flaky_transport, max_rows=5)
        assert rc == 0
        assert len(calls) == 3


class TestCache:
    def test_cache_hit_skips_request(self, tmp_path):
        transport = FakeTransport()
        transport.add_page("america", 0, FakeResponse(200, make_rows(["S1"])))
        transport.add_page("uk", 0, FakeResponse(200, make_rows(["U1"])))

        out_dir = str(tmp_path / "out")
        spec = BatchSpec.from_dict(make_spec_dict(max_rows=5))
        rc = BatchCollector(
            spec=spec, out_dir=out_dir, transport_fn=transport, sleep_fn=lambda _: None
        ).run()
        assert rc == 0
        assert len(transport.calls) == 2
        assert os.path.isdir(os.path.join(out_dir, "cache"))

        def fail_transport(url, json=None, headers=None, timeout=None):
            raise AssertionError("transport must not be called on cache hit")

        rc = BatchCollector(
            spec=spec, out_dir=out_dir, fresh=True,
            transport_fn=fail_transport, sleep_fn=lambda _: None,
        ).run()
        assert rc == 0
        dataset = read_jsonl(os.path.join(out_dir, "dataset.jsonl"))
        assert [r["symbol"] for r in dataset] == ["S1", "U1"]

    def test_refresh_bypasses_cache(self, tmp_path):
        transport = FakeTransport()
        transport.add_page("america", 0, FakeResponse(200, make_rows(["S1"])))
        transport.add_page("america", 0, FakeResponse(200, make_rows(["S1"])))
        transport.add_page("uk", 0, FakeResponse(200, make_rows(["U1"])))
        transport.add_page("uk", 0, FakeResponse(200, make_rows(["U1"])))

        out_dir = str(tmp_path / "out")
        spec = BatchSpec.from_dict(make_spec_dict(max_rows=5))
        rc = BatchCollector(
            spec=spec, out_dir=out_dir, transport_fn=transport, sleep_fn=lambda _: None
        ).run()
        assert rc == 0

        rc = BatchCollector(
            spec=spec, out_dir=out_dir, fresh=True, refresh=True,
            transport_fn=transport, sleep_fn=lambda _: None,
        ).run()
        assert rc == 0
        assert len(transport.calls) == 4


def paged_transport():
    """Transport serving three america pages and one uk page."""
    transport = FakeTransport()
    transport.add_page("america", 0, FakeResponse(200, {
        "data": [
            {"s": "NYSE:A", "d": ["A", 1.0, 0.1, 100, 1e6]},
            {"s": "NYSE:B", "d": ["B", 2.0, 0.2, 200, 2e6]},
        ],
        "totalCount": 5,
    }))
    transport.add_page("america", 2, FakeResponse(200, {
        "data": [
            {"s": "NYSE:C", "d": ["C", 3.0, 0.3, 300, 3e6]},
            {"s": "NYSE:D", "d": ["D", 4.0, 0.4, 400, 4e6]},
        ],
        "totalCount": 5,
    }))
    transport.add_page("america", 4, FakeResponse(200, {
        "data": [{"s": "NYSE:E", "d": ["E", 5.0, 0.5, 500, 5e6]}],
        "totalCount": 5,
    }))
    transport.add_page("uk", 0, FakeResponse(200, {
        "data": [{"s": "LSE:U", "d": ["U", 9.0, 0.9, 900, 9e6]}],
        "totalCount": 1,
    }))
    return transport


class TestCheckpoint:
    def test_resume_after_crash_matches_single_run(self, tmp_path):
        spec = BatchSpec.from_dict(make_spec_dict(max_rows=10))

        # Reference: one uninterrupted run.
        out_ref = str(tmp_path / "out_ref")
        rc = BatchCollector(
            spec=spec, out_dir=out_ref,
            transport_fn=paged_transport(), sleep_fn=lambda _: None,
        ).run()
        assert rc == 0

        # Crashed run: america's second page fails non-retryably.
        out_resumed = str(tmp_path / "out_resumed")
        crashed = paged_transport()
        crashed.pages[("america", 2)] = [FakeResponse(400, text="bad request")]
        rc = BatchCollector(
            spec=spec, out_dir=out_resumed,
            transport_fn=crashed, sleep_fn=lambda _: None,
        ).run()
        assert rc == 2

        # Resume: only the missing pages may be fetched.
        resumed = paged_transport()
        rc = BatchCollector(
            spec=spec, out_dir=out_resumed,
            transport_fn=resumed, sleep_fn=lambda _: None,
        ).run()
        assert rc == 0
        assert ("america", 0) not in resumed.calls
        assert ("uk", 0) not in resumed.calls
        assert resumed.calls == [("america", 2), ("america", 4)]

        with open(os.path.join(out_ref, "dataset.jsonl"), "rb") as fh:
            reference = fh.read()
        with open(os.path.join(out_resumed, "dataset.jsonl"), "rb") as fh:
            resumed_bytes = fh.read()
        assert resumed_bytes == reference

        manifest = json.loads(
            open(os.path.join(out_resumed, "manifest.json"), encoding="utf-8").read()
        )
        assert manifest["markets"]["america"]["status"] == "success"
        assert manifest["markets"]["america"]["pages_fetched"] == 3

    def test_rerun_after_success_is_byte_identical(self, tmp_path):
        spec = BatchSpec.from_dict(make_spec_dict(max_rows=10))
        out_dir = str(tmp_path / "out")
        rc = BatchCollector(
            spec=spec, out_dir=out_dir,
            transport_fn=paged_transport(), sleep_fn=lambda _: None,
        ).run()
        assert rc == 0
        dataset_path = os.path.join(out_dir, "dataset.jsonl")
        with open(dataset_path, "rb") as fh:
            first = fh.read()

        def fail_transport(url, json=None, headers=None, timeout=None):
            raise AssertionError("no requests expected on completed checkpoint")

        rc = BatchCollector(
            spec=spec, out_dir=out_dir,
            transport_fn=fail_transport, sleep_fn=lambda _: None,
        ).run()
        assert rc == 0
        with open(dataset_path, "rb") as fh:
            assert fh.read() == first

    def test_fresh_discards_checkpoint(self, tmp_path):
        spec = BatchSpec.from_dict(make_spec_dict(max_rows=10))
        out_dir = str(tmp_path / "out")
        rc = BatchCollector(
            spec=spec, out_dir=out_dir,
            transport_fn=paged_transport(), sleep_fn=lambda _: None,
        ).run()
        assert rc == 0

        transport = paged_transport()
        rc = BatchCollector(
            spec=spec, out_dir=out_dir, fresh=True, refresh=True,
            transport_fn=transport, sleep_fn=lambda _: None,
        ).run()
        assert rc == 0
        assert len(transport.calls) == 4
        dataset = read_jsonl(os.path.join(out_dir, "dataset.jsonl"))
        assert [r["symbol"] for r in dataset] == [
            "NYSE:A", "NYSE:B", "NYSE:C", "NYSE:D", "NYSE:E", "LSE:U",
        ]


class TestValidation:
    def test_bad_rows_quarantined_and_counted(self, tmp_path):
        transport = FakeTransport()
        transport.add_page("america", 0, FakeResponse(200, {
            "data": [
                {"s": "NYSE:OK", "d": ["Good", 1.0, 0.1, 100, 1e6]},
                {"s": "", "d": ["NoSymbol", 1.0, 0.1, 100, 1e6]},
                {"s": "NYSE:SHORT", "d": ["TooFew", 1.0]},
                {"s": "NYSE:BADTYPE", "d": ["Bad", "not-a-number", 0.1, 100, 1e6]},
                {"s": "NYSE:BADNAME", "d": [123, 1.0, 0.1, 100, 1e6]},
                {"s": "NYSE:NULLS", "d": ["Nulls", None, None, None, None]},
            ],
            "totalCount": 6,
        }))
        transport.add_page("uk", 0, FakeResponse(200, {
            "data": [{"s": "LSE:U", "d": ["U", 9.0, 0.9, 900, 9e6]}],
            "totalCount": 1,
        }))

        rc, out_dir = run_collector(tmp_path, transport, max_rows=10, page_size=10)
        assert rc == 0

        dataset = read_jsonl(os.path.join(out_dir, "dataset.jsonl"))
        assert [r["symbol"] for r in dataset] == ["NYSE:OK", "NYSE:NULLS", "LSE:U"]

        quarantine = read_jsonl(os.path.join(out_dir, "quarantine.jsonl"))
        reasons = {r["symbol"]: r["reason"] for r in quarantine}
        assert reasons == {
            "": "empty_symbol",
            "NYSE:SHORT": "missing_columns",
            "NYSE:BADTYPE": "type_mismatch:close",
            "NYSE:BADNAME": "type_mismatch:name",
        }
        assert all(r["market"] == "america" for r in quarantine)

        manifest = json.loads(
            open(os.path.join(out_dir, "manifest.json"), encoding="utf-8").read()
        )
        america = manifest["markets"]["america"]
        assert america["received"] == 6
        assert america["written"] == 2
        assert america["quarantined"] == 4
        assert america["skipped"] == 0
        assert (
            america["written"] + america["quarantined"] + america["skipped"]
            == america["received"]
        )

    def test_quarantined_symbol_blocks_later_duplicate(self, tmp_path):
        transport = FakeTransport()
        transport.add_page("america", 0, FakeResponse(200, {
            "data": [
                {"s": "NYSE:DUP", "d": ["short"]},
                {"s": "NYSE:DUP", "d": ["Dup", 1.0, 0.1, 100, 1e6]},
            ],
            "totalCount": 2,
        }))
        transport.add_page("uk", 0, FakeResponse(200, {
            "data": [{"s": "LSE:U", "d": ["U", 9.0, 0.9, 900, 9e6]}],
            "totalCount": 1,
        }))

        rc, out_dir = run_collector(tmp_path, transport, max_rows=10, page_size=10)
        assert rc == 0
        manifest = json.loads(
            open(os.path.join(out_dir, "manifest.json"), encoding="utf-8").read()
        )
        america = manifest["markets"]["america"]
        assert america["quarantined"] == 1
        assert america["skipped"] == 1
        assert america["written"] == 0

    def test_bool_is_not_numeric(self, tmp_path):
        transport = FakeTransport()
        transport.add_page("america", 0, FakeResponse(200, {
            "data": [{"s": "NYSE:B", "d": ["B", True, 0.1, 100, 1e6]}],
            "totalCount": 1,
        }))
        transport.add_page("uk", 0, FakeResponse(200, {
            "data": [{"s": "LSE:U", "d": ["U", 9.0, 0.9, 900, 9e6]}],
            "totalCount": 1,
        }))

        rc, out_dir = run_collector(tmp_path, transport, max_rows=10, page_size=10)
        assert rc == 0
        quarantine = read_jsonl(os.path.join(out_dir, "quarantine.jsonl"))
        assert [r["reason"] for r in quarantine] == ["type_mismatch:close"]

    def test_expired_cache_refetches(self, tmp_path):
        transport = FakeTransport()
        transport.add_page("america", 0, FakeResponse(200, make_rows(["S1"])))
        transport.add_page("america", 0, FakeResponse(200, make_rows(["S1"])))
        transport.add_page("uk", 0, FakeResponse(200, make_rows(["U1"])))
        transport.add_page("uk", 0, FakeResponse(200, make_rows(["U1"])))

        out_dir = str(tmp_path / "out")
        spec = BatchSpec.from_dict(make_spec_dict(max_rows=5))
        rc = BatchCollector(
            spec=spec, out_dir=out_dir, transport_fn=transport,
            sleep_fn=lambda _: None, cache_ttl_seconds=0,
        ).run()
        assert rc == 0

        rc = BatchCollector(
            spec=spec, out_dir=out_dir, fresh=True, transport_fn=transport,
            sleep_fn=lambda _: None, cache_ttl_seconds=0,
        ).run()
        assert rc == 0
        assert len(transport.calls) == 4
