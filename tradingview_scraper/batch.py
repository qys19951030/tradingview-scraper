"""Batch collection of screener data across markets.

Provides the BatchCollector class and a command line entry point:

    python -m tradingview_scraper.batch --spec spec.json --out outdir

The spec file is JSON with the following keys:
    markets (list[str]): markets to collect, processed in order.
    columns (list[str]): columns to request from the scanner.
    filters (list[dict]): passed through to the scanner as-is.
    page_size (int): rows per page request.
    max_rows (int): per-market row cap.
    requests_per_second (float): per-host throttle rate.
"""

import argparse
import hashlib
import json
import logging
import os
import sys
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

import requests

from tradingview_scraper.symbols import http_transport
from tradingview_scraper.symbols.screener import Screener
from tradingview_scraper.symbols.utils import generate_user_agent


logger = logging.getLogger(__name__)

DEFAULT_PAGE_SIZE = 100
DEFAULT_MAX_ROWS = 1000
DEFAULT_REQUESTS_PER_SECOND = 1.0
DEFAULT_CACHE_TTL_SECONDS = 30 * 60
REQUEST_TIMEOUT_SECONDS = 30
MAX_PAGE_ATTEMPTS = 5


class SpecError(ValueError):
    """Raised when the batch spec file is invalid."""


class PageFetchError(RuntimeError):
    """Raised when a page cannot be fetched after all retries."""


@dataclass
class BatchSpec:
    """Parsed and validated batch job description."""

    markets: List[str]
    columns: List[str]
    filters: Optional[List[Dict[str, Any]]] = None
    page_size: int = DEFAULT_PAGE_SIZE
    max_rows: int = DEFAULT_MAX_ROWS
    requests_per_second: float = DEFAULT_REQUESTS_PER_SECOND

    @classmethod
    def from_dict(cls, raw: Dict[str, Any]) -> "BatchSpec":
        if not isinstance(raw, dict):
            raise SpecError("Spec must be a JSON object")

        markets = raw.get("markets")
        if not isinstance(markets, list) or not markets or not all(
            isinstance(m, str) and m for m in markets
        ):
            raise SpecError("Spec 'markets' must be a non-empty list of market names")
        unknown = [m for m in markets if m not in Screener.SUPPORTED_MARKETS]
        if unknown:
            raise SpecError(f"Unsupported markets: {', '.join(unknown)}")

        columns = raw.get("columns")
        if not isinstance(columns, list) or not columns or not all(
            isinstance(c, str) and c for c in columns
        ):
            raise SpecError("Spec 'columns' must be a non-empty list of column names")

        filters = raw.get("filters")
        if filters is not None and not isinstance(filters, list):
            raise SpecError("Spec 'filters' must be a list when provided")

        page_size = raw.get("page_size", DEFAULT_PAGE_SIZE)
        if not isinstance(page_size, int) or isinstance(page_size, bool) or page_size <= 0:
            raise SpecError("Spec 'page_size' must be a positive integer")

        max_rows = raw.get("max_rows", DEFAULT_MAX_ROWS)
        if not isinstance(max_rows, int) or isinstance(max_rows, bool) or max_rows <= 0:
            raise SpecError("Spec 'max_rows' must be a positive integer")

        rps = raw.get("requests_per_second", DEFAULT_REQUESTS_PER_SECOND)
        if not isinstance(rps, (int, float)) or isinstance(rps, bool) or rps <= 0:
            raise SpecError("Spec 'requests_per_second' must be a positive number")

        return cls(
            markets=list(markets),
            columns=list(columns),
            filters=filters,
            page_size=page_size,
            max_rows=max_rows,
            requests_per_second=float(rps),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "markets": list(self.markets),
            "columns": list(self.columns),
            "filters": self.filters,
            "page_size": self.page_size,
            "max_rows": self.max_rows,
            "requests_per_second": self.requests_per_second,
        }

    def fingerprint(self) -> str:
        raw = json.dumps(self.to_dict(), sort_keys=True)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def load_spec(path: str) -> BatchSpec:
    """Load and validate a batch spec from a JSON file."""
    try:
        with open(path, "r", encoding="utf-8") as fh:
            raw = json.load(fh)
    except OSError as exc:
        raise SpecError(f"Cannot read spec file {path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise SpecError(f"Spec file {path} is not valid JSON: {exc}") from exc
    return BatchSpec.from_dict(raw)


class BatchCollector:
    """Collects full screener datasets for multiple markets."""

    def __init__(
        self,
        spec: BatchSpec,
        out_dir: str,
        refresh: bool = False,
        fresh: bool = False,
        transport_fn: Optional[Callable] = None,
        sleep_fn: Callable[[float], None] = time.sleep,
        cache_ttl_seconds: int = DEFAULT_CACHE_TTL_SECONDS,
    ):
        self.spec = spec
        self.out_dir = out_dir
        self.refresh = refresh
        self.fresh = fresh
        self.transport_fn = transport_fn or http_transport.post
        self.sleep_fn = sleep_fn
        self.cache_ttl_seconds = cache_ttl_seconds
        self.headers = {"User-Agent": generate_user_agent()}
        self._checkpoint: Dict[str, Any] = {"markets": {}}

    def run(self) -> int:
        """Run the batch job and return the process exit code."""
        os.makedirs(self.out_dir, exist_ok=True)
        http_transport.configure_rate_limit(self.spec.requests_per_second)

        if self.fresh:
            self._reset_outputs()
        self._checkpoint = self._load_checkpoint()

        results = {}
        for market in self.spec.markets:
            previous = self._checkpoint["markets"].get(market)
            if previous and previous.get("status") == "success":
                logger.info("market %s already complete, skipping", market)
                results[market] = previous
                continue
            logger.info("collecting market %s", market)
            results[market] = self._collect_market(market, previous)
            self._checkpoint["markets"][market] = results[market]
            self._save_checkpoint()

        self._assemble_outputs()
        self._write_manifest(results)
        if not self._verify_counts(results):
            return 1

        statuses = [state["status"] for state in results.values()]
        if all(status == "success" for status in statuses):
            return 0
        if any(status == "success" for status in statuses):
            return 2
        return 1

    def _collect_market(
        self, market: str, previous: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Fetch all pages for one market, writing rows as pages complete."""
        state = previous or {
            "status": "in_progress",
            "pages_fetched": 0,
            "received": 0,
            "written": 0,
            "quarantined": 0,
            "skipped": 0,
            "last_error": None,
            "next_offset": 0,
        }
        state["status"] = "in_progress"
        state["last_error"] = None
        seen = self._load_seen_symbols(market) if state["next_offset"] else set()
        offset = state["next_offset"]
        dataset_path = self._part_path(market, "dataset")
        os.makedirs(os.path.dirname(dataset_path), exist_ok=True)

        with open(dataset_path, "a", encoding="utf-8") as dataset_fh:
            while offset < self.spec.max_rows:
                page_end = min(offset + self.spec.page_size, self.spec.max_rows)
                payload = self._build_payload(offset, page_end)
                try:
                    response = self._fetch_page(market, payload)
                except PageFetchError as exc:
                    state["status"] = "failed"
                    state["last_error"] = str(exc)
                    logger.error("market %s failed at offset %d: %s", market, offset, exc)
                    break

                rows = response.get("data") or []
                total_count = response.get("totalCount")
                state["pages_fetched"] += 1

                if not rows:
                    break

                new_symbols = 0
                for item in rows:
                    state["received"] += 1
                    symbol = item.get("s") or ""
                    if symbol in seen:
                        state["skipped"] += 1
                        continue
                    seen.add(symbol)
                    new_symbols += 1
                    record = {"market": market, "symbol": symbol}
                    values = item.get("d") or []
                    for idx, column in enumerate(self.spec.columns):
                        record[column] = values[idx] if idx < len(values) else None
                    dataset_fh.write(json.dumps(record) + "\n")
                    state["written"] += 1
                dataset_fh.flush()

                offset = page_end
                state["next_offset"] = offset
                self._checkpoint["markets"][market] = state
                self._save_checkpoint()
                if new_symbols == 0:
                    break
                if total_count is not None and offset >= total_count:
                    break

        if state["status"] == "in_progress":
            state["status"] = "success"
        return state

    def _build_payload(self, start: int, end: int) -> Dict[str, Any]:
        payload = {
            "columns": list(self.spec.columns),
            "options": {"lang": "en"},
            "range": [start, end],
        }
        if self.spec.filters:
            payload["filter"] = self.spec.filters
        return payload

    def _fetch_page(self, market: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Fetch a single page with retries; raises PageFetchError on exhaustion."""
        url = Screener.SUPPORTED_MARKETS[market]
        cache_key = self._cache_key(market, payload)
        cached = self._cache_read(cache_key)
        if cached is not None:
            logger.debug("cache hit for %s %s", market, payload["range"])
            return cached
        last_error = "unknown error"
        for attempt in range(MAX_PAGE_ATTEMPTS):
            try:
                response = self.transport_fn(
                    url,
                    json=payload,
                    headers=self.headers,
                    timeout=REQUEST_TIMEOUT_SECONDS,
                )
            except (requests.ReadTimeout, requests.ConnectionError) as exc:
                last_error = f"request error: {exc}"
                logger.warning("page fetch attempt %d failed: %s", attempt + 1, exc)
                self._backoff(attempt)
                continue

            status = response.status_code
            if status == 200:
                try:
                    body = response.json()
                except ValueError as exc:
                    last_error = f"invalid JSON response: {exc}"
                    self._backoff(attempt)
                    continue
                self._cache_write(cache_key, body)
                return body
            if status == 429:
                last_error = "HTTP 429 (rate limited)"
                retry_after = response.headers.get("Retry-After")
                try:
                    wait = float(retry_after) if retry_after is not None else None
                except (TypeError, ValueError):
                    wait = None
                logger.warning("rate limited on %s", market)
                self._backoff(attempt, wait)
                continue
            if 500 <= status < 600:
                last_error = f"HTTP {status}"
                logger.warning("server error %d on %s, retrying", status, market)
                self._backoff(attempt)
                continue
            raise PageFetchError(f"HTTP {status}: {getattr(response, 'text', '')}")

        raise PageFetchError(f"exhausted {MAX_PAGE_ATTEMPTS} attempts: {last_error}")

    def _backoff(self, attempt: int, wait: Optional[float] = None) -> None:
        if wait is None:
            wait = min(2 ** attempt, 8)
        self.sleep_fn(wait)

    @staticmethod
    def _cache_key(market: str, payload: Dict[str, Any]) -> str:
        raw = market + "\n" + json.dumps(payload, sort_keys=True)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def _cache_path(self, key: str) -> str:
        return os.path.join(self.out_dir, "cache", f"{key}.json")

    def _cache_read(self, key: str) -> Optional[Dict[str, Any]]:
        if self.refresh:
            return None
        path = self._cache_path(key)
        try:
            with open(path, "r", encoding="utf-8") as fh:
                entry = json.load(fh)
        except (OSError, ValueError):
            return None
        fetched_at = entry.get("fetched_at", 0)
        if time.time() - fetched_at > self.cache_ttl_seconds:
            return None
        return entry.get("response")

    def _cache_write(self, key: str, body: Dict[str, Any]) -> None:
        path = self._cache_path(key)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        entry = {"fetched_at": time.time(), "response": body}
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(entry, fh)

    def _write_manifest(self, results: Dict[str, Dict[str, Any]]) -> None:
        manifest = {"markets": results}
        manifest_path = os.path.join(self.out_dir, "manifest.json")
        with open(manifest_path, "w", encoding="utf-8") as fh:
            json.dump(manifest, fh, indent=2, sort_keys=True)
            fh.write("\n")

    def _verify_counts(self, results: Dict[str, Dict[str, Any]]) -> bool:
        """Check that per-market counters and jsonl line counts agree."""
        ok = True
        for market, state in results.items():
            total = state["written"] + state["quarantined"] + state["skipped"]
            if total != state["received"]:
                logger.error(
                    "count mismatch for %s: received=%d but written+quarantined+skipped=%d",
                    market, state["received"], total,
                )
                ok = False
        expected_dataset = sum(s["written"] for s in results.values())
        expected_quarantine = sum(s["quarantined"] for s in results.values())
        if self._count_lines("dataset.jsonl") != expected_dataset:
            logger.error("dataset.jsonl line count mismatch")
            ok = False
        if self._count_lines("quarantine.jsonl") != expected_quarantine:
            logger.error("quarantine.jsonl line count mismatch")
            ok = False
        return ok

    def _count_lines(self, name: str) -> int:
        path = os.path.join(self.out_dir, name)
        if not os.path.exists(path):
            return 0
        with open(path, "r", encoding="utf-8") as fh:
            return sum(1 for _ in fh)

    def _checkpoint_path(self) -> str:
        return os.path.join(self.out_dir, "checkpoint.json")

    def _load_checkpoint(self) -> Dict[str, Any]:
        try:
            with open(self._checkpoint_path(), "r", encoding="utf-8") as fh:
                checkpoint = json.load(fh)
        except (OSError, ValueError):
            return {"spec_hash": self.spec.fingerprint(), "markets": {}}
        if checkpoint.get("spec_hash") != self.spec.fingerprint():
            logger.warning("spec changed since last run, starting over")
            self._reset_outputs()
            return {"spec_hash": self.spec.fingerprint(), "markets": {}}
        checkpoint.setdefault("markets", {})
        return checkpoint

    def _save_checkpoint(self) -> None:
        with open(self._checkpoint_path(), "w", encoding="utf-8") as fh:
            json.dump(self._checkpoint, fh, indent=2, sort_keys=True)
            fh.write("\n")

    def _reset_outputs(self) -> None:
        for name in ("dataset.jsonl", "quarantine.jsonl", "checkpoint.json"):
            try:
                os.remove(os.path.join(self.out_dir, name))
            except OSError:
                pass
        parts_dir = os.path.join(self.out_dir, ".parts")
        if os.path.isdir(parts_dir):
            for entry in os.listdir(parts_dir):
                try:
                    os.remove(os.path.join(parts_dir, entry))
                except OSError:
                    pass

    def _part_path(self, market: str, kind: str) -> str:
        return os.path.join(self.out_dir, ".parts", f"{market}.{kind}.jsonl")

    def _assemble_outputs(self) -> None:
        """Concatenate per-market part files into the final jsonl outputs."""
        for kind, target in (("dataset", "dataset.jsonl"),
                             ("quarantine", "quarantine.jsonl")):
            lines = []
            for market in self.spec.markets:
                part = self._part_path(market, kind)
                if not os.path.exists(part):
                    continue
                with open(part, "r", encoding="utf-8") as fh:
                    lines.extend(fh.readlines())
            target_path = os.path.join(self.out_dir, target)
            if not lines:
                if os.path.exists(target_path):
                    os.remove(target_path)
                continue
            tmp_path = target_path + ".tmp"
            with open(tmp_path, "w", encoding="utf-8") as fh:
                fh.writelines(lines)
            os.replace(tmp_path, target_path)

    def _load_seen_symbols(self, market: str) -> set:
        """Rebuild the dedup set for a market from rows already on disk."""
        seen = set()
        for kind in ("dataset", "quarantine"):
            path = self._part_path(market, kind)
            if not os.path.exists(path):
                continue
            with open(path, "r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        record = json.loads(line)
                    except ValueError:
                        continue
                    seen.add(record.get("symbol") or "")
        return seen


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m tradingview_scraper.batch",
        description="Batch-collect screener data for multiple markets.",
    )
    parser.add_argument("--spec", required=True, help="Path to the batch spec JSON file")
    parser.add_argument("--out", required=True, help="Output directory")
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Ignore cached responses and fetch everything again",
    )
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="Discard any previous checkpoint and start from scratch",
    )
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        spec = load_spec(args.spec)
    except SpecError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    collector = BatchCollector(
        spec=spec,
        out_dir=args.out,
        refresh=args.refresh,
        fresh=args.fresh,
    )
    return collector.run()


if __name__ == "__main__":
    sys.exit(main())
