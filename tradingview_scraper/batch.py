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
import json
import logging
import os
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from tradingview_scraper.symbols import http_transport
from tradingview_scraper.symbols.screener import Screener
from tradingview_scraper.symbols.utils import generate_user_agent


logger = logging.getLogger(__name__)

DEFAULT_PAGE_SIZE = 100
DEFAULT_MAX_ROWS = 1000
DEFAULT_REQUESTS_PER_SECOND = 1.0
DEFAULT_CACHE_TTL_SECONDS = 30 * 60
REQUEST_TIMEOUT_SECONDS = 30


class SpecError(ValueError):
    """Raised when the batch spec file is invalid."""


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
    ):
        self.spec = spec
        self.out_dir = out_dir
        self.refresh = refresh
        self.fresh = fresh
        self.transport_fn = transport_fn or http_transport.post
        self.sleep_fn = sleep_fn
        self.headers = {"User-Agent": generate_user_agent()}

    def run(self) -> int:
        """Run the batch job and return the process exit code."""
        raise NotImplementedError


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
