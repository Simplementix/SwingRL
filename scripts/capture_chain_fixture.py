# scripts/capture_chain_fixture.py
"""One-shot: pull one real CBOE chain; save a bounded fixture OR probe the delay (D8).

Usage:
    uv run python scripts/capture_chain_fixture.py --symbol _SPX --out tests/fixtures/cboe_chain_spx.json
    uv run python scripts/capture_chain_fixture.py --symbol _SPX --probe
Public, unauthenticated data — fixtures are bounded (size), not sanitized (no account fields).
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import structlog

from swingrl.config.schema import load_config
from swingrl.data.options.cboe_client import CboeChainClient
from swingrl.utils.logging import configure_logging

log = structlog.get_logger(__name__)
_MAX_OPTIONS = 40


def bound_chain(raw: dict, max_options: int = _MAX_OPTIONS) -> dict:
    """Keep the full header but only an evenly-spaced slice of contracts."""
    out = {k: v for k, v in raw.items() if k != "data"}
    data = dict(raw["data"])
    opts = data.get("options", [])
    step = max(1, len(opts) // max_options)
    data["options"] = opts[::step][:max_options]
    out["data"] = data
    return out


def main() -> int:
    """Capture a fixture or print the delay-probe readings."""
    parser = argparse.ArgumentParser(description="Capture a CBOE chain fixture / probe delay")
    parser.add_argument("--symbol", required=True)
    parser.add_argument("--out")
    parser.add_argument("--probe", action="store_true")
    args = parser.parse_args()
    config = load_config("config/swingrl.yaml")
    configure_logging(json_logs=config.logging.json_logs, log_level=config.logging.level)
    client = CboeChainClient(config.options_collector)
    raw = client.get_option_chain(args.symbol)
    if args.probe:
        print(f"wall_clock_utc={datetime.now(UTC).isoformat()}")
        print(f"payload_timestamp={raw.get('timestamp')}")
        print(f"header_last_trade_time={raw['data'].get('last_trade_time')}")
        print(f"contracts={len(raw['data'].get('options', []))}")
        return 0
    if not args.out:
        print("ERROR: --out required unless --probe", file=sys.stderr)
        return 2
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(bound_chain(raw), indent=1))
    log.info("fixture_written", symbol=args.symbol, path=str(out_path), rows=_MAX_OPTIONS)
    return 0


if __name__ == "__main__":
    sys.exit(main())
