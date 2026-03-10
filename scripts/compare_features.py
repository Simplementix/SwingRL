"""CLI entry point for feature set A/B comparison.

Compares baseline vs candidate feature sets using validation Sharpe improvement.
Exits 0 when candidate is accepted, 1 when rejected or on error.

Usage:
    python scripts/compare_features.py --baseline baseline.json --candidate candidate.json
    python scripts/compare_features.py --baseline b.json --candidate c.json --threshold 0.10
    python scripts/compare_features.py --baseline b.json --candidate c.json --format json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import structlog

from swingrl.config.schema import load_config
from swingrl.features.pipeline import compare_features
from swingrl.utils.logging import configure_logging

log = structlog.get_logger(__name__)


def build_parser() -> argparse.ArgumentParser:
    """Build CLI argument parser.

    Returns:
        Configured ArgumentParser.
    """
    parser = argparse.ArgumentParser(
        description="Compare baseline vs candidate feature sets for A/B testing.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  python scripts/compare_features.py --baseline baseline.json --candidate candidate.json
  python scripts/compare_features.py --baseline b.json --candidate c.json --threshold 0.10
  python scripts/compare_features.py --baseline b.json --candidate c.json --format json
""",
    )

    parser.add_argument(
        "--baseline",
        required=True,
        help="Path to JSON file with baseline metrics (requires train_sharpe, validation_sharpe).",
    )
    parser.add_argument(
        "--candidate",
        required=True,
        help="Path to JSON file with candidate metrics (requires train_sharpe, validation_sharpe).",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.05,
        help="Minimum validation Sharpe improvement to accept (default: 0.05).",
    )
    parser.add_argument(
        "--format",
        choices=["text", "json"],
        default="text",
        help="Output format: text (human-readable) or json (machine-readable, default: text).",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/swingrl.yaml",
        help="Path to config YAML (default: config/swingrl.yaml).",
    )

    return parser


def _load_metrics(path: str) -> dict[str, float] | None:
    """Load and validate metrics from a JSON file.

    Args:
        path: File path to JSON metrics file.

    Returns:
        Dict with train_sharpe and validation_sharpe, or None on failure.
    """
    try:
        text = Path(path).read_text()
        data: dict[str, float] = json.loads(text)
    except FileNotFoundError:
        print(f"Error: file not found: {path}")  # noqa: T201
        return None
    except json.JSONDecodeError as exc:
        print(f"Error: invalid JSON in {path}: {exc}")  # noqa: T201
        return None

    required_keys = {"train_sharpe", "validation_sharpe"}
    missing = required_keys - data.keys()
    if missing:
        missing_str = ", ".join(sorted(missing))
        print(f"Error: missing required keys in {path}: {missing_str}")  # noqa: T201
        return None

    return data


def main(argv: list[str] | None = None) -> int:
    """Run feature comparison.

    Args:
        argv: Command-line arguments (default: sys.argv[1:]).

    Returns:
        Exit code: 0 = accepted, 1 = rejected or error.
    """
    parser = build_parser()
    args = parser.parse_args(argv)

    config = load_config(args.config)
    configure_logging(json_logs=config.logging.json_logs, log_level=config.logging.level)

    # Load metrics from JSON files
    baseline = _load_metrics(args.baseline)
    if baseline is None:
        return 1

    candidate = _load_metrics(args.candidate)
    if candidate is None:
        return 1

    # Run comparison
    result = compare_features(baseline, candidate, threshold=args.threshold)

    # Output in requested format
    if args.format == "json":
        print(json.dumps(result))  # noqa: T201
    else:
        verdict = "ACCEPTED" if result["accepted"] else "REJECTED"
        sharpe_improvement = result["sharpe_improvement"]
        reason = result["reason"]
        print(f"Verdict: {verdict}")  # noqa: T201
        print(f"Sharpe improvement: {sharpe_improvement:.4f}")  # noqa: T201
        print(f"Reason: {reason}")  # noqa: T201

    log.info(
        "feature_comparison_complete",
        accepted=result["accepted"],
        sharpe_improvement=result["sharpe_improvement"],
    )

    return 0 if result["accepted"] else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        log.info("comparison_interrupted")
        sys.exit(130)
