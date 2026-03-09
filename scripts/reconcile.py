"""CLI entry point for position reconciliation.

Compares DB positions against broker state and auto-corrects mismatches.
Broker is treated as source of truth.

Usage:
    python scripts/reconcile.py --env equity
    python scripts/reconcile.py --env crypto
    python scripts/reconcile.py --env equity --config config/swingrl.yaml
"""

from __future__ import annotations

import argparse
import sys

import structlog

from swingrl.config.schema import load_config
from swingrl.utils.logging import configure_logging

log = structlog.get_logger(__name__)


def build_parser() -> argparse.ArgumentParser:
    """Build CLI argument parser.

    Returns:
        Configured ArgumentParser.
    """
    parser = argparse.ArgumentParser(
        description="Reconcile SwingRL positions against broker state.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  python scripts/reconcile.py --env equity
  python scripts/reconcile.py --env crypto
  python scripts/reconcile.py --env equity --config config/swingrl.yaml
""",
    )

    parser.add_argument(
        "--env",
        choices=["equity", "crypto"],
        required=True,
        help="Environment to reconcile positions for.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/swingrl.yaml",
        help="Path to config YAML (default: config/swingrl.yaml).",
    )

    return parser


def main(argv: list[str] | None = None) -> int:
    """Run position reconciliation.

    Args:
        argv: Command-line arguments (default: sys.argv[1:]).

    Returns:
        Exit code (0 = success, 1 = error).
    """
    parser = build_parser()
    args = parser.parse_args(argv)

    config = load_config(args.config)
    configure_logging(json_logs=config.logging.json_logs, log_level=config.logging.level)

    env_name: str = args.env

    log.info("reconciliation_started", env=env_name)

    try:
        from swingrl.data.db import DatabaseManager
        from swingrl.execution.reconciliation import PositionReconciler
        from swingrl.monitoring.alerter import Alerter

        db = DatabaseManager(config)
        db.init_schema()

        alerter = Alerter(
            webhook_url="",
            cooldown_minutes=config.alerting.alert_cooldown_minutes,
            db=db,
        )

        if env_name == "equity":
            from swingrl.execution.adapters.alpaca_adapter import AlpacaAdapter

            adapter = AlpacaAdapter(config=config, alerter=alerter)
        else:
            from swingrl.execution.adapters.binance_sim import BinanceSimAdapter

            adapter = BinanceSimAdapter(config=config, db=db, alerter=alerter)  # type: ignore[assignment]

        reconciler = PositionReconciler(
            config=config,
            db=db,
            adapter=adapter,
            alerter=alerter,  # type: ignore[arg-type]
        )
        adjustments = reconciler.reconcile(env_name)

        if adjustments:
            print(f"\nReconciliation complete: {len(adjustments)} adjustment(s)")
            for adj in adjustments:
                print(f"  {adj['symbol']}: {adj['action']}")
        else:
            print("\nNo mismatches found.")

    except Exception:
        log.exception("reconciliation_failed")
        return 1

    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        log.info("reconciliation_interrupted")
        sys.exit(130)
