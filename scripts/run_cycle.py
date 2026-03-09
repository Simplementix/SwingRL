"""CLI entry point for manual trading cycle execution.

Runs a full trading cycle for a given environment, optionally in dry-run mode.
Performs startup reconciliation before trading (equity only).

Usage:
    python scripts/run_cycle.py --env equity --dry-run
    python scripts/run_cycle.py --env crypto
    python scripts/run_cycle.py --env equity --config config/swingrl.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

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
        description="Run a full SwingRL trading cycle for equity or crypto.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  python scripts/run_cycle.py --env equity --dry-run
  python scripts/run_cycle.py --env crypto
  python scripts/run_cycle.py --env equity --config config/swingrl.yaml
""",
    )

    parser.add_argument(
        "--env",
        choices=["equity", "crypto"],
        required=True,
        help="Environment to run trading cycle for.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Run stages 1-3 only; log actions without submitting orders.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/swingrl.yaml",
        help="Path to config YAML (default: config/swingrl.yaml).",
    )
    parser.add_argument(
        "--models-dir",
        type=str,
        default="models",
        help="Root directory for trained models (default: models).",
    )

    return parser


def main(argv: list[str] | None = None) -> int:
    """Run trading cycle.

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
    dry_run: bool = args.dry_run
    models_dir = Path(args.models_dir)

    log.info(
        "run_cycle_started",
        env=env_name,
        dry_run=dry_run,
        config=args.config,
    )

    try:
        from swingrl.data.db import DatabaseManager
        from swingrl.execution.pipeline import ExecutionPipeline
        from swingrl.execution.reconciliation import PositionReconciler
        from swingrl.features.pipeline import FeaturePipeline
        from swingrl.monitoring.alerter import Alerter

        # Initialize components
        db = DatabaseManager(config)
        db.init_schema()

        import duckdb

        duckdb_path = Path(config.system.duckdb_path)
        duckdb_path.parent.mkdir(parents=True, exist_ok=True)
        conn = duckdb.connect(str(duckdb_path))

        feature_pipeline = FeaturePipeline(config, conn)

        alerter = Alerter(
            webhook_url="",  # No webhook in paper mode
            cooldown_minutes=config.alerting.alert_cooldown_minutes,
            db=db,
        )

        pipeline = ExecutionPipeline(
            config=config,
            db=db,
            feature_pipeline=feature_pipeline,
            alerter=alerter,
            models_dir=models_dir,
        )

        # Startup reconciliation (equity only)
        if env_name == "equity" and not dry_run:
            log.info("startup_reconciliation", env=env_name)
            try:
                from swingrl.execution.adapters.alpaca_adapter import AlpacaAdapter

                adapter = AlpacaAdapter(config=config, alerter=alerter)
                reconciler = PositionReconciler(
                    config=config, db=db, adapter=adapter, alerter=alerter
                )
                adjustments = reconciler.reconcile(env_name)
                if adjustments:
                    log.info("reconciliation_adjustments", count=len(adjustments))
            except Exception:
                log.warning("startup_reconciliation_failed", exc_info=True)

        # Execute trading cycle
        fills = pipeline.execute_cycle(env_name, dry_run=dry_run)

        # Print summary
        if fills:
            total_notional = sum(abs(f.quantity * f.fill_price) for f in fills)
            symbols = [f.symbol for f in fills]
            log.info(
                "cycle_summary",
                fills=len(fills),
                symbols=symbols,
                total_notional=round(total_notional, 2),
            )
            print(f"\nCycle complete: {len(fills)} fill(s)")
            print(f"  Symbols: {', '.join(symbols)}")
            print(f"  Total notional: ${total_notional:.2f}")
        else:
            msg = "dry-run" if dry_run else "no signals"
            print(f"\nCycle complete: 0 fills ({msg})")

        conn.close()

    except Exception:
        log.exception("run_cycle_failed")
        return 1

    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        log.info("run_cycle_interrupted")
        sys.exit(130)
