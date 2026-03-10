"""CLI for full four-tier emergency stop protocol.

Executes all 4 tiers automatically: halt + cancel, liquidate crypto,
time-aware equity liquidation, verify + Discord alert.

Usage:
    python scripts/emergency_stop.py --reason "manual intervention required"
    python scripts/emergency_stop.py --config config/swingrl.yaml --reason "drawdown limit hit"
    docker exec swingrl-bot python scripts/emergency_stop.py --reason "description"
"""

from __future__ import annotations

import argparse
import json
import sys

from swingrl.config.schema import load_config
from swingrl.data.db import DatabaseManager
from swingrl.execution.emergency import execute_emergency_stop
from swingrl.monitoring.alerter import Alerter


def main() -> None:
    """Parse arguments and execute the full four-tier emergency stop."""
    parser = argparse.ArgumentParser(
        description="Execute four-tier emergency stop: halt, cancel, liquidate, verify + alert."
    )
    parser.add_argument(
        "--config",
        default="config/swingrl.yaml",
        help="Path to SwingRL config YAML (default: config/swingrl.yaml)",
    )
    parser.add_argument(
        "--reason",
        required=True,
        help="Reason for the emergency stop (required)",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    db = DatabaseManager(config)
    alerter = Alerter(
        webhook_url=config.alerting.alerts_webhook_url,
        alerts_webhook_url=config.alerting.alerts_webhook_url,
        daily_webhook_url=config.alerting.daily_webhook_url,
        cooldown_minutes=0,
        db=db,
    )

    report = execute_emergency_stop(
        config=config,
        db=db,
        alerter=alerter,
        reason=args.reason,
    )

    print(json.dumps(report, indent=2, default=str))  # noqa: T201

    if report.get("all_success"):
        print("\nEMERGENCY STOP COMPLETE: All 4 tiers succeeded.")  # noqa: T201
        sys.exit(0)
    else:
        print("\nEMERGENCY STOP COMPLETE: One or more tiers had errors.")  # noqa: T201
        sys.exit(1)


if __name__ == "__main__":
    main()
