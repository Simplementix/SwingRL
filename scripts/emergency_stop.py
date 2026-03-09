"""CLI to set the emergency halt flag and stop all trading cycles.

Usage:
    python scripts/emergency_stop.py --reason "manual intervention required"
    python scripts/emergency_stop.py --config config/swingrl.yaml --reason "drawdown limit hit"
"""

from __future__ import annotations

import argparse
import sys

from swingrl.config.schema import load_config
from swingrl.data.db import DatabaseManager
from swingrl.scheduler.halt_check import set_halt


def main() -> None:
    """Parse arguments and set the emergency halt flag."""
    parser = argparse.ArgumentParser(
        description="Set the emergency halt flag to stop all trading cycles."
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
    set_halt(db, reason=args.reason, set_by="emergency_stop_cli")

    print(f"HALT FLAG SET: {args.reason}")  # noqa: T201
    print("All trading cycles will skip until reset.")  # noqa: T201
    sys.exit(0)


if __name__ == "__main__":
    main()
