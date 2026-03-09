"""CLI to clear the emergency halt flag and resume trading cycles.

Usage:
    python scripts/reset_halt.py
    python scripts/reset_halt.py --config config/swingrl.yaml
"""

from __future__ import annotations

import argparse
import sys

from swingrl.config.schema import load_config
from swingrl.data.db import DatabaseManager
from swingrl.scheduler.halt_check import clear_halt


def main() -> None:
    """Parse arguments and clear the emergency halt flag."""
    parser = argparse.ArgumentParser(
        description="Clear the emergency halt flag to resume trading cycles."
    )
    parser.add_argument(
        "--config",
        default="config/swingrl.yaml",
        help="Path to SwingRL config YAML (default: config/swingrl.yaml)",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    db = DatabaseManager(config)
    clear_halt(db)

    print("HALT FLAG CLEARED: Trading cycles will resume on next schedule.")  # noqa: T201
    sys.exit(0)


if __name__ == "__main__":
    main()
