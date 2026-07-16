"""Apply pending Stage 2.R schema migrations.

Usage:
    uv run python scripts/apply_migrations.py
    uv run python scripts/apply_migrations.py config/swingrl.yaml
"""

from __future__ import annotations

import sys

from swingrl.config.schema import load_config
from swingrl.data.db import DatabaseManager
from swingrl.data.migration_runner import apply_migrations, current_schema_version
from swingrl.utils.logging import configure_logging


def main() -> int:
    """Apply all pending migrations against the configured database.

    Returns:
        0 on success.
    """
    config = load_config(sys.argv[1] if len(sys.argv) > 1 else "config/swingrl.yaml")
    configure_logging(json_logs=config.logging.json_logs, log_level=config.logging.level)
    db = DatabaseManager(config)
    n = apply_migrations(db)
    print(f"applied={n} current_version={current_schema_version(db)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
