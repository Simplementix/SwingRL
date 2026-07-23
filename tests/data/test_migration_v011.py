"""RULING-2/3: V011 adds decision_price + disposition to pending_orders."""

from __future__ import annotations

import os

import pytest

from swingrl.config.schema import load_config
from swingrl.data.db import DatabaseManager
from swingrl.data.migration_runner import (
    EXPECTED_SCHEMA_VERSION,
    apply_migrations,
    current_schema_version,
)


@pytest.fixture
def db(valid_config_yaml: str, tmp_path):  # type: ignore[no-untyped-def]
    db_url = os.environ.get("DATABASE_URL", "")
    if not db_url:
        pytest.skip("DATABASE_URL not set — no PostgreSQL available for testing")
    config_file = tmp_path / "swingrl.yaml"
    config_file.write_text(valid_config_yaml)
    config = load_config(config_file)
    config.system.database_url = db_url
    DatabaseManager.reset()
    database = DatabaseManager(config)
    database.init_schema()
    yield database
    DatabaseManager.reset()


def test_v011_adds_lifecycle_columns(db: DatabaseManager) -> None:
    """RULING-2/3: after apply_migrations, pending_orders has decision_price + disposition."""
    apply_migrations(db)
    assert current_schema_version(db) >= 11
    assert EXPECTED_SCHEMA_VERSION == 11
    with db.connection() as conn:
        cols = {
            r["column_name"]
            for r in conn.execute(
                "SELECT column_name FROM information_schema.columns "
                "WHERE table_name = 'pending_orders'"
            ).fetchall()
        }
    assert "decision_price" in cols
    assert "disposition" in cols
