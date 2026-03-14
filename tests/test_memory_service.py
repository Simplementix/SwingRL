"""Tests for DuckDB training telemetry schema migrations.

TRAIN-06, TRAIN-07: Training telemetry tables (training_epochs, meta_decisions,
reward_adjustments) and idempotent ALTER TABLE for training_runs.
"""

from __future__ import annotations

import pytest


class TestTrainingTelemetryDDL:
    """Tests for apply_training_telemetry_ddl() DuckDB schema migrations."""

    def test_training_epochs_table_created(self, tmp_path: pytest.TempPathFactory) -> None:
        """TRAIN-06: apply_training_telemetry_ddl creates training_epochs table in DuckDB."""
        import duckdb

        from swingrl.features.schema import apply_training_telemetry_ddl

        db_path = str(tmp_path / "test.ddb")  # type: ignore[arg-type]
        conn = duckdb.connect(db_path)
        try:
            apply_training_telemetry_ddl(conn)
            result = conn.execute(
                "SELECT COUNT(*) FROM information_schema.tables WHERE table_name = 'training_epochs'"
            ).fetchone()
            assert result[0] == 1
        finally:
            conn.close()

    def test_training_epochs_has_all_columns(self, tmp_path: pytest.TempPathFactory) -> None:
        """TRAIN-06: training_epochs table has all required columns."""
        import duckdb

        from swingrl.features.schema import apply_training_telemetry_ddl

        db_path = str(tmp_path / "test.ddb")  # type: ignore[arg-type]
        conn = duckdb.connect(db_path)
        try:
            apply_training_telemetry_ddl(conn)
            cols = conn.execute(
                "SELECT column_name FROM information_schema.columns WHERE table_name = 'training_epochs'"
            ).fetchall()
            col_names = {row[0] for row in cols}
            expected = {
                "id",
                "run_id",
                "epoch",
                "algo",
                "env",
                "sharpe",
                "mdd",
                "reward_weight_profit",
                "reward_weight_sharpe",
                "reward_weight_drawdown",
                "reward_weight_turnover",
                "stop_training",
                "rationale",
                "created_at",
            }
            assert expected.issubset(col_names), f"Missing columns: {expected - col_names}"
        finally:
            conn.close()

    def test_meta_decisions_table_created(self, tmp_path: pytest.TempPathFactory) -> None:
        """TRAIN-06: apply_training_telemetry_ddl creates meta_decisions table."""
        import duckdb

        from swingrl.features.schema import apply_training_telemetry_ddl

        db_path = str(tmp_path / "test.ddb")  # type: ignore[arg-type]
        conn = duckdb.connect(db_path)
        try:
            apply_training_telemetry_ddl(conn)
            result = conn.execute(
                "SELECT COUNT(*) FROM information_schema.tables WHERE table_name = 'meta_decisions'"
            ).fetchone()
            assert result[0] == 1
        finally:
            conn.close()

    def test_meta_decisions_has_all_columns(self, tmp_path: pytest.TempPathFactory) -> None:
        """TRAIN-06: meta_decisions table has all required columns."""
        import duckdb

        from swingrl.features.schema import apply_training_telemetry_ddl

        db_path = str(tmp_path / "test.ddb")  # type: ignore[arg-type]
        conn = duckdb.connect(db_path)
        try:
            apply_training_telemetry_ddl(conn)
            cols = conn.execute(
                "SELECT column_name FROM information_schema.columns WHERE table_name = 'meta_decisions'"
            ).fetchall()
            col_names = {row[0] for row in cols}
            expected = {
                "id",
                "run_id",
                "algo",
                "env",
                "decision_type",
                "decision_json",
                "rationale",
                "created_at",
            }
            assert expected.issubset(col_names), f"Missing columns: {expected - col_names}"
        finally:
            conn.close()

    def test_reward_adjustments_table_created(self, tmp_path: pytest.TempPathFactory) -> None:
        """TRAIN-06: apply_training_telemetry_ddl creates reward_adjustments table."""
        import duckdb

        from swingrl.features.schema import apply_training_telemetry_ddl

        db_path = str(tmp_path / "test.ddb")  # type: ignore[arg-type]
        conn = duckdb.connect(db_path)
        try:
            apply_training_telemetry_ddl(conn)
            result = conn.execute(
                "SELECT COUNT(*) FROM information_schema.tables "
                "WHERE table_name = 'reward_adjustments'"
            ).fetchone()
            assert result[0] == 1
        finally:
            conn.close()

    def test_reward_adjustments_has_all_columns(self, tmp_path: pytest.TempPathFactory) -> None:
        """TRAIN-06: reward_adjustments table has all required columns."""
        import duckdb

        from swingrl.features.schema import apply_training_telemetry_ddl

        db_path = str(tmp_path / "test.ddb")  # type: ignore[arg-type]
        conn = duckdb.connect(db_path)
        try:
            apply_training_telemetry_ddl(conn)
            cols = conn.execute(
                "SELECT column_name FROM information_schema.columns "
                "WHERE table_name = 'reward_adjustments'"
            ).fetchall()
            col_names = {row[0] for row in cols}
            expected = {
                "id",
                "run_id",
                "epoch_trigger",
                "epoch_outcome",
                "algo",
                "env",
                "weight_before",
                "weight_after",
                "outcome_sharpe",
                "created_at",
            }
            assert expected.issubset(col_names), f"Missing columns: {expected - col_names}"
        finally:
            conn.close()

    def test_ddl_is_idempotent(self, tmp_path: pytest.TempPathFactory) -> None:
        """TRAIN-06: apply_training_telemetry_ddl is idempotent (run twice, no error)."""
        import duckdb

        from swingrl.features.schema import apply_training_telemetry_ddl

        db_path = str(tmp_path / "test.ddb")  # type: ignore[arg-type]
        conn = duckdb.connect(db_path)
        try:
            apply_training_telemetry_ddl(conn)
            # Second call must not raise
            apply_training_telemetry_ddl(conn)
            # Tables still exist
            result = conn.execute(
                "SELECT COUNT(*) FROM information_schema.tables "
                "WHERE table_name IN ('training_epochs', 'meta_decisions', 'reward_adjustments')"
            ).fetchone()
            assert result[0] == 3
        finally:
            conn.close()

    def test_alter_training_runs_adds_run_type_column(
        self, tmp_path: pytest.TempPathFactory
    ) -> None:
        """TRAIN-07: ALTER TABLE migration adds run_type column to training_runs."""
        import duckdb

        from swingrl.features.schema import apply_training_telemetry_ddl

        db_path = str(tmp_path / "test.ddb")  # type: ignore[arg-type]
        conn = duckdb.connect(db_path)
        try:
            # Create a minimal training_runs table first (as if from previous phase)
            conn.execute(
                "CREATE TABLE IF NOT EXISTS training_runs ("
                "  id INTEGER PRIMARY KEY,"
                "  run_id VARCHAR NOT NULL,"
                "  algo VARCHAR NOT NULL,"
                "  env VARCHAR NOT NULL,"
                "  created_at TIMESTAMP DEFAULT NOW()"
                ")"
            )
            apply_training_telemetry_ddl(conn)
            cols = conn.execute(
                "SELECT column_name FROM information_schema.columns "
                "WHERE table_name = 'training_runs'"
            ).fetchall()
            col_names = {row[0] for row in cols}
            assert "run_type" in col_names, "run_type column not added by ALTER TABLE"
        finally:
            conn.close()

    def test_alter_training_runs_adds_meta_rationale_column(
        self, tmp_path: pytest.TempPathFactory
    ) -> None:
        """TRAIN-07: ALTER TABLE migration adds meta_rationale column to training_runs."""
        import duckdb

        from swingrl.features.schema import apply_training_telemetry_ddl

        db_path = str(tmp_path / "test.ddb")  # type: ignore[arg-type]
        conn = duckdb.connect(db_path)
        try:
            conn.execute(
                "CREATE TABLE IF NOT EXISTS training_runs ("
                "  id INTEGER PRIMARY KEY,"
                "  run_id VARCHAR NOT NULL"
                ")"
            )
            apply_training_telemetry_ddl(conn)
            cols = conn.execute(
                "SELECT column_name FROM information_schema.columns "
                "WHERE table_name = 'training_runs'"
            ).fetchall()
            col_names = {row[0] for row in cols}
            assert "meta_rationale" in col_names, "meta_rationale column not added by ALTER TABLE"
        finally:
            conn.close()

    def test_alter_adds_dominant_regime_column(self, tmp_path: pytest.TempPathFactory) -> None:
        """TRAIN-07: ALTER TABLE migration adds dominant_regime column to training_runs."""
        import duckdb

        from swingrl.features.schema import apply_training_telemetry_ddl

        db_path = str(tmp_path / "test.ddb")  # type: ignore[arg-type]
        conn = duckdb.connect(db_path)
        try:
            conn.execute("CREATE TABLE IF NOT EXISTS training_runs (id INTEGER PRIMARY KEY)")
            apply_training_telemetry_ddl(conn)
            cols = conn.execute(
                "SELECT column_name FROM information_schema.columns "
                "WHERE table_name = 'training_runs'"
            ).fetchall()
            col_names = {row[0] for row in cols}
            assert "dominant_regime" in col_names, "dominant_regime column not added by ALTER TABLE"
        finally:
            conn.close()

    def test_alter_idempotent_when_columns_exist(self, tmp_path: pytest.TempPathFactory) -> None:
        """TRAIN-07: ALTER TABLE is idempotent if columns already exist (no error on re-run)."""
        import duckdb

        from swingrl.features.schema import apply_training_telemetry_ddl

        db_path = str(tmp_path / "test.ddb")  # type: ignore[arg-type]
        conn = duckdb.connect(db_path)
        try:
            # Create training_runs with the new columns already present
            conn.execute(
                "CREATE TABLE IF NOT EXISTS training_runs ("
                "  id INTEGER PRIMARY KEY,"
                "  run_type VARCHAR DEFAULT 'completed',"
                "  meta_rationale TEXT,"
                "  dominant_regime VARCHAR"
                ")"
            )
            # Running twice must not raise
            apply_training_telemetry_ddl(conn)
            apply_training_telemetry_ddl(conn)
        finally:
            conn.close()

    def test_telemetry_tables_insertable(self, tmp_path: pytest.TempPathFactory) -> None:
        """TRAIN-06: Can insert rows into training_epochs, meta_decisions, reward_adjustments."""
        import duckdb

        from swingrl.features.schema import apply_training_telemetry_ddl

        db_path = str(tmp_path / "test.ddb")  # type: ignore[arg-type]
        conn = duckdb.connect(db_path)
        try:
            apply_training_telemetry_ddl(conn)
            # Insert into training_epochs
            conn.execute(
                "INSERT INTO training_epochs (run_id, epoch, algo, env, sharpe, mdd) "
                "VALUES ('run-001', 1, 'ppo', 'equity', 1.5, -0.05)"
            )
            # Insert into meta_decisions
            conn.execute(
                "INSERT INTO meta_decisions (run_id, algo, env, decision_type, decision_json) "
                "VALUES ('run-001', 'ppo', 'equity', 'run_config', '{\"lr\": 3e-4}')"
            )
            # Insert into reward_adjustments
            conn.execute(
                "INSERT INTO reward_adjustments (run_id, algo, env) "
                "VALUES ('run-001', 'ppo', 'equity')"
            )
            count = conn.execute("SELECT COUNT(*) FROM training_epochs").fetchone()[0]
            assert count == 1
            count = conn.execute("SELECT COUNT(*) FROM meta_decisions").fetchone()[0]
            assert count == 1
            count = conn.execute("SELECT COUNT(*) FROM reward_adjustments").fetchone()[0]
            assert count == 1
        finally:
            conn.close()
