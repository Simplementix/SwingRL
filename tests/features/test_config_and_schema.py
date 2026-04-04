"""Tests for FeaturesConfig schema and DuckDB DDL.

FEAT-10, FEAT-11: Config schema additions and feature table DDL.
"""

from __future__ import annotations

from typing import Any

import pytest

from swingrl.config.schema import SwingRLConfig


class TestFeaturesConfig:
    """Tests for FeaturesConfig section in SwingRLConfig."""

    def test_features_config_default_zscore_windows(self, feature_config: SwingRLConfig) -> None:
        """FEAT-06: Default normalization windows load correctly."""
        assert feature_config.features.equity_zscore_window == 252
        assert feature_config.features.crypto_zscore_window == 360

    def test_features_config_hmm_params(self, feature_config: SwingRLConfig) -> None:
        """FEAT-05: HMM parameters load with correct defaults."""
        assert feature_config.features.equity_hmm_window == 1260
        assert feature_config.features.crypto_hmm_window == 2000
        assert feature_config.features.hmm_n_iter == 200
        assert feature_config.features.hmm_n_inits == 5

    def test_features_config_correlation_threshold(self, feature_config: SwingRLConfig) -> None:
        """FEAT-09: Correlation threshold loads correctly."""
        assert feature_config.features.correlation_threshold == 0.85

    def test_features_config_ridge_and_epsilon(self, feature_config: SwingRLConfig) -> None:
        """FEAT-05/06: Ridge and epsilon defaults load correctly."""
        assert feature_config.features.hmm_ridge == pytest.approx(1e-6)
        assert feature_config.features.zscore_epsilon == pytest.approx(1e-8)


class TestFeatureDDL:
    """Tests for DuckDB feature table DDL."""

    def test_init_feature_schema_creates_tables(self, pg_conn: Any) -> None:
        """FEAT-11: init_feature_schema creates all 4 required tables."""
        from swingrl.data.postgres_schema import init_postgres_schema

        init_postgres_schema(pg_conn)
        pg_conn.commit()

        tables = pg_conn.execute(
            "SELECT tablename FROM pg_tables WHERE schemaname = 'public'"
        ).fetchall()
        table_names = [row[0] for row in tables]

        assert "features_equity" in table_names
        assert "features_crypto" in table_names
        assert "fundamentals" in table_names
        assert "hmm_state_history" in table_names

    def test_features_equity_accepts_insert(self, pg_conn: Any) -> None:
        """FEAT-11: features_equity table accepts valid data insertion."""
        from swingrl.data.postgres_schema import init_postgres_schema

        init_postgres_schema(pg_conn)

        pg_conn.execute("""
            INSERT INTO features_equity VALUES (
                'SPY', '2024-01-02',
                1.02, 0.98, 55.0, 0.005, 0.002,
                0.65, 0.015, 1.2, 25.0,
                1.0, 58.0,
                -0.5, 0.10, 0.45, 0.018
            ) ON CONFLICT DO NOTHING
        """)
        pg_conn.commit()

        result = pg_conn.execute("SELECT * FROM features_equity WHERE symbol = 'SPY'").fetchone()
        assert result is not None
        assert result[0] == "SPY"

    def test_features_crypto_accepts_insert(self, pg_conn: Any) -> None:
        """FEAT-11: features_crypto table accepts valid data insertion."""
        from swingrl.data.postgres_schema import init_postgres_schema

        init_postgres_schema(pg_conn)

        pg_conn.execute("""
            INSERT INTO features_crypto VALUES (
                'BTCUSDT', '2024-01-01 04:00:00',
                1.01, 0.99, 60.0, 0.003, 0.001,
                0.55, 0.02, 0.9, 22.0,
                1.0, 55.0, 62.0, 1.05
            ) ON CONFLICT DO NOTHING
        """)
        pg_conn.commit()

        result = pg_conn.execute(
            "SELECT * FROM features_crypto WHERE symbol = 'BTCUSDT'"
        ).fetchone()
        assert result is not None
        assert result[0] == "BTCUSDT"

    def test_init_feature_schema_is_idempotent(self, pg_conn: Any) -> None:
        """FEAT-11: Calling init_feature_schema twice does not error."""
        from swingrl.data.postgres_schema import init_postgres_schema

        init_postgres_schema(pg_conn)
        init_postgres_schema(pg_conn)  # Should not raise
        pg_conn.commit()

        tables = pg_conn.execute(
            "SELECT tablename FROM pg_tables WHERE schemaname = 'public'"
        ).fetchall()
        assert len(tables) >= 4
