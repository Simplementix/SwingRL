"""Era-0 deployed-model bootstrap: best-CPS vintage selection (Plan A Task 5, amended).

Seeds the V002 identity spine (training_runs/models/ensemble_weight_history) with
the best-CPS era-0 vintage per (environment, algorithm): crypto iter 0 (uncoached
baseline beat every coached season) and equity iter 4. Real iteration_number/seed
are recoverable from ``models/iterations/{iter}/active/{env}/{algo}/`` — sentinels
(-1) are the fallback only, for a vintage genuinely unresolvable from disk.
"""

from __future__ import annotations

import hashlib
import logging
import os
import textwrap
from collections.abc import Generator
from pathlib import Path

import pytest

from swingrl.config.schema import load_config
from swingrl.data.db import DatabaseManager

pytestmark = pytest.mark.skipif(
    not os.environ.get("DATABASE_URL"),
    reason="DATABASE_URL not set — no PostgreSQL available for testing",
)


@pytest.fixture
def db_with_legacy_schema(tmp_path: Path) -> Generator[DatabaseManager, None, None]:
    """DatabaseManager on DATABASE_URL with init_schema() already run.

    Mirrors ``tests/data/test_migrations_content.py``'s fixture of the same name:
    V001/V002's plain ``CREATE TABLE`` / ``ALTER TABLE ADD COLUMN`` statements are
    not idempotent by themselves, so teardown drops the V002 artifacts (FK-safe
    order), the V001 artifacts, and clears the version-1/2 ledger rows — keeping
    the persistent scratch database re-runnable across test sessions.
    """
    db_url = os.environ.get(
        "DATABASE_URL",
        "postgresql://test:test@localhost:5432/swingrl_test",  # pragma: allowlist secret
    )
    config_yaml = textwrap.dedent(f"""\
        trading_mode: paper
        equity:
          symbols: [SPY, QQQ]
          max_position_size: 0.25
          max_drawdown_pct: 0.10
          daily_loss_limit_pct: 0.02
        crypto:
          symbols: [BTCUSDT, ETHUSDT]
          max_position_size: 0.50
          max_drawdown_pct: 0.12
          daily_loss_limit_pct: 0.03
          min_order_usd: 10.0
        capital:
          equity_usd: 400.0
          crypto_usd: 47.0
        paths:
          data_dir: data/
          db_dir: db/
          models_dir: models/
          logs_dir: logs/
        logging:
          level: INFO
          json_logs: false
        system:
          database_url: "{db_url}"
        alerting:
          alert_cooldown_minutes: 30
          consecutive_failures_before_alert: 3
    """)
    config_file = tmp_path / "swingrl.yaml"
    config_file.write_text(config_yaml)
    config = load_config(config_file)
    DatabaseManager.reset()
    mgr = DatabaseManager(config)
    mgr.init_schema()
    yield mgr
    with mgr.connection() as conn:
        conn.execute("DROP TABLE IF EXISTS ensemble_weight_history")
        conn.execute("DROP TABLE IF EXISTS models")
        conn.execute("DROP TABLE IF EXISTS training_runs")
        conn.execute(
            "ALTER TABLE backtest_results "
            "DROP COLUMN IF EXISTS era_id, DROP COLUMN IF EXISTS gate_version_id"
        )
        conn.execute(
            "ALTER TABLE iteration_results "
            "DROP COLUMN IF EXISTS era_id, DROP COLUMN IF EXISTS gate_version_ensemble_id"
        )
        conn.execute("DROP TABLE IF EXISTS eras")
        conn.execute("DROP TABLE IF EXISTS gate_versions")
        conn.execute(
            "DO $$ BEGIN "
            "IF to_regclass('public.schema_migrations') IS NOT NULL THEN "
            "DELETE FROM schema_migrations WHERE version IN (1, 2); "
            "END IF; "
            "END $$;"
        )
    DatabaseManager.reset()


def _write_vintage(models_root: Path, iteration: int, environment: str, algorithm: str) -> None:
    """Create a fake model.zip + vec_normalize.pkl under the loader-expected layout."""
    base = models_root / "iterations" / f"iter_{iteration}" / "active" / environment / algorithm
    base.mkdir(parents=True, exist_ok=True)
    (base / "model.zip").write_bytes(f"model-{environment}-{algorithm}-{iteration}".encode())
    (base / "vec_normalize.pkl").write_bytes(
        f"vecnorm-{environment}-{algorithm}-{iteration}".encode()
    )


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def test_bootstrap_era0_models(
    db_with_legacy_schema: DatabaseManager, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """REQ-T5: best-CPS vintage rows resolved; missing vintage falls back to sentinels; idempotent."""
    from scripts.migrations.bootstrap_era0_models import bootstrap_era0_models

    from swingrl.data.migration_runner import apply_migrations

    apply_migrations(db_with_legacy_schema)

    models_root = tmp_path / "models"
    # crypto iter 0: full set (all 3 algos present)
    _write_vintage(models_root, 0, "crypto", "ppo")
    _write_vintage(models_root, 0, "crypto", "a2c")
    _write_vintage(models_root, 0, "crypto", "sac")
    # equity iter 4: deliberately missing "sac" to exercise the fallback path
    _write_vintage(models_root, 4, "equity", "ppo")
    _write_vintage(models_root, 4, "equity", "a2c")

    fallback_model_id = "equity-vfallback-sac-2026-01-01"
    fallback_model_path = str(tmp_path / "legacy_equity_sac" / "model.zip")  # never created
    fallback_vecnorm_path = str(tmp_path / "legacy_equity_sac" / "vec_normalize.pkl")

    try:
        with db_with_legacy_schema.connection() as conn:
            conn.execute(
                "INSERT INTO model_metadata ("
                " model_id, environment, algorithm, version, training_start_date,"
                " training_end_date, total_timesteps, converged_at_step,"
                " validation_sharpe, ensemble_weight, model_path, vec_normalize_path"
                ") VALUES (%s, 'equity', 'sac', 'v0', '2026-01-01', '2026-01-01',"
                " 100000, 90000, 0.5, 0.4, %s, %s)",
                (fallback_model_id, fallback_model_path, fallback_vecnorm_path),
            )

        with caplog.at_level(logging.WARNING):
            counts = bootstrap_era0_models(db_with_legacy_schema, models_root)

        assert counts == {"vintage": 5, "fallback": 1, "unresolved": 0, "skipped_existing": 0}
        assert any(
            "fallback" in r.message.lower() or "missing" in r.message.lower()
            for r in caplog.records
        )

        with db_with_legacy_schema.connection() as conn:
            crypto_ppo = conn.execute(
                "SELECT tr.iteration_number, tr.seed, tr.era_id, tr.status, tr.fold_number,"
                " tr.run_type, tr.code_version, tr.data_fingerprint,"
                " m.model_id, m.artifact_path, m.vecnormalize_path, m.artifact_sha256,"
                " m.vecnormalize_sha256, m.status AS model_status"
                " FROM training_runs tr JOIN models m ON m.run_pk = tr.run_pk"
                " WHERE tr.environment = 'crypto' AND tr.algorithm = 'ppo'"
            ).fetchone()
        assert crypto_ppo is not None
        assert crypto_ppo["iteration_number"] == 0
        assert crypto_ppo["seed"] == 42
        assert crypto_ppo["era_id"] == 0
        assert crypto_ppo["status"] == "completed"
        assert crypto_ppo["fold_number"] == -1
        assert crypto_ppo["run_type"] == "final_train"
        assert crypto_ppo["code_version"] == "unknown_era0"
        assert crypto_ppo["data_fingerprint"] == "unknown_era0"
        assert crypto_ppo["model_status"] == "active"
        assert crypto_ppo["artifact_path"].endswith("iter_0/active/crypto/ppo/model.zip")
        assert crypto_ppo["artifact_sha256"] == _sha256_bytes(b"model-crypto-ppo-0")
        assert crypto_ppo["vecnormalize_sha256"] == _sha256_bytes(b"vecnorm-crypto-ppo-0")

        with db_with_legacy_schema.connection() as conn:
            crypto_a2c = conn.execute(
                "SELECT seed FROM training_runs WHERE environment='crypto' AND algorithm='a2c'"
            ).fetchone()
            crypto_sac = conn.execute(
                "SELECT seed FROM training_runs WHERE environment='crypto' AND algorithm='sac'"
            ).fetchone()
        assert crypto_a2c["seed"] == 43
        assert crypto_sac["seed"] == 44

        with db_with_legacy_schema.connection() as conn:
            equity_a2c = conn.execute(
                "SELECT iteration_number, seed FROM training_runs"
                " WHERE environment='equity' AND algorithm='a2c'"
            ).fetchone()
        assert equity_a2c["iteration_number"] == 4
        assert equity_a2c["seed"] == 43

        # Fallback row: sentinels, existing model_metadata identity reused, missing
        # artifact files -> sha256 columns NULL (files never existed on this host).
        with db_with_legacy_schema.connection() as conn:
            fallback = conn.execute(
                "SELECT tr.iteration_number, tr.seed, tr.era_id, tr.status,"
                " m.model_id, m.artifact_path, m.vecnormalize_path, m.artifact_sha256,"
                " m.vecnormalize_sha256"
                " FROM training_runs tr JOIN models m ON m.run_pk = tr.run_pk"
                " WHERE tr.environment = 'equity' AND tr.algorithm = 'sac'"
            ).fetchone()
        assert fallback is not None
        assert fallback["iteration_number"] == -1
        assert fallback["seed"] == -1
        assert fallback["era_id"] == 0
        assert fallback["model_id"] == fallback_model_id
        assert fallback["artifact_path"] == fallback_model_path
        assert fallback["vecnormalize_path"] == fallback_vecnorm_path
        assert fallback["artifact_sha256"] is None
        assert fallback["vecnormalize_sha256"] is None

        with db_with_legacy_schema.connection() as conn:
            weights = conn.execute(
                "SELECT m.model_id, ewh.weight_frac, ewh.set_by FROM ensemble_weight_history ewh"
                " JOIN models m ON m.model_id = ewh.model_id"
            ).fetchall()
        weight_by_model = {r["model_id"]: r for r in weights}
        assert len(weight_by_model) == 6
        assert weight_by_model[fallback_model_id]["weight_frac"] == pytest.approx(0.4)
        assert weight_by_model[fallback_model_id]["set_by"] == "training"
        vintage_model_id = crypto_ppo["model_id"]
        assert weight_by_model[vintage_model_id]["weight_frac"] == pytest.approx(1.0 / 3)
        assert weight_by_model[vintage_model_id]["set_by"] == "training"

        # --- Idempotent re-run: no new/duplicate rows anywhere ---
        counts_second = bootstrap_era0_models(db_with_legacy_schema, models_root)
        assert counts_second == {
            "vintage": 5,
            "fallback": 1,
            "unresolved": 0,
            "skipped_existing": 6,
        }

        with db_with_legacy_schema.connection() as conn:
            tr_count = conn.execute("SELECT count(*) AS n FROM training_runs").fetchone()["n"]
            model_count = conn.execute("SELECT count(*) AS n FROM models").fetchone()["n"]
            ewh_count = conn.execute(
                "SELECT count(*) AS n FROM ensemble_weight_history"
            ).fetchone()["n"]
        assert tr_count == 6
        assert model_count == 6
        assert ewh_count == 6
    finally:
        with db_with_legacy_schema.connection() as conn:
            conn.execute("DELETE FROM model_metadata WHERE model_id = %s", (fallback_model_id,))


def test_bootstrap_era0_models_no_vintage_no_fallback(
    db_with_legacy_schema: DatabaseManager, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """REQ-T5: an (env, algo) with neither a vintage dir nor a model_metadata row is skipped."""
    from scripts.migrations.bootstrap_era0_models import bootstrap_era0_models

    from swingrl.data.migration_runner import apply_migrations

    apply_migrations(db_with_legacy_schema)

    models_root = tmp_path / "models"
    with caplog.at_level(logging.WARNING):
        counts = bootstrap_era0_models(db_with_legacy_schema, models_root)

    assert counts["unresolved"] == 6  # nothing on disk, nothing in model_metadata
    assert counts["vintage"] == 0
    assert counts["fallback"] == 0

    with db_with_legacy_schema.connection() as conn:
        n = conn.execute("SELECT count(*) AS n FROM training_runs").fetchone()["n"]
    assert n == 0
