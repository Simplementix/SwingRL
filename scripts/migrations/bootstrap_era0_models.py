"""One-time script: seed the V002 identity spine with era-0 deployed-model identity.

Plan A Task 5 (AMENDED 2026-07-12). Verified facts that shaped this script:
(a) ``models/iterations/iter_{0..5}/active/{env}/{algo}/model.zip`` +
``vec_normalize.pkl`` exist for every era-0 iteration in the loader-expected
layout — model vintages are recoverable by directory, so real
``iteration_number``/``seed`` can be stamped instead of ``-1`` sentinels;
(b) live ``iteration_results.cps_v1_multiplicative`` picks crypto iter 0
(0.1531 — the uncoached baseline beat every coached season) and equity iter 4
(0.0153) as the best era-0 vintages; (c) era-0 seeds are per-algo constants
(``SEED_MAP`` = 42/43/44, ``trainer.py:71``).

For each (environment, algorithm) pair:
  - If ``models/iterations/iter_{BEST_ERA0_VINTAGE[env]}/active/{env}/{algo}/``
    has both artifact files, insert a ``training_runs`` row with the REAL
    iteration_number/seed (era 0, ``run_type='final_train'``,
    ``fold_number=-1``, ``code_version='unknown_era0'``,
    ``data_fingerprint='unknown_era0'``, ``status='completed'``) + a
    ``models`` row (``status='active'``, sha256 of both artifacts) + an
    initial ``ensemble_weight_history`` row (``set_by='training'``).
  - Otherwise fall back to the newest ``model_metadata`` row for that pair,
    with ``-1`` sentinels for iteration_number/seed and a warning log
    (P-A1: sentinel fallback preserved for a vintage genuinely unresolvable).

Idempotent: re-running is a no-op (``ON CONFLICT DO NOTHING`` on
``models.model_id``; the ``ensemble_weight_history`` row is only inserted
when the ``models`` row is newly created).

Deployment note: copying the selected vintage into ``models/active/{env}/{algo}/``
so the live loader serves it is a SEPARATE runbook step (Task 17) — this
script only writes spine rows, it never touches the filesystem models tree.

Usage:
    uv run python scripts/migrations/bootstrap_era0_models.py
    uv run python scripts/migrations/bootstrap_era0_models.py --config config/swingrl.yaml
    uv run python scripts/migrations/bootstrap_era0_models.py --models-root /app/models
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import structlog

from swingrl.config.schema import load_config
from swingrl.data.db import DatabaseManager
from swingrl.data.migration_runner import current_schema_version
from swingrl.utils.exceptions import ConfigError, DataError
from swingrl.utils.logging import configure_logging

if TYPE_CHECKING:
    import psycopg

log = structlog.get_logger(__name__)

# CPS evidence (live pg16 iteration_results, read 2026-07-12):
# crypto iter 0 = 0.1531 (baseline HPs — best era-0 crypto season);
# equity iter 4 = 0.0153. Frozen era-0 evidence; values never change.
BEST_ERA0_VINTAGE: dict[str, int] = {"crypto": 0, "equity": 4}
ERA0_SEED_MAP: dict[str, int] = {"ppo": 42, "a2c": 43, "sac": 44}  # trainer.py:71
SENTINEL = -1  # fallback only: vintage genuinely unresolvable (P-A1)

# This script writes training_runs/models/ensemble_weight_history rows, which only
# exist from V002 onward. Deliberately NOT migration_runner.assert_schema_current()
# / EXPECTED_SCHEMA_VERSION: that floor is the TRADER's requirement and will climb
# past 2 as later tasks ship (V003 in Task 8, V004 in Task 12) while this script's
# own requirement stays pinned at the V002 spine tables it actually writes to.
_REQUIRED_SCHEMA_VERSION = 2

ENVIRONMENTS: tuple[str, ...] = ("equity", "crypto")
ALGORITHMS: tuple[str, ...] = ("ppo", "a2c", "sac")

# No historical record of per-algo ensemble weight for a real-vintage bootstrap
# row (it isn't stored anywhere on disk) — equal weighting matches the same
# fallback the live execution pipeline uses (see
# ``src.swingrl.execution.pipeline._get_ensemble_weights``).
_DEFAULT_ENSEMBLE_WEIGHT = 1.0 / 3

_FALLBACK_QUERY = (
    "SELECT DISTINCT ON (environment, algorithm) * FROM model_metadata "
    "ORDER BY environment, algorithm, training_end_date DESC NULLS LAST"
)

_INSERT_TRAINING_RUN = (
    "INSERT INTO training_runs ("
    " iteration_number, environment, algorithm, fold_number, run_type, seed,"
    " attempt, status, era_id, code_version, data_fingerprint"
    ") VALUES (%s, %s, %s, -1, 'final_train', %s, 1, 'completed', 0,"
    " 'unknown_era0', 'unknown_era0')"
    " ON CONFLICT (iteration_number, environment, algorithm, fold_number, run_type, attempt)"
    " DO NOTHING RETURNING run_pk"
)

_SELECT_TRAINING_RUN = (
    "SELECT run_pk FROM training_runs"
    " WHERE iteration_number = %s AND environment = %s AND algorithm = %s"
    " AND fold_number = -1 AND run_type = 'final_train' AND attempt = 1"
)

_INSERT_MODEL = (
    "INSERT INTO models ("
    " model_id, run_pk, artifact_path, vecnormalize_path,"
    " artifact_sha256, vecnormalize_sha256, status"
    ") VALUES (%s, %s, %s, %s, %s, %s, 'active')"
    " ON CONFLICT (model_id) DO NOTHING RETURNING model_id"
)

_INSERT_ENSEMBLE_WEIGHT_HISTORY = (
    "INSERT INTO ensemble_weight_history (model_id, weight_frac, set_by)"
    " VALUES (%s, %s, 'training')"
)


@dataclass(frozen=True)
class _ResolvedIdentity:
    """Identity to stamp for one (environment, algorithm): real vintage or fallback."""

    iteration_number: int
    seed: int
    model_id: str
    artifact_path: Path
    vecnormalize_path: Path
    ensemble_weight: float
    source: Literal["vintage", "fallback"]


def _sha256(path: Path) -> str | None:
    """Return the sha256 hex digest of ``path``, or None (+ warning) if it is missing."""
    if not path.exists():
        log.warning("artifact_missing_sha_skipped", path=str(path))
        return None
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _fetch_fallback_rows(conn: psycopg.Connection[dict[str, Any]]) -> dict[tuple[str, str], Any]:
    """Return the newest model_metadata row per (environment, algorithm)."""
    rows = conn.execute(_FALLBACK_QUERY).fetchall()
    return {(row["environment"], row["algorithm"]): row for row in rows}


def _resolve_identity(
    environment: str,
    algorithm: str,
    models_root: Path,
    fallback_rows: dict[tuple[str, str], Any],
) -> _ResolvedIdentity | None:
    """Resolve the best-CPS vintage from disk, or fall back to model_metadata.

    Returns None when neither source has an identity for this (environment,
    algorithm) pair — logged as a warning; the caller skips it.
    """
    iteration = BEST_ERA0_VINTAGE[environment]
    base = models_root / "iterations" / f"iter_{iteration}" / "active" / environment / algorithm
    model_path = base / "model.zip"
    vecnorm_path = base / "vec_normalize.pkl"

    if model_path.exists() and vecnorm_path.exists():
        return _ResolvedIdentity(
            iteration_number=iteration,
            seed=ERA0_SEED_MAP[algorithm],
            model_id=f"era0-{environment}-{algorithm}",
            artifact_path=model_path,
            vecnormalize_path=vecnorm_path,
            ensemble_weight=_DEFAULT_ENSEMBLE_WEIGHT,
            source="vintage",
        )

    log.warning(
        "era0_vintage_missing_falling_back_to_model_metadata",
        environment=environment,
        algorithm=algorithm,
        expected_dir=str(base),
    )
    row = fallback_rows.get((environment, algorithm))
    if row is None:
        log.warning(
            "era0_identity_unresolvable_no_vintage_no_fallback",
            environment=environment,
            algorithm=algorithm,
        )
        return None

    weight = row["ensemble_weight"]
    return _ResolvedIdentity(
        iteration_number=SENTINEL,
        seed=SENTINEL,
        model_id=str(row["model_id"]),
        artifact_path=Path(row["model_path"]),
        vecnormalize_path=Path(row["vec_normalize_path"]),
        ensemble_weight=float(weight) if weight is not None else _DEFAULT_ENSEMBLE_WEIGHT,
        source="fallback",
    )


def _insert_identity(
    conn: psycopg.Connection[dict[str, Any]],
    environment: str,
    algorithm: str,
    identity: _ResolvedIdentity,
) -> bool:
    """Insert training_runs + models (+ ensemble_weight_history if new). Return True if new."""
    run_row = conn.execute(
        _INSERT_TRAINING_RUN,
        (identity.iteration_number, environment, algorithm, identity.seed),
    ).fetchone()
    if run_row is None:
        run_row = conn.execute(
            _SELECT_TRAINING_RUN,
            (identity.iteration_number, environment, algorithm),
        ).fetchone()
    if run_row is None:
        raise DataError(
            "era-0 bootstrap: training_runs row not found after insert/select for "
            f"iteration_number={identity.iteration_number} environment={environment} "
            f"algorithm={algorithm}"
        )
    run_pk = run_row["run_pk"]

    artifact_sha256 = _sha256(identity.artifact_path)
    vecnormalize_sha256 = _sha256(identity.vecnormalize_path)

    model_row = conn.execute(
        _INSERT_MODEL,
        (
            identity.model_id,
            run_pk,
            str(identity.artifact_path),
            str(identity.vecnormalize_path),
            artifact_sha256,
            vecnormalize_sha256,
        ),
    ).fetchone()
    if model_row is None:
        log.info("era0_model_already_bootstrapped", model_id=identity.model_id)
        return False

    conn.execute(_INSERT_ENSEMBLE_WEIGHT_HISTORY, (identity.model_id, identity.ensemble_weight))
    log.info(
        "era0_model_bootstrapped",
        environment=environment,
        algorithm=algorithm,
        model_id=identity.model_id,
        source=identity.source,
        iteration_number=identity.iteration_number,
        seed=identity.seed,
    )
    return True


def bootstrap_era0_models(db: DatabaseManager, models_root: Path) -> dict[str, int]:
    """Seed era-0 identity-spine rows for every (environment, algorithm) pair.

    Args:
        db: DatabaseManager providing pooled PostgreSQL connections. V001/V002
            must already be applied (schema_migrations at version >= 2).
        models_root: Root directory containing ``iterations/iter_{N}/active/...``.
            Never hardcoded — pass ``config.paths.models_dir`` or a CLI override.

    Returns:
        Counts: ``vintage`` (real-identity rows resolved from disk),
        ``fallback`` (sentinel rows resolved from model_metadata),
        ``unresolved`` (neither source available, skipped),
        ``skipped_existing`` (already bootstrapped by a prior run).

    Raises:
        ConfigError: The database schema is behind V002 (training_runs/models/
            ensemble_weight_history do not exist yet) — run apply_migrations first.
    """
    schema_version = current_schema_version(db)
    if schema_version < _REQUIRED_SCHEMA_VERSION:
        log.error(
            "era0_bootstrap_schema_behind",
            schema_version=schema_version,
            required=_REQUIRED_SCHEMA_VERSION,
        )
        raise ConfigError(
            f"Database schema version {schema_version} is behind the version "
            f"{_REQUIRED_SCHEMA_VERSION} required by era-0 bootstrap "
            "(training_runs/models/ensemble_weight_history); run "
            "scripts/apply_migrations.py before this script."
        )

    counts = {"vintage": 0, "fallback": 0, "unresolved": 0, "skipped_existing": 0}

    with db.connection() as conn:
        fallback_rows = _fetch_fallback_rows(conn)

    for environment in ENVIRONMENTS:
        for algorithm in ALGORITHMS:
            identity = _resolve_identity(environment, algorithm, models_root, fallback_rows)
            if identity is None:
                counts["unresolved"] += 1
                continue
            with db.connection() as conn:
                created = _insert_identity(conn, environment, algorithm, identity)
            counts[identity.source] += 1
            if not created:
                counts["skipped_existing"] += 1

    log.info("era0_bootstrap_complete", **counts)
    return counts


def main(argv: list[str] | None = None) -> int:
    """CLI entry point: bootstrap era-0 models against the configured database."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else "")
    parser.add_argument(
        "--config", type=Path, default=Path("config/swingrl.yaml"), help="SwingRL config YAML path"
    )
    parser.add_argument(
        "--models-root",
        type=Path,
        default=None,
        help="Override config.paths.models_dir as the models tree root",
    )
    args = parser.parse_args(argv)

    config = load_config(args.config)
    configure_logging(json_logs=config.logging.json_logs, log_level=config.logging.level)
    models_root = args.models_root or Path(config.paths.models_dir)

    db = DatabaseManager(config)
    counts = bootstrap_era0_models(db, models_root)
    print(f"bootstrap_era0_models: {counts}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
