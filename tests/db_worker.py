"""Per-session/per-worker isolated test databases cloned from a template.

Every xdist worker (and, opt-in via SWINGRL_TEST_ISOLATED_DB=1, every serial
session) gets its own scratch DB cloned file-level from ``swingrl_test_template``
(~100-300 ms via CREATE DATABASE … TEMPLATE — no migration replay), so parallel
workers and concurrent sessions can never share mutable DB state (the
2026-07-22 two-session corruption incident becomes structurally impossible).

Naming: derived names always end in ``_test`` so the existing name guard
(tests/db_guard.py:67 — any ``*_test`` passes) admits them UNCHANGED: gw0 on
base ``swingrl_test`` maps to ``swingrl_gw0_test``. (The scoping note's literal
``swingrl_test_gw0`` would be refused by the guard — pre-approved deviation,
see the plan's Task 8 rationale.)

The template is built/refreshed by scripts/prepare-test-db-template.sh (local)
or CI stage 2.8 (scripts/ci-homelab.sh) whenever a migration ships.
"""

from __future__ import annotations

import os
import time

import psycopg
from psycopg import sql

from swingrl.data.migration_runner import EXPECTED_SCHEMA_VERSION
from tests.db_guard import classify_db_url

TEMPLATE_DB = "swingrl_test_template"

_active_worker_url: str | None = None
_admin_base_url: str | None = None


def isolation_token() -> str | None:
    """Token for this process's isolated DB, or None to use DATABASE_URL as-is.

    xdist workers always isolate (PYTEST_XDIST_WORKER=gw0..gwN — set by xdist in
    each worker process, never in the controller). Serial sessions isolate when
    SWINGRL_TEST_ISOLATED_DB=1 — opt-in so single-session workflows keep working
    on hosts where the template has not been prepared yet.
    """
    worker = os.environ.get("PYTEST_XDIST_WORKER", "").strip()
    if worker:
        return worker
    if os.environ.get("SWINGRL_TEST_ISOLATED_DB", "").strip() == "1":
        return f"main{os.getpid()}"
    return None


def derive_isolated_db_url(base_url: str, token: str) -> str:
    """``…/swingrl_test`` + ``gw0`` -> ``…/swingrl_gw0_test`` (query string preserved)."""
    verdict, name = classify_db_url(base_url)
    if verdict != "safe" or name is None:
        raise RuntimeError(
            f"Cannot derive an isolated test DB from a non-test base URL "
            f"(verdict={verdict!r}, name={name!r})."
        )
    stem = name[: -len("_test")] if name.endswith("_test") else name
    new_name = f"{stem}_{token}_test"
    before, sep, after = base_url.rpartition(f"/{name}")
    if not sep:
        raise RuntimeError(f"Base URL does not contain database name {name!r}.")
    return f"{before}/{new_name}{after}"


def ensure_isolated_db(base_url: str, worker_url: str) -> None:
    """Create the isolated DB from TEMPLATE_DB (dropping any stale copy first).

    Runs on an admin connection to the BASE test DB (CREATE DATABASE cannot run
    from inside the database being created). Retries the transient template-busy
    error two workers can hit when cloning simultaneously, then verifies the
    clone's ledger is at EXPECTED_SCHEMA_VERSION so a stale template fails loud.
    """
    _, worker_name = classify_db_url(worker_url)
    with psycopg.connect(base_url, autocommit=True) as admin:
        admin.execute(
            sql.SQL("DROP DATABASE IF EXISTS {} WITH (FORCE)").format(sql.Identifier(worker_name))
        )
        for attempt in (1, 2, 3):
            try:
                admin.execute(
                    sql.SQL("CREATE DATABASE {} TEMPLATE {}").format(
                        sql.Identifier(worker_name), sql.Identifier(TEMPLATE_DB)
                    )
                )
                break
            except psycopg.errors.InvalidCatalogName as exc:
                raise RuntimeError(
                    f"Template database {TEMPLATE_DB!r} does not exist — run "
                    f"scripts/prepare-test-db-template.sh (or ci-homelab stage 2.8)."
                ) from exc
            except psycopg.errors.ObjectInUse:
                if attempt == 3:
                    raise
                time.sleep(0.5 * attempt)
    with psycopg.connect(worker_url, autocommit=True) as conn:
        row = conn.execute("SELECT max(version) FROM schema_migrations").fetchone()
    version = row[0] if row is not None else None
    if version != EXPECTED_SCHEMA_VERSION:
        raise RuntimeError(
            f"{TEMPLATE_DB} clone is at schema version {version}, expected "
            f"{EXPECTED_SCHEMA_VERSION} — re-run scripts/prepare-test-db-template.sh."
        )


def activate_isolated_db() -> None:
    """Rewrite DATABASE_URL to this process's isolated clone (no-op when not applicable)."""
    global _active_worker_url, _admin_base_url
    token = isolation_token()
    base_url = os.environ.get("DATABASE_URL", "").strip()
    if token is None or not base_url:
        return
    worker_url = derive_isolated_db_url(base_url, token)
    ensure_isolated_db(base_url, worker_url)
    os.environ["DATABASE_URL"] = worker_url
    _active_worker_url = worker_url
    _admin_base_url = base_url


def drop_isolated_db() -> None:
    """Drop this process's clone at session end (best-effort).

    Stale clones from killed sessions are reaped by
    scripts/prepare-test-db-template.sh and by CI stage 6.
    """
    global _active_worker_url
    if _active_worker_url is None or _admin_base_url is None:
        return
    _, worker_name = classify_db_url(_active_worker_url)
    if worker_name is None:
        return
    try:
        with psycopg.connect(_admin_base_url, autocommit=True) as admin:
            admin.execute(
                sql.SQL("DROP DATABASE IF EXISTS {} WITH (FORCE)").format(
                    sql.Identifier(worker_name)
                )
            )
    except psycopg.OperationalError:
        pass
    _active_worker_url = None
