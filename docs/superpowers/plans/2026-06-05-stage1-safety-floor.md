# Stage 1 — Test Safety Floor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make DB-dependent tests clean up after themselves so homelab CI is green (0 failures), with a three-layer guarantee that cleanup can never touch the production `swingrl` database — then drain the 249-commit branch to `main`.

**Architecture:** An autouse, function-scoped pytest fixture `TRUNCATE`s every `public`-schema table after each test, using a dedicated short-lived `psycopg` connection (not the pooled `DatabaseManager`). Two software guards wrap it: a pre-suite guard (`pytest_configure`) that refuses to start unless the **resolved** DB name is a test DB, and a pre-wipe re-check that refuses to TRUNCATE unless the resolved DB ends in `_test`. Both resolve the target the same way `DatabaseManager` does (env `DATABASE_URL` first, then `config.system.database_url`), closing the blank-env fallback hole. Postgres connection-binding is the backbone — a `*_test` connection physically cannot reach `swingrl`.

**Tech Stack:** Python 3.11, pytest, psycopg 3 (`psycopg.sql` for safe identifier composition), PostgreSQL (`pg16` on homelab), `scripts/ci-homelab.sh` as the only sanctioned test runner against a live DB.

**Spec:** [`docs/superpowers/specs/2026-06-05-stage1-safety-floor-design.md`](../specs/2026-06-05-stage1-safety-floor-design.md)
**Tracker:** [`.planning/V1.1_EXECUTION_PLAN.md`](../../../.planning/V1.1_EXECUTION_PLAN.md) → Stage 1

---

## Files at a glance

| File | Responsibility | Action |
|---|---|---|
| `tests/db_guard.py` | Pure helpers: resolve DB URL (env→config) + classify (blank/safe/unsafe/unparseable). Shared by conftest + fixture; no pytest/DB deps → no circular import. | **Create** |
| `tests/fixtures/db_cleanup.py` | Autouse wipe fixture + pre-wipe re-check + the TRUNCATE-all routine. | **Create** |
| `tests/conftest.py` | Harden `pytest_configure` to use the resolver; register the wipe fixture by import; drop the now-duplicated regex/constants. | **Modify** (`:36-60`) |
| `tests/test_db_guard.py` | Unit tests for resolver + classifier. | **Create** |
| `tests/test_db_cleanup_guard.py` | Unit tests for the pre-wipe re-check. | **Create** |
| `tests/test_db_isolation.py` | DB-gated regression: two tests sharing a table prove the wipe isolates state. | **Create** |
| `tests/execution/conftest.py` | Remove dead commented-out TRUNCATE block. | **Modify** (`:89-96`) |
| `tests/data/test_db.py` | Remove dead commented-out TRUNCATE block. | **Modify** (`:88-96`) |
| `tests/data/test_parquet_to_duckdb.py` | Remove dead commented-out TRUNCATE block. | **Modify** |
| `tests/data/test_ingestion_logging.py` | Remove dead commented-out TRUNCATE block. | **Modify** |
| `tests/data/test_cross_source.py` | Remove dead commented-out TRUNCATE block. | **Modify** |
| `tests/data/test_corporate_actions.py` | Remove dead commented-out TRUNCATE block. | **Modify** |
| `.planning/V1.1_EXECUTION_PLAN.md` | Sync the stale Stage 1 deliverables checklist (lists deferred pg-test/SAVEPOINT items) to the locked wipe approach. | **Modify** (`:138-146`) |

**Out of scope (deferred — do NOT do here):** separate `pg-test` container, SAVEPOINT rollback fixture, raw-connection/DDL test refactor, repository pattern, server-side privilege revocation, F1 turbulence bug. See spec "Non-goals".

---

### Task 1: Shared DB-guard helpers (`tests/db_guard.py`)

Pure, dependency-light functions both the suite guard and the wipe fixture call. Lives outside `conftest.py` to avoid a conftest↔fixture circular import.

**Files:**
- Create: `tests/db_guard.py`
- Test: `tests/test_db_guard.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_db_guard.py`:

```python
"""STAGE1: Pure DB-guard helpers — URL classification and target resolution."""

from __future__ import annotations

import pytest

import tests.db_guard as guard
from tests.db_guard import classify_db_url, resolve_target_db_url


@pytest.mark.parametrize(
    ("url", "expected_verdict", "expected_name"),
    [
        ("", "blank", None),
        ("   ", "blank", None),
        ("postgresql://swingrl:pw@pg16:5432/swingrl_test", "safe", "swingrl_test"),
        ("postgresql://u:pw@h:5432/anything_test", "safe", "anything_test"),
        ("postgresql://u:pw@h:5432/swingrl_test?sslmode=require", "safe", "swingrl_test"),
        ("postgresql://swingrl:pw@pg16:5432/swingrl", "unsafe", "swingrl"),
        ("postgresql://u:pw@h:5432/prod?x=1", "unsafe", "prod"),
        ("not-a-url", "unparseable", None),
    ],
)
def test_classify_db_url(url: str, expected_verdict: str, expected_name: str | None) -> None:
    """REQ-STAGE1: classify_db_url labels blank/safe/unsafe/unparseable correctly."""
    verdict, name = classify_db_url(url)
    assert verdict == expected_verdict
    assert name == expected_name


def test_resolve_prefers_env_over_config(monkeypatch: pytest.MonkeyPatch) -> None:
    """REQ-STAGE1: DATABASE_URL env wins over the config fallback."""
    monkeypatch.setenv("DATABASE_URL", "postgresql://u:pw@h:5432/swingrl_test")
    assert resolve_target_db_url() == "postgresql://u:pw@h:5432/swingrl_test"


def test_resolve_falls_back_to_config_url(monkeypatch: pytest.MonkeyPatch) -> None:
    """REQ-STAGE1: blank env resolves to config.system.database_url (the hole the guard closes)."""
    monkeypatch.delenv("DATABASE_URL", raising=False)

    class _Sys:
        database_url = "postgresql://u:pw@h:5432/swingrl"  # production!

    class _Cfg:
        system = _Sys()

    monkeypatch.setattr(guard, "load_config", lambda _path: _Cfg())
    assert resolve_target_db_url() == "postgresql://u:pw@h:5432/swingrl"
    # And that resolved value classifies unsafe → the suite would refuse to run.
    assert classify_db_url(resolve_target_db_url())[0] == "unsafe"
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_db_guard.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'tests.db_guard'`.

- [ ] **Step 3: Write the helper module**

Create `tests/db_guard.py`:

```python
"""Pure helpers for the test-database safety guard.

Lives in its own module (not conftest) so both ``tests/conftest.py`` and
``tests/fixtures/db_cleanup.py`` can import it without a circular dependency.

A Postgres connection is bound to exactly one database, so a ``*_test`` connection
can never see or TRUNCATE the production ``swingrl`` database. These helpers add
two software checks on top of that backbone (a pre-suite guard and a pre-wipe
re-check), both resolving the target the *same way* ``DatabaseManager`` does:
env ``DATABASE_URL`` first, then ``config.system.database_url``.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

from swingrl.config.schema import load_config

# Database names that are always safe to mutate/wipe.
SAFE_DB_NAMES = frozenset({"swingrl_test"})

# Extract the database name (last path segment) from a postgres URL:
#   postgresql://user:pw@host:5432/<name>?opts  ->  <name>
_DB_NAME_RE = re.compile(r"/([^/?]+)(?:\?|$)")

# The YAML config DatabaseManager falls back to when DATABASE_URL is unset.
_DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "swingrl.yaml"


def resolve_target_db_url() -> str:
    """Resolve the DB URL the same way ``DatabaseManager.__init__`` does.

    Order: ``DATABASE_URL`` env var, then ``config.system.database_url`` from the
    default YAML config. Returns ``""`` when neither is set (no DB → tests skip).
    """
    env_url = os.environ.get("DATABASE_URL", "").strip()
    if env_url:
        return env_url
    try:
        config = load_config(_DEFAULT_CONFIG_PATH)
    except Exception:  # noqa: BLE001 — guard must never crash the whole suite
        return ""
    return (config.system.database_url or "").strip()


def classify_db_url(db_url: str) -> tuple[str, str | None]:
    """Classify a DB URL for guard decisions.

    Returns ``(verdict, db_name)`` where verdict is one of:
      - ``"blank"``       — no URL; no DB configured (safe; tests skip)
      - ``"unparseable"`` — URL present but no database name found (refuse)
      - ``"safe"``        — resolved name is a recognised test database
      - ``"unsafe"``      — resolved name is NOT a test database (refuse)
    """
    if not db_url or not db_url.strip():
        return ("blank", None)
    match = _DB_NAME_RE.search(db_url.strip())
    if not match:
        return ("unparseable", None)
    name = match.group(1)
    if name in SAFE_DB_NAMES or name.endswith("_test"):
        return ("safe", name)
    return ("unsafe", name)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_db_guard.py -v`
Expected: PASS (8 parametrized + 2 = 10 passed).

- [ ] **Step 5: Commit**

```bash
git add tests/db_guard.py tests/test_db_guard.py
git commit -m "test(19.1): add shared DB-guard helpers (resolve + classify)"
```

---

### Task 2: Harden the pre-suite guard (`tests/conftest.py`)

Route `pytest_configure` through the resolver so it sees the *real* target (closing the blank-env→config-fallback hole), and delete the now-duplicated regex/constant.

**Files:**
- Modify: `tests/conftest.py:36-60` (the `_SAFE_DB_NAMES` / `_DB_NAME_RE` / `pytest_configure` block)

- [ ] **Step 1: Replace the guard block**

In `tests/conftest.py`, the current top-of-file imports include (line 23-24):

```python
from swingrl.config.schema import SwingRLConfig, load_config
from swingrl.data.db import DatabaseManager
```

Add immediately after them:

```python
from tests.db_guard import SAFE_DB_NAMES, classify_db_url, resolve_target_db_url
```

Then **delete** lines 36-60 (the `_SAFE_DB_NAMES`, `_DB_NAME_RE`, and old `pytest_configure`):

```python
_SAFE_DB_NAMES = {"swingrl_test"}
_DB_NAME_RE = re.compile(r"/([^/?]+)(?:\?|$)")


def pytest_configure(config: pytest.Config) -> None:
    """Refuse to run if DATABASE_URL points at a non-test database."""
    db_url = os.environ.get("DATABASE_URL", "").strip()
    if not db_url:
        return  # No DB configured — tests that need one will skip.
    match = _DB_NAME_RE.search(db_url)
    if not match:
        pytest.exit(
            f"DATABASE_URL has no parseable database name; refusing to run pytest. Got: {db_url!r}",
            returncode=2,
        )
    db_name = match.group(1)
    if db_name in _SAFE_DB_NAMES or db_name.endswith("_test"):
        return
    pytest.exit(
        f"REFUSING TO RUN: DATABASE_URL points at database {db_name!r}, which is "
        f"not a test database. Test fixtures DROP/TRUNCATE/DELETE production tables. "
        f"Use a database name in {sorted(_SAFE_DB_NAMES)} or one ending in '_test'. "
        f"In CI, scripts/ci-homelab.sh overrides DATABASE_URL automatically.",
        returncode=2,
    )
```

Replace it with:

```python
def pytest_configure(config: pytest.Config) -> None:
    """Refuse to run the suite unless the *resolved* DB target is a test database.

    Resolves the target the same way DatabaseManager does (env DATABASE_URL, then
    config.system.database_url), so a blank env var with a production URL in YAML
    config cannot slip past the guard.
    """
    verdict, db_name = classify_db_url(resolve_target_db_url())
    if verdict in {"blank", "safe"}:
        return
    if verdict == "unparseable":
        pytest.exit(
            "Resolved database URL has no parseable database name; refusing to run pytest.",
            returncode=2,
        )
    pytest.exit(
        f"REFUSING TO RUN: resolved database {db_name!r} is not a test database. "
        f"Test fixtures TRUNCATE/DELETE tables. Use a name in "
        f"{sorted(SAFE_DB_NAMES)} or one ending in '_test'. In CI, "
        f"scripts/ci-homelab.sh overrides DATABASE_URL automatically.",
        returncode=2,
    )
```

The module-level `import re` (line 11) is now unused **only if** no other code in conftest uses it — leave the `import re` removal to ruff in Step 3 (it will flag `F401` if truly unused; remove only if flagged). `import os` is still used by other fixtures — keep it.

- [ ] **Step 2: Verify the guard still blocks a production DB (manual, safe — it only refuses)**

Run: `DATABASE_URL="postgresql://u:pw@h:5432/swingrl" uv run pytest tests/test_db_guard.py --co -q`
Expected: exits with `REFUSING TO RUN: resolved database 'swingrl' is not a test database …`, returncode 2. (Collection-only `--co` never touches the DB; the guard fires before collection.)

- [ ] **Step 3: Verify collection still works with no DB configured**

Run: `uv run ruff check tests/conftest.py && uv run pytest tests/test_db_guard.py -v`
Expected: ruff clean (remove `import re` if `F401`); tests PASS.

- [ ] **Step 4: Commit**

```bash
git add tests/conftest.py
git commit -m "test(19.1): harden pytest_configure guard to resolve env+config DB target"
```

---

### Task 3: Autouse wipe fixture (`tests/fixtures/db_cleanup.py`)

The core of the floor. Truncates all `public` tables after each test; no-ops when no test DB; re-checks the resolved DB name ends in `_test` before every TRUNCATE.

**Files:**
- Create: `tests/fixtures/db_cleanup.py`
- Modify: `tests/conftest.py` (register the fixture by import)
- Test: `tests/test_db_cleanup_guard.py`

- [ ] **Step 1: Write the failing test for the pre-wipe re-check**

Create `tests/test_db_cleanup_guard.py`:

```python
"""STAGE1: Pre-wipe re-check (safety layer #3) refuses non-test databases."""

from __future__ import annotations

import pytest

from tests.fixtures.db_cleanup import ensure_wipe_target_is_test_db


def test_refuses_production_db() -> None:
    """REQ-STAGE1: pre-wipe re-check raises on a production DB name."""
    with pytest.raises(RuntimeError, match="not a test database"):
        ensure_wipe_target_is_test_db("postgresql://u:pw@h:5432/swingrl")


def test_refuses_unparseable_db() -> None:
    """REQ-STAGE1: pre-wipe re-check raises when the DB name cannot be parsed."""
    with pytest.raises(RuntimeError, match="not a test database"):
        ensure_wipe_target_is_test_db("not-a-url")


def test_allows_test_db() -> None:
    """REQ-STAGE1: pre-wipe re-check permits a *_test database."""
    assert ensure_wipe_target_is_test_db("postgresql://u:pw@h:5432/swingrl_test") is True


def test_skips_when_blank() -> None:
    """REQ-STAGE1: no DB configured → skip wipe (returns False, no raise)."""
    assert ensure_wipe_target_is_test_db("") is False
```

- [ ] **Step 2: Run it to verify it fails**

Run: `uv run pytest tests/test_db_cleanup_guard.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'tests.fixtures.db_cleanup'`.

- [ ] **Step 3: Write the fixture module**

Create `tests/fixtures/db_cleanup.py`:

```python
"""Autouse fixture that wipes the test database after every test.

Registered by importing ``wipe_db_after_test`` into ``tests/conftest.py``.
Function-scoped and autouse, so every test starts from a clean database —
killing the inter-test state pollution that fails ~50 tests on homelab CI.

Three safety layers (see ``tests/db_guard.py``):
  1. Postgres binds a connection to one database, so a ``*_test`` connection
     physically cannot reach production ``swingrl``.
  2. The suite-level guard (conftest ``pytest_configure``) refuses to start
     unless the resolved DB is a test database.
  3. ``ensure_wipe_target_is_test_db`` re-checks the resolved DB name ends in
     ``_test`` immediately before every TRUNCATE and refuses otherwise.
"""

from __future__ import annotations

from collections.abc import Generator

import psycopg
import pytest
from psycopg import sql

from swingrl.data.db import DatabaseManager
from tests.db_guard import classify_db_url, resolve_target_db_url


def ensure_wipe_target_is_test_db(db_url: str) -> bool:
    """Pre-wipe re-check (safety layer #3).

    Returns ``True`` when the resolved DB is a test database and should be wiped,
    ``False`` when no DB is configured (skip). Raises ``RuntimeError`` when a DB
    is present but is NOT a recognised test database — cleanup must never run
    against production.
    """
    verdict, db_name = classify_db_url(db_url)
    if verdict == "blank":
        return False
    if verdict != "safe":
        raise RuntimeError(
            f"Refusing to TRUNCATE: resolved database {db_name!r} is not a test "
            f"database (verdict={verdict!r}). The suite guard should have already "
            f"aborted — this is a defence-in-depth backstop."
        )
    return True


def _truncate_all_public_tables(db_url: str) -> None:
    """TRUNCATE every table in the public schema of the connected test database.

    Uses a dedicated short-lived autocommit connection (not the pooled
    DatabaseManager connection) to avoid pool-affinity surprises, with a bounded
    ``lock_timeout`` so a stray lock from a misbehaving test fails fast rather
    than hanging CI. Catalog-enumerates tables so ad-hoc tables created by raw
    tests are covered; a single ``TRUNCATE … CASCADE`` handles FK ordering.
    """
    with psycopg.connect(db_url, autocommit=True) as conn:
        conn.execute("SET lock_timeout = '10s'")
        rows = conn.execute(
            "SELECT tablename FROM pg_tables WHERE schemaname = 'public'"
        ).fetchall()
        tables = [row[0] for row in rows]
        if not tables:
            return
        statement = sql.SQL("TRUNCATE TABLE {} RESTART IDENTITY CASCADE").format(
            sql.SQL(", ").join(sql.Identifier(name) for name in tables)
        )
        conn.execute(statement)


@pytest.fixture(autouse=True)
def wipe_db_after_test() -> Generator[None, None, None]:
    """Wipe all test-database tables after each test (no-op when no test DB)."""
    yield
    db_url = resolve_target_db_url()
    if not ensure_wipe_target_is_test_db(db_url):
        return
    # Close any pooled connections so their locks are released before TRUNCATE.
    DatabaseManager.reset()
    _truncate_all_public_tables(db_url)
```

- [ ] **Step 4: Register the fixture in `tests/conftest.py`**

Add to the import block in `tests/conftest.py` (just below the `from tests.db_guard import …` line added in Task 2):

```python
# Autouse fixture — imported so it registers globally (wipes test DB after each test).
from tests.fixtures.db_cleanup import wipe_db_after_test  # noqa: F401
```

- [ ] **Step 5: Run the unit tests to verify they pass**

Run: `uv run pytest tests/test_db_cleanup_guard.py -v`
Expected: PASS (4 passed).

- [ ] **Step 6: Verify the suite still collects cleanly (no DB) and nothing regressed**

Run: `uv run pytest tests/ -q -x`
Expected: same pass/skip counts as before this task (DB tests skip without `DATABASE_URL`; the autouse fixture no-ops on blank). No new failures, no collection errors.

- [ ] **Step 7: Commit**

```bash
git add tests/fixtures/db_cleanup.py tests/conftest.py tests/test_db_cleanup_guard.py
git commit -m "test(19.1): autouse wipe-after-each-test fixture with pre-wipe re-check"
```

---

### Task 4: DB-gated isolation regression test (`tests/test_db_isolation.py`)

Proves end-to-end that the wipe isolates state between two tests sharing one table. Self-contained (creates its own probe table) so it does not couple to any production schema.

**Files:**
- Create: `tests/test_db_isolation.py`

- [ ] **Step 1: Write the regression test**

Create `tests/test_db_isolation.py`:

```python
"""STAGE1: The autouse wipe isolates DB state between tests.

Two tests insert into the same probe table and each assert exactly one row.
Without the wipe the second test would see two rows (the first test's leftover);
with the autouse wipe each test starts clean. DB-gated: skips without a test DB.
"""

from __future__ import annotations

import os

import psycopg
import pytest

pytestmark = pytest.mark.skipif(
    not os.environ.get("DATABASE_URL"),
    reason="requires a *_test PostgreSQL database (set DATABASE_URL via ci-homelab.sh)",
)

_PROBE_TABLE = "stage1_isolation_probe"


def _insert_probe_and_count() -> int:
    """Create the probe table if needed, insert one row, return its row count."""
    db_url = os.environ["DATABASE_URL"]
    with psycopg.connect(db_url, autocommit=True) as conn:
        conn.execute(f"CREATE TABLE IF NOT EXISTS {_PROBE_TABLE} (n integer)")
        conn.execute(f"INSERT INTO {_PROBE_TABLE} (n) VALUES (1)")
        row = conn.execute(f"SELECT count(*) FROM {_PROBE_TABLE}").fetchone()
    assert row is not None
    return int(row[0])


def test_isolation_first_insert() -> None:
    """REQ-STAGE1: first test inserts the probe row and sees exactly one."""
    assert _insert_probe_and_count() == 1


def test_isolation_second_insert() -> None:
    """REQ-STAGE1: after the wipe, the second test sees a clean table — one again."""
    assert _insert_probe_and_count() == 1
```

- [ ] **Step 2: Verify RED — without the wipe, the pair pollutes (one-time manual check)**

Temporarily comment out the `from tests.fixtures.db_cleanup import wipe_db_after_test` line in `tests/conftest.py`, then run against the test DB **via the safe path** (never ad-hoc against `pg16`):

Run (on homelab, with a `*_test` DATABASE_URL exported by you, NOT production):
`DATABASE_URL="postgresql://swingrl:<pw>@pg16:5432/swingrl_test" uv run pytest tests/test_db_isolation.py -v`
Expected: `test_isolation_second_insert` FAILS (`assert 2 == 1`) — proving the pollution the fixture fixes.

Then **un-comment** the import line. This RED check is a one-time manual verification; do not leave the import commented.

- [ ] **Step 3: Verify GREEN — with the wipe, the pair passes**

Run: `DATABASE_URL="postgresql://swingrl:<pw>@pg16:5432/swingrl_test" uv run pytest tests/test_db_isolation.py -v`
Expected: both tests PASS.

> If you cannot safely export a `*_test` DATABASE_URL in this environment, defer Steps 2-3 to the full `ci-homelab.sh` run in Task 6 (the test will run there); still commit the test now.

- [ ] **Step 4: Commit**

```bash
git add tests/test_db_isolation.py
git commit -m "test(19.1): DB-gated regression proving wipe isolates inter-test state"
```

---

### Task 5: Remove the dead commented-out TRUNCATE blocks

These per-module TRUNCATE hacks are commented out and superseded by the single autouse fixture (DRY). Remove all six.

**Files:**
- Modify: `tests/execution/conftest.py`, `tests/data/test_db.py`, `tests/data/test_parquet_to_duckdb.py`, `tests/data/test_ingestion_logging.py`, `tests/data/test_cross_source.py`, `tests/data/test_corporate_actions.py`

- [ ] **Step 1: Locate every block**

Run: `grep -rn "EXECUTE 'TRUNCATE TABLE ' || quote_ident" tests/`
Expected: six hits (one per file above), each inside a `# with …:` / `# conn.execute(` commented block.

- [ ] **Step 2: Remove each block**

In `tests/execution/conftest.py`, delete these lines (the commented block between the two `DatabaseManager.reset()` calls — currently `:89-96`):

```python
    # Truncate all tables for test isolation
    # with db.connection() as conn:
    #     conn.execute(
    #         "DO $$ DECLARE r RECORD; BEGIN "
    #         "FOR r IN SELECT tablename FROM pg_tables WHERE schemaname = 'public' LOOP "
    #         "EXECUTE 'TRUNCATE TABLE ' || quote_ident(r.tablename) || ' CASCADE'; "
    #         "END LOOP; END $$"
    #     )
```

In `tests/data/test_db.py`, delete the equivalent commented block (currently `:88-96`):

```python
    # Truncate all tables for test isolation
    # with mgr.connection() as conn:
    #     conn.execute(
    #         "DO $$ DECLARE r RECORD; BEGIN "
    #         "FOR r IN SELECT tablename FROM pg_tables WHERE schemaname = 'public' LOOP "
    #         "EXECUTE 'TRUNCATE TABLE ' || quote_ident(r.tablename) || ' CASCADE'; "
    #         "END LOOP; END $$"
    #     )
```

Repeat for the remaining four files (`test_parquet_to_duckdb.py`, `test_ingestion_logging.py`, `test_cross_source.py`, `test_corporate_actions.py`) — each has the same commented `# Truncate all tables …` / `# DO $$ …` block; remove the whole commented block, leaving the surrounding `DatabaseManager.reset()` / teardown lines intact.

> Leave the **active** (non-commented) TRUNCATE statements alone — `tests/data/test_gap_fill.py` and `tests/test_memory_service.py` truncate as deliberate setup and are harmless alongside the autouse teardown. Scope here is commented-out blocks only.

- [ ] **Step 3: Verify nothing references the removed blocks and lint is clean**

Run: `grep -rn "EXECUTE 'TRUNCATE TABLE ' || quote_ident" tests/ ; uv run ruff check tests/`
Expected: grep returns nothing; ruff clean.

- [ ] **Step 4: Verify suite still collects and passes (no DB)**

Run: `uv run pytest tests/ -q`
Expected: same pass/skip counts as Task 3 Step 6; 0 failures.

- [ ] **Step 5: Commit**

```bash
git add tests/execution/conftest.py tests/data/test_db.py tests/data/test_parquet_to_duckdb.py \
        tests/data/test_ingestion_logging.py tests/data/test_cross_source.py tests/data/test_corporate_actions.py
git commit -m "test(19.1): remove dead commented-out per-module TRUNCATE blocks (DRY)"
```

---

### Task 6: Sync the stale tracker checklist + G3 homelab CI

Bring the tracker's Stage 1 deliverables in line with the locked wipe approach, then run the only sanctioned live-DB gate.

**Files:**
- Modify: `.planning/V1.1_EXECUTION_PLAN.md:138-146`

- [ ] **Step 1: Replace the stale deliverables checklist**

In `.planning/V1.1_EXECUTION_PLAN.md`, replace the Stage 1 **Deliverables** list (currently lines 138-143, which still names the deferred `pg-test` service and SAVEPOINT fixture) with the locked approach:

```markdown
**Deliverables**
- [ ] `tests/db_guard.py` — shared resolve(env→config)+classify helpers (L1 backbone software checks)
- [ ] Harden `tests/conftest.py::pytest_configure` to resolve env **and** config fallback (Check #1)
- [ ] `tests/fixtures/db_cleanup.py` — autouse wipe-after-each-test fixture + pre-wipe `_test` re-check (Check #2)
- [ ] DB-gated isolation regression test (`tests/test_db_isolation.py`)
- [ ] Remove the six dead commented-out per-module TRUNCATE blocks (DRY)
- [ ] No CI/DATABASE_URL change — `ci-homelab.sh` already creates a fresh `swingrl_test` on `pg16`; the fixture handles intra-run isolation
- [ ] _(deferred)_ separate `pg-test` server + SAVEPOINT rollback fixture → **Stage 3.0**
```

- [ ] **Step 2: Append the dated Changelog line**

In the same file's `## Changelog` section, add:

```markdown
- **2026-06-05** — Stage 1 plan approved (G2) and implemented: wipe-after-each-test fixture
  + 3-layer guard; six dead TRUNCATE blocks removed. Deliverables checklist synced to the
  locked approach (pg-test/SAVEPOINT remain deferred to Stage 3.0).
```

- [ ] **Step 3: Commit the doc sync**

```bash
git add .planning/V1.1_EXECUTION_PLAN.md
git commit -m "docs(19.1): sync Stage 1 deliverables checklist to locked wipe approach"
```

- [ ] **Step 4: 🚩 G3 — run homelab CI (the SAFE path) and require 0 failures**

Push the branch, then run CI in the `~/swingrl` live checkout per `CLAUDE.md`:

```bash
git push origin gsd/phase-19.1-memory-agent-infrastructure-and-training
cd ~/swingrl && git fetch origin \
  && git checkout gsd/phase-19.1-memory-agent-infrastructure-and-training \
  && git pull origin gsd/phase-19.1-memory-agent-infrastructure-and-training \
  && bash scripts/ci-homelab.sh --no-cache
```

Expected: `=== CI PASSED ===` with pytest reporting **0 failures** (the ~50 previously-failing DB tests now pass; `test_db_isolation.py` passes against the fresh `swingrl_test`).

- [ ] **Step 5: If any test still fails**

Use `superpowers:systematic-debugging`. The likely cause is a test that depended on cross-test accumulated state (the spec's risk #4) — fix that test to assume empty state (it encodes the bug being removed), or a table the wipe didn't reach (confirm it is in the `public` schema). Do **not** weaken the guards. Re-run Step 4 until green.

---

## G4 — Consolidation merge (user-gated, after G3 green)

Not a code task — performed once CI is green and you approve:

- Open one consolidation PR: `gsd/phase-19.1-memory-agent-infrastructure-and-training` → `main`.
- Merge with **`--no-ff`** to preserve all 249 commits (Postgres migration + iter 0-4 recovery forensics).
- Retire the `gsd/` branch. From Stage 2, new work uses `swingrl/<N>-<slug>` cut from the updated `main`.

---

## Self-Review

**Spec coverage** (against `2026-06-05-stage1-safety-floor-design.md`):
- §"Autouse wipe fixture" → Task 3 (catalog-enumerate, single `TRUNCATE … CASCADE`, dedicated short-lived conn, pre-wipe re-check, no-op when blank). ✓
- §"Guard hardening (Check #1)" → Task 1 (`resolve_target_db_url`) + Task 2 (conftest uses it). ✓ Note: the spec names a `_resolve_db_name()` helper; this plan splits it into `resolve_target_db_url()` + `classify_db_url()` for unit-testability — same behaviour, better tested.
- §"Remove the dead TRUNCATE blocks" → Task 5 (all six, vs the spec's partial list — superset, intentional). ✓
- §"Raw-connection / DDL tests handled, not refactored" → no refactor in plan; wipe covers them. ✓
- §"CI: no change to fresh-DB-per-run" → Task 6 Step 1 deliverable note; `ci-homelab.sh` untouched. ✓
- §"Acceptance criteria" 1-5 → Task 6 (CI 0 failures), Task 1/2 (guard rejects non-test, env+config), Task 3 (fixture refuses non-`_test`), grep for `swingrl` confirms no prod name in test path, Task 4/6 (50 pass, none regress). ✓
- §"TDD order" 1-4 → Tasks 1→2 (guard), 3 (refuse), 4 (pollution pair), 6 (full CI). ✓
- §"Open questions" → catalog-enumerate (Task 3, recommended), dedicated connection (Task 3, recommended), failing-test list enumerated by Task 6's CI run. ✓

**Placeholder scan:** every code step shows complete code; every command shows expected output. No TBD/TODO. ✓

**Type consistency:** `resolve_target_db_url() -> str`, `classify_db_url(str) -> tuple[str, str|None]`, `ensure_wipe_target_is_test_db(str) -> bool`, `wipe_db_after_test` fixture used consistently across Tasks 1-4. `SAFE_DB_NAMES` (public) imported in conftest and used in db_guard. ✓
