# Stage 1 — Test Safety Floor (Wipe-Based Isolation + Guard Hardening)

**Status:** Draft — awaiting G1 (spec) approval
**Date:** 2026-06-05 ET
**Part of:** [`V1.1_EXECUTION_PLAN.md`](../../../.planning/V1.1_EXECUTION_PLAN.md) → Stage 1
**Design source / supersedes premise of:** [`2026-05-19-phase-19.2-testing-foundation-design.md`](2026-05-19-phase-19.2-testing-foundation-design.md)
(this spec **scopes down** 19.2.0 to the safety-critical subset and **revises one premise** — see "Key finding")

---

## Why this exists

~50 tests fail on homelab CI from **inter-test state pollution**: DB-dependent tests share one
`swingrl_test` database and do **not clean up after themselves**. The per-test `TRUNCATE`
teardown blocks exist but are **commented out** (`tests/execution/conftest.py`,
`tests/data/test_db.py`, and others); tests lean on `ON CONFLICT DO NOTHING` (which masks
duplicates but leaves stale rows) and assume an empty/known table state. The next test reads the
previous test's leftovers and fails.

This blocks the Stage 1 consolidation merge (the 249-commit work branch → `main`). Production
data has been destroyed by tests before, so the fix must make it **impossible for cleanup to
touch production**, not merely unlikely.

### Key finding (revises the 19.2 spec's premise)

The 19.2 spec assumed an L4 **SAVEPOINT rollback** fixture fixes the 50 failures. The
investigation showed it **cannot, alone**: a meaningful subset of DB tests use raw
`psycopg.connect(..., autocommit=True)` **outside** `DatabaseManager` and run **DDL**
(`init_postgres_schema` / `CREATE TABLE`) per test, which auto-commits and escapes transaction
rollback (e.g. `tests/test_phase15.py:81-162`, `tests/monitoring/test_stuck_agent.py:26-52`).
The pooled connection (max=20) also breaks SAVEPOINT→ROLLBACK affinity. A **wipe-after-each-test**
(TRUNCATE) approach covers all tests regardless of connection style or DDL — so Stage 1 uses
wipe; the elegant rollback + raw-test refactor is deferred to Stage 3.0.

## Goal

Turn homelab CI green (0 failures) by making DB tests clean up after themselves, with a
**three-layer guarantee** that cleanup only ever touches a test database — never production.
Minimal floor; defer the polish.

## Non-goals (explicitly deferred)

- SAVEPOINT/transaction rollback fixture → Stage 3.0
- Separate `pg-test` Postgres container (physical isolation, "different building") → Stage 3.0
- Refactoring raw-connection/DDL tests to route through `DatabaseManager` → Stage 3.0
- Repository-pattern refactor → Stage 3.1+
- Server-side privilege revocation (`swingrl_admin`, revoke TRUNCATE/DROP from app role) → Stage 3.5
- F1 turbulence-column bug → Stage 4 (optional ride-along only if trivial)

## The three-layer safety model (answers "how do we never wipe production?")

**Backbone — Postgres connection binding.** A connection is bound to exactly one database. A
connection to `swingrl_test` cannot see or `TRUNCATE` tables in `swingrl` (production) — they are
separate databases, even on the same server. So the only risk is the connection itself being
pointed at production. Two checks prevent that:

| Layer | Mechanism | File |
|---|---|---|
| **Check #1 — pre-suite guard** | `pytest_configure` refuses to start the suite unless the **resolved** DB name is `swingrl_test` or `*_test`. Hardened to resolve env **and** config fallback (closes the blank-`DATABASE_URL` hole where `DatabaseManager` falls back to `config.system.database_url`). | `tests/conftest.py:40-60` |
| **Check #2 — pre-wipe re-check** | The wipe fixture re-confirms the connection's database name ends in `_test` **immediately before every TRUNCATE**; raises and refuses otherwise. | new fixture |
| **Backbone — connection binding** | Postgres: a `*_test` connection physically cannot reach `swingrl`. | (DB engine) |

Stage 3 later adds the "different building" (`pg-test` server) + server-side revocation so it
becomes *architecturally* impossible. Stage 1's three layers make it *safe*.

## Design

### 1. Autouse wipe fixture (new)

- New file `tests/fixtures/db_cleanup.py`, registered in `tests/conftest.py`.
- `function`-scoped, `autouse=True`, runs in teardown (after each test).
- **No-op when no test DB is configured** (mirrors the existing skip behaviour) — only acts when a
  resolved `*_test` database is present.
- Enumerates user tables from the connected test database's catalog
  (`information_schema.tables` / `pg_tables`, `public` schema) and issues a single
  `TRUNCATE <all> RESTART IDENTITY CASCADE` — catalog-enumerate so new tables are auto-covered
  and FK ordering is handled by `CASCADE` in one statement.
- **Pre-wipe re-check (Check #2):** asserts the resolved DB name ends in `_test` before issuing
  the TRUNCATE; raises `RuntimeError` otherwise (never silently proceeds).
- Uses a **dedicated short-lived psycopg connection** (not the pooled `DatabaseManager`
  connection) to avoid pool-affinity surprises; closes it in teardown.

### 2. Guard hardening (Check #1)

- Add `_resolve_db_name()` helper shared by `pytest_configure` and the wipe fixture: resolves the
  target the same way `DatabaseManager` does (env `DATABASE_URL` → `config.system.database_url`),
  so the guard sees the *real* target, closing the blank-env fallback hole.
- `pytest_configure` uses it; behaviour otherwise unchanged (hard `pytest.exit` on non-test DB).

### 3. Remove the dead TRUNCATE blocks

- Delete the commented-out per-module TRUNCATE hacks (`tests/execution/conftest.py`,
  `tests/data/test_db.py`, `tests/data/test_parquet_to_duckdb.py`,
  `tests/data/test_ingestion_logging.py`) — superseded by the single autouse fixture (DRY).

### 4. Raw-connection / DDL tests — handled, not refactored

The wipe fixture truncates the test DB's committed data regardless of how a test wrote it, so
`test_phase15.py`, `test_stuck_agent.py`, etc. are fixed **without refactoring**. Their
`CREATE TABLE IF NOT EXISTS` DDL is idempotent and harmless to leave between tests; only the
*data* needs wiping. (Schema re-creation per test is wasteful but correct — optimizing to a
session-scoped schema init is a Stage 3.0 nicety.)

### 5. CI

- No change to the fresh-DB-per-run flow (`scripts/ci-homelab.sh:52-59` keeps creating/dropping a
  fresh `swingrl_test` on `pg16`). The wipe fixture handles **intra-run** isolation, which is the
  actual cause of the 50 failures. Separate server deferred.

## Acceptance criteria

1. `scripts/ci-homelab.sh` (the SAFE path — **not** ad-hoc `pytest` against `pg16`) passes with
   **0 failures**.
2. Unit test proves the guard rejects a non-`_test` resolved DB name (env and config-fallback
   cases).
3. Unit test proves the wipe fixture refuses to TRUNCATE when the resolved DB is not `*_test`.
4. No reference to the production database name (`swingrl`) anywhere in the test path.
5. The ~50 previously-failing tests pass; nothing previously passing regresses.

## TDD order

1. RED: guard rejects non-test DB (env + config fallback) → GREEN: `_resolve_db_name()` + guard.
2. RED: wipe fixture refuses non-`_test` DB → GREEN: pre-wipe re-check.
3. RED: representative pollution pair fails together, passes with fixture → GREEN: wipe fixture.
4. Full `ci-homelab.sh` → 0 failures.

## Risks & mitigations

- **TRUNCATE-all-per-test is slower.** Acceptable for Stage 1; measure. If too slow, scope to
  touched tables in Stage 3 (where the separate server + rollback land anyway).
- **FK ordering.** Single `TRUNCATE … CASCADE` statement handles it.
- **Parallel test execution (pytest-xdist).** Per-test TRUNCATE on a shared DB would collide.
  Stage 1 assumes **serial** execution (current default). Parallelism needs the separate
  `pg-test` server — noted for Stage 3.
- **A test that genuinely depends on cross-test accumulated state** would break. Investigation
  found tests assume *empty* state, so wiping aligns with their assumptions; if any exception
  surfaces, fix that test (it encodes the very bug we're removing).

## Open questions for the plan phase

1. Confirm catalog-enumerate vs `postgres_schema.py` table list (recommend catalog-enumerate).
2. Confirm the dedicated-connection approach for the fixture (recommend yes, avoids pool affinity).
3. Enumerate the exact currently-failing test list via one controlled `ci-homelab.sh` run, to use
   as the acceptance checklist.
