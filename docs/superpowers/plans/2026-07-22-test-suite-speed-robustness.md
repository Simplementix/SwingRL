# Test-Suite Speed + Robustness Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development
> (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use
> checkbox (`- [ ]`) syntax for tracking.

**Goal:** Cut the ~60-minute full test suite to ≤20 minutes and make scratch-DB corruption
(the 2026-07-22 `hmm_state_history` PK-loss incident) structurally impossible, without weakening
any of the three existing DB-safety guard layers.

**Architecture:** Five approved improvements land as ten tasks: (1) the autouse per-test
TRUNCATE becomes conditional on an auto-derived `db` marker and reuses one persistent cleanup
connection; (2) `--durations=25` measurement lands first so every later claim is calibrated;
(3) parallelization ships in two phases — pytest-xdist for the no-DB fast lane (Phase A), then
per-worker databases cloned from a pre-migrated template (Phase B); (4) the five destructive-DDL
test sites are converted to canonical-DDL/non-destructive form and a session-start
schema-integrity preflight hard-fails on a poisoned scratch DB; (5) a three-lane workflow
(fast / db / full) is documented with the targeted-preflight (`--lf`-first) rule.

**Tech Stack:** Python 3.11, pytest 8, pytest-xdist (new dev dep), psycopg 3, PostgreSQL 16
(homelab `pg16` container, no host port — reached via container IP), uv, ruff/mypy/pre-commit.

**Evidence base:** Three review reports (speed / robustness / practices, 2026-07-22) plus direct
reads of every file this plan modifies. Timing numbers marked *(target)* are unverified until
Task 1's calibration run converts them to measurements.

## Execution Order (USER RULING 2026-07-23 — overrides task numbering)

Quick wins first; no standalone calibration run. The 2026-07-22 session's four full-suite runs
(54:48 final, near-uniform ~1.8 s/test overhead fingerprint) are the accepted BEFORE baseline.

1. **Wave 1 (quick wins):** Task 2 (auto-derived `db` marker — dependency of the wipe) →
   Task 3 (conditional wipe + persistent connection) → Task 4 (fast-lane xdist, Phase A).
   Task 1 is REDUCED to its one-line `--durations=25` addopts change, folded into Wave 1's
   first commit — no dedicated baseline run.
2. **Wave 1 gate:** ONE full-suite run validates Wave 1 AND (via `--durations`) produces the
   measured hotspot list. Compare against the 54:48 baseline.
3. **Wave 2 (informed by the measurements):** Tasks 5-7 (destructive-DDL fixes + schema
   preflight), Task 8 (per-worker DBs, Phase B) — re-scope against the durations data first.
4. Task 9 (docs) and Task 10 (final verification) close out as written.

## Global Constraints

Carried verbatim from the approved scope — every task's requirements implicitly include these:

- **Branch:** `swingrl/2.R-F-test-infra` off `origin/swingrl/2.R-training-redesign`, created
  AFTER the in-flight rulings branch (`swingrl/2.R-E-rulings`) merges; PRs target the
  integration branch, never main.
- **Python 3.11; `from __future__ import annotations`; type hints on all defs; absolute imports
  in src/; 100-char lines; structlog kwargs only.**
- **TDD** where the change has testable I/O (fixture behavior IS testable — this plan tests that
  the wipe skips clean tests and fires for DB tests); **pre-commit always** (never
  `--no-verify`).
- **DB-backed verification uses a FRESH scratch DB per full run until item 4 lands** (standing
  precaution); pg16 has no host port — scratch DBs are reached via container IP; scratch DB
  names must end `_test`, owner `swingrl`.
- **No production `src/` behavior changes** — src/ edits are out of scope; NONE are expected.
  If a task seems to need one, STOP AND ASK (see "Stop-and-ask triggers" below).
- **Success criteria:** full suite ≤20 min after Tasks 1–4 (items 1+3A, calibrated by Task 1's
  measurement); fast lane <2 min; zero shared-DB corruption possible after Tasks 5–8 (3B+4).
- **Never weaken the DB-safety stack:** suite-refusal guard (`tests/conftest.py:35-56`),
  `_test`-suffix classifier (`tests/db_guard.py:50-69`), pre-wipe re-check
  (`tests/fixtures/db_cleanup.py:29-46`). Every task routes *around* it, never through it.

### Stop-and-ask triggers

- Any change that would touch a file under `src/swingrl/` (this plan only *imports* read-only
  DDL constants from `src/swingrl/data/postgres_schema.py` — importing is fine, editing is not).
- Any change to the three guard layers beyond what is written verbatim in a task here.
- The Task 8 worker-DB naming deviation (`swingrl_gw0_test` instead of the scoping note's
  literal `swingrl_test_gw0`) is pre-decided in Task 8's rationale; if a reviewer wants the
  literal spelling instead, that requires touching the guard — stop and ask.

### Glossary (house rule: no undefined shorthand)

| Term | Meaning here |
|---|---|
| **Lane** | A named subset of the suite with its own command + budget: fast (no DB), db (targeted, real scratch DB), full (everything, pre-push). |
| **`db` marker** | A pytest marker meaning "this test can reach the real PostgreSQL scratch DB". Auto-derived at collection time — never hand-edited onto the 51 existing files. |
| **Autouse fixture** | A pytest fixture that runs around *every* test without being requested. |
| **Wipe** | `wipe_db_after_test` — the autouse teardown that TRUNCATEs all non-registry tables. |
| **TRUNCATE** | Postgres statement that empties a table (fast row deletion, keeps the table). |
| **DDL** | Data Definition Language — `CREATE/DROP/ALTER TABLE` statements (shape, not rows). |
| **PK** | PRIMARY KEY constraint. Its loss on `hmm_state_history` caused the 2026-07-22 incident. |
| **Canonical DDL** | The production table definitions in `src/swingrl/data/postgres_schema.py`. |
| **xdist / worker** | pytest-xdist runs tests in N parallel worker processes (`gw0`…`gwN`). |
| **Template DB** | A pre-migrated Postgres database (`swingrl_test_template`) that worker DBs are cloned from via `CREATE DATABASE … TEMPLATE …` (file-level copy, ~100-300 ms). |
| **Ledger** | The `schema_migrations` table — records which migration versions were applied. |
| **Preflight** | A session-start check in `pytest_configure` that refuses to run against a bad DB. |
| **`--lf`** | pytest `--last-failed`: rerun only the tests that failed last run (cache already works in this repo — nothing disables the cacheprovider). |

### Scratch-DB preparation (referenced by Tasks 1, 5, 6, 7, 10)

Standing precaution: every DB-backed verification run in this plan uses a FRESH scratch DB.
On homelab, from the repo root (needs the `DATABASE_URL` password from `~/swingrl/.env` — ask
the user if you cannot read it):

```bash
PG_IP=$(docker inspect -f '{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}' pg16)
PG_PASS=$(grep DATABASE_URL ~/swingrl/.env | sed 's/.*:\/\/swingrl:\([^@]*\)@.*/\1/')
docker exec pg16 psql -U temporal -d postgres -c "DROP DATABASE IF EXISTS swingrl_plan_test;"
docker exec pg16 psql -U temporal -d postgres -c "CREATE DATABASE swingrl_plan_test OWNER swingrl;"
export DATABASE_URL="postgresql://swingrl:${PG_PASS}@${PG_IP}:5432/swingrl_plan_test"
uv run python -c "
from swingrl.config.schema import load_config
from swingrl.data.db import DatabaseManager
from swingrl.data.migration_runner import apply_migrations, current_schema_version
db = DatabaseManager(load_config('config/swingrl.yaml'))
db.init_schema()
n = apply_migrations(db)
print(f'applied={n} current_version={current_schema_version(db)}')
"
```

Expected final line: `applied=11 current_version=11` (fresh DB; `EXPECTED_SCHEMA_VERSION = 11`
per `src/swingrl/data/migration_runner.py:34`). This mirrors CI stage 2.7
(`scripts/ci-homelab.sh:66-91`) so the migrated-DB invariant every fixture assumes holds.

### Calibration Record (filled by Task 1 step 5 and Task 10)

| Metric | Baseline (Task 1) | After Tasks 1–4 (Task 10) | Target |
|---|---|---|---|
| Full suite wall time | 54:48 (accepted 2026-07-22 baseline, user ruling — no standalone run) | **17:36** (`-n 4`, per-worker clones, fresh `swingrl_final_test`, 2026-07-23: 1964 passed / 0 failed / 7 skipped) — intermediate: 30:03 serial at the Wave 1 gate | ≤ 20 min ✅ |
| Fast lane wall time | n/a (lane doesn't exist yet) | 35–51 s (`-n auto`, 996 selected, 995 passed / 1 skipped) | < 2 min ✅ |
| Top-25 durations | n/a (no standalone baseline run, user ruling) | See "Wave 1 gate durations" note below | — |
| Tests marked `db` | n/a | 961 of 1955 collected (994 fast lane) | — |

**Final `-n 4` durations (2026-07-23, Task 10):** slowest entries are `test_cleanup_connection_is_reused_across_wipes` (8.4 s call — it deliberately double-truncates), `test_pipeline` feature writes (7.0 s), and `tests/data/test_migrations_content.py` teardowns at 5.6–6.6 s (TRUNCATE under 4-way parallel I/O contention — higher per-test than serial, but 4 run at once). Second-tier costs match the reviews' forecast (`db_with_legacy_schema` re-migration, seeded pipeline writes); parallelism is the remedy already applied. No follow-up plan needed — both success criteria met.

**Wave 1 gate durations (2026-07-23):** top of the slowest-25 block is dominated by ~2.9–3.4 s
**teardown** entries on db-lane tests (`tests/data/test_migrations_content.py`,
`tests/data/test_views.py` — the per-test TRUNCATE itself), with the slowest calls at 4.5 s
(`test_seed_pinning`, `test_trainer_memory_wiring`, `test_db_cleanup_conn`). Total user CPU was
7:39 of 30:03 wall — DB wait dominates. Conclusion: Task 8 (per-worker DBs, `-n 4`) is the
remaining lever to reach ≤20 min; Tasks 5–7 are unaffected (robustness, not speed).

### Execution prerequisites

- [ ] Confirm `swingrl/2.R-E-rulings` has merged into the integration branch (ask the user if
      unsure — do NOT start before it merges).
- [ ] Create the branch:

```bash
git fetch origin
git checkout -b swingrl/2.R-F-test-infra origin/swingrl/2.R-training-redesign
```

---

### Task 1: Measurement first — `--durations=25` + baseline calibration run

Every speed estimate in the reviews is derived, not measured (uniform-rate inference from log
timestamps). This task makes the very next run produce facts, before any optimization lands.

**Files:**
- Modify: `pyproject.toml:106-116` (the `[tool.pytest.ini_options]` section — currently has NO
  `addopts`)
- Modify: `docs/superpowers/plans/2026-07-22-test-suite-speed-robustness.md` (this file —
  Calibration Record table)

**Interfaces:**
- Consumes: nothing.
- Produces: `addopts = "--durations=25 --durations-min=1.0"` in pytest config (Task 3 later
  extends this exact line with `-p pytester`); measured baseline numbers in the Calibration
  Record that Task 10 compares against.

- [ ] **Step 1: Add addopts to pyproject.toml**

The section currently reads (lines 106-117):

```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
pythonpath = ["src", "."]
python_files = ["test_*.py"]
python_classes = ["Test*"]
python_functions = ["test_*"]
markers = [
    "integration: requires real API keys and network access (skipped in CI)",
    "slow: long-running determinism/reproducibility tests (included in CI, no network)",
]
timeout = 600
```

Insert one line after `python_functions = ["test_*"]`:

```toml
addopts = "--durations=25 --durations-min=1.0"
```

- [ ] **Step 2: Sanity-run a small package to confirm the durations report appears**

Run: `env -u DATABASE_URL uv run pytest tests/config -q`
Expected: all tests pass, and the tail of the output contains a
`== slowest 25 durations ==` section (it may say
`(24 durations < 1s hidden. Use -vv to show these durations.)` — that is the
`--durations-min=1.0` filter working).

- [ ] **Step 3: Commit the config change**

```bash
git add pyproject.toml
git commit -m "chore(test): pytest addopts --durations=25 — measure before optimizing"
```

- [ ] **Step 4: Baseline calibration run (full suite, fresh scratch DB, background)**

Prepare a fresh scratch DB per the header block ("Scratch-DB preparation"), then launch the
full suite **in the background, harness-tracked** (house rule: no-poll long jobs — announce
once, let completion notify):

```bash
uv run pytest tests/ -q 2>&1 | tee /tmp/claude-1000/-home-varun-Projects-Simplementix-SwingRL/9f5ffdda-0924-42df-b90d-26302a2745f0/scratchpad/baseline-durations.log
```

Expected: ~1,949 collected, ~55-65 minutes wall time (the 2026-07-22 reference run was 59:26
for 1,934 pass / 8 fail / 7 skip — the 8 fails were the rulings-branch `ON CONFLICT` regression
and should be gone on this branch; expect 0 unrelated failures). The log tail contains the
`slowest 25 durations` block.

- [ ] **Step 5: Record the baseline in this plan's Calibration Record**

Copy the total wall time and the full `slowest 25 durations` block into the Calibration Record
table above (Baseline column). Sanity-check the reviews' central claim while you are there: if
the wipe tax is real, the *median* test should show no entry ≥1 s but the run total should be
≈1.8 s/test.

- [ ] **Step 6: Commit the calibration record**

```bash
git add docs/superpowers/plans/2026-07-22-test-suite-speed-robustness.md
git commit -m "docs(test): record baseline suite timing (durations=25 calibration)"
```

---

### Task 2: Auto-derived `db` marker (one selectable signal for four skip spellings)

DB gating is spelled four different ways across 51 files (module `pytestmark` skipif, per-test
skipif, 43 inline `pytest.skip` calls, fixture-level skips) and none is selectable with `-m`.
This task derives one `db` marker at collection time — **zero edits to the 51 files**.

Derivation rule (deliberately coarse-but-safe): a test is `db` if (a) it requests a known
real-DB fixture, OR (b) its module's source mentions `DATABASE_URL` anywhere. Over-marking a
mock-only test that lives in a DB-mentioning module (e.g. `tests/monitoring/test_alerter.py`)
only costs it a place in the fast lane; under-marking is impossible without hardcoding a DB URL
in a test, which the no-hardcoded-values rule already forbids. Explicit `@pytest.mark.db` is
always honored as a force-override.

**Files:**
- Create: `tests/db_marker.py`
- Create: `tests/test_db_marker_derivation.py`
- Modify: `tests/conftest.py` (add `pytest_collection_modifyitems` after `pytest_configure`,
  plus one import)
- Modify: `pyproject.toml:112-115` (register the `db` marker)

**Interfaces:**
- Consumes: nothing new.
- Produces (used by Tasks 3, 4, 9):
  - `tests/db_marker.py::DB_FIXTURE_NAMES: frozenset[str]`
  - `tests/db_marker.py::is_db_test(fixturenames: Iterable[str], module_source: str) -> bool`
  - `tests/db_marker.py::module_mentions_database_url(path: str) -> bool` (lru_cached per path)
  - Collection-time behavior: every DB-capable test item carries `pytest.mark.db`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_db_marker_derivation.py`:

```python
"""Auto-derived ``db`` marker: pure derivation-logic tests (fast lane, no DB).

The four historical DATABASE_URL gating spellings across 51 test files (module
pytestmark, per-test skipif, inline pytest.skip, fixture-level skip) collapse to
one collection-time signal: the test requests a known real-DB fixture, or its
module source mentions DATABASE_URL. See tests/db_marker.py for the rationale.
"""

from __future__ import annotations

from pathlib import Path

from tests.db_marker import DB_FIXTURE_NAMES, is_db_test, module_mentions_database_url


def test_real_db_fixture_triggers_db() -> None:
    """A test requesting a known real-DB fixture is a db test."""
    assert is_db_test(["tmp_path", "pg_conn"], "") is True


def test_module_mention_triggers_db() -> None:
    """A module that reads DATABASE_URL anywhere makes all its tests db tests."""
    source = 'db_url = os.environ.get("DATABASE_URL", "")'
    assert is_db_test(["tmp_path"], source) is True


def test_plain_test_is_not_db() -> None:
    """No DB fixture + no DATABASE_URL mention -> fast lane."""
    assert is_db_test(["tmp_path", "loaded_config"], "import pandas as pd") is False


def test_known_real_db_fixtures_registered() -> None:
    """The audited real-DB fixture names are all present (practices review §0)."""
    expected = {
        "mock_db",
        "pg_conn",
        "seeded_duckdb",
        "db_with_legacy_schema",
        "db_config_yaml",
        "db",
        "db_manager",
        "memory_db_env",
        "api_client",
    }
    assert expected <= DB_FIXTURE_NAMES


def test_module_mentions_database_url_reads_file(tmp_path: Path) -> None:
    """File-content check is exact and cached per path."""
    mentions = tmp_path / "test_mentions.py"
    mentions.write_text('URL = os.environ["DATABASE_URL"]\n')
    clean = tmp_path / "test_clean.py"
    clean.write_text("def test_ok():\n    assert True\n")
    assert module_mentions_database_url(str(mentions)) is True
    assert module_mentions_database_url(str(clean)) is False


def test_module_mentions_database_url_missing_file_is_false() -> None:
    """Unreadable path never crashes collection."""
    assert module_mentions_database_url("/nonexistent/never/test_x.py") is False
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `env -u DATABASE_URL uv run pytest tests/test_db_marker_derivation.py -q`
Expected: collection ERROR — `ModuleNotFoundError: No module named 'tests.db_marker'`.

- [ ] **Step 3: Implement `tests/db_marker.py`**

```python
"""Auto-derivation of the ``db`` marker at collection time.

A test is ``db`` (touches the real PostgreSQL scratch database) when it requests
one of the known real-DB fixtures below, or when its module's source mentions
``DATABASE_URL`` anywhere. This converts the four existing skip spellings
(module pytestmark / per-test skipif / inline pytest.skip / fixture-level skip,
51 files total) into ONE selectable marker without editing any of those files.

Deliberately coarse: over-marking a mock-only test inside a DATABASE_URL-mentioning
module only excludes it from the fast lane; under-marking is impossible without
hardcoding a DB URL in a test, which the no-hardcoded-values rule forbids.
An explicit ``@pytest.mark.db`` always wins (conftest checks it first).
"""

from __future__ import annotations

from collections.abc import Iterable
from functools import lru_cache
from pathlib import Path

# Fixtures that hand tests a real PostgreSQL connection/manager (audited 2026-07-22).
# NOTE the two naming traps flagged by the practices review: ``mock_db`` (execution
# conftest) and ``seeded_duckdb`` (features test_pipeline) are BOTH real Postgres.
DB_FIXTURE_NAMES: frozenset[str] = frozenset(
    {
        "mock_db",  # tests/execution/conftest.py:73 — real DatabaseManager
        "pg_conn",  # tests/features/conftest.py:103
        "seeded_duckdb",  # tests/features/test_pipeline.py:104 — real Postgres
        "db_with_legacy_schema",  # tests/data/conftest.py:64
        "db_config_yaml",  # tests/data/conftest.py:25 — embeds DATABASE_URL
        "db",  # tests/data/test_migration_runner.py:28
        "db_manager",  # tests/data/test_db.py:78
        "memory_db_env",  # tests/test_memory_service.py:81
        "api_client",  # tests/test_memory_service.py:115 — real pool via init_db()
    }
)


def is_db_test(fixturenames: Iterable[str], module_source: str) -> bool:
    """Return True when a test can reach the real database (see module docstring)."""
    if DB_FIXTURE_NAMES.intersection(fixturenames):
        return True
    return "DATABASE_URL" in module_source


@lru_cache(maxsize=None)
def module_mentions_database_url(path: str) -> bool:
    """Whether the module file at ``path`` mentions DATABASE_URL (cached per path)."""
    try:
        return "DATABASE_URL" in Path(path).read_text(encoding="utf-8")
    except OSError:
        return False
```

- [ ] **Step 4: Wire the collection hook into `tests/conftest.py`**

Add to the imports block (after the existing
`from tests.db_guard import SAFE_DB_NAMES, classify_db_url, resolve_target_db_url` line):

```python
from tests.db_marker import DB_FIXTURE_NAMES, module_mentions_database_url
```

Add this hook directly after the `pytest_configure` function (after line 56):

```python
def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Auto-stamp the ``db`` marker on every DB-capable test (tests/db_marker.py).

    Explicit ``@pytest.mark.db`` is honored as-is (checked first). Everything the
    conditional wipe and the fast/db lanes do keys off this marker.
    """
    for item in items:
        if item.get_closest_marker("db") is not None:
            continue
        fixturenames: list[str] = getattr(item, "fixturenames", [])
        if DB_FIXTURE_NAMES.intersection(fixturenames) or module_mentions_database_url(
            str(item.path)
        ):
            item.add_marker(pytest.mark.db)
```

- [ ] **Step 5: Register the marker in `pyproject.toml`**

Change the markers list (lines 112-115) to:

```toml
markers = [
    "integration: requires real API keys and network access (skipped in CI)",
    "slow: long-running determinism/reproducibility tests (included in CI, no network)",
    "db: touches the real PostgreSQL test database (auto-derived at collection; explicit mark forces)",
]
```

- [ ] **Step 6: Run the new tests + verify derivation on real modules**

Run: `env -u DATABASE_URL uv run pytest tests/test_db_marker_derivation.py -q`
Expected: `6 passed`.

Run: `env -u DATABASE_URL uv run pytest tests/features/test_pipeline.py --collect-only -q -m db | tail -1`
Expected: every collected test selected, `0 deselected` (the module has a DATABASE_URL
pytestmark at line 25 — all its tests must be `db`).

Run: `env -u DATABASE_URL uv run pytest tests/config --collect-only -q -m "not db" | tail -1`
Expected: all 21 tests selected, 0 deselected (`tests/config` never mentions DATABASE_URL).

Run: `env -u DATABASE_URL uv run pytest tests/ --collect-only -q -m db | tail -1`
Expected: several hundred selected (record the exact number in the Calibration Record — this is
the db-lane population), remainder deselected, `0 errors`.

- [ ] **Step 7: Commit**

```bash
git add tests/db_marker.py tests/test_db_marker_derivation.py tests/conftest.py pyproject.toml
git commit -m "feat(test): auto-derived db marker — one selectable signal for 4 skip spellings"
```

---

### Task 3: Conditional wipe + persistent cleanup connection

The autouse wipe (`tests/fixtures/db_cleanup.py:230-245`) opens a brand-new psycopg connection
and TRUNCATEs ~33 tables after **every** test — including the ~75-80% that never touch the DB.
Estimated ~42 of the 59 minutes (speed review §1.2). After this task the wipe (a) only fires
for `db`-marked tests and (b) reuses one persistent autocommit connection per process. Wipe
safety semantics for DB tests are unchanged: same three guard layers, same registry-table
exclusions, same `RESTART IDENTITY CASCADE`.

**Files:**
- Modify: `tests/fixtures/db_cleanup.py:70-92` (persistent connection) and `:230-245` (marker
  gate)
- Modify: `tests/db_guard.py:33-47` (memoize the YAML fallback — a Pydantic parse currently
  runs up to 1,905× per run)
- Modify: `pyproject.toml` addopts (enable the `pytester` plugin for the meta-tests)
- Create: `tests/test_wipe_conditionality.py` (fast-lane meta-tests via pytester)
- Create: `tests/test_db_cleanup_conn.py` (db-lane persistent-connection tests)
- Modify: `tests/test_db_cleanup_guard.py` (add the YAML-memoization test)

**Interfaces:**
- Consumes: the `db` marker from Task 2.
- Produces (used by Tasks 8, 10):
  - `tests/fixtures/db_cleanup.py::_get_cleanup_conn(db_url: str) -> psycopg.Connection[Any]`
    — persistent, autocommit, re-keys automatically when `db_url` changes (Task 8's worker
    rewrite relies on this).
  - `tests/fixtures/db_cleanup.py::_run_truncate(conn: psycopg.Connection[Any]) -> None`
  - `_truncate_all_public_tables(db_url: str) -> None` keeps its existing signature (external
    callers unaffected).
  - `wipe_db_after_test` gains a `request: pytest.FixtureRequest` parameter (autouse fixtures
    may take fixtures — no caller changes needed).
  - `tests/db_guard.py::_yaml_fallback_url() -> str` (lru_cached, `cache_clear()` available).

- [ ] **Step 1: Enable the pytester plugin via addopts**

In `pyproject.toml`, change the Task 1 line to:

```toml
addopts = "-p pytester --durations=25 --durations-min=1.0"
```

(`pytester` is pytest's built-in plugin-testing fixture; it is not enabled by default. A
`pytest_plugins` declaration in `tests/conftest.py` is NOT an option — that file is not the
rootdir conftest, and non-root `pytest_plugins` is a pytest error.)

- [ ] **Step 2: Write the failing conditional-wipe meta-tests**

Create `tests/test_wipe_conditionality.py`:

```python
"""Conditional-wipe semantics: the autouse wipe fires ONLY for db-marked tests.

Runs a miniature pytest suite in-process (pytester) using the REAL wipe fixture
and the REAL marker-derivation logic, with the TRUNCATE function stubbed to a
recorder — no PostgreSQL needed, so these meta-tests live in the fast lane.

(The literal string DATABASE_URL below auto-marks THIS module ``db`` too; that
is harmless — in the fast lane the wipe no-ops on a blank URL, in the db lane it
truncates an already-clean scratch DB.)
"""

from __future__ import annotations

import pytest

from tests.fixtures import db_cleanup

_FAKE_TEST_URL = "postgresql://u:pw@host:5432/swingrl_test"  # pragma: allowlist secret

_INNER_CONFTEST = """
from __future__ import annotations

import pytest

from tests.db_marker import DB_FIXTURE_NAMES, module_mentions_database_url
from tests.fixtures.db_cleanup import wipe_db_after_test  # noqa: F401  (autouse)


def pytest_collection_modifyitems(config, items):
    for item in items:
        if item.get_closest_marker("db") is None and (
            DB_FIXTURE_NAMES.intersection(getattr(item, "fixturenames", []))
            or module_mentions_database_url(str(item.path))
        ):
            item.add_marker(pytest.mark.db)
"""


def test_wipe_fires_only_for_db_marked_tests(
    pytester: pytest.Pytester, monkeypatch: pytest.MonkeyPatch
) -> None:
    """One explicitly db-marked test + one plain test -> exactly one TRUNCATE."""
    calls: list[str] = []
    monkeypatch.setattr(db_cleanup, "_truncate_all_public_tables", calls.append)
    monkeypatch.setenv("DATABASE_URL", _FAKE_TEST_URL)
    pytester.makeconftest(_INNER_CONFTEST)
    pytester.makepyfile(
        test_plain="def test_no_db():\n    assert 1 + 1 == 2\n",
        test_marked=(
            "import pytest\n\n"
            "@pytest.mark.db\n"
            "def test_db_marked():\n"
            "    assert True\n"
        ),
    )
    result = pytester.runpytest_inprocess("-p", "no:cacheprovider", "-q")
    result.assert_outcomes(passed=2)
    assert calls == [_FAKE_TEST_URL]


def test_wipe_fires_for_auto_derived_db_module(
    pytester: pytest.Pytester, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A module mentioning DATABASE_URL is auto-marked -> its test gets wiped."""
    calls: list[str] = []
    monkeypatch.setattr(db_cleanup, "_truncate_all_public_tables", calls.append)
    monkeypatch.setenv("DATABASE_URL", _FAKE_TEST_URL)
    pytester.makeconftest(_INNER_CONFTEST)
    pytester.makepyfile(
        test_mentions=(
            "import os\n\n"
            "def test_reads_env():\n"
            '    assert os.environ.get("DATABASE_URL")\n'
        ),
    )
    result = pytester.runpytest_inprocess("-p", "no:cacheprovider", "-q")
    result.assert_outcomes(passed=1)
    assert calls == [_FAKE_TEST_URL]
```

- [ ] **Step 3: Run the meta-tests to verify they fail**

Run: `env -u DATABASE_URL uv run pytest tests/test_wipe_conditionality.py -q`
Expected: `2 failed` — the FIRST test fails on `assert calls == [_FAKE_TEST_URL]` with TWO
recorded calls (current wipe fires unconditionally, so the plain test also wipes). If instead
you see a `fixture 'pytester' not found` error, Step 1's addopts change did not land — fix that
first.

- [ ] **Step 4: Write the failing persistent-connection tests (db lane)**

Create `tests/test_db_cleanup_conn.py`:

```python
"""Persistent cleanup connection: reuse across wipes + transparent reconnect.

db-lane tests — gated on DATABASE_URL like every real-DB test.
"""

from __future__ import annotations

import os

import pytest

from tests.fixtures import db_cleanup

pytestmark = pytest.mark.skipif(
    not os.environ.get("DATABASE_URL"), reason="DATABASE_URL not set"
)


def test_cleanup_connection_is_reused_across_wipes() -> None:
    """Two consecutive wipes use the SAME connection object (no per-test connect)."""
    db_url = os.environ["DATABASE_URL"]
    first = db_cleanup._get_cleanup_conn(db_url)
    db_cleanup._truncate_all_public_tables(db_url)
    db_cleanup._truncate_all_public_tables(db_url)
    second = db_cleanup._get_cleanup_conn(db_url)
    assert first is second
    assert not first.closed


def test_cleanup_connection_reopens_after_close() -> None:
    """A dropped connection (server restart) is reopened transparently."""
    db_url = os.environ["DATABASE_URL"]
    db_cleanup._get_cleanup_conn(db_url).close()  # simulate a dropped connection
    db_cleanup._truncate_all_public_tables(db_url)  # must not raise
    assert not db_cleanup._get_cleanup_conn(db_url).closed
```

Run: `uv run pytest tests/test_db_cleanup_conn.py -q` (with `DATABASE_URL` exported per the
scratch-DB header block)
Expected: `2 errors/failures` — `AttributeError: module ... has no attribute '_get_cleanup_conn'`.

- [ ] **Step 5: Write the failing YAML-memoization test**

Append to `tests/test_db_cleanup_guard.py`:

```python
def test_yaml_fallback_parsed_at_most_once(monkeypatch: pytest.MonkeyPatch) -> None:
    """REQ-SPEED: with DATABASE_URL unset, the YAML fallback is parsed once, not per call."""
    from tests import db_guard

    calls: list[int] = []
    real_load = db_guard.load_config

    def counting_load(path):  # type: ignore[no-untyped-def]
        calls.append(1)
        return real_load(path)

    db_guard._yaml_fallback_url.cache_clear()
    monkeypatch.setattr(db_guard, "load_config", counting_load)
    monkeypatch.delenv("DATABASE_URL", raising=False)
    db_guard.resolve_target_db_url()
    db_guard.resolve_target_db_url()
    assert len(calls) == 1
    db_guard._yaml_fallback_url.cache_clear()  # do not leak a counting-fn cache entry
```

Run: `env -u DATABASE_URL uv run pytest tests/test_db_cleanup_guard.py -q`
Expected: 4 old tests pass, the new one fails —
`AttributeError: module ... has no attribute '_yaml_fallback_url'`.

- [ ] **Step 6: Implement the persistent connection in `tests/fixtures/db_cleanup.py`**

Add `from typing import Any` to the stdlib imports (after
`from collections.abc import Generator`). Then replace `_truncate_all_public_tables`
(lines 70-92) with:

```python
# Session-persistent cleanup connection. Opened lazily on the first db-marked
# test's wipe, reused for every later wipe in this process, re-keyed when the
# resolved URL changes (per-worker DB rewrite), reopened once on OperationalError
# (server restart mid-session). Replaces a per-test psycopg.connect that charged
# every test a TCP+auth round trip (speed review 2026-07-22 §1.2).
_cleanup_conn: psycopg.Connection[Any] | None = None
_cleanup_conn_url: str = ""


def _get_cleanup_conn(db_url: str) -> psycopg.Connection[Any]:
    """Return the persistent autocommit cleanup connection for ``db_url``."""
    global _cleanup_conn, _cleanup_conn_url
    if (
        _cleanup_conn is not None
        and not _cleanup_conn.closed
        and _cleanup_conn_url == db_url
    ):
        return _cleanup_conn
    if _cleanup_conn is not None and not _cleanup_conn.closed:
        _cleanup_conn.close()
    _cleanup_conn = psycopg.connect(db_url, autocommit=True)
    # Bounded lock_timeout: a stray lock from a misbehaving test fails fast and
    # loudly (naming the offending test) rather than hanging CI.
    _cleanup_conn.execute("SET lock_timeout = '10s'")
    _cleanup_conn_url = db_url
    return _cleanup_conn


def _run_truncate(conn: psycopg.Connection[Any]) -> None:
    """TRUNCATE every non-registry public table on the given connection.

    Catalog-enumerates tables so ad-hoc tables created by raw tests are covered;
    a single ``TRUNCATE … CASCADE`` handles FK ordering. Migration-managed
    registry tables (``_MIGRATION_REGISTRY_TABLES``) are excluded -- see that
    constant's docstring for why.
    """
    rows = conn.execute(
        "SELECT tablename FROM pg_tables WHERE schemaname = 'public'"
    ).fetchall()
    tables = [row[0] for row in rows if row[0] not in _MIGRATION_REGISTRY_TABLES]
    if not tables:
        return
    statement = sql.SQL("TRUNCATE TABLE {} RESTART IDENTITY CASCADE").format(
        sql.SQL(", ").join(sql.Identifier(name) for name in tables)
    )
    conn.execute(statement)


def _truncate_all_public_tables(db_url: str) -> None:
    """TRUNCATE all non-registry tables via the persistent cleanup connection.

    One transparent reconnect+retry on OperationalError covers a Postgres
    restart mid-session; any second failure propagates loudly.
    """
    global _cleanup_conn
    try:
        _run_truncate(_get_cleanup_conn(db_url))
    except psycopg.OperationalError:
        if _cleanup_conn is not None and not _cleanup_conn.closed:
            _cleanup_conn.close()
        _cleanup_conn = None
        _run_truncate(_get_cleanup_conn(db_url))
```

- [ ] **Step 7: Make the wipe conditional on the `db` marker**

Replace `wipe_db_after_test` (previously lines 230-245) with:

```python
@pytest.fixture(autouse=True)
def wipe_db_after_test(request: pytest.FixtureRequest) -> Generator[None, None, None]:
    """Wipe test-DB tables after each ``db``-marked test (no-op otherwise).

    The ``db`` marker is auto-derived at collection time (tests/db_marker.py):
    any test that can reach the real database carries it, so skipping the wipe
    for unmarked tests never skips a wipe that mattered. Pre-2026-07-22 this
    fixture connect+TRUNCATEd after ALL ~1,900 tests including the ~75-80% that
    are mock-only — the dominant share of the 59-minute wall time.
    """
    yield
    if request.node.get_closest_marker("db") is None:
        return
    db_url = resolve_target_db_url()
    if not ensure_wipe_target_is_test_db(db_url):
        return
    # Close the pool so idle connections release their locks before TRUNCATE.
    # NOTE: reset() closes only idle/returned connections — a connection still
    # checked out when teardown runs is NOT force-closed. That can only happen if
    # a test leaks a checked-out connection past its own body (a bug); the
    # lock_timeout on the cleanup connection bounds such a case to a loud ~10s
    # failure that names the offending test, rather than a silent hang.
    # A proper drain/rollback fixture is deferred to Stage 3.0 (see the spec).
    DatabaseManager.reset()
    _truncate_all_public_tables(db_url)
```

- [ ] **Step 8: Memoize the YAML fallback in `tests/db_guard.py`**

Add `from functools import lru_cache` to the stdlib imports. Replace
`resolve_target_db_url` (lines 33-47) with:

```python
@lru_cache(maxsize=1)
def _yaml_fallback_url() -> str:
    """``config.system.database_url`` from the default YAML, parsed once per process.

    The YAML file is static within a test session; re-parsing it per wipe charged
    up to ~1,900 Pydantic loads per DATABASE_URL-less run (practices review §0).
    """
    try:
        config = load_config(_DEFAULT_CONFIG_PATH)
    except Exception:  # noqa: BLE001 — guard must never crash the whole suite
        return ""
    return (config.system.database_url or "").strip()


def resolve_target_db_url() -> str:
    """Resolve the DB URL the same way ``DatabaseManager.__init__`` does.

    Order: ``DATABASE_URL`` env var (read fresh every call — tests monkeypatch
    it), then the memoized ``config.system.database_url`` from the default YAML.
    Returns ``""`` when neither is set (no DB -> tests skip). Whitespace-only
    DATABASE_URL is normalised to blank (treated as unset).
    """
    env_url = os.environ.get("DATABASE_URL", "").strip()
    if env_url:
        return env_url
    return _yaml_fallback_url()
```

- [ ] **Step 9: Run all Task 3 tests to verify they pass**

Run: `env -u DATABASE_URL uv run pytest tests/test_wipe_conditionality.py tests/test_db_cleanup_guard.py tests/test_db_marker_derivation.py -q`
Expected: all pass (2 + 5 + 6 = `13 passed`).

Run (with `DATABASE_URL` exported per scratch-DB header):
`uv run pytest tests/test_db_cleanup_conn.py tests/test_db_isolation.py -q`
Expected: `4 passed` — the two isolation probes still each see exactly one row, proving the
wipe still fires between consecutive db-marked tests.

- [ ] **Step 10: Regression sanity on a mixed package (db lane)**

Run (with `DATABASE_URL` exported): `uv run pytest tests/execution -q`
Expected: same pass/skip counts as before this task (execution tests are db-marked via the
`mock_db` fixture and their wipes still run); noticeably faster wall time.

- [ ] **Step 11: Commit**

```bash
git add tests/fixtures/db_cleanup.py tests/db_guard.py tests/test_wipe_conditionality.py \
        tests/test_db_cleanup_conn.py tests/test_db_cleanup_guard.py pyproject.toml
git commit -m "feat(test): conditional wipe (db marker) + persistent cleanup connection"
```

---

### Task 4: Parallelization Phase A — pytest-xdist for the no-DB fast lane

Pure-mock tests share no DB, so parallelism is safe there with zero isolation work. The db/full
lanes stay **serial** until Task 8 delivers per-worker databases — a single shared scratch DB
with TRUNCATE-based isolation is incompatible with parallel workers.

**Files:**
- Modify: `pyproject.toml` (`[dependency-groups]` dev list, after `"pytest-timeout>=2.1.0",`)
- Modify: `uv.lock` (regenerated by uv)

**Interfaces:**
- Consumes: the `db` marker (Task 2) to define the lane; conditional wipe (Task 3) so the lane
  pays zero DB tax.
- Produces: the fast-lane command Tasks 9-10 document and use:
  `env -u DATABASE_URL uv run pytest tests/ -m "not db and not slow and not integration" -n auto -q`

- [ ] **Step 1: Add pytest-xdist to dev dependencies**

In `pyproject.toml` `[dependency-groups]` dev list, after `"pytest-timeout>=2.1.0",` add:

```toml
    "pytest-xdist>=3.5",
```

- [ ] **Step 2: Sync the environment**

Run: `uv lock && uv sync`
Expected: lockfile updated; `pytest-xdist` installed without dependency conflicts.

- [ ] **Step 3: Verify the fast lane runs green in parallel**

Run:

```bash
time env -u DATABASE_URL uv run pytest tests/ -m "not db and not slow and not integration" -n auto -q
```

Expected: ~1,100-1,400 tests selected (exact count depends on Task 2's derivation — record it),
`0 failed`, `0 errors`; wall time **< 2 min** *(target — record actual in the Calibration
Record)*. `DATABASE_URL` is unset, so the suite guard resolves "blank", db tests were already
deselected by `-m`, and no worker ever touches Postgres.

If any test fails ONLY under `-n auto` (order/parallel dependence), rerun it serially to
confirm, then fix the test's isolation in this task before proceeding — do not exclude it with
markers to force green.

- [ ] **Step 4: Verify serial invocation still works (no xdist regression)**

Run: `env -u DATABASE_URL uv run pytest tests/config tests/utils -q`
Expected: all pass, no xdist involvement (no `-n` flag given → plain serial run).

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml uv.lock
git commit -m "feat(test): pytest-xdist — parallel no-DB fast lane (Phase A)"
```

---

### Task 5: Robustness — `hmm_state_history` canonical DDL in test_meta_orchestrator (both blocks)

`tests/memory/test_meta_orchestrator.py` DROP+hand-recreates `hmm_state_history` WITHOUT its
`PRIMARY KEY (environment, date)` and without teardown, in two duplicate blocks
(`_create_hmm_db`, lines 171-195, and inline in `test_current_regime_vector_handles_null_columns`,
lines 225-246). Because every later schema init is `CREATE TABLE IF NOT EXISTS`, the PK-less
table persists forever — this is the exact mechanism of the 2026-07-22 incident (production
upsert `ON CONFLICT (environment, date)` at `src/swingrl/features/hmm_regime.py:320` then fails
suites later). Fix: never DROP; ensure the canonical table exists and insert by column name.

**Files:**
- Modify: `tests/memory/test_meta_orchestrator.py:171-195` and `:225-246`, plus one import
- Test: same file (new regression test in `TestMetaOrchestratorRegimeVector`)

**Interfaces:**
- Consumes: `src/swingrl/data/postgres_schema._HMM_STATE_HISTORY_DDL` (read-only import — the
  canonical `CREATE TABLE IF NOT EXISTS` with the PK; lines 282-293 of that file).
- Produces: `_create_hmm_db(self, tmp_path, row=None) -> str` keeps its exact signature and
  return (existing 4 caller tests unchanged).

**Precondition:** fresh, migrated scratch DB (header block). If the scratch DB was already
poisoned by an earlier session, the RED test below would fail even after the fix — the fresh-DB
precaution (and, after Task 7, the preflight) removes that ambiguity.

- [ ] **Step 1: Write the failing regression test**

Add as the LAST method of `TestMetaOrchestratorRegimeVector` (after
`test_current_regime_vector_uses_defaults_on_exception`, line 259):

```python
    def test_create_hmm_db_preserves_canonical_pk(self, tmp_path: Path) -> None:
        """Ruling 2026-07-22: the helper must never strip hmm_state_history's PRIMARY KEY.

        The pre-fix helper DROP+recreated the table without the PK; production
        upserts (ON CONFLICT (environment, date), hmm_regime.py:320) then fail
        suites later with an error that looks nothing like its cause.
        """
        db_url = self._create_hmm_db(tmp_path, row=("2026-01-01", "equity", 0.6, 0.2, 0.1))
        conn = psycopg.connect(db_url, autocommit=True)
        row = conn.execute(
            "SELECT con.conname FROM pg_constraint con "
            "JOIN pg_class c ON c.oid = con.conrelid "
            "WHERE c.relname = 'hmm_state_history' AND con.contype = 'p'"
        ).fetchone()
        conn.close()
        assert row is not None, "hmm_state_history lost its PRIMARY KEY"
```

- [ ] **Step 2: Run it to verify it fails**

Run (with `DATABASE_URL` exported per scratch-DB header):
`uv run pytest "tests/memory/test_meta_orchestrator.py::TestMetaOrchestratorRegimeVector::test_create_hmm_db_preserves_canonical_pk" -q`
Expected: FAIL on `assert row is not None` (the current helper's hand-rolled CREATE has no PK).

- [ ] **Step 3: Implement — canonical DDL, no DROP, named-column inserts**

Add to the imports block of `tests/memory/test_meta_orchestrator.py` (after
`from swingrl.memory.training.meta_orchestrator import MetaTrainingOrchestrator`):

```python
from swingrl.data.postgres_schema import _HMM_STATE_HISTORY_DDL
```

Replace `_create_hmm_db` (lines 171-195) with:

```python
    def _create_hmm_db(self, tmp_path: Path, row: tuple | None = None) -> str:
        """Ensure the CANONICAL hmm_state_history exists; optionally seed one row.

        Never DROPs the production-named table: the canonical DDL is IF NOT
        EXISTS, the autouse wipe empties rows between db-marked tests, and a
        hand-rolled replacement shape is exactly the 2026-07-22 PK-loss
        incident. Insert is by column name because the canonical column order
        (environment first, plus log_likelihood/fitted_at) differs from the old
        hand-rolled table. Returns DATABASE_URL.
        """
        db_url = os.environ.get("DATABASE_URL", "")
        if not db_url:
            pytest.skip("DATABASE_URL not set")
        conn = psycopg.connect(db_url, autocommit=True)
        conn.execute(_HMM_STATE_HISTORY_DDL)
        if row is not None:
            conn.execute(
                "INSERT INTO hmm_state_history (date, environment, p_bull, p_bear, p_crisis) "
                "VALUES (%s, %s, %s, %s, %s)",
                list(row),
            )
        conn.close()
        return db_url
```

Replace the inline duplicate inside `test_current_regime_vector_handles_null_columns`
(the body previously at lines 225-251) with:

```python
    def test_current_regime_vector_handles_null_columns(self, tmp_path: Path) -> None:
        """TRAIN-09: NULL columns treated as 0.33/0.17 defaults."""
        db_url = self._create_hmm_db(tmp_path)
        conn = psycopg.connect(db_url, autocommit=True)
        conn.execute(
            "INSERT INTO hmm_state_history (date, environment, p_bull, p_bear, p_crisis) "
            "VALUES ('2026-01-01', 'equity', NULL, NULL, NULL)"
        )
        conn.close()
        orch = _make_orchestrator(tmp_path, database_url=db_url)
        vec = orch._current_regime_vector("equity")
        assert abs(vec["bull"] - 0.33) < 1e-6
        assert abs(vec["bear"] - 0.33) < 1e-6
        assert abs(vec["crisis"] - 0.17) < 1e-6
```

- [ ] **Step 4: Run the whole class to verify GREEN**

Run: `uv run pytest tests/memory/test_meta_orchestrator.py -q`
Expected: all pass (the 4 pre-existing regime-vector tests + the new PK regression test + the
mock-only classes), 0 failed.

- [ ] **Step 5: Verify the DB is left healthy (manual probe)**

Run:

```bash
uv run python -c "
import os, psycopg
conn = psycopg.connect(os.environ['DATABASE_URL'], autocommit=True)
row = conn.execute(\"SELECT count(*) FROM pg_constraint con JOIN pg_class c ON c.oid = con.conrelid WHERE c.relname = 'hmm_state_history' AND con.contype = 'p'\").fetchone()
print('pk_count =', row[0])
"
```

Expected: `pk_count = 1`.

- [ ] **Step 6: Commit**

```bash
git add tests/memory/test_meta_orchestrator.py
git commit -m "fix(test): meta_orchestrator uses canonical hmm_state_history DDL — never drops PK"
```

---

### Task 6: Robustness — the sibling destructive-DDL tests (backtest, validation, fundamentals)

Same anti-pattern, three more files (robustness review §1.4):

1. `tests/agents/test_backtest.py:201-281` — `_create_backtest_schema` DROPs
   `backtest_results`/`iteration_results` and hand-recreates pre-V001 shapes (no `era_id`/
   `gate_version_id` back-stamp columns) while the ledger still says V001 is applied.
   ~20 call sites.
2. `tests/agents/test_validation.py:152` (hand-rolled `model_metadata`), `:190` and `:220-221`
   (`DROP … CASCADE` then canonical init — the CASCADE kills the V001 columns and any dependent
   objects, with no restore).
3. `tests/features/test_fundamentals.py:241-253, 297-299` — hand-rolled `fundamentals` variant
   + a terminal `DROP TABLE` that leaves the table absent for later sessions.

Fix strategy (uniform): **never DROP a production-named table from a test.** On the migrated
scratch DB the canonical tables already exist (CI stage 2.7 invariant) and the wipe empties
rows between db-marked tests; `init_postgres_schema` / the canonical single-table DDL are
`IF NOT EXISTS`, so "ensure" replaces "drop+create" with identical test semantics and zero
schema damage.

**Files:**
- Modify: `tests/agents/test_backtest.py:201-281` (helper body + import; call sites unchanged)
- Modify: `tests/agents/test_validation.py:147-235` (three test bodies)
- Modify: `tests/features/test_fundamentals.py:241-253, 295-299` (+ one import)
- Test: `tests/agents/test_backtest.py` (new regression test)

**Interfaces:**
- Consumes: `src/swingrl/data/postgres_schema.init_postgres_schema` (already imported by
  test_validation; new import in test_backtest) and `postgres_schema._FUNDAMENTALS_DDL`
  (read-only, `src/swingrl/data/postgres_schema.py:268`).
- Produces: `_create_backtest_schema(conn: Any) -> None` — same name and signature (so the ~20
  existing call sites need no edits), now non-destructive.

- [ ] **Step 1: Write the failing regression test (backtest helper preserves V001 columns)**

Add to `tests/agents/test_backtest.py`, directly after the `_make_test_fold` helper (before
`class TestStoreFoldResultsToDuckdb`):

```python
def test_create_backtest_schema_preserves_migrated_columns() -> None:
    """Ruling 2026-07-22: the schema helper must not strip V001 back-stamp columns.

    Pre-fix, _create_backtest_schema DROPped backtest_results/iteration_results
    and hand-rolled pre-V001 replacements while schema_migrations still said V001
    was applied — a ledger/reality desync on the shared scratch DB that only
    healed if a db_with_legacy_schema test happened to run later.
    """
    db_url = os.environ.get("DATABASE_URL", "")
    if not db_url:
        pytest.skip("DATABASE_URL not set")
    conn = psycopg.connect(db_url, autocommit=True)
    _create_backtest_schema(conn)
    cols = {
        r[0]
        for r in conn.execute(
            "SELECT column_name FROM information_schema.columns "
            "WHERE table_name = 'backtest_results'"
        ).fetchall()
    }
    conn.close()
    assert "era_id" in cols, "V001 back-stamp column era_id lost by the test helper"
    assert "gate_version_id" in cols
```

- [ ] **Step 2: Run it to verify it fails**

Run (with `DATABASE_URL` exported, migrated scratch DB):
`uv run pytest tests/agents/test_backtest.py::test_create_backtest_schema_preserves_migrated_columns -q`
Expected: FAIL on `assert "era_id" in cols` (the current helper's hand-rolled CREATE predates
V001).

- [ ] **Step 3: Implement the non-destructive helper in test_backtest.py**

Add to the imports block (after `from swingrl.agents.backtest import (...)`):

```python
from swingrl.data.postgres_schema import init_postgres_schema
```

Replace the whole `_create_backtest_schema` function (lines 201-281, both hand-rolled DDL
blocks included) with:

```python
def _create_backtest_schema(conn: Any) -> None:
    """Ensure canonical backtest_results/iteration_results exist (non-destructive).

    Replaces the pre-2026-07-22 DROP + hand-rolled pre-V001 DDL. On the migrated
    scratch DB the canonical tables already exist (CI stage 2.7 invariant) and
    the autouse wipe truncates rows between db-marked tests, so this is a no-op
    there — and a full canonical bootstrap (IF NOT EXISTS DDL) on an empty DB.
    Never DROP a production-named table from a test: CREATE TABLE IF NOT EXISTS
    can never repair the shape afterwards (2026-07-22 hmm_state_history incident).
    """
    init_postgres_schema(conn)
    if not conn.autocommit:
        conn.commit()
```

(The `Any`-typed `conn` matches the existing signature; callers pass both autocommit and
non-autocommit connections — `test_write_does_not_persist_without_autocommit_or_explicit_commit`
relies on the helper committing its own work on a non-autocommit connection, hence the guarded
commit.)

- [ ] **Step 4: Run the backtest storage tests to verify GREEN**

Run: `uv run pytest tests/agents/test_backtest.py -q`
Expected: all pass, including the new regression test and every
`TestStoreFoldResultsToDuckdb`/`TestStoreIterationResultsToDuckdb` test (the canonical tables
are a column superset of the old hand-rolled ones; all inserts in
`src/swingrl/agents/backtest.py` are by column name).

- [ ] **Step 5: Fix the three test_validation.py bodies**

In `tests/agents/test_validation.py` (`init_postgres_schema` is already imported at line 19):

(a) `test_model_metadata_table_created` — replace the DROP + 15-line hand-rolled CREATE
(the statements at lines 152-168) so the test body becomes:

```python
    @pytest.mark.skipif(not os.environ.get("DATABASE_URL"), reason="DATABASE_URL not set")
    def test_model_metadata_table_created(self) -> None:
        """TRAIN-12: model_metadata table exists after schema init.

        Uses the canonical schema init (idempotent IF NOT EXISTS) instead of the
        pre-2026-07-22 DROP + hand-rolled DDL — never drop a production-named
        table from a test. Rows are cleared by the autouse wipe, so the insert
        below always starts from an empty table.
        """
        db_url = os.environ["DATABASE_URL"]
        conn = psycopg.connect(db_url, autocommit=True)
        init_postgres_schema(conn)
        # Verify table accepts inserts (column order matches canonical DDL).
        conn.execute("""
            INSERT INTO model_metadata VALUES (
                'model-001', 'equity', 'PPO', 'v1.0',
                '2025-01-01', '2025-06-30', 500000, 350000,
                1.5, 0.33, '/models/ppo_equity.zip',
                '/models/ppo_equity_vecnorm.pkl',
                current_timestamp
            )
        """)
        result = conn.execute("SELECT * FROM model_metadata").fetchall()
        assert len(result) == 1
        assert result[0][0] == "model-001"
        conn.close()
```

(b) `test_backtest_results_table_created` — delete the single line
`conn.execute("DROP TABLE IF EXISTS backtest_results CASCADE")` (line 190). Everything else
(init_postgres_schema + named-column INSERT + asserts) stays.

(c) `test_db_init_schema_creates_tables` — delete BOTH lines
`conn.execute("DROP TABLE IF EXISTS model_metadata CASCADE")` and
`conn.execute("DROP TABLE IF EXISTS backtest_results CASCADE")` (lines 220-221). The
init + `pg_tables` membership asserts stay and remain meaningful (idempotent-init verification
against the canonical shape).

- [ ] **Step 6: Fix test_fundamentals.py**

Add to the imports of `tests/features/test_fundamentals.py`:

```python
from swingrl.data.postgres_schema import _FUNDAMENTALS_DDL
```

In `test_store_writes_to_pg` (line 235):

(a) Replace the 13-line hand-rolled `CREATE TABLE IF NOT EXISTS fundamentals (...)` block
(lines 241-253) with:

```python
        conn.execute(_FUNDAMENTALS_DDL)  # canonical shape — never hand-roll a prod table
```

(b) Delete the two teardown lines at 297-298:

```python
        conn.execute("DROP TABLE IF EXISTS fundamentals")
        conn.commit()
```

so the test ends with the row-count assert followed directly by `conn.close()`. (The autouse
wipe clears the rows; the canonical table must SURVIVE the test — dropping it left the DB
table-less until something re-ran init_schema.)

- [ ] **Step 7: Run all three files to verify GREEN**

Run: `uv run pytest tests/agents/test_backtest.py tests/agents/test_validation.py tests/features/test_fundamentals.py -q`
Expected: all pass, 0 failed (skips only if DATABASE_URL missing — it must be set for this
step).

- [ ] **Step 8: Commit**

```bash
git add tests/agents/test_backtest.py tests/agents/test_validation.py tests/features/test_fundamentals.py
git commit -m "fix(test): destructive-DDL tests use canonical schema — never drop prod-named tables"
```

---

### Task 7: Schema-integrity preflight — hard-fail on a poisoned scratch DB

The three existing guard layers are all NAME-based; nothing checks table SHAPE, which is why
the PK loss cost three suite-hours (robustness review §1.5). This adds safety layer #4 to
`pytest_configure`: every canonical table whose DDL declares a PRIMARY KEY must — IF it exists —
carry a `contype = 'p'` row in `pg_constraint`. Absent tables are fine (they get created
canonically on demand); present-but-PK-less tables are unrepairable poison and abort the run
with the table named.

**Files:**
- Create: `tests/fixtures/schema_preflight.py`
- Create: `tests/test_schema_preflight.py`
- Modify: `tests/conftest.py:35-56` (`pytest_configure`) + one import

**Interfaces:**
- Consumes: `postgres_schema._ALL_TABLE_DDL` (read-only; 36 DDL strings), `classify_db_url`/
  `resolve_target_db_url` (existing).
- Produces (used by Task 8's worker flow and Task 10's verification):
  - `tests/fixtures/schema_preflight.py::expected_pk_tables() -> frozenset[str]`
  - `tests/fixtures/schema_preflight.py::schema_integrity_errors(db_url: str) -> list[str]`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_schema_preflight.py`:

```python
"""Schema-integrity preflight: PK fingerprint of canonical tables.

This check would have failed run 1 of 3 on 2026-07-22 with the poisoned table's
name in the message, instead of costing three suite-hours (robustness review P1).
"""

from __future__ import annotations

import os

import psycopg
import pytest

from swingrl.data.postgres_schema import _HMM_STATE_HISTORY_DDL
from tests.fixtures.schema_preflight import expected_pk_tables, schema_integrity_errors


def test_expected_pk_tables_cover_known_pk_tables() -> None:
    """Pure (fast lane): the canonical PK map includes the incident table and friends."""
    tables = expected_pk_tables()
    assert "hmm_state_history" in tables
    assert "fundamentals" in tables
    assert "model_metadata" in tables
    # Sanity: parsing found a substantial share of the 36 canonical tables.
    assert len(tables) >= 10


@pytest.mark.skipif(not os.environ.get("DATABASE_URL"), reason="DATABASE_URL not set")
def test_detects_pk_less_table_and_passes_when_canonical() -> None:
    """db lane: a PK-less hmm_state_history (the incident shape) is named; canonical is clean.

    This test intentionally performs the forbidden DROP+hand-recreate pattern —
    WITH a finally-block that restores the canonical DDL, which is exactly the
    teardown discipline the pattern was missing.
    """
    db_url = os.environ["DATABASE_URL"]
    conn = psycopg.connect(db_url, autocommit=True)
    try:
        conn.execute("DROP TABLE IF EXISTS hmm_state_history CASCADE")
        conn.execute(
            "CREATE TABLE hmm_state_history ("
            " date DATE, environment VARCHAR,"
            " p_bull DOUBLE PRECISION, p_bear DOUBLE PRECISION, p_crisis DOUBLE PRECISION)"
        )
        errors = schema_integrity_errors(db_url)
        assert any("hmm_state_history" in e for e in errors), errors
    finally:
        conn.execute("DROP TABLE IF EXISTS hmm_state_history CASCADE")
        conn.execute(_HMM_STATE_HISTORY_DDL)
        conn.close()
    clean_errors = [e for e in schema_integrity_errors(db_url) if "hmm_state_history" in e]
    assert clean_errors == []
```

- [ ] **Step 2: Run them to verify they fail**

Run: `env -u DATABASE_URL uv run pytest tests/test_schema_preflight.py -q`
Expected: collection ERROR — `ModuleNotFoundError: No module named 'tests.fixtures.schema_preflight'`.

- [ ] **Step 3: Implement `tests/fixtures/schema_preflight.py`**

```python
"""Session-start schema-integrity preflight (safety layer #4 — SHAPE-based).

The three existing guard layers are NAME-based (suite guard, URL classifier,
pre-wipe re-check) and cannot see a poisoned scratch DB: a test that
DROP+hand-recreated a production-named table without its PRIMARY KEY leaves a
shape that ``CREATE TABLE IF NOT EXISTS`` will never repair, and version-number
checks (``assert_schema_current``) pass regardless. That exact blind spot cost
three suite-hours on 2026-07-22 (hmm_state_history PK loss, found only by a
human diffing pg_constraint against production).

Rule: every canonical table (postgres_schema._ALL_TABLE_DDL) whose DDL declares
a PRIMARY KEY must, IF it exists in the target DB, carry a ``contype = 'p'``
row in pg_constraint. Absent tables are fine — they are created canonically on
demand. Cost: one catalog query per session.
"""

from __future__ import annotations

import re

import psycopg

from swingrl.data import postgres_schema

_CREATE_TABLE_RE = re.compile(r"CREATE TABLE IF NOT EXISTS\s+(\w+)", re.IGNORECASE)


def expected_pk_tables() -> frozenset[str]:
    """Canonical table names whose DDL declares a PRIMARY KEY."""
    names: set[str] = set()
    for ddl in postgres_schema._ALL_TABLE_DDL:
        match = _CREATE_TABLE_RE.search(ddl)
        if match and "PRIMARY KEY" in ddl.upper():
            names.add(match.group(1))
    return frozenset(names)


def schema_integrity_errors(db_url: str) -> list[str]:
    """Messages naming existing canonical tables that are missing their PRIMARY KEY."""
    expected = sorted(expected_pk_tables())
    with psycopg.connect(db_url, autocommit=True) as conn:
        rows = conn.execute(
            """
            SELECT c.relname
            FROM pg_class c
            JOIN pg_namespace n ON n.oid = c.relnamespace
            WHERE n.nspname = 'public'
              AND c.relkind = 'r'
              AND c.relname = ANY(%s)
              AND NOT EXISTS (
                  SELECT 1 FROM pg_constraint con
                  WHERE con.conrelid = c.oid AND con.contype = 'p'
              )
            ORDER BY c.relname
            """,
            (expected,),
        ).fetchall()
    return [
        f"table {row[0]!r} exists WITHOUT its canonical PRIMARY KEY — the scratch "
        f"DB is poisoned (some test DROP+hand-recreated it). Recreate the scratch "
        f"database (or DROP the table and re-run init_schema) before testing."
        for row in rows
    ]
```

- [ ] **Step 4: Wire the preflight into `pytest_configure`**

In `tests/conftest.py`, add two imports (with the other first-party test imports):

```python
import psycopg
```

(in the third-party block, after `import pandas as pd`), and

```python
from tests.fixtures.schema_preflight import schema_integrity_errors
```

(after the `tests.db_marker` import from Task 2). Then replace the whole `pytest_configure`
function (lines 35-56) with:

```python
def pytest_configure(config: pytest.Config) -> None:
    """Refuse to run the suite unless the *resolved* DB target is a safe, healthy test DB.

    Layer 2 (name): resolves the target the same way DatabaseManager does (env
    DATABASE_URL, then config.system.database_url), so a blank env var with a
    production URL in YAML config cannot slip past the guard.
    Layer 4 (shape): a cheap pg_constraint fingerprint refuses a poisoned scratch
    DB — any canonical table present without its PRIMARY KEY aborts the session
    with the table named (2026-07-22 incident cost three suite-hours without it).
    """
    db_url = resolve_target_db_url()
    verdict, db_name = classify_db_url(db_url)
    if verdict == "blank":
        return
    if verdict == "unparseable":
        pytest.exit(
            "Resolved database URL has no parseable database name; refusing to run pytest.",
            returncode=2,
        )
    if verdict != "safe":
        pytest.exit(
            f"REFUSING TO RUN: resolved database {db_name!r} is not a test database. "
            f"Test fixtures TRUNCATE/DELETE tables. Use a name in "
            f"{sorted(SAFE_DB_NAMES)} or one ending in '_test'. In CI, "
            f"scripts/ci-homelab.sh overrides DATABASE_URL automatically.",
            returncode=2,
        )
    try:
        errors = schema_integrity_errors(db_url)
    except psycopg.OperationalError as exc:
        pytest.exit(
            f"Cannot reach test database {db_name!r} for the schema preflight: {exc}",
            returncode=2,
        )
    if errors:
        pytest.exit(
            "SCHEMA PREFLIGHT FAILED — refusing to run against a poisoned test DB:\n  "
            + "\n  ".join(errors),
            returncode=2,
        )
```

- [ ] **Step 5: Run the preflight tests to verify GREEN**

Run: `env -u DATABASE_URL uv run pytest tests/test_schema_preflight.py -q`
Expected: `1 passed, 1 skipped` (pure test passes; db test skips without a URL).

Run (with `DATABASE_URL` exported per scratch-DB header):
`uv run pytest tests/test_schema_preflight.py -q`
Expected: `2 passed`.

- [ ] **Step 6: End-to-end proof the preflight aborts a poisoned session**

Poison the scratch DB, attempt a run, confirm refusal, repair:

```bash
uv run python -c "
import os, psycopg
conn = psycopg.connect(os.environ['DATABASE_URL'], autocommit=True)
conn.execute('DROP TABLE IF EXISTS hmm_state_history CASCADE')
conn.execute('CREATE TABLE hmm_state_history (date DATE, environment VARCHAR)')
print('poisoned')
"
uv run pytest tests/config -q; echo "exit=$?"
uv run python -c "
import os, psycopg
from swingrl.data.postgres_schema import _HMM_STATE_HISTORY_DDL
conn = psycopg.connect(os.environ['DATABASE_URL'], autocommit=True)
conn.execute('DROP TABLE IF EXISTS hmm_state_history CASCADE')
conn.execute(_HMM_STATE_HISTORY_DDL)
print('repaired')
"
```

Expected: the pytest invocation prints
`SCHEMA PREFLIGHT FAILED — refusing to run against a poisoned test DB:` naming
`hmm_state_history`, runs **zero tests**, and exits `exit=2`; after the repair,
`uv run pytest tests/config -q` runs normally.

- [ ] **Step 7: Commit**

```bash
git add tests/fixtures/schema_preflight.py tests/test_schema_preflight.py tests/conftest.py
git commit -m "feat(test): schema-integrity preflight — abort on poisoned scratch DB (layer 4)"
```

---

### Task 8: Parallelization Phase B — per-worker DBs cloned from a pre-migrated template

Each xdist worker (and, opt-in, each serial session) gets its own database cloned file-level
from `swingrl_test_template` (~100-300 ms, no migration replay). Workers can then run the FULL
DB suite in parallel, and two pytest processes can never share a database.

**Naming decision (pre-approved deviation):** the scoping note's literal `swingrl_test_gw<N>`
does NOT end in `_test` and would be REFUSED by the existing guard
(`tests/db_guard.py:67` — `name.endswith("_test")`). Respecting the guard unchanged (global
constraint: never weaken the safety stack) means `gw0` maps to **`swingrl_gw0_test`** instead.
The template `swingrl_test_template` is never a pytest target (admin connections only), so its
name is exempt from the `_test` rule.

**Files:**
- Create: `tests/db_worker.py`
- Create: `tests/test_db_worker.py` (pure, fast-lane tests)
- Modify: `tests/conftest.py` (activate at the top of `pytest_configure`; add
  `pytest_unconfigure`)
- Create: `scripts/prepare-test-db-template.sh`
- Modify: `scripts/ci-homelab.sh` (new stage 2.8; stage 3 gains `-n 4`; stage 6 cleanup)

**Interfaces:**
- Consumes: `classify_db_url` (Task 0 state), `EXPECTED_SCHEMA_VERSION`
  (`src/swingrl/data/migration_runner.py:34`, read-only import), the persistent cleanup
  connection's URL re-keying (Task 3).
- Produces:
  - `tests/db_worker.py::TEMPLATE_DB = "swingrl_test_template"`
  - `tests/db_worker.py::isolation_token() -> str | None`
  - `tests/db_worker.py::derive_isolated_db_url(base_url: str, token: str) -> str`
  - `tests/db_worker.py::ensure_isolated_db(base_url: str, worker_url: str) -> None`
  - `tests/db_worker.py::activate_isolated_db() -> None` / `drop_isolated_db() -> None`
  - Env contract: workers see `DATABASE_URL` rewritten to their clone; every existing fixture
    resolves through the env var, so **zero fixture changes** (speed review R2).

- [ ] **Step 1: Write the failing pure tests**

Create `tests/test_db_worker.py`:

```python
"""Per-worker isolated-DB derivation: pure tests (fast lane, no DB, no xdist).

Deliberately avoids the literal D-A-T-A-B-A-S-E_URL env-var name in source so
this module stays out of the auto-derived db lane (it never touches Postgres).
"""

from __future__ import annotations

import pytest

from tests.db_worker import derive_isolated_db_url, isolation_token

_BASE = "postgresql://u:pw@172.18.5.246:5432/swingrl_test"  # pragma: allowlist secret


def test_derive_gw0() -> None:
    """swingrl_test + gw0 -> swingrl_gw0_test (guard-compatible _test suffix)."""
    assert (
        derive_isolated_db_url(_BASE, "gw0")
        == "postgresql://u:pw@172.18.5.246:5432/swingrl_gw0_test"  # pragma: allowlist secret
    )


def test_derive_preserves_query_string() -> None:
    """URL options after the DB name survive the rewrite."""
    assert (
        derive_isolated_db_url(_BASE + "?sslmode=disable", "gw3")
        == "postgresql://u:pw@172.18.5.246:5432/swingrl_gw3_test?sslmode=disable"  # pragma: allowlist secret
    )


def test_derive_refuses_non_test_base() -> None:
    """A production-named base URL can never spawn clones."""
    with pytest.raises(RuntimeError, match="non-test base URL"):
        derive_isolated_db_url("postgresql://u:pw@h:5432/swingrl", "gw0")  # pragma: allowlist secret


def test_isolation_token_xdist_worker(monkeypatch: pytest.MonkeyPatch) -> None:
    """xdist workers always isolate, keyed by their worker id."""
    monkeypatch.setenv("PYTEST_XDIST_WORKER", "gw2")
    assert isolation_token() == "gw2"


def test_isolation_token_serial_opt_in(monkeypatch: pytest.MonkeyPatch) -> None:
    """Serial sessions isolate only when SWINGRL_TEST_ISOLATED_DB=1 (pid-keyed)."""
    monkeypatch.delenv("PYTEST_XDIST_WORKER", raising=False)
    monkeypatch.setenv("SWINGRL_TEST_ISOLATED_DB", "1")
    token = isolation_token()
    assert token is not None
    assert token.startswith("main")


def test_isolation_token_default_none(monkeypatch: pytest.MonkeyPatch) -> None:
    """No xdist, no opt-in -> use the configured URL as-is (backward compatible)."""
    monkeypatch.delenv("PYTEST_XDIST_WORKER", raising=False)
    monkeypatch.delenv("SWINGRL_TEST_ISOLATED_DB", raising=False)
    assert isolation_token() is None
```

Run: `env -u DATABASE_URL uv run pytest tests/test_db_worker.py -q`
Expected: collection ERROR — `ModuleNotFoundError: No module named 'tests.db_worker'`.

- [ ] **Step 2: Implement `tests/db_worker.py`**

```python
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
            sql.SQL("DROP DATABASE IF EXISTS {} WITH (FORCE)").format(
                sql.Identifier(worker_name)
            )
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
```

Run: `env -u DATABASE_URL uv run pytest tests/test_db_worker.py -q`
Expected: `6 passed`.

- [ ] **Step 3: Wire activation/teardown into `tests/conftest.py`**

Add the import (after the `tests.db_marker` import):

```python
from tests.db_worker import activate_isolated_db, drop_isolated_db
```

At the very TOP of `pytest_configure` (before `db_url = resolve_target_db_url()` — the guard
and preflight must run against the ALREADY-rewritten worker URL), insert:

```python
    # Per-worker/per-session DB isolation (tests/db_worker.py). Must run FIRST:
    # the name guard and schema preflight below validate the rewritten URL.
    try:
        activate_isolated_db()
    except RuntimeError as exc:
        pytest.exit(str(exc), returncode=2)
```

Add after `pytest_collection_modifyitems`:

```python
def pytest_unconfigure(config: pytest.Config) -> None:
    """Drop this process's isolated DB clone at session end (best-effort)."""
    drop_isolated_db()
```

- [ ] **Step 4: Create `scripts/prepare-test-db-template.sh`**

```bash
#!/usr/bin/env bash
# Build/refresh the pre-migrated test-DB template (swingrl_test_template) and
# reap stale per-worker clones. Run on homelab from the repo root whenever a
# migration ships (EXPECTED_SCHEMA_VERSION changes) or the template goes stale.
# Requires: docker access to pg16, a .env with the swingrl DATABASE_URL password.
set -euo pipefail

PG_CONTAINER="${PG_CONTAINER:-pg16}"
TEMPLATE_DB="swingrl_test_template"
ENV_FILE="${ENV_FILE:-.env}"

PG_IP=$(docker inspect -f '{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}' "$PG_CONTAINER")
PG_PASS=$(grep DATABASE_URL "$ENV_FILE" | sed 's/.*:\/\/swingrl:\([^@]*\)@.*/\1/')

echo "=== [1/4] Grant CREATEDB to swingrl (idempotent; workers clone their own DBs) ==="
docker exec "$PG_CONTAINER" psql -U temporal -d postgres -c "ALTER ROLE swingrl CREATEDB;"

echo "=== [2/4] Recreate ${TEMPLATE_DB} ==="
docker exec "$PG_CONTAINER" psql -U temporal -d postgres \
    -c "DROP DATABASE IF EXISTS ${TEMPLATE_DB} WITH (FORCE);"
docker exec "$PG_CONTAINER" psql -U temporal -d postgres \
    -c "CREATE DATABASE ${TEMPLATE_DB} OWNER swingrl;"

echo "=== [3/4] Apply schema + migrations to ${TEMPLATE_DB} ==="
DATABASE_URL="postgresql://swingrl:${PG_PASS}@${PG_IP}:5432/${TEMPLATE_DB}" uv run python -c "
from swingrl.config.schema import load_config
from swingrl.data.db import DatabaseManager
from swingrl.data.migration_runner import apply_migrations, current_schema_version
db = DatabaseManager(load_config('config/swingrl.yaml'))
db.init_schema()
n = apply_migrations(db)
print(f'applied={n} current_version={current_schema_version(db)}')
"

echo "=== [4/4] Reap stale worker/session clones ==="
docker exec "$PG_CONTAINER" psql -U temporal -d postgres -tAc \
    "SELECT datname FROM pg_database WHERE datname ~ '^swingrl_(gw[0-9]+|main[0-9]+)_test$'" |
while read -r db; do
    [ -n "$db" ] && docker exec "$PG_CONTAINER" psql -U temporal -d postgres \
        -c "DROP DATABASE IF EXISTS \"$db\" WITH (FORCE);"
done

echo "=== TEMPLATE READY: ${TEMPLATE_DB} on ${PG_CONTAINER} (${PG_IP}) ==="
```

Run: `chmod +x scripts/prepare-test-db-template.sh`

- [ ] **Step 5: Update `scripts/ci-homelab.sh`**

(a) After the stage 2.7 block (the `$DEV_COMPOSE run … apply_migrations` heredoc, ends line 91),
insert:

```bash
echo "=== [2.8/6] Create test-DB template for per-worker clones ==="
# Workers (tests/db_worker.py) clone swingrl_gwN_test from this template at
# session start — file-level copy, no migration replay. swingrl needs CREATEDB
# to create its own clones. Stage 2.7 has just left swingrl_test fully migrated
# and its container has exited, so no connections block the TEMPLATE copy.
docker exec pg16 psql -U temporal -d postgres -c "ALTER ROLE swingrl CREATEDB;"
docker exec pg16 psql -U temporal -d postgres -c "DROP DATABASE IF EXISTS swingrl_test_template WITH (FORCE);"
docker exec pg16 psql -U temporal -d postgres -c "CREATE DATABASE swingrl_test_template TEMPLATE swingrl_test OWNER swingrl;"
```

(b) Change stage 3 (line 93-94) from:

```bash
$DEV_COMPOSE run --rm --entrypoint "" -e DATABASE_URL="$TEST_DB_URL" swingrl uv run pytest tests/ -v
```

to:

```bash
# -n 4: per-worker DBs (tests/db_worker.py) make the FULL suite parallel-safe —
# each gwN worker clones swingrl_gwN_test from the stage-2.8 template.
$DEV_COMPOSE run --rm --entrypoint "" -e DATABASE_URL="$TEST_DB_URL" swingrl uv run pytest tests/ -v -n 4
```

(c) In stage 6 cleanup, after the existing
`docker exec pg16 psql -U temporal -d postgres -c "DROP DATABASE IF EXISTS swingrl_test;" || true`
line, add:

```bash
docker exec pg16 psql -U temporal -d postgres -tAc \
    "SELECT datname FROM pg_database WHERE datname ~ '^swingrl_(gw[0-9]+|main[0-9]+)_test$'" |
while read -r db; do
    [ -n "$db" ] && docker exec pg16 psql -U temporal -d postgres \
        -c "DROP DATABASE IF EXISTS \"$db\" WITH (FORCE);" || true
done
docker exec pg16 psql -U temporal -d postgres -c "DROP DATABASE IF EXISTS swingrl_test_template;" || true
```

- [ ] **Step 6: Local end-to-end verification (template + parallel db package)**

On homelab, from the repo root (uses `~/swingrl/.env` for the password —
`ENV_FILE=~/swingrl/.env` if the dev checkout has no `.env`):

```bash
ENV_FILE=~/swingrl/.env bash scripts/prepare-test-db-template.sh
```

Expected: `applied=11 current_version=11` and `TEMPLATE READY`.

Then run a DB-heavy package in parallel against the base scratch DB (workers self-clone):

```bash
uv run pytest tests/data -n 2 -q
```

(with `DATABASE_URL` exported per the scratch-DB header). Expected: all pass; during the run
`docker exec pg16 psql -U temporal -d postgres -tAc "SELECT datname FROM pg_database WHERE datname LIKE 'swingrl_gw%_test'"`
shows `swingrl_gw0_test` and `swingrl_gw1_test`; after the run both are gone
(`pytest_unconfigure` dropped them).

Also verify serial opt-in isolation:

```bash
SWINGRL_TEST_ISOLATED_DB=1 uv run pytest tests/test_db_isolation.py -q
```

Expected: `2 passed`; no leftover `swingrl_main*_test` databases afterward.

- [ ] **Step 7: Commit**

```bash
git add tests/db_worker.py tests/test_db_worker.py tests/conftest.py \
        scripts/prepare-test-db-template.sh scripts/ci-homelab.sh
git commit -m "feat(test): per-worker DBs cloned from template — parallel full suite (Phase B)"
```

---

### Task 9: Three-lane model docs + command defaults + CLAUDE.md pointer (one commit)

House rule: docs update with code — the command-file change (code) and the docs describing the
lane machinery ship in the SAME commit.

**Files:**
- Create: `docs/testing/best-practices.md`
- Modify: `.claude/commands/test.md` (fast lane becomes the default)
- Modify: `CLAUDE.md` (Testing Conventions section: lanes + `--lf` rule + pointer)

**Interfaces:**
- Consumes: everything Tasks 1-8 built (marker, conditional wipe, xdist lanes, preflight,
  worker DBs).
- Produces: the documented workflow Task 10 verifies against.

- [ ] **Step 1: Create `docs/testing/best-practices.md`**

```markdown
# SwingRL Testing Best Practices

Source: test-infra reviews + implementation plan of 2026-07-22
(`docs/superpowers/plans/2026-07-22-test-suite-speed-robustness.md`). Each rule cites the
evidence that motivated it. Timing budgets were calibrated by that plan's measurement task.

## The three lanes

| Lane | Command | DB? | Budget | When |
|---|---|---|---|---|
| **fast** | `env -u DATABASE_URL uv run pytest tests/ -m "not db and not slow and not integration" -n auto -q` | none | < 2 min | every commit, every TDD RED/GREEN cycle |
| **db** | `DATABASE_URL=postgresql://swingrl:…@<pg16-ip>:5432/<name>_test uv run pytest tests/<pkg> -m "not slow" -q` | `*_test` scratch | < 5 min / package | before pushing a change that touches SQL / stores / migrations / that package |
| **full** | `bash scripts/ci-homelab.sh` (stage 3: `pytest tests/ -v -n 4`) | per-worker clones | ≤ 20 min | pre-push gate only |

pg16 has no host port — reach scratch DBs via the container IP
(`docker inspect -f '{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}' pg16`).

## The targeted-preflight rule (anti-2-hour rule)

**After ANY test failure, the next pytest invocation must be the failing subset, never the
full suite.** Escalation order:

1. `uv run pytest --lf -q` — the failure cache works out of the box (nothing disables the
   cacheprovider). Seconds to a minute.
2. Cache stale/gone (e.g. the failure was inside Docker CI — its cache died with the
   container): rerun the exact node IDs from the failure output
   (`ci-homelab.sh` runs `-v`, so node IDs are always in the log).
3. Only when the subset is green: relaunch the lane that failed.

A full-suite relaunch to verify a fix whose failing set is known is a rule violation
(2026-07-22 incident: 2 hours lost exactly this way).

## The checklist

1. **Always know which lane you are in** — fast / db / full, budgets above. *(Evidence: the
   only pre-2026-07-22 gate was one monolithic `pytest tests/ -v`, scripts/ci-homelab.sh:92 →
   60-minute minimum feedback.)*
2. **After any failure, rerun the failing subset first** (`--lf` / node IDs; rule above).
   *(Evidence: cacheprovider enabled but unused; 2026-07-22 incident cost 2 h.)*
3. **New DB-touching test ⇒ carries the `db` marker automatically** — derivation is
   collection-time (tests/db_marker.py: real-DB fixture requested, or module mentions
   DATABASE_URL). Do not hand-mark except to force with `@pytest.mark.db`. *(Evidence: DB
   gating was spelled 4 non-selectable ways across 51 files.)*
4. **One spelling for DB gating in new code**: module-level
   `pytestmark = pytest.mark.skipif(not os.environ.get("DATABASE_URL"), ...)` — no new inline
   mid-test `pytest.skip("DATABASE_URL...")`. *(Evidence: 43 inline skips were invisible to
   collection-time selection, e.g. tests/data/test_attribution_migration.py:78.)*
5. **Real DB only when SQL semantics are the subject** (migrations, upserts, FK/view order,
   store layers); mock (`make_mock_db`, tests/conftest.py:344) when the DB is a bystander.
   Rule of thumb: if changing the SQL string could break production while the test stays
   green, the test belongs on the real DB. *(Evidence: 61 mock files vs 16 real-DB files —
   split existed, unlabeled.)*
6. **Never name a real-DB fixture "mock"** — and vice versa. *(Evidence:
   tests/execution/conftest.py:73 `mock_db` is a real PostgreSQL DatabaseManager; root
   `make_mock_db` is a MagicMock — same word, opposite meanings, ~219 usages. Rename to
   `real_test_db` as one mechanical refactor when convenient.)*
7. **Session scope only for immutable values; freeze or don't promote.** DB fixtures stay
   function-scoped, non-negotiable (the autouse singleton reset at tests/conftest.py would
   hand wider-scoped fixtures a closed pool). *(Evidence: scope policy at
   tests/conftest.py:3-6; only `repo_root` is session-scoped.)*
8. **New autouse fixture ⇒ docstring must justify the per-test tax across ~1,900 tests** and
   state its per-test cost. *(Evidence: the unconditional wipe cost ~40 min/run before
   2026-07-22; contrast the well-justified structlog isolator, tests/conftest.py:66-93.)*
9. **Expensive schema fixtures (`db_with_legacy_schema`-class: de-migrate + re-migrate per
   test) live only in db/full lanes, never the commit loop.** *(Evidence:
   tests/data/conftest.py:64-112, ~79 test sites, full V001-V011 drop+reapply per test.)*
10. **NEVER `DROP TABLE` a production-named table from a test.** Use the canonical DDL
    (`src/swingrl/data/postgres_schema.py`) via `init_postgres_schema`/single-table constants,
    or a throwaway `*_probe`/`_mig_test_*` name. `CREATE TABLE IF NOT EXISTS` will never
    repair a wrong shape for you. *(Evidence: the 2026-07-22 hmm_state_history PK loss —
    3 suite-hours; the schema preflight now aborts the session if this rule is broken.)*
11. **Seeded factories over inline data**: pinned-RNG frames/arrays from conftest factories;
    package-local sizes in the package conftest. *(Evidence: good pattern at
    tests/conftest.py:166-321, tests/features/conftest.py — extend, don't fork.)*
12. **One config-YAML factory** — no fourth copy of the ~40-line YAML block. *(Evidence:
    three near-identical blocks: tests/conftest.py:105-143, 217-255,
    tests/execution/conftest.py:21-61.)*
13. **Docstring states the asserted behavior; cite ruling/REQ IDs when they exist.**
    *(Evidence: the literal `REQ-ID:` rule was followed by 23/1,905 tests — this matches live
    good practice instead, e.g. the rulings RED commits.)*
14. **Never weaken the DB-safety stack** — suite guard (tests/conftest.py::pytest_configure),
    `_test` classifier (tests/db_guard.py), pre-wipe re-check
    (tests/fixtures/db_cleanup.py:29-46), schema preflight
    (tests/fixtures/schema_preflight.py). Route around it, never through it.
15. **Long tests declare their own budget**: anything legitimately > 60 s gets
    `@pytest.mark.slow` + a tighter `@pytest.mark.timeout(n)`. *(Evidence: one global
    `timeout = 600` and zero per-test overrides — a hang burns 10 silent minutes.)*
16. **TDD stays RED-commit-then-GREEN; run the RED test via the fast lane or `--lf`, never
    the full suite.** *(Evidence: discipline alive in git history — keep the loop cheap so it
    survives.)*

## Isolated DBs and the template

- xdist workers automatically clone `swingrl_gwN_test` from `swingrl_test_template`
  (tests/db_worker.py). Serial sessions opt in with `SWINGRL_TEST_ISOLATED_DB=1`.
- Refresh the template after ANY new migration:
  `bash scripts/prepare-test-db-template.sh` (also reaps stale clones). A stale template
  fails loud (schema-version check at clone time).
- The schema preflight aborts any session whose target DB has a canonical table without its
  PRIMARY KEY — if you see `SCHEMA PREFLIGHT FAILED`, recreate the scratch DB; do not "fix"
  the guard.
```

- [ ] **Step 2: Make the fast lane the `/project:test` default**

Replace the content of `.claude/commands/test.md` with:

```markdown
---
description: Run pytest FAST lane natively (no DB, parallel). Usage: /project:test [path] [-k filter] [-x]
---

Run the fast lane by default — no DB, `db`/`slow`/`integration` tests excluded, parallel:

```bash
env -u DATABASE_URL uv run pytest ${ARGUMENTS:-tests/} -m "not db and not slow and not integration" -n auto -q
```

Common patterns:
- Fast lane, whole suite (default): `/project:test`
- One package: `/project:test tests/execution`
- Single test by name: `/project:test -k test_valid_config_loads`
- Stop on first failure: `/project:test -x`
- Rerun last failures only: `uv run pytest --lf -q` (ALWAYS do this first after a failure)
- DB lane (targeted, real scratch DB — see docs/testing/best-practices.md for the URL recipe):
  `DATABASE_URL=postgresql://swingrl:…@<pg16-ip>:5432/<name>_test uv run pytest tests/<pkg> -m "not slow" -q`
- Full suite (pre-push only): `bash scripts/ci-homelab.sh` on homelab
```

- [ ] **Step 3: Update CLAUDE.md's Testing Conventions section**

In `CLAUDE.md`, inside "## Testing Conventions", replace the line:

```markdown
- Run before commit: `uv run pytest tests/ -v`
```

with:

```markdown
- Run before commit (FAST lane, <2 min):
  `env -u DATABASE_URL uv run pytest tests/ -m "not db and not slow and not integration" -n auto -q`
- Three lanes (fast / db / full) + the after-failure `--lf`-first rule:
  see `docs/testing/best-practices.md`. Full suite = `scripts/ci-homelab.sh`, pre-push only.
- After ANY test failure: rerun the failing subset first (`uv run pytest --lf -q` or explicit
  node IDs); relaunch a full lane only after the subset is green.
- NEVER `DROP TABLE` a production-named table from a test — use canonical DDL from
  `src/swingrl/data/postgres_schema.py` or a throwaway table name (the schema preflight
  aborts poisoned-DB sessions).
```

- [ ] **Step 4: Verify the documented commands actually run**

Run the fast-lane command verbatim from the new docs:
`env -u DATABASE_URL uv run pytest tests/ -m "not db and not slow and not integration" -n auto -q`
Expected: green, < 2 min *(record)*.

- [ ] **Step 5: Commit (docs + the command code they describe, together)**

```bash
git add docs/testing/best-practices.md .claude/commands/test.md CLAUDE.md
git commit -m "docs(test): three-lane workflow + --lf preflight rule; /project:test defaults to fast lane"
```

---

### Task 10: Final verification, calibration vs success criteria, homelab CI

**Files:**
- Modify: `docs/superpowers/plans/2026-07-22-test-suite-speed-robustness.md` (Calibration
  Record — "After" column)

**Interfaces:**
- Consumes: everything.
- Produces: the evidence for the PR description; go/no-go against the success criteria.

- [ ] **Step 1: Fresh scratch DB + full serial-equivalent parallel run**

Prepare a fresh scratch DB (header block) AND the template
(`ENV_FILE=~/swingrl/.env bash scripts/prepare-test-db-template.sh`), then run the full suite
the way CI now does, **in the background, harness-tracked** (house rule: no-poll long jobs):

```bash
time uv run pytest tests/ -v -n 4 2>&1 | tee /tmp/claude-1000/-home-varun-Projects-Simplementix-SwingRL/9f5ffdda-0924-42df-b90d-26302a2745f0/scratchpad/final-durations.log
```

Expected: 0 failed (1,949+ collected — the plan added ~20 tests), wall time **≤ 20 min**
against the success criterion. Record total + top-25 durations in the Calibration Record
("After" column).

- [ ] **Step 2: Fast-lane timing**

Run: `time env -u DATABASE_URL uv run pytest tests/ -m "not db and not slow and not integration" -n auto -q`
Expected: green, **< 2 min**. Record.

- [ ] **Step 3: If a criterion misses, diagnose from the durations, don't guess**

If the full run exceeds 20 min: the top-25 durations name the offenders. Known second-tier
costs the reviews already scoped (NOT in this plan — do not silently expand scope):
`db_with_legacy_schema` re-migration cycles (legitimate — parallelism is the remedy),
`seeded_duckdb` row-by-row seeding, HMM `n_inits=10` refits, per-test memory-service pools.
Report the measured gap and the named offenders to the user; propose the follow-up as a new
plan rather than patching here.

- [ ] **Step 4: Commit the calibration record**

```bash
git add docs/superpowers/plans/2026-07-22-test-suite-speed-robustness.md
git commit -m "docs(test): final calibration — suite timing vs success criteria"
```

- [ ] **Step 5: Push and run homelab CI (house workflow — must pass before PR)**

```bash
git push -u origin swingrl/2.R-F-test-infra
cd ~/swingrl && git fetch origin && git checkout swingrl/2.R-F-test-infra && \
  git pull origin swingrl/2.R-F-test-infra && bash scripts/ci-homelab.sh --no-cache
```

Launch harness-tracked in the background (CI takes tens of minutes with `--no-cache`).
Expected: `=== CI PASSED ===`, with stage 2.8 creating the template and stage 3 running
`pytest tests/ -v -n 4` green.

- [ ] **Step 6: Create the PR to the integration branch (never main)**

Only after CI passes. PR targets `swingrl/2.R-training-redesign`. Body: goal, the five items →
tasks mapping, before/after calibration numbers, and the Task 8 naming deviation
(`swingrl_gw0_test` vs the scoping note's `swingrl_test_gw0` — guard-compatibility rationale).
Per house rule, deployment/merge is the user's call — stop after the PR is created.

---

## Self-Review (performed while writing — findings fixed inline)

- **Spec coverage:** item 1 → Task 3; item 2 → Tasks 1 + 10; item 3 Phase A → Task 4,
  Phase B → Task 8; item 4a → Tasks 5 + 6, 4b → Task 7; item 5 → Tasks 2 + 9. All five covered.
- **Placeholder scan:** no TBDs; the only intentionally deferred numbers are the calibration
  measurements, which are the *product* of Tasks 1/10, marked *(record)*.
- **Name/type consistency:** `_get_cleanup_conn` / `_run_truncate` / `_truncate_all_public_tables`
  consistent across Tasks 3, 8, and tests; `DB_FIXTURE_NAMES` / `is_db_test` /
  `module_mentions_database_url` consistent across Tasks 2, 3; `derive_isolated_db_url` /
  `isolation_token` / `activate_isolated_db` / `drop_isolated_db` consistent across Task 8 and
  conftest wiring; `_create_backtest_schema` keeps its original name on purpose (20 call sites).
- **Known accepted tradeoffs (documented in-task):** module-text derivation over-marks mock-only
  tests inside DATABASE_URL-mentioning modules (safe direction — extra wipes, never missing
  ones); Task 8's `swingrl_gw0_test` naming deviation; serial-session isolation is opt-in until
  the template exists everywhere.
