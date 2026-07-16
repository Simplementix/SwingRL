# Plan A — Paper-Trading Capture Foundation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development
> (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps
> use checkbox (`- [ ]`) syntax for tracking.

> **Status: WALKTHROUGH-REVIEWED (2026-07-07); AMENDED 2026-07-11 after the
> execution-path code review — now 21 implementation tasks (1–17 plus A–D, after the
> Task 0 gate). AMENDED 2026-07-12: Task 5 selects the best-CPS era-0 vintage
> (crypto iter 0, equity iter 4, from live `iteration_results`) and stamps real
> iteration/seed — P-A1 sentinels demoted to fallback (vintages recoverable via
> `models/iterations/`). AMENDED 2026-07-12 (A30): new Task E — trader/trainer deploy
> isolation (compose split, pinned trader tag, floor-semantics schema assertion,
> `models/active/` write ban) — now 22 implementation tasks; Task E before Task 16 and
> before any Plan B homelab deploy.**
> Review findings + user-approved disposition:
> `docs/superpowers/reviews/2026-07-07-execution-path-code-review.md` and the
> "Code-review disposition" section below.
> Companion: Plan B (`2026-07-07-training-redesign-plan.md`, written after Plan A is
> settled). Spec: `docs/superpowers/specs/2026-06-12-training-system-redesign-design.md`
> (G1 signed off; amendments A1–A29).

**Goal:** Stand up the durable capture layer for paper trading — migration machinery,
identity-spine subset, trade-time record tables (§3.7 / §4.7), the F1 + F1b turbulence
fixes, and the event calendar — so per-algo live behavior is attributable from the first
paper-trading cycle — **and a verified, secured paper-trading runtime**: broker-API
currency confirmed, the crypto fill simulator audited against real Binance.US behavior,
dependencies CVE-scanned, secrets hardened, and Discord + circuit-breaker paths proven
end-to-end (Tasks 13–16).

**Architecture:** New versioned SQL migrations (a real ledger replaces the
`CREATE IF NOT EXISTS`-at-startup pattern) create the registries (`eras`,
`gate_versions`), the spine subset (`training_runs`, `models`), and the trade-time tables.
A fail-open `CycleRecorder` hooks into `ExecutionPipeline.execute_cycle` where all
intermediates (per-algo actions, weights, blend) are already in scope — capture must never
block the money path. LLM-touching pieces (`llm_calls` + Meta-Trader commentary) are
DDL-ready but runtime-gated behind key rotation.

**Tech Stack:** Python 3.11, psycopg/psycopg_pool (Postgres 16), Stable Baselines 3,
APScheduler, structlog, pydantic v2 config, pytest.

## Glossary — read this first

Shorthand used throughout this plan, defined once here (per the project's
plain-English rule):

| Term | Meaning |
|---|---|
| **DDL** | Data Definition Language — the SQL statements that create or change *table structures* (`CREATE TABLE`, `ALTER TABLE`), as opposed to SQL that reads/writes rows |
| **Migration** | A numbered one-time script (`V001__...sql`) that upgrades the database structure; applied in order and recorded in the `schema_migrations` ledger so "which version is this database on" is always answerable |
| **Schema** | The database's overall shape — which tables exist, with which columns and rules |
| **Identity spine** | The `training_runs` table that every training-scoped record points at via `run_pk`, so any row can answer "which iteration/env/algo/fold produced me" structurally |
| **Era 0** | The comparability label stamped on all pre-redesign data (iterations 0–4): kept as evidence, never score-compared with the new regime (spec §4.1) |
| **Sentinel** | A documented placeholder (e.g. `-1`) meaning "genuinely unknown", used where legacy values are unrecoverable — instead of making the column nullable, which would weaken the rules for all future rows |
| **Fail-open** | On error, step aside rather than block: if capture fails, the trade still executes and an alert fires. (Fail-closed = no record → no trade.) |
| **Fingerprint assertion** | At startup the container checks the `schema_migrations` ledger against the version it was built for: **behind → refuse to run** (code merged but database never migrated); **ahead → warn and run** (a newer additive migration landed via a trainer deploy — floor semantics, A30) |
| **F1** | Turbulence bug 1: the live halt's *historical baseline* query reads a column that doesn't exist → silent 0.0 → the turbulence circuit breaker never fires (`execution/pipeline.py:537`, `:517`) |
| **F1b** | Turbulence bug 2: models were *trained* with the turbulence observation input frozen at 0.0, so feeding real values at inference multiplies them by untrained weights = noise |
| **MT / Meta-Trader** | The trade-time LLM coach (spec §3) — the "game-day manager" that watches paper/live trading and comments; distinct from the meta-*trainer* (the between-seasons coach). Advisory-only in this plan |
| **Intent record ("bet slip")** | The five-block form every coach call writes: identity, evidence snapshot, proposal, falsifiable bet, verdict (spec §2.4) — tables `intent_records`/`intent_verdicts` |
| **Rotation-gated** | Disabled until the 2026-03-24 leaked API keys are replaced (config `meta_trader.enabled: false`) |
| **CVE audit** | Scanning third-party dependencies against the public registry of known vulnerabilities (CVE = Common Vulnerabilities and Exposures) |
| **Sim fidelity** | How closely `binance_sim` (our simulated crypto broker — Binance.US has no real paper venue) matches real fees, minimum-order/lot-size filters, and fill behavior — mock trades must mirror live |
| **CB** | Circuit breaker — the deterministic tripwires that halt trading (drawdown breach, turbulence spike) |
| **VecNormalize** | The per-algorithm observation normalizer (SB3) saved next to each model; observations pass through it before prediction |
| **Spine FK** | A foreign-key column pointing at the identity spine (`run_pk` → `training_runs`) |

## Global Constraints

- **Execution order gate — SATISFIED 2026-07-11:** the execution/inference-path code
  review ran (2026-07-07/08, four agents) and its findings were dispositioned by the user
  (2026-07-11) — see the "Code-review disposition" section below. New ordering gates from
  the disposition: **Tasks A and B complete before capture Tasks 9–10; Tasks C and D
  complete before Task 16.** Tasks 1–5 remain free to start first. **A30 addition
  (2026-07-12): Task E (deploy isolation) completes before Task 16's go/no-go AND before
  any Plan B code deploys to homelab** — once paper trading runs, all training-side
  deploys go through the trainer service, never the trader.
- **Key rotation: COMPLETED (user-confirmed 2026-07-07).** Task 12's runtime is
  unblocked in principle; the flag still defaults false until Task 16's go/no-go. Task 15
  Step 2 verifies the old keys were *revoked*, not just replaced.
- 🛑 **Live-DB gate:** every migration against live pg16 runs only at deployment, under
  the standing backup gate (plan-mode approval + verified backup). Dev/CI use
  `swingrl_test`.
- CLAUDE.md rules bind throughout: no hardcoded symbols/paths/amounts (use
  `SwingRLConfig`); `load_config()` only; UTC everywhere; broker middleware only; typed
  `SwingRLError` subclasses; structlog kwargs (never f-strings); TDD — RED commit before
  GREEN; never `--no-verify`; line length 100; `from __future__ import annotations`.
- **Branch strategy (user-directed 2026-07-07):** `swingrl/2.R-training-redesign` is the
  long-lived redesign **integration branch** — it merges to `main` only when the entire
  training-system redesign is complete and verified working. Plan A work happens on
  `swingrl/2.R-A-capture-foundation`, branched **from the integration branch**; any Plan-A
  PR targets the integration branch, never `main`.
- Tests requiring Postgres follow the repo pattern: module-level
  `pytest.mark.skipif(not os.environ.get("DATABASE_URL"), ...)` (see
  `tests/data/test_db.py:21–24`); GHA stays red per issue #18 — homelab CI is the gate.
- **Paper trading is not declared ready until Task 16's end-to-end verification passes**
  (Discord alerts proven, circuit breakers proven to trip, capture rows verified, sim
  fidelity audit accepted, CVE audit clean or triaged, security checklist executed).
- **Quiet window (added 2026-07-14, master-sequence reconciliation):** once
  `swingrl-collector` is live (Track C, Wave 1 — before this plan's main body), no
  container recreation and no homelab CI runs spanning **15:30–16:45 ET** on trading
  days. Prerequisite already landed in Wave 0: `ci-homelab.sh` cleanup is scoped to the
  dev compose project (an unscoped `docker compose down` would kill always-on services
  on every CI run). Task 11 is amended (2026-07-14): its calendar jobs register in the
  collector's scheduler, adding a dependency on the collector being deployed.

## Verified change-site register (2026-07-07, read from code + live pg16)

| Site | Fact | Status |
|---|---|---|
| `src/swingrl/data/postgres_schema.py:808` | `init_postgres_schema()` — all legacy DDL, applied at startup via `db.init_schema()` (`scripts/main.py:246`) | VERIFIED |
| `services/memory/db.py:88–210` | Redundant second DDL copy for 7 memory tables (startup race documented for Plan B cutover) | VERIFIED |
| `src/swingrl/execution/pipeline.py:127–343` | `execute_cycle`; per-algo `actions` (:205–209), `weights` (:213), `blended_actions` (:218), `target_weights` (:223) in scope | VERIFIED |
| `src/swingrl/execution/pipeline.py:537` | F1: `SELECT PERCENTILE_CONT(0.9) ... ORDER BY turbulence` — column absent from `features_equity/crypto` (`postgres_schema.py:224–266`); bare except → 0.0 (:541–544); guard `:517` disables halt | VERIFIED |
| `training/data_loader.py:237,350` + `envs/base.py:359–365` | F1b: training obs turbulence hardcoded 0.0, no overrides; live obs get real values (`features/pipeline.py:363,407`) | VERIFIED |
| `src/swingrl/execution/fill_processor.py:115–143` | `_record_trade` — sole signal-trade writer; `:64–105` adjustment writer | VERIFIED |
| `src/swingrl/features/turbulence.py:47` | `BaseTurbulenceCalculator.compute_series(returns) -> np.ndarray` exists — F1 fix needs no schema change | VERIFIED |
| `model_metadata` (live pg16) | 13 cols, **no iteration column**; `model_id` = `{env}-v1.1.0-{algo}-{date}`; newest-per-algo read by `pipeline.py:410–445` | VERIFIED |
| Live pg16 | Postgres 16 → `UNIQUE NULLS NOT DISTINCT` available (needed by `calendar_events`) | VERIFIED |
| `scripts/ci-homelab.sh:47–54` | Tests run against `swingrl_test` DB inside persistent pg16 | VERIFIED |
| `monitoring/alerter.py` + `scripts/main.py:248–255` | Discord Alerter wired in production execution path (end-to-end proof = Task 16) | VERIFIED |
| `pyproject.toml:31` | `alpaca-py>=0.20` — floating lower bound, **no upper pin**; breaking SDK changes can arrive silently | VERIFIED |
| `pyproject.toml` / `scripts/ci-homelab.sh` | `bandit` runs in CI (static code security) but **no dependency CVE scanner** (`pip-audit`/`osv`) exists anywhere | VERIFIED |
| `execution/adapters/binance_sim.py` | Crypto fills are simulated in-process (`trade_id=uuid4`, `:94,160`) — no Binance SDK dependency exists; Binance.US is only called by data ingestion | VERIFIED |

## Assumptions register — each with its concrete verification method

| # | Assumption | Confidence | How it gets verified (task-wired, not hand-waved) |
|---|---|---|---|
| P-A1 | Era-0 model bootstrap sentinels: `seed=-1`, `iteration_number=-1`, `code_version='unknown_era0'`, `data_fingerprint='unknown_era0'` for back-filled `training_runs` rows (legacy models' true values are unrecoverable) — **PARTIALLY SUPERSEDED (user-approved 2026-07-12):** `models/iterations/iter_{0..5}/` on homelab makes deployed-model vintages recoverable, and era-0 seeds were per-algo constants (42/43/44) — Task 5 now stamps **real** iteration/seed and selects the **best-CPS vintage** (crypto iter 0, equity iter 4); sentinels remain the fallback for genuinely unresolvable pairs; `code_version`/`data_fingerprint` sentinels stand (still unrecoverable) | **SIGNED OFF (2026-07-07); amended 2026-07-12** | Task 5 test asserts real identity on resolvable pairs + sentinel fallback path; grep-audit step unchanged: no new code compares/does arithmetic on `iteration_number`/`seed` without handling `-1` |
| P-A2 | `gate_versions` needs a **surrogate PK** (`gate_version_id`) — the spec's `gate_version SMALLINT PK` cannot hold per_fold v0 and ensemble v0 simultaneously. Proposed as **amendment A29** | High (schema arithmetic) | Provable by construction: Task 2's migration test inserts both v0 rows — under the spec's original PK this collides. Needs only the A29 sign-off |
| P-A3 | Zeroing the raw obs turbulence slot *before* per-algo VecNormalize reproduces the training distribution for era-0 models (they trained on raw 0.0) | High | **Empirical, homelab, real artifacts (Task 7 Step 3b):** inspect each deployed `vec_normalize.pkl` — `obs_rms.mean[turb_idx]` must be ≈0 and `obs_rms.var[turb_idx]` ≈ epsilon (proof the dim never varied in training); then assert normalized obs + predicted action for a zeroed-slot input match the training-world case |
| P-A4 | FRED release IDs: CPI=10, Employment Situation (NFP)=50, GDP=53; FOMC not in FRED → yaml-seeded schedule | Medium | Task 11 Step 0: one `curl "https://api.stlouisfed.org/fred/release?release_id={id}&api_key=$FRED_API_KEY&file_type=json"` per ID — response `name` must match the expected release; recorded in the test file header |
| P-A5 | Decision price for `fill_quality` = the `adapter.get_current_price()` value used for sizing (`pipeline.py:258`) | High | Execution-path review confirms what `get_current_price` returns per broker (Alpaca: quote vs last trade; sim: bar source); Task 10 test asserts captured `decision_price_usd` == the sizing price of that same order |
| P-A6 | Release times (FRED gives dates only): CPI/NFP/GDP print at 08:30 ET; FOMC statement at 14:00 ET | High | Task 11: spot-check one historical print per event type against the publishing agency's archive before hardcoding; times are config per event_type so corrections are one-line |

## Code-review disposition (2026-07-11, user-approved)

The execution-path review (findings + evidence:
`docs/superpowers/reviews/2026-07-07-execution-path-code-review.md`) found 2 critical +
5 high defects. User-approved disposition — finding labels (C/H/M/L = severity) are
defined in the review doc's glossary:

| Destination | Findings |
|---|---|
| **Task A — Real portfolio valuation** | C1 (circular snapshots — breakers can never trip), M4 (cash math), global-breaker high-water mark in-memory |
| **Task B — Fill lifecycle + schedule** | C2 (fire-and-forget fills + after-close cron → phantom $0 trades), M11 (FillResult has no timestamps), M10-equity (alert on record-after-fill failure) |
| **Task C — Risk-layer honesty** | H4 (post-halt ramp never enforced), M1 (breaker trips never alerted), H5-minimal (crypto stop-loss: correct book + alert + record; auto-sell stays deferred, documented) |
| **Task D — Model-loading hygiene** | H2 (promotion layout mismatch), H3 (cache never invalidated + empty-cache poison), M5 (blend crash/drift), M7 (silent raw obs on missing VecNormalize), M3 (emergency-sell ghosts), hardcoded-value sweep |
| **Existing-task folds** | Task 6: M6 (single bounded turbulence compute). Task 9: M2-capture, canonical timestamp, early-exit capture, dry-run tag. Task 10: P-A5 resolution, M11 consumption. Task 13: sim-fidelity gap list, IEX-feed check, slippage-model outcome decision |
| **Plan B inputs** | M8 (DDL type divergence → cutover runbook), M9 (unseeded eval env → seed-pinning task), A3 (risk penalty discarded — confirmed live; L1-harness precondition), all 11 training-side verdicts (review doc §6) |
| **Documented deferrals** | Crypto stop auto-sell execution; crypto reconciliation (only meaningful against real Binance) |

**User decisions locked with the disposition:** equity cycle moves to **~15:45 ET (before
close, weekdays, market-calendar-gated)**; fill confirmation = short post-submit polling
(follows from the timing choice — no persistent order stream needed at daily cadence).

**Ordering:** Tasks A/B before Tasks 9–10 (capture must not record fictions); Tasks C/D
before Task 16 (the go/no-go must test working breakers).

---

### Task 0: Preconditions gate (no code)

**Files:** none.

- [x] **Step 1: DONE 2026-07-11.** The execution/inference-path code review ran
  (2026-07-07/08; findings doc:
  `docs/superpowers/reviews/2026-07-07-execution-path-code-review.md`) and the user
  approved the disposition (2026-07-11): 2 critical + 5 high findings become Tasks A–D
  below; folds into Tasks 6/9/10/13; the rest → Plan B inputs or documented deferrals.
  Remaining gates: Tasks A/B before Tasks 9–10; Tasks C/D before Task 16.
- [x] **Step 2:** Key rotation — **COMPLETED, user-confirmed 2026-07-07.** Task 15
  Step 2 still verifies the old keys were revoked.
- [ ] **Step 3:** Create branch `swingrl/2.R-A-capture-foundation` from
  `swingrl/2.R-training-redesign` (the integration branch — see Global Constraints).

### Task 1: Migration runner + `schema_migrations` ledger

**Files:**
- Create: `src/swingrl/data/migration_runner.py`
- Create: `src/swingrl/data/migrations/` (SQL files live here; V001 arrives in Task 2)
- Create: `scripts/apply_migrations.py`
- Test: `tests/data/test_migration_runner.py`

**Interfaces:**
- Produces: `apply_migrations(db: DatabaseManager, migrations_dir: Path | None = None) -> int`
  (count of newly applied); `assert_schema_current(db: DatabaseManager) -> None` with
  **floor semantics (A30, 2026-07-12)**: raises `ConfigError` when the DB version is
  **behind** `EXPECTED_SCHEMA_VERSION` (missing migrations — genuinely broken);
  **warns-and-returns when the DB is ahead** (newer additive migrations applied by a
  trainer-side deploy — the running trader must survive its next restart); module
  constant `EXPECTED_SCHEMA_VERSION: int` (bumped by every task that adds a migration
  file); file naming contract `V{NNN}__{slug}.sql`.
- Consumes: `DatabaseManager.connection()` (`src/swingrl/data/db.py`), `ConfigError`
  (`src/swingrl/utils/exceptions.py`).

- [ ] **Step 1: Write the failing tests**

```python
"""Migration runner tests.

D-T3.1/A7b: versioned ledger; which DDL is this database running becomes queryable.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from swingrl.utils.exceptions import ConfigError

pytestmark = pytest.mark.skipif(
    not os.environ.get("DATABASE_URL"),
    reason="DATABASE_URL not set — no PostgreSQL available for testing",
)


@pytest.fixture
def migrations_dir(tmp_path: Path) -> Path:
    d = tmp_path / "migrations"
    d.mkdir()
    (d / "V001__widgets.sql").write_text(
        "CREATE TABLE IF NOT EXISTS _mig_test_widgets (id BIGINT PRIMARY KEY);"
    )
    (d / "V002__widgets_name.sql").write_text(
        "ALTER TABLE _mig_test_widgets ADD COLUMN name TEXT;"
    )
    return d


def test_apply_migrations_applies_in_order_and_records(db, migrations_dir: Path) -> None:
    """A7b: runner applies V-files in order and records each in schema_migrations."""
    from swingrl.data.migration_runner import apply_migrations

    applied = apply_migrations(db, migrations_dir=migrations_dir)
    assert applied == 2
    with db.connection() as conn:
        rows = conn.execute(
            "SELECT version, description FROM schema_migrations ORDER BY version"
        ).fetchall()
    assert [r["version"] for r in rows] == [1, 2]


def test_apply_migrations_is_idempotent(db, migrations_dir: Path) -> None:
    """Re-running applies nothing new."""
    from swingrl.data.migration_runner import apply_migrations

    apply_migrations(db, migrations_dir=migrations_dir)
    assert apply_migrations(db, migrations_dir=migrations_dir) == 0


def test_assert_schema_current_raises_on_stale(db, migrations_dir: Path, monkeypatch) -> None:
    """Merged ≠ deployed guard: stale schema refuses to run."""
    import swingrl.data.migration_runner as mr

    mr.apply_migrations(db, migrations_dir=migrations_dir)
    monkeypatch.setattr(mr, "EXPECTED_SCHEMA_VERSION", 99)
    with pytest.raises(ConfigError):
        mr.assert_schema_current(db)
```

Add a `db` fixture to this module mirroring `tests/data/test_db.py`'s pattern (build a
`DatabaseManager` from a tmp config whose `system.database_url` is `DATABASE_URL`), and a
teardown dropping `_mig_test_widgets` + the test's `schema_migrations` rows.

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/data/test_migration_runner.py -v`
Expected: FAIL — `ModuleNotFoundError: swingrl.data.migration_runner`

- [ ] **Step 3: Implement `src/swingrl/data/migration_runner.py`**

```python
"""Versioned SQL migration runner (spec §4.1 A7b — schema_migrations ledger).

Migrations are files named V{NNN}__{slug}.sql in src/swingrl/data/migrations/.
Each file is applied in one transaction and recorded in schema_migrations.
Legacy tables continue to come from postgres_schema.init_postgres_schema();
all NEW (Stage 2.R) tables arrive only through this runner.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import TYPE_CHECKING

import structlog

from swingrl.utils.exceptions import ConfigError, DataError

if TYPE_CHECKING:
    from swingrl.data.db import DatabaseManager

log = structlog.get_logger(__name__)

# Bumped by every task that ships a new V{NNN} file. Deployed containers refuse
# to start against a database whose ledger does not match (assert_schema_current).
EXPECTED_SCHEMA_VERSION = 0  # becomes 1 in Task 2, 2 in Task 4, 3 in Task 8, 4 in Task 12

MIGRATIONS_DIR = Path(__file__).parent / "migrations"
_FILE_RE = re.compile(r"^V(\d{3})__[a-z0-9_]+\.sql$")

_LEDGER_DDL = (
    "CREATE TABLE IF NOT EXISTS schema_migrations ("
    " version SMALLINT PRIMARY KEY,"
    " description TEXT NOT NULL,"
    " applied_at TIMESTAMPTZ NOT NULL DEFAULT now())"
)


def _discover(migrations_dir: Path) -> list[tuple[int, str, Path]]:
    """Return sorted (version, description, path); raise DataError on bad names/gaps."""
    found: list[tuple[int, str, Path]] = []
    for path in sorted(migrations_dir.glob("V*.sql")):
        m = _FILE_RE.match(path.name)
        if not m:
            raise DataError(f"Bad migration filename: {path.name}")
        found.append((int(m.group(1)), path.stem.split("__", 1)[1], path))
    versions = [v for v, _, _ in found]
    if versions != sorted(set(versions)):
        raise DataError(f"Duplicate migration versions: {versions}")
    return found


def apply_migrations(db: DatabaseManager, migrations_dir: Path | None = None) -> int:
    """Apply all unapplied migrations in version order. Returns count applied."""
    mdir = migrations_dir or MIGRATIONS_DIR
    applied = 0
    with db.connection() as conn:
        conn.execute(_LEDGER_DDL)
        done = {
            r["version"]
            for r in conn.execute("SELECT version FROM schema_migrations").fetchall()
        }
    for version, description, path in _discover(mdir):
        if version in done:
            continue
        sql = path.read_text()
        with db.connection() as conn:  # one transaction per migration
            conn.execute(sql)
            conn.execute(
                "INSERT INTO schema_migrations (version, description) VALUES (%s, %s)",
                (version, description),
            )
        log.info("migration_applied", version=version, description=description)
        applied += 1
    return applied


def current_schema_version(db: DatabaseManager) -> int:
    """Highest applied version; 0 if the ledger is empty/absent."""
    with db.connection() as conn:
        conn.execute(_LEDGER_DDL)
        row = conn.execute("SELECT max(version) AS v FROM schema_migrations").fetchone()
    return int(row["v"] or 0)


def assert_schema_current(db: DatabaseManager) -> None:
    """Refuse to run against a stale schema (the merged-≠-deployed guard)."""
    actual = current_schema_version(db)
    if actual != EXPECTED_SCHEMA_VERSION:
        log.error(
            "schema_version_mismatch", expected=EXPECTED_SCHEMA_VERSION, actual=actual
        )
        raise ConfigError(
            f"Database schema version {actual} != expected {EXPECTED_SCHEMA_VERSION}; "
            "run scripts/apply_migrations.py before starting."
        )
    log.info("schema_version_ok", version=actual)
```

Note: `DatabaseManager.connection()` semantics (autocommit vs transaction) must be
confirmed at implementation time — if it is autocommit, wrap the per-migration block in
an explicit transaction.

- [ ] **Step 4: Implement `scripts/apply_migrations.py`**

```python
"""Apply pending Stage 2.R schema migrations. Usage: python scripts/apply_migrations.py [config]"""

from __future__ import annotations

import sys

from swingrl.config.schema import load_config
from swingrl.data.db import DatabaseManager
from swingrl.data.migration_runner import apply_migrations, current_schema_version
from swingrl.utils.logging import configure_logging


def main() -> int:
    config = load_config(sys.argv[1] if len(sys.argv) > 1 else "config/swingrl.yaml")
    configure_logging(json_logs=config.logging.json_logs, log_level=config.logging.level)
    db = DatabaseManager(config)
    n = apply_migrations(db)
    print(f"applied={n} current_version={current_schema_version(db)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 5: Run tests to verify pass**

Run: `DATABASE_URL=postgresql://swingrl:...@localhost:5432/swingrl_test uv run pytest tests/data/test_migration_runner.py -v`
Expected: PASS (all 3)

- [ ] **Step 6: Commit**

```bash
git add src/swingrl/data/migration_runner.py scripts/apply_migrations.py tests/data/test_migration_runner.py
git commit -m "feat(2.R-A): versioned migration runner + schema_migrations ledger (A7b)"
```

### Task 2: V001 — registries + era-0/gate-v0 bootstrap + 574-row back-stamp

**Files:**
- Create: `src/swingrl/data/migrations/V001__registries_era0.sql`
- Modify: `src/swingrl/data/migration_runner.py` (EXPECTED_SCHEMA_VERSION = 1)
- Test: `tests/data/test_migrations_content.py`

**Interfaces:**
- Produces: tables `gate_versions` (surrogate `gate_version_id` PK — assumption P-A2 /
  proposed A29), `eras`; columns `backtest_results.era_id/gate_version_id`,
  `iteration_results.era_id/gate_version_ensemble_id` (all DEFAULT 0 → the back-stamp).
- Consumes: Task 1 runner.

- [ ] **Step 1: Write the failing test**

```python
"""V001: registries exist; era 0 seeded; 574 legacy rows back-stamped era 0."""

from __future__ import annotations

import os

import pytest

pytestmark = pytest.mark.skipif(
    not os.environ.get("DATABASE_URL"),
    reason="DATABASE_URL not set — no PostgreSQL available for testing",
)


def test_v001_era0_bootstrap(db_with_legacy_schema) -> None:
    """D-T3.4/A7: era 0 + gate v0 rows exist; legacy result rows stamped era 0."""
    from swingrl.data.migration_runner import apply_migrations

    apply_migrations(db_with_legacy_schema)
    with db_with_legacy_schema.connection() as conn:
        era = conn.execute("SELECT * FROM eras WHERE era_id = 0").fetchone()
        assert era is not None and era["first_iteration"] == 0
        gates = conn.execute(
            "SELECT gate_type FROM gate_versions WHERE version_number = 0 ORDER BY gate_type"
        ).fetchall()
        assert [g["gate_type"] for g in gates] == ["ensemble", "per_fold"]
        stamped = conn.execute(
            "SELECT count(*) AS n FROM backtest_results WHERE era_id = 0"
        ).fetchone()
        total = conn.execute("SELECT count(*) AS n FROM backtest_results").fetchone()
        assert stamped["n"] == total["n"]
```

`db_with_legacy_schema` fixture: `DatabaseManager` on `DATABASE_URL` with
`init_schema()` already run (legacy tables exist, possibly empty — the stamp must hold
for 0 rows in CI and 564 in production).

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/data/test_migrations_content.py::test_v001_era0_bootstrap -v`
Expected: FAIL — relation "eras" does not exist

- [ ] **Step 3: Write `V001__registries_era0.sql`**

```sql
-- Spec §4.1 (D-T3.4, A7, A7b). Registries are written ONLY by human-approved migrations.

CREATE TABLE gate_versions (
    gate_version_id     SMALLINT PRIMARY KEY,          -- surrogate (P-A2 / proposed A29)
    gate_type           TEXT NOT NULL CHECK (gate_type IN ('per_fold', 'ensemble')),
    version_number      SMALLINT NOT NULL,
    definition          JSONB NOT NULL,                -- machine-readable rules, units per key
    derivation_evidence TEXT,
    approved_by         TEXT NOT NULL,
    approved_at         TIMESTAMPTZ NOT NULL,
    UNIQUE (gate_type, version_number)
);

CREATE TABLE eras (
    era_id                 SMALLINT PRIMARY KEY,
    reason                 TEXT NOT NULL,
    gate_version_per_fold  SMALLINT NOT NULL REFERENCES gate_versions (gate_version_id),
    gate_version_ensemble  SMALLINT NOT NULL REFERENCES gate_versions (gate_version_id),
    first_iteration        SMALLINT NOT NULL,
    started_at             TIMESTAMPTZ NOT NULL DEFAULT now()
);
-- A7 monotonicity: enforced procedurally — the runner is the only writer and each new
-- era migration must assert first_iteration > max(existing). (CHECK cannot see other rows.)

INSERT INTO gate_versions
    (gate_version_id, gate_type, version_number, definition, derivation_evidence, approved_by, approved_at)
VALUES
    (0, 'per_fold', 0,
     '{"schema_version": 1, "sharpe_min": 0.7, "mdd_max_frac": 0.15, "profit_factor_min": 1.5, "overfitting_gap_max": 0.20}',
     'Pre-CPS legacy per-fold gate (spec §2.8); retro-registered at era-0 bootstrap',
     'era0-bootstrap-migration', now()),
    (1, 'ensemble', 0,
     '{"schema_version": 1, "sharpe_min": 1.0, "mdd_abs_max_frac": 0.15}',
     'Pre-CPS legacy ensemble gate (spec §2.8); retro-registered at era-0 bootstrap',
     'era0-bootstrap-migration', now());

INSERT INTO eras (era_id, reason, gate_version_per_fold, gate_version_ensemble, first_iteration)
VALUES (0, 'Pre-redesign legacy: iterations 0-4 under the pre-CPS gate (D-T3.18 kept evidence)', 0, 1, 0);

-- Back-stamp the kept rows (574 in production; count-agnostic by design).
ALTER TABLE backtest_results
    ADD COLUMN era_id SMALLINT NOT NULL DEFAULT 0 REFERENCES eras (era_id),
    ADD COLUMN gate_version_id SMALLINT NOT NULL DEFAULT 0 REFERENCES gate_versions (gate_version_id);
ALTER TABLE iteration_results
    ADD COLUMN era_id SMALLINT NOT NULL DEFAULT 0 REFERENCES eras (era_id),
    ADD COLUMN gate_version_ensemble_id SMALLINT NOT NULL DEFAULT 1 REFERENCES gate_versions (gate_version_id);
```

Set `EXPECTED_SCHEMA_VERSION = 1` in `migration_runner.py`.

- [ ] **Step 4: Run tests to verify pass**

Run: `uv run pytest tests/data/test_migrations_content.py -v` — Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/swingrl/data/migrations/V001__registries_era0.sql src/swingrl/data/migration_runner.py tests/data/test_migrations_content.py
git commit -m "feat(2.R-A): V001 registries (eras, gate_versions) + era-0 bootstrap + legacy back-stamp"
```

### Task 3: Schema-fingerprint assertion at container start

**Files:**
- Modify: `scripts/main.py` (immediately after `db.init_schema()` at `:246`)
- Modify: `services/memory/db.py` (`init_db()` at `:277`)
- Test: `tests/test_smoke.py` (extend) / `tests/memory/` service test

**Interfaces:**
- Consumes: Task 1 `assert_schema_current`.
- Produces: both containers refuse to start on a stale schema.

- [ ] **Step 1: Failing test** — unit test with `make_mock_db` (from `tests/conftest.py:298`)
  where `schema_migrations` max returns a wrong version; assert `build_app` (or a thin
  wrapper around the new call) raises `ConfigError`.

```python
def test_main_refuses_stale_schema(monkeypatch) -> None:
    """A25 cutover: container start asserts ledger version (merged ≠ deployed guard)."""
    from swingrl.data import migration_runner as mr
    from swingrl.utils.exceptions import ConfigError

    db, _conn = make_mock_db(fetchone_returns=[{"v": 0}])
    monkeypatch.setattr(mr, "EXPECTED_SCHEMA_VERSION", 3)
    with pytest.raises(ConfigError):
        mr.assert_schema_current(db)
```

- [ ] **Step 2:** Run: `uv run pytest tests/test_smoke.py -k stale_schema -v` — Expected: FAIL
- [ ] **Step 3:** Wire `assert_schema_current(db)` into `scripts/main.py` right after
  `db.init_schema()`. For the memory service (cannot import `swingrl.*` — verified
  pattern at `services/memory/memory_agents/query.py:97`), add to `services/memory/db.py`:
  a module constant `_EXPECTED_SCHEMA_VERSION` (same value, comment cross-referencing
  `migration_runner.py`; duplication is the established pattern there, converges in the
  Stage 3 refactor) and a check in `init_db()` that queries
  `SELECT max(version) FROM schema_migrations` — same floor semantics: raise
  `RuntimeError` when behind, warn when ahead (A30).
- [ ] **Step 4:** Run: `uv run pytest tests/test_smoke.py -v` — Expected: PASS
- [ ] **Step 5: Commit** — `git commit -m "feat(2.R-A): schema-fingerprint assertion at container start"`

### Task 4: V002 — identity-spine subset (`training_runs`, `models`, `ensemble_weight_history`)

**Files:**
- Create: `src/swingrl/data/migrations/V002__spine_models.sql`
- Modify: `src/swingrl/data/migration_runner.py` (EXPECTED_SCHEMA_VERSION = 2)
- Test: `tests/data/test_migrations_content.py` (extend)

**Interfaces:**
- Produces: `training_runs` (spec §4.1 verbatim), `models`, `ensemble_weight_history`
  (spec §4.4; `intent_id` FK added later by V004).

- [ ] **Step 1: Failing test** — after `apply_migrations`, insert a `training_runs` row +
  `models` row; assert the UNIQUE spine constraint rejects a duplicate
  (iteration, env, algo, fold, run_type, attempt), and a second attempt row succeeds.

```python
def test_v002_spine_unique(db_with_legacy_schema) -> None:
    """D-T3.1: duplicates impossible; retries are new attempt rows."""
    from swingrl.data.migration_runner import apply_migrations

    apply_migrations(db_with_legacy_schema)
    ins = (
        "INSERT INTO training_runs (iteration_number, environment, algorithm, fold_number,"
        " run_type, seed, attempt, status, era_id, code_version, data_fingerprint)"
        " VALUES (5, 'equity', 'ppo', 0, 'reference', 42, %s, 'completed', 0, 'abc123', 'fp1')"
    )
    with db_with_legacy_schema.connection() as conn:
        conn.execute(ins, (1,))
        with pytest.raises(Exception):  # psycopg UniqueViolation
            conn.execute(ins, (1,))
        conn.execute(ins, (2,))  # new attempt OK
```

- [ ] **Step 2:** Run — Expected: FAIL (relation "training_runs" does not exist)
- [ ] **Step 3: Write `V002__spine_models.sql`**

```sql
-- Spec §4.1 (D-T3.1, A4/A5/A6/A12) + §4.4 models/ensemble_weight_history (D-T3.10, A22)

CREATE TABLE training_runs (
    run_pk            BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    iteration_number  SMALLINT NOT NULL,
    environment       TEXT NOT NULL CHECK (environment IN ('equity', 'crypto')),
    algorithm         TEXT NOT NULL CHECK (algorithm IN ('ppo', 'a2c', 'sac')),
    fold_number       SMALLINT NOT NULL,
    run_type          TEXT NOT NULL CHECK (run_type IN
        ('season', 'reference', 'harness_stage1', 'harness_stage2', 'final_train', 'l1_reearn_control')),
    seed              INTEGER NOT NULL,       -- -1 sentinel allowed for era-0 backfill only (P-A1)
    attempt           SMALLINT NOT NULL DEFAULT 1,
    status            TEXT NOT NULL CHECK (status IN ('running', 'completed', 'failed', 'aborted')),
    era_id            SMALLINT NOT NULL REFERENCES eras (era_id),
    code_version      TEXT NOT NULL,
    config_hash       TEXT,
    config_snapshot   JSONB,
    data_fingerprint  TEXT NOT NULL,
    started_at        TIMESTAMPTZ,
    finished_at       TIMESTAMPTZ,
    UNIQUE (iteration_number, environment, algorithm, fold_number, run_type, attempt)
);
CREATE INDEX idx_training_runs_iter_env_algo ON training_runs (iteration_number, environment, algorithm);
CREATE INDEX idx_training_runs_era ON training_runs (era_id);

CREATE TABLE models (
    model_id                       TEXT PRIMARY KEY,
    run_pk                         BIGINT NOT NULL REFERENCES training_runs (run_pk),
    artifact_path                  TEXT NOT NULL,
    vecnormalize_path              TEXT NOT NULL,
    artifact_sha256                TEXT,        -- NOT NULL from era 1 (A22); nullable for era-0 backfill
    vecnormalize_sha256            TEXT,
    training_window_start          DATE,
    training_window_end            DATE,
    converged_at_step              BIGINT,
    ensemble_weight_at_train_frac  DOUBLE PRECISION,
    status                         TEXT NOT NULL CHECK (status IN ('active', 'shadow', 'archived')),
    promoted_at                    TIMESTAMPTZ,
    created_at                     TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX idx_models_run ON models (run_pk);

CREATE TABLE ensemble_weight_history (
    id             BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    model_id       TEXT NOT NULL REFERENCES models (model_id),
    weight_frac    DOUBLE PRECISION NOT NULL,
    set_by         TEXT NOT NULL CHECK (set_by IN ('training', 'meta_trader', 'human')),
    intent_id      BIGINT,   -- FK to intent_records added by V004
    effective_from TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX idx_ewh_model_from ON ensemble_weight_history (model_id, effective_from DESC);
```

- [ ] **Step 4:** Run tests — Expected: PASS
- [ ] **Step 5: Commit** — `git commit -m "feat(2.R-A): V002 identity spine subset (training_runs, models, ensemble_weight_history)"`

### Task 5: Era-0 deployed-model bootstrap script — best-CPS vintage selection (AMENDED 2026-07-12)

> **Amendment (user-approved 2026-07-12).** Verified facts that reshaped this task:
> (a) `~/swingrl/models/iterations/iter_{0..5}/active/{env}/{algo}/model.zip` +
> `vec_normalize.pkl` exist for every iteration in the loader-expected layout — **model
> vintages are recoverable by directory**, partially invalidating P-A1's "unrecoverable"
> premise for the deployed set; (b) live `iteration_results.cps_v1_multiplicative` picks
> **crypto iter 0 (0.1531 — the uncoached baseline beat every coached season)** and
> **equity iter 4 (0.0153)** as the best era-0 vintages; (c) seeds are recoverable too —
> era-0 used per-algo constants (`SEED_MAP` = 42/43/44, `trainer.py:71`).
> Consequences: the bootstrap selects **best-CPS-per-env**, not newest-by-date; spine
> rows carry **real** `iteration_number` + `seed`; sentinels remain the fallback for any
> (env, algo) whose vintage genuinely can't be resolved. P-A1's sentinel machinery stays
> (the fallback path + the grep-audit are unchanged).

**Files:**
- Create: `scripts/migrations/bootstrap_era0_models.py`
- Test: `tests/data/test_bootstrap_era0_models.py`

**Interfaces:**
- Produces:
  - `BEST_ERA0_VINTAGE: dict[str, int] = {"crypto": 0, "equity": 4}` — module constant
    with the CPS evidence in a comment (values above, read from live pg16 2026-07-12);
    revisit only if `iteration_results` changes (it is frozen era-0 evidence, so it
    won't).
  - One `training_runs` row (`run_type='final_train'`, era 0, **real
    `iteration_number` from `BEST_ERA0_VINTAGE`**, **real `seed` from the era-0
    per-algo constants 42/43/44**, `fold_number = -1`,
    `code_version = 'unknown_era0'`, `data_fingerprint = 'unknown_era0'`) + one
    `models` row (+ initial `ensemble_weight_history` row, `set_by='training'`) per
    (environment, algorithm), pointing at
    `models/iterations/iter_{N}/active/{env}/{algo}/` artifacts.
  - **Deployment step (homelab, runbook'd in Task 17):** copy the selected vintage into
    `models/active/{env}/{algo}/` (both files) so the live loader serves the best-CPS
    set — replaces the implicit newest-by-date deployment. Recorded as an
    `operator_actions`-style note in the deployment runbook (the table itself is Plan
    B's V008).
  - Sentinel fallback preserved (P-A1): any (env, algo) missing from
    `models/iterations/` falls back to newest `model_metadata` + `-1` sentinels +
    warning log.
- Consumes: V002 tables; `models/iterations/` layout (verified on homelab 2026-07-12);
  live `model_metadata` (13 cols, fallback path only).

- [ ] **Step 1: Failing test** — fixture a `models/iterations/` tree (tmp_path) with
  iter_0 + iter_4 artifacts and a `model_metadata` fallback row; run
  `bootstrap_era0_models.main()`; assert: `models` rows point at the
  `BEST_ERA0_VINTAGE` artifacts per env; `training_runs` rows carry **real**
  `iteration_number` (0 for crypto pairs, 4 for equity pairs), `seed` ∈ {42, 43, 44}
  matching the algo, `era_id = 0`, `status = 'completed'`; an (env, algo) absent from
  the iterations tree falls back to sentinels with a warning; re-running is idempotent
  (ON CONFLICT DO NOTHING on `model_id`).
- [ ] **Step 2:** Run — Expected: FAIL (module missing)
- [ ] **Step 3: Implement** — core loop:

```python
# CPS evidence (live pg16 iteration_results, read 2026-07-12):
# crypto iter 0 = 0.1531 (baseline HPs — best era-0 crypto season);
# equity iter 4 = 0.0153. Frozen era-0 evidence; values never change.
BEST_ERA0_VINTAGE: dict[str, int] = {"crypto": 0, "equity": 4}
ERA0_SEED_MAP: dict[str, int] = {"ppo": 42, "a2c": 43, "sac": 44}  # trainer.py:71
SENTINEL = -1  # fallback only: vintage genuinely unresolvable (P-A1)

def _sha256(path: Path) -> str | None:
    if not path.exists():
        log.warning("artifact_missing_sha_skipped", path=str(path))
        return None
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()
```

For each (env, algo): resolve
`models/iterations/iter_{BEST_ERA0_VINTAGE[env]}/active/{env}/{algo}/` — if both files
exist, insert spine row with real `iteration_number`/`seed`; else fall back to newest
`model_metadata` (`SELECT DISTINCT ON (environment, algorithm) * FROM model_metadata
ORDER BY environment, algorithm, training_end_date DESC`) with `-1` sentinels + warning;
insert `models` row (`status='active'`) + `ensemble_weight_history` row from
`ensemble_weight`.

- [ ] **Step 4:** Run tests — Expected: PASS
- [ ] **Step 5: Commit** — `git commit -m "feat(2.R-A): era-0 bootstrap — best-CPS vintage selection (crypto iter 0, equity iter 4) + recoverable identity"`

### Task 6: F1 fix — turbulence baseline from `compute_series` (no schema change)

> **Method review RESOLVED (user adopted all recommendations, 2026-07-07).** Full memo:
> `.planning/research/turbulence-method-review.md`. This task is now a group of three
> sub-fixes (6a/6b/6c below) implementing the halt-side outcomes: equity hygiene fixes,
> crypto calculator replacement, and the F1 baseline plumbing with a **97th-percentile
> hard halt** (config-tunable). The era-1 observation-feature decomposition (magnitude +
> correlation surprise percentile-ranks) and the HMM-p_crisis redundancy check are
> **Plan B era-1 inputs**, recorded in the memo.
>
> - **6a — Equity hygiene** (`features/turbulence.py`): eigenvalue floor in
>   `_mahalanobis` (clip eigenvalues < 1e-3·λmax before inversion; the `abs()` becomes
>   an assert) + EWMA warm-start de-bias (divide stats by `1 − (1−α)^t`; kills the
>   verified ~15% post-warmup inflation). Tests: turbulence on synthetic correlated
>   returns is stable under a near-singular covariance; post-warmup series shows no
>   systematic early inflation vs a long-run baseline.
> - **6b — Crypto calculator replacement** (`features/turbulence.py`): new
>   `CryptoTurbulenceCalculator` = 2-asset EWMA-Mahalanobis (same class as equity,
>   half-life config ~750 4H bars) OR-gated with a **signed** realized-vol percentile;
>   `MIN_WARMUP` 360 → 1080 bars (config). Tests: a dead-calm stretch scores LOW (kills
>   the abs(vol-z) defect); a correlation flip +0.8→−0.8 scores HIGH (kills the
>   abs(corr) defect); both verified against the old composite's wrong answers.
> - **6c — F1 baseline plumbing** (below, updated): percentile from
>   `config.features.turbulence_halt_percentile` (default **0.97**), computed over a
>   trailing lookback (`turbulence_baseline_lookback_days`: 756 equity / full history
>   crypto), recomputed per cycle date, cached.

**Files:**
- Modify: `src/swingrl/features/turbulence.py` (6a hygiene + 6b crypto replacement)
- Modify: `src/swingrl/features/pipeline.py` (new public baseline method, 6c)
- Modify: `src/swingrl/execution/pipeline.py:524–544` (`_get_turbulence_90th_pct` → delegation)
- Modify: `src/swingrl/config/schema.py` + `config/swingrl.yaml` (`turbulence_halt_percentile: 0.97`, baseline lookbacks, crypto half-life/warmup)
- Test: `tests/features/test_turbulence.py` (6a/6b), `tests/features/test_turbulence_baseline.py` (6c), `tests/execution/` regression

**Interfaces:**
- Produces: `FeaturePipeline.turbulence_halt_baseline(env_name: str, date_or_datetime: str) -> float`
  (0.0 only on genuine data absence, logged); `ExecutionPipeline._get_turbulence_90th_pct`
  delegates to it (broken SQL removed; keep the private name at the call site or rename
  with its callers in the same commit); rebuilt `CryptoTurbulenceCalculator` with the
  same `compute`/`compute_series`/`min_warmup` interface (drop-in).
- Consumes: `BaseTurbulenceCalculator.compute_series(returns) -> np.ndarray` + `min_warmup`
  (`features/turbulence.py:30,47`) — verified interface.
- **Review fold (2026-07-11, M6):** turbulence is computed **once per cycle** and the value
  reused for both the halt check and the observation (today: two divergent computations,
  one via a consume-once cache — `features/pipeline.py:557–559` — the other a recompute);
  the obs-path recompute's `SELECT ... FROM ohlcv_daily WHERE date <= X` gets a lower bound
  from the baseline-lookback config (today it scans the entire history every cycle,
  `features/pipeline.py:565–572,607–613`). The single per-cycle value is what Task 9
  captures.

- [ ] **Step 1: Failing tests** (6a/6b tests per the sub-task notes above, plus:)

```python
def test_turbulence_halt_baseline_nonzero_on_synthetic(feature_pipeline_with_ohlcv) -> None:
    """F1: historical baseline computed from OHLCV series, not the phantom column."""
    baseline = feature_pipeline_with_ohlcv.turbulence_halt_baseline("equity", "2026-07-07")
    assert baseline > 0.0


def test_check_turbulence_reaches_risk_manager(monkeypatch) -> None:
    """F1 regression: nonzero baseline must reach RiskManager.check_turbulence
    (pipeline.py:517 guard no longer short-circuits)."""
    # Build ExecutionPipeline with mocked feature pipeline returning
    # compute_turbulence=5.0 and turbulence_halt_baseline=2.0; mock RiskManager;
    # assert risk_manager.check_turbulence called with (env, 5.0, 2.0).
```

(`feature_pipeline_with_ohlcv`: fixture seeding `ohlcv_daily` in the test DB with ≥
`min_warmup + 10` bars of synthetic closes for the configured symbols — reuse
`equity_prices_array` from `tests/conftest.py:254`.)

- [ ] **Step 2:** Run — Expected: FAIL (method missing / check never called)
- [ ] **Step 3: Implement.** In `FeaturePipeline`, reuse the exact data-prep already used
  by the fallback path (`_compute_turbulence_equity`, `pipeline.py:563–578`): load
  closes ≤ date from `ohlcv_daily` (or `ohlcv_4h`), pivot by symbol, log-returns, then:

```python
def turbulence_halt_baseline(self, env_name: str, date_or_datetime: str) -> float:
    """Historical halt-percentile of the turbulence series (F1 fix, spec §5).

    Computed from the OHLCV-derived series via compute_series — the
    features_* tables never had a turbulence column. Percentile + lookback
    from config (method review 2026-07-07: hard halt 97th, trailing window).
    """
    calc = self._turb_equity if env_name == "equity" else self._turb_crypto
    pct = self._config.features.turbulence_halt_percentile  # default 0.97
    lookback = self._config.features.turbulence_baseline_lookback_bars(env_name)
    log_returns = self._load_log_returns(env_name, date_or_datetime)  # extracted helper
    if log_returns is None or len(log_returns) < calc.min_warmup + 2:
        log.warning("turbulence_baseline_insufficient_data", env=env_name)
        return 0.0
    series = calc.compute_series(log_returns.values)
    valid = series[calc.min_warmup :]
    valid = valid[np.isfinite(valid)][-lookback:]  # trailing window, not frozen
    if valid.size == 0:
        log.warning("turbulence_baseline_empty_series", env=env_name)
        return 0.0
    return float(np.percentile(valid, pct * 100.0))
```

Extract `_load_log_returns` from the duplicated pivot/log-return code in
`_compute_turbulence_equity/_crypto` (DRY). In `execution/pipeline.py`, replace the
`_get_turbulence_90th_pct` body with a delegation +
per-(env, date) cache dict `self._turb_baseline_cache: dict[tuple[str, str], float]`.
Delete the broken SQL and the bare `except` (narrow to `DataError`/`Exception` with
`log.error("turbulence_baseline_failed", ...)` — never silent).

- [ ] **Step 4:** Run: `uv run pytest tests/features/test_turbulence_baseline.py tests/execution/ -v` — Expected: PASS
- [ ] **Step 5: Commit** — `git commit -m "fix(2.R-A): turbulence method fixes (shrinkage, de-bias, crypto Mahalanobis) + F1 halt baseline live at 97th pct"`

### Task 7: F1b — zero the turbulence obs input for era-0 models

**Files:**
- Modify: `src/swingrl/features/assembler.py` (index helper)
- Modify: `src/swingrl/config/schema.py` (`environment.zero_turbulence_obs: bool = True`)
- Modify: `src/swingrl/execution/pipeline.py` (`execute_cycle`, after Step 3 obs fetch)
- Modify: `config/swingrl.yaml` + `config/swingrl.prod.yaml.example` (documented flag)
- Test: `tests/features/test_assembler.py` (extend), `tests/execution/`

**Interfaces:**
- Produces: `turbulence_obs_index(env_name: str, n_symbols: int, sentiment_enabled: bool) -> int`
  in `assembler.py` (derived from the existing layout constants
  `SHARED_MACRO`/`HMM_REGIME` — never a magic number); config flag
  `environment.zero_turbulence_obs` (default **true**; flipped to false when era-1 models
  deploy — Plan B automates this off the `models` table).
- Consumes: verified F1b evidence (era-0 models trained with the slot frozen at 0.0).

- [ ] **Step 1: Failing tests** — (a) index helper returns the slot whose feature name is
  `turbulence_index` in the assembler's name list for both envs; (b) with the flag on,
  the observation passed to `model.predict` has 0.0 at that slot even when the feature
  pipeline produced a nonzero turbulence; (c) the **captured** turbulence (Task 9 uses
  it) still carries the real value.
- [ ] **Step 2:** Run — Expected: FAIL
- [ ] **Step 3: Implement.** In `execute_cycle` after the obs-health check (~`:196`):

```python
        # F1b: era-0 models trained with the turbulence slot frozen at 0.0 —
        # feeding real values would multiply them by untrained weights (noise).
        # The real sensor value is read out FIRST for capture (§4.7 / A27).
        turb_idx = turbulence_obs_index(
            env_name, len(symbols_for(env_name, self._config)),
            self._config.equity.sentiment_enabled if env_name == "equity" else False,
        )
        turbulence_at_decision = float(observation[turb_idx])
        if self._config.environment.zero_turbulence_obs:
            observation[turb_idx] = 0.0
```

(Exact accessor names for symbols/sentiment flags verified at implementation time against
`schema.py`; the pattern — read real value, then zero — is the contract.)

- [ ] **Step 3b: P-A3 empirical verification (homelab, real artifacts).** For each
  deployed `models/active/{env}/{algo}/vec_normalize.pkl`: load it and assert the
  turbulence slot's running stats prove the dimension never varied in training —
  `obs_rms.mean[turb_idx]` ≈ 0.0 and `obs_rms.var[turb_idx]` ≤ epsilon. Then build one
  observation with a nonzero turbulence, zero the slot, normalize, and assert the
  normalized slot value and `model.predict` output are identical to the raw-0.0 case.
  Record results (per model) in the commit message. **If any model shows non-zero
  variance here, STOP — F1b's premise is wrong for that model; escalate to the user.**
- [ ] **Step 4:** Run tests — Expected: PASS
- [ ] **Step 5: Commit** — `git commit -m "fix(2.R-A): F1b — zero turbulence obs slot for era-0 models; keep real value for capture"`

### Task A: Real portfolio valuation (review C1, M4; global-breaker high-water mark)

> Fixes the review's C1: `portfolio_snapshots.total_value` is the previous snapshot copied
> forward (`position_tracker.py:47–72,162–196`), so drawdown/daily-loss breakers and the
> emergency trigger compare against a value that never moves. Prerequisite for Task 16's
> breaker proof and for any meaningful paper results. **Complete before Tasks 9–10.**

**Files:**
- Modify: `src/swingrl/execution/position_tracker.py` (`get_portfolio_value`,
  `record_snapshot`, `get_daily_pnl`)
- Modify: `src/swingrl/execution/pipeline.py` (snapshot block ~`:330–335`)
- Modify: `src/swingrl/execution/risk/circuit_breaker.py` (global high-water mark)
- Test: `tests/execution/test_position_tracker.py` (extend), `tests/execution/`

**Interfaces:**
- Produces: `PositionTracker.compute_portfolio_value(env: str, prices: dict[str, float])
  -> float` = Σ(position qty × current price) + cash; **cash is derived, never stored as a
  running balance**: initial capital (config `capital.{env}_usd`) + Σ signed fill flows −
  commissions from `trades` (append-only-friendly; one SQL aggregate). `daily_pnl` =
  today's value − last prior-day snapshot value. Global breaker high-water mark read from
  `MAX(total_value)` over persisted snapshots, not process memory.
- Consumes: the cycle's already-fetched `get_current_price` values (no extra API calls —
  pass the price map from `execute_cycle`).

- [ ] **Step 1: Failing tests** — (a) value moves when position prices move (no fills);
  (b) cash reflects buys negative / sells positive, commissions deducted; (c) daily_pnl
  compares against the previous *day*, not the previous cycle; (d) drawdown check trips
  at the configured threshold with a genuinely fallen portfolio value; (e) global HWM
  survives a simulated restart (re-read from snapshots).
- [ ] **Step 2:** Run — Expected: FAIL
- [ ] **Step 3: Implement.** Replace the `pipeline.py:330–335` snapshot block: compute
  value from the cycle's price map + derived cash; drop the broken
  `cash = value − Σ|qty×price|` line (M4). Keep snapshots append-only.
- [ ] **Step 4:** Run tests — Expected: PASS
- [ ] **Step 5: Commit** — `git commit -m "fix(2.R-A): mark-to-market portfolio valuation — breakers measure reality (review C1/M4)"`

### Task B: Fill lifecycle + schedule (review C2, M11; market calendar)

> Fixes the review's C2: submits are fire-and-forget (`alpaca_adapter.py:124–140`), the
> equity cron fires at 16:15 ET after close, 7 days/week (`main.py:73–81`), and unfilled
> orders are recorded as $0 trades that zero the position's price and trigger re-buys.
> **User decisions (2026-07-11): cycle → ~15:45 ET before close; polling, not websocket.**
> **Complete before Tasks 9–10.**

**Files:**
- Modify: `scripts/main.py` (equity cron from config), `src/swingrl/config/schema.py`
  (`equity.cycle_time_et: str = "15:45"`, `equity.market_calendar_gate: bool = true`),
  `config/swingrl.yaml` + `config/swingrl.prod.yaml.example`
- Modify: `src/swingrl/execution/types.py` (`FillResult` + `status:
  Literal["filled","pending","rejected"]`, `submitted_at`, `filled_at`)
- Modify: `src/swingrl/execution/adapters/alpaca_adapter.py` (post-submit poll),
  `src/swingrl/execution/pipeline.py` (only process filled results),
  `src/swingrl/execution/fill_processor.py` (reject qty=0/price=0 defensively)
- Test: `tests/execution/test_alpaca_adapter.py`, `tests/execution/test_fill_processor.py`

**Interfaces:**
- Produces: equity cycle at 15:45 ET **weekdays** (`day_of_week="mon-fri"`), gated by the
  Alpaca clock API (`get_clock`): market closed/holiday → skip + info log; clock call
  fails → skip + alert (fail-safe: when in doubt, don't trade). `submit_order` polls order
  status after submit (bounded: `order_fill_timeout_s: int = 60`, poll every 2s); filled →
  real fill price + `filled_at`; unfilled at timeout → **cancel + alert + return
  status="pending" — never a $0 trades row** (`fill_processor.process` drops non-filled
  results and `_record_trade` raises `DataError` on qty=0/price=0 as a backstop).
  M10-equity: if `process()` raises after a real fill, alert critical (trade executed but
  unrecorded).
- **Restart semantics (A30 addendum, user-approved 2026-07-12):**
  - `misfire_grace_time` on the cycle jobs — equity **720 s** (a restart shortly after
    15:45 still runs the cycle late; the clock gate + freshness guard make a late run
    safe, and past ~15:57 the graced window has lapsed → clean skip + log, never a
    post-close submit), crypto **3600 s** (4H cadence — an hour late is immaterial).
    Config: `scheduler.misfire_grace_s: dict[str, int] = {"equity": 720, "crypto": 3600}`
    — no hardcoded values. Missed-beyond-grace cycles are **skipped, never replayed**
    (documented behavior: at daily cadence a skipped cycle is one missed rebalance, and
    the next cycle's target-weight diff self-corrects).
  - **Startup reconciliation:** `scripts/main.py` runs the equity reconciliation job
    once at boot (same function the 17:00 ET cron calls) so any fill/position drift from
    downtime is audited immediately, not hours later. Skips with an info log when the
    market data needed isn't available (e.g. overnight restart).
- Consumes: Task 10 reads `submitted_at`/`filled_at` for `time_to_fill_ms`;
  `binance_sim` fills stay synchronous — it sets `status="filled"` + both timestamps;
  Task E's `deploy-process.md` references these restart semantics (trader restarts are
  safe by design: durable state in pg16, breaker state DB-derived, worst case = one
  graced-late or skipped cycle).

- [ ] **Step 1: Failing tests** — (a) submit that returns no fill → poll loop queried,
  then cancel + status="pending", no trades row; (b) synchronous fill → status="filled",
  timestamps set, trade recorded with the real price; (c) market-closed clock response →
  cycle skipped, no orders; (d) `_record_trade` raises `DataError` on a zero-quantity fill;
  (e) cron registration reads `equity.cycle_time_et` (no hardcoded hour); (f) cycle jobs
  registered with `misfire_grace_time` from `scheduler.misfire_grace_s` (equity 720,
  crypto 3600 under default config); (g) startup path invokes the equity reconciliation
  once before the scheduler starts (mocked job function called exactly once at boot);
  (h) a NO-FILL cycle still persists a portfolio snapshot whose `total_value` reflects
  moved prices; (i) a held-position crash on a no-fill cycle trips the drawdown breaker
  at that cycle's risk evaluation (amendment below).
- [ ] **Step 2:** Run — Expected: FAIL
- [ ] **Step 3: Implement.** Also verify at implementation (review honest-gap): OHLCV bar
  freshness at 15:45 ET — the cycle logs a warning when the latest `ohlcv_daily` bar is
  older than the previous trading day (data-freshness guard, log-only in this task).

> **Amendment (2026-07-16, user ruling — closes Task A review Q1):** the cycle persists a
> portfolio snapshot EVERY cycle (mark-to-market via `compute_portfolio_value` from the
> cycle's already-fetched price map) — the `if fills:` gate on the snapshot write is
> removed; and the pre-trade risk evaluation consumes the FRESH computed value, not the
> last stored snapshot, so a held-position drawdown with zero fills is visible to the
> drawdown/daily-loss breakers at every cycle. No extra broker API calls (prices come
> from the cycle's existing fetch). Tests (h)/(i) above pin both behaviors.
- [ ] **Step 4:** Run tests — Expected: PASS
- [ ] **Step 5: Commit** — `git commit -m "fix(2.R-A): honest fill lifecycle + 15:45 ET market-gated cycle (review C2/M11)"`

### Task C: Risk-layer honesty (review H4, M1, H5-minimal)

> Post-halt ramp capacity is logged but never applied (`risk_manager.py:182–192`); breaker
> trips/auto-resumes are never alerted (`circuit_breaker.py:189–208,129–132`); the crypto
> stop-poller only logs, and against the wrong book (`stop_polling.py:128` BTCUSD vs fills
> on BTCUSDT). **Complete before Task 16.**

**Files:**
- Modify: `src/swingrl/execution/risk_manager.py`, `src/swingrl/execution/risk/circuit_breaker.py`
- Modify: `src/swingrl/execution/stop_polling.py`
- Modify: `scripts/main.py` (inject `Alerter` into breakers/stop-poller)
- Test: `tests/execution/test_risk_manager.py`, `tests/execution/test_circuit_breaker.py` (extend)

**Interfaces:**
- Produces: RAMPING state actually scales orders —
  `dataclasses.replace(order, dollar_amount=order.dollar_amount * capacity_frac)` before
  validation; Discord alert on **every** breaker trip (env, global, turbulence) and on
  auto-resume, with state + trigger value in the embed; `stop_polling` reads config
  symbols verbatim (same USDT book as fills), and on breach: Discord alert + one
  `circuit_breaker_events`-style DB record (append-only). **Auto-sell on stop breach
  remains out of scope — documented in the module docstring with the risk statement**
  (paper-trading positions are not auto-protected; revisit before live).
- Consumes: `Alerter` (wired in `scripts/main.py:248–255`, verified).

- [ ] **Step 1: Failing tests** — (a) RAMPING at 25% halves-then-quarters a $1,000 order
  to $250 before the validator sees it; (b) breaker trip calls `alerter.send_alert`;
  (c) auto-resume alerts; (d) stop-poller queries the configured symbol string unchanged;
  (e) stop breach writes the DB record and alerts.
- [ ] **Step 2:** Run — Expected: FAIL
- [ ] **Step 3: Implement.**
- [ ] **Step 4:** Run tests — Expected: PASS
- [ ] **Step 5: Commit** — `git commit -m "fix(2.R-A): enforce ramp capacity; alert breaker trips; crypto stops on the right book (review H4/M1/H5)"`

### Task D: Model-loading hygiene (review H2, H3, M5, M7, M3 + hardcode sweep)

> Promotion writes a flat layout the loader never reads (`lifecycle.py:102–186` vs
> `pipeline.py:364–365`); the model cache is never invalidated and caches empty forever
> (`pipeline.py:354–355,407`); blending crashes or silently drifts on partial loads
> (`ensemble.py:114`); missing VecNormalize silently feeds raw observations
> (`pipeline.py:379,492–496`); emergency crypto sells leave ghost rows and no trade record
> (`binance_sim.py:143–190`, `emergency.py:194,286–287`). **Complete before Task 16.**

**Files:**
- Create: `src/swingrl/execution/model_paths.py` (single source of the layout)
- Modify: `src/swingrl/execution/pipeline.py` (`_load_models`, min-order constant),
  `src/swingrl/shadow/lifecycle.py`, `src/swingrl/shadow/promoter.py`,
  `src/swingrl/execution/ensemble.py`, `src/swingrl/execution/adapters/binance_sim.py`,
  `src/swingrl/monitoring/emergency.py`, `src/swingrl/config/schema.py` (if
  `equity.min_order_usd` needs surfacing)
- Test: `tests/execution/test_model_loading.py`, `tests/shadow/` (extend)

**Interfaces:**
- Produces: `active_model_paths(models_dir: Path, env: str, algo: str) ->
  tuple[Path, Path]` (model.zip, vec_normalize.pkl) used by loader, lifecycle
  promote/archive/rollback (which now move **both** files per-algo), and
  `_verify_deployment`; cache keyed by artifact mtimes — changed mtime or previously-empty
  cache → reload; ensemble weights renormalized over actually-loaded algos (missing
  `model_metadata` row → warn + equal-share, never KeyError); missing/failed VecNormalize
  → **skip that algo + alert** (fail closed, per A22 direction); `emergency_sell` routes
  through `fill_processor` (trades row + position DELETE — no zero-qty ghosts; tier-4
  verification counts only `quantity > 0`); `$5` literal → `config.equity.min_order_usd`;
  `1/3` → named `DEFAULT_ENSEMBLE_WEIGHT` constant.
- Consumes: Task 5's `models` table is unaffected (disk layout only).

- [ ] **Step 1: Failing tests** — (a) promote → loader finds the new model at the per-algo
  path incl. vec_normalize; (b) touching model.zip mtime busts the cache; empty first load
  is retried next cycle; (c) 2-of-3 models loaded → weights renormalize to sum 1.0;
  (d) missing vec_normalize.pkl → algo excluded + alert, cycle proceeds with the rest;
  (e) emergency sell → trades row written, position row deleted, tier-4 reports
  all_closed=True; (f) min order honors config.
- [ ] **Step 2:** Run — Expected: FAIL
- [ ] **Step 3: Implement.**
- [ ] **Step 4:** Run tests — Expected: PASS
- [ ] **Step 5: Commit** — `git commit -m "fix(2.R-A): unified model layout + cache invalidation + fail-closed loading (review H2/H3/M5/M7/M3)"`

### Task E: Trader/trainer deploy isolation (A30, user-approved 2026-07-12)

Training deploys must never interrupt running paper trading. Verified basis: one
`swingrl` compose service runs the scheduler (`Dockerfile` CMD `python scripts/main.py`)
AND hosts training; code is baked into the image (bind mounts are data-only) — so today
an image rebuild + recreate kills the scheduler, and (post-Task D) a training write to
`models/active/` would hot-reload into the live trader.

**Files:**
- Modify: `docker-compose.yml`, `docker-compose.prod.yml`
- Create: `docs/training/deploy-process.md`
- Test: `tests/test_deploy_isolation.py` (new)

**Interfaces:**
- Produces:
  - Compose service **`swingrl-trader`** (renames the `swingrl` service; same build,
    scheduler CMD unchanged; `image:` pinned to an explicit tag, e.g.
    `swingrl:trader-YYYY-MM-DD`, bumped only by hand in a deploy window) and
    **`swingrl-trainer`** (same build context, `profiles: ["training"]` so plain
    `up -d` never starts or recreates it; command = idle entrypoint; training invoked
    via `docker compose run --rm swingrl-trainer python scripts/train_pipeline.py ...`).
    Container name `swingrl` is kept on the trader service so existing runbooks/`docker
    exec` references keep working.
  - `docs/training/deploy-process.md` — the standing process: training deploy = build +
    `compose run` (trader untouched); trading deploy = tag bump + recreate **only** in a
    market-safe window (equity: after the 15:45 ET cycle + fill polling complete, ~16:05
    ET or later; crypto: between 4H cycles) with a no-in-flight-cycle check
    (`status/` heartbeat + log tail documented); additive-only migration rule while the
    trader runs; rollback = re-pin the previous tag; **restart-semantics section** (per
    Task B's A30 addendum): durable state in pg16, breaker state DB-derived, jobstore
    persistent, misfire grace 720 s equity / 3600 s crypto, startup reconciliation at
    boot — worst case of any restart = one graced-late or cleanly-skipped cycle.
  - **Third service (amended 2026-07-14): `swingrl-collector`** — the deploy doc covers
    the full 3-service topology. Collector rules (from the options plan D9/C4): its own
    pinned image tag (Plan A/B image churn must never recreate it via bare `up -d` —
    service-scoped compose commands only); recreation only **outside 15:30–16:45 ET** on
    trading days (the shared quiet window — also protects the 15:45 equity cycle);
    restarts are safe by design (persistent jobstore + boot self-check: reconcile +
    lookback health check). The collector keeps running through trader deploys AND
    through Plan B's cutover (its tables are untouched by V010).
  - **Discord delivery wiring (amendment 2026-07-15, from T16 collector deploy — two
    latent gaps proven live; both bite when paper trading's notifications come into
    play, so they are fixed at container setup, verified at Task 16 Step 1):**
    1. **Webhook env:** the Alerter reads `config.alerting.alerts_webhook_url` /
       `daily_webhook_url` — populated only by the env overrides
       `SWINGRL_ALERTING__ALERTS_WEBHOOK_URL` / `SWINGRL_ALERTING__DAILY_WEBHOOK_URL`.
       The legacy `DISCORD_WEBHOOK_URL` in `.env` is read by NOTHING (it appears only in
       compose comments); with it alone, every alert boots disabled
       (`alert_disabled reason=no_webhook_url`) — this is why Discord was never proven
       end-to-end before 2026-07-15. Homelab `.env` now sets both overrides (fixed
       2026-07-15); this task's compose split must carry them into `swingrl-trader` and
       `swingrl-trainer` (shared `env_file: .env` already does — verify, don't assume).
    2. **INFO alerts never send:** `Alerter.send_daily_digest()` has ZERO production
       callers — INFO-level alerts buffer in memory forever and die on restart. The
       collector was fixed via `Alerter(info_immediate=True)` (2026-07-15, ~2 INFOs/day).
       The trader keeps digest semantics, so this task must wire a daily digest flush
       (e.g. end-of-day scheduler job calling `send_daily_digest()`) or consciously
       choose `info_immediate` for the trader too — decide here, prove at Task 16 Step 1
       (an INFO-path delivery is part of the Discord live proof, not just
       critical/warning embeds).
  - **`models/active/` write ban, tested**: a grep-based test (same pattern as the
    layout-constant audits) asserting no module under `src/swingrl/training/`,
    `src/swingrl/memory/training/`, or `scripts/train*` writes to `models/active` —
    the only writers of the active tree are the promotion/lifecycle module and the
    Task 5 bootstrap deployment step (both gated, documented).
- Consumes: Task D's `active_model_paths` + mtime-cache behavior (the reason the write
  ban is load-bearing); Task 3's floor-semantics assertion (the reason trainer-side
  migrations don't brick the trader).

- [ ] **Step 1: Failing tests** — (a) compose config test: `docker compose config`
  parses; `swingrl-trainer` carries the `training` profile; `swingrl-trader` has an
  explicit non-`latest` image tag; (b) the `models/active` grep ban (fails today only if
  a violation exists — if it passes immediately, commit it as the regression guard, RED
  step waived with a note).
- [ ] **Step 2:** Implement compose split + write `deploy-process.md`.
- [ ] **Step 3:** Verify on homelab (read-only to trading): `docker compose config`
  renders both services; `docker compose up -d` with the stack down starts trader only.
- [ ] **Step 4:** Run tests — PASS. Full suite green.
- [ ] **Step 5: Commit** — `git commit -m "feat(2.R-A): trader/trainer deploy isolation (A30) — compose split + deploy process + models/active write ban"`

### Task 8: V003 — trade-time tables (§4.7 + A27)

**Files:**
- Create: `src/swingrl/data/migrations/V003__trade_time.sql`
- Modify: `src/swingrl/data/migration_runner.py` (EXPECTED_SCHEMA_VERSION = 3)
- Test: `tests/data/test_migrations_content.py` (extend)

**Interfaces:**
- Produces: `inference_cycles` (incl. `turbulence` per A27), `cycle_algo_proposals`,
  `trades.cycle_id`, `fill_quality`, `calendar_events`, `event_outcomes`.

- [ ] **Step 1: Failing test** — apply migrations; assert `inference_cycles` has a
  `turbulence` column; `cycle_algo_proposals` UNIQUE(cycle_id, model_id) rejects a dup;
  `calendar_events` rejects a duplicate macro row **with NULL symbol** (the
  NULLS-NOT-DISTINCT check); `trades` has `cycle_id`.
- [ ] **Step 2:** Run — Expected: FAIL
- [ ] **Step 3: Write `V003__trade_time.sql`**

```sql
-- Spec §4.7 (D-T3.13/14/15) + A27 turbulence stamp.

CREATE TABLE inference_cycles (
    cycle_id           BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    environment        TEXT NOT NULL CHECK (environment IN ('equity', 'crypto')),
    mode               TEXT NOT NULL CHECK (mode IN ('paper', 'live')),
    cycle_ts           TIMESTAMPTZ NOT NULL,
    deployed_iteration SMALLINT,            -- derived/display only (A20)
    hmm_p_bull         DOUBLE PRECISION,    -- probability 0-1
    hmm_p_bear         DOUBLE PRECISION,    -- probability 0-1
    vix                DOUBLE PRECISION,    -- index points
    turbulence         DOUBLE PRECISION,    -- A27: decision-time sensor value (pre-zeroing)
    active_event_ids   BIGINT[],
    blended_actions    JSONB,               -- {"schema_version":1,"raw":{sym:val},"target_weights_frac":{sym:val}}
    created_at         TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX idx_inference_cycles_env_ts ON inference_cycles (environment, cycle_ts);

CREATE TABLE cycle_algo_proposals (
    id                   BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    cycle_id             BIGINT NOT NULL REFERENCES inference_cycles (cycle_id),
    model_id             TEXT NOT NULL REFERENCES models (model_id),
    algorithm            TEXT NOT NULL CHECK (algorithm IN ('ppo', 'a2c', 'sac')),
    proposed_actions     JSONB NOT NULL,    -- {"schema_version":1,"raw":{sym:val}} — same shape as blend
    weight_in_blend_frac DOUBLE PRECISION NOT NULL,  -- snapshotted (D-T3.13)
    UNIQUE (cycle_id, model_id)
);

ALTER TABLE trades ADD COLUMN cycle_id BIGINT REFERENCES inference_cycles (cycle_id);
CREATE INDEX idx_trades_cycle ON trades (cycle_id);

CREATE TABLE fill_quality (
    id                      BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    trade_id                TEXT NOT NULL UNIQUE REFERENCES trades (trade_id),
    decision_price_usd      NUMERIC(18, 8),
    expected_fill_price_usd NUMERIC(18, 8),
    fill_price_usd          NUMERIC(18, 8) NOT NULL,
    slippage_frac           DOUBLE PRECISION,
    expected_cost_frac      DOUBLE PRECISION,   -- snapshotted from config in force
    realized_cost_frac      DOUBLE PRECISION,
    time_to_fill_ms         INTEGER,
    created_at              TIMESTAMPTZ NOT NULL DEFAULT now()
);
COMMENT ON COLUMN fill_quality.slippage_frac IS
    'fraction, signed side-aware: positive = adverse to the order';

CREATE TABLE calendar_events (
    event_id     BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    event_type   TEXT NOT NULL CHECK (event_type IN ('fomc', 'cpi', 'nfp', 'gdp')),
    symbol       TEXT,                        -- NULL for macro (all current types)
    scheduled_at TIMESTAMPTZ NOT NULL,
    window_start TIMESTAMPTZ NOT NULL,        -- materialized at ingest (D-T3.14)
    window_end   TIMESTAMPTZ NOT NULL,
    importance   TEXT NOT NULL CHECK (importance IN ('high', 'medium', 'low')),
    source       TEXT NOT NULL,
    ingested_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE NULLS NOT DISTINCT (event_type, symbol, scheduled_at)   -- pg16; idempotent re-ingest
);
CREATE INDEX idx_calendar_events_sched ON calendar_events (scheduled_at);

CREATE TABLE event_outcomes (
    id          BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    event_id    BIGINT NOT NULL REFERENCES calendar_events (event_id),
    payload     JSONB NOT NULL,               -- units per key, schema_version inside
    recorded_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
```

- [ ] **Step 4:** Run tests — Expected: PASS
- [ ] **Step 5: Commit** — `git commit -m "feat(2.R-A): V003 trade-time capture tables (§4.7 + A27)"`

### Task 9: `CycleRecorder` + capture hooks in `execute_cycle`

**Files:**
- Create: `src/swingrl/execution/cycle_recorder.py`
- Modify: `src/swingrl/execution/pipeline.py` (`execute_cycle` Steps 3/7/8; constructor)
- Modify: `src/swingrl/features/pipeline.py` (public `regime_snapshot`)
- Test: `tests/execution/test_cycle_recorder.py`

**Interfaces:**
- Produces:
  - `RegimeStamp` frozen dataclass: `hmm_p_bull: float | None`, `hmm_p_bear: float | None`,
    `vix: float | None`, `turbulence: float | None`, `active_event_ids: list[int]`.
  - `CycleRecorder.record_cycle(*, env_name: str, mode: str, cycle_ts: datetime,
    regime: RegimeStamp, raw_actions: dict[str, list[float]], target_weights:
    dict[str, float], proposals: list[AlgoProposal], deployed_iteration: int | None)
    -> int | None` — returns `cycle_id`, or **None on any failure (fail-open: capture
    must never block the money path; failure logged `cycle_capture_failed` AND alerted
    via `Alerter.send_alert(level="warning", title="Cycle capture failed", ...)` — a
    silent capture outage must not survive a day unnoticed)**.
  - `AlgoProposal` dataclass: `algorithm: str`, `model_id: str`,
    `raw_actions: dict[str, float]`, `weight_in_blend_frac: float`.
  - `CycleRecorder.active_model_ids(env_name: str) -> dict[str, str]` — algo → model_id
    via `models ⋈ training_runs WHERE status='active' AND environment=%s`, cached per call.
  - `FeaturePipeline.regime_snapshot(env_name: str, date_or_datetime: str) -> dict[str, float | None]`
    — public wrapper over `_get_hmm_probs` (`features/pipeline.py:465–487`) + VIX from
    `_get_macro_array` (`:414–463`, element 0 = `VIXCLS`).
- Consumes: V003 tables; Task 7's `turbulence_at_decision`; Task 5's `models` rows.
- **Review folds (2026-07-11):** (a) ONE canonical `cycle_ts` captured at cycle start and
  reused everywhere (today `execute_cycle` calls `datetime.now(UTC)` several times);
  (b) **early-exit cycles still write their `inference_cycles` row** with a halt/skip
  reason (CB halt, turbulence halt, degraded features, NaN obs, zero portfolio —
  `pipeline.py:151–242` exits) — otherwise halted cycles are invisible to capture;
  (c) sub-minimum-delta skips (M2, `pipeline.py:252–253`) are recorded on the cycle
  payload (per-symbol skip reason) so "model signaled, order too small" is
  distinguishable from "model held"; (d) rows carry a `dry_run` tag (the `run_cycle.py`
  CLI is a second writer process); (e) no shared mutable per-cycle state on the pipeline
  object — equity and crypto cycles can overlap; (f) HMM/VIX values come from
  `regime_snapshot`, turbulence from Task 6's single per-cycle value — never a re-call of
  `compute_turbulence` (consume-once cache makes a third computation inconsistent).

- [ ] **Step 1: Failing tests** — with `make_mock_db`: (a) `record_cycle` issues the
  `inference_cycles` INSERT (RETURNING cycle_id) then one `cycle_algo_proposals` INSERT
  per proposal; (b) when the DB raises, `record_cycle` returns None and does not raise;
  (c) active-event stamping: events whose `window_start <= cycle_ts <= window_end` land
  in `active_event_ids`.
- [ ] **Step 2:** Run — Expected: FAIL
- [ ] **Step 3: Implement `cycle_recorder.py`** (single writer for both tables; every
  JSONB payload carries `schema_version: 1`; `json.dumps` with symbol keys from config).
  Wire into `execute_cycle`:
  - after Step 3 (obs) — build `RegimeStamp` (regime_snapshot + `turbulence_at_decision`
    from Task 7 + active events query),
  - after Step 8 (`target_weights`) — call `record_cycle(...)`; keep `cycle_id` for Task 10.
  - `mode` from `config.trading_mode`; `deployed_iteration` = max iteration over
    `active_model_ids` spine rows (display only, A20).
- [ ] **Step 4:** Run: `uv run pytest tests/execution/test_cycle_recorder.py -v` — Expected: PASS
- [ ] **Step 5: Commit** — `git commit -m "feat(2.R-A): CycleRecorder — per-cycle regime + per-algo proposal capture (fail-open)"`

### Task 10: `cycle_id` threading + `fill_quality` writer

**Files:**
- Modify: `src/swingrl/execution/pipeline.py` (Step 9 fill loop, `:292–297`)
- Modify: `src/swingrl/execution/fill_processor.py` (`process`, `_record_trade`; new
  `_record_fill_quality`)
- Test: `tests/execution/test_fill_processor.py` (extend)

**Interfaces:**
- Produces: `FillProcessor.process(fill, sized_order=None, *, cycle_id: int | None = None,
  decision_price: float | None = None) -> None`; `trades` INSERT gains `cycle_id`;
  one `fill_quality` row per signal fill (never for adjustments — `record_adjustment`
  unchanged, `cycle_id` stays NULL there).
- Consumes: Task 9's `cycle_id`; P-A5 **as resolved by the review (2026-07-11)**:
  `decision_price_usd` := the sizing-time `get_current_price()` value (`pipeline.py:258`).
  Column comments must record the nuances: equity submits by notional (the price converts
  dollars → quantity; Alpaca's source is the last IEX trade), and the crypto sim fetches a
  second mid-price for the fill, so decision ≠ fill by construction. `time_to_fill_ms`
  comes from Task B's `FillResult.submitted_at`/`filled_at` (M11).

- [ ] **Step 1: Failing tests** — (a) `_record_trade` INSERT includes `cycle_id`;
  (b) `fill_quality` row computes side-aware slippage:
  `slippage_frac = (fill - decision)/decision` for buys, `(decision - fill)/decision`
  for sells (positive = adverse); (c) `realized_cost_frac =
  commission/(fill_price*quantity) + max(slippage_frac, 0.0) `— formula stated in code
  comment; (d) adjustments produce no `fill_quality` row.
- [ ] **Step 2:** Run — Expected: FAIL
- [ ] **Step 3: Implement** — pipeline Step 9 passes
  `cycle_id=cycle_id, decision_price=current_price` into `process`; `_record_fill_quality`
  writes NUMERIC prices as `str(round(value, 8))` (psycopg adapts), `expected_cost_frac`
  from config where a modeled cost exists (crypto fee config; equity commission-free →
  0.0), `expected_fill_price_usd = decision_price * (1 + expected_cost_frac)` side-aware.
  Fail-open like Task 9 (capture error logged, fill processing unaffected).
- [ ] **Step 4:** Run tests — Expected: PASS
- [ ] **Step 5: Commit** — `git commit -m "feat(2.R-A): thread cycle_id through fills + fill_quality sidecar (§3.7.5)"`

### Task 11: Event-calendar ingest (FRED + FOMC yaml) + staleness alarm (AMENDED 2026-07-14)

> **Amendment (user-approved 2026-07-14, master-sequence reconciliation):** the ingest +
> staleness jobs register in the **`swingrl-collector` container's scheduler**
> (`scripts/collector_main.py` — Track C, deployed in Wave 1 before this task runs), NOT
> the trader's `scheduler/jobs.py`/`scripts/main.py`. Rationale: the collector is the
> market-data plane (options plan D10); homing ingest there means calendar-code updates
> never require trader rebuilds (A30). The ingestor code, config, tests, and backfill
> Step 0a are unchanged. The staleness alarm sends Discord via the collector's own
> `Alerter` (amended routing rule: trader scripts + collector send; memory never sends).
> **New dependency: options plan T13/T14 merged + `swingrl-collector` deployed.** The
> trader still *consumes* `calendar_events` at cycle time (Task 9) — reads only.

**Files:**
- Create: `src/swingrl/data/calendar.py`
- Modify: `src/swingrl/config/schema.py` (new `CalendarConfig` section)
- Modify: `config/swingrl.yaml`, `config/swingrl.prod.yaml.example`
- Modify: `scripts/collector_main.py` (register weekly ingest job + daily staleness check — amended 2026-07-14; was `scheduler/jobs.py` + `scripts/main.py`)
- Test: `tests/data/test_calendar.py`

**Interfaces:**
- Produces: `CalendarIngestor(config, db).run() -> int` (events upserted; logs one
  `data_ingestion_log` row with `environment='calendar'`, following `base.py:230`'s
  `_log_ingestion` pattern — standalone class, NOT a `BaseIngestor` subclass: the
  parquet-centric fetch/validate/store contract doesn't fit calendar data, deviation
  documented in the module docstring); `calendar_staleness_check(ctx) -> None` scheduler
  job (alerts when `max(scheduled_at) < now + config.calendar.min_future_days`).
- Config: `calendar.enabled: bool = true`, `calendar.fred_release_ids: dict[str, int] =
  {"cpi": 10, "nfp": 50, "gdp": 53}` (P-A4 — verify at implementation),
  `calendar.fomc_dates: list[str] = []` (ISO datetimes, seeded from the Fed's published
  schedule), `calendar.window_hours: dict[str, list[int]] = {"fomc": [24, 24], "cpi":
  [12, 12], "nfp": [12, 12], "gdp": [12, 12]}` (before/after, materialized at ingest),
  `calendar.min_future_days: int = 10`.
- Consumes: V003 `calendar_events`; `FRED_API_KEY` env (name verified against
  `data/fred.py` at implementation); `Alerter` (verified wired in `scripts/main.py`).

- [ ] **Step 0a: Historical backfill (user-added 2026-07-07).** The same ingestor runs
  once in backfill mode: FRED returns the full historical date series per release
  (`sort_order=asc`, no limit — ~300 rows per series for 25 years); FOMC historical
  meeting dates seeded from the Fed's published past calendars into the same yaml
  (or a committed CSV seed if the list is long). Backfill from
  `calendar.backfill_start` (default `2015-01-01` — must cover the earliest OHLCV bar;
  verify against `SELECT min(date) FROM ohlcv_daily` at implementation). Windows
  materialize at ingest with current config (analysis stamps, not trade-time stamps —
  acceptable, documented). **Purpose: NOT model training** (events are not obs features,
  by locked design D-MT.2) — it enables Plan B's backtest-side event stamping
  (`backtest_trades.bar_ts` joined against event windows), `event_shock_sensitivity`
  weakness-profile seeding (§4.8), and empirical window-size validation.
- [ ] **Step 0: P-A4 verification** — one live call per release ID:
  `curl "https://api.stlouisfed.org/fred/release?release_id={10|50|53}&api_key=$FRED_API_KEY&file_type=json"`;
  the response `name` must be "Consumer Price Index", "Employment Situation", "Gross
  Domestic Product" respectively. Record the three responses in the test file header.
  Any mismatch → correct `calendar.fred_release_ids` before writing code.
- [ ] **Step 1: Failing tests** — (a) ingest with a mocked FRED HTTP response inserts
  rows with materialized windows and `source='fred'`; (b) re-ingest of the same payload
  inserts 0 new rows (UNIQUE NULLS NOT DISTINCT); (c) FOMC yaml dates ingest with
  `source='config'`; (d) staleness check calls `alerter.send_alert` when the newest
  future event is nearer than `min_future_days`.
- [ ] **Step 2:** Run — Expected: FAIL
- [ ] **Step 3: Implement** (FRED endpoint:
  `https://api.stlouisfed.org/fred/release/dates?release_id={id}&api_key={key}&file_type=json&include_release_dates_with_no_data=false&sort_order=desc&limit=30`;
  release date → `scheduled_at` at the standard 08:30 ET print time for cpi/nfp/gdp
  converted to UTC, `importance='high'`; ON CONFLICT DO NOTHING).
- [ ] **Step 4:** Run tests — Expected: PASS
- [ ] **Step 5: Commit** — `git commit -m "feat(2.R-A): calendar_events ingest (FRED + FOMC yaml) + staleness alarm"`

### Task 12: V004 — `llm_calls` + intent tables (DDL) + rotation-gated MT commentary skeleton

**Files:**
- Create: `src/swingrl/data/migrations/V004__llm_calls_intents.sql`
- Modify: `src/swingrl/data/migration_runner.py` (EXPECTED_SCHEMA_VERSION = 4)
- Create: `services/memory/routers/trade.py` (POST `/trade/commentary`)
- Modify: `services/memory/app.py` (router include), `services/memory/db.py` (writers)
- Modify: `src/swingrl/config/schema.py` (`meta_trader.enabled: bool = False`,
  `meta_trader.commentary_provider: str = "cerebras"`)
- Modify: `src/swingrl/scheduler/jobs.py` (post-cycle commentary call, config-gated)
- Test: `tests/data/test_migrations_content.py` (extend), `tests/memory/test_trade_router.py`

**Interfaces:**
- Produces: `llm_calls` (all 8 `call_type`s + A15 identity CHECK matrix),
  `intent_records`, `intent_applications`, `intent_verdicts` (spec §4.4 verbatim);
  `ensemble_weight_history.intent_id` FK backfilled. Runtime: one shadow
  `MT_commentary` intent per cycle max (A14 cap enforced by UNIQUE partial index),
  `prompt_version = 'mt-commentary-v0'`.
- **Runtime gate:** `meta_trader.enabled` default **false**. Key rotation is complete
  (2026-07-07), so the remaining gate is Task 16's go/no-go. Graders/verdicts land in
  Plan B — day-one intents accumulate ungraded until then (freshness alarm ships with
  the graders; documented, accepted).

- [ ] **Step 1: Failing DDL test** — apply migrations; assert the CHECK matrix rejects an
  `epoch_advice` row with NULL `run_pk` and accepts a `trade_commentary` row with
  `cycle_id` set; `intent_verdicts` UNIQUE(intent_id, grader_version) holds; partial
  UNIQUE index rejects a second `MT_commentary` intent for the same cycle.
- [ ] **Step 2:** Run — Expected: FAIL
- [ ] **Step 3: Write `V004__llm_calls_intents.sql`** — spec §4.4 field lists verbatim
  (llm_calls with `coach`/`call_type` CHECKs, provider/model/prompt_version NOT NULL,
  `cycle_id REFERENCES inference_cycles`, the A15 CHECK matrix as one table CHECK;
  intent_records blocks 1–4 with per-lever CHECKs and `horizon_spec JSONB NOT NULL`;
  intent_applications `(intent_id UNIQUE, applied JSONB, applied_at)`; intent_verdicts
  per A16); plus:

```sql
ALTER TABLE ensemble_weight_history
    ADD CONSTRAINT fk_ewh_intent FOREIGN KEY (intent_id) REFERENCES intent_records (intent_id);
CREATE UNIQUE INDEX uq_mt_commentary_per_cycle
    ON intent_records (llm_call_id)
    WHERE lever = 'MT_commentary';   -- plus one-call-per-cycle enforced via llm_calls UNIQUE below
CREATE UNIQUE INDEX uq_llm_commentary_cycle
    ON llm_calls (cycle_id) WHERE call_type = 'trade_commentary';  -- A14 volume cap
```

- [ ] **Step 4:** Implement the endpoint + job skeleton: the swingrl scheduler job (after
  each cycle, when `meta_trader.enabled`) POSTs cycle context (cycle_id, regime stamp,
  per-algo proposals summary) to `/trade/commentary`; the memory service renders the v0
  prompt template (structured: diagnosis / matched-weakness / proposal / falsifiable bet
  fields required in the JSON schema, mirroring `advise_epoch`'s response-schema
  pattern at `query.py:996`), calls the configured provider, writes `llm_calls` +
  one `intent_records` row (`coach='meta_trader'`, `lever='MT_commentary'`,
  `mode='shadow'`, `horizon_spec` system-written from config:
  `{"type": "wall_clock_hours", "hours": 24}` equity / `{"type": "next_n_cycles", "n": 6}`
  crypto).
- [ ] **Step 5:** Run tests — Expected: PASS
- [ ] **Step 6: Commit** — `git commit -m "feat(2.R-A): V004 llm_calls + intent tables; rotation-gated MT commentary skeleton"`

### Task 13: Broker-API currency + binance_sim fidelity audit

**Files:**
- Modify: `pyproject.toml` (alpaca-py pin), `uv.lock`
- Modify: `src/swingrl/execution/adapters/binance_sim.py` (fidelity fixes found by audit)
- Create: `docs/execution/sim-fidelity.md` (divergence list — the audit's deliverable)
- Test: `tests/execution/test_binance_sim_fidelity.py`

**Interfaces:**
- Produces: a pinned, changelog-reviewed `alpaca-py>=0.20,<{verified_upper}`; a written
  divergence list (sim vs real Binance.US: fee schedule, min-notional and lot-size/step
  filters, price source, partial-fill behavior, time-in-force semantics); fixes for
  every divergence classified high-impact.
- Consumes: `alpaca_adapter.py` call inventory from the execution-path review (Task 0);
  **the review's sim-fidelity gap list** (review doc §5) as the audit's starting inventory:
  constant-slippage fills (captured slippage would be tautological), no
  rejections/lot-size/min-notional/partial fills, hardcoded fee never deducted, USDT-book
  thinness, zero time-to-fill, blocking retries. Plus the Alpaca IEX-feed check (free-plan
  `StockHistoricalDataClient` default — decision prices can be stale/off-consolidated).
- **Explicit Task 13 outcome decision:** improve the sim fill model (e.g. fill at best
  bid/ask instead of mid ± constant) vs accept + document the distortion — decided with
  the user when the audit's impact table exists.

- [ ] **Step 1 (Alpaca):** Read the alpaca-py changelog from the currently-locked version
  to latest; list breaking changes touching the calls `alpaca_adapter.py` makes (order
  submission, positions, account). Pin the upper bound at the highest verified-compatible
  minor. Run the adapter's existing tests against the pinned version.
- [ ] **Step 2 (Binance.US reality check):** Pull the current Binance.US fee schedule and
  the exchange-info filters (min notional, lot size/step) for the configured symbols via
  its public REST; record them in `docs/execution/sim-fidelity.md` next to what
  `binance_sim.py` actually models. Every divergence gets a row: behavior, sim value,
  real value, impact (high/low), disposition (fix now / accept + document).
- [ ] **Step 3 (failing tests):** For each high-impact divergence, write the failing test
  first (e.g. sim rejects an order below current min-notional; sim commission matches
  the documented fee tier; fill price sourced from the same bar the decision used).
- [ ] **Step 4:** Fix `binance_sim.py` until fidelity tests pass; low-impact divergences
  stay documented with rationale.
- [ ] **Step 5:** Run: `uv run pytest tests/execution/ -v` — Expected: PASS.
- [ ] **Step 6: Commit** — `git commit -m "feat(2.R-A): alpaca-py pinned after changelog review; binance_sim fidelity audit + fixes"`

### Task 14: CI health — dependency CVE audit + GHA Postgres (issue #18) (AMENDED 2026-07-13)

> **Amendment (user-approved 2026-07-13):** Task 14 also closes **issue #18** — the
> required GHA check `Tests (coverage >= 85%)` has been red on every PR since April
> because `ci.yml` runs pytest with no Postgres service and no `DATABASE_URL`
> (verified: `ci.yml:64,76`), so ~394 DB tests skip and coverage lands ~63%. With the
> redesign about to produce a steady stream of PRs, a permanently-red required check is
> an alarm-fatigue hazard and forces admin-bypass merges. The fix became cheap once
> Tasks 1–2 exist: the migration runner gives GHA a schema. Spec §1.4's out-of-scope
> line is overridden by this user decision for the *plan* (the spec governs the design,
> not CI plumbing). Homelab CI remains the authoritative gate.

**Files:**
- Modify: `pyproject.toml` (dev dependency `pip-audit`)
- Modify: `scripts/ci-homelab.sh` (new stage after lint)
- Modify: `.github/workflows/ci.yml` (Postgres 16 service + `DATABASE_URL` + schema init)
- Create: `docs/execution/cve-triage.md` (first-run findings + dispositions)

- [ ] **Step 1:** Add `pip-audit>=2.7` to dev dependencies; run locally:
  `uv run pip-audit --strict` — record every finding in `docs/execution/cve-triage.md`
  with disposition (upgrade / not-exploitable-here rationale / accepted-with-date).
- [ ] **Step 2:** Upgrade or pin packages for every fixable finding; re-run until clean
  or fully triaged.
- [ ] **Step 3:** Add a CI stage to `scripts/ci-homelab.sh` (after the lint stage,
  following its existing echo/step conventions): `uv run pip-audit --strict` — CI fails
  on new findings. Known-accepted findings are suppressed via `--ignore-vuln <id>` flags
  read from the triage doc, each with an expiry note.
- [ ] **Step 4: GHA Postgres (issue #18).** In `ci.yml`'s test job: add a
  `services: postgres:` block (image `postgres:16`, `POSTGRES_USER` / `POSTGRES_PASSWORD`
  / `POSTGRES_DB` all set to the CI-only throwaway value `swingrl_test`, health-checked
  on `pg_isready`); set `DATABASE_URL` on the test step to the standard
  `postgresql://` URL for that service on `localhost:5432` (same throwaway
  user/password/db — no real credential exists here); before pytest, initialize the
  schema exactly as homelab CI does
  (legacy `init_postgres_schema` + `scripts/apply_migrations.py` — mirror
  `scripts/ci-homelab.sh:47–54`'s sequence). The Stage-1 `db_guard`/`db_cleanup`
  fixtures handle per-test isolation unchanged.
- [ ] **Step 5:** Push and verify the GHA run: DB tests execute (skip count drops from
  ~394 to single digits), and coverage clears `--cov-fail-under=85`. **If coverage
  lands below 85 with all tests running, STOP and present the actual number + gap
  analysis to the user** — the threshold is adjusted or tests are added by decision,
  never silently. Update issue #18 with the result and close it on green.
- [ ] **Step 6:** Run homelab CI to prove the pip-audit stage works and passes.
- [ ] **Step 7: Commit** — `git commit -m "feat(2.R-A): pip-audit CVE stage + GHA Postgres service (closes #18)"`

### Task 15: Paper-trading security hardening checklist

**Files:**
- Create: `docs/execution/paper-security-checklist.md` (executed checklist, dated)
- Modify: whatever the checklist findings require (each fix = its own commit)

Checklist to execute and document (each item: verified-by command/evidence, date,
result):

- [ ] **Step 1 — Key scoping:** Alpaca **paper** keys in the paper deployment are
  distinct from any live keys; live keys absent from the homelab `.env` entirely until
  live go-live. Binance.US data keys are read-only market-data scope.
- [ ] **Step 2 — Key rotation executed:** the 2026-03-24 leaked keys (Mistral, Gemini,
  OpenRouter — see `project_llm_providers` note) are rotated and old keys revoked.
  This closes Task 0 Step 2 and unblocks Task 12's runtime.
- [ ] **Step 3 — Secrets never in image/repo/logs:** `.env` in `.dockerignore` +
  `.gitignore` (verify); `git log -S` spot-check for leaked values; grep structlog calls
  in `execution/`, `services/memory/` for any kwarg carrying a key/webhook; Alerter
  webhook URLs treated as secrets (env-only).
- [ ] **Step 4 — Network surface:** memory-service port 8889 bound to the docker network
  only (not host-published in `docker-compose.prod.yml`); dashboard read-only mounts
  confirmed; pg16 not exposed beyond the `br0` network.
- [ ] **Step 5 — Container posture:** `swingrl` runs as non-root `trader` user
  (Dockerfile); no `privileged`; restart policies sane.
- [ ] **Step 6 — Backup hygiene:** `pg_dump` output directory permissions; offsite sync
  encrypts or at minimum restricts access (check `backup/offsite_sync.py` behavior).
- [ ] **Step 7: Commit** — checklist doc + any fixes:
  `git commit -m "chore(2.R-A): paper-trading security hardening checklist executed"`

### Task 16: End-to-end readiness verification (Discord, breakers, capture) — paper go/no-go

**Files:**
- Create: `docs/execution/paper-readiness-runbook.md` (procedure + dated results)
- No production code (fixes discovered here loop back into earlier tasks).

Run on homelab against the paper deployment, after Tasks 1–15:

- [ ] **Step 1 — Discord live proof:** startup test alert (`Alerter.send_alert` smoke
  call on deploy); confirm arrival in the Discord channel. Then a real cycle's trade
  embed (paper fill) arrives. **Must also prove the INFO path** (digest flush or
  immediate INFO, per Task E's 2026-07-15 Discord-wiring amendment) — critical/warning
  arriving does NOT prove INFO can send; they take different code paths.
- [ ] **Step 2 — Drawdown CB trip:** in paper mode, force a drawdown breach (test hook or
  temporarily lowered `max_drawdown_pct` in a scratch config) → verify: cycle halts,
  `circuit_breaker_events` row written, Discord alert received, `risk_decisions` row
  records the veto. Restore config.
- [ ] **Step 3 — Turbulence halt (F1 proof):** with the Task 6 fix deployed, temporarily
  lower the halt threshold (or replay a known high-turbulence date) → verify
  `cycle_halted_by_turbulence` fires and alerts. This is the first time this breaker
  will EVER have fired — treat any anomaly as a finding.
- [ ] **Step 4 — Capture verification:** after ≥2 scheduled cycles per env:
  `inference_cycles` rows present with non-NULL regime stamps (incl. `turbulence`),
  3 `cycle_algo_proposals` per cycle, fills carry `cycle_id`, `fill_quality` rows for
  signal fills only. Force one capture failure (e.g. revoke INSERT on
  `cycle_algo_proposals` in a scratch DB run) → trading proceeds + Discord warning
  arrives (fail-open proof).
- [ ] **Step 5 — Sim mirror check:** one full crypto cycle in paper; verify the sim's
  fills respect the Task 13 filters/fees; compare a paper equity fill's `fill_quality`
  against the sim's for shape parity.
- [ ] **Step 6 — Go/no-go:** all above green → paper trading declared READY (recorded in
  the runbook doc with date + evidence links). Any red → fix, re-run this task.
- [ ] **Step 7: Commit** — `git commit -m "docs(2.R-A): paper-readiness runbook executed — go/no-go record"`

### Task 17: Full verification + homelab CI + deployment runbook

**Files:**
- Modify: `docs/training/` — only if any doc references changed code (CLAUDE.md rule)
- No new code.

- [ ] **Step 1:** Full suite locally: `uv run pytest tests/ -v` (background, 10-min
  timeout per feedback rule) — **0 failures required**.
- [ ] **Step 2:** Lint/type: `uv run ruff check . && uv run ruff format --check . && uv run mypy src/`
- [ ] **Step 3:** Homelab CI: `cd ~/swingrl && git fetch origin && git checkout swingrl/2.R-A-capture-foundation && git pull && bash scripts/ci-homelab.sh --no-cache` — must pass.
- [ ] **Step 4: Deployment runbook (gated 🛑 — plan-mode approval + verified backup before running against live pg16):**
  1. `pg_dump` live DB; restore into scratch DB; row-count check (backup gate).
  2. `python scripts/apply_migrations.py` (V001–V004) against live.
  3. `python scripts/migrations/bootstrap_era0_models.py` (homelab — artifacts on disk).
  4. Rebuild + restart `swingrl` and `swingrl-memory` (lockstep; both now assert the
     schema fingerprint at start).
  5. First paper cycle verification queries:
     `SELECT count(*) FROM inference_cycles;` (≥1 per env after one cycle),
     `SELECT count(*) FROM cycle_algo_proposals;` (= 3 per cycle),
     `SELECT turbulence FROM inference_cycles ORDER BY cycle_id DESC LIMIT 2;` (non-NULL),
     fills present → `SELECT count(*) FROM fill_quality;`.
- [ ] **Step 5:** Update `.planning/V1.1_EXECUTION_PLAN.md` tracker + ask the user about
  PR creation (per-phase workflow; **user merges**).

---

## §4.14 coverage — Plan A scope

| Hand-off item | Where |
|---|---|
| Migration/writer change-site inventory (trade-time subset) | Register above + Tasks 8–10 |
| Era-0/gate-v0 bootstrap incl. 574-row back-stamp (A7) | Task 2 |
| `schema_migrations` runner + fingerprint assertion (A7b/A25 cutover guard) | Tasks 1, 3 |
| Event-feed selection (FRED candidate) | Task 11 (FRED releases + FOMC yaml; P-A4) |
| Calendar staleness alarm (A25 alerting) | Task 11 |
| F1 re-triage — fix before capture (A25) | Task 6 (+ F1b Task 7, A27 capture) |
| Key rotation precondition for new call types | Task 0 Step 2 + Task 12 gate |
| Per-table index plan (trade-time subset: spine FKs, `trades.cycle_id`, `(call_type, created_at)`) | Tasks 4, 8, 12 DDL |
| A14 trade-time volume cap (≤1 intent/cycle) | Task 12 partial unique indexes |
| **User-added readiness scope (2026-07-07):** broker-API currency + sim fidelity | Task 13 |
| Dependency CVE audit in CI + GHA Postgres (issue #18, user-added 2026-07-13) | Task 14 |
| Paper-trading security hardening (incl. key rotation execution) | Task 15 |
| Discord + circuit-breaker + capture end-to-end proof (go/no-go) | Task 16 |
| **Code-review additions (2026-07-11):** mark-to-market valuation (C1/M4) | Task A |
| Honest fill lifecycle + 15:45 ET market-gated schedule (C2/M11) | Task B |
| Ramp enforcement + breaker alerting + crypto-stop book fix (H4/M1/H5) | Task C |
| Unified model layout, cache invalidation, fail-closed loading (H2/H3/M5/M7/M3) | Task D |
| **A30 (2026-07-12):** trader/trainer deploy isolation — compose split, pinned trader tag, deploy windows, floor-semantics assertion, `models/active/` write ban | Task E (+ Task 1/3 semantics) |

Deferred to Plan B: everything training-side (remaining §4 tables, caps/triggers, gate
re-derivation, harness, graders + freshness alarms, A26 numbers, cutover runbook REVOKE
step, archive-and-drop 🛑, nightly dumps + restore drill, Stage-3.5 grants, S4 gate,
K digest depth, S2 margin, `operator_actions`, provider/tier table, era-1 env definition
(A28) incl. train-with-real-turbulence).
