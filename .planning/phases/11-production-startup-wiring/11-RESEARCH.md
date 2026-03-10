# Phase 11: Production Startup Wiring - Research

**Researched:** 2026-03-10
**Domain:** Cross-phase integration bug fixes (main.py, db.py, jobs.py)
**Confidence:** HIGH

## Summary

Phase 11 fixes three cross-phase wiring bugs preventing `main.py` from starting without error. All three bugs are well-localized with exact line numbers and straightforward fixes. The research confirms the CONTEXT.md decisions are correct but surfaces additional bugs in the FRED import fix (wrong class name and wrong method call, not just wrong module path).

**Primary recommendation:** Fix all three bugs in a single plan wave. The fixes are isolated enough to be safe together, and testing each in isolation is straightforward with existing mock patterns.

<user_constraints>

## User Constraints (from CONTEXT.md)

### Locked Decisions
- INT-01: ExecutionPipeline constructor (main.py:231) -- Create FeaturePipeline with per-cycle DuckDB connection from DatabaseManager, pass Alerter and models_dir
- INT-02: Feature table init (db.py) -- Call init_feature_schema(conn) from DatabaseManager._init_duckdb_schema() at end of existing DDL
- INT-05: FRED import path (jobs.py:304,328) -- Change from swingrl.data.ingestors.fred to swingrl.data.fred, remove type: ignore comment

### Claude's Discretion
- Whether FeaturePipeline gets DatabaseManager reference or a factory callable for connections
- Test structure and fixture design
- Order of fixes within the plan

### Deferred Ideas (OUT OF SCOPE)
None -- discussion stayed within phase scope

</user_constraints>

<phase_requirements>

## Phase Requirements

| ID | Description | Research Support |
|----|-------------|-----------------|
| PAPER-01 | Alpaca paper trading connection for equity environment | ExecutionPipeline must construct correctly for equity_cycle to reach broker adapter |
| PAPER-02 | Binance.US simulated fills for crypto environment | ExecutionPipeline must construct correctly for crypto_cycle to reach broker adapter |
| PAPER-12 | APScheduler: equity daily, crypto 4H, pre-cycle halt checks | Scheduler jobs must import correctly (FRED fix), pipeline must init (constructor fix) |
| FEAT-11 | Per-environment feature tables in DuckDB | init_feature_schema() wired into DatabaseManager.init_schema() creates these tables |
| DATA-09 | Store lowest, aggregate up -- daily/4H bars, compute weekly/monthly on-the-fly | Aggregation views already in db.py; feature tables need to exist for pipeline queries |
| DATA-04 | FRED macro pipeline for Tier 1 series | FRED import path fix enables weekly_fundamentals_job and monthly_macro_job |
| FEAT-04 | Macro regime features via ASOF JOIN | Requires FRED data ingestion working (DATA-04) and feature tables existing (FEAT-11) |

</phase_requirements>

## Architecture Patterns

### Bug 1: ExecutionPipeline Constructor (INT-01)

**Current state (main.py:231):**
```python
pipeline = ExecutionPipeline(config=config, db=db)
```

**Required signature (execution/pipeline.py:55-62):**
```python
def __init__(
    self,
    config: SwingRLConfig,
    db: DatabaseManager,
    feature_pipeline: FeaturePipeline,
    alerter: Alerter,
    models_dir: Path,
) -> None:
```

**Missing arguments:** `feature_pipeline`, `alerter`, `models_dir`

**Fix approach -- DatabaseManager reference pattern (recommended):**

FeaturePipeline takes `(config, conn)` where `conn` is a raw DuckDB connection. The CONTEXT.md notes that a long-lived connection at startup risks DuckDB single-writer contention. Two options:

1. **DatabaseManager reference** -- Pass `db` to a wrapper that calls `db.duckdb()` on demand. Requires modifying FeaturePipeline to accept DatabaseManager instead of raw conn, or creating a thin adapter.
2. **Factory callable** -- Pass a `Callable[[], DuckDBConnection]` that the pipeline calls each cycle.

**Recommendation: Use DatabaseManager reference directly.** FeaturePipeline currently stores `self._conn` and passes it to sub-components. The cleanest fix is to construct FeaturePipeline with a connection obtained via `db.duckdb()` context manager at the time `build_app` runs. The DuckDB connection in DatabaseManager is already persistent (single connection, cursor-per-call), so there is no contention risk -- the context manager just creates a cursor. The CONTEXT.md concern about "long-lived connection" is already handled by DatabaseManager's existing architecture.

```python
# In build_app():
with db.duckdb() as conn:
    # conn is a cursor, but FeaturePipeline just calls conn.execute()
    # This pattern may not work because the cursor closes on context exit.
```

**Better: pass the underlying DuckDB connection directly.**
DatabaseManager._get_duckdb_conn() returns the persistent connection. FeaturePipeline can use this safely since DuckDB handles concurrent reads from the same connection. However, accessing private methods is an anti-pattern.

**Best approach: Add a public method to DatabaseManager** that exposes the persistent connection for components that need long-lived access (like FeaturePipeline). Or, refactor FeaturePipeline to accept DatabaseManager and call `db.duckdb()` internally when it needs a cursor.

Given Claude's discretion on this, the simplest fix is:
```python
# Get the persistent DuckDB connection (not a cursor)
duckdb_conn = db._get_duckdb_conn()
feature_pipeline = FeaturePipeline(config, duckdb_conn)
```
This works because `_get_duckdb_conn()` returns the persistent `duckdb.DuckDBPyConnection` which supports `.execute()` directly. FeaturePipeline's sub-components all use `conn.execute()` which works on both connections and cursors.

**Alerter and models_dir are straightforward:**
```python
alerter = Alerter(...)  # Already constructed at main.py:232-239
models_dir = Path(config.paths.models_dir)  # PathsConfig.models_dir defaults to "models/"
```

### Bug 2: Feature Table Init (INT-02)

**Current state (db.py:155-298):** `_init_duckdb_schema()` creates 7 tables + 2 views but does NOT create feature tables (features_equity, features_crypto, fundamentals, hmm_state_history).

**Fix:** Add `init_feature_schema(cursor)` call at the end of `_init_duckdb_schema()`.

```python
from swingrl.features.schema import init_feature_schema

# At end of _init_duckdb_schema(), inside the `with self.duckdb() as cursor:` block:
init_feature_schema(cursor)
```

**Verified:** `init_feature_schema()` accepts `Any` type for conn and calls `conn.execute(ddl)` -- works with both DuckDB connections and cursors. All DDL uses `CREATE TABLE IF NOT EXISTS` so it is idempotent.

### Bug 3: FRED Import Path (INT-05)

**Current state (jobs.py:304,328):**
```python
from swingrl.data.ingestors.fred import FredIngestor  # type: ignore[import-not-found]
```

**Actual module location:** `src/swingrl/data/fred.py`
**Actual class name:** `FREDIngestor` (not `FredIngestor`)
**Actual constructor:** `FREDIngestor(config)` (not `FREDIngestor(config, db)`)
**Actual method:** `run_all()` (not `refresh()`)

**CRITICAL FINDING: The CONTEXT.md identified the wrong module path but missed two additional bugs:**
1. Class name is `FREDIngestor`, not `FredIngestor` (all-caps FRED)
2. Constructor takes only `config`, not `(config, db)` -- jobs.py passes `(ctx.config, ctx.db)`
3. Method called is `.refresh()` which does not exist -- should be `.run_all()`

**Correct fix for both job functions:**
```python
from swingrl.data.fred import FREDIngestor

ingestor = FREDIngestor(ctx.config)
ingestor.run_all()
```

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Feature table DDL | Duplicate DDL in db.py | `init_feature_schema()` from features/schema.py | Already exists, tested, idempotent |
| DuckDB connection management | Manual connection pooling | DatabaseManager singleton | Thread-safe, lazy init, already handles all edge cases |

## Common Pitfalls

### Pitfall 1: DuckDB Cursor vs Connection Confusion
**What goes wrong:** Passing a cursor (from `db.duckdb()` context manager) to FeaturePipeline, then the cursor gets closed when the context exits.
**Why it happens:** `db.duckdb()` yields a cursor, not the connection. The cursor is only valid inside the `with` block.
**How to avoid:** Use the persistent DuckDB connection via `db._get_duckdb_conn()`, or refactor FeaturePipeline to accept DatabaseManager.
**Warning signs:** `duckdb.InvalidInputException: Connection already closed` errors during execute_cycle.

### Pitfall 2: Import Cycle Between db.py and features/schema.py
**What goes wrong:** Adding `from swingrl.features.schema import init_feature_schema` at top of db.py could create a circular import if features/schema.py imports from data/db.py.
**Why it happens:** features/schema.py currently only imports structlog -- no circular risk. But verify before committing.
**How to avoid:** The import can safely be at module level. If needed, use a local import inside `_init_duckdb_schema()`.

### Pitfall 3: FREDIngestor Constructor Mismatch
**What goes wrong:** Jobs.py passes `(ctx.config, ctx.db)` but FREDIngestor only takes `config`.
**Why it happens:** The original jobs.py code was written speculatively before the FRED ingestor existed.
**How to avoid:** Match the actual FREDIngestor signature: `FREDIngestor(config)`.

### Pitfall 4: Test Regression in test_main.py
**What goes wrong:** Existing test at test_main.py:54 asserts `mock_scheduler.add_job.call_count == 6`, but main.py currently registers 11 jobs. The test was written for an earlier version.
**Why it happens:** Phase 10 added backup, shadow, and trigger jobs but the test count was not updated.
**How to avoid:** Update the test assertion to match current job count (11), or verify the test is already correct for the current codebase.
**Note:** This may already be passing if the test was updated in Phase 10. Verify during implementation.

## Code Examples

### Fix 1: main.py ExecutionPipeline Constructor
```python
# Source: execution/pipeline.py:55-62 (verified constructor signature)
# Source: features/pipeline.py:90 (verified FeaturePipeline(config, conn))
# Source: config/schema.py:118 (verified PathsConfig.models_dir)

from swingrl.features.pipeline import FeaturePipeline

# In build_app(), after db = DatabaseManager(config):
duckdb_conn = db._get_duckdb_conn()
feature_pipeline = FeaturePipeline(config, duckdb_conn)

alerter = Alerter(...)  # existing code at lines 232-239

pipeline = ExecutionPipeline(
    config=config,
    db=db,
    feature_pipeline=feature_pipeline,
    alerter=alerter,
    models_dir=Path(config.paths.models_dir),
)
```

### Fix 2: db.py Feature Table Init
```python
# Source: features/schema.py:93-107 (verified init_feature_schema signature)

# At end of _init_duckdb_schema(), inside `with self.duckdb() as cursor:` block:
from swingrl.features.schema import init_feature_schema
init_feature_schema(cursor)
```

### Fix 3: jobs.py FRED Import
```python
# Source: data/fred.py:47 (class FREDIngestor)
# Source: data/fred.py:60 (__init__ takes only config)
# Source: data/fred.py:242 (run_all method)

# Replace lines 303-307 in weekly_fundamentals_job:
from swingrl.data.fred import FREDIngestor

ingestor = FREDIngestor(ctx.config)
ingestor.run_all()

# Replace lines 327-331 in monthly_macro_job:
from swingrl.data.fred import FREDIngestor  # noqa: F811

ingestor = FREDIngestor(ctx.config)
ingestor.run_all()
```

## Validation Architecture

### Test Framework
| Property | Value |
|----------|-------|
| Framework | pytest 8.x |
| Config file | pyproject.toml [tool.pytest.ini_options] |
| Quick run command | `uv run pytest tests/scheduler/ -v -x` |
| Full suite command | `uv run pytest tests/ -v` |

### Phase Requirements to Test Map
| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| PAPER-01 | ExecutionPipeline receives all 5 args, equity_cycle reachable | unit | `uv run pytest tests/scheduler/test_main.py -x -k "init"` | Yes (needs update) |
| PAPER-02 | ExecutionPipeline receives all 5 args, crypto_cycle reachable | unit | `uv run pytest tests/scheduler/test_main.py -x -k "init"` | Yes (needs update) |
| PAPER-12 | Scheduler jobs import without error, FRED jobs resolve | unit | `uv run pytest tests/scheduler/test_jobs.py -x` | Yes (needs new tests) |
| FEAT-11 | init_schema() creates features_equity and features_crypto | unit | `uv run pytest tests/data/test_db.py -x -k "feature"` | Partial (test_db.py exists, needs feature table test) |
| DATA-09 | Aggregation views + feature tables all present after init | unit | `uv run pytest tests/data/test_db.py -x` | Yes (existing) |
| DATA-04 | FRED import resolves from correct path | unit | `uv run pytest tests/scheduler/test_jobs.py -x -k "fred"` | Needs new test |
| FEAT-04 | Macro features depend on FRED + feature tables | integration | Covered by DATA-04 + FEAT-11 tests | N/A |

### Sampling Rate
- **Per task commit:** `uv run pytest tests/scheduler/ tests/data/test_db.py -v -x`
- **Per wave merge:** `uv run pytest tests/ -v`
- **Phase gate:** Full suite green before `/gsd:verify-work`

### Wave 0 Gaps
- [ ] `tests/scheduler/test_main.py` -- needs test for ExecutionPipeline receiving all 5 constructor args
- [ ] `tests/scheduler/test_jobs.py` -- needs test for FRED import path resolution (correct module, class, and method)
- [ ] `tests/data/test_db.py` -- needs test that init_schema() creates feature tables (features_equity, features_crypto)

## Open Questions

1. **FeaturePipeline connection strategy**
   - What we know: FeaturePipeline stores `self._conn` and passes it to sub-components. DatabaseManager has a persistent DuckDB connection accessed via `_get_duckdb_conn()` (private method).
   - What's unclear: Whether accessing `_get_duckdb_conn()` is acceptable or if a public accessor should be added.
   - Recommendation: Add a public `get_duckdb_connection()` method to DatabaseManager, or construct FeaturePipeline lazily inside execute_cycle using a cursor from `db.duckdb()`. The private accessor approach works but violates encapsulation.

2. **weekly_fundamentals vs monthly_macro distinction**
   - What we know: Both jobs currently call `ingestor.refresh()` (which does not exist). Both should call `run_all()`.
   - What's unclear: Whether the weekly job should only refresh a subset of series (DAILY_SERIES) vs the monthly job refreshing VINTAGE_SERIES.
   - Recommendation: For now, both call `run_all()` which fetches all 5 series. The weekly job handles VIXCLS/T10Y2Y/DFF updates; CPI/UNRATE are monthly-published so weekly fetches are harmless (FRED returns no new data). This matches the existing architecture where `run_all()` handles incremental fetching via "auto" since parameter.

## Sources

### Primary (HIGH confidence)
- `src/swingrl/execution/pipeline.py:55-62` -- ExecutionPipeline constructor signature (5 required args)
- `src/swingrl/features/pipeline.py:90` -- FeaturePipeline(config, conn) constructor
- `src/swingrl/features/schema.py:93-107` -- init_feature_schema(conn) function
- `src/swingrl/data/fred.py:47-66` -- FREDIngestor class (name, constructor, methods)
- `src/swingrl/data/db.py:155-298` -- _init_duckdb_schema() current DDL
- `src/swingrl/scheduler/jobs.py:291-336` -- FRED import bugs (lines 304, 328)
- `scripts/main.py:231-241` -- ExecutionPipeline construction site
- `src/swingrl/config/schema.py:118` -- PathsConfig.models_dir field

### Secondary (MEDIUM confidence)
- `tests/scheduler/test_main.py` -- Existing test patterns for mocking main.py components

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH -- all code read directly, no external library research needed
- Architecture: HIGH -- all integration points verified against actual source files
- Pitfalls: HIGH -- identified through direct code inspection (DuckDB cursor lifecycle, import cycles, class name mismatch)

**Research date:** 2026-03-10
**Valid until:** Indefinite (internal codebase research, no external dependencies)
