# Phase 11: Production Startup Wiring - Context

**Gathered:** 2026-03-10
**Status:** Ready for planning

<domain>
## Phase Boundary

Fix 3 cross-phase wiring bugs so that `main.py` starts without error, feature tables are created during standard DB init, and scheduler FRED jobs import from the correct module path. Pure bug-fix phase — no new capabilities.

</domain>

<decisions>
## Implementation Decisions

### INT-01: ExecutionPipeline constructor (main.py:231)
- Create FeaturePipeline with a per-cycle DuckDB connection obtained from DatabaseManager (not a long-lived connection at startup) to avoid DuckDB single-writer contention
- ExecutionPipeline already stores `self._db`, so FeaturePipeline can be constructed lazily inside `execute_cycle()` or passed a DatabaseManager reference that yields connections on demand
- Alerter is already constructed at main.py:232-239 — pass it to ExecutionPipeline constructor
- `models_dir` uses `Path("models/active")` resolved relative to project root from config (convention already established by model lifecycle in Phase 10)

### INT-02: Feature table init (db.py)
- Call `init_feature_schema(conn)` from `DatabaseManager._init_duckdb_schema()` at the end of existing DDL
- Import is `from swingrl.features.schema import init_feature_schema`
- This makes feature tables available immediately after `db.init_schema()` without requiring a prior `compute_features.py` run

### INT-05: FRED import path (jobs.py:304,328)
- Change `from swingrl.data.ingestors.fred import FredIngestor` to `from swingrl.data.fred import FredIngestor`
- Both `weekly_fundamentals_job` and `monthly_macro_job` have the same wrong import
- Remove `type: ignore[import-not-found]` comment that was masking the error

### Claude's Discretion
- Whether FeaturePipeline gets DatabaseManager reference or a factory callable for connections
- Test structure and fixture design
- Order of fixes within the plan

</decisions>

<specifics>
## Specific Ideas

No specific requirements — fixes are well-defined by the audit findings. The existing code patterns (lazy init, DatabaseManager context managers, Phase 5 schema module) dictate the approach.

</specifics>

<code_context>
## Existing Code Insights

### Reusable Assets
- `FeaturePipeline(config, conn)` at `src/swingrl/features/pipeline.py:90` — takes config + DuckDB connection
- `init_feature_schema(conn)` at `src/swingrl/features/schema.py:93` — idempotent DDL for feature tables
- `Alerter` already constructed in `main.py:232-239` — just needs to be passed through
- `DatabaseManager.duckdb()` context manager — yields DuckDB cursors on demand

### Established Patterns
- Lazy component init inside ExecutionPipeline (see `self._initialized` flag at pipeline.py:80)
- `DatabaseManager._init_duckdb_schema()` uses sequential `CREATE TABLE IF NOT EXISTS` calls
- Import-guarded job functions with `try/except ImportError` in jobs.py

### Integration Points
- `main.py:231` — ExecutionPipeline construction (currently missing 3 args)
- `main.py:241` — `init_job_context()` already passes pipeline and alerter to job context
- `db.py:155` — `_init_duckdb_schema()` where feature schema call should be added
- `jobs.py:304,328` — two import statements to fix

</code_context>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope

</deferred>

---

*Phase: 11-production-startup-wiring*
*Context gathered: 2026-03-10*
