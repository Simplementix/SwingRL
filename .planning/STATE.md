---
gsd_state_version: 1.0
milestone: v1.0
milestone_name: milestone
status: completed
stopped_at: Completed 10-05-PLAN.md
last_updated: "2026-03-10T03:35:20.242Z"
last_activity: "2026-03-10 -- Phase 10 Plan 05: FinBERT sentiment pipeline"
progress:
  total_phases: 10
  completed_phases: 9
  total_plans: 42
  completed_plans: 38
  percent: 100
---

# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-03-06)

**Core value:** Capital preservation through disciplined, automated risk management — never lose more than you can recover from
**Current focus:** Phase 6: RL Environments

## Current Position

Phase: 10 of 10 (Production Hardening)
Plan: 5 of 5 in phase 10
Status: Plan 10-05 complete -- FinBERT sentiment pipeline
Last activity: 2026-03-10 -- Phase 10 Plan 05: FinBERT sentiment pipeline

Progress: [██████████] 100%

## Performance Metrics

**Velocity:**
- Total plans completed: 0
- Average duration: - min
- Total execution time: 0 hours

**By Phase:**

| Phase | Plans | Total | Avg/Plan |
|-------|-------|-------|----------|
| - | - | - | - |

**Recent Trend:**
- Last 5 plans: none yet
- Trend: -

*Updated after each plan completion*
| Phase 01-dev-foundation P01 | 11 | 2 tasks | 35 files |
| Phase 01-dev-foundation P02 | 3 | 1 tasks | 2 files |
| Phase 01-dev-foundation P03 | ~2 sessions | 3 tasks | 5 files |
| Phase 02-developer-experience P02 | 2 min | 2 tasks | 6 files |
| Phase 02-developer-experience P01 | 6 | 2 tasks | 8 files |
| Phase 02-developer-experience P03 | 6 | 2 tasks | 8 files |
| Phase 02-developer-experience P04 | 2 | 2 tasks | 3 files |
| Phase 03-data-ingestion P01 | 8 min | 2 tasks | 9 files |
| Phase 03-data-ingestion P02 | 5 | 1 tasks | 4 files |
| Phase 03-data-ingestion P03 | 13 min | 2 tasks | 5 files |
| Phase 03 P04 | 7 | 1 tasks | 5 files |
| Phase 04 P01 | 7 | 2 tasks | 9 files |
| Phase 04 P03 | 5 | 1 tasks | 4 files |
| Phase 04 P02 | 11 | 1 tasks | 8 files |
| Phase 04 P04 | 13 | 2 tasks | 6 files |
| Phase 05 P01 | 5 | 2 tasks | 10 files |
| Phase 05 P03 | 5 | 2 tasks | 7 files |
| Phase 05 P02 | 9 | 2 tasks | 7 files |
| Phase 05 P04 | 4 | 2 tasks | 4 files |
| Phase 05 P05 | 9 | 2 tasks | 5 files |
| Phase 06 P01 | 4 | 1 tasks | 7 files |
| Phase 06 P02 | 6 | 1 tasks | 6 files |
| Phase 06 P03 | 5 | 2 tasks | 3 files |
| Phase 07 P01 | 7 | 2 tasks | 8 files |
| Phase 07 P02 | 11 | 2 tasks | 6 files |
| Phase 07 P01 | 8 | 2 tasks | 5 files |
| Phase 07 P03 | 6 | 2 tasks | 6 files |
| Phase 08 P02 | 5 | 2 tasks | 7 files |
| Phase 08 P03 | 7 | 2 tasks | 8 files |
| Phase 08 P01 | 14 | 2 tasks | 11 files |
| Phase 08 P04 | 8 | 2 tasks | 9 files |
| Phase 08 P05 | 5 | 2 tasks | 4 files |
| Phase 09 P03 | 3 | 2 tasks | 9 files |
| Phase 09 P02 | 5 | 2 tasks | 8 files |
| Phase 09 P01 | 7 | 2 tasks | 11 files |
| Phase 09 P04 | 8 | 3 tasks | 9 files |
| Phase 10 P01 | 5 | 2 tasks | 10 files |
| Phase 10 P05 | 8 | 2 tasks | 7 files |
| Phase 10 P03 | 5 | 2 tasks | 6 files |
| Phase 10-production-hardening P02 | 6 | 2 tasks | 9 files |

## Accumulated Context

### Decisions

Decisions are logged in PROJECT.md Key Decisions table.
Recent decisions affecting current work:

- Python 3.11 locked (FinRL/pyfolio compatibility — do not upgrade)
- uv over pip/poetry — must install on both M1 Mac and homelab
- Docker only for CI — validate what ships to production
- Sequential milestones — M2 finishes before M3 starts (no interleaving)
- Incremental DB schema — create tables as each milestone needs them, not all 28 upfront
- Discord webhooks as canonical alert channel (not Telegram, not email)
- [Phase 01-dev-foundation]: pandas-ta removed from Phase 1 deps — PyPI 0.4.x requires Python>=3.12; replaced with stockstats>=0.4.0 (FinRL native TA library). Re-evaluation Phase 6.
- [Phase 01-dev-foundation]: tool.uv.environments constrained to darwin for Phase 1 — Linux Docker lockfile generation deferred to Phase 2 (ENV-06). Remove constraint before Phase 2 Plan 1.
- [Phase 01-dev-foundation]: ruff-format replaces black in pre-commit to avoid formatting conflicts; black retained in pyproject.toml for direct CLI use
- [Phase 01-dev-foundation]: bandit[toml] additional_dependency required in pre-commit hook — without it bandit cannot read pyproject.toml exclusions
- [Phase 01-dev-foundation Plan 03]: CPU-only torch in Docker — homelab has no GPU; MPS stays on M1 Mac natively
- [Phase 01-dev-foundation Plan 03]: Single-stage Dockerfile with dev deps — no production split until Phase 8+
- [Phase 01-dev-foundation Plan 03]: ruff format --check replaces standalone black check inside container (ruff formatter is drop-in black replacement)
- [Phase 01-dev-foundation Plan 03]: ci-homelab.sh 5-stage pattern established as canonical CI runner for all future phases
- [Phase 02-developer-experience Plan 02]: CLAUDE.md written verbatim from plan spec — content was fully specified
- [Phase 02-developer-experience Plan 02]: ci-local.md 4 stages map to tests, lint, typecheck, security — mirrors ci-homelab.sh stages 2-5 natively
- [Phase 02-developer-experience]: structlog.* added to mypy ignore_missing_imports; pre-commit mypy hook needs structlog+pydantic-settings as additional_dependencies
- [Phase 02-developer-experience]: configure_logging uses stdlib ProcessorFormatter bridge so third-party libs emit structured logs
- [Phase 02-developer-experience]: pydantic-settings 2.13.1 uses file_secret_settings (not secrets_dir) in settings_customise_sources — fixed to match actual API
- [Phase 02-developer-experience]: types-PyYAML must be in pre-commit mypy additional_dependencies (isolated env) AND pyproject.toml dev deps
- [Phase 02-developer-experience]: load_config() uses inner _ConfigWithYaml subclass to bind yaml_path at call time — keeps SwingRLConfig cleanly importable
- [Phase 02-developer-experience]: conftest.py fixture scopes: session-scoped only for repo_root; function-scoped for all others to prevent cross-test state mutation
- [Phase 02-developer-experience]: valid_config_yaml as separate string fixture allows bad YAML tests to construct invalid variants independently of the valid baseline
- [Phase 03-data-ingestion]: pyarrow added as explicit dep (not transitive via pandas as research assumed)
- [Phase 03-data-ingestion]: responses moved to dev dependency group (test-only library)
- [Phase 03-data-ingestion]: Staleness threshold: 4 calendar days equity, 8H crypto, 35 days FRED
- [Phase 03-data-ingestion]: Quarantine reasons stored as semicolon-delimited string in reason column
- [Phase 03-data-ingestion]: Mock SDK client directly rather than HTTP responses for alpaca-py testing
- [Phase 03]: FRED validate() uses custom null-only row check instead of OHLCV-based validate_rows()
- [Phase 03-data-ingestion]: All klines fetched from api.binance.us (not api.binance.com) per Binance.US broker architecture
- [Phase 03-data-ingestion]: Stitch point at 2019-09-01 (Binance.US launch) separates archive from API data
- [Phase 03-data-ingestion]: Microsecond threshold at 2_000_000_000_000 to detect 2025+ archive timestamp format
- [Phase 04]: DuckDB connection type annotated as Any due to missing type stubs
- [Phase 04]: consecutive_failures_before_alert=1 in default test fixture; threshold=3 tested in dedicated TestConsecutiveFailures
- [Phase 04]: DuckDB replacement scan for DataFrame-to-table sync (sync_df variable referenced by name in SQL)
- [Phase 04]: Lazy DatabaseManager init in BaseIngestor via _get_db() for backward compatibility
- [Phase 04]: yfinance Adj Close used as reference price for cross-source comparison
- [Phase 04]: as_of_date parameter added to validate_prices for testability with historical dates
- [Phase 04]: Step 12 cross-source check is warning-only, never quarantines data
- [Phase 04]: Corporate action thresholds: 30% equity overnight spike, 40% crypto
- [Phase 05]: init_feature_schema accepts raw DuckDB conn for testability with in-memory connections
- [Phase 05]: stockstats retype(ohlcv.copy()) pattern established for copy-safe indicator computation
- [Phase 05]: HMM label ordering via mean-return sort ensures bull=state0 consistency across refits
- [Phase 05]: Ridge regularization (1e-6) on covariance matrices prevents ValueError on warm-start
- [Phase 05]: Turbulence uses np.linalg.pinv for near-singular crypto covariance (BTC/ETH r=0.9)
- [Phase 05]: ASOF JOIN on release_date for look-ahead-free macro alignment
- [Phase 05]: Alpha Vantage fallback is lazy-imported and disabled without API key
- [Phase 05]: Rolling z-score epsilon floor via clip(lower=epsilon) on rolling std
- [Phase 05]: Domain priority dict drives deterministic correlation pruning drop selection
- [Phase 05]: SMA exception threshold (0.90) higher than general correlation threshold (0.85)
- [Phase 05]: Observation assembly order: [per-asset alpha-sorted] + [macro] + [HMM] + [turbulence] + [overnight crypto-only] + [portfolio state]
- [Phase 05]: Default portfolio state = 100% cash until Phase 6 RL environments provide real positions
- [Phase 05]: Feature A/B comparison threshold: validation Sharpe >= 0.05 with overfitting guard
- [Phase 06]: deque(maxlen=window) for RollingSharpeReward automatic rolling window
- [Phase 06]: step_count-based termination avoids off-by-one in episode length
- [Phase 06]: equity_env_config fixture with 8 symbols separate from loaded_config to prevent fixture coupling
- [Phase 06]: gymnasium added to pre-commit mypy additional_dependencies for isolated hook environment
- [Phase 06]: register() calls before class imports in envs/__init__.py to avoid circular imports
- [Phase 06]: E402 noqa suppression for intentional post-registration class imports
- [Phase 07]: Sortino uses sqrt(mean(neg^2)) for downside deviation (full lower partial moment)
- [Phase 07]: Overfitting gap boundaries: <0.20 healthy, 0.20-0.50 marginal, >0.50 reject (strict inequalities)
- [Phase 07]: compute_trade_metrics treats zero-PnL trades as losses for win_rate
- [Phase 07]: stable_baselines3 added to mypy ignore_missing_imports and pre-commit additional_dependencies
- [Phase 07]: type: ignore[assignment] for SB3 VecEnv step/reset type stubs mismatches
- [Phase 07]: Episode bars minimum 50 in test fixtures (schema constraint)
- [Phase 07]: No log_interval parameter to model.learn() -- ZeroDivisionError in A2C on-policy
- [Phase 07]: Default turbulence threshold 1.0 for adaptive ensemble window shrink (configurable)
- [Phase 07]: Ensemble validation windows: 63 bars equity, 126 bars crypto (CONTEXT.md spec)
- [Phase 07]: nosec B608 on DuckDB table name interpolation (env_name constrained by CLI enum)
- [Phase 08]: Deadzone boundary (exactly +/-0.02) treated as hold, strict inequality for buy/sell
- [Phase 08]: Cost gate uses proportional rate (0.06% equity, 0.22% crypto) -- always passes under 2% threshold
- [Phase 08]: ValidatedOrder.order field name (not sized_order) per frozen dataclass convention
- [Phase 08]: ValidatedOrder.order field name matched existing types.py frozen dataclass
- [Phase 08]: Broker literal binance_us (not binance_sim) to match FillResult Literal constraint
- [Phase 08]: [Phase 08-01]: portfolio_snapshots gains environment column (composite PK) for per-env queries
- [Phase 08]: [Phase 08-01]: CB ramp-up uses 5-interval scheme (20% halt + 4x 20% ramp) to avoid float boundary issues
- [Phase 08]: [Phase 08-01]: exchange_calendars NYSE calendar for equity business day cooldown counting
- [Phase 08]: [Phase 08-04]: Exchange adapter instantiated lazily during execute_cycle to avoid API credential requirements at init
- [Phase 08]: [Phase 08-04]: Startup reconciliation runs before first trade cycle in run_cycle.py (equity only, per CONTEXT.md)
- [Phase 08]: [Phase 08-04]: Turbulence 90th percentile approximated from atr_14_pct until dedicated turbulence history table
- [Phase 08]: [Phase 08-05]: Phase 8 placeholder CMD (sleep loop) in production Dockerfile -- Phase 9 replaces with APScheduler
- [Phase 08]: [Phase 08-05]: HEALTHCHECK skips missing DBs gracefully on first startup (no false unhealthy)
- [Phase 08]: [Phase 08-05]: Production image selective COPY (src/, config/, scripts/) excludes tests/docs/.planning
- [Phase 09]: Dashboard as separate repo-root directory with own requirements.txt (not inside src/swingrl/)
- [Phase 09]: Traffic-light windows: 26h equity, 5h crypto; read-only SQLite URI mode enforced
- [Phase 09]: Stuck agent threshold 10 for equity, 30 for crypto; two-webhook routing with single-URL fallback
- [Phase 09]: Module-level _ctx: JobContext | None pattern for APScheduler job functions
- [Phase 09]: threading.Event().wait() over signal.pause() for cross-platform main blocking
- [Phase 09]: Stop-price polling daemon thread at 60s intervals with daemon=True
- [Phase 09]: Dashboard service 512MB/0.5 CPU with read-only DB mounts
- [Phase 10]: tenacity added to mypy ignore_missing_imports (no type stubs available)
- [Phase 10]: transformers in optional [sentiment] dep group to avoid 2GB+ install when disabled
- [Phase 10]: File handler always uses JSONRenderer regardless of json_logs flag
- [Phase 10-production-hardening]: Backup jobs skip halt check -- backups must run even when trading is halted
- [Phase 10-production-hardening]: nosec B608 on DuckDB backup table name interpolation (module constant tuple)
- [Phase 10-production-hardening]: _transformers module-level global for lazy import to avoid ruff N814 CamelCase alias violations
- [Phase 10-production-hardening]: nosec B107 on NewsFetcher __init__ for empty-string API key defaults (test-friendly)

### Pending Todos

None yet.

### Blockers/Concerns

None. Phase 1 complete.

**Action required before Phase 2 Plan 1:** Remove `tool.uv.environments` darwin constraint from pyproject.toml — deferred from Phase 1, needed for Linux Docker lockfile generation (ENV-06 follow-up).

## Session Continuity

Last session: 2026-03-10T03:36:23Z
Stopped at: Completed 10-05-PLAN.md
Resume file: .planning/phases/10-production-hardening/10-05-SUMMARY.md
