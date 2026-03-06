# Phase 2: Developer Experience - Context

**Gathered:** 2026-03-06
**Status:** Ready for planning

<domain>
## Phase Boundary

Establish SwingRL development conventions, config schema enforcement, and Claude Code skills so any session can work within the project consistently. Covers CLAUDE.md, .claude/commands/, Pydantic v2 config schema (swingrl.yaml), models/ scaffold completion, and expanded smoke tests with shared fixtures.

</domain>

<decisions>
## Implementation Decisions

### CLAUDE.md Conventions
- Comprehensive rulebook (~100-150 lines) covering all major patterns: imports, error handling, logging, config access, test structure, naming, docstrings
- Zero ambiguity for new sessions — no need to look up spec docs for coding patterns
- Custom exception hierarchy: SwingRLError base class with typed subclasses (BrokerError, DataError, ConfigError, etc.)
- structlog for structured JSON logging — pairs with HARD-05 requirement, works well with Docker log aggregation
- Include SwingRL domain rules: never hardcode ticker symbols, UTC internally with ET display conversion, broker interactions through execution middleware, config via Pydantic model (never raw YAML)

### Claude Skills (.claude/commands/)
- Only create skills usable today — dev workflow skills only (test, lint, typecheck, Docker build)
- Data/training/ops skills added in the phases that need them (no dead stubs)
- Native execution by default, with --docker flag option for container-based execution
- Include a local CI mirror skill that runs the same 5 stages as ci-homelab.sh but natively on Mac (no SSH, no Docker build) for quick pre-push validation

### Config Schema & Defaults
- Dev defaults in config/swingrl.yaml (paper mode, conservative limits, local paths), committed to repo
- Separate config/swingrl.prod.yaml.example with production values commented and annotated, committed as reference
- Environment variable overrides via Pydantic v2 SettingsConfigDict with SWINGRL_ prefix — perfect for Docker where .env sets trading mode
- Config file lives in config/swingrl.yaml (existing config/ directory)
- Explicit load pattern: call load_config(path) to get validated config — no side effects on import, tests can pass custom configs easily, fail-fast with clear ValidationError
- Schema itself follows Doc 05 §10.7 as-is (locked constraint from PROJECT.md)

### Smoke Tests & Fixtures
- Keep existing 8 smoke tests, add config validation tests in new tests/test_config.py
- conftest.py provides: config fixtures (tmp_config with valid YAML in temp dir), sample data fixtures (OHLCV DataFrames for equity daily + crypto 4H), and temp directory fixtures (data/, db/, models/, logs/ with auto-cleanup)
- Function-scoped fixtures by default (fresh per test, prevents coupling). Session-scope only for expensive read-only setup like config loading
- Must-have test scenario: config validation roundtrip — valid YAML loads, invalid YAML raises ValidationError with clear message, env vars override YAML values

### Claude's Discretion
- Exact structlog configuration and formatter setup
- Exception hierarchy granularity (which subclasses beyond the obvious)
- Specific dev skill implementations (exact commands, flags)
- Sample data fixture values (realistic but synthetic OHLCV data)
- Fixture helper utilities in conftest.py

</decisions>

<specifics>
## Specific Ideas

- Local CI mirror skill should match ci-homelab.sh's 5-stage pattern but skip Docker build (run natively for speed)
- Config validation error messages should be clear enough that a developer knows exactly which field failed and why
- structlog chosen specifically because HARD-05 requires "structured JSON logging" — get the pattern right from Phase 2

</specifics>

<code_context>
## Existing Code Insights

### Reusable Assets
- `pyproject.toml`: Already has tool configs for ruff, black, mypy, bandit, pytest — CLAUDE.md should reference these
- `tests/test_smoke.py`: 8 existing smoke tests with ENV-XX requirement annotations — pattern to follow
- `tests/conftest.py`: Exists but minimal (225 bytes) — needs expansion with shared fixtures
- `scripts/ci-homelab.sh`: 5-stage CI pattern to mirror in local skill

### Established Patterns
- Test naming: `test_<what>` with docstring referencing requirement ID (e.g., `"""ENV-01: Python 3.11 required."""`)
- Import checks: `importlib.import_module` pattern for dynamic import testing
- Path handling: `Path(__file__).parent.parent` for repo root — pathlib already standard
- ruff-format replaces black for formatting (Phase 1 decision)

### Integration Points
- `src/swingrl/config/`: Empty subpackage — config schema module goes here
- `config/`: Directory with .gitkeep — YAML config files go here
- `.claude/commands/`: Does not exist yet — create from scratch
- `models/active/`, `models/shadow/`, `models/archive/`: Already exist (ENV-12 partially done, just need .gitkeep verification)

</code_context>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope

</deferred>

---

*Phase: 02-developer-experience*
*Context gathered: 2026-03-06*
