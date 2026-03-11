# Project Retrospective

*A living document updated after each milestone. Lessons feed forward into future planning.*

## Milestone: v1.0 — MVP

**Shipped:** 2026-03-11
**Phases:** 17 | **Plans:** 49 | **Requirements:** 74/74
**Timeline:** 8 days (2026-03-03 → 2026-03-10) | **Commits:** 325

### What Was Built
- Complete RL swing trading system: data ingestion, feature engineering, RL training, paper trading execution
- Multi-source pipeline: Alpaca (8 equity ETFs), Binance.US (BTC/ETH 4H), FRED (5 macro series)
- 156-dim equity and 45-dim crypto observation vectors with HMM regime detection
- PPO/A2C/SAC Sharpe-weighted ensemble with walk-forward validation (4 performance gates)
- 5-stage execution middleware with two-tier risk veto, circuit breakers, bracket orders
- Production hardening: APScheduler automation, Discord alerting, Streamlit dashboard, backup pipeline, shadow mode, emergency stop
- Gap closure phases (11-17): 7 integration/wiring fixes discovered via milestone audit

### What Worked
- **Specification-first approach**: 15 detailed design docs before any code eliminated most design-time ambiguity
- **Milestone audit → gap closure pattern**: Audit after Phase 10 found 6 integration gaps; dedicated gap closure phases (11-17) resolved all of them before shipping
- **TDD workflow**: 848+ tests caught regressions early, especially during gap closure refactoring
- **ci-homelab.sh validation**: Docker builds on x86 caught ARM/x86 compatibility issues before they became production problems
- **Incremental DB schema**: Creating tables per phase kept each phase self-contained and testable

### What Was Inefficient
- **Gap closure overhead**: 7 extra phases (11-17) were needed because cross-module wiring wasn't verified during original phases 1-10. Earlier integration testing could have caught these during development.
- **REQUIREMENTS.md description drift**: Requirement descriptions didn't always get updated when gap closure phases changed how requirements were satisfied (FEAT-09/10, PAPER-02/09)
- **Plan checkbox inconsistency**: Some ROADMAP.md plan checkboxes weren't updated to `[x]` after execution, requiring Phase 17 doc housekeeping

### Patterns Established
- **Phase branch workflow**: `gsd/phase-{N}-{slug}` branches with CI validation before PR creation
- **5-stage CI pipeline**: ci-homelab.sh (docker build → pytest → lint → typecheck → security) as the canonical quality gate
- **Pydantic v2 config schema**: All config via `load_config()` → `SwingRLConfig`, never raw YAML parsing
- **structlog context logging**: `log.info("event", key=value)` pattern, never f-strings in log calls
- **Typed exception hierarchy**: `SwingRLError` subclasses (ConfigError, BrokerError, DataError, etc.) — never bare Exception
- **nosec annotations**: Explicit `# nosec B608` on DuckDB SQL interpolation where inputs are constrained by enums

### Key Lessons
1. **Integration testing must happen during development, not just at milestone end.** The gap closure phases (11-17) existed because individual phases were verified in isolation without testing cross-module flows.
2. **Audit before you ship.** The `/gsd:audit-milestone` workflow caught real issues that would have been painful to fix after archival.
3. **Doc-as-code needs maintenance too.** ROADMAP.md and REQUIREMENTS.md drifted from reality during rapid phase execution — a periodic sync step would help.
4. **stockstats over pandas_ta was the right call.** pandas_ta's Python 3.12+ requirement would have been a blocker.
5. **Incremental schema beats upfront DDL.** Creating 28 tables upfront would have created coupling between phases and made testing harder.

### Cost Observations
- Model mix: Balanced profile (opus for planning/verification, sonnet for execution, haiku for research)
- Sessions: ~15-20 across 8 days
- Notable: Gap closure phases (11-17) were small and fast (1 plan each) — the audit pattern efficiently scoped the remaining work

---

## Cross-Milestone Trends

### Process Evolution

| Milestone | Timeline | Phases | Key Change |
|-----------|----------|--------|------------|
| v1.0 | 8 days | 17 | Established audit → gap closure pattern |

### Cumulative Quality

| Milestone | Tests | LOC (src) | LOC (tests) |
|-----------|-------|-----------|-------------|
| v1.0 | 848+ | 15,288 | 18,028 |

### Top Lessons (Verified Across Milestones)

1. Specification-first eliminates design ambiguity but requires maintenance during execution
2. Integration testing during development prevents expensive gap closure phases at milestone end
