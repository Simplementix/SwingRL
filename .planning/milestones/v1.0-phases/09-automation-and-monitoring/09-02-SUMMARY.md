---
phase: 09-automation-and-monitoring
plan: 02
subsystem: monitoring
tags: [discord, webhooks, embeds, wash-sale, stuck-agent, alerter]

# Dependency graph
requires:
  - phase: 04-storage
    provides: "Alerter class, DatabaseManager, SQLite schema (portfolio_snapshots, wash_sale_tracker)"
  - phase: 08-paper-trading
    provides: "FillResult dataclass, execution types"
provides:
  - "Discord embed builders for trade, daily summary, stuck agent, circuit breaker alerts"
  - "Two-webhook Alerter routing (critical/warning vs info/daily)"
  - "send_embed method for pre-built embed payloads"
  - "Stuck agent detection with per-environment thresholds"
  - "Wash sale scanner for equity fills with 30-day window tracking"
affects: [09-automation-and-monitoring, 10-hardening]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Discord embed builder functions returning webhook-compatible dicts"
    - "Two-webhook routing with single-webhook fallback"
    - "In-memory SQLite mock pattern for DatabaseManager testing"

key-files:
  created:
    - src/swingrl/monitoring/embeds.py
    - src/swingrl/monitoring/stuck_agent.py
    - src/swingrl/monitoring/wash_sale.py
    - tests/monitoring/test_embeds.py
    - tests/monitoring/test_stuck_agent.py
    - tests/monitoring/test_wash_sale.py
  modified:
    - src/swingrl/monitoring/alerter.py
    - tests/monitoring/test_alerter.py

key-decisions:
  - "Stuck agent threshold 10 for equity (trading days), 30 for crypto (4H cycles)"
  - "Two-webhook routing via _get_webhook_for_level helper with fallback to single URL"

patterns-established:
  - "Embed builder pattern: functions return dict payloads, caller sends via alerter.send_embed"
  - "Mock DatabaseManager with in-memory SQLite for isolated integration tests"

requirements-completed: [PAPER-13, PAPER-14, PAPER-17]

# Metrics
duration: 5min
completed: 2026-03-09
---

# Phase 09 Plan 02: Discord Alerting Layer Summary

**Discord embed builders for 4 alert types, two-webhook Alerter routing, stuck agent detection, and wash sale scanner**

## Performance

- **Duration:** 5 min
- **Started:** 2026-03-09T21:24:37Z
- **Completed:** 2026-03-09T21:29:39Z
- **Tasks:** 2
- **Files modified:** 8

## Accomplishments
- 4 embed builders (trade, daily summary, stuck agent, circuit breaker) producing Discord webhook-compatible payloads
- Extended Alerter with two-webhook routing (alerts vs daily) and backward compatibility
- Stuck agent detection querying portfolio_snapshots with environment-specific thresholds
- Wash sale scanner flagging equity buys within 30-day loss windows, ignoring crypto

## Task Commits

Each task was committed atomically:

1. **Task 1: Discord embed builders and stuck agent detection (TDD)**
   - `ba46149` (test: RED - failing tests for embeds and stuck agent)
   - `63133bb` (feat: GREEN - implement embed builders and stuck agent detection)
2. **Task 2: Alerter two-webhook routing and wash sale scanner (TDD)**
   - `cb2a712` (test: RED - failing tests for alerter extension and wash sale)
   - `887afe9` (feat: GREEN - implement two-webhook alerter and wash sale scanner)

## Files Created/Modified
- `src/swingrl/monitoring/embeds.py` - Discord embed builder functions for all 4 alert types
- `src/swingrl/monitoring/stuck_agent.py` - Stuck agent detection with 10/30 thresholds
- `src/swingrl/monitoring/wash_sale.py` - Wash sale scanner and realized loss recording
- `src/swingrl/monitoring/alerter.py` - Extended with two-webhook routing and send_embed
- `tests/monitoring/test_embeds.py` - 22 tests for embed builders
- `tests/monitoring/test_stuck_agent.py` - 8 tests for stuck agent detection
- `tests/monitoring/test_wash_sale.py` - 7 tests for wash sale scanner
- `tests/monitoring/test_alerter.py` - 7 new tests for routing/send_embed/backward compat

## Decisions Made
- Stuck agent uses abs(cash_balance - total_value) < 0.01 threshold for "all cash" detection
- Two-webhook routing implemented via _get_webhook_for_level helper -- critical/warning to alerts URL, info to daily URL, with fallback to single webhook_url for backward compatibility
- send_embed method bypasses _build_embed for pre-built payloads from embeds.py

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered
None.

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- Monitoring modules ready for integration with APScheduler in Plan 03
- Embed builders can be called from cycle runners and passed to alerter.send_embed
- Stuck agent check can be scheduled as periodic job

---
*Phase: 09-automation-and-monitoring*
*Completed: 2026-03-09*
