# Phase 21: Discord Alert Suite - Context

**Gathered:** 2026-03-15
**Status:** Ready for planning

<domain>
## Phase Boundary

Wire all Discord alert types, route to the correct channels, and confirm delivery. Build 4 new retrain lifecycle embeds required by Phase 22. Expand alert coverage to include memory agent failures, broker auth, system health, container lifecycle, and data pipeline warnings. Add alert history tab to Streamlit dashboard. Update .env.example with three webhook URLs.

</domain>

<decisions>
## Implementation Decisions

### Discord Channel Structure
- **Three channels:** #swingrl-alerts (critical/warning), #swingrl-daily (info/summaries), #swingrl-trades (fill notifications + pending orders)
- **Three webhook URLs in config:** `alerts_webhook_url`, `daily_webhook_url`, `trades_webhook_url`
- `trades_webhook_url` falls back to `daily_webhook_url` if empty (backward compatible)
- Manual channel + webhook creation by operator, documented with detailed setup instructions
- .env.example updated with all three `SWINGRL_ALERTING__*_WEBHOOK_URL` fields with inline comments

### Alert Severity Matrix
- **CRITICAL (#swingrl-alerts, red 0xFF0000):** circuit breaker trigger, broker auth failure, emergency stop activated, container shutdown (graceful), retrain failed, Healthchecks.io ping failed
- **WARNING (#swingrl-alerts, orange 0xFFA500):** stuck agent (10+ equity / 30+ crypto), memory agent 3 consecutive failures, memory latency >3s (3 consecutive), data staleness / quarantine trigger, system health threshold crossed, 3 consecutive cycle failures
- **INFO (#swingrl-daily, blue 0x3498DB):** daily trading summary, memory agent daily digest, retrain started, retrain completed, shadow promoted/rejected, container startup
- **TRADE (#swingrl-trades):** individual trade fill notifications, pending/open order updates
- This matrix goes in CONTEXT.md AND Phase 24 runbook

### Daily Summary Content
- **Rolling 30-day metrics:** Sharpe ratio, Max Drawdown, Sortino Ratio, Calmar Ratio
- **Per-ticker position breakdown:** ticker, qty, entry price, current price, unrealized P&L, % of portfolio
- **Transactions section (separate from ticker summary):** new fills today + pending/open orders
- **Circuit breaker status with thresholds:** "CB: All Clear (MDD: 3.2% / 8% limit)" — always shown, even when not triggered
- **One embed, multiple field groups:** Performance Metrics | Position Breakdown | Today's Transactions | Pending Orders | Circuit Breaker Status
- **Post time:** 5:00 PM ET daily
- **Color:** P&L-based — green if positive, red if negative, blue if flat/no trades
- **Weekend handling:** skip equity section (market closed), crypto section still posts. Memory digest included if crypto cycles ran.

### Memory Agent Daily Digest
- **Separate embed** from trading summary, posted immediately after at 5 PM ET to #swingrl-daily
- **Content:** cycle gates fired (count), trade vetoes (count + tickers), blend weight adjustments (deltas), risk threshold overrides, avg memory latency, top memory insight of the day
- Activity stats + qualitative decisions from the LLM

### Retrain Embeds (4 types for Phase 22)
- **Retrain started (blue 0x3498DB, INFO #daily):** environment, algos queued, estimated duration based on last run
- **Retrain completed (green 0x00FF00, INFO #daily):** full comparison old vs new for all 4 metrics (Sharpe, MDD, Sortino, Calmar) per algo. Training duration, fold pass rate, whether deploying to shadow.
- **Retrain failed (red 0xFF0000, CRITICAL #alerts):** error type, traceback summary (last 3 lines), which algo/env failed, when last successful retrain was, **memory agent context on why it failed**, recommended next steps from memory agent
- **Shadow promoted (gold 0xFFD700, INFO #daily):** active vs shadow comparison table (Sharpe, MDD, Sortino, Calmar), promotion decision rationale, **memory agent's assessment** of model quality and market context
- **Shadow rejected (orange 0xFFA500, INFO #daily):** same comparison table, rejection reason, memory agent's assessment

### Alert Coverage Expansion
- **Broker auth failure:** dedicated `build_broker_auth_embed()` — broker name, error type, which env affected, last successful auth time
- **Memory agent failures:** WARNING after 3 consecutive failures (already fail-open, trading continues)
- **System health checks:** disk space, container memory usage, DB file sizes. WARNING when thresholds crossed (<1GB disk, >90% memory). Configurable interval via `config.alerting.health_check_interval_hours` (default 6)
- **Container lifecycle:** startup embed (green, after startup_checks pass) and shutdown embed (red, on graceful shutdown) to #swingrl-alerts
- **Data pipeline:** stale OHLCV data (>2 days equity, >12h crypto), gap detection failures, quarantine triggers — all as warnings to #swingrl-alerts
- **Healthchecks.io mirror:** if HC.io ping fails (HTTP error), also send warning to Discord as fallback
- **Resolved alerts:** green (0x00FF00) "Resolved" embed when a previously-alerting condition clears, showing issue duration

### Rate Limiting & Notification Fatigue
- **Queue + batch:** if >20 alerts fire in a minute, queue remaining and batch-send as single "Multiple alerts" embed. Never exceed 25 req/min per webhook.
- **Escalating cooldown:** first alert immediate, second at 30 min, third at 2 hours, fourth+ at 6 hours. Resets when issue resolves.
- **Discord only:** no email/SMS escalation. Discord push notifications sufficient for solo operator.
- **Multi-environment:** separate per-environment embeds (one embed per env per issue). No combined alerts.

### Notification Preferences
- **Per-type toggles:** `config.alerting.enabled_types` — list of alert types that are active. Default: all enabled. Remove a type to mute.
- **Severity threshold:** `config.alerting.min_severity` — "critical" | "warning" | "info". Mutes everything below threshold. Works alongside per-type toggles.

### Embed Branding
- **Footer format:** "SwingRL v1.1 | {environment} | {TYPE}" on all embeds
- **Visual style:** clean fields only — no thumbnails, no author icons. Discord markdown (bold, code blocks) in descriptions where helpful.

### Alert History
- **SQLite alert_log table** already exists — extend if needed for new alert types
- **Streamlit dashboard tab:** alert history with timestamp, level, title, channel, sent status. Filterable by severity and date range.
- **Retention:** configurable via `config.alerting.retention_days` (default 90). Auto-purge older records daily.

### Webhook Setup & Smoke Test
- **Manual setup:** operator creates channels + webhooks in Discord server settings, pastes URLs into .env. Detailed step-by-step instructions provided.
- **Startup validation:** `startup_checks.py` validates URL format (non-empty, matches discord.com/api/webhooks/ pattern). Warns (not fails) if missing — trading works without Discord.
- **Smoke test:** extend Phase 20's `smoke_test.py` with Discord section. Real ping to all three webhook URLs. Branded test card: purple embed (0x9B59B6), title "SwingRL Smoke Test", fields: Container ID, Timestamp, Webhook Type.
- **CI tests:** schema validation for all embed builders — required fields (title, color, timestamp, footer), field count <25, description <4096 chars, field values <1024 chars. No HTTP calls in CI.

### Claude's Discretion
- Exact field layout within embeds (inline vs block, ordering)
- System health threshold values (disk, memory, DB size)
- Alert log schema extensions if needed for new alert types
- Queue implementation details for rate limiting
- Escalating cooldown state management (in-memory vs SQLite)
- How to detect "resolved" state for each alert type

</decisions>

<code_context>
## Existing Code Insights

### Reusable Assets
- `src/swingrl/monitoring/alerter.py`: Full Alerter class with two-webhook routing, cooldown, dedup, thread safety, alert_log. Extend with third webhook + rate limit queue + escalating cooldown.
- `src/swingrl/monitoring/embeds.py`: 4 embed builders (trade, daily summary, stuck agent, circuit breaker). Add retrain embeds, broker auth embed, system health embed, lifecycle embeds, resolved embed.
- `src/swingrl/monitoring/stuck_agent.py`: Stuck agent detection with configurable thresholds.
- `src/swingrl/config/schema.py`: AlertingConfig with alerts_webhook_url, daily_webhook_url, healthchecks URLs. Add trades_webhook_url, enabled_types, min_severity, health_check_interval_hours, retention_days.
- `tests/monitoring/test_alerter.py` (561 lines) + `test_embeds.py` (375 lines): Comprehensive test coverage to extend.

### Established Patterns
- httpx for webhook HTTP calls (pinned at 0.28.1)
- Thread-safe with threading.Lock() for mutable alerter state
- SHA-256 hash dedup for info buffer
- Embed dict format: {"embeds": [{"title": ..., "color": ..., "fields": [...], "footer": ..., "timestamp": ...}]}
- structlog keyword args for all logging

### Integration Points
- `src/swingrl/scheduler/jobs.py`: All scheduler jobs already call alerter — extend with new alert types
- `scripts/smoke_test.py` (Phase 20): Add Discord verification section
- `scripts/startup_checks.py` (Phase 20): Add webhook URL validation
- `scripts/main.py`: Wire lifecycle alerts (startup/shutdown)
- Streamlit dashboard: Add alert history page/tab

</code_context>

<specifics>
## Specific Ideas

- Severity matrix is the source of truth for all alert routing decisions — replicate in Phase 24 runbook
- Memory agent context on retrain failures is key — the LLM should analyze why training failed and suggest operator actions
- Per-ticker position breakdown in daily summary — full visibility into what the system holds
- Escalating cooldown prevents notification fatigue without losing awareness of ongoing issues
- Phase 25 (Dashboard Alignment) should be added to roadmap for comprehensive dashboard updates post-v1.1

</specifics>

<deferred>
## Deferred Ideas

- Phase 25: Dashboard Alignment — comprehensive dashboard updates (memory agent tab, model performance views, broader visualization) after v1.1 phases complete. Add to roadmap.
- Email/SMS escalation for critical alerts — Discord sufficient for solo operator, revisit if paper trading reveals gaps
- Embed thumbnails/author icons — minimal value for monitoring use case
- Discord bot (vs webhooks) — would enable interactive commands but adds complexity

</deferred>

---

*Phase: 21-discord-alert-suite*
*Context gathered: 2026-03-15*
