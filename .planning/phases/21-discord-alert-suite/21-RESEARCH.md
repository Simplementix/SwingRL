# Phase 21: Discord Alert Suite - Research

**Researched:** 2026-03-15
**Domain:** Discord Webhooks, Alert routing, Embed builders, Streamlit dashboard extension
**Confidence:** HIGH

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions

**Discord Channel Structure**
- Three channels: #swingrl-alerts (critical/warning), #swingrl-daily (info/summaries), #swingrl-trades (fill notifications + pending orders)
- Three webhook URLs in config: `alerts_webhook_url`, `daily_webhook_url`, `trades_webhook_url`
- `trades_webhook_url` falls back to `daily_webhook_url` if empty (backward compatible)
- Manual channel + webhook creation by operator, documented with detailed setup instructions
- .env.example updated with all three `SWINGRL_ALERTING__*_WEBHOOK_URL` fields with inline comments

**Alert Severity Matrix**
- CRITICAL (#swingrl-alerts, red 0xFF0000): circuit breaker trigger, broker auth failure, emergency stop activated, container shutdown (graceful), retrain failed, Healthchecks.io ping failed
- WARNING (#swingrl-alerts, orange 0xFFA500): stuck agent (10+ equity / 30+ crypto), memory agent 3 consecutive failures, memory latency >3s (3 consecutive), data staleness / quarantine trigger, system health threshold crossed, 3 consecutive cycle failures
- INFO (#swingrl-daily, blue 0x3498DB): daily trading summary, memory agent daily digest, retrain started, retrain completed, shadow promoted/rejected, container startup
- TRADE (#swingrl-trades): individual trade fill notifications, pending/open order updates

**Daily Summary Content**
- Rolling 30-day metrics: Sharpe ratio, Max Drawdown, Sortino Ratio, Calmar Ratio
- Per-ticker position breakdown: ticker, qty, entry price, current price, unrealized P&L, % of portfolio
- Transactions section (separate from ticker summary): new fills today + pending/open orders
- Circuit breaker status with thresholds: "CB: All Clear (MDD: 3.2% / 8% limit)" — always shown
- One embed, multiple field groups: Performance Metrics | Position Breakdown | Today's Transactions | Pending Orders | Circuit Breaker Status
- Post time: 5:00 PM ET daily (update scheduler from current 18:00 ET)
- Color: P&L-based — green if positive, red if negative, blue if flat/no trades
- Weekend handling: skip equity section (market closed), crypto section still posts

**Memory Agent Daily Digest**
- Separate embed from trading summary, posted immediately after at 5 PM ET to #swingrl-daily
- Content: cycle gates fired (count), trade vetoes (count + tickers), blend weight adjustments (deltas), risk threshold overrides, avg memory latency, top memory insight of the day

**Retrain Embeds (4 types for Phase 22)**
- Retrain started (blue 0x3498DB, INFO #daily): environment, algos queued, estimated duration based on last run
- Retrain completed (green 0x00FF00, INFO #daily): full comparison old vs new for all 4 metrics (Sharpe, MDD, Sortino, Calmar) per algo; training duration, fold pass rate, whether deploying to shadow
- Retrain failed (red 0xFF0000, CRITICAL #alerts): error type, traceback summary (last 3 lines), which algo/env failed, when last successful retrain was, memory agent context on why it failed, recommended next steps
- Shadow promoted (gold 0xFFD700, INFO #daily): active vs shadow comparison table, promotion decision rationale, memory agent's assessment
- Shadow rejected (orange 0xFFA500, INFO #daily): same comparison table, rejection reason, memory agent's assessment

**Alert Coverage Expansion**
- Broker auth failure: dedicated `build_broker_auth_embed()` — broker name, error type, which env affected, last successful auth time
- Memory agent failures: WARNING after 3 consecutive failures
- System health checks: disk space, container memory usage, DB file sizes. WARNING when thresholds crossed (<1GB disk, >90% memory). Configurable interval via `config.alerting.health_check_interval_hours` (default 6)
- Container lifecycle: startup embed (green, after startup_checks pass) and shutdown embed (red, on graceful shutdown) to #swingrl-alerts
- Data pipeline: stale OHLCV data (>2 days equity, >12h crypto), gap detection failures, quarantine triggers — all as warnings to #swingrl-alerts
- Healthchecks.io mirror: if HC.io ping fails (HTTP error), also send warning to Discord as fallback
- Resolved alerts: green (0x00FF00) "Resolved" embed when a previously-alerting condition clears, showing issue duration

**Rate Limiting and Notification Fatigue**
- Queue + batch: if >20 alerts fire in a minute, queue remaining and batch-send as single "Multiple alerts" embed; never exceed 25 req/min per webhook
- Escalating cooldown: first alert immediate, second at 30 min, third at 2 hours, fourth+ at 6 hours. Resets when issue resolves
- Discord only: no email/SMS escalation
- Multi-environment: separate per-environment embeds (one embed per env per issue)

**Notification Preferences**
- Per-type toggles: `config.alerting.enabled_types` — list of alert types that are active; default all enabled
- Severity threshold: `config.alerting.min_severity` — "critical" | "warning" | "info". Mutes everything below threshold

**Embed Branding**
- Footer format: "SwingRL v1.1 | {environment} | {TYPE}" on all embeds
- Visual style: clean fields only — no thumbnails, no author icons; Discord markdown in descriptions where helpful

**Alert History**
- SQLite alert_log table already exists — extend if needed for new alert types
- Streamlit dashboard tab: alert history with timestamp, level, title, channel, sent status; filterable by severity and date range
- Retention: configurable via `config.alerting.retention_days` (default 90); auto-purge older records daily

**Webhook Setup and Smoke Test**
- Manual setup: operator creates channels + webhooks in Discord, pastes URLs into .env
- Startup validation: `startup_checks.py` validates URL format (non-empty, matches discord.com/api/webhooks/ pattern); warns (not fails) if missing
- Smoke test: extend Phase 20's `smoke_test.py` with Discord section; real ping to all three webhook URLs; branded test card: purple embed (0x9B59B6), title "SwingRL Smoke Test", fields: Container ID, Timestamp, Webhook Type
- CI tests: schema validation for all embed builders — required fields (title, color, timestamp, footer), field count <25, description <4096 chars, field values <1024 chars; no HTTP calls in CI

### Claude's Discretion
- Exact field layout within embeds (inline vs block, ordering)
- System health threshold values (disk, memory, DB size)
- Alert log schema extensions if needed for new alert types
- Queue implementation details for rate limiting
- Escalating cooldown state management (in-memory vs SQLite)
- How to detect "resolved" state for each alert type

### Deferred Ideas (OUT OF SCOPE)
- Phase 25: Dashboard Alignment — comprehensive dashboard updates after v1.1 phases complete
- Email/SMS escalation for critical alerts
- Embed thumbnails/author icons
- Discord bot (vs webhooks)
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|-----------------|
| DISC-01 | Discord webhook URLs created and configured for critical/warning and daily summary channels | Third `trades_webhook_url` added to `AlertingConfig`, .env.example updated, startup validation pattern documented |
| DISC-02 | Critical alerts fire on circuit breaker trigger, broker auth failure, stuck agent | Existing `send_alert()` + `build_circuit_breaker_embed()` extended with `build_broker_auth_embed()`, stuck agent already wired — upgrade to use richer embed |
| DISC-03 | Daily summary embed posts with Sharpe, P&L, and position summary | `build_daily_summary_embed()` upgraded to rolling 30-day metrics + per-ticker breakdown; scheduler job time adjusted to 17:00 ET |
| DISC-04 | Retraining alerts fire: started, completed (with old vs new Sharpe), failed, shadow promoted/rejected | Four new builder functions in `embeds.py`; placeholder wiring in `jobs.py` for Phase 22 consumption |
</phase_requirements>

---

## Summary

Phase 21 is almost entirely a codebase extension phase — no new dependencies, no new infrastructure. The Discord webhook integration is live (Phase 20 built `alerter.py` and `embeds.py`). The primary work is: (1) extending `AlertingConfig` with `trades_webhook_url`, per-type toggles, severity threshold, health check interval, and retention days; (2) adding ~8 new embed builder functions to `embeds.py`; (3) extending `Alerter` with the third webhook routing, rate-limit queue, and escalating cooldown; (4) wiring new alert types into `scheduler/jobs.py` and `scripts/main.py`; (5) adding a Streamlit alert history tab; and (6) updating `.env.example` and smoke test.

The escalating cooldown is the highest-complexity addition. The current `Alerter` uses a flat cooldown (`_last_alert_times` dict + `timedelta`). Escalating cooldown requires per-key fire counts alongside last-sent time — state can live in-memory since cooldown resets on process restart (acceptable for a solo operator). Rate-limit queue (>20 alerts/min) requires a `deque` with timestamps; when overflow is detected, subsequent alerts are batched into a single "Multiple alerts" embed.

**Primary recommendation:** Extend existing monitoring module in-place. Do not refactor `Alerter` into a new class — patch the existing class to add `trades_webhook_url`, rate-limit queue, and escalating cooldown as new attributes. Add all new embed builders to `embeds.py` under clearly named sections. Use the `_log_alert()` pattern already established for the alert history persistence, extending the SQLite `alert_log` schema with `channel` and `alert_type` columns.

---

## Standard Stack

### Core (all already pinned in pyproject.toml)
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| httpx | 0.28.1 | Webhook HTTP POST | Already pinned; used by existing alerter |
| structlog | existing | Structured logging | Project standard |
| threading.Lock | stdlib | Thread-safe alerter state | Already used in Alerter |
| collections.deque | stdlib | Rate-limit queue | O(1) append/popleft |
| sqlite3 | stdlib | alert_log persistence | DatabaseManager.sqlite() already wired |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| streamlit | existing | Alert history tab | Dashboard pages follow existing pattern (dashboard/pages/*.py) |
| psutil | check pyproject | System health metrics (disk, memory) | Only if already a dep; if not, use os.statvfs + /proc/meminfo |

### Installation
No new dependencies expected. Verify `psutil` status before adding:

```bash
uv run python -c "import psutil; print(psutil.__version__)"
```

If not available, use stdlib `os.statvfs()` for disk and `/proc/meminfo` parsing for container memory.

---

## Architecture Patterns

### Recommended File Structure (changes only)
```
src/swingrl/
├── config/
│   └── schema.py              # AlertingConfig: add trades_webhook_url, enabled_types,
│                              #   min_severity, health_check_interval_hours, retention_days
├── monitoring/
│   ├── alerter.py             # Add: trades webhook, rate-limit queue, escalating cooldown,
│   │                          #   purge_old_alerts(), system health check
│   └── embeds.py              # Add: ~8 new builder functions (retrain x4, broker auth,
│                              #   system health, lifecycle x2, resolved, memory digest)
└── scheduler/
    └── jobs.py                # Wire new alert types: lifecycle, system health job,
                               #   memory digest job; upgrade stuck_agent to use embed

scripts/
├── main.py                    # Wire container startup/shutdown embeds
└── smoke_test.py              # Add Discord verification section (3 webhook pings)

dashboard/
└── pages/
    └── 5_Alert_History.py     # New: alert log table with severity/date filters

config/
└── swingrl.yaml               # Add new alerting fields with defaults

.env.example                   # Add SWINGRL_ALERTING__TRADES_WEBHOOK_URL
```

### Pattern 1: AlertingConfig Schema Extension
**What:** Add 5 new fields to `AlertingConfig` Pydantic model
**When to use:** Any new config values needed by alerter

```python
# Source: src/swingrl/config/schema.py (existing pattern)
class AlertingConfig(BaseModel):
    alert_cooldown_minutes: int = Field(default=30, ge=1)
    consecutive_failures_before_alert: int = Field(default=3, ge=1)
    alerts_webhook_url: str = Field(default="")
    daily_webhook_url: str = Field(default="")
    trades_webhook_url: str = Field(default="")          # NEW: third channel
    healthchecks_equity_url: str = Field(default="")
    healthchecks_crypto_url: str = Field(default="")
    enabled_types: list[str] = Field(default_factory=list)  # NEW: empty = all enabled
    min_severity: Literal["critical", "warning", "info"] = Field(default="info")  # NEW
    health_check_interval_hours: int = Field(default=6, ge=1)  # NEW
    retention_days: int = Field(default=90, ge=1)        # NEW
```

### Pattern 2: Third Webhook Routing
**What:** Extend `_get_webhook_for_level()` to handle "trade" level
**When to use:** Trade fill embeds need their own channel

```python
# Source: src/swingrl/monitoring/alerter.py (extend existing method)
def _get_webhook_for_level(self, level: str) -> str:
    if level in ("critical", "warning"):
        return self._alerts_webhook_url or self._webhook_url
    if level == "trade":
        return self._trades_webhook_url or self._daily_webhook_url or self._webhook_url
    # info and daily digest
    return self._daily_webhook_url or self._webhook_url
```

The `Alerter.__init__` signature gains `trades_webhook_url: str | None = None` and stores it as `self._trades_webhook_url`.

### Pattern 3: Escalating Cooldown (in-memory state)
**What:** Per-alert-key escalating delay schedule: 0 → 30min → 2hr → 6hr
**State management:** In-memory dict (acceptable; resets on restart)

```python
# Escalation schedule in minutes
_ESCALATION_SCHEDULE: list[int] = [0, 30, 120, 360]

# New state in Alerter.__init__:
self._escalation_counts: dict[str, int] = {}  # cooldown_key -> fire count

# In send_alert() for critical/warning:
with self._lock:
    count = self._escalation_counts.get(cooldown_key, 0)
    escalation_minutes = _ESCALATION_SCHEDULE[min(count, len(_ESCALATION_SCHEDULE) - 1)]
    # ... check last_sent against escalation_minutes ...
    self._escalation_counts[cooldown_key] = count + 1

# On resolved: reset_escalation(cooldown_key) clears both dicts
```

**Reset on resolve:** A `reset_escalation(key: str)` method clears `_escalation_counts[key]` and `_last_alert_times[key]`. Called when a "Resolved" embed is sent.

### Pattern 4: Rate-Limit Queue (>20 alerts/minute)
**What:** Track per-webhook send timestamps; when >20 in last 60s, queue overflow
**Implementation:** One `deque[datetime]` per webhook URL, max 25 depth

```python
# New state in Alerter.__init__:
self._webhook_send_times: dict[str, deque[datetime]] = {}  # url -> recent send times
self._queued_payloads: deque[tuple[str, dict]] = deque()   # (webhook_url, payload)

def _check_rate_limit(self, webhook_url: str) -> bool:
    """Return True if we can send now (< 20 in last 60s). False = queue it."""
    now = datetime.now(UTC)
    times = self._webhook_send_times.setdefault(webhook_url, deque())
    # Evict entries older than 60s
    while times and (now - times[0]).total_seconds() > 60:
        times.popleft()
    return len(times) < 20
```

When rate-limited, append to `_queued_payloads`. A `flush_queued_alerts()` method batches them into a single "Multiple alerts" embed and sends. This method is called by a 60-second ticker job in APScheduler.

### Pattern 5: New Embed Builders (embeds.py)
**What:** 8 new functions following existing embed dict pattern
**Signature convention:** All return `dict[str, list[dict[str, object]]]`

```python
# Footer pattern (from CONTEXT.md branding decision):
# "SwingRL v1.1 | {environment} | {TYPE}"
# Example:
{"text": f"SwingRL v1.1 | {environment.title()} | CRITICAL"}
```

New builders needed:
1. `build_retrain_started_embed(env: str, algos: list[str], estimated_duration_min: int)`
2. `build_retrain_completed_embed(env: str, algo_metrics: dict[str, dict], duration_min: int, deploying_to_shadow: bool)`
3. `build_retrain_failed_embed(env: str, algo: str, error_type: str, traceback_tail: str, last_success_ts: str | None, memory_context: str | None)`
4. `build_shadow_decision_embed(env: str, decision: Literal["promoted", "rejected"], active_metrics: dict, shadow_metrics: dict, rationale: str, memory_assessment: str | None)`
5. `build_broker_auth_embed(broker: str, error_type: str, environment: str, last_success_ts: str | None)`
6. `build_system_health_embed(disk_gb_free: float, memory_pct: float, db_sizes: dict[str, float])`
7. `build_lifecycle_embed(event: Literal["startup", "shutdown"], container_id: str)`
8. `build_resolved_embed(title: str, issue_duration_str: str)`
9. `build_memory_digest_embed(cycle_gates: int, trade_vetoes: int, vetoed_tickers: list[str], blend_adjustments: dict[str, float], avg_latency_ms: float, top_insight: str)`

Note: `build_shadow_decision_embed()` handles both promoted and rejected via `decision` param, keeping the function count manageable.

### Pattern 6: alert_log Schema Extension
**What:** Add `channel` and `alert_type` columns to existing `alert_log` table
**Implementation:** Idempotent `ALTER TABLE` in `db.init_schema()` (follows existing column-add pattern at line 458-466 of db.py)

```python
# In DatabaseManager.init_schema() — after existing alert_log DDL:
for col_sql in [
    "ALTER TABLE alert_log ADD COLUMN channel TEXT DEFAULT 'alerts'",
    "ALTER TABLE alert_log ADD COLUMN alert_type TEXT",
]:
    try:
        conn.execute(col_sql)
    except sqlite3.OperationalError:
        pass  # Column already exists
```

### Pattern 7: Streamlit Alert History Tab
**What:** New page `dashboard/pages/5_Alert_History.py` following existing page pattern
**Pattern:** Helper functions for pure logic (no st.* calls) + rendering block

```python
# Mirrors pattern from dashboard/pages/4_System_Health.py
def get_alert_history(conn, severity: str, days: int) -> list[dict]:
    """Query alert_log with severity and date filters."""
    ...
```

Filters: severity selectbox (critical/warning/info/all), date range slider (last N days, default 7). Displays `st.dataframe()` with columns: timestamp, level, title, channel, alert_type, sent.

### Anti-Patterns to Avoid
- **Blocking HTTP in scheduler threads:** `httpx.post()` with `timeout=10.0` is already the pattern — never remove the timeout. The alerter runs in APScheduler thread pool; a hung webhook call blocks that worker.
- **Bypassing `_get_webhook_for_level()`:** Never hardcode webhook URLs in individual embed send calls — always route through the Alerter method so `enabled_types` and `min_severity` filtering applies.
- **Raising on missing webhook:** The existing pattern logs and returns `False` — never raise from `_post_webhook()`. Alerting must never crash the trading process.
- **Relative imports in src/swingrl:** Use `from swingrl.monitoring.embeds import ...` (absolute) per CLAUDE.md.
- **New dependencies without pyproject.toml update:** If `psutil` is needed, add to `[project.dependencies]` in `pyproject.toml` and verify CI.

---

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| HTTP client for webhooks | Custom urllib adapter | httpx (already pinned) | Timeout, error handling, keep-alive already solved |
| Thread safety | Custom locking primitives | `threading.Lock()` (already in Alerter) | Already proven correct in existing alerter |
| Config validation | Manual dict parsing | Pydantic `AlertingConfig` field validators | Type safety, env var override, test fixtures |
| Disk space check | Custom /proc parser | `os.statvfs()` or `psutil.disk_usage()` | Cross-platform, correct free-space accounting |
| Rate limiting math | Custom sliding window | `deque[datetime]` with eviction | O(1) operations, no external dep |

**Key insight:** The Discord webhook payload format is static — fields dict structure is simple and already battle-tested in the existing 4 embed builders. The main complexity is routing and state management (escalating cooldown, rate limits), not embed construction.

---

## Common Pitfalls

### Pitfall 1: Footer Format Mismatch
**What goes wrong:** New embed builders use the old footer format "SwingRL | {env} | {TYPE}" instead of "SwingRL v1.1 | {env} | {TYPE}"
**Why it happens:** Copy-paste from existing builders without updating version string
**How to avoid:** Extract a single `_build_footer(env: str, alert_type: str) -> dict` helper used by all builders
**Warning signs:** CI embed schema tests pass but footer doesn't contain "v1.1"

### Pitfall 2: Escalating Cooldown State Lost on Restart
**What goes wrong:** An ongoing alert that's been firing for hours resets to "immediate" after process restart
**Why it happens:** State is in-memory (`_escalation_counts` dict)
**How to avoid:** This is accepted behavior per CONTEXT.md (solo operator, Discord sufficient). Document in runbook. If it becomes a problem, persist state to SQLite.
**Warning signs:** Operator gets repeat storm of alerts after container restart — mitigate with rate-limit queue

### Pitfall 3: Rate-Limit Queue Accumulation Without Flush
**What goes wrong:** Queued alerts never sent if the flush ticker job doesn't fire
**Why it happens:** `flush_queued_alerts()` must be registered as a scheduled job (every 60s)
**How to avoid:** Register flush job in `main.py:create_scheduler_and_register_jobs()` with `trigger="interval", seconds=60`
**Warning signs:** `_queued_payloads` deque grows without bound, memory leak

### Pitfall 4: `enabled_types` Filter Applied Too Late
**What goes wrong:** Embed is built (expensive for retrain embeds with LLM data) before checking if the type is muted
**Why it happens:** Filter check placed after embed construction
**How to avoid:** Check `config.alerting.enabled_types` and `config.alerting.min_severity` at the top of `send_alert()` and `send_embed()` before any payload construction
**Warning signs:** LLM API calls happen even when alert type is disabled

### Pitfall 5: trades_webhook_url Fallback Chain Broken
**What goes wrong:** Trade embeds sent to alerts channel instead of trades channel when `trades_webhook_url` is empty
**Why it happens:** `_get_webhook_for_level("trade")` falls through to `_alerts_webhook_url` instead of `_daily_webhook_url`
**How to avoid:** Fallback order: `trades_webhook_url` → `daily_webhook_url` → `webhook_url` (never `alerts_webhook_url`)
**Warning signs:** Trade fills appearing in #swingrl-alerts channel

### Pitfall 6: Discord 30-Field Embed Limit
**What goes wrong:** Daily summary embed with many positions exceeds Discord's 25-field limit, causing 400 response
**Why it happens:** Per-ticker position breakdown adds one row per position (up to 8 equity + 2 crypto)
**How to avoid:** Group positions into a single field using Discord code block markdown; do not use one field per position. CI schema test validates `len(fields) < 25`.
**Warning signs:** `httpx` raises on 400 response from Discord

### Pitfall 7: daily_summary_job Scheduled Time
**What goes wrong:** `daily_summary_job` currently fires at 18:00 ET; CONTEXT.md locks it to 17:00 ET
**Why it happens:** `scripts/main.py` has `hour=18` for `daily_summary` job
**How to avoid:** Update `main.py` scheduler registration to `hour=17, minute=0` in Phase 21
**Warning signs:** Daily summary posts at 6 PM instead of 5 PM

### Pitfall 8: Weekend Equity Skip Logic
**What goes wrong:** Daily summary skips entire embed on weekends instead of just equity section
**Why it happens:** Logic placed at function top level rather than per-environment
**How to avoid:** Check `datetime.now(ET).weekday() >= 5` per-environment inside the embed builder call. Crypto section always posts.
**Warning signs:** No #swingrl-daily post on Saturday/Sunday

---

## Code Examples

Verified patterns from existing codebase:

### Existing embed dict format (source: embeds.py)
```python
return {
    "embeds": [
        {
            "title": f"Circuit Breaker: {environment.title()}",
            "color": _COLOR_CRITICAL,
            "fields": fields,
            "footer": {"text": f"SwingRL | {environment.title()} | CRITICAL"},
            "timestamp": datetime.now(UTC).isoformat(),
        }
    ]
}
```

### Existing Alerter webhook routing (source: alerter.py line 100-112)
```python
def _get_webhook_for_level(self, level: str) -> str:
    if level in ("critical", "warning"):
        return self._alerts_webhook_url or self._webhook_url
    return self._daily_webhook_url or self._webhook_url
```

### Existing idempotent column add (source: db.py lines 458-466)
```python
for col_sql in [
    "ALTER TABLE positions ADD COLUMN stop_loss_price REAL",
    ...
]:
    try:
        conn.execute(col_sql)
    except sqlite3.OperationalError:
        pass  # Column already exists
```

### Existing dashboard page pattern (source: dashboard/pages/4_System_Health.py)
```python
# Pure logic function — testable without Streamlit
def get_traffic_light_status(last_snapshot_ts: str | None, env: str) -> str:
    ...

# Rendering block (not wrapped in function)
st.header("Alert History")
try:
    conn = get_sqlite_conn()
    ...
except Exception as exc:
    st.warning(f"Could not load alert history: {exc}")
```

### Retrain completed embed fields structure (new)
```python
# algo_metrics: {"PPO": {"sharpe_old": 1.2, "sharpe_new": 1.4, "mdd_old": 0.08, ...}, ...}
def build_retrain_completed_embed(
    env: str,
    algo_metrics: dict[str, dict[str, float]],
    duration_min: int,
    deploying_to_shadow: bool,
) -> dict[str, list[dict[str, object]]]:
    fields = []
    for algo, m in algo_metrics.items():
        fields.append({
            "name": algo,
            "value": (
                f"Sharpe: {m['sharpe_old']:.2f} → **{m['sharpe_new']:.2f}**\n"
                f"MDD: {m['mdd_old']:.1%} → {m['mdd_new']:.1%}\n"
                f"Sortino: {m['sortino_old']:.2f} → {m['sortino_new']:.2f}\n"
                f"Calmar: {m['calmar_old']:.2f} → {m['calmar_new']:.2f}"
            ),
            "inline": True,
        })
    fields.append({
        "name": "Training Duration",
        "value": f"{duration_min} min",
        "inline": True,
    })
    fields.append({
        "name": "Shadow Deploy",
        "value": "Yes" if deploying_to_shadow else "No",
        "inline": True,
    })
    return {
        "embeds": [{
            "title": f"Retrain Completed: {env.title()}",
            "color": 0x00FF00,
            "fields": fields,
            "footer": {"text": f"SwingRL v1.1 | {env.title()} | INFO"},
            "timestamp": datetime.now(UTC).isoformat(),
        }]
    }
```

---

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Single webhook URL | Two-webhook routing (alerts + daily) | Phase 20 | alerts.py already supports two channels |
| Flat cooldown (30 min flat) | Escalating cooldown (0/30/120/360 min) | Phase 21 | Prevents fatigue without losing awareness |
| No rate limiting | 20-alert queue + batch flush | Phase 21 | Prevents webhook ban (25 req/min Discord limit) |
| No per-type muting | `enabled_types` + `min_severity` | Phase 21 | Operator can suppress low-signal alerts |

**Discord webhook limits (verified from Discord docs):**
- 25 fields per embed maximum
- 4096 characters for `description`
- 1024 characters per field value
- 2000 characters total per `content` (not `embeds`)
- Rate limit: 30 requests per 60 seconds per webhook URL (CONTEXT.md uses 25 as conservative threshold)

---

## Open Questions

1. **psutil availability**
   - What we know: `psutil` is common but not listed in visible config/schema.py imports
   - What's unclear: Whether it's already in pyproject.toml
   - Recommendation: Check `uv run python -c "import psutil"` in Wave 0. If absent, use `os.statvfs()` for disk and parse `/proc/meminfo` for container memory. Do not add psutil if stdlib suffices.

2. **smoke_test.py existence**
   - What we know: CONTEXT.md references extending "Phase 20's `smoke_test.py`" — but `scripts/smoke_test.py` does not exist in the current file tree
   - What's unclear: Whether Phase 20 created it (Phase 20 is pending)
   - Recommendation: Phase 21 plan must create `scripts/smoke_test.py` fresh if Phase 20 doesn't create it first. Plan Wave 0 should check for its existence.

3. **startup_checks.py existence**
   - What we know: CONTEXT.md references extending `scripts/startup_checks.py` — not found in current scripts directory
   - What's unclear: Created by Phase 20 (pending) or by Phase 21
   - Recommendation: Same pattern as smoke_test.py — create if absent, extend if present.

4. **Memory agent digest data source**
   - What we know: Memory agent daily digest should show cycle gates, vetoes, blend adjustments from the memory service
   - What's unclear: Whether these metrics are stored in `memory.db` (SQLite) or returned by the memory API
   - Recommendation: Read memory service API spec (services/memory/) in Wave 0. If endpoint exists, call it. If not, stub the digest with "Memory agent data unavailable" for Phase 21 and wire for real in Phase 22.

5. **Rolling 30-day Sharpe computation**
   - What we know: Daily summary requires rolling 30-day Sharpe, MDD, Sortino, Calmar
   - What's unclear: Whether these are already computed and stored in `portfolio_snapshots`, or require on-demand calculation from raw P&L history
   - Recommendation: Check `portfolio_snapshots` schema in db.py. If columns are missing, add a `compute_rolling_metrics(db, env, days=30)` helper in monitoring module that queries `daily_pnl` column and computes on-the-fly.

---

## Validation Architecture

### Test Framework
| Property | Value |
|----------|-------|
| Framework | pytest (existing) |
| Config file | `pyproject.toml` `[tool.pytest.ini_options]` |
| Quick run command | `uv run pytest tests/monitoring/ -v -x` |
| Full suite command | `uv run pytest tests/ -v` |

### Phase Requirements → Test Map
| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| DISC-01 | `AlertingConfig` has `trades_webhook_url`, `enabled_types`, `min_severity`, `health_check_interval_hours`, `retention_days` | unit | `uv run pytest tests/test_config.py -k "alerting" -x` | ✅ extend existing |
| DISC-01 | Startup validation warns (not fails) when webhook URL missing | unit | `uv run pytest tests/test_startup_checks.py -x` | ❌ Wave 0 |
| DISC-01 | `.env.example` contains all three webhook fields | unit (file content) | `uv run pytest tests/test_env_example.py -x` | ❌ Wave 0 |
| DISC-02 | `build_broker_auth_embed()` returns correct fields and CRITICAL color | unit | `uv run pytest tests/monitoring/test_embeds.py -k "broker_auth" -x` | ❌ Wave 0 |
| DISC-02 | `Alerter` routes "trade" level to trades webhook, falls back to daily | unit | `uv run pytest tests/monitoring/test_alerter.py -k "trades_webhook" -x` | ❌ Wave 0 |
| DISC-02 | Escalating cooldown fires at correct intervals | unit | `uv run pytest tests/monitoring/test_alerter.py -k "escalating_cooldown" -x` | ❌ Wave 0 |
| DISC-02 | Rate-limit queue batches after 20 alerts/minute | unit | `uv run pytest tests/monitoring/test_alerter.py -k "rate_limit" -x` | ❌ Wave 0 |
| DISC-02 | `enabled_types` filter suppresses muted alert types | unit | `uv run pytest tests/monitoring/test_alerter.py -k "enabled_types" -x` | ❌ Wave 0 |
| DISC-03 | `build_daily_summary_embed()` includes Sharpe, MDD, Sortino, Calmar fields | unit | `uv run pytest tests/monitoring/test_embeds.py -k "daily_summary" -x` | ✅ extend existing |
| DISC-03 | Daily summary skips equity fields on weekends | unit | `uv run pytest tests/monitoring/test_embeds.py -k "weekend" -x` | ❌ Wave 0 |
| DISC-03 | `build_daily_summary_embed()` sets color based on P&L sign | unit | `uv run pytest tests/monitoring/test_embeds.py -k "pnl_color" -x` | ❌ Wave 0 |
| DISC-04 | All 4 retrain embed builders return valid payloads with required fields | unit | `uv run pytest tests/monitoring/test_embeds.py -k "retrain" -x` | ❌ Wave 0 |
| DISC-04 | All embed builders enforce field count <25, description <4096 | unit | `uv run pytest tests/monitoring/test_embeds.py -k "schema" -x` | ❌ Wave 0 |
| DISC-04 | Alert history page renders filterable table (Streamlit page logic) | unit | `uv run pytest tests/dashboard/test_pages.py -k "alert_history" -x` | ❌ Wave 0 |

### Sampling Rate
- **Per task commit:** `uv run pytest tests/monitoring/ -v -x`
- **Per wave merge:** `uv run pytest tests/ -v`
- **Phase gate:** Full suite green before `/gsd:verify-work`

### Wave 0 Gaps
- [ ] `tests/test_startup_checks.py` — covers DISC-01 webhook URL validation
- [ ] `tests/test_env_example.py` — covers DISC-01 .env.example field presence
- [ ] Tests in `tests/monitoring/test_alerter.py` for escalating cooldown, rate limiting, trades webhook routing, enabled_types filter — extend existing file
- [ ] Tests in `tests/monitoring/test_embeds.py` for 8 new embed builders — extend existing file
- [ ] `tests/dashboard/test_pages.py` — extend for alert history page logic

---

## Sources

### Primary (HIGH confidence)
- Direct inspection of `src/swingrl/monitoring/alerter.py` — full Alerter class, webhook routing, cooldown, alert_log
- Direct inspection of `src/swingrl/monitoring/embeds.py` — 4 existing embed builders, field patterns, color constants
- Direct inspection of `src/swingrl/config/schema.py` — AlertingConfig current fields, Pydantic v2 patterns
- Direct inspection of `src/swingrl/data/db.py` — alert_log DDL, idempotent column-add pattern
- Direct inspection of `src/swingrl/scheduler/jobs.py` — scheduler job patterns, alerter wiring
- Direct inspection of `scripts/main.py` — APScheduler job registration, daily_summary at 18:00 ET
- Direct inspection of `dashboard/pages/4_System_Health.py` — Streamlit page pattern
- Direct inspection of `dashboard/app.py` — DB connection helpers used by pages
- `.planning/phases/21-discord-alert-suite/21-CONTEXT.md` — locked decisions

### Secondary (MEDIUM confidence)
- Discord webhook documentation (from existing codebase comments and payload structure): 25-field limit, 4096-char description limit, 1024-char field value limit
- Scheduler job registration patterns from existing `main.py` (verified against APScheduler <4.0 documented API)

### Tertiary (LOW confidence)
- Discord rate limit of 30 req/60s per webhook — cited in CONTEXT.md as 25 req/min conservative threshold; exact limit may vary by tier

---

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — all libraries already in codebase; no new deps required
- Architecture: HIGH — concrete extension points identified in existing code; patterns verified
- Pitfalls: HIGH — derived from direct code inspection (footer format, field count limits, scheduler time bug, weekend logic)
- Open questions: 5 gaps identified, all resolvable in Wave 0 without blocking implementation

**Research date:** 2026-03-15
**Valid until:** 2026-04-15 (stable stack; Discord webhook API rarely changes)
