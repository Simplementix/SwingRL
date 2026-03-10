# Phase 9: Automation and Monitoring - Research

**Researched:** 2026-03-09
**Domain:** Scheduled job execution, Discord alerting, Streamlit dashboards, dead man's switch monitoring
**Confidence:** HIGH

## Summary

Phase 9 transforms SwingRL from a manually-triggered system into a fully autonomous trading bot. The core technical domains are: (1) APScheduler 3.x with SQLAlchemy jobstore for crash-resilient cron scheduling, (2) Discord webhook embeds for trade alerts, circuit breaker notifications, daily summaries, and stuck agent detection, (3) Streamlit multi-page dashboard in a separate Docker container for system health monitoring, (4) Healthchecks.io HTTP pings as dead man's switch for both environments, and (5) wash sale tracking for equity tax compliance.

The existing codebase provides strong foundations: `Alerter` (Phase 4) handles Discord webhooks with cooldown/dedup, `ExecutionPipeline.execute_cycle()` (Phase 8) is the complete trading cycle callable, `DatabaseManager` has all required tables including `wash_sale_tracker` and `circuit_breaker_events`, and the Dockerfile has a placeholder `CMD` explicitly awaiting APScheduler replacement.

**Primary recommendation:** Build `scripts/main.py` as the single process entrypoint that initializes APScheduler BackgroundScheduler with SQLite jobstore, registers all cron jobs, starts the crypto stop-price polling daemon thread, and blocks. Each job wraps `ExecutionPipeline.execute_cycle()` with pre-cycle halt checks, post-cycle HC pings, and Discord alerts.

<user_constraints>

## User Constraints (from CONTEXT.md)

### Locked Decisions
- **Single main.py entrypoint**: one Python process initializes config, DB, APScheduler (BackgroundScheduler with SQLAlchemy jobstore), registers all jobs, and blocks. Replaces Phase 8's sleep-loop placeholder CMD in Dockerfile
- **Pre-cycle halt check**: every job checks emergency_flags table in SQLite before executing. If halt flag is active, skip the cycle and log it (PAPER-12 requirement)
- **Job failure handling**: log the error, send Discord critical alert, skip this cycle. Next scheduled cycle starts fresh -- no retry queue, no immediate retry
- **Graceful shutdown**: Claude's discretion on SIGTERM handling
- **Register what's needed**: don't force-match Doc 05's estimate of 13 jobs. HC pings and status page regen are post-cycle callbacks, not separate scheduled jobs
- **Equity trading cycle**: 1 cron job, daily at 4:15 PM ET
- **Crypto trading cycle**: 1 cron job firing 6x/day at 5 min past each 4H bar close (00:05, 04:05, 08:05, 12:05, 16:05, 20:05 UTC)
- **Daily summary**: 1 cron job at 6:00 PM ET
- **Stuck agent check**: 1 cron job (daily). Thresholds: 10 consecutive equity trading days or 30 consecutive crypto cycles all-cash
- **Wash sale scan**: runs after each equity cycle (callback, not separate job). Equities only
- **Weekly fundamentals refresh**: 1 cron job, Sunday ~6 PM ET
- **Monthly macro refresh**: 1 cron job, 1st of month ~6 PM ET
- **Data pipeline integrated**: data fetch + validate + store runs as part of each trading cycle, not as a separate job
- **Trade alerts**: detailed per-trade -- separate embed per fill with symbol, side, quantity, fill price, stop/TP levels, risk check results, observation summary
- **Daily summary (6 PM ET)**: portfolio + P&L focus -- per-environment current value, daily P&L ($ and %), positions held, trades today. Combined: total portfolio value, combined daily P&L, circuit breaker status
- **Stuck agent alerts**: include diagnostics -- consecutive all-cash cycle count, last non-trivial action date, current regime state, turbulence level
- **Channel split by severity**: critical/warning alerts -> #swingrl-alerts channel, daily summary/info -> #swingrl-daily channel. Two webhook URLs in config
- **Existing Alerter**: Phase 4's Alerter (cooldown, dedup, daily digest, thread-safety) is the foundation. Extend with trade-specific and summary-specific embed builders
- **Streamlit only**: no static HTML status page layer. Streamlit serves as both quick-glance status and analytical dashboard
- **Separate Docker container**: swingrl-dashboard service in docker-compose. Reads from same DB bind mounts (read-only). Isolates web surface from trading bot
- **Full multi-page dashboard**: Page 1: Portfolio overview, Page 2: Trade log with filters, Page 3: Risk metrics, Page 4: System health
- **Auto-refresh every 5 minutes** via streamlit-autorefresh component
- **Healthchecks.io ping after each successful cycle**: crypto check pinged after each crypto cycle, equity check after each equity cycle. If cycle fails, no ping -> dead man's switch fires
- **Alerts via Discord webhook**: configure Healthchecks.io to alert via Discord webhook. No email fallback
- **Post-cycle callback**: HC ping is a callback at the end of each cycle, not a separate scheduled job
- **Wash sale warn only, don't block**: send Discord warning alert when buying a symbol within its 30-day wash window. Log triggered=true. Don't prevent the trade
- **Equities only**: crypto exempt under current IRS rules
- **Emergency stop DB flag + docker exec**: emergency_stop.py sets halt flag in SQLite (emergency_flags table)
- **DB flag only (no scheduler.shutdown())**: the flag prevents all future cycles. APScheduler keeps running but every job is a no-op
- **Stop-price polling daemon thread**: main.py starts APScheduler AND launches the crypto stop-price polling daemon thread

### Claude's Discretion
- APScheduler configuration details (jobstore path, executor type, max_instances)
- Streamlit page layout and styling (chart libraries, color schemes)
- Discord embed field layout and formatting details
- SIGTERM graceful shutdown implementation specifics
- Status page regen callback implementation
- Healthchecks.io ping implementation (HTTP GET to check URL)
- Emergency flags table schema
- Wash sale scanner implementation details

### Deferred Ideas (OUT OF SCOPE)
- Comprehensive error handling with exponential backoff for all API calls -- Phase 10 (HARD-02)
- Structured JSON logging to bind-mounted logs/ volume -- Phase 10 (HARD-05)
- Backup automation (daily SQLite, weekly DuckDB, monthly off-site rsync) -- Phase 10 (PROD-01)
- deploy_model.sh (M1 Mac to homelab SCP + smoke test) -- Phase 10 (PROD-02)
- Shadow mode for new models -- Phase 10 (PROD-03/04)
- Security review (non-root containers, IP allowlisting, key rotation) -- Phase 10 (PROD-06)
- emergency_stop.py four-tier protocol with Tier 2-4 (liquidation, extended hours) -- Phase 10 (PROD-07)
- Disaster recovery testing -- Phase 10 (PROD-08/09)
- Jupyter analysis notebooks -- Phase 10 (HARD-01)

</user_constraints>

<phase_requirements>

## Phase Requirements

| ID | Description | Research Support |
|----|-------------|-----------------|
| PAPER-12 | APScheduler: equity daily at 4:15 PM ET, crypto every 4H at 5 min past bar close, pre-cycle halt checks | APScheduler 3.x CronTrigger with timezone, SQLAlchemy jobstore for crash resilience, emergency_flags table for halt checks |
| PAPER-13 | Discord webhook alerting: trade executions, circuit breakers, daily summary (6 PM ET), stuck agent detection | Extend existing Alerter with embed builders, two-webhook channel routing |
| PAPER-14 | Stuck agent detection: alert if environment stays 100% cash for 10 equity days or 30 crypto cycles | Query trades/portfolio_snapshots tables for consecutive all-cash cycles |
| PAPER-15 | Streamlit dashboard for system health monitoring with traffic-light status | Streamlit multi-page app with streamlit-autorefresh, separate Docker container |
| PAPER-16 | Healthchecks.io dead man's switch (70-min crypto window, 25-hr equity window) | Simple HTTP GET to hc-ping.com/<uuid> after each successful cycle |
| PAPER-17 | Wash sale tracker: log realized losses and flag 30-day wash sale violations | wash_sale_tracker table exists; build scanner logic + Discord warning alerts |

</phase_requirements>

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| apscheduler | 3.11.x | Background job scheduling with cron triggers | Stable production release; v4 still alpha. SQLAlchemy jobstore for crash resilience |
| streamlit | >=1.35 | Multi-page monitoring dashboard | Standard Python dashboard framework; separate container isolates web surface |
| streamlit-autorefresh | >=1.0 | Frontend timer for 5-minute auto-refresh | Official community component for dashboard auto-rerun without blocking server |
| plotly | >=5.18 | Interactive charts (equity curves, drawdown) | Works natively with Streamlit via st.plotly_chart; better interactivity than matplotlib |
| httpx | >=0.27 | Healthchecks.io HTTP pings | Already installed in project; async-capable, timeout-aware |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| sqlalchemy | >=2.0 | APScheduler jobstore backend | Required by apscheduler SQLAlchemy jobstore; uses SQLite URL |
| pytz | >=2024.1 | APScheduler 3.x timezone for CronTrigger | APScheduler 3.x uses pytz for timezone specification (not zoneinfo) |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| plotly | altair | Altair is Streamlit's native charting; plotly has richer financial chart support (candlesticks, range sliders) |
| streamlit-autorefresh | st.rerun + time.sleep | Blocks server resources; autorefresh uses frontend timer, much cleaner |
| APScheduler 3.x | APScheduler 4.x | v4 is alpha (4.0.0a6); breaking API changes; not production-ready |

**Installation:**
```bash
uv add "apscheduler>=3.11,<4" sqlalchemy streamlit streamlit-autorefresh plotly
```

Note: `pytz` is typically already available as a transitive dependency. `httpx` is already installed.

## Architecture Patterns

### Recommended Project Structure
```
src/swingrl/
├── monitoring/
│   ├── alerter.py           # Existing (Phase 4) -- extend
│   ├── embeds.py            # NEW: Discord embed builders (trade, summary, stuck, CB)
│   ├── stuck_agent.py       # NEW: Stuck agent detection logic
│   └── wash_sale.py         # NEW: Wash sale scanner
├── scheduler/
│   ├── __init__.py
│   ├── jobs.py              # NEW: Job functions (equity_cycle, crypto_cycle, etc.)
│   └── halt_check.py        # NEW: Pre-cycle emergency halt check
scripts/
├── main.py                  # NEW: Single entrypoint -- APScheduler + daemon thread
├── emergency_stop.py        # NEW: Set halt flag + cancel broker orders
├── reset_halt.py            # NEW: Clear halt flag to resume operations
dashboard/
├── app.py                   # NEW: Streamlit multi-page entry
├── pages/
│   ├── 1_Portfolio.py       # Page 1: Equity curves, P&L
│   ├── 2_Trade_Log.py       # Page 2: Filterable trade history
│   ├── 3_Risk_Metrics.py    # Page 3: Drawdown, CB status
│   └── 4_System_Health.py   # Page 4: Traffic light, heartbeats, last 5 trades
├── Dockerfile.dashboard     # NEW: Separate container for dashboard
```

### Pattern 1: Main Entrypoint with APScheduler
**What:** Single process that initializes all components, registers cron jobs, and blocks.
**When to use:** Always -- this is the production CMD.
**Example:**
```python
# Source: APScheduler 3.x docs + project CONTEXT.md
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.jobstores.sqlalchemy import SQLAlchemyJobStore
from apscheduler.executors.pool import ThreadPoolExecutor
import signal

jobstores = {
    "default": SQLAlchemyJobStore(url="sqlite:///db/apscheduler_jobs.sqlite")
}
executors = {
    "default": ThreadPoolExecutor(max_workers=4)
}
job_defaults = {
    "coalesce": True,        # Merge missed runs into one
    "max_instances": 1,      # Never overlap same job
    "misfire_grace_time": 300  # 5 min grace for misfired jobs
}

scheduler = BackgroundScheduler(
    jobstores=jobstores,
    executors=executors,
    job_defaults=job_defaults,
)

# Register cron jobs
scheduler.add_job(equity_cycle, "cron", hour=16, minute=15,
                  timezone="America/New_York", id="equity_cycle",
                  replace_existing=True)
scheduler.add_job(crypto_cycle, "cron", hour="0,4,8,12,16,20", minute=5,
                  timezone="UTC", id="crypto_cycle",
                  replace_existing=True)
scheduler.add_job(daily_summary, "cron", hour=18, minute=0,
                  timezone="America/New_York", id="daily_summary",
                  replace_existing=True)

scheduler.start()
# Block main thread
signal.pause()  # or Event().wait()
```

### Pattern 2: Pre-Cycle Halt Check
**What:** Every job checks the emergency_flags table before executing.
**When to use:** First line of every scheduled job function.
**Example:**
```python
def equity_cycle() -> None:
    """Scheduled equity trading cycle."""
    if is_halted():
        log.warning("cycle_skipped_halt_active", env="equity")
        return

    try:
        fills = pipeline.execute_cycle("equity")
        # Post-cycle callbacks
        send_trade_alerts(fills, "equity")
        scan_wash_sales(fills)
        ping_healthcheck(config.healthchecks.equity_check_url)
    except Exception:
        log.exception("equity_cycle_failed")
        alerter.send_alert("critical", "Equity Cycle Failed", str(e))
        # No HC ping on failure -- dead man's switch triggers
```

### Pattern 3: Two-Webhook Channel Routing
**What:** Critical/warning alerts go to #swingrl-alerts, daily/info to #swingrl-daily.
**When to use:** Extend Alerter to accept two webhook URLs or add routing logic.
**Example:**
```python
# Config extension
class AlertingConfig(BaseModel):
    alert_cooldown_minutes: int = Field(default=30, ge=1)
    consecutive_failures_before_alert: int = Field(default=3, ge=1)
    alerts_webhook_url: str = Field(default="")  # #swingrl-alerts
    daily_webhook_url: str = Field(default="")    # #swingrl-daily
    healthchecks_equity_url: str = Field(default="")
    healthchecks_crypto_url: str = Field(default="")
```

### Pattern 4: Streamlit Multi-Page with Auto-Refresh
**What:** Separate Docker container reads from DB bind mounts (read-only).
**When to use:** Dashboard service in docker-compose.
**Example:**
```python
# dashboard/app.py
import streamlit as st
from streamlit_autorefresh import st_autorefresh

st.set_page_config(page_title="SwingRL Dashboard", layout="wide")

# Auto-refresh every 5 minutes (300000 ms)
st_autorefresh(interval=300_000, key="dashboard_refresh")

st.title("SwingRL Dashboard")
```

### Anti-Patterns to Avoid
- **Separate scheduled jobs for post-cycle callbacks**: HC pings and wash sale scans are callbacks within the cycle job function, not separate cron jobs. This ensures they only fire when the cycle completes.
- **APScheduler scheduler.shutdown() for emergency stop**: The halt flag pattern is simpler and survives container restart. APScheduler keeps running but jobs are no-ops.
- **Retry queue for failed cycles**: Failed cycles are logged and skipped. Next scheduled cycle starts fresh. No retry queue or immediate retry.
- **Static HTML status page**: Streamlit serves both quick-glance and analytical needs. Skip Doc 09 Layer 2.
- **Single Discord webhook**: Split by severity to allow muting daily without missing critical alerts.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Cron scheduling | Custom time-checking loop | APScheduler CronTrigger | DST handling, misfire grace, crash resilience via jobstore |
| Job persistence | Manual job state serialization | SQLAlchemy jobstore | Automatic persistence and recovery across restarts |
| Dashboard auto-refresh | time.sleep + st.rerun loop | streamlit-autorefresh | Frontend timer avoids blocking server resources |
| Interactive financial charts | matplotlib static images | plotly | Zoom, pan, hover tooltips for equity curves and drawdown |
| Discord embed building | Raw JSON construction | Typed embed builder functions | Consistent formatting, less error-prone, DRY |
| Timezone-aware scheduling | Manual timezone math | APScheduler timezone parameter + pytz | DST transitions handled correctly |

**Key insight:** APScheduler's misfire_grace_time and coalesce settings handle edge cases (container restart during scheduled time, DST transition skipping a trigger) that a hand-rolled scheduler would miss.

## Common Pitfalls

### Pitfall 1: APScheduler Duplicate Job Registration
**What goes wrong:** If `add_job()` is called on every startup without `replace_existing=True`, APScheduler raises `ConflictingIdError` because the SQLAlchemy jobstore already has the job from the previous run.
**Why it happens:** Persistent jobstore remembers jobs from prior process lifecycle.
**How to avoid:** Always use `id="job_name", replace_existing=True` on every `add_job()` call.
**Warning signs:** `ConflictingIdError` on container restart.

### Pitfall 2: DST and Timezone Misalignment
**What goes wrong:** Equity cycle at "4:15 PM ET" might fire at 3:15 PM or 5:15 PM during DST transitions.
**Why it happens:** Using UTC offsets instead of timezone names.
**How to avoid:** Use `timezone="America/New_York"` (not UTC-5 or UTC-4). APScheduler 3.x CronTrigger handles DST correctly with pytz zone objects.
**Warning signs:** Jobs firing at wrong wall-clock time twice a year.

### Pitfall 3: APScheduler SQLAlchemy Jobstore with SQLite Concurrency
**What goes wrong:** APScheduler jobstore SQLite and trading_ops.db SQLite compete for WAL locks if they share the same file.
**Why it happens:** SQLite write locking is per-database.
**How to avoid:** Use a **separate** SQLite file for APScheduler jobstore (e.g., `db/apscheduler_jobs.sqlite`), distinct from `trading_ops.db`.
**Warning signs:** `database is locked` errors.

### Pitfall 4: Streamlit Container Can't Access DB Files
**What goes wrong:** Dashboard container fails to connect to DuckDB/SQLite because paths don't match.
**Why it happens:** Streamlit container has different bind mount paths than the bot container.
**How to avoid:** Use identical bind mount paths in both services (`/app/db`). Open DuckDB with `read_only=True` in the dashboard to prevent write contention.
**Warning signs:** `FileNotFoundError` or DuckDB write lock errors in dashboard.

### Pitfall 5: Healthchecks.io Ping Timeout Blocking Cycle
**What goes wrong:** HC ping HTTP request hangs and delays the next cycle.
**Why it happens:** No timeout on the HTTP GET call.
**How to avoid:** Always use `timeout=10` on HC ping requests. Wrap in try/except -- failed pings should log a warning but never crash the cycle.
**Warning signs:** Cycle duration spikes correlated with HC ping failures.

### Pitfall 6: Stuck Agent False Positives During Market Holidays
**What goes wrong:** Stuck agent alert fires because equity env was 100% cash for 10 calendar days (includes weekends/holidays).
**Why it happens:** Counting calendar days instead of trading days.
**How to avoid:** Count only trading days for equity (use `exchange_calendars` NYSE sessions). Count cycles for crypto (no holidays).
**Warning signs:** False stuck agent alerts on long weekends / holiday weeks.

### Pitfall 7: Emergency Halt Flag Not Checked By Stop-Price Daemon
**What goes wrong:** Emergency stop sets halt flag but the crypto stop-price polling thread keeps executing stops.
**Why it happens:** Daemon thread has its own loop and doesn't check the flag.
**How to avoid:** Per CONTEXT.md: "Thread runs independently, checks emergency flag each iteration."
**Warning signs:** Stop orders executing after emergency halt.

## Code Examples

### Emergency Flags Table Schema
```python
# Source: CONTEXT.md locked decision
conn.execute("""
    CREATE TABLE IF NOT EXISTS emergency_flags (
        flag_name TEXT PRIMARY KEY,
        active INTEGER NOT NULL DEFAULT 0,
        set_at TEXT,
        set_by TEXT,
        reason TEXT
    )
""")
```

### Halt Check Function
```python
# Source: CONTEXT.md pattern
def is_halted(db: DatabaseManager) -> bool:
    """Check if emergency halt flag is active."""
    with db.sqlite() as conn:
        row = conn.execute(
            "SELECT active FROM emergency_flags WHERE flag_name = 'halt'"
        ).fetchone()
        return row is not None and row[0] == 1
```

### Healthchecks.io Ping
```python
# Source: healthchecks.io Pinging API docs
import httpx

def ping_healthcheck(check_url: str) -> None:
    """Ping Healthchecks.io after successful cycle."""
    if not check_url:
        return
    try:
        httpx.get(check_url, timeout=10.0)
        log.info("healthcheck_pinged", url=check_url)
    except Exception:
        log.warning("healthcheck_ping_failed", url=check_url, exc_info=True)
```

### Discord Trade Alert Embed
```python
# Source: CONTEXT.md trade alert spec
def build_trade_embed(fill: FillResult, env: str) -> dict:
    """Build Discord embed for a single trade fill."""
    side_emoji = "BUY" if fill.side == "buy" else "SELL"
    color = 0x00FF00 if fill.side == "buy" else 0xFF4444
    return {
        "title": f"{side_emoji} {fill.symbol}",
        "color": color,
        "fields": [
            {"name": "Side", "value": fill.side.upper(), "inline": True},
            {"name": "Quantity", "value": str(fill.quantity), "inline": True},
            {"name": "Fill Price", "value": f"${fill.fill_price:.2f}", "inline": True},
            {"name": "Notional", "value": f"${abs(fill.quantity * fill.fill_price):.2f}", "inline": True},
        ],
        "footer": {"text": f"SwingRL | {env.title()} | TRADE"},
        "timestamp": fill.timestamp,
    }
```

### Wash Sale Scanner
```python
# Source: CONTEXT.md wash sale spec
def scan_wash_sales(fills: list[FillResult], db: DatabaseManager) -> list[dict]:
    """Check new equity buys against 30-day wash sale windows."""
    warnings = []
    buy_symbols = [f.symbol for f in fills if f.side == "buy"]
    if not buy_symbols:
        return warnings

    with db.sqlite() as conn:
        for symbol in buy_symbols:
            row = conn.execute(
                "SELECT wash_window_end FROM wash_sale_tracker "
                "WHERE symbol = ? AND wash_window_end > datetime('now') "
                "AND triggered = 0 ORDER BY sale_date DESC LIMIT 1",
                [symbol],
            ).fetchone()
            if row:
                conn.execute(
                    "UPDATE wash_sale_tracker SET triggered = 1 "
                    "WHERE symbol = ? AND wash_window_end = ?",
                    [symbol, row[0]],
                )
                warnings.append({"symbol": symbol, "window_end": row[0]})
    return warnings
```

### Stuck Agent Detection
```python
# Source: CONTEXT.md stuck agent spec
def check_stuck_agents(db: DatabaseManager, config: SwingRLConfig) -> list[dict]:
    """Detect environments that are stuck in 100% cash."""
    alerts = []
    # Equity: 10 consecutive trading days all-cash
    # Crypto: 30 consecutive cycles all-cash
    for env, threshold in [("equity", 10), ("crypto", 30)]:
        with db.sqlite() as conn:
            rows = conn.execute(
                "SELECT cash_balance, total_value FROM portfolio_snapshots "
                "WHERE environment = ? ORDER BY timestamp DESC LIMIT ?",
                [env, threshold],
            ).fetchall()
        if len(rows) >= threshold and all(
            abs(r["cash_balance"] - r["total_value"]) < 0.01 for r in rows
        ):
            alerts.append({"environment": env, "consecutive_cash_cycles": len(rows)})
    return alerts
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| APScheduler 4.x alpha | APScheduler 3.11.x stable | 4.x still alpha (Apr 2025) | Stick with 3.x for production stability |
| st.experimental_rerun | streamlit-autorefresh component | 2024 | Frontend timer is cleaner than server-side rerun loops |
| Static HTML status pages | Streamlit multi-page apps | 2023+ | Dynamic, interactive, no custom web server needed |
| pytz for timezones | Still pytz for APScheduler 3.x | APScheduler 3.x requirement | APScheduler 4.x will switch to zoneinfo, but we use 3.x |

**Deprecated/outdated:**
- `st.experimental_rerun` has been replaced by `st.rerun` in Streamlit 1.27+
- APScheduler 4.0 alpha should NOT be used in production

## Open Questions

1. **Streamlit Docker image size**
   - What we know: Streamlit + plotly + pandas adds significant deps to the dashboard container
   - What's unclear: Whether to use the same base image or a separate slimmer one
   - Recommendation: Use the same `python:3.11-slim` base but only install dashboard deps (no torch, no SB3). Separate Dockerfile.

2. **APScheduler jobstore cleanup**
   - What we know: SQLAlchemy jobstore accumulates job execution history
   - What's unclear: Whether old job records need periodic cleanup
   - Recommendation: APScheduler 3.x jobstore only stores current job state (not history). No cleanup needed.

3. **Stop-price polling daemon thread integration**
   - What we know: Phase 8 CONTEXT says "main.py starts APScheduler AND launches the crypto stop-price polling daemon thread"
   - What's unclear: The stop-price polling code doesn't exist yet in the codebase (no dedicated module found)
   - Recommendation: Build the stop-price polling daemon as part of this phase. Simple thread that polls Binance.US REST API every 60s for open positions' current price vs stop/TP levels.

## Validation Architecture

### Test Framework
| Property | Value |
|----------|-------|
| Framework | pytest 8.x |
| Config file | pyproject.toml [tool.pytest.ini_options] |
| Quick run command | `uv run pytest tests/ -x -q` |
| Full suite command | `uv run pytest tests/ -v` |

### Phase Requirements -> Test Map
| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| PAPER-12 | APScheduler registers jobs with correct cron triggers; pre-cycle halt check skips when flag active | unit | `uv run pytest tests/scheduler/test_jobs.py -x` | No -- Wave 0 |
| PAPER-12 | Emergency flags table CRUD (set, check, clear) | unit | `uv run pytest tests/scheduler/test_halt_check.py -x` | No -- Wave 0 |
| PAPER-13 | Trade embed builder produces valid Discord payload | unit | `uv run pytest tests/monitoring/test_embeds.py -x` | No -- Wave 0 |
| PAPER-13 | Daily summary embed includes both env P&L and combined | unit | `uv run pytest tests/monitoring/test_embeds.py::test_daily_summary_embed -x` | No -- Wave 0 |
| PAPER-13 | Two-webhook routing sends critical to alerts URL, info to daily URL | unit | `uv run pytest tests/monitoring/test_alerter.py::test_channel_routing -x` | No -- Wave 0 |
| PAPER-14 | Stuck agent detection returns alert for 10+ consecutive cash equity snapshots | unit | `uv run pytest tests/monitoring/test_stuck_agent.py -x` | No -- Wave 0 |
| PAPER-14 | Stuck agent detection returns empty for mixed cash/positions | unit | `uv run pytest tests/monitoring/test_stuck_agent.py::test_no_false_positive -x` | No -- Wave 0 |
| PAPER-15 | Dashboard pages render without error (Streamlit AppTest) | unit | `uv run pytest tests/dashboard/test_pages.py -x` | No -- Wave 0 |
| PAPER-16 | HC ping sends GET to configured URL on cycle success | unit | `uv run pytest tests/monitoring/test_healthcheck_ping.py -x` | No -- Wave 0 |
| PAPER-16 | HC ping does not fire when cycle raises exception | unit | `uv run pytest tests/monitoring/test_healthcheck_ping.py::test_no_ping_on_failure -x` | No -- Wave 0 |
| PAPER-17 | Wash sale scanner flags buy within 30-day window | unit | `uv run pytest tests/monitoring/test_wash_sale.py -x` | No -- Wave 0 |
| PAPER-17 | Wash sale scanner ignores crypto fills | unit | `uv run pytest tests/monitoring/test_wash_sale.py::test_crypto_exempt -x` | No -- Wave 0 |

### Sampling Rate
- **Per task commit:** `uv run pytest tests/ -x -q`
- **Per wave merge:** `uv run pytest tests/ -v`
- **Phase gate:** Full suite green before `/gsd:verify-work`

### Wave 0 Gaps
- [ ] `tests/scheduler/test_jobs.py` -- covers PAPER-12 job registration and halt checks
- [ ] `tests/scheduler/test_halt_check.py` -- covers emergency flag CRUD
- [ ] `tests/monitoring/test_embeds.py` -- covers PAPER-13 embed builders
- [ ] `tests/monitoring/test_stuck_agent.py` -- covers PAPER-14
- [ ] `tests/monitoring/test_healthcheck_ping.py` -- covers PAPER-16
- [ ] `tests/monitoring/test_wash_sale.py` -- covers PAPER-17
- [ ] `tests/dashboard/test_pages.py` -- covers PAPER-15 (Streamlit AppTest)
- [ ] `tests/scheduler/__init__.py` -- package init
- [ ] `tests/dashboard/__init__.py` -- package init

## Sources

### Primary (HIGH confidence)
- [APScheduler 3.x User Guide](https://apscheduler.readthedocs.io/en/3.x/userguide.html) -- BackgroundScheduler, jobstores, executors, cron triggers
- [APScheduler 3.x SQLAlchemy JobStore docs](https://apscheduler.readthedocs.io/en/3.x/modules/jobstores/sqlalchemy.html) -- SQLAlchemy jobstore configuration
- [APScheduler 3.x CronTrigger docs](https://apscheduler.readthedocs.io/en/latest/modules/triggers/cron.html) -- Cron trigger fields and timezone support
- [Healthchecks.io Pinging API](https://healthchecks.io/docs/http_api/) -- Ping endpoints, response codes, start/fail signals
- [streamlit-autorefresh GitHub](https://github.com/kmcgrady/streamlit-autorefresh) -- Auto-refresh component usage
- [APScheduler PyPI](https://pypi.org/project/APScheduler/) -- Version 3.11.2 latest stable (Dec 2025)

### Secondary (MEDIUM confidence)
- [Streamlit 2025 Release Notes](https://docs.streamlit.io/develop/quick-reference/release-notes/2025) -- st.context, multi-page improvements
- [Streamlit Auto Refresh Forum Discussion](https://discuss.streamlit.io/t/auto-refresh-streamlit-application-with-many-pages/93242) -- Multi-page auto-refresh patterns
- [APScheduler 4.0 Migration Guide](https://apscheduler.readthedocs.io/en/master/migration.html) -- Confirmed v4 still alpha, v3 is production choice

### Tertiary (LOW confidence)
- None

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH -- APScheduler 3.x is well-documented stable release; Streamlit is mature; Healthchecks.io has simple HTTP API
- Architecture: HIGH -- CONTEXT.md decisions are extremely detailed and prescriptive; existing codebase patterns are clear
- Pitfalls: HIGH -- APScheduler SQLite concurrency, DST handling, and jobstore conflicts are well-documented issues

**Research date:** 2026-03-09
**Valid until:** 2026-04-09 (stable libraries, 30-day validity)
