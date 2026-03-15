# Phase 20: Production Deployment - Research

**Researched:** 2026-03-15
**Domain:** Docker production stack, memory live enrichment pipeline, websocket fill ingestion, APScheduler timezone handling, startup validation
**Confidence:** HIGH

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions

- **Startup validation**: Standalone `scripts/startup_checks.py` called by `main.py` before scheduler init, hard fail exit(1) on any validation failure. Checks: broker credentials (Alpaca + Binance.US API ping), DB tables (SQLite + DuckDB), model files (warn, don't block), memory service reachability (warn, fail-open), memory seeding status (warn if empty), bind-mount writability (touch temp file). Structlog output only.
- **Deployment method**: `scripts/deploy.sh` — git pull → docker compose build → down → up -d → wait for healthy → run `scripts/setup_ollama.sh`. .env is manual and never touched by deploy.sh. No rollback mechanism (git revert + rebuild is sufficient). Smoke test is separate.
- **Memory live enrichment included in Phase 20** — not deferred. All 5 endpoints enabled immediately (seeded memories provide warm start). No obs space expansion.
- **Memory agent controls pipeline from outside**: hooks inside `ExecutionPipeline.execute_cycle()`. Per-endpoint config toggles via `memory_agent.live_endpoints.*`.
- **All live endpoints use qwen2.5:3b** (fast model, latency-sensitive).
- **Blend weight bounds**: each algo weight clamped to ±0.20 from Sharpe-softmax baseline; configurable via `memory_agent.live_bounds.blend_weight_delta=0.20`. Weights reset each cycle.
- **CircuitBreaker override pattern**: save + restore. `apply_overrides()` saves to `_saved_*` fields; `reset_overrides()` restores. Always log original and override thresholds.
- **Memory seeding workflow**: operator runs `seed_memory_from_backtest.py` manually once after training; add POST /consolidate at end. Seed only once.
- **Alpaca FillIngestor**: full websocket thread using TradingStream (`trade_updates` stream). Daemon thread. Mock TradingStream in unit tests.
- **Binance FillIngestor**: python-binance (BinanceSocketManager, `tld='us'`). Stub for paper mode (connects in live, no-op in paper). Auto-reconnect with exponential backoff (1→2→4→8→16, max 60s). Daemon thread.
- **MacroWatcher**: researcher finds best free economic calendar API; fallback is hardcoded FOMC/CPI/NFP schedule. Poll interval configurable via YAML.
- **`scripts/show_schedule.py`**: CLI that prints all 12 jobs with next fire times in both ET and UTC. Reconstructs from config (no running scheduler required).
- **`scripts/smoke_test.py`**: triggers one equity + one crypto cycle via docker exec. Real paper mode. Full pipeline with memory. Pass/fail checklist to stdout. Exit 0 all pass, exit 1 any fail.
- **Docker resource limits**: swingrl-ollama 24GB/8CPU (unchanged), swingrl-memory 1GB/1CPU (up from 512MB/0.5), swingrl 8GB/4CPU (up from 2.5GB/1), swingrl-dashboard 512MB/0.5CPU (unchanged).
- **New dependencies**: SQLAlchemy>=2.0,<3 (APScheduler jobstore), requests (undeclared existing dep), python-binance (Binance websocket). Full production Docker build-and-trace required.
- **Testing**: mock Ollama in all unit tests. Explicit fail-open tests. Mock both websocket streams. Test startup_checks.py with mocked APIs. No deploy.sh tests.
- **Healthcheck improvements**: add memory service (HTTP to swingrl-memory:8889/health) + Ollama reachability (HTTP to swingrl-ollama:11434/api/tags) checks. UNHEALTHY with degraded flag on fail (trading continues fail-open).
- **Error recovery**: track consecutive failures per env in SQLite; after 3 consecutive failures fire Discord warning + structlog error.
- **Status file**: write `status/status.json` after each cycle (last_cycle_time, positions, memory_health, scheduler_state, next_fire_times).
- **Logging**: structured log with rationale for each memory call. DuckDB table `memory_live_decisions` (timestamp, env, endpoint, decision_json, rationale, latency_ms, clamped). Memory latency alert if avg > 3s for 3 consecutive calls.
- **CI updates**: test both containers; Docker compose build only (no start).
- **Config schema additions**: `memory_agent.live_endpoints.*`, `memory_agent.live_bounds.*`, `memory_agent.macro_watcher.*`.

### Claude's Discretion

- Exact system prompts for each live endpoint (cycle_gate, blend_weights, risk_thresholds, position_advice, trade_veto)
- MacroWatcher config field structure
- Whether Alpaca streaming needs separate auth vs same keys
- Background ingest queue implementation details (queue depth, batch size)
- Exact status.json schema
- How to handle edge cases in client_order_id parsing for signal confidence threading

### Deferred Ideas (OUT OF SCOPE)

- Obs space expansion (memory dims 156→164, 45→53) — requires retraining, defer to Phase 22 or later
- Memory dashboard tab in Streamlit — DuckDB table ready for it
- Progressive ramp-up of live endpoints — decided against (all enabled immediately)
- Inbox file watcher for memory ingestion — deferred from Phase 19
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|-----------------|
| DEPLOY-01 | Docker production stack starts on homelab with both services healthy (swingrl + dashboard) | Resource limit updates, deploy.sh pattern, healthcheck extensions, memory/Ollama checks |
| DEPLOY-02 | Homelab .env configured with all required API keys and webhook URLs — startup validation confirms | startup_checks.py design, env key validation, .env.example update |
| DEPLOY-03 | Startup validation runs on container boot: DB tables, active models, broker creds, Discord ping | Full validation checklist, exit(1) on hard failures, warn on soft failures |
| DEPLOY-04 | APScheduler fires equity cycle (16:15 ET) and crypto cycle (every 4H) correctly with timezone handling | Existing scheduler code confirmed correct, show_schedule.py for verification, TZ double-conversion pitfall |
| DEPLOY-05 | Paper trading executes end-to-end: signal → size → validate → submit → fill for both environments | smoke_test.py design, memory pipeline hooks, FillIngestor threads, full cycle verification |
</phase_requirements>

---

## Summary

Phase 20 deploys the production Docker stack on homelab while simultaneously activating the memory live enrichment pipeline that was built in Phase 19.1. The two workstreams are tightly coupled: the memory agent must be running and seeded before the first live trading cycle fires.

The existing codebase is in very good shape for deployment. The `docker-compose.prod.yml` defines all four services, `scripts/main.py` already wires APScheduler with correct ET/UTC job registrations, and `ExecutionPipeline.execute_cycle()` is the single integration point for all memory hooks. The primary work is: (1) adding the `/live/` router to the memory service with 5 new endpoints, (2) hooking those endpoints into `execute_cycle()`, (3) adding FillIngestor daemon threads for Alpaca and Binance, (4) adding startup validation and operational scripts, (5) fixing missing production dependencies, and (6) updating Docker resource limits.

The memory agent pattern is well-established from the QueryAgent in Phase 19.1: Ollama structured JSON output with qwen2.5:3b (fast model), XML-wrapped context, clamped responses with safe defaults, fail-open on any exception. The live endpoints follow the same pattern but are synchronous (stdlib urllib) rather than async (httpx) to match the existing MemoryClient.

**Primary recommendation:** Implement memory live endpoints as a new `/live/` FastAPI router in `services/memory/`, hook into `ExecutionPipeline.execute_cycle()` with minimal disruption to existing logic, add FillIngestor daemon threads, and run `seed_memory_from_backtest.py` before first deploy. The APScheduler timezone configuration is already correct — verification script (`show_schedule.py`) is needed to confirm, not fix.

---

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| APScheduler | >=3.10,<4 | Cron + interval job scheduling | Already in use; pin <4 (breaking API change in 4.x) |
| SQLAlchemy | >=2.0,<3 | APScheduler SQLAlchemyJobStore backend | Required companion — APScheduler declares sqlalchemy>=2.0.24 as optional dep |
| alpaca-py | >=0.20 | TradingStream websocket for fill ingestion | Already in use for order submission; same auth keys |
| python-binance | >=1.0 | BinanceSocketManager user data stream | Official python client; ThreadedWebsocketManager avoids asyncio complexity |
| stdlib urllib | built-in | Live memory endpoint HTTP calls | Consistent with existing MemoryClient pattern; no new dep |
| FastAPI | >=0.110 | Memory service live router | Already in use in services/memory/ |
| structlog | >=24.0 | Structured logging | Project-wide standard |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| requests | >=2.28 | HTTP calls in 4 existing modules | Add as explicit dep — currently undeclared but used |
| finnhub-python | >=2.4.27 | Economic calendar API | Already declared in pyproject.toml; free tier 60 req/min |
| threading | built-in | Daemon threads for FillIngestors | Background threads exit with main process |
| queue | built-in | Background post-trade ingest queue | Zero-latency async ingest without new dep |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| ThreadedWebsocketManager (Binance) | BinanceSocketManager (async) | ThreadedWebsocketManager is simpler — no asyncio context needed in daemon thread |
| Finnhub economic calendar | Trading Economics API / hardcoded | Finnhub already in pyproject.toml, free tier sufficient; hardcoded fallback is more reliable for fixed events (FOMC 8x/year) |
| stdlib urllib for live endpoints | httpx (already in project) | Consistency with MemoryClient; httpx would work equally well |

**Installation (new additions only):**
```bash
# Add to pyproject.toml [project.dependencies]:
"sqlalchemy>=2.0,<3",
"requests>=2.28",
"python-binance>=1.0",
```

---

## Architecture Patterns

### Recommended Project Structure (new files)

```
scripts/
├── startup_checks.py      # Pre-scheduler validation; exit(1) on hard fail
├── deploy.sh              # git pull → build → down → up -d → setup_ollama.sh
├── show_schedule.py       # Print all 12 jobs with ET + UTC next fire times
├── smoke_test.py          # E2E cycle trigger; pass/fail checklist; exit 0/1
└── seed_memory_from_backtest.py  # One-time memory seeding (operator runs manually)

src/swingrl/
├── execution/
│   ├── pipeline.py        # Add memory hook calls to execute_cycle()
│   └── risk/
│       └── circuit_breaker.py  # Add apply_overrides() / reset_overrides()
├── memory/
│   ├── client.py          # Existing MemoryClient — no change needed
│   └── live/
│       ├── __init__.py
│       ├── client.py      # LiveMemoryClient (sync urllib, fail-open, 5 endpoints)
│       └── fill_ingestor.py  # AlpacaFillIngestor + BinanceFillIngestor daemon threads
└── scheduler/
    └── macro_watcher.py   # MacroWatcher polling thread

services/memory/
├── routers/
│   └── live.py            # New /live/ router (5 endpoints)
└── memory_agents/
    └── live_agent.py      # LiveAgent: qwen2.5:3b, per-endpoint prompts

config/swingrl.yaml        # Add live_endpoints, live_bounds, macro_watcher sections
```

### Pattern 1: Memory Live Endpoint (fail-open sync HTTP)

**What:** Each live endpoint follows the same pattern as MemoryClient — stdlib urllib POST, try/except, return safe default on any failure, log with structlog.
**When to use:** All 5 live endpoints (cycle_gate, blend_weights, risk_thresholds, position_advice, trade_veto).

```python
# Source: existing src/swingrl/memory/client.py pattern
def query_cycle_gate(self, env: str, context: dict[str, Any]) -> bool:
    """Check if memory agent approves this cycle. Fail-open: returns True on error."""
    import json, urllib.request
    url = f"{self._base_url}/live/cycle_gate"
    try:
        data = json.dumps({"env": env, "context": context}).encode("utf-8")
        headers = {"Content-Type": "application/json", "X-API-Key": self._api_key}
        req = urllib.request.Request(url, data=data, headers=headers, method="POST")
        with urllib.request.urlopen(req, timeout=self._timeout) as resp:  # nosec B310
            body = json.loads(resp.read())
            return bool(body.get("approved", True))
    except Exception as exc:
        log.warning("cycle_gate_failed_open", env=env, error=str(exc))
        return True  # fail-open: allow cycle to proceed
```

### Pattern 2: CircuitBreaker Override (save + restore)

**What:** Memory risk thresholds are applied by temporarily overriding CB thresholds, then restoring originals at cycle start.
**When to use:** `risk_thresholds` endpoint integration in `execute_cycle()`.

```python
# In CircuitBreaker — new methods:
def apply_overrides(self, max_dd: float | None, daily_limit: float | None) -> None:
    """Save current thresholds and apply memory-advised overrides (tighten only)."""
    self._saved_max_dd = self._max_dd
    self._saved_daily_limit = self._daily_limit
    if max_dd is not None:
        self._max_dd = min(self._max_dd, max_dd)  # tighten only
    if daily_limit is not None:
        self._daily_limit = min(self._daily_limit, daily_limit)  # tighten only
    log.info("cb_overrides_applied",
             original_max_dd=self._saved_max_dd, new_max_dd=self._max_dd,
             original_daily_limit=self._saved_daily_limit, new_daily_limit=self._daily_limit)

def reset_overrides(self) -> None:
    """Restore original thresholds (call at cycle start)."""
    if hasattr(self, "_saved_max_dd"):
        self._max_dd = self._saved_max_dd
        self._daily_limit = self._saved_daily_limit
        log.debug("cb_overrides_reset", max_dd=self._max_dd, daily_limit=self._daily_limit)
```

### Pattern 3: Blend Weight Override (±0.20 clamp from Sharpe baseline)

**What:** Memory advises per-algo weight adjustments; clamped to ±blend_weight_delta from Sharpe-softmax baseline.
**When to use:** Between `_get_ensemble_weights()` and `blender.blend_actions()` in `execute_cycle()`.

```python
# In execute_cycle(), after weights = self._get_ensemble_weights(env_name):
if (self._config.memory_agent.enabled
        and self._config.memory_agent.live_endpoints.blend_weights):
    memory_weights = self._live_client.query_blend_weights(env_name, weights)
    delta = self._config.memory_agent.live_bounds.blend_weight_delta
    for algo in weights:
        baseline = weights[algo]
        advised = memory_weights.get(algo, baseline)
        weights[algo] = max(baseline - delta, min(baseline + delta, advised))
    # Re-normalize weights to sum=1.0
    total = sum(weights.values())
    if total > 0:
        weights = {k: v / total for k, v in weights.items()}
```

### Pattern 4: Alpaca FillIngestor (TradingStream daemon thread)

**What:** Daemon thread subscribing to Alpaca's `trade_updates` stream. Converts fill events to memory ingest payloads via background queue.
**When to use:** Started in `build_app()` alongside stop_polling_thread.

```python
# Alpaca TradingStream approach (alpaca-py):
from alpaca.trading.stream import TradingStream

class AlpacaFillIngestor:
    def __init__(self, api_key: str, secret_key: str, paper: bool,
                 memory_client: MemoryClient) -> None:
        self._stream = TradingStream(api_key=api_key, secret_key=secret_key, paper=paper)
        self._memory_client = memory_client
        self._stream.subscribe_trade_updates(self._handle_trade_update)

    async def _handle_trade_update(self, data: Any) -> None:
        """Async handler called by TradingStream on each event."""
        if data.event in ("fill", "partial_fill"):
            # Enqueue to background queue — do not block stream loop
            self._queue.put_nowait(data)

    def start(self) -> None:
        """Start as daemon thread (exits with main process)."""
        t = threading.Thread(target=self._stream.run, daemon=True)
        t.start()
```

### Pattern 5: Binance FillIngestor (ThreadedWebsocketManager, stub in paper)

**What:** Daemon thread using python-binance ThreadedWebsocketManager for user data stream. No-op in paper mode. Auto-reconnect with exponential backoff.
**When to use:** Started in `build_app()` only when `trading_mode != "paper"`.

```python
from binance import ThreadedWebsocketManager  # python-binance

class BinanceFillIngestor:
    def __init__(self, api_key: str, secret_key: str, memory_client: MemoryClient) -> None:
        self._bm = ThreadedWebsocketManager(api_key=api_key, api_secret=secret_key, tld="us")
        self._memory_client = memory_client
        self._backoff = [1, 2, 4, 8, 16, 60]  # seconds, max 60

    def start(self) -> None:
        """Start manager and user data stream as daemon thread."""
        self._bm.start()
        self._bm.start_user_socket(callback=self._handle_message)

    def _handle_message(self, msg: dict[str, Any]) -> None:
        """Handle user data stream events — filter for order fills."""
        if msg.get("e") == "executionReport" and msg.get("X") in ("FILLED", "PARTIALLY_FILLED"):
            # Post to background ingest queue
            self._queue.put_nowait(msg)
```

### Pattern 6: execute_cycle() Memory Hooks (hook order)

**What:** All 5 memory hooks integrate into `execute_cycle()` at defined points, each gated by its config toggle.
**When to use:** Integration into `ExecutionPipeline.execute_cycle()`.

Hook integration order in `execute_cycle()`:
1. **At start**: `reset_overrides()` on all circuit breakers
2. **Step 1.5** (after CB state check): `query_cycle_gate(env_name)` — if False, log and return []
3. **Step 3.5** (after risk_thresholds endpoint): `apply_overrides()` on circuit breakers
4. **Step 7.5** (after `_get_ensemble_weights()`): `query_blend_weights()` → clamp and override weights
5. **Step 9.5** (after signals generated, before processing loop): `query_position_advice()` — advisory only, logged
6. **Step 10.1** (per signal, before broker submission): `query_trade_veto(signal)` — if vetoed, skip signal
7. **After cycle complete**: enqueue post-trade ingest to background queue (non-blocking)

### Pattern 7: startup_checks.py (fail-hard / warn pattern)

**What:** Standalone script called before scheduler init. Hard fail (exit 1) for critical checks, warn for advisory checks.
**When to use:** Called from `build_app()` before `BackgroundScheduler` creation.

```python
# scripts/startup_checks.py
HARD_FAIL_CHECKS = [
    check_bind_mount_writability,   # touch temp file in data/, db/, models/, logs/, status/
    check_broker_credentials,       # ping Alpaca + Binance.US APIs
    check_db_tables,                # verify SQLite + DuckDB tables exist, create if absent
    check_env_keys,                 # all required .env keys present and non-empty
]
WARN_CHECKS = [
    check_model_files,              # models/active/{equity,crypto}/{ppo,a2c,sac}/model.zip
    check_memory_service,           # GET swingrl-memory:8889/health
    check_memory_seeding,           # query memory.db row count, warn if empty
]
```

### Anti-Patterns to Avoid

- **Double TZ conversion**: APScheduler jobs registered with `timezone="America/New_York"` — container also sets `TZ=America/New_York`. APScheduler uses its own `timezone` param independently of the system TZ. Do NOT apply any additional offset correction. Verify with `show_schedule.py` before first live cycle.
- **Blocking memory calls in trade loop**: All live endpoint calls must be wrapped with a tight timeout (3s from `memory_agent.timeout_sec`). Never allow Ollama latency to delay broker submission.
- **Raising exceptions from FillIngestor callbacks**: Websocket callback exceptions kill the stream. Always `try/except Exception` everything inside handlers.
- **Importing from src/swingrl/ in services/memory/**: Cross-container boundary. Live router follows the same pattern as existing routers — all config via env vars, duplicate any needed constants inline.
- **Using httpx in live endpoint calls from swingrl container**: The existing `MemoryClient` uses stdlib urllib. `LiveMemoryClient` must also use stdlib urllib for consistency (fail-open, no async context required).

---

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Scheduler persistence across restarts | Custom SQLite job store | `SQLAlchemyJobStore` (already in use) | APScheduler jobstore handles persistence natively |
| Alpaca fill stream | Manual WebSocket with raw websockets lib | `alpaca-py TradingStream` | Official client handles auth, reconnect, paper/live routing |
| Binance fill stream reconnect | Custom reconnect loop | `ThreadedWebsocketManager` built-in reconnect | Manager handles 5 retries with exponential backoff |
| Timezone-aware scheduling | Manual UTC offset math | APScheduler `timezone` param | APScheduler handles DST transitions automatically |
| Blend weight normalization | Ad-hoc re-normalization | Standard sum/divide pattern | Simple but must always happen after clamping |
| Memory latency tracking | In-memory counters | DuckDB `memory_live_decisions` table | Already decided — enables post-hoc analysis |

**Key insight:** The memory agent pattern (fail-open HTTP, structured JSON, clamped defaults) is already established in Phase 19.1. Copy the pattern exactly — do not design a new one.

---

## Common Pitfalls

### Pitfall 1: APScheduler + TZ=America/New_York Double Conversion
**What goes wrong:** Container sets `TZ=America/New_York`. Developer sees ET times in logs and assumes APScheduler is also reading system TZ. They do not set `timezone=` param on jobs. Job fires at wrong time OR fires correctly but developer thinks UTC offset is wrong and "fixes" it — introducing the actual bug.
**Why it happens:** APScheduler uses its own `timezone` parameter independent of the system TZ env var.
**How to avoid:** Keep `timezone="America/New_York"` on all ET jobs (already done in existing code). Crypto jobs use `timezone="UTC"` (already done). Run `show_schedule.py` and verify next fire times before first live cycle.
**Warning signs:** `show_schedule.py` shows next ET fire at unexpected UTC offset.

### Pitfall 2: Memory Endpoint Latency Blocking Trade Submission
**What goes wrong:** Ollama (qwen2.5:3b) occasionally takes 5-10s under load. If called synchronously without timeout, the trade submission for a time-sensitive signal is delayed past market close.
**Why it happens:** LLM inference time is variable; no timeout means unbounded wait.
**How to avoid:** All `LiveMemoryClient` calls must use `timeout=self._config.memory_agent.timeout_sec` (default 3s). On timeout, return safe default immediately.
**Warning signs:** `memory_latency_alert` structlog warning + consecutive_slow_calls counter increments.

### Pitfall 3: python-binance Stub in Paper Mode
**What goes wrong:** `BinanceFillIngestor.start()` is called regardless of trading_mode. In paper mode it tries to authenticate with Binance.US for user data stream, which fails (no live account or different keys).
**Why it happens:** Binance paper mode (sim adapter) doesn't have a real Binance account to stream from.
**How to avoid:** Decision says "stub for paper — connects in live mode, no-op in paper". In `build_app()`, only call `BinanceFillIngestor.start()` when `config.trading_mode == "live"`. The class still exists but `start()` is not called.
**Warning signs:** BrokerError on startup in paper mode related to Binance authentication.

### Pitfall 4: .env Missing DISCORD_WEBHOOK_URL for DAILY_WEBHOOK_URL
**What goes wrong:** `startup_checks.py` validates `.env` keys. The `.env.example` shows `DISCORD_WEBHOOK_URL` but `AlertingConfig` has two separate fields: `alerts_webhook_url` and `daily_webhook_url`. The .env needs both mapped.
**Why it happens:** The YAML field names differ from the env var names used in production.
**How to avoid:** `.env.example` must include `SWINGRL_ALERTING__ALERTS_WEBHOOK_URL` and `SWINGRL_ALERTING__DAILY_WEBHOOK_URL` (or the single `DISCORD_WEBHOOK_URL` mapped to both in `build_app()`). Verify during startup_checks.
**Warning signs:** Daily summary never fires; alerts fire but daily summary webhook returns 404.

### Pitfall 5: Bind Mount Permissions for Non-Root Trader User
**What goes wrong:** Host creates `./status/` as root. Container trader user (UID 1000) cannot write `status/status.json`. Container healthcheck passes but status file write fails silently.
**Why it happens:** Docker bind mounts inherit host file permissions. `./status:/app/status` requires the trader user (UID 1000) to own the directory.
**How to avoid:** `startup_checks.py` checks writability by touching a temp file in each bind mount. `deploy.sh` pre-creates directories with correct permissions: `mkdir -p data db models logs status && chown 1000:1000 data db models logs status`.
**Warning signs:** `bind_mount_writability_failed` in startup logs; status.json never appears.

### Pitfall 6: Memory DB Empty After Seeding Returns Warning (Not Error)
**What goes wrong:** `startup_checks.py` warns but continues when memory.db is empty. Operator assumes empty memory is OK and starts live trading without seeding. Memory agent falls back to safe defaults for every cycle — technically works but memory is useless.
**Why it happens:** Empty memory is a warn (not hard fail) by design — trading can proceed but memory is not contributing.
**How to avoid:** Make the warning prominent (log.warning with clear message "MEMORY NOT SEEDED — run seed_memory_from_backtest.py before first live cycle"). Document in deploy.sh output. Startup checks output should be reviewed by operator.
**Warning signs:** Every memory endpoint logs `rationale: "cold_start_defaults"`.

### Pitfall 7: SQLAlchemy Not in Production Dependencies
**What goes wrong:** `pyproject.toml` has `apscheduler>=3.10,<4` in dev dependencies but `SQLAlchemy` is missing entirely. Production Docker image (`uv sync --no-dev`) does not include APScheduler or SQLAlchemy. `scripts/main.py` fails with `ImportError: No module named 'apscheduler'` in production.
**Why it happens:** APScheduler was added to `dev` group only (for tests). Production needs it too.
**How to avoid:** Move `apscheduler>=3.10,<4` from `[dependency-groups.dev]` to `[project.dependencies]`. Add `sqlalchemy>=2.0,<3` and `python-binance>=1.0` and `requests>=2.28` to `[project.dependencies]`. Run production Docker build after to verify no ImportErrors.
**Warning signs:** Container exits immediately after start with ImportError in Docker logs.

---

## Code Examples

### startup_checks.py — Broker Credential Ping Pattern

```python
# scripts/startup_checks.py
from __future__ import annotations

import os
import sys
import urllib.error
import urllib.request

import structlog

log = structlog.get_logger(__name__)


def check_alpaca_credentials() -> None:
    """Ping Alpaca paper API to confirm auth. Hard fail on error."""
    api_key = os.environ.get("ALPACA_API_KEY", "")
    secret_key = os.environ.get("ALPACA_SECRET_KEY", "")
    if not api_key or not secret_key:
        log.error("startup_check_failed", check="alpaca_credentials",
                  reason="env vars missing")
        sys.exit(1)
    try:
        from alpaca.trading.client import TradingClient
        client = TradingClient(api_key=api_key, secret_key=secret_key, paper=True)
        client.get_account()
        log.info("startup_check_passed", check="alpaca_credentials")
    except Exception as exc:
        log.error("startup_check_failed", check="alpaca_credentials", error=str(exc))
        sys.exit(1)
```

### show_schedule.py — Next Fire Time Display

```python
# scripts/show_schedule.py
from __future__ import annotations

from datetime import timezone
from zoneinfo import ZoneInfo

from scripts.main import create_scheduler_and_register_jobs
from swingrl.config.schema import load_config

ET = ZoneInfo("America/New_York")
UTC = timezone.utc

def main() -> None:
    config = load_config("config/swingrl.yaml")
    # Import APScheduler here (not at module level — allow ImportError to surface)
    from apscheduler.schedulers.background import BackgroundScheduler
    scheduler = BackgroundScheduler()
    create_scheduler_and_register_jobs(scheduler, config)
    # Do NOT call scheduler.start() — just read next_run_time from job definitions
    print(f"{'Job ID':<30} {'Next ET':<25} {'Next UTC':<25}")
    print("-" * 80)
    for job in scheduler.get_jobs():
        # next_run_time is None until scheduler starts; use trigger.get_next_fire_time()
        import datetime
        next_utc = job.trigger.get_next_fire_time(None, datetime.datetime.now(UTC))
        if next_utc:
            next_et = next_utc.astimezone(ET)
            print(f"{job.id:<30} {str(next_et):<25} {str(next_utc):<25}")
        else:
            print(f"{job.id:<30} {'N/A':<25} {'N/A':<25}")
```

### Live Router Pattern (services/memory/routers/live.py)

```python
# services/memory/routers/live.py
from __future__ import annotations

from fastapi import APIRouter, Depends
from pydantic import BaseModel
from auth import verify_api_key
from memory_agents.live_agent import LiveAgent

router = APIRouter()
_AGENT = LiveAgent()  # qwen2.5:3b, dedicated system prompts per endpoint


class CycleGateRequest(BaseModel):
    env: str
    context: dict  # HMM regime, volatility, recent P&L

class CycleGateResponse(BaseModel):
    approved: bool
    rationale: str
    latency_ms: float


@router.post("/live/cycle_gate", response_model=CycleGateResponse)
async def cycle_gate(
    body: CycleGateRequest,
    _key: str = Depends(verify_api_key),
) -> CycleGateResponse:
    """Memory agent decides if this trading cycle should proceed."""
    import time
    t0 = time.monotonic()
    result = await _AGENT.query_cycle_gate(body.env, body.context)
    latency_ms = (time.monotonic() - t0) * 1000
    return CycleGateResponse(
        approved=result.get("approved", True),
        rationale=result.get("rationale", "safe_default"),
        latency_ms=latency_ms,
    )
```

### config/swingrl.yaml additions

```yaml
memory_agent:
  enabled: true                     # Set true for production with memory
  live_endpoints:
    cycle_gate: true
    blend_weights: true
    risk_thresholds: true
    position_advice: true
    trade_veto: true
    obs_enrichment: false           # Never enabled — no obs space expansion
  live_bounds:
    blend_weight_delta: 0.20        # Max ±deviation from Sharpe baseline per algo
    max_dd_tighten_min: 0.02        # CB can tighten max_dd by at most 2pp
    daily_limit_tighten_min: 0.005  # CB can tighten daily_limit by at most 0.5pp
  macro_watcher:
    enabled: true
    poll_interval_seconds: 300      # 5 minutes
    lookforward_hours: 24           # Warn if high-impact event within 24h
    source: finnhub                 # "finnhub" | "hardcoded"
    high_impact_only: true          # Filter to high-impact events only
```

---

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| APScheduler <4.0 SQLAlchemy jobstore | Still APScheduler 3.x (v4 has breaking API) | Stable, don't upgrade | APScheduler 3.x is production-stable; v4 rewrites API entirely |
| Binance Global stream | Binance.US ThreadedWebsocketManager with `tld='us'` | Binance.US migration 2019 | Must use `tld='us'` or connections route to wrong endpoint |
| Alpaca v1 trade API stream | alpaca-py `TradingStream` | alpaca-py >=0.20 | Modern SDK; handles paper/live routing via `paper=True` param |
| Manual healthchecks | DB + memory service + Ollama reachability | Phase 20 | Three-layer health: DB connectivity, memory service HTTP, Ollama API |

**Deprecated/outdated:**
- `alpaca-trade-api` (old Python SDK): replaced by `alpaca-py` — project already uses `alpaca-py`
- `apscheduler>=4`: Do NOT use — breaking API; project pins `<4`

---

## Open Questions

1. **MacroWatcher: Finnhub vs hardcoded**
   - What we know: Finnhub free tier allows 60 req/min; has economic calendar endpoint at `/api/v1/calendar/economic`; already declared in `pyproject.toml` as `finnhub-python>=2.4.27`. Filtering by `impact="high"` would cover FOMC, CPI, NFP, PCE, GDP.
   - What's unclear: Whether Finnhub economic calendar includes event impact classification on free tier. Rate limits are generous (60/min vs 300s poll = 1 req per 5 min).
   - Recommendation: Implement Finnhub as primary source with `source: finnhub` config. Hardcoded fallback as backup when Finnhub returns empty or errors. FOMC dates are public (8x/year); CPI/NFP follow monthly patterns. Hardcoded dates should cover 2026 calendar year.

2. **Alpaca TradingStream auth in paper mode**
   - What we know: TradingStream takes same `api_key` + `secret_key` as TradingClient. `paper=True` routes to `wss://paper-api.alpaca.markets/stream`. Same Alpaca account keys work for both paper trading and paper streaming.
   - What's unclear: Nothing — alpaca-py docs confirm `paper=True` uses paper endpoint with same keys.
   - Recommendation: Use same `ALPACA_API_KEY` / `ALPACA_SECRET_KEY` from env. `paper = (config.trading_mode == "paper")`.

3. **Background post-trade ingest queue depth**
   - What we know: Queue must be non-blocking (put_nowait). Post-trade ingest is fire-and-forget.
   - Recommendation: `queue.Queue(maxsize=100)`. If queue is full (trading very active), drop silently with log.warning. Background worker thread drains queue and calls `MemoryClient.ingest()`. Single worker thread is sufficient — 2 fills/day at most in paper mode.

4. **seed_memory_from_backtest.py: what does it seed?**
   - What we know: Phase 19.1 trained with memory enabled. Training summaries, walk-forward results, regime patterns were ingested. The seeding script narrates backtest performance for each env/algo combo.
   - What's unclear: Whether this script was created in Phase 19.1 or needs to be created in Phase 20.
   - Recommendation: Create in Phase 20 as part of the memory workflow. Script reads DuckDB `model_runs` table, formats per-run narratives, calls `MemoryClient.ingest_training()` in batch, then calls `MemoryClient.consolidate()`.

---

## Validation Architecture

### Test Framework
| Property | Value |
|----------|-------|
| Framework | pytest >=8.0 |
| Config file | `pyproject.toml` `[tool.pytest.ini_options]` |
| Quick run command | `uv run pytest tests/ -v -k "test_startup or test_live or test_fill or test_macro" -x` |
| Full suite command | `uv run pytest tests/ -v` |

### Phase Requirements → Test Map
| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| DEPLOY-01 | Docker compose build succeeds (both containers) | smoke | `docker compose -f docker-compose.prod.yml build` | ❌ Wave 0 (CI step) |
| DEPLOY-01 | Resource limits updated in docker-compose.prod.yml | unit (config check) | `uv run pytest tests/test_deploy_config.py::test_resource_limits -x` | ❌ Wave 0 |
| DEPLOY-02 | startup_checks.py validates all required env keys | unit | `uv run pytest tests/test_startup_checks.py::test_env_key_validation -x` | ❌ Wave 0 |
| DEPLOY-03 | startup_checks.py hard-fails on missing broker creds | unit | `uv run pytest tests/test_startup_checks.py::test_broker_credential_hard_fail -x` | ❌ Wave 0 |
| DEPLOY-03 | startup_checks.py warns (not fails) on empty memory | unit | `uv run pytest tests/test_startup_checks.py::test_memory_empty_warn -x` | ❌ Wave 0 |
| DEPLOY-04 | scheduler jobs registered with correct timezones | unit | `uv run pytest tests/scheduler/test_jobs.py::test_equity_cycle_timezone -x` | ❌ Wave 0 |
| DEPLOY-04 | show_schedule.py prints next fire times (ET + UTC) | smoke | `uv run pytest tests/test_show_schedule.py -x` | ❌ Wave 0 |
| DEPLOY-05 | AlpacaFillIngestor: fill event → memory ingest | unit | `uv run pytest tests/execution/test_fill_ingestor.py::test_alpaca_fill_ingest -x` | ❌ Wave 0 |
| DEPLOY-05 | BinanceFillIngestor: no-op in paper mode | unit | `uv run pytest tests/execution/test_fill_ingestor.py::test_binance_stub_paper -x` | ❌ Wave 0 |
| DEPLOY-05 | LiveMemoryClient: cycle_gate fail-open on connection refused | unit | `uv run pytest tests/memory/test_live_client.py::test_cycle_gate_fail_open -x` | ❌ Wave 0 |
| DEPLOY-05 | LiveMemoryClient: trade_veto fail-open on timeout | unit | `uv run pytest tests/memory/test_live_client.py::test_trade_veto_fail_open -x` | ❌ Wave 0 |
| DEPLOY-05 | execute_cycle() calls cycle_gate before proceeding | unit | `uv run pytest tests/execution/test_pipeline.py::test_cycle_gate_blocks_cycle -x` | ❌ Wave 0 |
| DEPLOY-05 | execute_cycle() skips vetoed signals | unit | `uv run pytest tests/execution/test_pipeline.py::test_trade_veto_skips_signal -x` | ❌ Wave 0 |
| DEPLOY-05 | CircuitBreaker apply_overrides tightens only | unit | `uv run pytest tests/execution/test_circuit_breaker.py::test_apply_overrides_tighten_only -x` | ❌ Wave 0 |
| DEPLOY-05 | smoke_test.py exits 0 with full pipeline mock | integration | Manual / docker exec | Manual-only in CI |
| DEPLOY-05 | memory_live_decisions DuckDB table populated after cycle | unit | `uv run pytest tests/execution/test_pipeline.py::test_memory_decisions_logged -x` | ❌ Wave 0 |

### Sampling Rate
- **Per task commit:** `uv run pytest tests/test_startup_checks.py tests/execution/test_fill_ingestor.py tests/memory/test_live_client.py -x`
- **Per wave merge:** `uv run pytest tests/ -v`
- **Phase gate:** Full suite green before `/gsd:verify-work`

### Wave 0 Gaps
- [ ] `tests/test_startup_checks.py` — covers DEPLOY-02, DEPLOY-03
- [ ] `tests/test_show_schedule.py` — covers DEPLOY-04 schedule display
- [ ] `tests/test_deploy_config.py` — covers DEPLOY-01 resource limits
- [ ] `tests/execution/test_fill_ingestor.py` — covers DEPLOY-05 FillIngestors
- [ ] `tests/memory/test_live_client.py` — covers DEPLOY-05 live endpoint fail-open
- [ ] `tests/execution/test_circuit_breaker.py` — extend with `test_apply_overrides_*` tests
- [ ] `tests/execution/test_pipeline.py` — extend with memory hook integration tests
- [ ] `services/memory/routers/live.py` — new live router (tested via memory service tests)
- [ ] Framework install: all deps present — add SQLAlchemy, requests, python-binance to `pyproject.toml [project.dependencies]`; move apscheduler from dev to prod deps

*(Existing test infrastructure covers scheduler, pipeline, and circuit breaker basics — new test files cover the new production deployment code paths)*

---

## Sources

### Primary (HIGH confidence)
- Codebase: `scripts/main.py` — APScheduler job registration, existing timezone handling confirmed correct
- Codebase: `src/swingrl/execution/pipeline.py` — execute_cycle() integration points identified
- Codebase: `src/swingrl/memory/client.py` — fail-open urllib pattern for live endpoint replication
- Codebase: `services/memory/memory_agents/query.py` — QueryAgent pattern (qwen2.5:3b, structured JSON, clamped defaults)
- Codebase: `services/memory/routers/core.py` — FastAPI router pattern for live.py
- Codebase: `src/swingrl/execution/risk/circuit_breaker.py` — `_max_dd`, `_daily_limit` fields for override pattern
- Codebase: `pyproject.toml` — confirmed missing deps (SQLAlchemy, python-binance, requests)
- Codebase: `docker-compose.prod.yml` — existing resource limits, service definitions, bind mounts
- [APScheduler 3.x SQLAlchemyJobStore docs](https://apscheduler.readthedocs.io/en/3.x/modules/jobstores/sqlalchemy.html) — confirmed SQLAlchemy>=2.0.24 required
- [alpaca-py TradingStream SDK](https://alpaca.markets/sdks/python/api_reference/trading/stream.html) — confirmed `subscribe_trade_updates`, `paper=True` param

### Secondary (MEDIUM confidence)
- [python-binance websockets docs](https://python-binance.readthedocs.io/en/latest/websockets.html) — ThreadedWebsocketManager pattern, `tld='us'` param, built-in 5-retry reconnect
- [Finnhub economic calendar API](https://finnhub.io/docs/api/economic-calendar) — 60 req/min free tier; `finnhub-python` already in pyproject.toml

### Tertiary (LOW confidence)
- Finnhub free tier economic calendar impact classification — not confirmed from docs; may require testing with real API key to verify `impact` field availability on free tier

---

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — all libraries are existing project dependencies or documented companion deps
- Architecture: HIGH — all integration points confirmed from codebase inspection; patterns replicate established Phase 19.1 patterns
- Pitfalls: HIGH — most derive from actual codebase state (missing deps confirmed, TZ config confirmed correct, bind mount ownership is a real Docker pitfall)
- MacroWatcher source: MEDIUM — Finnhub is the right library but free tier impact classification needs validation

**Research date:** 2026-03-15
**Valid until:** 2026-04-15 (stable libraries; APScheduler 3.x and python-binance 1.x are stable)
