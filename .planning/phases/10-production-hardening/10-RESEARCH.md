# Phase 10: Production Hardening - Research

**Researched:** 2026-03-09
**Domain:** Backup automation, model deployment, shadow mode, FinBERT sentiment, emergency stop, disaster recovery, security hardening
**Confidence:** HIGH

## Summary

Phase 10 hardens the SwingRL system for sustained autonomous operation across seven workstreams: backup automation (SQLite daily, DuckDB weekly, off-site monthly), model deployment pipeline (SCP + smoke test), shadow mode validation (parallel inference with auto-promotion), FinBERT sentiment experiment (A/B with +0.05 Sharpe threshold), four-tier emergency stop (halt + cancel + liquidate + verify), comprehensive retry/logging, and disaster recovery testing. The existing codebase provides strong foundations: `emergency_stop.py` (Tier 1 halt flag), `Alerter` (Discord routing), `ExecutionPipeline` (cycle orchestration), `DatabaseManager` (dual-DB context managers), and both exchange adapters with retry patterns.

The primary technical challenges are: (1) shadow mode must run a second inference pipeline within each cycle job without blocking the active pipeline, (2) emergency stop Tiers 2-4 require authenticated Binance.US order placement (currently only simulated fills exist) and Alpaca time-aware liquidation, (3) FinBERT adds ~500-800MB RAM within the 2.5GB Docker limit, requiring careful lazy loading. All required libraries are either already in dependencies (`exchange-calendars`, `alpaca-py`, `structlog`) or are standard Python (`shutil`, `sqlite3.backup`) or need adding (`transformers`, `finnhub-python`, `tenacity`).

**Primary recommendation:** Organize into 5-6 plans following dependency order: (1) config + DB schema extensions, (2) backup automation + retry hardening, (3) shadow mode + model deployment, (4) FinBERT sentiment pipeline, (5) emergency stop four-tier protocol, (6) security review + disaster recovery + notebooks.

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions
- Daily SQLite backups: retain 14 days, rotate oldest
- DuckDB backups: never rotate -- market data is permanent training asset
- Config and secrets (.env, swingrl.yaml) copied alongside trading_ops.db during daily backup
- Each backup job verifies integrity before considering successful (PRAGMA integrity_check for SQLite, table count + row count for DuckDB)
- Monthly off-site rsync via Tailscale to separate NAS/drive
- Discord alerts on every backup (success and failure) routed to #swingrl-daily channel
- Shadow mode minimum: 10 equity trading days / 30 crypto 4H cycles
- Fully automatic promotion when all 3 criteria met: Shadow Sharpe >= Active Sharpe, Shadow MDD <= 120% Active MDD, no circuit breaker triggers
- Shadow hypothetical trades stored in SQLite shadow_trades table (same schema as trades + model_version column)
- Failed shadow models: move to models/archive/, Discord alert, active model continues
- Model lifecycle: Training -> Shadow -> Active -> Archive -> Deletion
- FinBERT feature toggle: sentiment.enabled in swingrl.yaml
- Sharpe improvement threshold: +0.05 (matches FEAT-08 protocol)
- FinBERT runs within each trading cycle (not separate job)
- Headline sources: Alpaca News API (primary, 200 calls/min) + Finnhub (secondary, 60 calls/min free)
- ProsusAI/finbert from Hugging Face transformers
- Emergency stop: all 4 tiers run automatically in sequence, no flags, no prompts
- Tier 1 (<1s): halt flag + cancel ALL open orders
- Tier 2 (<30s): market-sell all crypto on Binance.US
- Tier 3 (<30s): auto time-aware equity liquidation via exchange_calendars
- Tier 4 (<1min): verify + Discord critical alert
- 3 automated triggers: VIX>40 + global CB within 2% of firing, 2+ consecutive NaN inferences in 24h, Binance.US IP ban (418) with open positions
- deploy_model.sh: SCP from M1 Mac, 6-point smoke test, model goes to models/shadow/ first
- Full 9-step quarterly recovery checklist per Docs 14/15
- Jupyter analysis notebooks in notebooks/ directory
- Structured JSON logging to bind-mounted logs/ volume
- Security review: non-root containers, env_file chmod 600, Binance.US IP allowlisting, 90-day staggered key rotation

### Claude's Discretion
- Backup cron schedule times (exact hours for daily/weekly/monthly jobs)
- Shadow mode inference integration with existing scheduler jobs
- Jupyter notebook layout and visualization library choices
- Retry backoff parameters (base delay, max retries, jitter)
- Security review checklist script implementation
- Disaster recovery test automation level (fully scripted vs manual checklist)
- deploy_model.sh shell script implementation details

### Deferred Ideas (OUT OF SCOPE)
None -- discussion stayed within phase scope.
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|-----------------|
| HARD-01 | Jupyter analysis notebooks for weekly performance review | Matplotlib/plotly for visualization, notebooks/ directory, portfolio curves + trade logs + risk metrics |
| HARD-02 | Error handling with retry logic and exponential backoff | tenacity library for decorator-based retry, extends Phase 8's 3-retry pattern across all API calls |
| HARD-03 | FinBERT sentiment pipeline (ProsusAI/finbert) | transformers library, Alpaca News API + Finnhub, lazy loading pattern for memory, sentiment table in SQLite |
| HARD-04 | A/B sentiment experiment | Feature toggle in config, compare_features() from Phase 5 pipeline, +0.05 Sharpe threshold |
| HARD-05 | Structured JSON logging to logs/ volume | Extend configure_logging() with FileHandler to bind-mounted logs/, rotation via logging.handlers.RotatingFileHandler |
| PROD-01 | Backup automation (SQLite daily, DuckDB weekly, monthly rsync) | Python shutil + sqlite3.backup API, integrity verification, cron jobs via APScheduler, Tailscale rsync |
| PROD-02 | deploy_model.sh with SCP + smoke test | Shell script with scp via Tailscale, 6-point smoke test (deserialize, shape, non-degenerate, inference speed, VecNormalize, no NaN) |
| PROD-03 | Shadow mode parallel inference (10 equity days, 30 crypto cycles) | Second pipeline inference per cycle, shadow_trades SQLite table, shadow model loading from models/shadow/ |
| PROD-04 | Shadow mode auto-promotion criteria | Sharpe comparison, MDD ratio check, CB trigger monitoring, auto-move from shadow/ to active/ |
| PROD-05 | Model lifecycle management | Training -> Shadow -> Active -> Archive -> Deletion, directory-based state machine |
| PROD-06 | Security review | Non-root trader user (existing), env_file permissions, API key security checklist, rotation runbook |
| PROD-07 | emergency_stop.py four-tier kill switch | Extend existing Tier 1, add authenticated Binance.US market sell, exchange_calendars for equity hours, verify+alert |
| PROD-08 | Disaster recovery test | Stop container, delete volumes, restore from backup, verify system resumes |
| PROD-09 | 9-step quarterly recovery checklist | Scripted verification: DB integrity, row counts, model loading, feature compute, scheduler start, first cycle |
</phase_requirements>

## Standard Stack

### Core (already in project)
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| exchange-calendars | >=4.13 | NYSE market hours detection for Tier 3 equity liquidation | Already a dependency; provides `is_open_on_minute()` |
| alpaca-py | >=0.20 | News API + cancel_orders + close_position for emergency stop | Already a dependency; TradingClient has cancel/liquidation methods |
| structlog | existing | JSON logging to file handler for HARD-05 | Already configured in utils/logging.py |
| sqlite3 | stdlib | backup() API for safe SQLite backups | Built-in, supports online backup API |
| shutil | stdlib | File copy for DuckDB backup | Built-in, simpler than DuckDB export for file-level backup |

### New Dependencies
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| transformers | >=4.40 | ProsusAI/finbert model loading and inference | HARD-03 FinBERT sentiment pipeline |
| finnhub-python | >=2.4 | Secondary news headlines source | HARD-03 Finnhub fallback headlines |
| tenacity | >=8.2 | Decorator-based retry with exponential backoff + jitter | HARD-02 retry hardening across all API calls |
| matplotlib | >=3.8 | Jupyter notebook visualization | HARD-01 performance review notebooks |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| tenacity | Manual retry loops (existing) | tenacity provides jitter, exception filtering, logging hooks -- much cleaner than hand-rolled |
| matplotlib | plotly | plotly is heavier; matplotlib is sufficient for static notebook analysis and doesn't add to Docker image |
| transformers | FinBERT standalone package | transformers is the standard HF inference API; FinBERT standalone is outdated |

**Installation:**
```bash
uv add transformers finnhub-python tenacity matplotlib
```

Note: `transformers` will pull in `torch` which is already a project dependency. The FinBERT model (~420MB) must be downloaded at first run or pre-cached in Docker build.

## Architecture Patterns

### Recommended New Module Structure
```
src/swingrl/
  backup/
    __init__.py
    sqlite_backup.py      # Daily SQLite backup with integrity check
    duckdb_backup.py      # Weekly DuckDB backup with verification
    offsite_sync.py       # Monthly rsync via Tailscale
  sentiment/
    __init__.py
    finbert.py            # FinBERT model wrapper with lazy loading
    news_fetcher.py       # Alpaca News + Finnhub headline fetcher
  shadow/
    __init__.py
    shadow_runner.py      # Shadow inference alongside active pipeline
    promoter.py           # Auto-promotion logic (3 criteria)
    lifecycle.py          # Model state machine (Training->Shadow->Active->Archive->Delete)
scripts/
  deploy_model.sh         # SCP + smoke test shell script
  security_checklist.py   # Security review verification script
  disaster_recovery.py    # Automated DR test (scripted 9-step)
notebooks/
  weekly_review.ipynb     # Performance analysis notebook
```

### Pattern 1: Lazy FinBERT Loading
**What:** Load the FinBERT model only on first use, not at module import or container startup.
**When to use:** Always for FinBERT -- the model is ~420MB RAM and takes several seconds to load.
**Example:**
```python
from __future__ import annotations
from typing import Any
import structlog

log = structlog.get_logger(__name__)

class FinBERTScorer:
    """Lazy-loading FinBERT sentiment scorer."""

    def __init__(self) -> None:
        self._model: Any = None
        self._tokenizer: Any = None

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        self._tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
        self._model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
        log.info("finbert_model_loaded")

    def score_headlines(self, headlines: list[str]) -> list[dict[str, float]]:
        self._ensure_loaded()
        # Batch tokenize and predict
        ...
```

### Pattern 2: Shadow Mode Dual Inference
**What:** Run shadow model inference after active model inference within the same cycle job.
**When to use:** Every trading cycle when a shadow model exists in models/shadow/.
**Example:**
```python
# In scheduler/jobs.py, after active pipeline.execute_cycle():
from swingrl.shadow.shadow_runner import run_shadow_inference

def equity_cycle() -> list[FillResult]:
    ctx = _get_ctx()
    if is_halted(ctx.db):
        return []
    fills = ctx.pipeline.execute_cycle("equity")
    # Shadow inference (non-blocking, never affects active)
    try:
        run_shadow_inference(ctx, "equity")
    except Exception:
        log.exception("shadow_inference_failed")
    return fills
```

### Pattern 3: SQLite Online Backup
**What:** Use Python's sqlite3.backup() API for safe live database backup.
**When to use:** Daily SQLite backup job -- this is safer than shutil.copy during active writes.
**Example:**
```python
import sqlite3
from pathlib import Path

def backup_sqlite(src_path: Path, dst_path: Path) -> None:
    """Create a consistent backup using SQLite's online backup API."""
    src = sqlite3.connect(str(src_path))
    dst = sqlite3.connect(str(dst_path))
    try:
        src.backup(dst)
    finally:
        dst.close()
        src.close()

    # Verify integrity of backup
    verify = sqlite3.connect(str(dst_path))
    try:
        result = verify.execute("PRAGMA integrity_check").fetchone()
        if result is None or result[0] != "ok":
            raise RuntimeError(f"Backup integrity check failed: {result}")
    finally:
        verify.close()
```

### Pattern 4: Tenacity Retry Decorator
**What:** Replace hand-rolled retry loops with tenacity decorators across all external API calls.
**When to use:** All broker API calls, news API calls, FRED API calls, Binance.US price fetches.
**Example:**
```python
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from swingrl.utils.exceptions import BrokerError

@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=1, max=30),
    retry=retry_if_exception_type((ConnectionError, TimeoutError)),
    reraise=True,
)
def fetch_with_retry(url: str) -> dict:
    ...
```

### Pattern 5: Emergency Stop Sequential Tiers
**What:** Run all 4 tiers in strict sequence, each with timeout protection.
**When to use:** emergency_stop.py invocation (manual or automated triggers).
**Example:**
```python
def execute_emergency_stop(config, db, alerter, reason: str) -> dict:
    """Execute all 4 tiers sequentially. Returns status report."""
    report = {}
    # Tier 1: Halt flag + cancel orders (<1s)
    set_halt(db, reason=reason, set_by="emergency_stop")
    report["tier1"] = cancel_all_orders(config)

    # Tier 2: Liquidate crypto immediately (<30s)
    report["tier2"] = liquidate_crypto(config, db)

    # Tier 3: Equity liquidation (time-aware) (<30s)
    report["tier3"] = liquidate_equity(config, db)

    # Tier 4: Verify + alert (<1min)
    report["tier4"] = verify_and_alert(config, db, alerter, report, reason)
    return report
```

### Anti-Patterns to Avoid
- **Loading FinBERT at module import:** Wastes 500-800MB RAM if sentiment is disabled; use lazy loading.
- **Running shadow inference in a separate thread:** Adds concurrency complexity; run sequentially after active cycle (shadow is non-time-critical).
- **DuckDB backup via EXPORT DATABASE:** File-level copy is simpler and faster for full backup; EXPORT is for selective data extraction.
- **Hardcoding backup retention:** Use config fields for retention days, not magic numbers in code.
- **Shared retry logic via inheritance:** Use tenacity decorators instead of custom base class retry methods.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Retry with backoff | Custom retry loops per adapter | tenacity decorators | Handles jitter, exception filtering, logging hooks, stop conditions |
| SQLite backup | shutil.copy() during writes | sqlite3.backup() API | Online backup API is transaction-safe during concurrent reads/writes |
| Market hours detection | Manual NYSE schedule mapping | exchange_calendars XNYS | Handles holidays, half-days, extended hours, DST changes |
| NLP sentiment scoring | Custom BERT fine-tuning | ProsusAI/finbert via transformers | Pre-trained, validated on financial text, zero fine-tuning needed |
| Log rotation | Custom file rotation logic | logging.handlers.RotatingFileHandler | Handles max size, backup count, atomic rotation |

**Key insight:** Every "seems simple" problem in this phase (market hours, backup safety, retry logic) has edge cases that existing libraries handle correctly. The risk of hand-rolling is silent data loss (backup during write), missed holidays (market hours), or thundering herd (retry without jitter).

## Common Pitfalls

### Pitfall 1: SQLite Backup During Active Writes
**What goes wrong:** Using shutil.copy() on trading_ops.db while APScheduler jobs are writing trades/positions corrupts the backup file.
**Why it happens:** SQLite WAL mode has a separate WAL file; copying only the .db file misses uncommitted transactions.
**How to avoid:** Use `sqlite3.backup()` API which handles WAL mode correctly and creates a consistent snapshot.
**Warning signs:** Backup file passes `PRAGMA integrity_check` but is missing recent trades.

### Pitfall 2: FinBERT Memory Pressure
**What goes wrong:** Loading FinBERT model consumes ~500-800MB RAM, pushing Docker container past 2.5GB limit and triggering OOM kill.
**Why it happens:** The transformers library loads the full model into memory alongside existing PyTorch models.
**How to avoid:** Lazy-load FinBERT only when `sentiment.enabled=True`. Unload after scoring batch. Monitor container memory usage in Streamlit dashboard.
**Warning signs:** Container restarts with exit code 137 (OOM killed).

### Pitfall 3: Shadow Mode Interfering with Active Trading
**What goes wrong:** Shadow inference error propagates and crashes the active trading cycle.
**Why it happens:** Shadow code called within the same try/except block as active pipeline.
**How to avoid:** Wrap all shadow logic in its own try/except that logs errors but never raises. Shadow failures are informational only.
**Warning signs:** Active fills stop after shadow model deployment.

### Pitfall 4: Emergency Stop Equity Liquidation Outside Market Hours
**What goes wrong:** Attempting market sell for equity positions at 2 AM -- order rejected or fills at bad price.
**Why it happens:** Not checking NYSE market hours before choosing order type.
**How to avoid:** Use exchange_calendars to check if NYSE is open. If open: limit sell at bid. If closed: queue with `time_in_force='opg'` (on-open). If extended hours: limit sell.
**Warning signs:** BrokerError on emergency stop equity liquidation.

### Pitfall 5: DuckDB Backup While Connection is Open
**What goes wrong:** Copying market_data.ddb while the DuckDB connection is held by the running process results in a corrupt backup.
**Why it happens:** DuckDB uses a write-ahead log similar to SQLite; file-level copy during writes is unsafe.
**How to avoid:** Temporarily close the DuckDB connection in DatabaseManager before backup, or use DuckDB's `CHECKPOINT` command to flush WAL before copying.
**Warning signs:** Backup DuckDB file fails to open or shows missing recent data.

### Pitfall 6: Binance.US Authenticated API for Emergency Liquidation
**What goes wrong:** The current BinanceSimAdapter only fetches prices from public API -- it cannot place real orders or cancel orders.
**Why it happens:** Paper trading mode uses simulated fills. Emergency stop needs real API calls.
**How to avoid:** Create an authenticated Binance.US client (using python-binance or direct REST) specifically for emergency operations. The adapter pattern already supports this via the ExchangeAdapter Protocol.
**Warning signs:** Emergency stop Tier 2 does nothing because simulated adapter's cancel is a no-op.

## Code Examples

### Backup Job Registration in APScheduler
```python
# Add to scripts/main.py create_scheduler_and_register_jobs()
from swingrl.backup.sqlite_backup import daily_sqlite_backup
from swingrl.backup.duckdb_backup import weekly_duckdb_backup

scheduler.add_job(
    daily_sqlite_backup,
    trigger="cron",
    hour=2, minute=0,  # 2:00 AM ET -- after daily summary, before market open
    timezone="America/New_York",
    id="daily_sqlite_backup",
    replace_existing=True,
)

scheduler.add_job(
    weekly_duckdb_backup,
    trigger="cron",
    day_of_week="sun",
    hour=3, minute=0,  # 3:00 AM ET Sunday
    timezone="America/New_York",
    id="weekly_duckdb_backup",
    replace_existing=True,
)
```

### Alpaca Cancel All Orders + Close Positions
```python
# Source: alpaca-py TradingClient API
from alpaca.trading.client import TradingClient

def cancel_all_and_liquidate_equity(client: TradingClient) -> dict:
    """Cancel all open orders and close all positions on Alpaca."""
    # Cancel all open orders
    cancel_statuses = client.cancel_orders()

    # Close all positions
    close_responses = client.close_all_positions(cancel_orders=True)

    return {
        "cancelled_orders": len(cancel_statuses),
        "closed_positions": len(close_responses),
    }
```

### Exchange Calendar Market Hours Check
```python
import exchange_calendars as xcals
from datetime import datetime, timezone

def get_equity_liquidation_strategy(now: datetime | None = None) -> str:
    """Determine equity liquidation strategy based on market hours."""
    nyse = xcals.get_calendar("XNYS")
    if now is None:
        now = datetime.now(tz=timezone.utc)

    # Convert to pandas Timestamp for exchange_calendars API
    import pandas as pd
    ts = pd.Timestamp(now)

    if nyse.is_open_on_minute(ts):
        return "limit_at_bid"  # Market is open, sell at current bid
    elif nyse.is_session(ts.normalize()):
        return "limit_extended"  # Trading day but outside core hours
    else:
        return "queue_for_open"  # Overnight/weekend: time_in_force='opg'
```

### Config Extensions for Phase 10
```python
class BackupConfig(BaseModel):
    """Backup automation configuration."""
    sqlite_retention_days: int = Field(default=14, ge=1)
    duckdb_rotate: bool = Field(default=False)  # Never rotate -- permanent asset
    backup_dir: str = Field(default="backups/")
    offsite_host: str = Field(default="")  # Tailscale hostname
    offsite_path: str = Field(default="")  # Remote backup directory

class ShadowConfig(BaseModel):
    """Shadow mode configuration."""
    equity_eval_days: int = Field(default=10, ge=5)
    crypto_eval_cycles: int = Field(default=30, ge=10)
    auto_promote: bool = Field(default=True)
    mdd_tolerance_ratio: float = Field(default=1.2, gt=1.0)  # 120% of active MDD

class SentimentConfig(BaseModel):
    """FinBERT sentiment configuration."""
    enabled: bool = Field(default=False)
    model_name: str = Field(default="ProsusAI/finbert")
    max_headlines_per_asset: int = Field(default=10, ge=1)
    finnhub_api_key: str = Field(default="")  # Set via env var

class SecurityConfig(BaseModel):
    """Security review configuration."""
    key_rotation_days: int = Field(default=90, ge=30)
    env_file_permissions: str = Field(default="600")
```

### Shadow Trades Table DDL
```sql
CREATE TABLE IF NOT EXISTS shadow_trades (
    trade_id TEXT PRIMARY KEY,
    timestamp TEXT NOT NULL,
    symbol TEXT NOT NULL,
    side TEXT NOT NULL,
    quantity REAL NOT NULL,
    price REAL NOT NULL,
    commission REAL DEFAULT 0.0,
    slippage REAL DEFAULT 0.0,
    environment TEXT NOT NULL,
    broker TEXT,
    order_type TEXT,
    trade_type TEXT,
    model_version TEXT NOT NULL  -- links to model in models/shadow/
)
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Manual retry loops per adapter | tenacity decorators | Library stable since 2022 | Cleaner code, consistent retry behavior, jitter built-in |
| shutil.copy for SQLite backup | sqlite3.backup() API | Python 3.7+ | Safe online backup during concurrent access |
| Custom sentiment models | Pre-trained FinBERT via transformers | 2020+ | Zero fine-tuning needed for financial sentiment |
| Manual cron via crontab | APScheduler in-process (already used) | Phase 9 | Backup/shadow jobs integrate naturally |

**Deprecated/outdated:**
- `alpaca-trade-api` package: Replaced by `alpaca-py` (already using the newer SDK)
- Standalone FinBERT repository: Use `transformers` pipeline instead

## Open Questions

1. **FinBERT Model Caching in Docker**
   - What we know: The model is ~420MB and downloads from HuggingFace on first use
   - What's unclear: Whether to pre-download during Docker build or lazy-download at runtime
   - Recommendation: Lazy-download at runtime with model cache in a bind-mounted volume. Pre-downloading in Docker build bloats the image and slows CI.

2. **Binance.US Authenticated API for Emergency Liquidation**
   - What we know: Current BinanceSimAdapter only does simulated fills. Emergency stop needs real market sells.
   - What's unclear: Whether to extend BinanceSimAdapter or create a separate authenticated client for emergency operations
   - Recommendation: Create a dedicated `BinanceEmergencyClient` in the emergency stop module. Keep the sim adapter unchanged for normal paper trading. This aligns with the broker middleware rule (all orders through execution/).

3. **DuckDB Backup Safety**
   - What we know: DuckDB doesn't expose a Python backup() API like SQLite
   - What's unclear: Safest way to backup while the process is running
   - Recommendation: Call `CHECKPOINT` via DuckDB cursor to flush WAL, then shutil.copy. This is the documented approach for DuckDB file-level backup. Alternatively, temporarily close and reopen the connection.

## Validation Architecture

### Test Framework
| Property | Value |
|----------|-------|
| Framework | pytest (existing) |
| Config file | pyproject.toml [tool.pytest] |
| Quick run command | `uv run pytest tests/ -x -q` |
| Full suite command | `uv run pytest tests/ -v` |

### Phase Requirements -> Test Map
| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| PROD-01 | SQLite backup creates valid copy with integrity check | unit | `uv run pytest tests/backup/test_sqlite_backup.py -x` | Wave 0 |
| PROD-01 | DuckDB backup creates valid copy with table/row verification | unit | `uv run pytest tests/backup/test_duckdb_backup.py -x` | Wave 0 |
| PROD-02 | Model smoke test validates 6 checkpoints | unit | `uv run pytest tests/test_deploy_smoke.py -x` | Wave 0 |
| PROD-03 | Shadow inference records hypothetical trades without affecting active | unit | `uv run pytest tests/shadow/test_shadow_runner.py -x` | Wave 0 |
| PROD-04 | Auto-promotion fires when all 3 criteria met | unit | `uv run pytest tests/shadow/test_promoter.py -x` | Wave 0 |
| PROD-05 | Model lifecycle moves through all states correctly | unit | `uv run pytest tests/shadow/test_lifecycle.py -x` | Wave 0 |
| PROD-06 | Security checklist passes all verification points | integration | `uv run pytest tests/test_security.py -x` | Wave 0 |
| PROD-07 | Emergency stop sets halt, cancels orders, liquidates per tier | unit | `uv run pytest tests/test_emergency_stop.py -x` | Wave 0 |
| PROD-08 | Disaster recovery restores DB and verifies integrity | integration | `uv run pytest tests/test_disaster_recovery.py -x` | Wave 0 |
| HARD-01 | Notebook execution completes without errors | smoke | manual-only (Jupyter kernel) | Wave 0 |
| HARD-02 | Retry decorator fires correct number of times with backoff | unit | `uv run pytest tests/test_retry.py -x` | Wave 0 |
| HARD-03 | FinBERT scores headlines and returns sentiment dict | unit | `uv run pytest tests/sentiment/test_finbert.py -x` | Wave 0 |
| HARD-04 | A/B comparison detects Sharpe improvement above threshold | unit | `uv run pytest tests/sentiment/test_ab_experiment.py -x` | Wave 0 |
| HARD-05 | JSON logging writes structured events to file | unit | `uv run pytest tests/utils/test_file_logging.py -x` | Wave 0 |

### Sampling Rate
- **Per task commit:** `uv run pytest tests/ -x -q`
- **Per wave merge:** `uv run pytest tests/ -v`
- **Phase gate:** Full suite green before `/gsd:verify-work`

### Wave 0 Gaps
- [ ] `tests/backup/test_sqlite_backup.py` -- covers PROD-01 SQLite
- [ ] `tests/backup/test_duckdb_backup.py` -- covers PROD-01 DuckDB
- [ ] `tests/shadow/test_shadow_runner.py` -- covers PROD-03
- [ ] `tests/shadow/test_promoter.py` -- covers PROD-04
- [ ] `tests/shadow/test_lifecycle.py` -- covers PROD-05
- [ ] `tests/sentiment/test_finbert.py` -- covers HARD-03
- [ ] `tests/sentiment/test_ab_experiment.py` -- covers HARD-04
- [ ] `tests/test_emergency_stop.py` -- covers PROD-07 (extend existing)
- [ ] `tests/test_retry.py` -- covers HARD-02
- [ ] `tests/utils/test_file_logging.py` -- covers HARD-05
- [ ] `tests/test_deploy_smoke.py` -- covers PROD-02
- [ ] `tests/test_security.py` -- covers PROD-06
- [ ] `tests/test_disaster_recovery.py` -- covers PROD-08/09
- [ ] `tests/backup/__init__.py`, `tests/shadow/__init__.py`, `tests/sentiment/__init__.py` -- package init
- [ ] Framework install: N/A -- pytest already configured

## Sources

### Primary (HIGH confidence)
- Existing codebase: `scripts/emergency_stop.py`, `src/swingrl/scheduler/jobs.py`, `src/swingrl/execution/pipeline.py`, `src/swingrl/monitoring/alerter.py`, `src/swingrl/data/db.py`, `src/swingrl/config/schema.py`
- Python stdlib documentation: `sqlite3.backup()` API
- exchange-calendars GitHub: market hours checking API (already a project dependency)

### Secondary (MEDIUM confidence)
- [ProsusAI/finbert on Hugging Face](https://huggingface.co/ProsusAI/finbert) - Model card, usage patterns
- [Alpaca News API docs](https://docs.alpaca.markets/reference/news-3) - News endpoint, rate limits
- [Finnhub Python client](https://github.com/Finnhub-Stock-API/finnhub-python) - company_news API, free tier limits
- [Tenacity documentation](https://tenacity.readthedocs.io/) - Retry patterns, exponential backoff, jitter
- [SQLite backup best practices](https://www.slingacademy.com/article/best-practices-for-managing-sqlite-backups-in-production/) - Online backup API vs file copy

### Tertiary (LOW confidence)
- Binance.US authenticated order placement via python-binance -- needs verification against current Binance.US API docs for emergency liquidation

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH - all core libraries verified, most already in project dependencies
- Architecture: HIGH - patterns follow existing codebase conventions, clear integration points
- Pitfalls: HIGH - backed by understanding of existing code (WAL mode, adapter patterns, memory limits)
- FinBERT integration: MEDIUM - model size/memory verified via HuggingFace, but Docker memory interaction needs runtime validation
- Emergency stop Binance.US auth: MEDIUM - current adapter is sim-only; authenticated client needs implementation verification

**Research date:** 2026-03-09
**Valid until:** 2026-04-09 (stable domain, no fast-moving dependencies)
