# Stack Research

**Domain:** Operational deployment additions — automated CPU retraining, Discord alerting, production Docker
**Researched:** 2026-03-10
**Confidence:** HIGH

---

## Context: What Already Exists (Do Not Re-add)

The following are locked in v1.0 and confirmed working. Research covers only **additions and gaps**:

| Existing | Version | Status |
|----------|---------|--------|
| Python | 3.11 | Locked — FinRL/pyfolio compatibility |
| Stable Baselines3 | 2.7.1 (locked) | In pyproject + uv.lock |
| torch (CPU-only Linux) | 2.3.1+cpu (locked) | pytorch-cpu index, linux marker |
| httpx | 0.28.1 (locked) | Already used by Alerter for Discord webhooks |
| DuckDB | 1.4.4 (locked) | In pyproject + uv.lock |
| APScheduler (3.x) | **NOT in pyproject.toml** | Used in scripts/main.py with try/except guard only |
| SQLAlchemy | **NOT in pyproject.toml** | Required by APSchedulerJobStore — missing dep |
| Discord alerter | Custom (Alerter class) | Already built; uses httpx, two-webhook routing |

---

## Additions and Gaps for v1.1

### Critical Missing Dependencies (Gaps Identified)

APScheduler and SQLAlchemy are imported in `scripts/main.py` behind a `try/except ImportError` guard but are **not declared in `pyproject.toml`**. The Docker production container will fail at runtime if APScheduler is not installed. Both must be added to `pyproject.toml` before v1.1 ships.

| Package | Version Constraint | Why | Source |
|---------|-------------------|-----|--------|
| `APScheduler` | `>=3.10,<4` | Already used for 12 scheduled jobs; 3.x is production-stable (4.0 is alpha as of 2025-04). Must pin to 3.x to avoid auto-upgrade to incompatible 4.0 API | PyPI confirmed: 3.11.2 is stable, 4.0.0a6 is alpha |
| `SQLAlchemy` | `>=2.0,<3` | APScheduler SQLAlchemyJobStore requires it; v2.x is stable and supports Python 3.11 | APScheduler docs |

### Automated CPU Retraining (Homelab Intel i5, 64GB RAM)

No new libraries are required. The retraining capability is a **scheduling + scripting concern**, not a new library concern. Analysis of what exists:

- `scripts/train.py` — full CLI for PPO/A2C/SAC training on equity or crypto
- `src/swingrl/training/trainer.py` — TrainingOrchestrator wrapping SB3 learn()
- `src/swingrl/shadow/` — lifecycle, promoter, shadow_runner already built
- `src/swingrl/scheduler/jobs.py` — APScheduler jobs wired (shadow_promotion_check_job exists)

**What is missing:** A scheduler job that invokes `scripts/train.py` programmatically for monthly equity and biweekly crypto retraining with shadow promotion. This is a new job function in `jobs.py`, not a new library.

**CPU threading configuration:** SB3 uses PyTorch CPU ops. On the i5-13th gen (8 P-cores + 4 E-cores), set `OMP_NUM_THREADS=4` and `MKL_NUM_THREADS=4` in Docker environment to avoid over-subscription. Using `DummyVecEnv` (not `SubprocVecEnv`) is correct for the low-complexity trading environment — multiprocessing overhead exceeds compute gains for small obs vectors (156-dim equity, 45-dim crypto).

### Discord Alert Suite (Full Coverage)

The Discord alerter (`src/swingrl/monitoring/alerter.py`) is already built with:
- Level-based routing (critical/warning → alerts webhook, info/digest → daily webhook)
- Rate-limit-aware cooldown deduplication
- Rich embed formatting (embeds.py)
- httpx async-capable HTTP client

**What is missing:** The alert is wired but the webhook URLs are not configured in the production Docker environment. The v1.1 work is **configuration and alert coverage expansion** (adding retraining-specific alerts), not new library work.

**Verdict:** No new libraries needed for Discord. The custom httpx-based alerter already handles all Discord webhook requirements and avoids the `discord-webhook` PyPI package's `requests` dependency (which would conflict with the project's httpx-first pattern).

### Production Docker Deployment

The production Dockerfile and `docker-compose.prod.yml` already exist and are structured correctly. Identified gaps:

| Gap | Resolution | New Dependency? |
|-----|-----------|-----------------|
| APScheduler + SQLAlchemy not in pyproject | Add to pyproject.toml | YES (see Critical gaps above) |
| `OMP_NUM_THREADS` / `MKL_NUM_THREADS` not set | Add to docker-compose.prod.yml environment | No |
| No separate retraining service/profile | Use APScheduler cron job within existing container OR add `retrain` Docker Compose profile | No |
| Training resource limits in prod compose: 1 CPU cap | Increase temporarily for retraining job, or allow APScheduler job to use host CPU | No |

**Retraining service pattern:** Use APScheduler within the existing `swingrl` container (not a separate container). Training is a monthly/biweekly batch job that can run inside the production container alongside the scheduler loop. Separate container would add complexity without benefit at this scale (single operator, one box). Memory ceiling: 6-model training (PPO+A2C+SAC x equity+crypto) peaks at ~4GB RAM on CPU — raise `mem_limit` from 2.5g to 6g for the production container.

---

## Recommended Stack Additions

### Core Technologies (New Additions Only)

| Technology | Version | Purpose | Why Recommended |
|------------|---------|---------|-----------------|
| `APScheduler` | `>=3.10,<4` | 12-job production scheduler already in main.py | 3.x is production-stable (3.11.2 current); 4.x is alpha — must pin below 4 to prevent auto-upgrade via `uv lock` refresh |
| `SQLAlchemy` | `>=2.0,<3` | APScheduler SQLAlchemyJobStore backend for persistent job state | v2.0 modernized API; APScheduler 3.x SQLAlchemyJobStore tested with v2.0 |

### Supporting Libraries (No New Additions)

| Library | Status | Notes |
|---------|--------|-------|
| `httpx` (0.28.1) | Already locked | Sufficient for Discord webhooks — do not add `discord-webhook` package |
| `torch` CPU-only | Already locked | pytorch-cpu index on linux marker handles homelab correctly |
| `stable-baselines3` (2.7.1) | Already locked | CPU training is native; no GPU-specific additions needed |
| `mkdocs-material` | NOT needed | Operator runbook is Markdown files in `docs/` — no static site generation needed for single-operator system |

### Development Tools (No New Additions)

All pre-commit hooks, ruff, mypy, bandit, detect-secrets remain unchanged.

---

## Installation

```bash
# Add to pyproject.toml [project.dependencies]:
# "APScheduler>=3.10,<4",
# "SQLAlchemy>=2.0,<3",

uv add "APScheduler>=3.10,<4" "SQLAlchemy>=2.0,<3"
```

---

## Alternatives Considered

| Recommended | Alternative | Why Not |
|-------------|-------------|---------|
| APScheduler 3.x (pin `<4`) | APScheduler 4.0 | 4.0 is alpha (4.0.0a6 as of Apr 2025); completely redesigned API — add_job() renamed to add_schedule(), jobstores renamed to data stores, data incompatible with 3.x persistent stores. Not production-ready. |
| Custom httpx alerter (existing) | `discord-webhook` PyPI package (v1.4.1) | `discord-webhook` adds `requests` as a hard dependency alongside the project's existing `httpx`. Introducing two HTTP clients for the same purpose adds bloat and confusion. The existing custom alerter already handles rate limiting, embeds, two-webhook routing, and deduplication. |
| DummyVecEnv for retraining | SubprocVecEnv with n_envs=4 | For the low-complexity trading environment (156-dim obs, simple step logic), SubprocVecEnv multiprocessing overhead exceeds computation time. DummyVecEnv is faster for this use case per SB3 documentation. |
| APScheduler cron job for retraining | Separate Docker Compose service | A dedicated retrain container would require orchestration (when to start, shared volume access, health signaling). APScheduler already owns the scheduling; a new job function in jobs.py is simpler and shares all existing context (config, db, alerter). |
| Markdown files in `docs/` for runbook | MkDocs Material static site | Operator runbook is used by one person SSHed into a server. A rendered static site adds build/serve complexity with no UX benefit. Raw Markdown in `docs/runbook/` is sufficient and immediately viewable on GitHub. |

---

## What NOT to Use

| Avoid | Why | Use Instead |
|-------|-----|-------------|
| `APScheduler>=4.0` (unpinned) | 4.0 alpha has incompatible API and data store format; uv can resolve to 4.x if not pinned | `APScheduler>=3.10,<4` |
| `discord-webhook` PyPI package | Adds `requests` dependency when project already uses `httpx`; the custom Alerter already provides all needed functionality | Existing `src/swingrl/monitoring/alerter.py` |
| `celery` or `rq` for task queuing | Massively over-engineered for a single-machine, single-operator system with 12 jobs and ~2 executions/day | APScheduler 3.x (already wired) |
| `ray` or `multiprocessing` for parallel training | 6 models train sequentially in ~30-60 min CPU total. Parallel training risks OOM on 64GB with current mem_limit. Sequential is safe and sufficient | Sequential training in TrainingOrchestrator |
| `prometheus-client` + Grafana | Full metrics stack is over-engineered for solo operator. Healthchecks.io dead man's switch (already integrated) + Discord alerts + Streamlit dashboard cover all monitoring needs | Existing monitoring stack |
| PyTorch 2.4+ | pyproject.toml pins `torch>=2.2,<2.4`; upgrading beyond pin is untested and not validated for this milestone | `torch 2.3.1` (locked) |

---

## Stack Patterns by Condition

**If retraining job runs out of memory on homelab:**
- Reduce `n_envs` to 1 (already `DummyVecEnv` default)
- Set `mem_limit: 6g` in docker-compose.prod.yml during training window
- Train equity and crypto sequentially (never simultaneously)

**If APScheduler cron job needs to survive Docker restart mid-training:**
- SQLAlchemy jobstore persists job state to SQLite
- APScheduler resumes missed jobs on restart (with misfire_grace_time set appropriately)
- Set `misfire_grace_time=3600` for monthly retraining jobs (1-hour grace window)

**If Discord webhook rate limit is hit (30 req/min per webhook):**
- Existing `Alerter` has cooldown deduplication to prevent burst alerts
- For retraining, send one "training started" + one "training complete" alert only
- Two-webhook routing already separates high-frequency daily digest from critical alerts

---

## Version Compatibility

| Package A | Compatible With | Notes |
|-----------|-----------------|-------|
| `APScheduler 3.10-3.11.x` | `SQLAlchemy 2.x` | APScheduler 3.x SQLAlchemyJobStore tested with SQLAlchemy 2.0+; confirmed in APScheduler docs |
| `APScheduler 3.10-3.11.x` | `Python 3.11` | Full support confirmed |
| `torch 2.3.1` | `stable-baselines3 2.7.1` | Locked in uv.lock; confirmed in CI |
| `APScheduler <4` | APScheduler 4.0 persistent stores | **Incompatible** — do not mix data stores between major versions |

---

## Sources

- PyPI APScheduler — confirmed 3.11.2 stable, 4.0.0a6 alpha: https://pypi.org/project/APScheduler/
- APScheduler migration guide (4.0 breaking changes): https://apscheduler.readthedocs.io/en/master/migration.html
- APScheduler 3.x user guide: https://apscheduler.readthedocs.io/en/3.x/userguide.html
- PyPI discord-webhook — confirmed 1.4.1, uses requests core + httpx async extra: https://pypi.org/project/discord-webhook/
- SB3 vectorized environments (DummyVecEnv vs SubprocVecEnv guidance): https://stable-baselines3.readthedocs.io/en/master/guide/vec_envs.html
- PyTorch CPU threading: https://docs.pytorch.org/docs/stable/generated/torch.set_num_threads.html
- Healthchecks.io (already integrated in v1.0): https://healthchecks.io/docs/

---

*Stack research for: SwingRL v1.1 Operational Deployment additions*
*Researched: 2026-03-10*
