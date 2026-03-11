# Project Research Summary

**Project:** SwingRL v1.1 — Operational Deployment
**Domain:** RL trading system — automated retraining, production Docker, Discord alerting, operator runbook
**Researched:** 2026-03-10
**Confidence:** HIGH

## Executive Summary

SwingRL v1.0 shipped a complete 17-phase MVP with all trading infrastructure built and tested. The v1.1 milestone is not greenfield development — it is operational deployment of that infrastructure onto the homelab Intel i5 server. The recommended approach is a strict sequential bootstrap: ingest historical data, train models on homelab CPU, deploy Docker stack, wire Discord alerts, implement automated retraining, then document everything in an operator runbook. The ordering is load-bearing: each step is a prerequisite for the next, and skipping steps or reordering them causes silent failures rather than obvious errors.

The dominant risk pattern for this milestone is "looks done but isn't." Docker containers start healthy while silently writing nothing due to bind-mount permission mismatches. Discord alerts are configured but fire to the wrong channel or drop under burst conditions. APScheduler jobs schedule at UTC instead of ET. Models load without error but operate outside their training distribution because VecNormalize statistics were not loaded. Every phase must include explicit verification steps with defined pass/fail criteria — not just "did the code run."

The one genuine new capability required is automated retraining: two new APScheduler jobs (`equity_retrain_job` monthly, `crypto_retrain_job` biweekly) plus four new Discord embed types and a bootstrap guard in the shadow promoter. All other v1.1 work is wire-up, configuration, and documentation of existing v1.0 modules. The codebase is production-ready; the deployment is not.

## Key Findings

### Recommended Stack

Two dependencies are missing from `pyproject.toml` and will cause Docker container failures at runtime: `APScheduler>=3.10,<4` and `SQLAlchemy>=2.0,<3`. These are used in `scripts/main.py` behind a `try/except ImportError` guard but are not declared as formal dependencies. Both must be added via `uv add` before any other v1.1 work ships. APScheduler must be pinned below 4.0 — the 4.0 alpha has a completely incompatible API and data store format; `uv lock` refresh can silently upgrade to it if unpinned.

No other new libraries are needed. The existing `httpx`-based Discord alerter handles all webhook requirements and must not be replaced with the `discord-webhook` PyPI package (which introduces a `requests` dependency). SB3 training on the Intel i5 uses `DummyVecEnv` (not `SubprocVecEnv`) — multiprocessing overhead exceeds computation gains for the low-complexity observation vectors. SAC replay buffer must be capped at `buffer_size=200_000` for Docker (default 1M would consume ~6GB during training). Container memory limit should be raised from 2.5GB to 6GB for retraining windows.

**Core technologies (additions only):**
- `APScheduler>=3.10,<4`: production scheduler already wired in main.py — declare the dependency
- `SQLAlchemy>=2.0,<3`: required by APScheduler SQLAlchemyJobStore for persistent job state
- `OMP_NUM_THREADS=4`, `MKL_NUM_THREADS=4`: Docker env vars to prevent CPU over-subscription on i5

### Expected Features

The v1.1 feature set is entirely P1 (blocking for hands-off paper trading) or P2 (add during the paper trading period). Nothing is truly greenfield — the distinction is between wire-up work and new code.

**Must have (table stakes — system cannot run hands-off without these):**
- Full historical data ingestion on homelab — prerequisite for training; DuckDB must be populated with aligned OHLCV + macro + feature data before train.py can run
- Agent training on homelab CPU — PPO/A2C/SAC for equity and crypto passing all 4 walk-forward validation gates, models placed in models/active/
- Production Docker stack running — both containers healthy, bind mounts writable, .env populated with secrets
- Discord webhook channel wiring — critical/warning/daily webhooks configured and smoke-tested
- Automated retraining job — equity monthly + crypto biweekly, with inline walk-forward validation gate check before shadow deployment (this is the only genuinely new code in v1.1)
- Automated shadow promotion — existing promoter wired to retraining output, with bootstrap guard preventing zero-baseline spurious promotions
- Operator runbook — step-by-step procedures for every workflow; the single largest documentation gap

**Should have (add during paper trading period):**
- Model performance degradation alert (rolling Sharpe below threshold)
- Retraining dry-run validation gate (programmatic backtest invocation inside retraining job)
- Startup validation sequence (pre-flight checks on container start)
- 6-month paper trading validation log with defined exit criteria for live promotion

**Defer (v2+):**
- Live trading promotion — requires completed 6-month paper validation period
- SPX options environment — requires Charles Schwab account + $2K+ capital

### Architecture Approach

The architecture is single-container APScheduler daemon with 12 existing jobs, expanding to 14 with two retraining jobs added. The key architectural decision is that retraining and shadow promotion remain decoupled: the retraining job trains and calls `ModelLifecycle.deploy_to_shadow()` only; the existing `shadow_promotion_check_job` (daily 7pm ET) handles promotion via 3-criteria evaluation the following day. This is load-bearing for capital preservation — it prevents a freshly trained model from immediately replacing the active model without shadow evaluation. Retraining jobs follow the existing `jobs.py` pattern exactly: `_get_ctx()`, halt check, try/except with alerter on exception.

**Major components:**

1. `scheduler/jobs.py` + `scripts/main.py` — APScheduler daemon, 14 jobs after v1.1; add equity_retrain_job and crypto_retrain_job following the existing function pattern
2. `training/trainer.py` + `agents/validation.py` — TrainingOrchestrator wrapped by retraining job; validation gates must be called between train() and deploy_to_shadow()
3. `shadow/lifecycle.py` + `shadow/promoter.py` — Model state machine; promoter needs a `min_active_trades` bootstrap guard before enabling auto-promotion
4. `monitoring/alerter.py` + `monitoring/embeds.py` — Discord delivery; four new embed builders needed for retrain lifecycle events
5. `docker-compose.prod.yml` — Production container; needs mem_limit raised to 6g, OMP/MKL thread vars, and HEALTHCHECK stanzas with `depends_on: condition: service_healthy`

### Critical Pitfalls

1. **MPS-trained models produce wrong outputs on CPU** — All training must happen on homelab CPU from day one. Never transfer VecNormalize from M1 Mac to homelab. Add post-load assertion: `assert str(next(model.policy.parameters()).device) == "cpu"`. Warning sign: shadow Sharpe drifts to 0.0 within first week despite varied market conditions.

2. **Retraining overlaps trading cycles causing silent missed trades** — Schedule retraining jobs on Saturday 2 AM ET (equity) and Sunday 3 AM ET (crypto), with at least 4 hours gap from any trading cycle. Set `misfire_grace_time=600` for trading jobs. Run retraining as subprocess to isolate CPU pressure. Warning sign: APScheduler logs show equity_cycle missed by N seconds on monthly retraining dates.

3. **Shadow auto-promotes during bootstrap when active baseline is zero** — `trades` table has no rows on fresh deployment, so any shadow Sharpe (even 0.1) beats the 0.0 active baseline. Add `min_active_trades` bootstrap guard to `promoter.py` that returns False when trades table has fewer than N rows. Warning sign: `shadow_promoted` fires within the first week of deployment.

4. **Docker bind mount permissions silently break all writes** — Bind-mounted directories (`./db`, `./data`, `./models`, `./logs`) may be owned by root on the homelab host; container runs as trader (UID 1000) and cannot write. Container shows healthy but writes nothing. Mitigation: `sudo chown -R 1000:1000 db/ data/ models/ logs/` before first launch; post-deploy write test: `docker exec swingrl touch /app/db/.write_test`.

5. **VecNormalize not loaded at inference causes silent policy degradation** — Loading `model.zip` without the corresponding `vec_normalize.pkl` passes raw (unnormalized) observations to a network trained on normalized inputs. No error is thrown; action outputs are nonsensical. Add assertion: `assert vec_path.exists()` in every model-loading path. Verify with an integration test: load both files, feed known raw obs, verify bounded action output.

6. **Retraining skips walk-forward validation gates** — `TrainingOrchestrator.train()` does not automatically call `AgentValidator.validate()`. A retraining job that chains `train() → deploy_to_shadow()` without the intermediate validation step puts an unvalidated model into the shadow pipeline. The retraining job must explicitly call all 4 gates and abort shadow deployment on any failure.

7. **APScheduler CronTrigger fires at UTC not ET** — `TZ=America/New_York` in Docker environment sets OS timezone but does not propagate into APScheduler trigger objects. Always pass `timezone="America/New_York"` explicitly to every CronTrigger. Post-deploy verification: `scheduler.get_jobs()` must show 4:15 PM ET next run time for equity_cycle, not 4:15 PM UTC.

## Implications for Roadmap

Based on research, dependencies are strict and sequential. The suggested phase structure maps directly to the dependency chain identified in FEATURES.md and ARCHITECTURE.md.

### Phase 1: Data Ingestion and Homelab Baseline
**Rationale:** Everything downstream reads DuckDB. Training, inference, and feature computation all require populated market_data.ddb with dimension-aligned OHLCV + macro + feature columns. This must come absolutely first.
**Delivers:** Populated DuckDB with validated feature dimensions (156-dim equity, 45-dim crypto), no NaN in observation columns, confirmed row counts for all symbols and date ranges.
**Addresses:** Data ingestion run (P1 table stake)
**Avoids:** Training on misaligned data that produces degenerate models passing smoke tests but failing on live data

### Phase 2: Initial Model Training and Validation
**Rationale:** Trained models are required before Docker stack is useful. The initial bootstrap must happen on the homelab CPU (not M1 Mac) to avoid the MPS/CPU device mismatch pitfall.
**Delivers:** PPO/A2C/SAC models for equity and crypto, validated through all 4 walk-forward gates, placed in models/active/. VecNormalize statistics confirmed present alongside each model.zip.
**Addresses:** Agent training on homelab (P1 table stake); VecNormalize load integration
**Avoids:** MPS-trained model pitfall (Pitfall 1), VecNormalize missing at inference (Pitfall 5)
**Note:** This phase establishes the no-MPS-transfer rule documented in the runbook

### Phase 3: Production Docker Deployment
**Rationale:** Requires active models to exist (Phase 2) and DuckDB populated (Phase 1). This phase establishes the paper trading baseline — all subsequent automated work runs inside these containers.
**Delivers:** Both containers running on homelab, healthchecks green, bind mounts writable, .env with all secrets, paper trading scheduler firing equity_cycle and crypto_cycle at correct ET times.
**Addresses:** Production Docker stack (P1 table stake), pyproject.toml gap for APScheduler/SQLAlchemy
**Avoids:** Bind mount permission mismatch (Pitfall 4), APScheduler timezone misconfiguration (Pitfall 6)
**Stack additions:** Add `APScheduler>=3.10,<4` and `SQLAlchemy>=2.0,<3` to pyproject.toml in this phase

### Phase 4: Discord Alert Suite
**Rationale:** Requires running Docker stack (Phase 3) to test end-to-end. Can partially overlap with Phase 3 (webhook URL creation), but smoke testing requires containers running. Also establishes the alert infrastructure needed by the retraining job (Phase 5).
**Delivers:** Both Discord channels (#alerts, #daily) receiving correct embed types at correct severity levels. Four new retrain embed builders in embeds.py. Rate-limit retry logic added to Alerter._post_webhook(). Smoke test confirming each alert level routes correctly.
**Addresses:** Discord webhook channel wiring (P1 table stake), retraining Discord notification (P2 differentiator)
**Avoids:** Discord rate limit dropping critical alerts (Pitfall 5)

### Phase 5: Automated Retraining Jobs
**Rationale:** Requires paper trading running (Phase 3) to provide shadow_trades data for promotion evaluation. Requires Discord alert suite (Phase 4) because retrain alerts are the first test of new embed types. This is the only phase with genuinely new production code.
**Delivers:** equity_retrain_job (monthly, Saturday 2 AM ET) and crypto_retrain_job (biweekly, Sunday 3 AM ET) in jobs.py and main.py. Inline walk-forward validation gate check before shadow deployment. Bootstrap guard in promoter.py. Job count increases from 12 to 14.
**Addresses:** Automated retraining job (P1 table stake), automated shadow promotion (P1 table stake)
**Avoids:** Retraining overlapping trading cycles (Pitfall 2), shadow bootstrap promotion (Pitfall 3), retraining skipping validation gates (Pitfall 8)
**New code:** ~150-200 lines — two job functions, two add_job() registrations, four embed builders, one bootstrap guard condition

### Phase 6: Operator Runbook
**Rationale:** Must come last — can only be written accurately once all components are deployed and verified. Documents the final behavior of all other features. Covers all 10 runbook sections identified in FEATURES.md research.
**Delivers:** Complete operator runbook in docs/runbook/ covering: initial setup, data ingestion walkthrough, first training run, Docker deployment, Discord verification, automated retraining expectations, emergency stop (4 tiers), model rollback procedure, common failure diagnosis, disaster recovery.
**Addresses:** Operator runbook (P1 table stake — the single largest documentation gap)
**Avoids:** Operator unable to recover from failures without source code archaeology

### Phase Ordering Rationale

- Data before training before Docker is strict (each phase's prerequisite is the prior phase's output)
- Discord before retraining is not strictly required by code but is required by operational safety: the first automated retraining run should deliver alerts; debugging alert delivery during retraining adds confounding variables
- Runbook last is mandatory: it documents verified behavior, not intended behavior
- No phase can be safely parallelized with an earlier phase except Discord channel creation (a 5-minute operator task) with late Phase 3 container startup

### Research Flags

Phases with standard, well-documented patterns (research-phase not needed):
- **Phase 1 (Data Ingestion):** Existing pipeline.py and ingestion scripts exist; this is execution and validation, not design
- **Phase 3 (Docker Deployment):** docker-compose.prod.yml is production-grade; work is configuration and verification
- **Phase 4 (Discord Alerts):** Alerter architecture is established; work is wiring and new embed builders
- **Phase 6 (Runbook):** Documentation task; no implementation research needed

Phases that may benefit from targeted implementation research:
- **Phase 2 (Model Training):** VecNormalize device handling under SB3 2.7.1 should be verified against SB3 release notes before assuming the two-file contract is sufficient — 30 minutes of verification against official docs during planning
- **Phase 5 (Retraining Jobs):** The SAC buffer_size reduction (200K from default 1M) and its impact on policy quality for the crypto env should be validated against SB3 documentation during planning

## Confidence Assessment

| Area | Confidence | Notes |
|------|------------|-------|
| Stack | HIGH | Research derived from direct codebase audit + PyPI source verification; APScheduler version pinning verified against release history |
| Features | HIGH | Existing v1.0 codebase audited directly; feature gaps are confirmed code gaps, not inferred gaps; community patterns for RL retraining schedules are MEDIUM confidence |
| Architecture | HIGH | All component responsibilities derived from reading production source files; retraining job pattern derived from existing job function implementations |
| Pitfalls | HIGH | Critical pitfalls derived from direct code inspection of failure modes; 3 of 8 pitfalls have direct GitHub issue citations for confirmation |

**Overall confidence:** HIGH

### Gaps to Address

- **Ensemble Sharpe weights (known tech debt):** pyproject.toml and PITFALLS.md both flag that ensemble weights use placeholder Sharpe ratios rather than walk-forward computed values. This must be resolved before the 6-month paper trading period begins — not before v1.1 ships, but within the paper trading period while there is still time to correct it. Flag in Phase 5 planning.

- **SAC buffer_size for Docker:** The recommended cap of 200K (from 1M default) is conservative but unvalidated against policy quality metrics for the crypto env. Monitor shadow Sharpe on first post-retrain crypto shadow evaluation and be prepared to adjust.

- **Paper trading exit criteria undefined:** FEATURES.md identifies that the 6-month validation period needs defined pass/fail criteria (minimum Sharpe, maximum MDD, minimum trade count) before it starts. These should be defined as part of Phase 3 (when paper trading begins), not at month 6 when they would influence a high-stakes decision.

- **HMM regime with insufficient live data:** hmmlearn diverges with fewer than 60 observations. After initial deployment, if macro data is sparse, HMM features may produce NaN that propagates into the observation vector on the first retraining cycle. Add minimum observation assertion to FeaturePipeline before retraining phase.

## Sources

### Primary (HIGH confidence — direct codebase inspection)
- `src/swingrl/scheduler/jobs.py` — APScheduler job function pattern, existing 12 jobs
- `src/swingrl/monitoring/alerter.py` — Discord webhook routing, cooldown logic, rate limit behavior
- `src/swingrl/shadow/promoter.py` — 3-criteria promotion logic, bootstrap vulnerability identified
- `src/swingrl/training/trainer.py` — TrainingOrchestrator, SAC buffer_size default, DummyVecEnv usage
- `docker-compose.prod.yml` — Resource caps, bind mounts, two-container architecture
- `Dockerfile` — Two-stage build, non-root trader user, HEALTHCHECK stanza
- `.planning/PROJECT.md` — v1.1 milestone goals, active requirements, known tech debt
- `.planning/RETROSPECTIVE.md` — v1.0 lesson: integration gaps from cross-module wiring not verified during development

### Secondary (HIGH confidence — official documentation)
- PyPI APScheduler release history — 3.11.2 stable, 4.0.0a6 alpha; version pinning rationale confirmed
- Discord developer docs — 30 requests/minute per webhook URL rate limit
- APScheduler docs — CronTrigger timezone parameter requirement; SQLAlchemyJobStore SQLAlchemy 2.x compatibility
- SB3 docs — DummyVecEnv vs SubprocVecEnv guidance for low-complexity environments

### Secondary (MEDIUM confidence — community sources)
- MLOps for quantitative trading teams (Luxoft, Medium) — scheduled retraining patterns
- Walk-forward optimization guides — 18-24 month window for swing trading strategies
- Docker Compose healthcheck + restart policy interaction — `unless-stopped` behavior
- GitHub issue: APScheduler CronTrigger timezone not defaulting to scheduler timezone (#346)
- GitHub issue: SB3 GPU-to-CPU model loading behavior (#159 SB3-contrib)

---
*Research completed: 2026-03-10*
*Ready for roadmap: yes*
