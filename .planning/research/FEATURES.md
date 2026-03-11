# Feature Research

**Domain:** Operational deployment — automated retraining, Discord alerting, production Docker, operator documentation for RL trading system.
**Researched:** 2026-03-10
**Confidence:** HIGH (existing codebase audited directly; patterns verified against 2025-2026 community sources)

---

## Context: What Already Exists (v1.0)

The following are **already built** and must NOT be re-implemented. They are dependency prerequisites, not new work.

| Existing Capability | Module |
|--------------------|--------|
| Discord Alerter with level routing and cooldown | `src/swingrl/monitoring/alerter.py` |
| Embed builders (trade fill, daily summary, circuit breaker) | `src/swingrl/monitoring/embeds.py` |
| Shadow model lifecycle state machine (Training → Shadow → Active → Archive) | `src/swingrl/shadow/lifecycle.py` |
| Shadow promoter with smoke test | `src/swingrl/shadow/promoter.py` |
| APScheduler jobs: equity_cycle, crypto_cycle, monthly_macro, monthly_offsite | `src/swingrl/scheduler/jobs.py` |
| train.py CLI — PPO/A2C/SAC training + Sharpe-weighted ensemble | `scripts/train.py` |
| docker-compose.prod.yml with swingrl + swingrl-dashboard services | `docker-compose.prod.yml` |
| deploy_model.sh, emergency_stop.py, disaster_recovery.py | `scripts/` |
| Healthchecks.io dead man's switch ping | `src/swingrl/scheduler/healthcheck_ping.py` |

**The v1.1 feature work is wire-up, integration, and documentation — not greenfield implementation.**

---

## Feature Landscape

### Table Stakes (Operator Expects These)

Features the operator assumes exist for the system to be usable. Missing = system cannot run hands-off.

| Feature | Why Expected | Complexity | Notes |
|---------|--------------|------------|-------|
| **Data ingestion run** — full historical depth with dimension alignment | Cannot train agents without aligned OHLCV + macro + feature data | MEDIUM | Existing pipeline.py + ingestion scripts exist; need end-to-end run on homelab with validation that all 156-dim / 45-dim observation columns populate without NaN |
| **Trained agents on homelab** — PPO/A2C/SAC for equity and crypto | System cannot trade without trained active/ models | MEDIUM | train.py CLI exists; need homelab CPU training run (estimated 4-8 hours equity + 2-4 hours crypto), backtest validation gates must pass |
| **Discord webhook channel wiring** — critical/warning/daily channels configured | Operator must know system is alive and what it did overnight | LOW | Alerter + embeds exist; need webhook URLs configured in .env and smoke-test each alert type fires correctly |
| **Production Docker stack running on homelab** — both services healthy | System must run 24/7 without developer machine | LOW | docker-compose.prod.yml exists; need first-time deploy verification: containers start, healthchecks pass, volumes mount, .env populated |
| **Retraining job in APScheduler** — equity monthly, crypto biweekly | Without retraining, model degrades as market regimes shift | HIGH | jobs.py has no retraining job yet; this is genuinely new code: trigger train.py, run smoke test, promote shadow if validation passes |
| **Automated shadow promotion** — new model replaces active after retraining | Completing the retraining loop without operator intervention | MEDIUM | Shadow lifecycle and promoter exist; new work is wiring retraining job output → shadow placement → auto-promotion with Discord notification |
| **Operator runbook** — step-by-step walkthroughs for all workflows | Solo operator must be able to recover from any failure without source archaeology | MEDIUM | Nothing exists; needs: initial setup, first training run, routine retraining, emergency stop, model rollback, disaster recovery, common failure diagnosis |

### Differentiators (Competitive Advantage)

Features that improve operational confidence beyond bare minimum. Not blocking, but high value for a capital-preservation-first system.

| Feature | Value Proposition | Complexity | Notes |
|---------|-------------------|------------|-------|
| **Retraining Discord notification** — start/finish/fail alert with Sharpe deltas | Operator knows retraining happened and whether new model is better or worse without logging in | LOW | New embed type needed; alerter infrastructure handles delivery; show old vs new Sharpe, decision (promoted/rejected) |
| **Model performance degradation alert** — warn when rolling Sharpe drops below threshold during paper trading | Early warning before model degrades to the point of risky decisions | MEDIUM | Requires rolling Sharpe calculation on live fills stored in SQLite; compare against baseline from training |
| **Retraining dry-run validation gate** — run walk-forward backtest on new model before promoting to shadow | Prevents deploying a worse model to shadow | MEDIUM | backtest.py already exists with 4 validation gates; new work is invoking it programmatically inside the retraining job and aborting promotion on gate failure |
| **6-month paper trading validation log** — structured record of daily P&L, drawdown, fill latency | Required evidence base before the operator considers any live deployment | LOW | Mostly automatic given existing fill recording; need a validation tracker document and periodic review checklist |
| **Startup validation sequence** — on container start, verify: DB tables exist, active models present, broker credentials reachable, Discord pingable | Fail fast with actionable error rather than silent degradation | LOW | Can be a pre-flight script called from container CMD; similar to existing healthcheck.py |

### Anti-Features (Commonly Requested, Often Problematic)

| Feature | Why Requested | Why Problematic | Alternative |
|---------|---------------|-----------------|-------------|
| **Continuous/online retraining** — retrain after every N trades | Appealing for "always fresh" model | RL agents need sufficient environment interactions before policy stabilizes; online retraining on swing trading frequency (~1-3 trades/day) produces models trained on statistical noise, not regimes | Scheduled monthly/biweekly retraining on 18-24 months of data; confirmed best practice for swing-frequency strategies |
| **Separate Docker container for retraining** — Kubernetes-style job container | Seems cleaner as microservice | Homelab has 64GB RAM / 4 cores; orchestrating a separate training container adds complexity (shared volume locking, job queuing) without benefit at this scale | APScheduler job in main container triggers retraining subprocess; simpler, proven, uses existing job context |
| **MLflow / experiment tracking server** — full MLOps platform | Professional-grade tooling | Adds a third service (PostgreSQL or SQLite backend, UI container); DuckDB model_runs table already stores training metadata; overhead not justified for single-operator system | Use existing DuckDB model_runs table for experiment history; add Sharpe delta column for comparison |
| **Slack notifications** — parallel alert channel | Operator might prefer Slack | Splits attention across two platforms; v1.0 already has Discord infrastructure (Alerter, embeds, two-webhook routing) hardened and tested | Discord only; one notification plane |
| **Automated live trading promotion** — auto-promote from paper to live when paper Sharpe > threshold | "Remove the human from the loop" | Capital at risk; silent promotion of a model with hidden overfitting could cause significant losses; shadow mode was explicitly designed to require human decision for live | Keep paper trading promotion as a deliberate manual operator action documented in the runbook |
| **Kubernetes or Docker Swarm** — container orchestration | "Production-grade" orchestration | Homelab is a single-node Intel i5 server; Kubernetes overhead (etcd, scheduler, API server) consumes ~1-2GB RAM and 1 CPU; provides no benefit without multi-node | Docker Compose with `restart: unless-stopped`; add Portainer or Komodo for UI if desired |

---

## Feature Dependencies

```
[Data Ingestion Run — full historical depth]
    └──required-before──> [Trained Agents on Homelab]
                              └──required-before──> [Production Docker Stack Running]
                                                        └──required-before──> [Retraining Job in APScheduler]

[Retraining Job in APScheduler]
    └──triggers──> [Automated Shadow Promotion]
                      └──fires──> [Retraining Discord Notification]

[Production Docker Stack Running]
    └──prerequisite-for──> [Discord Webhook Channel Wiring verification]
    └──prerequisite-for──> [6-Month Paper Trading Validation Log]

[Operator Runbook]
    └──documents-all-of──> [every feature above]
```

### Dependency Notes

- **Data ingestion required before training:** train.py reads directly from DuckDB market_data.ddb; if feature columns are misaligned or contain NaN, training will produce a degenerate model that passes smoke test but fails on live data.
- **Training required before Docker stack is useful:** Without active/ model files, the scheduler jobs will raise ModelError on first cycle; startup validation (differentiator) catches this, but retraining job cannot fix it — initial training must be manual.
- **Retraining job requires running stack:** The APScheduler retraining job lives inside the main container; homelab must be running for automated retraining to occur.
- **Shadow promotion enhances retraining:** The retraining job is only complete if it handles the promotion decision; a job that trains but requires manual file-copy is not hands-off.
- **Runbook documents everything:** The runbook has no code dependencies but is blocked by knowing the final behavior of all other features; write it last.

---

## MVP Definition

### Launch With (v1.1 — Operational Deployment)

Minimum required for the system to run hands-off in paper trading on the homelab.

- [ ] **Data ingestion completed** — all historical data ingested, dimensions aligned, feature columns validated non-NaN
- [ ] **Agents trained** — PPO/A2C/SAC models for equity and crypto passing walk-forward validation gates, placed in models/active/
- [ ] **Docker stack deployed** — homelab running both containers, healthchecks green, volumes mounted
- [ ] **Discord channels configured** — critical/warning/daily webhooks wired, smoke test each fires
- [ ] **Retraining job implemented** — APScheduler triggers equity monthly + crypto biweekly; invokes train.py subprocess, runs backtest gate, promotes to shadow on pass, alerts on result
- [ ] **Operator runbook** — covers: initial setup, first training walkthrough, routine retraining, emergency stop, model rollback, Discord test procedure, common failure diagnosis

### Add After Validation (v1.x — During Paper Trading Period)

Features to add once the system is running and generating 30+ days of paper data.

- [ ] **Model performance degradation alert** — trigger: 30 days of fill history exists to compute rolling Sharpe
- [ ] **Retraining dry-run validation gate** — trigger: first automated retraining cycle runs to reveal if promotion gate needs tightening
- [ ] **Startup validation sequence** — trigger: first container restart failure or silent startup failure observed

### Future Consideration (v2+)

- [ ] **Live trading promotion workflow** — defer until 6-month paper trading validation period completes with consistent positive Sharpe
- [ ] **SPX options environment** — requires IBKR account + $2K+ capital; explicitly out of scope for v1

---

## Feature Prioritization Matrix

| Feature | Operator Value | Implementation Cost | Priority |
|---------|---------------|---------------------|----------|
| Data ingestion run (full depth) | HIGH | MEDIUM | P1 |
| Agent training on homelab | HIGH | MEDIUM | P1 |
| Production Docker stack running | HIGH | LOW | P1 |
| Discord webhook channel wiring | HIGH | LOW | P1 |
| Retraining job (APScheduler) | HIGH | HIGH | P1 |
| Automated shadow promotion | HIGH | MEDIUM | P1 |
| Operator runbook | HIGH | MEDIUM | P1 |
| Retraining Discord notification | MEDIUM | LOW | P2 |
| Model performance degradation alert | MEDIUM | MEDIUM | P2 |
| Retraining dry-run validation gate | MEDIUM | MEDIUM | P2 |
| Startup validation sequence | MEDIUM | LOW | P2 |
| 6-month paper trading validation log | HIGH | LOW | P2 |
| Live trading promotion | HIGH (deferred) | HIGH | P3 |

**Priority key:**
- P1: Must have for v1.1 launch (hands-off paper trading)
- P2: Should have, add during paper trading period
- P3: Future milestone

---

## Retraining Pipeline — Specific Behavior (What "Table Stakes" Means)

Based on research into RL trading retraining practices and the existing SwingRL codebase:

### Trigger Schedule
- **Equity:** Monthly (first Sunday of the month, or configurable day; off-market hours)
- **Crypto:** Biweekly (every other Sunday; crypto runs 24/7 but avoid high-volatility periods)
- **Emergency:** Manual trigger via CLI or Discord command on model degradation alert

### Training Window
- Standard practice for swing trading: 18-24 months of historical data
- Existing walk-forward setup uses 200-bar purge gap between train and validation — preserve this
- Do NOT use all available history; recency bias toward recent regime outperforms long-history for swing strategies (MEDIUM confidence, consistent with multiple sources)

### Validation Gates (existing, must pass before promotion)
The existing backtest.py has 4 validation gates. The retraining job must invoke these programmatically and abort promotion if any gate fails. This is the safety layer.

### Promotion Decision Logic
```
Retraining job result:
  ALL 4 gates pass → place new model in models/shadow/{env}/
                   → run 6-point smoke test (lifecycle.py)
                   → if smoke test passes → promote to models/active/{env}/
                                          → archive old active model
                                          → send Discord: "Model promoted, Sharpe delta: +X.XX"
  ANY gate fails   → discard new model
                   → keep existing active model
                   → send Discord: "Retraining failed validation — keeping current model"
```

### What is NOT Needed
- MLflow or separate experiment tracking server (DuckDB model_runs table sufficient)
- Online/continuous retraining (confirmed anti-pattern at swing frequency)
- Human approval step (shadow + backtest gates provide sufficient safety for paper mode)

---

## Discord Alert Suite — What "Full Alert Suite" Means

The Alerter + embeds infrastructure exists. The gaps are:

### Already Implemented (v1.0)
- Trade fill embed (buy/sell with symbol, qty, fill price, stop, take profit)
- Daily digest (buffered info alerts flushed as single embed)
- Circuit breaker event embed
- Stuck agent warning
- Level routing: critical/warning → alerts channel, info/daily → daily channel

### New Embeds Needed for v1.1
| Embed Type | Trigger | Channel | Priority |
|-----------|---------|---------|----------|
| Retraining started | Retraining job begins | Daily | P1 |
| Retraining succeeded + promoted | All gates pass, model promoted | Alerts | P1 |
| Retraining failed validation | Gate failure; existing model retained | Alerts (warning) | P1 |
| Retraining error (exception) | Job raised exception | Alerts (critical) | P1 |

### Rate Limit Awareness
- Discord webhooks: 30 requests/minute per webhook URL (confirmed from Discord docs)
- Retraining runs at most 2x/month per env; rate limit irrelevant for retraining alerts
- Daily digest pattern (existing) is the correct pattern for trade-frequency info alerts
- Use `discord-webhook` library with `rate_limit_retry=True` (already used in alerter.py)

---

## Docker Production Deployment — What "Table Stakes" Means

### What docker-compose.prod.yml Already Provides (v1.0)
- Two services: swingrl (main) + swingrl-dashboard (Streamlit, read-only DB)
- Resource caps: 2.5GB RAM / 1 CPU (main), 512MB / 0.5 CPU (dashboard)
- `restart: unless-stopped`
- Named bind mounts for data/, db/, models/, logs/, config/
- env_file: .env for secrets

### Gaps for First Production Deployment
- **Healthcheck definitions** — neither service has a HEALTHCHECK stanza; Docker cannot distinguish healthy from unhealthy containers; add `test: ["CMD", "python", "scripts/healthcheck.py"]` to main service (healthcheck.py already exists)
- **depends_on with condition: service_healthy** — dashboard should wait for main to be healthy before starting
- **First-time volume initialization** — models/, db/ will be empty on first deploy; startup validation must detect this and guide operator to run initial training before scheduler fires

### What Is NOT Needed
- Traefik or reverse proxy — Streamlit is accessed directly on :8501 from LAN; no public internet exposure
- Portainer or Komodo — single docker-compose.prod.yml file; operator manages via SSH; UI adds complexity not value at this scale
- Docker Swarm or multi-node orchestration — single-node homelab

---

## Operator Runbook — What It Must Cover

A runbook exists nowhere in the repository. This is the single largest documentation gap. Based on MLOps practices for trading systems, a complete runbook must cover:

| Section | Contents | Why Critical |
|---------|----------|-------------|
| Initial setup | Clone repo, configure .env, create Discord webhooks, run init_db.py, seed_production.py | Without this, operator cannot do first deploy |
| Data ingestion walkthrough | Run ingestion scripts, verify row counts, validate feature dimensions, check for NaN | Training on misaligned data produces silent failure |
| First training run | Run train.py for equity then crypto, verify models/active/ populated, run smoke test | Manual first-time; subsequent runs are automated |
| Docker deployment | First-deploy checklist, verify containers healthy, verify scheduler fired | System is not running until verified |
| Discord verification | Test each webhook with send_test_alert, verify correct channel routing | Silent alerting failure is worse than no alerting |
| Automated retraining | What to expect, how to confirm it ran, what to do if it fails | Operator must know retraining is a background event |
| Emergency stop | Four tiers (scheduler halt → position flatten → full stop → kill switch), when to use each | Capital preservation requires fast, correct response |
| Model rollback | How to restore archived model when active model performs poorly | Archive/ directory exists; procedure must be documented |
| Common failures | Broker API down, circuit breaker stuck, scheduler not firing, Discord silent, disk full | Operator cannot do production archaeology in a crisis |
| Disaster recovery | disaster_recovery.py procedure, offsite backup restore, full system rebuild | Hardware failure recovery path must be pre-written |

---

## Sources

- [AI Model Drift & Retraining Guide](https://smartdev.com/ai-model-drift-retraining-a-guide-for-ml-system-maintenance/) — retraining trigger and window sizing patterns (MEDIUM confidence)
- [Model Drift in Production 2026 Runbook](https://alldaystech.com/guides/artificial-intelligence/model-drift-detection-monitoring-response) — drift → retrain → shadow promotion pipeline (MEDIUM confidence)
- [Walk Forward Optimization — therobusttrader.com](https://therobusttrader.com/walk-forward-analysis-testing-optimization-wfa/) — 18-24 month window recommendation for swing trading (MEDIUM confidence)
- [Discord Webhooks Complete Guide 2025](https://inventivehq.com/blog/discord-webhooks-guide) — embed format limits, rate limits (HIGH confidence — matches Discord official docs)
- [Discord Rate Limits — birdie0 guide](https://birdie0.github.io/discord-webhooks-guide/other/rate_limits.html) — 30 req/min per webhook URL (HIGH confidence)
- [Docker for Trading Applications — DEV Community](https://dev.to/propfirmkey/docker-for-trading-applications-a-practical-setup-guide-1n41) — healthcheck and restart policy patterns (MEDIUM confidence)
- [Docker Compose Restart Policies and Healthchecks 2025](https://blog.justanotheruptime.com/posts/2025_07_07_docker_compose_restart_policies_and_healthchecks/) — `unless-stopped` + healthcheck interaction (HIGH confidence)
- [MLOps Best Practices for Quantitative Trading Teams](https://medium.com/@online-inference/mlops-best-practices-for-quantitative-trading-teams-59f063d3aaf8) — model registry, continuous training, monitoring (MEDIUM confidence)
- [Mastering MLOps for Trading Bots — Luxoft](https://www.luxoft.com/blog/mastering-mlops-for-trading-bots) — CI/CD/CT/CM framework for trading (MEDIUM confidence)
- Direct codebase audit: `src/swingrl/monitoring/`, `src/swingrl/shadow/`, `src/swingrl/scheduler/`, `scripts/train.py`, `docker-compose.prod.yml` (HIGH confidence — primary source)

---

*Feature research for: SwingRL v1.1 Operational Deployment*
*Researched: 2026-03-10*
