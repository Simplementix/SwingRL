# Milestones

## v1.0 MVP (Shipped: 2026-03-11)

**Phases:** 17 | **Plans:** 49 | **Requirements:** 74/74
**Timeline:** 8 days (2026-03-03 → 2026-03-10) | **Commits:** 325
**Code:** 15,288 LOC src/ + 18,028 LOC tests/ = 33,316 Python LOC
**Git range:** Phase 1 (Dev Foundation) → Phase 17 (Doc Housekeeping)

**Delivered:** Complete RL swing trading system with PPO/A2C/SAC ensemble agents for equity (8 ETFs) and crypto (BTC/ETH), paper trading execution pipeline, automated scheduling, and production hardening — ready for homelab deployment.

**Key accomplishments:**
1. Reproducible dev environment with Docker CI on x86 homelab, Pydantic v2 config schema
2. Multi-source data pipeline: Alpaca equities, Binance.US crypto, FRED macro with 12-step validation
3. Feature engineering: 156-dim equity and 45-dim crypto observation vectors with HMM regime detection
4. Gymnasium RL environments and PPO/A2C/SAC Sharpe-weighted ensemble with walk-forward validation
5. Full paper trading: 5-stage execution middleware, two-tier risk veto, circuit breakers, bracket orders
6. Production hardening: APScheduler automation, Discord alerting, backup pipeline, shadow mode, emergency stop

**Tech Debt (non-blocking):**
- pipeline.py sets fundamental columns to 0.0 (live FundamentalFetcher integration deferred)
- Placeholder Sharpe ratios in train.py for initial ensemble weights (by design)

**Archives:** `.planning/milestones/v1.0-ROADMAP.md`, `.planning/milestones/v1.0-REQUIREMENTS.md`

---

