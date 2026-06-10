# Training Documentation

Living reference cards for SwingRL training. Thin, scannable, source-of-truth for values and invariants. Update when the referenced code changes.

## Conventions

- Each doc is a reference card, not a tutorial.
- Every doc includes **Configurable values (yaml)** + **Hardcoded values** sections — so "can I change X from yaml?" is always answerable from the doc alone.
- "Source of truth" section at the bottom of every doc lists authoritative files.
- Update the **Changelog** block in each doc when you touch it.
- When shipping a new doc: flip the checkbox **and** append the short commit SHA in the same edit. This file is the single source of truth for series progress — no separate handoff memory.
- **Deprecating a doc:** If a subsystem is removed or merged into another doc, prefix the entry with `~~strikethrough~~` for one milestone, then delete the entry in the milestone-closeout pass. Do not delete the file itself until at least one minor version has passed — historical links may still reference it.
- **Staleness check:** Run `git log --since="3 months ago" -- docs/training/` quarterly. Any doc with no commits while its referenced code did change is a staleness candidate. The Changelog block in each doc is the primary signal.

## Documents

### Tier 1 — Foundation

- [x] [rl-environments.md](rl-environments.md) `b811dc1` — Obs/action/reward/episode per env
- [x] [feature-catalog.md](feature-catalog.md) `60ac088` — Every feature: source, dtype, range, normalization, which env consumes it
- [x] [reward-shaping.md](reward-shaping.md) `5707cd8` — Reward formula, penalties, memory-driven adjustments, yaml weights
- [x] [agent-architecture.md](agent-architecture.md) `5707cd8` — PPO/A2C/SAC configs, ensemble weights, epoch cadence

### Tier 2 — Training loop

- [x] [training-data-capture.md](training-data-capture.md) `5707cd8` — Per pg16 table: who writes, when, cardinality, readers
- [x] [memory-system.md](memory-system.md) `ef35984` — Memory types, consolidation phases, LLM chain, retrieval, known issues
- [x] [training-pipeline.md](training-pipeline.md) `32d38e8` — Iteration lifecycle, walk-forward structure, entry points
- [x] [validation-promotion.md](validation-promotion.md) `b980a88` — Sharpe/MDD gates, shadow → active → archive flow

### Tier 3

- [x] [hyperparameter-tuning.md](hyperparameter-tuning.md) `5707cd8` — LLM-driven HP selection, bounds, history (folds `.planning/research/hp-tuning-reference.md`)

## Cross-cutting topics

Some subsystems are documented across multiple reference cards rather than in a single doc. Use these pointers:

- **Iteration reporting & regression detection** — Lifecycle covered across [`validation-promotion.md`](validation-promotion.md) (CPS gates, regression deltas), [`training-data-capture.md`](training-data-capture.md) (`iteration_results` table), and [`agent-architecture.md`](agent-architecture.md) (walk-forward backtester). Code: `src/swingrl/reporting/iteration_report.py`.
- **LLM provider chain** — Provider inventory, routing per call_type, 429/calendar-day backoff, audit logging: [`memory-system.md`](memory-system.md) "LLM provider chain" section. HP-tuning-specific provider behavior: [`hyperparameter-tuning.md`](hyperparameter-tuning.md).
- **Epoch advisor** — Cadence, trigger thresholds (KL=0.10, MDD=-25.0), 9-gate guardrail chain: [`reward-shaping.md`](reward-shaping.md) "Memory-driven weight adjustments" section. Endpoint contract: [`memory-system.md`](memory-system.md) "/training/epoch_advice" section.
- **Training orchestration & resume** — `training_state.json`, atomic writes, per-env retry-once, multi-iteration resume: [`training-pipeline.md`](training-pipeline.md) "Iteration counter & resume" section. Production trading scheduler (12 cron jobs) is operational scope — see `docs/TRAINING_RUNBOOK.md`.
- **Risk & safety controls** — Circuit breaker, emergency halt, position limits, wash-sale tracker are all **live-trading concerns** with zero training-loop interaction. See `docs/TRAINING_RUNBOOK.md`. Training-touch points (shadow-promotion gating) are documented in [`agent-architecture.md`](agent-architecture.md) and [`validation-promotion.md`](validation-promotion.md).

## Related

- `docs/TRAINING_RUNBOOK.md` — Operational (build/deploy/run/monitor)
- `.planning/ROADMAP.md` — Project roadmap & phase status
- `.planning/research/` — Phase research (kept separate from these reference cards)
