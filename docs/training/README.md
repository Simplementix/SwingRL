# Training Documentation

Living reference cards for SwingRL training. Thin, scannable, source-of-truth for values and invariants. Update when the referenced code changes.

## Conventions

- Each doc is a reference card, not a tutorial.
- Every doc includes **Configurable values (yaml)** + **Hardcoded values** sections — so "can I change X from yaml?" is always answerable from the doc alone.
- "Source of truth" section at the bottom of every doc lists authoritative files.
- Update the **Changelog** block in each doc when you touch it.

## Documents

### Tier 1 — Foundation

- [x] [rl-environments.md](rl-environments.md) — Obs/action/reward/episode per env
- [x] [feature-catalog.md](feature-catalog.md) — Every feature: source, dtype, range, normalization, which env consumes it
- [ ] reward-shaping.md — Reward formula, penalties, memory-driven adjustments, yaml weights
- [ ] agent-architecture.md — PPO/A2C/SAC configs, ensemble weights, epoch cadence

### Tier 2 — Training loop

- [ ] training-data-capture.md — Per pg16 table: who writes, when, cardinality, readers
- [ ] memory-system.md — Memory types, consolidation phases, LLM chain, retrieval, known issues
- [ ] training-pipeline.md — Iteration lifecycle, walk-forward structure, entry points
- [ ] validation-promotion.md — Sharpe/MDD gates, shadow → active → archive flow

### Tier 3 — Deferred

- [ ] hyperparameter-tuning.md — LLM-driven HP selection, bounds, history (folds `.planning/research/hp-tuning-reference.md`)

## Related

- `docs/TRAINING_RUNBOOK.md` — Operational (build/deploy/run/monitor)
- `.planning/ROADMAP.md` — Project roadmap & phase status
- `.planning/research/` — Phase research (kept separate from these reference cards)
