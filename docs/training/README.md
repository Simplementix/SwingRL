# Training Documentation

Living reference cards for SwingRL training. Thin, scannable, source-of-truth for values and invariants. Update when the referenced code changes.

## Conventions

- Each doc is a reference card, not a tutorial.
- Every doc includes **Configurable values (yaml)** + **Hardcoded values** sections — so "can I change X from yaml?" is always answerable from the doc alone.
- "Source of truth" section at the bottom of every doc lists authoritative files.
- Update the **Changelog** block in each doc when you touch it.
- When shipping a new doc: flip the checkbox **and** append the short commit SHA in the same edit. This file is the single source of truth for series progress — no separate handoff memory.

## Documents

### Tier 1 — Foundation

- [x] [rl-environments.md](rl-environments.md) `b811dc1` — Obs/action/reward/episode per env
- [x] [feature-catalog.md](feature-catalog.md) `60ac088` — Every feature: source, dtype, range, normalization, which env consumes it
- [x] [reward-shaping.md](reward-shaping.md) `de0ed8c` — Reward formula, penalties, memory-driven adjustments, yaml weights
- [x] [agent-architecture.md](agent-architecture.md) `17caae0` — PPO/A2C/SAC configs, ensemble weights, epoch cadence

### Tier 2 — Training loop

- [x] [training-data-capture.md](training-data-capture.md) `b297636` — Per pg16 table: who writes, when, cardinality, readers
- [x] [memory-system.md](memory-system.md) `ef35984` — Memory types, consolidation phases, LLM chain, retrieval, known issues
- [x] [training-pipeline.md](training-pipeline.md) `32d38e8` — Iteration lifecycle, walk-forward structure, entry points
- [x] [validation-promotion.md](validation-promotion.md) `b980a88` — Sharpe/MDD gates, shadow → active → archive flow

### Tier 3

- [x] [hyperparameter-tuning.md](hyperparameter-tuning.md) `PENDING` — LLM-driven HP selection, bounds, history (folds `.planning/research/hp-tuning-reference.md`)

## Related

- `docs/TRAINING_RUNBOOK.md` — Operational (build/deploy/run/monitor)
- `.planning/ROADMAP.md` — Project roadmap & phase status
- `.planning/research/` — Phase research (kept separate from these reference cards)
