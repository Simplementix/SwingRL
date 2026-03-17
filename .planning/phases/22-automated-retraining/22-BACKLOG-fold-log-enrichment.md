# Backlog: Enrich fold_complete log with algo_name and env_name

**Priority:** Include in Phase 22 execution
**Effort:** Small (one line change)
**Why:** fold_complete logs currently don't identify which algo or environment produced the result. When 3 algos run walk-forward in parallel across 2 environments, it's impossible to attribute a fold result to a specific algo/env pair without cross-referencing timestamps with training_complete logs.

## Current State

`src/swingrl/agents/backtest.py` — the `fold_complete` log in `WalkForwardBacktester.run()` (~line 378):

```python
log.info(
    "fold_complete",
    fold=fold_idx,
    gate_passed=gate.passed,
    failures=gate.failures,
    oos_sharpe=round(oos_metrics.get("sharpe", 0.0), 4),
    oos_mdd=round(oos_metrics.get("mdd", 0.0), 4),
    oos_profit_factor=round(oos_metrics.get("profit_factor", 0.0), 4),
    overfit_gap=round(overfit.get("gap", 0.0), 4),
    overfit_class=overfit.get("classification"),
    total_trades=int(oos_metrics.get("total_trades", 0)),
    win_rate=round(oos_metrics.get("win_rate", 0.0), 4),
)
```

Missing: `algo_name` and `env_name` — both are available in scope (passed as args to `run()`).

## Change Required

Add `algo_name` and `env_name` to the log call:

```python
log.info(
    "fold_complete",
    env_name=env_name,
    algo_name=algo_name,
    fold=fold_idx,
    gate_passed=gate.passed,
    failures=gate.failures,
    oos_sharpe=round(oos_metrics.get("sharpe", 0.0), 4),
    oos_mdd=round(oos_metrics.get("mdd", 0.0), 4),
    oos_profit_factor=round(oos_metrics.get("profit_factor", 0.0), 4),
    overfit_gap=round(overfit.get("gap", 0.0), 4),
    overfit_class=overfit.get("classification"),
    total_trades=int(oos_metrics.get("total_trades", 0)),
    win_rate=round(oos_metrics.get("win_rate", 0.0), 4),
)
```

## File

`src/swingrl/agents/backtest.py` — single log statement in `WalkForwardBacktester.run()`, around line 378.

## Result

Monitoring command output changes from:
```
fold_complete  fold=0  gate_passed=False  oos_sharpe=0.6795 ...
```
To:
```
fold_complete  env_name=equity  algo_name=a2c  fold=0  gate_passed=False  oos_sharpe=0.6795 ...
```

Enables filtering by algo (`grep algo_name=ppo`) and env (`grep env_name=crypto`) when reviewing training runs.
