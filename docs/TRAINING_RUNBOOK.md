# SwingRL Training Runbook

Operational guide for building, deploying, running, and monitoring training on the homelab.

## Infrastructure

| Component | Container | Port | Purpose |
|-----------|-----------|------|---------|
| Training + Scheduler | `swingrl` | — | APScheduler (paper trading) + training pipeline |
| Memory Service | `swingrl-memory` | 8889 | FastAPI for memories, consolidation, epoch advice |
| Ollama | `swingrl-ollama` | 11435 | Epoch advice fallback (qwen3:1.7b) |
| Dashboard | `swingrl-dashboard` | 8501 | Streamlit monitoring dashboard |
| PostgreSQL | `pg16` | 5432 | All data storage (OHLCV, features, results, memories) |

## Docker Image Build

### Production image (includes training deps)

```bash
cd ~/swingrl
docker compose build --no-cache swingrl
```

### CI image (includes test + lint deps)

```bash
docker compose -f docker-compose-dev.yml build --no-cache
```

### Memory service image

```bash
docker compose build --no-cache swingrl-memory
```

## Stack Management

### Start full stack

```bash
cd ~/swingrl
docker compose up -d
```

### Stop full stack

```bash
docker compose down
```

### Restart single service

```bash
docker compose restart swingrl
```

### View logs

```bash
docker logs swingrl --tail 50
docker logs swingrl-memory --tail 50
docker logs pg16 --tail 50
```

## Running Training

### CLI Arguments

| Flag | Default | Description |
|------|---------|-------------|
| `--env` | `all` | `equity`, `crypto`, or `all` (sequential) |
| `--iterations` | `0` | 0 = baseline only. N > 0 = iter 0 (baseline, memory off) + N iters with memory on |
| `--force` | `False` | Ignore checkpoints, retrain everything |
| `--skip-ingest` | `False` | Skip data ingestion before each iteration |
| `--config` | `config/swingrl.yaml` | Config YAML path |
| `--models-dir` | `models` | Model storage root |
| `--state-path` | `data/training_state.json` | Checkpoint file for resume |
| `--report` | `data/training_report.json` | Output report path |
| `--comparison-path` | `data/training_comparison.json` | Best-model comparison output |

### Common Commands

```bash
# Full run: baseline + 5 memory-enhanced iterations (detached)
docker exec -d swingrl python scripts/train_pipeline.py --env all --iterations 5

# Equity only, skip data ingestion
docker exec -d swingrl python scripts/train_pipeline.py --env equity --iterations 5 --skip-ingest

# Force retrain from scratch (ignores checkpoints)
docker exec -d swingrl python scripts/train_pipeline.py --env all --iterations 5 --force

# Single baseline run (no memory iterations)
docker exec -d swingrl python scripts/train_pipeline.py --env all

# Resume from checkpoint (default behavior)
docker exec -d swingrl python scripts/train_pipeline.py --env all --iterations 5
```

### Data Ingestion Only (no training)

```bash
# Incremental (new data since last run)
docker exec swingrl python -m swingrl.data.ingest_all

# Full backfill (re-fetch all history from APIs)
docker exec swingrl python -m swingrl.data.ingest_all --backfill
```

## Monitoring Training

### Live log follow

```bash
docker exec swingrl tail -f logs/training_iter5.log
```

### Track fold completions

```bash
docker exec swingrl tail -f logs/training_iter5.log | grep -E "fold_complete|iteration_complete|gate_passed|env_failed"
```

### Track errors only

```bash
docker exec swingrl tail -f logs/training_iter5.log | grep '"level": "error"'
```

### Check if training process is running

```bash
docker top swingrl
```

### Quick snapshot (last 50 lines)

```bash
docker exec swingrl tail -50 logs/training_iter5.log
```

## Verifying Data During Training

### Data Write Map

Training writes data at four granularities: per-epoch, per-fold, per-environment, and per-iteration.

#### Per-Epoch (during each fold's training)

| Table | Cadence | Data |
|-------|---------|------|
| `training_epochs` | PPO: every 60 epochs, A2C: every 8000, SAC: every 40000 | Rolling sharpe/mdd, reward weights, policy loss, notable events |
| `reward_adjustments` | When LLM suggests weight change | Weight before/after, trigger metric, then outcome 10 epochs later |
| Memory service (`/ingest`) | Same as training_epochs | Epoch snapshots for memory accumulation |
| Memory service (`/training/epoch_advice`) | Same cadence | LLM epoch advice request + response |

#### Per-Fold (after each WF fold completes)

| Table | Data |
|-------|------|
| `backtest_results` | OOS sharpe/mdd/sortino/calmar, win_rate, profit_factor, total_trades, overfitting gap, regime context, date ranges, convergence info |

#### Per-Environment (after all folds for equity or crypto)

| Table | Data |
|-------|------|
| `iteration_results` | Ensemble sharpe/mdd, per-algo weights, per-algo mean metrics, gate_passed, HP source |
| `model_metadata` | Model path, ensemble weight, timesteps (after final deployment training) |
| Memory service | Walk-forward results, trading patterns, run summaries |

#### Per-Iteration (after both envs complete)

| Table/File | Data |
|------------|------|
| Memory service: `cross_iteration:{env}` | VS baseline/previous/best sharpe deltas (iter 1+ only) |
| Memory service: consolidation | LLM consolidation of new memories into patterns |
| `data/training_state.json` | Checkpoint for resume-on-crash |
| `data/training_comparison.json` | Best models per algo x env |

### Verification Queries

```bash
# Fold results for current iteration
docker exec pg16 psql -U swingrl -d swingrl -c "
  SELECT algorithm, COUNT(*) as folds,
    ROUND(AVG(sharpe)::numeric, 3) as avg_sharpe,
    ROUND(AVG(mdd)::numeric, 4) as avg_mdd
  FROM backtest_results
  WHERE iteration_number = 5
  GROUP BY algorithm ORDER BY algorithm;"

# Training epochs being captured
docker exec pg16 psql -U swingrl -d swingrl -c "
  SELECT algo, env, COUNT(*) as epochs,
    ROUND(AVG(rolling_sharpe)::numeric, 3) as avg_sharpe
  FROM training_epochs
  GROUP BY algo, env ORDER BY algo, env;"

# Reward adjustments
docker exec pg16 psql -U swingrl -d swingrl -c "
  SELECT algo, env, COUNT(*) as adjustments,
    COUNT(*) FILTER (WHERE effective = true) as effective
  FROM reward_adjustments
  GROUP BY algo, env ORDER BY algo, env;"

# Iteration results (ensemble metrics)
docker exec pg16 psql -U swingrl -d swingrl -c "
  SELECT iteration_number, environment,
    ROUND(ensemble_sharpe::numeric, 3) as sharpe,
    ROUND(ensemble_mdd::numeric, 4) as mdd,
    gate_passed
  FROM iteration_results
  ORDER BY iteration_number, environment;"

# Memory ingestion count (new memories since migration)
docker exec pg16 psql -U swingrl -d swingrl -c "
  SELECT COUNT(*) as new_memories
  FROM memories WHERE id > 7904649;"

# Consolidation patterns
docker exec pg16 psql -U swingrl -d swingrl -c "
  SELECT status, COUNT(*) as patterns
  FROM consolidations GROUP BY status;"

# Model metadata
docker exec pg16 psql -U swingrl -d swingrl -c "
  SELECT model_id, environment, algorithm,
    ROUND(ensemble_weight::numeric, 3) as weight
  FROM model_metadata
  ORDER BY created_at DESC LIMIT 10;"

# Overall table row counts
docker exec pg16 psql -U swingrl -d swingrl -c "
  SELECT 'backtest_results' as tbl, COUNT(*) FROM backtest_results
  UNION ALL SELECT 'iteration_results', COUNT(*) FROM iteration_results
  UNION ALL SELECT 'training_epochs', COUNT(*) FROM training_epochs
  UNION ALL SELECT 'reward_adjustments', COUNT(*) FROM reward_adjustments
  UNION ALL SELECT 'model_metadata', COUNT(*) FROM model_metadata
  UNION ALL SELECT 'memories', COUNT(*) FROM memories
  UNION ALL SELECT 'consolidations', COUNT(*) FROM consolidations
  ORDER BY 1;"
```

## CI Pipeline

CI creates an isolated `swingrl_test` database — production data is never touched.

```bash
# Run CI (from local Mac)
ssh homelab "cd ~/swingrl && bash scripts/ci-homelab.sh --no-cache"

# Cached build (faster, for code-only changes)
ssh homelab "cd ~/swingrl && bash scripts/ci-homelab.sh"
```

CI stages:
1. Git pull
2. Docker build (CI + memory service)
3. Create `swingrl_test` database
4. Run tests (against `swingrl_test`)
5. Lint + type check (ruff, mypy)
6. Memory service lint
7. Cleanup (drop `swingrl_test`, prune images)

## Database Operations

### Check PostgreSQL connectivity

```bash
docker exec pg16 psql -U swingrl -d swingrl -c "SELECT 1;"
```

### Re-migrate from DuckDB + SQLite (disaster recovery)

```bash
# Wipe and recreate (requires user to run DROP DATABASE manually)
# Then:
docker exec -e DUCKDB_PATH=data/db/market_data.ddb \
  -e SQLITE_PATH=data/db/trading_ops.db \
  -e MEMORY_DB_PATH=db/memory.db \
  swingrl python scripts/migrate_to_postgres.py
```

Source databases (preserved on homelab):
- `data/db/market_data.ddb` — OHLCV, features, backtest results, iteration results
- `data/db/trading_ops.db` — Trades, positions, risk decisions
- `db/memory.db` — Memories, consolidations, patterns

### Check database sizes

```bash
docker exec pg16 psql -U swingrl -d swingrl -c "
  SELECT relname, n_live_tup as rows
  FROM pg_stat_user_tables
  WHERE n_live_tup > 0
  ORDER BY n_live_tup DESC;"
```

## Training State Management

### View checkpoint state

```bash
cat ~/swingrl/data/training_state.json | python3 -m json.tool | head -20
```

### Check which iterations are completed

```bash
python3 -c "import json; s=json.load(open('data/training_state.json')); print('Completed:', s.get('completed_iterations', []))"
```

### Remove a failed iteration from checkpoints

```python
import json
with open("data/training_state.json") as f:
    state = json.load(f)
state["completed_iterations"].remove(5)  # remove iter 5
state.pop("current_iteration", None)
state.pop("iteration_5_env_equity", None)
state.pop("iteration_5_env_crypto", None)
with open("data/training_state.json", "w") as f:
    json.dump(state, f, indent=2)
```

## Troubleshooting

### Training fails with "No data found in features_equity table"

Features not computed. Run ingestion:
```bash
docker exec swingrl python -m swingrl.data.ingest_all --backfill
```

### Alpaca "end should not be before start"

Data is already current (market closed or today's bar not available yet). The completed-bar guard handles this — it's a no-op, not an error.

### FRED CPIAUCSL stale data error

CPI is monthly with 45-65 day release lag. Threshold is set to 95 days. If it still fails, macro ingestion is non-fatal — training continues with existing macro data.

### PostgreSQL "duplicate key value violates unique constraint"

Identity sequences may be behind migrated data. Reset them:
```bash
docker exec pg16 psql -U swingrl -d swingrl -c "
  SELECT setval(pg_get_serial_sequence('TABLE_NAME', 'id'),
    (SELECT MAX(id) FROM TABLE_NAME));"
```

### Container restart loop

Check logs: `docker logs swingrl --tail 20`. Common cause: wrong entrypoint or missing module.

### Binance Global 451 errors during gap fill

Binance Global blocks US IPs. Crypto gaps from 2018-2019 are known and unfillable. Training handles them via episode splitting.
