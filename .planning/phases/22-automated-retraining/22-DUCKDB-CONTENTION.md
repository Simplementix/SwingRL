# Phase 22: DuckDB Contention Issues & Resolutions

**Discovered:** 2026-03-16 during initial training parallelization
**Status:** Needs resolution in Phase 22 implementation

## Problem

DuckDB enforces a **single-writer lock** on the database file. Only one process can hold a write connection to `market_data.ddb` at a time. This creates conflicts when:

1. **Training container vs main service** (observed today): `docker compose up -d` starts `swingrl-1` (main.py) which opens a persistent DuckDB connection. `docker compose run` creates a second container for training that also needs DuckDB. Second container gets `IOException: Conflicting lock`.

2. **Retrain subprocess vs live trading** (Phase 22 scenario): Automated retraining runs as a subprocess inside the main `swingrl` container. `main.py` holds a persistent DuckDB connection for live trading. Retrain subprocess needs DuckDB to load features and write results. Both in the same container.

3. **Parallel algo workers vs parent process**: `ProcessPoolExecutor` workers are forked from the parent. DuckDB connections cannot be shared across process boundaries. Workers must NOT inherit parent's DuckDB connection.

## Current Mitigations (Initial Training)

- Workers use `db=None` (no DuckDB access in subprocesses)
- Parent process handles all DuckDB reads (feature loading) and writes (model metadata) sequentially
- Config serialized as `config.model_dump()` dict to avoid unpicklable `_ConfigWithYaml` local class
- Only start dependencies (`swingrl-ollama`, `swingrl-memory`), NOT the main service during training

## Phase 22 Resolution Options

### Option A: Short-lived connections in main.py (Recommended)
- Change `main.py` from persistent DuckDB connection to open/close per operation
- Each trading cycle: open DuckDB -> read -> close -> trade -> open -> write -> close
- Retrain subprocess does the same: open -> read features -> close -> train -> open -> write results -> close
- **Pro:** Simple, no architectural change. DuckDB lock held only milliseconds per operation.
- **Con:** Slight overhead from repeated connection open/close (~5ms each, negligible for trading cycles)

### Option B: Read-only for training reads
- Retrain subprocess opens `duckdb.connect(path, read_only=True)` for feature loading
- DuckDB allows multiple concurrent readers
- Write connection opened briefly only for deferred result writes (after training completes)
- **Pro:** No lock conflict during the long training phase (hours)
- **Con:** Still conflicts during the brief write phase; need coordination

### Option C: Combination (A + B)
- main.py uses short-lived connections (Option A)
- Retrain subprocess uses `read_only=True` for feature loading (Option B)
- Retrain defers all writes to after training completes, opens brief write connection
- **Pro:** Maximum concurrency, minimal lock contention
- **Con:** Most code changes

## Parallel Training Architecture (Current Implementation)

```
Main Process (train_pipeline.py)
  |
  |-- Opens DuckDB (read features, close immediately)
  |
  |-- ProcessPoolExecutor(max_workers=3)
  |     |-- Worker PPO (db=None, SubprocVecEnv fork, 6 envs)
  |     |-- Worker A2C (db=None, SubprocVecEnv fork, 6 envs)
  |     |-- Worker SAC (db=None, SubprocVecEnv fork, 6 envs)
  |
  |-- Collect results in-memory
  |-- Open DuckDB (write model metadata, close)
```

## Related Config Serialization Issue

`load_config()` creates a local class `_ConfigWithYaml` that can't be serialized by `ProcessPoolExecutor`. Fix: serialize config as `config.model_dump()` dict in parent, reconstruct via `SwingRLConfig(**config_dict)` in each worker. See `_reconstruct_config()` in `train_pipeline.py`.

## SubprocVecEnv Start Method

- `forkserver`: Broken pipe errors in Docker (tested, failed)
- `spawn`: Fails on `cv2/libGL.so.1` missing in `python:3.11-slim` (tested, failed)
- `fork`: Works in Docker on Linux, CPU-only (no CUDA concerns). **Current choice.**

## Files Modified

- `src/swingrl/config/schema.py` -- Added `n_envs=6`, `vecenv_backend="subproc"` to TrainingConfig
- `src/swingrl/training/trainer.py` -- SubprocVecEnv(fork) + torch.set_num_threads()
- `scripts/train_pipeline.py` -- Algo-level ProcessPoolExecutor, deferred DuckDB writes, config dict serialization
- `docker-compose.prod.yml` -- swingrl bumped to 16GB/8CPU, container_name added
- `config/swingrl.yaml` -- n_envs=6, vecenv_backend=subproc

## Operational Notes

- During initial training: only start `swingrl-ollama` + `swingrl-memory`, NOT `swingrl` main service
- Training runs via: `docker compose run -d --name swingrl-training swingrl /app/.venv/bin/python scripts/train_pipeline.py --env all --iterations 5 --force`
- Monitor: `docker stats swingrl-training` (expect ~800% CPU with 3 algo workers x 6 envs)
- Check progress: `docker logs swingrl-training 2>&1 | grep -E '(fold_complete|training_complete|iteration_start)'`

---
*Discovered during Phase 19.1 training, documented for Phase 22 implementation*
