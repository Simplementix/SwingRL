# Phase 19: Model Training - Context

**Gathered:** 2026-03-13
**Status:** Ready for planning (updated — memory agent integration)

<domain>
## Phase Boundary

Train PPO/A2C/SAC agents for equity and crypto on homelab CPU with an LLM-powered meta-training loop adapted from Google's Always-On Memory Agent pattern. The meta-trainer controls hyperparameters, reward shaping, curriculum selection, and early stopping — with all LLM outputs hard-clamped via bounds.py before touching the trainer. Store training data in memory (SQLite) and structured metrics (DuckDB) to support future live trading decisions. Expand HMM from 2 to 3 regimes (bull/bear/crisis). Validate via walk-forward, pass the training success gate (ensemble Sharpe > 1.0, MDD < 15%), and deploy models to models/active/{env}/{algo}/. Automated retraining, Discord alerts, and production deployment are separate phases.

</domain>

<decisions>
## Implementation Decisions

### Memory Agent Architecture
- Custom FastAPI service — no LangChain, no framework dependencies. Direct Ollama API calls with tool definitions
- 3 separate focused agents (Google's pattern): IngestAgent, ConsolidateAgent, QueryAgent — each with its own system prompt
- Runs as `swingrl-memory` Docker container on port 8889
- HTTP-only ingestion via POST /ingest — no inbox file watcher
- Emergency stop (emergency_stop.py, reset_halt.py) bypasses memory entirely — do not touch

### Storage Architecture
- **SQLite (memory.db)**: memories table, consolidations table, archived_memories table — LLM-readable text for pattern detection
- **DuckDB (market_data.db)**: training_epochs, meta_decisions, reward_adjustments tables + ALTER TABLE training_runs — structured metrics for SQL queries
- Two-tier retrieval: consolidations first (compact patterns), then targeted raw memories by ID when detail needed
- No embeddings, no vector search — SQL filtering by source tag + recency, LLM does the reasoning
- Plain text key=value ingestion format (not JSON) — designed for LLM consumption

### Memory Lifecycle
- Consolidation: 30-min background schedule + explicit POST /consolidate after each training run
- Archival: raw memories archived (moved to archived_memories table, not deleted) 90 days after consolidation
- Archived memories retrievable via QueryAgent tool call when consolidation references them
- DuckDB training tables kept forever (trivial volume: ~2,000 rows/year)

### Ollama Configuration
- **Dedicated Ollama container** for SwingRL — separate from existing RAGFlow Ollama (prevents live trading contention)
- Both models loaded 24/7: OLLAMA_KEEP_ALIVE=24h, OLLAMA_MAX_LOADED_MODELS=2
- i5-13500 CPU (14C/20T), 64GB DDR4-3200. ~30GB free RAM. Models + training fit in ~19-27GB with headroom

### Model Selection
- **qwen2.5:3b** (fast, ~3-4GB RAM, 15-25 t/s): IngestAgent + QueryAgent — frequent, latency-sensitive ops
- **qwen3:14b** (smart, ~12-16GB RAM, 4-7 t/s): ConsolidateAgent + run_config + epoch_advice — strategic decisions
- **No nomic-embed-text** — embeddings not needed (SQL filtering + LLM reasoning is sufficient)
- Fallback: if qwen3:14b fails → fail-open (no LLM advice), NO intermediate 3b fallback for strategic tasks
- If qwen3:14b has tool calling issues, one-line YAML swap to qwen2.5:14b

### Failure Handling
- All memory calls fail-open — training never blocks on memory unavailability
- Timeouts: epoch_advice=8s, run_config=15s, consolidation=60s, general ingest=5s
- Health endpoint (/health) on swingrl-memory. Startup warning if unavailable. Discord alert if down >1hr
- Consolidation failures: unconsolidated memories stay in queue, next cycle picks them up

### Meta-Trainer Control Scope
- **Outer loop** (once per run, before trainer.learn()): LLM sets hyperparams (learning_rate, batch_size, n_epochs, gamma, clip_range, entropy_coeff) + curriculum window weights + initial reward weights
- **Inner loop** (every rollout via MemoryEpochCallback): LLM adjusts reward weights + can trigger early stop
- All LLM outputs pass through bounds.py before any trainer contact — hard-clamped with warnings logged
- Cold-start guard: meta-trainer passive until 3 completed runs exist in DuckDB. No pause needed between runs — DuckDB writes are immediate

### Safety Bounds
- Hyperparams: learning_rate [1e-5, 1e-3], entropy_coeff [0, 0.05], clip_range [0.1, 0.4], n_epochs [3, 20], batch_size [32, 512] (power of 2), gamma [0.90, 0.9999]
- Reward weights: profit [0.10, 0.70], sharpe [0.10, 0.60], drawdown [0.05, 0.50], turnover [0.00, 0.20] — renormalized to sum=1.0 after clamping
- min_epochs_before_stop = 10 (hard floor — never lower)
- max_epochs = 200, min_window_years = 1.0, crisis_period_pct [5%, 50%]

### Training Data Storage
- Epoch snapshots: every 5th epoch + notable events (KL spike >0.02, MDD breach <-0.08)
- Two-pass reward adjustment tracking: trigger ingested immediately on weight change, outcome ingested 10 epochs later with Sharpe/MDD delta
- Source tags with ':historical' suffix enforced by MemoryClient.ingest_training() assertion
- 5 training source tags: training_run:historical, training_epoch:historical, reward_adjustment:historical, backtest_fill:historical, curriculum_performance:historical
- Backtest seeding (seed_memory_from_backtest.py) runs after first successful training run
- Pre-live validation: automated script (scripts/validate_memory.py) + manual curl commands documented in runbook

### Meta-Trainer Audit Trail
- meta_decisions DuckDB table stores every LLM decision: proposed_config (raw LLM output) + clamped_config (after bounds.py) + rationale text
- Proposed vs clamped comparison reveals if the LLM consistently hits bounds (prompt tuning signal)

### HMM Regime Expansion
- Expand from 2 states (bull/bear) to 3 states (bull/bear/crisis) as part of Phase 19
- Existing hmm_state_history table in DuckDB — adjust _current_regime_vector() to query this table (not the spec's hmm_regimes table which doesn't exist)
- 3-state HMM: bull (positive returns, low-moderate vol), bear (negative returns, moderate vol), crisis (sharply negative returns, very high vol)
- 3 states fit cleanly with available data (10yr equity, 8.6yr crypto). Crisis periods: 2020 COVID, 2022 bear, 2018 crypto crash, Luna collapse
- Meta-trainer designed to work with any regime count — gracefully handles 2 or 3 keys

### Curriculum Sampler
- Part of Phase 19 (not deferred to Phase 22)
- LLM weights which historical periods to train on (e.g., more crisis exposure)
- Validated: min_window_years=1.0, crisis_period_pct between 5-50%
- Falls back to uniform sampling if memory agent unavailable

### Reward Shaping
- MemoryVecRewardWrapper: VecEnvWrapper (not gym.RewardWrapper — cannot wrap VecEnv)
- Intercepts step_wait(), applies weighted combination: profit + sharpe - drawdown - turnover
- Requires env info dict keys: raw_return, raw_sharpe_contrib, raw_drawdown_penalty, raw_turnover_cost
- If keys absent: rewards pass through unchanged (backwards-compatible)

### Retained Decisions from Original Context
- Gate failure: 2 tuning rounds if ensemble Sharpe < 0.5, BLOCK if both fail
- No wall-clock limit, sequential equity → crypto, 1M/500K timesteps with escalation to 2M/1M
- Walk-forward on FULL history, deployed model on RECENT data (3yr equity, 1yr crypto)
- SAC buffer_size = 1,000,000 (64GB handles it)
- Console summary + JSON report (data/training_report.json) + DuckDB fold results
- scripts/train_pipeline.py with --env equity|crypto|all via Docker exec

### Claude's Discretion
- Exact checkpoint format and resume detection logic
- How to structure the tuning hyperparameter grid within the specified ranges
- JSON report schema details beyond the specified content
- Console summary formatting
- MemoryVecRewardWrapper internal buffer sizes and rolling window parameters
- Exact system prompts for IngestAgent, ConsolidateAgent, QueryAgent
- How to detect convergence failure vs successful convergence for timestep escalation
- Exact HMM 3-state fitting parameters and label assignment thresholds

</decisions>

<specifics>
## Specific Ideas

- Adapt Google's Always-On Memory Agent pattern (github.com/GoogleCloudPlatform/generative-ai/tree/main/gemini/agents/always-on-memory-agent) but replace Google ADK with custom FastAPI + Ollama
- Training on homelab ONLY — not on M1 MacBook Pro. All training via Docker exec
- Fold-level data in DuckDB is diagnostic gold — regime-specific failures signal data problems, not hyperparameter problems
- The pipeline should be fully hands-off: operator runs one command, walks away, checks results later
- Two-layer knowledge: DuckDB for hard numbers (exact Sharpe, MDD), memory.db for soft knowledge (patterns, causal insights, cross-run learnings)
- Consolidation mimics "sleep consolidation" — finds cross-run patterns that individual run summaries don't reveal
- Reference spec files: memory_meta_trainer.md (architecture + code), memory_storage_spec.md (data formats + seeding)

</specifics>

<code_context>
## Existing Code Insights

### Reusable Assets
- `src/swingrl/training/trainer.py`: TrainingOrchestrator with locked hyperparams, VecNormalize, 6 smoke tests — core training engine
- `src/swingrl/training/ensemble.py`: EnsembleBlender with sharpe_softmax_weights — produces real ensemble weights
- `src/swingrl/training/callbacks.py`: ConvergenceCallback — detects training stagnation (1% improvement, 10 patience)
- `src/swingrl/agents/backtest.py`: WalkForwardBacktester with generate_folds(), per-fold retraining, DuckDB storage
- `src/swingrl/agents/validation.py`: check_validation_gates() (4 gates), diagnose_overfitting()
- `src/swingrl/agents/metrics.py`: annualized_sharpe, max_drawdown, sortino, calmar, profit_factor
- `src/swingrl/features/hmm_regime.py`: HMMRegimeDetector with store_hmm_state() → hmm_state_history table (2-state, needs expansion to 3)
- `scripts/train.py`: Existing CLI with --env, --algo, --timesteps, --dry-run, feature loading from DuckDB

### Established Patterns
- Docker exec for operator commands (Phase 18: `docker exec swingrl python -m swingrl.data.ingest_all --backfill`)
- JSON report files at data/ path (Phase 18: data/verification.json)
- Config loaded via load_config() from SwingRLConfig — MemoryAgentConfig must be added to schema.py
- structlog with keyword args for all logging
- Typed exceptions: ModelError for training/model failures
- Docker inter-container communication via Docker DNS (container_name:port)

### Integration Points
- scripts/train.py already has _load_features_prices() that assembles observation matrices from DuckDB
- WalkForwardBacktester._store_results() writes to DuckDB backtest_results table
- _write_model_metadata() writes to DuckDB model_metadata table
- Models save to models/active/{env}/{algo}/model.zip + vec_normalize.pkl
- HYPERPARAMS dict in trainer.py holds the locked starting values — meta-trainer creates variants within bounds
- hmm_state_history table already exists — stores p_bull, p_bear per environment per date
- Existing Ollama container on homelab (RAGFlow) — SwingRL gets its own dedicated container

</code_context>

<deferred>
## Deferred Ideas

- Memory dashboard: add "Memory" tab to existing Streamlit dashboard to browse memories and consolidation insights — separate phase
- Inbox file watcher: support dropping research notes/files into inbox/ for auto-ingestion — future enhancement
- Expand HMM to 4 states (adding sideways) — revisit when more data accumulated or if 3-state fit quality is insufficient
- nomic-embed-text embedding search — add if unstructured data sources (research papers, market commentary) are integrated later

</deferred>

---

*Phase: 19-model-training*
*Context gathered: 2026-03-13*
