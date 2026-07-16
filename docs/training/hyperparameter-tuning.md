# Hyperparameter Tuning Reference

Living reference for the LLM-driven hyperparameter advice loop. Covers the request/response chain (`MetaTrainingOrchestrator` ↔ memory service `/training/run_config` ↔ cloud LLM), the double-clamp safety layer, the cold-start guard, and the audit trail across pg16 tables. Update when any referenced module changes.

**Last verified against code:** 2026-07-15 (U2 fix — cold-start fallback `reward_weights` now equal canonical `DEFAULT_WEIGHTS`)

For the mechanistic background of each HP (what it controls, financial-RL ranges, diagnostic patterns) see `.planning/research/hp-tuning-reference.md`. For the SB3 baseline `HYPERPARAMS` dict, see [`agent-architecture.md`](agent-architecture.md).

## End-to-end flow

```
train_pipeline.py per fold/iteration
  └── if memory_agent.enabled AND memory_agent.meta_training:
        ├── (WF folds)         wf_meta.query_hyperparams(env, algo, iter=N)        meta_orchestrator.py:229-260
        └── (final retrain)    meta.run(... hyperparams_override=baseline ...)     meta_orchestrator.py:114-227
              │
              ▼
      _query_run_config(env, algo, iter)                                           meta_orchestrator.py:262-336
        ├── _get_pattern_count(env)  →  GET /debug/consolidations?limit=100        meta_orchestrator.py:354-390
        │     └── if count < _COLD_START_MIN_PATTERNS (=1) → return {}             meta_orchestrator.py:283-293
        ├── _current_regime_vector(env)  →  hmm_state_history (pg16)               meta_orchestrator.py:392-431
        └── POST {base_url}/training/run_config { query: "...regime=..." }
              │     timeout = memory_agent.meta_training_timeout_sec (300s default)
              ▼
      services/memory/routers/training.py:97-108
        └── QueryAgent.advise_run_config(query)                                    query.py:1035-1104
              ├── _build_context_async(env, algo, request_type="run_config")
              │     └── pulls active consolidations + raw memories, XML-wraps them
              ├── schema = _build_run_config_schema(algo)                          query.py:876-903
              ├── system_prompt = _build_algo_system_prompt(bounds, algo)
              ├── _call_lm(user, schema, system_prompt) — primary then backup      query.py:1646-1694
              │     └── _is_provider_blocked() — skips providers in 429 backoff    query.py:361-378
              ├── _track_presentations_async(...)  →  pattern_presentations
              ├── _audit_log(call_type="run_config", ...)  →  llm_audit_log
              ├── merged = _SAFE_DEFAULTS | result
              ├── clamped = _clamp_run_config(merged, algo=algo)  ◄── service-side clamp
              └── return JSON { learning_rate, entropy_coeff, ..., reward_weights, rationale }
              │
              ▼
      back in MetaTrainingOrchestrator:
        ├── safe_config = clamp_run_config(advised, algo)  ◄── trainer-side clamp  meta_orchestrator.py:160
        ├── _write_meta_decision(run_id, algo, env, json, rationale)               meta_orchestrator.py:525-545
        │     └── INSERT INTO meta_decisions (decision_type='hp_tuning', ...)
        ├── filter to _VALID_HP_KEYS[algo], rename "entropy_coeff" → "ent_coef"    meta_orchestrator.py:172-178
        ├── merged_hp = hyperparams_override (baseline) | memory_hp                meta_orchestrator.py:181-186
        └── trainer.train(... hyperparams_override=merged_hp ...)
              │
              ▼
      TrainingOrchestrator.train()  →  params = HYPERPARAMS[algo] | merged_hp     trainer.py:259-260
                                        SB3 model = AlgoClass(**params)
```

**Key invariant: double clamp.** LLM output passes through `_clamp_run_config` *inside the service* (`query.py:936-966`) before the JSON leaves, then through `clamp_run_config` *inside the trainer process* (`bounds.py:181-256`) again. Same logic, same yaml-loaded bounds — idempotent in normal flow, but the trainer-side clamp is a backstop in case any caller ever bypasses the network path.

**Two production entry points** (both gated by `memory_agent.enabled AND memory_agent.meta_training`, `train_pipeline.py:2143-2145`):

| Path | Caller | Method | Writes `meta_decisions`? |
|------|--------|--------|--------------------------|
| WF fold HP query | `train_pipeline.py:2161` | `query_hyperparams()` | No (lighter; just returns HP dict) |
| Final retrain | `train_pipeline.py:1923` | `run()` (full wrapper) | Yes |

## Per-algo reachable HPs

The set of HPs the LLM can actually move through to SB3 = **intersection of the service schema (`_ALGO_HP_FIELDS`, `query.py:849-873`) and the client whitelist (`_VALID_HP_KEYS`, `meta_orchestrator.py:56-80`)**. Keys outside the intersection are silently dropped on one side or the other.

| HP | PPO | A2C | SAC | Notes |
|-----|:---:|:---:|:---:|-------|
| `learning_rate` | ✅ | ✅ | ✅ | |
| `entropy_coeff` (→ `ent_coef`) | ✅ | ✅ | ✅ | renamed `meta_orchestrator.py:175-178`. SAC baseline is `"auto_0.1"` (string); a numeric override replaces the auto schedule. |
| `gamma` | ✅ | ✅ | ✅ | algo-specific bounds (see Clamping). |
| `clip_range` | ✅ | — | — | PPO-only by design. |
| `n_epochs` | ✅ | — | — | PPO-only. |
| `batch_size` | ✅ | — | ✅ | A2C has no minibatch concept. |
| `target_kl` | ✅ | — | — | service offers; client allows. |
| `gae_lambda` | — | ✅ | — | A2C-reachable; PPO client allows it but service does NOT offer it for PPO (mismatch — see Known issues). |
| `gradient_steps` | — | — | ✅ | |
| `target_entropy` | — | — | ✅ | |

### Dead keys (in whitelist but unreachable via LLM)

These keys are in `_VALID_HP_KEYS` (`meta_orchestrator.py:56-80`) but **not** in the service schema (`_ALGO_HP_FIELDS`, `query.py:849-873`), so the LLM has no path to set them. They keep their hardcoded `HYPERPARAMS` defaults always:

- **PPO:** `n_steps`, `gae_lambda`, `vf_coef`
- **A2C:** `n_steps`, `vf_coef`
- **SAC:** `tau`, `learning_starts`

## Bounds & clamping rules

Both `clamp_run_config` implementations (service `query.py:936-966`, trainer-side `bounds.py:181-256`) load bounds **once at module import** from `config/swingrl.yaml` `training.bounds.hyperparam_bounds.*` (with hardcoded fallbacks if the yaml key is missing). **Yaml edits to bounds require a container restart on both the swingrl and swingrl-memory containers.**

### Global HP bounds (`HYPERPARAM_BOUNDS`, loaded at `bounds.py:95`)

| Key | Default `(lo, hi)` | Yaml key | Cast |
|-----|:------------------:|----------|------|
| `learning_rate` | `(1e-5, 1e-3)` | `training.bounds.hyperparam_bounds.learning_rate` | float |
| `entropy_coeff` | `(0.0, 0.05)` | `…entropy_coeff` | float |
| `clip_range` | `(0.1, 0.4)` | `…clip_range` | float |
| `n_epochs` | `(3, 20)` | `…n_epochs` | **int** |
| `batch_size` | `(32, 512)` | `…batch_size` | **int → nearest power of 2** (`bounds.py:163-173, 224-233`) |
| `gamma` | `(0.95, 0.995)` | `…gamma` | float (algo-overridden, see below) |
| `target_kl` | `(0.01, 0.05)` | `…target_kl` (fallback only) | float |
| `gae_lambda` | `(0.85, 1.0)` | `…gae_lambda` (fallback only) | float |
| `gradient_steps` | `(1, 8)` | `…gradient_steps` (fallback only) | **int** |
| `target_entropy` | `(-9.0, -0.5)` | `…target_entropy` (fallback only) | float |

Current yaml only declares the first 6 keys; the rest fall back to hardcoded defaults (`bounds.py:31-42`).

### Algo-specific gamma overrides (`_ALGO_GAMMA_BOUNDS`, `bounds.py:44-48`)

| Algo | gamma bounds | Reason |
|------|:------------:|--------|
| PPO | `(0.95, 0.995)` | global default |
| A2C | `(0.95, 0.985)` | tighter ceiling — see mismatch invariant |
| SAC | `(0.95, 0.995)` | global default |

### A2C mismatch-factor invariant (`bounds.py:235-254`)

```
(1 / (1 - gamma)) / n_steps  <  8.0
```

If a clamped A2C `gamma` would violate this with the current `n_steps` (default 5), `gamma` is further clamped down to satisfy the constraint. Provenance: iter 3 collapse with `gamma=0.999, n_steps=5` produced a 200× mismatch — the agent bootstrapped a ~1000-step horizon from 5 real reward steps.

### Reward weights

`_clamp_reward_weights` in both processes clamps each component to `REWARD_BOUNDS`, then **renormalizes to sum=1.0**. If all clamped weights are 0 it returns the midpoint defaults normalized. Bounds + behavior live in [`reward-shaping.md`](reward-shaping.md); the same `reward_weights` block ships back as part of the `/training/run_config` JSON.

### Other safety constants (`bounds.py:97-98`)

| Constant | Value | Purpose |
|----------|------:|---------|
| `MIN_TRAINING_PROGRESS` | 0.20 | hard floor — never lower (declared but currently informational; not consumed in HP path) |
| `MAX_EPOCHS` | 200 | declared but currently informational |

## Cold-start guard

`_COLD_START_MIN_PATTERNS = 1` (`meta_orchestrator.py:52`). `_get_pattern_count(env)` calls `GET /debug/consolidations?limit=100` and counts entries where `env_name == env_name OR env in affected_envs`. **If the count is below the threshold, `_query_run_config` returns `{}` and the trainer falls through to the baseline `HYPERPARAMS` dict** (`meta_orchestrator.py:283-293`).

**Fail-open everywhere.** Any exception in the chain — DNS failure, 5xx, JSON parse error, timeout — is caught and logged at `WARNING`; the orchestrator returns `{}` and training proceeds with baseline HPs (`meta_orchestrator.py:329-336`). Training never blocks on memory unavailability.

## Provider chain (HP tuning is cloud-only)

`_call_lm` (`query.py:1646-1694`) iterates `("primary", "backup")` and returns the first non-`None` result. There is **no Ollama fallback for HP advice** — Ollama is wired into `advise_epoch`, not `advise_run_config` (the comment at `query.py:250` and the `_call_lm` docstring are explicit). If both cloud tiers fail, `advise_run_config` returns `_SAFE_DEFAULTS` (`query.py:1087-1089`) and the trainer-side caller treats that the same as a successful cloud reply (clamped, double-checked, applied).

### Provider selection (`_load_query_cloud_config`, `query.py:146-196`)

- **Primary** = `memory_agent.query_provider` (yaml; default `"gemini"`).
- **Backup** = `"openrouter"` if primary ≠ openrouter, else `"nvidia"` — **hardcoded in code** (`query.py:174`); not yaml-tunable.
- Per-provider `base_url`, `default_model`, `timeout_sec`, `max_tokens` come from `memory_agent.consolidation.providers.{provider}` in yaml (the same map shared with consolidation).
- API key resolution: env var `{PROVIDER}_API_KEY` overrides the yaml `api_key` field at startup.

### 429 / rate-limit handling (`query.py:361-408`)

Per-provider exponential backoff, calendar-day reset:

| Failure count | Block window |
|--------------:|--------------|
| 1 | 5 min |
| 2 | 15 min |
| 3 | 60 min |
| 4+ | rest of UTC day |

Auto-resets when `now >= blocked_until`. Tracked in process memory (`_CLOUD_BLOCKED` dict, lock-protected). Disabled by setting `memory_agent.cloud_block_on_429 = false`. Trigger codes are `memory_agent.cloud_block_codes` (default `[429]`).

### Structured-output enforcement per provider (`query.py:1559-1597`)

| Provider | Mode |
|----------|------|
| `gemini`, `openrouter`, `cerebras` | `response_format = {type: json_schema, strict: true, schema: …}` |
| `nvidia` | `response_format = {type: json_object}` + `guided_json` field |
| `mistral`, `groq`, others | `response_format = {type: json_object}` (no schema enforcement) |

## Tuning heuristics (folded from research)

The full mechanistic background — what each HP controls, financial-RL recommended ranges, key interactions — lives in `.planning/research/hp-tuning-reference.md`. The diagnostic tables below are the action-oriented summaries the LLM is implicitly choosing from.

### PPO

| Symptom | Primary HP lever | Secondary |
|---------|------------------|-----------|
| High overfit_gap (IS ≫ OOS Sharpe) | Reduce `n_epochs` (10 → 4–6) | Add `target_kl=0.015` |
| Exploding `approx_kl` | Reduce `learning_rate` | Narrow `clip_range` |
| `clip_fraction > 0.3` | Reduce `n_epochs` | Narrow `clip_range` |
| Policy collapse (entropy → 0) | Increase `entropy_coeff` | Verify LR not too high |
| Value function overfit (`explained_var > 0.9`) | Reduce `vf_coef` ⚠ unreachable via LLM — code edit | Reduce `n_epochs` |

### A2C

| Symptom | Primary HP lever | Secondary |
|---------|------------------|-----------|
| High overfit_gap | Reduce `learning_rate` (first!) | Reduce `gae_lambda` (1.0 → 0.95) |
| Erratic policy (whipsawing) | Reduce `learning_rate` | Increase `n_steps` ⚠ unreachable via LLM |
| High gradient variance | Increase `n_steps` ⚠ unreachable via LLM | Reduce `gae_lambda` |
| Policy collapse | Increase `entropy_coeff` | Reduce `learning_rate` |
| Myopic trading (too reactive) | Increase `gamma` | Increase `n_steps` ⚠ unreachable |

### SAC

| Symptom | Primary HP lever | Secondary |
|---------|------------------|-----------|
| Q-values exploding | Reduce `learning_rate` | Reduce `tau` ⚠ unreachable via LLM |
| High overfit_gap | Reduce `learning_rate` | `buffer_size` is yaml-only (not LLM) |
| Over-exploration (random trades) | Set `entropy_coeff` (numeric override of `auto_0.1`) | Tighten reward penalties |
| Premature exploitation | Increase `entropy_coeff` | `learning_starts` is unreachable via LLM |
| Regime confusion (bull/bear averaging) | yaml `training.sac_buffer_size` (smaller) | Reduce `gamma` |

### Cross-algo: when to adjust what

| Problem | PPO | A2C | SAC |
|---------|-----|-----|-----|
| Overfitting | Reduce `n_epochs`; `target_kl` | Reduce `learning_rate`; reduce `gae_lambda` | Reduce `learning_rate` |
| Underfitting | Increase `learning_rate` / `n_epochs` | Increase `learning_rate` | Increase `learning_rate` / `batch_size` |
| Policy instability | Narrow `clip_range` | Reduce `learning_rate` (only lever) | Reduce `learning_rate` |
| Capital preservation | Lower `gamma` (0.95–0.97), narrow `clip` | Lower `gamma`, lower `learning_rate` | Lower `gamma`, numeric `entropy_coeff` |

## History & audit trail

Four pg16 tables back the HP-tuning audit chain:

| Table | Writer | Granularity | Notable columns |
|-------|--------|-------------|-----------------|
| `meta_decisions` | `_write_meta_decision` (`meta_orchestrator.py:525-545`) | per `meta.run()` call | `run_id`, `algo`, `env`, `decision_type='hp_tuning'`, `decision_json` (post-clamp), `rationale`, `created_at`. DDL: `postgres_schema.py:325-336`. |
| `llm_audit_log` | `_audit_log` in QueryAgent (`query.py:1003-1033`) | per LLM call (any provider) | `call_type='run_config'`, `provider`, `model_name`, `prompt_text`, `response_text`, `response_parsed`, `latency_ms`, `success`, `error_text`, plus `algo`/`env`/`fold_number`/`iteration_number` context. DDL: `services/memory/db.py:170-189`. |
| `pattern_presentations` | `insert_pattern_presentation_async` (`query.py:1500-1510`) | per consolidation × per query | `consolidation_id`, `iteration`, `env_name`, `request_type`, `advice_response` (rationale ≤200 chars). DDL: `services/memory/db.py:143-151`. |
| `iteration_results` | `store_iteration_results_to_duckdb` (`backtest.py:882-1015`) | per iteration × env | `ppo_hyperparams`, `a2c_hyperparams`, `sac_hyperparams` (post-clamp JSON of what actually shipped to SB3), `hp_source` (`'baseline'` or `'llm'`). DDL: `postgres_schema.py:171-208`. |

Note: `model_metadata` (`postgres_schema.py:105-121`) does **not** carry per-fold HP JSON — only `validation_sharpe`, `ensemble_weight`, `total_timesteps`, `converged_at_step`. For "what HPs shipped in iter N", join `iteration_results` on `iteration_number` + `environment`.

## Configurable values (yaml)

### Memory-agent HP-tuning provider (`memory_agent.*`)

| Key | Schema default | Validator | Current yaml | File:Line |
|-----|---------------:|-----------|:-------------|-----------|
| `memory_agent.enabled` | `false` | bool | `true` | `schema.py:362`, `swingrl.yaml:89` |
| `memory_agent.meta_training` | `false` | bool | `false` | `schema.py:366`, `swingrl.yaml:90` |
| `memory_agent.query_provider` | `"gemini"` | str (gemini\|openrouter\|nvidia\|mistral\|cerebras\|groq) | gemini (default) | `schema.py:374`, yaml omitted |
| `memory_agent.meta_training_timeout_sec` | 300.0 | float | default | `schema.py:367` |
| `memory_agent.cloud_block_on_429` | `true` | bool | `true` | `schema.py:378`, `swingrl.yaml:100` |
| `memory_agent.cloud_block_codes` | `[429]` | `list[int]` | `[429]` | `schema.py:381`, `swingrl.yaml:101` |
| `memory_agent.timeout_sec` | 3.0 | float | default | `schema.py:364` (live-endpoint timeout, not HP path) |

### Per-provider config (`memory_agent.consolidation.providers.{name}`)

Shared with consolidation. Each entry: `base_url`, `api_key` (env-var-overridable via `{NAME}_API_KEY`), `default_model`, `timeout_sec`, `max_tokens`. Current yaml: `swingrl.yaml:133-165`.

### LLM HP bounds (`training.bounds.hyperparam_bounds.*`)

Used by `clamp_run_config` (both processes) to clip LLM output before it reaches the trainer.

| Key | Default `(lo, hi)` | Current yaml | File:Line |
|-----|-------------------:|:-------------|-----------|
| `learning_rate` | `(1e-5, 1e-3)` | `[1e-5, 1e-3]` | `schema.py:419`, `swingrl.yaml:170` |
| `entropy_coeff` | `(0.0, 0.05)` | `[0.0, 0.05]` | `schema.py:420`, `swingrl.yaml:171` |
| `clip_range` | `(0.1, 0.4)` | `[0.1, 0.4]` | `schema.py:421`, `swingrl.yaml:172` |
| `n_epochs` | `(3, 20)` | `[3, 20]` | `schema.py:422`, `swingrl.yaml:173` |
| `batch_size` | `(32, 512)` | `[32, 512]` | `schema.py:423`, `swingrl.yaml:174` |
| `gamma` | `(0.95, 0.995)` | `[0.95, 0.995]` | `schema.py:424`, `swingrl.yaml:175` |
| `target_kl` | `(0.01, 0.05)` | not in yaml | `schema.py:425` (fallback only) |
| `gae_lambda` | `(0.85, 1.0)` | not in yaml | `schema.py:426` (fallback only) |
| `gradient_steps` | `(1, 8)` | not in yaml | `schema.py:427` (fallback only) |
| `target_entropy` | `(-9.0, -0.5)` | not in yaml | `schema.py:428` (fallback only) |

Reward-weight bounds (`training.bounds.reward_bounds.*`) are documented in [`reward-shaping.md`](reward-shaping.md).

## Hardcoded values (not yaml-tunable — code edit required)

### Cold-start & filters

| Constant | Value | Location |
|----------|------:|----------|
| `_COLD_START_MIN_PATTERNS` | 1 | `meta_orchestrator.py:52` |
| `_get_pattern_count` HTTP limit | 100 | `meta_orchestrator.py:368` |

### Algo-specific

| Constant | Value | Location |
|----------|-------|----------|
| `_VALID_HP_KEYS["ppo"]` | `{learning_rate, n_steps, batch_size, n_epochs, gamma, gae_lambda, clip_range, ent_coef, vf_coef, target_kl}` | `meta_orchestrator.py:57-68` |
| `_VALID_HP_KEYS["a2c"]` | `{learning_rate, n_steps, gamma, gae_lambda, ent_coef, vf_coef}` | `meta_orchestrator.py:69` |
| `_VALID_HP_KEYS["sac"]` | `{learning_rate, batch_size, tau, gamma, ent_coef, learning_starts, gradient_steps, target_entropy}` | `meta_orchestrator.py:70-79` |
| `_ALGO_HP_FIELDS["ppo"]` | `{learning_rate, entropy_coeff, clip_range, n_epochs, batch_size, gamma, target_kl}` | `query.py:850-858` |
| `_ALGO_HP_FIELDS["a2c"]` | `{learning_rate, entropy_coeff, gamma, gae_lambda}` | `query.py:859-864` |
| `_ALGO_HP_FIELDS["sac"]` | `{learning_rate, entropy_coeff, batch_size, gamma, gradient_steps, target_entropy}` | `query.py:865-872` |
| `_ALGO_GAMMA_BOUNDS` | PPO `(0.95, 0.995)`, A2C `(0.95, 0.985)`, SAC `(0.95, 0.995)` | `bounds.py:44-48` |
| A2C mismatch-factor ceiling | 8.0 | `bounds.py:237` |

### Service-side fallback safe defaults (`query.py:122-131`)

```
learning_rate=3e-4, entropy_coeff=0.01, clip_range=0.2, n_epochs=10, batch_size=64, gamma=0.99,
reward_weights={profit: 0.50, sharpe: 0.25, drawdown: 0.15, turnover: 0.10},
rationale="cold_start_defaults"
```

Returned by `advise_run_config` when both cloud tiers fail. Not used on the trainer side directly — the trainer falls back to `HYPERPARAMS` baseline when `_query_run_config` returns `{}`.

**U2 fix (spec §2.2):** `reward_weights` in the fallback was `{profit: 0.4, sharpe: 0.35, drawdown: 0.20, turnover: 0.05}` prior to the fix — a value that diverged from the canonical `DEFAULT_WEIGHTS` (`reward_wrapper.py:33-38`). It now equals `DEFAULT_WEIGHTS` exactly, cross-checked by `tests/memory/test_query_safe_defaults.py`.

### 429 backoff schedule (`query.py:381-408`)

Hardcoded: 5 min → 15 min → 60 min → rest-of-UTC-day.

### Other

| Constant | Value | Location |
|----------|------:|----------|
| `MIN_TRAINING_PROGRESS` | 0.20 | `bounds.py:97` |
| `MAX_EPOCHS` | 200 | `bounds.py:98` |
| Backup provider for HP-tuning query | OpenRouter (NVIDIA when primary IS OpenRouter) | `query.py:174` — not yaml-tunable. See Known issues for rationale. |

## Invariants

- **Double clamp.** Every LLM output is clamped twice — service-side (`query.py:1097`) and trainer-side (`meta_orchestrator.py:160`). Both load bounds at module import from the same yaml file (mounted into both containers); a yaml change requires restarting both `swingrl` and `swingrl-memory` containers to take effect on both sides.
- **Fail-open everywhere.** Any failure (LLM 4xx/5xx, network timeout, JSON parse error, DB error, cold start) returns `{}` from `_query_run_config` → trainer uses baseline `HYPERPARAMS`. Training never blocks on memory unavailability.
- **`entropy_coeff → ent_coef` rename** is performed exactly once on the trainer side (`meta_orchestrator.py:175-178`, `:250`); the service emits `entropy_coeff`, SB3 expects `ent_coef`.
- **Baseline override beats LLM advice.** When `hyperparams_override` is provided to `meta.run()`, conflicting LLM keys are ignored (`meta_orchestrator.py:181-186`).
- **Reward weights always normalize to sum = 1.0** (`bounds.py:288-297`). All-zero clamped weights → midpoint defaults normalized.
- **Integer cast after clamping.** `n_epochs`, `batch_size`, `gradient_steps` are forced to `int` after numeric clamp (`bounds.py:220-222`). `batch_size` additionally rounds to nearest power of 2 (`bounds.py:225-233`).
- **Algo-specific gamma always wins over global.** `_ALGO_GAMMA_BOUNDS[algo]` overrides the global `(0.95, 0.995)` window when algo is provided to `clamp_run_config` (`bounds.py:204-205`).
- **A2C mismatch invariant is post-clamp.** Even if global gamma bounds permit it, A2C `gamma` is further clamped down whenever `(1/(1−γ))/n_steps ≥ 8.0` (`bounds.py:237-254`).
- **Bounds are loaded once.** Both `bounds.py:95` and `query.py:120` execute `_load_bounds()` at module import. Hot-reload after yaml edit requires container restart.

## Known issues / open questions

- **Service ↔ client whitelist mismatch.** `_VALID_HP_KEYS` (client, `meta_orchestrator.py:56-80`) includes keys (`n_steps`, `vf_coef`, `tau`, `learning_starts`) that `_ALGO_HP_FIELDS` (service, `query.py:849-873`) never offers. The LLM cannot suggest them, so they're effectively dead — but a future service-schema edit could enable them without touching the client. Conversely, removing a key from the client whitelist without updating the service schema means the LLM's suggestion is silently dropped (`meta_orchestrator.py:177`).
- **PPO `gae_lambda` is asymmetric.** Client whitelist allows it; service schema does not include it for PPO (only A2C). PPO can never receive an LLM-suggested `gae_lambda`.
- **`min_run_history_for_meta` is dead config.** Declared in `schema.py:368` (default 3) but never read anywhere in `src/` or `services/`. The actual cold-start gate is `_COLD_START_MIN_PATTERNS = 1` (`meta_orchestrator.py:52`). Either consume the field or remove it.
- **`_get_pattern_count` caps at 100 consolidations.** `GET /debug/consolidations?limit=100` then filters client-side. Not currently a problem but worth knowing if pattern count grows past 100 per env — the LLM could be misled about cold-start status.
- **Backup provider is hardcoded.** `_load_query_cloud_config` (`query.py:174`) sets backup to `"openrouter"` unless primary is openrouter, in which case backup is `"nvidia"`. There's no yaml knob to choose the backup; configuring a third tier needs a code change.
- **HP tuning has no Ollama fallback.** Unlike `advise_epoch`, `_call_lm` is cloud-only (`query.py:1660`). If both `query_provider` and the implicit backup are 429-blocked, HP advice falls through to `_SAFE_DEFAULTS` for the remainder of the block window.
- **`_SAFE_DEFAULTS` is PPO-shaped.** The cold-start safe-default dict (`query.py:122-131`) carries PPO-flavored keys (`clip_range`, `n_epochs`); after `_clamp_run_config` and the algo-aware filter the algo-incompatible ones are dropped, but it means an A2C/SAC cold-start emits a noisy "clamp" log line for keys that wouldn't have been used anyway.
- **`MIN_TRAINING_PROGRESS=0.20` and `MAX_EPOCHS=200`** are declared in `bounds.py:97-98` but not consumed by the HP path itself — they're informational ceilings. Verify before assuming they constrain anything.
- **Bounds yaml is not hot-reloadable.** Editing `training.bounds.*` and *not* restarting both containers gives an inconsistent picture: trainer uses old bounds, service uses old bounds, and the yaml on disk lies about what's enforced.

## Source of truth

| Concern | File |
|---------|------|
| Trainer-side meta-orchestrator + cold-start guard + DB writes | `src/swingrl/memory/training/meta_orchestrator.py` |
| Trainer-side HP / reward bounds + clamp + A2C mismatch | `src/swingrl/memory/training/bounds.py` |
| Service-side QueryAgent + LLM call + service-side clamp | `services/memory/memory_agents/query.py` |
| Service-side HTTP routes (`/training/run_config`, `/training/epoch_advice`) | `services/memory/routers/training.py` |
| Production callers (WF + final retrain entry points) | `scripts/train_pipeline.py` |
| Trainer baseline HPs | `src/swingrl/training/trainer.py` (`HYPERPARAMS`) |
| pg16 DDL (`meta_decisions`, `iteration_results`, `model_metadata`) | `src/swingrl/data/postgres_schema.py` |
| Service-side DDL (`llm_audit_log`, `pattern_presentations`, `pattern_outcomes`) | `services/memory/db.py` |
| Config schema (`MemoryAgentConfig`, `HyperparamBoundsConfig`, `RewardBoundsConfig`, `TrainingBoundsConfig`) | `src/swingrl/config/schema.py` |
| Provider yaml (per-provider URLs / models / timeouts) | `config/swingrl.yaml` (`memory_agent.consolidation.providers.*`) |
| Mechanistic HP background (per-HP semantics, financial-RL ranges) | `.planning/research/hp-tuning-reference.md` |

## Changelog

- **2026-05-07** — Initial version.
- **2026-05-15** — Promoted "Dead keys" to its own subsection above the reachability table. Added hardcoded-backup-provider row to Hardcoded values table (also remains in Known issues for context).
