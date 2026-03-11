# Phase 15: Training CLI Observation Assembly - Context

**Gathered:** 2026-03-10
**Status:** Ready for planning

<domain>
## Phase Boundary

Fix `train.py._load_features_prices()` to produce pre-assembled observation matrices matching env observation_space dimensions (156 equity / 45 crypto). Currently passes raw DuckDB feature columns (15/13 cols) directly, causing SB3 shape mismatch on model construction.

</domain>

<decisions>
## Implementation Decisions

### Transformation approach
- Pre-compute full `(N, obs_dim)` observation matrix before passing to TrainingOrchestrator
- Use `ObservationAssembler.assemble_equity()` / `assemble_crypto()` per timestep, stack into matrix
- Default portfolio state (100% cash via `_default_portfolio_state()`) — env overwrites at step time

### Data flow
- Read per-symbol feature rows from DuckDB `features_equity` / `features_crypto`
- Group by timestep, split into per-asset dicts + macro + HMM + turbulence components
- Pass grouped components to ObservationAssembler for each timestep
- Stack assembled vectors into `(N, obs_dim)` numpy array

### Price extraction
- Extract close prices per-asset for the price array the env/trainer needs
- Maintain existing `_load_features_prices` return signature: `tuple[np.ndarray, np.ndarray]`

### Claude's Discretion
- Exact DuckDB query structure for grouping features by timestep
- Whether to add HMM/macro/turbulence column identification via naming convention or config
- Error handling for missing feature groups (e.g., HMM not computed yet)

</decisions>

<specifics>
## Specific Ideas

No specific requirements — the fix is fully constrained by the existing assembler API and env contract.

</specifics>

<code_context>
## Existing Code Insights

### Reusable Assets
- `ObservationAssembler` (`src/swingrl/features/assembler.py`): Full assembly API with validation
- `equity_obs_dim()` / `CRYPTO_OBS_DIM`: Dynamic dimension computation
- `_default_portfolio_state()`: 100% cash default for training pre-assembly
- Feature name lists (`_EQUITY_FEATURE_NAMES`, `_MACRO_FEATURE_NAMES`, etc.): Column identification

### Established Patterns
- `BaseTradingEnv._get_observation()` reads `self._features[step]` and overlays portfolio state on last `_portfolio_dim` positions
- `TrainingOrchestrator._create_env()` passes features/prices directly to env constructor
- DuckDB `features_equity` / `features_crypto` tables store per-symbol rows with feature columns

### Integration Points
- `train.py:_load_features_prices()` → only function that needs changes
- `TrainingOrchestrator.train(features, prices)` → unchanged, receives pre-assembled matrix
- `BaseTradingEnv.__init__(features=...)` → unchanged, receives `(N, obs_dim)` array

</code_context>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope

</deferred>

---

*Phase: 15-training-cli-observation-assembly*
*Context gathered: 2026-03-10*
