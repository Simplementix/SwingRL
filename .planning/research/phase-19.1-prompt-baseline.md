# Phase 19.1 — C0 Prompt/Payload/Weights Baseline

**Purpose:** Archival snapshot of the LLM advice system *before* Stage 2 training-refocus
changes. This is the "before picture." Later tasks (C1–C5) change these; this doc is the
QA comparison baseline and the source of QA criteria for pattern regeneration (Group D).

**Do not modify this file** once committed — it is a historical record.

---

> **TASK 2 FLAG — ENV_MEDIAN_RETURN CONSTANTS:**
> Actual `med_ret` values from production data differ from plan estimates.
> Task 2 must use these actual values:
> - equity: **0.0535** (estimate was 0.053 — matches)
> - crypto: **0.3893** (estimate was 0.70 — **significantly lower**)
>
> The crypto estimate was materially wrong. Task 2's `ENV_MEDIAN_RETURN` constant for
> crypto must be set to **0.3893**, not 0.70.

---

## 1. Current Epoch-Advice Payload

Source: `src/swingrl/memory/training/epoch_callback.py`, lines 608–617 (verified).

```python
iter_part = f" iteration={self._iteration}" if self._iteration is not None else ""
payload = {
    "query": (
        f"EPOCH ADVICE: run_id={self._run_id} algo={self._algo} "
        f"env={self._env} epoch={self._epoch}{iter_part} "
        f"rolling_sharpe={self._wrapper.rolling_sharpe():.4f} "
        f"rolling_mdd={self._wrapper.rolling_mdd():.4f} "
        f"current_weights={_json.dumps(self._wrapper.weights)}"
    )
}
```

**What is missing from this payload (C3 will add):**
- No `total_trades` or trade-rate signal — the LLM cannot distinguish trade-shy collapse
  from healthy low-activity.
- No `win_rate` — cannot distinguish profitable silence from loss-avoiding silence.
- No fold context (fold number, fold start/end date, regime) — advice is context-free.

---

## 2. Current Run-Config Payload

Source: `src/swingrl/memory/training/meta_orchestrator.py`, lines 298–304 (verified).

```python
iter_part = f" iteration={iteration}" if iteration is not None else ""
payload = {
    "query": (
        f"TRAINING RUN CONFIG ADVICE: env={env_name} algo={algo_name}{iter_part} "
        f"current_regime={json.dumps(regime)}"
    )
}
```

**What is missing from this payload (C3 will add):**
- No prior-fold trade-rate statistics — LLM cannot see whether the last fold was trade-shy.
- No worst-fold metrics — LLM cannot anchor to the actual distribution of outcomes.
- No ENV_MEDIAN_RETURN baseline for context — advice is regime-only, no return anchor.

---

## 3. Current Prompt Builder Functions

Source: `services/memory/memory_agents/query.py`

### 3.1 `_build_system_prompt` — lines 429–463 (verified)

```python
def _build_system_prompt(
    hp_bounds: dict[str, tuple[Any, Any]],
    rw_bounds: dict[str, tuple[float, float]],
) -> str:
    """Build the system prompt dynamically from config-loaded bounds.

    Args:
        hp_bounds: Hyperparameter bounds dict.
        rw_bounds: Reward weight bounds dict.

    Returns:
        System prompt string with bounds inlined.
    """
    hp_lines = "\n".join(f"- {k}: [{lo}, {hi}]" for k, (lo, hi) in hp_bounds.items())
    rw_lines = "\n".join(f"- {k}: [{lo}, {hi}]" for k, (lo, hi) in rw_bounds.items())
    return (
        "You are the training advisor agent for SwingRL, an RL-based "
        "swing trading system.\n\n"
        "SwingRL context:\n"
        "- Two environments: equity daily (8 ETFs) and crypto 4H (BTC/ETH)\n"
        "- Algorithms: PPO (on-policy), A2C (on-policy), SAC (off-policy, "
        "entropy-maximizing)\n"
        "- Capital preservation is the PRIMARY constraint — Sortino ratio and "
        "MDD are the main metrics\n"
        "- Market regimes: bull (0), bear (1), crisis (2) — detected by HMM "
        "from FRED indicators\n\n"
        f"Hyperparameter bounds (you MUST stay within these):\n{hp_lines}\n\n"
        "Reward weight bounds (you MUST stay within these, weights should "
        f"sum to ~1.0):\n{rw_lines}\n\n"
        "You will receive recent memory patterns from past training runs and "
        "queries about current training state.\n"
        "Provide specific, numerical advice grounded in the memory patterns.\n"
        "If memory patterns are insufficient, stay close to safe defaults "
        "and explain why."
    )
```

**Note:** This generic system prompt is used for non-algo-specific run_config paths.
The algo-specific path uses `_build_algo_system_prompt` instead (see §3.2).

### 3.2 `_build_algo_system_prompt` — lines 650–695 (verified)

```python
def _build_algo_system_prompt(
    hp_bounds: dict[str, tuple[Any, Any]],
    rw_bounds: dict[str, tuple[float, float]],
    algo_name: str,
) -> str:
    """Build algo-specific system prompt with filtered HP bounds and tuning guide.

    Args:
        hp_bounds: Hyperparameter bounds dict (all algos).
        rw_bounds: Reward weight bounds dict.
        algo_name: Algorithm name ('ppo', 'a2c', 'sac').

    Returns:
        System prompt string with algo-specific bounds, HP guide, and hallucination guards.
    """
    valid_hp = set(_ALGO_HP_FIELDS.get(algo_name, {}).keys())
    filtered_bounds = {k: v for k, v in hp_bounds.items() if k in valid_hp}
    # Override gamma bounds with algo-specific values when available
    if "gamma" in filtered_bounds and algo_name.lower() in _ALGO_GAMMA_BOUNDS:
        filtered_bounds["gamma"] = _ALGO_GAMMA_BOUNDS[algo_name.lower()]
    hp_lines = "\n".join(f"- {k}: [{lo}, {hi}]" for k, (lo, hi) in filtered_bounds.items())
    rw_lines = "\n".join(f"- {k}: [{lo}, {hi}]" for k, (lo, hi) in rw_bounds.items())
    algo_upper = algo_name.upper()
    valid_fields = ", ".join(valid_hp)
    hp_guide = _ALGO_HP_GUIDES.get(algo_name, "")
    return (
        "You are the training advisor agent for SwingRL, an RL-based "
        "swing trading system.\n\n"
        f"You are advising hyperparameters for the {algo_upper} algorithm.\n"
        f"ONLY include these hyperparameter fields: {valid_fields}\n\n"
        "SwingRL context:\n"
        "- Two environments: equity daily (8 ETFs) and crypto 4H (BTC/ETH)\n"
        f"- Algorithm: {algo_upper} ("
        f"{'on-policy' if algo_name in ('ppo', 'a2c') else 'off-policy, entropy-maximizing'}"
        f")\n"
        "- Capital preservation is the PRIMARY constraint — Sortino ratio and "
        "MDD are the main metrics\n"
        "- Market regimes: bull (0), bear (1), crisis (2) — detected by HMM "
        "from FRED indicators\n\n"
        f"Hyperparameter bounds for {algo_upper} (you MUST stay within these):\n{hp_lines}\n\n"
        "Reward weight bounds (you MUST stay within these, weights should "
        f"sum to ~1.0):\n{rw_lines}\n\n"
        f"{_HALLUCINATION_GUARD}\n\n"
        f"{hp_guide}\n\n"
        f"{_RUN_CONFIG_INSTRUCTIONS}"
    )
```

### 3.3 `_build_epoch_system_prompt` — lines 698–741 (verified)

```python
def _build_epoch_system_prompt(
    hp_bounds: dict[str, tuple[Any, Any]],
    rw_bounds: dict[str, tuple[float, float]],
    algo_name: str,
) -> str:
    """Build algo-specific system prompt for mid-training epoch advice.

    Includes the algo HP guide, reward weight adjustment guide, and hallucination guards.

    Args:
        hp_bounds: Hyperparameter bounds dict (all algos).
        rw_bounds: Reward weight bounds dict.
        algo_name: Algorithm name ('ppo', 'a2c', 'sac').

    Returns:
        System prompt for epoch_advice requests.
    """
    rw_lines = "\n".join(f"- {k}: [{lo}, {hi}]" for k, (lo, hi) in rw_bounds.items())
    algo_upper = algo_name.upper()
    hp_guide = _ALGO_HP_GUIDES.get(algo_name, "")
    epoch_guide = _ALGO_EPOCH_GUIDES.get(algo_name, "")
    return (
        "You are the training advisor agent for SwingRL, an RL-based "
        "swing trading system.\n\n"
        f"You are advising mid-training reward weight adjustments for {algo_upper}.\n\n"
        "SwingRL context:\n"
        "- Two environments: equity daily (8 ETFs) and crypto 4H (BTC/ETH)\n"
        f"- Algorithm: {algo_upper} ("
        f"{'on-policy' if algo_name in ('ppo', 'a2c') else 'off-policy, entropy-maximizing'}"
        f")\n"
        "- Capital preservation is the PRIMARY constraint — Sortino ratio and "
        "MDD are the main metrics\n\n"
        "Reward weight bounds (you MUST stay within these, weights should "
        f"sum to ~1.0):\n{rw_lines}\n\n"
        "FOLD ADJUSTMENT HISTORY: The user message may include recent reward weight\n"
        "adjustments from THIS fold with measured sharpe/mdd deltas and effectiveness.\n"
        "Use this to avoid repeating ineffective adjustments. If a specific dimension\n"
        "change was ineffective, try a different dimension or keep current weights.\n\n"
        f"{_HALLUCINATION_GUARD}\n\n"
        f"{hp_guide}\n\n"
        f"{epoch_guide}\n\n"
        f"{_REWARD_WEIGHT_GUIDE}\n\n"
        "You MUST respond with a valid JSON object matching the schema."
    )
```

---

## 4. Reward Weight / Bounds Inventory and Divergences

### 4.1 DEFAULT_WEIGHTS (production runtime)

Source: `src/swingrl/memory/training/reward_wrapper.py`, lines 28–33 (verified).

```python
DEFAULT_WEIGHTS: dict[str, float] = {
    "profit": 0.50,
    "sharpe": 0.25,
    "drawdown": 0.15,
    "turnover": 0.10,
}
```

This is the starting state for every training run before any LLM advice is applied.

### 4.2 `_SAFE_DEFAULTS["reward_weights"]` (memory service cold-start)

Source: `services/memory/memory_agents/query.py`, lines 122–131 (verified).

```python
_SAFE_DEFAULTS: dict[str, Any] = {
    "learning_rate": 3e-4,
    "entropy_coeff": 0.01,
    "clip_range": 0.2,
    "n_epochs": 10,
    "batch_size": 64,
    "gamma": 0.99,
    "reward_weights": {"profit": 0.4, "sharpe": 0.35, "drawdown": 0.20, "turnover": 0.05},
    "rationale": "cold_start_defaults",
}
```

**DIVERGENCE:** `_SAFE_DEFAULTS["reward_weights"]` differs from `DEFAULT_WEIGHTS` on every
component. The memory service cold-start advice returns weights that contradict the
production runtime's starting weights:

| Component | DEFAULT_WEIGHTS (runtime) | `_SAFE_DEFAULTS` (memory service) | Delta |
|-----------|--------------------------|-----------------------------------|-------|
| profit    | 0.50                     | 0.40                              | −0.10 |
| sharpe    | 0.25                     | 0.35                              | +0.10 |
| drawdown  | 0.15                     | 0.20                              | +0.05 |
| turnover  | 0.10                     | 0.05                              | −0.05 |

On the first run (before enough patterns exist for the cold-start guard to pass), the
memory service returns `_SAFE_DEFAULTS` weights and the runtime applies them as an
adjustment delta against `DEFAULT_WEIGHTS` — silently shifting weights away from the
production baseline on the very first query. This is a latent incoherence; fix scope is
Stage 3.

### 4.3 `_FALLBACK_REWARD_BOUNDS` — duplicated in two files

Source A: `src/swingrl/memory/training/bounds.py`, lines 50–55 (verified).

```python
_FALLBACK_REWARD_BOUNDS: dict[str, tuple[float, float]] = {
    "profit": (0.10, 0.70),
    "sharpe": (0.10, 0.60),
    "drawdown": (0.05, 0.50),
    "turnover": (0.00, 0.20),
}
```

Source B: `services/memory/memory_agents/query.py`, lines 85–90 (verified).

```python
_FALLBACK_REWARD_BOUNDS: dict[str, tuple[float, float]] = {
    "profit": (0.10, 0.70),
    "sharpe": (0.10, 0.60),
    "drawdown": (0.05, 0.50),
    "turnover": (0.00, 0.20),
}
```

Values are identical today. The duplication is a maintenance risk (one could drift).
Deduplication is deferred to Stage 3.4 per plan.

---

## 5. Data Bugs (documented here; fixes deferred to later tasks)

### Bug A — `max_single_loss` unit mismatch (breaks CPS v2)

The `backtest_results.max_single_loss` column stores values in **dollars** (e.g., −19871.5).

The `FoldMetrics` TypedDict docstring in `src/swingrl/metrics/cps.py` (lines 59–61)
specifies it as a signed **fraction**:

```
max_single_loss: Largest single-trade loss as a signed fraction
    (e.g. -0.08 = lost 8% of equity on the worst trade). May be None
    for legacy rows; treated as 0.0 in the CPS computation.
```

The CPS v2 formula (line 12) applies the penalty as:

```
- 2.0·max(0, |max_single_loss| - 0.10)
```

When `max_single_loss = −19871.5` (dollars), the term becomes
`2.0 × max(0, 19871.5 − 0.10) = 39742.8` — completely dominating the score and
rendering CPS v2 meaningless for any fold with a real dollar loss stored. This bug
makes CPS v2 unreliable until the column is normalized. Fix is scoped to Task 7.

### Bug B — `sharpe_delta` written into `outcome_sharpe` column

Source: `src/swingrl/memory/training/epoch_callback.py`, line 580 (verified).

The `_adjustment_outcome_queue.append` call (lines 576–588) constructs a params list:

```python
[
    self._epoch,      # -> epoch_outcome
    sharpe_delta,     # -> outcome_sharpe   ← BUG: should be post-adjustment rolling_sharpe
    sharpe_delta,     # -> sharpe_delta     ← correct
    mdd_delta,
    effective,
]
```

The UPDATE statement at lines 285–291 maps positionally:
`epoch_outcome = %s, outcome_sharpe = %s, sharpe_delta = %s, mdd_delta = %s, effective = %s`

So `outcome_sharpe` receives `sharpe_delta` (a delta, e.g. +0.12) instead of the
post-adjustment absolute rolling Sharpe value. Any query that uses `outcome_sharpe` to
gauge post-adjustment performance is reading a delta, not a level. Fix is scoped to Task 7.

---

## 6. Iter 3–4 Harm Decomposition

### 6.1 Actual query output (verified against production DB)

```
 environment | iteration_number | is_control_fold | med_ret | max_mdd
-------------+------------------+-----------------+---------+---------
 crypto      |                3 | f               |  0.3749 |   0.483
 crypto      |                3 | t               |  0.1368 |   0.413
 crypto      |                4 | f               |  0.4624 |   0.463
 crypto      |                4 | t               |  0.3047 |   0.300
 equity      |                3 | f               |  0.0500 |   0.361
 equity      |                3 | t               |  0.0510 |   0.069
 equity      |                4 | f               |  0.0629 |   0.382
 equity      |                4 | t               |  0.0529 |   0.067
```

(`f` = treatment fold, `t` = control fold)

### 6.2 Worst-case single losses (from plan-time queries, as documented in plan)

- Equity treatment (iter 3–4): max single losses −2360 / −3714 (dollars)
- Equity control (iter 3–4): max single losses −422 / −555 (dollars)
- Note: these are dollar values; the unit-mismatch bug (§5A) means these cannot be
  directly compared to the CPS v2 threshold of −0.10.

### 6.3 Failure mode taxonomy

**Failure mode 1 — Tail blowups (dominant in equity, iter 3–4)**

Treatment max_mdd: equity iter 3 = 0.361, iter 4 = 0.382.
Control max_mdd: equity iter 3 = 0.069, iter 4 = 0.067.
Treatment MDD is 5–6× worse than control in equity. The LLM advice is inducing
catastrophic drawdown events that do not appear in control folds. Single-loss dollars
are 5–7× larger in treatment than control (−2360/−3714 vs −422/−555).

Diagnosis: the advice system pushes the agent toward larger position sizing or
drawdown-tolerating weight configurations, causing tail events that control folds avoid.

**Failure mode 2 — Trade-shy collapse (identified in iter 4–5)**

Crypto SAC median trades = 66 (p10=14, p25=29). The epoch payload carries no
`total_trades` signal, so the LLM cannot detect when an agent has stopped trading.
Consolidation patterns that say "reduce risk" or "damp participation" (see §7)
map to reward weight shifts that push the agent toward inactivity.

The advice system has no guardrail against advising into a zero-trade regime.
The epoch prompt for SAC explicitly acknowledges "SAC Symptom → Fix: Over-exploration
(random trades): Lower target_entropy" — but has no symmetric guard for the opposite
problem (under-trading / trade-shy collapse).

---

## 7. Representative Consolidation Patterns (iter 0–4 sample)

Query: `SELECT id, left(pattern_text, 400) FROM consolidations ORDER BY id LIMIT 20`
(20 rows returned; 5 representative patterns selected below — IDs 61, 62, 67, 69, 73)

---

**Pattern ID 61** (taxonomy: `single_disaster`)

> "SAC exhibits extreme rolling_mdd outliers (e.g., fold17: -940.3081, fold18: -5507.8952)
> in 8 of 22 folds, with outlier rates up to 24% (fold8: 83/759), suggesting
> regime-dependent instability despite reward IQM improvements in late training."

Annotation: The mdd values here (-940, -5507) are rolling_mdd in dollar terms during
training (not OOS fraction), but the pattern text does not clarify units. A downstream
LLM reading this pattern will infer "catastrophic drawdown" and apply
**heavy drawdown-weight increases** or **trade-frequency damping**. Both responses map
to the `single_disaster` taxonomy (position reduction → concentration risk, or
drawdown-weight overshoot → reward signal dominated by MDD term).

Harm hypothesis cross-reference: The 5–7× equity single-loss amplification in iter 3–4
treatment vs control (§6.2) is consistent with a reward system that over-rotated into
drawdown-weight increases triggered by patterns like this one — eventually causing the
agent to hold large losing positions rather than exit early.

---

**Pattern ID 62** (taxonomy: `trade_shy`)

> "Reward weight adjustments are absent across all algorithms and folds (0/69 folds
> show changes), despite consistent mdd worsening trends in 38 of 69 folds, indicating
> static reward shaping may be suboptimal for drawdown control."

Annotation: This pattern tells the LLM "the system never adjusts weights, yet MDD is
worsening — therefore you should adjust weights." It is a blanket license to intervene,
with no success-rate evidence attached. Any LLM reading this will feel justified making
drawdown-weight increases on every fold. The lack of a trade-rate counter-signal (nothing
says "but increasing drawdown weight in prior tests caused trade-shy collapse") makes this
pattern a reliable trigger for `trade_shy` induction.

Harm hypothesis: The cold-start iterations (iter 0–2) had no control folds and no
pattern-effectiveness tracking. The first consolidation pass saw "MDD worsening and zero
adjustments" and generated this blanket-license pattern. Later iterations then acted on
it, driving the iter 3–4 treatment MDD from 0.069 (control) to 0.361–0.382 (treatment).

---

**Pattern ID 69** (taxonomy: `trade_shy`)

> "Reward weight adjustments are absent across all algorithms and folds (0 changes in 42
> folds), yet A2C and SAC show statistically significant improvements in mdd trends
> without explicit drawdown penalty increases. This suggests intrinsic drawdown recovery
> mechanisms in these algorithms for crypto environments."

Annotation: This pattern is the logical *opposite* of Pattern 62 — it says "A2C and SAC
fix their own MDD without help." In isolation it is beneficial. But because the pattern
corpus also contains Pattern 62 (and 73, below) as competing instructions, the LLM faces
contradictory evidence. In practice the patterns that advocate intervention (62, 73)
outnumber the patterns that advocate restraint (69, 71), so intervention dominates.

Harm hypothesis: The ratio of "intervene" to "leave alone" patterns in the iter 0–2 corpus
structurally biases the LLM toward reward-weight adjustment, even for algo/env combinations
where control folds consistently outperform treatment.

---

**Pattern ID 73** (taxonomy: `trade_shy`, `churning`)

> "Reward weight adjustments are universally absent across all algorithms and folds in both
> equity (0/69 folds) and crypto (0/42 folds) environments, despite evidence of drawdown
> worsening or recovery trends. This indicates a cross-environment reliance on static reward
> shaping, which may be suboptimal for dynamic drawdown control."

Annotation: Cross-environment repeat of Pattern 62. The phrase "cross-environment reliance
on static reward shaping" frames inaction as a systematic flaw. A downstream LLM will
weight this highly (cross-environment evidence is treated as stronger than single-env).
This is the highest-risk pattern in the corpus for inducing `trade_shy` or `churning`
collapses — it licenses aggressive cross-environment weight changes with no success-rate
guard. Must be retired in Group D.

Harm hypothesis: Iter 3–4 equity treatment max_mdd = 0.361–0.382 vs control 0.069–0.067
(§6.1). The 5× divergence is consistent with a cross-environment weight-adjustment mandate
operating without any feedback loop on whether previous adjustments improved or worsened
outcomes.

---

**Pattern ID 67** (taxonomy: `poor_selection`)

> "A2C and SAC algorithms exhibit consistent recovery trends in rolling_mdd (maximum
> drawdown) during training, with positive trend slopes in 10 of 14 A2C folds (e.g.,
> fold1: +0.0003, fold10: +0.0001) and 4 of 10 SAC folds (e.g., fold2: +0.0005,
> fold4: +0.0004). Trajectory analysis shows mid->late improvement in mdd for 12 of
> 14 A2C folds (e.g., fold1: -31.2717 -> -33.3356) and 5 of 10 SAC folds..."

Annotation: The mdd trajectory values here (-31.2, -33.3) are again training-time
rolling_mdd in dollar terms, not OOS fractions. A downstream LLM that does not track
units will interpret "-31 to -33" as worsening (more negative = worse MDD), when the
actual interpretation depends on whether rolling_mdd is stored as a signed dollar running
sum or a fraction. This unit confusion in pattern text is a `poor_selection` failure
mode — the pattern selects misleading evidence due to unit mismatch in the stored
metrics.

Harm hypothesis: If the LLM consistently misreads "mid-late mdd trajectory" as worsening
when it is actually improving, it will trigger unnecessary drawdown-weight interventions
in folds that are self-correcting. Cross-reference: Pattern 69 correctly identifies A2C
MDD self-correction, but Pattern 67's numeric framing obscures this conclusion.

---

## 8. Raw Query Outputs (Section 1 Data Tables)

### Trade distribution by env/algo

```
 environment | algorithm | count | p10 | p25 | med | p75 |  p90 | med_wr | p25_wr
-------------+-----------+-------+-----+-----+-----+-----+------+--------+--------
 crypto      | a2c       |    70 |  67 | 160 | 376 | 619 |  777 |  0.576 |  0.490
 crypto      | ppo       |    70 | 913 | 943 | 974 | 996 | 1013 |  0.615 |  0.576
 crypto      | sac       |    70 |  14 |  29 |  66 | 225 |  403 |  0.425 |  0.232
 equity      | a2c       |   121 |  99 | 180 | 311 | 381 |  419 |  0.647 |  0.480
 equity      | ppo       |   118 | 430 | 446 | 458 | 469 |  478 |  0.644 |  0.562
 equity      | sac       |   115 |  59 | 100 | 171 | 256 |  440 |  0.674 |  0.479
```

All values match plan-time expectations exactly.

### Median total return by environment

```
 environment | med_ret
-------------+---------
 crypto      |  0.3893
 equity      |  0.0535
```

**Note:** Crypto `med_ret = 0.3893` differs materially from the plan estimate of `0.70`.
Equity `med_ret = 0.0535` matches the estimate of `0.053` (within rounding). See Task 2
flag at the top of this document.
