# Phase 19.1 — C6 Post-Change Documentation (After Picture)

**Purpose:** Documents the LLM advice system *after* all Stage 2 Group C code changes
(Tasks 2–11 on `swingrl/19.1-training-refocus`). This is the "after picture" that mirrors
the C0 baseline (`.planning/research/phase-19.1-prompt-baseline.md`) section-for-section.
Together they are the QA reference for Group D (pattern regeneration) and the control-arm
audit for Group E (training run).

**Spec authority:** `docs/superpowers/specs/2026-06-11-stage2-training-refocus-design.md`
(referred to below as "the spec"). Section numbers (§2, §4.1, etc.) refer to that document.

**Do not modify this file** once committed — it is a historical record of the Group C
completion state.

---

## 1. New Prompt Blocks

Source: `services/memory/memory_agents/query.py`, lines 652–718 (verified).

Five new module-level string constants were added in Task 11 (C3). Two are spliced into
the epoch builder; two into the run-config builder; one (`_GOAL_BLOCK`) is spliced into
**both** builders.

---

### 1.1 `_GOAL_BLOCK` (epoch + run-config)

```python
_GOAL_BLOCK = (
    "YOUR OBJECTIVE:\n"
    "Your single objective metric is CPS v1 (Capital Preservation Score):\n"
    "  CPS v1 = median_return x (1 - max_mdd)^2 x tanh(mean_winner_sharpe / 2) x pass_ratio\n"
    "It is MULTIPLICATIVE: collapsing any factor - including profit - kills your score.\n"
    "Reducing trading activity to reduce risk LOWERS your score: a return collapse from "
    "8% to 2% costs ~4x while the paired drawdown improvement earns only ~1.25x back.\n"
    "Pass rate is NOT your goal. Risk-damping is NOT automatically safe.\n\n"
)
```

**Why it exists:** Before C3, the string `cps` appeared zero times in the advice path
(verified in the spec §1). The LLM had no stated objective, so it defaulted to local
risk-damping — the "safe" choice under any generic capital-preservation framing, but
catastrophic under CPS v1's multiplicative form where profit collapse costs ~4× the
corresponding drawdown gain. The goal block makes the formula explicit so the LLM knows
that trade-shyness lowers its score.

**Splice point in `_build_epoch_system_prompt`:** inserted between the SwingRL context
section and the antipattern block (before reward-weight bounds).

**Splice point in `_build_algo_system_prompt`:** inserted between the SwingRL context
section and the antipattern block (before the HP bounds and hallucination guard).

---

### 1.2 `_ANTIPATTERN_BLOCK_EPOCH` (epoch builder only)

```python
_ANTIPATTERN_BLOCK_EPOCH = (
    "EMPIRICAL ANTI-PATTERNS (from this system's own history):\n"
    "Across iterations 3-4, folds receiving LLM advice scored 2.7x-5.1x WORSE CPS than "
    "control folds with no advice. Two observed failure modes:\n"
    "1. Tail blowups (iter 3-4): advised folds reached max drawdown 0.36-0.38 vs 0.067-0.069 "
    "for controls - adjustments amplified worst-case risk.\n"
    "2. Trade-shy collapse (iter 4-5): advised folds cut trading, suppressing profit AND risk "
    "together, collapsing CPS (e.g. A2C returns fell 6.0% -> 4.9% while pass rate rose).\n"
    "Your payload includes a deterministic context JSON with a 'diagnosis' field. Match your "
    "advice to the labeled cause:\n"
    "- trade_shy -> increase participation; do NOT damp risk further\n"
    "- poor_selection -> tighten entry quality; reduce frequency, not size\n"
    "- single_disaster -> cap per-trade risk; leave frequency alone\n"
    "- churning -> reduce frequency; raise conviction threshold\n"
    "- healthy -> no adjustment warranted\n"
    "If diagnosis confidence is 'mixed', be conservative: prefer no change.\n\n"
)
```

**Shape difference from run-config variant:** references the `diagnosis` field in the
per-epoch `context` JSON (mid-fold, single fold). The run-config variant references
`prev_iter_diagnoses` (a map of fold_number → label from the previous full iteration).

---

### 1.3 `_ANTIPATTERN_BLOCK_RUNCONFIG` (run-config builder only)

```python
_ANTIPATTERN_BLOCK_RUNCONFIG = (
    "EMPIRICAL ANTI-PATTERNS (from this system's own history):\n"
    "Across iterations 3-4, folds receiving LLM advice scored 2.7x-5.1x WORSE CPS than "
    "control folds with no advice. Two observed failure modes:\n"
    "1. Tail blowups (iter 3-4): advised folds reached max drawdown 0.36-0.38 vs 0.067-0.069 "
    "for controls - adjustments amplified worst-case risk.\n"
    "2. Trade-shy collapse (iter 4-5): advised folds cut trading, suppressing profit AND risk "
    "together, collapsing CPS (e.g. A2C returns fell 6.0% -> 4.9% while pass rate rose).\n"
    "Your payload includes prev_iter_diagnoses, a map of fold_number -> diagnosis label from "
    "the previous iteration. Match each fold's advice to its labeled cause:\n"
    "- trade_shy -> increase participation; do NOT damp risk further\n"
    "- poor_selection -> tighten entry quality; reduce frequency, not size\n"
    "- single_disaster -> cap per-trade risk; leave frequency alone\n"
    "- churning -> reduce frequency; raise conviction threshold\n"
    "- healthy -> no adjustment warranted\n"
    "If a fold's diagnosis was unavailable, be conservative for that fold: prefer no change.\n\n"
)
```

**Shape difference from epoch variant:** references `prev_iter_diagnoses` (a
`dict[fold_number_as_str → label]` assembled pre-iteration from the previous iteration's
`backtest_results` rows). The epoch variant references the single-fold live `diagnosis`
field delivered in real time.

---

### 1.4 `_FOLD_PROTECTION_BLOCK_EPOCH` (epoch builder only)

```python
_FOLD_PROTECTION_BLOCK_EPOCH = (
    "FOLD PROTECTION:\n"
    "The context JSON includes fold_role and the chronic_failure/protected_winner fold lists.\n"
    "- fold_role=protected_winner: return the current weights UNCHANGED. This fold wins "
    "without intervention; history shows intervention degrades it.\n"
    "- fold_role=chronic_failure: regime-conditional shaping is allowed; you may lean toward "
    "drawdown emphasis, always within the stated bounds.\n"
    "- fold_role=neutral: adjust only with clear diagnosis-backed cause.\n\n"
)
```

---

### 1.5 `_FOLD_PROTECTION_BLOCK_RUNCONFIG` (run-config builder only)

```python
_FOLD_PROTECTION_BLOCK_RUNCONFIG = (
    "FOLD PROTECTION:\n"
    "The context JSON includes the chronic_failure_folds and protected_winner_folds lists "
    "(by fold number).\n"
    "- Folds in protected_winner_folds: return the current weights UNCHANGED for these folds. "
    "They win without intervention; history shows intervention degrades them.\n"
    "- Folds in chronic_failure_folds: regime-conditional shaping is allowed; you may lean "
    "toward drawdown emphasis, always within the stated bounds.\n"
    "- All other folds: adjust only with clear cause from prev_iter_diagnoses.\n\n"
)
```

**Shape difference from epoch variant:** the epoch variant uses singular `fold_role` (a
pre-classified string for the one fold currently training). The run-config variant uses
two lists (`chronic_failure_folds`, `protected_winner_folds`) because the run-config
advice covers all upcoming folds simultaneously, not a single fold.

---

### 1.6 Builder splice order (verified against `_build_epoch_system_prompt` and `_build_algo_system_prompt`)

**Epoch builder** (`_build_epoch_system_prompt`, lines 772–818):
```
role preamble → SwingRL context → _GOAL_BLOCK → _ANTIPATTERN_BLOCK_EPOCH
→ _FOLD_PROTECTION_BLOCK_EPOCH → reward-weight bounds → fold-adjustment history
→ _HALLUCINATION_GUARD → hp_guide → epoch_guide → _REWARD_WEIGHT_GUIDE → schema instruction
```

**Run-config builder** (`_build_algo_system_prompt`, lines 721–769):
```
role preamble → SwingRL context → _GOAL_BLOCK → _ANTIPATTERN_BLOCK_RUNCONFIG
→ _FOLD_PROTECTION_BLOCK_RUNCONFIG → HP bounds → reward-weight bounds
→ _HALLUCINATION_GUARD → hp_guide → _RUN_CONFIG_INSTRUCTIONS
```

---

## 2. New Payload Schemas

### 2.1 Epoch context JSON

Source: `src/swingrl/memory/training/epoch_callback.py`, lines 662–685 (verified).

**Keys constructed:**

```python
context = {
    "fold_number": self._fold_number,
    "fold_role": self._fold_context["fold_role"],
    "prev_iter_cps_v1": self._fold_context["prev_iter_cps_v1"],
    "target_metric": "cps_v1_multiplicative",
    "leading_indicators": {
        "rolling_sharpe": round(self._wrapper.rolling_sharpe(), 4),
        "rolling_mdd": round(self._wrapper.rolling_mdd(), 4),
        "rolling_win_rate": round(self._wrapper.rolling_win_rate(), 4),
        "trade_rate": round(self._wrapper.rolling_trade_rate(), 4),
        "baseline_trade_rate": round(self._wrapper.baseline_trade_rate(), 4),
    },
    "diagnosis": diagnosis,
}
```

The assembled query string (line 678–685):
```python
payload = {
    "query": (
        f"EPOCH ADVICE: run_id={self._run_id} algo={self._algo} "
        f"env={self._env} epoch={self._epoch}{iter_part} "
        f"current_weights={_json.dumps(self._wrapper.weights)} "
        f"context={_json.dumps(context, allow_nan=False)}"
    )
}
```

**Realistic example** (equity SAC, fold 3, mid-fold iter 6):

```json
{
  "fold_number": 3,
  "fold_role": "neutral",
  "prev_iter_cps_v1": 0.0149,
  "target_metric": "cps_v1_multiplicative",
  "leading_indicators": {
    "rolling_sharpe": 0.8124,
    "rolling_mdd": -0.0312,
    "rolling_win_rate": 0.5340,
    "trade_rate": 0.0081,
    "baseline_trade_rate": 0.0171
  },
  "diagnosis": {
    "label": "trade_shy",
    "fired": ["trade_shy"],
    "confidence": "clear",
    "evidence": {
      "trade_rate": 0.0081,
      "baseline_trade_rate": 0.0171
    }
  }
}
```

**Fail-open paths (all verified in code):**

- Unknown `(env, algo)` → `diagnose_rolling()` raises `DataError` → caught at line 653–660
  → `diagnosis` defaults to `{"label": "healthy", "fired": [], "confidence": "clear", "evidence": {}}`.
- NaN in `trade_rate`, `baseline_trade_rate`, or `rolling_win_rate` → `diagnose_rolling()`
  raises `DataError` → same healthy-fallback path.
- DB unavailable or `fold_number=None` → `load_fold_context()` returns the neutral dict
  (`fold_role="neutral"`, `chronic_failure_folds=[]`, `protected_winner_folds=[]`,
  `prev_iter_cps_v1=None`).
- Any NaN in `context` → serialization via `json.dumps(context, allow_nan=False)` raises
  `ValueError` → caught by the outer `except Exception` at line 806 → `_advice_timed_out`
  incremented, advice skipped for this epoch, training continues.
- The fold context is lazy-loaded once per fold and cached in `self._fold_context`. On the
  first epoch query, a 5-second DB timeout prevents stalling training.

**C0 comparison — what changed:**

| Field | C0 payload | C6 payload |
|-------|-----------|-----------|
| `rolling_sharpe` | Bare f-string field | Inside `leading_indicators` |
| `rolling_mdd` | Bare f-string field | Inside `leading_indicators` |
| `total_trades` | **Missing** | Represented via `trade_rate` + `baseline_trade_rate` |
| `win_rate` | **Missing** | Inside `leading_indicators.rolling_win_rate` |
| `fold_number` | **Missing** | Present |
| `fold_role` | **Missing** | Present (`neutral` / `chronic_failure` / `protected_winner`) |
| `diagnosis` | **Missing** | Present (label + fired rules + confidence + evidence) |
| `target_metric` | **Missing** | `"cps_v1_multiplicative"` |
| `prev_iter_cps_v1` | **Missing** | Float from `iteration_results` (null on iter 0) |

---

### 2.2 Run-config context JSON

Source: `src/swingrl/memory/training/meta_orchestrator.py`, lines 300–348 (verified).

**Keys constructed:**

```python
context: dict[str, Any] = {"target_metric": "cps_v1_multiplicative"}
# then populated from load_fold_context + fold_history if database_url present:
context["chronic_failure_folds"] = ctx["chronic_failure_folds"]
context["protected_winner_folds"] = ctx["protected_winner_folds"]
context["prev_iter_cps_v1"] = ctx["prev_iter_cps_v1"]
context["prev_iter_diagnoses"] = diagnoses  # dict[int, str] fold → label
```

The assembled query string (lines 341–347):
```python
payload = {
    "query": (
        f"TRAINING RUN CONFIG ADVICE: env={env_name} algo={algo_name}{iter_part} "
        f"current_regime={json.dumps(regime)} "
        f"context={json.dumps(context, allow_nan=False)}"
    )
}
```

**Realistic example** (equity PPO, iteration 6):

```json
{
  "target_metric": "cps_v1_multiplicative",
  "chronic_failure_folds": [2, 5],
  "protected_winner_folds": [7],
  "prev_iter_cps_v1": 0.0407,
  "prev_iter_diagnoses": {
    "0": "healthy",
    "2": "single_disaster",
    "3": "trade_shy",
    "5": "churning",
    "7": "healthy"
  }
}
```

**Fail-open paths (all verified in code):**

- `database_url` absent → `context` stays `{"target_metric": "cps_v1_multiplicative"}` only.
- Any DB/context assembly error → `except Exception` at line 338 logs `run_config_context_failed`
  → same minimal `{"target_metric": "cps_v1_multiplicative"}` fallback.
- A fold's `backtest_results` row has a `None` or `NaN` in `mdd`, `win_rate`, `total_trades`,
  `profit_factor`, or `total_return` → `diagnose_fold()` raises `DataError` → the fold is
  skipped from `prev_iter_diagnoses` (per-fold try/except at lines 328–334), not fatal.
- Any NaN that survives into `context` → `json.dumps(context, allow_nan=False)` raises
  `ValueError` → caught by outer `except Exception` at line 373 → `_query_run_config`
  returns `{}` → meta-orchestrator uses baseline HPs.

**C0 comparison — what changed:**

| Field | C0 payload | C6 payload |
|-------|-----------|-----------|
| `current_regime` | Present | Present (unchanged) |
| `target_metric` | **Missing** | `"cps_v1_multiplicative"` |
| `chronic_failure_folds` | **Missing** | List of fold numbers |
| `protected_winner_folds` | **Missing** | List of fold numbers |
| `prev_iter_cps_v1` | **Missing** | Float or null |
| `prev_iter_diagnoses` | **Missing** | Dict fold_number→label |

---

## 3. Reward Weights: UNCHANGED by Design

`DEFAULT_WEIGHTS` in `src/swingrl/memory/training/reward_wrapper.py` (lines 28–33) is:

```python
DEFAULT_WEIGHTS: dict[str, float] = {
    "profit": 0.50,
    "sharpe": 0.25,
    "drawdown": 0.15,
    "turnover": 0.10,
}
```

**This was deliberately not changed.** Spec decision D2 (§2) states:

> "C4 (base reward-weight rebalance) is DROPPED. `DEFAULT_WEIGHTS` stays
> `{profit 0.50, sharpe 0.25, drawdown 0.15, turnover 0.10}`. The control fold — the
> empirical *winner* — runs on these exact weights, untouched (control's early-return at
> `epoch_callback.py:602` never calls `update_weights`). The harm is the LLM *moving*
> weights/HPs, not the base values. Changing the base would also change the control baseline,
> turning Group E into a two-variable experiment."

The handoff's planned `CHRONIC_FAILURE_WEIGHTS` constant was not added as a code constant.
Instead, chronic-failure guidance lives exclusively in `_FOLD_PROTECTION_BLOCK_EPOCH` and
`_FOLD_PROTECTION_BLOCK_RUNCONFIG` — the LLM is told it may lean toward drawdown emphasis
for chronic-failure folds, but the steering happens within the existing clamp bounds,
without a separate hardcoded weight set.

The `_SAFE_DEFAULTS["reward_weights"]` divergence documented in C0 §4.1
(`{profit: 0.40, sharpe: 0.35, drawdown: 0.20, turnover: 0.05}` vs the runtime's
`{0.50, 0.25, 0.15, 0.10}`) is still present. Deduplication and convergence of these two
weight sets is deferred to Stage 3.4 per plan.

---

## 4. Diagnosis Taxonomy and Thresholds

Source: `src/swingrl/memory/training/cps_diagnosis.py` (verified, full file).

### 4.1 Five diagnosis labels

| Label | Detection condition | Paired correction (verbatim from `DIAGNOSIS_CORRECTIONS`) |
|-------|---------------------|------------------------------------------------------------|
| `trade_shy` | `total_trades < p25` AND `total_return < ENV_MEDIAN_RETURN` | `"increase participation; do NOT damp risk further"` |
| `poor_selection` | `total_trades >= p25` AND `win_rate < p25_win_rate` | `"tighten entry quality; reduce frequency, not size"` |
| `single_disaster` | `mdd > MDD_DISASTER_THRESHOLD[env]` AND `win_rate >= p25_win_rate` | `"cap per-trade risk; leave frequency alone"` |
| `churning` | `total_trades > p90` AND `profit_factor < 1.5` | `"reduce frequency; raise conviction threshold"` |
| `healthy` | None of the above fired | `"no adjustment warranted"` |

The code constant mirrors the prompt verbatim:

```python
DIAGNOSIS_CORRECTIONS: dict[DiagnosisLabel, str] = {
    "trade_shy": "increase participation; do NOT damp risk further",
    "poor_selection": "tighten entry quality; reduce frequency, not size",
    "single_disaster": "cap per-trade risk; leave frequency alone",
    "churning": "reduce frequency; raise conviction threshold",
    "healthy": "no adjustment warranted",
}
```

### 4.2 Rule precedence

`single_disaster > churning > trade_shy > poor_selection` (constant `_PRECEDENCE`).

Rationale: a tail blowup dominates CPS v1 via the squared drawdown factor — it outranks
activity anomalies. Selection quality is judged only when activity is normal.

When multiple rules fire, `confidence = "mixed"` and the precedence-resolved label is
returned. Mixed confidence → the prompt tells the LLM to be conservative and prefer no change.

### 4.3 Per-(env, algo) `TRADE_BASELINES` table

Derived from iter 0–4 `backtest_results` (564 folds). Each entry is a `TradeBaseline`
TypedDict with keys: `p10`, `p25`, `med`, `p75`, `p90`, `med_win_rate`, `p25_win_rate`.

| (env, algo) | p10 | p25 | med | p75 | p90 | med_wr | p25_wr |
|-------------|-----|-----|-----|-----|-----|--------|--------|
| crypto / a2c | 67 | 160 | 376 | 619 | 777 | 0.576 | 0.490 |
| crypto / ppo | 913 | 943 | 974 | 996 | 1013 | 0.615 | 0.576 |
| crypto / sac | 14 | 29 | 66 | 225 | 403 | 0.425 | 0.232 |
| equity / a2c | 99 | 180 | 311 | 381 | 419 | 0.647 | 0.480 |
| equity / ppo | 430 | 446 | 458 | 469 | 478 | 0.644 | 0.562 |
| equity / sac | 59 | 100 | 171 | 256 | 440 | 0.674 | 0.479 |

Per-(env, algo) is mandatory — SAC's median is 66 trades/fold (crypto) vs PPO's 974. A
shared threshold would mislabel SAC as perpetually trade-shy.

**Baseline staleness note:** the comment in `cps_diagnosis.py` says "Regenerate after each
completed training iteration via the percentile SQL in
`.planning/research/phase-19.1-prompt-baseline.md` §8; stale baselines mislabel as the
policy improves." This is a live operational concern; Group D does not regenerate them.

### 4.4 `MDD_DISASTER_THRESHOLD` and `ENV_MEDIAN_RETURN`

```python
MDD_DISASTER_THRESHOLD: dict[str, float] = {"equity": 0.20, "crypto": 0.40}
ENV_MEDIAN_RETURN: dict[str, float] = {"equity": 0.0535, "crypto": 0.3893}
```

Thresholds sit between the control-fold ceiling (equity 0.069, crypto 0.413) and the
treatment blowup peaks (equity 0.361–0.382, crypto 0.463–0.483) documented in C0 §6.1.

`ENV_MEDIAN_RETURN` values are the actual production medians from the C0 baseline query
output (§8 of C0 doc). The crypto estimate in the original handoff plan was 0.70;
the actual value is 0.3893, materially lower — this is why the C0 doc carries the "Task 2
flag" notice and why 0.3893 is used here.

### 4.5 `None`/`NaN` → `DataError` policy

`_validate_fold_fields()` raises `DataError` for any of the five consumed fields
(`mdd`, `win_rate`, `profit_factor`, `total_return`, `total_trades`) that is `None` or
`NaN`. This is a typed refusal: a `None` or `NaN` value would silently produce a false
label or raise an untyped `TypeError` — both worse than rejecting the row. Callers catch
`DataError` and fall back to no-diagnosis rather than receiving a false label.

### 4.6 Mid-fold self-baseline design (`diagnose_rolling`)

The mid-fold variant (`diagnose_rolling`) uses only `trade_shy` and `poor_selection`
(the two detectable mid-fold). `single_disaster` and `churning` require completed
backtest rows.

The critical design: `baseline_trade_rate` is the **fold's own first-full-window rate**
provided by `MemoryVecRewardWrapper.baseline_trade_rate()`. This is locked at the first
completed rolling window within the fold — it is not a cross-fold average. Two consequences:

1. No cross-fold steps-per-trade conversion is needed — the rate is comparable within
   the same fold's time scale.
2. If `baseline_trade_rate == 0.0` (the window has not filled yet — early in the fold),
   the `trade_shy` rule is **disabled** for that epoch. The code:

   ```python
   if (
       baseline_trade_rate > 0.0
       and trade_rate < TRADE_RATE_COLLAPSE_FRACTION * baseline_trade_rate
   ):
       fired.append("trade_shy")
   ```

   `TRADE_RATE_COLLAPSE_FRACTION = 0.5` (constant in `cps_diagnosis.py`). A trade rate
   that falls to less than half the fold's own baseline triggers the rule.

---

## 5. Attribution Loop

Source: `src/swingrl/data/postgres_schema.py` (DDL), `src/swingrl/memory/training/epoch_callback.py` (writers), `src/swingrl/memory/training/fold_context.py` (`record_fold_attribution`).

### 5.1 Six new `reward_adjustments` columns

The six attribution columns added in Task 7 (C5):

| Column | Type | Written at | Value |
|--------|------|------------|-------|
| `fold_number` | `INTEGER` | Trigger flush | Fold index from `self._fold_number` |
| `iteration_number` | `INTEGER` | Trigger flush | `self._iteration` |
| `advice_id` | `TEXT` | Trigger flush | UUID v4, unique per accepted advice call |
| `fold_cps_v1_before` | `DOUBLE PRECISION` | Trigger flush | Most recent `cps_v1_multiplicative` from `iteration_results` at the time of the call; NULL on iter 0 |
| `fold_cps_v1_after` | `DOUBLE PRECISION` | `record_fold_attribution()` post-fold | Single-fold CPS v1 from the completed backtest row |
| `advice_was_effective` | `BOOLEAN` | `record_fold_attribution()` post-fold | `fold_cps_v1_after > fold_cps_v1_before`; NULL when `fold_cps_v1_before IS NULL` |

### 5.2 `advice_id` — unique per accepted advice call

`self._advice_id = str(uuid.uuid4())` is set exactly when a weight change passes all
guards (cooldown, max-delta, change-detection) and is written to the wrapper (line 796 of
`epoch_callback.py`). A UUID is generated per accepted call; multiple accepted calls in
the same fold get distinct UUIDs. The UUID is written into `reward_adjustments.advice_id`
at trigger flush time.

### 5.3 `fold_cps_v1_before` at advice time

The value is read from `self._fold_context["prev_iter_cps_v1"]`, which is the most recent
`cps_v1_multiplicative` row from `iteration_results` for this env — loaded once per fold
at first epoch-advice query. This is the "before" baseline: the CPS the system achieved in
the previous complete iteration before this advice was given.

### 5.4 Post-fold closure via `record_fold_attribution`

Source: `src/swingrl/memory/training/fold_context.py`, lines 125–160.

After the fold's backtest completes:

```python
def record_fold_attribution(conn: Any, run_id: str, fold: dict[str, Any]) -> None:
    cps_after = compute_cps_v1_multiplicative([fold_metrics])
    with conn.cursor() as cur:
        cur.execute(
            "UPDATE reward_adjustments "
            "SET fold_cps_v1_after = %s, "
            "    advice_was_effective = CASE WHEN fold_cps_v1_before IS NULL "
            "        THEN NULL ELSE %s > fold_cps_v1_before END "
            "WHERE run_id = %s",
            (cps_after, cps_after, run_id),
        )
```

All `reward_adjustments` rows with this `run_id` get `fold_cps_v1_after` set. Rows with
`fold_cps_v1_before IS NULL` (iter 0, no prior CPS) get `advice_was_effective = NULL` —
NULL-safe effectiveness, no false positives on the cold start.

### 5.5 `fold_run_id` single-source-of-truth helper

The `run_id` written at trigger flush (`self._run_id`) is the same run_id used in
`record_fold_attribution`. It is assembled once per fold by the meta-orchestrator at
`self._generate_run_id(env_name, algo_name)` and threaded through the callback's
`__init__`. There is no separate "fold_run_id" lookup — the same `run_id` serves both
trigger writes (at flush) and the attribution closure (post-fold).

### 5.6 `outcome_sharpe` bug fix

The C0 baseline (§5, Bug B) documented that `outcome_sharpe` was receiving `sharpe_delta`
(a delta) instead of the post-adjustment absolute rolling Sharpe. This was fixed in Task 7:
the outcome queue now correctly writes `current_sharpe` (the post-outcome absolute value)
into `outcome_sharpe` position 2 (index 1 of the params list), and `sharpe_delta` into
position 3 (index 2). Verified against lines 592–603 of `epoch_callback.py`:

```python
self._adjustment_outcome_queue.append(
    (
        [
            self._epoch,
            current_sharpe,   # -> outcome_sharpe  (FIXED: was sharpe_delta)
            sharpe_delta,     # -> sharpe_delta     (correct)
            mdd_delta,
            effective,
        ],
        self._run_id,
        adj["epoch_triggered"],
    )
)
```

---

## 6. Diff Summary Table

Each row is one change (C0 → C6) with the expected CPS mechanism.

| Change | C0 state | C6 state | Expected CPS mechanism |
|--------|----------|----------|------------------------|
| **Goal block in both prompts** | No objective stated; LLM defaults to risk-damping | Formula stated explicitly; multiplicative penalty for trade-shy explained | Counters failure mode 2 (trade-shy collapse): LLM now knows that cutting trades lowers its score, not just risk |
| **Antipattern blocks** | No empirical history shared | Iter 3-4 CPS ratios (2.7×–5.1×) stated; failure modes 1 and 2 spelled out | LLM anchors to observed harm rather than a theoretically "safe" but empirically destructive prior |
| **Diagnosis in epoch payload** | No trade signal; no cause label | `diagnosis.label` + `diagnosis.confidence` + `evidence` in `context` JSON | Prevents opposite-advice failures: a `trade_shy` fold gets "increase participation" not "damp risk"; eliminates the main source of opposite-advice cited in spec §1 |
| **Fold protection blocks** | No fold role; LLM advises all folds equally | `fold_role` (epoch) + `protected_winner_folds` list (run-config) | Stops the LLM adjusting winners; the iter 3-4 tail-blowup channel on protected folds is removed (intervention on a winning fold cannot decrease it if the fold is gated out) |
| **Prev-iter diagnoses in run-config** | Regime vector only | Regime + per-fold diagnosis labels from prior iteration | Gives the run-config LLM a causal breakdown before HP advice; prevents it recommending the same HP changes that led to `single_disaster` folds in the previous iteration |
| **`trade_rate` + `baseline_trade_rate` in epoch payload** | No trade signal at all | Self-baseline rate (fold's first window) vs current rate | The only signal that detects trade-shy collapse mid-fold (CPS catches it post-fold; MDD looks fine while the agent stops trading) |
| **Attribution columns in `reward_adjustments`** | `fold_number`, `iteration_number`, `advice_id`, `fold_cps_v1_before/after`, `advice_was_effective` missing | All six columns present; `record_fold_attribution` wires before/after CPS per accepted advice call | By iter 6 we can measure which advice categories helped. Group E training run produces the first population of attributed rows with non-null `fold_cps_v1_before` |
| **`outcome_sharpe` fix** | `outcome_sharpe` stored `sharpe_delta` (a delta, not a level) | `outcome_sharpe` stores the post-outcome absolute rolling Sharpe | Fixes the column so future per-fold attribution queries that use `outcome_sharpe` to gauge post-adjustment absolute performance are reading a level, not a delta |
| **`DEFAULT_WEIGHTS` unchanged (D2)** | `{0.50, 0.25, 0.15, 0.10}` | Same — unchanged by design | Control folds run on these exact weights; Group E is a single-variable experiment (prompt changes only, not base weights); changing the base would make the control arm incomparable |

**Honest uncertainty:** the above mechanisms are the expected pathways. Whether they are
*sufficient* to close the 2.7–5.1× gap is unknown before Group E runs. The control arm
(unchanged from iter 3-4 weights and behavior) remains the arbiter — if treatment still
underperforms control by the end of iter 6, that is informative: it means the failure mode
is not fully captured by these five changes, or is caused by factors not yet modeled (e.g.,
stale consolidation patterns, HP advice quality, replay-buffer staleness in SAC).

---

## 7. Group D QA Criteria

Group D's task is to retire the iter 0–4 harmful patterns and regenerate them against the
new prompt blocks and payload context. QA must verify the following:

### 7.1 Regenerated patterns must reflect the CPS goal block

Every new consolidation pattern that pertains to reward adjustment must reference CPS
(or the multiplicative profit × drawdown × Sharpe structure) as the objective. A pattern
that says "increase drawdown weight to reduce MDD" without acknowledging that reduced
trading collapses the profit factor of CPS v1 is a regression to the C0 failure mode.

**Test question:** do the new patterns cite the CPS formula terms, or do they fall back to
"maximize Sharpe / minimize MDD" as separate objectives?

### 7.2 Must not reproduce C0-annotated harmful patterns

The following five patterns (IDs from C0 §7) must not appear verbatim or in semantically
equivalent form in the regenerated corpus:

| Pattern ID | Taxonomy | What to check |
|------------|----------|---------------|
| **61** | `single_disaster` | No pattern should recommend drawdown-weight increases based on dollar-denominated rolling MDD values from training (which are not portfolio fractions). The unit confusion is the diagnostic. |
| **62** | `trade_shy` | No pattern should license blanket reward-weight increases because "adjustments are absent and MDD is worsening." Any pattern recommending intervention must cite a diagnosis label, not just MDD trend. |
| **67** | `poor_selection` | No pattern should cite mid-training rolling MDD trajectory values (e.g., −31 → −33) as evidence of worsening performance without clarifying that these are dollar-denominated training metrics, not OOS fraction. |
| **69** | `trade_shy` (beneficial, but weak) | Patterns that correctly identify A2C/SAC self-correction should survive; the issue is that they must not be outnumbered by intervention-licensing patterns (62, 73). Check corpus balance. |
| **73** | `trade_shy` / `churning` | No cross-environment pattern should frame zero reward-adjustment as a "systematic flaw." The C3 prompt explicitly states that healthy folds warrant no adjustment. |

**Retirement SQL** (run against production DB before Group D consolidation):

```sql
-- Verify these IDs are 'retired' before regeneration
SELECT id, status, left(pattern_text, 100)
FROM consolidations
WHERE id IN (61, 62, 67, 69, 73);
```

Expected: `status = 'retired'` for all five rows.

### 7.3 Iter 3-4 patterns must surface the harm evidence

The regenerated Phase A consolidation patterns should include at least one pattern that:

- Cites the actual iter 3–4 treatment vs control CPS ratios (2.7×–5.1×) or the max_mdd
  divergence (treatment 0.36–0.38 vs control 0.067–0.069 in equity).
- Categorizes this as `iteration_regression` or `hp_effectiveness`.
- Maps the observation to the correct failure mode (tail blowup vs trade-shy, not a generic
  "drawdown control needed" instruction).

This pattern is critical for the run-config LLM: it needs to anchor HP advice to the
empirical harm record, not just the regime vector.

### 7.4 Diagnosis corrections in patterns must match the taxonomy

If any new pattern mentions a diagnosis label (`trade_shy`, `single_disaster`, etc.), the
paired correction must match the `DIAGNOSIS_CORRECTIONS` map verbatim or in intent:

| Label | Correct direction |
|-------|------------------|
| `trade_shy` | Increase participation — do NOT damp risk |
| `single_disaster` | Cap per-trade risk — leave frequency alone |
| `churning` | Reduce frequency — raise conviction threshold |
| `poor_selection` | Tighten entry quality — reduce frequency, not size |

A new pattern that says "for SAC with `trade_shy` diagnosis, increase drawdown weight" is
incorrect (drawdown emphasis further suppresses trading). Flag and retire immediately.

### 7.5 Verify `fold_cps_v1_before` populates on iter 6 advice rows

After Group E training completes one fold, query:

```sql
SELECT run_id, fold_number, advice_id, fold_cps_v1_before, fold_cps_v1_after,
       advice_was_effective
FROM reward_adjustments
WHERE iteration_number = 6
LIMIT 10;
```

Expected: `fold_cps_v1_before` is non-null for treatment folds (iter 5 CPS exists as
the "before"). `advice_was_effective` is non-null where `fold_cps_v1_before IS NOT NULL`.
Any row with `fold_cps_v1_before IS NULL` in iter 6 indicates the attribution context
was not loaded correctly.
