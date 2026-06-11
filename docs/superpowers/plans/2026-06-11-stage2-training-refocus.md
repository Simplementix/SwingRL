# Stage 2 Training Refocus Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give the LLM advice path an objective (CPS v1), a live dashboard (leading indicators incl. trade activity), a deterministic cause diagnosis, and per-advice attribution — reversing the 2.7–5.1× treatment-vs-control CPS harm.

**Architecture:** New pure modules (`cps_diagnosis.py`, `fold_context.py`) feed an enriched epoch/run-config payload (client-side JSON context block appended to the existing `query` string — the memory-service API contract is unchanged); three new system-prompt blocks on the service side; six attribution columns on `reward_adjustments`. Base reward weights are untouched (spec D2).

**Tech Stack:** Python 3.11, pytest (class-based, REQ-ID docstrings), psycopg, structlog, Stable Baselines3 callbacks. TDD: RED commit → GREEN commit per the project's CLAUDE.md.

**Spec:** `docs/superpowers/specs/2026-06-11-stage2-training-refocus-design.md`
**Branch:** `swingrl/19.1-training-refocus`

---

## §11 open items — RESOLVED (from pg16 iter 0–4 data, queried 2026-06-11)

**1. Trade-rate baseline = per-(env, algo) percentile constants** derived from 564 iter 0–4
`backtest_results` rows (query documented in Task 1). Per-(env,algo) is mandatory: SAC's
median is 66 trades/fold (crypto) vs PPO's 974 — a global threshold would permanently
mislabel SAC as trade-shy.

| env | algo | p10 | p25 | med | p75 | p90 | med_wr | p25_wr |
|---|---|---|---|---|---|---|---|---|
| crypto | a2c | 67 | 160 | 376 | 619 | 777 | 0.576 | 0.490 |
| crypto | ppo | 913 | 943 | 974 | 996 | 1013 | 0.615 | 0.576 |
| crypto | sac | 14 | 29 | 66 | 225 | 403 | 0.425 | 0.232 |
| equity | a2c | 99 | 180 | 311 | 381 | 419 | 0.647 | 0.480 |
| equity | ppo | 430 | 446 | 458 | 469 | 478 | 0.644 | 0.562 |
| equity | sac | 59 | 100 | 171 | 256 | 440 | 0.674 | 0.479 |

**2. Diagnosis thresholds** (Task 2 rules). Empirical anchor: iter 3–4 treatment harm is
dominated by **tail blowups** — treatment worst-fold MDD 0.36–0.38 vs control 0.067–0.069
(equity); worst single losses 4–7× larger. Disaster MDD thresholds: equity 0.20, crypto 0.40
(control max_mdd never exceeded 0.069 equity / 0.413 crypto; treatment reached 0.382 / 0.483).
**`max_single_loss` is stored in DOLLARS, not the fraction `cps.py`'s docstring claims** —
so the single-disaster rule uses `mdd` + `win_rate`, not `max_single_loss` (C0 documents
this data bug; CPS v2's single-loss penalty is broken by it).

**3. Payload shape:** client-side only — a compact JSON `context` block appended to the
existing single-string `query` payload. No memory-service API change; token cost ≈ 300–400
tokens, fine for Cerebras/Groq. Epoch payload carries fold_number, fold_role,
prev_iter_cps_v1, leading_indicators, diagnosis, target_metric. Regime fields (hmm_regime,
vix_mean) and chronic/protected lists go in the **run-config** payload only — the
orchestrator already has `_current_regime_vector()` and DB access; the epoch callback has
neither (documented deviation from handoff C1).

**4. Orchestrator recomputes** diagnoses via the pure function (single source of truth, no
schema addition).

---

## File structure

| File | Action | Responsibility |
|---|---|---|
| `src/swingrl/memory/training/cps_diagnosis.py` | Create | Pure diagnosis: types, baselines, `diagnose_fold`, `diagnose_rolling` |
| `src/swingrl/memory/training/fold_context.py` | Create | Fold role classification (pure) + thin DB loaders + post-fold attribution writer |
| `src/swingrl/envs/base.py` | Modify | Emit `trades_this_step` in step info dict |
| `src/swingrl/memory/training/reward_wrapper.py` | Modify | `rolling_trade_rate()` + self-baseline |
| `src/swingrl/memory/training/epoch_callback.py` | Modify | Enriched payload, fold_number, attribution columns, outcome_sharpe bugfix |
| `src/swingrl/training/trainer.py` | Modify | Pass `fold_number` through to callback |
| `src/swingrl/agents/backtest.py` | Modify | Pass `fold_number=fold_idx` + post-fold attribution call |
| `src/swingrl/memory/training/meta_orchestrator.py` | Modify | Enriched run-config payload |
| `src/swingrl/data/postgres_schema.py` | Modify | 6 new `reward_adjustments` columns in DDL |
| `scripts/migrations/add_attribution_columns.py` | Create | Idempotent ALTER TABLE (copies `add_cps_columns.py` pattern) |
| `services/memory/memory_agents/query.py` | Modify | Goal / anti-pattern / fold-protection prompt blocks |
| `tests/memory/test_cps_diagnosis.py` | Create | Table-driven diagnosis tests |
| `tests/memory/test_fold_context.py` | Create | Role classification + attribution tests |
| `tests/memory/test_reward_wrapper.py` | Modify | Trade-rate tests |
| `tests/memory/test_epoch_callback_extended.py` | Modify | Payload + attribution tests |
| `tests/test_memory_service_prompts.py` | Create | Prompt block tests (service-side) |
| `tests/data/test_attribution_migration.py` | Create | DB-gated migration tests |
| `.planning/research/phase-19.1-prompt-baseline.md` | Create | C0 baseline doc |
| `.planning/research/phase-19.1-prompt-refocus.md` | Create | C6 post-change doc |
| `docs/training/*.md` | Modify | Same-commit doc updates (per feedback memory) |

---

### Task 1: C0 baseline documentation (no production code)

**Files:**
- Create: `.planning/research/phase-19.1-prompt-baseline.md`

- [ ] **Step 1: Re-derive the data tables.** Run (containers are down; pg16 is up — run from homelab directly):

```bash
docker exec pg16 psql -U swingrl -d swingrl -c "
SELECT environment, algorithm, count(*),
       percentile_cont(0.10) WITHIN GROUP (ORDER BY total_trades)::int AS p10,
       percentile_cont(0.25) WITHIN GROUP (ORDER BY total_trades)::int AS p25,
       percentile_cont(0.5)  WITHIN GROUP (ORDER BY total_trades)::int AS med,
       percentile_cont(0.75) WITHIN GROUP (ORDER BY total_trades)::int AS p75,
       percentile_cont(0.90) WITHIN GROUP (ORDER BY total_trades)::int AS p90,
       round(percentile_cont(0.5) WITHIN GROUP (ORDER BY win_rate)::numeric,3) AS med_wr,
       round(percentile_cont(0.25) WITHIN GROUP (ORDER BY win_rate)::numeric,3) AS p25_wr
FROM backtest_results GROUP BY 1,2 ORDER BY 1,2"
docker exec pg16 psql -U swingrl -d swingrl -c "
SELECT environment,
       round(percentile_cont(0.5) WITHIN GROUP (ORDER BY total_return)::numeric,4) AS med_ret
FROM backtest_results GROUP BY 1"
```

Expected: first query matches the baseline table above; second gives the per-env median
return constants for Task 2 (estimates from plan-time aggregates: equity ≈ 0.053,
crypto ≈ 0.70 — **use the actual query output**, and if it differs from the Task 2 constants,
update them in Task 2 Step 3).

- [ ] **Step 2: Write the baseline doc.** Contents (all verbatim sources verified at plan time):
  1. Current epoch payload — the single f-string at `epoch_callback.py:609–617` (quote it).
  2. Current run-config payload — `meta_orchestrator.py:299–304` (quote it).
  3. Current prompt builders — `services/memory/memory_agents/query.py`:
     `_build_system_prompt` (429–463), `_build_algo_system_prompt` (650–695),
     `_build_epoch_system_prompt` (698–741) (quote each in full).
  4. Reward weight/bounds inventory and divergences:
     - `DEFAULT_WEIGHTS` `{0.50/0.25/0.15/0.10}` (`reward_wrapper.py:28–33`)
     - `_SAFE_DEFAULTS["reward_weights"]` `{0.40/0.35/0.20/0.05}` (`query.py:122–131`) — **diverges from DEFAULT_WEIGHTS**
     - `_FALLBACK_REWARD_BOUNDS` duplicated in `src/swingrl/memory/training/bounds.py:50–55` and `services/.../query.py:85` (left for Stage 3.4)
  5. Data bugs found at plan time:
     - `max_single_loss` stored in dollars vs fractional docstring/formula (breaks CPS v2's single-loss penalty)
     - `epoch_callback.py:580` writes `sharpe_delta` into `outcome_sharpe` (fixed in Task 7)
  6. The iter 3–4 harm decomposition tables (from plan-time queries): treatment max_mdd
     0.36–0.38 vs control 0.067–0.069 (equity); both failure modes (tail blowups iter 3–4,
     trade-shy iter 4–5) mapped to the §4.1 taxonomy.
  7. 3–5 representative `consolidations.pattern_text` rows from iter 0–4
     (`docker exec pg16 psql -U swingrl -d swingrl -c "SELECT id, left(pattern_text, 400) FROM consolidations ORDER BY id LIMIT 20"` — pick representative ones), each annotated
     with the taxonomy label it likely induces. These annotations are Group D's QA criteria.

- [ ] **Step 3: Commit**

```bash
git add .planning/research/phase-19.1-prompt-baseline.md
git commit -m "docs(19.1): C0 prompt/payload/weights baseline + data-bug inventory"
```

---

### Task 2: `cps_diagnosis.py` — types, baselines, `diagnose_fold`

**Files:**
- Create: `src/swingrl/memory/training/cps_diagnosis.py`
- Test: `tests/memory/test_cps_diagnosis.py`

- [ ] **Step 1: Write failing tests** (conventions: class-based, REQ-ID docstrings, `make_fold` helper mirroring `tests/metrics/test_cps.py:30–56`):

```python
"""DIAG-01..DIAG-05: Deterministic CPS diagnosis (spec §4.1 / §5.2)."""

from __future__ import annotations

import pytest

from swingrl.metrics.cps import FoldMetrics
from swingrl.utils.exceptions import DataError


def make_fold(
    fold_number: int = 0,
    sharpe: float = 2.0,
    mdd: float = 0.05,
    total_return: float = 0.08,
    profit_factor: float = 3.0,
    win_rate: float = 0.65,
    total_trades: int = 450,
    sortino: float = 2.5,
    max_single_loss: float = -0.04,
    overfitting_class: str = "healthy",
    is_control_fold: bool = False,
) -> FoldMetrics:
    """Construct a FoldMetrics dict with healthy equity-ppo-like defaults."""
    return {
        "fold_number": fold_number, "sharpe": sharpe, "mdd": mdd,
        "total_return": total_return, "profit_factor": profit_factor,
        "win_rate": win_rate, "total_trades": total_trades, "sortino": sortino,
        "max_single_loss": max_single_loss, "overfitting_class": overfitting_class,
        "is_control_fold": is_control_fold,
    }


class TestDiagnoseFold:
    """DIAG-01: post-fold diagnosis labels match spec §4.1 signatures."""

    def test_healthy_fold_labeled_healthy(self) -> None:
        """DIAG-01: A fold matching all baselines is healthy/clear."""
        from swingrl.memory.training.cps_diagnosis import diagnose_fold

        d = diagnose_fold(make_fold(), env="equity", algo="ppo")
        assert d["label"] == "healthy"
        assert d["confidence"] == "clear"
        assert d["fired"] == []

    def test_trade_shy_low_trades_low_return(self) -> None:
        """DIAG-01: trades < p25 baseline AND return < env median → trade_shy."""
        from swingrl.memory.training.cps_diagnosis import diagnose_fold

        # equity/ppo p25 = 446; equity median return = 0.053
        d = diagnose_fold(
            make_fold(total_trades=200, total_return=0.01), env="equity", algo="ppo"
        )
        assert d["label"] == "trade_shy"
        assert "trade_shy" in d["fired"]

    def test_low_trades_high_return_not_trade_shy(self) -> None:
        """DIAG-01: few trades but strong return is NOT trade_shy."""
        from swingrl.memory.training.cps_diagnosis import diagnose_fold

        d = diagnose_fold(
            make_fold(total_trades=200, total_return=0.12), env="equity", algo="ppo"
        )
        assert d["label"] == "healthy"

    def test_poor_selection_normal_trades_low_winrate(self) -> None:
        """DIAG-02: trades ≥ p25 AND win_rate < p25 baseline → poor_selection."""
        from swingrl.memory.training.cps_diagnosis import diagnose_fold

        # equity/ppo p25_wr = 0.562
        d = diagnose_fold(make_fold(win_rate=0.45), env="equity", algo="ppo")
        assert d["label"] == "poor_selection"

    def test_single_disaster_deep_mdd_healthy_selection(self) -> None:
        """DIAG-03: mdd > disaster threshold with healthy win_rate → single_disaster."""
        from swingrl.memory.training.cps_diagnosis import diagnose_fold

        # equity disaster threshold = 0.20 (control never exceeded 0.069)
        d = diagnose_fold(make_fold(mdd=0.30), env="equity", algo="ppo")
        assert d["label"] == "single_disaster"

    def test_churning_high_trades_low_pf(self) -> None:
        """DIAG-04: trades > p90 AND profit_factor < 1.5 → churning."""
        from swingrl.memory.training.cps_diagnosis import diagnose_fold

        # equity/ppo p90 = 478
        d = diagnose_fold(
            make_fold(total_trades=600, profit_factor=1.1), env="equity", algo="ppo"
        )
        assert d["label"] == "churning"

    def test_mixed_confidence_when_multiple_fire(self) -> None:
        """DIAG-05: ≥2 rules fired → precedence label + mixed confidence."""
        from swingrl.memory.training.cps_diagnosis import diagnose_fold

        # deep mdd (disaster) + low win_rate (poor_selection... but disaster requires
        # healthy win_rate, so use disaster + churning): mdd 0.30 + 600 trades + pf 1.1
        d = diagnose_fold(
            make_fold(mdd=0.30, total_trades=600, profit_factor=1.1),
            env="equity", algo="ppo",
        )
        assert d["label"] == "single_disaster"  # precedence: disaster first
        assert d["confidence"] == "mixed"
        assert set(d["fired"]) >= {"single_disaster", "churning"}

    def test_unknown_env_algo_raises_data_error(self) -> None:
        """DIAG-05: unknown (env, algo) raises DataError, never silent default."""
        from swingrl.memory.training.cps_diagnosis import diagnose_fold

        with pytest.raises(DataError):
            diagnose_fold(make_fold(), env="forex", algo="ppo")

    def test_evidence_contains_fired_values(self) -> None:
        """DIAG-05: evidence dict carries the numbers that fired."""
        from swingrl.memory.training.cps_diagnosis import diagnose_fold

        d = diagnose_fold(make_fold(mdd=0.30), env="equity", algo="ppo")
        assert d["evidence"]["mdd"] == 0.30
        assert d["evidence"]["mdd_disaster_threshold"] == 0.20
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/memory/test_cps_diagnosis.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'swingrl.memory.training.cps_diagnosis'`

- [ ] **Step 3: Implement `cps_diagnosis.py`**

```python
"""Deterministic CPS diagnosis — labels WHY a fold's CPS is low (spec §4.1/§5.2).

Pure functions only: no DB access, no side effects. Baselines are per-(env, algo)
percentile constants derived from iter 0–4 backtest_results (564 folds, query in
.planning/research/phase-19.1-prompt-baseline.md). Per-(env, algo) is mandatory:
SAC's median is 66 trades/fold (crypto) vs PPO's 974.

Rule precedence (single_disaster > churning > trade_shy > poor_selection): a tail
blowup dominates CPS v1 via the squared drawdown factor, so it outranks activity
anomalies; selection quality is judged only when activity is normal.

NOTE: max_single_loss is stored in dollars (not the fraction cps.py's docstring
claims), so the single_disaster rule uses mdd + win_rate instead.
"""

from __future__ import annotations

from typing import Literal, TypedDict

import structlog

from swingrl.metrics.cps import FoldMetrics
from swingrl.utils.exceptions import DataError

log = structlog.get_logger(__name__)

DiagnosisLabel = Literal[
    "trade_shy", "poor_selection", "single_disaster", "churning", "healthy"
]


class TradeBaseline(TypedDict):
    """Per-(env, algo) activity/quality percentiles from iter 0-4 history."""

    p10: int
    p25: int
    med: int
    p75: int
    p90: int
    med_win_rate: float
    p25_win_rate: float


class CpsDiagnosis(TypedDict):
    """Labeled cause for a fold's CPS state."""

    label: DiagnosisLabel
    fired: list[str]
    confidence: Literal["clear", "mixed"]
    evidence: dict[str, float]


# Derived 2026-06-11 from iter 0-4 backtest_results (SQL in C0 baseline doc).
TRADE_BASELINES: dict[tuple[str, str], TradeBaseline] = {
    ("crypto", "a2c"): {"p10": 67, "p25": 160, "med": 376, "p75": 619, "p90": 777,
                        "med_win_rate": 0.576, "p25_win_rate": 0.490},
    ("crypto", "ppo"): {"p10": 913, "p25": 943, "med": 974, "p75": 996, "p90": 1013,
                        "med_win_rate": 0.615, "p25_win_rate": 0.576},
    ("crypto", "sac"): {"p10": 14, "p25": 29, "med": 66, "p75": 225, "p90": 403,
                        "med_win_rate": 0.425, "p25_win_rate": 0.232},
    ("equity", "a2c"): {"p10": 99, "p25": 180, "med": 311, "p75": 381, "p90": 419,
                        "med_win_rate": 0.647, "p25_win_rate": 0.480},
    ("equity", "ppo"): {"p10": 430, "p25": 446, "med": 458, "p75": 469, "p90": 478,
                        "med_win_rate": 0.644, "p25_win_rate": 0.562},
    ("equity", "sac"): {"p10": 59, "p25": 100, "med": 171, "p75": 256, "p90": 440,
                        "med_win_rate": 0.674, "p25_win_rate": 0.479},
}

# Control folds never exceeded mdd 0.069 (equity) / 0.413 (crypto); treatment
# blowups reached 0.382 / 0.483. Thresholds sit between the two regimes.
MDD_DISASTER_THRESHOLD: dict[str, float] = {"equity": 0.20, "crypto": 0.40}

# Median total_return across all iter 0-4 folds per env (C0 Step 1 query;
# verify against actual output and update if it differs).
ENV_MEDIAN_RETURN: dict[str, float] = {"equity": 0.053, "crypto": 0.70}

CHURNING_PROFIT_FACTOR_MAX: float = 1.5

# Diagnosis → correction map; mirrored verbatim in the C3 anti-pattern block.
DIAGNOSIS_CORRECTIONS: dict[DiagnosisLabel, str] = {
    "trade_shy": "increase participation; do NOT damp risk further",
    "poor_selection": "tighten entry quality; reduce frequency, not size",
    "single_disaster": "cap per-trade risk; leave frequency alone",
    "churning": "reduce frequency; raise conviction threshold",
    "healthy": "no adjustment warranted",
}

_PRECEDENCE: tuple[DiagnosisLabel, ...] = (
    "single_disaster", "churning", "trade_shy", "poor_selection"
)


def _baseline(env: str, algo: str) -> TradeBaseline:
    """Look up the (env, algo) baseline or raise DataError."""
    key = (env.lower(), algo.lower())
    if key not in TRADE_BASELINES:
        log.error("diagnosis_unknown_env_algo", env=env, algo=algo)
        raise DataError(f"No trade baseline for env={env!r} algo={algo!r}")
    return TRADE_BASELINES[key]


def diagnose_fold(fold: FoldMetrics, env: str, algo: str) -> CpsDiagnosis:
    """Label why a completed fold's CPS contribution is degraded (or healthy).

    Args:
        fold: Per-fold metrics (same dict shape the CPS formulas consume).
        env: Environment name ("equity" | "crypto").
        algo: Algorithm name ("ppo" | "a2c" | "sac"), case-insensitive.

    Returns:
        CpsDiagnosis with precedence-resolved label, all fired rules,
        clear/mixed confidence, and the evidence values that fired.

    Raises:
        DataError: Unknown (env, algo) — never silently defaults.
    """
    b = _baseline(env, algo)
    mdd_max = MDD_DISASTER_THRESHOLD[env.lower()]
    med_ret = ENV_MEDIAN_RETURN[env.lower()]

    fired: list[str] = []
    evidence: dict[str, float] = {}

    if fold["mdd"] > mdd_max and fold["win_rate"] >= b["p25_win_rate"]:
        fired.append("single_disaster")
        evidence["mdd"] = fold["mdd"]
        evidence["mdd_disaster_threshold"] = mdd_max
    if fold["total_trades"] > b["p90"] and fold["profit_factor"] < CHURNING_PROFIT_FACTOR_MAX:
        fired.append("churning")
        evidence["total_trades"] = float(fold["total_trades"])
        evidence["profit_factor"] = fold["profit_factor"]
    if fold["total_trades"] < b["p25"] and fold["total_return"] < med_ret:
        fired.append("trade_shy")
        evidence["total_trades"] = float(fold["total_trades"])
        evidence["total_return"] = fold["total_return"]
    if fold["total_trades"] >= b["p25"] and fold["win_rate"] < b["p25_win_rate"]:
        fired.append("poor_selection")
        evidence["win_rate"] = fold["win_rate"]
        evidence["p25_win_rate"] = b["p25_win_rate"]

    if not fired:
        return {"label": "healthy", "fired": [], "confidence": "clear", "evidence": {}}

    label = next(lab for lab in _PRECEDENCE if lab in fired)
    confidence: Literal["clear", "mixed"] = "clear" if len(fired) == 1 else "mixed"
    return {"label": label, "fired": fired, "confidence": confidence, "evidence": evidence}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/memory/test_cps_diagnosis.py -v`
Expected: all PASS

- [ ] **Step 5: Commit (RED test commit was Step 1-2; if committing once, include both)**

```bash
git add tests/memory/test_cps_diagnosis.py src/swingrl/memory/training/cps_diagnosis.py
git commit -m "feat(19.1): deterministic CPS diagnosis — per-(env,algo) baselines + diagnose_fold"
```

---

### Task 3: `diagnose_rolling` — mid-fold diagnosis

**Files:**
- Modify: `src/swingrl/memory/training/cps_diagnosis.py`
- Test: `tests/memory/test_cps_diagnosis.py`

Mid-fold has no `backtest_results` row, so the rolling variant uses the wrapper's
indicators with a **self-baseline**: the fold's own first-full-window trade rate. This
avoids inventing a steps-per-fold conversion for the per-fold percentiles and directly
catches the actual harm pattern (advice kills trading *mid*-fold). Rolling MDD is in
reward units (not portfolio fraction), so the disaster rule is post-fold-only.

- [ ] **Step 1: Write failing tests** (append to `tests/memory/test_cps_diagnosis.py`):

```python
class TestDiagnoseRolling:
    """DIAG-06: mid-fold diagnosis from rolling indicators."""

    def test_trade_rate_collapse_labeled_trade_shy(self) -> None:
        """DIAG-06: trade rate < 50% of the fold's own baseline → trade_shy."""
        from swingrl.memory.training.cps_diagnosis import diagnose_rolling

        d = diagnose_rolling(
            trade_rate=0.10, baseline_trade_rate=0.40,
            rolling_win_rate=0.60, env="equity", algo="ppo",
        )
        assert d["label"] == "trade_shy"
        assert d["evidence"]["trade_rate"] == 0.10
        assert d["evidence"]["baseline_trade_rate"] == 0.40

    def test_winrate_collapse_normal_activity_poor_selection(self) -> None:
        """DIAG-06: activity normal but win rate < p25 baseline → poor_selection."""
        from swingrl.memory.training.cps_diagnosis import diagnose_rolling

        d = diagnose_rolling(
            trade_rate=0.40, baseline_trade_rate=0.40,
            rolling_win_rate=0.30, env="equity", algo="ppo",
        )
        assert d["label"] == "poor_selection"

    def test_no_signals_healthy(self) -> None:
        """DIAG-06: nothing fired → healthy/clear."""
        from swingrl.memory.training.cps_diagnosis import diagnose_rolling

        d = diagnose_rolling(
            trade_rate=0.40, baseline_trade_rate=0.40,
            rolling_win_rate=0.60, env="equity", algo="ppo",
        )
        assert d["label"] == "healthy"

    def test_zero_baseline_never_divides(self) -> None:
        """DIAG-06: baseline 0.0 (window not yet full) → healthy, no ZeroDivisionError."""
        from swingrl.memory.training.cps_diagnosis import diagnose_rolling

        d = diagnose_rolling(
            trade_rate=0.0, baseline_trade_rate=0.0,
            rolling_win_rate=0.60, env="equity", algo="ppo",
        )
        assert d["label"] == "healthy"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/memory/test_cps_diagnosis.py::TestDiagnoseRolling -v`
Expected: FAIL — `ImportError: cannot import name 'diagnose_rolling'`

- [ ] **Step 3: Implement** (append to `cps_diagnosis.py`):

```python
TRADE_RATE_COLLAPSE_FRACTION: float = 0.5


def diagnose_rolling(
    trade_rate: float,
    baseline_trade_rate: float,
    rolling_win_rate: float,
    env: str,
    algo: str,
) -> CpsDiagnosis:
    """Label mid-fold degradation from the wrapper's rolling indicators.

    Self-baseline design: baseline_trade_rate is the fold's own first-full-window
    rate (wrapper-provided), so no cross-fold steps conversion is needed. Only
    trade_shy and poor_selection are detectable mid-fold; disaster/churning need
    the completed backtest row (rolling MDD is in reward units, not portfolio
    fraction).

    Args:
        trade_rate: Current rolling trades-per-step rate.
        baseline_trade_rate: The fold's first-full-window trades-per-step rate
            (0.0 while the window is still filling — disables the trade_shy rule).
        rolling_win_rate: Wrapper's rolling fraction of positive-reward steps.
        env: Environment name. algo: Algorithm name.

    Returns:
        CpsDiagnosis (same shape as diagnose_fold).

    Raises:
        DataError: Unknown (env, algo).
    """
    b = _baseline(env, algo)
    fired: list[str] = []
    evidence: dict[str, float] = {}

    if baseline_trade_rate > 0.0 and trade_rate < TRADE_RATE_COLLAPSE_FRACTION * baseline_trade_rate:
        fired.append("trade_shy")
        evidence["trade_rate"] = trade_rate
        evidence["baseline_trade_rate"] = baseline_trade_rate
    elif rolling_win_rate < b["p25_win_rate"]:
        fired.append("poor_selection")
        evidence["rolling_win_rate"] = rolling_win_rate
        evidence["p25_win_rate"] = b["p25_win_rate"]

    if not fired:
        return {"label": "healthy", "fired": [], "confidence": "clear", "evidence": {}}
    return {"label": fired[0], "fired": fired, "confidence": "clear", "evidence": evidence}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/memory/test_cps_diagnosis.py -v`
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add tests/memory/test_cps_diagnosis.py src/swingrl/memory/training/cps_diagnosis.py
git commit -m "feat(19.1): diagnose_rolling — mid-fold trade-shy/poor-selection detection"
```

---

### Task 4: Envs emit `trades_this_step`

**Files:**
- Modify: `src/swingrl/envs/base.py` (step ~lines 215–255; `_build_info` lines 372–400)
- Test: `tests/test_envs.py` (append)
- Modify: `docs/training/` env reference doc in the same commit (per `feedback_docs_update_with_code` — find the file documenting env info keys: `grep -rln "reward_components" docs/training/`)

The portfolio already appends to `trade_log` per executed trade (`portfolio.py:128–137`);
the env just doesn't surface a per-step count.

- [ ] **Step 1: Write failing test** (append to `tests/test_envs.py`, matching its existing style — check the file's imports/fixtures first and reuse its env-construction helper):

```python
def test_step_info_contains_trades_this_step(equity_env: Any) -> None:
    """ENV-TRADES-01: step() info dict reports how many trades executed this step."""
    equity_env.reset()
    action = equity_env.action_space.sample()
    _obs, _r, _term, _trunc, info = equity_env.step(action)
    assert "trades_this_step" in info
    assert isinstance(info["trades_this_step"], int)
    assert info["trades_this_step"] >= 0
```

(If `tests/test_envs.py` has no `equity_env` fixture, construct the env exactly as the
file's existing step tests do — copy their setup verbatim.)

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_envs.py -k trades_this_step -v`
Expected: FAIL — KeyError/assert on `"trades_this_step"`

- [ ] **Step 3: Implement.** In `base.py` `step()`, capture the trade-log length before
rebalancing and pass the delta into `_build_info`. Around the rebalance call (the step that
computes `cost`), add:

```python
trades_before = len(self._portfolio.trade_log)
# ... existing rebalance call computing `cost` ...
trades_this_step = len(self._portfolio.trade_log) - trades_before
```

Extend `_build_info` (line 372) signature and body:

```python
def _build_info(
    self,
    portfolio_value: float,
    daily_return: float,
    transaction_cost: float,
    reward_components: dict[str, float] | None = None,
    trades_this_step: int = 0,
) -> dict[str, Any]:
```

and in the dict construction add:

```python
        "trades_this_step": trades_this_step,
```

then pass `trades_this_step=trades_this_step` at the `_build_info` call site in `step()`
(line ~246). Also update the `_build_info` call in `reset()` if it exists (it has no trades:
pass nothing — the default 0 covers it).

- [ ] **Step 4: Run env tests**

Run: `uv run pytest tests/test_envs.py -v`
Expected: all PASS (existing tests unaffected — key is additive)

- [ ] **Step 5: Update `docs/training/` env doc + commit**

```bash
git add src/swingrl/envs/base.py tests/test_envs.py docs/training/
git commit -m "feat(19.1): envs report trades_this_step in step info (trade-activity signal)"
```

---

### Task 5: Wrapper `rolling_trade_rate()` + self-baseline

**Files:**
- Modify: `src/swingrl/memory/training/reward_wrapper.py`
- Test: `tests/memory/test_reward_wrapper.py` (append to `TestRollingMetrics`)
- Modify: relevant `docs/training/` page in same commit

- [ ] **Step 1: Write failing tests** (match the file's `_make_mock_venv` pattern — its mock `step_wait` returns infos; set `"trades_this_step"` on them):

```python
class TestRollingTradeRate:
    """TRAIN-TRADE-01: rolling trade rate + first-window self-baseline."""

    def test_trade_rate_counts_trades_per_step(self) -> None:
        """TRAIN-TRADE-01: rate = mean trades_this_step over the window."""
        from swingrl.memory.training.reward_wrapper import MemoryVecRewardWrapper

        mock_venv = _make_mock_venv()
        wrapper = MemoryVecRewardWrapper(mock_venv)
        # Simulate 10 steps: 1 trade every other step → rate 0.5
        for i in range(10):
            _set_mock_step(mock_venv, trades_this_step=i % 2)
            wrapper.step_wait()
        assert abs(wrapper.rolling_trade_rate() - 0.5) < 1e-9

    def test_empty_history_rate_zero(self) -> None:
        """TRAIN-TRADE-01: no steps → 0.0 (matches other rolling metrics)."""
        from swingrl.memory.training.reward_wrapper import MemoryVecRewardWrapper

        wrapper = MemoryVecRewardWrapper(_make_mock_venv())
        assert wrapper.rolling_trade_rate() == 0.0

    def test_baseline_locks_at_first_full_window(self) -> None:
        """TRAIN-TRADE-02: baseline_trade_rate() is 0.0 until the 500-step window
        first fills, then locks to that window's rate permanently."""
        from swingrl.memory.training.reward_wrapper import (
            _ROLLING_WINDOW,
            MemoryVecRewardWrapper,
        )

        mock_venv = _make_mock_venv()
        wrapper = MemoryVecRewardWrapper(mock_venv)
        _set_mock_step(mock_venv, trades_this_step=1)
        for _ in range(_ROLLING_WINDOW - 1):
            wrapper.step_wait()
        assert wrapper.baseline_trade_rate() == 0.0  # window not yet full
        wrapper.step_wait()  # window fills
        assert abs(wrapper.baseline_trade_rate() - 1.0) < 1e-9
        _set_mock_step(mock_venv, trades_this_step=0)
        for _ in range(_ROLLING_WINDOW):
            wrapper.step_wait()
        assert abs(wrapper.baseline_trade_rate() - 1.0) < 1e-9  # still locked
        assert wrapper.rolling_trade_rate() == 0.0  # current rate collapsed

    def test_reset_clears_trade_history_and_baseline(self) -> None:
        """TRAIN-TRADE-02: reset() clears trade history and the locked baseline."""
        from swingrl.memory.training.reward_wrapper import MemoryVecRewardWrapper

        mock_venv = _make_mock_venv()
        wrapper = MemoryVecRewardWrapper(mock_venv)
        _set_mock_step(mock_venv, trades_this_step=1)
        for _ in range(5):
            wrapper.step_wait()
        wrapper.reset()
        assert wrapper.rolling_trade_rate() == 0.0
        assert wrapper.baseline_trade_rate() == 0.0
```

Add a module-level helper next to the file's existing mock helpers:

```python
def _set_mock_step(mock_venv: Any, trades_this_step: int) -> None:
    """Point the mock venv's next step_wait() at an info carrying a trade count."""
    obs, rewards, dones, infos = mock_venv.step_wait.return_value
    for info in infos:
        info["trades_this_step"] = trades_this_step
```

(Adapt to the file's actual mock construction — read `_make_mock_venv` first and mirror it;
if `step_wait.return_value` infos are shared dicts, build fresh tuples instead.)

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/memory/test_reward_wrapper.py::TestRollingTradeRate -v`
Expected: FAIL — `AttributeError: ... no attribute 'rolling_trade_rate'`

- [ ] **Step 3: Implement.** In `reward_wrapper.py`:

In `__init__` (after line 77 `self._positive_steps`):

```python
        self._trades_per_step: deque[float] = deque(maxlen=_ROLLING_WINDOW)
        self._baseline_trade_rate: float = 0.0
```

In `step_wait()` (inside the existing per-step tracking loop, lines 102–105 region):

```python
        trades = 0.0
        for info in infos:
            trades += float(info.get("trades_this_step", 0))
        self._trades_per_step.append(trades)
        if (
            self._baseline_trade_rate == 0.0
            and len(self._trades_per_step) == _ROLLING_WINDOW
        ):
            self._baseline_trade_rate = float(
                sum(self._trades_per_step) / len(self._trades_per_step)
            )
```

In `reset()` (after the existing `.clear()` calls):

```python
        self._trades_per_step.clear()
        self._baseline_trade_rate = 0.0
```

New methods (after `rolling_win_rate`):

```python
    def rolling_trade_rate(self) -> float:
        """Mean trades per step over the rolling window. 0.0 if empty."""
        if not self._trades_per_step:
            return 0.0
        return float(sum(self._trades_per_step) / len(self._trades_per_step))

    def baseline_trade_rate(self) -> float:
        """The fold's first-full-window trade rate (0.0 until the window fills).

        Locked once: later windows never overwrite it, so a mid-fold activity
        collapse is measured against how the fold STARTED trading.
        """
        return self._baseline_trade_rate
```

- [ ] **Step 4: Run the full wrapper suite**

Run: `uv run pytest tests/memory/test_reward_wrapper.py -v`
Expected: all PASS

- [ ] **Step 5: Update `docs/training/` + commit**

```bash
git add src/swingrl/memory/training/reward_wrapper.py tests/memory/test_reward_wrapper.py docs/training/
git commit -m "feat(19.1): rolling_trade_rate + locked first-window baseline in reward wrapper"
```

---

### Task 6: `fold_context.py` — role classification + context loader

**Files:**
- Create: `src/swingrl/memory/training/fold_context.py`
- Test: `tests/memory/test_fold_context.py`

- [ ] **Step 1: Write failing tests:**

```python
"""FOLD-CTX-01..03: fold role classification and context assembly."""

from __future__ import annotations

import pandas as pd


def _history_df(rows: list[dict]) -> pd.DataFrame:
    """Build a load_fold_history-shaped DataFrame."""
    return pd.DataFrame(rows)


def _row(iteration: int, fold: int, sharpe: float, oc: str) -> dict:
    return {
        "iteration_number": iteration, "environment": "equity", "algorithm": "ppo",
        "fold_number": fold, "sharpe": sharpe, "overfitting_class": oc,
    }


class TestClassifyFoldRole:
    """FOLD-CTX-01: chronic_failure / protected_winner / neutral from history."""

    def test_chronic_failure_fold(self) -> None:
        """FOLD-CTX-01: fold failing in every recent iteration → chronic_failure."""
        from swingrl.memory.training.fold_context import classify_fold_role

        df = _history_df([_row(i, 3, 0.2, "reject") for i in range(5)])
        assert classify_fold_role(df, env="equity", fold_number=3) == "chronic_failure"

    def test_protected_winner_fold(self) -> None:
        """FOLD-CTX-01: consistently healthy high-sharpe fold → protected_winner."""
        from swingrl.memory.training.fold_context import classify_fold_role

        df = _history_df([_row(i, 7, 2.5, "healthy") for i in range(5)])
        assert classify_fold_role(df, env="equity", fold_number=7) == "protected_winner"

    def test_unlisted_fold_neutral(self) -> None:
        """FOLD-CTX-01: fold in neither detector output → neutral."""
        from swingrl.memory.training.fold_context import classify_fold_role

        df = _history_df(
            [_row(i, 3, 0.2, "reject") for i in range(5)]
            + [_row(i, 5, 1.0, "healthy") for i in range(2)]  # mixed → neutral
        )
        assert classify_fold_role(df, env="equity", fold_number=5) == "neutral"

    def test_empty_history_neutral(self) -> None:
        """FOLD-CTX-01: no history (iter 0 cold start) → neutral."""
        from swingrl.memory.training.fold_context import classify_fold_role

        assert classify_fold_role(pd.DataFrame(), env="equity", fold_number=0) == "neutral"
```

(Exact pass/fail windows come from `detect_chronic_failures`/`detect_protected_winners`
defaults — `CHRONIC_DEFAULT_WINDOW`/`MIN_FAILS` etc. in `iteration_report.py:208–309`.
Read those constants first; if 5 same-class iterations don't trigger them, adjust the row
counts in these tests so they do — the *behavior* under test is delegation, not the
detectors' internals, which already have their own tests.)

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/memory/test_fold_context.py -v`
Expected: FAIL — module not found

- [ ] **Step 3: Implement:**

```python
"""Fold-scoped training context: role classification + context assembly.

classify_fold_role is the single source of truth for chronic-failure /
protected-winner determination (spec §5.2) — it delegates to the existing
detectors in iteration_report.py rather than reimplementing their windows.
"""

from __future__ import annotations

from typing import Any, Literal

import pandas as pd
import structlog

from swingrl.reporting.iteration_report import (
    detect_chronic_failures,
    detect_protected_winners,
)

log = structlog.get_logger(__name__)

FoldRole = Literal["chronic_failure", "protected_winner", "neutral"]


def classify_fold_role(
    fold_history: pd.DataFrame, env: str, fold_number: int
) -> FoldRole:
    """Classify a fold from its cross-iteration history.

    Args:
        fold_history: DataFrame shaped like load_fold_history() output
            (iteration_number, environment, algorithm, fold_number, sharpe,
            overfitting_class, ...). May be empty (iter 0 cold start).
        env: Environment name.
        fold_number: Fold to classify.

    Returns:
        "chronic_failure" | "protected_winner" | "neutral".
    """
    if fold_history.empty:
        return "neutral"
    chronic = detect_chronic_failures(fold_history).get(env, [])
    if fold_number in chronic:
        return "chronic_failure"
    protected = detect_protected_winners(fold_history).get(env, [])
    if fold_number in protected:
        return "protected_winner"
    return "neutral"


def load_fold_context(
    database_url: str, env: str, fold_number: int
) -> dict[str, Any]:
    """Assemble the per-fold advice context (thin I/O wrapper; spec §5.1).

    Returns dict with: fold_role, chronic_failure_folds, protected_winner_folds,
    prev_iter_cps_v1 (None when absent). Fails open: any DB error returns the
    neutral cold-start context — advice must never block training.
    """
    import psycopg

    from swingrl.reporting.iteration_report import load_fold_history

    neutral: dict[str, Any] = {
        "fold_role": "neutral",
        "chronic_failure_folds": [],
        "protected_winner_folds": [],
        "prev_iter_cps_v1": None,
    }
    try:
        with psycopg.connect(database_url) as conn:
            history = load_fold_history(conn, env)
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT cps_v1_multiplicative FROM iteration_results "
                    "WHERE environment = %s ORDER BY iteration_number DESC LIMIT 1",
                    (env,),
                )
                row = cur.fetchone()
    except Exception as exc:  # noqa: BLE001 — fail-open by design (see docstring)
        log.warning("fold_context_load_failed", env=env, error=str(exc))
        return neutral

    return {
        "fold_role": classify_fold_role(history, env, fold_number),
        "chronic_failure_folds": detect_chronic_failures(history).get(env, []),
        "protected_winner_folds": detect_protected_winners(history).get(env, []),
        "prev_iter_cps_v1": row[0] if row else None,
    }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/memory/test_fold_context.py -v`
Expected: PASS (the `load_fold_context` I/O path is exercised by Task 8's callback tests
with a mocked loader; its fail-open branch gets a direct test there too)

- [ ] **Step 5: Commit**

```bash
git add src/swingrl/memory/training/fold_context.py tests/memory/test_fold_context.py
git commit -m "feat(19.1): fold_context — role classification + fail-open context loader"
```

---

### Task 7: `reward_adjustments` attribution columns + migration + outcome_sharpe bugfix

**Files:**
- Modify: `src/swingrl/data/postgres_schema.py:338–359` (DDL)
- Create: `scripts/migrations/add_attribution_columns.py`
- Modify: `src/swingrl/memory/training/epoch_callback.py:580` (bugfix)
- Test: `tests/data/test_attribution_migration.py` (DB-gated)

- [ ] **Step 1: Write failing DB-gated test:**

```python
"""C5-MIG-01: reward_adjustments attribution columns exist after migration."""

from __future__ import annotations

import os

import pytest

pytestmark = pytest.mark.skipif(
    not os.environ.get("DATABASE_URL"), reason="live-DB test; needs DATABASE_URL"
)

_NEW_COLS = {
    "fold_number", "iteration_number", "advice_id",
    "fold_cps_v1_before", "fold_cps_v1_after", "advice_was_effective",
}


def test_attribution_columns_present_after_schema_init() -> None:
    """C5-MIG-01: init_postgres_schema creates the 6 attribution columns."""
    import psycopg

    from swingrl.data.postgres_schema import init_postgres_schema

    url = os.environ["DATABASE_URL"]
    init_postgres_schema(url)
    with psycopg.connect(url) as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT column_name FROM information_schema.columns "
            "WHERE table_name = 'reward_adjustments'"
        )
        cols = {r[0] for r in cur.fetchall()}
    assert _NEW_COLS <= cols


def test_migration_script_idempotent() -> None:
    """C5-MIG-01: running the migration twice is a no-op the second time."""
    from scripts.migrations.add_attribution_columns import migrate

    url = os.environ["DATABASE_URL"]
    first = migrate(url)
    second = migrate(url)
    assert second["added"] == 0
    assert second["already_present"] == len(_NEW_COLS)
    assert first["added"] + first["already_present"] == len(_NEW_COLS)
```

(Check how `init_postgres_schema` is imported/named in `postgres_schema.py` first and match
it; check how other scripts are imported in tests — if `scripts/` is not importable, invoke
the migration via `subprocess.run([sys.executable, "scripts/migrations/add_attribution_columns.py"], ...)`
the way existing migration tests do, if any exist — `grep -rn "migrations" tests/` first.)

- [ ] **Step 2: Run to verify it fails** (needs a test DB per Stage 1 rules — name must end `_test`):

Run: `DATABASE_URL=postgresql://swingrl:***@localhost/swingrl_test uv run pytest tests/data/test_attribution_migration.py -v` (on homelab CI this is automatic; locally skip if no test DB)
Expected: FAIL — columns missing / module not found

- [ ] **Step 3: Implement.**

(a) In `postgres_schema.py` `reward_adjustments` DDL (after `outcome_sharpe` line, before `created_at`):

```sql
    fold_number         INTEGER,
    iteration_number    INTEGER,
    advice_id           TEXT,
    fold_cps_v1_before  DOUBLE PRECISION,
    fold_cps_v1_after   DOUBLE PRECISION,
    advice_was_effective BOOLEAN,
```

(b) `scripts/migrations/add_attribution_columns.py` — copy the structure of
`add_cps_columns.py` (information_schema pre-check + `ADD COLUMN IF NOT EXISTS` + counts
dict + structlog), with:

```python
_NEW_COLUMNS: list[tuple[str, str]] = [
    ("fold_number", "INTEGER"),
    ("iteration_number", "INTEGER"),
    ("advice_id", "TEXT"),
    ("fold_cps_v1_before", "DOUBLE PRECISION"),
    ("fold_cps_v1_after", "DOUBLE PRECISION"),
    ("advice_was_effective", "BOOLEAN"),
]
_TABLE = "reward_adjustments"
```

and a `migrate(db_url: str) -> dict[str, int]` entry point returning the counts dict
(`added` / `already_present`), callable from `__main__`.

(c) Bugfix `epoch_callback.py:580`: `outcome_sharpe` must store the *current* sharpe
(`current_sharpe` computed at line 544), not `sharpe_delta`. Change the UPDATE parameter
ordering accordingly (the SQL at 286–291 is fine; the bug is the Python argument at 580).
Add a regression test in `tests/memory/test_epoch_callback_extended.py` asserting the
outcome-queue tuple carries `current_sharpe` in the outcome_sharpe position.

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/data/test_attribution_migration.py tests/memory/test_epoch_callback_extended.py -v`
Expected: PASS locally for callback tests; migration tests skip without DATABASE_URL (pass on homelab CI)

- [ ] **Step 5: Commit**

```bash
git add src/swingrl/data/postgres_schema.py scripts/migrations/add_attribution_columns.py \
       tests/data/test_attribution_migration.py src/swingrl/memory/training/epoch_callback.py \
       tests/memory/test_epoch_callback_extended.py
git commit -m "feat(19.1): reward_adjustments attribution columns + migration; fix outcome_sharpe write"
```

---

### Task 8: Epoch callback — enriched payload, fold wiring, attribution writes

**Files:**
- Modify: `src/swingrl/memory/training/epoch_callback.py` (constructor 101–113; `_query_epoch_advice` 592–746; INSERT params ~520–531)
- Modify: `src/swingrl/training/trainer.py:303–315` (pass-through)
- Modify: `src/swingrl/agents/backtest.py:375–376` (pass `fold_number=fold_idx`)
- Test: `tests/memory/test_epoch_callback_extended.py`
- Modify: `docs/training/` memory/advice page in same commit

- [ ] **Step 1: Write failing tests** (mirror the file's `_make_mock_wrapper` / `_make_callback` helpers; extend `_make_callback` to accept `fold_number`):

```python
class TestEnrichedEpochPayload:
    """C1-PAYLOAD-01: epoch advice payload carries context JSON."""

    def test_payload_query_contains_context_block(self) -> None:
        """C1-PAYLOAD-01: query string embeds context={...} with required keys."""
        import json

        cb = _make_callback(advice_enabled=True, fold_number=3)
        cb._fold_context = {
            "fold_role": "neutral", "chronic_failure_folds": [],
            "protected_winner_folds": [], "prev_iter_cps_v1": 0.034,
        }
        cb._epoch = cb._cadence  # make it a storage epoch
        cb._query_epoch_advice()
        payload = cb._client.epoch_advice.call_args[0][0]
        query = payload["query"]
        assert "context=" in query
        ctx = json.loads(query.split("context=", 1)[1])
        assert ctx["fold_number"] == 3
        assert ctx["fold_role"] == "neutral"
        assert ctx["target_metric"] == "cps_v1_multiplicative"
        assert ctx["prev_iter_cps_v1"] == 0.034
        assert "leading_indicators" in ctx
        assert set(ctx["leading_indicators"]) == {
            "rolling_sharpe", "rolling_mdd", "rolling_win_rate",
            "trade_rate", "baseline_trade_rate",
        }
        assert ctx["diagnosis"]["label"] in {
            "trade_shy", "poor_selection", "single_disaster", "churning", "healthy"
        }

    def test_control_fold_sends_no_advice_query(self) -> None:
        """C1-PAYLOAD-01: advice_enabled=False short-circuits (existing behavior kept)."""
        cb = _make_callback(advice_enabled=False, fold_number=3)
        cb._epoch = cb._cadence
        cb._query_epoch_advice()
        cb._client.epoch_advice.assert_not_called()


class TestAttributionWrites:
    """C5-ATTR-01: adjustment INSERT carries fold/iteration/advice_id/cps_before."""

    def test_adjustment_trigger_row_has_attribution_fields(self) -> None:
        """C5-ATTR-01: trigger queue row includes the 4 new identity values."""
        cb = _make_callback(advice_enabled=True, fold_number=3, iteration=6)
        cb._fold_context = {"fold_role": "neutral", "chronic_failure_folds": [],
                            "protected_winner_folds": [], "prev_iter_cps_v1": 0.034}
        # drive the same path the existing trigger-queue tests drive, then:
        row = cb._adjustment_trigger_queue[-1]
        # new tail fields: fold_number, iteration_number, advice_id, fold_cps_v1_before
        assert row[-4] == 3
        assert row[-3] == 6
        assert isinstance(row[-2], str) and len(row[-2]) > 0  # advice_id uuid
        assert row[-1] == 0.034
```

(Adapt the drive-the-path setup to exactly how the existing `TestEpochCallback*` tests
trigger `_ingest_adjustment_trigger` — read those tests first and copy their arrangement.
The `_fold_context` injection bypasses the lazy DB load, keeping tests offline.)

- [ ] **Step 2: Run to verify failures**

Run: `uv run pytest tests/memory/test_epoch_callback_extended.py -k "Enriched or Attribution" -v`
Expected: FAIL — unknown `fold_number` kwarg / missing context

- [ ] **Step 3: Implement in `epoch_callback.py`:**

(a) Constructor: add `fold_number: int | None = None` param; store
`self._fold_number = fold_number`; add `self._fold_context: dict[str, Any] | None = None`
and `self._advice_id: str = ""`.

(b) Lazy context load at the top of `_query_epoch_advice` (after the existing
advice-enabled/storage-epoch early-returns):

```python
        if self._fold_context is None:
            if self._database_url and self._fold_number is not None:
                from swingrl.memory.training.fold_context import load_fold_context

                self._fold_context = load_fold_context(
                    self._database_url, self._env, self._fold_number
                )
            else:
                self._fold_context = {
                    "fold_role": "neutral", "chronic_failure_folds": [],
                    "protected_winner_folds": [], "prev_iter_cps_v1": None,
                }
```

(c) Replace the payload construction (lines 609–617) with:

```python
        from swingrl.memory.training.cps_diagnosis import diagnose_rolling

        diagnosis = diagnose_rolling(
            trade_rate=self._wrapper.rolling_trade_rate(),
            baseline_trade_rate=self._wrapper.baseline_trade_rate(),
            rolling_win_rate=self._wrapper.rolling_win_rate(),
            env=self._env,
            algo=self._algo,
        )
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
        payload = {
            "query": (
                f"EPOCH ADVICE: run_id={self._run_id} algo={self._algo} "
                f"env={self._env} epoch={self._epoch}{iter_part} "
                f"current_weights={_json.dumps(self._wrapper.weights)} "
                f"context={_json.dumps(context)}"
            )
        }
```

(`rolling_sharpe`/`rolling_mdd` move inside the JSON block; keep the query prefix shape so
the service's existing parsing of the leading fields is unaffected. `DataError` from
`diagnose_rolling` on an unknown algo must not kill training: wrap the diagnose call in
`try/except DataError` and fall back to the healthy/neutral diagnosis dict with a
`log.warning("diagnosis_unavailable", ...)`.)

(d) Attribution identity: where the advice is accepted and the trigger ingested
(line ~728–734), first set `self._advice_id = str(uuid.uuid4())` (add `import uuid` at
top, stdlib group). Extend `_ingest_adjustment_trigger` to append
`self._fold_number, self._iteration, self._advice_id,
self._fold_context["prev_iter_cps_v1"]` to the queued row, extend the INSERT SQL
(lines 271–275) with `fold_number, iteration_number, advice_id, fold_cps_v1_before`
columns + 4 placeholders.

(e) `trainer.py`: add `fold_number: int | None = None` to `train()` signature (line 129
region, after `is_control_fold`), document it in the docstring, pass
`fold_number=fold_number` in the `MemoryEpochCallback(...)` call (line 303).

(f) `backtest.py:375` region: pass `fold_number=fold_idx` into the `.train(...)` call.

- [ ] **Step 4: Run the callback + trainer test suites**

Run: `uv run pytest tests/memory/test_epoch_callback_extended.py tests/training/ -v`
Expected: all PASS

- [ ] **Step 5: Update `docs/training/` + commit**

```bash
git add src/swingrl/memory/training/epoch_callback.py src/swingrl/training/trainer.py \
       src/swingrl/agents/backtest.py tests/memory/test_epoch_callback_extended.py docs/training/
git commit -m "feat(19.1): enriched epoch-advice payload (context JSON + diagnosis) + attribution identity"
```

---

### Task 9: Post-fold attribution closure

**Files:**
- Modify: `src/swingrl/memory/training/fold_context.py`
- Modify: `src/swingrl/agents/backtest.py` (after the fold's backtest row is persisted)
- Test: `tests/memory/test_fold_context.py`

`fold_cps_v1_after` = single-fold CPS v1 of the completed fold —
`compute_cps_v1_multiplicative([fold])` degenerates cleanly (median = the fold's return,
winner ratio ∈ {0, 1}). `advice_was_effective = fold_cps_v1_after > fold_cps_v1_before`
(NULL-safe: stays NULL when before is NULL — iter 0 has no baseline).

- [ ] **Step 1: Write failing tests** (append to `tests/memory/test_fold_context.py`):

```python
class TestRecordFoldAttribution:
    """C5-ATTR-02: post-fold UPDATE closes the attribution loop."""

    def test_update_sets_after_and_effectiveness(self) -> None:
        """C5-ATTR-02: computes single-fold CPS and effectiveness, updates by run_id."""
        from unittest.mock import MagicMock

        from swingrl.memory.training.fold_context import record_fold_attribution

        conn = MagicMock()
        cur = conn.cursor.return_value.__enter__.return_value
        fold = {
            "fold_number": 3, "sharpe": 2.0, "mdd": 0.05, "total_return": 0.08,
            "profit_factor": 3.0, "win_rate": 0.65, "total_trades": 450,
            "sortino": 2.5, "max_single_loss": -0.04,
            "overfitting_class": "healthy", "is_control_fold": False,
        }
        record_fold_attribution(conn, run_id="equity_ppo_fold3", fold=fold)
        sql, params = cur.execute.call_args[0]
        assert "UPDATE reward_adjustments" in sql
        assert "fold_cps_v1_after" in sql
        assert "advice_was_effective" in sql
        cps_after = params[0]
        assert cps_after > 0.0
        assert params[-1] == "equity_ppo_fold3"
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/memory/test_fold_context.py::TestRecordFoldAttribution -v`
Expected: FAIL — no `record_fold_attribution`

- [ ] **Step 3: Implement** (append to `fold_context.py`):

```python
def record_fold_attribution(conn: Any, run_id: str, fold: dict[str, Any]) -> None:
    """Close the advice-attribution loop after a fold's backtest completes.

    Sets fold_cps_v1_after = single-fold CPS v1 and
    advice_was_effective = (after > before) on every advice row for this run_id.
    Rows with NULL fold_cps_v1_before keep NULL effectiveness (no baseline).
    """
    from swingrl.metrics.cps import compute_cps_v1_multiplicative

    cps_after = compute_cps_v1_multiplicative([fold])  # type: ignore[list-item]
    with conn.cursor() as cur:
        cur.execute(
            "UPDATE reward_adjustments "
            "SET fold_cps_v1_after = %s, "
            "    advice_was_effective = CASE WHEN fold_cps_v1_before IS NULL "
            "        THEN NULL ELSE %s > fold_cps_v1_before END "
            "WHERE run_id = %s",
            (cps_after, cps_after, run_id),
        )
    log.info("fold_attribution_recorded", run_id=run_id, fold_cps_v1_after=cps_after)
```

Wire the call in `backtest.py` immediately after the fold's `backtest_results` row is
persisted (find the persistence call in the fold loop — same scope where `run_id` from
line 375 is still in hand; reuse the connection that wrote the row; wrap in
`try/except Exception` with `log.warning` — attribution must never fail a fold).

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/memory/test_fold_context.py tests/agents/ -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/swingrl/memory/training/fold_context.py src/swingrl/agents/backtest.py \
       tests/memory/test_fold_context.py
git commit -m "feat(19.1): post-fold attribution closure — fold_cps_v1_after + advice_was_effective"
```

---

### Task 10: Meta-orchestrator run-config payload enrichment

**Files:**
- Modify: `src/swingrl/memory/training/meta_orchestrator.py:262–336`
- Test: locate the orchestrator's existing tests first (`grep -rln "meta_orchestrator\|_query_run_config" tests/`) and extend them; if none exist, create `tests/memory/test_meta_orchestrator_payload.py` with the same mock style as the epoch-callback tests

- [ ] **Step 1: Write failing test:**

```python
class TestRunConfigPayload:
    """C1-PAYLOAD-02: run-config payload carries fold context + diagnoses."""

    def test_payload_contains_context_json(self) -> None:
        """C1-PAYLOAD-02: query embeds context with target_metric + fold lists +
        per-fold prev-iteration diagnoses."""
        import json
        # Arrange an orchestrator with mocked _current_regime_vector and a mocked
        # context provider (mirror the module's existing test arrangement for
        # urllib mocking), then:
        payload = ...  # captured request body dict
        ctx = json.loads(payload["query"].split("context=", 1)[1])
        assert ctx["target_metric"] == "cps_v1_multiplicative"
        assert "chronic_failure_folds" in ctx
        assert "protected_winner_folds" in ctx
        assert "prev_iter_diagnoses" in ctx  # {fold_number: label} for prev iteration
```

(Complete the arrangement by mirroring how existing orchestrator tests — or, if none, the
epoch-callback tests — mock the HTTP call: patch `urllib.request.urlopen` and capture
`req.data`. The assertion block above is the contract.)

- [ ] **Step 2: Run to verify failure.** Expected: FAIL — no `context=` in query.

- [ ] **Step 3: Implement.** In `_query_run_config` (lines 299–304), build context before the payload:

```python
        context: dict[str, Any] = {"target_metric": "cps_v1_multiplicative"}
        if self._database_url:
            try:
                import psycopg

                from swingrl.memory.training.cps_diagnosis import diagnose_fold
                from swingrl.memory.training.fold_context import load_fold_context
                from swingrl.reporting.iteration_report import load_fold_history

                ctx = load_fold_context(self._database_url, env_name, fold_number=-1)
                context["chronic_failure_folds"] = ctx["chronic_failure_folds"]
                context["protected_winner_folds"] = ctx["protected_winner_folds"]
                context["prev_iter_cps_v1"] = ctx["prev_iter_cps_v1"]
                with psycopg.connect(self._database_url) as conn:
                    history = load_fold_history(conn, env_name)
                if not history.empty:
                    last_iter = int(history["iteration_number"].max())
                    prev = history[
                        (history["iteration_number"] == last_iter)
                        & (history["algorithm"].str.lower() == algo_name.lower())
                    ]
                    context["prev_iter_diagnoses"] = {
                        int(r["fold_number"]): diagnose_fold(
                            r.to_dict(), env=env_name, algo=algo_name  # type: ignore[arg-type]
                        )["label"]
                        for _, r in prev.iterrows()
                    }
                else:
                    context["prev_iter_diagnoses"] = {}
            except Exception as exc:  # noqa: BLE001 — fail-open by design
                log.warning("run_config_context_failed", error=str(exc))
        payload = {
            "query": (
                f"TRAINING RUN CONFIG ADVICE: env={env_name} algo={algo_name}{iter_part} "
                f"current_regime={json.dumps(regime)} context={json.dumps(context)}"
            )
        }
```

(Check whether the orchestrator already holds a database URL attribute —
`grep -n "database_url\|_db" src/swingrl/memory/training/meta_orchestrator.py`. If not,
add `database_url: str | None = None` to `__init__` and wire it from the orchestrator's
construction site the same way trainer.py wires `self._config.system.database_url`.
`load_fold_history` rows must satisfy `diagnose_fold`'s FoldMetrics keys — verify the
DataFrame includes total_trades/profit_factor/win_rate/total_return/mdd and extend
`load_fold_history`'s SELECT if any are missing.)

- [ ] **Step 4: Run tests.** Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/swingrl/memory/training/meta_orchestrator.py tests/
git commit -m "feat(19.1): run-config payload — fold lists, prev-iter diagnoses, CPS target"
```

---

### Task 11: C3 prompt blocks (memory service)

**Files:**
- Modify: `services/memory/memory_agents/query.py` (`_build_epoch_system_prompt` 698–741, `_build_algo_system_prompt` 650–695)
- Test: Create `tests/test_memory_service_prompts.py` (copy the sys.path bootstrap from `tests/test_memory_service.py:28–40`)

- [ ] **Step 1: Write failing tests:**

```python
"""C3-PROMPT-01..03: goal / anti-pattern / fold-protection blocks present."""

from __future__ import annotations

import sys
from pathlib import Path

_MEMORY_SERVICE_DIR = Path(__file__).parent.parent / "services" / "memory"
sys.path.insert(0, str(_MEMORY_SERVICE_DIR))  # mirror tests/test_memory_service.py

from memory_agents import query as q  # noqa: E402


class TestGoalBlock:
    """C3-PROMPT-01: CPS v1 is stated as the single objective."""

    def test_epoch_prompt_states_cps_objective(self) -> None:
        p = q._build_epoch_system_prompt("ppo")
        assert "CPS v1" in p
        assert "single objective" in p.lower()
        assert "pass rate is not your goal" in p.lower()
        assert "multiplicative" in p.lower()

    def test_run_config_prompt_states_cps_objective(self) -> None:
        p = q._build_algo_system_prompt("ppo", {}, {})
        assert "CPS v1" in p


class TestAntiPatternBlock:
    """C3-PROMPT-02: empirical harm numbers + diagnosis→correction map."""

    def test_epoch_prompt_cites_harm_numbers(self) -> None:
        p = q._build_epoch_system_prompt("ppo")
        assert "2.7" in p and "5.1" in p  # control/treatment CPS ratios
        assert "trade_shy" in p
        assert "single_disaster" in p
        assert "do NOT damp risk further" in p

    def test_corrections_match_diagnosis_module(self) -> None:
        """The prompt's correction lines mirror DIAGNOSIS_CORRECTIONS verbatim."""
        p = q._build_epoch_system_prompt("ppo")
        for correction in (
            "increase participation; do NOT damp risk further",
            "tighten entry quality; reduce frequency, not size",
            "cap per-trade risk; leave frequency alone",
            "reduce frequency; raise conviction threshold",
        ):
            assert correction in p


class TestFoldProtectionBlock:
    """C3-PROMPT-03: protected winners untouched; chronic failures bounded."""

    def test_epoch_prompt_has_fold_protection(self) -> None:
        p = q._build_epoch_system_prompt("ppo")
        assert "protected_winner" in p
        assert "return the current weights unchanged" in p.lower()
        assert "chronic_failure" in p
```

(Verify `_build_algo_system_prompt`'s exact signature first — the recon shows it takes
algo + bounds dicts; match it.)

- [ ] **Step 2: Run to verify failures**

Run: `uv run pytest tests/test_memory_service_prompts.py -v`
Expected: FAIL — blocks absent

- [ ] **Step 3: Implement.** Add three module constants near the existing prompt builders in
`services/memory/memory_agents/query.py` (~line 640):

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

_ANTIPATTERN_BLOCK = (
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

_FOLD_PROTECTION_BLOCK = (
    "FOLD PROTECTION:\n"
    "The context JSON includes fold_role and the chronic_failure/protected_winner fold lists.\n"
    "- fold_role=protected_winner: return the current weights UNCHANGED. This fold wins "
    "without intervention; history shows intervention degrades it.\n"
    "- fold_role=chronic_failure: regime-conditional shaping is allowed; you may lean toward "
    "drawdown emphasis, always within the stated bounds.\n"
    "- fold_role=neutral: adjust only with clear diagnosis-backed cause.\n\n"
)
```

Insert `_GOAL_BLOCK + _ANTIPATTERN_BLOCK + _FOLD_PROTECTION_BLOCK` into the return
expressions of **both** `_build_epoch_system_prompt` and `_build_algo_system_prompt`,
immediately after each prompt's opening context section (read each builder and splice at
the natural seam — after the "You are..." intro, before bounds/instructions).

- [ ] **Step 4: Run service tests**

Run: `uv run pytest tests/test_memory_service_prompts.py tests/test_memory_service.py -v`
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add services/memory/memory_agents/query.py tests/test_memory_service_prompts.py
git commit -m "feat(19.1): goal/anti-pattern/fold-protection prompt blocks (CPS as driving force)"
```

---

### Task 12: C6 post-change documentation

**Files:**
- Create: `.planning/research/phase-19.1-prompt-refocus.md`
- Modify: `docs/training/` advice-flow page (final consistency pass)

- [ ] **Step 1: Write the C6 doc** — symmetric to C0: new prompt full text (all three blocks),
new payload schemas (epoch context JSON + run-config context JSON, with one real example
each), unchanged reward weights + rationale (spec D2 — control baseline preserved), the
diagnosis taxonomy + thresholds table, and a diff summary: each change → expected CPS
mechanism (e.g. "fold-protection block → stops adjustment of winners → removes the iter 3-4
tail-blowup channel").

- [ ] **Step 2: Commit**

```bash
git add .planning/research/phase-19.1-prompt-refocus.md docs/training/
git commit -m "docs(19.1): C6 post-change documentation — prompts, payloads, expected CPS impact"
```

---

### Task 13: Full suite → homelab CI (G3) → PR (G4)

- [ ] **Step 1: Full local suite** (background, 10-min timeout per `feedback_test_timeout`):

Run: `uv run pytest tests/ -q` (in background; check after completion)
Expected: 0 failures (DB tests skip without DATABASE_URL)

- [ ] **Step 2: Lint + types**: `uv run ruff check . && uv run ruff format --check . && uv run mypy src/`
Expected: clean

- [ ] **Step 3: Push branch + homelab CI** (CI bed is `~/swingrl`):

```bash
git push -u origin swingrl/19.1-training-refocus
cd ~/swingrl && git fetch origin && git checkout swingrl/19.1-training-refocus && \
  git pull origin swingrl/19.1-training-refocus && bash scripts/ci-homelab.sh --no-cache
```

Expected: `=== CI PASSED ===`, 0 failures. **G3 gate.**

- [ ] **Step 4: Update tracker** (`.planning/V1.1_EXECUTION_PLAN.md`): Stage 2 status → CI
green; changelog line. Commit.

- [ ] **Step 5: Open PR** to `main` with the Stage 2 Group C summary. **G4 = user merges.**
Note in the PR body: GHA coverage check expected RED per issue #18 (pre-existing).

---

## Runbook: Groups A / B / D / E (not code tasks — runtime gates)

Execute per `.planning/PHASE_19.1_HANDOFF.md`; sequence and gates:

| Step | When | Gate |
|---|---|---|
| **A** — wipe iter-5 artifacts (A1–A9) | before D, while no training runs | 🛑 separate plan-mode approval + **verified backup** immediately before the DELETEs |
| **B** — rebuild 3 containers (B1–B4) | before E; stack is currently DOWN (host restart 2026-06-11) so no in-flight-training risk, but get user approval before `up -d` | ⚠ user approval to start stack (standing rule) |
| **C7** — second rebuild of `swingrl` + `swingrl-memory` | after this plan's code merges | rides Group B verification greps |
| **D** — D1 wipe + per-iteration regeneration D2–D8 | after C merges + B live | 🛑 backup gate on D1; QA criteria from C0 annotations + spec §8 (halt → revise C → restart on failure) |
| **E** — fresh iter-5 run E1–E4 | after A–D | success = treatment/control CPS ratio → 1.0; capture in harm table |

---

## Self-review notes (resolved during plan writing)

- **Spec coverage:** C0→Task 1, C1→Tasks 8+10, C2→Tasks 2/3/6 (+4/5 for the trade signal the
  spec's leading indicator requires), C3→Task 11, C5→Tasks 7/8/9, C6→Task 12, C7+A/B/D/E→runbook.
  Spec D2 (no weight change) — verified: no task touches `DEFAULT_WEIGHTS`.
- **Deviation from handoff C1 (documented in §11-resolution 3):** `hmm_regime`/`vix_mean` are
  run-config-payload-only (orchestrator has them; the epoch callback doesn't and adding env
  plumbing for them is YAGNI when the regime is fold-constant anyway).
- **`fold_history` payload field from handoff C1** is intentionally narrowed to
  `prev_iter_cps_v1` + `prev_iter_diagnoses`: full 6-iteration per-fold history at epoch
  cadence is token-heavy for the Cerebras fast path and the diagnosis labels carry the
  decision-relevant signal. C6 documents this.
- **Type consistency check:** `CpsDiagnosis`/`diagnose_fold`/`diagnose_rolling`/
  `rolling_trade_rate`/`baseline_trade_rate`/`load_fold_context`/`record_fold_attribution`
  names match across Tasks 2–10. `FoldMetrics` reused from `cps.py` (not redefined).
