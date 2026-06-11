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

import math
from typing import Literal, TypedDict

import structlog

from swingrl.metrics.cps import FoldMetrics
from swingrl.utils.exceptions import DataError

log = structlog.get_logger(__name__)

DiagnosisLabel = Literal["trade_shy", "poor_selection", "single_disaster", "churning", "healthy"]


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
    """Labeled cause for a fold's CPS state.

    Contract: ``evidence`` values are JSON-finite floats — infinite or NaN values
    are excluded by the rule logic so that ``json.dumps(evidence, allow_nan=False)``
    always succeeds.  This is required because the dict is serialised directly into
    LLM payloads.
    """

    label: DiagnosisLabel
    fired: list[str]
    confidence: Literal["clear", "mixed"]
    evidence: dict[str, float]


# Derived 2026-06-11 from iter 0-4 backtest_results (SQL in C0 baseline doc).
# Regenerate after each completed training iteration via the percentile SQL in
# .planning/research/phase-19.1-prompt-baseline.md §8; stale baselines mislabel
# as the policy improves.
TRADE_BASELINES: dict[tuple[str, str], TradeBaseline] = {
    ("crypto", "a2c"): {
        "p10": 67,
        "p25": 160,
        "med": 376,
        "p75": 619,
        "p90": 777,
        "med_win_rate": 0.576,
        "p25_win_rate": 0.490,
    },
    ("crypto", "ppo"): {
        "p10": 913,
        "p25": 943,
        "med": 974,
        "p75": 996,
        "p90": 1013,
        "med_win_rate": 0.615,
        "p25_win_rate": 0.576,
    },
    ("crypto", "sac"): {
        "p10": 14,
        "p25": 29,
        "med": 66,
        "p75": 225,
        "p90": 403,
        "med_win_rate": 0.425,
        "p25_win_rate": 0.232,
    },
    ("equity", "a2c"): {
        "p10": 99,
        "p25": 180,
        "med": 311,
        "p75": 381,
        "p90": 419,
        "med_win_rate": 0.647,
        "p25_win_rate": 0.480,
    },
    ("equity", "ppo"): {
        "p10": 430,
        "p25": 446,
        "med": 458,
        "p75": 469,
        "p90": 478,
        "med_win_rate": 0.644,
        "p25_win_rate": 0.562,
    },
    ("equity", "sac"): {
        "p10": 59,
        "p25": 100,
        "med": 171,
        "p75": 256,
        "p90": 440,
        "med_win_rate": 0.674,
        "p25_win_rate": 0.479,
    },
}

# Control folds never exceeded mdd 0.069 (equity) / 0.413 (crypto); treatment
# blowups reached 0.382 / 0.483. Thresholds sit between the two regimes.
MDD_DISASTER_THRESHOLD: dict[str, float] = {"equity": 0.20, "crypto": 0.40}

# Median total_return across all iter 0-4 folds per env (C0 baseline doc,
# actual query output 2026-06-11).
ENV_MEDIAN_RETURN: dict[str, float] = {"equity": 0.0535, "crypto": 0.3893}

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
    "single_disaster",
    "churning",
    "trade_shy",
    "poor_selection",
)


def _baseline(env: str, algo: str) -> TradeBaseline:
    """Look up the (env, algo) baseline or raise DataError."""
    key = (env.lower(), algo.lower())
    if key not in TRADE_BASELINES:
        log.error("diagnosis_unknown_env_algo", env=env, algo=algo)
        raise DataError(f"No trade baseline for env={env!r} algo={algo!r}")
    return TRADE_BASELINES[key]


def _validate_fold_fields(fold: FoldMetrics) -> None:
    """Raise DataError if any consumed field is None or NaN.

    FoldMetrics annotations declare these as non-Optional, but real DB rows are
    nullable.  A None or NaN value would silently produce a false label or raise
    an untyped TypeError — both worse than a typed refusal.  Callers catch
    DataError and fall back to no-diagnosis.

    Args:
        fold: Per-fold metrics dict to validate.

    Raises:
        DataError: If any of the five consumed fields is None or NaN.
    """
    float_fields = ("mdd", "win_rate", "profit_factor", "total_return")
    for field in float_fields:
        val: float | None = fold.get(field)  # type: ignore[assignment]
        if val is None:
            log.error("diagnosis_null_field", field=field, fold_number=fold.get("fold_number"))
            raise DataError(
                f"diagnose_fold: field '{field}' is None for fold {fold.get('fold_number')!r}"
            )
        if math.isnan(val):
            log.error("diagnosis_nan_field", field=field, fold_number=fold.get("fold_number"))
            raise DataError(
                f"diagnose_fold: field '{field}' is NaN for fold {fold.get('fold_number')!r}"
            )

    trades: int | None = fold.get("total_trades")  # type: ignore[assignment]
    if trades is None:
        log.error("diagnosis_null_field", field="total_trades", fold_number=fold.get("fold_number"))
        raise DataError(
            f"diagnose_fold: field 'total_trades' is None for fold {fold.get('fold_number')!r}"
        )


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
        DataError: Any of the five consumed fields (mdd, win_rate, total_trades,
            profit_factor, total_return) is None or NaN — typed refusal so callers
            can fall back to no-diagnosis rather than receiving a false label.
    """
    _validate_fold_fields(fold)
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
