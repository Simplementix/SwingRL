"""DIAG-01..DIAG-05: Deterministic CPS diagnosis (spec §4.1 / §5.2)."""

from __future__ import annotations

import json

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
        "fold_number": fold_number,
        "sharpe": sharpe,
        "mdd": mdd,
        "total_return": total_return,
        "profit_factor": profit_factor,
        "win_rate": win_rate,
        "total_trades": total_trades,
        "sortino": sortino,
        "max_single_loss": max_single_loss,
        "overfitting_class": overfitting_class,
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

        # equity/ppo p25 = 446; equity median return = 0.0535
        d = diagnose_fold(make_fold(total_trades=200, total_return=0.01), env="equity", algo="ppo")
        assert d["label"] == "trade_shy"
        assert "trade_shy" in d["fired"]

    def test_low_trades_high_return_not_trade_shy(self) -> None:
        """DIAG-01: few trades but strong return is NOT trade_shy."""
        from swingrl.memory.training.cps_diagnosis import diagnose_fold

        d = diagnose_fold(make_fold(total_trades=200, total_return=0.12), env="equity", algo="ppo")
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
        d = diagnose_fold(make_fold(total_trades=600, profit_factor=1.1), env="equity", algo="ppo")
        assert d["label"] == "churning"

    def test_mixed_confidence_when_multiple_fire(self) -> None:
        """DIAG-05: ≥2 rules fired → precedence label + mixed confidence."""
        from swingrl.memory.training.cps_diagnosis import diagnose_fold

        d = diagnose_fold(
            make_fold(mdd=0.30, total_trades=600, profit_factor=1.1),
            env="equity",
            algo="ppo",
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


class TestDiagnoseFoldRobustness:
    """DIAG-07: None/NaN guard, inf-pf safety, boundary pins."""

    # ------------------------------------------------------------------
    # DIAG-07a: None in any consumed field → DataError (not TypeError)
    # ------------------------------------------------------------------

    @pytest.mark.parametrize(
        "field,value",
        [
            ("mdd", None),
            ("win_rate", None),
            ("total_trades", None),
            ("profit_factor", None),
            ("total_return", None),
        ],
    )
    def test_none_field_raises_data_error(self, field: str, value: None) -> None:
        """DIAG-07a: None in any consumed field raises DataError naming the field."""
        from swingrl.memory.training.cps_diagnosis import diagnose_fold

        fold = make_fold(**{field: value})  # type: ignore[arg-type]
        with pytest.raises(DataError, match=field):
            diagnose_fold(fold, env="equity", algo="ppo")

    # ------------------------------------------------------------------
    # DIAG-07b: NaN silently → DataError (not false "healthy")
    # ------------------------------------------------------------------

    @pytest.mark.parametrize(
        "field,value",
        [
            ("mdd", float("nan")),
            ("win_rate", float("nan")),
            ("profit_factor", float("nan")),
            ("total_return", float("nan")),
        ],
    )
    def test_nan_field_raises_data_error(self, field: str, value: float) -> None:
        """DIAG-07b: NaN in any float consumed field raises DataError (not 'healthy')."""
        from swingrl.memory.training.cps_diagnosis import diagnose_fold

        fold = make_fold(**{field: value})
        with pytest.raises(DataError, match=field):
            diagnose_fold(fold, env="equity", algo="ppo")

    # ------------------------------------------------------------------
    # DIAG-07c: profit_factor=inf, high trades → NOT churning; evidence JSON-safe
    # ------------------------------------------------------------------

    def test_inf_profit_factor_no_churning_and_json_safe_evidence(self) -> None:
        """DIAG-07c: inf profit_factor does not fire churning; evidence is JSON-finite."""
        from swingrl.memory.training.cps_diagnosis import diagnose_fold

        # equity/ppo p90 = 478; trades=600 > p90, but inf pf < 1.5 is False → no churning
        fold = make_fold(total_trades=600, profit_factor=float("inf"))
        d = diagnose_fold(fold, env="equity", algo="ppo")
        assert "churning" not in d["fired"]
        # evidence must be serialisable without allow_nan (LLM payload contract)
        json.dumps(d["evidence"], allow_nan=False)

    # ------------------------------------------------------------------
    # DIAG-07d: boundary pin — total_trades == p25 (446) does NOT fire trade_shy
    # ------------------------------------------------------------------

    def test_trades_equal_p25_not_trade_shy(self) -> None:
        """DIAG-07d: total_trades == p25 (446 for equity/ppo) does NOT fire trade_shy."""
        from swingrl.memory.training.cps_diagnosis import diagnose_fold

        # trade_shy requires trades < p25 (strict); at p25 it should not fire
        fold = make_fold(total_trades=446, total_return=0.01)  # low return, but trades == p25
        d = diagnose_fold(fold, env="equity", algo="ppo")
        assert "trade_shy" not in d["fired"]

    def test_trades_equal_p25_poor_selection_eligible(self) -> None:
        """DIAG-07d: total_trades == p25 with low win_rate IS eligible for poor_selection."""
        from swingrl.memory.training.cps_diagnosis import diagnose_fold

        # poor_selection requires trades >= p25; at p25 with win_rate < p25_win_rate it fires
        fold = make_fold(total_trades=446, win_rate=0.40)
        d = diagnose_fold(fold, env="equity", algo="ppo")
        assert "poor_selection" in d["fired"]

    # ------------------------------------------------------------------
    # DIAG-07e: boundary pin — mdd == 0.20 equity does NOT fire single_disaster
    # ------------------------------------------------------------------

    def test_mdd_at_threshold_not_single_disaster(self) -> None:
        """DIAG-07e: mdd == 0.20 equity (at threshold, not above) does NOT fire single_disaster."""
        from swingrl.memory.training.cps_diagnosis import diagnose_fold

        # single_disaster requires mdd > 0.20 (strict); at exactly 0.20 it must not fire
        fold = make_fold(mdd=0.20)
        d = diagnose_fold(fold, env="equity", algo="ppo")
        assert "single_disaster" not in d["fired"]

    # ------------------------------------------------------------------
    # DIAG-07f: boundary pin — win_rate == p25_win_rate (0.562) does NOT fire poor_selection
    # ------------------------------------------------------------------

    def test_win_rate_at_p25_not_poor_selection(self) -> None:
        """DIAG-07f: win_rate == p25_win_rate (0.562 equity/ppo) does NOT fire poor_selection."""
        from swingrl.memory.training.cps_diagnosis import diagnose_fold

        # poor_selection requires win_rate < p25_win_rate (strict); at p25_win_rate it must not fire
        fold = make_fold(win_rate=0.562, total_trades=450)
        d = diagnose_fold(fold, env="equity", algo="ppo")
        assert "poor_selection" not in d["fired"]


class TestDiagnoseRolling:
    """DIAG-06: mid-fold diagnosis from rolling indicators."""

    def test_trade_rate_collapse_labeled_trade_shy(self) -> None:
        """DIAG-06: trade rate < 50% of the fold's own baseline → trade_shy."""
        from swingrl.memory.training.cps_diagnosis import diagnose_rolling

        d = diagnose_rolling(
            trade_rate=0.10,
            baseline_trade_rate=0.40,
            rolling_win_rate=0.60,
            env="equity",
            algo="ppo",
        )
        assert d["label"] == "trade_shy"
        assert d["evidence"]["trade_rate"] == 0.10
        assert d["evidence"]["baseline_trade_rate"] == 0.40

    def test_winrate_collapse_normal_activity_poor_selection(self) -> None:
        """DIAG-06: activity normal but win rate < p25 baseline → poor_selection."""
        from swingrl.memory.training.cps_diagnosis import diagnose_rolling

        d = diagnose_rolling(
            trade_rate=0.40,
            baseline_trade_rate=0.40,
            rolling_win_rate=0.30,
            env="equity",
            algo="ppo",
        )
        assert d["label"] == "poor_selection"

    def test_no_signals_healthy(self) -> None:
        """DIAG-06: nothing fired → healthy/clear."""
        from swingrl.memory.training.cps_diagnosis import diagnose_rolling

        d = diagnose_rolling(
            trade_rate=0.40,
            baseline_trade_rate=0.40,
            rolling_win_rate=0.60,
            env="equity",
            algo="ppo",
        )
        assert d["label"] == "healthy"

    def test_zero_baseline_never_divides(self) -> None:
        """DIAG-06: baseline 0.0 (window not yet full) → healthy, no ZeroDivisionError."""
        from swingrl.memory.training.cps_diagnosis import diagnose_rolling

        d = diagnose_rolling(
            trade_rate=0.0,
            baseline_trade_rate=0.0,
            rolling_win_rate=0.60,
            env="equity",
            algo="ppo",
        )
        assert d["label"] == "healthy"

    def test_nan_input_raises_data_error(self) -> None:
        """DIAG-06: NaN in any float input raises DataError (not a false label)."""
        from swingrl.memory.training.cps_diagnosis import diagnose_rolling

        with pytest.raises(DataError):
            diagnose_rolling(
                trade_rate=float("nan"),
                baseline_trade_rate=0.40,
                rolling_win_rate=0.60,
                env="equity",
                algo="ppo",
            )
        with pytest.raises(DataError):
            diagnose_rolling(
                trade_rate=0.40,
                baseline_trade_rate=float("nan"),
                rolling_win_rate=0.60,
                env="equity",
                algo="ppo",
            )
        with pytest.raises(DataError):
            diagnose_rolling(
                trade_rate=0.40,
                baseline_trade_rate=0.40,
                rolling_win_rate=float("nan"),
                env="equity",
                algo="ppo",
            )

    def test_unknown_env_algo_raises_data_error(self) -> None:
        """DIAG-06: unknown (env, algo) pair raises DataError, never silent default."""
        from swingrl.memory.training.cps_diagnosis import diagnose_rolling

        with pytest.raises(DataError):
            diagnose_rolling(
                trade_rate=0.40,
                baseline_trade_rate=0.40,
                rolling_win_rate=0.60,
                env="forex",
                algo="ppo",
            )
