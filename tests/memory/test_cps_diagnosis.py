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
