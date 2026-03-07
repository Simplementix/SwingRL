"""Tests for HMM regime detection.

FEAT-05: HMM regime produces P(bull)+P(bear)=1.0 with warm-start and label consistency.
"""

from __future__ import annotations

from datetime import date
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from swingrl.config.schema import SwingRLConfig


def _make_bull_bear_close(n_bull: int = 300, n_bear: int = 300, seed: int = 42) -> pd.Series:
    """Create synthetic close prices with distinct bull/bear regimes.

    Bull regime: positive drift, low volatility.
    Bear regime: negative drift, high volatility.
    """
    rng = np.random.default_rng(seed)
    bull_returns = rng.normal(0.001, 0.005, n_bull)  # +0.1%/day, low vol
    bear_returns = rng.normal(-0.001, 0.02, n_bear)  # -0.1%/day, high vol
    all_returns = np.concatenate([bull_returns, bear_returns])
    prices = 100.0 * np.exp(np.cumsum(all_returns))
    return pd.Series(prices, name="close")


@pytest.fixture()
def config() -> SwingRLConfig:
    """Minimal config for HMM tests."""
    return SwingRLConfig(
        features={
            "equity_hmm_window": 500,
            "crypto_hmm_window": 500,
            "hmm_n_iter": 100,
            "hmm_n_inits": 3,
            "hmm_ridge": 1e-6,
        }
    )


@pytest.fixture()
def bull_bear_close() -> pd.Series:
    """600-bar synthetic close with bull then bear regimes."""
    return _make_bull_bear_close()


class TestComputeHmmInputs:
    """Test compute_hmm_inputs produces correct features."""

    def test_output_shape_and_columns(self, config: SwingRLConfig) -> None:
        """compute_hmm_inputs returns (n, 2) array of log_return + realized_vol_20d."""
        from swingrl.features.hmm_regime import HMMRegimeDetector

        detector = HMMRegimeDetector(environment="equity", config=config)
        close = pd.Series(np.linspace(100, 150, 100))
        inputs = detector.compute_hmm_inputs(close)
        assert inputs.ndim == 2
        assert inputs.shape[1] == 2
        # No NaN in output (dropped)
        assert not np.isnan(inputs).any()

    def test_window_limits_output(self, config: SwingRLConfig) -> None:
        """Output is limited to last `window` rows."""
        from swingrl.features.hmm_regime import HMMRegimeDetector

        detector = HMMRegimeDetector(environment="equity", config=config)
        close = pd.Series(np.linspace(100, 200, 1000))
        inputs = detector.compute_hmm_inputs(close)
        # Window is 500, but after dropping NaN from rolling(20), max possible is <= 500
        assert inputs.shape[0] <= 500


class TestInitialFit:
    """Test initial_fit on synthetic data."""

    def test_produces_two_state_model(
        self, config: SwingRLConfig, bull_bear_close: pd.Series
    ) -> None:
        """initial_fit returns a GaussianHMM with 2 states."""
        from swingrl.features.hmm_regime import HMMRegimeDetector

        detector = HMMRegimeDetector(environment="equity", config=config)
        model = detector.initial_fit(bull_bear_close)
        assert model.n_components == 2

    def test_label_ordering_bull_state0(
        self, config: SwingRLConfig, bull_bear_close: pd.Series
    ) -> None:
        """After fit, state 0 has higher mean return (bull)."""
        from swingrl.features.hmm_regime import HMMRegimeDetector

        detector = HMMRegimeDetector(environment="equity", config=config)
        model = detector.initial_fit(bull_bear_close)
        assert model.means_[0, 0] >= model.means_[1, 0]

    def test_multiple_inits_selects_best(self, config: SwingRLConfig) -> None:
        """Multiple random initializations select the model with best log-likelihood."""
        from swingrl.features.hmm_regime import HMMRegimeDetector

        detector = HMMRegimeDetector(environment="equity", config=config)
        close = _make_bull_bear_close(seed=123)
        model = detector.initial_fit(close)
        # Model should be fitted and have a score
        inputs = detector.compute_hmm_inputs(close)
        score = model.score(inputs)
        assert np.isfinite(score)


class TestPredictProba:
    """Test predict_proba returns valid probabilities."""

    def test_shape_and_sum(self, config: SwingRLConfig, bull_bear_close: pd.Series) -> None:
        """predict_proba returns (n, 2) where each row sums to 1.0."""
        from swingrl.features.hmm_regime import HMMRegimeDetector

        detector = HMMRegimeDetector(environment="equity", config=config)
        detector.initial_fit(bull_bear_close)
        probs = detector.predict_proba(bull_bear_close)
        assert probs.ndim == 2
        assert probs.shape[1] == 2
        np.testing.assert_allclose(probs.sum(axis=1), 1.0, atol=1e-6)


class TestWarmStartRefit:
    """Test warm-start refit with ridge regularization."""

    def test_produces_valid_model(self, config: SwingRLConfig, bull_bear_close: pd.Series) -> None:
        """warm_start_refit produces a valid model without ValueError."""
        from swingrl.features.hmm_regime import HMMRegimeDetector

        detector = HMMRegimeDetector(environment="equity", config=config)
        detector.initial_fit(bull_bear_close)
        # Refit on slightly different data
        shifted = bull_bear_close.iloc[10:].reset_index(drop=True)
        model = detector.warm_start_refit(shifted)
        assert model.n_components == 2

    def test_preserves_label_ordering(
        self, config: SwingRLConfig, bull_bear_close: pd.Series
    ) -> None:
        """warm_start_refit preserves bull=state0 ordering."""
        from swingrl.features.hmm_regime import HMMRegimeDetector

        detector = HMMRegimeDetector(environment="equity", config=config)
        detector.initial_fit(bull_bear_close)
        shifted = bull_bear_close.iloc[10:].reset_index(drop=True)
        model = detector.warm_start_refit(shifted)
        assert model.means_[0, 0] >= model.means_[1, 0]

    def test_probabilities_sum_after_refit(
        self, config: SwingRLConfig, bull_bear_close: pd.Series
    ) -> None:
        """Probabilities still sum to 1.0 after warm-start refit."""
        from swingrl.features.hmm_regime import HMMRegimeDetector

        detector = HMMRegimeDetector(environment="equity", config=config)
        detector.initial_fit(bull_bear_close)
        shifted = bull_bear_close.iloc[10:].reset_index(drop=True)
        detector.warm_start_refit(shifted)
        probs = detector.predict_proba(shifted)
        np.testing.assert_allclose(probs.sum(axis=1), 1.0, atol=1e-6)


class TestColdStartFit:
    """Test cold-start with informed priors for limited data."""

    def test_works_with_few_bars(self, config: SwingRLConfig) -> None:
        """cold_start_fit works with < 500 bars of data."""
        from swingrl.features.hmm_regime import HMMRegimeDetector

        detector = HMMRegimeDetector(environment="equity", config=config)
        short_close = _make_bull_bear_close(n_bull=100, n_bear=100)
        model = detector.cold_start_fit(short_close)
        assert model.n_components == 2
        # Label ordering still holds
        assert model.means_[0, 0] >= model.means_[1, 0]


class TestCovarianceTypes:
    """Test environment-specific covariance types."""

    def test_equity_uses_full(self, config: SwingRLConfig) -> None:
        """Equity HMM uses covariance_type='full'."""
        from swingrl.features.hmm_regime import HMMRegimeDetector

        detector = HMMRegimeDetector(environment="equity", config=config)
        assert detector.covariance_type == "full"

    def test_crypto_uses_diag(self, config: SwingRLConfig) -> None:
        """Crypto HMM uses covariance_type='diag'."""
        from swingrl.features.hmm_regime import HMMRegimeDetector

        detector = HMMRegimeDetector(environment="crypto", config=config)
        assert detector.covariance_type == "diag"

    def test_crypto_fit_diag(self, config: SwingRLConfig) -> None:
        """Crypto HMM fits with diag covariance successfully."""
        from swingrl.features.hmm_regime import HMMRegimeDetector

        detector = HMMRegimeDetector(environment="crypto", config=config)
        close = _make_bull_bear_close(seed=99)
        model = detector.initial_fit(close)
        assert model.covariance_type == "diag"
        probs = detector.predict_proba(close)
        np.testing.assert_allclose(probs.sum(axis=1), 1.0, atol=1e-6)


class TestStoreHmmState:
    """Test store_hmm_state writes to DuckDB."""

    def test_calls_db_with_correct_data(self, config: SwingRLConfig) -> None:
        """store_hmm_state writes p_bull, p_bear, log_likelihood."""
        from swingrl.features.hmm_regime import HMMRegimeDetector

        detector = HMMRegimeDetector(environment="equity", config=config)
        mock_db = MagicMock()
        probs = np.array([[0.8, 0.2]])
        detector.store_hmm_state(
            db=mock_db,
            dt=date(2026, 1, 15),
            probs=probs,
            log_likelihood=-100.5,
        )
        mock_db.execute.assert_called_once()
        call_args = mock_db.execute.call_args
        sql = call_args[0][0]
        assert "hmm_state_history" in sql
        params = call_args[0][1]
        assert params[0] == "equity"
        assert params[2] == pytest.approx(0.8)
        assert params[3] == pytest.approx(0.2)
        assert params[4] == pytest.approx(-100.5)
