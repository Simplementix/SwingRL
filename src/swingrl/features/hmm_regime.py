"""HMM regime detection for equity and crypto environments.

Two-state Gaussian HMM produces P(bull) and P(bear) probabilities.
- Equity: SPY, full covariance, 1260-day rolling window, refit weekly
- Crypto: BTC (4H->daily aggregation), diag covariance, 2000 4H-bar window, refit daily

Inputs: log-returns + 20-day realized volatility (2 features).
Warm-start with ridge regularization prevents ValueError on near-singular covariance.
"""

from __future__ import annotations

from datetime import UTC, date, datetime
from typing import Any, Literal

import numpy as np
import pandas as pd
import structlog
from hmmlearn.hmm import GaussianHMM

from swingrl.config.schema import SwingRLConfig

log = structlog.get_logger(__name__)


class HMMRegimeDetector:
    """Two-state Gaussian HMM regime detector.

    Produces P(bull) and P(bear) probabilities for a given environment.
    Bull = state 0 (higher mean return), Bear = state 1 (lower mean return).
    """

    def __init__(
        self,
        environment: Literal["equity", "crypto"],
        config: SwingRLConfig,
    ) -> None:
        """Initialize HMM detector for a specific environment.

        Args:
            environment: "equity" or "crypto" — determines covariance type and window.
            config: Validated SwingRLConfig with features section.
        """
        self.environment = environment

        if environment == "equity":
            self.covariance_type = "full"
            self.window = config.features.equity_hmm_window
        else:
            self.covariance_type = "diag"
            self.window = config.features.crypto_hmm_window

        self.n_iter = config.features.hmm_n_iter
        self.n_inits = config.features.hmm_n_inits
        self.ridge = config.features.hmm_ridge

        self._model: GaussianHMM | None = None

    def compute_hmm_inputs(self, close: pd.Series) -> np.ndarray:
        """Compute log-return + realized volatility from close prices.

        Args:
            close: Close price series.

        Returns:
            (n, 2) array with log_return and realized_vol_20d, NaN-free,
            limited to the last `self.window` rows.
        """
        ratio = close / close.shift(1)
        log_returns = pd.Series(np.log(ratio), index=close.index)
        realized_vol = log_returns.rolling(20).std()

        features = pd.DataFrame({"log_return": log_returns, "realized_vol_20d": realized_vol})
        features = features.dropna()

        # Limit to window
        if len(features) > self.window:
            features = features.iloc[-self.window :]

        return features.values

    def initial_fit(self, close: pd.Series) -> GaussianHMM:
        """Fit HMM with multiple random initializations, keep best.

        Args:
            close: Close price series (must have enough bars for window + warmup).

        Returns:
            Fitted GaussianHMM with bull=state0, bear=state1 ordering.
        """
        data = self.compute_hmm_inputs(close)

        best_model: GaussianHMM | None = None
        best_score = -np.inf

        for i in range(self.n_inits):
            model = GaussianHMM(
                n_components=2,
                covariance_type=self.covariance_type,
                n_iter=self.n_iter,
                random_state=i,
            )
            try:
                model.fit(data)
                score = model.score(data)
                if score > best_score:
                    best_score = score
                    best_model = model
            except ValueError:
                log.warning(
                    "hmm_init_failed",
                    environment=self.environment,
                    init_idx=i,
                )
                continue

        if best_model is None:
            msg = f"All {self.n_inits} HMM initializations failed"
            log.error("hmm_all_inits_failed", environment=self.environment)
            raise ValueError(msg)

        best_model = self._ensure_label_order(best_model)
        self._model = best_model

        log.info(
            "hmm_initial_fit",
            environment=self.environment,
            covariance_type=self.covariance_type,
            n_inits=self.n_inits,
            best_score=best_score,
        )

        return best_model

    def cold_start_fit(self, close: pd.Series) -> GaussianHMM:
        """Fit HMM with informed priors for limited data (< 500 bars).

        Uses manual parameter initialization: bull state has positive mean
        and low volatility, bear state has negative mean and high volatility.

        Args:
            close: Close price series (can be short, < 500 bars).

        Returns:
            Fitted GaussianHMM with bull=state0, bear=state1 ordering.
        """
        data = self.compute_hmm_inputs(close)

        model = GaussianHMM(
            n_components=2,
            covariance_type=self.covariance_type,
            n_iter=self.n_iter,
            init_params="",
        )

        # Informed priors: bull = positive return, low vol; bear = negative return, high vol
        model.startprob_ = np.array([0.5, 0.5])
        model.transmat_ = np.array([[0.95, 0.05], [0.05, 0.95]])

        # Estimate vol from data for prior
        data_std = data.std(axis=0)
        low_vol = data_std[1] * 0.5
        high_vol = data_std[1] * 1.5

        model.means_ = np.array([[0.001, low_vol], [-0.001, high_vol]])

        n_features = data.shape[1]
        if self.covariance_type == "full":
            model.covars_ = np.array([np.eye(n_features) * 0.01, np.eye(n_features) * 0.01])
        else:
            model.covars_ = np.array([[0.01, 0.01], [0.01, 0.01]])

        model.fit(data)
        model = self._ensure_label_order(model)
        self._model = model

        log.info(
            "hmm_cold_start_fit",
            environment=self.environment,
            n_bars=len(data),
        )

        return model

    def warm_start_refit(self, close: pd.Series) -> GaussianHMM:
        """Refit HMM with warm-start from previous model parameters.

        Applies ridge regularization to covariance matrices to prevent
        ValueError from near-singular matrices.

        Args:
            close: Close price series for the new window.

        Returns:
            Refitted GaussianHMM with consistent label ordering.

        Raises:
            ValueError: If no previous model exists.
        """
        if self._model is None:
            msg = "Cannot warm-start without a previously fitted model"
            raise ValueError(msg)

        data = self.compute_hmm_inputs(close)
        prev = self._model

        model = GaussianHMM(
            n_components=2,
            covariance_type=prev.covariance_type,
            n_iter=self.n_iter,
            init_params="",
        )

        model.startprob_ = prev.startprob_.copy()
        model.transmat_ = prev.transmat_.copy()
        model.means_ = prev.means_.copy()

        # Ridge regularization for numerical stability
        covars = prev.covars_.copy()
        n_features = covars.shape[-1]

        if prev.covariance_type == "full":
            for i in range(covars.shape[0]):
                covars[i] = (covars[i] + covars[i].T) / 2  # enforce symmetry
                covars[i] += np.eye(n_features) * self.ridge
        elif prev.covariance_type == "diag":
            covars = np.maximum(covars, self.ridge)

        model.covars_ = covars
        model.fit(data)
        model = self._ensure_label_order(model)
        self._model = model

        log.info(
            "hmm_warm_start_refit",
            environment=self.environment,
            ridge=self.ridge,
        )

        return model

    def predict_proba(self, close: pd.Series) -> np.ndarray:
        """Predict regime probabilities for each bar.

        Args:
            close: Close price series.

        Returns:
            (n_samples, 2) array where col 0 = P(bull), col 1 = P(bear).

        Raises:
            ValueError: If no model has been fitted.
        """
        if self._model is None:
            msg = "No fitted model — call initial_fit() or cold_start_fit() first"
            raise ValueError(msg)

        data = self.compute_hmm_inputs(close)
        result: np.ndarray = self._model.predict_proba(data)
        return result

    def store_hmm_state(
        self,
        db: Any,
        dt: date,
        probs: np.ndarray,
        log_likelihood: float,
    ) -> None:
        """Write HMM state to DuckDB hmm_state_history table.

        Args:
            db: DuckDB connection (or mock with execute method).
            dt: Date for this observation.
            probs: (1, 2) or (2,) array with [P(bull), P(bear)].
            log_likelihood: Log-likelihood of the fitted model.
        """
        flat = probs.flatten()
        p_bull = float(flat[0])
        p_bear = float(flat[1])
        fitted_at = datetime.now(tz=UTC)

        db.execute(
            """INSERT INTO hmm_state_history
               (environment, date, p_bull, p_bear, log_likelihood, fitted_at)
               VALUES (?, ?, ?, ?, ?, ?)""",
            [self.environment, dt, p_bull, p_bear, log_likelihood, fitted_at],
        )

        log.info(
            "hmm_state_stored",
            environment=self.environment,
            date=str(dt),
            p_bull=p_bull,
            p_bear=p_bear,
        )

    def _ensure_label_order(self, model: GaussianHMM) -> GaussianHMM:
        """Ensure bull=state0 (higher mean return) and bear=state1.

        If state 0 has lower mean return than state 1, swap all parameters.

        Args:
            model: Fitted GaussianHMM.

        Returns:
            Model with consistent bull=state0, bear=state1 ordering.
        """
        if model.means_[0, 0] < model.means_[1, 0]:
            # Swap states
            swap = [1, 0]
            model.startprob_ = model.startprob_[swap]
            model.transmat_ = model.transmat_[np.ix_(swap, swap)]
            model.means_ = model.means_[swap]

            if model.covariance_type == "full":
                model.covars_ = model.covars_[swap]
            elif model.covariance_type == "diag":
                model.covars_ = model.covars_[swap]

            log.debug("hmm_labels_swapped", environment=self.environment)

        return model
