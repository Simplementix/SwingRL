"""HMM regime detection for equity and crypto environments.

Three-state Gaussian HMM produces P(bull), P(bear), and P(crisis) probabilities.
- Equity: SPY, full covariance, 1260-day rolling window, refit weekly
- Crypto: BTC (4H->daily aggregation), diag covariance, 2000 4H-bar window, refit daily

Inputs: log-returns + 20-day realized volatility (2 features).
Warm-start with ridge regularization prevents ValueError on near-singular covariance.
Label ordering: State 0 = bull (highest mean return), State 1 = bear (mid),
State 2 = crisis (lowest mean return + highest volatility).
"""

from __future__ import annotations

from datetime import UTC, date, datetime
from typing import Any, Literal

import numpy as np
import pandas as pd
import structlog
from hmmlearn.hmm import GaussianHMM

from swingrl.config.schema import SwingRLConfig
from swingrl.utils.exceptions import ModelError

log = structlog.get_logger(__name__)


class HMMRegimeDetector:
    """Three-state Gaussian HMM regime detector.

    Produces P(bull), P(bear), and P(crisis) probabilities for a given environment.
    Bull = state 0 (highest mean return), Bear = state 1 (mid), Crisis = state 2 (lowest + highest vol).
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
            Fitted GaussianHMM with bull=state0, bear=state1, crisis=state2 ordering.
        """
        data = self.compute_hmm_inputs(close)

        best_model: GaussianHMM | None = None
        best_score = -np.inf

        for i in range(self.n_inits):
            model = GaussianHMM(
                n_components=3,
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
            raise ModelError(msg)

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

        Uses manual parameter initialization:
        - Bull state: positive mean return, low volatility
        - Bear state: near-zero mean return, medium volatility
        - Crisis state: negative mean return, high volatility

        Args:
            close: Close price series (can be short, < 500 bars).

        Returns:
            Fitted GaussianHMM with bull=state0, bear=state1, crisis=state2 ordering.
        """
        data = self.compute_hmm_inputs(close)

        model = GaussianHMM(
            n_components=3,
            covariance_type=self.covariance_type,
            n_iter=self.n_iter,
            init_params="",
        )

        # Informed priors: bull=positive return/low vol, bear=neutral/mid vol, crisis=negative/high vol
        model.startprob_ = np.array([0.6, 0.3, 0.1])
        # 3x3 transmat: 0.90 stay, 0.05 transition each to the other two
        model.transmat_ = np.array(
            [
                [0.90, 0.05, 0.05],
                [0.05, 0.90, 0.05],
                [0.05, 0.05, 0.90],
            ]
        )

        # Estimate vol from data for prior
        data_std = data.std(axis=0)
        low_vol = data_std[1] * 0.5
        mid_vol = data_std[1] * 1.0
        high_vol = data_std[1] * 2.0

        # Means: [log_return, realized_vol_20d]
        model.means_ = np.array(
            [
                [0.001, low_vol],  # bull: positive return, low vol
                [0.000, mid_vol],  # bear: zero return, mid vol
                [-0.002, high_vol],  # crisis: negative return, high vol
            ]
        )

        n_features = data.shape[1]
        if self.covariance_type == "full":
            model.covars_ = np.array(
                [
                    np.eye(n_features) * 0.01,
                    np.eye(n_features) * 0.01,
                    np.eye(n_features) * 0.01,
                ]
            )
        else:
            model.covars_ = np.array([[0.01, 0.01], [0.01, 0.01], [0.01, 0.01]])

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
            raise ModelError(msg)

        data = self.compute_hmm_inputs(close)
        prev = self._model

        model = GaussianHMM(
            n_components=3,
            covariance_type=prev.covariance_type,
            n_iter=self.n_iter,
            init_params="",
        )

        model.startprob_ = prev.startprob_.copy()
        model.transmat_ = prev.transmat_.copy()
        model.means_ = prev.means_.copy()

        # Ridge regularization for numerical stability
        covars = prev.covars_.copy()

        if prev.covariance_type == "full":
            n_features = covars.shape[-1]
            for i in range(covars.shape[0]):
                covars[i] = (covars[i] + covars[i].T) / 2  # enforce symmetry
                covars[i] += np.eye(n_features) * self.ridge
            model.covars_ = covars
        elif prev.covariance_type == "diag":
            # hmmlearn stores diag covars as (n, d, d) internally but setter expects (n, d)
            if covars.ndim == 3:
                covars = np.array([np.diag(covars[i]) for i in range(covars.shape[0])])
            covars = np.maximum(covars, self.ridge)
            model.covars_ = covars
        else:
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
            (n_samples, 3) array where col 0 = P(bull), col 1 = P(bear), col 2 = P(crisis).

        Raises:
            ValueError: If no model has been fitted.
        """
        if self._model is None:
            msg = "No fitted model — call initial_fit() or cold_start_fit() first"
            raise ModelError(msg)

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
            probs: (1, 3) or (3,) array with [P(bull), P(bear), P(crisis)].
            log_likelihood: Log-likelihood of the fitted model.
        """
        flat = probs.flatten()
        p_bull = float(flat[0])
        p_bear = float(flat[1])
        p_crisis = float(flat[2]) if len(flat) > 2 else 0.0
        fitted_at = datetime.now(tz=UTC)

        db.execute(
            """INSERT INTO hmm_state_history
               (environment, date, p_bull, p_bear, p_crisis, log_likelihood, fitted_at)
               VALUES (%s, %s, %s, %s, %s, %s, %s)""",
            [self.environment, dt, p_bull, p_bear, p_crisis, log_likelihood, fitted_at],
        )

        log.info(
            "hmm_state_stored",
            environment=self.environment,
            date=str(dt),
            p_bull=p_bull,
            p_bear=p_bear,
            p_crisis=p_crisis,
        )

    def ensure_3state_schema(self, db: Any) -> None:
        """Ensure hmm_state_history table has p_crisis column.

        Runs ALTER TABLE ... ADD COLUMN IF NOT EXISTS to add p_crisis column
        for databases that were created with the 2-state schema.

        Args:
            db: DuckDB connection (or mock with execute method).
        """
        db.execute(
            "ALTER TABLE hmm_state_history ADD COLUMN IF NOT EXISTS p_crisis DOUBLE DEFAULT 0.0"
        )
        log.info("hmm_3state_schema_ensured", environment=self.environment)

    def _ensure_label_order(self, model: GaussianHMM) -> GaussianHMM:
        """Ensure bull=state0 (highest mean return), bear=state1, crisis=state2 (lowest mean).

        Sorts all 3 states by mean return descending. State 0 = bull (highest mean),
        State 1 = bear (mid), State 2 = crisis (lowest mean + highest vol).
        Applies ridge regularization to full covariance matrices before the setter
        to prevent ValueError from near-singular matrices after reordering.

        Args:
            model: Fitted GaussianHMM with 3 components.

        Returns:
            Model with consistent bull=state0, bear=state1, crisis=state2 ordering.
        """
        # Sort states by mean return (col 0) descending
        mean_returns = model.means_[:, 0]
        order = np.argsort(mean_returns)[::-1]  # descending

        # Always reorder to ensure consistent label assignment
        model.startprob_ = model.startprob_[order]
        model.transmat_ = model.transmat_[np.ix_(order, order)]
        model.means_ = model.means_[order]

        if model.covariance_type == "full":
            # Apply ridge regularization before setting (may have near-singular covariances)
            new_covars = model.covars_[order].copy()
            n_features = new_covars.shape[-1]
            for i in range(new_covars.shape[0]):
                # Enforce symmetry then add ridge
                new_covars[i] = (new_covars[i] + new_covars[i].T) / 2
                new_covars[i] += np.eye(n_features) * self.ridge
            model.covars_ = new_covars
        elif model.covariance_type == "diag":
            # hmmlearn stores diag covars as (n, d, d) internally but setter expects (n, d)
            raw = model.covars_[order].copy()
            if raw.ndim == 3:
                # Extract diagonal elements from each (d, d) matrix
                new_covars = np.array([np.diag(raw[i]) for i in range(raw.shape[0])])
            else:
                new_covars = raw
            new_covars = np.maximum(new_covars, self.ridge)
            model.covars_ = new_covars

        if list(order) != [0, 1, 2]:
            log.debug(
                "hmm_labels_reordered",
                environment=self.environment,
                new_order=order.tolist(),
            )

        return model
