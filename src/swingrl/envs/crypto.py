"""Crypto 4H trading environment for RL training.

CryptoTradingEnv specializes BaseTradingEnv for BTC/ETH 4-hour bar trading
with 540-step episodes (approximately 3 months) and random start positioning
within the training window for diverse episode sampling.

TRAIN-02: Crypto environment with Gymnasium-compatible step/reset contract.
"""

from __future__ import annotations

import numpy as np
import structlog

from swingrl.config.schema import SwingRLConfig
from swingrl.envs.base import BaseTradingEnv
from swingrl.utils.exceptions import DataError

log = structlog.get_logger(__name__)


class CryptoTradingEnv(BaseTradingEnv):
    """Crypto 4H bar trading environment for BTC/ETH.

    Extends BaseTradingEnv with crypto-specific configuration:
    - 540-step episodes (approximately 3 months of 4H bars)
    - Random start positioning within training window via np_random
    - 2 crypto assets (BTC, ETH)

    Args:
        features: Pre-loaded feature array, shape (n_steps, 47), float32.
        prices: Pre-loaded close price array, shape (n_steps, 2), float32.
        config: Validated SwingRLConfig instance.
        render_mode: Gymnasium render mode (optional).
    """

    def __init__(
        self,
        features: np.ndarray,
        prices: np.ndarray,
        config: SwingRLConfig,
        render_mode: str | None = None,
    ) -> None:
        super().__init__(
            features=features,
            prices=prices,
            config=config,
            environment="crypto",
            render_mode=render_mode,
        )

    def _select_start_step(self) -> int:
        """Select random start step within the training window.

        Uses self.np_random (set by super().reset(seed=seed)) for
        reproducible random start positions across episodes.

        Returns:
            Random starting index into features/prices arrays.
        """
        max_start = len(self._features) - self._episode_bars
        if max_start <= 0:
            return 0
        return int(self.np_random.integers(0, max_start))

    @classmethod
    def from_arrays(
        cls,
        features: np.ndarray,
        prices: np.ndarray,
        config: SwingRLConfig,
        render_mode: str | None = None,
    ) -> CryptoTradingEnv:
        """Factory method to create CryptoTradingEnv from pre-loaded arrays.

        Validates array shapes before construction.

        Args:
            features: Feature array, shape (n_steps, 47), float32.
            prices: Price array, shape (n_steps, n_assets), float32.
            config: Validated SwingRLConfig instance.
            render_mode: Gymnasium render mode (optional).

        Returns:
            Configured CryptoTradingEnv instance.

        Raises:
            ValueError: If array shapes are inconsistent.
        """
        n_assets = len(config.crypto.symbols)
        if features.ndim != 2:
            raise DataError(f"features must be 2D, got {features.ndim}D")
        if prices.ndim != 2:
            raise DataError(f"prices must be 2D, got {prices.ndim}D")
        if features.shape[0] != prices.shape[0]:
            raise DataError(
                f"features and prices must have same number of steps: "
                f"{features.shape[0]} != {prices.shape[0]}"
            )
        if prices.shape[1] != n_assets:
            raise DataError(
                f"prices must have {n_assets} columns (crypto symbols), got {prices.shape[1]}"
            )

        log.info(
            "crypto_env_created",
            n_steps=features.shape[0],
            obs_dim=features.shape[1],
            n_assets=n_assets,
        )

        return cls(
            features=features,
            prices=prices,
            config=config,
            render_mode=render_mode,
        )
