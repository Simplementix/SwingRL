"""RL trading environment components.

Provides portfolio simulation, reward functions, action processing,
and Gymnasium-compatible trading environments for equity and crypto.

Importing this module registers both environments with Gymnasium:
- StockTradingEnv-v0: 8-ETF equity daily bar environment (252-step episodes)
- CryptoTradingEnv-v0: BTC/ETH crypto 4H bar environment (540-step episodes)
"""

from __future__ import annotations

from gymnasium.envs.registration import register

register(
    id="StockTradingEnv-v0",
    entry_point="swingrl.envs.equity:StockTradingEnv",
)

register(
    id="CryptoTradingEnv-v0",
    entry_point="swingrl.envs.crypto:CryptoTradingEnv",
)

from swingrl.envs.crypto import CryptoTradingEnv  # noqa: E402
from swingrl.envs.equity import StockTradingEnv  # noqa: E402

__all__ = ["CryptoTradingEnv", "StockTradingEnv"]
