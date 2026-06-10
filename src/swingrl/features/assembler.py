"""Observation vector assembler for RL environments.

Assembles flat NumPy observation vectors from computed feature groups.
Contract between feature engineering (Phase 5) and RL environments (Phase 6).

- Equity (default): (164,) = 120 per-asset + 6 macro + 2 HMM + 1 turbulence + 35 portfolio
- Equity (sentiment): (180,) = 136 per-asset + 6 macro + 2 HMM + 1 turbulence + 35 portfolio
- Crypto: (47,) = 26 per-asset + 6 macro + 2 HMM + 1 turbulence + 1 overnight + 11 portfolio

Assembly order is deterministic: [per-asset alpha-sorted] + [macro] + [HMM] + [turbulence]
+ [overnight (crypto only)] + [portfolio state].

Use equity_obs_dim(sentiment_enabled, n_symbols) to compute the expected dimension at runtime.
Never hardcode 164 or 180 -- always derive via equity_obs_dim().

Usage:
    from swingrl.features.assembler import ObservationAssembler, equity_obs_dim

    assembler = ObservationAssembler(config)
    obs = assembler.assemble_equity(per_asset, macro, hmm, turb)
    obs_with_sentiment = assembler.assemble_equity(per_asset, macro, hmm, turb,
                                                    sentiment_features=sentiment)
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import structlog

from swingrl.config.schema import SwingRLConfig
from swingrl.utils.exceptions import DataError

log = structlog.get_logger(__name__)

# Feature group sizes
EQUITY_PER_ASSET = 15  # 9 price action + 2 weekly + 4 fundamentals (backward-compat alias)
EQUITY_PER_ASSET_BASE = 15  # base per-asset dimension without sentiment
SENTIMENT_FEATURES_PER_ASSET = 2  # sentiment_score + sentiment_confidence
CRYPTO_PER_ASSET = 13  # 9 price action + 4 multi-timeframe
SHARED_MACRO = 6
HMM_REGIME = 2
TURBULENCE = 1
EQUITY_PORTFOLIO = 35  # 3 fixed + 4*8 per-asset
CRYPTO_PORTFOLIO = 11  # 3 fixed + 4*2 per-asset
OVERNIGHT_CONTEXT = 1  # crypto only

# Expected total dimensions (disabled-sentiment baseline; backward-compat aliases)
EQUITY_OBS_DIM = 164
CRYPTO_OBS_DIM = 47

# Sentiment feature names appended per-asset when sentiment is enabled
_SENTIMENT_FEATURE_NAMES = ["sentiment_score", "sentiment_confidence"]


def equity_per_asset_dim(sentiment_enabled: bool = False) -> int:
    """Return per-asset feature dimension for equity observations.

    Args:
        sentiment_enabled: Whether sentiment features are included.

    Returns:
        15 when disabled, 17 when enabled.
    """
    return EQUITY_PER_ASSET_BASE + (SENTIMENT_FEATURES_PER_ASSET if sentiment_enabled else 0)


def equity_obs_dim(sentiment_enabled: bool = False, n_equity_symbols: int = 8) -> int:
    """Return total equity observation dimension.

    Never hardcode 164 or 180 -- always call this function.

    Args:
        sentiment_enabled: Whether sentiment features are included.
        n_equity_symbols: Number of equity symbols in config.

    Returns:
        Total observation dimension (164 by default, 180 with sentiment and 8 symbols).
    """
    per_asset = equity_per_asset_dim(sentiment_enabled)
    return (
        (per_asset * n_equity_symbols) + SHARED_MACRO + HMM_REGIME + TURBULENCE + EQUITY_PORTFOLIO
    )


# Per-asset feature names
_EQUITY_FEATURE_NAMES = [
    "price_sma50_ratio",
    "price_sma200_ratio",
    "rsi_14",
    "macd_line",
    "macd_histogram",
    "bb_position",
    "atr_14_pct",
    "volume_sma20_ratio",
    "adx_14",
    "weekly_trend_dir",
    "weekly_rsi_14",
    "pe_zscore",
    "earnings_growth",
    "debt_to_equity",
    "dividend_yield",
]

_CRYPTO_FEATURE_NAMES = [
    "price_sma50_ratio",
    "price_sma200_ratio",
    "rsi_14",
    "macd_line",
    "macd_histogram",
    "bb_position",
    "atr_14_pct",
    "volume_sma20_ratio",
    "adx_14",
    "daily_trend_dir",
    "daily_rsi_14",
    "four_h_rsi_14",
    "four_h_price_sma20_ratio",
]

_MACRO_FEATURE_NAMES = [
    "macro_vix_zscore",
    "macro_yield_curve_spread",
    "macro_yield_curve_direction",
    "macro_fed_funds_90d_change",
    "macro_cpi_yoy",
    "macro_unemployment_3m_direction",
]

_HMM_FEATURE_NAMES = ["hmm_p_bull", "hmm_p_bear"]

_TURBULENCE_FEATURE_NAMES = ["turbulence_index"]

_OVERNIGHT_FEATURE_NAMES = ["overnight_hours_since_equity_close"]

_PORTFOLIO_FIXED_NAMES = ["portfolio_cash_ratio", "portfolio_exposure", "portfolio_daily_return"]

_PORTFOLIO_PER_ASSET_NAMES = [
    "weight",
    "weight_deviation",
    "unrealized_pnl_pct",
    "bars_since_trade",
]


class ObservationAssembler:
    """Assembles flat observation vectors for RL environments.

    Deterministic ordering: per-asset features (alpha-sorted by symbol),
    then shared macro, HMM regime, turbulence, overnight (crypto), portfolio state.
    """

    def __init__(self, config: SwingRLConfig) -> None:
        """Initialize with config-driven symbol lists.

        Args:
            config: SwingRLConfig with equity and crypto symbol lists.
        """
        self._equity_symbols: list[str] = sorted(config.equity.symbols)
        self._crypto_symbols: list[str] = sorted(config.crypto.symbols)

    def assemble_equity(
        self,
        per_asset_features: dict[str, np.ndarray],
        macro: np.ndarray,
        hmm_probs: np.ndarray,
        turbulence: float,
        portfolio_state: np.ndarray | None = None,
        sentiment_features: dict[str, tuple[float, float]] | None = None,
    ) -> np.ndarray:
        """Assemble equity observation vector.

        Args:
            per_asset_features: {symbol: (15,) array} for each equity ETF.
            macro: (6,) shared macro features.
            hmm_probs: (2,) array [P(bull), P(bear)].
            turbulence: Turbulence index scalar.
            portfolio_state: (35,) array or None for defaults.
            sentiment_features: Optional {symbol: (score, confidence)} dict.
                When provided, appends 2 sentiment features per asset, producing
                a (180,) vector instead of (164,) for 8-symbol configs.

        Returns:
            (164,) observation vector when sentiment_features is None,
            or (180,) when sentiment_features is provided (for 8-symbol configs).

        Raises:
            DataError: If assembled vector contains NaN values or shape mismatch.
        """
        if portfolio_state is None:
            portfolio_state = self._default_portfolio_state("equity")

        sentiment_enabled = sentiment_features is not None
        parts: list[np.ndarray] = []

        # Per-asset features in alpha-sorted order
        for symbol in self._equity_symbols:
            parts.append(per_asset_features[symbol])
            if sentiment_enabled and symbol in sentiment_features:  # type: ignore[operator]
                score, confidence = sentiment_features[symbol]  # type: ignore[index]
                parts.append(np.array([score, confidence]))

        parts.append(macro)
        parts.append(hmm_probs)
        parts.append(np.array([turbulence]))
        parts.append(portfolio_state)

        obs = np.concatenate(parts)

        expected_dim = equity_obs_dim(
            sentiment_enabled=sentiment_enabled,
            n_equity_symbols=len(self._equity_symbols),
        )
        if obs.shape != (expected_dim,):
            msg = f"Equity observation shape {obs.shape} != expected ({expected_dim},)"
            log.error("assembly_shape_mismatch", expected=expected_dim, actual=obs.shape)
            raise DataError(msg)

        if np.any(np.isnan(obs)):
            nan_count = int(np.sum(np.isnan(obs)))
            msg = f"Equity observation contains {nan_count} NaN values"
            log.error("assembly_nan_detected", nan_count=nan_count)
            raise DataError(msg)

        return np.asarray(obs)

    def assemble_crypto(
        self,
        per_asset_features: dict[str, np.ndarray],
        macro: np.ndarray,
        hmm_probs: np.ndarray,
        turbulence: float,
        overnight_context: float,
        portfolio_state: np.ndarray | None = None,
    ) -> np.ndarray:
        """Assemble crypto observation vector.

        Args:
            per_asset_features: {symbol: (13,) array} for BTC, ETH.
            macro: (6,) shared macro features.
            hmm_probs: (2,) array [P(bull), P(bear)].
            turbulence: Turbulence index scalar.
            overnight_context: Hours since equity market close.
            portfolio_state: (11,) array or None for defaults.

        Returns:
            (47,) observation vector.

        Raises:
            DataError: If assembled vector contains NaN values.
        """
        if portfolio_state is None:
            portfolio_state = self._default_portfolio_state("crypto")

        parts: list[np.ndarray] = []

        # Per-asset features in alpha-sorted order
        for symbol in self._crypto_symbols:
            parts.append(per_asset_features[symbol])

        parts.append(macro)
        parts.append(hmm_probs)
        parts.append(np.array([turbulence]))
        parts.append(np.array([overnight_context]))
        parts.append(portfolio_state)

        obs = np.concatenate(parts)

        if obs.shape != (CRYPTO_OBS_DIM,):
            msg = f"Crypto observation shape {obs.shape} != expected ({CRYPTO_OBS_DIM},)"
            log.error("assembly_shape_mismatch", expected=CRYPTO_OBS_DIM, actual=obs.shape)
            raise DataError(msg)

        if np.any(np.isnan(obs)):
            nan_count = int(np.sum(np.isnan(obs)))
            msg = f"Crypto observation contains {nan_count} NaN values"
            log.error("assembly_nan_detected", nan_count=nan_count)
            raise DataError(msg)

        return np.asarray(obs)

    def get_feature_names_equity(self, sentiment_enabled: bool = False) -> list[str]:
        """Return ordered list of feature names for equity observation.

        Args:
            sentiment_enabled: When True, appends sentiment_score and
                sentiment_confidence names after each asset's 15 base features,
                producing a 180-element list for 8-symbol configs.

        Returns:
            List of 164 feature name strings (default) or 180 when sentiment_enabled=True.
        """
        names: list[str] = []

        # Per-asset features
        for symbol in self._equity_symbols:
            for feat in _EQUITY_FEATURE_NAMES:
                names.append(f"{symbol}_{feat}")
            if sentiment_enabled:
                for feat in _SENTIMENT_FEATURE_NAMES:
                    names.append(f"{symbol}_{feat}")

        # Shared features
        names.extend(_MACRO_FEATURE_NAMES)
        names.extend(_HMM_FEATURE_NAMES)
        names.extend(_TURBULENCE_FEATURE_NAMES)

        # Portfolio state
        names.extend(_PORTFOLIO_FIXED_NAMES)
        for symbol in self._equity_symbols:
            for feat in _PORTFOLIO_PER_ASSET_NAMES:
                names.append(f"portfolio_{symbol}_{feat}")

        return names

    def get_feature_names_crypto(self) -> list[str]:
        """Return ordered list of 47 feature names for crypto observation.

        Returns:
            List of feature name strings matching assembly order.
        """
        names: list[str] = []

        # Per-asset features
        for symbol in self._crypto_symbols:
            for feat in _CRYPTO_FEATURE_NAMES:
                names.append(f"{symbol}_{feat}")

        # Shared features
        names.extend(_MACRO_FEATURE_NAMES)
        names.extend(_HMM_FEATURE_NAMES)
        names.extend(_TURBULENCE_FEATURE_NAMES)

        # Overnight context
        names.extend(_OVERNIGHT_FEATURE_NAMES)

        # Portfolio state
        names.extend(_PORTFOLIO_FIXED_NAMES)
        for symbol in self._crypto_symbols:
            for feat in _PORTFOLIO_PER_ASSET_NAMES:
                names.append(f"portfolio_{symbol}_{feat}")

        return names

    def _default_portfolio_state(self, environment: Literal["equity", "crypto"]) -> np.ndarray:
        """Generate default portfolio state (100% cash, no positions).

        Args:
            environment: "equity" or "crypto".

        Returns:
            (35,) for equity or (11,) for crypto with cash_ratio=1.0, rest zeros.
        """
        size = EQUITY_PORTFOLIO if environment == "equity" else CRYPTO_PORTFOLIO
        state = np.zeros(size)
        state[0] = 1.0  # cash_ratio = 100%
        return state
