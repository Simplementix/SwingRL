"""Tests for observation vector assembler.

Verifies ObservationAssembler produces correct shapes (164 equity, 47 crypto),
deterministic assembly order, NaN-free output post-warmup, and default portfolio state.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from swingrl.config.schema import SwingRLConfig, load_config
from swingrl.features.assembler import ObservationAssembler


@pytest.fixture
def assembler_config(tmp_path: Path) -> SwingRLConfig:
    """Config with 8 equity symbols and 2 crypto symbols."""
    yaml_content = """\
trading_mode: paper
equity:
  symbols: [DIA, IWM, QQQ, SPY, VTI, XLE, XLF, XLK]
  max_position_size: 0.25
  max_drawdown_pct: 0.10
  daily_loss_limit_pct: 0.02
crypto:
  symbols: [BTCUSDT, ETHUSDT]
  max_position_size: 0.50
  max_drawdown_pct: 0.12
  daily_loss_limit_pct: 0.03
  min_order_usd: 10.0
capital:
  equity_usd: 400.0
  crypto_usd: 47.0
paths:
  data_dir: data/
  db_dir: db/
  models_dir: models/
  logs_dir: logs/
logging:
  level: INFO
  json_logs: false
system:
  database_url: "postgresql://test:test@localhost:5432/swingrl_test"  # pragma: allowlist secret
  duckdb_path: data/db/market_data.ddb
  sqlite_path: data/db/trading_ops.db
alerting:
  alert_cooldown_minutes: 30
  consecutive_failures_before_alert: 3
features:
  equity_zscore_window: 252
  crypto_zscore_window: 360
"""
    config_file = tmp_path / "swingrl.yaml"
    config_file.write_text(yaml_content)
    return load_config(config_file)


@pytest.fixture
def assembler(assembler_config: SwingRLConfig) -> ObservationAssembler:
    """ObservationAssembler with standard 8-equity / 2-crypto config."""
    return ObservationAssembler(assembler_config)


class TestEquityAssembly:
    """Tests for equity observation vector assembly."""

    def test_equity_shape_164(self, assembler: ObservationAssembler) -> None:
        """FEAT-07: assemble_equity returns (164,) ndarray."""
        per_asset = {
            s: np.ones(15) for s in sorted(["DIA", "IWM", "QQQ", "SPY", "VTI", "XLE", "XLF", "XLK"])
        }
        macro = np.ones(6)
        hmm = np.array([0.7, 0.3])
        turb = 1.5
        obs = assembler.assemble_equity(per_asset, macro, hmm, turb)
        assert obs.shape == (164,)

    def test_equity_per_asset_120_dims(self, assembler: ObservationAssembler) -> None:
        """FEAT-07: 15 features x 8 symbols = 120 per-asset dims."""
        symbols = sorted(["DIA", "IWM", "QQQ", "SPY", "VTI", "XLE", "XLF", "XLK"])
        per_asset = {s: np.full(15, i + 1.0) for i, s in enumerate(symbols)}
        macro = np.zeros(6)
        hmm = np.zeros(2)
        turb = 0.0
        obs = assembler.assemble_equity(per_asset, macro, hmm, turb)
        # First 120 dims are per-asset, each symbol gets 15 features
        for i, _sym in enumerate(symbols):
            block = obs[i * 15 : (i + 1) * 15]
            assert np.all(block == i + 1.0), f"Symbol {_sym} block mismatch"

    def test_equity_assembly_order(self, assembler: ObservationAssembler) -> None:
        """FEAT-07: Order is [per-asset alpha-sorted] + [macro] + [hmm] + [turb] + [portfolio]."""
        per_asset = {
            s: np.zeros(15)
            for s in sorted(["DIA", "IWM", "QQQ", "SPY", "VTI", "XLE", "XLF", "XLK"])
        }
        macro = np.full(6, 2.0)
        hmm = np.full(2, 3.0)
        turb = 4.0
        portfolio = np.full(35, 5.0)
        obs = assembler.assemble_equity(per_asset, macro, hmm, turb, portfolio)
        # Per-asset: 0:120
        assert np.all(obs[0:120] == 0.0)
        # Macro: 120:126
        assert np.all(obs[120:126] == 2.0)
        # HMM: 126:128
        assert np.all(obs[126:128] == 3.0)
        # Turbulence: 128
        assert obs[128] == 4.0
        # Portfolio: 129:164
        assert np.all(obs[129:164] == 5.0)

    def test_equity_default_portfolio_state(self, assembler: ObservationAssembler) -> None:
        """FEAT-07: Default portfolio = 100% cash, zero positions."""
        per_asset = {
            s: np.ones(15) for s in sorted(["DIA", "IWM", "QQQ", "SPY", "VTI", "XLE", "XLF", "XLK"])
        }
        macro = np.ones(6)
        hmm = np.array([0.5, 0.5])
        turb = 1.0
        obs = assembler.assemble_equity(per_asset, macro, hmm, turb)
        portfolio_slice = obs[129:164]
        assert portfolio_slice[0] == 1.0  # cash_ratio
        assert portfolio_slice[1] == 0.0  # exposure
        assert portfolio_slice[2] == 0.0  # daily_return
        assert np.all(portfolio_slice[3:] == 0.0)  # per-asset zeros

    def test_equity_no_nan(self, assembler: ObservationAssembler) -> None:
        """FEAT-07: No NaN in assembled observation (post-warmup guarantee)."""
        per_asset = {
            s: np.random.default_rng(42).random(15)
            for s in sorted(["DIA", "IWM", "QQQ", "SPY", "VTI", "XLE", "XLF", "XLK"])
        }
        macro = np.random.default_rng(43).random(6)
        hmm = np.array([0.6, 0.4])
        turb = 0.8
        obs = assembler.assemble_equity(per_asset, macro, hmm, turb)
        assert not np.any(np.isnan(obs))

    def test_equity_nan_raises_data_error(self, assembler: ObservationAssembler) -> None:
        """FEAT-07: NaN in input raises DataError."""
        from swingrl.utils.exceptions import DataError

        per_asset = {
            s: np.ones(15) for s in sorted(["DIA", "IWM", "QQQ", "SPY", "VTI", "XLE", "XLF", "XLK"])
        }
        per_asset["SPY"][3] = np.nan  # Inject NaN
        macro = np.ones(6)
        hmm = np.array([0.5, 0.5])
        turb = 1.0
        with pytest.raises(DataError, match="NaN"):
            assembler.assemble_equity(per_asset, macro, hmm, turb)


class TestCryptoAssembly:
    """Tests for crypto observation vector assembly."""

    def test_crypto_shape_47(self, assembler: ObservationAssembler) -> None:
        """FEAT-07: assemble_crypto returns (47,) ndarray."""
        per_asset = {s: np.ones(13) for s in ["BTCUSDT", "ETHUSDT"]}
        macro = np.ones(6)
        hmm = np.array([0.7, 0.3])
        turb = 1.5
        overnight = 4.0
        obs = assembler.assemble_crypto(per_asset, macro, hmm, turb, overnight)
        assert obs.shape == (47,)

    def test_crypto_per_asset_26_dims(self, assembler: ObservationAssembler) -> None:
        """FEAT-07: 13 features x 2 symbols = 26 per-asset dims."""
        per_asset = {"BTCUSDT": np.full(13, 1.0), "ETHUSDT": np.full(13, 2.0)}
        macro = np.zeros(6)
        hmm = np.zeros(2)
        turb = 0.0
        overnight = 0.0
        obs = assembler.assemble_crypto(per_asset, macro, hmm, turb, overnight)
        # BTCUSDT first (alpha-sorted)
        assert np.all(obs[0:13] == 1.0)
        assert np.all(obs[13:26] == 2.0)

    def test_crypto_includes_overnight_context(self, assembler: ObservationAssembler) -> None:
        """FEAT-07: Crypto has overnight_context (1 dim) between turbulence and portfolio."""
        per_asset = {s: np.zeros(13) for s in ["BTCUSDT", "ETHUSDT"]}
        macro = np.zeros(6)
        hmm = np.zeros(2)
        turb = 0.0
        overnight = 7.5
        obs = assembler.assemble_crypto(per_asset, macro, hmm, turb, overnight)
        # Layout: per_asset(26) + macro(6) + hmm(2) + turb(1) + overnight(1) + portfolio(11)
        overnight_idx = 26 + 6 + 2 + 1  # = 35
        assert obs[overnight_idx] == 7.5

    def test_crypto_default_portfolio_state(self, assembler: ObservationAssembler) -> None:
        """FEAT-07: Default portfolio = 100% cash for crypto."""
        per_asset = {s: np.ones(13) for s in ["BTCUSDT", "ETHUSDT"]}
        macro = np.ones(6)
        hmm = np.array([0.5, 0.5])
        turb = 1.0
        overnight = 2.0
        obs = assembler.assemble_crypto(per_asset, macro, hmm, turb, overnight)
        portfolio_slice = obs[36:47]
        assert portfolio_slice[0] == 1.0  # cash_ratio
        assert portfolio_slice[1] == 0.0  # exposure
        assert portfolio_slice[2] == 0.0  # daily_return
        assert np.all(portfolio_slice[3:] == 0.0)  # per-asset zeros

    def test_crypto_no_nan(self, assembler: ObservationAssembler) -> None:
        """FEAT-07: No NaN in crypto observation."""
        per_asset = {s: np.random.default_rng(42).random(13) for s in ["BTCUSDT", "ETHUSDT"]}
        macro = np.random.default_rng(43).random(6)
        hmm = np.array([0.6, 0.4])
        turb = 0.8
        overnight = 3.0
        obs = assembler.assemble_crypto(per_asset, macro, hmm, turb, overnight)
        assert not np.any(np.isnan(obs))


class TestFeatureNames:
    """Tests for feature name generation."""

    def test_equity_feature_names_count(self, assembler: ObservationAssembler) -> None:
        """FEAT-07: get_feature_names_equity returns 164 names."""
        names = assembler.get_feature_names_equity()
        assert len(names) == 164

    def test_crypto_feature_names_count(self, assembler: ObservationAssembler) -> None:
        """FEAT-07: get_feature_names_crypto returns 47 names."""
        names = assembler.get_feature_names_crypto()
        assert len(names) == 47

    def test_equity_feature_names_alpha_sorted(self, assembler: ObservationAssembler) -> None:
        """FEAT-07: Per-asset names start with alpha-sorted symbols."""
        names = assembler.get_feature_names_equity()
        # First symbol should be DIA (alpha-sorted)
        assert names[0].startswith("DIA_")
        # Second symbol block at index 15
        assert names[15].startswith("IWM_")

    def test_crypto_feature_names_overnight(self, assembler: ObservationAssembler) -> None:
        """FEAT-07: Crypto names include overnight context."""
        names = assembler.get_feature_names_crypto()
        assert "overnight_hours_since_equity_close" in names

    def test_feature_names_unique(self, assembler: ObservationAssembler) -> None:
        """FEAT-07: All feature names are unique."""
        equity_names = assembler.get_feature_names_equity()
        crypto_names = assembler.get_feature_names_crypto()
        assert len(equity_names) == len(set(equity_names))
        assert len(crypto_names) == len(set(crypto_names))


class TestSymbolOrdering:
    """Tests for alpha-sorted symbol ordering."""

    def test_symbols_alpha_sorted(self, assembler: ObservationAssembler) -> None:
        """FEAT-07: Symbols are alpha-sorted regardless of config order."""
        assert assembler._equity_symbols == sorted(assembler._equity_symbols)
        assert assembler._crypto_symbols == sorted(assembler._crypto_symbols)
