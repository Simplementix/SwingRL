"""Phase 14 tests: FEAT-09 (compare_features CLI) and FEAT-10 (sentiment integration).

Tests:
  - TestCompareFeaturesScript: CLI entrypoint for A/B feature comparison
  - TestEquityDimHelpers: Config-aware dimension helpers
  - TestAssembleSentiment: Sentiment-aware assembly producing (172,) vectors
  - TestFeatureNames: Feature name lists with and without sentiment
  - TestEnvObsSpace: BaseTradingEnv observation space shape with sentiment toggle
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from swingrl.config.schema import SwingRLConfig, load_config
from swingrl.features.assembler import (
    EQUITY_PER_ASSET_BASE,
    SENTIMENT_FEATURES_PER_ASSET,
    ObservationAssembler,
    equity_obs_dim,
    equity_per_asset_dim,
)

# Add scripts/ to sys.path so we can import compare_features
_SCRIPTS_DIR = Path(__file__).parent.parent / "scripts"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from compare_features import main as cf_main  # noqa: E402

# ---------------------------------------------------------------------------
# Helpers / shared fixtures
# ---------------------------------------------------------------------------


def _make_accepted_metrics() -> dict[str, float]:
    """Metrics where candidate beats baseline by > 0.05."""
    return {"train_sharpe": 1.5, "validation_sharpe": 1.1}


def _make_baseline_metrics() -> dict[str, float]:
    return {"train_sharpe": 1.0, "validation_sharpe": 1.0}


def _make_low_metrics() -> dict[str, float]:
    return {"train_sharpe": 0.8, "validation_sharpe": 0.9}


def _write_json(tmp_path: Path, name: str, data: dict[str, Any]) -> Path:
    p = tmp_path / name
    p.write_text(json.dumps(data))
    return p


# ---------------------------------------------------------------------------
# TestCompareFeaturesScript
# ---------------------------------------------------------------------------


class TestCompareFeaturesScript:
    """FEAT-09: CLI entrypoint tests."""

    def test_cli_returns_0_on_accept(self, tmp_path: Path) -> None:
        """FEAT-09: Exit code 0 when candidate is accepted."""
        baseline = _write_json(tmp_path, "b.json", _make_baseline_metrics())
        candidate = _write_json(tmp_path, "c.json", _make_accepted_metrics())
        result = cf_main(
            ["--baseline", str(baseline), "--candidate", str(candidate), "--threshold", "0.01"]
        )
        assert result == 0

    def test_cli_returns_1_on_reject(self, tmp_path: Path) -> None:
        """FEAT-09: Exit code 1 when candidate is rejected (Sharpe too low)."""
        baseline = _write_json(tmp_path, "b.json", _make_accepted_metrics())
        candidate = _write_json(tmp_path, "c.json", _make_low_metrics())
        result = cf_main(["--baseline", str(baseline), "--candidate", str(candidate)])
        assert result == 1

    def test_cli_reads_json_files(self, tmp_path: Path) -> None:
        """FEAT-09: Reads baseline/candidate from JSON file paths."""
        baseline = _write_json(tmp_path, "b.json", _make_baseline_metrics())
        candidate = _write_json(tmp_path, "c.json", _make_accepted_metrics())
        result = cf_main(["--baseline", str(baseline), "--candidate", str(candidate)])
        # accept: improvement = 1.1 - 1.0 = 0.1 > default threshold 0.05
        assert result == 0

    def test_cli_json_format(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        """FEAT-09: --format json produces valid JSON stdout output."""
        baseline = _write_json(tmp_path, "b.json", _make_baseline_metrics())
        candidate = _write_json(tmp_path, "c.json", _make_accepted_metrics())
        cf_main(
            [
                "--baseline",
                str(baseline),
                "--candidate",
                str(candidate),
                "--format",
                "json",
            ]
        )
        captured = capsys.readouterr()
        parsed = json.loads(captured.out)
        assert "accepted" in parsed
        assert "sharpe_improvement" in parsed
        assert "reason" in parsed

    def test_cli_missing_key_exits_1(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """FEAT-09: JSON missing validation_sharpe key exits 1 with human-readable message."""
        bad = _write_json(tmp_path, "bad.json", {"train_sharpe": 1.0})
        good = _write_json(tmp_path, "good.json", _make_baseline_metrics())
        result = cf_main(["--baseline", str(bad), "--candidate", str(good)])
        assert result == 1
        captured = capsys.readouterr()
        # Should print an error message mentioning the missing key
        assert "validation_sharpe" in captured.out or "validation_sharpe" in captured.err


# ---------------------------------------------------------------------------
# TestEquityDimHelpers
# ---------------------------------------------------------------------------


class TestEquityDimHelpers:
    """FEAT-10: Config-aware dimension helpers."""

    def test_disabled_returns_156(self) -> None:
        """FEAT-10: equity_obs_dim(False, 8) == 156."""
        assert equity_obs_dim(False, 8) == 156

    def test_enabled_returns_172(self) -> None:
        """FEAT-10: equity_obs_dim(True, 8) == 172."""
        assert equity_obs_dim(True, 8) == 172

    def test_per_asset_disabled(self) -> None:
        """FEAT-10: equity_per_asset_dim(False) == 15."""
        assert equity_per_asset_dim(False) == 15

    def test_per_asset_enabled(self) -> None:
        """FEAT-10: equity_per_asset_dim(True) == 17."""
        assert equity_per_asset_dim(True) == 17

    def test_constants_exported(self) -> None:
        """FEAT-10: EQUITY_PER_ASSET_BASE and SENTIMENT_FEATURES_PER_ASSET are importable."""
        assert EQUITY_PER_ASSET_BASE == 15
        assert SENTIMENT_FEATURES_PER_ASSET == 2

    def test_formula_consistency(self) -> None:
        """FEAT-10: Enabled adds 2 features per asset relative to disabled."""
        diff = equity_obs_dim(True, 8) - equity_obs_dim(False, 8)
        assert diff == 8 * SENTIMENT_FEATURES_PER_ASSET  # 16 extra features


# ---------------------------------------------------------------------------
# TestAssembleSentiment
# ---------------------------------------------------------------------------


class TestAssembleSentiment:
    """FEAT-10: Sentiment-aware assembly tests."""

    _SYMBOLS_8 = ["SPY", "QQQ", "VTI", "XLV", "XLI", "XLE", "XLF", "XLK"]

    def _make_assembler(self, loaded_config: SwingRLConfig) -> ObservationAssembler:
        return ObservationAssembler(loaded_config)

    def _make_per_asset(self, symbols: list[str]) -> dict[str, np.ndarray]:
        """Create per-asset (15,) arrays of zeros."""
        return {s: np.zeros(15, dtype=float) for s in symbols}

    def _make_macro(self) -> np.ndarray:
        return np.zeros(6, dtype=float)

    def _make_hmm(self) -> np.ndarray:
        return np.array([0.6, 0.4])

    def test_shape_172(self, equity_env_config: SwingRLConfig) -> None:
        """FEAT-10: assemble_equity() with sentiment_features produces (172,) vector."""
        assembler = ObservationAssembler(equity_env_config)
        symbols = sorted(equity_env_config.equity.symbols)
        per_asset = self._make_per_asset(symbols)
        sentiment: dict[str, tuple[float, float]] = dict.fromkeys(symbols, (0.5, 0.8))
        obs = assembler.assemble_equity(
            per_asset,
            self._make_macro(),
            self._make_hmm(),
            0.0,
            sentiment_features=sentiment,
        )
        assert obs.shape == (172,)

    def test_shape_156_when_disabled(self, equity_env_config: SwingRLConfig) -> None:
        """FEAT-10: assemble_equity() without sentiment_features produces (156,) vector."""
        assembler = ObservationAssembler(equity_env_config)
        symbols = sorted(equity_env_config.equity.symbols)
        per_asset = self._make_per_asset(symbols)
        obs = assembler.assemble_equity(
            per_asset,
            self._make_macro(),
            self._make_hmm(),
            0.0,
        )
        assert obs.shape == (156,)

    def test_sentiment_values_in_correct_positions(self, equity_env_config: SwingRLConfig) -> None:
        """FEAT-10: Sentiment values appear at index 15 and 16 within first asset's slice."""
        assembler = ObservationAssembler(equity_env_config)
        symbols = sorted(equity_env_config.equity.symbols)
        per_asset = self._make_per_asset(symbols)
        # Set known per-asset values for first symbol
        first_sym = symbols[0]
        per_asset[first_sym] = np.arange(15, dtype=float)  # values 0..14

        sentiment_score = 0.75
        sentiment_confidence = 0.90
        sentiment: dict[str, tuple[float, float]] = dict.fromkeys(symbols, (0.0, 0.0))
        sentiment[first_sym] = (sentiment_score, sentiment_confidence)

        obs = assembler.assemble_equity(
            per_asset,
            self._make_macro(),
            self._make_hmm(),
            0.0,
            sentiment_features=sentiment,
        )
        # First asset occupies indices [0:17] (15 base + 2 sentiment)
        assert obs[15] == pytest.approx(sentiment_score)
        assert obs[16] == pytest.approx(sentiment_confidence)


# ---------------------------------------------------------------------------
# TestFeatureNames
# ---------------------------------------------------------------------------


class TestFeatureNames:
    """FEAT-10: Feature name list correctness."""

    def test_feature_names_172_when_enabled(self, equity_env_config: SwingRLConfig) -> None:
        """FEAT-10: get_feature_names_equity(sentiment_enabled=True) returns 172-element list."""
        assembler = ObservationAssembler(equity_env_config)
        names = assembler.get_feature_names_equity(sentiment_enabled=True)
        assert len(names) == 172
        # Check sentiment names exist for first alpha-sorted symbol
        first_sym = sorted(equity_env_config.equity.symbols)[0]
        assert f"{first_sym}_sentiment_score" in names
        assert f"{first_sym}_sentiment_confidence" in names

    def test_feature_names_156_default(self, equity_env_config: SwingRLConfig) -> None:
        """FEAT-10: get_feature_names_equity() returns 156-element list (backward compat)."""
        assembler = ObservationAssembler(equity_env_config)
        names = assembler.get_feature_names_equity()
        assert len(names) == 156


# ---------------------------------------------------------------------------
# TestEnvObsSpace
# ---------------------------------------------------------------------------


class TestEnvObsSpace:
    """FEAT-10: BaseTradingEnv observation space shape with sentiment toggle."""

    def _make_sentiment_config_yaml(self, tmp_path: Path, sentiment_enabled: bool) -> Path:
        """Write config YAML with sentiment toggle and 8 equity symbols."""
        import textwrap

        yaml_text = textwrap.dedent(f"""\
            trading_mode: paper
            equity:
              symbols: [SPY, QQQ, VTI, XLV, XLI, XLE, XLF, XLK]
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
            environment:
              initial_amount: 100000.0
              equity_episode_bars: 252
              crypto_episode_bars: 540
              equity_transaction_cost_pct: 0.0006
              crypto_transaction_cost_pct: 0.0022
              signal_deadzone: 0.02
              position_penalty_coeff: 10.0
              drawdown_penalty_coeff: 5.0
            system:
              duckdb_path: data/db/market_data.ddb
              sqlite_path: data/db/trading_ops.db
            alerting:
              alert_cooldown_minutes: 30
              consecutive_failures_before_alert: 3
            sentiment:
              enabled: {str(sentiment_enabled).lower()}
        """)
        cfg_file = tmp_path / "swingrl_sentiment.yaml"
        cfg_file.write_text(yaml_text)
        return cfg_file

    def test_equity_env_obs_172_when_sentiment_enabled(self, tmp_path: Path) -> None:
        """FEAT-10: BaseTradingEnv with sentiment=True sets observation_space.shape == (172,)."""
        from swingrl.envs.base import BaseTradingEnv

        cfg = load_config(self._make_sentiment_config_yaml(tmp_path, True))
        rng = np.random.default_rng(50)
        features = rng.standard_normal((300, 172)).astype(np.float32)
        prices = rng.uniform(50, 500, (300, 8)).astype(np.float32)
        env = BaseTradingEnv(features, prices, cfg, "equity")
        assert env.observation_space.shape == (172,)

    def test_equity_env_obs_156_default(self, tmp_path: Path) -> None:
        """FEAT-10: BaseTradingEnv with sentiment=False sets observation_space.shape == (156,)."""
        from swingrl.envs.base import BaseTradingEnv

        cfg = load_config(self._make_sentiment_config_yaml(tmp_path, False))
        rng = np.random.default_rng(51)
        features = rng.standard_normal((300, 156)).astype(np.float32)
        prices = rng.uniform(50, 500, (300, 8)).astype(np.float32)
        env = BaseTradingEnv(features, prices, cfg, "equity")
        assert env.observation_space.shape == (156,)
