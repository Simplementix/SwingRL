"""Extended tests for MemoryEpochCallback.

TRAIN-10: MemoryEpochCallback stores epoch snapshots at cadence and notable
events, collects metrics from SB3 logger, and implements two-pass adjustment
tracking for reward weight changes.
"""

from __future__ import annotations

from unittest.mock import MagicMock

from swingrl.memory.training.epoch_callback import (
    EPOCH_STORE_CADENCE,
    NOTABLE_KL_THRESHOLD,
    NOTABLE_MDD_THRESHOLD,
    MemoryEpochCallback,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_mock_memory_client() -> MagicMock:
    """Create a mock MemoryClient."""
    client = MagicMock()
    client._base_url = "http://localhost:8889"
    client.ingest_training.return_value = True
    return client


def _make_mock_wrapper(n_envs: int = 1) -> MagicMock:
    """Create a mock MemoryVecRewardWrapper."""
    mock = MagicMock()
    mock.num_envs = n_envs
    mock.observation_space = MagicMock()
    mock.action_space = MagicMock()
    mock.rolling_sharpe.return_value = 1.2
    mock.rolling_mdd.return_value = -0.05
    mock.rolling_win_rate.return_value = 0.55
    mock.weights = {"profit": 0.50, "sharpe": 0.25, "drawdown": 0.15, "turnover": 0.10}
    return mock


def _make_callback(
    run_id: str = "test_run_001",
    algo: str = "PPO",
    env: str = "equity",
) -> MemoryEpochCallback:
    """Create a callback with mock dependencies, logger pre-wired."""
    client = _make_mock_memory_client()
    wrapper = _make_mock_wrapper()
    cb = MemoryEpochCallback(
        memory_client=client,
        wrapper=wrapper,
        run_id=run_id,
        algo=algo,
        env=env,
        verbose=0,
    )
    # SB3 exposes logger as a property: self.model.logger
    # Wire a mock model so logger is accessible in unit tests
    mock_logger = MagicMock()
    mock_logger.name_to_value = {
        "rollout/ep_rew_mean": 1.5,
        "train/approx_kl": 0.01,
        "train/policy_gradient_loss": -0.002,
        "train/value_loss": 0.05,
        "train/entropy_loss": -0.003,
        "train/clip_fraction": 0.12,
    }
    cb.model = MagicMock()
    cb.model.logger = mock_logger
    # SB3 stores completed episode stats in ep_info_buffer (deque of dicts).
    # mean_reward is computed from this buffer, not the logger.
    cb.model.ep_info_buffer = [{"r": 1.0, "l": 100, "t": 0.0}, {"r": 2.0, "l": 100, "t": 0.0}]
    cb.num_timesteps = 1000
    return cb


# ---------------------------------------------------------------------------
# TestEpochCallbackShouldStore
# ---------------------------------------------------------------------------


class TestEpochCallbackShouldStore:
    """TRAIN-10: _should_store() returns correct (should_store, event_label) tuples."""

    def test_should_store_every_5th_epoch(self) -> None:
        """TRAIN-10: Cadence epoch (multiple of EPOCH_STORE_CADENCE) returns (True, None)."""
        cb = _make_callback()
        should, event = cb._should_store(EPOCH_STORE_CADENCE, 0.0, -0.01)
        assert should is True
        assert event is None

    def test_should_store_not_on_non_cadence_epoch(self) -> None:
        """TRAIN-10: Non-cadence epoch without notable event returns (False, None)."""
        cb = _make_callback()
        should, event = cb._should_store(3, 0.0, -0.01)
        assert should is False
        assert event is None

    def test_should_store_kl_spike(self) -> None:
        """TRAIN-10: approx_kl > NOTABLE_KL_THRESHOLD returns (True, 'kl_spike')."""
        cb = _make_callback()
        kl_above = NOTABLE_KL_THRESHOLD + 0.001
        should, event = cb._should_store(3, kl_above, -0.01)
        assert should is True
        assert event == "kl_spike"

    def test_should_store_mdd_breach(self) -> None:
        """TRAIN-10: rolling_mdd < NOTABLE_MDD_THRESHOLD returns (True, 'mdd_breach')."""
        cb = _make_callback()
        mdd_below = NOTABLE_MDD_THRESHOLD - 0.001
        should, event = cb._should_store(3, 0.0, mdd_below)
        assert should is True
        assert event == "mdd_breach"

    def test_should_store_kl_boundary(self) -> None:
        """TRAIN-10: approx_kl == NOTABLE_KL_THRESHOLD (not >) returns False."""
        cb = _make_callback()
        should, event = cb._should_store(3, NOTABLE_KL_THRESHOLD, -0.01)
        assert should is False

    def test_should_store_mdd_boundary(self) -> None:
        """TRAIN-10: rolling_mdd == NOTABLE_MDD_THRESHOLD (not <) returns False."""
        cb = _make_callback()
        should, event = cb._should_store(3, 0.0, NOTABLE_MDD_THRESHOLD)
        assert should is False


# ---------------------------------------------------------------------------
# TestEpochCallbackCollectMetrics
# ---------------------------------------------------------------------------


class TestEpochCallbackCollectMetrics:
    """TRAIN-10: _collect_metrics() assembles the epoch metrics dict."""

    def test_collect_metrics_pulls_from_model_and_logger(self) -> None:
        """TRAIN-10: mean_reward from ep_info_buffer; policy_loss, approx_kl from logger."""
        cb = _make_callback()
        cb._epoch = 5
        metrics = cb._collect_metrics(None)
        # mean_reward comes from ep_info_buffer: mean([1.0, 2.0]) = 1.5
        assert abs(metrics["mean_reward"] - 1.5) < 1e-6
        assert abs(metrics["approx_kl"] - 0.01) < 1e-6
        assert abs(metrics["policy_loss"] - (-0.002)) < 1e-6

    def test_collect_metrics_missing_keys_default_to_zero(self) -> None:
        """TRAIN-10: Absent logger keys default to 0.0; empty buffer → mean_reward 0.0."""
        cb = _make_callback()
        cb.model.logger.name_to_value = {}  # empty — no keys present
        cb.model.ep_info_buffer = []  # empty buffer
        cb._epoch = 1
        metrics = cb._collect_metrics(None)
        assert metrics["mean_reward"] == 0.0
        assert metrics["approx_kl"] == 0.0
        assert metrics["policy_loss"] == 0.0

    def test_collect_metrics_mean_reward_from_ep_info_buffer(self) -> None:
        """TRAIN-10: mean_reward computed from model.ep_info_buffer, not logger."""
        cb = _make_callback()
        # Override buffer with known values
        cb.model.ep_info_buffer = [
            {"r": 10.0, "l": 50, "t": 0.0},
            {"r": 20.0, "l": 50, "t": 0.0},
            {"r": 30.0, "l": 50, "t": 0.0},
        ]
        cb._epoch = 5
        metrics = cb._collect_metrics(None)
        # mean([10, 20, 30]) = 20.0
        assert abs(metrics["mean_reward"] - 20.0) < 1e-6

    def test_collect_metrics_mean_reward_no_buffer_attribute(self) -> None:
        """TRAIN-10: mean_reward falls back to 0.0 if ep_info_buffer missing."""
        cb = _make_callback()
        del cb.model.ep_info_buffer
        cb._epoch = 5
        metrics = cb._collect_metrics(None)
        assert metrics["mean_reward"] == 0.0

    def test_collect_metrics_calls_wrapper_rolling_methods(self) -> None:
        """TRAIN-10: rolling_sharpe(), rolling_mdd(), rolling_win_rate() are called."""
        cb = _make_callback()
        cb._epoch = 5
        cb._collect_metrics(None)
        cb._wrapper.rolling_sharpe.assert_called()
        cb._wrapper.rolling_mdd.assert_called()
        cb._wrapper.rolling_win_rate.assert_called()

    def test_collect_metrics_includes_reward_weights(self) -> None:
        """TRAIN-10: metrics dict contains reward_weights key from wrapper."""
        cb = _make_callback()
        cb._epoch = 5
        metrics = cb._collect_metrics("kl_spike")
        assert "reward_weights" in metrics
        assert metrics["reward_weights"]["profit"] == 0.50


# ---------------------------------------------------------------------------
# TestEpochCallbackIngestSnapshot
# ---------------------------------------------------------------------------


class TestEpochCallbackIngestSnapshot:
    """TRAIN-10: _ingest_epoch_snapshot() delegates to memory client."""

    def test_ingest_epoch_snapshot_calls_ingest_training(self) -> None:
        """TRAIN-10: client.ingest_training() is called with formatted text."""
        cb = _make_callback()
        cb._epoch = 5
        metrics = cb._collect_metrics(None)
        cb._ingest_epoch_snapshot(metrics)
        cb._client.ingest_training.assert_called_once()

    def test_ingest_epoch_snapshot_text_contains_run_id(self) -> None:
        """TRAIN-10: run_id appears in the ingested text."""
        cb = _make_callback(run_id="run_xyz_999")
        cb._epoch = 5
        metrics = cb._collect_metrics(None)
        cb._ingest_epoch_snapshot(metrics)
        call_args = cb._client.ingest_training.call_args[0][0]
        assert "run_xyz_999" in call_args


# ---------------------------------------------------------------------------
# TestEpochCallbackAdjustmentTrigger
# ---------------------------------------------------------------------------


class TestEpochCallbackAdjustmentTrigger:
    """TRAIN-10: _ingest_adjustment_trigger() sets up two-pass tracking state."""

    def test_ingest_adjustment_trigger_sets_pending_adjustment(self) -> None:
        """TRAIN-10: _pending_adjustment dict is populated with trigger details."""
        cb = _make_callback()
        cb._epoch = 10
        cb._ingest_adjustment_trigger(
            new_weights={"profit": 0.6, "sharpe": 0.2, "drawdown": 0.1, "turnover": 0.1},
            old_weights={"profit": 0.5, "sharpe": 0.25, "drawdown": 0.15, "turnover": 0.1},
            trigger_metric="rolling_mdd_500",
            trigger_value=-0.09,
            trigger_reason="mdd too high",
        )
        assert cb._pending_adjustment is not None
        assert cb._pending_adjustment["epoch_triggered"] == 10
        assert cb._pending_adjustment["trigger_metric"] == "rolling_mdd_500"

    def test_ingest_adjustment_trigger_records_sharpe_at_trigger(self) -> None:
        """TRAIN-10: _sharpe_at_trigger is set from wrapper.rolling_sharpe()."""
        cb = _make_callback()
        cb._wrapper.rolling_sharpe.return_value = 0.75
        cb._epoch = 7
        cb._ingest_adjustment_trigger(
            new_weights={"profit": 0.6, "sharpe": 0.2, "drawdown": 0.1, "turnover": 0.1},
            old_weights={"profit": 0.5, "sharpe": 0.25, "drawdown": 0.15, "turnover": 0.1},
            trigger_metric="rolling_mdd_500",
            trigger_value=-0.09,
            trigger_reason="test",
        )
        assert abs(cb._sharpe_at_trigger - 0.75) < 1e-6

    def test_ingest_adjustment_trigger_calls_ingest_training(self) -> None:
        """TRAIN-10: client.ingest_training() is called with trigger text."""
        cb = _make_callback()
        cb._epoch = 5
        cb._ingest_adjustment_trigger(
            new_weights={"profit": 0.6, "sharpe": 0.2, "drawdown": 0.1, "turnover": 0.1},
            old_weights={"profit": 0.5, "sharpe": 0.25, "drawdown": 0.15, "turnover": 0.1},
            trigger_metric="epoch_advice",
            trigger_value=-0.05,
            trigger_reason="LLM advised",
        )
        cb._client.ingest_training.assert_called_once()
        text = cb._client.ingest_training.call_args[0][0]
        assert "REWARD_ADJUSTMENT_TRIGGER" in text


# ---------------------------------------------------------------------------
# TestEpochCallbackResolvePendingAdjustment
# ---------------------------------------------------------------------------


class TestEpochCallbackResolvePendingAdjustment:
    """TRAIN-10: _resolve_pending_adjustment() computes outcome deltas."""

    def _setup_with_pending(
        self,
        sharpe_at_trigger: float = 1.0,
        mdd_at_trigger: float = -0.05,
        current_sharpe: float = 1.3,
        current_mdd: float = -0.03,
    ) -> MemoryEpochCallback:
        """Create callback with a pre-populated pending adjustment."""
        cb = _make_callback()
        cb._epoch = 20
        cb._pending_adjustment = {
            "epoch_triggered": 10,
            "trigger_metric": "rolling_mdd_500",
            "trigger_value": -0.09,
            "trigger_reason": "test",
            "weights_before": {"profit": 0.5, "sharpe": 0.25, "drawdown": 0.15, "turnover": 0.1},
            "weights_after": {"profit": 0.6, "sharpe": 0.2, "drawdown": 0.1, "turnover": 0.1},
            "curriculum_window_at_trigger": "2022_bear",
            "regime_at_trigger": "bear",
        }
        cb._sharpe_at_trigger = sharpe_at_trigger
        cb._mdd_at_trigger = mdd_at_trigger
        cb._wrapper.rolling_sharpe.return_value = current_sharpe
        cb._wrapper.rolling_mdd.return_value = current_mdd
        return cb

    def test_resolve_pending_computes_sharpe_delta(self) -> None:
        """TRAIN-10: sharpe_delta = current_sharpe - sharpe_at_trigger."""
        cb = self._setup_with_pending(sharpe_at_trigger=1.0, current_sharpe=1.3)
        cb._resolve_pending_adjustment()
        text = cb._client.ingest_training.call_args[0][0]
        assert "post_adjustment_sharpe_delta=0.3000" in text

    def test_resolve_pending_computes_mdd_delta(self) -> None:
        """TRAIN-10: mdd_delta = current_mdd - mdd_at_trigger."""
        cb = self._setup_with_pending(mdd_at_trigger=-0.05, current_mdd=-0.03)
        cb._resolve_pending_adjustment()
        text = cb._client.ingest_training.call_args[0][0]
        assert "post_adjustment_mdd_delta=0.0200" in text

    def test_resolve_pending_effective_when_sharpe_improves(self) -> None:
        """TRAIN-10: sharpe_delta > 0 → adjustment_effective=True."""
        cb = self._setup_with_pending(sharpe_at_trigger=1.0, current_sharpe=1.2, current_mdd=-0.06)
        cb._resolve_pending_adjustment()
        text = cb._client.ingest_training.call_args[0][0]
        assert "adjustment_effective=True" in text

    def test_resolve_pending_effective_when_mdd_improves(self) -> None:
        """TRAIN-10: mdd_delta > 0 (less drawdown) → adjustment_effective=True."""
        cb = self._setup_with_pending(
            sharpe_at_trigger=1.0,
            current_sharpe=0.9,  # sharpe got worse
            mdd_at_trigger=-0.10,
            current_mdd=-0.05,  # mdd improved
        )
        cb._resolve_pending_adjustment()
        text = cb._client.ingest_training.call_args[0][0]
        assert "adjustment_effective=True" in text

    def test_resolve_pending_clears_pending_adjustment(self) -> None:
        """TRAIN-10: _pending_adjustment is set to None after resolve."""
        cb = self._setup_with_pending()
        assert cb._pending_adjustment is not None
        cb._resolve_pending_adjustment()
        assert cb._pending_adjustment is None

    def test_resolve_pending_calls_ingest_training(self) -> None:
        """TRAIN-10: outcome text is ingested via client.ingest_training()."""
        cb = self._setup_with_pending()
        cb._resolve_pending_adjustment()
        cb._client.ingest_training.assert_called_once()
        text = cb._client.ingest_training.call_args[0][0]
        assert "REWARD_ADJUSTMENT_OUTCOME" in text
