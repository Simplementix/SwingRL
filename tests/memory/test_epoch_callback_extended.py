"""Extended tests for MemoryEpochCallback.

TRAIN-10: MemoryEpochCallback stores epoch snapshots at cadence and notable
events, collects metrics from SB3 logger, and implements two-pass adjustment
tracking for reward weight changes.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from swingrl.memory.training.epoch_callback import MemoryEpochCallback
from swingrl.utils.exceptions import ConfigError

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
    """Create a mock MemoryVecRewardWrapper.

    ``window_metrics("short" | "trend")`` returns a copy of ``mock._window_data[window]``
    -- tests that need to drive §4.10 triggers mutate that dict directly (e.g.
    ``wrapper._window_data["short"]["mdd_frac_worst"] = 0.11``) rather than
    reconfiguring the MagicMock's side_effect each time.
    """
    mock = MagicMock()
    mock.num_envs = n_envs
    mock.observation_space = MagicMock()
    mock.action_space = MagicMock()
    mock.rolling_mean_reward.return_value = 1.5
    mock.rolling_sharpe.return_value = 1.2
    mock.rolling_mdd.return_value = -0.05
    mock.rolling_win_rate.return_value = 0.55
    mock.rolling_trade_rate.return_value = 0.12
    mock.baseline_trade_rate.return_value = 0.10
    mock.weights = {"profit": 0.50, "sharpe": 0.25, "drawdown": 0.15, "turnover": 0.10}

    window_data = {
        "short": {
            "pct_of_fold": 0.01,
            "steps": 1_000,
            "sharpe_annualized": 1.2,
            "mdd_frac_worst": 0.02,
            "mdd_frac_mean": 0.01,
            "win_rate": 0.55,
            "trade_rate": 0.10,
        },
        "trend": {
            "pct_of_fold": 0.15,
            "steps": 15_000,
            "sharpe_annualized": 1.2,
            "mdd_frac_worst": 0.02,
            "mdd_frac_mean": 0.01,
            "win_rate": 0.55,
            "trade_rate": 0.10,
        },
    }
    mock._window_data = window_data
    mock.window_metrics.side_effect = lambda window: dict(window_data[window])
    return mock


def _make_callback(
    run_id: str = "test_run_001",
    algo: str = "PPO",
    env: str = "equity",
    fold_number: int | None = None,
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
        fold_number=fold_number,
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
    cb.num_timesteps = 1000
    return cb


# ---------------------------------------------------------------------------
# TestEpochCallbackShouldStore
# ---------------------------------------------------------------------------


class TestEpochCallbackShouldStore:
    """§4.10 (D-T3.19): _should_store() five-trigger set + rate cap + hard cap.

    Replaces the retired NOTABLE_KL_THRESHOLD/NOTABLE_MDD_THRESHOLD pure-thresholding
    design. The default mock wrapper (``_make_mock_wrapper``) starts "healthy": short
    window mdd_frac_worst=0.02 (< equity ceiling 0.10), trade_rate=0.10 == baseline
    (neither trade_shy nor churning), and callbacks default to approx_kl=0.0 /
    mean_reward=1.5 (finite) unless a test overrides them.
    """

    def test_should_store_cadence_epoch_returns_true(self) -> None:
        """Cadence epoch (multiple of callback cadence) returns (True, None) unconditionally."""
        cb = _make_callback()
        should, event = cb._should_store(cb._cadence, 0.0, 1.5)
        assert should is True
        assert event is None

    def test_should_store_healthy_epoch_returns_false(self) -> None:
        """Non-cadence epoch with all triggers healthy returns (False, None)."""
        cb = _make_callback()
        should, event = cb._should_store(3, 0.0, 1.5)
        assert should is False
        assert event is None

    def test_should_store_kl_spike(self) -> None:
        """approx_kl > kl_spike_threshold (config default 0.10) returns (True, 'kl_spike')."""
        cb = _make_callback()
        should, event = cb._should_store(3, 0.101, 1.5)
        assert should is True
        assert event == "kl_spike"

    def test_should_store_kl_boundary(self) -> None:
        """approx_kl == kl_spike_threshold (not >) returns (False, None)."""
        cb = _make_callback()
        should, event = cb._should_store(3, 0.10, 1.5)
        assert should is False
        assert event is None

    def test_should_store_mdd_breach_uses_worst_not_mean(self) -> None:
        """mdd_breach fires on mdd_frac_worst 0.11 (equity ceiling 0.10) even when
        mdd_frac_mean is 0.03 -- the worst basis is load-bearing (spec 2026-07-12
        locked decision), not the old cumsum scale, and never the mean."""
        cb = _make_callback(env="equity")
        cb._wrapper._window_data["short"]["mdd_frac_worst"] = 0.11
        cb._wrapper._window_data["short"]["mdd_frac_mean"] = 0.03
        should, event = cb._should_store(3, 0.0, 1.5)
        assert should is True
        assert event == "mdd_breach"

    def test_should_store_mdd_boundary(self) -> None:
        """mdd_frac_worst == mdd_breach_frac ceiling (not >) returns (False, None)."""
        cb = _make_callback(env="equity")
        cb._wrapper._window_data["short"]["mdd_frac_worst"] = 0.10
        should, event = cb._should_store(3, 0.0, 1.5)
        assert should is False
        assert event is None

    def test_should_store_trade_shy(self) -> None:
        """trade_rate < 0.5x locked baseline_trade_rate fires trade_shy."""
        cb = _make_callback()
        cb._wrapper.baseline_trade_rate.return_value = 0.10
        cb._wrapper._window_data["short"]["trade_rate"] = 0.04
        should, event = cb._should_store(3, 0.0, 1.5)
        assert should is True
        assert event == "trade_shy"

    def test_should_store_churning(self) -> None:
        """trade_rate > 3x locked baseline_trade_rate fires churning."""
        cb = _make_callback()
        cb._wrapper.baseline_trade_rate.return_value = 0.10
        cb._wrapper._window_data["short"]["trade_rate"] = 0.35
        should, event = cb._should_store(3, 0.0, 1.5)
        assert should is True
        assert event == "churning"

    def test_should_store_trade_shy_and_churning_disabled_before_baseline_locks(self) -> None:
        """baseline_trade_rate() == 0.0 (window not yet locked) disables both triggers,
        matching cps_diagnosis.diagnose_rolling's own guard -- not a false alarm."""
        cb = _make_callback()
        cb._wrapper.baseline_trade_rate.return_value = 0.0
        cb._wrapper._window_data["short"]["trade_rate"] = 0.0
        should, event = cb._should_store(3, 0.0, 1.5)
        assert should is False
        assert event is None

    def test_should_store_numeric_anomaly_on_nan_reward(self) -> None:
        """numeric_anomaly fires on NaN mean_reward."""
        cb = _make_callback()
        should, event = cb._should_store(3, 0.0, float("nan"))
        assert should is True
        assert event == "numeric_anomaly"

    def test_should_store_numeric_anomaly_on_inf_approx_kl(self) -> None:
        """numeric_anomaly fires on +inf approx_kl."""
        cb = _make_callback()
        should, event = cb._should_store(3, float("inf"), 1.5)
        assert should is True
        assert event == "numeric_anomaly"

    def test_should_store_numeric_anomaly_precedes_kl_spike(self) -> None:
        """When both numeric_anomaly and kl_spike would fire, numeric_anomaly wins
        (checked first -- a NaN/inf reading makes every other metric this epoch
        suspect, so it is reported ahead of a comparison against a possibly
        corrupted approx_kl)."""
        cb = _make_callback()
        should, event = cb._should_store(3, 0.5, float("nan"))
        assert should is True
        assert event == "numeric_anomaly"

    def test_should_store_rate_cap_suppresses_second_same_trigger_in_window(self) -> None:
        """Rate cap: the same trigger type firing twice inside one trend window ->
        the second occurrence is suppressed (should_store=False)."""
        cb = _make_callback()
        should1, event1 = cb._should_store(3, 0.20, 1.5)
        should2, event2 = cb._should_store(5, 0.20, 1.5)
        assert (should1, event1) == (True, "kl_spike")
        assert (should2, event2) == (False, None)

    def test_should_store_rate_cap_resets_on_new_trend_window(self) -> None:
        """Rate cap resets once num_timesteps crosses into a new trend window."""
        cb = _make_callback()
        trend_steps = cb._wrapper._window_data["trend"]["steps"]
        cb.num_timesteps = 0
        should1, event1 = cb._should_store(3, 0.20, 1.5)
        cb.num_timesteps = trend_steps  # crosses into trend-window index 1
        should2, event2 = cb._should_store(5, 0.20, 1.5)
        assert (should1, event1) == (True, "kl_spike")
        assert (should2, event2) == (True, "kl_spike")

    def test_should_store_hard_cap_drops_row_51_and_fires_alarm_once(self) -> None:
        """Hard cap (default 50/run): 51 distinct events (each in its own trend
        window, so the rate cap never suppresses them) -> row 51 is dropped, a
        capture_alarm ingests + logs exactly once, and the run's next cadence
        epoch is STILL stored (fail-safe direction: lose telemetry, never the
        heartbeat)."""
        cb = _make_callback()
        trend_steps = cb._wrapper._window_data["trend"]["steps"]

        for i in range(50):
            cb.num_timesteps = i * trend_steps
            should, event = cb._should_store(3, 0.20, 1.5)
            assert (should, event) == (True, "kl_spike"), f"event {i} should be accepted"
        assert cb._event_rows_this_run == 50
        assert cb._hard_cap_alarm_fired is False
        cb._client.ingest_training.assert_not_called()

        # 51st distinct-window event: hard cap reached -> dropped + alarm fires once.
        cb.num_timesteps = 50 * trend_steps
        should_51, event_51 = cb._should_store(3, 0.20, 1.5)
        assert (should_51, event_51) == (False, None)
        assert cb._hard_cap_alarm_fired is True
        assert cb._event_rows_this_run == 50  # dropped row does not increment
        cb._client.ingest_training.assert_called_once()
        call = cb._client.ingest_training.call_args
        assert call.kwargs["source"] == "capture_alarm:equity:ppo"

        # A second breach past the cap must NOT fire a second alarm.
        cb.num_timesteps = 51 * trend_steps
        should_52, event_52 = cb._should_store(3, 0.20, 1.5)
        assert (should_52, event_52) == (False, None)
        cb._client.ingest_training.assert_called_once()  # still just the one call

        # Cadence path is uncapped: the next cadence epoch still stores.
        should_cadence, event_cadence = cb._should_store(cb._cadence, 0.20, 1.5)
        assert (should_cadence, event_cadence) == (True, None)


# ---------------------------------------------------------------------------
# TestEpochCallbackCollectMetrics
# ---------------------------------------------------------------------------


class TestEpochCallbackCollectMetrics:
    """TRAIN-10: _collect_metrics() assembles the epoch metrics dict."""

    def test_collect_metrics_pulls_from_wrapper_and_logger(self) -> None:
        """TRAIN-10: mean_reward from wrapper; policy_loss, approx_kl from logger."""
        cb = _make_callback()
        cb._epoch = 5
        metrics = cb._collect_metrics(None)
        # mean_reward comes from wrapper.rolling_mean_reward() = 1.5
        assert abs(metrics["mean_reward"] - 1.5) < 1e-6
        assert abs(metrics["approx_kl"] - 0.01) < 1e-6
        assert abs(metrics["policy_loss"] - (-0.002)) < 1e-6

    def test_collect_metrics_missing_keys_default_to_zero(self) -> None:
        """TRAIN-10: Absent logger keys default to 0.0."""
        cb = _make_callback()
        cb.model.logger.name_to_value = {}  # empty — no keys present
        cb._wrapper.rolling_mean_reward.return_value = 0.0
        cb._epoch = 1
        metrics = cb._collect_metrics(None)
        assert metrics["mean_reward"] == 0.0
        assert metrics["approx_kl"] == 0.0
        assert metrics["policy_loss"] == 0.0

    def test_collect_metrics_mean_reward_from_wrapper(self) -> None:
        """TRAIN-10: mean_reward from wrapper.rolling_mean_reward(), not SB3 logger."""
        cb = _make_callback()
        cb._wrapper.rolling_mean_reward.return_value = 42.5
        cb._epoch = 5
        metrics = cb._collect_metrics(None)
        assert abs(metrics["mean_reward"] - 42.5) < 1e-6
        cb._wrapper.rolling_mean_reward.assert_called()

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


# ---------------------------------------------------------------------------
# TestOutcomeSharpeRegression
# ---------------------------------------------------------------------------


class TestOutcomeSharpeRegression:
    """TRAIN-10: outcome_sharpe must store current_sharpe, not sharpe_delta."""

    def _setup_with_pending_and_db(
        self,
        sharpe_at_trigger: float = 1.0,
        current_sharpe: float = 1.5,
        current_mdd: float = -0.03,
    ) -> MemoryEpochCallback:
        """Callback pre-loaded with a pending adjustment and a fake DATABASE_URL
        so _adjustment_outcome_queue is populated on _resolve_pending_adjustment().
        """
        cb = _make_callback()
        cb._epoch = 20
        cb._pending_adjustment = {
            "epoch_triggered": 10,
            "trigger_metric": "rolling_mdd_500",
            "trigger_value": -0.09,
            "trigger_reason": "test",
            "weights_before": {
                "profit": 0.5,
                "sharpe": 0.25,
                "drawdown": 0.15,
                "turnover": 0.1,
            },
            "weights_after": {
                "profit": 0.6,
                "sharpe": 0.2,
                "drawdown": 0.1,
                "turnover": 0.1,
            },
            "curriculum_window_at_trigger": "2022_bear",
            "regime_at_trigger": "bear",
        }
        cb._sharpe_at_trigger = sharpe_at_trigger
        cb._mdd_at_trigger = -0.05
        cb._wrapper.rolling_sharpe.return_value = current_sharpe
        cb._wrapper.rolling_mdd.return_value = current_mdd
        # Provide a fake DATABASE_URL so the queue-append branch executes.
        cb._database_url = "postgresql://fake:fake@localhost/fake"  # pragma: allowlist secret
        return cb

    def test_outcome_sharpe_stores_current_sharpe_not_delta(self) -> None:
        """TRAIN-10: outcome_queue tuple position 1 (outcome_sharpe) == current_sharpe.

        The UPDATE SQL is:
          SET epoch_outcome=%s, outcome_sharpe=%s, sharpe_delta=%s, mdd_delta=%s, effective=%s
        so params[0]=epoch, params[1]=outcome_sharpe, params[2]=sharpe_delta.

        Before the fix, params[1] was sharpe_delta (the difference), not
        current_sharpe (the absolute value).  With sharpe_at_trigger=1.0 and
        current_sharpe=1.5, the delta is 0.5 — clearly distinct from 1.5.
        """
        cb = self._setup_with_pending_and_db(
            sharpe_at_trigger=1.0,
            current_sharpe=1.5,
        )
        cb._resolve_pending_adjustment()

        assert len(cb._adjustment_outcome_queue) == 1, (
            "Expected exactly one item in the outcome queue"
        )
        params, _run_id, _epoch_trigger = cb._adjustment_outcome_queue[0]
        # params layout: [epoch_outcome, outcome_sharpe, sharpe_delta, mdd_delta, effective]
        outcome_sharpe_value = params[1]
        sharpe_delta_value = params[2]

        # The bug: outcome_sharpe_value == sharpe_delta_value (both 0.5)
        # The fix: outcome_sharpe_value == current_sharpe (1.5)
        assert abs(outcome_sharpe_value - 1.5) < 1e-9, (
            f"outcome_sharpe should be current_sharpe=1.5, got {outcome_sharpe_value}. "
            f"sharpe_delta was {sharpe_delta_value}. "
            "Likely the outcome tuple still passes sharpe_delta instead of current_sharpe."
        )


# ---------------------------------------------------------------------------
# TestEnrichedEpochPayload
# ---------------------------------------------------------------------------


class TestEnrichedEpochPayload:
    """C1-PAYLOAD-01: epoch advice payload carries context JSON."""

    def _make_advice_callback(
        self,
        fold_number: int | None = 3,
        algo: str = "PPO",
        env: str = "equity",
        advice_enabled: bool = True,
        iteration: int | None = 6,
    ) -> MemoryEpochCallback:
        """Create an advice-enabled callback at a storage epoch, fold context pre-injected."""
        client = _make_mock_memory_client()
        # Return a non-empty body so the accepted-advice path runs
        client.epoch_advice.return_value = {
            "reward_weights": {
                "profit": 0.55,
                "sharpe": 0.25,
                "drawdown": 0.15,
                "turnover": 0.05,
            },
            "stop_training": False,
            "rationale": "test rationale",
            "provider": "test",
            "model": "test",
        }
        wrapper = _make_mock_wrapper()
        cb = MemoryEpochCallback(
            memory_client=client,
            wrapper=wrapper,
            run_id=f"equity_{algo}_fold{fold_number or 0}",
            algo=algo,
            env=env,
            verbose=0,
            advice_enabled=advice_enabled,
            iteration=iteration,
            fold_number=fold_number,
        )
        # Wire mock model
        mock_logger = MagicMock()
        mock_logger.name_to_value = {}
        cb.model = MagicMock()
        cb.model.logger = mock_logger
        cb.num_timesteps = 1000
        # Set epoch to a storage epoch so the cadence check passes
        cb._epoch = cb._cadence  # noqa: SLF001
        # Inject fold context to bypass the lazy DB load
        cb._fold_context = {  # noqa: SLF001
            "fold_role": "neutral",
            "chronic_failure_folds": [],
            "protected_winner_folds": [],
            "prev_iter_cps_v1": 0.034,
        }
        return cb

    def test_payload_query_contains_context_block(self) -> None:
        """C1-PAYLOAD-01: query embeds context={...} with required keys."""
        import json

        cb = self._make_advice_callback(fold_number=3, iteration=6)
        cb._query_epoch_advice()  # noqa: SLF001

        payload = cb._client.epoch_advice.call_args[0][0]
        assert "context=" in payload["query"], "payload query must contain context= key"

        ctx = json.loads(payload["query"].split("context=", 1)[1])
        assert ctx["fold_number"] == 3
        assert ctx["fold_role"] == "neutral"
        assert ctx["target_metric"] == "cps_v1_multiplicative"
        assert ctx["prev_iter_cps_v1"] == 0.034

        li = ctx["leading_indicators"]
        assert set(li) == {
            "rolling_sharpe",
            "rolling_mdd",
            "rolling_win_rate",
            "trade_rate",
            "baseline_trade_rate",
        }
        # rolling_sharpe and rolling_mdd must be in leading_indicators (not bare in query)
        assert "rolling_sharpe=" not in payload["query"].split("context=")[0], (
            "rolling_sharpe must be inside context JSON, not in bare query string"
        )

        valid_labels = {"trade_shy", "poor_selection", "single_disaster", "churning", "healthy"}
        assert ctx["diagnosis"]["label"] in valid_labels

    def test_lazy_context_defaults_without_db(self) -> None:
        """C1-PAYLOAD-01: no database_url → neutral context, no crash."""
        import json

        cb = self._make_advice_callback(fold_number=3)
        # Remove the pre-injected fold_context so the lazy-load path runs
        cb._fold_context = None  # noqa: SLF001
        # No database_url set → neutral defaults
        cb._database_url = None  # noqa: SLF001

        cb._query_epoch_advice()  # noqa: SLF001

        payload = cb._client.epoch_advice.call_args[0][0]
        ctx = json.loads(payload["query"].split("context=", 1)[1])
        assert ctx["fold_role"] == "neutral"
        assert ctx["prev_iter_cps_v1"] is None

    def test_diagnosis_dataerror_falls_back_healthy(self) -> None:
        """C1-PAYLOAD-01: unknown algo → DataError inside diagnose_rolling → healthy fallback."""
        import json

        # Use an unknown algo so _baseline() raises DataError
        cb = self._make_advice_callback(fold_number=3, algo="XGB", env="equity")
        # Need a run_id that keeps advice_enabled logic working
        cb._algo = "xgb"  # noqa: SLF001  # ensure lowercase for diagnose_rolling lookup

        cb._query_epoch_advice()  # noqa: SLF001

        # epoch_advice must still have been called (advice proceeds despite error)
        assert cb._client.epoch_advice.called
        payload = cb._client.epoch_advice.call_args[0][0]
        ctx = json.loads(payload["query"].split("context=", 1)[1])
        assert ctx["diagnosis"]["label"] == "healthy"

    def test_control_fold_sends_no_advice_query(self) -> None:
        """C1-PAYLOAD-01: advice_enabled=False short-circuits (existing behavior kept)."""
        cb = self._make_advice_callback(fold_number=3, advice_enabled=False)
        cb._advice_enabled = False  # noqa: SLF001
        cb._query_epoch_advice()  # noqa: SLF001
        cb._client.epoch_advice.assert_not_called()


# ---------------------------------------------------------------------------
# TestAttributionIdentity
# ---------------------------------------------------------------------------


class TestAttributionIdentity:
    """C5-ATTR-01: trigger row carries fold/iteration/advice_id/cps_before.

    D-T2.1: the L1 lever is benched by default (max_reward_delta=0.0 for every
    algo/env pair), which would short-circuit acceptance before the attribution
    logic under test ever runs. These tests simulate a re-earned/harness-passed
    lever by monkeypatching get_max_reward_delta() back to a nonzero value for
    the PPO/equity pair used here — the attribution mechanics are independent
    of the bench posture and must keep working once a lever is re-earned.
    """

    def _make_accepted_advice_callback(
        self,
        monkeypatch: pytest.MonkeyPatch,
        fold_number: int = 3,
        iteration: int = 6,
        prev_iter_cps_v1: float | None = 0.034,
    ) -> MemoryEpochCallback:
        """Create a callback where advice is accepted (big enough delta) with DB url."""
        # Minor (Task 4 review): use a MagicMock (not a bare lambda) so callers can
        # assert it was invoked with the expected (algo, env) pair — the callback's
        # run_id ("equity_ppo_fold{N}") parses to algo="ppo", env="equity".
        mock_get_max_reward_delta = MagicMock(return_value=0.03)
        monkeypatch.setattr(
            "swingrl.memory.training.bounds.get_max_reward_delta",
            mock_get_max_reward_delta,
        )
        client = _make_mock_memory_client()
        # Return weights with a delta large enough to exceed the 0.01 min
        client.epoch_advice.return_value = {
            "reward_weights": {
                "profit": 0.62,  # was 0.50 → delta 0.12 > 0.01
                "sharpe": 0.20,
                "drawdown": 0.12,
                "turnover": 0.06,
            },
            "stop_training": False,
            "rationale": "test",
            "provider": "test",
            "model": "test",
        }
        wrapper = _make_mock_wrapper()
        cb = MemoryEpochCallback(
            memory_client=client,
            wrapper=wrapper,
            run_id=f"equity_ppo_fold{fold_number}",
            algo="PPO",
            env="equity",
            verbose=0,
            advice_enabled=True,
            iteration=iteration,
            fold_number=fold_number,
            database_url="postgresql://fake:fake@localhost/fake",  # pragma: allowlist secret
        )
        mock_logger = MagicMock()
        mock_logger.name_to_value = {}
        cb.model = MagicMock()
        cb.model.logger = mock_logger
        cb.num_timesteps = 0
        cb._epoch = cb._cadence  # noqa: SLF001  # storage epoch
        # Pre-inject fold context to bypass DB load
        cb._fold_context = {  # noqa: SLF001
            "fold_role": "neutral",
            "chronic_failure_folds": [],
            "protected_winner_folds": [],
            "prev_iter_cps_v1": prev_iter_cps_v1,
        }
        cb._test_max_delta_mock = mock_get_max_reward_delta  # noqa: SLF001
        return cb

    def test_adjustment_trigger_row_has_attribution_tail(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """C5-ATTR-01: queue row tail = (fold_number, iteration, advice_id, cps_before)."""
        cb = self._make_accepted_advice_callback(
            monkeypatch, fold_number=3, iteration=6, prev_iter_cps_v1=0.034
        )
        cb._query_epoch_advice()  # noqa: SLF001

        # Minor (Task 4 review): confirm the lever check was actually exercised for
        # this callback's (algo, env) pair, not just returning a stubbed constant.
        cb._test_max_delta_mock.assert_called_once_with("ppo", "equity")  # noqa: SLF001

        assert len(cb._adjustment_trigger_queue) == 1, (
            "Expected one trigger row in queue after accepted advice"
        )
        row = cb._adjustment_trigger_queue[0]

        # Existing 11 columns: run_id, epoch, algo, env, trigger_metric,
        #   trigger_value, trigger_reason, weight_before, weight_after,
        #   sharpe_at_trigger, mdd_at_trigger
        # New 4: fold_number, iteration_number, advice_id, fold_cps_v1_before
        assert len(row) == 15, f"Expected 15 columns in trigger row, got {len(row)}"

        fold_col = row[-4]
        iteration_col = row[-3]
        advice_id_col = row[-2]
        cps_before_col = row[-1]

        assert fold_col == 3, f"fold_number should be 3, got {fold_col}"
        assert iteration_col == 6, f"iteration_number should be 6, got {iteration_col}"
        assert isinstance(advice_id_col, str) and len(advice_id_col) == 36, (
            f"advice_id should be a UUID string (len 36), got {advice_id_col!r}"
        )
        assert abs(cps_before_col - 0.034) < 1e-9, (
            f"fold_cps_v1_before should be 0.034, got {cps_before_col}"
        )

    def test_advice_id_is_unique_per_call(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """C5-ATTR-01: each accepted advice produces a distinct advice_id UUID."""
        cb = self._make_accepted_advice_callback(monkeypatch)
        # First call
        cb._query_epoch_advice()  # noqa: SLF001
        row1 = cb._adjustment_trigger_queue[0]
        advice_id_1 = row1[-2]

        # Reset pending so second call also accepts
        cb._pending_adjustment = None  # noqa: SLF001
        cb._epoch = cb._cadence * 2  # noqa: SLF001  # next storage epoch
        cb._query_epoch_advice()  # noqa: SLF001
        row2 = cb._adjustment_trigger_queue[1]
        advice_id_2 = row2[-2]

        assert advice_id_1 != advice_id_2, "Each advice call should produce a unique advice_id"

    def test_attribution_none_cps_when_no_context(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """C5-ATTR-01: fold_cps_v1_before is None when prev_iter_cps_v1 not set."""
        cb = self._make_accepted_advice_callback(monkeypatch, prev_iter_cps_v1=None)
        cb._query_epoch_advice()  # noqa: SLF001

        row = cb._adjustment_trigger_queue[0]
        assert row[-1] is None, f"cps_before should be None when no prior iter, got {row[-1]}"


# ---------------------------------------------------------------------------
# TestL1BenchDefaultVetoesEndToEnd
# ---------------------------------------------------------------------------


class TestL1BenchDefaultVetoesEndToEnd:
    """Finding 2 (Task 4 review): prove the shipped all-zero default actually vetoes.

    TestAttributionIdentity deliberately monkeypatches get_max_reward_delta back to
    a nonzero value to exercise attribution mechanics for a re-earned lever. Nothing
    previously proved the OPPOSITE: that the real shipped config (all algo/env pairs
    at 0.0, config/swingrl.yaml training.bounds.max_reward_delta) actually vetoes a
    reward-weight adjustment at the enforcement site in _query_epoch_advice(). This
    test drives that path with nothing monkeypatched — get_max_reward_delta and
    get_adjustment_cooldown resolve through the real load_config().
    """

    def test_default_config_vetoes_nonzero_delta_advice(self) -> None:
        """D-T2.1: shipped max_reward_delta=0.0 vetoes the reward-weight adjustment."""
        client = _make_mock_memory_client()
        client.epoch_advice.return_value = {
            "reward_weights": {
                "profit": 0.62,  # was 0.50 -> delta 0.12, well above the 0.01 no-op floor
                "sharpe": 0.20,
                "drawdown": 0.12,
                "turnover": 0.06,
            },
            "stop_training": False,
            "rationale": "test",
            "provider": "test",
            "model": "test",
        }
        wrapper = _make_mock_wrapper()
        original_weights = dict(wrapper.weights)
        cb = MemoryEpochCallback(
            memory_client=client,
            wrapper=wrapper,
            run_id="equity_ppo_fold3",
            algo="PPO",
            env="equity",
            verbose=0,
            advice_enabled=True,
            iteration=6,
            fold_number=3,
            database_url="postgresql://fake:fake@localhost/fake",  # pragma: allowlist secret
        )
        mock_logger = MagicMock()
        mock_logger.name_to_value = {}
        cb.model = MagicMock()
        cb.model.logger = mock_logger
        cb.num_timesteps = 0
        cb._epoch = cb._cadence  # noqa: SLF001  # storage epoch
        # Pre-inject fold context to bypass the lazy DB load
        cb._fold_context = {  # noqa: SLF001
            "fold_role": "neutral",
            "chronic_failure_folds": [],
            "protected_winner_folds": [],
            "prev_iter_cps_v1": 0.034,
        }

        cb._query_epoch_advice()  # noqa: SLF001

        assert cb._adjustment_trigger_queue == [], (
            "shipped default (0.0) must veto: no trigger row should be queued"
        )
        wrapper.update_weights.assert_not_called()
        assert wrapper.weights == original_weights, (
            "wrapper weights must be untouched when the L1 lever is benched"
        )


# ---------------------------------------------------------------------------
# TestNaNContextGuard
# ---------------------------------------------------------------------------


class TestNaNContextGuard:
    """C5-ATTR-02 carry-over: context JSON must not emit non-finite float tokens."""

    def _make_nan_callback(self) -> MemoryEpochCallback:
        """Callback whose wrapper returns NaN for rolling_sharpe (simulating early training)."""
        client = _make_mock_memory_client()
        client.epoch_advice.return_value = {
            "reward_weights": {
                "profit": 0.55,
                "sharpe": 0.25,
                "drawdown": 0.15,
                "turnover": 0.05,
            },
            "stop_training": False,
            "rationale": "test",
            "provider": "test",
            "model": "test",
        }
        wrapper = _make_mock_wrapper()
        # Make rolling_sharpe return NaN — simulates an early fold with no return data yet
        wrapper.rolling_sharpe.return_value = float("nan")

        cb = MemoryEpochCallback(
            memory_client=client,
            wrapper=wrapper,
            run_id="equity_ppo_fold0",
            algo="PPO",
            env="equity",
            verbose=0,
            advice_enabled=True,
            iteration=6,
            fold_number=0,
        )
        mock_logger = MagicMock()
        mock_logger.name_to_value = {}
        cb.model = MagicMock()
        cb.model.logger = mock_logger
        cb.num_timesteps = 1000
        cb._epoch = cb._cadence  # noqa: SLF001  # storage epoch
        # Pre-inject fold context to bypass DB load
        cb._fold_context = {  # noqa: SLF001
            "fold_role": "neutral",
            "chronic_failure_folds": [],
            "protected_winner_folds": [],
            "prev_iter_cps_v1": None,
        }
        return cb

    def test_nan_rolling_sharpe_skips_advice_no_crash(self) -> None:
        """C5-ATTR-02: NaN rolling_sharpe → allow_nan=False raises → advice skipped, no crash.

        Before the fix, json.dumps(context) would silently emit 'NaN' (a JS token,
        not valid JSON) to the LLM API call.  With allow_nan=False the ValueError
        is caught by the outer try/except in _query_epoch_advice, advice is skipped
        (fail-open), and epoch_advice is NOT called on the client.
        """
        cb = self._make_nan_callback()
        cb._query_epoch_advice()  # noqa: SLF001

        # Advice must have been skipped (not called) because context JSON raised
        cb._client.epoch_advice.assert_not_called()
        # Also verify the advice-failed counter was incremented (the except branch ran)
        assert cb._advice_timed_out == 1, (
            f"expected 1 advice_timed_out (skipped via exception), got {cb._advice_timed_out}"
        )


# ---------------------------------------------------------------------------
# TestStopTrainingAdviceOnly
# ---------------------------------------------------------------------------


class TestStopTrainingAdviceOnly:
    """U1 (spec §2.2): stop_training advice is logged/recorded, never actuated."""

    def _callback_with_mock_client(self) -> MemoryEpochCallback:
        """Create a callback whose model has enough progress to pass the floor check.

        Mirrors the brief's `callback_with_mock_client` fixture (no such fixture exists
        in this file's helper set — `_make_callback` is the equivalent local pattern).
        """
        cb = _make_callback()
        # _make_callback() sets cb.num_timesteps = 1000 and cb.model = MagicMock().
        # MagicMock auto-vivifies unset attrs, so getattr(model, "_total_timesteps", 1)
        # would return a MagicMock (not the intended default) and blow up the
        # progress-floor comparison in _query_epoch_advice. Pin it explicitly so
        # progress >= MIN_TRAINING_PROGRESS and the real branch under test executes.
        cb.model._total_timesteps = 1000  # noqa: SLF001
        # Real SB3 models start with stop_training=False; a bare MagicMock would
        # auto-vivify this attr to a truthy Mock on first access, which would make
        # "never actuated" trivially unverifiable. Pin the real initial state.
        cb.model.stop_training = False
        return cb

    def test_stop_training_advice_is_never_actuated(self) -> None:
        """U1 (spec §2.2): a stop_training=true response must not stop the run."""
        cb = self._callback_with_mock_client()
        cb._client.epoch_advice.return_value = {  # noqa: SLF001
            "reward_weights": {},
            "stop_training": True,
            "rationale": "test stop",
        }
        cb._epoch = cb._cadence - 1  # noqa: SLF001  # next rollout end is an advice epoch
        cb._on_rollout_end()  # noqa: SLF001

        assert getattr(cb.model, "stop_training", False) is False
        assert cb._on_step() is True  # noqa: SLF001
        assert len(cb._stop_requests) == 1  # noqa: SLF001
        assert cb._stop_requests[0]["reason"] == "test stop"  # noqa: SLF001


# ---------------------------------------------------------------------------
# TestOnTrainingStartWindowConfig (Task 5, spec §2.6)
# ---------------------------------------------------------------------------


class TestOnTrainingStartWindowConfig:
    """Task 5 (spec §2.6): _on_training_start sizes the wrapper's percent-of-fold
    windows from the run's REAL total_timesteps and enforces the startup guard.

    Both tests resolve get_adjustment_cooldown() and training.windows.*_pct_of_fold
    through the REAL load_config() (nothing monkeypatched) -- same style as
    TestL1BenchDefaultVetoesEndToEnd, proving the actual shipped defaults behave as
    documented: trend_pct_of_fold=0.15 (N2), SAC adjustment_cooldown_steps=20_000.
    """

    def test_on_training_start_configures_wrapper_windows(self) -> None:
        """Happy path: PPO/equity at 1,000,000 total_timesteps (DEFAULT_TIMESTEPS,
        pipeline_helpers.py:45-48) -> both windows sized from the real 0.01/0.15
        pct_of_fold defaults; no guard trip (trend 150,000 >> PPO cooldown 24,576)."""
        cb = _make_callback(run_id="equity_ppo_fold0", algo="PPO", env="equity")
        cb.model._total_timesteps = 1_000_000  # noqa: SLF001

        cb._on_training_start()  # noqa: SLF001

        cb._wrapper.configure_windows.assert_called_once_with(10_000, 150_000)

    def test_on_training_start_guard_raises_when_trend_window_too_short(self) -> None:
        """Guard (spec §2.6): trend_steps < get_adjustment_cooldown(algo) -> ConfigError.
        At a 100,000-step fold, trend_pct_of_fold=0.15 -> trend_steps=15,000, which is
        below the real SAC adjustment_cooldown_steps=20,000 -- refuses to start."""
        cb = _make_callback(run_id="equity_sac_fold0", algo="SAC", env="equity")
        cb.model._total_timesteps = 100_000  # noqa: SLF001

        with pytest.raises(ConfigError):
            cb._on_training_start()  # noqa: SLF001

        cb._wrapper.configure_windows.assert_not_called()
