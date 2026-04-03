"""Memory epoch callback for training-time event ingestion and LLM advice.

MemoryEpochCallback fires on every rollout end and:
1. Stores epoch snapshots to memory every 5th epoch or on notable events
   (KL spike, MDD breach)
2. Queries LLM for epoch advice (reward weight adjustments) — fail-open
3. Implements two-pass adjustment tracking: ingests trigger immediately,
   ingests outcome 10 epochs later

Usage:
    from swingrl.memory.training.epoch_callback import MemoryEpochCallback
    callback = MemoryEpochCallback(
        memory_client=client,
        wrapper=reward_wrapper,
        run_id="run_042",
        algo="PPO",
        env="equity",
    )
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

import structlog
from stable_baselines3.common.callbacks import BaseCallback

if TYPE_CHECKING:
    from swingrl.memory.client import MemoryClient
    from swingrl.memory.training.reward_wrapper import MemoryVecRewardWrapper

log = structlog.get_logger(__name__)

# Per-algo epoch store cadence. Reduced frequency for smarter 14b model (fewer calls,
# each one higher quality). YAML config takes precedence; these are fallback defaults.
# PPO: n_steps=2048, n_envs=6 → ~82 rollouts/fold → cadence 60 → ~1.4 calls
# A2C: n_steps=5, n_envs=6 → ~33,334 rollouts/fold → cadence 8000 → ~4 calls
# SAC: n_steps=1, n_envs=6 → ~166,667 rollouts/fold → cadence 40000 → ~4 calls
ALGO_EPOCH_CADENCE: dict[str, int] = {
    "PPO": 60,
    "A2C": 8000,
    "SAC": 40000,
}
EPOCH_STORE_CADENCE: int = 500  # Fallback for unknown algos

# Per-algo SB3 logger key mappings. PPO/A2C/SAC use different internal names
# for the same conceptual metrics. None means the metric does not exist for that algo.
_ALGO_LOGGER_KEYS: dict[str, dict[str, str | None]] = {
    "ppo": {
        "policy_loss": "train/policy_gradient_loss",
        "value_loss": "train/value_loss",
        "entropy_loss": "train/entropy_loss",
        "approx_kl": "train/approx_kl",
        "clip_fraction": "train/clip_fraction",
    },
    "a2c": {
        "policy_loss": "train/policy_loss",
        "value_loss": "train/value_loss",
        "entropy_loss": "train/entropy_loss",
        "approx_kl": None,
        "clip_fraction": None,
    },
    "sac": {
        "policy_loss": "train/actor_loss",
        "value_loss": "train/critic_loss",
        "entropy_loss": "train/ent_coef_loss",
        "approx_kl": None,
        "clip_fraction": None,
    },
}
# Thresholds for "notable" epoch events (P99/P1 based on iter 1 data analysis).
# Previous values (0.02 KL, -0.08 MDD) fired on 80% of epochs, generating 2.8M
# memories instead of the target ~45K. These thresholds capture only the top/bottom
# ~1% of events.
NOTABLE_KL_THRESHOLD: float = 0.10
NOTABLE_MDD_THRESHOLD: float = -25.0
# Epochs to wait before resolving pending adjustment
ADJUSTMENT_RESOLVE_EPOCHS: int = 10


class MemoryEpochCallback(BaseCallback):
    """Stable-Baselines3 callback that ingests epoch snapshots and drives reward adjustments.

    On every rollout end:
    - Increments epoch counter
    - Checks if epoch snapshot should be stored (cadence + notable events)
    - Queries LLM for reward weight advice (fail-open)
    - Applies approved weight changes to the MemoryVecRewardWrapper
    - Resolves pending two-pass adjustment outcomes

    Args:
        memory_client: MemoryClient for ingesting training data.
        wrapper: MemoryVecRewardWrapper to update weights on.
        run_id: Training run identifier (e.g. "run_042").
        algo: Algorithm name (e.g. "PPO").
        env: Environment name (e.g. "equity").
        verbose: Verbosity level (0 = silent).
    """

    def __init__(
        self,
        memory_client: MemoryClient,
        wrapper: MemoryVecRewardWrapper,
        run_id: str,
        algo: str,
        env: str,
        verbose: int = 0,
        advice_enabled: bool = True,
        is_control_fold: bool = False,
        iteration: int | None = None,
        duckdb_path: str | Path | None = None,
    ) -> None:
        """Initialize the epoch callback.

        Args:
            memory_client: MemoryClient for ingesting training data.
            wrapper: MemoryVecRewardWrapper for weight updates.
            run_id: Training run identifier.
            algo: Algorithm name.
            env: Environment name.
            verbose: Verbosity level.
            advice_enabled: If False, epoch snapshots are stored but LLM advice is skipped.
            is_control_fold: If True, this fold is a scientific control group
                (no reward adjustments). Tagged in epoch snapshot text.
            iteration: Training iteration number for pattern presentation tracking.
            duckdb_path: Optional path to DuckDB database for direct epoch/adjustment writes.
        """
        super().__init__(verbose=verbose)
        self._client = memory_client
        self._wrapper = wrapper
        self._run_id = run_id
        self._env = env
        self._advice_enabled = advice_enabled
        self._is_control_fold = is_control_fold
        self._iteration = iteration
        self._duckdb_path = Path(duckdb_path) if duckdb_path else None

        # Queue-based DuckDB telemetry: buffer writes during training, flush post-fold.
        # Avoids DuckDB single-writer lock contention with WF backtester.
        self._epoch_queue: list[list[Any]] = []
        self._adjustment_trigger_queue: list[list[Any]] = []
        self._adjustment_outcome_queue: list[tuple[list[Any], str, int]] = []

        # LLM advice call counters
        self._advice_calls: int = 0
        self._advice_succeeded: int = 0
        self._advice_timed_out: int = 0
        self._advice_provider_used: str = ""

        # Parse algo from run_id (format: {env}_{algo}_fold{N}), fallback to algo param.
        try:
            parts = run_id.split("_")
            self._algo = parts[1].lower() if len(parts) >= 3 else algo.lower()
        except (IndexError, AttributeError):
            self._algo = "ppo"
        self._epoch: int = 0
        self._cadence: int = self._load_cadence(algo)
        self._curriculum_window_active: str = "unknown"
        self._curriculum_window_year_range: str = "unknown"

        # Tracks whether epoch_advice has failed at least once (for log-level escalation)
        self._advice_failed_once: bool = False

        # Per-algo reward adjustment cooldown tracking
        self._last_adjustment_step: int = 0

        log.info(
            "epoch_callback_cadence",
            algo=algo,
            cadence=self._cadence,
            run_id=run_id,
        )

        # Two-pass adjustment tracking
        self._pending_adjustment: dict[str, Any] | None = None
        self._sharpe_at_trigger: float = 0.0
        self._mdd_at_trigger: float = 0.0

    @staticmethod
    def _load_cadence(algo: str) -> int:
        """Read algo-specific cadence from validated config.

        Uses load_config() for consistent Pydantic validation and env var overrides.
        Re-reads from disk each time a new callback is created (once per fold),
        so config changes take effect on the next fold without container restart.
        Falls back to hardcoded ALGO_EPOCH_CADENCE dict if config load fails.
        """
        try:
            from pathlib import Path

            from swingrl.config.schema import load_config

            config_path = Path("/app/config/swingrl.yaml")
            if config_path.exists():
                config = load_config(config_path)
                key = f"epoch_cadence_{algo.lower()}"
                val = getattr(config.memory_agent, key, None)
                if val is not None:
                    return int(val)
        except Exception:  # nosec B110  # Fail-open: config load failure → use hardcoded defaults
            pass
        cadence = ALGO_EPOCH_CADENCE.get(algo)
        if cadence is None:
            log.warning(
                "unknown_algo_epoch_cadence_using_fallback",
                algo=algo,
                fallback=EPOCH_STORE_CADENCE,
                known_algos=list(ALGO_EPOCH_CADENCE.keys()),
            )
            return EPOCH_STORE_CADENCE
        return cadence

    @property
    def advice_stats(self) -> dict[str, Any]:
        """Return summary of LLM advice call statistics."""
        return {
            "advice_calls": self._advice_calls,
            "advice_succeeded": self._advice_succeeded,
            "advice_timed_out": self._advice_timed_out,
            "advice_provider_used": self._advice_provider_used,
        }

    def flush_telemetry(self) -> None:
        """Flush buffered epoch/adjustment data to DuckDB.

        Call this AFTER model.learn() completes for each fold, when the WF
        backtester is not holding the DuckDB write lock. All buffered rows
        are inserted in a single transaction.

        Safe to call multiple times (idempotent — clears queues after flush).
        Safe to call when duckdb_path is None (no-op).
        """
        if self._duckdb_path is None:
            return
        if (
            not self._epoch_queue
            and not self._adjustment_trigger_queue
            and not self._adjustment_outcome_queue
        ):
            return

        try:
            import duckdb

            con = duckdb.connect(str(self._duckdb_path))

            if self._epoch_queue:
                con.executemany(
                    """INSERT INTO training_epochs (
                        run_id, epoch, algo, env, timestep, mean_reward,
                        policy_loss, value_loss, entropy_loss, approx_kl,
                        clip_fraction, rolling_sharpe, rolling_mdd,
                        rolling_win_rate, reward_weights, notable_event,
                        is_control_fold
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    self._epoch_queue,
                )
                log.info(
                    "duckdb_epoch_telemetry_flushed",
                    rows=len(self._epoch_queue),
                    run_id=self._run_id,
                )

            if self._adjustment_trigger_queue:
                con.executemany(
                    """INSERT INTO reward_adjustments (
                        run_id, epoch_trigger, algo, env, trigger_metric,
                        trigger_value, trigger_reason, weight_before,
                        weight_after, sharpe_at_trigger, mdd_at_trigger
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    self._adjustment_trigger_queue,
                )
                log.info(
                    "duckdb_adjustment_triggers_flushed",
                    rows=len(self._adjustment_trigger_queue),
                )

            for params, run_id, epoch_trigger in self._adjustment_outcome_queue:
                con.execute(
                    """UPDATE reward_adjustments
                    SET epoch_outcome = ?, outcome_sharpe = ?,
                        sharpe_delta = ?, mdd_delta = ?, effective = ?
                    WHERE run_id = ? AND epoch_trigger = ?
                      AND epoch_outcome IS NULL""",
                    params + [run_id, epoch_trigger],
                )
            if self._adjustment_outcome_queue:
                log.info(
                    "duckdb_adjustment_outcomes_flushed",
                    rows=len(self._adjustment_outcome_queue),
                )

            con.close()
        except Exception as exc:
            log.error("duckdb_telemetry_flush_failed", error=str(exc))
        finally:
            self._epoch_queue.clear()
            self._adjustment_trigger_queue.clear()
            self._adjustment_outcome_queue.clear()

    def _on_step(self) -> bool:
        """Check if training should continue.

        Returns:
            False if stop_training was set (e.g. by LLM advice), True otherwise.
        """
        return not getattr(self.model, "stop_training", False)

    def _on_rollout_end(self) -> None:
        """Called at the end of each rollout (epoch). Main callback entry point.

        Note: SAC uses a continuous replay buffer rather than explicit rollouts,
        so SB3 calls _on_rollout_end less frequently for SAC than for PPO/A2C.
        Epoch snapshots will therefore be sparser during SAC training — this is
        expected and does not affect correctness.
        """
        self._epoch += 1

        kl_key = _ALGO_LOGGER_KEYS.get(self._algo, _ALGO_LOGGER_KEYS["ppo"])["approx_kl"]
        approx_kl = float(self.logger.name_to_value.get(kl_key, 0.0)) if kl_key else 0.0
        rolling_mdd = self._wrapper.rolling_mdd()

        # Check if epoch should be stored
        should_store, notable_event = self._should_store(self._epoch, approx_kl, rolling_mdd)

        if should_store:
            metrics = self._collect_metrics(notable_event)
            self._ingest_epoch_snapshot(metrics)

        # Resolve pending two-pass adjustment
        if self._pending_adjustment is not None:
            epoch_triggered = self._pending_adjustment.get("epoch_triggered", 0)
            if self._epoch - epoch_triggered >= ADJUSTMENT_RESOLVE_EPOCHS:
                self._resolve_pending_adjustment()

        # Query LLM for epoch advice (fail-open)
        self._query_epoch_advice()

    def _should_store(
        self,
        epoch: int,
        approx_kl: float,
        rolling_mdd: float,
    ) -> tuple[bool, str | None]:
        """Determine if this epoch warrants a snapshot.

        Args:
            epoch: Current epoch number.
            approx_kl: Approximate KL divergence from training logs.
            rolling_mdd: Rolling maximum drawdown.

        Returns:
            Tuple of (should_store, notable_event_label or None).
        """
        if epoch % self._cadence == 0:
            return True, None
        if approx_kl > NOTABLE_KL_THRESHOLD:
            return True, "kl_spike"
        if rolling_mdd < NOTABLE_MDD_THRESHOLD:
            return True, "mdd_breach"
        return False, None

    def _collect_metrics(self, notable_event: str | None) -> dict[str, Any]:
        """Collect epoch metrics from the training logger and model state.

        Args:
            notable_event: Optional notable event label.

        Returns:
            Dict of epoch metrics for memory ingestion.
        """
        get = self.logger.name_to_value.get
        keys = _ALGO_LOGGER_KEYS.get(self._algo, _ALGO_LOGGER_KEYS["ppo"])

        # SB3's rollout/ep_rew_mean requires Monitor wrapper (not used here).
        # Use the MemoryVecRewardWrapper's rolling mean instead — it tracks
        # the same shaped per-step rewards used for rolling_sharpe/mdd/win_rate.
        mean_reward = self._wrapper.rolling_mean_reward()

        return {
            "run_id": self._run_id,
            "algo": self._algo,
            "env": self._env,
            "epoch": self._epoch,
            "timestep": self.num_timesteps,
            "mean_reward": mean_reward,
            "policy_loss": float(get(keys["policy_loss"], 0.0)) if keys["policy_loss"] else 0.0,
            "value_loss": float(get(keys["value_loss"], 0.0)) if keys["value_loss"] else 0.0,
            "entropy_loss": float(get(keys["entropy_loss"], 0.0)) if keys["entropy_loss"] else 0.0,
            "approx_kl": float(get(keys["approx_kl"], 0.0)) if keys["approx_kl"] else 0.0,
            "clip_fraction": float(get(keys["clip_fraction"], 0.0))
            if keys["clip_fraction"]
            else 0.0,
            "rolling_sharpe_500": self._wrapper.rolling_sharpe(),
            "rolling_mdd_500": self._wrapper.rolling_mdd(),
            "rolling_win_rate_500": self._wrapper.rolling_win_rate(),
            "reward_weights": self._wrapper.weights,
            "curriculum_window_active": self._curriculum_window_active,
            "curriculum_window_year_range": self._curriculum_window_year_range,
            "hmm_regime_at_timestep": "",
            "hmm_regime_confidence": 0.0,
            "notable_event": notable_event,
            "notable_detail": None,
            "source": "training_epoch:historical",
        }

    def _ingest_epoch_snapshot(self, metrics: dict[str, Any]) -> None:
        """Ingest epoch snapshot to memory agent.

        Args:
            metrics: Epoch metrics dict from _collect_metrics().
        """
        text = (
            f"EPOCH SNAPSHOT: run_id={metrics['run_id']} algo={metrics['algo']} "
            f"env={metrics['env']} epoch={metrics['epoch']} "
            f"timestep={metrics['timestep']} "
            f"mean_reward={metrics['mean_reward']:.4f} "
            f"policy_loss={metrics['policy_loss']:.6f} "
            f"value_loss={metrics['value_loss']:.6f} "
            f"entropy_loss={metrics['entropy_loss']:.6f} "
            f"approx_kl={metrics['approx_kl']:.6f} "
            f"clip_fraction={metrics['clip_fraction']:.4f} "
            f"rolling_sharpe_500={metrics['rolling_sharpe_500']:.4f} "
            f"rolling_mdd_500={metrics['rolling_mdd_500']:.4f} "
            f"rolling_win_rate_500={metrics['rolling_win_rate_500']:.4f} "
            f"reward_weights={json.dumps(metrics['reward_weights'])} "
            f"curriculum_window={metrics['curriculum_window_active']} "
            f"notable_event={metrics['notable_event']} "
            f"is_control_fold={self._is_control_fold}"
        )
        ok = self._client.ingest_training(text, source="training_epoch:historical")
        log.debug(
            "epoch_snapshot_ingested",
            epoch=metrics["epoch"],
            ok=ok,
            notable=metrics["notable_event"],
        )

        # Buffer epoch snapshot for post-fold DuckDB flush (avoids single-writer deadlock)
        if self._duckdb_path is not None:
            self._epoch_queue.append(
                [
                    metrics["run_id"],
                    metrics["epoch"],
                    metrics["algo"],
                    metrics["env"],
                    metrics["timestep"],
                    metrics["mean_reward"],
                    metrics["policy_loss"],
                    metrics["value_loss"],
                    metrics["entropy_loss"],
                    metrics["approx_kl"],
                    metrics["clip_fraction"],
                    metrics["rolling_sharpe_500"],
                    metrics["rolling_mdd_500"],
                    metrics["rolling_win_rate_500"],
                    json.dumps(metrics["reward_weights"]),
                    metrics["notable_event"],
                    self._is_control_fold,
                ]
            )

    def _ingest_adjustment_trigger(
        self,
        new_weights: dict[str, float],
        old_weights: dict[str, float],
        trigger_metric: str,
        trigger_value: float,
        trigger_reason: str,
    ) -> None:
        """Ingest Pass 1 of a reward adjustment event (trigger moment).

        Args:
            new_weights: New reward weights after adjustment.
            old_weights: Previous reward weights.
            trigger_metric: Metric that triggered adjustment (e.g. "rolling_mdd_500").
            trigger_value: Value of the trigger metric.
            trigger_reason: LLM rationale text.
        """
        self._pending_adjustment = {
            "epoch_triggered": self._epoch,
            "trigger_metric": trigger_metric,
            "trigger_value": trigger_value,
            "trigger_reason": trigger_reason,
            "weights_before": old_weights,
            "weights_after": new_weights,
            "curriculum_window_at_trigger": self._curriculum_window_active,
            "regime_at_trigger": "unknown",
        }
        self._sharpe_at_trigger = self._wrapper.rolling_sharpe()
        self._mdd_at_trigger = self._wrapper.rolling_mdd()

        text = (
            f"REWARD_ADJUSTMENT_TRIGGER: run_id={self._run_id} algo={self._algo} "
            f"env={self._env} epoch_triggered={self._epoch} "
            f"trigger_metric={trigger_metric} trigger_value={trigger_value:.4f} "
            f"trigger_reason={trigger_reason} "
            f"weights_before={json.dumps(old_weights)} "
            f"weights_after={json.dumps(new_weights)} "
            f"curriculum_window={self._curriculum_window_active}"
        )
        ok = self._client.ingest_training(text, source="reward_adjustment:historical")
        log.info(
            "adjustment_trigger_ingested",
            epoch=self._epoch,
            trigger_metric=trigger_metric,
            ok=ok,
        )

        # Buffer adjustment trigger for post-fold DuckDB flush
        if self._duckdb_path is not None:
            self._adjustment_trigger_queue.append(
                [
                    self._run_id,
                    self._epoch,
                    self._algo,
                    self._env,
                    trigger_metric,
                    trigger_value,
                    trigger_reason,
                    json.dumps(old_weights),
                    json.dumps(new_weights),
                    self._sharpe_at_trigger,
                    self._mdd_at_trigger,
                ]
            )

    def _resolve_pending_adjustment(self) -> None:
        """Ingest Pass 2 of a reward adjustment event (10-epoch outcome).

        Computes the effectiveness of the prior weight adjustment and ingests
        the outcome with before/after Sharpe and MDD deltas.
        """
        if self._pending_adjustment is None:
            return

        current_sharpe = self._wrapper.rolling_sharpe()
        current_mdd = self._wrapper.rolling_mdd()

        sharpe_delta = current_sharpe - self._sharpe_at_trigger
        mdd_delta = current_mdd - self._mdd_at_trigger  # positive = less drawdown
        effective = sharpe_delta > 0 or mdd_delta > 0

        adj = self._pending_adjustment
        text = (
            f"REWARD_ADJUSTMENT_OUTCOME: run_id={self._run_id} algo={self._algo} "
            f"env={self._env} epoch_triggered={adj['epoch_triggered']} "
            f"epochs_measured_over={ADJUSTMENT_RESOLVE_EPOCHS} "
            f"post_adjustment_sharpe_delta={sharpe_delta:.4f} "
            f"post_adjustment_mdd_delta={mdd_delta:.4f} "
            f"adjustment_effective={effective} "
            f"weights_before={json.dumps(adj['weights_before'])} "
            f"weights_after={json.dumps(adj['weights_after'])}"
        )
        ok = self._client.ingest_training(text, source="reward_adjustment:historical")
        log.info(
            "adjustment_outcome_ingested",
            epoch=self._epoch,
            sharpe_delta=round(sharpe_delta, 4),
            mdd_delta=round(mdd_delta, 4),
            effective=effective,
            ok=ok,
        )

        # Buffer adjustment outcome for post-fold DuckDB flush
        if self._duckdb_path is not None:
            self._adjustment_outcome_queue.append(
                (
                    [
                        self._epoch,
                        sharpe_delta,
                        sharpe_delta,
                        mdd_delta,
                        effective,
                    ],
                    self._run_id,
                    adj["epoch_triggered"],
                )
            )

        self._pending_adjustment = None

    def _query_epoch_advice(self) -> None:
        """Query memory agent for epoch-level reward weight advice.

        Fail-open: if the memory agent is unavailable or returns invalid advice,
        the current weights are unchanged and training continues uninterrupted.
        """
        # Only query on storage epochs to avoid hammering the API
        if self._epoch % self._cadence != 0:
            return

        if not self._advice_enabled:
            return

        try:
            import json as _json

            iter_part = f" iteration={self._iteration}" if self._iteration is not None else ""
            payload = {
                "query": (
                    f"EPOCH ADVICE: run_id={self._run_id} algo={self._algo} "
                    f"env={self._env} epoch={self._epoch}{iter_part} "
                    f"rolling_sharpe={self._wrapper.rolling_sharpe():.4f} "
                    f"rolling_mdd={self._wrapper.rolling_mdd():.4f} "
                    f"current_weights={_json.dumps(self._wrapper.weights)}"
                )
            }
            self._advice_calls += 1
            body = self._client.epoch_advice(payload)
            if not body:
                self._advice_timed_out += 1
                return

            reason = body.get("rationale", "")

            stop_training = body.get("stop_training", False)
            if stop_training:
                from swingrl.memory.training.bounds import MIN_TRAINING_PROGRESS

                progress = self.num_timesteps / max(getattr(self.model, "_total_timesteps", 1), 1)
                if progress < MIN_TRAINING_PROGRESS:
                    log.info(
                        "stop_training_ignored_too_early",
                        progress=f"{progress:.1%}",
                        min_required="20%",
                        epoch=self._epoch,
                    )
                else:
                    log.warning(
                        "llm_advises_stop_training",
                        epoch=self._epoch,
                        reason=reason,
                    )
                    self.model.stop_training = True  # type: ignore[attr-defined]
                return

            new_weights = body.get("reward_weights")

            if isinstance(new_weights, dict) and new_weights:
                from swingrl.memory.training.bounds import (
                    clamp_reward_weights,
                    get_adjustment_cooldown,
                    get_max_reward_delta,
                )

                # Per-algo cooldown: reject if too soon after last adjustment
                cooldown = get_adjustment_cooldown(self._algo)
                steps_since = self.num_timesteps - self._last_adjustment_step
                if self._last_adjustment_step > 0 and steps_since < cooldown:
                    log.debug(
                        "epoch_advice_cooldown_active",
                        epoch=self._epoch,
                        algo=self._algo,
                        steps_since=steps_since,
                        cooldown=cooldown,
                    )
                    return

                # Per-algo/env max_delta: cap or disable reward adjustments
                algo_max_delta = get_max_reward_delta(self._algo, self._env)
                if algo_max_delta <= 0.0:
                    log.debug(
                        "epoch_advice_adjustments_disabled",
                        epoch=self._epoch,
                        algo=self._algo,
                        env=self._env,
                    )
                    return

                clamped = clamp_reward_weights(new_weights)
                old_weights = self._wrapper.weights

                # Change detection: skip if max absolute delta < 0.01
                max_delta = max(
                    abs(clamped.get(k, 0.0) - old_weights.get(k, 0.0))
                    for k in set(clamped) | set(old_weights)
                )
                if max_delta < 0.01:
                    log.debug(
                        "epoch_advice_no_change",
                        epoch=self._epoch,
                        max_delta=round(max_delta, 4),
                    )
                    return

                # Cap per-component delta to algo/env max_delta
                if max_delta > algo_max_delta:
                    scale = algo_max_delta / max_delta
                    clamped = {
                        k: old_weights.get(k, 0.0)
                        + (clamped.get(k, 0.0) - old_weights.get(k, 0.0)) * scale
                        for k in set(clamped) | set(old_weights)
                    }
                    # Renormalize to sum=1.0
                    total = sum(clamped.values())
                    if total > 0:
                        clamped = {k: v / total for k, v in clamped.items()}
                    log.info(
                        "epoch_advice_delta_capped",
                        epoch=self._epoch,
                        algo=self._algo,
                        env=self._env,
                        raw_max_delta=round(max_delta, 4),
                        capped_to=round(algo_max_delta, 4),
                    )

                # Resolve existing pending adjustment before overwriting
                if self._pending_adjustment is not None:
                    log.warning(
                        "epoch_advice_resolving_pending_early",
                        epoch=self._epoch,
                        pending_epoch=self._pending_adjustment.get("epoch_triggered"),
                    )
                    self._resolve_pending_adjustment()

                self._wrapper.update_weights(clamped)
                self._last_adjustment_step = self.num_timesteps
                self._ingest_adjustment_trigger(
                    new_weights=clamped,
                    old_weights=old_weights,
                    trigger_metric="epoch_advice",
                    trigger_value=self._wrapper.rolling_mdd(),
                    trigger_reason=reason,
                )
                self._advice_succeeded += 1
                self._advice_provider_used = body.get("provider", "unknown")
        except Exception as exc:
            self._advice_timed_out += 1
            # Log first failure at info so it's visible in production; subsequent
            # failures at debug to avoid log spam during prolonged unavailability.
            if not self._advice_failed_once:
                log.info("epoch_advice_failed_first", epoch=self._epoch, error=str(exc))
                self._advice_failed_once = True
            else:
                log.debug("epoch_advice_failed", epoch=self._epoch, error=str(exc))
