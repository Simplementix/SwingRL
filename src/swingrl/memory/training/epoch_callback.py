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
from typing import TYPE_CHECKING, Any

import structlog
from stable_baselines3.common.callbacks import BaseCallback

if TYPE_CHECKING:
    from swingrl.memory.client import MemoryClient
    from swingrl.memory.training.reward_wrapper import MemoryVecRewardWrapper

log = structlog.get_logger(__name__)

# How often to store epoch snapshots
EPOCH_STORE_CADENCE: int = 5
# Thresholds for "notable" epoch events
NOTABLE_KL_THRESHOLD: float = 0.02
NOTABLE_MDD_THRESHOLD: float = -0.08
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
    ) -> None:
        """Initialize the epoch callback.

        Args:
            memory_client: MemoryClient for ingesting training data.
            wrapper: MemoryVecRewardWrapper for weight updates.
            run_id: Training run identifier.
            algo: Algorithm name.
            env: Environment name.
            verbose: Verbosity level.
        """
        super().__init__(verbose=verbose)
        self._client = memory_client
        self._wrapper = wrapper
        self._run_id = run_id
        self._algo = algo
        self._env = env
        # Cache base URL at init; avoids repeated private attr access during training
        self._memory_base_url: str = memory_client._base_url  # noqa: SLF001

        self._epoch: int = 0
        self._curriculum_window_active: str = "unknown"
        self._curriculum_window_year_range: str = "unknown"

        # Two-pass adjustment tracking
        self._pending_adjustment: dict[str, Any] | None = None
        self._sharpe_at_trigger: float = 0.0
        self._mdd_at_trigger: float = 0.0

    def _on_step(self) -> bool:
        """Required by BaseCallback. Always returns True (never stops training)."""
        return True

    def _on_rollout_end(self) -> None:
        """Called at the end of each rollout (epoch). Main callback entry point."""
        self._epoch += 1

        approx_kl = float(self.logger.name_to_value.get("train/approx_kl", 0.0))
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
        if epoch % EPOCH_STORE_CADENCE == 0:
            return True, None
        if approx_kl > NOTABLE_KL_THRESHOLD:
            return True, "kl_spike"
        if rolling_mdd < NOTABLE_MDD_THRESHOLD:
            return True, "mdd_breach"
        return False, None

    def _collect_metrics(self, notable_event: str | None) -> dict[str, Any]:
        """Collect epoch metrics from the training logger.

        Args:
            notable_event: Optional notable event label.

        Returns:
            Dict of epoch metrics for memory ingestion.
        """
        get = self.logger.name_to_value.get

        return {
            "run_id": self._run_id,
            "algo": self._algo,
            "env": self._env,
            "epoch": self._epoch,
            "timestep": self.num_timesteps,
            "mean_reward": float(get("rollout/ep_rew_mean", 0.0)),
            "policy_loss": float(get("train/policy_gradient_loss", 0.0)),
            "value_loss": float(get("train/value_loss", 0.0)),
            "entropy_loss": float(get("train/entropy_loss", 0.0)),
            "approx_kl": float(get("train/approx_kl", 0.0)),
            "clip_fraction": float(get("train/clip_fraction", 0.0)),
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
            f"notable_event={metrics['notable_event']}"
        )
        ok = self._client.ingest_training(text, source="training_epoch:historical")
        log.debug(
            "epoch_snapshot_ingested",
            epoch=metrics["epoch"],
            ok=ok,
            notable=metrics["notable_event"],
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
        self._pending_adjustment = None

    def _query_epoch_advice(self) -> None:
        """Query memory agent for epoch-level reward weight advice.

        Fail-open: if the memory agent is unavailable or returns invalid advice,
        the current weights are unchanged and training continues uninterrupted.
        """
        # Only query on storage epochs to avoid hammering the API
        if self._epoch % EPOCH_STORE_CADENCE != 0:
            return

        try:
            import json as _json
            import urllib.request

            payload = {
                "query": (
                    f"EPOCH ADVICE: run_id={self._run_id} algo={self._algo} "
                    f"env={self._env} epoch={self._epoch} "
                    f"rolling_sharpe={self._wrapper.rolling_sharpe():.4f} "
                    f"rolling_mdd={self._wrapper.rolling_mdd():.4f} "
                    f"current_weights={_json.dumps(self._wrapper.weights)}"
                )
            }
            data = _json.dumps(payload).encode("utf-8")
            req = urllib.request.Request(
                f"{self._memory_base_url}/training/epoch_advice",
                data=data,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=5.0) as resp:  # noqa: S310  # nosec B310
                body = _json.loads(resp.read().decode("utf-8"))

            new_weights = body.get("reward_weights")
            reason = body.get("rationale", "")

            if isinstance(new_weights, dict) and new_weights:
                from swingrl.memory.training.bounds import clamp_reward_weights

                clamped = clamp_reward_weights(new_weights)
                old_weights = self._wrapper.weights
                self._wrapper.update_weights(clamped)
                self._ingest_adjustment_trigger(
                    new_weights=clamped,
                    old_weights=old_weights,
                    trigger_metric="epoch_advice",
                    trigger_value=self._wrapper.rolling_mdd(),
                    trigger_reason=reason,
                )
        except Exception as exc:
            log.debug("epoch_advice_failed", epoch=self._epoch, error=str(exc))
