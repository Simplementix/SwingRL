"""Meta-training orchestrator that wraps TrainingOrchestrator with LLM advice.

MetaTrainingOrchestrator is an outer loop that:
1. Queries LLM for run configuration (hyperparams + curriculum) before training
2. Clamps all LLM suggestions through bounds.py (non-negotiable safety layer)
3. Sets up MemoryVecRewardWrapper and MemoryEpochCallback
4. Runs trainer.learn() with memory-driven callbacks
5. Ingests training run summary and curriculum performance to memory agent

Cold-start guard: meta-trainer is passive until 3 completed runs exist in history.
All LLM call failures are silent — training never blocks on memory availability.

Usage:
    from swingrl.memory.training.meta_orchestrator import MetaTrainingOrchestrator
    orchestrator = MetaTrainingOrchestrator(config=config, memory_client=client)
    result = orchestrator.run(
        env_name="equity", algo_name="ppo", trainer=trainer_instance,
        features=features, prices=prices
    )
"""

from __future__ import annotations

import json
import time
import urllib.request
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import duckdb
import numpy as np
import structlog

from swingrl.memory.training.bounds import (
    clamp_reward_weights,
    clamp_run_config,
)

if TYPE_CHECKING:
    from swingrl.config.schema import SwingRLConfig
    from swingrl.memory.client import MemoryClient
    from swingrl.training.trainer import TrainingOrchestrator, TrainingResult

log = structlog.get_logger(__name__)

# Minimum consolidated patterns required before meta-trainer queries LLM for advice.
# Below this threshold, there isn't enough synthesized data for the LLM to produce
# meaningful hyperparameter/reward weight suggestions. The baseline iteration's
# walk-forward results must be consolidated first.
_COLD_START_MIN_PATTERNS: int = 1


class MetaTrainingOrchestrator:
    """Outer loop that wraps TrainingOrchestrator with LLM-guided meta-training.

    Provides hyperparameter advice, curriculum selection, and reward shaping
    via the memory agent. All suggestions are hard-clamped before reaching
    the trainer. Training never blocks on memory agent unavailability.

    Args:
        config: Validated SwingRLConfig with memory_agent section.
        memory_client: MemoryClient for LLM queries and ingestion.
        db_path: Optional DuckDB path for run history queries.
    """

    def __init__(
        self,
        config: SwingRLConfig,
        memory_client: MemoryClient,
        db_path: str | Path | None = None,
    ) -> None:
        """Initialize the meta-training orchestrator.

        Args:
            config: Validated SwingRLConfig.
            memory_client: MemoryClient for ingestion and queries.
            db_path: Optional DuckDB path. Defaults to config.system.duckdb_path.
        """
        self._config = config
        self._client = memory_client
        self._db_path = Path(db_path) if db_path else Path(config.system.duckdb_path)
        self._meta_cfg = config.memory_agent

    def run(
        self,
        env_name: str,
        algo_name: str,
        trainer: TrainingOrchestrator,
        features: np.ndarray,
        prices: np.ndarray,
        total_timesteps: int | None = None,
        hyperparams_override: dict[str, Any] | None = None,
    ) -> TrainingResult:
        """Run a single training iteration with LLM-guided meta-training.

        1. Queries run_config (hyperparams + curriculum) from memory agent
        2. Clamps all suggestions through bounds.py
        3. Applies config to trainer
        4. Wraps trainer env with MemoryVecRewardWrapper
        5. Sets up MemoryEpochCallback
        6. Calls trainer.train() and returns result
        7. Ingests run summary + curriculum performance

        Args:
            env_name: Environment name ("equity" or "crypto").
            algo_name: Algorithm name ("ppo", "a2c", or "sac").
            trainer: TrainingOrchestrator instance to wrap.
            features: Feature array (full or recent depending on use case).
            prices: Price array aligned with features.
            total_timesteps: Training timesteps (overrides config default).
            hyperparams_override: Optional baseline hyperparams to merge with LLM advice.

        Returns:
            TrainingResult from trainer.train().
        """
        run_id = self._generate_run_id(env_name, algo_name)
        start_time = datetime.now(tz=UTC)
        wall_start = time.monotonic()

        log.info(
            "meta_training_started",
            run_id=run_id,
            env_name=env_name,
            algo_name=algo_name,
        )

        # Query full run config (HP + reward weights + rationale) — single call
        advised_config = self._query_run_config(env_name, algo_name)
        safe_config = clamp_run_config(advised_config) if advised_config else {}

        # Extract SB3-compatible HP keys
        hp_keys = {
            "learning_rate",
            "entropy_coeff",
            "clip_range",
            "n_epochs",
            "batch_size",
            "gamma",
        }
        memory_hp: dict[str, Any] = {}
        for k, v in safe_config.items():
            if k in hp_keys:
                key = "ent_coef" if k == "entropy_coeff" else k
                memory_hp[key] = v

        # Merge: baseline overrides win on conflict
        merged_hp: dict[str, Any] = {}
        if hyperparams_override:
            merged_hp.update(hyperparams_override)
        for k, v in memory_hp.items():
            if k not in merged_hp:
                merged_hp[k] = v

        # Reward weights from LLM advice
        reward_weights_raw = safe_config.get("reward_weights", {})
        if reward_weights_raw:
            reward_weights = clamp_reward_weights(reward_weights_raw)
        else:
            reward_weights = None  # use MemoryVecRewardWrapper defaults

        # Determine timesteps
        from swingrl.training.pipeline_helpers import DEFAULT_TIMESTEPS

        ts = total_timesteps or DEFAULT_TIMESTEPS.get(env_name, 1_000_000)

        # Train with memory client injected — trainer wires MemoryVecRewardWrapper
        # and MemoryEpochCallback internally when memory_client is provided.
        result = trainer.train(
            env_name=env_name,
            algo_name=algo_name,
            features=features,
            prices=prices,
            total_timesteps=ts,
            hyperparams_override=merged_hp if merged_hp else None,
            memory_client=self._client,
            run_id=run_id,
            initial_reward_weights=reward_weights,
        )

        # Ingest run summary
        end_time = datetime.now(tz=UTC)
        wall_elapsed = time.monotonic() - wall_start

        regime_vector = self._current_regime_vector(env_name)
        final_metrics = self._compute_final_metrics(result)
        summary_text = self._build_run_summary_text(
            run_id=run_id,
            algo_name=algo_name,
            env_name=env_name,
            start_time=start_time,
            end_time=end_time,
            result=result,
            final_metrics=final_metrics,
            merged_hp=merged_hp,
            reward_weights=reward_weights,
            regime_vector=regime_vector,
            rationale=advised_config.get("rationale", "cold_start"),
        )
        self._client.ingest_training(summary_text, source="training_run:historical")

        log.info(
            "meta_training_complete",
            run_id=run_id,
            wall_elapsed_s=round(wall_elapsed, 1),
        )

        return result

    def query_hyperparams(self, env_name: str, algo_name: str) -> dict[str, Any]:
        """Query memory agent for walk-forward hyperparameter advice.

        Returns SB3-compatible HP keys (learning_rate, ent_coef, clip_range,
        n_epochs, batch_size, gamma). Returns empty dict on cold-start or
        any failure (fail-open).
        """
        advised = self._query_run_config(env_name, algo_name)
        if not advised:
            return {}

        safe = clamp_run_config(advised)

        hp_keys = {
            "learning_rate",
            "entropy_coeff",
            "clip_range",
            "n_epochs",
            "batch_size",
            "gamma",
        }
        result: dict[str, Any] = {}
        for k, v in safe.items():
            if k in hp_keys:
                key = "ent_coef" if k == "entropy_coeff" else k
                result[key] = v

        log.info(
            "wf_hyperparams_queried",
            env_name=env_name,
            algo_name=algo_name,
            hp=result,
        )
        return result

    def _query_run_config(self, env_name: str, algo_name: str) -> dict[str, Any]:
        """Query memory agent for run configuration advice.

        Cold-start guard: returns empty dict (passthrough) until
        _COLD_START_MIN_RUNS completed runs exist.

        Args:
            env_name: Environment name.
            algo_name: Algorithm name.

        Returns:
            Dict with optional keys: learning_rate, entropy_coeff, clip_range,
            n_epochs, batch_size, gamma, reward_weights, rationale.
            Empty dict on cold-start or any failure.
        """
        pattern_count = self._get_pattern_count(env_name)
        if pattern_count < _COLD_START_MIN_PATTERNS:
            log.info(
                "meta_cold_start_guard",
                pattern_count=pattern_count,
                min_patterns=_COLD_START_MIN_PATTERNS,
                env=env_name,
                algo=algo_name,
                action="using_baseline_params",
            )
            return {}

        try:
            regime = self._current_regime_vector(env_name)

            payload = {
                "query": (
                    f"TRAINING RUN CONFIG ADVICE: env={env_name} algo={algo_name} "
                    f"current_regime={json.dumps(regime)}"
                )
            }
            data = json.dumps(payload).encode("utf-8")
            timeout = self._meta_cfg.meta_training_timeout_sec

            headers: dict[str, str] = {"Content-Type": "application/json"}
            if self._client.api_key:
                headers["X-API-Key"] = self._client.api_key

            req = urllib.request.Request(
                f"{self._meta_cfg.base_url}/training/run_config",
                data=data,
                headers=headers,
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310  # nosec B310
                body = json.loads(resp.read().decode("utf-8"))

            log.info(
                "meta_run_config_received",
                env_name=env_name,
                algo_name=algo_name,
                keys=list(body.keys()),
            )
            return dict(body)  # type: ignore[return-value]

        except Exception as exc:
            log.debug("meta_run_config_failed", error=str(exc))
            return {}

    def _apply_run_config(
        self,
        trainer: TrainingOrchestrator,
        safe_config: dict[str, Any],
    ) -> None:
        """Apply clamped run config to trainer instance attributes.

        Args:
            trainer: TrainingOrchestrator to configure.
            safe_config: Clamped config dict from clamp_run_config().
        """
        # TrainingOrchestrator reads hyperparams from HYPERPARAMS dict;
        # overrides are applied at train() call time via hyperparams_override.
        # This method is a hook for future direct attribute injection.
        log.debug("meta_apply_run_config", keys=list(safe_config.keys()))

    def _get_pattern_count(self, env_name: str) -> int:
        """Check how many consolidated patterns exist for this environment.

        Queries the memory service debug/consolidations endpoint and filters
        by env_name client-side. Used by the cold-start guard to determine
        if enough data has been synthesized for meaningful LLM advice.

        Args:
            env_name: Environment name ("equity" or "crypto").

        Returns:
            Number of active consolidated patterns for this env. 0 on any error.
        """
        try:
            url = f"{self._meta_cfg.base_url}/debug/consolidations?limit=100"
            headers: dict[str, str] = {}
            if self._client.api_key:
                headers["X-API-Key"] = self._client.api_key
            req = urllib.request.Request(url, headers=headers, method="GET")
            with urllib.request.urlopen(req, timeout=5.0) as resp:  # noqa: S310  # nosec B310
                patterns = json.loads(resp.read().decode("utf-8"))
            # Filter to this env's patterns (env_name field or affected_envs)
            count = sum(
                1
                for p in patterns
                if p.get("env_name") == env_name or env_name in (p.get("affected_envs") or [])
            )
            log.info("meta_pattern_count", env=env_name, count=count)
            return count
        except Exception as exc:
            log.warning(
                "meta_pattern_count_query_failed",
                env=env_name,
                error=str(exc),
                fallback="cold_start_baseline",
            )
            return 0

    def _current_regime_vector(self, env_name: str) -> dict[str, float]:
        """Query latest HMM regime probabilities from DuckDB.

        Returns:
            Dict with keys: bull, bear, crisis, sideways.
            Falls back to {"bull": 0.33, "bear": 0.33, "crisis": 0.17, "sideways": 0.17}
            on any error.
        """
        try:
            with duckdb.connect(str(self._db_path), read_only=True) as conn:
                row = conn.execute(
                    """
                    SELECT p_bull, p_bear, p_crisis
                    FROM hmm_state_history
                    WHERE environment = ?
                    ORDER BY date DESC
                    LIMIT 1
                    """,
                    [env_name],
                ).fetchone()

            if row is None:
                return {"bull": 0.33, "bear": 0.33, "crisis": 0.17, "sideways": 0.17}

            p_bull = float(row[0]) if row[0] is not None else 0.33
            p_bear = float(row[1]) if row[1] is not None else 0.33
            p_crisis = float(row[2]) if row[2] is not None else 0.17
            p_sideways = max(0.0, 1.0 - p_bull - p_bear - p_crisis)

            return {
                "bull": p_bull,
                "bear": p_bear,
                "crisis": p_crisis,
                "sideways": p_sideways,
            }

        except Exception as exc:
            log.debug("meta_regime_query_failed", error=str(exc))
            return {"bull": 0.33, "bear": 0.33, "crisis": 0.17, "sideways": 0.17}

    def _compute_final_metrics(
        self,
        result: TrainingResult,
    ) -> dict[str, float]:
        """Compute final training metrics from TrainingResult.

        Args:
            result: TrainingResult from trainer.train().

        Returns:
            Dict with sharpe, mdd, sortino, mean_reward placeholders.
            Actual values should come from walk-forward evaluation.
        """
        # TrainingResult doesn't carry walk-forward eval metrics — those are
        # computed later in train_pipeline.py and ingested separately via
        # walk_forward:{env}:{algo} source tags. Report convergence info here.
        return {
            "converged": result.converged_at_step is not None,
            "converged_at_step": result.converged_at_step or result.total_timesteps,
            "total_timesteps": result.total_timesteps,
            "convergence_ratio": (
                round(result.converged_at_step / result.total_timesteps, 3)
                if result.converged_at_step
                else 1.0
            ),
        }

    def _build_run_summary_text(
        self,
        run_id: str,
        algo_name: str,
        env_name: str,
        start_time: datetime,
        end_time: datetime,
        result: TrainingResult,
        final_metrics: dict[str, float],
        merged_hp: dict[str, Any],
        reward_weights: dict[str, float] | None,
        regime_vector: dict[str, float],
        rationale: str,
    ) -> str:
        """Build training run summary text for memory ingestion.

        Args:
            run_id: Training run identifier.
            algo_name: Algorithm name.
            env_name: Environment name.
            start_time: Training start UTC datetime.
            end_time: Training end UTC datetime.
            result: TrainingResult from trainer.
            final_metrics: Final performance metrics.
            merged_hp: Hyperparameters used.
            reward_weights: Reward shaping weights (or None).
            regime_vector: HMM regime probabilities at training time.
            rationale: LLM rationale text.

        Returns:
            Formatted text string for memory ingestion.
        """
        rw = reward_weights or {"profit": 0.50, "sharpe": 0.25, "drawdown": 0.15, "turnover": 0.10}

        # Determine dominant regime
        dominant_regime = max(regime_vector, key=lambda k: regime_vector[k])

        return (
            f"TRAINING RUN SUMMARY: run_id={run_id} algo={algo_name.upper()} env={env_name} "
            f"timestamp_start={start_time.isoformat()} timestamp_end={end_time.isoformat()} "
            f"final_sharpe={final_metrics.get('final_sharpe', 0.0):.4f} "
            f"final_mdd={final_metrics.get('final_mdd', 0.0):.4f} "
            f"final_sortino={final_metrics.get('final_sortino', 0.0):.4f} "
            f"final_mean_reward={final_metrics.get('final_mean_reward', 0.0):.4f} "
            f"total_timesteps={result.total_timesteps} "
            f"epochs_to_convergence={result.converged_at_step or result.total_timesteps} "
            f"stopped_early={result.converged_at_step is not None} "
            f"learning_rate={merged_hp.get('learning_rate', 'default')} "
            f"batch_size={merged_hp.get('batch_size', 'default')} "
            f"n_epochs={merged_hp.get('n_epochs', 'default')} "
            f"gamma={merged_hp.get('gamma', 'default')} "
            f"clip_range={merged_hp.get('clip_range', 'default')} "
            f"entropy_coeff={merged_hp.get('entropy_coeff', merged_hp.get('ent_coef', 'default'))} "
            f"reward_weight_profit={rw.get('profit', 0.50):.4f} "
            f"reward_weight_sharpe={rw.get('sharpe', 0.25):.4f} "
            f"reward_weight_drawdown={rw.get('drawdown', 0.15):.4f} "
            f"reward_weight_turnover={rw.get('turnover', 0.10):.4f} "
            f"dominant_regime={dominant_regime} "
            f"regime_bull={regime_vector.get('bull', 0.33):.4f} "
            f"regime_bear={regime_vector.get('bear', 0.33):.4f} "
            f"regime_crisis={regime_vector.get('crisis', 0.17):.4f} "
            f"regime_sideways={regime_vector.get('sideways', 0.17):.4f} "
            f"meta_rationale={rationale}"
        )

    @staticmethod
    def _generate_run_id(env_name: str, algo_name: str) -> str:
        """Generate a unique run identifier.

        Args:
            env_name: Environment name.
            algo_name: Algorithm name.

        Returns:
            Run ID string (e.g. "equity_ppo_20260313T145526Z").
        """
        ts = datetime.now(tz=UTC).strftime("%Y%m%dT%H%M%SZ")
        return f"{env_name}_{algo_name}_{ts}"
