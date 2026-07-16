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
import math
import uuid
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
# §4.10 (D-T3.19, F2 class-fix) fallback notable-event trigger thresholds -- used only
# if config load fails entirely. Must match NotableEventsConfig's pydantic defaults
# (config/schema.py), same fail-open pattern as _FALLBACK_SHORT_PCT_OF_FOLD /
# _FALLBACK_TREND_PCT_OF_FOLD below. RETIRES the old NOTABLE_KL_THRESHOLD=0.10 /
# NOTABLE_MDD_THRESHOLD=-25.0 pair: KL survives unchanged (well-defined, rare) but MDD
# moves off a cumsum-of-shaped-rewards scale (quasi-permanently true for crypto SAC --
# Task 5 made rolling_mdd() a bounded [-1, 0] equity fraction, which left -25.0
# permanently FALSE, i.e. the old trigger was silently inert) onto Task 5's
# window_metrics("short")["mdd_frac_worst"] equity-fraction basis, with three siblings
# (trade_shy, churning, numeric_anomaly) added per the redesigned five-trigger set.
_FALLBACK_KL_SPIKE_THRESHOLD: float = 0.10
_FALLBACK_MDD_BREACH_FRAC: dict[str, float] = {"equity": 0.10, "crypto": 0.12}
_FALLBACK_TRADE_SHY_RATIO: float = 0.5
_FALLBACK_CHURNING_RATIO: float = 3.0
_FALLBACK_HARD_CAP_PER_RUN: int = 50
# Epochs to wait before resolving pending adjustment
ADJUSTMENT_RESOLVE_EPOCHS: int = 10

# Fallback percent-of-fold window targets (spec §2.6, N1/N2) for _on_training_start's
# startup guard, used only if config load fails entirely. Must match
# TrainingWindowsConfig's pydantic defaults (config/schema.py) and
# reward_wrapper.py's own copy of these constants (each module independently
# config-loads and falls back, per this codebase's established bounds.py pattern).
_FALLBACK_SHORT_PCT_OF_FOLD: float = 0.01
_FALLBACK_TREND_PCT_OF_FOLD: float = 0.15


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
        database_url: str | None = None,
        fold_number: int | None = None,
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
            database_url: Optional PostgreSQL connection URL for epoch/adjustment writes.
            fold_number: Sequential fold index (0-based) within the current walk-forward run.
                Used to assemble the per-fold advice context and written to the
                reward_adjustments attribution columns.
        """
        super().__init__(verbose=verbose)
        self._client = memory_client
        self._wrapper = wrapper
        self._run_id = run_id
        self._env = env
        self._advice_enabled = advice_enabled
        self._is_control_fold = is_control_fold
        self._iteration = iteration
        self._database_url = database_url
        self._fold_number = fold_number
        # Lazy-loaded once on first _query_epoch_advice call; None = not yet loaded.
        self._fold_context: dict[str, Any] | None = None
        # UUID string for the most recent accepted advice event (attribution).
        self._advice_id: str = ""

        # Queue-based telemetry: buffer writes during training, flush post-fold.
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

        # §4.10 (D-T3.19): F2 trigger-set thresholds, env-scoped, config-sourced once
        # per fold (same fail-open pattern as _load_cadence). Sets self._kl_spike_threshold,
        # self._mdd_breach_frac, self._trade_shy_ratio, self._churning_ratio,
        # self._hard_cap_per_run.
        self._load_notable_events()

        # Event-path bounding (§4.10): rate cap = one row per trigger TYPE per trend
        # window; hard cap = total event rows per run. The cadence path above is never
        # subject to either -- it is arithmetically bounded (total_epochs / cadence)
        # already, and must always flow (fail-safe direction: lose telemetry, never the
        # heartbeat).
        self._events_this_window: dict[str, int] = {}
        self._event_rows_this_run: int = 0
        self._trend_window_idx: int = 0
        self._hard_cap_alarm_fired: bool = False

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

        # U1 (spec §2.2): stop_training is advice-only. Requests are recorded here
        # (epoch, timestep, pct_complete, reason) for Task 17's intent writer; the
        # runtime never actuates them — folds always run to completion.
        self._stop_requests: list[dict[str, Any]] = []

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

    def _load_notable_events(self) -> None:
        """Load training.notable_events from validated config, env-scoped (§4.10).

        Same fail-open pattern as _load_cadence / _on_training_start's window-pct
        load: reads once per fold (call site: __init__) so config changes take
        effect on the next fold without a container restart. Falls back to the
        hardcoded _FALLBACK_* module constants if config load fails entirely.
        """
        try:
            from swingrl.config.schema import load_config

            cfg = load_config()
            ne = cfg.training.notable_events
            self._kl_spike_threshold: float = ne.kl_spike_threshold
            self._mdd_breach_frac: float = ne.mdd_breach_frac.get(
                self._env.lower(), _FALLBACK_MDD_BREACH_FRAC.get(self._env.lower(), 0.10)
            )
            self._trade_shy_ratio: float = ne.trade_shy_ratio
            self._churning_ratio: float = ne.churning_ratio
            self._hard_cap_per_run: int = ne.hard_cap_per_run
        except Exception as exc:  # Fail-open: config load failure → use hardcoded defaults
            log.warning("notable_events_config_load_failed", error=str(exc))
            self._kl_spike_threshold = _FALLBACK_KL_SPIKE_THRESHOLD
            self._mdd_breach_frac = _FALLBACK_MDD_BREACH_FRAC.get(self._env.lower(), 0.10)
            self._trade_shy_ratio = _FALLBACK_TRADE_SHY_RATIO
            self._churning_ratio = _FALLBACK_CHURNING_RATIO
            self._hard_cap_per_run = _FALLBACK_HARD_CAP_PER_RUN

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
        """Flush buffered epoch/adjustment data to PostgreSQL.

        Call this AFTER model.learn() completes for each fold. All buffered rows
        are inserted in a single transaction.

        Safe to call multiple times (idempotent — clears queues after flush).
        Safe to call when database_url is None (no-op).
        """
        if self._database_url is None:
            return
        if (
            not self._epoch_queue
            and not self._adjustment_trigger_queue
            and not self._adjustment_outcome_queue
        ):
            return

        try:
            import os

            import psycopg
            from psycopg.rows import dict_row

            db_url = os.environ.get("DATABASE_URL") or self._database_url
            with psycopg.connect(db_url, row_factory=dict_row) as con:
                if self._epoch_queue:
                    with con.cursor() as cur:
                        cur.executemany(
                            """INSERT INTO training_epochs (
                                run_id, epoch, algo, env, timestep, mean_reward,
                                policy_loss, value_loss, entropy_loss, approx_kl,
                                clip_fraction, rolling_sharpe, rolling_mdd,
                                rolling_win_rate, reward_weights, notable_event,
                                is_control_fold
                            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)""",
                            self._epoch_queue,
                        )
                    con.commit()
                    log.info(
                        "pg_epoch_telemetry_flushed",
                        rows=len(self._epoch_queue),
                        run_id=self._run_id,
                    )

                if self._adjustment_trigger_queue:
                    with con.cursor() as cur:
                        cur.executemany(
                            """INSERT INTO reward_adjustments (
                                run_id, epoch_trigger, algo, env, trigger_metric,
                                trigger_value, trigger_reason, weight_before,
                                weight_after, sharpe_at_trigger, mdd_at_trigger,
                                fold_number, iteration_number, advice_id, fold_cps_v1_before
                            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)""",
                            self._adjustment_trigger_queue,
                        )
                    con.commit()
                    log.info(
                        "pg_adjustment_triggers_flushed",
                        rows=len(self._adjustment_trigger_queue),
                    )

                for params, run_id, epoch_trigger in self._adjustment_outcome_queue:
                    con.execute(
                        """UPDATE reward_adjustments
                        SET epoch_outcome = %s, outcome_sharpe = %s,
                            sharpe_delta = %s, mdd_delta = %s, effective = %s
                        WHERE run_id = %s AND epoch_trigger = %s
                          AND epoch_outcome IS NULL""",
                        params + [run_id, epoch_trigger],
                    )
                if self._adjustment_outcome_queue:
                    con.commit()
                    log.info(
                        "pg_adjustment_outcomes_flushed",
                        rows=len(self._adjustment_outcome_queue),
                    )
        except Exception as exc:
            log.error("pg_telemetry_flush_failed", error=str(exc))
        finally:
            self._epoch_queue.clear()
            self._adjustment_trigger_queue.clear()
            self._adjustment_outcome_queue.clear()

    def _on_step(self) -> bool:
        """Check if training should continue.

        U1 (spec §2.2): stop_training is advice-only. This always returns True —
        the LLM cannot halt a fold; it can only log a stop request for review.

        Returns:
            True unconditionally.
        """
        return True

    def _on_training_start(self) -> None:
        """SB3 hook fired once before rollout collection begins for this fold.

        Computes percent-of-fold window sizes from the run's ACTUAL total_timesteps
        (model._total_timesteps -- set by SB3's _setup_learn() before this hook
        fires, so an escalated run, spec: ESCALATED_TIMESTEPS, resizes correctly
        rather than inheriting a size baked in at construction time), configures the
        wrapper's short/trend windows, and enforces the §2.6 startup guard: the
        trend window must be long enough to observe at least one full
        adjustment-cooldown cycle for this fold's algo, or training refuses to start.
        Logs window_config with both units (dual-unit capture, D-T2.7).

        Raises:
            ConfigError: the trend window is shorter than this algo's adjustment
                cooldown -- refuses to start training (spec §2.6 guard).
        """
        from swingrl.config.schema import load_config
        from swingrl.memory.training.bounds import get_adjustment_cooldown
        from swingrl.utils.exceptions import ConfigError

        total_timesteps = int(getattr(self.model, "_total_timesteps", 0))

        try:
            cfg = load_config()
            short_pct = cfg.training.windows.short_pct_of_fold
            trend_pct = cfg.training.windows.trend_pct_of_fold
        except Exception as exc:
            log.warning("training_windows_config_load_failed", error=str(exc))
            short_pct = _FALLBACK_SHORT_PCT_OF_FOLD
            trend_pct = _FALLBACK_TREND_PCT_OF_FOLD

        short_steps = max(1, round(short_pct * total_timesteps))
        trend_steps = max(1, round(trend_pct * total_timesteps))

        cooldown = get_adjustment_cooldown(self._algo)
        if trend_steps < cooldown:
            log.error(
                "trend_window_shorter_than_cooldown",
                algo=self._algo,
                env=self._env,
                run_id=self._run_id,
                trend_steps=trend_steps,
                cooldown=cooldown,
                total_timesteps=total_timesteps,
            )
            raise ConfigError(
                f"trend window ({trend_steps} steps) is shorter than the "
                f"{self._algo} adjustment cooldown ({cooldown} steps) for run "
                f"{self._run_id!r} -- refusing to start training (spec §2.6 "
                "startup guard)."
            )

        self._wrapper.configure_windows(short_steps, trend_steps)

        log.info(
            "window_config",
            algo=self._algo,
            env=self._env,
            run_id=self._run_id,
            total_timesteps=total_timesteps,
            short_pct_of_fold=short_pct,
            short_steps=short_steps,
            trend_pct_of_fold=trend_pct,
            trend_steps=trend_steps,
        )

    def _on_rollout_end(self) -> None:
        """Called at the end of each rollout (epoch). Main callback entry point.

        Honest correction (§4.10, D-T3.19; this docstring previously claimed the
        opposite): SAC does NOT fire _on_rollout_end less often than PPO/A2C. SAC's
        train_freq=1 (continuous replay buffer, off-policy) means SB3 calls
        _on_rollout_end roughly every environment step -- ~167K times per fold, vs
        PPO's ~82 rollouts/fold. "Epoch" is therefore not a fixed unit of training
        progress across algos; the per-algo cadence (ALGO_EPOCH_CADENCE) and the
        §4.10 event-path rate/hard caps in _should_store exist BECAUSE of this fact,
        not despite it -- the old design's silent 688K-row blowup (F2) was this
        exact gap between assumption and reality. See _on_training_end's
        rollout_cadence_observed log for the actual per-fold fire count.
        """
        self._epoch += 1

        kl_key = _ALGO_LOGGER_KEYS.get(self._algo, _ALGO_LOGGER_KEYS["ppo"])["approx_kl"]
        approx_kl = float(self.logger.name_to_value.get(kl_key, 0.0)) if kl_key else 0.0
        mean_reward = self._wrapper.rolling_mean_reward()

        # Check if epoch should be stored
        should_store, notable_event = self._should_store(self._epoch, approx_kl, mean_reward)

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

    def _on_training_end(self) -> None:
        """SB3 hook fired once when model.learn() completes for this fold.

        F2 instrumentation (§4.10, D-T3.19): logs the actual _on_rollout_end fire
        count for this fold (self._epoch), replacing the prior undocumented and
        incorrect assumption with a measured fact, once per run.
        """
        log.info(
            "rollout_cadence_observed",
            run_id=self._run_id,
            algo=self._algo,
            env=self._env,
            rollout_end_count=self._epoch,
            event_rows_this_run=self._event_rows_this_run,
            hard_cap_per_run=self._hard_cap_per_run,
        )

    def _should_store(
        self,
        epoch: int,
        approx_kl: float,
        mean_reward: float,
    ) -> tuple[bool, str | None]:
        """Determine if this epoch warrants a snapshot (§4.10, D-T3.19 trigger set).

        The cadence path (``epoch % cadence == 0``) is unconditional and uncapped --
        it always returns ``(True, None)`` regardless of rate/hard cap state, so the
        fold's heartbeat telemetry is never starved by event-storm bookkeeping.

        Otherwise, evaluates five triggers against Task 5's short (acute-detector)
        window plus this call's ``approx_kl``/``mean_reward``, in numeric_anomaly-first
        precedence (a NaN/inf reading makes every other metric this epoch suspect --
        checked before trusting kl/mdd/trade-rate values that may themselves be
        corrupted): ``numeric_anomaly``, ``kl_spike``, ``mdd_breach``, ``trade_shy``,
        ``churning``. A fired trigger is subject to two independent bounds:

        - **Rate cap:** at most one event row per trigger TYPE per trend window
          (``self._events_this_window``, reset at each trend-window boundary via
          ``_roll_trend_window_if_needed``). A second occurrence of the same trigger
          type inside the same trend window is suppressed -- the onset is the signal.
        - **Hard cap:** at most ``self._hard_cap_per_run`` event rows per run
          (``self._event_rows_this_run``). Past it, the row drops and a
          ``capture_alarm`` fires exactly once (``self._hard_cap_alarm_fired``) --
          fail-safe direction: lose telemetry, never the run.

        Args:
            epoch: Current epoch number.
            approx_kl: Approximate KL divergence from training logs.
            mean_reward: Wrapper's rolling mean shaped reward (numeric_anomaly probe).

        Returns:
            Tuple of (should_store, notable_event_label or None).
        """
        self._roll_trend_window_if_needed()

        if epoch % self._cadence == 0:
            return True, None

        trigger = self._fired_trigger(approx_kl, mean_reward)
        if trigger is None:
            return False, None

        if self._events_this_window.get(trigger, 0) >= 1:
            log.debug("notable_event_rate_capped", trigger=trigger, epoch=epoch)
            return False, None

        if self._event_rows_this_run >= self._hard_cap_per_run:
            self._fire_hard_cap_alarm_once(trigger)
            return False, None

        self._events_this_window[trigger] = self._events_this_window.get(trigger, 0) + 1
        self._event_rows_this_run += 1
        return True, trigger

    def _fired_trigger(self, approx_kl: float, mean_reward: float) -> str | None:
        """Evaluate the five §4.10 triggers, numeric_anomaly-first.

        Args:
            approx_kl: Approximate KL divergence from training logs.
            mean_reward: Wrapper's rolling mean shaped reward.

        Returns:
            The first trigger label that fires, or None if the epoch is healthy.
        """
        if (
            math.isnan(mean_reward)
            or math.isinf(mean_reward)
            or math.isnan(approx_kl)
            or math.isinf(approx_kl)
        ):
            return "numeric_anomaly"

        if approx_kl > self._kl_spike_threshold:
            return "kl_spike"

        short = self._wrapper.window_metrics("short")
        if float(short.get("mdd_frac_worst", 0.0)) > self._mdd_breach_frac:
            return "mdd_breach"

        # trade_shy/churning read the wrapper's LOCKED baseline_trade_rate (existing
        # lock mechanism, reward_wrapper.py: set once the first full _ROLLING_WINDOW
        # fills, never overwritten thereafter). baseline == 0.0 means the window
        # hasn't locked yet (or locked on a zero-trade fold) -- both triggers are
        # disabled until then, matching cps_diagnosis.diagnose_rolling's own guard.
        baseline = self._wrapper.baseline_trade_rate()
        if baseline > 0.0:
            trade_rate = float(short.get("trade_rate", 0.0))
            if trade_rate < self._trade_shy_ratio * baseline:
                return "trade_shy"
            if trade_rate > self._churning_ratio * baseline:
                return "churning"

        return None

    def _roll_trend_window_if_needed(self) -> None:
        """Reset the per-trend-window rate-cap bookkeeping at each trend-window boundary.

        Trend-window index = ``num_timesteps // trend_steps`` (Task 5's
        ``configure_windows`` sizing) -- a tumbling (non-overlapping) count, distinct
        from the wrapper's own sliding deque used for metric computation.
        ``trend_steps == 0`` (windows not yet configured, e.g. before
        ``_on_training_start`` has run) means no boundary tracking is possible;
        treated as a single window 0 (rate cap effectively suspended until windows
        are configured, matching this callback's established fail-open posture).
        """
        trend_steps = int(self._wrapper.window_metrics("trend").get("steps", 0))
        window_idx = self.num_timesteps // trend_steps if trend_steps > 0 else 0
        if window_idx != self._trend_window_idx:
            self._trend_window_idx = window_idx
            self._events_this_window = {}

    def _fire_hard_cap_alarm_once(self, trigger: str) -> None:
        """Fire the D-T3.19 hard-cap alarm exactly once per run.

        Training containers have no Discord alerter today (Discord wiring lands in
        Task 19 with the grader alarms). This surfaces the breach through the memory
        client's existing ingest path as a ``capture_alarm`` row plus a structlog
        error, so the event-storm is recorded rather than silent -- noted, not silent.

        Args:
            trigger: The trigger label that was being evaluated when the hard cap
                was hit (for diagnostic context only -- the row itself still drops).
        """
        if self._hard_cap_alarm_fired:
            return
        self._hard_cap_alarm_fired = True

        text = (
            f"CAPTURE_ALARM: run_id={self._run_id} algo={self._algo} env={self._env} "
            f"epoch={self._epoch} hard_cap_per_run={self._hard_cap_per_run} "
            f"trigger_at_breach={trigger} reason=event_rows_exceeded_hard_cap"
        )
        ok = self._client.ingest_training(text, source=f"capture_alarm:{self._env}:{self._algo}")
        log.error(
            "notable_event_hard_cap_breached",
            run_id=self._run_id,
            epoch=self._epoch,
            hard_cap_per_run=self._hard_cap_per_run,
            trigger_at_breach=trigger,
            ingest_ok=ok,
        )

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
            "source": f"training_epoch:{self._env}:{self._algo}",
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
        ok = self._client.ingest_training(text, source=f"training_epoch:{self._env}:{self._algo}")
        log.debug(
            "epoch_snapshot_ingested",
            epoch=metrics["epoch"],
            ok=ok,
            notable=metrics["notable_event"],
        )

        # Buffer epoch snapshot for post-fold PostgreSQL flush
        if self._database_url is not None:
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
        ok = self._client.ingest_training(
            text, source=f"reward_adjustment:{self._env}:{self._algo}"
        )
        log.info(
            "adjustment_trigger_ingested",
            epoch=self._epoch,
            trigger_metric=trigger_metric,
            ok=ok,
        )

        # Buffer adjustment trigger for post-fold PostgreSQL flush
        if self._database_url is not None:
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
                    # Attribution identity (Task 8): fold / iteration / advice UUID / CPS before
                    self._fold_number,
                    self._iteration,
                    self._advice_id,
                    (self._fold_context or {}).get("prev_iter_cps_v1"),
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
        ok = self._client.ingest_training(
            text, source=f"reward_adjustment:{self._env}:{self._algo}"
        )
        log.info(
            "adjustment_outcome_ingested",
            epoch=self._epoch,
            sharpe_delta=round(sharpe_delta, 4),
            mdd_delta=round(mdd_delta, 4),
            effective=effective,
            ok=ok,
        )

        # Buffer adjustment outcome for post-fold PostgreSQL flush
        if self._database_url is not None:
            self._adjustment_outcome_queue.append(
                (
                    [
                        self._epoch,
                        current_sharpe,
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

            # Lazy-load fold context once per fold (skipped if no DB or no fold_number).
            if self._fold_context is None:
                if self._database_url and self._fold_number is not None:
                    from swingrl.memory.training.fold_context import load_fold_context

                    self._fold_context = load_fold_context(
                        self._database_url, self._env, self._fold_number
                    )
                else:
                    self._fold_context = {
                        "fold_role": "neutral",
                        "chronic_failure_folds": [],
                        "protected_winner_folds": [],
                        "prev_iter_cps_v1": None,
                    }

            from swingrl.memory.training.cps_diagnosis import diagnose_rolling
            from swingrl.utils.exceptions import DataError

            try:
                diagnosis: dict[str, Any] = dict(
                    diagnose_rolling(
                        trade_rate=self._wrapper.rolling_trade_rate(),
                        baseline_trade_rate=self._wrapper.baseline_trade_rate(),
                        rolling_win_rate=self._wrapper.rolling_win_rate(),
                        env=self._env,
                        algo=self._algo,
                    )
                )
            except DataError as exc:
                log.warning("diagnosis_unavailable", env=self._env, algo=self._algo, error=str(exc))
                diagnosis = {
                    "label": "healthy",
                    "fired": [],
                    "confidence": "clear",
                    "evidence": {},
                }

            context = {
                "fold_number": self._fold_number,
                "fold_role": self._fold_context["fold_role"],
                "prev_iter_cps_v1": self._fold_context["prev_iter_cps_v1"],
                "target_metric": "cps_v1_multiplicative",
                "leading_indicators": {
                    "rolling_sharpe": round(self._wrapper.rolling_sharpe(), 4),
                    "rolling_mdd": round(self._wrapper.rolling_mdd(), 4),
                    "rolling_win_rate": round(self._wrapper.rolling_win_rate(), 4),
                    "trade_rate": round(self._wrapper.rolling_trade_rate(), 4),
                    "baseline_trade_rate": round(self._wrapper.baseline_trade_rate(), 4),
                },
                "diagnosis": diagnosis,
            }

            iter_part = f" iteration={self._iteration}" if self._iteration is not None else ""
            payload = {
                "query": (
                    f"EPOCH ADVICE: run_id={self._run_id} algo={self._algo} "
                    f"env={self._env} epoch={self._epoch}{iter_part} "
                    f"current_weights={_json.dumps(self._wrapper.weights)} "
                    f"context={_json.dumps(context, allow_nan=False)}"
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
                # U1 (spec §2.2): advice-only — 0 stop requests in 850,430 live epochs,
                # no case for keeping actuation. The request is logged and recorded for
                # Task 17's intent writer; model.stop_training is never set.
                from swingrl.memory.training.bounds import MIN_TRAINING_PROGRESS

                progress = self.num_timesteps / max(getattr(self.model, "_total_timesteps", 1), 1)
                log.warning(
                    "llm_stop_request_advice_only",
                    epoch=self._epoch,
                    timestep=self.num_timesteps,
                    pct_complete=round(progress, 4),
                    min_required=MIN_TRAINING_PROGRESS,
                    reason=reason,
                )
                self._stop_requests.append(
                    {
                        "epoch": self._epoch,
                        "timestep": self.num_timesteps,
                        "pct_complete": round(progress, 4),
                        "reason": reason,
                    }
                )
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
                self._advice_id = str(uuid.uuid4())
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
