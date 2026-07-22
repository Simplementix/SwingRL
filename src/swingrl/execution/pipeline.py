"""ExecutionPipeline: orchestrates weight-based rebalancing trading middleware.

Wires together model inference, ensemble blending, process_actions (softmax +
deadzone — same function used during training), order validation, broker
submission, and fill processing into a single execute_cycle() call.

Usage:
    from swingrl.execution.pipeline import ExecutionPipeline
    pipeline = ExecutionPipeline(config, db, feature_pipeline, alerter, models_dir)
    fills = pipeline.execute_cycle("equity", dry_run=False)
"""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal
from zoneinfo import ZoneInfo

import numpy as np
import structlog

from swingrl.envs.portfolio import process_actions
from swingrl.execution.cycle_recorder import AlgoProposal, CycleRecorder, RegimeStamp
from swingrl.execution.fill_processor import FillProcessor
from swingrl.execution.model_paths import active_model_paths
from swingrl.execution.order_validator import OrderValidator
from swingrl.execution.risk.circuit_breaker import CBState, CircuitBreaker, GlobalCircuitBreaker
from swingrl.execution.risk.position_tracker import PositionTracker
from swingrl.execution.risk.risk_manager import RiskManager
from swingrl.execution.types import FillResult, SizedOrder
from swingrl.features.assembler import turbulence_obs_index
from swingrl.features.health import FeatureHealthTracker
from swingrl.utils.exceptions import CircuitBreakerError, DataError, RiskVetoError

if TYPE_CHECKING:
    from swingrl.config.schema import SwingRLConfig
    from swingrl.data.db import DatabaseManager
    from swingrl.features.pipeline import FeaturePipeline
    from swingrl.monitoring.alerter import Alerter

log = structlog.get_logger(__name__)

# Algorithms used for ensemble inference
_ALGO_NAMES: list[str] = ["ppo", "a2c", "sac"]


class ExecutionPipeline:
    """Orchestrates weight-based rebalancing execution middleware.

    Flow matches training environment:
        1. Model inference: per-algo predictions with per-algo VecNormalize
        2. Ensemble blending: weighted sum of per-algo actions
        3. process_actions: softmax + deadzone (same function used in training)
        4. Weight-based rebalancing: target_weight * portfolio_value -> delta orders
        5. OrderValidator: SizedOrder -> ValidatedOrder (cost gate + risk)
        6. ExchangeAdapter: ValidatedOrder -> FillResult (broker submission)
        7. FillProcessor: FillResult -> DB recording + position update
    """

    def __init__(
        self,
        config: SwingRLConfig,
        db: DatabaseManager,
        feature_pipeline: FeaturePipeline,
        alerter: Alerter,
        models_dir: Path,
    ) -> None:
        """Initialize execution pipeline.

        Args:
            config: Validated SwingRLConfig.
            db: DatabaseManager for database access.
            feature_pipeline: FeaturePipeline for observation assembly.
            alerter: Discord alerter for critical/warning notifications.
            models_dir: Root directory for trained models.
        """
        self._config = config
        self._db = db
        self._feature_pipeline = feature_pipeline
        self._alerter = alerter
        self._models_dir = models_dir

        # Feature health tracking for live inference
        self._health_tracker = FeatureHealthTracker()

        # Lazy-initialized components. Models are cached per env, keyed by the
        # artifact mtimes so a hot-swapped model (changed mtime) or a previously
        # empty load (no models on disk yet) reloads next cycle (review H3).
        self._models: dict[str, dict[str, tuple[Any, Any]]] = {}
        self._model_cache_keys: dict[str, tuple[Any, ...]] = {}
        self._adapters: dict[str, Any] = {}
        self._initialized = False

        # Per-(env, date) turbulence halt-baseline cache (one delegate call/cycle)
        self._turb_baseline_cache: dict[tuple[str, str], float] = {}

        # Real turbulence sensor value read from the observation before the F1b
        # slot-zeroing (kept for capture — §4.7 / A27). None until the first cycle.
        self._turbulence_at_decision: float | None = None

        # Eagerly create components that don't need lazy loading
        self._position_tracker = PositionTracker(db=db, config=config)
        self._fill_processor = FillProcessor(db=db, config=config)

        # Fail-open per-cycle capture writer (regime + per-algo proposals, §4.7).
        self._cycle_recorder = CycleRecorder(db=db, config=config, alerter=alerter)

        # Circuit breakers — alerter injected so every trip/auto-resume alerts (review M1)
        self._circuit_breakers: dict[str, CircuitBreaker] = {
            "equity": CircuitBreaker("equity", db, config, alerter=alerter),
            "crypto": CircuitBreaker("crypto", db, config, alerter=alerter),
        }
        self._global_cb = GlobalCircuitBreaker(self._circuit_breakers, config, db, alerter=alerter)

        # Risk manager
        self._risk_manager = RiskManager(
            config=config,
            db=db,
            position_tracker=self._position_tracker,
            circuit_breakers=self._circuit_breakers,
            global_cb=self._global_cb,
        )

        self._order_validator = OrderValidator(config=config, risk_manager=self._risk_manager)

        log.info("execution_pipeline_initialized", models_dir=str(models_dir))

    @property
    def config(self) -> SwingRLConfig:
        """Public accessor for the pipeline config."""
        return self._config

    @property
    def feature_pipeline(self) -> FeaturePipeline:
        """Public accessor for the feature pipeline."""
        return self._feature_pipeline

    @property
    def db(self) -> DatabaseManager:
        """Public accessor for the database manager."""
        return self._db

    def execute_cycle(
        self,
        env_name: str,
        dry_run: bool = False,
    ) -> list[FillResult]:
        """Run a full trading cycle for the given environment.

        Uses weight-based rebalancing that mirrors the training environment:
        model output -> process_actions (softmax + deadzone) -> target weights ->
        delta orders via broker.

        Args:
            env_name: Environment name ("equity" or "crypto").
            dry_run: If True, skip broker submission.

        Returns:
            List of FillResult for successful fills. Empty on CB halt or turbulence.
        """
        log.info("cycle_started", env=env_name, dry_run=dry_run)

        # One canonical timestamp for the whole cycle (review fold a): the date
        # string, the inference-outcome row, and the capture row all derive from
        # this single value rather than re-reading the clock several times.
        cycle_ts = datetime.now(UTC)

        # Step 1: Check circuit breaker state
        cb = self._circuit_breakers.get(env_name)
        if cb is not None:
            state = cb.get_state()
            if state == CBState.HALTED:
                log.warning("cycle_halted_by_cb", env=env_name)
                self._record_halt(env_name, cycle_ts, "circuit_breaker", dry_run)
                return []

        # Resolve the cycle date once so the turbulence halt check and the
        # observation share a single per-(env, date) turbulence value (M6). Derived
        # from the canonical cycle_ts so date and timestamp never disagree.
        current_date_str = self._get_current_date_str(env_name, cycle_ts)

        # Step 2: Check turbulence crash protection (PAPER-20)
        if self._check_turbulence(env_name, current_date_str):
            log.warning("cycle_halted_by_turbulence", env=env_name)
            # Record the halt with the turbulence that triggered it. compute_turbulence
            # is cached per (env, date), so this reuses the value _check_turbulence just
            # computed — a cache hit, not a re-computation (fold f consistency holds).
            self._record_halt(
                env_name,
                cycle_ts,
                "turbulence_halt",
                dry_run,
                self._feature_pipeline.compute_turbulence(env_name, current_date_str),
            )
            return []

        # Step 2.5: Equity market-calendar gate (fail-safe). Crypto trades 24/7.
        # Market closed/holiday -> skip + info log; clock unreachable -> skip + alert
        # (when in doubt, don't trade — review C2).
        if env_name == "equity" and self._config.equity.market_calendar_gate:
            if not self._equity_market_open():
                return []
            # Data-freshness guard (log-only): warn on a stale latest daily bar.
            self._warn_if_stale_ohlcv(env_name)

        # Step 3: Get observation from FeaturePipeline
        env_literal: Literal["equity", "crypto"] = "equity" if env_name == "equity" else "crypto"
        observation = self._feature_pipeline.get_observation(env_literal, current_date_str)

        # Step 3b: Check feature health — block trading on degraded features
        obs_health = self._health_tracker.assess(env_name)
        if obs_health.should_block:
            log.error(
                "trading_blocked_degraded_features",
                env=env_name,
                reason=obs_health.reason,
            )
            if self._alerter:
                self._alerter.send_alert(
                    level="warning",
                    title="Trading Blocked",
                    message=f"Trading blocked for {env_name}: {obs_health.reason}",
                )
            self._record_halt(env_name, cycle_ts, "degraded_features", dry_run)
            return []

        # Step 3c: Track NaN observations in inference_outcomes table
        had_nan = bool(np.isnan(observation).any())
        try:
            with self._db.connection() as conn:
                conn.execute(
                    "INSERT INTO inference_outcomes (timestamp, environment, had_nan) "
                    "VALUES (%s, %s, %s)",
                    (cycle_ts.isoformat(), env_name, int(had_nan)),
                )
        except Exception:
            log.warning("inference_outcome_tracking_failed", exc_info=True)

        if had_nan:
            log.warning("nan_observation_detected", env=env_name)
            self._record_halt(env_name, cycle_ts, "nan_obs", dry_run)
            return []

        # Step 3d: F1b — era-0 models were trained with the turbulence slot frozen
        # at 0.0, so feeding a real value would multiply it by untrained weights
        # (noise). Read the real sensor value out FIRST for capture (§4.7 / A27),
        # then zero the slot in the observation handed to the models. The flag is
        # flipped off once era-1 models (trained with a live turbulence input) deploy.
        env_symbols = (
            self._config.equity.symbols if env_name == "equity" else self._config.crypto.symbols
        )
        turb_idx = turbulence_obs_index(
            env_name,
            len(env_symbols),
            self._config.sentiment.enabled if env_name == "equity" else False,
        )
        # Consume the real sensor value into a cycle-local BEFORE zeroing. The
        # local (not self._turbulence_at_decision) feeds capture, so an overlapping
        # equity/crypto cycle overwriting the shared attribute cannot corrupt this
        # cycle's stamp (review folds e/f — single per-cycle value, no re-compute).
        turbulence_at_decision = float(observation[turb_idx])
        self._turbulence_at_decision = turbulence_at_decision
        if self._config.environment.zero_turbulence_obs:
            observation[turb_idx] = 0.0

        # Step 4: Get portfolio state (used by ObservationAssembler in future)
        _portfolio_state = self._position_tracker.get_portfolio_state_array(env_name)

        # Step 5: Load models first (needed for VecNormalize lookup)
        models = self._load_models(env_name)

        # Step 6: Per-algo VecNormalize observations
        normalized_obs = self._normalize_observation(env_name, observation)
        actions: dict[str, np.ndarray] = {}
        for algo_name, (model, _vec_norm) in models.items():
            algo_obs = normalized_obs.get(algo_name, observation)
            action, _ = model.predict(algo_obs, deterministic=True)
            actions[algo_name] = action
            log.debug("model_predicted", env=env_name, algo=algo_name)

        # Step 7: Load ensemble weights and blend actions
        weights = self._get_ensemble_weights(env_name)

        from swingrl.training.ensemble import EnsembleBlender

        blender = EnsembleBlender(self._config)
        blended_actions = blender.blend_actions(actions, weights)

        # Step 8: Weight-based rebalancing (mirrors training env)
        current_weights = self._get_current_weights(env_name)
        deadzone = self._config.environment.signal_deadzone
        target_weights = process_actions(blended_actions, current_weights, deadzone=deadzone)

        log.info(
            "target_weights_computed",
            env=env_name,
            target_weights=target_weights.tolist(),
            current_weights=current_weights.tolist(),
        )

        # Step 9: Generate rebalancing orders from weight deltas
        symbols = (
            self._config.equity.symbols if env_name == "equity" else self._config.crypto.symbols
        )
        fills: list[FillResult] = []
        # Prices fetched this cycle, reused to mark the portfolio to market for the
        # end-of-cycle snapshot (review C1 — no extra broker calls).
        cycle_prices: dict[str, float] = {}
        adapter = self._get_adapter(env_name)
        portfolio_value = self._position_tracker.get_portfolio_value(env_name)

        if portfolio_value <= 0:
            log.error("zero_portfolio_value", env=env_name, value=portfolio_value)
            # Early-exit cycle still recorded so a zero-portfolio halt is visible (fold b).
            self._record_halt(env_name, cycle_ts, "zero_portfolio", dry_run, turbulence_at_decision)
            return []

        # Minimum order value comes from config per env (no hardcoded floor).
        min_order_value = self._min_order_value(env_name)

        # Step 9b: Capture this cycle (regime + per-algo proposals + blended weights +
        # per-symbol M2 skip reasons + dry_run tag). Placed after the min-order
        # threshold is known so skip reasons are classified from the same inputs Step 9
        # uses, but BEFORE any order-submission side effect. Fail-open — never blocks
        # the money path (§4.7). Dry-runs are recorded and tagged (fold d), not skipped.
        skip_reasons = self._classify_skips(
            env_symbols, target_weights, current_weights, portfolio_value, min_order_value
        )
        cycle_id = self._capture_cycle(
            env_name=env_name,
            cycle_ts=cycle_ts,
            current_date_str=current_date_str,
            turbulence_at_decision=turbulence_at_decision,
            symbols=env_symbols,
            actions=actions,
            ensemble_weights=weights,
            target_weights=target_weights,
            dry_run=dry_run,
            skip_reasons=skip_reasons,
        )

        for i, symbol in enumerate(symbols):
            target_value = float(target_weights[i]) * portfolio_value
            current_value = float(current_weights[i]) * portfolio_value
            delta_value = target_value - current_value

            if abs(delta_value) < min_order_value:
                continue

            side: Literal["buy", "sell"] = "buy" if delta_value > 0 else "sell"

            try:
                current_price = adapter.get_current_price(symbol)
                if current_price <= 0:
                    log.warning("zero_price_skip", symbol=symbol)
                    continue

                # Record the fresh price to mark the portfolio to market later.
                cycle_prices[symbol] = current_price

                quantity = abs(delta_value) / current_price

                sized_order = SizedOrder(
                    symbol=symbol,
                    side=side,
                    quantity=quantity,
                    dollar_amount=abs(delta_value),
                    stop_loss_price=None,
                    take_profit_price=None,
                    environment=env_literal,
                )

                # Fresh mark-to-market value for the pre-trade risk evaluation
                # (amendment 2026-07-16): the drawdown/daily-loss breakers see this
                # cycle's fetched prices, not the last stored snapshot, so a held-position
                # drawdown is visible even when no order fills.
                fresh_value = self._position_tracker.compute_portfolio_value(env_name, cycle_prices)

                # Post-halt ramp enforcement (review H4): scale the order to the
                # breaker's current capacity fraction BEFORE validation and submission,
                # so the validator, the broker, and the fill record all see the scaled
                # amount (an ACTIVE breaker returns it unchanged).
                sized_order = self._risk_manager.apply_ramp_capacity(sized_order)

                # Risk validation (guardrail)
                validated_order = self._order_validator.validate(
                    sized_order, portfolio_value=fresh_value
                )

                # Dry-run: log but skip broker submission
                if dry_run:
                    log.info(
                        "dry_run_would_submit",
                        symbol=symbol,
                        side=side,
                        dollar_amount=abs(delta_value),
                        quantity=quantity,
                    )
                    continue

                # Broker submission — idempotency relies on PositionReconciler
                # running post-cycle to detect and correct any duplicate fills
                # caused by network timeouts (Alpaca bracket orders are atomic).
                fill = adapter.submit_order(validated_order)

                # Only real fills are recorded (review C2). A pending/rejected result
                # (unfilled + cancelled) is dropped — never a $0 trades row.
                if fill.status != "filled":
                    log.warning(
                        "order_unfilled_skipped",
                        symbol=fill.symbol,
                        side=fill.side,
                        status=fill.status,
                    )
                    continue

                # M10-equity backstop: a real fill that fails to record is money that
                # moved without a ledger entry — alert critical for manual reconciliation.
                # cycle_id/decision_price thread this fill back to its inference cycle and
                # sizing-time price (Task 10, §3.7.5) — decision_price is current_price,
                # the get_current_price() value Step 9 used to size this order.
                try:
                    self._fill_processor.process(
                        fill,
                        sized_order=sized_order,
                        cycle_id=cycle_id,
                        decision_price=current_price,
                    )
                except Exception:
                    log.critical(
                        "fill_recorded_failed_after_execution",
                        symbol=fill.symbol,
                        trade_id=fill.trade_id,
                        exc_info=True,
                    )
                    if self._alerter is not None:
                        self._alerter.send_alert(
                            level="critical",
                            title="Fill Executed But Not Recorded",
                            message=(
                                f"{env_name} {fill.symbol} {fill.side} executed but recording "
                                "failed — manual reconciliation required."
                            ),
                            environment=env_literal,
                        )
                    continue

                fills.append(fill)

                log.info(
                    "fill_complete",
                    symbol=fill.symbol,
                    side=fill.side,
                    quantity=fill.quantity,
                    fill_price=fill.fill_price,
                )

            except RiskVetoError as exc:
                log.warning(
                    "order_vetoed",
                    symbol=symbol,
                    reason=str(exc),
                )
                continue

            except CircuitBreakerError as exc:
                log.error(
                    "circuit_breaker_during_cycle",
                    symbol=symbol,
                    reason=str(exc),
                )
                break

            except Exception:
                log.exception(
                    "order_processing_failed",
                    symbol=symbol,
                )
                continue

        # Step 10: Record portfolio snapshot EVERY cycle (review C1/M4 + amendment
        # 2026-07-16). Mark positions to this cycle's fetched prices and derive cash from
        # the trades ledger so total_value reflects reality — including moved prices on a
        # zero-fill cycle — not the previous snapshot copied forward. Snapshots stay
        # append-only. Dry-run is a simulation and never writes a snapshot.
        if not dry_run:
            # Mark EVERY held position to market (MTM-D6): a deadzone cycle (zero
            # orders) fetched no prices in Step 9, so without this the snapshot kept
            # copying forward the last stored mark and its value froze while the
            # market moved (live defect: frozen since 07-19). Fetch a fresh price for
            # each held symbol missing from cycle_prices, fail-open per symbol — a
            # fetch failure or non-positive price warns and falls back to the stored
            # last_price (via compute_portfolio_value), never blocks the cycle.
            for pos in self._position_tracker.get_positions(env_name):
                sym = pos["symbol"]
                if pos["quantity"] and sym not in cycle_prices:
                    try:
                        price = adapter.get_current_price(sym)
                    except Exception:
                        log.warning("mark_price_fetch_failed", symbol=sym, exc_info=True)
                        continue
                    if price is not None and price > 0:
                        cycle_prices[sym] = price
                    else:
                        log.warning("mark_price_non_positive", symbol=sym, price=price)
            self._position_tracker.mark_positions(env_name, cycle_prices)
            cash = self._position_tracker.compute_cash(env_name)
            new_portfolio_value = self._position_tracker.compute_portfolio_value(
                env_name, cycle_prices
            )
            daily_pnl = self._position_tracker.compute_daily_pnl(env_name, new_portfolio_value)
            self._position_tracker.record_snapshot(env_name, new_portfolio_value, cash, daily_pnl)

        log.info(
            "cycle_complete",
            env=env_name,
            fills=len(fills),
            dry_run=dry_run,
            cycle_id=cycle_id,
        )
        return fills

    def _load_models(self, env_name: str) -> dict[str, tuple[Any, Any]]:
        """Load trained models for the environment (lazy, mtime-keyed cache).

        The cache is keyed by the artifact mtimes (review H3): a changed model.zip
        (hot swap) or a previously-empty load (no models on disk yet) reloads on the
        next cycle rather than serving a stale or permanently-empty result.

        Fail-closed (review M7 / A22): an algo whose ``vec_normalize.pkl`` is missing
        or fails to load is SKIPPED and a Discord alert is sent — never fed raw,
        un-normalized observations. The ensemble renormalizes over the algos that
        did load.

        Args:
            env_name: Environment name.

        Returns:
            Dict mapping algo name to (model, vec_normalize) tuple. May be empty
            (never cached as a permanent empty — retried next cycle).
        """
        cache_key = self._model_cache_key(env_name)
        # A non-empty key that matches the cached key is a hit. An empty key means
        # no models on disk — never treated as a cache hit, so it retries next cycle.
        if (
            cache_key
            and self._model_cache_keys.get(env_name) == cache_key
            and env_name in self._models
        ):
            return self._models[env_name]

        models: dict[str, tuple[Any, Any]] = {}

        for algo_name in _ALGO_NAMES:
            model_path, vec_path = active_model_paths(self._models_dir, env_name, algo_name)

            if not model_path.exists():
                log.warning(
                    "model_not_found",
                    env=env_name,
                    algo=algo_name,
                    path=str(model_path),
                )
                continue

            # Fail closed: no VecNormalize stats means we would feed raw observations
            # (out-of-distribution) to a model trained on normalized ones (M7).
            if not vec_path.exists():
                log.error(
                    "vec_normalize_missing",
                    env=env_name,
                    algo=algo_name,
                    path=str(vec_path),
                )
                self._alert_model_skipped(env_name, algo_name, "VecNormalize stats file missing")
                continue

            try:
                model, vec_norm = self._load_one_algo(algo_name, model_path, vec_path)
            except Exception:
                log.error(
                    "model_load_failed",
                    env=env_name,
                    algo=algo_name,
                    path=str(model_path),
                    exc_info=True,
                )
                self._alert_model_skipped(
                    env_name, algo_name, "model or VecNormalize failed to load"
                )
                continue

            models[algo_name] = (model, vec_norm)
            log.info("model_loaded", env=env_name, algo=algo_name, has_vec_normalize=True)

        self._models[env_name] = models
        self._model_cache_keys[env_name] = cache_key
        return models

    def _model_cache_key(self, env_name: str) -> tuple[Any, ...]:
        """Return a cache key from the on-disk artifact mtimes for the environment.

        The key changes whenever a model.zip or vec_normalize.pkl is added, removed,
        or rewritten, so the loader reloads. An empty tuple means no models on disk.

        Args:
            env_name: Environment name.

        Returns:
            Tuple of ``(algo, model_mtime, vec_mtime_or_None)`` per present model.
        """
        key: list[tuple[str, float, float | None]] = []
        for algo_name in _ALGO_NAMES:
            model_path, vec_path = active_model_paths(self._models_dir, env_name, algo_name)
            if not model_path.exists():
                continue
            vec_mtime = vec_path.stat().st_mtime if vec_path.exists() else None
            key.append((algo_name, model_path.stat().st_mtime, vec_mtime))
        return tuple(key)

    def _load_one_algo(self, algo_name: str, model_path: Path, vec_path: Path) -> tuple[Any, Any]:
        """Load a single algo's SB3 model and its VecNormalize stats.

        Args:
            algo_name: Algorithm name ("ppo", "a2c", or "sac").
            model_path: Path to the model.zip file.
            vec_path: Path to the vec_normalize.pkl file.

        Returns:
            Tuple of (loaded model, loaded VecNormalize with training disabled).
        """
        from stable_baselines3 import A2C, PPO, SAC  # noqa: PLC0415
        from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize  # noqa: PLC0415

        algo_map = {"ppo": PPO, "a2c": A2C, "sac": SAC}
        model = algo_map[algo_name].load(str(model_path))  # type: ignore[attr-defined]

        # Build a minimal stub env with correct spaces from the loaded model.
        obs_space = model.observation_space
        act_space = model.action_space

        def _make_stub_env(_obs: Any = obs_space, _act: Any = act_space) -> Any:
            """Return a minimal gymnasium env stub for VecNormalize loading."""
            import gymnasium  # noqa: PLC0415

            env: Any = gymnasium.Env()  # type: ignore[abstract]
            env.observation_space = _obs
            env.action_space = _act
            return env

        dummy_env = DummyVecEnv([_make_stub_env])
        vec_norm = VecNormalize.load(str(vec_path), venv=dummy_env)
        vec_norm.training = False
        vec_norm.norm_reward = False
        return model, vec_norm

    def _alert_model_skipped(self, env_name: str, algo_name: str, reason: str) -> None:
        """Send a fail-closed warning that an algo was excluded from the ensemble.

        Args:
            env_name: Environment name.
            algo_name: Algorithm that was skipped.
            reason: Human-readable reason for the skip.
        """
        if self._alerter is None:
            return
        env_literal: Literal["equity", "crypto"] = "equity" if env_name == "equity" else "crypto"
        self._alerter.send_alert(
            level="warning",
            title="Model Skipped (Fail-Closed)",
            message=(
                f"{env_name}/{algo_name} excluded from the ensemble: {reason}. "
                "Blending renormalizes over the remaining algos."
            ),
            environment=env_literal,
        )

    def _min_order_value(self, env_name: str) -> float:
        """Return the minimum order dollar value for the environment (config-driven).

        Args:
            env_name: Environment name.

        Returns:
            ``config.crypto.min_order_usd`` for crypto, else ``config.equity.min_order_usd``.
        """
        if env_name == "crypto":
            return self._config.crypto.min_order_usd
        return self._config.equity.min_order_usd

    def _get_ensemble_weights(self, env_name: str) -> dict[str, float]:
        """Query model_metadata table for ensemble weights.

        Args:
            env_name: Environment name.

        Returns:
            Dict mapping algo name to weight. Defaults to equal weights.
        """
        from swingrl.training.ensemble import DEFAULT_ENSEMBLE_WEIGHT  # noqa: PLC0415

        try:
            with self._db.connection() as conn:
                rows = conn.execute(
                    "SELECT algorithm, ensemble_weight FROM model_metadata "
                    "WHERE environment = %s ORDER BY training_end_date DESC",
                    [env_name],
                ).fetchall()

            if rows:
                weights: dict[str, float] = {}
                seen: set[str] = set()
                for row in rows:
                    algo = str(row["algorithm"])
                    if algo not in seen:
                        weights[algo] = (
                            float(row["ensemble_weight"])
                            if row["ensemble_weight"] is not None
                            else DEFAULT_ENSEMBLE_WEIGHT
                        )
                        seen.add(algo)
                if weights:
                    return weights
        except Exception:
            log.warning("ensemble_weights_query_failed", env=env_name)

        # Default: equal weights
        return dict.fromkeys(_ALGO_NAMES, DEFAULT_ENSEMBLE_WEIGHT)

    def _get_adapter(self, env_name: str) -> Any:
        """Get the exchange adapter for the environment (cached).

        Args:
            env_name: Environment name.

        Returns:
            ExchangeAdapter instance.
        """
        if env_name not in self._adapters:
            if env_name == "equity":
                from swingrl.execution.adapters.alpaca_adapter import AlpacaAdapter

                self._adapters[env_name] = AlpacaAdapter(config=self._config, alerter=self._alerter)
            else:
                from swingrl.execution.adapters.binance_sim import BinanceSimAdapter

                self._adapters[env_name] = BinanceSimAdapter(
                    config=self._config, db=self._db, alerter=self._alerter
                )
        return self._adapters[env_name]

    def _normalize_observation(
        self, env_name: str, observation: np.ndarray
    ) -> dict[str, np.ndarray]:
        """Return per-algo normalized observations via each algo's VecNormalize.

        Each algorithm trains with its own VecNormalize statistics. Using PPO's
        normalization for A2C/SAC produces out-of-distribution inputs. This method
        returns a dict so each algo receives correctly normalized observations.

        Args:
            env_name: Environment name.
            observation: Raw observation array.

        Returns:
            Dict mapping algo name to its normalized observation array.
        """
        result: dict[str, np.ndarray] = {}

        if env_name in self._models:
            for algo_name, (_, vec_norm) in self._models[env_name].items():
                if vec_norm is not None:
                    try:
                        result[algo_name] = vec_norm.normalize_obs(observation)
                    except Exception:
                        log.warning("vec_normalize_failed", env=env_name, algo=algo_name)
                        result[algo_name] = observation
                else:
                    result[algo_name] = observation

        return result

    def _check_turbulence(self, env_name: str, date_str: str) -> bool:
        """Check turbulence for crash protection (PAPER-20).

        Args:
            env_name: Environment name.
            date_str: Cycle date/datetime string — shared with the observation
                path so both use the single per-cycle turbulence value (M6).

        Returns:
            True if turbulence exceeds the hard-halt baseline (should halt trading).
        """
        try:
            # Single per-cycle turbulence value (reused by the observation path)
            turbulence = self._feature_pipeline.compute_turbulence(env_name, date_str)

            # Historical hard-halt baseline percentile (F1 fix — OHLCV series)
            historical_pct = self._get_turbulence_90th_pct(env_name, date_str)

            if turbulence > 0 and historical_pct > 0:
                return self._risk_manager.check_turbulence(env_name, turbulence, historical_pct)
        except Exception:
            log.warning("turbulence_check_failed", env=env_name)

        return False

    def _get_turbulence_90th_pct(self, env_name: str, date_str: str) -> float:
        """Get the historical turbulence hard-halt baseline for the cycle date.

        Delegates to ``FeaturePipeline.turbulence_halt_baseline`` (the F1 fix —
        the ``features_*`` tables never had a turbulence column, so the old
        percentile query always returned 0.0 and the halt never fired). Cached
        per (env, date) so the delegate runs at most once per cycle.

        Args:
            env_name: Environment name.
            date_str: Cycle date/datetime string.

        Returns:
            The hard-halt baseline percentile, or 0.0 if unavailable (logged).
        """
        key = (env_name, date_str)
        if key in self._turb_baseline_cache:
            return self._turb_baseline_cache[key]
        try:
            value = self._feature_pipeline.turbulence_halt_baseline(env_name, date_str)
        except DataError as exc:
            log.error("turbulence_baseline_failed", env=env_name, error=str(exc))
            value = 0.0
        except Exception as exc:  # never let a baseline lookup crash the cycle — but log it
            log.error("turbulence_baseline_unexpected_error", env=env_name, error=str(exc))
            value = 0.0
        self._turb_baseline_cache[key] = value
        return value

    def _get_current_weights(self, env_name: str) -> np.ndarray:
        """Get current portfolio weights per symbol.

        Args:
            env_name: Environment name.

        Returns:
            Array of current weights per symbol.
        """
        symbols = (
            self._config.equity.symbols if env_name == "equity" else self._config.crypto.symbols
        )
        portfolio_value = self._position_tracker.get_portfolio_value(env_name)
        positions = self._position_tracker.get_positions(env_name)
        pos_by_symbol = {p["symbol"]: p for p in positions}

        weights = np.zeros(len(symbols), dtype=np.float32)
        for i, symbol in enumerate(symbols):
            pos = pos_by_symbol.get(symbol)
            if pos is not None and portfolio_value > 0:
                weights[i] = abs(pos["quantity"] * (pos["last_price"] or 0.0)) / portfolio_value

        return weights

    def _equity_market_open(self) -> bool:
        """Return True if the equity market is open per the Alpaca clock (fail-safe).

        Market closed/holiday -> False + info log. Clock unreachable (any error, incl.
        adapter construction) -> False + critical alert: when in doubt, don't trade.

        Returns:
            True only when the clock explicitly reports the market open.
        """
        try:
            adapter = self._get_adapter("equity")
            clock = adapter.get_clock()
        except Exception:
            log.error("market_clock_check_failed", env="equity", exc_info=True)
            if self._alerter is not None:
                self._alerter.send_alert(
                    level="critical",
                    title="Market Clock Unreachable",
                    message="Alpaca clock check failed — skipping equity cycle (fail-safe).",
                    environment="equity",
                )
            return False

        is_open = bool(getattr(clock, "is_open", False))
        if not is_open:
            log.info("equity_cycle_skipped_market_closed")
        return is_open

    def _warn_if_stale_ohlcv(self, env_name: str) -> None:
        """Log a warning when the latest ``ohlcv_daily`` bar is older than the prior session.

        Log-only data-freshness guard (never raises, never halts the cycle): a stale
        latest bar at decision time means the cycle would trade on old prices.

        Args:
            env_name: Environment name (only "equity" has a daily-bar table here).
        """
        try:
            symbols = self._config.equity.symbols
            with self._db.connection() as conn:
                row = conn.execute(
                    "SELECT MAX(date) AS latest FROM ohlcv_daily WHERE symbol = ANY(%s)",
                    (symbols,),
                ).fetchone()
            latest = row["latest"] if row is not None else None
            if latest is None:
                log.warning("ohlcv_freshness_no_data", env=env_name)
                return

            import exchange_calendars  # noqa: PLC0415

            nyse = exchange_calendars.get_calendar("XNYS")
            today_et = datetime.now(tz=UTC).astimezone(ZoneInfo("America/New_York")).date()
            sessions = nyse.sessions_in_range(
                today_et - timedelta(days=10), today_et - timedelta(days=1)
            )
            if len(sessions) == 0:
                return
            prev_session_date = sessions[-1].date()

            latest_date = (
                latest if isinstance(latest, date) else datetime.fromisoformat(str(latest)).date()
            )
            if latest_date < prev_session_date:
                log.warning(
                    "ohlcv_stale_bar",
                    env=env_name,
                    latest_bar=latest_date.isoformat(),
                    expected_min=prev_session_date.isoformat(),
                )
        except Exception:
            log.warning("ohlcv_freshness_check_failed", exc_info=True)

    def _get_current_date_str(self, env_name: str, now: datetime | None = None) -> str:
        """Get current date string for the environment.

        Args:
            env_name: Environment name.
            now: Timestamp to derive the string from. Defaults to ``datetime.now(UTC)``
                so direct callers stay unchanged; ``execute_cycle`` passes the single
                canonical ``cycle_ts`` (review fold a).

        Returns:
            Date string (YYYY-MM-DD for equity, ISO datetime for crypto).
        """
        now = now or datetime.now(tz=UTC)
        if env_name == "equity":
            return now.strftime("%Y-%m-%d")
        return now.isoformat()

    def _capture_cycle(
        self,
        *,
        env_name: str,
        cycle_ts: datetime,
        current_date_str: str,
        turbulence_at_decision: float,
        symbols: list[str],
        actions: dict[str, np.ndarray],
        ensemble_weights: dict[str, float],
        target_weights: np.ndarray,
        dry_run: bool,
        skip_reasons: dict[str, str],
    ) -> int | None:
        """Assemble the RegimeStamp + proposals and hand them to the CycleRecorder.

        Fail-open backstop around the assembly (the recorder's own writes are
        already fail-open): any error here is logged and swallowed so capture never
        blocks the money path.

        Args:
            env_name: Environment name.
            cycle_ts: The canonical cycle timestamp.
            current_date_str: Cycle date/datetime string (regime lookup cutoff).
            turbulence_at_decision: Real sensor value read before F1b zeroing.
            symbols: Ordered env symbols (per-symbol action/weight keys).
            actions: Per-algo raw action vectors.
            ensemble_weights: Ensemble weights keyed by algo (may be partial).
            target_weights: Blended per-symbol target weights (array over symbols).
            dry_run: Whether this cycle was a dry-run (tagged in the payload, fold d).
            skip_reasons: Per-symbol M2 skip reasons for the payload (fold c).

        Returns:
            The new ``cycle_id``, or ``None`` on any capture failure.
        """
        try:
            regime_dict = self._feature_pipeline.regime_snapshot(env_name, current_date_str)
            regime = RegimeStamp(
                hmm_p_bull=regime_dict.get("hmm_p_bull"),
                hmm_p_bear=regime_dict.get("hmm_p_bear"),
                vix=regime_dict.get("vix"),
                turbulence=turbulence_at_decision,
                active_event_ids=self._cycle_recorder.active_event_ids(cycle_ts),
            )
            active_ids = self._cycle_recorder.active_model_ids(env_name)
            proposals = self._build_proposals(symbols, actions, ensemble_weights, active_ids)
            raw_actions = {algo: [float(x) for x in vec] for algo, vec in actions.items()}
            target_weights_map = {symbols[i]: float(target_weights[i]) for i in range(len(symbols))}
            return self._cycle_recorder.record_cycle(
                env_name=env_name,
                mode=self._config.trading_mode,
                cycle_ts=cycle_ts,
                regime=regime,
                raw_actions=raw_actions,
                target_weights=target_weights_map,
                proposals=proposals,
                deployed_iteration=self._cycle_recorder.deployed_iteration(env_name),
                dry_run=dry_run,
                skip_reasons=skip_reasons,
            )
        except Exception:
            log.warning("cycle_capture_failed", env=env_name, exc_info=True)
            return None

    def _record_halt(
        self,
        env_name: str,
        cycle_ts: datetime,
        reason: str,
        dry_run: bool,
        turbulence: float | None = None,
    ) -> None:
        """Fail-open early-exit capture (fold b): record a halt row, never affect the exit.

        Wrapped so a capture failure can never change the halted cycle's behavior or
        return value. The CycleRecorder's write is itself fail-open; this is a backstop.

        Args:
            env_name: Environment name.
            cycle_ts: The canonical cycle timestamp.
            reason: Short halt-reason code.
            dry_run: Whether the halted cycle was a dry-run.
            turbulence: Decision-time turbulence when known at the exit, else None.
        """
        try:
            self._cycle_recorder.record_halt(
                env_name=env_name,
                mode=self._config.trading_mode,
                cycle_ts=cycle_ts,
                reason=reason,
                turbulence=turbulence,
                dry_run=dry_run,
            )
        except Exception:
            log.warning("cycle_capture_failed", env=env_name, reason=reason, exc_info=True)

    @staticmethod
    def _classify_skips(
        symbols: list[str],
        target_weights: np.ndarray,
        current_weights: np.ndarray,
        portfolio_value: float,
        min_order_value: float,
    ) -> dict[str, str]:
        """Classify per-symbol M2 (sub-minimum-delta) skips for the capture payload (fold c).

        Mirrors the Step 9 order-gen filter (``abs(delta_value) < min_order_value``). A
        symbol the model signaled to change (``target != current``) but whose dollar
        delta is below the minimum order value is tagged ``below_min_delta`` — the
        "model signaled, order too small" case, distinguishable from "model held"
        (``target == current``), which is intentionally absent from the map.

        Args:
            symbols: Ordered env symbols.
            target_weights: Post-process target weights (array over symbols).
            current_weights: Current portfolio weights (array over symbols).
            portfolio_value: Portfolio value used to size the dollar delta.
            min_order_value: Minimum order dollar value (the M2 threshold).

        Returns:
            ``{symbol: "below_min_delta"}`` for signaled-but-too-small symbols only.
        """
        reasons: dict[str, str] = {}
        for i, symbol in enumerate(symbols):
            target = float(target_weights[i])
            current = float(current_weights[i])
            delta_value = (target - current) * portfolio_value
            if target != current and abs(delta_value) < min_order_value:
                reasons[symbol] = "below_min_delta"
        return reasons

    def _build_proposals(
        self,
        symbols: list[str],
        actions: dict[str, np.ndarray],
        ensemble_weights: dict[str, float],
        active_ids: dict[str, str],
    ) -> list[AlgoProposal]:
        """Build one AlgoProposal per algo, snapshotting its blend weight (D-T3.13).

        The blend fraction renormalizes the ensemble weights over the algos that
        actually produced actions this cycle — mirroring ``EnsembleBlender`` so the
        snapshot matches the weight each algo really carried. An algo with no active
        DB model row is skipped (a proposal row would violate the ``model_id`` FK;
        the cycle row still records).

        Args:
            symbols: Ordered env symbols.
            actions: Per-algo raw action vectors (length n_assets + 1, cash last).
            ensemble_weights: Ensemble weights keyed by algo (may be partial).
            active_ids: Active ``{algo: model_id}`` map.

        Returns:
            AlgoProposal list (may be empty when no active models are registered).
        """
        from swingrl.training.ensemble import DEFAULT_ENSEMBLE_WEIGHT  # noqa: PLC0415

        loaded = list(actions.keys())
        raw_w = {a: float(ensemble_weights.get(a, DEFAULT_ENSEMBLE_WEIGHT)) for a in loaded}
        total = sum(raw_w.values()) or 1.0

        proposals: list[AlgoProposal] = []
        for algo in loaded:
            model_id = active_ids.get(algo)
            if model_id is None:
                continue
            vec = actions[algo]
            per_symbol = {symbols[i]: float(vec[i]) for i in range(len(symbols))}
            proposals.append(
                AlgoProposal(
                    algorithm=algo,
                    model_id=model_id,
                    raw_actions=per_symbol,
                    weight_in_blend_frac=raw_w[algo] / total,
                )
            )
        return proposals
