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

from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import structlog

from swingrl.envs.portfolio import process_actions
from swingrl.execution.fill_processor import FillProcessor
from swingrl.execution.order_validator import OrderValidator
from swingrl.execution.risk.circuit_breaker import CBState, CircuitBreaker, GlobalCircuitBreaker
from swingrl.execution.risk.position_tracker import PositionTracker
from swingrl.execution.risk.risk_manager import RiskManager
from swingrl.execution.types import FillResult, SizedOrder
from swingrl.features.health import FeatureHealthTracker
from swingrl.utils.exceptions import CircuitBreakerError, RiskVetoError

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
            db: DatabaseManager for SQLite/DuckDB access.
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

        # Lazy-initialized components
        self._models: dict[str, dict[str, tuple[Any, Any]]] = {}
        self._adapters: dict[str, Any] = {}
        self._initialized = False

        # Eagerly create components that don't need lazy loading
        self._position_tracker = PositionTracker(db=db, config=config)
        self._fill_processor = FillProcessor(db=db)

        # Circuit breakers
        self._circuit_breakers: dict[str, CircuitBreaker] = {
            "equity": CircuitBreaker("equity", db, config),
            "crypto": CircuitBreaker("crypto", db, config),
        }
        self._global_cb = GlobalCircuitBreaker(self._circuit_breakers, config)

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

        # Step 1: Check circuit breaker state
        cb = self._circuit_breakers.get(env_name)
        if cb is not None:
            state = cb.get_state()
            if state == CBState.HALTED:
                log.warning("cycle_halted_by_cb", env=env_name)
                return []

        # Step 2: Check turbulence crash protection (PAPER-20)
        if self._check_turbulence(env_name):
            log.warning("cycle_halted_by_turbulence", env=env_name)
            return []

        # Step 3: Get observation from FeaturePipeline
        current_date_str = self._get_current_date_str(env_name)
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
            return []

        # Step 3c: Track NaN observations in inference_outcomes table
        had_nan = bool(np.isnan(observation).any())
        try:
            with self._db.connection() as conn:
                conn.execute(
                    "INSERT INTO inference_outcomes (timestamp, environment, had_nan) "
                    "VALUES (%s, %s, %s)",
                    (datetime.now(UTC).isoformat(), env_name, int(had_nan)),
                )
        except Exception:
            log.warning("inference_outcome_tracking_failed", exc_info=True)

        if had_nan:
            log.warning("nan_observation_detected", env=env_name)
            return []

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
        adapter = self._get_adapter(env_name)
        portfolio_value = self._position_tracker.get_portfolio_value(env_name)

        if portfolio_value <= 0:
            log.error("zero_portfolio_value", env=env_name, value=portfolio_value)
            return []

        # Minimum order value: use crypto min_order_usd or $5 for equity
        min_order_value = self._config.crypto.min_order_usd if env_name == "crypto" else 5.0

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

                # Risk validation (guardrail)
                validated_order = self._order_validator.validate(sized_order)

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

                # Fill processing
                self._fill_processor.process(fill, sized_order=sized_order)

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

        # Step 10: Record portfolio snapshot
        if fills:
            new_portfolio_value = self._position_tracker.get_portfolio_value(env_name)
            daily_pnl = self._position_tracker.get_daily_pnl(env_name)
            cash = new_portfolio_value - sum(abs(f.quantity * f.fill_price) for f in fills)
            self._position_tracker.record_snapshot(env_name, new_portfolio_value, cash, daily_pnl)

        log.info(
            "cycle_complete",
            env=env_name,
            fills=len(fills),
            dry_run=dry_run,
        )
        return fills

    def _load_models(self, env_name: str) -> dict[str, tuple[Any, Any]]:
        """Load trained models for the environment (lazy, cached).

        Args:
            env_name: Environment name.

        Returns:
            Dict mapping algo name to (model, vec_normalize) tuple.
        """
        if env_name in self._models:
            return self._models[env_name]

        from stable_baselines3 import A2C, PPO, SAC
        from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

        algo_map = {"ppo": PPO, "a2c": A2C, "sac": SAC}
        models: dict[str, tuple[Any, Any]] = {}

        for algo_name in _ALGO_NAMES:
            model_path = self._models_dir / "active" / env_name / algo_name / "model.zip"
            vec_path = self._models_dir / "active" / env_name / algo_name / "vec_normalize.pkl"

            if not model_path.exists():
                log.warning(
                    "model_not_found",
                    env=env_name,
                    algo=algo_name,
                    path=str(model_path),
                )
                continue

            algo_cls = algo_map[algo_name]
            model = algo_cls.load(str(model_path))  # type: ignore[attr-defined]

            vec_norm = None
            if vec_path.exists():
                # Build a minimal stub env with correct spaces from the loaded model
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

            models[algo_name] = (model, vec_norm)
            log.info(
                "model_loaded",
                env=env_name,
                algo=algo_name,
                has_vec_normalize=vec_norm is not None,
            )

        self._models[env_name] = models
        return models

    def _get_ensemble_weights(self, env_name: str) -> dict[str, float]:
        """Query DuckDB model_metadata table for ensemble weights.

        Args:
            env_name: Environment name.

        Returns:
            Dict mapping algo name to weight. Defaults to equal weights.
        """
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
                            else 1.0 / 3
                        )
                        seen.add(algo)
                if weights:
                    return weights
        except Exception:
            log.warning("ensemble_weights_query_failed", env=env_name)

        # Default: equal weights
        return dict.fromkeys(_ALGO_NAMES, 1.0 / 3)

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

    def _check_turbulence(self, env_name: str) -> bool:
        """Check turbulence for crash protection (PAPER-20).

        Args:
            env_name: Environment name.

        Returns:
            True if turbulence exceeds threshold (should halt trading).
        """
        try:
            # Get current turbulence from feature pipeline
            current_date_str = self._get_current_date_str(env_name)
            turbulence = self._feature_pipeline.compute_turbulence(env_name, current_date_str)

            # Get historical 90th percentile from DuckDB
            historical_90th = self._get_turbulence_90th_pct(env_name)

            if turbulence > 0 and historical_90th > 0:
                return self._risk_manager.check_turbulence(env_name, turbulence, historical_90th)
        except Exception:
            log.warning("turbulence_check_failed", env=env_name)

        return False

    def _get_turbulence_90th_pct(self, env_name: str) -> float:
        """Get historical 90th percentile turbulence from DuckDB.

        Args:
            env_name: Environment name.

        Returns:
            90th percentile turbulence value, or 0.0 if unavailable.
        """
        table = "features_equity" if env_name == "equity" else "features_crypto"
        try:
            with self._db.connection() as conn:
                row = conn.execute(
                    f"SELECT PERCENTILE_CONT(0.9) WITHIN GROUP (ORDER BY turbulence) as p90 FROM {table}",  # nosec B608
                ).fetchone()
                if row and row["p90"] is not None:
                    return float(row["p90"])
        except Exception:
            log.warning("turbulence_90th_query_failed", env=env_name)

        return 0.0

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

    def _get_current_date_str(self, env_name: str) -> str:
        """Get current date string for the environment.

        Args:
            env_name: Environment name.

        Returns:
            Date string (YYYY-MM-DD for equity, ISO datetime for crypto).
        """
        now = datetime.now(tz=UTC)
        if env_name == "equity":
            return now.strftime("%Y-%m-%d")
        return now.isoformat()
