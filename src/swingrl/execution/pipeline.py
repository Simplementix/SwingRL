"""ExecutionPipeline: orchestrates the 5-stage trading middleware.

Wires together model inference, ensemble blending, signal interpretation,
position sizing, order validation, broker submission, and fill processing
into a single execute_cycle() call.

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

from swingrl.execution.fill_processor import FillProcessor
from swingrl.execution.order_validator import OrderValidator
from swingrl.execution.position_sizer import PositionSizer
from swingrl.execution.risk.circuit_breaker import CBState, CircuitBreaker, GlobalCircuitBreaker
from swingrl.execution.risk.position_tracker import PositionTracker
from swingrl.execution.risk.risk_manager import RiskManager
from swingrl.execution.signal_interpreter import SignalInterpreter
from swingrl.execution.types import FillResult
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
    """Orchestrates the full 5-stage execution middleware.

    Stages:
        1. SignalInterpreter: blended actions -> TradeSignals
        2. PositionSizer: TradeSignal -> SizedOrder (Kelly + ATR stops)
        3. OrderValidator: SizedOrder -> ValidatedOrder (cost gate + risk)
        4. ExchangeAdapter: ValidatedOrder -> FillResult (broker submission)
        5. FillProcessor: FillResult -> DB recording + position update
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

        # Lazy-initialized components
        self._models: dict[str, dict[str, tuple[Any, Any]]] = {}
        self._initialized = False

        # Eagerly create components that don't need lazy loading
        self._position_tracker = PositionTracker(db=db, config=config)
        self._fill_processor = FillProcessor(db=db)
        self._signal_interpreter = SignalInterpreter(config=config)
        self._position_sizer = PositionSizer(config=config)

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

    def execute_cycle(
        self,
        env_name: str,
        dry_run: bool = False,
    ) -> list[FillResult]:
        """Run a full trading cycle for the given environment.

        Args:
            env_name: Environment name ("equity" or "crypto").
            dry_run: If True, skip broker submission (stages 4-5).

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

        # Step 3b: Track NaN observations in inference_outcomes table
        had_nan = bool(np.isnan(observation).any())
        try:
            with self._db.sqlite() as conn:
                conn.execute(
                    "INSERT INTO inference_outcomes (timestamp, environment, had_nan) "
                    "VALUES (?, ?, ?)",
                    (datetime.now(UTC).isoformat(), env_name, int(had_nan)),
                )
        except Exception:
            log.warning("inference_outcome_tracking_failed", exc_info=True)

        if had_nan:
            log.warning("nan_observation_detected", env=env_name)
            return []

        # Step 4: Get portfolio state (used by ObservationAssembler in future)
        _portfolio_state = self._position_tracker.get_portfolio_state_array(env_name)

        # Step 5: Normalize observation via VecNormalize
        obs = self._normalize_observation(env_name, observation)

        # Step 6: Load models and run predict
        models = self._load_models(env_name)
        actions: dict[str, np.ndarray] = {}
        for algo_name, (model, _vec_norm) in models.items():
            action, _ = model.predict(obs, deterministic=True)
            actions[algo_name] = action
            log.debug("model_predicted", env=env_name, algo=algo_name)

        # Step 7: Load ensemble weights and blend actions
        weights = self._get_ensemble_weights(env_name)

        from swingrl.training.ensemble import EnsembleBlender

        blender = EnsembleBlender(self._config)
        blended_actions = blender.blend_actions(actions, weights)

        # Step 8: Get current asset weights
        current_weights = self._get_current_weights(env_name)

        # Step 9: Stage 1 - Signal interpretation
        signals = self._signal_interpreter.interpret(env_name, blended_actions, current_weights)

        log.info(
            "signals_generated",
            env=env_name,
            signal_count=len(signals),
        )

        # Step 10: Process each signal through stages 2-5
        fills: list[FillResult] = []
        adapter = self._get_adapter(env_name)
        portfolio_value = self._position_tracker.get_portfolio_value(env_name)

        for signal in signals:
            if signal.action == "hold":
                continue

            try:
                # Get current price
                current_price = adapter.get_current_price(signal.symbol)

                # Get ATR value
                atr_value = self._get_latest_atr(env_name, signal.symbol)

                # Stage 2: Position sizing
                sized_order = self._position_sizer.size(
                    signal, current_price, atr_value, portfolio_value
                )
                if sized_order is None:
                    log.info("signal_skipped_kelly", symbol=signal.symbol)
                    continue

                # Stage 3: Order validation
                validated_order = self._order_validator.validate(sized_order)

                # Dry-run: log but skip stages 4-5
                if dry_run:
                    log.info(
                        "dry_run_would_submit",
                        symbol=sized_order.symbol,
                        side=sized_order.side,
                        dollar_amount=sized_order.dollar_amount,
                        quantity=sized_order.quantity,
                        stop_loss=sized_order.stop_loss_price,
                        take_profit=sized_order.take_profit_price,
                    )
                    continue

                # Stage 4: Broker submission
                fill = adapter.submit_order(validated_order)

                # Stage 5: Fill processing
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
                    "signal_vetoed",
                    symbol=signal.symbol,
                    reason=str(exc),
                )
                continue

            except CircuitBreakerError as exc:
                log.error(
                    "circuit_breaker_during_cycle",
                    symbol=signal.symbol,
                    reason=str(exc),
                )
                break

            except Exception:
                log.exception(
                    "signal_processing_failed",
                    symbol=signal.symbol,
                )
                continue

        # Step 11: Record portfolio snapshot
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
                # Create a dummy env for VecNormalize loading
                dummy_env = DummyVecEnv([lambda: None])  # type: ignore[arg-type,return-value,list-item]
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
            with self._db.duckdb() as conn:
                rows = conn.execute(
                    "SELECT algorithm, ensemble_weight FROM model_metadata "
                    "WHERE environment = ? ORDER BY training_end_date DESC",
                    [env_name],
                ).fetchall()

            if rows:
                weights: dict[str, float] = {}
                seen: set[str] = set()
                for row in rows:
                    algo = str(row[0])
                    if algo not in seen:
                        weights[algo] = float(row[1]) if row[1] is not None else 1.0 / 3
                        seen.add(algo)
                if weights:
                    return weights
        except Exception:
            log.warning("ensemble_weights_query_failed", env=env_name)

        # Default: equal weights
        return dict.fromkeys(_ALGO_NAMES, 1.0 / 3)

    def _get_adapter(self, env_name: str) -> Any:
        """Get the exchange adapter for the environment.

        Args:
            env_name: Environment name.

        Returns:
            ExchangeAdapter instance.
        """
        if env_name == "equity":
            from swingrl.execution.adapters.alpaca_adapter import AlpacaAdapter

            return AlpacaAdapter(config=self._config, alerter=self._alerter)

        from swingrl.execution.adapters.binance_sim import BinanceSimAdapter

        return BinanceSimAdapter(config=self._config, db=self._db, alerter=self._alerter)

    def _normalize_observation(self, env_name: str, observation: np.ndarray) -> np.ndarray:
        """Normalize observation via VecNormalize if available.

        Args:
            env_name: Environment name.
            observation: Raw observation array.

        Returns:
            Normalized observation array.
        """
        # If models are loaded and have vec_normalize, use first available
        if env_name in self._models:
            for _algo, (_, vec_norm) in self._models[env_name].items():
                if vec_norm is not None:
                    try:
                        return vec_norm.normalize_obs(observation)  # type: ignore[no-any-return]
                    except Exception:
                        log.warning("vec_normalize_failed", env=env_name)
                        break

        return observation

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
            if env_name == "equity":
                turbulence = self._feature_pipeline._compute_turbulence_equity(current_date_str)
            else:
                turbulence = self._feature_pipeline._compute_turbulence_crypto(current_date_str)

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
            with self._db.duckdb() as conn:
                row = conn.execute(
                    f"SELECT QUANTILE(atr_14_pct, 0.9) as p90 FROM {table}",  # nosec B608
                ).fetchone()
                if row and row[0] is not None:
                    return float(row[0])
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

    def _get_latest_atr(self, env_name: str, symbol: str) -> float:
        """Get latest ATR value for a symbol from DuckDB features.

        Args:
            env_name: Environment name.
            symbol: Ticker symbol.

        Returns:
            ATR value, or a default fallback.
        """
        table = "features_equity" if env_name == "equity" else "features_crypto"
        date_col = "date" if env_name == "equity" else "datetime"
        try:
            with self._db.duckdb() as conn:
                row = conn.execute(
                    f"SELECT atr_14_pct FROM {table} "  # nosec B608
                    f"WHERE symbol = ? ORDER BY {date_col} DESC LIMIT 1",  # nosec B608
                    [symbol],
                ).fetchone()
                if row and row[0] is not None:
                    return float(row[0])
        except Exception:
            log.warning("atr_query_failed", env=env_name, symbol=symbol)

        # Default fallback: 2% of price as ATR estimate
        return 0.02

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
