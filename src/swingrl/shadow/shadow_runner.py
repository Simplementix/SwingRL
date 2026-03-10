"""Shadow inference runner for parallel model evaluation.

Runs shadow model inference after active trading cycle, recording hypothetical
trades in the shadow_trades SQLite table. Failures are logged but never raised
to protect the active trading cycle.

Usage:
    from swingrl.shadow.shadow_runner import run_shadow_inference
    run_shadow_inference(ctx, "equity")
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import structlog

if TYPE_CHECKING:
    pass

log = structlog.get_logger(__name__)


def run_shadow_inference(ctx: Any, env_name: str) -> None:
    """Run shadow model inference and record hypothetical trades.

    If no shadow model exists for the environment, returns silently (no-op).
    All exceptions are caught and logged -- this function never raises.

    Args:
        ctx: JobContext with config, db, pipeline, alerter.
        env_name: Environment name (equity or crypto).
    """
    try:
        models_dir = Path(ctx.config.paths.models_dir)
        shadow_dir = models_dir / "shadow" / env_name

        # No shadow model => no-op
        shadow_models = list(shadow_dir.glob("*.zip")) if shadow_dir.exists() else []
        if not shadow_models:
            log.info("shadow_no_model", env=env_name)
            return

        shadow_model_path = shadow_models[0]
        model_version = shadow_model_path.stem

        # Load shadow model
        model = _load_shadow_model(shadow_model_path)

        # Generate hypothetical trades
        trades = _generate_hypothetical_trades(model, ctx, env_name, model_version)

        # Record trades in shadow_trades table
        if trades:
            _record_shadow_trades(ctx.db, trades)

        log.info(
            "shadow_inference_complete",
            env=env_name,
            model=model_version,
            trade_count=len(trades),
        )

    except Exception:
        log.exception("shadow_inference_error", env=env_name)


def _load_shadow_model(model_path: Path) -> Any:
    """Load a shadow SB3 model from file.

    Uses the same loading pattern as lifecycle.py.

    Args:
        model_path: Path to the .zip model file.

    Returns:
        Loaded SB3 model instance.
    """
    from swingrl.shadow.lifecycle import _load_sb3_model  # noqa: PLC0415

    return _load_sb3_model(model_path)


def _generate_hypothetical_trades(
    model: Any,
    ctx: Any,
    env_name: str,
    model_version: str,
) -> list[dict[str, Any]]:
    """Generate hypothetical trades from shadow model predictions.

    Runs the shadow model predict() to get actions, then interprets and
    sizes them as hypothetical trades. Does NOT submit to any exchange adapter.

    Args:
        model: Loaded SB3 model.
        ctx: JobContext with pipeline (ExecutionPipeline) and config.
        env_name: Environment name ("equity" or "crypto").
        model_version: Model version string from filename.

    Returns:
        List of trade dicts ready for shadow_trades insertion.
    """
    try:
        # Lazy imports to avoid circular dependency risk
        import numpy as np  # noqa: PLC0415

        from swingrl.execution.position_sizer import PositionSizer  # noqa: PLC0415
        from swingrl.execution.signal_interpreter import SignalInterpreter  # noqa: PLC0415

        config = ctx.pipeline._config
        env_literal: Literal["equity", "crypto"] = "equity" if env_name == "equity" else "crypto"

        # Step 1: Current date string (same logic as ExecutionPipeline)
        now = datetime.now(tz=UTC)
        if env_name == "equity":
            current_date_str = now.strftime("%Y-%m-%d")
        else:
            current_date_str = now.isoformat()

        log.info(
            "shadow_trade_gen_start",
            env=env_name,
            model_version=model_version,
            date=current_date_str,
        )

        # Step 2: Get observation from feature pipeline
        observation = ctx.pipeline._feature_pipeline.get_observation(env_literal, current_date_str)
        log.debug("shadow_observation_obtained", env=env_name, shape=observation.shape)

        # Step 3: Normalize observation if VecNormalize exists
        obs = _maybe_normalize_obs(model, observation)

        # Step 4: Model predict (single shadow model, no ensemble blending)
        actions, _ = model.predict(obs, deterministic=True)
        log.debug("shadow_model_predicted", env=env_name, actions_shape=actions.shape)

        # Step 5: Current weights -- shadow has no real positions, use zeros
        symbols = config.equity.symbols if env_name == "equity" else config.crypto.symbols
        current_weights = np.zeros(len(symbols), dtype=np.float32)

        # Step 6: Interpret actions into trade signals
        interpreter = SignalInterpreter(config)
        signals = interpreter.interpret(env_name, actions, current_weights)
        log.info("shadow_signals_interpreted", env=env_name, signal_count=len(signals))

        if not signals:
            return []

        # Step 7: Size each signal
        sizer = PositionSizer(config)
        portfolio_value = (
            config.capital.equity_usd if env_name == "equity" else config.capital.crypto_usd
        )
        default_atr = 0.02  # conservative fallback

        trades: list[dict[str, Any]] = []
        for signal in signals:
            if signal.action == "hold":
                continue

            # Use a reasonable estimated price fallback
            estimated_price = 100.0 if env_name == "equity" else 50000.0

            sized_order = sizer.size(signal, estimated_price, default_atr, portfolio_value)
            if sized_order is None:
                log.debug(
                    "shadow_signal_skipped",
                    symbol=signal.symbol,
                    reason="sizer_returned_none",
                )
                continue

            # Step 8: Build trade dict matching shadow_trades schema
            price = (
                sized_order.dollar_amount / sized_order.quantity
                if sized_order.quantity > 0
                else estimated_price
            )
            trade: dict[str, Any] = {
                "trade_id": str(uuid.uuid4()),
                "timestamp": datetime.now(tz=UTC).isoformat(),
                "symbol": signal.symbol,
                "side": sized_order.side,
                "quantity": sized_order.quantity,
                "price": price,
                "commission": 0.0,
                "slippage": 0.0,
                "environment": env_literal,
                "broker": None,
                "order_type": "market",
                "trade_type": "shadow",
                "model_version": model_version,
            }
            trades.append(trade)
            log.info(
                "shadow_trade_generated",
                symbol=signal.symbol,
                side=sized_order.side,
                quantity=sized_order.quantity,
                dollar_amount=sized_order.dollar_amount,
            )

        return trades

    except Exception:
        log.exception("shadow_trade_gen_error", env=env_name)
        return []


def _maybe_normalize_obs(model: Any, observation: Any) -> Any:
    """Normalize observation via VecNormalize if available alongside model.

    Args:
        model: Loaded SB3 model (may have get_vec_normalize_env).
        observation: Raw observation array.

    Returns:
        Normalized observation, or raw observation if no normalizer.
    """
    try:
        vec_env = getattr(model, "get_vec_normalize_env", None)
        if vec_env is not None:
            norm_env = vec_env()
            if norm_env is not None:
                return norm_env.normalize_obs(observation)  # type: ignore[no-any-return]
    except Exception:
        log.warning("shadow_vec_normalize_failed")

    return observation


def _record_shadow_trades(db: Any, trades: list[dict[str, Any]]) -> None:
    """Insert hypothetical trades into shadow_trades SQLite table.

    Args:
        db: DatabaseManager instance.
        trades: List of trade dicts with shadow_trades column keys.
    """
    with db.sqlite() as conn:
        for trade in trades:
            conn.execute(
                "INSERT INTO shadow_trades "
                "(trade_id, timestamp, symbol, side, quantity, price, "
                "commission, slippage, environment, broker, order_type, "
                "trade_type, model_version) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    trade.get("trade_id", str(uuid.uuid4())),
                    trade.get("timestamp", datetime.now(tz=UTC).isoformat()),
                    trade["symbol"],
                    trade["side"],
                    trade["quantity"],
                    trade["price"],
                    trade.get("commission", 0.0),
                    trade.get("slippage", 0.0),
                    trade["environment"],
                    trade.get("broker"),
                    trade.get("order_type"),
                    trade.get("trade_type"),
                    trade["model_version"],
                ),
            )

    log.info("shadow_trades_recorded", count=len(trades))
