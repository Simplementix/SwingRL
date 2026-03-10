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
from typing import TYPE_CHECKING, Any

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

    Runs the shadow model predict() to get actions, then interprets them
    as hypothetical trades. Does NOT submit to any exchange adapter.

    Args:
        model: Loaded SB3 model.
        ctx: JobContext.
        env_name: Environment name.
        model_version: Model version string from filename.

    Returns:
        List of trade dicts ready for shadow_trades insertion.
    """
    # Placeholder: in production, this would use the feature pipeline
    # to get the latest observation and run through SignalInterpreter
    # + PositionSizer. For now, returns an empty list.
    return []


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
