"""Position reconciliation: DB vs broker state comparison and correction.

Detects mismatches between SQLite positions and broker-reported positions,
auto-corrects DB to match broker (source of truth), inserts adjustment
trades, and sends Discord warning alerts.

Usage:
    from swingrl.execution.reconciliation import PositionReconciler
    reconciler = PositionReconciler(config, db, adapter, alerter)
    adjustments = reconciler.reconcile("equity")
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

import structlog

from swingrl.execution.fill_processor import FillProcessor
from swingrl.execution.risk.position_tracker import PositionTracker

if TYPE_CHECKING:
    from swingrl.config.schema import SwingRLConfig
    from swingrl.data.db import DatabaseManager
    from swingrl.execution.adapters.base import ExchangeAdapter
    from swingrl.monitoring.alerter import Alerter

log = structlog.get_logger(__name__)


class PositionReconciler:
    """Reconcile DB positions against broker state.

    Trusts broker as source of truth. Auto-corrects DB and inserts
    adjustment trade rows for audit trail.
    """

    def __init__(
        self,
        config: SwingRLConfig,
        db: DatabaseManager,
        adapter: ExchangeAdapter,
        alerter: Alerter,
    ) -> None:
        """Initialize position reconciler.

        Args:
            config: Validated SwingRLConfig.
            db: DatabaseManager for SQLite access.
            adapter: Exchange adapter for broker position reads (equity only).
            alerter: Discord alerter for mismatch warnings.
        """
        self._config = config
        self._db = db
        self._adapter = adapter
        self._alerter = alerter
        self._tracker = PositionTracker(db=db, config=config)
        self._fill_processor = FillProcessor(db=db)

    def reconcile(self, env_name: str) -> list[dict[str, Any]]:
        """Compare DB positions vs broker positions and correct mismatches.

        Args:
            env_name: Environment name ("equity" or "crypto").

        Returns:
            List of adjustment dicts describing corrections made.
        """
        # Crypto uses virtual balance -- no broker-side state to reconcile
        if env_name == "crypto":
            log.info("reconciliation_skipped_crypto")
            return []

        log.info("reconciliation_started", env=env_name)

        # Get DB positions
        db_positions = self._tracker.get_positions(env_name)
        db_by_symbol: dict[str, dict[str, Any]] = {p["symbol"]: p for p in db_positions}

        # Get broker positions
        broker_positions = self._adapter.get_positions()
        broker_by_symbol: dict[str, dict[str, Any]] = {
            str(p["symbol"]): p for p in broker_positions
        }

        adjustments: list[dict[str, Any]] = []

        # Check broker positions against DB
        for symbol, broker_pos in broker_by_symbol.items():
            broker_qty = float(broker_pos.get("quantity", 0))
            db_pos = db_by_symbol.get(symbol)

            if db_pos is None:
                # Position in broker but not in DB: create it
                adj = self._create_position(env_name, symbol, broker_pos)
                adjustments.append(adj)
                log.warning(
                    "reconcile_missing_db_position",
                    symbol=symbol,
                    broker_qty=broker_qty,
                )
            else:
                db_qty = float(db_pos.get("quantity", 0))
                if abs(broker_qty - db_qty) > 0.0001:
                    # Quantity mismatch: adjust DB to match broker
                    adj = self._adjust_quantity(env_name, symbol, db_qty, broker_qty, broker_pos)
                    adjustments.append(adj)
                    log.warning(
                        "reconcile_quantity_mismatch",
                        symbol=symbol,
                        db_qty=db_qty,
                        broker_qty=broker_qty,
                    )

        # Check DB positions not in broker (stale): remove them
        for symbol in db_by_symbol:
            if symbol not in broker_by_symbol:
                adj = self._remove_stale_position(env_name, symbol)
                adjustments.append(adj)
                log.warning(
                    "reconcile_stale_db_position",
                    symbol=symbol,
                )

        # Send alert if any mismatches
        if adjustments:
            adj_summary = "; ".join(f"{a['symbol']}: {a['action']}" for a in adjustments)
            self._alerter.send_alert(
                level="warning",
                title="Position Reconciliation Mismatch",
                message=f"Reconciled {len(adjustments)} position(s) for {env_name}: {adj_summary}",
                environment=env_name,
            )

        log.info(
            "reconciliation_complete",
            env=env_name,
            adjustments=len(adjustments),
        )
        return adjustments

    def _create_position(
        self,
        env_name: str,
        symbol: str,
        broker_pos: dict[str, Any],
    ) -> dict[str, Any]:
        """Create a DB position row to match broker.

        Args:
            env_name: Environment name.
            symbol: Ticker symbol.
            broker_pos: Broker position dict.

        Returns:
            Adjustment record dict.
        """
        qty = float(broker_pos.get("quantity", 0))
        price = float(broker_pos.get("avg_entry_price", 0))
        now = datetime.now(UTC).isoformat()

        with self._db.connection() as conn:
            conn.execute(
                "INSERT INTO positions "
                "(symbol, environment, quantity, cost_basis, last_price, "
                "unrealized_pnl, updated_at) "
                "VALUES (%s, %s, %s, %s, %s, %s, %s)",
                (symbol, env_name, qty, price, price, 0.0, now),
            )

        # Record adjustment trade
        self._fill_processor.record_adjustment(
            symbol=symbol,
            environment=env_name,
            quantity_delta=qty,
            price=price,
            reason=f"reconcile: created position from broker (qty={qty})",
        )

        return {
            "symbol": symbol,
            "action": f"created (qty={qty})",
            "delta": qty,
        }

    def _adjust_quantity(
        self,
        env_name: str,
        symbol: str,
        db_qty: float,
        broker_qty: float,
        broker_pos: dict[str, Any],
    ) -> dict[str, Any]:
        """Adjust DB position quantity to match broker.

        Args:
            env_name: Environment name.
            symbol: Ticker symbol.
            db_qty: Current DB quantity.
            broker_qty: Broker-reported quantity.
            broker_pos: Broker position dict.

        Returns:
            Adjustment record dict.
        """
        delta = broker_qty - db_qty
        price = float(broker_pos.get("avg_entry_price", 0))
        now = datetime.now(UTC).isoformat()

        with self._db.connection() as conn:
            conn.execute(
                "UPDATE positions SET quantity = %s, updated_at = %s "
                "WHERE symbol = %s AND environment = %s",
                (broker_qty, now, symbol, env_name),
            )

        self._fill_processor.record_adjustment(
            symbol=symbol,
            environment=env_name,
            quantity_delta=delta,
            price=price,
            reason=f"reconcile: adjusted qty from {db_qty} to {broker_qty}",
        )

        return {
            "symbol": symbol,
            "action": f"adjusted (db={db_qty} -> broker={broker_qty})",
            "delta": delta,
        }

    def _remove_stale_position(self, env_name: str, symbol: str) -> dict[str, Any]:
        """Remove a DB position that no longer exists at broker.

        Args:
            env_name: Environment name.
            symbol: Ticker symbol.

        Returns:
            Adjustment record dict.
        """
        # Read current qty before delete for adjustment record
        with self._db.connection() as conn:
            row = conn.execute(
                "SELECT quantity, cost_basis FROM positions WHERE symbol = %s AND environment = %s",
                (symbol, env_name),
            ).fetchone()

            qty = float(row["quantity"]) if row else 0.0
            price = float(row["cost_basis"]) if row else 0.0

            conn.execute(
                "DELETE FROM positions WHERE symbol = %s AND environment = %s",
                (symbol, env_name),
            )

        if qty > 0:
            self._fill_processor.record_adjustment(
                symbol=symbol,
                environment=env_name,
                quantity_delta=-qty,
                price=price,
                reason=f"reconcile: removed stale position (qty={qty})",
            )

        return {
            "symbol": symbol,
            "action": f"removed (was qty={qty})",
            "delta": -qty,
        }
