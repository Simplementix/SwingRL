"""Portfolio state reader from SQLite positions and snapshots.

Provides real-time portfolio state for risk evaluation and observation assembly.
All reads from SQLite tables: positions, portfolio_snapshots, trades.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

import numpy as np
import structlog

if TYPE_CHECKING:
    from swingrl.config.schema import SwingRLConfig
    from swingrl.data.db import DatabaseManager

log = structlog.get_logger(__name__)


class PositionTracker:
    """Read portfolio state from SQLite for risk checks and observation assembly.

    Args:
        db: DatabaseManager for SQLite access.
        config: SwingRLConfig for initial capital and symbol lists.
    """

    def __init__(self, db: DatabaseManager, config: SwingRLConfig) -> None:
        """Initialize position tracker."""
        self._db = db
        self._config = config

    def _initial_capital(self, env: str) -> float:
        """Return initial capital for the given environment."""
        if env == "equity":
            return self._config.capital.equity_usd
        return self._config.capital.crypto_usd

    def _symbols(self, env: str) -> list[str]:
        """Return symbol list for the given environment."""
        if env == "equity":
            return self._config.equity.symbols
        return self._config.crypto.symbols

    def get_portfolio_value(self, env: str) -> float:
        """Return total portfolio value for environment from latest snapshot.

        Falls back to initial capital from config if no snapshot exists.

        Args:
            env: "equity" or "crypto".

        Returns:
            Portfolio value as float.
        """
        with self._db.connection() as conn:
            row = conn.execute(
                "SELECT total_value FROM portfolio_snapshots "
                "WHERE environment = %s ORDER BY timestamp DESC LIMIT 1",
                (env,),
            ).fetchone()
        if row is not None:
            return float(row["total_value"])
        fallback = self._initial_capital(env)
        log.warning(
            "portfolio_value_fallback_to_initial_capital",
            environment=env,
            initial_capital=fallback,
        )
        return fallback

    def get_positions(self, env: str) -> list[dict[str, Any]]:
        """Return list of positions for environment.

        Args:
            env: "equity" or "crypto".

        Returns:
            List of position dicts with keys: symbol, quantity, cost_basis,
            last_price, unrealized_pnl, updated_at.
        """
        with self._db.connection() as conn:
            rows = conn.execute(
                "SELECT symbol, quantity, cost_basis, last_price, "
                "unrealized_pnl, updated_at FROM positions WHERE environment = %s",
                (env,),
            ).fetchall()
        return [dict(row) for row in rows]

    def get_high_water_mark(self, env: str) -> float:
        """Return high water mark for environment from latest snapshot.

        Falls back to initial capital if no snapshot exists.

        Args:
            env: "equity" or "crypto".

        Returns:
            High water mark as float.
        """
        with self._db.connection() as conn:
            row = conn.execute(
                "SELECT high_water_mark FROM portfolio_snapshots "
                "WHERE environment = %s ORDER BY timestamp DESC LIMIT 1",
                (env,),
            ).fetchone()
        if row is not None:
            return float(row["high_water_mark"])
        fallback = self._initial_capital(env)
        log.warning(
            "high_water_mark_fallback_to_initial_capital",
            environment=env,
            initial_capital=fallback,
        )
        return fallback

    def get_daily_pnl(self, env: str) -> float:
        """Return today's P&L for environment.

        Returns 0.0 if no snapshot exists for today.

        Args:
            env: "equity" or "crypto".

        Returns:
            Daily P&L as float.
        """
        today_prefix = datetime.now(tz=UTC).strftime("%Y-%m-%d")
        with self._db.connection() as conn:
            row = conn.execute(
                "SELECT daily_pnl FROM portfolio_snapshots "
                "WHERE environment = %s AND timestamp LIKE %s "
                "ORDER BY timestamp DESC LIMIT 1",
                (env, f"{today_prefix}%"),
            ).fetchone()
        if row is not None:
            return float(row["daily_pnl"])
        return 0.0

    def get_exposure(self, env: str) -> float:
        """Return current exposure ratio for environment.

        Exposure = sum(abs(quantity * last_price)) / portfolio_value.

        Args:
            env: "equity" or "crypto".

        Returns:
            Exposure ratio between 0.0 and 1.0+ (can exceed 1.0 with leverage).
        """
        positions = self.get_positions(env)
        if not positions:
            return 0.0
        invested: float = sum(abs(p["quantity"] * (p["last_price"] or 0.0)) for p in positions)
        portfolio_value = self.get_portfolio_value(env)
        if portfolio_value <= 0:
            return 0.0
        return float(invested / portfolio_value)

    def record_snapshot(
        self,
        env: str,
        portfolio_value: float,
        cash: float,
        daily_pnl: float,
    ) -> None:
        """Write a portfolio snapshot row with auto-computed HWM and drawdown.

        Args:
            env: "equity" or "crypto".
            portfolio_value: Current total portfolio value.
            cash: Current cash balance.
            daily_pnl: Today's profit/loss.
        """
        prev_hwm = self.get_high_water_mark(env)
        hwm = max(prev_hwm, portfolio_value)
        drawdown_pct = 1.0 - portfolio_value / hwm if hwm > 0 else 0.0
        timestamp = datetime.now(tz=UTC).isoformat()

        with self._db.connection() as conn:
            conn.execute(
                "INSERT INTO portfolio_snapshots "
                "(timestamp, environment, total_value, cash_balance, "
                "high_water_mark, daily_pnl, drawdown_pct) "
                "VALUES (%s, %s, %s, %s, %s, %s, %s)",
                (timestamp, env, portfolio_value, cash, hwm, daily_pnl, drawdown_pct),
            )
        log.info(
            "snapshot_recorded",
            environment=env,
            portfolio_value=portfolio_value,
            hwm=hwm,
            drawdown_pct=drawdown_pct,
        )

    def get_portfolio_state_array(self, env: str) -> np.ndarray:
        """Build numpy array for ObservationAssembler.

        Equity (35,): [cash_ratio, exposure, daily_return,
                       per-asset interleaved: weight, weight_dev, unrealized_pnl_pct,
                       days_since_trade] x 8
        Crypto (11,): [cash_ratio, exposure, daily_return,
                       per-asset interleaved: weight, weight_dev, unrealized_pnl_pct,
                       days_since_trade] x 2

        Args:
            env: "equity" or "crypto".

        Returns:
            Numpy array with portfolio state.
        """
        symbols = self._symbols(env)
        n_assets = len(symbols)
        # 3 global + 4 per asset (weight, weight_deviation, unrealized_pnl_pct, days_since_trade)
        state = np.zeros(3 + 4 * n_assets, dtype=np.float32)

        portfolio_value = self.get_portfolio_value(env)
        exposure = self.get_exposure(env)
        daily_pnl = self.get_daily_pnl(env)
        daily_return = daily_pnl / portfolio_value if portfolio_value > 0 else 0.0

        state[0] = 1.0 - exposure  # cash_ratio
        state[1] = exposure
        state[2] = daily_return

        positions = self.get_positions(env)
        pos_by_symbol = {p["symbol"]: p for p in positions}

        now = datetime.now(tz=UTC)

        for i, symbol in enumerate(symbols):
            base_idx = 3 + i * 4
            pos = pos_by_symbol.get(symbol)
            if pos is None:
                continue

            qty = pos["quantity"]
            last_price = pos["last_price"] or 0.0
            cost_basis = pos["cost_basis"] or 0.0
            position_value = abs(qty * last_price)

            # Weight
            weight = position_value / portfolio_value if portfolio_value > 0 else 0.0
            state[base_idx] = weight

            # Weight deviation from equal-weight target (0.0 when no exposure)
            state[base_idx + 1] = weight - (1.0 / n_assets) if exposure > 0 else 0.0

            # Unrealized PnL %
            if cost_basis > 0:
                state[base_idx + 2] = (last_price - cost_basis) / cost_basis
            else:
                state[base_idx + 2] = 0.0

            # Days since last trade for this symbol
            days = self._days_since_trade(env, symbol, now)
            state[base_idx + 3] = float(days)

        return state

    def _days_since_trade(self, env: str, symbol: str, now: datetime) -> int:
        """Return days since last trade for symbol in environment.

        Args:
            env: "equity" or "crypto".
            symbol: Ticker symbol.
            now: Current UTC datetime.

        Returns:
            Number of days since last trade, or 0 if no trades found.
        """
        with self._db.connection() as conn:
            row = conn.execute(
                "SELECT timestamp FROM trades "
                "WHERE symbol = %s AND environment = %s "
                "ORDER BY timestamp DESC LIMIT 1",
                (symbol, env),
            ).fetchone()
        if row is None:
            return 0
        try:
            last_trade_ts = datetime.fromisoformat(row["timestamp"])
            if last_trade_ts.tzinfo is None:
                last_trade_ts = last_trade_ts.replace(tzinfo=UTC)
            return max(0, (now - last_trade_ts).days)
        except (ValueError, TypeError):
            return 0
