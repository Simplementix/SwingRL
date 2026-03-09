"""Pipeline data types for the execution layer.

Internal DTOs used between execution stages. These are plain dataclasses,
not Pydantic models -- they carry validated data through the pipeline.

Flow: TradeSignal -> SizedOrder -> ValidatedOrder -> FillResult
Plus: RiskDecision for audit logging.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Literal
from uuid import uuid4


@dataclass(frozen=True)
class TradeSignal:
    """Stage 1 output: ensemble action interpreted as trade intent."""

    environment: Literal["equity", "crypto"]
    symbol: str
    action: Literal["buy", "sell", "hold"]
    raw_weight: float


@dataclass(frozen=True)
class SizedOrder:
    """Stage 2 output: trade sized with Kelly, stops, and dollar amount."""

    symbol: str
    side: Literal["buy", "sell"]
    quantity: float
    dollar_amount: float
    stop_loss_price: float | None
    take_profit_price: float | None
    environment: Literal["equity", "crypto"]


@dataclass(frozen=True)
class ValidatedOrder:
    """Stage 3 output: order that passed all risk checks."""

    order: SizedOrder
    risk_checks_passed: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class FillResult:
    """Stage 5 output: broker fill confirmation."""

    trade_id: str
    symbol: str
    side: Literal["buy", "sell"]
    quantity: float
    fill_price: float
    commission: float
    slippage: float
    environment: Literal["equity", "crypto"]
    broker: Literal["alpaca", "binance_us"]


@dataclass
class RiskDecision:
    """Risk evaluation record logged to risk_decisions table."""

    decision_id: str = field(default_factory=lambda: str(uuid4()))
    timestamp: str = field(default_factory=lambda: datetime.now(tz=UTC).isoformat())
    environment: Literal["equity", "crypto"] = "equity"
    symbol: str = ""
    proposed_action: str = ""
    final_action: str = ""
    risk_rule_triggered: str = ""
    reason: str = ""
