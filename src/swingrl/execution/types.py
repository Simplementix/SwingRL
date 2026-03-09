"""Pipeline data types for the execution layer.

Internal DTOs used between execution stages. These are plain dataclasses,
not Pydantic models -- they carry validated data through the pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class TradeSignal:
    """Stage 1 output: ensemble action interpreted as trade intent."""

    environment: str  # "equity" | "crypto"
    symbol: str
    action: str  # "buy" | "sell" | "hold"
    raw_weight: float  # target portfolio weight from ensemble


@dataclass
class SizedOrder:
    """Stage 2 output: trade sized with Kelly, stops, and dollar amount."""

    symbol: str
    side: str  # "buy" | "sell"
    quantity: float
    dollar_amount: float
    stop_loss_price: float
    take_profit_price: float
    environment: str


@dataclass
class ValidatedOrder:
    """Stage 3 output: order that passed all risk checks."""

    sized_order: SizedOrder
    risk_checks_passed: list[str] = field(default_factory=list)


@dataclass
class FillResult:
    """Stage 5 output: broker fill confirmation."""

    trade_id: str
    symbol: str
    side: str
    quantity: float
    fill_price: float
    commission: float
    slippage: float
    environment: str
    broker: str


@dataclass
class RiskDecision:
    """Risk evaluation record logged to risk_decisions table."""

    decision_id: str
    timestamp: str
    environment: str
    symbol: str
    proposed_action: str
    final_action: str
    risk_rule_triggered: str
    reason: str
