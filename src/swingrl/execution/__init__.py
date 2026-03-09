"""Execution layer: order flow pipeline and risk management.

Exports pipeline data types for convenient access:
    from swingrl.execution import TradeSignal, SizedOrder, ValidatedOrder, FillResult
"""

from __future__ import annotations

from swingrl.execution.types import (
    FillResult,
    RiskDecision,
    SizedOrder,
    TradeSignal,
    ValidatedOrder,
)

__all__ = [
    "FillResult",
    "RiskDecision",
    "SizedOrder",
    "TradeSignal",
    "ValidatedOrder",
]
