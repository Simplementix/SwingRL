"""SwingRL custom exception hierarchy.

All modules raise typed subclasses of SwingRLError.
Callers catch specific types (e.g., except BrokerError) rather than bare Exception.
"""

from __future__ import annotations


class SwingRLError(Exception):
    """Base class for all SwingRL application errors."""


class ConfigError(SwingRLError, ValueError):
    """Config file missing, unreadable, or fails Pydantic validation."""


class BrokerError(SwingRLError):
    """Broker API errors: auth failure, order rejection, rate limit exceeded."""


class DataError(SwingRLError):
    """Data pipeline errors: missing bars, validation failure, quarantine trigger."""


class ModelError(SwingRLError):
    """Model lifecycle errors: file missing, incompatible shape, load failure."""


class CircuitBreakerError(SwingRLError):
    """Circuit breaker is active; trading halted for this environment."""


class RiskVetoError(SwingRLError):
    """Risk management layer vetoed the trade; order not submitted."""
