"""Tests for SwingRL utility modules: exception hierarchy and logging."""

import structlog


def test_swingrl_error_is_base_exception() -> None:
    """SwingRLError must be catchable as Exception."""
    from swingrl.utils.exceptions import SwingRLError

    # Verify SwingRLError is a proper Exception subclass
    assert issubclass(SwingRLError, Exception)
    err = SwingRLError("base error")
    assert str(err) == "base error"


def test_config_error_is_subclass() -> None:
    """ConfigError must be a subclass of SwingRLError."""
    from swingrl.utils.exceptions import ConfigError, SwingRLError

    assert issubclass(ConfigError, SwingRLError)
    assert isinstance(ConfigError("test"), SwingRLError)


def test_broker_error_is_subclass() -> None:
    """BrokerError must be a subclass of SwingRLError."""
    from swingrl.utils.exceptions import BrokerError, SwingRLError

    assert issubclass(BrokerError, SwingRLError)
    assert isinstance(BrokerError("test"), SwingRLError)


def test_data_error_is_subclass() -> None:
    """DataError must be a subclass of SwingRLError."""
    from swingrl.utils.exceptions import DataError, SwingRLError

    assert issubclass(DataError, SwingRLError)
    assert isinstance(DataError("test"), SwingRLError)


def test_model_error_is_subclass() -> None:
    """ModelError must be a subclass of SwingRLError."""
    from swingrl.utils.exceptions import ModelError, SwingRLError

    assert issubclass(ModelError, SwingRLError)
    assert isinstance(ModelError("test"), SwingRLError)


def test_circuit_breaker_error_is_subclass() -> None:
    """CircuitBreakerError must be a subclass of SwingRLError."""
    from swingrl.utils.exceptions import CircuitBreakerError, SwingRLError

    assert issubclass(CircuitBreakerError, SwingRLError)
    assert isinstance(CircuitBreakerError("test"), SwingRLError)


def test_risk_veto_error_is_subclass() -> None:
    """RiskVetoError must be a subclass of SwingRLError."""
    from swingrl.utils.exceptions import RiskVetoError, SwingRLError

    assert issubclass(RiskVetoError, SwingRLError)
    assert isinstance(RiskVetoError("test"), SwingRLError)


def test_all_seven_classes_importable_from_utils() -> None:
    """All 7 exception classes must be importable from swingrl.utils."""
    from swingrl.utils import (
        BrokerError,
        CircuitBreakerError,
        ConfigError,
        DataError,
        ModelError,
        RiskVetoError,
        SwingRLError,
    )

    # All should be importable without error
    assert SwingRLError is not None
    assert ConfigError is not None
    assert BrokerError is not None
    assert DataError is not None
    assert ModelError is not None
    assert CircuitBreakerError is not None
    assert RiskVetoError is not None


def test_configure_logging_console_renderer_does_not_raise() -> None:
    """configure_logging(json_logs=False) must not raise."""
    from swingrl.utils import configure_logging

    configure_logging(json_logs=False)


def test_configure_logging_json_renderer_does_not_raise() -> None:
    """configure_logging(json_logs=True) must not raise."""
    from swingrl.utils import configure_logging

    configure_logging(json_logs=True)


def test_structlog_logger_works_after_configure() -> None:
    """After configure_logging(), structlog logger must produce output without raising."""
    from swingrl.utils import configure_logging

    configure_logging(json_logs=False)
    log = structlog.get_logger("test")
    log.info("smoke_check", status="ok")  # must not raise
