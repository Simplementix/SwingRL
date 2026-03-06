"""SwingRL structured logging configuration using structlog.

Usage:
    from swingrl.utils.logging import configure_logging
    configure_logging(json_logs=False)  # local dev — coloured console output
    configure_logging(json_logs=True)   # production/Docker — JSON lines output

After calling configure_logging(), obtain a logger with:
    import structlog
    log = structlog.get_logger(__name__)
    log.info("event", key="value")
"""

from __future__ import annotations

import logging

import structlog


def configure_logging(json_logs: bool = False, log_level: str = "INFO") -> None:
    """Configure structlog for the SwingRL application.

    Args:
        json_logs: When True, emit JSON lines (production/Docker).
                   When False, use coloured console output (local dev).
        log_level: Standard Python logging level string (e.g., "INFO", "DEBUG").
    """
    # Shared processors run for every log entry regardless of renderer
    shared_processors: list[structlog.types.Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
    ]

    if json_logs:
        # Production: machine-readable JSON lines for log aggregators
        renderer: structlog.types.Processor = structlog.processors.JSONRenderer()
    else:
        # Local dev: human-readable coloured output
        renderer = structlog.dev.ConsoleRenderer(colors=True)

    structlog.configure(
        processors=[
            *shared_processors,
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        wrapper_class=structlog.stdlib.BoundLogger,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )

    formatter = structlog.stdlib.ProcessorFormatter(
        processor=renderer,
        foreign_pre_chain=shared_processors,
    )

    handler = logging.StreamHandler()
    handler.setFormatter(formatter)

    root_logger = logging.getLogger()
    root_logger.handlers = [handler]
    root_logger.setLevel(log_level)
