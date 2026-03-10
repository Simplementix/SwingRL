"""SwingRL structured logging configuration using structlog.

Usage:
    from swingrl.utils.logging import configure_logging
    configure_logging(json_logs=False)  # local dev — coloured console output
    configure_logging(json_logs=True)   # production/Docker — JSON lines output

Optional file logging (always JSON, with rotation):
    configure_logging(json_logs=False, log_file=Path("logs/swingrl.log"))

After calling configure_logging(), obtain a logger with:
    import structlog
    log = structlog.get_logger(__name__)
    log.info("event", key="value")
"""

from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

import structlog


def configure_logging(
    json_logs: bool = False,
    log_level: str = "INFO",
    log_file: Path | None = None,
    max_bytes: int = 10_485_760,
    backup_count: int = 5,
) -> None:
    """Configure structlog for the SwingRL application.

    Args:
        json_logs: When True, emit JSON lines (production/Docker).
                   When False, use coloured console output (local dev).
        log_level: Standard Python logging level string (e.g., "INFO", "DEBUG").
        log_file: Optional path for a rotating JSON log file.
                  When provided, a RotatingFileHandler is added alongside the stream handler.
                  The file handler always uses JSONRenderer regardless of json_logs flag.
        max_bytes: Maximum bytes per log file before rotation (default 10MB).
        backup_count: Number of rotated backup files to keep (default 5).
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

    # Console handler with chosen renderer
    console_formatter = structlog.stdlib.ProcessorFormatter(
        processor=renderer,
        foreign_pre_chain=shared_processors,
    )
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(console_formatter)

    root_logger = logging.getLogger()
    root_logger.handlers = [stream_handler]
    root_logger.setLevel(log_level)

    # Optional file handler — always JSON for machine parsing
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        json_formatter = structlog.stdlib.ProcessorFormatter(
            processor=structlog.processors.JSONRenderer(),
            foreign_pre_chain=shared_processors,
        )
        file_handler = RotatingFileHandler(
            filename=str(log_file),
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding="utf-8",
        )
        file_handler.setFormatter(json_formatter)
        root_logger.addHandler(file_handler)
