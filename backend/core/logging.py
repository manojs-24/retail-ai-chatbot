"""
Logging configuration for the Retail AI backend.

Provides:
- ``setup_logging()`` — call once at application startup.
- ``get_logger(name)``  — returns a named logger ready for use.

Log output goes to both the console (stdout) and a rotating file
(default: ``./logs/retail_ai.log``, max 10 MB, 5 backups).

Note: This module is named ``logging.py`` intentionally to match the project
convention; it does **not** shadow the stdlib because imports resolve the
package-qualified path ``backend.core.logging`` rather than the bare name.
"""

from __future__ import annotations

import logging
import logging.handlers
from pathlib import Path

__all__ = [
    "get_logger",
    "setup_logging",
]


_LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

_configured: bool = False


def setup_logging(
    log_level: str = "INFO",
    log_file_path: str = "./logs/retail_ai.log",
) -> None:
    """
    Configure the root logger with a console handler and a rotating file handler.

    This function is idempotent — calling it more than once is safe.

    Args:
        log_level:     One of DEBUG / INFO / WARNING / ERROR / CRITICAL.
        log_file_path: Path to the rotating log file.
    """
    global _configured
    if _configured:
        return

    numeric_level = getattr(logging, log_level.upper(), logging.INFO)

    # Ensure the log directory exists.
    log_path = Path(log_file_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    formatter = logging.Formatter(fmt=_LOG_FORMAT, datefmt=_DATE_FORMAT)

    # ------------------------------------------------------------------
    # Console handler
    # ------------------------------------------------------------------
    console_handler = logging.StreamHandler()
    console_handler.setLevel(numeric_level)
    console_handler.setFormatter(formatter)

    # ------------------------------------------------------------------
    # Rotating file handler (10 MB per file, keep 5 backups)
    # ------------------------------------------------------------------
    file_handler = logging.handlers.RotatingFileHandler(
        filename=log_file_path,
        maxBytes=10 * 1024 * 1024,  # 10 MB
        backupCount=5,
        encoding="utf-8",
    )
    file_handler.setLevel(numeric_level)
    file_handler.setFormatter(formatter)

    # ------------------------------------------------------------------
    # Root logger
    # ------------------------------------------------------------------
    root_logger = logging.getLogger()
    root_logger.setLevel(numeric_level)
    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)

    _configured = True
    root_logger.info("Logging initialised — level=%s  file=%s", log_level, log_file_path)


def get_logger(name: str) -> logging.Logger:
    """
    Return a named logger.

    Args:
        name: Typically ``__name__`` of the calling module.

    Returns:
        A :class:`logging.Logger` instance.
    """
    return logging.getLogger(name)
