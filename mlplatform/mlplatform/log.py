"""Framework logging with configurable log levels."""

from __future__ import annotations

import logging


def get_logger(name: str = "mlplatform", level: str = "INFO") -> logging.Logger:
    """Return a configured logger. Reuses existing handlers to avoid duplication."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
        handler.setFormatter(fmt)
        logger.addHandler(handler)
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    return logger
