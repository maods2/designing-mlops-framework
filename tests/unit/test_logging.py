"""Unit tests for mlplatform.utils.logging."""

from __future__ import annotations

from mlplatform.utils.logging import get_logger


def test_get_logger_returns_logger():
    logger = get_logger("test_logger")
    assert logger.name == "test_logger"
    assert logger.level is not None


def test_get_logger_custom_name():
    logger = get_logger(name="my_module")
    assert logger.name == "my_module"
