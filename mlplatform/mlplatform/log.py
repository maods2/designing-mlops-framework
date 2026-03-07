"""Backward-compatible shim — logging has moved to mlplatform.utils.logging."""

from mlplatform.utils.logging import get_logger  # noqa: F401

__all__ = ["get_logger"]
