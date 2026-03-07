"""Backward-compatible shim — data I/O has moved to mlplatform.data.io."""

from mlplatform.data.io import (  # noqa: F401
    load_prediction_input,
    write_prediction_output,
)

__all__ = ["load_prediction_input", "write_prediction_output"]
