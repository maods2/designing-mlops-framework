"""Core framework abstractions."""

from mlplatform.core.context import ExecutionContext
from mlplatform.core.predictor import BasePredictor
from mlplatform.core.trainer import BaseTrainer

__all__ = [
    "ExecutionContext",
    "BaseTrainer",
    "BasePredictor",
]
