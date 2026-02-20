"""Core framework abstractions."""

from mlplatform.core.artifact_registry import ArtifactRegistry
from mlplatform.core.context import ExecutionContext
from mlplatform.core.predictor import BasePredictor
from mlplatform.core.trainer import BaseTrainer

__all__ = [
    "ArtifactRegistry",
    "ExecutionContext",
    "BaseTrainer",
    "BasePredictor",
]
