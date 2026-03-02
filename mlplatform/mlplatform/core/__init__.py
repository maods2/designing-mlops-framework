"""Core framework abstractions."""

from mlplatform.core.artifact_path_builder import ArtifactPathBuilder, ArtifactPaths
from mlplatform.core.artifact_registry import ArtifactRegistry
from mlplatform.core.context import ExecutionContext
from mlplatform.core.predictor import BasePredictor
from mlplatform.core.trainer import BaseTrainer

__all__ = [
    "ArtifactPathBuilder",
    "ArtifactPaths",
    "ArtifactRegistry",
    "ExecutionContext",
    "BaseTrainer",
    "BasePredictor",
]
