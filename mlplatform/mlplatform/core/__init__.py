"""Core framework abstractions."""

from mlplatform.core.artifact_path_builder import ArtifactPathBuilder, ArtifactPaths
from mlplatform.core.artifact_registry import (
    ArtifactRegistry,
    ArtifactRegistryProtocol,
    PathArtifactRegistry,
)
from mlplatform.core.context import ExecutionContext
from mlplatform.core.predictor import BasePredictor
from mlplatform.core.trainer import BaseTrainer
from mlplatform.core.version_resolver import resolve_version

__all__ = [
    "ArtifactPathBuilder",
    "ArtifactPaths",
    "ArtifactRegistry",
    "ArtifactRegistryProtocol",
    "PathArtifactRegistry",
    "ExecutionContext",
    "BasePredictor",
    "BaseTrainer",
    "resolve_version",
]
