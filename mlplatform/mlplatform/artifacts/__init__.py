"""Artifact store backends for model versioning and resolution."""

from mlplatform.artifacts.base import ArtifactStore
from mlplatform.artifacts.local import LocalArtifactStore

__all__ = ["ArtifactStore", "LocalArtifactStore"]
