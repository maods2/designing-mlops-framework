"""Abstract ArtifactStore interface for model versioning and resolution."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class ArtifactStore(ABC):
    """Logical model versioning and resolution, separate from physical storage."""

    @abstractmethod
    def register_model(self, model_name: str, metadata: dict[str, Any]) -> None:
        """Register a trained model with metadata (version, metrics, etc.)."""
        ...

    @abstractmethod
    def resolve_model(self, model_name: str, version: str) -> dict[str, Any]:
        """Resolve a model by name and version, returning metadata including storage path."""
        ...
