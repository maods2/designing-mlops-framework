"""ArtifactRegistry - protocol and path-based implementation for artifact persistence."""

from __future__ import annotations

from typing import Any, Callable, Optional, Protocol

from mlplatform.storage.base import Storage


class ArtifactRegistryProtocol(Protocol):
    """Protocol for artifact registry implementations."""

    storage: Storage

    def save(self, name: str, obj: Any) -> None:
        """Persist an artifact under the current model's versioned path."""
        ...

    def load(
        self,
        name: str,
        *,
        model_name: str | None = None,
        version: str | None = None,
    ) -> Any:
        """Load an artifact. Override model_name/version for cross-model loading."""
        ...

    def resolve_path(
        self,
        name: str,
        *,
        model_name: str | None = None,
        version: str | None = None,
    ) -> str:
        """Build a storage key for an artifact."""
        ...


class PathArtifactRegistry:
    """Path-based artifact registry. Manages artifact path conventions and delegates persistence to Storage.

    Supports two modes:
    - Standardized: storage_base_path points to model/ subdir; artifacts saved as {name}.
    - Legacy: path = {feature}/{model}/{version}/{name} under storage root.
    """

    def __init__(
        self,
        storage: Storage,
        feature_name: str,
        model_name: str,
        version: str,
        *,
        storage_base_path: Optional[str] = None,
        artifact_path: Optional[str] = None,
        artifact_base_path: Optional[str] = None,
        model_artifact_dir: Optional[str] = None,
        metrics_artifact_dir: Optional[str] = None,
        storage_factory: Optional[Callable[[str], Storage]] = None,
    ) -> None:
        self.storage = storage
        self.feature_name = feature_name
        self.model_name = model_name
        self.version = version
        self._storage_base_path = storage_base_path
        self.artifact_path = artifact_path
        self._artifact_base_path = artifact_base_path
        self.model_artifact_dir = model_artifact_dir
        self.metrics_artifact_dir = metrics_artifact_dir
        self._storage_factory = storage_factory

    @property
    def base_path(self) -> str:
        """Path under which artifacts are stored (for Storage, relative to its base)."""
        if self._storage_base_path is not None:
            return self._storage_base_path
        return f"{self.feature_name}/{self.model_name}/{self.version}"

    def resolve_path(
        self,
        name: str,
        *,
        model_name: str | None = None,
        version: str | None = None,
    ) -> str:
        """Build a storage key for an artifact, with optional cross-model/version override."""
        if self._storage_base_path is not None:
            if model_name is not None or version is not None:
                # Cross-version in standardized mode: path relative to artifact_base_path
                m = model_name or self.model_name
                v = version or self.version
                model_train_version = f"{self.feature_name}_{m}_train_{v}"
                return f"{model_train_version}/model/{name}"
            return name
        m = model_name or self.model_name
        v = version or self.version
        return f"{self.feature_name}/{m}/{v}/{name}"

    def _get_storage_for_load(
        self,
        model_name: str | None,
        version: str | None,
    ) -> Storage:
        """Get the storage backend to use for loading."""
        if self._storage_base_path is not None and (model_name is not None or version is not None):
            if not self._artifact_base_path or not self._storage_factory:
                raise NotImplementedError(
                    "Cross-model/version loading in standardized path mode requires "
                    "artifact_base_path and storage_factory"
                )
            return self._storage_factory(self._artifact_base_path)
        return self.storage

    def save(self, name: str, obj: Any) -> None:
        """Persist *obj* under the current model's versioned path."""
        path = self.resolve_path(name)
        self.storage.save(path, obj)

    def load(
        self,
        name: str,
        *,
        model_name: str | None = None,
        version: str | None = None,
    ) -> Any:
        """Load an artifact. Defaults to current model/version.

        Override *model_name*/*version* for cross-model loading (e.g., ensembles).
        """
        path = self.resolve_path(name, model_name=model_name, version=version)
        storage = self._get_storage_for_load(model_name, version)
        return storage.load(path)


# Backward-compatible alias
ArtifactRegistry = PathArtifactRegistry
