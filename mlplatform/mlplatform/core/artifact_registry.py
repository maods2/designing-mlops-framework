"""ArtifactRegistry - manages artifact path conventions and persistence."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from mlplatform.storage.base import Storage


class ArtifactRegistry:
    """Manages artifact path conventions and delegates persistence to Storage.

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
        model_artifact_dir: Optional[str] = None,
        metrics_artifact_dir: Optional[str] = None,
    ) -> None:
        self.storage = storage
        self.feature_name = feature_name
        self.model_name = model_name
        self.version = version
        self._storage_base_path = storage_base_path
        self.artifact_path = artifact_path
        self.model_artifact_dir = model_artifact_dir
        self.metrics_artifact_dir = metrics_artifact_dir

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
        """Build a storage key for an artifact, with optional cross-model override."""
        if self._storage_base_path is not None:
            if model_name is not None or version is not None:
                raise NotImplementedError(
                    "Cross-model loading not yet supported in standardized path mode"
                )
            return name
        m = model_name or self.model_name
        v = version or self.version
        return f"{self.feature_name}/{m}/{v}/{name}"

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
        return self.storage.load(path)
