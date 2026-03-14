"""ArtifactRegistry - manages artifact path conventions and persistence.

Unified registry merging plat/core (path conventions, cross-model loading,
storage property) with mlplatform/artifacts (format dispatch).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from mlplatform.storage.base import Storage


class ArtifactRegistry:
    """Manages artifact path conventions and delegates persistence to Storage.

    Encapsulates the ``{feature}/{model}/{version}/{artifact_name}`` convention
    so that callers never build paths manually.
    """

    def __init__(
        self,
        storage: Storage,
        feature_name: str,
        model_name: str,
        version: str,
    ) -> None:
        self._storage = storage
        self.feature_name = feature_name
        self.model_name = model_name
        self.version = version

    @property
    def storage(self) -> Storage:
        """Direct access to the underlying Storage backend."""
        return self._storage

    @property
    def base_path(self) -> str:
        return f"{self.feature_name}/{self.model_name}/{self.version}"

    def resolve_path(
        self,
        name: str,
        *,
        model_name: str | None = None,
        version: str | None = None,
    ) -> str:
        """Build a storage key for an artifact, with optional cross-model override."""
        m = model_name or self.model_name
        v = version or self.version
        return f"{self.feature_name}/{m}/{v}/{name}"

    def save(self, name: str, obj: Any) -> None:
        """Persist *obj* under the current model's versioned path."""
        self._storage.save(f"{self.base_path}/{name}", obj)

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
        return self._storage.load(path)

    def exists(self, name: str) -> bool:
        """Return True if artifact exists."""
        return self._storage.exists(self.resolve_path(name))
