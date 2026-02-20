"""ArtifactRegistry - manages artifact path conventions and persistence."""

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
        self.storage = storage
        self.feature_name = feature_name
        self.model_name = model_name
        self.version = version

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
        self.storage.save(f"{self.base_path}/{name}", obj)

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
