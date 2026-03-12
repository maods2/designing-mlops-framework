"""Abstract Storage interface for artifact persistence."""

from abc import ABC, abstractmethod
from typing import Any


class Storage(ABC):
    """Abstract interface for artifact storage backends.

    Subclasses **must** implement :meth:`save` and :meth:`load`.

    The remaining methods (``exists``, ``list_artifacts``, ``delete``,
    ``save_bytes``) have default implementations so that existing backends
    continue to work without changes.  Override them for better performance
    or native support.
    """

    @abstractmethod
    def save(self, path: str, obj: Any) -> None:
        """Save an object to the given path."""
        ...

    @abstractmethod
    def load(self, path: str) -> Any:
        """Load an object from the given path."""
        ...

    # ------------------------------------------------------------------
    # Optional methods with sensible defaults
    # ------------------------------------------------------------------

    def exists(self, path: str) -> bool:
        """Return ``True`` if an artifact exists at *path*.

        Default implementation attempts a load and catches exceptions.
        Subclasses should override with a cheaper check (e.g. HEAD request).
        """
        try:
            self.load(path)
            return True
        except Exception:
            return False

    def list_artifacts(self, prefix: str = "") -> list[str]:
        """List artifact keys under *prefix*.

        Not all backends can enumerate keys efficiently — the default
        raises ``NotImplementedError``.  Subclasses should override when
        listing is supported (e.g. GCS, S3).
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not support listing artifacts."
        )

    def delete(self, path: str) -> None:
        """Delete the artifact at *path*.

        Default raises ``NotImplementedError``.  Subclasses should override.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not support deleting artifacts."
        )

    def save_bytes(self, path: str, data: bytes) -> None:
        """Save raw bytes to *path*.

        Default delegates to :meth:`save` (which typically wraps with joblib).
        Override if the backend can store raw bytes more efficiently.
        """
        self.save(path, data)
