"""Abstract Storage interface for artifact persistence."""

from abc import ABC, abstractmethod
from typing import Any


class Storage(ABC):
    """Abstract interface for artifact storage backends."""

    @abstractmethod
    def save(self, path: str, obj: Any) -> None:
        """Save an object to the given path."""
        ...

    @abstractmethod
    def load(self, path: str) -> Any:
        """Load an object from the given path."""
        ...

    @abstractmethod
    def list(self, prefix: str = "") -> list[str]:
        """List immediate children (files or subdirs) under base_path/prefix.

        Returns names only (no full paths). Used for version resolution (e.g. listing
        version directories to find 'latest').
        """
        ...
