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
