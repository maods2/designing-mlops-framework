"""Storage backends for artifact persistence."""

from mlops_framework.backends.storage.base import StorageBackend
from mlops_framework.backends.storage.local import LocalStorage

__all__ = ["StorageBackend", "LocalStorage"]
