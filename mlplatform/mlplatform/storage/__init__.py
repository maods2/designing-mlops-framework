"""Storage backends for artifact persistence."""

from mlplatform.storage.base import Storage
from mlplatform.storage.gcs import GCSStorage
from mlplatform.storage.local import LocalFileSystem

__all__ = ["Storage", "LocalFileSystem", "GCSStorage"]
