"""Storage backends for artifact persistence."""

from mlplatform.storage.base import Storage
from mlplatform.storage.local import LocalFileSystem


def __getattr__(name: str):
    """Lazy-load GCSStorage to avoid import failure when google-cloud-storage is not installed."""
    if name == "GCSStorage":
        from mlplatform.storage.gcs import GCSStorage
        return GCSStorage
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["Storage", "LocalFileSystem", "GCSStorage"]
