"""GCS storage backend for cloud execution."""

import io
import pickle
from typing import Optional

from mlops_framework.backends.storage.base import StorageBackend


class GCSStorage(StorageBackend):
    """
    Google Cloud Storage backend for artifacts.
    
    Requires google-cloud-storage (install via: pip install mlops-framework[gcs])
    """
    
    def __init__(self, bucket_name: str, prefix: str = "artifacts/"):
        """
        Initialize GCS storage.
        
        Args:
            bucket_name: GCS bucket name
            prefix: Blob prefix for artifacts (enables run-scoping)
        """
        try:
            from google.cloud import storage
        except ImportError:
            raise ImportError(
                "google-cloud-storage is required for GCSStorage. "
                "Install with: pip install google-cloud-storage"
            )
        self.client = storage.Client()
        self.bucket = self.client.bucket(bucket_name)
        self.prefix = prefix.rstrip("/") + "/"
    
    def save(self, name: str, obj: object) -> None:
        """Save object to GCS as {prefix}{name}.pkl."""
        blob = self.bucket.blob(f"{self.prefix}{name}.pkl")
        buffer = io.BytesIO()
        pickle.dump(obj, buffer)
        blob.upload_from_string(buffer.getvalue(), content_type="application/octet-stream")
    
    def load(self, name: str) -> object:
        """Load object from GCS."""
        blob = self.bucket.blob(f"{self.prefix}{name}.pkl")
        if not blob.exists():
            raise FileNotFoundError(f"Artifact '{name}' not found in GCS")
        data = blob.download_as_bytes()
        return pickle.loads(data)
