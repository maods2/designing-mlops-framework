"""Google Cloud Storage backend for artifact persistence."""

from __future__ import annotations

import io
import tempfile
from typing import Any

import joblib

from mlplatform.storage.base import Storage


class GCSStorage(Storage):
    """Google Cloud Storage backend.

    ``base_path`` should be a ``gs://bucket/prefix`` URI. Artifacts are
    stored as blobs under ``{base_path}/{path}`` using joblib serialization.
    """

    def __init__(self, base_path: str) -> None:
        from google.cloud import storage as gcs

        self.base_path = base_path.rstrip("/")
        self._client = gcs.Client()
        bucket_name, self._prefix = self._parse_gs_uri(self.base_path)
        self._bucket = self._client.bucket(bucket_name)

    @staticmethod
    def _parse_gs_uri(uri: str) -> tuple[str, str]:
        """Split ``gs://bucket/prefix`` into ``(bucket, prefix)``."""
        if not uri.startswith("gs://"):
            raise ValueError(f"GCS base_path must start with gs://, got: {uri}")
        without_scheme = uri[5:]
        parts = without_scheme.split("/", 1)
        bucket = parts[0]
        prefix = parts[1] if len(parts) > 1 else ""
        return bucket, prefix

    def _blob_path(self, path: str) -> str:
        if self._prefix:
            return f"{self._prefix}/{path}"
        return path

    def save(self, path: str, obj: Any) -> None:
        buf = io.BytesIO()
        joblib.dump(obj, buf)
        buf.seek(0)
        blob = self._bucket.blob(self._blob_path(path))
        blob.upload_from_file(buf, content_type="application/octet-stream")

    def load(self, path: str) -> Any:
        blob = self._bucket.blob(self._blob_path(path))
        buf = io.BytesIO()
        blob.download_to_file(buf)
        buf.seek(0)
        return joblib.load(buf)
